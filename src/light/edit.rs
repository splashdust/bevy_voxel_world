//! Voxel edits: what an edit changes for light, worked out on the main thread
//! the frame it is committed, and the relights it starts.
//!
//! An edited chunk's inputs in the world view take in its new voxels at once:
//! its occupancy and its emissive voxels. Its own direct sky follows from
//! them. The sky record takes in the position's new mask, and every loaded
//! chunk below whose column answer the edit changes gets its new answer: a
//! placed solid blocks a column, and a dug one can open it, provisionally if
//! some position above isn't known.
//!
//! A relight ripple then starts at the edited chunks and at the chunks whose
//! direct sky changed, and goes on to neighbours only across faces whose
//! hashes change (see [`relight`](super::relight)). Where an edit touches no
//! light, nothing relights: a placed solid relights only if its voxel held
//! light, a dug one only if a voxel beside it did, in either channel. Placing
//! or removing an emissive voxel always relights. Every edit committed in one
//! frame is worked out together, as one change.
//!
//! The ripple compares a solve's face hashes with those the world view holds.
//! While an earlier ripple is still out, those hashes are about to be
//! replaced, so an edit made then relights every chunk within light reach of
//! the voxels it changed instead, and doesn't look at light to skip any.
//!
//! These relights are urgent: they start at once, ahead of the pool budget,
//! even for chunks being re-meshed, whose chunk tasks skip their own solve.
//! The light of a chunk being re-meshed waits for its new mesh to land.
//!
//! Light tells whether what it holds for a position is current by the
//! position's edit stamp, [`VoxelEdits::edit_stamp`]: the chunk's edit
//! sequence, plus a forget epoch that each forget raises past every stamp
//! handed out before it, so the stamp never goes back. The world view, the
//! sky record and running probes each note the stamp their voxels were read
//! at. A chunk task's light that lands with a smaller stamp than the world
//! view holds is dropped as stale, and equal stamps mean the same voxels.
//! Forgetting sets the edit sequence back to 0, so the sequence alone would
//! have the chunk task that regenerates a forgotten chunk dropped as stale,
//! and could come back to a value light noted for different voxels.
//!
//! The epoch moves every position's stamp, not only those of the forgotten
//! chunks, so after a forget light treats all it holds as read before the
//! last edit: each chunk takes the miss path once, and the sky record and
//! running probes start over. A stamp above the epoch,
//! [`VoxelEdits::stamp_has_edits`], tells a position with edits, whose fill
//! hint light ignores.
//!
//! A removed edit leaves the voxel the world generates there, which light
//! looks up for a chunk that lost a few edits, so that, for instance, a
//! cleared dig closes its columns at once. A chunk that lost more, as a clear
//! or forget of a dense chunk does, has its removed voxels read as unset
//! rather than make thousands of lookups on the main thread. Light can then
//! be wrong about them only for a moment: such a chunk regenerates, and its
//! chunk task works out its inputs and solves its light again.
//!
//! An edit is worked out against what light held before the commit, at the
//! stamp after it less the number of voxels the commit changed. A commit that
//! replaced a chunk's edits, a clear, a forget or an applied full delta,
//! can't be counted back that way, so light treats that chunk as one it holds
//! nothing for, and the chunk's next chunk task works out its inputs. That
//! chunk regenerates, and a chunk task that regenerates always solves its
//! light rather than keep the light the world view holds, since its voxels
//! can change while its edit stamp doesn't: see
//! [`LightSettings::task_for`](super::LightSettings::task_for).

use std::sync::atomic::Ordering;

use bevy::platform::collections::{HashMap, HashSet};
use bevy::platform::time::Instant;
use bevy::prelude::*;

use super::fill::{cell_bit, open_cells, CELL};
use super::hinted::air_inputs;
use super::inputs::LightInputs;
use super::probe::SkyProbes;
use super::relight::LightWork;
use super::sky::{walkable, Columns, Hinted, SkyRule};
use super::solver::ChunkLight;
use super::view::{ViewMaps, ENTRY};
use super::LightSettings;
use crate::chunk::{chunk_containing, ChunkKey, CHUNK_SIZE_I, CHUNK_SIZE_U};
use crate::chunk_lifecycle::ChunkRequests;
use crate::configuration::VoxelWorldConfig;
use crate::voxel::WorldVoxel;
use crate::voxel_edits::VoxelEdits;

const CHUNK: usize = CHUNK_SIZE_U as usize;

/// How many chunk positions below an edit a sky ceiling's window is followed
/// down, looking for loaded chunks whose direct sky changed.
const CEILING_DEPTH: i32 = 64;

/// How far from an edit, in voxels, chunks relight for sky fill.
const FILL_RUN_REACH: i32 = 8;

/// A chunk losing at most this many edits in a commit has the voxels left
/// in their place looked up; past it, they read as unset.
///
/// The lookups run on the main thread, one per voxel, and a clear or forget
/// of a whole chunk would make thousands of them.
const LOOKED_UP_REMOVALS: usize = 64;

/// The six steps to a voxel's face neighbours.
const STEPS: [IVec3; 6] = [
    IVec3::X,
    IVec3::NEG_X,
    IVec3::Y,
    IVec3::NEG_Y,
    IVec3::Z,
    IVec3::NEG_Z,
];

type View = ViewMaps;

/// Takes this frame's committed voxel edits into the light of world `C`, and
/// triggers the relights they call for.
#[allow(clippy::too_many_arguments)]
pub(crate) fn relight_edits<C: VoxelWorldConfig>(
    requests: Res<ChunkRequests<C>>,
    mut work: ResMut<LightWork<C>>,
    settings: Res<LightSettings<C>>,
    edits: Res<VoxelEdits<C, C::MaterialIndex>>,
    mut probes: Option<ResMut<SkyProbes<C>>>,
    configuration: Res<C>,
) {
    if requests.edited.is_empty() {
        return;
    }
    #[cfg(feature = "trace")]
    let _span = bevy::log::info_span!("relight_edits").entered();
    let edited: Vec<IVec3> = requests
        .edited
        .iter()
        .flat_map(|set| set.voxels.iter().map(|(position, _)| *position))
        .collect();
    // How far each edited chunk's edit stamp moved in this commit, or `None`
    // where the commit replaced the chunk's edits, so the stamp before it
    // can't be told and light takes the path for a chunk it has no inputs
    // for.
    let mut moved: HashMap<IVec3, Option<u64>> = HashMap::new();
    for set in &requests.edited {
        let n = moved.entry(set.chunk).or_insert(Some(0));
        *n = n
            .filter(|_| !set.replaced)
            .map(|n| n + set.voxels.len() as u64);
    }
    let sequence_before =
        |position: IVec3| moved[&position].map(|n| edits.edit_stamp(position) - n);
    let started = Instant::now();
    settings.edit_frame.fetch_add(1, Ordering::Relaxed);
    settings.walks.lock().unwrap().clear();
    let quiet = work.is_settled();
    let reader = edits.reader();
    // A removed edit leaves the voxel the world generates there, which only
    // the world's voxel lookup knows. It is looked up for a chunk that lost
    // at most `LOOKED_UP_REMOVALS` edits.
    let mut generated = HashMap::new();
    for set in &requests.edited {
        let removed: Vec<_> = set
            .voxels
            .iter()
            .filter(|(p, voxel)| voxel.is_none() && reader.get(*p).is_none())
            .take(LOOKED_UP_REMOVALS + 1)
            .collect();
        if removed.is_empty() || removed.len() > LOOKED_UP_REMOVALS {
            continue;
        }
        let mut lookup = (configuration.voxel_lookup_delegate())(set.chunk, 0, None);
        for (p, _) in removed {
            generated.insert(*p, lookup(*p, None));
        }
    }
    let voxel = |p: IVec3| {
        reader
            .get(p)
            .or_else(|| generated.get(&p).copied())
            .unwrap_or(WorldVoxel::Unset)
    };
    let emission = &settings.task.emission;
    let level_of = |voxel: WorldVoxel<C::MaterialIndex>| match voxel {
        WorldVoxel::Solid(material) => emission
            .iter()
            .find(|(emissive, _)| *emissive == material)
            .map(|(_, level)| *level),
        _ => None,
    };
    let reach = settings
        .task
        .sky_scale
        .as_ref()
        .map_or(0, |scale| scale.reach())
        .max(settings.task.voxel_scale.reach()) as i32;

    let mut by_chunk: HashMap<IVec3, Vec<IVec3>> = HashMap::new();
    for p in &edited {
        by_chunk.entry(chunk_containing(*p)).or_default().push(*p);
    }
    let mut relight: HashSet<IVec3> = HashSet::new();
    let mut view = settings.view.write();

    // Which edits touch light, read before any input changes.
    let fill = settings.task.fill.is_some();
    for p in &edited {
        if quiet {
            if touches_light(&view, *p, voxel(*p).is_solid()) {
                relight.insert(chunk_containing(*p));
            }
        } else {
            relight.extend(within_reach(*p, reach));
        }
        // Sky fill's seeds run up to 7 voxels from their cell, so a change
        // can move them in the chunks within 8 voxels.
        if fill {
            relight.extend(within_reach(*p, FILL_RUN_REACH));
        }
    }

    // A solid placed in the padding of a chunk of air makes it take sky fill.
    if fill {
        for p in edited.iter().filter(|p| voxel(**p).is_solid()) {
            let own = chunk_containing(*p);
            for position in within_reach(*p, 1).filter(|position| *position != own) {
                let Some(chunk) = view.get_mut(&position) else {
                    continue;
                };
                let inputs = &chunk.light.inputs;
                if inputs.ring
                    || !matches!(inputs.occupancy, super::inputs::Occupancy::Open)
                {
                    continue;
                }
                let inputs = LightInputs {
                    ring: true,
                    ..inputs.with_columns(inputs.columns.clone())
                };
                chunk.light = std::sync::Arc::new(ChunkLight {
                    inputs: std::sync::Arc::new(inputs),
                    sky: chunk.light.sky.clone(),
                    voxel: chunk.light.voxel.clone(),
                    precision: chunk.light.precision,
                });
                relight.insert(position);
            }
        }
    }

    // The edited chunks' inputs, and the columns whose solids changed.
    let mut current: HashSet<IVec3> = HashSet::new();
    let mut columns: Vec<(IVec3, usize, usize)> = Vec::new();
    let mut old_occupancy = HashMap::new();
    for (position, voxels) in &by_chunk {
        let count = edits.edit_stamp(*position);
        let before = sequence_before(*position);
        if view.is_air(*position) {
            // Hinted air's voxels are its hint with the edits over them.
            let old = view.air[position].inputs.clone();
            let air = air_inputs(*position, &edits, emission, old.columns.clone());
            for p in voxels {
                let local = local(*p);
                if old.occupancy.is_solid(local.x, local.y, local.z)
                    != voxel(*p).is_solid()
                {
                    columns.push((*position, local.x, local.z));
                }
            }
            old_occupancy.insert(*position, old.occupancy.solid_columns());
            view.air.insert(*position, air);
            relight.insert(*position);
            current.insert(*position);
            work.fill_touch(*position);
            continue;
        }
        let chunk = view
            .get_mut(position)
            .filter(|chunk| Some(chunk.edits) == before);
        let Some(chunk) = chunk else {
            // The chunk's next chunk task works out its inputs.
            for p in voxels {
                let local = local(*p);
                columns.push((*position, local.x, local.z));
            }
            continue;
        };
        let old = &chunk.light.inputs;
        let mut occupancy = old.occupancy.clone();
        let mut emissive = old.emissive.clone();
        for p in voxels {
            let local = local(*p);
            let new = voxel(*p);
            if occupancy.is_solid(local.x, local.y, local.z) != new.is_solid() {
                occupancy.set(local.x, local.y, local.z, new.is_solid());
                columns.push((*position, local.x, local.z));
            }
            let at = |emitter: &super::inputs::EmissiveVoxel| {
                (emitter.x, emitter.y, emitter.z)
                    == (local.x as u8, local.y as u8, local.z as u8)
            };
            let was = emissive.iter().position(at);
            let now = level_of(new);
            if was.map(|index| emissive[index].level) != now {
                if let Some(index) = was {
                    emissive.swap_remove(index);
                }
                if let Some(level) = now {
                    emissive.push(super::inputs::EmissiveVoxel {
                        x: local.x as u8,
                        y: local.y as u8,
                        z: local.z as u8,
                        level,
                    });
                }
                relight.insert(*position);
            }
        }
        occupancy.normalize();
        old_occupancy.insert(*position, old.occupancy.solid_columns());
        let inputs = LightInputs {
            occupancy,
            emissive,
            columns: old.columns.clone(),
            ring: old.ring,
        };
        chunk.light = std::sync::Arc::new(ChunkLight {
            inputs: std::sync::Arc::new(inputs),
            sky: chunk.light.sky.clone(),
            voxel: chunk.light.voxel.clone(),
            precision: chunk.light.precision,
        });
        chunk.edits = count;
        current.insert(*position);
        work.hinted_touch(*position);
    }

    // The sky record, and the direct sky of the chunks below.
    if let (Some(rule), Some(probes)) = (settings.sky_rule, probes.as_mut()) {
        columns.sort_by_key(|(position, x, z)| (position.to_array(), *x, *z));
        columns.dedup();
        let mut record = settings.record.lock().unwrap();
        let mut masks: HashMap<IVec3, Option<Columns>> = HashMap::new();
        let mut flips = Vec::new();
        for (position, x, z) in &columns {
            let mask = masks.entry(*position).or_insert_with(|| {
                if let Some(old) = old_occupancy.get(position) {
                    return Some(*old);
                }
                sequence_before(*position)
                    .and_then(|before| record.get(*position, before))
            });
            let placed = by_chunk[position].iter().any(|p| {
                let local = local(*p);
                (local.x, local.z) == (*x, *z) && voxel(*p).is_solid()
            });
            // Whether the column holds a solid now, or `None` if nothing
            // tells: at an unloaded position the record doesn't hold, its
            // voxels are left for a sky probe to look up.
            let blocked = if current.contains(position) {
                Some(column_blocked_in(view.inputs(*position).unwrap(), *x, *z))
            } else if placed {
                Some(true)
            } else if mask.is_some() {
                Some(column_looked_up(
                    configuration.as_ref(),
                    &reader,
                    *position,
                    *x,
                    *z,
                ))
            } else {
                None
            };
            let was = mask.map(|mask| mask[*x] >> *z & 1 == 1);
            if let (Some(mask), Some(blocked)) = (mask.as_mut(), blocked) {
                if blocked {
                    mask[*x] |= 1 << *z;
                } else {
                    mask[*x] &= !(1 << *z);
                }
            }
            if was.is_none() || was != blocked {
                flips.push((*position, *x, *z, blocked == Some(true)));
            }
        }
        for (position, mask) in &masks {
            let count = edits.edit_stamp(*position);
            if current.contains(position) {
                let occupancy = &view.inputs(*position).unwrap().occupancy;
                let open = record.keeps_cells().then(|| open_cells(occupancy));
                record.insert(*position, &occupancy.solid_columns(), open, count);
            } else if let Some(mask) = mask {
                // A placed solid closes its cell. A dug voxel may open its
                // cell, which only reading the rest of the cell can tell, so
                // its cells are left unknown.
                let voxels = &by_chunk[position];
                let cells = sequence_before(*position)
                    .and_then(|before| record.cells(*position, before));
                let open = cells.and_then(|cells| {
                    voxels.iter().try_fold(cells.open, |open, p| {
                        let local = local(*p);
                        voxel(*p).is_solid().then(|| {
                            open & !cell_bit(
                                local.x / CELL,
                                local.y / CELL,
                                local.z / CELL,
                            )
                        })
                    })
                });
                record.insert(*position, mask, open, count);
            }
        }
        drop(record);

        let status = |view: &View, above: IVec3, x: usize, z: usize| -> Option<bool> {
            let count = edits.edit_stamp(above);
            if let Some(chunk) = view.get(&above).filter(|chunk| chunk.edits == count) {
                return Some(column_blocked_in(&chunk.light.inputs, x, z));
            }
            if let Some(air) = view.air.get(&above).filter(|air| air.edits == count) {
                return Some(column_blocked_in(&air.inputs, x, z));
            }
            if let Some(mask) = settings.record.lock().unwrap().get(above, count) {
                return Some(mask[x] >> z & 1 == 1);
            }
            let hint = configuration.chunk_fill_hint(ChunkKey::new(above, 0));
            match Hinted::of(hint, edits.stamp_has_edits(count)) {
                Hinted::Air => Some(false),
                Hinted::Solid => Some(true),
                Hinted::Unknown => None,
            }
        };
        for (position, x, z, blocked) in flips {
            let depth = match rule {
                SkyRule::Clearance(chunks) => chunks as i32,
                SkyRule::Ceiling(top) if position.y <= top => CEILING_DEPTH,
                SkyRule::Ceiling(_) => 0,
            };
            for k in 1..=depth {
                let below = position - IVec3::Y * k;
                // A solid between the edit and this chunk decides its column
                // whatever the edit did.
                if k > 1 && status(&view, below + IVec3::Y, x, z) == Some(true) {
                    break;
                }
                let (entity, inputs) = match view.get(&below) {
                    Some(chunk) => (chunk.entity, &chunk.light.inputs),
                    None => match view.air.get(&below) {
                        Some(air) => (ENTRY, &air.inputs),
                        None => continue,
                    },
                };
                let answer = &inputs.columns;
                let has_sky = answer.sky[x] >> z & 1 == 1;
                let new = if blocked {
                    if !has_sky {
                        continue;
                    }
                    probes.column_blocked(below, x, z).unwrap_or_else(|| {
                        let mut answer = answer.clone();
                        answer.sky[x] &= !(1 << z);
                        answer.provisional[x] &= !(1 << z);
                        answer
                    })
                } else {
                    if has_sky {
                        continue;
                    }
                    let window = rule.window(below.y);
                    let reachable = walkable(window);
                    let found: Vec<Option<bool>> = (1..=reachable as i32)
                        .map(|y| status(&view, below + IVec3::Y * y, x, z))
                        .collect();
                    if found.contains(&Some(true)) {
                        continue;
                    }
                    let known = found.iter().map(Option::is_some).collect();
                    let answer = answer.clone();
                    probes.column_opened(
                        below,
                        entity,
                        (x, z),
                        &answer,
                        known,
                        reachable == window,
                    )
                };
                match view.get_mut(&below) {
                    Some(chunk) => {
                        let light = &chunk.light;
                        chunk.light = std::sync::Arc::new(ChunkLight {
                            inputs: std::sync::Arc::new(light.inputs.with_columns(new)),
                            sky: light.sky.clone(),
                            voxel: light.voxel.clone(),
                            precision: light.precision,
                        });
                    }
                    None => {
                        let air = view.air.get_mut(&below).unwrap();
                        air.inputs = std::sync::Arc::new(air.inputs.with_columns(new));
                    }
                }
                work.fill_touch(below);
                if quiet {
                    relight.insert(below);
                } else {
                    relight.extend(within_reach_of_column(below, x, z, reach));
                }
            }
        }
    }
    drop(view);

    // Hinted air holding no light entry passes its relight on.
    for position in relight {
        if by_chunk.contains_key(&position) {
            work.trigger_edited(position);
        } else {
            work.trigger_urgent(position);
        }
    }
    work.add_edit_time(started);
}

/// The voxel at `p`'s place in its chunk.
fn local(p: IVec3) -> UVec3Usize {
    let local = p.rem_euclid(IVec3::splat(CHUNK_SIZE_I));
    UVec3Usize {
        x: local.x as usize,
        y: local.y as usize,
        z: local.z as usize,
    }
}

#[derive(Clone, Copy)]
struct UVec3Usize {
    x: usize,
    y: usize,
    z: usize,
}

/// Whether making the voxel at `p` solid, or open, as `solid` says, could
/// change light: a solid placed where light is, or a voxel dug beside light.
/// Reads the light the world view holds.
fn touches_light(view: &View, p: IVec3, solid: bool) -> bool {
    let was_open = open_light(view, p);
    if solid {
        return was_open == Some(true);
    }
    was_open.is_none()
        && STEPS.iter().any(|step| {
            let beside = p + *step;
            open_light(view, beside).unwrap_or_else(|| emits(view, beside))
        })
}

/// Whether the open voxel at `p` holds light in either channel, or `None` if
/// it is solid or its chunk isn't in the view. An open voxel of hinted air
/// holding no light entry may hold light, which nothing keeps.
fn open_light(view: &View, p: IVec3) -> Option<bool> {
    let local = local(p);
    let Some(chunk) = view.get(&chunk_containing(p)) else {
        let air = view.air.get(&chunk_containing(p))?;
        return (!air.inputs.occupancy.is_solid(local.x, local.y, local.z))
            .then_some(true);
    };
    let light = &chunk.light;
    if light.inputs.occupancy.is_solid(local.x, local.y, local.z) {
        return None;
    }
    Some(
        light.sky.at(local.x, local.y, local.z) > 0
            || light.voxel.at(local.x, local.y, local.z) > 0,
    )
}

/// Whether the voxel at `p` is emissive.
fn emits(view: &View, p: IVec3) -> bool {
    let Some(inputs) = view.inputs(chunk_containing(p)) else {
        return false;
    };
    let local = local(p);
    inputs.emissive.iter().any(|emitter| {
        (emitter.x as usize, emitter.y as usize, emitter.z as usize)
            == (local.x, local.y, local.z)
    })
}

/// Whether the column at `x`, `z` of the chunk with `inputs` holds a solid.
fn column_blocked_in(inputs: &LightInputs, x: usize, z: usize) -> bool {
    (0..CHUNK).any(|y| inputs.occupancy.is_solid(x, y, z))
}

/// Whether the column at `x`, `z` of the chunk position `position` holds a
/// solid, looked up through the world's lookups and voxel edits: at most 32
/// lookups.
fn column_looked_up<C: VoxelWorldConfig>(
    world: &C,
    edits: &crate::voxel_edits::VoxelEditReader<C::MaterialIndex>,
    position: IVec3,
    x: usize,
    z: usize,
) -> bool {
    let mut lookup = (world.voxel_lookup_delegate())(position, 0, None);
    let min = position * CHUNK_SIZE_I;
    (0..CHUNK_SIZE_I).any(|y| {
        let at = min + IVec3::new(x as i32, y, z as i32);
        edits.get(at).unwrap_or_else(|| lookup(at, None)).is_solid()
    })
}

/// The chunk positions whose light a change at the voxel `p` can reach: those
/// whose box, `reach` past each face, holds it.
fn within_reach(p: IVec3, reach: i32) -> impl Iterator<Item = IVec3> {
    let chunk = chunk_containing(p);
    let local = p - chunk * CHUNK_SIZE_I;
    let range = move |axis: usize| {
        let low = if local[axis] < reach { -1 } else { 0 };
        let high = if local[axis] >= CHUNK_SIZE_I - reach {
            1
        } else {
            0
        };
        low..=high
    };
    range(2).flat_map(move |z| {
        range(1).flat_map(move |y| range(0).map(move |x| chunk + IVec3::new(x, y, z)))
    })
}

/// The chunk positions whose light a change of direct sky in the column at
/// `x`, `z` of the chunk at `position` can reach.
fn within_reach_of_column(
    position: IVec3,
    x: usize,
    z: usize,
    reach: i32,
) -> impl Iterator<Item = IVec3> {
    let column = position * CHUNK_SIZE_I + IVec3::new(x as i32, 0, z as i32);
    (-1..=1).flat_map(move |y| {
        within_reach(column, reach)
            .filter(move |chunk| chunk.y == position.y)
            .map(move |chunk| chunk + IVec3::Y * y)
    })
}
