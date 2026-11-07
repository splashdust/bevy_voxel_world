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

use std::sync::atomic::Ordering;
use std::time::Instant;

use bevy::platform::collections::{HashMap, HashSet};
use bevy::prelude::*;

use super::fill::{cell_bit, open_cells, CELL};
use super::inputs::LightInputs;
use super::probe::SkyProbes;
use super::relight::LightWork;
use super::sky::{walkable, Columns, Hinted, SkyRule};
use super::solver::ChunkLight;
use super::view::ViewChunk;
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

/// The six steps to a voxel's face neighbours.
const STEPS: [IVec3; 6] = [
    IVec3::X,
    IVec3::NEG_X,
    IVec3::Y,
    IVec3::NEG_Y,
    IVec3::Z,
    IVec3::NEG_Z,
];

type View = HashMap<IVec3, ViewChunk>;

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
    let started = Instant::now();
    settings.edit_frame.fetch_add(1, Ordering::Relaxed);
    let quiet = work.is_settled();
    let reader = edits.reader();
    let voxel = |p: IVec3| reader.get(p).unwrap_or(WorldVoxel::Unset);
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
    for p in &requests.edited {
        by_chunk.entry(chunk_containing(*p)).or_default().push(*p);
    }
    let mut relight: HashSet<IVec3> = HashSet::new();
    let mut view = settings.view.write();

    // Which edits touch light, read before any input changes.
    let fill = settings.task.fill.is_some();
    for p in &requests.edited {
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
        for p in requests.edited.iter().filter(|p| voxel(**p).is_solid()) {
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
        let count = edits.edit_count(*position);
        let before = count - voxels.len() as u64;
        let chunk = view.get_mut(position).filter(|chunk| chunk.edits == before);
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
                let before =
                    edits.edit_count(*position) - by_chunk[position].len() as u64;
                record.get(*position, before)
            });
            let placed = by_chunk[position].iter().any(|p| {
                let local = local(*p);
                (local.x, local.z) == (*x, *z) && voxel(*p).is_solid()
            });
            // Whether the column holds a solid now, or `None` if nothing
            // tells: at an unloaded position the record doesn't hold, its
            // voxels are left for a sky probe to look up.
            let blocked = if current.contains(position) {
                Some(column_blocked_in(&view[position].light.inputs, *x, *z))
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
            let count = edits.edit_count(*position);
            if current.contains(position) {
                let occupancy = &view[position].light.inputs.occupancy;
                let open = record.keeps_cells().then(|| open_cells(occupancy));
                record.insert(*position, &occupancy.solid_columns(), open, count);
            } else if let Some(mask) = mask {
                // A placed solid closes its cell. A dug voxel may open its
                // cell, which only reading the rest of the cell can tell, so
                // its cells are left unknown.
                let voxels = &by_chunk[position];
                let before = count - voxels.len() as u64;
                let open = record.cells(*position, before).and_then(|cells| {
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
            let count = edits.edit_count(above);
            if let Some(chunk) = view.get(&above).filter(|chunk| chunk.edits == count) {
                return Some(column_blocked_in(&chunk.light.inputs, x, z));
            }
            if let Some(mask) = settings.record.lock().unwrap().get(above, count) {
                return Some(mask[x] >> z & 1 == 1);
            }
            let hint = configuration.chunk_fill_hint(ChunkKey::new(above, 0));
            match Hinted::of(hint, count > 0) {
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
                let Some(chunk) = view.get(&below) else {
                    continue;
                };
                let entity = chunk.entity;
                let answer = &chunk.light.inputs.columns;
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
                let chunk = view.get_mut(&below).unwrap();
                let light = &chunk.light;
                chunk.light = std::sync::Arc::new(ChunkLight {
                    inputs: std::sync::Arc::new(light.inputs.with_columns(new)),
                    sky: light.sky.clone(),
                    voxel: light.voxel.clone(),
                    precision: light.precision,
                });
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

    for position in relight {
        if work.has_light(position) {
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
/// it is solid or its chunk isn't in the view.
fn open_light(view: &View, p: IVec3) -> Option<bool> {
    let chunk = view.get(&chunk_containing(p))?;
    let local = local(p);
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
    let Some(chunk) = view.get(&chunk_containing(p)) else {
        return false;
    };
    let local = local(p);
    chunk.light.inputs.emissive.iter().any(|emitter| {
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
