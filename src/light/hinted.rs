//! Hinted air: chunk positions whose fill hint says `Uniform(Air)`, which the
//! streaming policy doesn't spawn outside the protected radius.
//!
//! Light reads hinted air as what its hint says, not as a neighbour that
//! isn't loaded: every voxel open, with the voxel edits over it, and its own
//! direct sky, worked out like a chunk's from the fill hints and the sky
//! record above it. A column nothing has proved clear shows direct sky until a
//! sky probe disproves it, as in a chunk. So light crosses hinted air in a
//! solver's box, and hinting air only saves work. A `Uniform` solid hint
//! already reads right, as a neighbour that isn't loaded reads solid.
//!
//! The world view keeps the inputs of the hinted air around each loaded chunk
//! and light entry, the positions their solves read. A chunk task's start
//! adds those around its chunk, so its first solve reads them.
//!
//! A light entry holds the light of hinted air where a face can read it: it
//! is made where a loaded chunk's solid touches hinted air by a face, an
//! edge or a corner, and dropped once none does. It has no voxels, entity,
//! mesh or slot, but to light it's a chunk: it's solved in relight tasks,
//! takes arrivals, ripples and every other relight, and the chunks beside it
//! read its light through their shells. A chunk retiring in hinted air that a
//! loaded solid still touches leaves a light entry with its light, since its
//! voxels are the same, and a chunk spawning where a light entry is takes its
//! place.
//!
//! Hinted air holding no light entry has no light of its own to compare, so
//! relights that would stop there pass on through it instead: see
//! `LightWork::pass_on`. A relight passed on while a chunk's task runs misses
//! that chunk, so a chunk task's light landing beside such air relights its
//! chunk again if anything it read through the air changed meanwhile: see
//! [`changed_across_air`].

use std::sync::Arc;

use bevy::platform::collections::HashSet;
use bevy::prelude::*;

use super::config::LightPrecision;
use super::faces::{Face, FaceHashes};
use super::inputs::{EmissiveVoxel, LightInputs, Occupancy};
use super::probe::{air_answer, replace_columns, SkyProbes};
use super::record::SkyRecord;
use super::relight::LightWork;
use super::route::LightRoute;
use super::sky::{ColumnAnswer, SkyRule};
use super::solver::{neighbour_index, ChannelLight, ChunkLight, Neighbourhood};
use super::view::{AirInputs, ViewChunk, ViewMaps, ENTRY};
use super::LightSettings;
use crate::chunk::{ChunkKey, CHUNK_SIZE_I, CHUNK_SIZE_U};
use crate::configuration::{FillHint, VoxelWorldConfig};
use crate::voxel_edits::VoxelEdits;

const CHUNK: usize = CHUNK_SIZE_U as usize;

/// Whether the fill hint of the level-0 chunk position `position` says its
/// voxels are all air.
pub(crate) fn is_hinted_air<C: VoxelWorldConfig>(world: &C, position: IVec3) -> bool {
    matches!(
        world.chunk_fill_hint(ChunkKey::new(position, 0)),
        FillHint::Uniform(voxel) if !voxel.is_solid()
    )
}

/// The inputs of the hinted air at `position`: every voxel open, with the
/// voxel edits over it, under the emissive list `emission`, and the column
/// answer `columns`.
pub(crate) fn air_inputs<C: VoxelWorldConfig>(
    position: IVec3,
    edits: &VoxelEdits<C, C::MaterialIndex>,
    emission: &[(C::MaterialIndex, u8)],
    columns: ColumnAnswer,
) -> AirInputs {
    let count = edits.edit_sequence(position);
    let mut occupancy = Occupancy::Open;
    let mut emissive = Vec::new();
    if count > 0 {
        if let Some(bucket) = edits.buckets(std::iter::once(position)).pop().flatten() {
            let min = position * CHUNK_SIZE_I;
            for (at, voxel) in bucket.iter() {
                let local = (at - min).as_uvec3();
                let (x, y, z) = (local.x as usize, local.y as usize, local.z as usize);
                occupancy.set(x, y, z, voxel.is_solid());
                let level = match voxel {
                    crate::voxel::WorldVoxel::Solid(material) => emission
                        .iter()
                        .find(|(emissive, _)| *emissive == material)
                        .map(|(_, level)| *level),
                    _ => None,
                };
                if let Some(level) = level {
                    emissive.push(EmissiveVoxel {
                        x: x as u8,
                        y: y as u8,
                        z: z as u8,
                        level,
                    });
                }
            }
            occupancy.normalize();
        }
    }
    AirInputs {
        inputs: Arc::new(LightInputs {
            occupancy,
            emissive,
            columns,
            ring: false,
        }),
        edits: count,
    }
}

/// Whether a solid voxel of a chunk with `occupancy` touches, by a face, an
/// edge or a corner, a voxel of the chunk position `offset` from it, one of
/// its 26 neighbours.
pub(crate) fn solid_touches(occupancy: &Occupancy, offset: IVec3) -> bool {
    match occupancy {
        Occupancy::Open => false,
        Occupancy::Solid => true,
        Occupancy::Mixed(_) => {
            let range = |d: i32| match d {
                1 => CHUNK - 1..CHUNK,
                -1 => 0..1,
                _ => 0..CHUNK,
            };
            let z: u32 = match offset.z {
                1 => 1 << (CHUNK - 1),
                -1 => 1,
                _ => !0,
            };
            range(offset.y)
                .any(|y| range(offset.x).any(|x| occupancy.solid_column(x, y) & z != 0))
        }
    }
}

/// The 26 chunk positions around `position`.
pub(crate) fn around(position: IVec3) -> impl Iterator<Item = IVec3> {
    (-1..=1).flat_map(move |z| {
        (-1..=1).flat_map(move |y| {
            (-1..=1)
                .map(move |x| IVec3::new(x, y, z))
                .filter(|offset| *offset != IVec3::ZERO)
                .map(move |offset| position + offset)
        })
    })
}

/// Whether a loaded chunk's solid touches the chunk position `position`.
pub(crate) fn touched(view: &ViewMaps, position: IVec3) -> bool {
    around(position).any(|neighbour| {
        view.get(&neighbour).is_some_and(|chunk| {
            !chunk.is_entry()
                && solid_touches(&chunk.light.inputs.occupancy, position - neighbour)
        })
    })
}

/// Whether a loaded chunk or a light entry lies around `position`, so its
/// solve may read hinted air there.
fn read_around(view: &ViewMaps, position: IVec3) -> bool {
    around(position).any(|neighbour| view.contains_key(&neighbour))
}

/// The light hinted air with `inputs` is known to hold, with no light entry,
/// in steps of `precision`: direct sky, and nothing else.
pub(crate) fn known_light(
    inputs: &Arc<LightInputs>,
    precision: LightPrecision,
) -> ChunkLight {
    let sky = match inputs.direct_sky() {
        None => ChannelLight::Uniform(0),
        Some(_) if inputs.all_open_direct() => ChannelLight::Uniform(precision.top()),
        Some(direct) => {
            ChannelLight::Planes(vec![*direct; precision.bits()].into_boxed_slice())
        }
    };
    ChunkLight {
        inputs: inputs.clone(),
        sky,
        voxel: ChannelLight::Uniform(0),
        precision,
    }
}

/// Adds the inputs of the hinted air around `position` that the world view
/// doesn't hold yet, with column answers from the fill hints and `record`,
/// under `rule`, or none with sky light off. Returns where it added them.
pub(crate) fn add_air_around<C: VoxelWorldConfig>(
    view: &mut ViewMaps,
    position: IVec3,
    world: &C,
    edits: &VoxelEdits<C, C::MaterialIndex>,
    emission: &[(C::MaterialIndex, u8)],
    rule: Option<SkyRule>,
    mut record: Option<&mut SkyRecord>,
) -> Vec<IVec3> {
    let mut added = Vec::new();
    for neighbour in around(position) {
        if view.contains_key(&neighbour)
            || view.air.contains_key(&neighbour)
            || !is_hinted_air(world, neighbour)
        {
            continue;
        }
        let columns = match (rule, record.as_deref_mut()) {
            (Some(rule), Some(record)) => {
                air_answer(rule, neighbour, record, edits, world)
            }
            _ => ColumnAnswer::none(),
        };
        let air = air_inputs(neighbour, edits, emission, columns);
        view.air.insert(neighbour, air);
        added.push(neighbour);
    }
    added
}

/// Keeps the hinted air of world `C` and its light entries in line with the
/// loaded chunks: adds the hinted air around chunks that landed, makes and
/// drops light entries where loaded solids came to touch hinted air or stopped
/// touching it, forgets hinted air no solve reads any more, and lets the
/// world's sky probes settle the columns of hinted air added since the last
/// frame.
pub(crate) fn follow<C: VoxelWorldConfig>(
    mut work: ResMut<LightWork<C>>,
    settings: Res<LightSettings<C>>,
    mut probes: Option<ResMut<SkyProbes<C>>>,
    mut route: Option<ResMut<LightRoute<C>>>,
    world: Res<C>,
    edits: Res<VoxelEdits<C, C::MaterialIndex>>,
) {
    let mut added: Vec<IVec3> = std::mem::take(&mut *settings.new_air.lock().unwrap());
    let landed = work.take_hinted_touched();
    if added.is_empty() && landed.is_empty() {
        return;
    }
    #[cfg(feature = "trace")]
    let _span = bevy::log::info_span!("follow_hinted_air").entered();
    let emission = &settings.task.emission;
    let mut record = settings.sky_rule.map(|_| settings.record.lock().unwrap());
    let mut relight = Vec::new();
    let mut forgotten = Vec::new();
    {
        let mut view = settings.view.write();
        let view = &mut *view;
        let mut seen = HashSet::new();
        for position in landed {
            if !seen.insert(position) {
                continue;
            }
            // A chunk retiring in hinted air may leave a light entry, whose
            // solves read the air around it too.
            if view.contains_key(&position) {
                let new = add_air_around(
                    view,
                    position,
                    world.as_ref(),
                    &edits,
                    emission,
                    settings.sky_rule,
                    record.as_deref_mut(),
                );
                // Its last solve read the new air as a neighbour that isn't
                // loaded.
                if !new.is_empty() {
                    relight.push(position);
                }
                added.extend(new);
            }
            let mut dropped = Vec::new();
            for neighbour in around(position).chain(std::iter::once(position)) {
                if view.is_air(neighbour) && touched(view, neighbour) {
                    let air = view.air.remove(&neighbour).unwrap();
                    make_entry(
                        view,
                        &mut work,
                        neighbour,
                        air,
                        settings.task.emission_generation,
                        settings.task.precision,
                    );
                    added.extend(add_air_around(
                        view,
                        neighbour,
                        world.as_ref(),
                        &edits,
                        emission,
                        settings.sky_rule,
                        record.as_deref_mut(),
                    ));
                } else if view.get(&neighbour).is_some_and(ViewChunk::is_entry)
                    && !touched(view, neighbour)
                {
                    let entry = view.remove(&neighbour).unwrap();
                    view.air.insert(
                        neighbour,
                        AirInputs {
                            inputs: entry.light.inputs.clone(),
                            edits: entry.edits,
                        },
                    );
                    work.forget_entry(neighbour);
                    if let Some(route) = route.as_mut() {
                        route.entry_dropped(neighbour);
                    }
                    dropped.push(neighbour);
                }
            }
            let near = std::iter::once(position)
                .chain(dropped)
                .flat_map(|centre| around(centre).chain(std::iter::once(centre)))
                .collect::<HashSet<IVec3>>();
            for neighbour in near {
                if view.is_air(neighbour) && !read_around(view, neighbour) {
                    view.air.remove(&neighbour);
                    forgotten.push(neighbour);
                }
            }
        }
    }
    for position in forgotten {
        if let Some(probes) = probes.as_mut() {
            probes.forget(position, ENTRY);
        }
        work.fill_touch(position);
    }
    for position in added {
        // Air a chunk has landed in since is the chunk's to answer for.
        let current = {
            let view = settings.view.read();
            let hinted = view.is_air(position)
                || view.get(&position).is_some_and(ViewChunk::is_entry);
            hinted.then(|| view.inputs(position).cloned()).flatten()
        };
        let Some(inputs) = current else {
            continue;
        };
        if let (Some(probes), Some(record)) = (probes.as_mut(), record.as_mut()) {
            let answer = probes.answer_air(position, record, &edits, world.as_ref());
            if answer != inputs.columns {
                replace_columns(&settings.view, position, ENTRY, answer.clone());
                if answer.sky != inputs.columns.sky {
                    relight.push(position);
                }
            }
        }
        work.fill_touch(position);
    }
    drop(record);
    for position in relight {
        work.trigger(position);
    }
}

/// Makes a light entry of the hinted air `air` at `position`, and queues its
/// first solve. Until that lands, it holds direct sky where its column answer
/// lets sky light in, and no other light.
fn make_entry<C: VoxelWorldConfig>(
    view: &mut ViewMaps,
    work: &mut LightWork<C>,
    position: IVec3,
    air: AirInputs,
    emission: u64,
    precision: LightPrecision,
) {
    let inputs = Arc::new(LightInputs {
        ring: true,
        ..air.inputs.with_columns(air.inputs.columns.clone())
    });
    let light = Arc::new(known_light(&inputs, precision));
    let version = work.next_version();
    view.insert(
        position,
        ViewChunk {
            entity: ENTRY,
            light,
            faces: FaceHashes {
                sky: [0; 6],
                voxel: [0; 6],
            },
            version,
            edits: air.edits,
            emission,
            fill_seeds: None,
        },
    );
    work.add_entry(position);
}

/// Whether the inputs of a chunk, light entry or hinted air that the chunk at
/// `position` reaches through hinted air holding no light entry, within the
/// 3×3×3 around it, are other than those its solve `read`.
///
/// A face neighbour's change shows in the light versions a solve compares
/// against, but one beyond hinted air changes no face neighbour's light, and
/// a relight passed on through the air while the chunk's task ran found no
/// chunk there to relight.
pub(crate) fn changed_across_air(
    view: &ViewMaps,
    position: IVec3,
    read: &Neighbourhood,
) -> bool {
    let within = |at: IVec3| (at - position).abs().max_element() <= 1;
    let mut next: Vec<IVec3> = Face::ALL
        .iter()
        .map(|face| position + face.offset())
        .filter(|beside| view.is_air(*beside))
        .collect();
    let mut seen: HashSet<IVec3> = next.iter().copied().collect();
    seen.insert(position);
    let mut checked = next.clone();
    while let Some(air) = next.pop() {
        for face in Face::ALL {
            let beside = air + face.offset();
            if !within(beside) || !seen.insert(beside) {
                continue;
            }
            checked.push(beside);
            if view.is_air(beside) {
                next.push(beside);
            }
        }
    }
    checked.into_iter().any(|at| {
        let was = &read[neighbour_index(at - position)];
        match (view.inputs(at), was) {
            (Some(now), Some(was)) => !Arc::ptr_eq(now, was),
            (now, was) => now.is_some() != was.is_some(),
        }
    })
}

/// Whether the arriving light `light` of a chunk passes on into the hinted
/// air with `inputs` across `face`: its light reaches an open voxel there
/// whose neighbour isn't direct sky, or it carries voxel light there.
pub(crate) fn passes_into_air(
    light: &ChunkLight,
    inputs: &Arc<LightInputs>,
    face: Face,
) -> bool {
    let known = known_light(inputs, light.precision);
    let back = face.opposite();
    (0..CHUNK).any(|u| {
        (0..CHUNK).any(|v| {
            let (fx, fy, fz) = face.voxel(u, v);
            let (tx, ty, tz) = back.voxel(u, v);
            if light.inputs.occupancy.is_solid(fx, fy, fz)
                || inputs.occupancy.is_solid(tx, ty, tz)
            {
                return false;
            }
            light.voxel.at(fx, fy, fz) > 0
                || light.sky.at(fx, fy, fz) > 0
                    && known.sky.at(tx, ty, tz) < light.precision.top()
        })
    })
}
