//! A sub-grid's own light: the light its emissive voxels give its chunks.
//! Compiled only with the `sub_grids` feature.
//!
//! A sub-grid's voxels never enter the world's light. They change neither
//! the world's occupancy nor its sky record, and spread no light into it.
//! Instead each sub-grid chunk holds light of its own, which its faces show.
//! Its chunks join the [world view](super::view) under their grid's keys,
//! beside the world's chunks, and the world's [solver](super::solver)
//! solves them chunk by chunk, with the world's voxel reach, precision and
//! shape. No world solve reads them, since a world solve finds the chunks
//! around it by the world's keys.
//!
//! # Inputs from the commit
//!
//! A commit writes the occupancy and emissive voxels of each sub-grid chunk
//! whose voxels it changed, and of each sub-grid chunk it spawns, into the
//! world view straight from the sub-grid's voxel edits. So every chunk of a
//! sub-grid solves with all of its grid's chunks around it present, one
//! spawned by the same commit included. A new chunk's first solve is a
//! relight the commit starts, ahead of the pool budget, rather than part of
//! its chunk task: its chunk task keeps the light the world view holds, and
//! the relight's light waits for the chunk's mesh, so
//! [`light_is_settled`](crate::prelude::VoxelWorld::light_is_settled) waits
//! for it.
//!
//! A sub-grid's solve reads a chunk position of its grid that holds no chunk
//! as air: light passes through it and nothing starts there. Every open voxel
//! of a sub-grid chunk has direct sky, so its own sky is the top level
//! everywhere.
//!
//! # What relights
//!
//! A voxel edit relights every chunk of the sub-grid whose solve box, the
//! chunk and the world's reach past each face, holds a voxel the edit
//! changed. A solve reads only its box, so no other chunk's light can
//! change, and these relights relight nothing further when they land. A
//! settings relight relights the sub-grid's chunks too, nearest a camera
//! first, placed by each sub-grid's last pose. Nothing else solves a
//! sub-grid's light again: moving or turning a sub-grid, world edits and
//! changes of the world's light start nothing. A sub-grid chunk that
//! despawns, as every chunk of a despawned sub-grid and a chunk gone all air
//! do, drops its light.
//!
//! # Cost
//!
//! A sub-grid chunk's solve grows with reach, as a world chunk's does. An
//! edit's cost grows with the sub-grid chunks within reach of the voxels it
//! changed, and a burst of lifts with the chunks lifted. Moving a sub-grid
//! costs no solve.

use std::marker::PhantomData;
use std::sync::atomic::Ordering;
use std::sync::Arc;

use bevy::platform::collections::HashSet;
use bevy::platform::time::Instant;
use bevy::prelude::*;

use super::edit::{reach_of, within_reach};
use super::faces::FaceHashes;
use super::inputs::{EmissiveVoxel, LightInputs, Occupancy};
use super::levels::LightLevels;
use super::relight::LightWork;
use super::sky::ColumnAnswer;
use super::solver::{ChannelLight, ChunkLight};
use super::view::ViewChunk;
use super::LightSettings;
use crate::chunk::{chunk_containing, CHUNK_SIZE_I};
use crate::chunk_lifecycle::{ChunkIndex, ChunkRequests};
use crate::configuration::VoxelWorldConfig;
use crate::grid::{GridId, GridKey};
use crate::voxel_edits::{EditBucket, VoxelEdits};

/// The sub-grid chunks of world `C` whose inputs a commit wrote into the
/// world view before their chunk entities spawned, waiting for
/// [`name_new_chunks`] to name their entities.
#[derive(Resource)]
pub(crate) struct NewChunks<C> {
    keys: Vec<GridKey>,
    _world: PhantomData<fn() -> C>,
}

impl<C> Default for NewChunks<C> {
    fn default() -> Self {
        Self {
            keys: Vec::new(),
            _world: PhantomData,
        }
    }
}

/// Takes this frame's committed sub-grid voxel edits, and the sub-grid
/// chunks the commit spawns, into the world view, and triggers the relights
/// they call for. Runs after the world's edits are taken in.
///
/// A chunk the world view holds nothing for gets its entry with no entity,
/// which [`name_new_chunks`] fills in once the chunk has spawned.
pub(crate) fn relight_edits<C: VoxelWorldConfig>(
    requests: Res<ChunkRequests<C>>,
    mut work: ResMut<LightWork<C>>,
    settings: Res<LightSettings<C>>,
    edits: Res<VoxelEdits<C, C::MaterialIndex>>,
    mut new: ResMut<NewChunks<C>>,
) {
    let sets = || {
        requests
            .edited
            .iter()
            .filter(|set| set.grid != GridId::WORLD)
    };
    let spawned = requests
        .spawn
        .iter()
        .map(|request| request.key)
        .filter(|key| !key.is_world());
    let changed: HashSet<GridKey> = sets()
        .map(|set| GridKey::level_0(set.grid, set.chunk))
        .chain(spawned)
        .collect();
    if changed.is_empty() {
        return;
    }
    #[cfg(feature = "trace")]
    let _span = bevy::log::info_span!("relight_sub_grid_edits").entered();
    let started = Instant::now();
    settings.edit_frame.fetch_add(1, Ordering::Relaxed);
    let task = &settings.task;
    {
        let mut view = settings.view.write();
        for key in &changed {
            let bucket = edits
                .buckets_in(key.grid(), std::iter::once(key.position()))
                .pop()
                .flatten();
            let inputs = Arc::new(inputs_of(bucket, key.position(), &task.emission));
            let stamp = edits.edit_stamp_in(*key);
            if let Some(chunk) = view.chunks.get_mut(key) {
                chunk.light = Arc::new(ChunkLight {
                    inputs,
                    sky: chunk.light.sky.clone(),
                    voxel: chunk.light.voxel.clone(),
                    precision: chunk.light.precision,
                });
                chunk.edits = stamp;
                continue;
            }
            // Before its first solve lands, a chunk shows its own sky and no
            // voxel light.
            let sky = task
                .sky_scale
                .as_ref()
                .map_or(0, |scale| scale.precision().top());
            let chunk = ViewChunk {
                entity: Entity::PLACEHOLDER,
                light: Arc::new(ChunkLight {
                    inputs,
                    sky: ChannelLight::Uniform(sky),
                    voxel: ChannelLight::Uniform(0),
                    precision: task.precision,
                }),
                faces: FaceHashes::default(),
                version: work.next_version(),
                edits: stamp,
                emission: task.emission_generation,
                fill_seeds: None,
            };
            view.chunks.insert(*key, chunk);
            new.keys.push(*key);
        }
    }
    let reach = reach_of(task);
    let mut reached: HashSet<GridKey> = HashSet::new();
    for set in sets() {
        for (p, _) in &set.voxels {
            reached.extend(
                within_reach(*p, reach)
                    .map(|position| GridKey::level_0(set.grid, position)),
            );
        }
    }
    // A chunk still waiting for its entity relights once it has one.
    for key in reached {
        if changed.contains(&key) {
            work.trigger_edited(key);
        } else {
            work.trigger_urgent(key);
        }
    }
    work.add_edit_time(started);
}

/// Names the chunk entity of each sub-grid chunk whose inputs a commit wrote
/// before it spawned, and starts its first solve. Runs once the commit's
/// chunks have spawned, before relights start.
///
/// A chunk position with no chunk spawned, or whose chunk is Retiring, drops
/// its entry: a chunk spawned there later gets its inputs from the commit
/// that spawns it.
pub(crate) fn name_new_chunks<C: VoxelWorldConfig>(
    mut new: ResMut<NewChunks<C>>,
    mut work: ResMut<LightWork<C>>,
    settings: Res<LightSettings<C>>,
    index: Res<ChunkIndex<C>>,
) {
    if new.keys.is_empty() {
        return;
    }
    let mut view = settings.view.write();
    for key in new.keys.drain(..) {
        let unnamed = view
            .chunks
            .get(&key)
            .is_some_and(|chunk| chunk.entity == Entity::PLACEHOLDER);
        if !unnamed {
            continue;
        }
        match index.get_in(key).filter(|_| !index.is_retiring_in(key)) {
            Some(entity) => {
                view.chunks.get_mut(&key).unwrap().entity = entity;
                work.add_sub_grid_chunk(key, entity);
            }
            None => {
                view.chunks.remove(&key);
            }
        }
    }
}

/// What the sub-grid chunk at `position` holds for light, from `bucket`, its
/// voxel edits: every edit is a solid voxel, and every other voxel is air.
/// Its emissive voxels are found under the emissive list `emission`. It has
/// no column answer, since a sub-grid's sky never comes from one.
fn inputs_of<I: Copy + PartialEq>(
    bucket: Option<EditBucket<I>>,
    position: IVec3,
    emission: &[(I, u8)],
) -> LightInputs {
    let mut occupancy = Occupancy::Open;
    let mut emissive = Vec::new();
    let min = position * CHUNK_SIZE_I;
    for (at, voxel) in bucket.iter().flat_map(|bucket| bucket.iter()) {
        let crate::voxel::WorldVoxel::Solid(material) = voxel else {
            continue;
        };
        let local = (at - min).as_uvec3();
        let (x, y, z) = (local.x as usize, local.y as usize, local.z as usize);
        occupancy.set(x, y, z, true);
        let level = emission
            .iter()
            .find(|(emissive, _)| *emissive == material)
            .map(|(_, level)| *level);
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
    LightInputs {
        occupancy,
        emissive,
        columns: ColumnAnswer::none(),
        ring: false,
    }
}

impl<C: VoxelWorldConfig> LightSettings<C> {
    /// The light the voxel at `position` of the sub-grid `grid` holds of its
    /// own, from its chunk's light in the world view: a solid voxel holds no
    /// light, except an emissive one, which holds its level. `None` where the
    /// sub-grid has no chunk.
    pub fn own_light_at(&self, grid: GridId, position: IVec3) -> Option<LightLevels> {
        let chunk = chunk_containing(position);
        let light = self
            .view
            .read()
            .chunks
            .get(&GridKey::level_0(grid, chunk))?
            .light
            .clone();
        let local = (position - chunk * CHUNK_SIZE_I).as_uvec3();
        let (x, y, z) = (local.x as usize, local.y as usize, local.z as usize);
        let inputs = &light.inputs;
        if inputs.occupancy.is_solid(x, y, z) {
            let level = inputs
                .emissive
                .iter()
                .find(|emitter| {
                    (emitter.x, emitter.y, emitter.z) == (x as u8, y as u8, z as u8)
                })
                .map_or(0, |emitter| emitter.level);
            return Some(LightLevels::new(0, level));
        }
        Some(LightLevels::of_steps(
            light.sky.at(x, y, z),
            light.voxel.at(x, y, z),
            light.precision,
        ))
    }
}
