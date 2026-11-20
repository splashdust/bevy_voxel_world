//! Light of sub-grids
//!
//! This module gives each sub-grid light of its own: the light its emissive
//! voxels give its chunks. It is compiled only with the `sub_grids` feature.
//!
//! The voxels of a sub-grid never enter the light of the world. They change
//! neither the occupancy of the world nor its sky record, and spread no light
//! into it. Instead each sub-grid chunk holds light of its own, which its
//! faces show. The chunks of a sub-grid join the [world view](super::view)
//! under the keys of their grid, beside the chunks of the world. The
//! [solver](super::solver) of the world solves them chunk by chunk, with the
//! voxel light reach, light precision and `LightShape` of the world. No world
//! solve reads them, since a world solve finds the chunks around it by the
//! keys of the world.
//!
//! The voxel light of a sub-grid chunk has light colours, as the voxel light
//! of a world chunk does. Its solve runs once per light colour in its light
//! box, and mixes the colours whose light reaches the chunk. The faces of the
//! chunk and `grid(g).light_at` mix the light of the world with that light by
//! the same rule. The enclosure of a sub-grid is sky light, which has no
//! colour.
//!
//! # Inputs from the commit
//!
//! A commit writes the occupancy and emissive voxels of two kinds of sub-grid
//! chunk into the world view, straight from the voxel edits of the sub-grid:
//! each chunk whose voxels it changed, and each chunk it spawns. So every
//! chunk of a sub-grid solves with all the chunks of its grid around it
//! present, including one spawned by the same commit.
//!
//! The first solve of a new chunk is a relight the commit starts, ahead of the
//! pool budget, rather than part of its chunk task. Its chunk task keeps the
//! light the world view holds, and the light of the relight waits for the mesh
//! of the chunk. So [`light_is_settled`](crate::prelude::VoxelWorld::light_is_settled)
//! waits for it.
//!
//! A solve of a sub-grid chunk reads a chunk position of its grid that holds
//! no chunk as air: light passes through it and nothing starts there. Under
//! [`SubGridSky::Enclosed`], the sky of the sub-grid is its enclosure. Sky
//! spreads, with the sky light reach, light precision and `LightShape` of the
//! world, from each open voxel that has a line along one of the axes of the
//! sub-grid that leaves the sub-grid without meeting a solid: see
//! [`enclosure`]. Under [`SubGridSky::World`], every open voxel has direct
//! sky, so the sky of the sub-grid is the top level everywhere.
//!
//! # What relights
//!
//! A voxel edit relights every chunk of the sub-grid whose solve box holds a
//! voxel the edit changed. The solve box is the chunk and the reach of the
//! world past each face. Under `Enclosed` the reach is the longer of the sky
//! light reach and the voxel light reach, and under `World` it is the voxel
//! light reach alone. Under `Enclosed` the edit also relights every chunk
//! whose solve box holds a voxel whose line test it flipped, however far from
//! the edit. A solve reads only its box and the line record, so the light of
//! no other chunk can change, and these relights relight nothing further when
//! they land. A commit that leaves a sub-grid with no solid voxel, as a stamp
//! does, relights none of its chunks, which all retire with it.
//!
//! These changes also relight sub-grid chunks:
//!
//! - Switching `sub_grid_sky` relights every sub-grid chunk with an open
//!   voxel, and no world chunk.
//! - A settings relight relights the chunks of sub-grids too, nearest a camera
//!   first, each placed by the last pose of its sub-grid.
//! - A change to the light colour of an emissive material relights, in the
//!   same way, only the sub-grid chunks whose solve box holds an emissive
//!   voxel of that material.
//!
//! Nothing else solves the light of a sub-grid again. Moving or turning a
//! sub-grid, world edits and changes of the light of the world start nothing.
//! A sub-grid chunk that despawns drops its light, as every chunk of a
//! despawned sub-grid does, and so does a chunk that has become all air.
//!
//! # Cost
//!
//! The solve of a sub-grid chunk grows with reach and with the light colours
//! whose light reaches it, as the solve of a world chunk does. The cost of an
//! edit grows with the sub-grid chunks within reach of the voxels it changed,
//! and the cost of a burst of lifts grows with the chunks lifted. On the main
//! thread, an edit updates the light inputs of each sub-grid chunk it changed
//! from the voxels it changed, and their emissive voxels. It builds the inputs
//! from all the voxel edits of the chunk only for a chunk it spawned or whose
//! edits it replaced. Moving a sub-grid costs no solve.
//!
//! Enclosure adds to that, and `World` skips all of it:
//!
//! - The sky light of a solve grows with the sky light reach, and ends at once
//!   for a chunk whose open voxels all have direct sky. Finding the voxels with
//!   direct sky grows with the voxels of the 3×3×3 chunks around the chunk,
//!   and with the extent of the sub-grid in chunks.
//! - An edit also follows the three lines through each edited voxel across the
//!   extent of the sub-grid. It relights the chunks around the voxels whose
//!   line test flipped.
//! - The memory of the line record grows with the chunks of the sub-grid.
//! - Switching to `Enclosed` builds the line record of every sub-grid, which
//!   grows with their voxel edits.

use std::marker::PhantomData;
use std::sync::atomic::Ordering;
use std::sync::Arc;

use bevy::platform::collections::{HashMap, HashSet};
use bevy::platform::time::Instant;
use bevy::prelude::*;

pub(crate) mod enclosure;
pub(crate) mod handover;
pub(crate) mod line_record;
pub(crate) mod tag_map;

use super::color::{unpack, ChunkColor, EmissionTable};
use super::config::SubGridSky;
use super::edit::{reach_of, within_reach, within_reach_of_box};
use super::faces::FaceHashes;
use super::inputs::{EmissiveVoxel, LightInputs, Occupancy};
use super::levels::LightLevels;
use super::relight::LightWork;
use super::sky::ColumnAnswer;
use super::solver::{ChunkLevels, ChunkLight};
use super::view::ViewChunk;
use super::LightSettings;
use crate::chunk::{chunk_containing, CHUNK_SIZE_I};
use crate::chunk_lifecycle::{ChunkIndex, ChunkRequests};
use crate::configuration::VoxelWorldConfig;
use crate::grid::{GridId, GridKey};
use crate::sub_grid::SubGrids;
use crate::voxel::WorldVoxel;
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

impl<C: VoxelWorldConfig> NewChunks<C> {
    /// Whether a chunk waits for [`name_new_chunks`] to name its entity.
    pub(crate) fn any(new: Res<Self>) -> bool {
        !new.keys.is_empty()
    }
}

/// Whether [`relight_edits`] has anything to do: a sub-grid in use, or a
/// line record of an ended sub-grid still to forget.
pub(crate) fn has_sub_grid_edits<C: VoxelWorldConfig>(
    sub_grids: Res<SubGrids<C>>,
    settings: Res<LightSettings<C>>,
) -> bool {
    sub_grids.in_use()
        || settings
            .view
            .read()
            .lines
            .as_ref()
            .is_some_and(|lines| !lines.is_empty())
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
    sub_grids: Res<SubGrids<C>>,
) {
    enclosure::forget_ended(&settings.view, &sub_grids);
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
    // A handover waits for the relights its commit starts.
    if requests.lands_as_handover() {
        work.capture_handover(Some(settings.edit_frame.load(Ordering::Relaxed)));
    }
    let task = &settings.task;
    let mut changes: HashMap<GridKey, ChunkChanges<C::MaterialIndex>> = HashMap::new();
    for set in sets() {
        let (voxels, moved) = changes
            .entry(GridKey::level_0(set.grid, set.chunk))
            .or_insert((Vec::new(), Some(0)));
        voxels.extend_from_slice(&set.voxels);
        *moved = moved
            .filter(|_| !set.replaced)
            .map(|n| n + set.voxels.len() as u64);
    }
    // Each changed chunk's new inputs: the view's updated by the voxels the
    // commit changed, where the view holds the inputs from just before it,
    // and otherwise built from the chunk's voxel edits, as for a chunk the
    // commit spawned or whose edits it replaced. Worked out under a read
    // lock, so solves reading the view don't wait for them.
    let stamps: Vec<(GridKey, u64)> = changed
        .iter()
        .map(|key| (*key, edits.edit_stamp_in(*key)))
        .collect();
    let mut inputs: Vec<(GridKey, u64, Option<LightInputs>)> = {
        let view = settings.view.read();
        stamps
            .into_iter()
            .map(|(key, stamp)| {
                let updated = changes.get(&key).and_then(|(voxels, moved)| {
                    let before = moved.and_then(|n| stamp.checked_sub(n))?;
                    let chunk = view.chunks.get(&key).filter(|c| c.edits == before)?;
                    Some(inputs_updated(
                        &chunk.light.inputs,
                        key.position(),
                        voxels,
                        &task.emission,
                    ))
                });
                (key, stamp, updated)
            })
            .collect()
    };
    for (key, _, inputs) in &mut inputs {
        if inputs.is_none() {
            let bucket = edits
                .buckets_in(key.grid(), std::iter::once(key.position()))
                .pop()
                .flatten();
            *inputs = Some(inputs_of(bucket, key.position(), &task.emission));
        }
    }
    let flips = {
        let mut view = settings.view.write();
        let flips = enclosure::follow_edits(&mut view, sets(), &edits, &sub_grids);
        for (key, stamp, inputs) in inputs {
            let inputs = Arc::new(inputs.unwrap());
            if let Some(chunk) = view.chunks.get_mut(&key) {
                chunk.light = Arc::new(ChunkLight {
                    inputs,
                    sky: chunk.light.sky.clone(),
                    voxel: chunk.light.voxel.clone(),
                    color: chunk.light.color.clone(),
                    borders: chunk.light.borders.clone(),
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
                    sky: ChunkLevels::Uniform(sky),
                    voxel: ChunkLevels::Uniform(0),
                    color: ChunkColor::White,
                    borders: Default::default(),
                    precision: task.precision,
                }),
                faces: FaceHashes::default(),
                version: work.next_version(),
                edits: stamp,
                emission: task.emission_generation,
                fill_seeds: None,
            };
            view.chunks.insert(key, chunk);
            new.keys.push(key);
            work.capture_new_chunk(key);
        }
        flips
    };
    // Sky reach counts only where enclosure spreads sky.
    let reach = match settings.config.sub_grid_sky {
        SubGridSky::Enclosed => reach_of(task),
        SubGridSky::World => task.voxel_scale.reach() as i32,
    };
    // A sub-grid the commit emptied, as a stamp empties the sub-grid it
    // stamps, has every chunk retiring with the commit, so none relights.
    let emptied = requests.emptied();
    let mut reached: HashSet<GridKey> = HashSet::new();
    for set in sets().filter(|set| !emptied.contains(&set.grid)) {
        for (p, _) in &set.voxels {
            reached.extend(
                within_reach(*p, reach)
                    .map(|position| GridKey::level_0(set.grid, position)),
            );
        }
    }
    // With sky light off, a flip changes no light.
    if task.sky_scale.is_some() {
        for (grid, flip) in flips.iter().filter(|(grid, _)| !emptied.contains(grid)) {
            let mut voxels = flip.voxels();
            let first = voxels.next().unwrap();
            let last = voxels.last().unwrap_or(first);
            reached.extend(
                within_reach_of_box(first.min(last), first.max(last), reach)
                    .map(|position| GridKey::level_0(*grid, position)),
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
    work.capture_handover(None);
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
                work.release_for_handover(key);
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
    emission: &EmissionTable<I>,
) -> LightInputs {
    let mut occupancy = Occupancy::Open;
    let mut emissive = Vec::new();
    let min = position * CHUNK_SIZE_I;
    for (at, voxel) in bucket.iter().flat_map(|bucket| bucket.iter()) {
        let WorldVoxel::Solid(material) = voxel else {
            continue;
        };
        let local = (at - min).as_uvec3();
        let (x, y, z) = (local.x as usize, local.y as usize, local.z as usize);
        occupancy.set(x, y, z, true);
        if let Some(emitter) = emission.get(material) {
            emissive.push(EmissiveVoxel::new(x as u8, y as u8, z as u8, emitter));
        }
    }
    occupancy.normalize();
    LightInputs {
        occupancy,
        emissive: emissive.into(),
        columns: ColumnAnswer::none(),
        ring: false,
    }
}

/// A voxel a commit changed: its position and its new edit, or `None` where
/// its edit was removed.
type VoxelChange<I> = (IVec3, Option<WorldVoxel<I>>);

/// The voxels a commit changed in one chunk, in commit order, and how far
/// the chunk's edit stamp moved, or `None` where the commit replaced its
/// edits, so the stamp before it can't be told.
type ChunkChanges<I> = (Vec<VoxelChange<I>>, Option<u64>);

/// `old`, the inputs of the sub-grid chunk at `position` before a commit,
/// with the voxels the commit changed there, `voxels`, applied in order. A
/// voxel whose new edit isn't solid is air. Its cost grows with `voxels`
/// and the chunk's emissive voxels, on top of copying its occupancy.
fn inputs_updated<I: Copy + PartialEq>(
    old: &LightInputs,
    position: IVec3,
    voxels: &[VoxelChange<I>],
    emission: &EmissionTable<I>,
) -> LightInputs {
    let mut occupancy = old.occupancy.clone();
    let min = position * CHUNK_SIZE_I;
    // The last new voxel of each changed position, by its place in the
    // chunk.
    let mut placed: HashMap<(u8, u8, u8), Option<I>> = HashMap::new();
    for (at, voxel) in voxels {
        let local = (*at - min).as_uvec3();
        let material = match voxel {
            Some(WorldVoxel::Solid(material)) => Some(*material),
            _ => None,
        };
        let (x, y, z) = (local.x as usize, local.y as usize, local.z as usize);
        occupancy.set(x, y, z, material.is_some());
        placed.insert((x as u8, y as u8, z as u8), material);
    }
    occupancy.normalize();
    let mut emissive: Vec<EmissiveVoxel> = old
        .emissive
        .iter()
        .filter(|emitter| !placed.contains_key(&emitter.at()))
        .copied()
        .collect();
    for ((x, y, z), material) in placed {
        if let Some(emitter) = material.and_then(|material| emission.get(material)) {
            emissive.push(EmissiveVoxel::new(x, y, z, emitter));
        }
    }
    LightInputs {
        occupancy,
        emissive: emissive.into(),
        columns: ColumnAnswer::none(),
        ring: false,
    }
}

impl<C: VoxelWorldConfig> LightSettings<C> {
    /// Returns the light of its own that the voxel at `position` of the
    /// sub-grid `grid` holds, from the light of its chunk in the world view, or
    /// `None` where the sub-grid has no chunk there.
    ///
    /// - A solid voxel holds no light, except an emissive voxel, which holds
    ///   its level and its light colour as voxel light.
    /// - An open voxel holds the levels its chunk holds for it, and its voxel
    ///   light has the light colour the chunk holds for it.
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
            let emitter = inputs
                .emissive
                .iter()
                .find(|emitter| emitter.at() == (x as u8, y as u8, z as u8));
            return Some(match emitter {
                Some(emitter) => LightLevels::new(0, emitter.level()).with_voxel_color(
                    unpack(self.task.emission.color(emitter.material())),
                ),
                None => LightLevels::new(0, 0),
            });
        }
        Some(light.open_levels_at(x, y, z))
    }
}
