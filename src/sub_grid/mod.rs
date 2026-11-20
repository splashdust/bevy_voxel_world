//! Sub-grids
//!
//! This module holds sub-grids: sets of voxels on grids of their own, each
//! belonging to one voxel world, that a game moves freely against the grid of
//! the world. Its parts spawn the chunks of each sub-grid from its commits,
//! keep the pose of each sub-grid, and land the handovers that move voxels
//! between grids.
//!
//! A sub-grid is an entity carrying [`SubGrid<C>`], made with
//! [`VoxelWorld::spawn_sub_grid`] or [`VoxelWorld::insert_sub_grid`], and
//! named by a [`Grid`]. [`VoxelWorld::grid`] and [`VoxelWorld::grid_mut`] read
//! and edit any grid with the calls the methods of the world itself use.
//!
//! A sub-grid has no generator, so every voxel it holds is a voxel edit, and
//! its edits are exactly its solid voxels. Setting air or `Unset` removes a
//! voxel, and a position with no solid reads air. The edits of a sub-grid
//! commit in the next `PreUpdate`, with those of the world.
//!
//! [`VoxelWorld::lift`](crate::prelude::VoxelWorld::lift) moves the solid
//! voxels a game picks out of any grid into a new sub-grid, which shows them
//! where they were, and leaves air behind. Lifting from a sub-grid cuts it in
//! two. [`Lifted`] says when the move has landed.
//!
//! [`VoxelWorld::stamp`](crate::prelude::VoxelWorld::stamp) writes the voxels
//! of a sub-grid into any other grid, snapped to the nearest pose that lines
//! up with it. It despawns the sub-grid once [`Stamped`] says the move has
//! landed. Together, lift and stamp move voxels every way: from the world into
//! a sub-grid, from a sub-grid into the world, from one sub-grid into a new
//! one, and from one sub-grid into another.
//!
//! # Chunks
//!
//! A sub-grid shows through chunks of its own. The chunk tasks of the world
//! build them, and the mesher, material and mesh cache of the world mesh them,
//! so identical voxels in two grids share one mesh. Each commit spawns and
//! retires them in place of the streaming policy. After the commit, the
//! sub-grid has a chunk at LOD level 0 at every chunk position whose padded
//! voxels hold one of its solid voxels, and no other. The voxels of such a
//! chunk are the voxel edits of the sub-grid over air, so the voxel lookup
//! delegate and fill hints of the world are never called for them. A sub-grid
//! with no solid voxel has no chunks, and lives on until it is despawned.
//!
//! The chunks of a sub-grid start their chunk tasks ahead of streamed chunks,
//! as edited chunks do. They count towards the
//! [`max_active_chunk_threads`](crate::prelude::VoxelWorldConfig::max_active_chunk_threads)
//! of the world together with its own chunks. The cost of a commit grows with
//! the chunks its edits reach, not with the size of the sub-grid. Moving the
//! camera never retires these chunks, and [`VoxelWorld::regenerate_chunks`]
//! leaves them alone.
//!
//! # Light
//!
//! In a world with light propagation, the voxels of a sub-grid stay out of
//! the light of the world: they block none of its light and spread none into
//! it. The emissive voxels of a sub-grid light the sub-grid itself, with the
//! voxel light reach, light precision and `LightShape` of the world.
//! `GridVoxels::light_at` reads that light on top of the light of the world
//! where the sub-grid is. The faces of the sub-grid show the same. The shader
//! reads the light of the world at the world position of each face, so a
//! sub-grid darkens as it enters a cave and brightens as it passes the
//! emissive voxels of the world. Only voxel edits to the sub-grid and changes
//! of light settings solve its light again, so moving it costs no solve.
//!
//! Each chunk is a child of the sub-grid entity, placed by its chunk key as
//! the chunks of the world are placed under the root of the world, so it
//! moves with the sub-grid. The crate copies no component from the sub-grid
//! to its chunks. `Visibility` inherits through the hierarchy. A game gives
//! the chunks `RenderLayers` or `NotShadowCaster` by putting a Bevy
//! `Propagate` of the component on the sub-grid. A sub-grid spawned under the
//! root of the world picks up a layer the root propagates, until the game
//! gives the sub-grid another parent.
//!
//! The meshing delegate and the texture index mapper get the position of a
//! sub-grid chunk on the grid of the sub-grid, and aren't told which grid it
//! is on. So a chunk meshes the same whichever grid holds its voxels.
//!
//! The chunk messages fire for the chunks of a sub-grid with `grid` naming the
//! sub-grid, and with keys on its grid. A physics integration rebuilds the
//! collider of a chunk on [`ChunkPublished`](crate::prelude::ChunkPublished)
//! by reading the chunk through its grid, which works for every grid:
//!
//! ```
//! use bevy::prelude::*;
//! use bevy_voxel_world::prelude::*;
//!
//! fn rebuild_colliders(
//!     mut published: MessageReader<ChunkPublished<DefaultWorld>>,
//!     voxel_world: VoxelWorld<DefaultWorld>,
//! ) {
//!     for message in published.read() {
//!         let Some(chunk) = voxel_world
//!             .grid(message.grid)
//!             .and_then(|grid| grid.get_chunk_data(message.chunk_key))
//!         else {
//!             continue;
//!         };
//!         // Build the collider from `chunk`'s voxels and put it on
//!         // `message.entity`. A sub-grid's chunk is a child of the
//!         // sub-grid, so its collider moves with the sub-grid.
//!         let _ = chunk;
//!     }
//! }
//! # bevy::ecs::system::assert_is_system(rebuild_colliders);
//! ```
//!
//! # Handovers
//!
//! Voxels moving from one grid to another never show twice or not at all. A
//! commit that gives a sub-grid its first solid voxel, or takes its last,
//! lands as one handover. Every chunk the commit spawns or makes stale, in
//! every grid, keeps showing what it showed, its mesh and its published voxel
//! data, until all of them have a finished chunk task that includes the
//! commit. Then all of them publish in the same frame, and the chunks of the
//! sub-grid that the commit retires despawn in that frame. So a system that
//! fills a new sub-grid and sets the same voxels to air in the world shows
//! them move in one frame. Colliders rebuilt on
//! [`ChunkPublished`](crate::prelude::ChunkPublished) swap in that frame too.
//! A commit that fills or empties no sub-grid publishes each chunk as its task
//! finishes.
//!
//! In a lit world, a handover also waits until the relights its commit starts
//! have solved, in every grid. These are the first solves of the new sub-grid
//! chunks, and the relights of the chunks around the hole of a lift and of
//! those under a stamp. Light then changes with the meshes, in the same frame.
//! So a lifted piece shows the light of its emissive voxels from its first
//! frame, and the walls of the hole never flash dark. The handover doesn't
//! wait for the relight ripple that light calls for. The ripple starts as the
//! handover lands, and lands later, as after any edit.
//!
//! So a lit handover lasts as long as its slowest relight. That relight grows
//! with the light reach of the world and with the light colours it runs: one
//! run for each light colour of the emissive voxels within light reach of its
//! chunk.
//!
//! The state of a held chunk turns Ready when its task finishes, before its
//! result publishes. So a game waits for `ChunkPublished` rather than for
//! Ready to read the chunk. An edit made every frame in a held chunk doesn't
//! hold the handover, since the task it supersedes started after the commit.
//!
//! Handovers that hold a chunk in common fold into one and land together. So
//! lifting voxels from the same chunks every frame holds the folded handover
//! until the lifting stops. A sub-grid that flips between empty and not every
//! frame, beside chunks edited every frame, keeps folding handovers, and can
//! hold its chunks back for as long. A grid despawned mid-handover drops out
//! of it, and the other chunks land when they are ready.
//! [`VoxelWorld::despawn_sub_grid`](crate::prelude::VoxelWorld::despawn_sub_grid)
//! waits for the handover holding the chunks of a sub-grid to land.
//!
//! The commit of a stamp lands as one handover even when it fills or empties
//! no sub-grid. That happens when one system lifts voxels out of a sub-grid
//! and stamps the new sub-grid into the world, since the new sub-grid is
//! empty before the commit and after it. A world applying that commit from
//! the edit deltas of another world gets only the deltas of the grid lifted
//! from and the grid stamped into. Those fill or empty no sub-grid, so there
//! the chunks of the two grids can publish a frame apart.
//!
//! The cost of a handover grows with the chunks it holds. Each chunk keeps its
//! newest finished chunk task until the handover lands.
//!
//! # Syncing and saving
//!
//! The voxel edits of a sub-grid travel and save as those of the world do. Its
//! commits write a
//! [`SubGridEditsCommitted`](crate::prelude::SubGridEditsCommitted) per
//! changed chunk, and [`GridVoxels::full_delta`],
//! [`GridVoxels::edited_chunks`] and [`GridVoxelsMut::apply_delta`] work on
//! any grid. The [`edits`](crate::edits) module docs give the rules, how a
//! lift and a stamp travel as their parts, and the order a load runs in.
//!
//! # Lifetime and pose
//!
//! The crate keeps a record of the live sub-grids of each world. A system
//! reads the record at once, before its commands apply, so the system that
//! spawns a sub-grid can fill it. As the entity of a sub-grid despawns, or its
//! [`SubGrid<C>`] is removed, the sub-grid ends and its voxel edits go with
//! it. Its chunks despawn in the next `PreUpdate`, each with its
//! [`ChunkWillDespawn`](crate::prelude::ChunkWillDespawn).
//!
//! After transform propagation in `PostUpdate`, the crate keeps the pose of
//! each sub-grid against the root of the world, which [`GridVoxels::pose`]
//! reads: the pose shown last frame.

use std::{
    marker::PhantomData,
    sync::{Arc, Mutex},
};

use bevy::{
    math::Affine3A,
    platform::collections::{HashMap, HashSet},
    prelude::*,
};

use crate::{
    chunk::{padding_depth, ChunkData, ChunkKey, CHUNK_SIZE_I},
    chunk_lifecycle::{
        ChunkIndex, ChunkRequests, ChunkStateOf, ChunkStatus, SpawnRequest,
    },
    chunk_map::{ChunkMap, ChunkMapRemoveBuffer},
    configuration::VoxelWorldConfig,
    edits::{EditDelta, EditGap},
    grid::{GridId, GridIds, GridKey},
    voxel::WorldVoxel,
    voxel_edits::{chunks_including_box, EditedBox, VoxelEditReader, VoxelEdits},
    voxel_world::voxel_at,
    voxel_world_internal::WorldRoot,
};

pub(crate) mod handover;
mod lift;
pub(crate) mod raycast;
pub(crate) mod snap;
mod stamp;

pub(crate) use lift::{lift, LiftParts};
pub use lift::{LiftError, Lifted};
pub use stamp::{incoming_replaces, StampError, StampPreview, StampVoxel, Stamped};
pub(crate) use stamp::{move_stamped_children, preview, stamp, StampParts};

/// Names a grid of a voxel world: its streamed voxels, or one of its
/// sub-grids by the sub-grid's entity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Grid {
    /// The world's streamed voxels, which the methods of
    /// [`VoxelWorld`](crate::prelude::VoxelWorld) that name no grid read and
    /// edit.
    World,
    /// The sub-grid whose entity this is.
    SubGrid(Entity),
}

/// Marks a sub-grid of the voxel world `C`, so a system can query a world's
/// sub-grids.
///
/// It is added by [`VoxelWorld::spawn_sub_grid`] and
/// [`VoxelWorld::insert_sub_grid`], and has no public constructor. Removing
/// it, or despawning its entity, ends the sub-grid as the removal applies:
/// its voxels go at once, and its chunks in the next `PreUpdate`.
/// [`VoxelWorld::despawn_sub_grid`] despawns it once the handover holding
/// its chunks, if one does, has landed.
///
/// [`VoxelWorld::despawn_sub_grid`]: crate::prelude::VoxelWorld::despawn_sub_grid
/// [`VoxelWorld::spawn_sub_grid`]: crate::prelude::VoxelWorld::spawn_sub_grid
/// [`VoxelWorld::insert_sub_grid`]: crate::prelude::VoxelWorld::insert_sub_grid
#[derive(Component)]
#[require(Transform, Visibility)]
#[component(on_remove)]
pub struct SubGrid<C: VoxelWorldConfig> {
    grid: GridId,
    _world: PhantomData<fn() -> C>,
}

impl<C: VoxelWorldConfig> SubGrid<C> {
    /// The grid this sub-grid's voxels are kept under.
    #[allow(dead_code)]
    pub(crate) fn grid(&self) -> GridId {
        self.grid
    }

    /// Ends the sub-grid whose marker was removed or whose entity despawned,
    /// so the record of live sub-grids drops it before any later system
    /// reads it, and a [`VoxelWorld::insert_sub_grid`] of the same entity
    /// makes a new sub-grid.
    ///
    /// [`VoxelWorld::insert_sub_grid`]: crate::prelude::VoxelWorld::insert_sub_grid
    fn on_remove(
        world: bevy::ecs::world::DeferredWorld,
        context: bevy::ecs::lifecycle::HookContext,
    ) {
        let (Some(sub_grids), Some(edits)) = (
            world.get_resource::<SubGrids<C>>(),
            world.get_resource::<VoxelEdits<C, C::MaterialIndex>>(),
        ) else {
            return;
        };
        sub_grids.end_despawned(context.entity, edits);
    }
}

/// The live sub-grids of the voxel world `C`, with the id and pose of each.
///
/// The record sits behind a lock, so `VoxelWorld` adds a sub-grid through a
/// shared borrow and the system that adds it sees it live at once.
#[derive(Resource)]
pub(crate) struct SubGrids<C> {
    record: Mutex<Record>,
    _world: PhantomData<fn() -> C>,
}

struct Record {
    /// The world's root, which a spawned sub-grid is a child of.
    root: Entity,
    live: HashMap<Entity, Live>,
    /// The entity of each live sub-grid, by its id.
    by_grid: HashMap<GridId, Entity>,
    ids: GridIds,
    /// How many pose snapshots have run.
    snapshots: u64,
    /// Chunks a commit found needed while their key's chunk was still
    /// Retiring, to look at again each frame until that chunk is gone.
    recheck: HashSet<GridKey>,
    /// The ids of ended sub-grids whose chunks or chunk data aren't gone yet,
    /// so the ids can't be handed out again.
    ending: Vec<GridId>,
}

/// A live sub-grid.
#[derive(Clone, Copy)]
struct Live {
    grid: GridId,
    /// The pose against the world's root.
    pose: GlobalTransform,
    /// The snapshot count when the sub-grid was added.
    added: u64,
    /// The nearest of its ancestors that is a live sub-grid, as the last
    /// pose snapshot found it, or as the call that spawned it says.
    inside: Option<Entity>,
}

impl<C: VoxelWorldConfig> SubGrids<C> {
    /// A record of no sub-grids, for the world whose root is `root`.
    pub fn new(root: Entity) -> Self {
        Self {
            record: Mutex::new(Record {
                root,
                live: HashMap::new(),
                by_grid: HashMap::new(),
                ids: GridIds::default(),
                snapshots: 0,
                recheck: HashSet::new(),
                ending: Vec::new(),
            }),
            _world: PhantomData,
        }
    }

    /// Makes `entity` a sub-grid at `pose`, with a new id, and returns the
    /// sub-grid's marker. `parent` is the entity it is a child of, where the
    /// caller knows it. An entity that is a live sub-grid already keeps its
    /// id and pose.
    ///
    /// # Panics
    ///
    /// When the world has 65,535 live sub-grids, the most its ids can name.
    fn add(
        &self,
        entity: Entity,
        pose: GlobalTransform,
        parent: Option<Entity>,
    ) -> SubGrid<C> {
        let mut record = self.record.lock().unwrap();
        let added = record.snapshots;
        let inside = parent.filter(|parent| record.live.contains_key(parent));
        let grid = match record.live.get(&entity) {
            Some(live) => live.grid,
            None => {
                let grid = record.ids.allocate().unwrap_or_else(|| {
                    panic!(
                        "{} has 65,535 live sub-grids, the most it can hold",
                        std::any::type_name::<C>()
                    )
                });
                record.live.insert(
                    entity,
                    Live {
                        grid,
                        pose,
                        added,
                        inside,
                    },
                );
                record.by_grid.insert(grid, entity);
                grid
            }
        };
        SubGrid {
            grid,
            _world: PhantomData,
        }
    }

    /// The id and pose of the live sub-grid `entity`, if it is one.
    pub(crate) fn live(&self, entity: Entity) -> Option<(GridId, GlobalTransform)> {
        let record = self.record.lock().unwrap();
        record.live.get(&entity).map(|live| (live.grid, live.pose))
    }

    /// The entity of the live sub-grid kept under `grid`, if there is one.
    pub(crate) fn entity_of(&self, grid: GridId) -> Option<Entity> {
        self.record.lock().unwrap().by_grid.get(&grid).copied()
    }

    /// The entity, id and pose of each live sub-grid.
    pub(crate) fn live_grids(&self) -> Vec<(Entity, GridId, GlobalTransform)> {
        let record = self.record.lock().unwrap();
        record
            .live
            .iter()
            .map(|(entity, live)| (*entity, live.grid, live.pose))
            .collect()
    }

    fn root(&self) -> Entity {
        self.record.lock().unwrap().root
    }

    /// Whether the live sub-grid `entity` is `ancestor` or one of its
    /// descendants, through the sub-grid ancestors the last pose snapshot
    /// found. Its cost grows with how deep sub-grids nest.
    pub(crate) fn is_within(&self, entity: Entity, ancestor: Entity) -> bool {
        let record = self.record.lock().unwrap();
        let mut at = Some(entity);
        // A hierarchy has no cycles, but the record is only as fresh as the
        // last snapshot, so the walk stops after one step per sub-grid.
        for _ in 0..=record.live.len() {
            match at {
                Some(at) if at == ancestor => return true,
                Some(entity) => at = record.live.get(&entity).and_then(|l| l.inside),
                None => return false,
            }
        }
        false
    }

    /// Whether the world has a live sub-grid, or an ended one whose chunks or
    /// chunk data aren't gone yet. The systems that keep this record run only
    /// then, so a world with no sub-grid pays one lock per system and frame.
    pub(crate) fn in_use(&self) -> bool {
        let record = self.record.lock().unwrap();
        !record.live.is_empty() || !record.ending.is_empty() || !record.recheck.is_empty()
    }

    /// Whether the world has a live sub-grid.
    #[cfg(feature = "light_propagation")]
    pub(crate) fn has_live(&self) -> bool {
        !self.record.lock().unwrap().live.is_empty()
    }

    /// Ends the sub-grid `entity`, if it is live: ahead of its queued
    /// despawn, or as its [`SubGrid<C>`] is removed.
    pub(crate) fn end_despawned(
        &self,
        entity: Entity,
        edits: &VoxelEdits<C, C::MaterialIndex>,
    ) {
        Self::end(&mut self.record.lock().unwrap(), entity, edits);
    }

    /// Ends the sub-grid `entity`, dropping its voxel edits at once. Its
    /// chunks go in the next [`forget_despawned`](Self::forget_despawned),
    /// and its id is freed for a later sub-grid once their data is gone too.
    fn end(record: &mut Record, entity: Entity, edits: &VoxelEdits<C, C::MaterialIndex>) {
        if let Some(live) = record.live.remove(&entity) {
            edits.drop_grid(live.grid);
            record.by_grid.remove(&live.grid);
            record.recheck.retain(|key| key.grid() != live.grid);
            record.ending.push(live.grid);
        }
    }

    /// Despawns the chunks of every ended sub-grid, dropping them from the
    /// chunk index at once as despawned chunks are. The chunk entities
    /// despawn later in the frame, with the retired chunks, each with its
    /// `ChunkWillDespawn`, so this system has no `Commands` and needs no
    /// sync point before the commit it runs ahead of.
    ///
    /// An ended sub-grid's id is freed once the chunk map holds none of its
    /// chunks' data, which the chunk lifecycle removes later in the frame,
    /// so a later sub-grid given the id never finds the old one's chunks.
    pub fn forget_despawned(
        mut sub_grids: ResMut<Self>,
        mut index: ResMut<ChunkIndex<C>>,
        mut chunk_map_remove_buffer: ResMut<ChunkMapRemoveBuffer<C>>,
        chunk_map: Res<ChunkMap<C, C::MaterialIndex>>,
    ) {
        let record = sub_grids.record.get_mut().unwrap();
        if record.ending.is_empty() {
            return;
        }
        let chunk_map = chunk_map.get_read_lock();
        let Record { ending, ids, .. } = record;
        ending.retain(|&grid| {
            index.despawn_grid(grid, &mut chunk_map_remove_buffer);
            if chunk_map.holds_grid(grid) {
                return true;
            }
            ids.release(grid);
            false
        });
    }

    /// Spawns and retires the chunks of each sub-grid whose voxel edits this
    /// frame's commit changed, so each sub-grid has a level-0 chunk at
    /// exactly the chunk positions whose padded voxels hold a solid.
    ///
    /// Only the chunks whose padded voxels hold a changed voxel can change,
    /// so its cost grows with the chunks the commit changed, not with the
    /// sub-grids' size. A chunk needed again while its key's chunk is still
    /// Retiring spawns once that chunk has despawned.
    pub fn plan_chunks(
        mut sub_grids: ResMut<Self>,
        mut requests: ResMut<ChunkRequests<C>>,
        index: Res<ChunkIndex<C>>,
        edits: Res<VoxelEdits<C, C::MaterialIndex>>,
        configuration: Res<C>,
    ) {
        let record = sub_grids.record.get_mut().unwrap();
        if record.live.is_empty() && record.recheck.is_empty() {
            return;
        }
        let padding = padding_depth(configuration.chunk_data_shape(0), 0);
        let committed = edits.committed();
        let mut decisions: Vec<(GridKey, bool)> = std::mem::take(&mut record.recheck)
            .into_iter()
            .map(|key| {
                let (min, max) = padded_box(key.position(), padding);
                (key, committed.holds_edit_within(key.grid(), min, max))
            })
            .collect();
        for set in requests
            .edited
            .iter()
            .filter(|set| set.grid != GridId::WORLD)
        {
            for changed in set.stale_boxes() {
                decisions.extend(chunks_needing(changed, padding, |min, max| {
                    committed.holds_edit_within(changed.grid, min, max)
                }));
            }
        }
        let mut seen: HashSet<GridKey> = HashSet::new();
        for (key, needed) in decisions {
            let Some(&sub_grid) = record.by_grid.get(&key.grid()) else {
                continue;
            };
            if !seen.insert(key) {
                continue;
            }
            match index.get_in(key) {
                None if needed => requests
                    .spawn
                    .push(SpawnRequest::sub_grid_chunk(key, sub_grid)),
                Some(_) if index.is_retiring_in(key) => {
                    if needed {
                        record.recheck.insert(key);
                    }
                }
                Some(_) if !needed => requests.retire.push(key),
                _ => {}
            }
        }
    }

    /// Keeps each sub-grid's `GlobalTransform` against the world root's as
    /// its pose, and the nearest of its ancestors that is a sub-grid. Runs
    /// after transform propagation, so the pose is the one shown this frame.
    ///
    /// A sub-grid added before the last snapshot whose entity never got its
    /// [`SubGrid<C>`], as an entity despawned before
    /// [`VoxelWorld::insert_sub_grid`](crate::prelude::VoxelWorld::insert_sub_grid)
    /// reached it, ends here. Its cost grows with the live sub-grids and how
    /// deep they sit in the hierarchy.
    pub fn snapshot_poses(
        mut sub_grids: ResMut<Self>,
        roots: Query<&GlobalTransform, With<WorldRoot<C>>>,
        grids: Query<&GlobalTransform, With<SubGrid<C>>>,
        parents: Query<&ChildOf>,
        edits: Res<VoxelEdits<C, C::MaterialIndex>>,
    ) {
        let record = sub_grids.record.get_mut().unwrap();
        let snapshots = record.snapshots;
        record.snapshots += 1;
        if record.live.is_empty() {
            return;
        }
        let from_root = roots
            .get(record.root)
            .map_or(Affine3A::IDENTITY, |root| root.affine().inverse());
        let nearest_sub_grid_above = |entity: Entity| {
            let mut at = entity;
            while let Ok(child_of) = parents.get(at) {
                at = child_of.parent();
                if at == record.root {
                    return None;
                }
                if grids.contains(at) {
                    return Some(at);
                }
            }
            None
        };
        let inside: Vec<Option<Entity>> = record
            .live
            .keys()
            .map(|&entity| nearest_sub_grid_above(entity))
            .collect();
        let mut gone = Vec::new();
        for ((entity, live), inside) in record.live.iter_mut().zip(inside) {
            match grids.get(*entity) {
                Ok(global) => {
                    live.pose = GlobalTransform::from(from_root * global.affine());
                    live.inside = inside;
                }
                Err(_) if live.added < snapshots => gone.push(*entity),
                Err(_) => {}
            }
        }
        for entity in gone {
            Self::end(record, entity, &edits);
        }
    }
}

/// The padded voxels of the level-0 chunk at `chunk`, whose padding is
/// `padding` voxels deep: the box from its minimum to its maximum corner,
/// both inclusive.
fn padded_box(chunk: IVec3, padding: IVec3) -> (IVec3, IVec3) {
    let first = chunk * CHUNK_SIZE_I;
    (first - padding, first + CHUNK_SIZE_I - 1 + padding)
}

/// Each level-0 chunk of the grid of `changed` whose padded voxels hold a
/// voxel of the box `changed`, with whether it needs a chunk: whether
/// `holds_solid` says its padded voxels hold a solid voxel. `holds_solid`
/// takes a box from its minimum to its maximum corner, both inclusive, and
/// `padding` is how many voxels deep a level-0 chunk's padding is.
///
/// A chunk position whose padded voxels hold none of the box's voxels keeps
/// its need when the box's voxels change, so a commit spawns and retires
/// only the chunks this gives for the voxels it changed.
pub(crate) fn chunks_needing(
    changed: EditedBox,
    padding: IVec3,
    holds_solid: impl Fn(IVec3, IVec3) -> bool,
) -> impl Iterator<Item = (GridKey, bool)> {
    chunks_including_box(changed, 0, padding).map(move |key| {
        let (min, max) = padded_box(key.position(), padding);
        (key, holds_solid(min, max))
    })
}

/// What a grid view reads and edits.
#[derive(Clone, Copy)]
enum Target {
    World,
    SubGrid { grid: GridId, pose: GlobalTransform },
}

/// Reads one grid of the voxel world `C`, from [`VoxelWorld::grid`].
///
/// [`VoxelWorld::grid`]: crate::prelude::VoxelWorld::grid
pub struct GridVoxels<'a, C: VoxelWorldConfig> {
    edits: &'a VoxelEdits<C, C::MaterialIndex>,
    chunk_map: &'a ChunkMap<C, C::MaterialIndex>,
    chunk_index: &'a ChunkIndex<C>,
    chunk_states: &'a Query<'a, 'a, &'static ChunkStateOf<C>>,
    configuration: &'a Arc<C>,
    /// The world's light, in a world with light propagation.
    #[cfg(feature = "light_propagation")]
    light: Option<crate::light::LightReads<'a, C>>,
    target: Target,
}

impl<'a, C: VoxelWorldConfig> GridVoxels<'a, C> {
    /// The view of `grid`, or `None` if it names an entity that isn't a live
    /// sub-grid of this world.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        grid: Grid,
        sub_grids: &SubGrids<C>,
        edits: &'a VoxelEdits<C, C::MaterialIndex>,
        chunk_map: &'a ChunkMap<C, C::MaterialIndex>,
        chunk_index: &'a ChunkIndex<C>,
        chunk_states: &'a Query<'a, 'a, &'static ChunkStateOf<C>>,
        configuration: &'a Arc<C>,
    ) -> Option<Self> {
        let target = match grid {
            Grid::World => Target::World,
            Grid::SubGrid(entity) => {
                let (grid, pose) = sub_grids.live(entity)?;
                Target::SubGrid { grid, pose }
            }
        };
        Some(Self {
            edits,
            chunk_map,
            chunk_index,
            chunk_states,
            configuration,
            #[cfg(feature = "light_propagation")]
            light: None,
            target,
        })
    }

    /// This view, reading the world's light through `light`.
    #[cfg(feature = "light_propagation")]
    pub(crate) fn with_light(
        self,
        light: Option<crate::light::LightReads<'a, C>>,
    ) -> Self {
        Self { light, ..self }
    }

    /// The id this view's grid is kept under.
    fn id(&self) -> GridId {
        match self.target {
            Target::World => GridId::WORLD,
            Target::SubGrid { grid, .. } => grid,
        }
    }

    /// Where this grid's chunk at `key` is in the chunk lifecycle, or `None`
    /// if the grid has no chunk spawned there.
    ///
    /// For the world, it reads what
    /// [`VoxelWorld::chunk_state`](crate::prelude::VoxelWorld::chunk_state)
    /// does. A sub-grid's chunk keys are on its own grid, and its chunks are
    /// all at LOD level 0.
    pub fn chunk_state(&self, key: ChunkKey) -> Option<ChunkStatus> {
        let entity = self.chunk_index.get_in(GridKey::new(self.id(), key))?;
        // A chunk spawned this frame has no state component until commands
        // apply.
        Some(
            self.chunk_states
                .get(entity)
                .map_or(ChunkStatus::Pending, |state| state.status()),
        )
    }

    /// The published data of this grid's chunk at `key`, or `None` if the
    /// grid has no chunk published there.
    ///
    /// For the world, it reads what
    /// [`VoxelWorld::get_chunk_data`](crate::prelude::VoxelWorld::get_chunk_data)
    /// does. A sub-grid's chunk holds its voxel edits over air, with the key
    /// and voxel positions of its own grid.
    pub fn get_chunk_data(&self, key: ChunkKey) -> Option<ChunkData<C::MaterialIndex>> {
        self.chunk_map
            .get_read_lock()
            .get_in(&GridKey::new(self.id(), key))
            .cloned()
    }

    /// The voxel at `position` on this grid.
    ///
    /// For the world, it reads what
    /// [`VoxelWorld::get_voxel`](crate::prelude::VoxelWorld::get_voxel)
    /// does. In a sub-grid, it reads its solid voxel there, including one set
    /// this frame, and air where it holds none, never `WorldVoxel::Unset`.
    pub fn get_voxel(&self, position: IVec3) -> WorldVoxel<C::MaterialIndex> {
        match self.target {
            Target::World => voxel_at(
                &self.edits.reader(),
                &self.chunk_map.get_map(),
                self.configuration.as_ref(),
                position,
            ),
            Target::SubGrid { grid, .. } => {
                solid_at(&self.edits.reader_in(grid), position)
            }
        }
    }

    /// A sendable closure that reads this grid's voxels as
    /// [`get_voxel`](Self::get_voxel) does, for tasks off the main thread.
    ///
    /// It sees the edits set before it was made, and committed edits as they
    /// are when it reads.
    pub fn get_voxel_fn(
        &self,
    ) -> Arc<dyn Fn(IVec3) -> WorldVoxel<C::MaterialIndex> + Send + Sync> {
        match self.target {
            Target::World => {
                let edits = self.edits.reader();
                let chunk_map = self.chunk_map.get_map();
                let configuration = self.configuration.clone();
                Arc::new(move |position| {
                    voxel_at(&edits, &chunk_map, configuration.as_ref(), position)
                })
            }
            Target::SubGrid { grid, .. } => {
                let edits = self.edits.reader_in(grid);
                Arc::new(move |position| solid_at(&edits, position))
            }
        }
    }

    /// The box holding a sub-grid's solid voxels, from its minimum corner
    /// inclusive to its maximum corner exclusive, as
    /// [`GridVoxelsMut::clear_voxels_in`] takes a box. It includes voxels set
    /// this frame, and is `None` for a sub-grid with no solid voxel and for
    /// the world.
    ///
    /// A sub-grid with no edits since the last commit reads the box that
    /// commit kept. Otherwise its cost grows with the sub-grid's chunks
    /// holding voxels, and with the voxels of the chunks edited since the
    /// last commit.
    pub fn bounds(&self) -> Option<(IVec3, IVec3)> {
        match self.target {
            Target::World => None,
            Target::SubGrid { grid, .. } => self
                .edits
                .solid_bounds_in(grid)
                .map(|(min, max)| (min, max + 1)),
        }
    }

    /// The full delta of this grid's level-0 chunk at `chunk_pos`: every
    /// voxel edit it has, at its edit sequence. A chunk with no edits gives
    /// an empty one at sequence 0. Its cost grows with the chunk's edits.
    ///
    /// For the world, it gives what
    /// [`VoxelWorld::full_delta`](crate::prelude::VoxelWorld::full_delta)
    /// does. A sub-grid's full delta holds only solid voxels, so an emptied
    /// chunk gives an empty one at its sequence. It reads committed edits
    /// only.
    pub fn full_delta(&self, chunk_pos: IVec3) -> EditDelta<C::MaterialIndex> {
        self.edits.full_delta_in(self.id(), chunk_pos)
    }

    /// Every level-0 chunk position of this grid with voxel edits, in no
    /// particular order, with its edit sequence. Its cost grows with the
    /// grid's chunks holding edits.
    ///
    /// For the world, it gives what
    /// [`VoxelWorld::edited_chunks`](crate::prelude::VoxelWorld::edited_chunks)
    /// does. A sub-grid's chunk whose solids were all removed stays listed,
    /// with its sequence, for as long as the sub-grid lives, so a join and a
    /// heartbeat carry the removal. It reads committed edits only.
    pub fn edited_chunks(&self) -> Vec<(IVec3, u64)> {
        self.edits.edited_chunks_in(self.id())
    }

    /// The grid's pose against the world's root: the identity for the world.
    ///
    /// Voxel `v` of the grid fills the box from `v` to `v + 1` in the pose's
    /// frame, so its centre is at `pose().transform_point(v.as_vec3() +
    /// 0.5)` in the root's frame.
    ///
    /// A sub-grid's pose is the one it was shown at last frame: its
    /// `GlobalTransform` against the root's, kept after transform propagation
    /// in `PostUpdate`. So a `Transform` changed this frame shows here from
    /// the next frame. A sub-grid spawned this frame reads the transform it
    /// was spawned with, and an entity made a sub-grid this frame reads the
    /// identity.
    pub fn pose(&self) -> GlobalTransform {
        match self.target {
            Target::World => GlobalTransform::IDENTITY,
            Target::SubGrid { pose, .. } => pose,
        }
    }

    /// Returns the light a face of this grid shows at the voxel at
    /// `position`, as levels from 0 to 15.
    ///
    /// For the world, it reads what
    /// [`VoxelWorld::light_at`](crate::prelude::VoxelWorld::light_at) reads.
    /// In a sub-grid, it reads the light of the world at the world voxel
    /// holding the centre of the voxel, placed by [`pose`](Self::pose). It
    /// adds the voxel light that the emissive voxels of the sub-grid give:
    ///
    /// - The voxel light level is the brighter of the two levels.
    /// - The light colour is the two light colours mixed, as light of several
    ///   colours mixes anywhere in the world.
    ///
    /// An emissive voxel on a sub-grid lights only the sub-grid. The light of
    /// the world never shows it.
    ///
    /// Under [`SubGridSky::Enclosed`](crate::prelude::SubGridSky::Enclosed),
    /// the enclosure of the sub-grid caps the light of the world at an open
    /// voxel. It caps the sky light, and with sky light on it caps the voxel
    /// light too. So the inside of a closed cabin reads no sky, and the
    /// emissive voxels of the world don't light it through its walls. Where
    /// the sub-grid has no chunk, such as deep inside a hangar, the voxel is
    /// open where a line from it along an axis of the sub-grid leaves the
    /// sub-grid. Elsewhere it is closed in, which reads darker than the truth
    /// within sky light reach of an opening.
    ///
    /// Returns `None` when the world has no light propagation, or where
    /// `VoxelWorld::light_at` says nothing of the world voxel.
    #[cfg(feature = "light_propagation")]
    pub fn light_at(&self, position: IVec3) -> Option<crate::light::LightLevels> {
        let light = self.light.as_ref()?;
        let world_light_at = |position: IVec3| {
            light.settings.light_at(
                &self.chunk_map.get_read_lock(),
                light.work,
                light.route,
                self.configuration.as_ref(),
                self.edits,
                position,
            )
        };
        let (grid, pose) = match self.target {
            Target::World => return world_light_at(position),
            Target::SubGrid { grid, pose } => (grid, pose),
        };
        let centre = pose.transform_point(position.as_vec3() + 0.5);
        let world = world_light_at(centre.floor().as_ivec3())?;
        let config = &light.settings.config;
        let caps_voxel = config.enclosure_caps_voxel_light();
        let Some(own) = light.settings.own_light_at(grid, position) else {
            if light.settings.closed_in_at(grid, position) {
                let voxel = if caps_voxel { 0.0 } else { world.voxel };
                return Some(crate::light::LightLevels {
                    sky: 0.0,
                    voxel,
                    voxel_color: if voxel > 0.0 {
                        world.voxel_color
                    } else {
                        LinearRgba::WHITE
                    },
                });
            }
            return Some(world);
        };
        let capped = config.sub_grid_sky == crate::light::config::SubGridSky::Enclosed
            && !self.get_voxel(position).is_solid();
        let world_voxel = if capped && caps_voxel {
            world.voxel.min(own.sky)
        } else {
            world.voxel
        };
        let voxel_color = crate::light::color::mix([
            (world.voxel_color, world_voxel),
            (own.voxel_color, own.voxel),
        ]);
        Some(crate::light::LightLevels {
            sky: if capped {
                world.sky.min(own.sky)
            } else {
                world.sky
            },
            voxel: world_voxel.max(own.voxel),
            voxel_color,
        })
    }
}

/// Edits one grid of the voxel world `C`, from [`VoxelWorld::grid_mut`]. It
/// reads the grid too, as [`GridVoxels`] does.
///
/// [`VoxelWorld::grid_mut`]: crate::prelude::VoxelWorld::grid_mut
pub struct GridVoxelsMut<'a, C: VoxelWorldConfig> {
    voxels: GridVoxels<'a, C>,
}

impl<'a, C: VoxelWorldConfig> std::ops::Deref for GridVoxelsMut<'a, C> {
    type Target = GridVoxels<'a, C>;

    fn deref(&self) -> &Self::Target {
        &self.voxels
    }
}

impl<'a, C: VoxelWorldConfig> GridVoxelsMut<'a, C> {
    pub(crate) fn new(voxels: GridVoxels<'a, C>) -> Self {
        Self { voxels }
    }

    /// Sets the voxel at `position` on this grid. `get_voxel` reads it back
    /// at once, and it commits in the next `PreUpdate`.
    ///
    /// For the world, it does what
    /// [`VoxelWorld::set_voxel`](crate::prelude::VoxelWorld::set_voxel)
    /// does. In a sub-grid, setting air or `WorldVoxel::Unset` removes the
    /// voxel there.
    pub fn set_voxel(&mut self, position: IVec3, voxel: WorldVoxel<C::MaterialIndex>) {
        match self.target {
            Target::World => self.edits.set(position, voxel),
            Target::SubGrid { grid, .. } => self.edits.set_in(grid, position, voxel),
        }
    }

    /// Sets each voxel in the box from `min` inclusive to `max` exclusive to
    /// the voxel `voxel` returns for its position, leaving those it returns
    /// `None` for alone.
    ///
    /// For the world, it does what
    /// [`VoxelWorld::set_voxels_in`](crate::prelude::VoxelWorld::set_voxels_in)
    /// does. In a sub-grid, air or `WorldVoxel::Unset` removes the voxel
    /// there, as with [`set_voxel`](Self::set_voxel).
    pub fn set_voxels_in(
        &mut self,
        min: IVec3,
        max: IVec3,
        voxel: impl FnMut(IVec3) -> Option<WorldVoxel<C::MaterialIndex>>,
    ) {
        match self.target {
            Target::World => self.edits.set_region(min, max, voxel),
            Target::SubGrid { grid, .. } => {
                self.edits.set_region_in(grid, min, max, voxel)
            }
        }
    }

    /// Clears the box from `min` inclusive to `max` exclusive.
    ///
    /// For the world, it does what
    /// [`VoxelWorld::clear_voxels_in`](crate::prelude::VoxelWorld::clear_voxels_in)
    /// does. In a sub-grid, it removes every voxel in the box, which reads
    /// air at once.
    pub fn clear_voxels_in(&mut self, min: IVec3, max: IVec3) {
        match self.target {
            Target::World => self.edits.clear_region(min, max),
            Target::SubGrid { grid, .. } => self.edits.remove_region_in(grid, min, max),
        }
    }

    /// Applies an edit delta from another world to this grid's chunk at the
    /// delta's chunk position, if it continues from that chunk's edit
    /// sequence here. Returns `Ok` for a delta applied or one already
    /// applied, and an [`EditGap`] for one that doesn't continue, which is
    /// dropped: ask the sender for the chunk's full delta and apply that.
    ///
    /// For the world, it does what
    /// [`VoxelWorld::apply_delta`](crate::prelude::VoxelWorld::apply_delta)
    /// does, by the rule the [`edits`](crate::edits) module docs give. In a
    /// sub-grid, the rule is the same, and a voxel in the delta that isn't
    /// solid removes the voxel there, as setting it does. A delta that fills
    /// an empty sub-grid or empties one lands as one handover when it
    /// commits, with the other deltas applied in the same frame.
    pub fn apply_delta(
        &mut self,
        delta: EditDelta<C::MaterialIndex>,
    ) -> Result<(), EditGap> {
        self.edits.apply_in(self.id(), delta)
    }
}

/// The solid voxel a sub-grid's `edits` hold at `position`, or air.
pub(crate) fn solid_at<I: Copy + PartialEq>(
    edits: &VoxelEditReader<I>,
    position: IVec3,
) -> WorldVoxel<I> {
    edits
        .get(position)
        .filter(WorldVoxel::is_solid)
        .unwrap_or(WorldVoxel::Air)
}

/// What [`VoxelWorld`](crate::prelude::VoxelWorld) needs to spawn, find and
/// view sub-grids.
pub(crate) mod world {
    use super::*;

    /// Spawns a sub-grid at `transform` under the world's root, and returns
    /// its entity.
    pub fn spawn<C: VoxelWorldConfig>(
        sub_grids: &SubGrids<C>,
        commands: &mut Commands,
        transform: Transform,
    ) -> Entity {
        let root = sub_grids.root();
        spawn_under(sub_grids, commands, root, transform, transform.into()).0
    }

    /// Spawns a sub-grid at `transform` under `parent`, whose pose against
    /// the world's root is `pose` until the next snapshot. Returns its
    /// entity and id.
    pub fn spawn_under<C: VoxelWorldConfig>(
        sub_grids: &SubGrids<C>,
        commands: &mut Commands,
        parent: Entity,
        transform: Transform,
        pose: GlobalTransform,
    ) -> (Entity, GridId) {
        let entity = commands.spawn_empty().id();
        let marker = sub_grids.add(entity, pose, Some(parent));
        let grid = marker.grid;
        commands
            .entity(entity)
            .try_insert((marker, transform, ChildOf(parent)));
        (entity, grid)
    }

    /// The parent entity, id and pose of the grid `grid` names, or `None`
    /// for an entity that isn't a live sub-grid: the world's root, its id and
    /// the identity for the world.
    pub fn place<C: VoxelWorldConfig>(
        sub_grids: &SubGrids<C>,
        grid: Grid,
    ) -> Option<(Entity, GridId, GlobalTransform)> {
        match grid {
            Grid::World => {
                Some((sub_grids.root(), GridId::WORLD, GlobalTransform::IDENTITY))
            }
            Grid::SubGrid(entity) => {
                let (id, pose) = sub_grids.live(entity)?;
                Some((entity, id, pose))
            }
        }
    }

    /// Makes `entity` a sub-grid, leaving its parent alone.
    pub fn insert<C: VoxelWorldConfig>(
        sub_grids: &SubGrids<C>,
        commands: &mut Commands,
        entity: Entity,
    ) {
        if sub_grids.live(entity).is_some() {
            return;
        }
        let marker = sub_grids.add(entity, GlobalTransform::IDENTITY, None);
        commands.entity(entity).try_insert(marker);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The level-0 chunk positions a sub-grid holding `solids` needs, as a
    /// commit setting each of them finds them.
    fn needed(solids: &[IVec3]) -> HashSet<IVec3> {
        let holds_solid = |min: IVec3, max: IVec3| {
            solids
                .iter()
                .any(|solid| solid.cmpge(min).all() && solid.cmple(max).all())
        };
        solids
            .iter()
            .flat_map(|&solid| {
                chunks_needing(EditedBox::world(solid, solid), IVec3::ONE, holds_solid)
                    .filter(|(_, needed)| *needed)
                    .map(|(key, _)| key.position())
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    fn chunks(positions: &[[i32; 3]]) -> HashSet<IVec3> {
        positions.iter().map(|&p| IVec3::from_array(p)).collect()
    }

    #[test]
    fn a_solid_needs_every_chunk_whose_padding_reaches_it() {
        assert_eq!(
            needed(&[IVec3::new(5, 6, 7)]),
            chunks(&[[0, 0, 0]]),
            "inside"
        );
        assert_eq!(
            needed(&[IVec3::new(0, 6, 7)]),
            chunks(&[[0, 0, 0], [-1, 0, 0]]),
            "on a face"
        );
        assert_eq!(
            needed(&[IVec3::new(31, -1, 7)]),
            chunks(&[[0, -1, 0], [1, -1, 0], [0, 0, 0], [1, 0, 0]]),
            "on an edge"
        );
        let corner: Vec<[i32; 3]> = (0..8)
            .map(|i| [i & 1, (i >> 1) & 1, i >> 2].map(|c| c - 1))
            .collect();
        assert_eq!(
            needed(&[IVec3::new(-1, -1, -1)]),
            chunks(&corner),
            "at a corner"
        );
        assert_eq!(
            needed(&[
                IVec3::new(5, 6, 7),
                IVec3::new(0, 6, 7),
                IVec3::new(70, 6, 7)
            ]),
            chunks(&[[0, 0, 0], [-1, 0, 0], [2, 0, 0]]),
            "several solids"
        );
    }

    #[test]
    fn a_changed_voxel_reaches_only_the_chunks_whose_padding_holds_it() {
        // The solid at the corner is gone, and one inside the chunk at the
        // origin remains.
        let remaining = IVec3::new(5, 6, 7);
        let holds_solid = |min: IVec3, max: IVec3| {
            remaining.cmpge(min).all() && remaining.cmple(max).all()
        };
        let removed = IVec3::splat(31);
        let reached: Vec<(IVec3, bool)> =
            chunks_needing(EditedBox::world(removed, removed), IVec3::ONE, holds_solid)
                .map(|(key, needed)| (key.position(), needed))
                .collect();
        assert_eq!(reached.len(), 8);
        for (position, needed) in reached {
            assert_eq!(needed, position == IVec3::ZERO, "{position}");
        }
    }
}
