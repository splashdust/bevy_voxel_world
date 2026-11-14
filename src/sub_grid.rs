//! Sub-grids: sets of voxels on grids of their own, each belonging to one
//! voxel world, that a game moves freely against the world's grid.
//!
//! A sub-grid is an entity carrying [`SubGrid<C>`], made with
//! [`VoxelWorld::spawn_sub_grid`] or [`VoxelWorld::insert_sub_grid`], and
//! named by a [`Grid`]. [`VoxelWorld::grid`] and [`VoxelWorld::grid_mut`]
//! read and edit any grid with the calls the world's own methods use.
//!
//! A sub-grid has no generator, so every voxel it holds is a voxel edit, and
//! its edits are exactly its solid voxels. Setting air or `Unset` removes a
//! voxel, and a position with no solid reads air. Its edits commit in the
//! next `PreUpdate`, with the world's.
//!
//! # Chunks
//!
//! A sub-grid shows through chunks of its own, built by the world's chunk
//! tasks and meshed by the world's mesher, material and mesh cache, so
//! identical voxels in two grids share one mesh. Each commit spawns and
//! retires them in place of the streaming policy: after it, the sub-grid
//! has a chunk at LOD level 0 at every chunk position whose padded voxels
//! hold one of its solid voxels, and no other. A chunk's voxels are the
//! sub-grid's voxel edits over air, so the world's voxel lookup delegate and
//! fill hints are never called for them. A sub-grid with no solid voxel has
//! no chunks, and lives on until it is despawned.
//!
//! A sub-grid's chunks start their chunk tasks ahead of streamed chunks, as
//! edited chunks do, and count towards the world's
//! [`max_active_chunk_threads`](crate::prelude::VoxelWorldConfig::max_active_chunk_threads)
//! with the world's own. A commit's cost grows with the chunks its edits
//! reach, not with the sub-grid's size. Moving the camera never retires them,
//! and [`VoxelWorld::regenerate_chunks`] leaves them alone.
//!
//! # Light
//!
//! In a world with light propagation, a sub-grid's voxels stay out of the
//! world's light: they block none of its light and spread none into it.
//! A sub-grid's emissive voxels light the sub-grid itself, with the world's
//! voxel reach, precision and shape, and `GridVoxels::light_at` reads that
//! light on top of the world's light where the sub-grid is. Only voxel edits
//! to the sub-grid and changes of light settings solve its light again:
//! moving it costs no solve.
//!
//! Each chunk is a child of the sub-grid's entity, placed by its chunk key as
//! the world's chunks are under the world's root, so it moves with the
//! sub-grid. The crate copies no component from the sub-grid to its chunks:
//! `Visibility` inherits through the hierarchy, and a game gives the chunks
//! `RenderLayers` or `NotShadowCaster` by putting Bevy's `Propagate` of them
//! on the sub-grid. A sub-grid spawned under the world's root picks up a
//! layer the root propagates, until the game gives the sub-grid another
//! parent.
//!
//! The meshing delegate and texture index mapper get a sub-grid chunk's
//! position on the sub-grid's own grid, and aren't told which grid it is on,
//! so a chunk meshes the same whichever grid holds its voxels.
//!
//! The chunk messages fire for a sub-grid's chunks with `grid` naming the
//! sub-grid, and keys on its grid. A physics integration rebuilds a chunk's
//! collider on [`ChunkPublished`](crate::prelude::ChunkPublished) by reading
//! the chunk through its grid, which works for every grid:
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
//! lands as one handover: every chunk the commit spawns or makes stale, in
//! every grid, keeps showing what it showed, its mesh and its published
//! voxel data, until all of them have a finished chunk task that includes
//! the commit. Then all of them publish in the same frame, and the
//! sub-grid's chunks the commit retires despawn in that frame. So a system
//! that fills a new sub-grid and sets the same voxels to air in the world
//! shows them move in one frame, and colliders rebuilt on
//! [`ChunkPublished`](crate::prelude::ChunkPublished) swap in that frame
//! too. A commit that fills or empties no sub-grid publishes each chunk as
//! its task finishes.
//!
//! A held chunk's state turns Ready when its task finishes, before its
//! result publishes, so a game waits for `ChunkPublished` rather than for
//! Ready to read it. An edit made every frame in a held chunk doesn't hold
//! the handover, since the task it supersedes started after the commit.
//!
//! Handovers that hold a chunk in common fold into one and land together.
//! So lifting voxels from the same chunks every frame holds the folded
//! handover until the lifting stops, and a sub-grid that flips between
//! empty and not every frame, beside chunks edited every frame, keeps
//! folding handovers and can hold its chunks back for as long. A grid
//! despawned mid-handover drops out of it, and the other chunks land when
//! they are ready.
//! [`VoxelWorld::despawn_sub_grid`](crate::prelude::VoxelWorld::despawn_sub_grid)
//! waits for the handover holding a sub-grid's chunks to land.
//!
//! A handover's cost grows with the chunks it holds. Each one keeps its
//! newest finished chunk task until the handover lands.
//!
//! # Lifetime and pose
//!
//! The crate keeps a record of each world's live sub-grids, which a system
//! reads at once, before its commands apply, so the system that spawns a
//! sub-grid can fill it. Once the crate sees a sub-grid's entity despawned,
//! or its [`SubGrid<C>`] removed, in the next `PreUpdate` at the latest, the
//! sub-grid's voxel edits and chunks go with it.
//!
//! After transform propagation in `PostUpdate`, the crate keeps each
//! sub-grid's pose against the world's root, which [`GridVoxels::pose`]
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
    grid::{GridId, GridIds, GridKey},
    voxel::WorldVoxel,
    voxel_edits::{chunks_including_box, EditedBox, VoxelEditReader, VoxelEdits},
    voxel_world::voxel_at,
    voxel_world_internal::WorldRoot,
};

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
/// it, or despawning its entity, ends the sub-grid and drops its voxels and
/// chunks. [`VoxelWorld::despawn_sub_grid`] despawns it once the handover
/// holding its chunks, if one does, has landed.
///
/// [`VoxelWorld::despawn_sub_grid`]: crate::prelude::VoxelWorld::despawn_sub_grid
/// [`VoxelWorld::spawn_sub_grid`]: crate::prelude::VoxelWorld::spawn_sub_grid
/// [`VoxelWorld::insert_sub_grid`]: crate::prelude::VoxelWorld::insert_sub_grid
#[derive(Component)]
#[require(Transform, Visibility)]
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
    /// sub-grid's marker. An entity that is a live sub-grid already keeps its
    /// id and pose.
    ///
    /// # Panics
    ///
    /// When the world has 65,535 live sub-grids, the most its ids can name.
    fn add(&self, entity: Entity, pose: GlobalTransform) -> SubGrid<C> {
        let mut record = self.record.lock().unwrap();
        let added = record.snapshots;
        let grid = match record.live.get(&entity) {
            Some(live) => live.grid,
            None => {
                let grid = record.ids.allocate().unwrap_or_else(|| {
                    panic!(
                        "{} has 65,535 live sub-grids, the most it can hold",
                        std::any::type_name::<C>()
                    )
                });
                record.live.insert(entity, Live { grid, pose, added });
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

    /// Ends each sub-grid whose entity was despawned or lost its
    /// [`SubGrid<C>`] since this last ran, and despawns the chunks of every
    /// ended sub-grid, dropping them from the chunk index at once as
    /// despawned chunks are.
    ///
    /// An ended sub-grid's id is freed once the chunk map holds none of its
    /// chunks' data, which the chunk lifecycle removes later in the frame,
    /// so a later sub-grid given the id never finds the old one's chunks.
    #[allow(clippy::too_many_arguments)]
    pub fn forget_despawned(
        mut commands: Commands,
        mut removed: RemovedComponents<SubGrid<C>>,
        mut sub_grids: ResMut<Self>,
        edits: Res<VoxelEdits<C, C::MaterialIndex>>,
        mut index: ResMut<ChunkIndex<C>>,
        mut chunk_map_remove_buffer: ResMut<ChunkMapRemoveBuffer<C>>,
        chunk_map: Res<ChunkMap<C, C::MaterialIndex>>,
    ) {
        let record = sub_grids.record.get_mut().unwrap();
        for entity in removed.read() {
            Self::end(record, entity, &edits);
        }
        if record.ending.is_empty() {
            return;
        }
        let chunk_map = chunk_map.get_read_lock();
        let Record { ending, ids, .. } = record;
        ending.retain(|&grid| {
            index.despawn_grid(grid, &mut commands, &mut chunk_map_remove_buffer);
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
    /// its pose. Runs after transform propagation, so the pose is the one
    /// shown this frame.
    ///
    /// A sub-grid added before the last snapshot whose entity never got its
    /// [`SubGrid<C>`], as an entity despawned before
    /// [`VoxelWorld::insert_sub_grid`](crate::prelude::VoxelWorld::insert_sub_grid)
    /// reached it, ends here. Its cost grows with the live sub-grids.
    pub fn snapshot_poses(
        mut sub_grids: ResMut<Self>,
        roots: Query<&GlobalTransform, With<WorldRoot<C>>>,
        grids: Query<&GlobalTransform, With<SubGrid<C>>>,
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
        let mut gone = Vec::new();
        for (entity, live) in record.live.iter_mut() {
            match grids.get(*entity) {
                Ok(global) => {
                    live.pose = GlobalTransform::from(from_root * global.affine())
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
    /// Its cost grows with the sub-grid's chunks holding voxels, and with
    /// the voxels of the chunks edited since the last commit.
    pub fn bounds(&self) -> Option<(IVec3, IVec3)> {
        match self.target {
            Target::World => None,
            Target::SubGrid { grid, .. } => self
                .edits
                .solid_bounds_in(grid)
                .map(|(min, max)| (min, max + 1)),
        }
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

    /// The light a face of this grid shows at the voxel at `position`, as
    /// levels from 0 to 15.
    ///
    /// For the world, it reads what
    /// [`VoxelWorld::light_at`](crate::prelude::VoxelWorld::light_at) does.
    /// In a sub-grid, it reads the world's light at the world voxel holding
    /// the voxel's centre, placed by [`pose`](Self::pose), with the
    /// sub-grid's own voxel light, from its emissive voxels, taking the
    /// brighter of the two. Sky light is the world's. A lamp on a sub-grid
    /// lights only the sub-grid: the world's light never shows it.
    ///
    /// Returns `None` when the world has no light propagation, or where the
    /// world's `light_at` says nothing of the world voxel.
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
        let Some(own) = light.settings.own_light_at(grid, position) else {
            return Some(world);
        };
        Some(crate::light::LightLevels {
            sky: world.sky,
            voxel: world.voxel.max(own.voxel),
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
        let entity = commands.spawn_empty().id();
        let marker = sub_grids.add(entity, GlobalTransform::from(transform));
        commands.entity(entity).try_insert((
            marker,
            transform,
            ChildOf(sub_grids.root()),
        ));
        entity
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
        let marker = sub_grids.add(entity, GlobalTransform::IDENTITY);
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
