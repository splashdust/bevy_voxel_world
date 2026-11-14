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
//! next `PreUpdate`, with the world's. A sub-grid has no chunks yet, so
//! nothing renders it: its voxels are read from its edits.
//!
//! The crate keeps a record of each world's live sub-grids, which a system
//! reads at once, before its commands apply, so the system that spawns a
//! sub-grid can fill it. Once the crate sees a sub-grid's entity despawned,
//! or its [`SubGrid<C>`] removed, in the next `PreUpdate` at the latest, the
//! sub-grid's voxel edits go with it.
//!
//! After transform propagation in `PostUpdate`, the crate keeps each
//! sub-grid's pose against the world's root, which [`GridVoxels::pose`]
//! reads: the pose shown last frame.

use std::{
    marker::PhantomData,
    sync::{Arc, Mutex},
};

use bevy::{math::Affine3A, platform::collections::HashMap, prelude::*};

use crate::{
    chunk_map::ChunkMap,
    configuration::VoxelWorldConfig,
    grid::{GridId, GridIds},
    voxel::WorldVoxel,
    voxel_edits::{VoxelEditReader, VoxelEdits},
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
/// it, or despawning its entity, ends the sub-grid and drops its voxels.
///
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
    ids: GridIds,
    /// How many pose snapshots have run.
    snapshots: u64,
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
                ids: GridIds::default(),
                snapshots: 0,
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
                grid
            }
        };
        SubGrid {
            grid,
            _world: PhantomData,
        }
    }

    /// The id and pose of the live sub-grid `entity`, if it is one.
    fn live(&self, entity: Entity) -> Option<(GridId, GlobalTransform)> {
        let record = self.record.lock().unwrap();
        record.live.get(&entity).map(|live| (live.grid, live.pose))
    }

    fn root(&self) -> Entity {
        self.record.lock().unwrap().root
    }

    /// Ends the sub-grid `entity`, dropping its voxel edits and freeing its
    /// id for a later sub-grid.
    fn end(record: &mut Record, entity: Entity, edits: &VoxelEdits<C, C::MaterialIndex>) {
        if let Some(live) = record.live.remove(&entity) {
            edits.drop_grid(live.grid);
            record.ids.release(live.grid);
        }
    }

    /// Ends each sub-grid whose entity was despawned or lost its
    /// [`SubGrid<C>`] since this last ran.
    pub fn forget_despawned(
        mut removed: RemovedComponents<SubGrid<C>>,
        mut sub_grids: ResMut<Self>,
        edits: Res<VoxelEdits<C, C::MaterialIndex>>,
    ) {
        let record = sub_grids.record.get_mut().unwrap();
        for entity in removed.read() {
            Self::end(record, entity, &edits);
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
    configuration: &'a Arc<C>,
    target: Target,
}

impl<'a, C: VoxelWorldConfig> GridVoxels<'a, C> {
    /// The view of `grid`, or `None` if it names an entity that isn't a live
    /// sub-grid of this world.
    pub(crate) fn new(
        grid: Grid,
        sub_grids: &SubGrids<C>,
        edits: &'a VoxelEdits<C, C::MaterialIndex>,
        chunk_map: &'a ChunkMap<C, C::MaterialIndex>,
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
            configuration,
            target,
        })
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
fn solid_at<I: Copy + PartialEq>(
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
