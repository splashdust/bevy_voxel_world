use std::{
    marker::PhantomData,
    sync::{Arc, RwLock, RwLockReadGuard},
};

use crate::{
    chunk::{self, ChunkData, ChunkKey, LevelCounts, CHUNK_SIZE_F},
    configuration::VoxelWorldConfig,
    grid::{GridId, GridKey, GridTable, PerGrid},
    voxel::VOXEL_SIZE,
    voxel_world::{ChunkPublished, ChunkWillSpawn},
};
use bevy::{math::Vec3A, prelude::*};
use hashbrown::DefaultHashBuilder;

/// The published data of chunks by grid key, with each grid's LOD levels in
/// use and the bounds of its published chunks.
///
/// The methods that take a `ChunkKey` or a voxel position read the world's
/// grid.
pub struct ChunkMapData<I> {
    data: GridTable<chunk::ChunkData<I>, DefaultHashBuilder>,
    levels: PerGrid<LevelCounts>,
    bounds: PerGrid<Aabb3d>,
}

impl<I> ChunkMapData<I> {
    /// The published data of the world's chunk at `key`, if it has any.
    pub fn get(&self, key: &ChunkKey) -> Option<&chunk::ChunkData<I>> {
        self.get_in(&GridKey::world(*key))
    }

    /// The published data of the chunk at `key`, if it has any.
    pub(crate) fn get_in(&self, key: &GridKey) -> Option<&chunk::ChunkData<I>> {
        self.data.get(key)
    }

    /// Whether no chunk of any grid has published data.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// The published data of the world's finest chunk that covers the voxel
    /// at `voxel_position`, if any chunk does.
    pub fn finest_covering(&self, voxel_position: IVec3) -> Option<&chunk::ChunkData<I>> {
        self.covering(voxel_position).next()
    }

    /// The published data of every chunk of the world that covers the voxel
    /// at `voxel_position`, finest first.
    pub fn covering(
        &self,
        voxel_position: IVec3,
    ) -> impl Iterator<Item = &chunk::ChunkData<I>> + '_ {
        self.covering_in(GridId::WORLD, voxel_position)
    }

    /// The published data of every chunk of `grid` that covers the voxel at
    /// `voxel_position` on that grid, finest first.
    pub(crate) fn covering_in(
        &self,
        grid: GridId,
        voxel_position: IVec3,
    ) -> impl Iterator<Item = &chunk::ChunkData<I>> + '_ {
        let levels = self.levels.get(grid).unwrap_or(&LevelCounts::NONE);
        levels.in_use().filter_map(move |lod_level| {
            self.data.get(&GridKey::new(
                grid,
                ChunkKey::containing(voxel_position, lod_level),
            ))
        })
    }

    /// The bounding box of `grid`'s published chunks, in chunk coordinates,
    /// both corners inclusive. A grid with no published chunk has an empty
    /// box at the origin.
    pub(crate) fn bounds_in(&self, grid: GridId) -> Aabb3d {
        self.bounds.get(grid).copied().unwrap_or(EMPTY_BOUNDS)
    }

    /// Publishes `chunk_data` as the data of the chunk at `key`, growing its
    /// grid's bounds and LOD levels. Returns the data it replaced, if any.
    pub(crate) fn insert(
        &mut self,
        key: GridKey,
        chunk_data: chunk::ChunkData<I>,
    ) -> Option<chunk::ChunkData<I>> {
        let previous = self.data.insert(key, chunk_data);
        let (min, max) = chunk_corners(&key.key());
        let bounds = self.bounds.get_or_insert_with(key.grid(), || EMPTY_BOUNDS);
        if min.cmplt(bounds.min).any() {
            bounds.min = min.min(bounds.min);
        }
        if max.cmpgt(bounds.max).any() {
            bounds.max = max.max(bounds.max);
        }
        if previous.is_none() {
            self.levels
                .get_or_insert_with(key.grid(), LevelCounts::new)
                .add(key.lod_level());
        }
        previous
    }

    /// Removes the data of the chunk at `key`. Returns whether the chunk was
    /// its grid's last or on the edge of its grid's bounds, so they need
    /// [`rebuild_bounds`](Self::rebuild_bounds).
    fn remove(&mut self, key: &GridKey) -> bool {
        if self.data.remove(key).is_none() {
            return false;
        }
        let levels = self.levels.get_or_insert_with(key.grid(), LevelCounts::new);
        levels.remove(key.lod_level());
        let last = levels.total() == 0;
        let (min, max) = chunk_corners(&key.key());
        let bounds = self.bounds_in(key.grid());
        last || bounds.min.floor().cmpeq(min).any() || bounds.max.floor().cmpeq(max).any()
    }

    /// Works `grid`'s bounds out again from its published chunks. Its cost
    /// grows with the published chunks of every grid.
    fn rebuild_bounds(&mut self, grid: GridId) {
        let empty = self
            .levels
            .get(grid)
            .is_none_or(|levels| levels.total() == 0);
        let bounds = if empty {
            EMPTY_BOUNDS
        } else {
            let corners: Vec<Vec3A> = self
                .data
                .grid_keys(grid)
                .flat_map(|key| {
                    let (min, max) = chunk_corners(&key.key());
                    [min, max]
                })
                .collect();
            Aabb3d::from_point_cloud(Isometry3d::IDENTITY, corners.into_iter())
        };
        *self.bounds.get_or_insert_with(grid, || EMPTY_BOUNDS) = bounds;
    }
}

/// The bounds of a grid with no published chunk.
const EMPTY_BOUNDS: Aabb3d = Aabb3d {
    min: Vec3A::ZERO,
    max: Vec3A::ZERO,
};

/// Holds the published data of chunks, by chunk key. A chunk has no entry here
/// until its first chunk task is published.
#[derive(Resource)]
pub struct ChunkMap<C, I> {
    map: Arc<RwLock<ChunkMapData<I>>>,
    _marker: PhantomData<C>,
}

impl<C: VoxelWorldConfig, I: Copy> ChunkMap<C, I> {
    pub fn get(
        key: &ChunkKey,
        read_lock: &RwLockReadGuard<ChunkMapData<I>>,
    ) -> Option<chunk::ChunkData<I>> {
        read_lock.get(key).cloned()
    }

    pub fn contains_chunk(
        key: &ChunkKey,
        read_lock: &RwLockReadGuard<ChunkMapData<I>>,
    ) -> bool {
        read_lock.get(key).is_some()
    }

    /// Get the current bounding box of loaded chunks in this map.
    ///
    /// Expressed in **chunk coordinates**. Bounds are **inclusive**.
    pub fn get_bounds(read_lock: &RwLockReadGuard<ChunkMapData<I>>) -> Aabb3d {
        read_lock.bounds_in(GridId::WORLD)
    }

    /// Get the current bounding box of loaded chunks in this map.
    ///
    /// Expressed in **world units**. Bounds are **inclusive**.
    pub fn get_world_bounds(read_lock: &RwLockReadGuard<ChunkMapData<I>>) -> Aabb3d {
        let mut world_bounds = ChunkMap::<C, I>::get_bounds(read_lock);
        world_bounds.min *= CHUNK_SIZE_F * VOXEL_SIZE;
        world_bounds.max = (world_bounds.max + Vec3A::ONE) * CHUNK_SIZE_F * VOXEL_SIZE;
        world_bounds
    }

    pub fn get_read_lock(&self) -> RwLockReadGuard<'_, ChunkMapData<I>> {
        #[cfg(feature = "trace")]
        let _span = bevy::log::info_span!("wait_chunk_map_read").entered();
        self.map.read().unwrap()
    }

    pub fn get_map(&self) -> Arc<RwLock<ChunkMapData<I>>> {
        self.map.clone()
    }

    pub(crate) fn apply_buffers(
        &self,
        update_buffer: &mut ChunkMapUpdateBuffer<C, I>,
        remove_buffer: &mut ChunkMapRemoveBuffer<C>,
        ev_chunk_published: &mut MessageWriter<ChunkPublished<C>>,
        ev_chunk_will_spawn: &mut MessageWriter<ChunkWillSpawn<C>>,
    ) {
        if update_buffer.is_empty() && remove_buffer.is_empty() {
            return;
        }

        if let Ok(mut write_lock) = self.map.try_write() {
            for (key, chunk_data, entity) in update_buffer.iter() {
                let previous = write_lock.insert(
                    *key,
                    ChunkData {
                        key: key.key(),
                        ..chunk_data.clone()
                    },
                );
                // A chunk is published for the first time when the key has no
                // data, or still holds the data of a despawned chunk.
                if previous.map(|data| data.entity) != Some(*entity) {
                    ev_chunk_will_spawn.write(ChunkWillSpawn::new(key.key(), *entity));
                }
                ev_chunk_published.write(ChunkPublished::new(key.key(), *entity));
            }
            update_buffer.clear();

            let mut rebuild: Vec<GridId> = Vec::new();
            for (key, entity) in remove_buffer.iter() {
                // The chunk may have spawned again and published while this
                // removal waited for the lock. Its data stays.
                if write_lock.get_in(key).map(|data| data.entity) != Some(*entity) {
                    continue;
                }
                if write_lock.remove(key) && !rebuild.contains(&key.grid()) {
                    rebuild.push(key.grid());
                }
            }
            remove_buffer.clear();

            for grid in rebuild {
                write_lock.rebuild_bounds(grid);
            }
        }
    }
}

/// The chunk's minimum and maximum corners in chunk coordinates, inclusive.
fn chunk_corners(key: &ChunkKey) -> (Vec3A, Vec3A) {
    (
        Vec3A::from(key.position.as_vec3()),
        Vec3A::from((key.position + key.extent() - 1).as_vec3()),
    )
}

impl<C, I> Default for ChunkMap<C, I> {
    fn default() -> Self {
        Self {
            map: Arc::new(RwLock::new(ChunkMapData {
                data: GridTable::with_capacity(1000),
                levels: PerGrid::new(LevelCounts::new()),
                bounds: PerGrid::new(EMPTY_BOUNDS),
            })),
            _marker: PhantomData,
        }
    }
}

#[derive(Resource, Deref, DerefMut, Default)]
pub(crate) struct ChunkMapUpdateBuffer<C: VoxelWorldConfig, I>(
    #[deref] Vec<(GridKey, chunk::ChunkData<I>, Entity)>,
    PhantomData<C>,
);

/// The despawned chunks whose data is still to be removed, by grid key and
/// entity.
#[derive(Resource, Deref, DerefMut, Default)]
pub(crate) struct ChunkMapRemoveBuffer<C>(
    #[deref] Vec<(GridKey, Entity)>,
    PhantomData<C>,
);
