use std::{
    marker::PhantomData,
    sync::{Arc, RwLock, RwLockReadGuard},
};

use crate::{
    chunk::{self, ChunkData, ChunkKey, LevelCounts, CHUNK_SIZE_F},
    configuration::VoxelWorldConfig,
    voxel::VOXEL_SIZE,
    voxel_world::{ChunkPublished, ChunkWillSpawn},
};
use bevy::{math::Vec3A, prelude::*};
use hashbrown::HashMap;

#[derive(Deref, DerefMut)]
pub struct ChunkMapData<I> {
    #[deref]
    data: HashMap<ChunkKey, chunk::ChunkData<I>>,
    levels: LevelCounts,
    bounds: Aabb3d,
}

impl<I> ChunkMapData<I> {
    /// The published data of the finest chunk that covers the voxel at
    /// `voxel_position`, if any chunk does.
    pub fn finest_covering(&self, voxel_position: IVec3) -> Option<&chunk::ChunkData<I>> {
        self.covering(voxel_position).next()
    }

    /// The published data of every chunk that covers the voxel at
    /// `voxel_position`, finest first.
    pub fn covering(
        &self,
        voxel_position: IVec3,
    ) -> impl Iterator<Item = &chunk::ChunkData<I>> + '_ {
        self.levels.in_use().filter_map(move |lod_level| {
            self.data
                .get(&ChunkKey::containing(voxel_position, lod_level))
        })
    }
}

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
        read_lock.data.get(key).cloned()
    }

    pub fn contains_chunk(
        key: &ChunkKey,
        read_lock: &RwLockReadGuard<ChunkMapData<I>>,
    ) -> bool {
        read_lock.data.contains_key(key)
    }

    /// Get the current bounding box of loaded chunks in this map.
    ///
    /// Expressed in **chunk coordinates**. Bounds are **inclusive**.
    pub fn get_bounds(read_lock: &RwLockReadGuard<ChunkMapData<I>>) -> Aabb3d {
        read_lock.bounds
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
                let previous = write_lock.data.insert(
                    *key,
                    ChunkData {
                        key: *key,
                        ..chunk_data.clone()
                    },
                );

                let (min, max) = chunk_corners(key);
                if min.cmplt(write_lock.bounds.min).any() {
                    write_lock.bounds.min = min.min(write_lock.bounds.min);
                }
                if max.cmpgt(write_lock.bounds.max).any() {
                    write_lock.bounds.max = max.max(write_lock.bounds.max);
                }

                if previous.is_none() {
                    write_lock.levels.add(key.lod_level);
                }
                // A chunk is published for the first time when the key has no
                // data, or still holds the data of a despawned chunk.
                if previous.map(|data| data.entity) != Some(*entity) {
                    ev_chunk_will_spawn.write(ChunkWillSpawn::new(*key, *entity));
                }
                ev_chunk_published.write(ChunkPublished::new(*key, *entity));
            }
            update_buffer.clear();

            let mut need_rebuild_aabb = false;
            for (key, entity) in remove_buffer.iter() {
                // The chunk may have spawned again and published while this
                // removal waited for the lock. Its data stays.
                if write_lock.data.get(key).map(|data| data.entity) != Some(*entity) {
                    continue;
                }
                write_lock.data.remove(key);
                write_lock.levels.remove(key.lod_level);

                let (min, max) = chunk_corners(key);
                need_rebuild_aabb |= write_lock.bounds.min.floor().cmpeq(min).any()
                    || write_lock.bounds.max.floor().cmpeq(max).any();
            }
            remove_buffer.clear();

            if write_lock.data.is_empty() {
                write_lock.bounds = Aabb3d::new(Vec3::ZERO, Vec3::ZERO);
            } else if need_rebuild_aabb {
                let mut corners = Vec::with_capacity(2 * write_lock.data.len());
                for key in write_lock.data.keys() {
                    let (min, max) = chunk_corners(key);
                    corners.push(min);
                    corners.push(max);
                }
                write_lock.bounds =
                    Aabb3d::from_point_cloud(Isometry3d::IDENTITY, corners.drain(0..));
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
                data: HashMap::with_capacity(1000),
                levels: LevelCounts::new(),
                bounds: Aabb3d::new(Vec3::ZERO, Vec3::ZERO),
            })),
            _marker: PhantomData,
        }
    }
}

#[derive(Resource, Deref, DerefMut, Default)]
pub(crate) struct ChunkMapUpdateBuffer<C: VoxelWorldConfig, I>(
    #[deref] Vec<(ChunkKey, chunk::ChunkData<I>, Entity)>,
    PhantomData<C>,
);

/// The despawned chunks whose data is still to be removed, by chunk key and
/// entity.
#[derive(Resource, Deref, DerefMut, Default)]
pub(crate) struct ChunkMapRemoveBuffer<C>(
    #[deref] Vec<(ChunkKey, Entity)>,
    PhantomData<C>,
);
