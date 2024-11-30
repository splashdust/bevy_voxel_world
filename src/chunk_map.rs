use std::{
    marker::PhantomData,
    sync::{Arc, RwLock, RwLockReadGuard},
};

use crate::{
    chunk::{self, ChunkData, CHUNK_SIZE_F},
    voxel::VOXEL_SIZE,
    voxel_world::ChunkWillSpawn,
};
use bevy::{
    math::{bounding::Aabb3d, Vec3A},
    prelude::*,
    utils::hashbrown::HashMap,
};

#[derive(Deref, DerefMut)]
pub struct ChunkMapData<I> {
    #[deref]
    data: HashMap<IVec3, chunk::ChunkData<I>>,
    bounds: Aabb3d,
}

/// Holds a map of all chunks that are currently spawned spawned
/// The chunks also exist as entities that can be queried in the ECS,
/// but having this map in addition allows for faster spatial lookups
#[derive(Resource)]
pub struct ChunkMap<C, I> {
    map: Arc<RwLock<ChunkMapData<I>>>,
    _marker: PhantomData<C>,
}

impl<C: Send + Sync + 'static, I: Copy> ChunkMap<C, I> {
    pub fn get(
        position: &IVec3,
        read_lock: &RwLockReadGuard<ChunkMapData<I>>,
    ) -> Option<chunk::ChunkData<I>> {
        read_lock.data.get(position).cloned()
    }

    pub fn contains_chunk(position: &IVec3, read_lock: &RwLockReadGuard<ChunkMapData<I>>) -> bool {
        read_lock.data.contains_key(position)
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

    pub fn get_read_lock(&self) -> RwLockReadGuard<ChunkMapData<I>> {
        self.map.read().unwrap()
    }

    pub fn get_map(&self) -> Arc<RwLock<ChunkMapData<I>>> {
        self.map.clone()
    }

    pub(crate) fn apply_buffers(
        &self,
        insert_buffer: &mut ChunkMapInsertBuffer<C, I>,
        update_buffer: &mut ChunkMapUpdateBuffer<C, I>,
        remove_buffer: &mut ChunkMapRemoveBuffer<C>,
        ev_chunk_will_spawn: &mut EventWriter<ChunkWillSpawn<C>>,
    ) {
        if insert_buffer.is_empty() && update_buffer.is_empty() && remove_buffer.is_empty() {
            return;
        }

        if let Ok(mut write_lock) = self.map.try_write() {
            for (position, chunk_data) in insert_buffer.iter() {
                write_lock.data.insert(
                    *position,
                    ChunkData {
                        position: *position,
                        ..chunk_data.clone()
                    },
                );

                let position_f = Vec3A::from(position.as_vec3());
                if position_f.cmplt(write_lock.bounds.min).any() {
                    write_lock.bounds.min = position_f.min(write_lock.bounds.min);
                } else if position_f.cmpgt(write_lock.bounds.max).any() {
                    write_lock.bounds.max = position_f.max(write_lock.bounds.max);
                }
            }
            insert_buffer.clear();

            for (position, chunk_data, evt) in update_buffer.iter() {
                write_lock.data.insert(
                    *position,
                    ChunkData {
                        position: *position,
                        ..chunk_data.clone()
                    },
                );

                let position_f = Vec3A::from(position.as_vec3());
                if position_f.cmplt(write_lock.bounds.min).any() {
                    write_lock.bounds.min = position_f.min(write_lock.bounds.min);
                } else if position_f.cmpgt(write_lock.bounds.max).any() {
                    write_lock.bounds.max = position_f.max(write_lock.bounds.max);
                }

                ev_chunk_will_spawn.send((*evt).clone());
            }
            update_buffer.clear();

            let mut need_rebuild_aabb = false;
            for position in remove_buffer.iter() {
                write_lock.data.remove(position);

                need_rebuild_aabb = write_lock.bounds.min.floor().as_ivec3() == *position
                    || write_lock.bounds.max.floor().as_ivec3() == *position;
            }
            remove_buffer.clear();

            if need_rebuild_aabb {
                let mut tmp_vec = Vec::with_capacity(write_lock.data.len());
                for v in write_lock.data.keys() {
                    tmp_vec.push(Vec3A::from(v.as_vec3()));
                }
                write_lock.bounds =
                    Aabb3d::from_point_cloud(Isometry3d::IDENTITY, tmp_vec.drain(0..));
            }
        }
    }
}

impl<C, I> Default for ChunkMap<C, I> {
    fn default() -> Self {
        Self {
            map: Arc::new(RwLock::new(ChunkMapData {
                data: HashMap::with_capacity(1000),
                bounds: Aabb3d::new(Vec3::ZERO, Vec3::ZERO),
            })),
            _marker: PhantomData,
        }
    }
}

#[derive(Resource, Deref, DerefMut, Default, Debug)]
pub(crate) struct ChunkMapInsertBuffer<C, I>(
    #[deref] Vec<(IVec3, chunk::ChunkData<I>)>,
    PhantomData<C>,
);

#[derive(Resource, Deref, DerefMut, Default)]
pub(crate) struct ChunkMapUpdateBuffer<C, I>(
    #[deref] Vec<(IVec3, chunk::ChunkData<I>, ChunkWillSpawn<C>)>,
    PhantomData<C>,
);

#[derive(Resource, Deref, DerefMut, Default)]
pub(crate) struct ChunkMapRemoveBuffer<C>(#[deref] Vec<IVec3>, PhantomData<C>);
