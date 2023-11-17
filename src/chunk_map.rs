use std::sync::{Arc, RwLock, RwLockReadGuard};

use bevy::{prelude::*, utils::hashbrown::HashMap};

use crate::{chunk, voxel_world::ChunkWillSpawn};

/// Holds a map of all chunks that are currently spawned spawned
/// The chunks also exist as entities that can be queried in the ECS,
/// but having this map in addition allows for faster spatial lookups
#[derive(Resource)]
pub struct ChunkMap {
    map: Arc<RwLock<HashMap<IVec3, chunk::ChunkData>>>,
}

impl ChunkMap {
    pub fn get(
        position: &IVec3,
        read_lock: &RwLockReadGuard<HashMap<IVec3, chunk::ChunkData>>,
    ) -> Option<chunk::ChunkData> {
        read_lock.get(position).cloned()
    }

    pub fn contains_chunk(
        position: &IVec3,
        read_lock: &RwLockReadGuard<HashMap<IVec3, chunk::ChunkData>>,
    ) -> bool {
        read_lock.contains_key(position)
    }

    pub fn get_read_lock(&self) -> RwLockReadGuard<HashMap<IVec3, chunk::ChunkData>> {
        self.map.read().unwrap()
    }

    pub fn get_map(&self) -> Arc<RwLock<HashMap<IVec3, chunk::ChunkData>>> {
        self.map.clone()
    }

    pub(crate) fn apply_buffers(
        &self,
        insert_buffer: &mut ChunkMapInsertBuffer,
        update_buffer: &mut ChunkMapUpdateBuffer,
        remove_buffer: &mut ChunkMapRemoveBuffer,
        ev_chunk_will_spawn: &mut EventWriter<ChunkWillSpawn>,
    ) {
        if insert_buffer.is_empty() && update_buffer.is_empty() && remove_buffer.is_empty() {
            return;
        }

        if let Ok(mut write_lock) = self.map.try_write() {
            for (position, chunk_data) in insert_buffer.iter() {
                write_lock.insert(*position, chunk_data.clone());
            }
            insert_buffer.clear();

            for (position, chunk_data, evt) in update_buffer.iter() {
                write_lock.insert(*position, chunk_data.clone());
                ev_chunk_will_spawn.send((*evt).clone());
            }
            update_buffer.clear();

            for position in remove_buffer.iter() {
                write_lock.remove(position);
            }
            remove_buffer.clear();
        }
    }
}

impl Default for ChunkMap {
    fn default() -> Self {
        Self {
            map: Arc::new(RwLock::new(HashMap::with_capacity(1000))),
        }
    }
}

#[derive(Resource, Deref, DerefMut, Default)]
pub(crate) struct ChunkMapInsertBuffer(Vec<(IVec3, chunk::ChunkData)>);

#[derive(Resource, Deref, DerefMut, Default)]
pub(crate) struct ChunkMapUpdateBuffer(Vec<(IVec3, chunk::ChunkData, ChunkWillSpawn)>);

#[derive(Resource, Deref, DerefMut, Default)]
pub(crate) struct ChunkMapRemoveBuffer(Vec<IVec3>);
