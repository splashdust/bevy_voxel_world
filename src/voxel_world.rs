///
/// VoxelWorld
/// This module implements most of the public API for bevy_voxel_world.
///
use bevy::{ecs::system::SystemParam, prelude::*};
use ndshape::ConstShape;
use std::sync::Arc;

use crate::{
    chunk,
    voxel::WorldVoxel,
    voxel_world_internal::{get_chunk_voxel_position, ChunkMap, ModifiedVoxels, VoxelWriteBuffer},
};

/// This component is used to mark the Camera that bevy_voxel_world should use to determine
/// which chunks to spawn and despawn.
#[derive(Component)]
pub struct VoxelWorldCamera;

/// Fired when a chunk is about to be despawned.
#[derive(Event)]
pub struct ChunkWillDespawn {
    pub chunk_key: IVec3,
    pub entity: Entity,
}

/// Fired when a chunk is about to be spawned.
#[derive(Event)]
pub struct ChunkWillSpawn {
    pub chunk_key: IVec3,
    pub entity: Entity,
}

/// Fired when a chunk is about to be remeshed.
#[derive(Event)]
pub struct ChunkWillRemesh {
    pub chunk_key: IVec3,
    pub entity: Entity,
}

/// Grants access to the VoxelWorld in systems
#[derive(SystemParam)]
pub struct VoxelWorld<'w> {
    chunk_map: Res<'w, ChunkMap>,
    modified_voxels: Res<'w, ModifiedVoxels>,
    voxel_write_buffer: ResMut<'w, VoxelWriteBuffer>,
}

impl<'w> VoxelWorld<'w> {
    /// Get the voxel at the given position, or None if there is no voxel at that position
    pub fn get_voxel(&self, position: IVec3) -> WorldVoxel {
        self.get_voxel_fn()(position)
    }

    /// Set the voxel at the given position. This will create a new chunk if one does not exist at
    /// the given position.
    pub fn set_voxel(&mut self, position: IVec3, voxel: WorldVoxel) {
        self.voxel_write_buffer.push((position, voxel));
    }

    /// Get a sendable closure that can be used to get the voxel at the given position
    /// This is useful for spawning tasks that need to access the voxel world
    pub fn get_voxel_fn(&self) -> Arc<dyn Fn(IVec3) -> WorldVoxel + Send + Sync> {
        let chunk_map = self.chunk_map.get_map();
        let write_buffer = self.voxel_write_buffer.clone();
        let modified_voxels = self.modified_voxels.clone();

        Arc::new(move |position| {
            let (chunk_pos, vox_pos) = get_chunk_voxel_position(position);

            if let Some(voxel) = write_buffer
                .iter()
                .find(|(pos, _)| *pos == position)
                .map(|(_, voxel)| *voxel)
            {
                return voxel;
            }

            {
                if let Some(voxel) = modified_voxels.get_voxel(&position) {
                    return voxel;
                }
            }

            let chunk_opt = {
                let chun_map_read = chunk_map.read().unwrap();
                chun_map_read.get(&chunk_pos).cloned()
            };

            if let Some(chunk_data) = chunk_opt {
                let i = chunk::PaddedChunkShape::linearize(vox_pos.to_array()) as usize;
                chunk_data.voxels[i]
            } else {
                WorldVoxel::Unset
            }
        })
    }

    /// Get the closes surface voxel to the given position
    /// Returns None if there is no surface voxel at or below the given position
    pub fn get_closest_surface_voxel(&self, position: IVec3) -> Option<(IVec3, WorldVoxel)> {
        let get_voxel = self.get_voxel_fn();
        let mut current_pos = position;
        let current_voxel = get_voxel(current_pos);

        let is_surface = |pos: IVec3| {
            let above = pos + IVec3::Y;
            (get_voxel(pos) != WorldVoxel::Unset && get_voxel(pos) != WorldVoxel::Air)
                && (get_voxel(above) == WorldVoxel::Unset || get_voxel(above) == WorldVoxel::Air)
        };

        if current_voxel == WorldVoxel::Unset || current_voxel == WorldVoxel::Air {
            while !is_surface(current_pos) {
                current_pos -= IVec3::Y;
                if current_pos.y < -256 {
                    return None;
                }
            }

            return Some((current_pos, get_voxel(current_pos)));
        }

        None
    }

    /// Get a randowm surface voxel within the given radius of the given position
    /// Returns None if no surface voxel was found within the given radius
    pub fn get_random_surface_voxel(
        &self,
        position: IVec3,
        radius: u32,
    ) -> Option<(IVec3, WorldVoxel)> {
        let mut tries = 0;

        while tries < 100 {
            tries += 1;

            let r = radius as f32;
            let x = rand::random::<f32>() * r * 2.0 - r;
            let y = rand::random::<f32>() * r * 2.0 - r;
            let z = rand::random::<f32>() * r * 2.0 - r;

            if y < 0.0 {
                continue;
            }

            let d = x * x + y * y + z * z;
            if d > r * r {
                continue;
            }

            let pos = position + IVec3::new(x as i32, y as i32, z as i32);
            if let Some(result) = self.get_closest_surface_voxel(pos) {
                return Some(result);
            }
        }

        None
    }

    /// Get first surface voxel at the given Vec2 position
    pub fn get_surface_voxel_at_2d_pos(&self, pos_2d: Vec2) -> Option<(IVec3, WorldVoxel)> {
        self.get_closest_surface_voxel(IVec3 {
            x: pos_2d.x.floor() as i32,
            y: 256,
            z: pos_2d.y.floor() as i32,
        })
    }
}
