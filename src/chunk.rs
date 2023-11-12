use bevy::{prelude::*, tasks::Task};
use ndshape::{ConstShape, ConstShape3u32};
use std::sync::Arc;

use crate::{meshing, voxel::WorldVoxel, voxel_world_internal::ModifiedVoxels};

// The size of a chunk in voxels
// TODO: implement a way to change this though the configuration
pub const CHUNK_SIZE_U: u32 = 32;
pub const CHUNK_SIZE_I: i32 = CHUNK_SIZE_U as i32;
pub const CHUNK_SIZE_F: f32 = CHUNK_SIZE_U as f32;

// A chunk with 1-voxel boundary padding.
pub(crate) const PADDED_CHUNK_SIZE: u32 = CHUNK_SIZE_U + 2;
pub(crate) type PaddedChunkShape =
    ConstShape3u32<PADDED_CHUNK_SIZE, PADDED_CHUNK_SIZE, PADDED_CHUNK_SIZE>;

pub(crate) type VoxelArray = [WorldVoxel; PaddedChunkShape::SIZE as usize];

#[derive(Component)]
#[component(storage = "SparseSet")]
pub(crate) struct ChunkThread(pub Task<ChunkTask>);

impl ChunkThread {
    pub fn new(task: Task<ChunkTask>) -> Self {
        Self(task)
    }
}

#[derive(Component)]
#[component(storage = "SparseSet")]
pub struct NeedsRemesh;

#[derive(Component)]
pub struct NeedsDespawn;

/// This is used to lookup voxel data from spawned chunks. Does not persist after
/// the chunk is despawned.
#[derive(Clone)]
pub struct ChunkData {
    pub voxels: Arc<VoxelArray>,
    pub voxels_hash: u64,
    pub is_full: bool,
    pub entity: Entity,
}

/// A marker component for chunks, with some helpful data
#[derive(Component, Clone)]
pub struct Chunk {
    pub position: IVec3,
    pub entity: Entity,
}

impl Chunk {
    pub fn from(chunk: &Chunk) -> Self {
        Self {
            position: chunk.position,
            entity: chunk.entity,
        }
    }
}

/// Holds all data needed to generate and mesh a chunk
#[derive(Component)]
pub(crate) struct ChunkTask {
    pub position: IVec3,
    pub voxels: Arc<VoxelArray>,
    pub voxels_hash: u64,
    pub modified_voxels: ModifiedVoxels,
    pub is_empty: bool,
    pub is_full: bool,
    pub mesh: Option<Mesh>,
}

impl ChunkTask {
    /// Generate voxel data for the chunk. The supplied `modified_voxels` map is first checked,
    /// and where no voxeles are modified, the `voxel_data_fn` is called to get data from the
    /// consumer.
    pub fn generate<F>(&mut self, mut voxel_data_fn: F)
    where
        F: FnMut(IVec3) -> WorldVoxel + Send + 'static,
    {
        let mut filled_count = 0;
        let modified_voxels = (*self.modified_voxels).read().unwrap();
        let mut voxels = [WorldVoxel::Unset; PaddedChunkShape::SIZE as usize];

        for i in 0..PaddedChunkShape::SIZE {
            let chunk_block = PaddedChunkShape::delinearize(i);

            let block_pos = IVec3 {
                x: chunk_block[0] as i32 + (self.position.x * CHUNK_SIZE_I) - 1,
                y: chunk_block[1] as i32 + (self.position.y * CHUNK_SIZE_I) - 1,
                z: chunk_block[2] as i32 + (self.position.z * CHUNK_SIZE_I) - 1,
            };

            if let Some(voxel) = modified_voxels.get(&block_pos) {
                voxels[i as usize] = *voxel;
                if !voxel.is_unset() && !voxel.is_air() {
                    filled_count += 1;
                }
                continue;
            }

            let voxel = voxel_data_fn(block_pos);

            voxels[i as usize] = voxel;

            if let WorldVoxel::Solid(_) = voxel {
                filled_count += 1;
            }
        }

        self.voxels = Arc::new(voxels);

        self.is_empty = filled_count == 0;
        self.is_full = filled_count == PaddedChunkShape::SIZE;
    }

    /// Generate a mesh for the chunk based on the currect voxel data
    pub fn mesh(&mut self, texture_index_mapper: Arc<dyn Fn(u8) -> [u32; 3] + Send + Sync>) {
        if self.mesh.is_none() {
            self.mesh = Some(meshing::generate_chunk_mesh(
                self.voxels.clone(),
                self.position,
                texture_index_mapper,
            ));
        }
    }
}
