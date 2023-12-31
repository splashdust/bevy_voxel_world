use bevy::{prelude::*, tasks::Task, utils::HashSet};
use ndshape::{ConstShape, ConstShape3u32};
use std::{
    hash::{Hash, Hasher},
    sync::Arc,
};

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
pub(crate) struct ChunkThread(pub Task<ChunkTask>, pub IVec3);

impl ChunkThread {
    pub fn new(task: Task<ChunkTask>, pos: IVec3) -> Self {
        Self(task, pos)
    }
}

#[derive(Component)]
#[component(storage = "SparseSet")]
pub struct NeedsRemesh;

#[derive(Component)]
pub struct NeedsDespawn;

#[derive(Clone)]
pub enum FillType {
    Empty,
    Mixed,
    Uniform(WorldVoxel),
}

/// This is used to lookup voxel data from spawned chunks. Does not persist after
/// the chunk is despawned.
#[derive(Clone)]
pub struct ChunkData {
    pub voxels: Option<Arc<VoxelArray>>,
    pub voxels_hash: u64,
    pub is_full: bool,
    pub is_empty: bool,
    pub fill_type: FillType,
    pub entity: Entity,
}

impl ChunkData {
    pub fn new() -> Self {
        Self {
            voxels: None,
            voxels_hash: 0,
            is_full: false,
            is_empty: true,
            fill_type: FillType::Empty,
            entity: Entity::PLACEHOLDER,
        }
    }

    pub fn with_entity(entity: Entity) -> Self {
        let new = Self::new();
        Self { entity, ..new }
    }

    pub fn generate_hash(&mut self) {
        if let Some(voxels) = &self.voxels {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            voxels.hash(&mut hasher);
            self.voxels_hash = hasher.finish();
        }
    }

    pub fn get_voxel(&self, position: UVec3) -> WorldVoxel {
        if self.voxels.is_some() {
            self.voxels.as_ref().unwrap()[PaddedChunkShape::linearize(position.to_array()) as usize]
        } else {
            match self.fill_type {
                FillType::Uniform(voxel) => voxel,
                FillType::Empty => WorldVoxel::Unset,
                FillType::Mixed => unreachable!(),
            }
        }
    }
}

impl Default for ChunkData {
    fn default() -> Self {
        Self::new()
    }
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
    pub chunk_data: ChunkData,
    pub modified_voxels: ModifiedVoxels,
    pub mesh: Option<Mesh>,
}

impl ChunkTask {
    pub fn new(entity: Entity, position: IVec3, modified_voxels: ModifiedVoxels) -> Self {
        Self {
            position,
            chunk_data: ChunkData::with_entity(entity),
            modified_voxels,
            mesh: None,
        }
    }

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
        let mut material_count = HashSet::new();

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

            if let WorldVoxel::Solid(m) = voxel {
                filled_count += 1;
                material_count.insert(m);
            }
        }

        self.chunk_data.is_empty = filled_count == 0;
        self.chunk_data.is_full = filled_count == PaddedChunkShape::SIZE;

        if self.chunk_data.is_full && material_count.len() == 1 {
            self.chunk_data.fill_type = FillType::Uniform(voxels[0]);
            self.chunk_data.voxels = None;
        } else if filled_count > 0 {
            self.chunk_data.fill_type = FillType::Mixed;
            self.chunk_data.voxels = Some(Arc::new(voxels));
        } else {
            self.chunk_data.fill_type = FillType::Empty;
            self.chunk_data.voxels = None;
        };

        self.chunk_data.generate_hash();
    }

    /// Generate a mesh for the chunk based on the currect voxel data
    pub fn mesh(&mut self, texture_index_mapper: Arc<dyn Fn(u8) -> [u32; 3] + Send + Sync>) {
        if self.mesh.is_none() && self.chunk_data.voxels.is_some() {
            self.mesh = Some(meshing::generate_chunk_mesh(
                self.chunk_data.voxels.as_ref().unwrap().clone(),
                self.position,
                texture_index_mapper,
            ));
        }
    }

    pub fn is_empty(&self) -> bool {
        self.chunk_data.is_empty
    }

    pub fn is_full(&self) -> bool {
        self.chunk_data.is_full
    }

    pub fn voxels_hash(&self) -> u64 {
        self.chunk_data.voxels_hash
    }
}
