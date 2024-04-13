use bevy::{prelude::*, render::primitives::Aabb, tasks::Task, utils::HashSet};
use ndshape::{ConstShape, ConstShape3u32};
use std::{
    hash::{Hash, Hasher},
    marker::PhantomData,
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
pub(crate) struct ChunkThread<C>(pub Task<ChunkTask<C>>, PhantomData<C>);

impl<C> ChunkThread<C>
where
    C: Send + Sync + 'static,
{
    pub fn new(task: Task<ChunkTask<C>>, _pos: IVec3) -> Self {
        Self(task, PhantomData)
    }
}

#[derive(Component)]
#[component(storage = "SparseSet")]
pub struct NeedsRemesh;

#[derive(Component)]
pub struct NeedsDespawn;

#[derive(Clone, Debug)]
pub enum FillType {
    Empty,
    Mixed,
    Uniform(WorldVoxel),
}

/// This is used to lookup voxel data from spawned chunks. Does not persist after
/// the chunk is despawned.
#[derive(Clone, Debug)]
pub struct ChunkData {
    pub position: IVec3,
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
            position: IVec3::ZERO,
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

    pub fn world_position(&self) -> Vec3 {
        self.position.as_vec3() * CHUNK_SIZE_F
    }

    pub fn aabb(&self) -> Aabb {
        let min = Vec3::ZERO;
        let max = min + Vec3::splat(CHUNK_SIZE_F);
        Aabb::from_min_max(min, max)
    }

    pub fn encloses_point(&self, point: Vec3) -> bool {
        let local_point = point - self.world_position();
        let aabb = self.aabb();
        let min = aabb.min();
        let max = aabb.max();
        local_point.x >= min.x
            && local_point.y >= min.y
            && local_point.z >= min.z
            && local_point.x <= max.x
            && local_point.y <= max.y
            && local_point.z <= max.z
    }
}

impl Default for ChunkData {
    fn default() -> Self {
        Self::new()
    }
}

/// A marker component for chunks, with some helpful data
#[derive(Component, Clone)]
pub struct Chunk<C> {
    pub position: IVec3,
    pub entity: Entity,
    _marker: PhantomData<C>,
}

impl<C> Chunk<C> {
    pub fn new(position: IVec3, entity: Entity) -> Self {
        Self {
            position,
            entity,
            _marker: PhantomData,
        }
    }

    pub fn from(chunk: &Chunk<C>) -> Self {
        Self {
            position: chunk.position,
            entity: chunk.entity,
            _marker: PhantomData,
        }
    }

    pub fn aabb(&self) -> Aabb {
        let min = Vec3::ZERO;
        let max = min + Vec3::splat(CHUNK_SIZE_F);
        Aabb::from_min_max(min, max)
    }
}

/// Holds all data needed to generate and mesh a chunk
#[derive(Component)]
pub(crate) struct ChunkTask<C> {
    pub position: IVec3,
    pub chunk_data: ChunkData,
    pub modified_voxels: ModifiedVoxels<C>,
    pub mesh: Option<Mesh>,
    _marker: PhantomData<C>,
}

impl<C: Send + Sync + 'static> ChunkTask<C> {
    pub fn new(entity: Entity, position: IVec3, modified_voxels: ModifiedVoxels<C>) -> Self {
        Self {
            position,
            chunk_data: ChunkData::with_entity(entity),
            modified_voxels,
            mesh: None,
            _marker: PhantomData,
        }
    }

    /// Generate voxel data for the chunk. The supplied `modified_voxels` map is first checked,
    /// and where no voxels are modified, the `voxel_data_fn` is called to get data from the
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

    /// Generate a mesh for the chunk based on the correct voxel data
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
