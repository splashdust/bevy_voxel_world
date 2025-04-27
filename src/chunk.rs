use bevy::{prelude::*, render::primitives::Aabb, tasks::Task, platform::collections::HashSet};
use ndshape::{ConstShape, ConstShape3u32};
use std::{
    hash::{Hash, Hasher},
    marker::PhantomData,
    sync::Arc,
};

use crate::{
    prelude::{ChunkMeshingFn, TextureIndexMapperFn, VoxelWorldConfig},
    voxel::WorldVoxel,
    voxel_world_internal::ModifiedVoxels,
};

// The size of a chunk in voxels
// TODO: implement a way to change this though the configuration
pub const CHUNK_SIZE_U: u32 = 32;
pub const CHUNK_SIZE_I: i32 = CHUNK_SIZE_U as i32;
pub const CHUNK_SIZE_F: f32 = CHUNK_SIZE_U as f32;

// A chunk with 1-voxel boundary padding.
pub(crate) const PADDED_CHUNK_SIZE: u32 = CHUNK_SIZE_U + 2;
pub type PaddedChunkShape =
    ConstShape3u32<PADDED_CHUNK_SIZE, PADDED_CHUNK_SIZE, PADDED_CHUNK_SIZE>;

pub type VoxelArray<I> = [WorldVoxel<I>; PaddedChunkShape::SIZE as usize];

#[derive(Component)]
#[component(storage = "SparseSet")]
pub(crate) struct ChunkThread<C: VoxelWorldConfig, I>(
    pub Task<ChunkTask<C, I>>,
    PhantomData<C>,
);

impl<C, I> ChunkThread<C, I>
where
    C: VoxelWorldConfig,
{
    pub fn new(task: Task<ChunkTask<C, I>>, _pos: IVec3) -> Self {
        Self(task, PhantomData)
    }
}

#[derive(Component)]
#[component(storage = "SparseSet")]
pub struct NeedsRemesh;

#[derive(Component)]
pub struct NeedsDespawn;

#[derive(Clone, Debug)]
pub enum FillType<I> {
    Empty,
    Mixed,
    Uniform(WorldVoxel<I>),
}

/// This is used to lookup voxel data from spawned chunks. Does not persist after
/// the chunk is despawned.
#[derive(Clone, Debug)]
pub struct ChunkData<I> {
    pub(crate) position: IVec3,
    pub(crate) voxels: Option<Arc<VoxelArray<I>>>,
    pub(crate) voxels_hash: u64,
    pub(crate) is_full: bool,
    pub(crate) is_empty: bool,
    pub(crate) fill_type: FillType<I>,
    pub(crate) entity: Entity,
    pub(crate) has_generated: bool,
}

impl<I: Hash + Copy + PartialEq> ChunkData<I> {
    pub(crate) fn new() -> Self {
        Self {
            position: IVec3::ZERO,
            voxels: None,
            voxels_hash: 0,
            is_full: false,
            is_empty: true,
            fill_type: FillType::Empty,
            entity: Entity::PLACEHOLDER,
            has_generated: false,
        }
    }

    pub(crate) fn with_entity(entity: Entity) -> Self {
        let new = Self::new();
        Self { entity, ..new }
    }

    pub(crate) fn generate_hash(&mut self) {
        if let Some(voxels) = &self.voxels {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            voxels.hash(&mut hasher);
            self.voxels_hash = hasher.finish();
        }
    }

    /// Get the voxel at the given position in the chunk
    /// The position is given in local chunk coordinates
    pub fn get_voxel(&self, position: UVec3) -> WorldVoxel<I> {
        if self.voxels.is_some() {
            self.voxels.as_ref().unwrap()
                [PaddedChunkShape::linearize(position.to_array()) as usize]
        } else {
            match self.fill_type {
                FillType::Uniform(voxel) => voxel,
                FillType::Empty => WorldVoxel::Unset,
                FillType::Mixed => unreachable!(),
            }
        }
    }

    /// Returns true if the chunk is full. No mesh will be generated for full chunks.
    pub fn is_full(&self) -> bool {
        self.is_full
    }

    /// Returns true if the chunk is empty. No mesh will be generated for empty chunks.
    pub fn is_empty(&self) -> bool {
        self.is_empty
    }

    /// Returns the fill type of the chunk.
    /// This is used to determine the type of content in the chunk.
    ///
    /// - FillType::Empty - The chunk is completely empty
    /// - FillType::Mixed - The chunk contains a mix of different voxels, either different materials or air
    /// - FillType::Uniform(WorldVoxel) - The chunk is full and contains only one type of voxel. The type can be retrieved from contained WorldVoxel
    pub fn get_fill_type(&self) -> &FillType<I> {
        &self.fill_type
    }

    /// Returns the entity of the corresponding Chunk
    pub fn get_entity(&self) -> Entity {
        self.entity
    }

    /// Returns the position of the chunk in world coordinates
    pub fn world_position(&self) -> Vec3 {
        self.position.as_vec3() * CHUNK_SIZE_F
    }

    /// Returns the AABB of the chunk
    pub fn aabb(&self) -> Aabb {
        let min = Vec3::ZERO;
        let max = min + Vec3::splat(CHUNK_SIZE_F);
        Aabb::from_min_max(min, max)
    }

    /// Returns true if the given point is inside the chunk
    /// The point is given in world coordinates
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

    /// Returns true if the given voxel is within the bounds of the chunk
    /// and the voxel data at the given position matcheso the given voxel
    pub fn has_voxel(&self, voxel_pos: IVec3, voxel: WorldVoxel<I>) -> bool {
        let chunk_pos = voxel_pos / CHUNK_SIZE_I;
        if self.position != chunk_pos {
            return false;
        }
        self.get_voxel(voxel_pos.as_uvec3() % CHUNK_SIZE_U) == voxel
    }

    /// Returns true if this chunk has been processed by the voxel generation system (typically to generate terrain)
    /// Before generation has happened, voxel data in the chunk is not initialized.
    pub fn has_generated(&self) -> bool {
        self.has_generated
    }
}

impl<I: Hash + Copy + PartialEq> Default for ChunkData<I> {
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
pub(crate) struct ChunkTask<C, I>
where
    C: VoxelWorldConfig,
{
    pub position: IVec3,
    pub chunk_data: ChunkData<I>,
    pub modified_voxels: ModifiedVoxels<C, I>,
    pub mesh: Option<Mesh>,
    pub user_bundle: Option<C::ChunkUserBundle>,
    _marker: PhantomData<C>,
}

impl<C: VoxelWorldConfig + Send + Sync + 'static, I: Hash + Copy + Eq> ChunkTask<C, I> {
    pub fn new(
        entity: Entity,
        position: IVec3,
        modified_voxels: ModifiedVoxels<C, I>,
    ) -> Self {
        Self {
            position,
            chunk_data: ChunkData::with_entity(entity),
            modified_voxels,
            mesh: None,
            user_bundle: None,
            _marker: PhantomData,
        }
    }

    /// Generate voxel data for the chunk. The supplied `modified_voxels` map is first checked,
    /// and where no voxeles are modified, the `voxel_data_fn` is called to get data from the
    /// consumer.
    pub fn generate<F>(&mut self, mut voxel_data_fn: F)
    where
        F: FnMut(IVec3) -> WorldVoxel<I> + Send + 'static,
    {
        let mut filled_count = 0;
        let modified_voxels = (*self.modified_voxels).read().unwrap();
        let mut voxels = [WorldVoxel::Unset; PaddedChunkShape::SIZE as usize];
        let mut material_count = HashSet::new();

        self.chunk_data.has_generated = true;

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
    pub fn mesh(
        &mut self,
        mut chunk_meshing_fn: ChunkMeshingFn<I, C::ChunkUserBundle>,
        texture_index_mapper: TextureIndexMapperFn<I>,
    ) {
        if self.mesh.is_none() && self.chunk_data.voxels.is_some() {
            let mesh_and_bundle = chunk_meshing_fn(
                self.chunk_data.voxels.as_ref().unwrap().clone(),
                texture_index_mapper,
            );
            self.mesh = Some(mesh_and_bundle.0);
            self.user_bundle = mesh_and_bundle.1;
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
