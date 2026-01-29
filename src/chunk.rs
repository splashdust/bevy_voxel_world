use bevy::{
    math::bounding::Aabb3d, platform::collections::HashSet, prelude::*, tasks::Task,
};
use ndshape::{ConstShape3u32, RuntimeShape, Shape};
use std::{
    hash::{Hash, Hasher},
    marker::PhantomData,
    sync::Arc,
};

use crate::{
    prelude::{
        ChunkMeshingFn, ChunkRegenerateStrategy, LodLevel, TextureIndexMapperFn,
        VoxelWorldConfig,
    },
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

pub type VoxelArray<I> = Arc<[WorldVoxel<I>]>;

fn voxel_size_from_shape(shape: &RuntimeShape<u32, 3>) -> Vec3 {
    let [ex, ey, ez] = shape.as_array();
    let ix = (ex.saturating_sub(2)).max(1);
    let iy = (ey.saturating_sub(2)).max(1);
    let iz = (ez.saturating_sub(2)).max(1);

    Vec3::new(
        CHUNK_SIZE_F / ix as f32,
        CHUNK_SIZE_F / iy as f32,
        CHUNK_SIZE_F / iz as f32,
    )
}

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
    pub(crate) lod_level: LodLevel,
    pub(crate) voxels: Option<VoxelArray<I>>,
    pub(crate) voxels_hash: u64,
    pub(crate) is_full: bool,
    pub(crate) is_empty: bool,
    pub(crate) fill_type: FillType<I>,
    pub(crate) entity: Entity,
    pub(crate) has_generated: bool,
    pub(crate) data_shape: UVec3,
    pub(crate) mesh_shape: UVec3,
}

impl<I: Hash + Copy + PartialEq> ChunkData<I> {
    pub(crate) fn new() -> Self {
        Self {
            position: IVec3::ZERO,
            lod_level: 0,
            voxels: None,
            voxels_hash: 0,
            is_full: false,
            is_empty: true,
            fill_type: FillType::Empty,
            entity: Entity::PLACEHOLDER,
            has_generated: false,
            data_shape: UVec3::splat(PADDED_CHUNK_SIZE),
            mesh_shape: UVec3::splat(PADDED_CHUNK_SIZE),
        }
    }

    pub(crate) fn with_entity(entity: Entity) -> Self {
        let new = Self::new();
        Self { entity, ..new }
    }

    pub fn data_shape(&self) -> UVec3 {
        self.data_shape
    }

    pub fn mesh_shape(&self) -> UVec3 {
        self.mesh_shape
    }

    pub(crate) fn generate_hash(&mut self) {
        if let Some(voxels) = &self.voxels {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            self.data_shape.to_array().hash(&mut hasher);
            self.mesh_shape.to_array().hash(&mut hasher);
            voxels.hash(&mut hasher);
            self.voxels_hash = hasher.finish();
        } else {
            self.voxels_hash = 0;
        }
    }

    /// Get the voxel at the given position in the chunk.
    ///
    /// The `position` is expressed in chunk-local **data** coordinates, meaning it
    /// indexes directly into the padded `data_shape` associated with this chunk.
    /// When using non-default LOD shapes, these coordinates may no longer map
    /// 1:1 to world-space voxels. Prefer [`ChunkData::get_voxel_at_world_position`]
    /// when you need to query by a world position.
    pub fn get_voxel(&self, position: UVec3) -> WorldVoxel<I> {
        if let Some(voxels) = &self.voxels {
            let shape = RuntimeShape::<u32, 3>::new(self.data_shape.to_array());
            voxels[shape.linearize(position.to_array()) as usize]
        } else {
            match self.fill_type {
                FillType::Uniform(voxel) => voxel,
                FillType::Empty => WorldVoxel::Unset,
                FillType::Mixed => unreachable!(),
            }
        }
    }

    /// Returns the voxel at the given world-space position if this chunk contains it.
    pub fn get_voxel_at_world_position(
        &self,
        world_position: IVec3,
    ) -> Option<WorldVoxel<I>> {
        let shape = RuntimeShape::<u32, 3>::new(self.data_shape.to_array());
        let scale = voxel_size_from_shape(&shape);
        let chunk_origin = self.position * CHUNK_SIZE_I;
        let offset = world_position - chunk_origin;

        let to_index = |component: i32, scale: f32, max: u32| -> Option<u32> {
            if scale <= f32::EPSILON || max == 0 {
                return None;
            }

            let raw = ((component as f32) + 1.0) / scale;
            let chunk_block = raw.ceil();

            if chunk_block < 0.0 {
                return None;
            }

            let chunk_block_i = chunk_block as i32;

            if chunk_block_i < 0 || chunk_block_i >= max as i32 {
                return None;
            }

            let reconstructed = ((chunk_block - 1.0) * scale) as i32;

            if reconstructed != component {
                return None;
            }

            Some(chunk_block_i as u32)
        };

        let chunk_block = match (
            to_index(offset.x, scale.x, self.data_shape.x),
            to_index(offset.y, scale.y, self.data_shape.y),
            to_index(offset.z, scale.z, self.data_shape.z),
        ) {
            (Some(x), Some(y), Some(z)) => UVec3::new(x, y, z),
            _ => return None,
        };

        Some(self.get_voxel(chunk_block))
    }

    /// Returns true if the chunk is full. No mesh will be generated for full chunks.
    pub fn is_full(&self) -> bool {
        self.is_full
    }

    /// Returns true if the chunk is empty. No mesh will be generated for empty chunks.
    pub fn is_empty(&self) -> bool {
        self.is_empty
    }

    /// Returns the chunk-space position of this chunk.
    pub fn chunk_position(&self) -> IVec3 {
        self.position
    }

    /// Returns the LOD level this chunk was generated with.
    pub fn lod_level(&self) -> LodLevel {
        self.lod_level
    }

    /// Returns the hash of the voxel payload, if any.
    pub fn voxels_hash(&self) -> u64 {
        self.voxels_hash
    }

    /// Returns a clone of the voxel array, if the chunk stores explicit voxels.
    pub fn voxels_arc(&self) -> Option<VoxelArray<I>> {
        self.voxels.as_ref().map(Arc::clone)
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
    pub fn aabb(&self) -> Aabb3d {
        let min = Vec3::ZERO;
        let max = min + Vec3::splat(CHUNK_SIZE_F);
        Aabb3d {
            min: min.into(),
            max: max.into(),
        }
    }

    /// Returns true if the given point is inside the chunk
    /// The point is given in world coordinates
    pub fn encloses_point(&self, point: Vec3) -> bool {
        let local_point = point - self.world_position();
        let aabb = self.aabb();
        let min: Vec3 = aabb.min.into();
        let max: Vec3 = aabb.max.into();
        local_point.x >= min.x
            && local_point.y >= min.y
            && local_point.z >= min.z
            && local_point.x <= max.x
            && local_point.y <= max.y
            && local_point.z <= max.z
    }

    /// Returns true if the given voxel is within the bounds of the chunk
    /// and the voxel data at the given position matches the given voxel
    pub fn has_voxel(&self, voxel_pos: IVec3, voxel: WorldVoxel<I>) -> bool {
        let chunk_pos = voxel_pos / CHUNK_SIZE_I;
        if self.position != chunk_pos {
            return false;
        }
        self.get_voxel_at_world_position(voxel_pos) == Some(voxel)
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
    pub lod_level: LodLevel,
    pub entity: Entity,
    pub data_shape: UVec3,
    pub mesh_shape: UVec3,
    _marker: PhantomData<C>,
}

impl<C> Chunk<C> {
    pub fn new(
        position: IVec3,
        lod_level: LodLevel,
        entity: Entity,
        data_shape: UVec3,
        mesh_shape: UVec3,
    ) -> Self {
        Self {
            position,
            lod_level,
            entity,
            data_shape,
            mesh_shape,
            _marker: PhantomData,
        }
    }

    pub fn from(chunk: &Chunk<C>) -> Self {
        Self {
            position: chunk.position,
            lod_level: chunk.lod_level,
            entity: chunk.entity,
            data_shape: chunk.data_shape,
            mesh_shape: chunk.mesh_shape,
            _marker: PhantomData,
        }
    }

    pub fn aabb(&self) -> Aabb3d {
        let min = Vec3::ZERO;
        let max = min + Vec3::splat(CHUNK_SIZE_F);
        Aabb3d {
            min: min.into(),
            max: max.into(),
        }
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
        lod_level: LodLevel,
        data_shape: UVec3,
        mesh_shape: UVec3,
        modified_voxels: ModifiedVoxels<C, I>,
    ) -> Self {
        let mut chunk_data = ChunkData::with_entity(entity);
        chunk_data.lod_level = lod_level;
        chunk_data.data_shape = data_shape;
        chunk_data.mesh_shape = mesh_shape;

        Self {
            position,
            chunk_data,
            modified_voxels,
            mesh: None,
            user_bundle: None,
            _marker: PhantomData,
        }
    }

    /// Generate voxel data for the chunk. The supplied `modified_voxels` map is first checked,
    /// and where no voxeles are modified, the `voxel_data_fn` is called to get data from the
    /// consumer.
    pub fn generate<F>(
        &mut self,
        mut voxel_data_fn: F,
        previous_data: Option<ChunkData<I>>,
        strategy: ChunkRegenerateStrategy,
    ) where
        F: FnMut(IVec3, Option<WorldVoxel<I>>) -> WorldVoxel<I> + Send + 'static,
    {
        let mut filled_count = 0;
        let modified_voxels = (*self.modified_voxels).read().unwrap();
        let mut material_count = HashSet::new();
        let reuse_previous =
            matches!(strategy, ChunkRegenerateStrategy::Reuse) && previous_data.is_some();

        let desired_shape = self.chunk_data.data_shape;
        let previous_shape = previous_data
            .as_ref()
            .map(|chunk| chunk.data_shape())
            .unwrap_or(desired_shape);
        let mut active_shape = desired_shape;

        if reuse_previous
            // Preserve the previously generated payload so the full-resolution voxels remain available when the chunk is promoted again.
            && previous_data
                .as_ref()
                .map(|chunk| chunk.has_generated())
                .unwrap_or(false)
            && desired_shape.x <= previous_shape.x
            && desired_shape.y <= previous_shape.y
            && desired_shape.z <= previous_shape.z
        {
            active_shape = previous_shape;
        }

        self.chunk_data.data_shape = active_shape;
        let data_shape = RuntimeShape::<u32, 3>::new(active_shape.to_array());

        let mut voxels = vec![WorldVoxel::Unset; data_shape.size() as usize];

        let scale = voxel_size_from_shape(&data_shape);

        self.chunk_data.has_generated = true;

        for i in 0..data_shape.size() {
            let chunk_block = data_shape.delinearize(i);

            let block_pos = IVec3 {
                x: ((chunk_block[0] as f32 - 1.0) * scale[0]) as i32
                    + (self.position.x * CHUNK_SIZE_I),
                y: ((chunk_block[1] as f32 - 1.0) * scale[1]) as i32
                    + (self.position.y * CHUNK_SIZE_I),
                z: ((chunk_block[2] as f32 - 1.0) * scale[2]) as i32
                    + (self.position.z * CHUNK_SIZE_I),
            };

            if let Some(voxel) = modified_voxels.get(&block_pos) {
                voxels[i as usize] = *voxel;
                if !voxel.is_unset() && !voxel.is_air() {
                    filled_count += 1;
                }
                continue;
            }

            let previous_voxel = previous_data
                .as_ref()
                .and_then(|chunk| chunk.get_voxel_at_world_position(block_pos));

            if reuse_previous {
                if let Some(prev_voxel) = previous_voxel {
                    if !prev_voxel.is_unset() {
                        voxels[i as usize] = prev_voxel;
                        if let WorldVoxel::Solid(m) = prev_voxel {
                            filled_count += 1;
                            material_count.insert(m);
                        }
                        continue;
                    }
                }
            }

            let voxel = voxel_data_fn(block_pos, previous_voxel);

            voxels[i as usize] = voxel;

            if let WorldVoxel::Solid(m) = voxel {
                filled_count += 1;
                material_count.insert(m);
            }
        }

        self.chunk_data.is_empty = filled_count == 0;
        self.chunk_data.is_full = filled_count == data_shape.size();

        if self.chunk_data.is_full && material_count.len() == 1 {
            self.chunk_data.fill_type = FillType::Uniform(voxels[0]);
            self.chunk_data.voxels = None;
        } else if filled_count > 0 {
            self.chunk_data.fill_type = FillType::Mixed;
            self.chunk_data.voxels = Some(Arc::from(voxels));
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
        if self.mesh.is_none() {
            if let Some(voxels) = self.chunk_data.voxels.as_ref() {
                let data_shape = self.chunk_data.data_shape;
                let mesh_shape = self.chunk_data.mesh_shape;
                let mesh_and_bundle = chunk_meshing_fn(
                    Arc::clone(voxels),
                    data_shape,
                    mesh_shape,
                    texture_index_mapper,
                );
                self.mesh = Some(mesh_and_bundle.0);
                self.user_bundle = mesh_and_bundle.1;
            }
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
