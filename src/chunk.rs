use bevy::{platform::hash::FixedHasher, prelude::*};
use ndshape::{ConstShape3u32, RuntimeShape, Shape};
use std::{
    hash::{BuildHasher, Hash, Hasher},
    marker::PhantomData,
    sync::Arc,
};

use crate::{
    prelude::{
        ChunkMeshingFn, ChunkRegenerateStrategy, LodLevel, TextureIndexMapperFn,
        VoxelWorldConfig,
    },
    voxel::WorldVoxel,
    voxel_edits::ChunkEdits,
};

// The size of a chunk in voxels
// TODO: implement a way to change this though the configuration
pub const CHUNK_SIZE_U: u32 = 32;
pub const CHUNK_SIZE_I: i32 = CHUNK_SIZE_U as i32;
pub const CHUNK_SIZE_F: f32 = CHUNK_SIZE_U as f32;

// A chunk with 1-voxel boundary padding.
pub(crate) const PADDED_CHUNK_SIZE: u32 = CHUNK_SIZE_U + 2;

/// The position of the chunk that contains a voxel. Rounds down, so voxel -1
/// is in chunk -1.
pub(crate) fn chunk_containing(voxel_position: IVec3) -> IVec3 {
    voxel_position.div_euclid(IVec3::splat(CHUNK_SIZE_I))
}
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

/// The data coordinate along one axis of the voxel `offset` voxels from a
/// chunk's origin, where the chunk's data voxels are `scale` voxels apart and
/// `extent` long. `None` when no data voxel sits there.
fn data_coordinate(offset: i32, scale: f32, extent: u32) -> Option<u32> {
    if scale <= f32::EPSILON || extent == 0 {
        return None;
    }

    let raw = ((offset as f32) + 1.0) / scale;
    let chunk_block = raw.ceil();

    if chunk_block < 0.0 {
        return None;
    }

    let chunk_block_i = chunk_block as i32;

    if chunk_block_i < 0 || chunk_block_i >= extent as i32 {
        return None;
    }

    let reconstructed = ((chunk_block - 1.0) * scale) as i32;

    if reconstructed != offset {
        return None;
    }

    Some(chunk_block_i as u32)
}

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
            let mut hasher = FixedHasher.build_hasher();
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
            self.fill_voxel()
        }
    }

    /// The voxel everywhere in a chunk that stores no voxels.
    fn fill_voxel(&self) -> WorldVoxel<I> {
        match self.fill_type {
            FillType::Uniform(voxel) => voxel,
            FillType::Empty => WorldVoxel::Unset,
            FillType::Mixed => unreachable!(),
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

        let chunk_block = match (
            data_coordinate(offset.x, scale.x, self.data_shape.x),
            data_coordinate(offset.y, scale.y, self.data_shape.y),
            data_coordinate(offset.z, scale.z, self.data_shape.z),
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
        if self.position != chunk_containing(voxel_pos) {
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

/// A chunk entity of the voxel world `C`. Its chunk state is read through
/// `VoxelWorld::chunk_state`.
#[derive(Component, Clone)]
pub struct Chunk<C> {
    pub(crate) position: IVec3,
    pub(crate) lod_level: LodLevel,
    pub(crate) entity: Entity,
    pub(crate) data_shape: UVec3,
    pub(crate) mesh_shape: UVec3,
    _marker: PhantomData<C>,
}

impl<C> Chunk<C> {
    pub(crate) fn new(
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

    /// The chunk position, in chunks.
    pub fn position(&self) -> IVec3 {
        self.position
    }

    pub fn lod_level(&self) -> LodLevel {
        self.lod_level
    }

    pub fn entity(&self) -> Entity {
        self.entity
    }

    pub fn data_shape(&self) -> UVec3 {
        self.data_shape
    }

    pub fn mesh_shape(&self) -> UVec3 {
        self.mesh_shape
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

/// Reads a chunk's previous voxels at the world positions of a task's data
/// voxels, as [`ChunkData::get_voxel_at_world_position`] would.
struct PreviousVoxels<'a, I> {
    data: &'a ChunkData<I>,
    // For each axis, what each of the task's coordinates along it adds to the
    // previous data's index, or `None` where no previous voxel sits.
    index_parts: [Vec<Option<u32>>; 3],
}

impl<'a, I: Hash + Copy + PartialEq> PreviousVoxels<'a, I> {
    /// `world_coordinates` holds the world coordinates of the task's data
    /// voxels along each axis.
    fn new(data: &'a ChunkData<I>, world_coordinates: &[Vec<i32>; 3]) -> Self {
        let shape = data.data_shape;
        let scale = voxel_size_from_shape(&RuntimeShape::<u32, 3>::new(shape.to_array()));
        let origin = data.position * CHUNK_SIZE_I;
        let strides = [1, shape.x, shape.x * shape.y];
        let index_parts = std::array::from_fn(|axis| {
            world_coordinates[axis]
                .iter()
                .map(|&world| {
                    data_coordinate(world - origin[axis], scale[axis], shape[axis])
                        .map(|coordinate| coordinate * strides[axis])
                })
                .collect()
        });
        Self { data, index_parts }
    }

    /// The previous voxel at the task's data coordinates `x`, `y` and `z`.
    fn get(&self, x: usize, y: usize, z: usize) -> Option<WorldVoxel<I>> {
        let [xs, ys, zs] = &self.index_parts;
        let index = xs[x]? + ys[y]? + zs[z]?;
        Some(match &self.data.voxels {
            Some(voxels) => voxels[index as usize],
            None => self.data.fill_voxel(),
        })
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
    pub edits: ChunkEdits<I>,
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
        edits: ChunkEdits<I>,
    ) -> Self {
        let mut chunk_data = ChunkData::with_entity(entity);
        chunk_data.lod_level = lod_level;
        chunk_data.data_shape = data_shape;
        chunk_data.mesh_shape = mesh_shape;

        Self {
            position,
            chunk_data,
            edits,
            mesh: None,
            user_bundle: None,
            _marker: PhantomData,
        }
    }

    /// Generate voxel data for the chunk. The task's edits are first checked, and
    /// where no voxels are edited, the `voxel_data_fn` is called to get data from
    /// the consumer.
    pub fn generate<F>(
        &mut self,
        mut voxel_data_fn: F,
        previous_data: Option<ChunkData<I>>,
        strategy: ChunkRegenerateStrategy,
    ) where
        F: FnMut(IVec3, Option<WorldVoxel<I>>) -> WorldVoxel<I> + Send + 'static,
    {
        #[cfg(feature = "trace")]
        let _span = bevy::log::info_span!("generate_chunk").entered();
        let has_edits = !self.edits.is_empty();
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
        let scale = voxel_size_from_shape(&data_shape);
        let origin = self.position * CHUNK_SIZE_I;
        // The world coordinates of the data voxels along each axis.
        let world_coordinates: [Vec<i32>; 3] = std::array::from_fn(|axis| {
            (0..active_shape[axis])
                .map(|c| ((c as f32 - 1.0) * scale[axis]) as i32 + origin[axis])
                .collect()
        });
        let previous_voxels = previous_data
            .as_ref()
            .map(|chunk| PreviousVoxels::new(chunk, &world_coordinates));

        self.chunk_data.has_generated = true;

        // The voxel at `block_pos`, which is at data coordinates `x`, `y` and `z`.
        let mut voxel_at = |block_pos: IVec3, x: usize, y: usize, z: usize| {
            if has_edits {
                if let Some(voxel) = self.edits.get(block_pos) {
                    return voxel;
                }
            }
            let previous_voxel = previous_voxels
                .as_ref()
                .and_then(|previous| previous.get(x, y, z));
            match previous_voxel {
                Some(voxel) if reuse_previous && !voxel.is_unset() => voxel,
                _ => voxel_data_fn(block_pos, previous_voxel),
            }
        };

        let mut voxels = Vec::with_capacity(data_shape.usize());
        let mut filled_count = 0;
        // The first solid voxel, and whether every solid voxel is the same as it.
        let mut first_solid = None;
        let mut one_material = true;

        // Data voxels are stored x fastest, then y, then z.
        let [xs, ys, zs] = &world_coordinates;
        for (z_index, &z) in zs.iter().enumerate() {
            for (y_index, &y) in ys.iter().enumerate() {
                for (x_index, &x) in xs.iter().enumerate() {
                    let voxel = voxel_at(IVec3::new(x, y, z), x_index, y_index, z_index);
                    voxels.push(voxel);

                    if voxel.is_solid() {
                        filled_count += 1;
                        match first_solid {
                            None => first_solid = Some(voxel),
                            Some(first) => one_material &= voxel == first,
                        }
                    }
                }
            }
        }

        self.chunk_data.is_empty = filled_count == 0;
        self.chunk_data.is_full = filled_count == voxels.len();

        match first_solid {
            Some(voxel) if self.chunk_data.is_full && one_material => {
                self.chunk_data.fill_type = FillType::Uniform(voxel);
                self.chunk_data.voxels = None;
            }
            Some(_) => {
                self.chunk_data.fill_type = FillType::Mixed;
                self.chunk_data.voxels = Some(Arc::from(voxels));
            }
            None => {
                self.chunk_data.fill_type = FillType::Empty;
                self.chunk_data.voxels = None;
            }
        }

        self.chunk_data.generate_hash();
    }

    /// Generate a mesh for the chunk based on the currect voxel data
    pub fn mesh(
        &mut self,
        mut chunk_meshing_fn: ChunkMeshingFn<I, C::ChunkUserBundle>,
        texture_index_mapper: TextureIndexMapperFn<I>,
    ) {
        #[cfg(feature = "trace")]
        let _span = bevy::log::info_span!("mesh_chunk").entered();
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
