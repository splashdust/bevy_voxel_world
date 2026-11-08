use bevy::{mesh::VertexAttributeValues, platform::hash::FixedHasher, prelude::*};
use ndshape::{ConstShape3u32, RuntimeShape, Shape};
use std::{
    fmt,
    hash::{BuildHasher, Hash, Hasher},
    marker::PhantomData,
    sync::{Arc, RwLock},
};

use crate::{
    mesh_cache::WeakMeshMap,
    prelude::{
        ChunkMeshingFn, FillHint, LodLevel, TextureIndexMapperFn, VoxelWorldConfig,
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

/// `log2(CHUNK_SIZE)`. Chunk and LOD level arithmetic shifts by this instead
/// of dividing, since dividing by a value only known at run time is slow.
const CHUNK_SIZE_SHIFT: i32 = CHUNK_SIZE_U.trailing_zeros() as i32;
const _: () = assert!(CHUNK_SIZE_U.is_power_of_two());

/// The coarsest LOD level a chunk can have. A chunk at this level spans
/// `CHUNK_SIZE << MAX_LOD_LEVEL` voxels on a side.
pub const MAX_LOD_LEVEL: LodLevel = 15;

/// The position of the level-0 chunk that contains a voxel. Rounds down, so
/// voxel -1 is in chunk -1.
pub(crate) fn chunk_containing(voxel_position: IVec3) -> IVec3 {
    voxel_position >> CHUNK_SIZE_SHIFT
}

/// Addresses a chunk of a voxel world: where it is, and at which LOD level.
///
/// A chunk at LOD level 0 spans `CHUNK_SIZE` voxels on a side. A chunk at level
/// `L` spans `CHUNK_SIZE << L` voxels and contains the eight level `L - 1`
/// chunks in that space. `position` is the chunk's minimum corner in level-0
/// chunk units, so for a level-0 chunk it is the plain chunk position.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ChunkKey {
    /// The chunk's minimum corner, in level-0 chunks: a multiple of
    /// `1 << lod_level` on every axis.
    pub position: IVec3,
    pub lod_level: LodLevel,
}

impl ChunkKey {
    pub fn new(position: IVec3, lod_level: LodLevel) -> Self {
        debug_assert!(
            lod_level <= MAX_LOD_LEVEL,
            "LOD level {lod_level} is above MAX_LOD_LEVEL"
        );
        debug_assert!(
            position.rem_euclid(IVec3::splat(1 << lod_level)) == IVec3::ZERO,
            "{position} is not the minimum corner of a level {lod_level} chunk"
        );
        Self {
            position,
            lod_level,
        }
    }

    /// The key of the chunk at `lod_level` that contains the voxel at
    /// `voxel_position`.
    pub fn containing(voxel_position: IVec3, lod_level: LodLevel) -> Self {
        // Arithmetic shifts round down, as the chunk grid needs.
        let lod_level_i = lod_level as i32;
        Self {
            position: (voxel_position >> (CHUNK_SIZE_SHIFT + lod_level_i)) << lod_level_i,
            lod_level,
        }
    }

    /// The key of the chunk at `lod_level` that contains the level-0 chunk at
    /// `chunk_position`.
    pub fn containing_chunk(chunk_position: IVec3, lod_level: LodLevel) -> Self {
        let lod_level_i = lod_level as i32;
        Self {
            position: (chunk_position >> lod_level_i) << lod_level_i,
            lod_level,
        }
    }

    /// The key of the chunk one level above this one that contains it.
    pub fn parent(&self) -> Self {
        debug_assert!(self.lod_level < MAX_LOD_LEVEL, "{self} has no parent");
        Self::containing_chunk(self.position, self.lod_level + 1)
    }

    /// The keys of the eight chunks one level below this one that it contains.
    pub fn children(&self) -> impl Iterator<Item = Self> {
        debug_assert!(self.lod_level > 0, "{self} has no children");
        let lod_level = self.lod_level.saturating_sub(1);
        let half = 1 << lod_level;
        let position = self.position;
        (0..8).map(move |corner: i32| Self {
            position: position
                + IVec3::new(corner & 1, (corner >> 1) & 1, corner >> 2) * half,
            lod_level,
        })
    }

    /// How many level-0 chunks the chunk spans on a side.
    pub fn extent(&self) -> i32 {
        1 << self.lod_level
    }

    /// How many voxels the chunk spans on a side.
    pub fn voxel_extent(&self) -> i32 {
        CHUNK_SIZE_I << self.lod_level
    }

    /// The voxel at the chunk's minimum corner.
    pub fn min_voxel(&self) -> IVec3 {
        self.position * CHUNK_SIZE_I
    }

    /// The chunk's minimum corner in world coordinates.
    pub fn world_position(&self) -> Vec3 {
        self.position.as_vec3() * CHUNK_SIZE_F
    }

    /// The chunk's centre in world coordinates.
    pub fn centre(&self) -> Vec3 {
        (self.position.as_vec3() + 0.5 * self.extent() as f32) * CHUNK_SIZE_F
    }

    /// Whether the voxel at `voxel_position` is in the chunk.
    pub fn contains_voxel(&self, voxel_position: IVec3) -> bool {
        let offset = voxel_position - self.min_voxel();
        offset.cmpge(IVec3::ZERO).all()
            && offset.cmplt(IVec3::splat(self.voxel_extent())).all()
    }
}

impl fmt::Display for ChunkKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} at level {}", self.position, self.lod_level)
    }
}

/// How many chunks there are at each LOD level, so that a lookup by position
/// visits only the levels in use.
pub(crate) struct LevelCounts {
    counts: [usize; MAX_LOD_LEVEL as usize + 1],
    /// Bit `L` is set while level `L` has chunks.
    in_use: u16,
}

impl LevelCounts {
    pub fn new() -> Self {
        Self {
            counts: [0; MAX_LOD_LEVEL as usize + 1],
            in_use: 0,
        }
    }

    pub fn add(&mut self, lod_level: LodLevel) {
        self.counts[lod_level as usize] += 1;
        self.in_use |= 1 << lod_level;
    }

    pub fn remove(&mut self, lod_level: LodLevel) {
        let count = &mut self.counts[lod_level as usize];
        *count -= 1;
        if *count == 0 {
            self.in_use &= !(1 << lod_level);
        }
    }

    /// The levels with chunks, finest first.
    pub fn in_use(&self) -> impl Iterator<Item = LodLevel> {
        let mut bits = self.in_use;
        std::iter::from_fn(move || {
            if bits == 0 {
                return None;
            }
            let lod_level = bits.trailing_zeros() as LodLevel;
            bits &= bits - 1;
            Some(lod_level)
        })
    }
}

pub type PaddedChunkShape =
    ConstShape3u32<PADDED_CHUNK_SIZE, PADDED_CHUNK_SIZE, PADDED_CHUNK_SIZE>;

pub type VoxelArray<I> = Arc<[WorldVoxel<I>]>;

/// The spacing in voxels between a chunk's data voxels along each axis.
///
/// It is the extent of a chunk at `lod_level` divided by the interior of its
/// padded data `shape`.
fn sample_spacing(shape: UVec3, lod_level: LodLevel) -> Vec3 {
    let interior = shape.saturating_sub(UVec3::splat(2)).max(UVec3::ONE);
    Vec3::splat((CHUNK_SIZE_I << lod_level) as f32) / interior.as_vec3()
}

/// How many voxels deep the padding of a chunk at `lod_level` is along each axis:
/// the spacing of its data voxels from the padded data `shape`, rounded up.
///
/// It is at most the chunk's voxel extent, so the padding never reaches past the
/// chunks next to it at the same level.
pub(crate) fn padding_depth(shape: UVec3, lod_level: LodLevel) -> IVec3 {
    sample_spacing(shape, lod_level).ceil().as_ivec3()
}

/// The data coordinate, along one axis, of the data voxel that covers the voxel
/// `offset` voxels from a chunk's origin.
///
/// The chunk's data voxels are `scale` voxels apart, and there are `extent` of
/// them along the axis. The covering data voxel is the last one at or before
/// the offset, so in a chunk above level 0 every voxel maps to a data voxel.
/// Returns `None` outside the padded data.
fn data_coordinate(offset: i32, scale: f32, extent: u32) -> Option<u32> {
    if scale <= f32::EPSILON || extent == 0 {
        return None;
    }

    // Data voxel `c` sits at `floor((c - 1) * scale)`, so it covers the offset
    // when `(c - 1) * scale < offset + 1`.
    let chunk_block = ((offset as f32 + 1.0) / scale).ceil();
    if chunk_block < 0.0 || chunk_block >= extent as f32 {
        return None;
    }

    Some(chunk_block as u32)
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
    pub(crate) key: ChunkKey,
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
            key: ChunkKey::new(IVec3::ZERO, 0),
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

    /// Hashes the voxels, their shapes and the LOD level, which together decide
    /// the chunk's mesh.
    pub(crate) fn generate_hash(&mut self) {
        if let Some(voxels) = &self.voxels {
            let mut hasher = FixedHasher.build_hasher();
            self.key.lod_level.hash(&mut hasher);
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
    /// Above LOD level 0, or with non-default shapes, these coordinates don't map
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

    /// Returns the voxel at the given world-space position, if the position is
    /// within the chunk's padded data.
    ///
    /// The answer is the data voxel covering the position. Above LOD level 0
    /// one data voxel covers several voxels, `1 << L` on a side with the default
    /// shapes, so every voxel it covers gets its value, as the chunk's mesh
    /// shows it.
    pub fn get_voxel_at_world_position(
        &self,
        world_position: IVec3,
    ) -> Option<WorldVoxel<I>> {
        let scale = sample_spacing(self.data_shape, self.key.lod_level);
        let offset = world_position - self.key.min_voxel();

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

    /// The chunk's key: its position and LOD level.
    pub fn key(&self) -> ChunkKey {
        self.key
    }

    /// Returns the position of this chunk, in level-0 chunks.
    pub fn chunk_position(&self) -> IVec3 {
        self.key.position
    }

    /// Returns the LOD level this chunk was generated with.
    pub fn lod_level(&self) -> LodLevel {
        self.key.lod_level
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
        self.key.world_position()
    }

    /// Returns the AABB of the chunk, relative to its world position
    pub fn aabb(&self) -> Aabb3d {
        let min = Vec3::ZERO;
        let max = min + Vec3::splat(self.key.voxel_extent() as f32);
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
        if !self.key.contains_voxel(voxel_pos) {
            return false;
        }
        self.get_voxel_at_world_position(voxel_pos) == Some(voxel)
    }

    /// Returns true if this chunk has been processed by the voxel generation system (typically to generate terrain)
    /// Before generation has happened, voxel data in the chunk is not initialized.
    pub fn has_generated(&self) -> bool {
        self.has_generated
    }

    /// Whether every voxel of the chunk, not counting the padding, is `voxel`.
    /// `Air` and `Unset` count as the same.
    #[cfg(debug_assertions)]
    pub(crate) fn interior_is(&self, voxel: WorldVoxel<I>) -> bool {
        let same = |other: WorldVoxel<I>| {
            if voxel.is_solid() {
                other == voxel
            } else {
                !other.is_solid()
            }
        };
        let Some(voxels) = &self.voxels else {
            return same(self.fill_voxel());
        };
        let shape = self.data_shape;
        let row = shape.x as usize;
        let slab = (shape.x * shape.y) as usize;
        (1..shape.z as usize - 1).all(|z| {
            (1..shape.y as usize - 1).all(|y| {
                let start = z * slab + y * row + 1;
                voxels[start..start + shape.x as usize - 2]
                    .iter()
                    .all(|&other| same(other))
            })
        })
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
    pub(crate) key: ChunkKey,
    pub(crate) entity: Entity,
    pub(crate) data_shape: UVec3,
    pub(crate) mesh_shape: UVec3,
    _marker: PhantomData<C>,
}

impl<C> Chunk<C> {
    pub(crate) fn new(
        key: ChunkKey,
        entity: Entity,
        data_shape: UVec3,
        mesh_shape: UVec3,
    ) -> Self {
        Self {
            key,
            entity,
            data_shape,
            mesh_shape,
            _marker: PhantomData,
        }
    }

    /// The chunk's key: its position and LOD level.
    pub fn key(&self) -> ChunkKey {
        self.key
    }

    /// The chunk position, in level-0 chunks.
    pub fn position(&self) -> IVec3 {
        self.key.position
    }

    pub fn lod_level(&self) -> LodLevel {
        self.key.lod_level
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

    /// The AABB of the chunk, relative to its world position.
    pub fn aabb(&self) -> Aabb3d {
        let min = Vec3::ZERO;
        let max = min + Vec3::splat(self.key.voxel_extent() as f32);
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
        let scale = sample_spacing(shape, data.key.lod_level);
        let origin = data.key.min_voxel();
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

/// Scales a chunk mesh, built as for a level-0 chunk, to the extent of a chunk
/// at `lod_level`.
///
/// A chunk mesh puts the corner of the chunk's first voxel at (1, 1, 1), just
/// inside the padding, and the scaling keeps that corner in place.
fn scale_mesh_to_level(mesh: &mut Mesh, lod_level: LodLevel) {
    if lod_level == 0 {
        return;
    }
    let scale = (1 << lod_level) as f32;
    if let Some(VertexAttributeValues::Float32x3(positions)) =
        mesh.attribute_mut(Mesh::ATTRIBUTE_POSITION)
    {
        for position in positions.iter_mut() {
            for coordinate in position.iter_mut() {
                *coordinate = (*coordinate - 1.0) * scale + 1.0;
            }
        }
    }
}

/// What sampling a chunk's data voxels found.
struct Sampled<I> {
    voxels: Vec<WorldVoxel<I>>,
    /// How many voxels are solid.
    filled_count: usize,
    /// The first solid voxel, and whether every solid voxel is the same as it.
    first_solid: Option<WorldVoxel<I>>,
    one_material: bool,
}

/// Samples the voxel at every data coordinate of a chunk, x fastest, then y,
/// then z, as the voxels are stored. `voxel_at` gets each voxel's world
/// position and data coordinates.
#[inline]
fn sample<I: Copy + PartialEq>(
    world_coordinates: &[Vec<i32>; 3],
    mut voxel_at: impl FnMut(IVec3, usize, usize, usize) -> WorldVoxel<I>,
) -> Sampled<I> {
    let [xs, ys, zs] = world_coordinates;
    let mut voxels = Vec::with_capacity(xs.len() * ys.len() * zs.len());
    let mut filled_count = 0;
    let mut first_solid = None;
    let mut one_material = true;
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
    Sampled {
        voxels,
        filled_count,
        first_solid,
        one_material,
    }
}

/// Holds all data needed to generate and mesh a chunk
#[derive(Component)]
pub(crate) struct ChunkTask<C, I>
where
    C: VoxelWorldConfig,
{
    pub key: ChunkKey,
    pub chunk_data: ChunkData<I>,
    pub edits: ChunkEdits<I>,
    /// The world's fill hint for the chunk, which the generated voxels are
    /// checked against in debug builds.
    pub hint: FillHint<I>,
    pub disagrees_with_hint: bool,
    pub mesh: Option<Mesh>,
    pub user_bundle: Option<C::ChunkUserBundle>,
    /// What the task needs to solve the chunk's light, in a world with light
    /// propagation.
    #[cfg(feature = "light_propagation")]
    pub light: Option<crate::light::TaskLight<I>>,
    /// The chunk's solved light, once `post_voxel` has run.
    #[cfg(feature = "light_propagation")]
    pub solved_light: Option<crate::light::SolvedLight>,
    _marker: PhantomData<C>,
}

impl<C: VoxelWorldConfig + Send + Sync + 'static, I: Hash + Copy + Eq> ChunkTask<C, I> {
    pub fn new(
        entity: Entity,
        key: ChunkKey,
        data_shape: UVec3,
        mesh_shape: UVec3,
        edits: ChunkEdits<I>,
        hint: FillHint<I>,
    ) -> Self {
        let mut chunk_data = ChunkData::with_entity(entity);
        chunk_data.key = key;
        chunk_data.data_shape = data_shape;
        chunk_data.mesh_shape = mesh_shape;

        Self {
            key,
            chunk_data,
            edits,
            hint,
            disagrees_with_hint: false,
            mesh: None,
            user_bundle: None,
            #[cfg(feature = "light_propagation")]
            light: None,
            #[cfg(feature = "light_propagation")]
            solved_light: None,
            _marker: PhantomData,
        }
    }

    /// Checks the generated voxels against the chunk's fill hint. Debug builds
    /// only.
    ///
    /// A fill hint doesn't account for voxel edits, so a chunk with voxel edits
    /// is not checked.
    #[cfg(debug_assertions)]
    pub fn check_hint(&mut self) {
        if let FillHint::Uniform(voxel) = self.hint {
            self.disagrees_with_hint =
                self.edits.is_empty() && !self.chunk_data.interior_is(voxel);
        }
    }

    /// Generate voxel data for the chunk. The task's edits are first checked, then
    /// the chunk's previous data, if it has any, and where neither has a voxel the
    /// `voxel_data_fn` is called to get data from the consumer.
    pub fn generate<F>(
        &mut self,
        mut voxel_data_fn: F,
        previous_data: Option<ChunkData<I>>,
    ) where
        F: FnMut(IVec3, Option<WorldVoxel<I>>) -> WorldVoxel<I> + Send + 'static,
    {
        #[cfg(feature = "trace")]
        let _span = bevy::log::info_span!("generate_chunk").entered();
        let shape = self.chunk_data.data_shape;
        let scale = sample_spacing(shape, self.key.lod_level);
        let origin = self.key.min_voxel();
        // The world coordinates of the data voxels along each axis.
        let world_coordinates: [Vec<i32>; 3] = std::array::from_fn(|axis| {
            (0..shape[axis])
                .map(|c| ((c as f32 - 1.0) * scale[axis]).floor() as i32 + origin[axis])
                .collect()
        });
        let previous_voxels = previous_data
            .as_ref()
            .map(|chunk| PreviousVoxels::new(chunk, &world_coordinates));

        self.chunk_data.has_generated = true;

        // The generated voxel at `block_pos`, which is at data coordinates `x`,
        // `y` and `z`. It is the previous data's voxel where there is one, and
        // otherwise the one `voxel_data_fn` returns.
        let mut generated = |block_pos: IVec3, x: usize, y: usize, z: usize| {
            let previous_voxel = previous_voxels
                .as_ref()
                .and_then(|previous| previous.get(x, y, z));
            match previous_voxel {
                Some(voxel) if !voxel.is_unset() => voxel,
                _ => voxel_data_fn(block_pos, previous_voxel),
            }
        };
        // Edits are looked up in a loop of their own, so that the loop for a
        // chunk without edits, which is most chunks, carries no edit lookup.
        let sampled = if self.edits.is_empty() {
            sample(&world_coordinates, generated)
        } else {
            let edits = &self.edits;
            sample(&world_coordinates, |block_pos, x, y, z| {
                edits
                    .get(block_pos)
                    .unwrap_or_else(|| generated(block_pos, x, y, z))
            })
        };
        let Sampled {
            voxels,
            filled_count,
            first_solid,
            one_material,
        } = sampled;

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

    /// Works out what the chunk needs from its finished voxels. Runs for every
    /// chunk task, after generating and before the mesh cache check, including
    /// for chunks that generated empty or full.
    ///
    /// It is the place for work that depends on the voxels and must be done
    /// before the chunk is published, whether or not it gets a mesh: in a world
    /// with light propagation, solving the chunk's light.
    pub fn post_voxel(&mut self) {
        #[cfg(feature = "trace")]
        let _span = bevy::log::info_span!("post_voxel").entered();
        #[cfg(feature = "light_propagation")]
        if let Some(light) = self.light.take() {
            self.solved_light = Some(light.solve(&self.chunk_data));
        }
    }

    /// Whether the mesh cache already holds a mesh for the chunk's voxels, so
    /// the chunk needn't be meshed.
    pub fn is_mesh_cached(&self, mesh_map: &RwLock<WeakMeshMap>) -> bool {
        #[cfg(feature = "trace")]
        let _span = bevy::log::info_span!("check_mesh_cache").entered();
        {
            #[cfg(feature = "trace")]
            let _span = bevy::log::info_span!("wait_mesh_cache_read").entered();
            mesh_map.read().unwrap()
        }
        .contains_key(&self.voxels_hash())
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
                let (mut mesh, user_bundle) = chunk_meshing_fn(
                    Arc::clone(voxels),
                    data_shape,
                    mesh_shape,
                    texture_index_mapper,
                );
                scale_mesh_to_level(&mut mesh, self.key.lod_level);
                self.mesh = Some(mesh);
                self.user_bundle = user_bundle;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_level_0_key_is_the_chunk_position_of_the_voxel() {
        for voxel in [
            IVec3::ZERO,
            IVec3::splat(31),
            IVec3::splat(32),
            IVec3::splat(-1),
        ] {
            assert_eq!(
                ChunkKey::containing(voxel, 0),
                ChunkKey::new(chunk_containing(voxel), 0),
                "voxel {voxel}"
            );
        }
    }

    #[test]
    fn a_coarse_key_is_the_corner_of_the_chunk_in_level_0_units() {
        assert_eq!(
            ChunkKey::containing(IVec3::new(0, 100, 127), 2),
            ChunkKey::new(IVec3::new(0, 0, 0), 2)
        );
        assert_eq!(
            ChunkKey::containing(IVec3::new(128, 100, 127), 2),
            ChunkKey::new(IVec3::new(4, 0, 0), 2)
        );
        assert_eq!(
            ChunkKey::containing(IVec3::splat(-1), 2),
            ChunkKey::new(IVec3::splat(-4), 2)
        );
    }

    #[test]
    fn a_chunk_spans_its_extent_from_its_minimum_corner() {
        let key = ChunkKey::new(IVec3::new(2, -2, 0), 1);
        assert_eq!(key.extent(), 2);
        assert_eq!(key.voxel_extent(), 64);
        assert_eq!(key.min_voxel(), IVec3::new(64, -64, 0));
        assert_eq!(key.centre(), Vec3::new(96.0, -32.0, 32.0));
        assert!(key.contains_voxel(IVec3::new(64, -64, 0)));
        assert!(key.contains_voxel(IVec3::new(127, -1, 63)));
        assert!(!key.contains_voxel(IVec3::new(63, -64, 0)));
        assert!(!key.contains_voxel(IVec3::new(64, 0, 0)));
        assert!(!key.contains_voxel(IVec3::new(64, -64, 64)));
    }

    #[test]
    fn a_chunk_contains_the_eight_chunks_below_it_and_has_one_parent() {
        let key = ChunkKey::new(IVec3::new(4, -4, 0), 2);
        let children: Vec<_> = key.children().collect();
        assert_eq!(children.len(), 8);
        for child in &children {
            assert_eq!(child.lod_level, 1);
            assert_eq!(child.parent(), key);
            assert_eq!(ChunkKey::containing_chunk(child.position, 2), key);
        }
        assert!(children.contains(&ChunkKey::new(IVec3::new(4, -4, 0), 1)));
        assert!(children.contains(&ChunkKey::new(IVec3::new(6, -2, 2), 1)));
        assert_eq!(
            ChunkKey::containing_chunk(IVec3::new(-1, 0, 7), 2),
            ChunkKey::new(IVec3::new(-4, 0, 4), 2)
        );
    }

    #[test]
    fn levels_in_use_are_those_with_chunks_finest_first() {
        let mut levels = LevelCounts::new();
        assert_eq!(levels.in_use().count(), 0);
        levels.add(3);
        levels.add(0);
        levels.add(3);
        assert_eq!(levels.in_use().collect::<Vec<_>>(), vec![0, 3]);
        levels.remove(3);
        assert_eq!(levels.in_use().collect::<Vec<_>>(), vec![0, 3]);
        levels.remove(3);
        assert_eq!(levels.in_use().collect::<Vec<_>>(), vec![0]);
    }

    #[test]
    fn a_coarse_chunk_samples_every_other_voxel_by_default() {
        let spacing = sample_spacing(UVec3::splat(PADDED_CHUNK_SIZE), 1);
        assert_eq!(spacing, Vec3::splat(2.0));
        let oversampled = sample_spacing(UVec3::splat(2 * CHUNK_SIZE_U + 2), 1);
        assert_eq!(oversampled, Vec3::splat(1.0));
    }

    #[test]
    fn a_data_voxel_covers_the_voxels_up_to_the_next_one() {
        // Level 0: one data voxel per voxel, padding at -1 and 32.
        assert_eq!(data_coordinate(-2, 1.0, 34), None);
        assert_eq!(data_coordinate(-1, 1.0, 34), Some(0));
        assert_eq!(data_coordinate(0, 1.0, 34), Some(1));
        assert_eq!(data_coordinate(32, 1.0, 34), Some(33));
        assert_eq!(data_coordinate(33, 1.0, 34), None);
        // Level 1: data voxels two voxels apart, each covering two voxels.
        assert_eq!(data_coordinate(-3, 2.0, 34), None);
        assert_eq!(data_coordinate(-2, 2.0, 34), Some(0));
        assert_eq!(data_coordinate(-1, 2.0, 34), Some(0));
        assert_eq!(data_coordinate(0, 2.0, 34), Some(1));
        assert_eq!(data_coordinate(1, 2.0, 34), Some(1));
        assert_eq!(data_coordinate(2, 2.0, 34), Some(2));
        assert_eq!(data_coordinate(63, 2.0, 34), Some(32));
        assert_eq!(data_coordinate(65, 2.0, 34), Some(33));
        assert_eq!(data_coordinate(66, 2.0, 34), None);
        // An uneven spacing still picks the last data voxel at or before the offset.
        assert_eq!(data_coordinate(5, 3.2, 12), Some(2));
        assert_eq!(data_coordinate(6, 3.2, 12), Some(3));
    }

    #[test]
    fn a_chunk_mesh_is_scaled_about_the_corner_of_its_first_voxel() {
        let mut mesh = Mesh::new(
            bevy::render::render_resource::PrimitiveTopology::TriangleList,
            bevy::asset::RenderAssetUsages::default(),
        );
        mesh.insert_attribute(
            Mesh::ATTRIBUTE_POSITION,
            vec![[1.0, 1.0, 1.0], [33.0, 1.0, 0.0]],
        );

        scale_mesh_to_level(&mut mesh, 1);

        let Some(VertexAttributeValues::Float32x3(positions)) =
            mesh.attribute(Mesh::ATTRIBUTE_POSITION)
        else {
            panic!("positions missing");
        };
        assert_eq!(positions, &vec![[1.0, 1.0, 1.0], [65.0, 1.0, -1.0]]);
    }
}
