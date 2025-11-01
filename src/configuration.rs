use std::hash::Hash;
use std::sync::Arc;

use crate::chunk::{ChunkData, VoxelArray, PADDED_CHUNK_SIZE};
use crate::meshing::generate_chunk_mesh_for_shape;
use crate::voxel::WorldVoxel;
use bevy::prelude::*;

pub type VoxelLookupFn<I = u8> =
    Box<dyn FnMut(IVec3, Option<WorldVoxel<I>>) -> WorldVoxel<I> + Send + Sync>;
pub type LodLevel = u8;
pub type VoxelLookupDelegate<I = u8> =
    Box<dyn Fn(IVec3, LodLevel, Option<ChunkData<I>>) -> VoxelLookupFn<I> + Send + Sync>;

pub type TextureIndexMapperFn<I = u8> = Arc<dyn Fn(I) -> [u32; 3] + Send + Sync>;

#[inline]
pub const fn padded_chunk_shape(interior: UVec3) -> UVec3 {
    UVec3::new(interior.x + 2, interior.y + 2, interior.z + 2)
}

#[inline]
pub const fn padded_chunk_shape_uniform(edge: u32) -> UVec3 {
    UVec3::splat(edge + 2)
}

pub type ChunkMeshingFn<I, UB> = Box<
    dyn FnMut(
            Arc<VoxelArray<I>>,
            UVec3,
            UVec3,
            TextureIndexMapperFn<I>,
        ) -> (Mesh, Option<UB>)
        + Send
        + Sync,
>;
pub type ChunkMeshingDelegate<I, UB> = Option<
    Box<
        dyn Fn(
                IVec3,
                LodLevel,
                UVec3,
                UVec3,
                Option<ChunkData<I>>,
            ) -> ChunkMeshingFn<I, UB>
            + Send
            + Sync,
    >,
>;

#[derive(Default, Clone, Copy, PartialEq, Eq)]
pub enum ChunkRegenerateStrategy {
    /// Attempt to reuse previously generated chunk data before invoking the voxel lookup delegate.
    #[default]
    Reuse,
    /// Always regenerate voxel data using the voxel lookup delegate.
    Repopulate,
}

#[derive(Default, PartialEq, Eq)]
pub enum ChunkDespawnStrategy {
    /// Despawn chunks that are further than `spawning_distance` away from the camera
    /// or outside of the viewport.
    #[default]
    FarAwayOrOutOfView,

    /// Only despawn chunks that are further than `spawning_distance` away from the camera.
    FarAway,
}

#[derive(Default, PartialEq, Eq)]
pub enum ChunkSpawnStrategy {
    /// Spawn chunks that are within `spawning_distance` of the camera
    /// and also inside the viewport.
    #[default]
    CloseAndInView,

    /// Spawn chunks that are within `spawning_distance` of the camera, regardless of whether
    /// they are in the viewport or not. Will only have an effect if the despawn strategy is
    /// `FarAway`. If this strategy is used a flood fill will be used to find unspawned chunks
    /// and therefore it might make sense to lower the `spawning_rays` option.
    Close,
}

/// `bevy_voxel_world` configuation structs need to implement this trait
pub trait VoxelWorldConfig: Resource + Default + Clone {
    /// The type used to index materials. A value of this type will be stored in each voxel,
    /// so it's a good idea to keep it small.
    type MaterialIndex: Copy + Hash + PartialEq + Eq + Default + Send + Sync;

    /// This type is used to insert a custom component bundle in generated chunks during meshing.
    /// It is part of the return type of the chunk_meshing_delegate function.
    /// If you are not using this feature, you can set this to `()`.
    type ChunkUserBundle: Bundle + Clone;

    /// Maximum distance in chunks to spawn chunks, depending on the [`ChunkSpawnStrategy`]
    fn spawning_distance(&self) -> u32 {
        10
    }

    /// Minimum distance in chunks to despawn chunks regardless of the [`ChunkSpawnStrategy`].
    /// As a result, this radius will always remain spawned around the camera.
    fn min_despawn_distance(&self) -> u32 {
        1
    }

    /// Strategy for despawning chunks
    fn chunk_despawn_strategy(&self) -> ChunkDespawnStrategy {
        ChunkDespawnStrategy::default()
    }

    /// Strategy for spawning chunks
    /// This is only used if the despawn strategy is `FarAway`
    fn chunk_spawn_strategy(&self) -> ChunkSpawnStrategy {
        ChunkSpawnStrategy::default()
    }

    /// Maximum number of chunks that can get queued for spawning in a given frame.
    /// In some scenarios, reducing this number can help with performance, due to less
    /// thread contention.
    fn max_spawn_per_frame(&self) -> usize {
        10000
    }

    /// Number of rays to cast when spawning chunks. Higher values will result in more
    /// chunks being spawned per frame, but will also increase cpu load, and can lead to
    /// thread contention.
    fn spawning_rays(&self) -> usize {
        100
    }

    /// How far outside of the viewports spawning rays should get cast. Higher values will
    /// will reduce the likelyhood of chunks popping in, but will also increase cpu load.
    fn spawning_ray_margin(&self) -> u32 {
        25
    }

    /// Debugging aids
    fn debug_draw_chunks(&self) -> bool {
        false
    }

    /// A function that maps voxel materials to texture coordinates.
    /// The input is the material index, and the output is a slice of three indexes into an array texture.
    /// The three values correspond to the top, sides and bottom of the voxel. For example,
    /// if the slice is `[1,2,2]`, the top will use texture index 1 and the sides and bottom will use texture
    /// index 2.
    fn texture_index_mapper(&self) -> TextureIndexMapperFn<Self::MaterialIndex> {
        Arc::new(|_mat| [0, 0, 0])
    }

    /// A function that returns a function that returns true if a voxel exists at the given position
    ///
    /// The delegate will be called every time a new chunk needs to be computed. The delegate should
    /// return a function that can be called to check if a voxel exists at a given position. This function
    /// needs to be thread-safe, since chunk computation happens on a separate thread.
    fn voxel_lookup_delegate(&self) -> VoxelLookupDelegate<Self::MaterialIndex> {
        Box::new(|_, _, _| Box::new(|_, _| WorldVoxel::Unset))
    }

    /// A function that returns a function that computes the mesh for a chunk
    ///
    /// The delegate will be called every time a new chunk needs to be computed. The delegate should
    /// return a function that returns a Mesh. This function needs to be thread-safe, since chunk computation
    /// happens on a separate thread.
    ///
    /// The input to the function is the voxel array for the chunk, the position of the chunk and the texture
    /// index mapper function
    fn chunk_meshing_delegate(
        &self,
    ) -> ChunkMeshingDelegate<Self::MaterialIndex, Self::ChunkUserBundle> {
        None
    }

    /// A tuple of the path to the texture and the number of indexes in the texture. `None` if no texture is used.
    fn voxel_texture(&self) -> Option<(String, u32)> {
        None
    }

    /// Custom material will not get initialized if this returns false. When this is false,
    /// `VoxelWorldMaterialHandle` needs to be manually added with a reference to the material handle.
    ///
    /// This can be used for example if you need to wait for a texture image to load before
    /// the material can be used.
    fn init_custom_materials(&self) -> bool {
        true
    }

    /// Compute the level of detail for a chunk at a world-space position.
    /// Defaults to `0` for all chunks.
    fn chunk_lod(&self, _chunk_position: IVec3, _camera_position: Vec3) -> LodLevel {
        0
    }

    /// Define the padded voxel dimensions used for data generation for a given LOD level.
    fn chunk_data_shape(&self, _lod_level: LodLevel) -> UVec3 {
        UVec3::splat(PADDED_CHUNK_SIZE)
    }

    /// Define the padded voxel dimensions used for meshing for a given LOD level.
    fn chunk_meshing_shape(&self, _lod_level: LodLevel) -> UVec3 {
        UVec3::splat(PADDED_CHUNK_SIZE)
    }

    /// Determine how voxel data should be regenerated for a chunk. Defaults to reusing previous data.
    fn chunk_regenerate_strategy(&self) -> ChunkRegenerateStrategy {
        ChunkRegenerateStrategy::default()
    }

    fn init_root(&self, mut _commands: Commands, _root: Entity) {}
}

pub fn default_chunk_meshing_delegate<I: PartialEq + Copy, UB: Bundle>(
    pos: IVec3,
    _lod: LodLevel,
    data_shape: UVec3,
    mesh_shape: UVec3,
    _previous_data: Option<ChunkData<I>>,
) -> ChunkMeshingFn<I, UB> {
    Box::new(
        move |voxels: Arc<VoxelArray<I>>,
              data_shape_in: UVec3,
              mesh_shape_in: UVec3,
              texture_index_mapper: TextureIndexMapperFn<I>| {
            let data_shape = if data_shape_in == UVec3::ZERO {
                data_shape
            } else {
                data_shape_in
            };
            let mesh_shape = if mesh_shape_in == UVec3::ZERO {
                mesh_shape
            } else {
                mesh_shape_in
            };

            let voxels_slice: Arc<[WorldVoxel<I>]> = voxels.clone();
            let mesh = generate_chunk_mesh_for_shape(
                voxels_slice,
                pos,
                data_shape,
                mesh_shape,
                texture_index_mapper,
            );
            (mesh, None)
        },
    )
}

#[derive(Resource, Clone, Default)]
pub struct DefaultWorld;

impl DefaultWorld {}

impl VoxelWorldConfig for DefaultWorld {
    type MaterialIndex = u8;
    type ChunkUserBundle = ();

    fn texture_index_mapper(
        &self,
    ) -> Arc<dyn Fn(Self::MaterialIndex) -> [u32; 3] + Send + Sync> {
        Arc::new(|mat| match mat {
            0 => [0, 0, 0],
            1 => [1, 1, 1],
            2 => [2, 2, 2],
            3 => [3, 3, 3],
            _ => [0, 0, 0],
        })
    }
}
