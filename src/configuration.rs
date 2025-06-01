use std::hash::Hash;
use std::sync::Arc;

use crate::chunk::VoxelArray;
use crate::meshing::generate_chunk_mesh;
use crate::voxel::WorldVoxel;
use bevy::prelude::*;

pub type VoxelLookupFn<I = u8> = Box<dyn FnMut(IVec3) -> WorldVoxel<I> + Send + Sync>;
pub type VoxelLookupDelegate<I = u8> =
    Box<dyn Fn(IVec3) -> VoxelLookupFn<I> + Send + Sync>;

pub type TextureIndexMapperFn<I = u8> = Arc<dyn Fn(I) -> [u32; 3] + Send + Sync>;

pub type ChunkMeshingFn<I, UB> = Box<
    dyn FnMut(Arc<VoxelArray<I>>, TextureIndexMapperFn<I>) -> (Mesh, Option<UB>)
        + Send
        + Sync,
>;
pub type ChunkMeshingDelegate<I, UB> =
    Option<Box<dyn Fn(IVec3) -> ChunkMeshingFn<I, UB> + Send + Sync>>;

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

    /// Condition to evaluate before running any voxel systems. This allows you
    /// to defer execution of bevy_voxel_world systems.
    fn get_run_if_condition(&self) -> impl Condition<()> {
        IntoSystem::into_system(|| true)
    }

    /// Distance in chunks to spawn chunks around the camera
    fn spawning_distance(&self) -> u32 {
        10
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
        Box::new(|_| Box::new(|_| WorldVoxel::Unset))
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

    fn init_root(&self, mut _commands: Commands, _root: Entity) {}
}

pub fn default_chunk_meshing_delegate<I: PartialEq + Copy, UB: Bundle>(
    pos: IVec3,
) -> ChunkMeshingFn<I, UB> {
    Box::new(
        move |voxels: Arc<VoxelArray<I>>,
              texture_index_mapper: TextureIndexMapperFn<I>| {
            let mesh = generate_chunk_mesh(voxels, pos, texture_index_mapper);
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
