use std::hash::Hash;
use std::sync::Arc;

use crate::chunk::{ChunkData, ChunkKey, VoxelArray, PADDED_CHUNK_SIZE};
use crate::meshing::{generate_chunk_mesh_for_shape, scale_texture_coordinates};
use crate::voxel::WorldVoxel;
use bevy::prelude::*;

pub type VoxelLookupFn<I = u8> =
    Box<dyn FnMut(IVec3, Option<WorldVoxel<I>>) -> WorldVoxel<I> + Send + Sync>;
/// How coarsely a chunk samples the voxel world, from 0 up to
/// [`MAX_LOD_LEVEL`].
///
/// A level-0 chunk is `CHUNK_SIZE` voxels on a side and samples every voxel. A
/// chunk at level `L` is `CHUNK_SIZE << L` voxels on a side and contains the
/// eight level `L - 1` chunks in that space. With the default
/// [`chunk_data_shape`](VoxelWorldConfig::chunk_data_shape) it holds as many
/// samples as a level-0 chunk, one every `1 << L` voxels.
///
/// [`MAX_LOD_LEVEL`]: crate::prelude::MAX_LOD_LEVEL
pub type LodLevel = u8;
/// Makes the voxel lookup function for one chunk.
///
/// It is given the chunk's position in level-0 chunks, its LOD level, and its
/// previous data if the chunk was generated before.
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
    dyn FnMut(VoxelArray<I>, UVec3, UVec3, TextureIndexMapperFn<I>) -> (Mesh, Option<UB>)
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

#[derive(Default, PartialEq, Eq)]
pub enum ChunkDespawnStrategy {
    /// Despawn chunks that are further than `spawning_distance` away from the camera
    /// or outside of the viewport. Chunks within `min_despawn_distance` of the
    /// camera are kept, even when they are outside of the viewport.
    #[default]
    FarAwayOrOutOfView,

    /// Only despawn chunks that are further than `spawning_distance` away from the camera.
    FarAway,
}

/// What a world knows about a chunk's voxels before the chunk is generated.
///
/// See [`VoxelWorldConfig::chunk_fill_hint`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum FillHint<I = u8> {
    /// Nothing is known, so the chunk must be generated to find out.
    #[default]
    Unknown,
    /// Every voxel in the chunk is this one. `Uniform(WorldVoxel::Air)` means
    /// the chunk is empty, and `Uniform(WorldVoxel::Unset)` means the same.
    Uniform(WorldVoxel<I>),
}

#[derive(Default, PartialEq, Eq)]
pub enum ChunkSpawnStrategy {
    /// Spawn chunks that are within `spawning_distance` of the camera
    /// and also inside the viewport.
    #[default]
    CloseAndInView,

    /// Spawn chunks that are within `spawning_distance` of the camera, regardless of whether
    /// they are in the viewport or not. With the `FarAwayOrOutOfView` despawn strategy,
    /// chunks outside of the viewport are still only spawned within `min_despawn_distance`,
    /// since they would be despawned right away. If this strategy is used a flood fill will
    /// be used to find unspawned chunks and therefore it might make sense to lower the
    /// `spawning_rays` option.
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

    /// Distance in chunks around the camera within which chunks are spawned and never
    /// despawned, whether or not they are in view, regardless of the [`ChunkSpawnStrategy`]
    /// and [`ChunkDespawnStrategy`].
    fn min_despawn_distance(&self) -> u32 {
        1
    }

    /// Strategy for despawning chunks
    fn chunk_despawn_strategy(&self) -> ChunkDespawnStrategy {
        ChunkDespawnStrategy::default()
    }

    /// Strategy for spawning chunks
    fn chunk_spawn_strategy(&self) -> ChunkSpawnStrategy {
        ChunkSpawnStrategy::default()
    }

    /// Maximum number of chunks spawned in a frame.
    ///
    /// Chunks within `min_despawn_distance` of a camera spawn first, nearest
    /// first. The others follow largest on screen first, ranked by their
    /// distance divided by their extent, so a coarse chunk far away ranks the
    /// same as a fine chunk nearby.
    fn max_spawn_per_frame(&self) -> usize {
        10000
    }

    /// Number of rays to cast when spawning chunks.
    ///
    /// A ray passes through generated chunks that are not completely solid, and
    /// through missing chunks whose [fill hint](Self::chunk_fill_hint) says they
    /// are empty. It spawns the first chunks past them. So chunks spawn from the
    /// camera outwards as the chunks in front of them are generated.
    ///
    /// Higher values will result in more chunks being spawned per frame, but
    /// will also increase cpu load.
    fn spawning_rays(&self) -> usize {
        100
    }

    /// What the world knows about the voxels of the chunk at `key` before it is
    /// generated.
    ///
    /// Returns [`FillHint::Uniform`] when every voxel in the chunk is the same,
    /// and [`FillHint::Unknown`] otherwise, which is the default. Hints are an
    /// optional speed-up: the world works fully without them, and a hint never
    /// fills a chunk's data.
    ///
    /// Spawning rays pass through missing chunks hinted empty and stop at
    /// missing chunks hinted solid, and spawn neither. A world that hints where
    /// it is empty and where it is solid therefore streams only the chunks in
    /// between. Its rays reach those chunks, up to the spawning distance, in one
    /// frame instead of finding them chunk by chunk. Where no chunk covers a
    /// voxel, [`get_voxel`](crate::prelude::VoxelWorld::get_voxel) and
    /// [`raycast`](crate::prelude::VoxelWorld::raycast) answer from the hint of
    /// the voxel's level-0 chunk.
    ///
    /// Chunks within `min_despawn_distance` of a camera spawn whatever their
    /// hint, and are generated through the voxel lookup delegate like any other.
    /// In debug builds, a generated chunk whose voxels disagree with its hint
    /// logs a warning, once per world. A chunk reached by a voxel edit, within
    /// the chunk or in the level-0 chunks its padding samples, is treated as
    /// `Unknown`, since the hint doesn't know about voxel edits.
    ///
    /// This is called for every missing chunk a spawning ray passes through.
    /// With [`ChunkSpawnStrategy::Close`] it is also called each frame for every
    /// hinted chunk within the spawning distance, since the flood fill continues
    /// past hinted chunks without spawning them. Keep it cheap: a few
    /// comparisons on the key, not a noise sample.
    ///
    /// The hint must hold at every LOD level: a chunk above level 0 is uniform
    /// only when its whole extent is.
    fn chunk_fill_hint(&self, _key: ChunkKey) -> FillHint<Self::MaterialIndex> {
        FillHint::Unknown
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
    /// index mapper function.
    ///
    /// The delegate builds the mesh as for a level-0 chunk, with the corner of
    /// the chunk's first voxel at (1, 1, 1), just inside the padding. The mesh
    /// of a chunk above level 0 is then scaled up to the chunk's extent.
    ///
    /// Only positions are scaled; other attributes are left as built. A delegate
    /// whose texture coordinates advance by one per voxel, as the default
    /// mesher's do, should scale them by `1 << lod_level` itself, so the texture
    /// stays the same size at every level.
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
    /// All worlds with the same material type that return false share that handle.
    ///
    /// This can be used for example if you need to wait for a texture image to load before
    /// the material can be used.
    fn init_custom_materials(&self) -> bool {
        true
    }

    /// The LOD level the chunk at `chunk_position`, in level-0 chunks, should
    /// have for the camera at `camera_position`. Defaults to `0` for all chunks.
    ///
    /// The streaming policy picks the level at a position from the top down.
    /// Starting from the coarsest chunk containing the position, it splits a
    /// chunk while this function asks for a finer level than the chunk has. It
    /// asks about the chunk's level-0 chunk nearest the camera, so each chunk is
    /// as coarse as its nearest point allows. The coarsest level in use is what
    /// this returns at the spawning distance.
    ///
    /// When a position's level changes, the chunks at the new level spawn first,
    /// and the old ones retire once the new ones cover them. The levels overlap
    /// briefly instead of leaving a hole. A new chunk hidden from every camera
    /// behind solid chunks isn't waited for, since it may never spawn. Chunks
    /// generate largest on screen first, so far coarse chunks fill in alongside
    /// near fine ones.
    ///
    /// The level must not get finer with distance: for one camera and one
    /// `previous_lod`, a chunk farther from the camera than another must get a
    /// level at least as coarse. The streaming policy relies on this to call the
    /// function for only a few chunks.
    ///
    /// `previous_lod` is the level of the chunk covering the position now, or
    /// `None` when no chunk covers it yet. Implementors can use it to add
    /// hysteresis. While a split or merge is under way, it is the finer of the
    /// two levels.
    fn chunk_lod(
        &self,
        _chunk_position: IVec3,
        _previous_lod: Option<LodLevel>,
        _camera_position: Vec3,
    ) -> LodLevel {
        0
    }

    /// The padded shape of the voxel data generated for a chunk at the given LOD
    /// level.
    ///
    /// The default is the padded `CHUNK_SIZE` cube at every level, so a chunk at
    /// level `L` samples the world every `1 << L` voxels. A larger shape at a
    /// level samples its chunks more densely.
    fn chunk_data_shape(&self, _lod_level: LodLevel) -> UVec3 {
        UVec3::splat(PADDED_CHUNK_SIZE)
    }

    /// The padded shape that chunks at the given LOD level are meshed at. The
    /// default is the padded `CHUNK_SIZE` cube at every level.
    fn chunk_meshing_shape(&self, _lod_level: LodLevel) -> UVec3 {
        UVec3::splat(PADDED_CHUNK_SIZE)
    }

    /// Maximum number of chunk tasks that may run at once.
    ///
    /// Chunk tasks start in the order chunks spawn in (see
    /// [`max_spawn_per_frame`](Self::max_spawn_per_frame)). Only as many start
    /// at a time as keep the task pool busy until the next frame. A cap below
    /// that holds chunks back further, leaving room on the pool for other work.
    fn max_active_chunk_threads(&self) -> usize {
        usize::MAX
    }

    /// Whether chunk entities should be parented to the world root entity.
    ///
    /// Parenting makes it easy to transform or hide the entire voxel world at once,
    /// but it also forces Bevy's transform propagation system to walk the entire
    /// hierarchy whenever chunks spawn/despawn. Worlds with large numbers of chunks
    /// streaming into and out of view will perform better without a world_root.
    fn attach_chunks_to_root(&self) -> bool {
        true
    }

    fn init_root(&self, mut _commands: Commands, _root: Entity) {}
}

pub fn default_chunk_meshing_delegate<I: PartialEq + Copy, UB: Bundle>(
    pos: IVec3,
    lod_level: LodLevel,
    data_shape: UVec3,
    mesh_shape: UVec3,
    _previous_data: Option<ChunkData<I>>,
) -> ChunkMeshingFn<I, UB> {
    Box::new(
        move |voxels: VoxelArray<I>,
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
            let mut mesh = generate_chunk_mesh_for_shape(
                voxels_slice,
                pos,
                data_shape,
                mesh_shape,
                texture_index_mapper,
            );
            // The texture coordinates advance by one per voxel. The mesh of a
            // chunk above level 0 is scaled up to its extent, so scale them too.
            scale_texture_coordinates(&mut mesh, (1 << lod_level) as f32);
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
