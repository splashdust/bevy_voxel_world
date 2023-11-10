use std::sync::Arc;

use crate::voxel::WorldVoxel;
use bevy::prelude::*;

pub type VoxelLookupFn = Box<dyn FnMut(IVec3) -> WorldVoxel + Send + Sync>;
pub type VoxelLookupDelegate = Box<dyn Fn(IVec3) -> VoxelLookupFn + Send + Sync>;

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
    /// `FarAway`.
    Close,
}

/// Configuration resource for bevy_voxel_world
#[derive(Resource)]
pub struct VoxelWorldConfiguration {
    /// Distance in chunks to spawn chunks around the camera
    pub spawning_distance: u32,

    /// Strategy for despawning chunks
    pub chunk_despawn_strategy: ChunkDespawnStrategy,

    /// Strategy for spawning chunks
    /// This is only used if the despawn strategy is `FarAway`
    pub chunk_spawn_strategy: ChunkSpawnStrategy,

    /// Maximum number of chunks that can get queued for spawning in a given frame.
    /// In some scenarios, reducing this number can help with performance, due to less
    /// thread contention.
    pub max_spawn_per_frame: usize,

    /// Debugging aids
    pub debug_draw_chunks: bool,

    /// A function that maps voxel materials to texture coordinates.
    /// The input is the material index, and the output is a slice of three indexes into an array texture.
    /// The three values correspond to the top, sides and bottom of the voxel. For example,
    /// if the slice is `[1,2,2]`, the top will use texture index 1 and the sides and bottom will use texture
    /// index 2.
    pub texture_index_mapper: Arc<dyn Fn(u8) -> [u32; 3] + Send + Sync>,

    /// A function that returns a function that returns true if a voxel exists at the given position
    /// The delegate will be called every time a new chunk needs to be computed. The delegate should
    /// return a function that can be called to check if a voxel exists at a given position. This function
    /// needs to be thread-safe, since chunk computation happens on a separate thread.
    pub voxel_lookup_delegate: VoxelLookupDelegate,
}

impl Default for VoxelWorldConfiguration {
    fn default() -> Self {
        Self {
            spawning_distance: 10,
            chunk_despawn_strategy: ChunkDespawnStrategy::default(),
            chunk_spawn_strategy: ChunkSpawnStrategy::default(),
            debug_draw_chunks: true,
            max_spawn_per_frame: 10000,
            texture_index_mapper: Arc::new(|mat| match mat {
                0 => [0, 0, 0],
                1 => [1, 1, 1],
                2 => [2, 2, 2],
                3 => [3, 3, 3],
                _ => [0, 0, 0],
            }),
            voxel_lookup_delegate: Box::new(|_| Box::new(|_| WorldVoxel::Unset)),
        }
    }
}
