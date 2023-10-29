use std::sync::Arc;

use crate::voxel::WorldVoxel;
use bevy::prelude::*;

///
/// Configuration for the voxel world
///
#[derive(Resource)]
//#[reflect(Resource)]
pub struct VoxelWorldConfiguration {
    /// Distance in chunks to spawn chunks around the camera
    pub spawning_distance: u32,

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
    pub voxel_lookup_delegate:
        Box<dyn Fn(IVec3) -> Box<dyn FnMut(IVec3) -> WorldVoxel + Send + Sync> + Send + Sync>,
}

impl Default for VoxelWorldConfiguration {
    fn default() -> Self {
        Self {
            spawning_distance: 10,
            debug_draw_chunks: false,
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
