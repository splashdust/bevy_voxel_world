mod chunk;
mod chunk_map;
mod configuration;
mod debug_draw;
mod mesh_cache;
mod meshing;
mod plugin;
mod voxel;
mod voxel_material;
mod voxel_traversal;
mod voxel_world;
mod voxel_world_internal;

pub mod prelude {
    pub use crate::chunk::{Chunk, NeedsDespawn};
    pub use crate::configuration::*;
    pub use crate::plugin::VoxelWorldPlugin;
    pub use crate::voxel::{VoxelFace, WorldVoxel, VOXEL_SIZE};
    pub use crate::voxel_world::{
        ChunkWillDespawn, ChunkWillRemesh, ChunkWillSpawn, ChunkWillUpdate,
    };
    pub use crate::voxel_world::{VoxelRaycastResult, VoxelWorld, VoxelWorldCamera};
}

pub mod debug {
    pub use crate::debug_draw::*;
}

pub mod rendering {
    pub use crate::plugin::VoxelWorldMaterialHandle;
    pub use crate::voxel_material::vertex_layout;
    pub use crate::voxel_material::VOXEL_TEXTURE_SHADER_HANDLE;
}

pub mod traversal_alg {
    pub use crate::voxel_traversal::*;
}

#[cfg(test)]
mod test;
