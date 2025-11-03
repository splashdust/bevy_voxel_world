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
        get_chunk_voxel_position, VoxelRaycastResult, VoxelWorld, VoxelWorldCamera,
    };
    pub use crate::voxel_world::{
        ChunkWillDespawn, ChunkWillRemesh, ChunkWillSpawn, ChunkWillUpdate,
    };
}

pub mod custom_meshing {
    pub use crate::chunk::PaddedChunkShape;
    pub use crate::chunk::CHUNK_SIZE_F;
    pub use crate::chunk::CHUNK_SIZE_I;
    pub use crate::chunk::CHUNK_SIZE_U;
    pub use crate::meshing::generate_chunk_mesh;
    pub use crate::meshing::mesh_from_quads;
    pub use crate::chunk::VoxelArray;
}

pub mod debug {
    pub use crate::debug_draw::*;
}

pub mod rendering {
    pub use crate::plugin::VoxelWorldMaterialHandle;
    pub use crate::voxel_material::vertex_layout;
    pub use crate::voxel_material::ATTRIBUTE_TEX_INDEX;
    pub use crate::voxel_material::VOXEL_TEXTURE_SHADER_HANDLE;
}

pub mod traversal_alg {
    pub use crate::voxel_traversal::*;
}

#[cfg(test)]
mod test;
