mod chunk;
mod chunk_map;
mod configuration;
mod mesh_cache;
mod meshing;
mod plugin;
mod voxel;
mod voxel_material;
mod voxel_world;
mod voxel_world_internal;

pub mod prelude {
    pub use crate::chunk::{Chunk, NeedsDespawn};
    pub use crate::configuration::*;
    pub use crate::plugin::VoxelWorldPlugin;
    pub use crate::voxel::WorldVoxel;
    pub use crate::voxel_world::{ChunkWillDespawn, ChunkWillRemesh, ChunkWillSpawn};
    pub use crate::voxel_world::{VoxelWorld, VoxelWorldCamera};
}

pub mod rendering {
    pub use crate::plugin::{VoxelWorldMaterialHandle, VoxelWorldMaterialPlugin};
    pub use crate::voxel_material::vertex_layout;
    pub use crate::voxel_material::VOXEL_TEXTURE_SHADER_HANDLE;
}
