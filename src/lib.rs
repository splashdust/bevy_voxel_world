mod chunk;
mod chunk_lifecycle;
mod chunk_map;
mod configuration;
mod debug_draw;
#[cfg(feature = "dislodged_fragments")]
mod dislodged_fragments;
pub mod edits;
mod grid;
#[cfg(feature = "light_propagation")]
mod light;
#[cfg(not(feature = "light_propagation"))]
mod light_off;
mod mesh_cache;
mod meshing;
mod plugin;
mod pool_budget;
#[cfg(feature = "sub_grids")]
mod snap;
mod streaming_policy;
mod voxel;
mod voxel_edits;
mod voxel_material;
mod voxel_traversal;
mod voxel_world;
mod voxel_world_internal;

pub mod prelude {
    pub use crate::chunk::{Chunk, ChunkKey, MAX_LOD_LEVEL};
    pub use crate::chunk_lifecycle::ChunkStatus;
    pub use crate::configuration::*;
    #[cfg(feature = "dislodged_fragments")]
    pub use crate::dislodged_fragments::Connectivity;
    pub use crate::edits::{EditDelta, EditGap, EditsCommitted};
    #[cfg(feature = "light_propagation")]
    pub use crate::light::config::{
        Emission, LevelCurve, LightAo, LightLook, LightPrecision, LightPropagation,
        LightShape, LightSmoothing, OpenFill, PathFill, Sky, SkyFill, SunGate,
    };
    #[cfg(feature = "light_propagation")]
    pub use crate::light::LightLevels;
    #[cfg(not(feature = "light_propagation"))]
    pub use crate::light_off::*;
    pub use crate::plugin::VoxelWorldPlugin;
    #[cfg(feature = "sub_grids")]
    pub use crate::snap::GridRotation;
    pub use crate::voxel::{VoxelFace, WorldVoxel, VOXEL_SIZE};
    pub use crate::voxel_world::{
        get_chunk_voxel_position, VoxelRaycastResult, VoxelWorld, VoxelWorldCamera,
        VoxelWorldSet,
    };
    pub use crate::voxel_world::{
        ChunkPublished, ChunkWillDespawn, ChunkWillRemesh, ChunkWillSpawn,
        ChunkWillUpdate,
    };
}

pub mod custom_meshing {
    pub use crate::chunk::PaddedChunkShape;
    pub use crate::chunk::VoxelArray;
    pub use crate::chunk::CHUNK_SIZE_F;
    pub use crate::chunk::CHUNK_SIZE_I;
    pub use crate::chunk::CHUNK_SIZE_U;
    pub use crate::meshing::generate_chunk_mesh;
    pub use crate::meshing::generate_chunk_mesh_for_shape;
    pub use crate::meshing::mesh_from_quads;
    pub use crate::meshing::scale_texture_coordinates;
}

pub mod debug {
    pub use crate::debug_draw::*;
}

pub mod rendering {
    //! Materials, vertex layout and shaders for chunk meshes.
    //!
    //! # Vertex outputs
    //!
    //! The crate's vertex shader, [`VOXEL_TEXTURE_SHADER_HANDLE`], returns the
    //! shader struct `VoxelVertexOutput`, which a custom fragment shader
    //! imports from `bevy_voxel_world::voxel_io` and takes as its input. Its
    //! fields, by location:
    //!
    //! | Location | Field | Holds |
    //! |---|---|---|
    //! | builtin | `position` | The clip-space position. |
    //! | 0 | `world_position` | The position in world space. |
    //! | 1 | `world_normal` | The normal in world space. |
    //! | 2 | `uv` | The mesher's texture coordinates. |
    //! | 5 | `color` | The mesher's vertex colour. The crate's mesher puts ambient occlusion in it. |
    //! | 6 | `instance_index` | Flat. Present under Bevy's `VERTEX_OUTPUT_INSTANCE_INDEX` shader def. |
    //! | 8 | `tex_idx` | Flat. The voxel's top, side and bottom texture indices from [`texture_index_mapper`](crate::prelude::VoxelWorldConfig::texture_index_mapper). |
    //! | 9 | `local_position` | The position in the chunk's padded local coordinates, centroid-sampled. |
    //! | 10 | `local_normal` | The normal in the chunk's padded local coordinates, as the mesh holds it. |
    //! | 11 | `light_tag` | Flat. The chunk's `MeshTag`, present only under the `VOXEL_LIGHT` shader def of a lit world. |
    //!
    //! Locations 3 and 4 hold Bevy's `uv_b` and `world_tangent` for a mesh
    //! that has them. The crate's mesher makes neither.
    //!
    //! # On a moving body
    //!
    //! `world_position`, `world_normal` and `world_tangent` follow the chunk's
    //! transform, so they change as the chunk moves or turns, for example
    //! when a world's root, set up in
    //! [`init_root`](crate::prelude::VoxelWorldConfig::init_root), is
    //! rotated. Every other output stays with the chunk and holds the same
    //! values wherever it is.
    //!
    //! So on a moving body, a texture read from `world_position` slides
    //! across the faces, and a face picked as top, side or bottom from
    //! `world_normal` swaps when the body turns. Read textures from `uv` or
    //! `local_position` and pick faces from `local_normal`, as the crate's
    //! fragment shader does. At an unrotated root both picks agree.
    //!
    //! # Local coordinates
    //!
    //! `local_position` and `local_normal` are in the chunk's padded local
    //! coordinates, the ones its mesh is built in. At LOD level 0, voxel `i`
    //! spans `i..i + 1` along each axis, and index 1 is the chunk's first
    //! voxel, since index 0 is the padding copied from its neighbours. A
    //! coarse chunk keeps the same units and origin, so its first voxel still
    //! begins at 1, but each of its cells spans `1 << L` voxels at LOD level
    //! `L`. At every level, `local_position - 1` is the offset in voxels from
    //! the chunk's minimum corner.
    //!
    //! A pattern worked out from `local_position` starts again in each chunk,
    //! so it is seamless across chunk borders only if its period divides the
    //! chunk size, [`CHUNK_SIZE_U`](crate::custom_meshing::CHUNK_SIZE_U)
    //! voxels.
    //!
    //! `local_position` is centroid-sampled: with MSAA, a pixel on a
    //! triangle's edge reads it at a point inside the triangle rather than at
    //! the pixel's centre, so that light is never read from a voxel beyond the
    //! face. Neighbouring pixels can then be sampled at points that aren't
    //! evenly spaced, so a texture whose mip level is chosen from
    //! `local_position` can flicker along triangle edges under MSAA.
    //!
    //! # Custom vertex shaders
    //!
    //! A custom material can keep the crate's vertex shader and
    //! [`vertex_layout`], and take `VoxelVertexOutput` in its own fragment
    //! shader. A custom vertex shader outputs `local_position` at location 9
    //! with `@interpolate(perspective, centroid)` and `local_normal` at
    //! location 10 in every world, and `light_tag` at location 11 with
    //! `@interpolate(flat)` under `VOXEL_LIGHT`. The crate's fragment shader
    //! picks textures from `local_normal`, so a vertex shader that declares
    //! location 10 but never writes it shows the side texture on every face.
    #![cfg_attr(
        feature = "light_propagation",
        doc = "",
        doc = "In a lit world, [`VoxelWorldBindings`] says what light needs from a mesh and from a custom vertex shader."
    )]

    #[cfg(feature = "light_propagation")]
    pub use crate::light::material::{VoxelWorldBindings, VoxelWorldMaterial};
    pub use crate::plugin::VoxelWorldMaterialHandle;
    pub use crate::voxel_material::vertex_layout;
    pub use crate::voxel_material::ATTRIBUTE_TEX_INDEX;
    pub use crate::voxel_material::VOXEL_TEXTURE_SHADER_HANDLE;
}

pub mod traversal_alg {
    pub use crate::voxel_traversal::*;
}

/// Hooks for the crate's own light benchmarks. Not public API: it can change
/// in any release.
#[cfg(feature = "light_propagation")]
#[doc(hidden)]
pub use crate::light::bench as light_bench;

#[cfg(test)]
mod test;
