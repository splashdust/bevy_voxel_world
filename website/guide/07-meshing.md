---
title: Custom meshing
api:
  - ChunkMeshingDelegate
  - ChunkMeshingFn
  - default_chunk_meshing_delegate
  - generate_chunk_mesh
  - generate_chunk_mesh_for_shape
  - mesh_from_quads
  - scale_texture_coordinates
  - padded_chunk_shape
  - padded_chunk_shape_uniform
  - PaddedChunkShape
  - VoxelArray
  - CHUNK_SIZE_U
  - CHUNK_SIZE_I
  - CHUNK_SIZE_F
---

Custom meshing is supported though a delegate function that can be specified in the config. The function recives an array of voxel data for the chunk to be meshed, and a reference to the texture index mapper functon. This delegate is then called from a thread in the task pool and needs to return a Bevy Mesh that can be inserted on the chunk entity.

The meshing delegate function can also return a Bevy component or bundle that will be inserted along with the mesh. This is useful for generating additional data based on the meshing data, such as nav meshes. It can be beneficial to do all such processing at once, while the voxel data is already hot in the caches.

See this [full example of custom meshing](https://github.com/splashdust/bevy_voxel_world/blob/main/examples/custom_meshing.rs) for more details.

<img width="558" src="https://github.com/user-attachments/assets/13f46fd6-81a8-4c93-943c-9c5fc7b9b38e"/>

## Other shapes

<img width="800" height="445" alt="" src="https://raw.githubusercontent.com/splashdust/bevy_voxel_world/main/assets/readme/marching_cubes.webp" />

The delegate doesn't have to produce cubical voxels. The [marching cubes example](https://github.com/splashdust/bevy_voxel_world/blob/main/examples/marching_cubes.rs) turns each chunk's voxels into a density field and extracts a smooth surface with the [transvoxel](https://crates.io/crates/transvoxel) crate, rendered with a `StandardMaterial` and vertex colours instead of the default voxel material.
