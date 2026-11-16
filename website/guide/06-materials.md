---
title: Materials and shaders
module: rendering
api:
  - TextureIndexMapperFn
  - VoxelWorldMaterialHandle
  - VOXEL_TEXTURE_SHADER_HANDLE
  - vertex_layout
  - ATTRIBUTE_TEX_INDEX
  - VoxelWorldBindings
  - VoxelWorldMaterial
---

`Solid` voxels holds a material type value. You can configure the type of the material index, but it's advisable to keep it small (like `u8`), since each voxel will hold one of these values. Material types can easily be mapped to indexes in a 2d texture array though a mapping callback. The mapping callback always returns a `[u32; 3]` which is passed along to the shader.

A custom array texture can be supplied in the config. It should be image with a size of `W x (W * n)`, where `n` is the number of indexes. So an array of 4 16x16 px textures would be 16x64 px in size. The number of indexes is specified in the second parameter.

Then, to map out which indexes belong to which material type, you can supply a `texture_index_mapper` callback:

```rust
impl VoxelWorldConfig for MyWorld {
    // In this example we use a `u8` for the index.
    type MaterialIndex = u8;
    type ChunkUserBundle = ();

    fn texture_index_mapper(&self) -> Arc<dyn Fn(u8) -> [u32; 3] + Send + Sync> {
        Arc::new(|vox_mat: u8| match vox_mat {
            SNOWY_BRICK => [0, 1, 2],
            FULL_BRICK => [2, 2, 2],
            GRASS | _ => [3, 3, 3],
        })
    }

    fn voxel_texture(&self) -> Option<(String, u32)> {
        Some(("example_voxel_texture.png".into(), 4)) // Array texture with 4 indexes
    }
}
```

The `texture_index_mapper` callback is supplied with a material type and should return an array with three values. The values indicate which texture index maps to `[top, sides, bottom]` of a voxel.

See the [textures example](https://github.com/splashdust/bevy_voxel_world/blob/main/examples/textures.rs) for a runnable example of this.

<img width="558" alt src="https://github.com/splashdust/bevy_voxel_world/assets/428824/382fdcf7-9d70-4432-b2ba-18479d34346f">

## Custom shader support

If you need to customize materials futher, you can use `.with_material(MyCustomVoxelMaterial)`, when adding the plugin, to register your own Bevy material. This allows you to use your own custom shader with `bevy_voxel_world`. See [this example](https://github.com/splashdust/bevy_voxel_world/blob/main/examples/custom_material.rs) for more details.
