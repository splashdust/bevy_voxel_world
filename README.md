# bevy_voxel_world

[![Crates.io](https://img.shields.io/crates/v/bevy_voxel_world.svg)](https://crates.io/crates/bevy_voxel_world)
[![License](https://img.shields.io/badge/license-MIT%2FApache-blue.svg)](https://github.com/bevyengine/bevy#license)
[![Bevy tracking](https://img.shields.io/badge/Bevy%20tracking-released%20version-lightblue)](https://github.com/bevyengine/bevy/blob/main/docs/plugins_guidelines.md#main-branch-tracking)

---

## What is `bevy_voxel_world`

This plugin makes it easy to generate and modify voxel terrains in Bevy. `bevy_voxel_world` handles multithreaded meshing, chunk spawning/despawning, texture mapping and provides an easy to use API that can be accessed from any system.

![bvw_480](https://github.com/splashdust/bevy_voxel_world/assets/428824/98d25cd1-0a6c-4618-b0de-0e16ca5af636)

```bash
$ cargo run -r --example noise_terrain
```

The world can be controlled in two main ways: through a terrain lookup function, and directly by `set_voxel` and `get_voxel` functions. The world has two "layers" of voxel information, one that is procedural and determined by the terrain lookup function, and one that is controlled by `set_voxel` and persisted in a `HashMap`. The persistent layer always overrides the procedural layer. This way, the world can be infinitely large, but we only need to store information about voxels that are deliberately changed. In the current implementation, the proceduaral layer is cached for spawned chunks, so it may still use a lot of memory if the spawning distance is large.

For an example on how to use a terrain lookup function, see [this example](https://github.com/splashdust/bevy_voxel_world/blob/main/examples/noise_terrain.rs).

## Modifying the world

The `set_voxel` and `get_voxel` access functions are easily reached from any normal Bevy system:

```rust
fn my_system(mut voxel_world: VoxelWorld) {
    voxel_world.set_voxel(IVec3 { ... }, WorldVoxel::Solid(0));
}
```

This will update the voxel value at the given location in the persisting `HashMap`, and cause `bevy_voxel_world` to queue the affected chunk for re-meshing.

Voxels are keyed by their XYZ coordinate in the world, specified by an `IVec3`. The type of voxel is specified by the `WorldVoxel` type. A voxel can be `Unset`, `Air` or `Solid`.

## Voxel materials

`Solid` voxels holds a `u8` material type value. Thus, a maximum of 256 material types are supported. Material types can easily be mapped to indexes in a 2d texture array though a mapping callback.

A custom array texture can be supplied when initializing the plugin:

```rust
VoxelWorldPlugin::default()
    .with_voxel_texture("images/materials.png", 6)
```

This should be image with a size of `W x (W * n)`, where `n` is the number of indexes. So an array of 4 16x16 px textures would be 16x64 px in size. The number of indexes is specified in the second parameter (6 in the example above).

Then, to map out which indexes belong to which material type, you can supply a `texture_index_mapper` callback:

```rust
commands.insert_resource(VoxelWorldConfiguration {
    texture_index_mapper: Arc::new(|vox_mat: u8| {
        match vox_mat {
            // Top brick
            0 => [0, 1, 2],

            // Full brick
            1 => [2, 2, 2],

            // Grass
            2 | _ => [3, 3, 3],
        }
    }),
    ..Default::default()
});
```

The `texture_index_mapper` callback is supplied with a material type and should return an array with three values. The values indicate which texture index maps to `[top, sides, bottom]` of a voxel.

See the [textures example](https://github.com/splashdust/bevy_voxel_world/blob/main/examples/textures.rs) for a runnable example of this.

<img width="558" alt="Screenshot 2023-11-06 at 21 50 05" src="https://github.com/splashdust/bevy_voxel_world/assets/428824/382fdcf7-9d70-4432-b2ba-18479d34346f">

### Custom shader support

If you need to customize materials futher, you can use `VoxelWorldMaterialPlugin` to register your own Bevy material. This allows you to use your own custom shader with `bevy_voxel_world`. See [this example](https://github.com/splashdust/bevy_voxel_world/blob/main/examples/custom_material.rs) for more details.

## Gotchas

`bevy_voxel_world` began as an internal part of a game that I'm working on, but I figured that it could be useful as a standalone plugin, for myself and perhaps for others, so I decided to break it out and make it public as a crate.

In its current state, there are still various hard-coded assumptions that works well enough for my usecase, but may not suit everyone. Over time, the aim is to generalize and make `bevy_voxel_world` more configurable. There are also many potential performance optimizations that I have not prioritized yet at this point.

Currently only "blocky", Minecraft-like, voxels are supported, and there is no support for "half-slabs". Meshing is handled by [block-mesh-rs](https://github.com/bonsairobo/block-mesh-rs), and only the "simple" algorithm is used (i.e, no greedy meshing.)

Feedback, issues and pull requests are welcomed!

---

### Bevy compatibility

| bevy | bevy_voxel_world |
| ---- | ---------------- |
| 0.13 | ^0.4.0           |
| 0.12 | ^0.3.0           |
| 0.11 | 0.2.2            |
