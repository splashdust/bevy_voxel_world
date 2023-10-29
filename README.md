# bevy_voxel_world

[![License](https://img.shields.io/badge/license-MIT%2FApache-blue.svg)](https://github.com/bevyengine/bevy#license)
[![Bevy tracking](https://img.shields.io/badge/Bevy%20tracking-released%20version-lightblue)](https://github.com/bevyengine/bevy/blob/main/docs/plugins_guidelines.md#main-branch-tracking)

---

## What is `bevy_voxel_world`

This plugin makes it easy to generate and modify voxel terrains in Bevy. `bevy_voxel_world` handles multithreaded meshing, chunk spawning/despawning, texture mapping and provides an easy to use API that can be accessed from any system.

![output](https://github.com/splashdust/bevy_voxel_world/assets/428824/24a9ffd0-6b9b-40d8-aa66-c72dac575f09)

The voxel terrain can be controlled in two main ways: through a terrain lookup function, and by direct `set_voxel` and `get_voxel` functions. The world has two "layers", one that is purely procedural and determined by the terrain lookup function, and one that is data based and controlled by `set_voxel`. The data layer always overrides the procedural layer. This way, the world can be infinitely large, but we only need to store information about voxels that are deliberately changed.

The `set_voxel` and `get_voxel` access functions are easily reached from any normal Bevy system:

```rust
fn my_system(mut voxel_world: VoxelWorld) {
    voxel_world.set_voxel(IVec3 { ... });
}
```

Voxels are keyed by their XYZ coordinate in the world, specified by an `IVec3`.

For an example on how to use a terrain lookup function, see [this example](https://github.com/splashdust/bevy_voxel_world/blob/main/examples/noise_terrain.rs).

## Gotchas

`bevy_voxel_world` began as an internal part of a game that I'm working on, but I figured that it could be useful as a standalone plugin, for myself and perhaps for others, so I decided to break it out and make it public as a crate.

In its current state (as of 0.1.0), there are still various hard-coded assumptions that works well enough for my usecase, but may not suit everyone. Over time, I aim to generalize and make `bevy_voxel_world` more configurable. There are also many potential performance optimizations that I have not prioritized yet at this point.

Currently only "blocky", Minecraft-like, voxels are supported, and there is no support for "half-slabs".

Feedback, issues and pull requests are welcomed!

---

### Bevy compatibility

| bevy | bevy_voxel_world |
| ---- | ---------------- |
| 0.11 | 0.1              |
