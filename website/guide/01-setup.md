---
title: Setting up a world
api:
  - VoxelWorldPlugin
  - VoxelWorldConfig
  - DefaultWorld
  - VoxelWorldCamera
  - VoxelWorldSet
  - ChunkSpawnStrategy
  - ChunkDespawnStrategy
---

Create a configuration struct for your world:

```rust
#[derive(Resource, Clone, Default)]
struct MyWorld;

impl VoxelWorldConfig for MyWorld {
    type MaterialIndex = u8;
    type ChunkUserBundle = ();

    // All options have defaults, so you only need to add the ones you want to modify.
    // For a full list, see src/configuration.rs
    fn spawning_distance(&self) -> u32 {
        25
    }
}
```

Then add the plugin with your config:

```rust
.add_plugins(VoxelWorldPlugin::with_config(MyWorld))
```

The config struct does two things:

- It supplies the configuration values
- Its type also acts as a world instance identifier. This means that you can create multiple worlds by adding multiple instances of the plugin as long as each instance has a unique configuration struct. [Here's an example of two worlds using different materials](https://github.com/splashdust/bevy_voxel_world/blob/main/examples/multiple_worlds.rs)
