# Changelog

## 0.6.0

New features:

- Add function that returns a sendable raycast function (3335f30)
- Add WorldConfig constraint to the Chunk events (ebbd03d)
- Add World Config type parameter to the VoxelWorldCamera and CameraInfo (0349632)
- Spawn chunks on a root node and add `init_root`callback (6e5df25, f1fc4c6, a0fc663)

Breaking changes:

- Move raycast functions to main VoxelWorld system param (262b124)

Thanks to @aligator for contributing to this release!

## 0.5.1

Fix lingering meshes when an existing chunk is emptied of voxels

## 0.5.0

Add support for multiple parallell world instances

**Breaking changes:**

Configuration is now supplied when adding the plugin

```rust
// First declare a config struct. It needs to derive `Resource`, `Clone` and `Default`
#[derive(Resource, Clone, Default)]
struct MyWorld;

// Then implement the `VoxelWorldConfig` trait for it:
impl VoxelWorldConfig for MyWorld {
    // All the trait methods have defaults, so you only need to add the ones you want to alter
    fn spawning_distance(&self) -> u32 {
       15
    }
}
```

Then when adding the plugin:

```rust
.add_plugins(VoxelWorldPlugin::with_config(MyWorld))
```

If you don't want to change any default config, you can simply do this:

```rust
.add_plugins(VoxelWorldPlugin::default())
```

Adding multiple worlds follows the same pattern. Just create different configuration structs and add a `VoxelWorldPlugin` for each.

```rust
.add_plugins(VoxelWorldPlugin::with_config(MyOtherWorld))
```

Each world instance can have it's own configuration and will keep track of it's own set of voxel data and chunks.

### The `VoxelWorld` system param now needs a type parameter to specify which world instance you want to select

The configuration struct adds the config values and its type also acts as a marker for the world instance.

```rust
fn my_system(
  my_voxel_world: VoxelWorld<MyWorld>,
  my_other_voxel_world: VoxelWorld<MyOtherWorld>
) {
  // Set a voxel in `my_voxel_world`
  my_voxel_world.set_voxel(pos, WorldVoxel::Solid(voxel_type))

  // Set a voxel in `my_other_voxel_world`
  my_other_voxel_world.set_voxel(pos, WorldVoxel::Solid(voxel_type))
}
```

If you initialized the plugin with `::default()`, you still need to explicitly specify the instance as `DefaultWorld`:

```rust
fn my_system(voxel_world: VoxelWorld<DefaultWorld>) { ... }
```

The `VoxelWorldRaycast` system param now also requires the same config type paramter as described above.

## 0.4.0

- Update to Bevy 0.13

## 0.3.6

- Fix some issues with ray casting

## 0.3.5

- Add support for using custom Bevy materials. This makes it easy to use custom shaders with `bevy_voxel_world`.
- Add a built-in method for ray casting into the world. Usefull if you want to know which voxe is under the mouse cursor for instance.

## 0.3.4

- Avoid issues with chuck entities that have already been despawned, by using `try_insert` instead of regular `insert`. #12 & #14

## 0.3.3

- Defer `ChunkWillSpawn` event until buffers are applied. Fixes issues caused by the event getting fired before the chunk data can actually be looked up.
- Fix texture bleeding issue on negative Y faces (#10)

## 0.3.2

- Performance improvements:
  - More granular resources for more parallelization opportunities.
  - Pre-calculate hash for voxels array for faster mesh cache lookups
  - Buffered iserts/updates and removes for ChunkMap, to reduce time spent waiting to aquire RwLock.

## 0.3.1

- Add lookup map for mesh handles. This allows `bevy_voxel_world` to re-use mesh handles for identical chunks and thereby utilising Bevy's automatic instancing while also avoiding redundant meshing.
- Change the default chunk discovery algorith to only use ray cating. This uses less CPU than the previous flood fill method, and also works with larger spawn distances. The flood fill method can still be used by setting `ChunkSpawnStrategy::Close`
- Add various config options for tuning spawning behaviour for different needs.

## 0.3.0

- Update to Bevy 0.12

## 0.2.2

- Fix an issue where filled underground chunks would never get meshed, even if they were modified
- `ChunkWillSpawn` event now only fire when actually meshed chunks spawn, preventing massive spam of this event.

## 0.2.1

- Move voxel data to `ChunkMap` instead of the `Chunk` component. This makes `get_voxel()` much faster, because we don't need to `collect()` all the `Chunk`s to find the correct voxel data.

## 0.2.0

- Rewrite spawning system\
   The old system would just spawn a cubic volume of chunks with a fixed height of 2 chunks. The new system spawns a spherical volume instead, and uses a combination of ray casting and flood fill to minimize the amount of work needed to find unspawned chunks within the volume.
- Add `ChunkDespawnStrategy` config option\
   Configure wheter chunks should despawned when not in view, or only when outside of `spawning_distance`

## 0.1.0

Initial release
