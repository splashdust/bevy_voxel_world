# Changelog

## 0.19 (unreleased)

- Add `VoxelWorldSet<C>`, the system set in `PreUpdate` that runs world `C`. Order your systems after it to see this frame's chunks.
- Add `VoxelWorld::chunk_state`, which returns a chunk's `ChunkStatus`: `Pending`, `Generating`, `Ready` or `Retiring`.
- Add `ChunkPublished<C>`, fired whenever a chunk's voxel data is published, including after each remesh.
- Add `ChunkKey`, a chunk's position and LOD level. Chunk messages carry one, and `chunk_state` and `get_chunk_data` take one.
- Add `VoxelWorld::chunk_at`, which finds the finest spawned chunk covering a voxel.
- `LodLevel` is a level of detail: 0 is full detail, and a chunk at level `L` spans `CHUNK_SIZE << L` voxels on a side, contains the eight level `L - 1` chunks in that space, and by default samples every `1 << L` voxels.
- Add the `voxel_animation` example, which animates water by setting voxels every frame.
- Systems that use `VoxelWorld` can run in parallel, including ones that call `set_voxel`.
- `minimal()` worlds publish voxel data, so `get_voxel` and `ChunkWillSpawn` work without rendering.
- Chunk data no longer waits for the voxel texture to load; only meshes do.
- A chunk is despawned one frame after `ChunkWillDespawn`, so systems reading that message can still query it.
- Chunk meshes merge neighbouring faces that look the same into one quad: a chunk of noise terrain has 41% fewer quads, and flat ground is one quad per chunk.
- Chunks are generated nearest the camera first, and spawning rays stop at chunks that aren't generated yet instead of spawning everything behind them, so chunks hidden underground are hardly ever spawned. With the `noise_terrain` example's settings at 60 fps, half of the visible terrain is ready after 2.6 s instead of 3.3 s and 95% after 4.1 s instead of 10 s, from 23,600 generated chunks instead of 64,000.
- Performance: in the `chunk_lifecycle` benchmark, chunks stream in 2.5× faster (4.9× on flat terrain), edits are published 2.6× sooner, and frames while voxels are being edited take a quarter of the time (worst 1% from 2.06 ms to 0.49 ms). Meshing a chunk of noise terrain is 1.5× faster.
- Fix: a chunk edited every frame now publishes. Before, each edit cancelled its running task.
- Fix: when `set_voxel` sets a position more than once, the last voxel set wins, in `get_voxel` too. Setting an edited voxel to another voxel and back within a frame no longer remeshes its chunk.
- Fix: a voxel set to another material inside a chunk that is otherwise all one material stays in that chunk's data. Before, `get_chunk_data` reported the chunk's material there.
- Fix: spawning rays cover the whole viewport on high-DPI displays and with custom viewports.
- Fix: chunk positions round down at negative coordinates, so the right chunks spawn and stay around a camera there, and `ChunkData::has_voxel` works there.
- Fix: `ChunkSpawnStrategy::Close` with `FarAwayOrOutOfView` no longer spawns chunks that are despawned the next frame.
- Fix: with several cameras, a chunk's LOD level comes from the camera nearest the chunk's centre.

### Migrating from 0.18

- `Chunk` fields are now methods.
  - Before: `chunk.position`
  - Now: `chunk.position()`. The same goes for `lod_level`, `entity`, `data_shape` and `mesh_shape`.
- `NeedsDespawn` is removed.
  - Before: `Query<&Chunk<C>, Without<NeedsDespawn>>`
  - Now: `voxel_world.chunk_state(chunk.key()) != Some(ChunkStatus::Retiring)`
- `ChunkWillSpawn` fires once per chunk.
  - Before: `ChunkWillSpawn` fired again after every remesh.
  - Now: read `ChunkPublished` to react to every remesh.
- `get_chunk_data` returns `None` for a chunk that hasn't been published yet.
  - Before: a newly spawned chunk returned empty data.
  - Now: wait for `ChunkWillSpawn` before reading it. Raycasts skip unpublished chunks too.
- `VoxelWorld` has a second lifetime.
  - Before: `VoxelWorld<'_, MyWorld>`
  - Now: `VoxelWorld<'_, '_, MyWorld>`, or just `VoxelWorld<MyWorld>`.
- `ChunkDespawnStrategy::FarAwayOrOutOfView`, the default, despawns chunks that are out of view.
  - Before: a bug kept every chunk out of view within `spawning_distance`.
  - Now: chunks out of view are despawned beyond `min_despawn_distance`. To keep them, return `ChunkDespawnStrategy::FarAway` from `chunk_despawn_strategy`.
- `max_spawn_per_frame` caps spawns.
  - Before: it limited a queue of candidate chunks, not how many spawned.
  - Now: it is the most chunks spawned in a frame, nearest a camera first. If you set it, check the value still suits you.
- A chunk mesh quad can cover many voxels.
  - Before: each quad was one voxel's face, with texture coordinates from 0 to 1.
  - Now: texture coordinates still count one per voxel, so they run from 0 to the quad's width and height. A custom material that samples a texture with them needs a repeating sampler (`ImageAddressMode::Repeat`), as the built-in material has.
- Chunk messages carry a `ChunkKey`.
  - Before: `msg.chunk_key` was the chunk's position, an `IVec3`.
  - Now: `msg.chunk_key.position` is the position and `msg.chunk_key.lod_level` the LOD level.
- `chunk_state` and `get_chunk_data` take a `ChunkKey`.
  - Before: `voxel_world.get_chunk_data(chunk_pos)`
  - Now: `voxel_world.get_chunk_data(ChunkKey::new(chunk_pos, 0))`, or `voxel_world.chunk_at(voxel_pos)` for the key of the chunk covering a voxel.
- `ChunkWillChangeLod` and `ChunkRegenerateStrategy` are removed.
  - Before: a chunk given a new LOD level by `chunk_lod` was regenerated in place, reusing or repopulating its voxels.
  - Now: a chunk keeps its LOD level for as long as it exists. A remesh after a voxel edit reuses the chunk's generated voxels, as `ChunkRegenerateStrategy::Reuse` did.

## 0.18

- Update to Bevy 0.20

## 0.17

- Update to Bevy 0.19

## 0.16

- Fix: `set_voxel` now queues loaded neighboring chunks for remeshing when a changed voxel is part of their padded chunk data. This keeps meshes correct across chunk boundaries without requiring user code to manually mark adjacent chunks.
- Fix issue with voxel textures with a single texture index.
- Use minimal bevy features in cargo manifest
- Adds WASM support
- Add support for multiple voxelworld cameras (i.e, spawn chunks at multiple locations)

Thanks to @nathanaelneveux

## 0.15.1

- Fix: `set_voxel` no longer triggers a chunk remesh when the written value is identical to the current value. Previously, writing the same voxel value every frame would cause perpetual remesh cancellation — the in-progress async meshing task was dropped each frame before completion, preventing the chunk from ever rendering.

## 0.15.0

- Upgrade to Bevy 0.18

Thanks to @AndreRoelofs

## 0.14.0

- Introduce per-chunk level-of-detail and configurable chunk shapes
- Add `ChunkRegenerateStrategy`
- Add `max_active_chunk_threads()` and `attach_chunks_to_root()` knobs for better control over task-pool pressure and transform hierarchy costs.

Thanks to @nathanaelneveux

## 0.13.0

- Upgrade to Bevy 0.17
- Ensure Linux CI installs the required Wayland dependencies

Thanks to @nathanaelneveux

## 0.12.0

- Upgrade to Bevy 0.16 (thanks to @GabrielCarpr)

## 0.11.0

- Adds possibility of customizing the meshing step through a `chunk_meshing_delegate` function in the config.
- Add possibility to insert components bundles from meshing function.
- Makes various types and functions associated with meshing public.

Breaking Changes:

- `type ChunkUserBundle` now needs to be implemented for `VoxelWorldConfig`:

```rust
impl VoxelWorldConfig for MainWorld {
    type MaterialIndex = u8;
    type ChunkUserBundle = ();
    ...
}
```

## 0.10.2

- `ChunkData` is now public. You can get the data for a chunk by calling `voxel_world.get_chunk_data(chunk_pos)`
- `get_closest_surface_voxel`, `get_random_surface_voxel` and `get_surface_voxel_at_2d_pos` have been deprecated. It's better to use ray-casting instead to find voxels according to your needs.
- A system for debug-drawing chunks have been added. This can be used to visualize chunk boundaries in the world.
- Add `ChunkWillUpdate` event which is fired when `set_voxel` has been called
- Fix an error in the events that caused events to fire incorrectly.

## 0.10.1

Fix crash on macOS (#43)

## 0.10.0

Upgrade to Bevy 0.15

Thanks to @Touma-Kazusa2 for contributing this release

## 0.9.0

Use a generic type for voxel material index. See the `textures_custom_idx.rs` example for details on how this can be used.

Breaking Changes:

- `type MaterialIndex` now needs to be implemented for `VoxelWorldConfig`:

```rust
impl VoxelWorldConfig for MainWorld {
    type MaterialIndex = u8;
    ...
}
```

## 0.8.1

Add some debug drawing helpers

## 0.8.0

Upgrade to Bevy 0.14

Thanks to @JohnathanFL for contributing this release

## 0.7.0

New features:

- Added `voxel_line_traversal` and `voxel_cartesian_traversal` for versatile and fast traversal of the voxel grid (#24)
- Created a new bevy_voxel_world::traversal_alg module to make it easy for crate users to import those algorithms (#24)
- Rewrite of `raycast` using the traversal algorithm. (#25)

Breaking changes:

- `VoxelRaycastResult.normal` is now wrapped in an `Option<Vec3>`

Thanks to @dtaralla for contributing to this release!

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
