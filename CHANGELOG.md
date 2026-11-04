# Changelog

## 0.19.1

### Bug Fixes

- Each world's chunks now get that world's material and wait only for that world's voxel texture. Before, all worlds using the built-in voxel material, or the same custom material type, rendered with the material of the last `VoxelWorldPlugin` built, and shared its texture loading state.

- A world that lets the plugin create its custom material now always uses that material, so adding a `VoxelWorldMaterialHandle<M>` over the plugin's one no longer changes its chunks. To supply a handle yourself, return false from `VoxelWorldConfig::init_custom_materials`.

## 0.19

### Architecture

- Add `VoxelWorldSet<C>`, the system set in `PreUpdate` that runs world `C`. Order your systems after it to see this frame's chunks.

- Add `VoxelWorld::chunk_state`, which returns a chunk's `ChunkStatus`: `Pending`, `Generating`, `Ready` or `Retiring`.

- Add `ChunkPublished<C>`, fired whenever a chunk's voxel data is published, including after each remesh.

- `minimal()` worlds publish voxel data, so `get_voxel` and `ChunkWillSpawn` work without rendering.

- A chunk is despawned one frame after `ChunkWillDespawn`, so systems reading that message can still query it.

### LOD Updates

- Add `ChunkKey`, a chunk's position and LOD level. Chunk messages carry one, and `chunk_state` and `get_chunk_data` take one.

- Add `VoxelWorld::chunk_at`, which finds the finest spawned chunk covering a voxel.

- `LodLevel` is a level of detail: 0 is full detail, and a chunk at level `L` spans `CHUNK_SIZE << L` voxels on a side, contains the eight level `L - 1` chunks in that space, and by default samples every `1 << L` voxels.

- Chunks stream at the LOD level such that a far region needs no more chunks than a much smaller near one, and spawning rays cross far empty space in fewer steps. When a position’s level changes, chunks at the new level spawn first and the old ones retire once they’re covered (see the README’s Level of Detail section).

- Chunk tasks now start largest on screen first: chunks within min_despawn_distance of a camera, then the rest by distance divided by extent. Far coarse chunks therefore generate alongside near fine ones, and a world with only level 0 still generates nearest first.

- Added `VoxelWorldConfig::chunk_fill_hint` and `FillHint`, which let a world declare that a chunk is all one voxel type before it’s generated, so spawning rays pass through chunks hinted empty and stop at chunks hinted solid without spawning either.

- Voxel edits apply at every LOD level: a coarse chunk whose padded samples include an edited voxel regenerates with it.

- The default mesher scales its texture coordinates with the LOD level, so a texture keeps its size at every level.

### Optimizations

- Systems that use `VoxelWorld` can run in parallel, including ones that call `set_voxel`.

- Chunk data no longer waits for the voxel texture to load; only meshes do.

- Default mesher changed to a custom binary greedy mesher, about 1.5x faster than the old one. This also means bevy_voxel_world now uses greedy meshing by default, so flat faces are joined into larger quads.

- Chunks are generated nearest the camera first, and spawning rays stop at chunks that aren't generated yet instead of spawning everything behind them, so hidden chunks are less likely to spawn.

- Plus various minor optimizations here and there. Overall performance improvements of around 2.5x.

### Bug fixes

- Fix: a chunk edited every frame now publishes. Before, each edit cancelled its running task. See the `voxel_animation` example.

- Fix: when `set_voxel` sets a position more than once, the last voxel set wins, in `get_voxel` too. Setting an edited voxel to another voxel and back within a frame no longer remeshes its chunk.

- Fix: a voxel set to another material inside a chunk that is otherwise all one material stays in that chunk's data. Before, `get_chunk_data` reported the chunk's material there.

- Fix: spawning rays cover the whole viewport on high-DPI displays and with custom viewports.

- Fix: chunk positions round down at negative coordinates, so the right chunks spawn and stay around a camera there, and `ChunkData::has_voxel` works there.

- Fix: `ChunkSpawnStrategy::Close` with `FarAwayOrOutOfView` no longer spawns chunks that are despawned the next frame.

- Fix: with several cameras, a chunk's LOD level comes from the camera nearest the chunk.

### Migrating from 0.18

Go through these in order. `MyWorld` stands for your `VoxelWorldConfig` type.

#### 1. `VoxelWorld` has a second lifetime

```rust
// 0.18
fn my_system(voxel_world: VoxelWorld<'_, MyWorld>) {}

// 0.19
fn my_system(voxel_world: VoxelWorld<MyWorld>) {}
// or, with explicit lifetimes:
fn my_system(voxel_world: VoxelWorld<'_, '_, MyWorld>) {}
```

#### 2. `Chunk` fields are now methods

```rust
// 0.18
let position = chunk.position;
let lod_level = chunk.lod_level;

// 0.19
let position = chunk.position();
let lod_level = chunk.lod_level();
```

The same goes for `entity()`, `data_shape()` and `mesh_shape()`. The new `chunk.key()` returns the position and LOD level together as a `ChunkKey`.

#### 3. Chunk messages carry a `ChunkKey`, not an `IVec3`

This applies to `ChunkWillSpawn`, `ChunkWillDespawn`, `ChunkWillRemesh` and `ChunkWillUpdate`.

```rust
// 0.18
for msg in spawned.read() {
    let position: IVec3 = msg.chunk_key;
}

// 0.19
for msg in spawned.read() {
    let position: IVec3 = msg.chunk_key.position;
    let lod_level: LodLevel = msg.chunk_key.lod_level;
}
```

#### 4. `get_chunk_data` and `get_chunk_data_fn` take a `ChunkKey`

```rust
// 0.18
let data = voxel_world.get_chunk_data(chunk_pos);

// 0.19, for the level-0 chunk at a chunk position
let data = voxel_world.get_chunk_data(ChunkKey::new(chunk_pos, 0));

// 0.19, for whichever chunk covers a voxel, at any LOD level
let data = voxel_world
    .chunk_at(voxel_pos)
    .and_then(|key| voxel_world.get_chunk_data(key));
```

The closure returned by `get_chunk_data_fn` also takes a `ChunkKey` now.

#### 5. `get_chunk_data` returns `None` until a chunk's data is published

In 0.18 a chunk that had just spawned returned empty data. Now it returns `None` until its first chunk task has finished, and raycasts skip such chunks too. Read a new chunk's data when its `ChunkWillSpawn` arrives, since that message now fires once the data is published:

```rust
// 0.19
fn on_chunk_spawned(
    mut spawned: MessageReader<ChunkWillSpawn<MyWorld>>,
    voxel_world: VoxelWorld<MyWorld>,
) {
    for msg in spawned.read() {
        if let Some(data) = voxel_world.get_chunk_data(msg.chunk_key) {
            // ...
        }
    }
}
```

#### 6. `ChunkWillSpawn` fires once per chunk

In 0.18 `ChunkWillSpawn` also fired after every remesh. If your system relied on that, read `ChunkPublished` instead, which fires on the first publish and after every remesh:

```rust
// 0.18
fn on_chunk_changed(mut msgs: MessageReader<ChunkWillSpawn<MyWorld>>) {}

// 0.19
fn on_chunk_changed(mut msgs: MessageReader<ChunkPublished<MyWorld>>) {}
```

#### 7. `NeedsDespawn` is removed

Ask the voxel world for the chunk's state instead:

```rust
// 0.18
fn my_system(chunks: Query<&Chunk<MyWorld>, Without<NeedsDespawn>>) {
    for chunk in &chunks {
        // ...
    }
}

// 0.19
fn my_system(chunks: Query<&Chunk<MyWorld>>, voxel_world: VoxelWorld<MyWorld>) {
    for chunk in &chunks {
        if voxel_world.chunk_state(chunk.key()) == Some(ChunkStatus::Retiring) {
            continue;
        }
        // ...
    }
}
```

#### 8. `ChunkWillChangeLod` is removed

A chunk now keeps its LOD level for as long as it exists. When `chunk_lod` asks for another level, new chunks at that level spawn and the old chunk is despawned once they cover it.

#### 9. `ChunkRegenerateStrategy` is removed

Delete your `chunk_regenerate_strategy` override, if you have one:

```rust
// 0.18: delete this
fn chunk_regenerate_strategy(&self) -> ChunkRegenerateStrategy {
    ChunkRegenerateStrategy::Repopulate
}
```

A remesh after a voxel edit always reuses the chunk's generated voxels, as `ChunkRegenerateStrategy::Reuse` did.

#### 10. `chunk_lod` returns a LOD level, not a sample stride

In 0.18 you returned a stride and turned it into a smaller shape yourself. Now level `L` means a chunk `CHUNK_SIZE << L` voxels wide, sampled every `1 << L` voxels with the default shapes. Return `0, 1, 2, 3` where you returned strides of `1, 2, 4, 8`, and delete the shape overrides that divided by the stride:

```rust
// 0.18
fn chunk_lod(&self, chunk_position: IVec3, _previous_lod: Option<LodLevel>, camera_position: Vec3) -> LodLevel {
    let distance = chunk_position.as_vec3().distance((camera_position / CHUNK_SIZE_F).floor());
    if distance < 10.0 {
        1
    } else if distance < 25.0 {
        2
    } else {
        4
    }
}

fn chunk_data_shape(&self, lod_level: LodLevel) -> UVec3 {
    padded_chunk_shape_uniform(CHUNK_SIZE_U / lod_level.max(1) as u32)
}

fn chunk_meshing_shape(&self, lod_level: LodLevel) -> UVec3 {
    padded_chunk_shape_uniform(CHUNK_SIZE_U / lod_level.max(1) as u32)
}

// 0.19: no chunk_data_shape or chunk_meshing_shape overrides
fn chunk_lod(&self, chunk_position: IVec3, _previous_lod: Option<LodLevel>, camera_position: Vec3) -> LodLevel {
    let distance = chunk_position.as_vec3().distance((camera_position / CHUNK_SIZE_F).floor());
    if distance < 10.0 {
        0
    } else if distance < 25.0 {
        1
    } else {
        2
    }
}
```

#### 11. A custom mesher used with LOD levels should scale its texture coordinates

The mesh your `chunk_meshing_delegate` returns is built as for a level-0 chunk, and the crate scales it up to the size of a chunk above level 0. Only the positions are scaled. If your texture coordinates advance by one per voxel, scale them yourself so the texture keeps its size, as the default mesher does:

```rust
// 0.19, inside your chunk_meshing_delegate, where `lod_level` is the
// delegate's second argument
use bevy_voxel_world::custom_meshing::scale_texture_coordinates;

let mut mesh = build_my_mesh(voxels);
scale_texture_coordinates(&mut mesh, (1 << lod_level) as f32);
```

#### 12. A custom material needs a repeating texture sampler

The default mesher now merges flat faces into larger quads. Texture coordinates still advance by one per voxel, so on a quad they run from 0 to the quad's width and height, not from 0 to 1. The built-in material handles this. A custom material that samples a texture with these coordinates must set its sampler to repeat:

```rust
// 0.19
image.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
    address_mode_u: ImageAddressMode::Repeat,
    address_mode_v: ImageAddressMode::Repeat,
    ..default()
});
```

#### 13. `ChunkDespawnStrategy::FarAwayOrOutOfView` now despawns chunks out of view

This is the default strategy. In 0.18 a bug kept every chunk within `spawning_distance`, in view or not. Now chunks out of view are despawned unless they are within `min_despawn_distance` of a camera. To keep the 0.18 behaviour:

```rust
// 0.19
fn chunk_despawn_strategy(&self) -> ChunkDespawnStrategy {
    ChunkDespawnStrategy::FarAway
}
```

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
