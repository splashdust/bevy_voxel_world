# bevy_voxel_world

[![Crates.io](https://img.shields.io/crates/v/bevy_voxel_world.svg)](https://crates.io/crates/bevy_voxel_world)
[![License](https://img.shields.io/badge/license-MIT%2FApache-blue.svg)](https://github.com/bevyengine/bevy#license)
[![Bevy tracking](https://img.shields.io/badge/Bevy%20tracking-released%20version-lightblue)](https://github.com/bevyengine/bevy/blob/main/docs/plugins_guidelines.md#main-branch-tracking)

---

## What is `bevy_voxel_world`

This plugin makes it easy to generate and modify voxel terrains in Bevy. `bevy_voxel_world` handles multithreaded meshing, chunk spawning/despawning, texture mapping and provides an easy to use API that can be accessed from any system.

<img width="700" height="389" alt="bombs" src="https://github.com/user-attachments/assets/6c768a17-c370-4480-98fc-6bf4b4fae22c" />

```bash
$ cargo run -r --example bombs
```

The world can be controlled in two main ways: through a terrain lookup function, and directly by `set_voxel` and `get_voxel` functions. The world has two "layers" of voxel information, one that is procedural and determined by the terrain lookup function, and one that is controlled by `set_voxel` and persisted in a `HashMap`. The persistent layer always overrides the procedural layer. This way, the world can be infinitely large, but we only need to store information about voxels that are deliberately changed. In the current implementation, the proceduaral layer is cached for spawned chunks, so it may still use a lot of memory if the spawning distance is large.

For an example on how to use a terrain lookup function, see [this example](https://github.com/splashdust/bevy_voxel_world/blob/main/examples/noise_terrain.rs).

## Basic setup

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

## Accessing the world

To access a voxel world instance in a system, you can use the `VoxelWorld` system param. `VoxelWorld` take one type parameter, which is the configuration struct for the world you want to access.

The `set_voxel` and `get_voxel` access functions can be used to manipulate the voxel data in the world.

```rust
fn my_system(mut voxel_world: VoxelWorld<MyWorld>) {
    voxel_world.set_voxel(IVec3 { ... }, WorldVoxel::Solid(0));
}
```

This will update the voxel value at the given location in the persisting `HashMap`, and cause `bevy_voxel_world` to queue the affected chunk for re-meshing.

Voxels are keyed by their XYZ coordinate in the world, specified by an `IVec3`. The type of voxel is specified by the `WorldVoxel` type. A voxel can be `Unset`, `Air` or `Solid`.

<img width="800" height="445" alt="voxel_animation" src="https://github.com/user-attachments/assets/7dafe892-4035-4cc5-92b3-51c73dd3071c" />

## Voxel materials

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

### Custom shader support

If you need to customize materials futher, you can use `.with_material(MyCustomVoxelMaterial)`, when adding the plugin, to register your own Bevy material. This allows you to use your own custom shader with `bevy_voxel_world`. See [this example](https://github.com/splashdust/bevy_voxel_world/blob/main/examples/custom_material.rs) for more details.

## Ray casting

To find a voxel location in the world from a pixel location on the screen, for example the mouse location, you can ray cast into the voxel world.

```rust
fn do_something_with_mouse_voxel_pos(
    voxel_world: VoxelWorld<MyWorld>,
    camera_info: Query<(&Camera, &GlobalTransform), With<VoxelWorldCamera<MyWorld>>>,
    mut cursor_evr: EventReader<CursorMoved>,
) {
    for ev in cursor_evr.read() {
        // Get a ray from the cursor position into the world
        let (camera, cam_gtf) = camera_info.single();
        let Some(ray) = camera.viewport_to_world(cam_gtf, ev.position) else {
            return;
        };

        if let Some(result) = voxel_world.raycast(ray, &|(_pos, _vox)| true) {
            // result.position will be the world location of the voxel as a Vec3
            // To get the empty location next to the voxel in the direction of the surface where the ray intersected you can use result.normal:
            // let empty_pos = result.position + result.normal;
        }
    }
}
```

See this [full example of ray casting](https://github.com/splashdust/bevy_voxel_world/blob/main/examples/ray_cast.rs) for more details.

## Custom meshing

Custom meshing is supported though a delegate function that can be specified in the config. The function recives an array of voxel data for the chunk to be meshed, and a reference to the texture index mapper functon. This delegate is then called from a thread in the task pool and needs to return a Bevy Mesh that can be inserted on the chunk entity.

The meshing delegate function can also return a Bevy component or bundle that will be inserted along with the mesh. This is useful for generating additional data based on the meshing data, such as nav meshes. It can be beneficial to do all such processing at once, while the voxel data is already hot in the caches.

See this [full example of custom meshing](https://github.com/splashdust/bevy_voxel_world/blob/main/examples/custom_meshing.rs) for more details.

<img width="558" src="https://github.com/user-attachments/assets/13f46fd6-81a8-4c93-943c-9c5fc7b9b38e"/>

### Other shapes

<img width="800" height="445" alt="marching_cubes" src="https://github.com/user-attachments/assets/b83b9861-206c-4bf8-9cfb-582e0ccbb500" />

The delegate doesn't have to produce cubical voxels. The [marching cubes example](https://github.com/splashdust/bevy_voxel_world/blob/main/examples/marching_cubes.rs) turns each chunk's voxels into a density field and extracts a smooth surface with the [transvoxel](https://crates.io/crates/transvoxel) crate, rendered with a `StandardMaterial` and vertex colours instead of the default voxel material.

## Level of Detail

<img width="800" height="445" alt="noise_terrain_lod" src="https://github.com/user-attachments/assets/a93bd929-36b1-44d5-bcdb-41d608d33399" />

Every chunk has a LOD level. A level-0 chunk is 32 voxels on a side and samples every voxel. A chunk at level `L` is `32 << L` voxels on a side and, by default, samples every `1 << L` voxels, so it holds as many samples as a level-0 chunk. Far regions therefore cost about as much as near ones, which lets a world stream out to a distant horizon.

To use LOD levels, override `VoxelWorldConfig::chunk_lod` and return the level wanted at a chunk position. The default returns `0` everywhere. The level must not get finer with distance. Check out the `noise_terrain_lod` example to see this in action.

When the level at a position changes, chunks at the new level spawn first, and the old ones are despawned once the new ones cover them, so no hole opens.

A chunk is addressed by a `ChunkKey`, which holds its position and its level. Chunk messages carry one, and `VoxelWorld::chunk_at` returns the key of the finest chunk covering a voxel.

Other `VoxelWorldConfig` methods that matter with LOD levels:

- `chunk_data_shape` and `chunk_meshing_shape` can be used for sampling a level more or less densely than the default.
- `voxel_lookup_delegate` is called with world positions, so it works unchanged at every level.
- `chunk_meshing_delegate` builds its mesh as for a level-0 chunk, and the crate scales it to the chunk's size. A custom mesher should scale its own texture coordinates with `scale_texture_coordinates`.
- `chunk_fill_hint` is an opt-in optimization that lets a world hint at the contents of a chunk. By hinting that a chunk is either 100% air or 100% solid material, the chunk streaming policy can decide to skip over it or stop at it, thereby avoiding unnecessary work.

The consuming code is responsible for hiding seams between levels. `examples/noise_terrain_lod.rs` streams terrain 500 chunks out in six levels, uses fill hints, and hides cracks with skirts by returning `WorldVoxel::Unset` in the padding of its level-1 chunks.

## AI Disclosure

From version 0.19.0 of `bevy_voxel_world`, AI has been used to implement new features, optimize existing ones and write some documentation (such as release notes). On prior versions, AI was used only sparingly for some of the upgrade work to keep up with Bevy versions.

Increased use of AI makes it possible for me to keep working on this crate, despite having very limited time to do so.

This note is here so that those who wish can steer clear of AI. Use versions prior to the one mentioned above, and feel free to fork or copy an earlier version if you wish to have a version of this crate untouched by AI.

## Contribution policy

AI written contributions are accepted. Code quality, such as maintainability, clarity, performance, and so on, still matter as much as they always have. It remains important to maintain sane abstractions and to understand how the code works. I may reject PRs that are hard to understand, are messy, or lack clear goal or architecture.

Please do disclose in the PR description where and how AI was used. It's also fine to let agents co-author commits. Doing so makes it clear which commits were written by AI.

It may sometimes take up to a month or so before I have a chance to process new PRs, especially during summer.

---

### Bevy compatibility

| bevy | bevy_voxel_world |
| ---- | ---------------- |
| 0.20 | ^0.18.0          |
| 0.19 | 0.17.0           |
| 0.18 | 0.15.0           |
| 0.17 | 0.13.0           |
| 0.16 | 0.12.0           |
| 0.15 | 0.10.0           |
| 0.14 | 0.8.0            |
| 0.13 | 0.4.0            |
| 0.12 | 0.3.6            |
| 0.11 | 0.2.2            |
