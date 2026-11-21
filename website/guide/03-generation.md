---
title: Generating voxels
api:
  - VoxelLookupDelegate
  - VoxelLookupFn
  - FillHint
---

## The generated layer

In the previous chapters, you learned how to set up a voxel world, and read and edit voxels within it. But a world isn't much without some terrain. Let's generate some!

As mentioned in the first chapter, there are two layers of voxel data: The generated layer and the edit layer. In this chapter we'll focus on the generated layer.

Generation esentially works by `bevy_voxel_world` asking your game: "Hey, what voxel should be at this location?", for every voxel in the world.

It does this by calling the `voxel_lookup_delegate`:

```rust
fn voxel_lookup_delegate(&self) -> VoxelLookupDelegate<Self::MaterialIndex> {
    Box::new(move |_chunk_pos, _lod, _previous| get_voxel_fn())
}
```

`voxel_lookup_delegate` is a function you declare in your VoxelWorldConfig. It returns a boxed closure, which in turn returns another boxed clousure. Wait, what!? You may be thinking now. That sounds really confusing!

## The nested closures

Let me explain:

The first closure, let us call it the *Chunk closure*, is called once per chunk. It let's you prepare data for that chunk. For example, terrain noise data can be prepared here.

The second closure we can call the *Voxel closure*. This one gets called once per voxel in the chunk, and it's the one that says what kind of voxel hould be generated at each location.

The reason for this seemingly convoluted setup is that these closures gets sent off to Bevys task pool to be executed off the main thread. This is how voxel chunks can stream in at high rates without causing the game to stutter. You put instructions in a box, the engine sends it off, and back comes a Bevy mesh in the shape of your voxels!

Here is an example `voxel_lookup_delegate`:

```rust
fn voxel_lookup_delegate(&self) -> VoxelLookupDelegate<Self::MaterialIndex> {
    // This is the Chunk closure, called once per chunk
    Box::new(move |_chunk_pos, _lod, _previous| {
        // And this is the Voxel closure, called for every voxel, 32^3, or 32768 times per chunk!
        Box::new(move |pos: IVec3, _previous| {
            let scale = 8.0;
            
            // Nice wavy sine hills!
            if (pos.y as f32)
                < (pos.x as f32 / scale).sin() * scale + (pos.z as f32 / scale).sin() * scale
            {
                // Every voxel below the sine threshold is solid
                return WorldVoxel::Solid(0);
            }

            // Everything else is left unset
            WorldVoxel::Unset
        })
    })
}
```

To try it, put it in your `MyWorld` struct from the previous chapters:

```rust
impl VoxelWorldConfig for MyWorld {
    type MaterialIndex = u8;
    type ChunkUserBundle = ();

    fn voxel_lookup_delegate(&self) -> VoxelLookupDelegate<Self::MaterialIndex> {
        //...
    }
}
```

You may also want to change the setup system to something like this, so you can actually see the world:

```rust
fn setup(mut commands: Commands) {
    // Camera
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(-50.0, 50.0, -50.0).looking_at(Vec3::ZERO, Vec3::Y), // A bit more zoomed out!
        VoxelWorldCamera::<MyWorld>::default(),
    ));

    // Let there be light
    commands.insert_resource(GlobalAmbientLight {
        brightness: 1000.0,
        ..default()
    });
}
```

Behold, sine hills forever!

<img alt="Sine hills" src="img/sine_hills.png" />

## Performance

As you can imagine, in an infinitely large world there are a lot of voxel locations to cover!

`bevy_voxel_world` handles this by only asking about the locations that are currently relevant. The streaming engine figures it out by continuosly casting detection rays out from the camera, discovering chunks the ray passes through. This ensures that only chunks that are in view are generated, and work is not wasted on things the player cannot see.

There are various ways you can configure your world for performance, such as `FillHint`s and `LodLevel`s. We'll go deeper into those topics in future chapters!
