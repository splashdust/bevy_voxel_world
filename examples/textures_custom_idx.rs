use bevy::prelude::*;
use bevy_voxel_world::prelude::*;
use std::sync::Arc;

// Using enum for material index allows for more than u8::MAX number of materials.
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq, Default)]
enum BlockTexture {
    #[default]
    SnowyBrick,
    FullBrick,
    Grass,
}

#[derive(Resource, Clone, Default)]
struct MyMainWorld;

impl VoxelWorldConfig for MyMainWorld {
    type MaterialIndex = BlockTexture;

    fn texture_index_mapper(&self) -> Arc<dyn Fn(Self::MaterialIndex) -> [u32; 3] + Send + Sync> {
        Arc::new(|vox_mat| match vox_mat {
            BlockTexture::SnowyBrick => [0, 1, 2],
            BlockTexture::FullBrick => [2, 2, 2],
            BlockTexture::Grass => [3, 3, 3],
        })
    }

    fn voxel_texture(&self) -> Option<(String, u32)> {
        Some(("example_voxel_texture.png".into(), 4))
    }
}

fn main() {
    assert_eq!(size_of::<WorldVoxel>(), 2);
    assert_eq!(size_of::<WorldVoxel<BlockTexture>>(), 1);

    App::new()
        .add_plugins(DefaultPlugins)
        // We can specify a custom texture when initializing the plugin.
        // This should just be a path to an image in your assets folder.
        .add_plugins(VoxelWorldPlugin::with_config(MyMainWorld))
        .add_systems(Startup, (setup, create_voxel_scene).chain())
        .run();
}

fn setup(mut commands: Commands) {
    // Camera
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(10.0, 10.0, 10.0).looking_at(Vec3::ZERO, Vec3::Y),
        // This tells bevy_voxel_world to use this cameras transform to calculate spawning area
        VoxelWorldCamera::<MyMainWorld>::default(),
    ));

    // light
    commands.spawn((
        PointLight {
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(4.0, 8.0, 4.0),
    ));
}

fn create_voxel_scene(mut voxel_world: VoxelWorld<MyMainWorld>) {
    // Then we can use the `u8` consts to specify the type of voxel

    // 20 by 20 floor
    for x in -10..10 {
        for z in -10..10 {
            voxel_world.set_voxel(IVec3::new(x, -1, z), WorldVoxel::Solid(BlockTexture::Grass));
            // Grassy floor
        }
    }

    // Some bricks
    voxel_world.set_voxel(
        IVec3::new(0, 0, 0),
        WorldVoxel::Solid(BlockTexture::SnowyBrick),
    );
    voxel_world.set_voxel(
        IVec3::new(1, 0, 0),
        WorldVoxel::Solid(BlockTexture::SnowyBrick),
    );
    voxel_world.set_voxel(
        IVec3::new(0, 0, 1),
        WorldVoxel::Solid(BlockTexture::SnowyBrick),
    );
    voxel_world.set_voxel(
        IVec3::new(0, 0, -1),
        WorldVoxel::Solid(BlockTexture::SnowyBrick),
    );
    voxel_world.set_voxel(
        IVec3::new(-1, 0, 0),
        WorldVoxel::Solid(BlockTexture::FullBrick),
    );
    voxel_world.set_voxel(
        IVec3::new(-2, 0, 0),
        WorldVoxel::Solid(BlockTexture::FullBrick),
    );
    voxel_world.set_voxel(
        IVec3::new(-1, 1, 0),
        WorldVoxel::Solid(BlockTexture::SnowyBrick),
    );
    voxel_world.set_voxel(
        IVec3::new(-2, 1, 0),
        WorldVoxel::Solid(BlockTexture::SnowyBrick),
    );
    voxel_world.set_voxel(
        IVec3::new(0, 1, 0),
        WorldVoxel::Solid(BlockTexture::SnowyBrick),
    );
}
