use bevy::prelude::*;
use bevy_voxel_world::prelude::*;
use std::sync::Arc;
use bevy_flycam::prelude::*;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        // We can specify a custom texture when initializing the plugin.
        // This should just be a path to an image in your assets folder.
        .add_plugins(VoxelWorldPlugin::default().with_voxel_texture(
            "example_voxel_texture.png",
            4, // number of indexes in the texture
        ))
        .add_plugins(NoCameraPlayerPlugin)
        .add_systems(Startup, setup)
        .add_systems(Update, create_voxel_scene)
        .run();
}

fn setup(mut commands: Commands) {
    commands.insert_resource(VoxelWorldConfiguration {
        // To specify how the texture map to different kind of voxels we add this mapping callback
        // For each material type, we specify the texture coordinates for the top, side and bottom faces.
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

    // Camera
    commands.spawn((
        Camera3dBundle {
            transform: Transform::from_xyz(5.0, 5.0, 5.0).looking_at(Vec3::ZERO, Vec3::Y),
            ..default()
        },
        // This tells bevy_voxel_world tos use this cameras transform to calculate spawning area
        VoxelWorldCamera,
        FlyCam
    ));

    // Ambient light
    commands.insert_resource(AmbientLight {
        color: Color::rgb(0.98, 0.95, 0.82),
        brightness: 1.0,
    });

    // Point light
    commands.spawn(PointLightBundle {
        transform: Transform::from_xyz(0.0, 5.0, 0.0),
        point_light: PointLight {
            color: Color::rgb(0.98, 0.95, 0.82),
            intensity: 100.0,
            range: 40.0,
            ..default()
        },
        ..default()
    });
}

fn create_voxel_scene(mut voxel_world: VoxelWorld, mut ran_once: Local<bool>) {
    if *ran_once {
        return;
    }
    *ran_once = true;

    // Then we just use the material type `u8` value to specify the type of voxel

    // 20 by 20 floor
    // for x in -10..10 {
    //     for z in -10..10 {
    //         voxel_world.set_voxel(IVec3::new(x, -1, z), WorldVoxel::Solid(2)); // Grassy floor
    //     }
    // }

    // Some bricks
    voxel_world.set_voxel(IVec3::new(0, 0, 0), WorldVoxel::Solid(0));
    voxel_world.set_voxel(IVec3::new(1, 0, 0), WorldVoxel::Solid(0));
    voxel_world.set_voxel(IVec3::new(0, 0, 1), WorldVoxel::Solid(0));
    voxel_world.set_voxel(IVec3::new(0, 0, -1), WorldVoxel::Solid(0));
    voxel_world.set_voxel(IVec3::new(-1, 0, 0), WorldVoxel::Solid(1));
    voxel_world.set_voxel(IVec3::new(-2, 0, 0), WorldVoxel::Solid(1));
    voxel_world.set_voxel(IVec3::new(-1, 1, 0), WorldVoxel::Solid(0));
    voxel_world.set_voxel(IVec3::new(-2, 1, 0), WorldVoxel::Solid(0));
    voxel_world.set_voxel(IVec3::new(0, 1, 0), WorldVoxel::Solid(0));
}

// Rotate the camera around the origin
fn move_camera(time: Res<Time>, mut query: Query<&mut Transform, With<VoxelWorldCamera>>) {
    let mut transform = query.single_mut();
    let time_seconds = time.elapsed_seconds();
    transform.translation.x = 10.0 * (time_seconds * 0.1).sin();
    transform.translation.z = 10.0 * (time_seconds * 0.1).cos();
    transform.look_at(Vec3::ZERO, Vec3::Y);
}
