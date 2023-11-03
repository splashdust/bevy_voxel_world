use bevy::prelude::*;
use bevy_voxel_world::prelude::*;
use rand::Rng;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(VoxelWorldPlugin::default())
        .add_systems(Startup, setup)
        .add_systems(Update, (set_solid_voxel, move_camera))
        .run();
}

fn setup(mut commands: Commands) {
    // Camera
    commands.spawn((
        Camera3dBundle {
            transform: Transform::from_xyz(20.0, 20.0, 20.0).looking_at(Vec3::ZERO, Vec3::Y),
            ..default()
        },
        // This tells bevy_voxel_world tos use this cameras transform to calculate spawning area
        VoxelWorldCamera,
    ));

    // Ambient light
    commands.insert_resource(AmbientLight {
        color: Color::rgb(0.98, 0.95, 0.82),
        brightness: 1.0,
    });
}

fn set_solid_voxel(mut voxel_world: VoxelWorld) {
    // Generate some random values
    let size = 10;
    let mut rng = rand::thread_rng();
    let x = rng.gen_range(-size..size);
    let y = rng.gen_range(-size..size);
    let z = rng.gen_range(-size..size);
    let voxel_type = rng.gen_range(0..4);
    let pos = IVec3::new(x, y, z);

    // Set a voxel at the random position with the random type
    if pos.distance_squared(IVec3::ZERO) < i32::pow(size, 2) {
        voxel_world.set_voxel(pos, WorldVoxel::Solid(voxel_type));
    }
}

// Rotate the camera around the origin
fn move_camera(time: Res<Time>, mut query: Query<&mut Transform, With<VoxelWorldCamera>>) {
    let mut transform = query.single_mut();
    let time_seconds = time.elapsed_seconds();
    transform.translation.x = 25.0 * (time_seconds * 0.1).sin();
    transform.translation.z = 25.0 * (time_seconds * 0.1).cos();
    transform.look_at(Vec3::ZERO, Vec3::Y);
}
