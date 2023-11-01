use bevy::{pbr::CascadeShadowConfigBuilder, prelude::*, utils::HashMap};
use bevy_voxel_world::prelude::*;
use noise::{HybridMulti, NoiseFn, Perlin};

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(VoxelWorldPlugin::default())
        .add_systems(Startup, setup)
        .add_systems(Update, move_camera)
        .run();
}

fn setup(mut commands: Commands) {
    commands.insert_resource(VoxelWorldConfiguration {
        // This is the spawn distance (in 32 meter chunks), centered around the camera.
        spawning_distance: 25,

        // Here we supply a closure that returns another closure that returns a voxel value for a given position.
        // This may seem a bit convoluted, but it allows us to capture data in a sendable closure to be sent off
        // to a differrent thread for the meshing process. A new closure is fetched for each chunk.
        voxel_lookup_delegate: Box::new(move |_chunk_pos| get_voxel_fn()), // `get_voxel_fn` is defined below
        ..Default::default()
    });

    // --- Just scene setup below ---

    // camera
    commands.spawn((
        Camera3dBundle {
            transform: Transform::from_xyz(-200.0, 180.0, -200.0).looking_at(Vec3::ZERO, Vec3::Y),
            ..default()
        },
        // This tells bevy_voxel_world tos use this cameras transform to calculate spawning area
        VoxelWorldCamera,
    ));

    // Sun
    let cascade_shadow_config = CascadeShadowConfigBuilder { ..default() }.build();
    commands.spawn(DirectionalLightBundle {
        directional_light: DirectionalLight {
            color: Color::rgb(0.98, 0.95, 0.82),
            shadows_enabled: true,
            ..default()
        },
        transform: Transform::from_xyz(0.0, 0.0, 0.0)
            .looking_at(Vec3::new(-0.15, -0.1, 0.15), Vec3::Y),
        cascade_shadow_config,
        ..default()
    });

    // Ambient light, same color as sun
    commands.insert_resource(AmbientLight {
        color: Color::rgb(0.98, 0.95, 0.82),
        brightness: 0.3,
    });
}

fn get_voxel_fn() -> Box<dyn FnMut(IVec3) -> WorldVoxel + Send + Sync> {
    // Set up some noise to use as the terrain height map
    let mut noise = HybridMulti::<Perlin>::new(1234);
    noise.octaves = 5;
    noise.frequency = 1.1;
    noise.lacunarity = 2.8;
    noise.persistence = 0.4;

    // We use this to cache the noise value for each y column so we only need
    // to calculate it once per x/z coordinate
    let mut cache = HashMap::<(i32, i32), f64>::new();

    // Then we return this boxed closure that captures the noise and the cache
    // This will get sent off to a separate thread for meshing by bevy_voxel_world
    Box::new(move |pos: IVec3| {
        // Sea level
        if pos.y < 1 {
            return WorldVoxel::Solid(3);
        }

        let [x, y, z] = pos.as_dvec3().to_array();

        // If y is less than the noise sample, we will set the voxel to solid
        let is_ground = y - 2.0
            < match cache.get(&(pos.x, pos.z)) {
                Some(sample) => *sample,
                None => {
                    let sample = noise.get([x / 1000.0, z / 1000.0]) * 50.0;
                    cache.insert((pos.x, pos.z), sample);
                    sample
                }
            };

        if is_ground {
            // Solid voxel of material type 0
            WorldVoxel::Solid(0)
        } else {
            WorldVoxel::Air
        }
    })
}

fn move_camera(time: Res<Time>, mut cam_transform: Query<&mut Transform, With<VoxelWorldCamera>>) {
    cam_transform.single_mut().translation.x += time.delta_seconds() * 30.0;
    cam_transform.single_mut().translation.z += time.delta_seconds() * 60.0;
}
