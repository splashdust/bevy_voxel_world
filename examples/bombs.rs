use bevy::{pbr::CascadeShadowConfigBuilder, prelude::*, utils::HashMap};
use bevy_voxel_world::prelude::*;
use noise::{HybridMulti, NoiseFn, Perlin};
use std::time::Duration;

#[derive(Clone, Default)]
struct MainWorld;

#[derive(Clone, Default)]
struct SecondWorld;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(VoxelWorldPlugin::<MainWorld>::default())
        .add_plugins(VoxelWorldPlugin::<SecondWorld>::default())
        .add_systems(Startup, setup)
        .add_systems(Update, move_camera)
        .add_systems(Update, explosion)
        .run();
}

#[derive(Component)]
struct ExplosionTimeout {
    timer: Timer,
}

fn setup(mut commands: Commands) {
    commands.insert_resource(VoxelWorldConfiguration::<MainWorld> {
        // This is the spawn distance (in 32 meter chunks), centered around the camera.
        spawning_distance: 15,

        // Here we supply a closure that returns another closure that returns a voxel value for a given position.
        // This may seem a bit convoluted, but it allows us to capture data in a sendable closure to be sent off
        // to a differrent thread for the meshing process. A new closure is fetched for each chunk.
        voxel_lookup_delegate: Box::new(move |_chunk_pos| get_voxel_fn()), // `get_voxel_fn` is defined below
        ..Default::default()
    });

    commands.insert_resource(VoxelWorldConfiguration::<SecondWorld> {
        // This is the spawn distance (in 32 meter chunks), centered around the camera.
        spawning_distance: 10,

        // Here we supply a closure that returns another closure that returns a voxel value for a given position.
        // This may seem a bit convoluted, but it allows us to capture data in a sendable closure to be sent off
        // to a differrent thread for the meshing process. A new closure is fetched for each chunk.
        voxel_lookup_delegate: Box::new(move |_chunk_pos| get_voxel_fn_2()), // `get_voxel_fn` is defined below
        ..Default::default()
    });

    commands.spawn(ExplosionTimeout {
        timer: Timer::from_seconds(0.25, TimerMode::Repeating),
    });

    // --- Just scene setup below ---

    // camera
    commands.spawn((
        Camera3dBundle {
            transform: Transform::from_xyz(-120.0, 150.0, -120.0).looking_at(Vec3::ZERO, Vec3::Y),
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
        brightness: 100.0,
    });
}

fn get_voxel_fn() -> Box<dyn FnMut(IVec3) -> WorldVoxel + Send + Sync> {
    // Set up some noise to use as the terrain height map
    let mut noise = HybridMulti::<Perlin>::new(1234);
    noise.octaves = 4;
    noise.frequency = 1.0;
    noise.lacunarity = 2.2;
    noise.persistence = 0.5;

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

        let sample = match cache.get(&(pos.x, pos.z)) {
            Some(sample) => *sample,
            None => {
                let sample = noise.get([x / 800.0, z / 800.0]) * 25.0;
                cache.insert((pos.x, pos.z), sample);
                sample
            }
        };

        // If y is less than the noise sample, we will set the voxel to solid
        let is_surface = y < sample;
        let is_sub_surface = y < sample - 1.0;

        if is_surface && !is_sub_surface {
            // Solid voxel of material type 0
            WorldVoxel::Solid(0)
        } else if is_sub_surface {
            // Solid voxel of material type 1
            WorldVoxel::Solid(1)
        } else {
            WorldVoxel::Air
        }
    })
}

fn get_voxel_fn_2() -> Box<dyn FnMut(IVec3) -> WorldVoxel + Send + Sync> {
    // Set up some noise to use as the terrain height map
    let mut noise = HybridMulti::<Perlin>::new(1234);
    noise.octaves = 4;
    noise.frequency = 1.0;
    noise.lacunarity = 2.2;
    noise.persistence = 0.5;

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

        let sample = match cache.get(&(pos.x, pos.z)) {
            Some(sample) => *sample,
            None => {
                let sample = noise.get([x / 400.0, z / 400.0]) * 50.0;
                cache.insert((pos.x, pos.z), sample);
                sample
            }
        };

        // If y is less than the noise sample, we will set the voxel to solid
        let is_surface = y < sample;
        let is_sub_surface = y < sample - 1.0;

        if is_surface && !is_sub_surface {
            // Solid voxel of material type 0
            WorldVoxel::Solid(1)
        } else if is_sub_surface {
            // Solid voxel of material type 1
            WorldVoxel::Solid(0)
        } else {
            WorldVoxel::Air
        }
    })
}

fn move_camera(time: Res<Time>, mut cam_transform: Query<&mut Transform, With<VoxelWorldCamera>>) {
    cam_transform.single_mut().translation.x += time.delta_seconds() * 7.0;
    cam_transform.single_mut().translation.z += time.delta_seconds() * 12.0;
}

fn explosion(
    mut voxel_world: VoxelWorld<MainWorld>,
    camera: Query<&Transform, With<VoxelWorldCamera>>,
    mut timeout: Query<&mut ExplosionTimeout>,
    time: Res<Time>,
) {
    let mut timeout = timeout.get_single_mut().unwrap();
    timeout
        .timer
        .tick(Duration::from_secs_f32(time.delta_seconds()));

    if !timeout.timer.finished() {
        return;
    }

    let camera_transform = camera.get_single().unwrap();
    let direction = Vec3::new(
        camera_transform.forward().x,
        1.0,
        camera_transform.forward().z,
    )
    .normalize();
    let impact_point = camera_transform.translation + (direction * 300.0) - Vec3::Y * 10.0;

    if let Some((impact_point, _)) =
        voxel_world.get_random_surface_voxel(impact_point.as_ivec3(), 70)
    {
        let vox = voxel_world.get_voxel(impact_point - IVec3::Y);

        // Dig out a spherical volume centered around the impact point
        let radius = 10;
        for x in -radius..=radius {
            for y in -radius..=radius {
                for z in -radius..=radius {
                    let pos =
                        IVec3::new(x + impact_point.x, y + impact_point.y, z + impact_point.z);

                    if pos.distance_squared(impact_point) <= radius.pow(2) {
                        voxel_world.set_voxel(pos, WorldVoxel::Air);
                    }
                }
            }
        }

        // Spread some voxels out around the impact zone
        let num_voxels = 50;
        match vox {
            WorldVoxel::Solid(mat) => {
                for _ in 0..num_voxels {
                    if let Some(rand_vox) = voxel_world.get_random_surface_voxel(impact_point, 25) {
                        voxel_world.set_voxel(rand_vox.0 + IVec3::Y, WorldVoxel::Solid(mat));
                    }
                }
            }
            _ => {}
        }
    }
}
