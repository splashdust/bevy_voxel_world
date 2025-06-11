use bevy::{pbr::CascadeShadowConfigBuilder, platform::collections::HashMap, prelude::*};
use bevy_voxel_world::prelude::*;
use noise::{HybridMulti, NoiseFn, Perlin};
use std::{sync::Arc, time::Duration};
#[derive(Resource, Clone, Default)]
struct MainWorld;

impl VoxelWorldConfig for MainWorld {
    type MaterialIndex = u8;
    type ChunkUserBundle = ();

    fn spawning_distance(&self) -> u32 {
        15
    }

    fn voxel_lookup_delegate(&self) -> VoxelLookupDelegate<Self::MaterialIndex> {
        Box::new(move |_chunk_pos| get_voxel_fn())
    }

    fn texture_index_mapper(
        &self,
    ) -> Arc<dyn Fn(Self::MaterialIndex) -> [u32; 3] + Send + Sync> {
        Arc::new(|mat| match mat {
            0 => [0, 0, 0],
            1 => [1, 1, 1],
            2 => [2, 2, 2],
            3 => [3, 3, 3],
            _ => [0, 0, 0],
        })
    }
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(VoxelWorldPlugin::with_config(MainWorld))
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
    commands.spawn(ExplosionTimeout {
        timer: Timer::from_seconds(0.25, TimerMode::Repeating),
    });

    // --- Just scene setup below ---

    // camera
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(-120.0, 150.0, -120.0).looking_at(Vec3::ZERO, Vec3::Y),
        // This tells bevy_voxel_world to use this cameras transform to calculate spawning area
        VoxelWorldCamera::<MainWorld>::default(),
    ));

    // Sun
    let cascade_shadow_config = CascadeShadowConfigBuilder { ..default() }.build();
    commands.spawn((
        DirectionalLight {
            color: Color::srgb(0.98, 0.95, 0.82),
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(0.0, 0.0, 0.0)
            .looking_at(Vec3::new(-0.15, -0.1, 0.15), Vec3::Y),
        cascade_shadow_config,
    ));

    // Ambient light, same color as sun
    commands.insert_resource(AmbientLight {
        color: Color::srgb(0.98, 0.95, 0.82),
        brightness: 100.0,
        affects_lightmapped_meshes: true,
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

fn move_camera(
    time: Res<Time>,
    mut cam_transform: Query<&mut Transform, With<VoxelWorldCamera<MainWorld>>>,
) {
    let mut transform = cam_transform.get_single_mut().unwrap();
    transform.translation.x += time.delta_secs() * 7.0;
    transform.translation.z += time.delta_secs() * 12.0;
}

fn explosion(
    mut voxel_world: VoxelWorld<MainWorld>,
    camera: Query<&Transform, With<VoxelWorldCamera<MainWorld>>>,
    mut timeout: Query<&mut ExplosionTimeout>,
    time: Res<Time>,
) {
    let mut timeout = timeout.get_single_mut().unwrap();
    timeout
        .timer
        .tick(Duration::from_secs_f32(time.delta_secs()));

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
    let impact_point =
        camera_transform.translation + (direction * 300.0) - Vec3::Y * 10.0;

    if let Some((impact_point, _)) =
        voxel_world.get_random_surface_voxel(impact_point.as_ivec3(), 70)
    {
        let vox = voxel_world.get_voxel(impact_point - IVec3::Y);

        // Dig out a spherical volume centered around the impact point
        let radius = 10;
        for x in -radius..=radius {
            for y in -radius..=radius {
                for z in -radius..=radius {
                    let pos = IVec3::new(
                        x + impact_point.x,
                        y + impact_point.y,
                        z + impact_point.z,
                    );

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
                    if let Some(rand_vox) =
                        voxel_world.get_random_surface_voxel(impact_point, 25)
                    {
                        voxel_world
                            .set_voxel(rand_vox.0 + IVec3::Y, WorldVoxel::Solid(mat));
                    }
                }
            }
            _ => {}
        }
    }
}
