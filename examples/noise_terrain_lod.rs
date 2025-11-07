use std::sync::Arc;

use bevy::{
    light::CascadeShadowConfigBuilder,
    pbr::{DistanceFog, FogFalloff},
    platform::collections::HashMap,
    prelude::*,
};
use bevy_voxel_world::custom_meshing::{CHUNK_SIZE_F, CHUNK_SIZE_I, CHUNK_SIZE_U};
use bevy_voxel_world::prelude::*;
use noise::{HybridMulti, NoiseFn, Perlin};

#[derive(Resource, Clone)]
struct MainWorld {
    noise: Arc<HybridMulti<Perlin>>,
}

impl Default for MainWorld {
    fn default() -> Self {
        let mut noise = HybridMulti::<Perlin>::new(1234);
        noise.octaves = 5;
        noise.frequency = 1.1;
        noise.lacunarity = 2.8;
        noise.persistence = 0.4;

        Self {
            noise: Arc::new(noise),
        }
    }
}

impl VoxelWorldConfig for MainWorld {
    type MaterialIndex = u8;
    type ChunkUserBundle = ();

    fn spawning_distance(&self) -> u32 {
        200
    }

    fn min_despawn_distance(&self) -> u32 {
        1
    }

    fn voxel_lookup_delegate(&self) -> VoxelLookupDelegate<Self::MaterialIndex> {
        let chunk_noise = Arc::clone(&self.noise);
        Box::new(move |chunk_pos, lod_level, _previous| {
            if chunk_pos.y < 0 {
                return Box::new(|_, _| WorldVoxel::Solid(3));
            }
            if chunk_pos.y > 4 {
                return Box::new(|_, _| WorldVoxel::Air);
            }

            let chunk_min = chunk_pos * CHUNK_SIZE_I;
            let chunk_max = chunk_min + IVec3::splat(CHUNK_SIZE_I);
            let skirt_enabled = lod_level > 1;
            let noise = Arc::clone(&chunk_noise);

            // We use this to cache the noise value for each y column so we only need
            // to calculate it once per x/z coordinate
            let mut cache = HashMap::<(i32, i32), f64>::new();

            // Then we return this boxed closure that captures the noise and the cache
            // This will get sent off to a separate thread for meshing by bevy_voxel_world
            Box::new(move |pos: IVec3, _previous_voxel| {
                if skirt_enabled {
                    let outside = pos.x < chunk_min.x
                        || pos.x >= chunk_max.x
                        || pos.y < chunk_min.y
                        || pos.y >= chunk_max.y
                        || pos.z < chunk_min.z
                        || pos.z >= chunk_max.z;
                    if outside {
                        return WorldVoxel::Unset;
                    }
                }
                // Sea level
                if pos.y < 1 {
                    return WorldVoxel::Solid(3);
                }

                let [x, y, z] = pos.as_dvec3().to_array();

                // If y is less than the noise sample, we will set the voxel to solid
                let is_ground = y < match cache.get(&(pos.x, pos.z)) {
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
        })
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

    fn chunk_data_shape(&self, lod_level: LodLevel) -> UVec3 {
        padded_chunk_shape_uniform(CHUNK_SIZE_U / lod_level.max(1) as u32)
    }

    fn chunk_meshing_shape(&self, lod_level: LodLevel) -> UVec3 {
        padded_chunk_shape_uniform(CHUNK_SIZE_U / lod_level.max(1) as u32)
    }

    fn chunk_lod(&self, chunk_position: IVec3, camera_position: Vec3) -> LodLevel {
        let camera_chunk = (camera_position / CHUNK_SIZE_F).floor();
        let distance = chunk_position.as_vec3().distance(camera_chunk);

        // directly set lod values to our stride lengths
        if distance < 10.0 {
            1
        } else if distance < 25.0 {
            2
        } else if distance < 50.0 {
            4
        } else if distance < 85.0 {
            8
        } else if distance < 125.0 {
            16
        } else {
            32
        }
    }
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(VoxelWorldPlugin::with_config(MainWorld::default()))
        .add_systems(Startup, setup)
        .add_systems(Update, move_camera)
        .run();
}

fn setup(mut commands: Commands) {
    // camera
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(-200.0, 180.0, -200.0)
            .looking_at(Vec3::new(0.0, 60.0, 0.0), Vec3::Y),
        // This tells bevy_voxel_world to use this cameras transform to calculate spawning area
        VoxelWorldCamera::<MainWorld>::default(),
        DistanceFog {
            color: *ClearColor::default(),
            falloff: FogFalloff::Linear {
                start: 125.0 * CHUNK_SIZE_F,
                end: 200.0 * CHUNK_SIZE_F,
            },
            ..default()
        },
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

fn move_camera(
    time: Res<Time>,
    mut cam_transform: Query<&mut Transform, With<VoxelWorldCamera<MainWorld>>>,
) {
    if let Ok(mut transform) = cam_transform.single_mut() {
        transform.translation.x += time.delta_secs() * 12.0;
        transform.translation.z += time.delta_secs() * 24.0;
    }
}
