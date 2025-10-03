use std::sync::Arc;

use bevy::{pbr::CascadeShadowConfigBuilder, platform::collections::HashMap, prelude::*};
use bevy_voxel_world::prelude::*;
use noise::{HybridMulti, NoiseFn, Perlin};

#[derive(Resource, Clone, Default)]
struct MainWorld;

// Using enum for material index allows for more than u8::MAX number of materials.
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq, Default)]
enum BlockTexture {
    #[default]
    Grass,
    Stone,
    Dirt,
    Snow,
}

impl VoxelWorldConfig for MainWorld {
    type MaterialIndex = BlockTexture;
    type ChunkUserBundle = ();

    fn spawning_distance(&self) -> u32 {
        25
    }

    fn min_despawn_distance(&self) -> u32 {
        1
    }

    fn voxel_lookup_delegate(&self) -> VoxelLookupDelegate<Self::MaterialIndex> {
        Box::new(move |_chunk_pos| get_voxel_fn())
    }

    fn texture_index_mapper(
        &self,
    ) -> Arc<dyn Fn(Self::MaterialIndex) -> [u32; 3] + Send + Sync> {
        Arc::new(|mat| match mat {
            BlockTexture::Grass => [0, 0, 0],
            BlockTexture::Stone => [1, 1, 1],
            BlockTexture::Dirt => [2, 2, 2],
            BlockTexture::Snow => [3, 3, 3],
            // _ => [0, 0, 0],
        })
    }
}

fn main() {
    assert_eq!(size_of::<WorldVoxel>(), 2);
    assert_eq!(size_of::<WorldVoxel<BlockTexture>>(), 1);
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(VoxelWorldPlugin::with_config(MainWorld))
        .add_systems(Startup, setup)
        .add_systems(Update, move_camera)
        .run();
}

fn setup(mut commands: Commands) {
    // camera
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(-200.0, 180.0, -200.0).looking_at(Vec3::ZERO, Vec3::Y),
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

fn get_voxel_fn() -> Box<dyn FnMut(IVec3) -> WorldVoxel<BlockTexture> + Send + Sync> {
    // Set up some noise to use as the terrain height map
    let mut noise = HybridMulti::<Perlin>::new(1234);
    noise.octaves = 5;
    noise.frequency = 1.1;
    noise.lacunarity = 2.8;
    noise.persistence = 0.4;

    // Set up a second noise for vegetation placement
    let mut veg_noise = HybridMulti::<Perlin>::new(2345);
    veg_noise.octaves = 3;
    veg_noise.frequency = 0.5;
    veg_noise.lacunarity = 2.0;
    veg_noise.persistence = 0.3;

    // We use this to cache the noise value for each y column so we only need
    // to calculate it once per x/z coordinate
    let mut cache = HashMap::<(i32, i32), (f64, f64)>::new();

    // Then we return this boxed closure that captures the noise and the cache
    // This will get sent off to a separate thread for meshing by bevy_voxel_world
    Box::new(move |pos: IVec3| {
        // Sea level
        if pos.y < 1 {
            return WorldVoxel::Solid(BlockTexture::Snow);
        }

        let [x, y, z] = pos.as_dvec3().to_array();

        // Get or calculate both terrain and vegetation noise values
        let (terrain_sample, veg_sample) = match cache.get(&(pos.x, pos.z)) {
            Some((terrain, veg)) => (*terrain, *veg),
            None => {
                let terrain = noise.get([x / 1000.0, z / 1000.0]) * 50.0;
                let veg = veg_noise.get([x / 50.0, z / 50.0]);
                cache.insert((pos.x, pos.z), (terrain, veg));
                (terrain, veg)
            }
        };

        // Check if this is a terrain block
        let is_ground = y < terrain_sample;
        // Check if this is a vegetation block (only on top of terrain)
        let is_vegetation = is_ground && veg_sample > 0.0; // Only place vegetation in some areas

        const SNOW_LEVEL: f64 = 50.0;
        if is_vegetation {
            if y > SNOW_LEVEL {
                WorldVoxel::Solid(BlockTexture::Snow)
            } else {
                WorldVoxel::Solid(BlockTexture::Stone)
            }
        } else if is_ground {
            if y > SNOW_LEVEL {
                WorldVoxel::Solid(BlockTexture::Snow)
            } else {
                WorldVoxel::Solid(BlockTexture::Grass)
            }
        } else {
            WorldVoxel::Air
        }
    })
}

fn move_camera(
    time: Res<Time>,
    mut cam_transform: Query<&mut Transform, With<VoxelWorldCamera<MainWorld>>>,
) {
    let Ok(mut transform) = cam_transform.get_single_mut() else {
        return;
    };
    transform.translation.x += time.delta_secs() * 30.0;
    transform.translation.z += time.delta_secs() * 60.0;
}
