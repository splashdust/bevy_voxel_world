//! Flies endlessly through hills, over open valleys and through tunnels.
//!
//! Shows light propagation while the world streams in around a moving camera:
//! sky light lights the open ground, and fades into a tunnel from its mouths
//! and from where its roof breaks open. Emissive voxels in the tunnel roof
//! light the rock around them. Frame times are logged to the console.
//!
//! Run with `cargo run -r --example light_flythrough --features light_propagation`.

use std::sync::Arc;

use bevy::{
    diagnostic::{FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin},
    light::CascadeShadowConfigBuilder,
    platform::collections::HashMap,
    prelude::*,
};
use bevy_voxel_world::prelude::*;
use noise::{Fbm, NoiseFn, Perlin};

const GRASS: u8 = 0;
const STONE: u8 = 1;
const LAMP: u8 = 2;

/// The height the camera flies at, along a path that winds sideways.
const FLIGHT_Y: f32 = 15.0;
/// The radius of the tunnel carved along the path, through any hill in its way.
const TUNNEL_RADIUS: f32 = 6.0;
/// Voxels per second.
const SPEED: f32 = 20.0;
/// The distance along x between lamps in the tunnel roof.
const LAMP_SPACING: i32 = 64;

fn flight_path(x: f32) -> Vec3 {
    Vec3::new(x, FLIGHT_Y, (x / 90.0).sin() * 40.0)
}

#[derive(Resource, Clone, Default)]
struct MainWorld;

impl VoxelWorldConfig for MainWorld {
    type MaterialIndex = u8;
    type ChunkUserBundle = ();

    fn spawning_distance(&self) -> u32 {
        16
    }

    fn chunk_despawn_strategy(&self) -> ChunkDespawnStrategy {
        ChunkDespawnStrategy::FarAway
    }

    fn voxel_lookup_delegate(&self) -> VoxelLookupDelegate<Self::MaterialIndex> {
        Box::new(|_chunk_pos, _lod, _previous| get_voxel_fn())
    }

    // The hills lie between y = -64 and y = 96, with rock below and air above.
    // Giving these as fill hints lets spawning rays pass through the air and
    // stop at the rock, without spawning chunks for either.
    fn chunk_fill_hint(&self, key: ChunkKey) -> FillHint {
        if key.position.y + key.extent() <= -2 {
            FillHint::Uniform(WorldVoxel::Solid(STONE))
        } else if key.position.y >= 3 {
            FillHint::Uniform(WorldVoxel::Air)
        } else {
            FillHint::Unknown
        }
    }

    fn texture_index_mapper(
        &self,
    ) -> Arc<dyn Fn(Self::MaterialIndex) -> [u32; 3] + Send + Sync> {
        Arc::new(|material| match material {
            GRASS => [0, 0, 1],
            STONE => [2, 2, 2],
            _ => [3, 3, 3],
        })
    }

    fn light_propagation(&self) -> Option<LightPropagation> {
        Some(LightPropagation::default())
    }

    fn emissive_materials(&self) -> Vec<(Self::MaterialIndex, Emission)> {
        vec![(LAMP, Emission::level(15.0))]
    }
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins((
            FrameTimeDiagnosticsPlugin::default(),
            LogDiagnosticsPlugin::default(),
        ))
        .add_plugins(VoxelWorldPlugin::with_config(MainWorld))
        .add_systems(Startup, setup)
        .add_systems(Update, fly)
        .run();
}

fn setup(mut commands: Commands) {
    commands.spawn((
        Camera3d::default(),
        Transform::from_translation(flight_path(0.0)),
        // This tells bevy_voxel_world to use this camera's transform to calculate spawning area
        VoxelWorldCamera::<MainWorld>::default(),
    ));

    // Sun. With light propagation it shines only where the sky is in direct view.
    commands.spawn((
        DirectionalLight {
            color: Color::srgb(0.98, 0.95, 0.82),
            shadow_maps_enabled: true,
            ..default()
        },
        Transform::from_xyz(0.0, 0.0, 0.0)
            .looking_at(Vec3::new(-0.15, -0.1, 0.15), Vec3::Y),
        CascadeShadowConfigBuilder::default().build(),
    ));

    // Ambient light. Each face gets a share of it set by its sky light level.
    commands.insert_resource(GlobalAmbientLight {
        color: Color::srgb(0.98, 0.95, 0.82),
        brightness: 600.0,
        affects_lightmapped_meshes: true,
    });
    commands.insert_resource(ClearColor(Color::srgb(0.6, 0.75, 0.95)));
}

fn get_voxel_fn() -> Box<dyn FnMut(IVec3, Option<WorldVoxel>) -> WorldVoxel + Send + Sync>
{
    let noise = Fbm::<Perlin>::new(1234);

    // We use this to cache the height of each column so we only need to
    // calculate it once per x/z coordinate
    let mut heights = HashMap::<(i32, i32), f32>::new();

    Box::new(move |pos: IVec3, _previous| {
        let p = pos.as_vec3();

        // The tunnel along the flight path
        let from_path = p.distance(flight_path(p.x));
        if from_path < TUNNEL_RADIUS {
            return WorldVoxel::Air;
        }

        let height = *heights.entry((pos.x, pos.z)).or_insert_with(|| {
            let sample = noise.get([p.x as f64 / 200.0, p.z as f64 / 200.0]);
            25.0 + 50.0 * sample as f32
        });

        if p.y >= height {
            WorldVoxel::Air
        } else if pos.x % LAMP_SPACING == 0
            && from_path < TUNNEL_RADIUS + 1.0
            && p.y > FLIGHT_Y + 4.0
        {
            // A lamp in the rock just above the tunnel
            WorldVoxel::Solid(LAMP)
        } else if p.y < height - 3.0 {
            WorldVoxel::Solid(STONE)
        } else {
            WorldVoxel::Solid(GRASS)
        }
    })
}

fn fly(
    time: Res<Time>,
    mut camera: Query<&mut Transform, With<VoxelWorldCamera<MainWorld>>>,
) {
    let x = time.elapsed_secs() * SPEED;
    if let Ok(mut transform) = camera.single_mut() {
        *transform = Transform::from_translation(flight_path(x))
            .looking_at(flight_path(x + 20.0), Vec3::Y);
    }
}
