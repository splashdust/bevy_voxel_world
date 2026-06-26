use std::sync::Arc;

use bevy::{
    color::palettes::css::{AQUA, FUCHSIA, LIME, ORANGE, YELLOW},
    light::CascadeShadowConfigBuilder,
    platform::collections::HashMap,
    prelude::*,
    text::DEFAULT_FONT_DATA,
    ui::{PositionType, Val},
};
use bevy_voxel_world::{
    custom_meshing::{CHUNK_SIZE_F, CHUNK_SIZE_U},
    prelude::*,
};
use noise::{HybridMulti, NoiseFn, Perlin};

const PATROL_RANGE: f32 = 180.0;
const PATROL_HEIGHT: f32 = 28.0;
const TERRAIN_NOISE_SCALE: f64 = 4000.0;
const TERRAIN_HEIGHT_SCALE: f64 = 20.0;

#[derive(Resource, Clone, Default)]
struct MainWorld;

impl VoxelWorldConfig for MainWorld {
    type MaterialIndex = u8;
    type ChunkUserBundle = ();

    fn spawning_distance(&self) -> u32 {
        8
    }

    fn min_despawn_distance(&self) -> u32 {
        1
    }

    fn chunk_spawn_strategy(&self) -> ChunkSpawnStrategy {
        ChunkSpawnStrategy::CloseAndInView
    }

    fn chunk_despawn_strategy(&self) -> ChunkDespawnStrategy {
        ChunkDespawnStrategy::FarAwayOrOutOfView
    }

    fn spawning_rays(&self) -> usize {
        12
    }

    fn max_spawn_per_frame(&self) -> usize {
        500
    }

    fn voxel_lookup_delegate(&self) -> VoxelLookupDelegate<Self::MaterialIndex> {
        Box::new(move |_chunk_pos, _lod, _previous| get_voxel_fn())
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

    fn chunk_lod(
        &self,
        chunk_position: IVec3,
        _previous_lod: Option<LodLevel>,
        camera_position: Vec3,
    ) -> LodLevel {
        let camera_chunk = (camera_position / CHUNK_SIZE_F).floor();
        let distance = chunk_position.as_vec3().distance(camera_chunk);

        // directly set lod values to our stride lengths
        if distance > 4.0 {
            return 4;
        }

        1
    }
}

#[derive(Component)]
struct PatrolCamera {
    target: Vec3,
    speed: f32,
}

#[derive(Component)]
struct CameraCountText;

#[derive(Resource, Clone)]
struct PatrolMarkerAssets {
    mesh: Handle<Mesh>,
    materials: Vec<Handle<StandardMaterial>>,
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(VoxelWorldPlugin::with_config(MainWorld))
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
                patrol_cameras,
                update_patrol_camera_count,
                update_camera_count_text,
            ),
        )
        .run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut fonts: ResMut<Assets<Font>>,
) {
    commands.insert_resource(ClearColor(Color::srgb(0.04, 0.06, 0.08)));

    let marker_assets = PatrolMarkerAssets {
        mesh: meshes.add(Sphere { radius: 4.0 }),
        materials: vec![
            materials.add(Color::from(AQUA)),
            materials.add(Color::from(FUCHSIA)),
            materials.add(Color::from(LIME)),
            materials.add(Color::from(ORANGE)),
            materials.add(Color::from(YELLOW)),
        ],
    };

    spawn_patrol_camera(&mut commands, &marker_assets, 0);
    spawn_patrol_camera(&mut commands, &marker_assets, 1);
    commands.insert_resource(marker_assets);

    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 300.0, 260.0)
            .looking_at(Vec3::new(0.0, 0.0, 0.0), Vec3::Y),
    ));

    commands.spawn((
        Camera2d,
        Camera {
            order: 1,
            ..default()
        },
    ));

    let font = fonts.add(Font::try_from_bytes(DEFAULT_FONT_DATA.to_vec()).unwrap());
    commands.spawn((
        Text::new(camera_count_text(2)),
        TextFont {
            font,
            font_size: 18.0,
            ..default()
        },
        TextColor(Color::WHITE),
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(12.0),
            left: Val::Px(12.0),
            ..default()
        },
        CameraCountText,
    ));

    let cascade_shadow_config = CascadeShadowConfigBuilder {
        maximum_distance: 650.0,
        ..default()
    }
    .build();
    commands.spawn((
        DirectionalLight {
            color: Color::srgb(0.98, 0.95, 0.82),
            shadow_maps_enabled: true,
            ..default()
        },
        Transform::from_xyz(0.0, 0.0, 0.0)
            .looking_at(Vec3::new(-0.2, -0.35, 0.15), Vec3::Y),
        cascade_shadow_config,
    ));

    commands.insert_resource(GlobalAmbientLight {
        color: Color::srgb(0.98, 0.95, 0.82),
        brightness: 90.0,
        affects_lightmapped_meshes: true,
    });
}

fn spawn_patrol_camera(
    commands: &mut Commands,
    marker_assets: &PatrolMarkerAssets,
    index: usize,
) {
    let start = random_patrol_point();
    let target = random_patrol_point();
    let material = marker_assets.materials[index % marker_assets.materials.len()].clone();

    commands.spawn((
        Camera3d::default(),
        Camera {
            is_active: false,
            ..default()
        },
        Mesh3d(marker_assets.mesh.clone()),
        MeshMaterial3d(material),
        Transform::from_translation(start).looking_at(target, Vec3::Y),
        PatrolCamera {
            target,
            speed: rand::random_range(12.0..28.0),
        },
        VoxelWorldCamera::<MainWorld>::default(),
    ));
}

fn update_patrol_camera_count(
    keys: Res<ButtonInput<KeyCode>>,
    mut commands: Commands,
    marker_assets: Res<PatrolMarkerAssets>,
    patrols: Query<Entity, With<PatrolCamera>>,
) {
    if keys.just_pressed(KeyCode::Space)
        || keys.just_pressed(KeyCode::Equal)
        || keys.just_pressed(KeyCode::NumpadAdd)
    {
        spawn_patrol_camera(&mut commands, &marker_assets, patrols.iter().count());
    }

    let remove_requested = keys.just_pressed(KeyCode::Backspace)
        || keys.just_pressed(KeyCode::Minus)
        || keys.just_pressed(KeyCode::NumpadSubtract);

    if remove_requested && patrols.iter().count() > 1 {
        let entity = patrols.iter().last().expect("at least one patrol camera");
        commands.entity(entity).despawn();
    }
}

fn patrol_cameras(
    time: Res<Time>,
    mut patrols: Query<(&mut Transform, &mut PatrolCamera)>,
) {
    for (mut transform, mut patrol) in patrols.iter_mut() {
        let to_target = patrol.target - transform.translation;
        let distance = to_target.length();

        if distance < 4.0 {
            patrol.target = random_patrol_point();
            patrol.speed = rand::random_range(12.0..28.0);
            continue;
        }

        let step = (patrol.speed * time.delta_secs()).min(distance);
        transform.translation += to_target.normalize_or_zero() * step;
        transform.look_at(patrol.target, Vec3::Y);
    }
}

fn update_camera_count_text(
    patrols: Query<(), With<PatrolCamera>>,
    mut text_query: Query<&mut Text, With<CameraCountText>>,
) {
    let Some(mut text) = text_query.iter_mut().next() else {
        return;
    };

    text.0 = camera_count_text(patrols.iter().count());
}

fn camera_count_text(count: usize) -> String {
    format!(
        "VoxelWorldCameras: {count}\n\
        Space / +: Add patrol camera\n\
        Backspace / -: Remove patrol camera"
    )
}

fn random_patrol_point() -> Vec3 {
    let x = rand::random_range(-PATROL_RANGE..PATROL_RANGE);
    let z = rand::random_range(-PATROL_RANGE..PATROL_RANGE);
    Vec3::new(x, terrain_height(x, z) + PATROL_HEIGHT, z)
}

fn terrain_height(x: f32, z: f32) -> f32 {
    let noise = terrain_noise();
    (noise.get([
        x as f64 / TERRAIN_NOISE_SCALE,
        z as f64 / TERRAIN_NOISE_SCALE,
    ]) * TERRAIN_HEIGHT_SCALE)
        .max(0.0) as f32
}

fn terrain_noise() -> HybridMulti<Perlin> {
    let mut noise = HybridMulti::<Perlin>::new(1234);
    noise.octaves = 5;
    noise.frequency = 1.1;
    noise.lacunarity = 2.8;
    noise.persistence = 0.4;
    noise
}

fn get_voxel_fn() -> Box<dyn FnMut(IVec3, Option<WorldVoxel>) -> WorldVoxel + Send + Sync>
{
    let noise = terrain_noise();
    let mut cache = HashMap::<(i32, i32), f64>::new();

    Box::new(move |pos: IVec3, _previous| {
        if pos.y < 0 {
            return WorldVoxel::Solid(3);
        }

        let [x, y, z] = pos.as_dvec3().to_array();
        let is_ground = y < match cache.get(&(pos.x, pos.z)) {
            Some(sample) => *sample,
            None => {
                let sample = noise
                    .get([x / TERRAIN_NOISE_SCALE, z / TERRAIN_NOISE_SCALE])
                    * TERRAIN_HEIGHT_SCALE;
                cache.insert((pos.x, pos.z), sample);
                sample
            }
        };

        if is_ground {
            WorldVoxel::Solid(0)
        } else {
            WorldVoxel::Air
        }
    })
}
