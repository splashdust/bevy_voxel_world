use std::sync::Arc;

use bevy::{
    diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin},
    light::CascadeShadowConfigBuilder,
    pbr::{DistanceFog, FogFalloff},
    platform::collections::HashMap,
    prelude::*,
    ui::{PositionType, Val},
};
use bevy_voxel_world::custom_meshing::{CHUNK_SIZE_F, CHUNK_SIZE_I};
use bevy_voxel_world::prelude::*;
use noise::{HybridMulti, NoiseFn, Perlin};

#[derive(Component)]
struct FpsText;

/// The terrain lies between these level-0 chunk rows: below the sea level at
/// y = 0 everything is rock, and the highest peaks (about y = 170) end in row
/// 5, so every chunk above that row is air.
const LOWEST_TERRAIN_CHUNK: i32 = -1;
const HIGHEST_TERRAIN_CHUNK: i32 = 5;

/// What fills the chunk at `key` when no terrain crosses it: rock below the
/// terrain, air above it. Used as the chunk's fill hint, so that the streaming
/// policy spawns neither, and by the voxel lookup to skip the noise there.
fn terrain_fill(key: ChunkKey) -> FillHint {
    if key.position.y + key.extent() <= LOWEST_TERRAIN_CHUNK {
        FillHint::Uniform(WorldVoxel::Solid(3))
    } else if key.position.y > HIGHEST_TERRAIN_CHUNK {
        FillHint::Uniform(WorldVoxel::Air)
    } else {
        FillHint::Unknown
    }
}

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
        500
    }

    fn min_despawn_distance(&self) -> u32 {
        1
    }

    fn voxel_lookup_delegate(&self) -> VoxelLookupDelegate<Self::MaterialIndex> {
        let chunk_noise = Arc::clone(&self.noise);
        Box::new(move |chunk_pos, lod_level, _previous| {
            if let FillHint::Uniform(voxel) =
                terrain_fill(ChunkKey::new(chunk_pos, lod_level))
            {
                return Box::new(move |_, _| voxel);
            }

            // A chunk at level L spans CHUNK_SIZE << L voxels.
            let chunk_min = chunk_pos * CHUNK_SIZE_I;
            let chunk_max = chunk_min + IVec3::splat(CHUNK_SIZE_I << lod_level);
            let skirt_enabled = lod_level == 1;
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
                if pos.y < 0 {
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

    fn chunk_lod(
        &self,
        chunk_position: IVec3,
        _previous_lod: Option<LodLevel>,
        camera_position: Vec3,
    ) -> LodLevel {
        let camera_chunk = (camera_position / CHUNK_SIZE_F).floor();
        let distance = chunk_position.as_vec3().distance(camera_chunk);

        // The LOD level by distance in chunks. A chunk at level L spans
        // 32 << L voxels and samples the terrain every 1 << L voxels, so the
        // far bands hold few, large chunks.
        if distance < 30.0 {
            0
        } else if distance < 70.0 {
            1
        } else if distance < 100.0 {
            2
        } else if distance < 150.0 {
            3
        } else if distance < 250.0 {
            4
        } else {
            5
        }
    }

    // With these fill hints, spawning rays pass through the air above the
    // terrain and stop at the rock below it, without spawning chunks for
    // either. Only the chunks the terrain crosses are generated, and the rays
    // reach the terrain at the horizon right away.
    fn chunk_fill_hint(&self, key: ChunkKey) -> FillHint {
        terrain_fill(key)
    }

    fn attach_chunks_to_root(&self) -> bool {
        false
    }
}

fn main() {
    App::new()
        .add_plugins((DefaultPlugins, FrameTimeDiagnosticsPlugin::default()))
        .add_plugins(VoxelWorldPlugin::with_config(MainWorld::default()))
        .add_systems(Startup, setup)
        .add_systems(Update, (move_camera, update_fps_text, edit_terrain))
        .run();
}

fn setup(mut commands: Commands) {
    // camera
    commands.spawn((
        Camera3d::default(),
        Camera {
            order: 0,
            ..default()
        },
        Transform::from_xyz(-200.0, 260.0, -200.0)
            .looking_at(Vec3::new(0.0, 170.0, 0.0), Vec3::Y),
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
    let cascade_shadow_config = CascadeShadowConfigBuilder {
        maximum_distance: 3000.0,
        first_cascade_far_bound: 200.0,
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
            .looking_at(Vec3::new(-0.15, -0.1, 0.15), Vec3::Y),
        cascade_shadow_config,
    ));

    // Ambient light, same color as sun
    commands.insert_resource(GlobalAmbientLight {
        color: Color::srgb(0.98, 0.95, 0.82),
        brightness: 100.0,
        affects_lightmapped_meshes: true,
    });

    // UI overlay camera
    commands.spawn((
        Camera2d,
        Camera {
            order: 1,
            ..default()
        },
    ));

    commands.spawn((
        Text::new("FPS: -- (--)\nFrame: -- ms"),
        TextFont {
            font_size: FontSize::Px(18.0),
            ..default()
        },
        TextColor(Color::WHITE),
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(12.0),
            left: Val::Px(12.0),
            ..default()
        },
        FpsText,
    ));
}

/// Left click digs a hole and shift + left click adds a mound where the
/// cursor points. Edits go through the usual voxel API and the terrain keeps
/// updating like any other voxel change.
fn edit_terrain(
    buttons: Res<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
    window: Query<&Window>,
    camera: Query<(&Camera, &GlobalTransform), With<VoxelWorldCamera<MainWorld>>>,
    mut voxel_world: VoxelWorld<MainWorld>,
) {
    if !buttons.just_pressed(MouseButton::Left) {
        return;
    }

    let (Ok(window), Ok((camera, camera_transform))) = (window.single(), camera.single())
    else {
        return;
    };
    let Some(cursor) = window.cursor_position() else {
        return;
    };
    let Ok(ray) = camera.viewport_to_world(camera_transform, cursor) else {
        return;
    };
    let Some(hit) = voxel_world.raycast(ray, &|_| true) else {
        return;
    };

    let add = keys.any_pressed([KeyCode::ShiftLeft, KeyCode::ShiftRight]);
    let center = hit.voxel_pos();
    let radius = 32;
    for x in -radius..=radius {
        for y in -radius..=radius {
            for z in -radius..=radius {
                let offset = IVec3::new(x, y, z);
                if offset.length_squared() <= radius * radius {
                    let voxel = if add {
                        WorldVoxel::Solid(0)
                    } else {
                        WorldVoxel::Air
                    };
                    voxel_world.set_voxel(center + offset, voxel);
                }
            }
        }
    }
}

fn move_camera(
    time: Res<Time>,
    mut cam_transform: Query<&mut Transform, With<VoxelWorldCamera<MainWorld>>>,
) {
    if let Ok(mut transform) = cam_transform.single_mut() {
        transform.translation.x += time.delta_secs() * 12.0;
        transform.translation.z += time.delta_secs() * 24.0;
        transform.rotate_y(time.delta_secs() * 0.1);
    }
}

fn update_fps_text(
    diagnostics: Res<DiagnosticsStore>,
    mut text_query: Query<&mut Text, With<FpsText>>,
) {
    let Some(mut text) = text_query.iter_mut().next() else {
        return;
    };

    let fps_diag = diagnostics.get(&FrameTimeDiagnosticsPlugin::FPS);
    let fps_current = fps_diag.and_then(|d| d.value());
    let fps_avg = fps_diag.and_then(|d| d.average());
    let frame_ms = diagnostics
        .get(&FrameTimeDiagnosticsPlugin::FRAME_TIME)
        .and_then(|d| d.average())
        .map(|seconds| seconds * 1000.0);

    let line_one = match (fps_current, fps_avg) {
        (Some(current), Some(avg)) => {
            format!("FPS: {:>5.1} (avg {:>5.1})\n", current, avg)
        }
        (Some(current), None) => format!("FPS: {:>5.1} (--)\n", current),
        _ => "FPS: -- (--)\n".to_string(),
    };

    let line_two = match frame_ms {
        Some(ms) => format!("Frame: {:>5.2} ms", ms),
        None => "Frame: -- ms".to_string(),
    };

    text.0 = format!("{line_one}{line_two}");
}
