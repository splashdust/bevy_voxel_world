//! The live demo on the website: a voxel world streaming around a flying camera,
//! with craters dug by falling bombs and by clicks on the canvas.

use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Duration,
};

use bevy::{
    core_pipeline::tonemapping::Tonemapping,
    light::CascadeShadowConfigBuilder,
    log::{Level, LogPlugin},
    pbr::{DistanceFog, FogFalloff},
    platform::collections::HashMap,
    prelude::*,
    winit::{UpdateMode, WinitSettings},
};
use bevy_voxel_world::custom_meshing::{CHUNK_SIZE_F, CHUNK_SIZE_U};
use bevy_voxel_world::prelude::*;
use noise::{HybridMulti, NoiseFn, Perlin};
use rand::RngExt;
use wasm_bindgen::prelude::*;

const SKY: Color = Color::srgb(0.66, 0.8, 0.92);
const SPAWNING_DISTANCE: u32 = 14;

const CAMERA_HEIGHT: f32 = 110.0;
const CAMERA_SPEED: f32 = 12.0;

const BOMB_INTERVAL_SECS: f32 = 4.0;
const CLICK_CRATER_RADIUS: i32 = 10;

const GRASS: u8 = 0;
const DIRT: u8 = 1;
const ROCK: u8 = 2;
const SNOW: u8 = 3;

const SNOW_LINE: f64 = 58.0;
/// Keeps the terrain below chunk row 3, and below the camera.
const MAX_TERRAIN_HEIGHT: f64 = 90.0;

#[derive(Resource, Clone, Default)]
struct DemoWorld;

impl VoxelWorldConfig for DemoWorld {
    type MaterialIndex = u8;
    type ChunkUserBundle = ();

    fn spawning_distance(&self) -> u32 {
        SPAWNING_DISTANCE
    }

    // A browser runs chunk tasks on the main thread, between frames. Running one
    // at a time keeps frames even while chunks stream in.
    fn max_active_chunk_threads(&self) -> usize {
        1
    }

    // The terrain's surface lies between chunk rows 0 and 2: everything below is
    // solid and everything above is air, so spawning rays need not spawn chunks
    // for either.
    fn chunk_fill_hint(&self, key: ChunkKey) -> FillHint {
        if key.position.y + key.extent() <= 0 {
            FillHint::Uniform(WorldVoxel::Solid(ROCK))
        } else if key.position.y > 2 {
            FillHint::Uniform(WorldVoxel::Air)
        } else {
            FillHint::Unknown
        }
    }

    fn voxel_lookup_delegate(&self) -> VoxelLookupDelegate<Self::MaterialIndex> {
        Box::new(|_chunk_pos, _lod_level, _previous| terrain_lookup())
    }

    fn texture_index_mapper(
        &self,
    ) -> Arc<dyn Fn(Self::MaterialIndex) -> [u32; 3] + Send + Sync> {
        Arc::new(|material| match material {
            GRASS => [GRASS as u32, DIRT as u32, DIRT as u32],
            other => [other as u32; 3],
        })
    }

    fn attach_chunks_to_root(&self) -> bool {
        false
    }
}

#[wasm_bindgen(inline_js = "export function announce_ready() {
    window.dispatchEvent(new Event('voxel-demo-ready'));
}")]
extern "C" {
    /// Tells the page that the demo has started drawing, so it can show the canvas.
    fn announce_ready();
}

static CANVAS_VISIBLE: AtomicBool = AtomicBool::new(true);

/// Tells the demo whether its canvas is on screen.
///
/// Off screen, the demo updates once a second, which leaves the main thread to
/// the rest of the page.
#[wasm_bindgen]
pub fn set_canvas_visible(visible: bool) {
    CANVAS_VISIBLE.store(visible, Ordering::Relaxed);
}

#[wasm_bindgen(start)]
fn start() {
    App::new()
        .insert_resource(ClearColor(SKY))
        .add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        canvas: Some("#voxel-demo".into()),
                        fit_canvas_to_parent: true,
                        // A focused canvas would scroll the page to itself on startup.
                        focused: false,
                        // Lets the page keep scrolling while the pointer is over the canvas.
                        prevent_default_event_handling: false,
                        ..default()
                    }),
                    ..default()
                })
                .set(LogPlugin {
                    level: Level::ERROR,
                    ..default()
                }),
        )
        .add_plugins(VoxelWorldPlugin::with_config(DemoWorld))
        .insert_resource(BombTimer(Timer::from_seconds(
            BOMB_INTERVAL_SECS,
            TimerMode::Repeating,
        )))
        .add_systems(Startup, setup)
        .add_systems(Update, (move_camera, drop_bombs, dig_where_clicked))
        .add_systems(Update, announce_ready.run_if(run_once))
        .add_systems(Update, follow_canvas_visibility)
        .run();
}

fn setup(mut commands: Commands) {
    let far = SPAWNING_DISTANCE as f32 * CHUNK_SIZE_F;

    commands.spawn((
        Camera3d::default(),
        // The default tonemapper needs lookup tables that would add to the download.
        Tonemapping::AcesFitted,
        Transform::from_xyz(0.0, CAMERA_HEIGHT, 0.0),
        VoxelWorldCamera::<DemoWorld>::default(),
        DistanceFog {
            color: SKY,
            falloff: FogFalloff::Linear {
                start: far * 0.5,
                end: far * 0.95,
            },
            ..default()
        },
    ));

    commands.spawn((
        DirectionalLight {
            color: Color::srgb(0.98, 0.95, 0.82),
            shadow_maps_enabled: true,
            ..default()
        },
        Transform::default().looking_at(Vec3::new(-0.5, -0.5, 0.4), Vec3::Y),
        // Bevy supports one shadow cascade per light on WebGL2.
        CascadeShadowConfigBuilder {
            num_cascades: 1,
            maximum_distance: far * 0.6,
            ..default()
        }
        .build(),
    ));

    commands.insert_resource(GlobalAmbientLight {
        color: Color::srgb(0.98, 0.95, 0.82),
        brightness: 420.0,
        affects_lightmapped_meshes: true,
    });
}

fn follow_canvas_visibility(mut settings: ResMut<WinitSettings>) {
    let mode = if CANVAS_VISIBLE.load(Ordering::Relaxed) {
        UpdateMode::Continuous
    } else {
        UpdateMode::reactive_low_power(Duration::from_secs(1))
    };
    settings.focused_mode = mode;
    settings.unfocused_mode = mode;
}

/// Returns a lookup for hills of grass on dirt on rock, with snow on the highest.
fn terrain_lookup() -> VoxelLookupFn {
    // A chunk's lookups include one voxel of padding on each side.
    const COLUMNS_PER_CHUNK: usize = (CHUNK_SIZE_U as usize + 2).pow(2);

    let mut noise = HybridMulti::<Perlin>::new(1234);
    noise.octaves = 4;
    noise.frequency = 1.0;
    noise.lacunarity = 2.2;
    noise.persistence = 0.5;

    // Every voxel in a column has the same height sample, so it is cached by column.
    let mut heights = HashMap::<(i32, i32), f64>::with_capacity(COLUMNS_PER_CHUNK);

    Box::new(move |pos: IVec3, _previous| {
        let height = *heights.entry((pos.x, pos.z)).or_insert_with(|| {
            let sample = noise.get([pos.x as f64 / 500.0, pos.z as f64 / 500.0]) * 30.0;
            sample.clamp(1.0, MAX_TERRAIN_HEIGHT)
        });

        let depth = height - pos.y as f64;
        if depth <= 0.0 {
            WorldVoxel::Air
        } else if depth <= 1.0 && height > SNOW_LINE {
            WorldVoxel::Solid(SNOW)
        } else if depth <= 1.0 {
            WorldVoxel::Solid(GRASS)
        } else if depth <= 6.0 {
            WorldVoxel::Solid(DIRT)
        } else {
            WorldVoxel::Solid(ROCK)
        }
    })
}

/// The direction the camera flies at `seconds`, level with the ground. It swings
/// slowly from side to side.
fn flight_heading(seconds: f32) -> Vec3 {
    let angle = 0.6 + (seconds * 0.04).sin() * 0.8;
    Vec3::new(angle.cos(), 0.0, angle.sin())
}

fn move_camera(
    time: Res<Time>,
    mut camera: Query<&mut Transform, With<VoxelWorldCamera<DemoWorld>>>,
) {
    let Ok(mut transform) = camera.single_mut() else {
        return;
    };
    let heading = flight_heading(time.elapsed_secs());
    transform.translation += heading * CAMERA_SPEED * time.delta_secs();
    transform.look_to(heading + Vec3::NEG_Y * 0.55, Vec3::Y);
}

#[derive(Resource)]
struct BombTimer(Timer);

/// Digs a crater at a random place ahead of the camera, once per `BombTimer` period.
fn drop_bombs(
    time: Res<Time>,
    mut timer: ResMut<BombTimer>,
    mut voxel_world: VoxelWorld<DemoWorld>,
    camera: Query<&Transform, With<VoxelWorldCamera<DemoWorld>>>,
) {
    if !timer.0.tick(time.delta()).just_finished() {
        return;
    }
    let Ok(camera) = camera.single() else {
        return;
    };

    let mut rng = rand::rng();
    let ahead = flight_heading(time.elapsed_secs());
    let beside = ahead.cross(Vec3::Y);
    let target = camera.translation
        + ahead * rng.random_range(100.0..300.0)
        + beside * rng.random_range(-120.0..120.0);

    let ray = Ray3d::new(target.with_y(CAMERA_HEIGHT), Dir3::NEG_Y);
    if let Some(hit) = voxel_world.raycast(ray, &is_solid) {
        dig_crater(&mut voxel_world, hit.voxel_pos(), rng.random_range(6..=10));
    }
}

/// Digs a crater at the first solid voxel under a click or a touch.
fn dig_where_clicked(
    mouse: Res<ButtonInput<MouseButton>>,
    touches: Res<Touches>,
    window: Query<&Window>,
    camera: Query<(&Camera, &GlobalTransform), With<VoxelWorldCamera<DemoWorld>>>,
    mut voxel_world: VoxelWorld<DemoWorld>,
) {
    let (Ok(window), Ok((camera, camera_transform))) = (window.single(), camera.single())
    else {
        return;
    };

    let clicked = mouse
        .just_pressed(MouseButton::Left)
        .then(|| window.cursor_position())
        .flatten();
    let touched = touches
        .iter_just_pressed()
        .next()
        .map(|touch| touch.position());

    let Some(position) = clicked.or(touched) else {
        return;
    };
    let Ok(ray) = camera.viewport_to_world(camera_transform, position) else {
        return;
    };
    if let Some(hit) = voxel_world.raycast(ray, &is_solid) {
        dig_crater(&mut voxel_world, hit.voxel_pos(), CLICK_CRATER_RADIUS);
    }
}

fn is_solid((_position, voxel): (Vec3, WorldVoxel)) -> bool {
    matches!(voxel, WorldVoxel::Solid(_))
}

fn dig_crater(voxel_world: &mut VoxelWorld<DemoWorld>, center: IVec3, radius: i32) {
    for x in -radius..=radius {
        for y in -radius..=radius {
            for z in -radius..=radius {
                let offset = IVec3::new(x, y, z);
                if offset.length_squared() <= radius * radius {
                    voxel_world.set_voxel(center + offset, WorldVoxel::Air);
                }
            }
        }
    }
}
