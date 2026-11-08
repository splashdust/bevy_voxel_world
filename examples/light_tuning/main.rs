//! Every light setting in a panel: change one, see what it does to the world,
//! and print the config as Rust to paste into your own.
//!
//! The panel on the right groups the settings into tabs: how sky light gets
//! in and spreads, lamps, how light looks on faces, and what it costs. Hover
//! over a setting to see what it does, when a change to it shows and where it
//! shows best; a dot marks each setting changed from its default, and its
//! cross resets it. Under the panel, how long the last change took to settle.
//! "Print as Rust", or P, prints the config with only what differs from the
//! default, ready to paste into a world's `VoxelWorldConfig`.
//!
//! The bar at the bottom moves the camera to the places each setting shows
//! best, as do the keys 1 to 8: a cave whose wide hall sky fill reaches down
//! from its mouth, a step darker at each chunk, with lanterns along its walls,
//! and a deep chamber behind a narrow passage that only crystals light; the
//! ground under a floating island, an arch the sun stops
//! at, a house lit through its door and windows, and a well sky light falls
//! straight down. The bar lights the place the camera is at, and underlines
//! the one the hovered setting shows best from. The right mouse button orbits
//! the camera, the middle one pans it and the wheel zooms. H hides the panel.
//!
//! The ball walking in and out of the cave isn't a chunk, so propagated light
//! doesn't shade it. It dims itself from `VoxelWorld::light_at` at its
//! position, through the look's `LevelCurve::at`, so it goes dark in the cave
//! as the rock around it does instead of standing out fully lit.
//!
//! Run with `cargo run --release --example light_tuning --features light_propagation`.
//! On WASM the world starts with a sky reach of 16, as a WASM application
//! should.
//!
//! With the argument `measure`, it renders offscreen instead, changes a few
//! settings one step and back once the world has settled, and prints how many
//! frames and milliseconds each change took to settle, at 60 frames a second.
//! With `audit`, it renders offscreen, sets each setting that relights to its
//! extremes, and prints how much of the view from the place the panel names
//! for it changes, listing the settings that barely show there.

use std::sync::Arc;

use bevy::{light::CascadeShadowConfigBuilder, prelude::*};
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
use bevy_voxel_world::prelude::*;

/// Writes a line of output: to standard output on native, and to the
/// browser's console on WASM.
macro_rules! report {
    ($($line:tt)*) => {{
        #[cfg(not(target_arch = "wasm32"))]
        println!($($line)*);
        #[cfg(target_arch = "wasm32")]
        info!($($line)*);
    }};
}

#[cfg(not(target_arch = "wasm32"))]
mod audit;
#[cfg(not(target_arch = "wasm32"))]
mod measure;
mod panel;
mod rust;
mod scene;
mod settings;

use panel::PanelPlugin;
use scene::{texture_indices, Place, Scene, TEXTURE, TILE_COUNT};
use settings::Tuning;

/// The world, whose light settings are the values being tuned.
#[derive(Resource, Clone, Default)]
pub struct TuningWorld {
    pub tuning: Tuning,
}

impl VoxelWorldConfig for TuningWorld {
    type MaterialIndex = u8;
    type ChunkUserBundle = ();

    fn voxel_lookup_delegate(&self) -> VoxelLookupDelegate<Self::MaterialIndex> {
        Box::new(|_, _, _| {
            let mut scene = Scene::default();
            Box::new(move |position, _| scene.voxel(position))
        })
    }

    // Every chunk within the spawning distance stays spawned, in view or
    // not, so turning the camera spawns and retires nothing, and a setting's
    // change relights the same chunks wherever the camera looks.
    fn chunk_spawn_strategy(&self) -> ChunkSpawnStrategy {
        ChunkSpawnStrategy::Close
    }

    fn chunk_despawn_strategy(&self) -> ChunkDespawnStrategy {
        ChunkDespawnStrategy::FarAway
    }

    // Everything lies between y = 0 and y = 160, with rock below and air
    // above, so neither needs chunks spawned.
    fn chunk_fill_hint(&self, key: ChunkKey) -> FillHint {
        if key.position.y + key.extent() <= 0 {
            FillHint::Uniform(WorldVoxel::Solid(scene::STONE))
        } else if key.position.y >= 5 {
            FillHint::Uniform(WorldVoxel::Air)
        } else {
            FillHint::Unknown
        }
    }

    fn texture_index_mapper(
        &self,
    ) -> Arc<dyn Fn(Self::MaterialIndex) -> [u32; 3] + Send + Sync> {
        Arc::new(texture_indices)
    }

    fn voxel_texture(&self) -> Option<(String, u32)> {
        Some((TEXTURE.into(), TILE_COUNT))
    }

    fn light_propagation(&self) -> Option<LightPropagation> {
        Some(self.tuning.light())
    }

    fn emissive_materials(&self) -> Vec<(Self::MaterialIndex, Emission)> {
        self.tuning.emissive_materials()
    }
}

const SKY: Color = Color::srgb(0.58, 0.74, 0.92);

fn main() {
    #[cfg(not(target_arch = "wasm32"))]
    if std::env::args().nth(1).as_deref() == Some("measure") {
        return measure::measure();
    }
    #[cfg(not(target_arch = "wasm32"))]
    if std::env::args().nth(1).as_deref() == Some("audit") {
        return audit::audit();
    }
    #[cfg(target_arch = "wasm32")]
    let plugins = DefaultPlugins.set(WindowPlugin {
        primary_window: Some(Window {
            canvas: Some("#bevy".into()),
            ..default()
        }),
        ..default()
    });
    #[cfg(not(target_arch = "wasm32"))]
    let plugins = DefaultPlugins;
    App::new()
        .add_plugins((plugins, PanOrbitCameraPlugin))
        .add_plugins(VoxelWorldPlugin::with_config(TuningWorld::default()))
        .add_plugins(PanelPlugin)
        .insert_resource(ClearColor(SKY))
        .init_resource::<WalkerScene>()
        .add_systems(Startup, setup)
        .add_systems(Update, (walk, dim_walker, follow_ambient).chain())
        .run();
}

/// The sun, from high in the south-east so the cave mouth faces it, and
/// Bevy's ambient light.
fn add_lights(commands: &mut Commands) {
    commands.spawn((
        DirectionalLight {
            color: Color::srgb(0.98, 0.95, 0.86),
            shadow_maps_enabled: true,
            ..default()
        },
        Transform::default().looking_at(Vec3::new(-0.35, -0.7, -0.6), Vec3::Y),
        CascadeShadowConfigBuilder {
            maximum_distance: 260.0,
            ..default()
        }
        .build(),
    ));
    // A lit world wants about 600: propagated light scales it down in shade,
    // and caves stay black at any value.
    commands.insert_resource(GlobalAmbientLight {
        color: Color::srgb(0.9, 0.94, 1.0),
        brightness: Tuning::default().ambient,
        affects_lightmapped_meshes: true,
    });
}

fn camera_transform(place: Place) -> Transform {
    let (eye, target) = place.view();
    Transform::from_translation(eye).looking_at(target, Vec3::Y)
}

/// The focus, radius, yaw and pitch a `PanOrbitCamera` needs to look at
/// `place`.
fn orbit_of(place: Place) -> (Vec3, f32, f32, f32) {
    let (eye, target) = place.view();
    let offset = eye - target;
    let radius = offset.length();
    let yaw = offset.x.atan2(offset.z);
    let pitch = (offset.y / radius).asin();
    (target, radius, yaw, pitch)
}

/// Points `orbit` at `place`, both where it is and where it is heading, so
/// the camera cuts straight there instead of swinging around.
fn orbit_at(orbit: &mut PanOrbitCamera, place: Place) {
    let (target, radius, yaw, pitch) = orbit_of(place);
    orbit.focus = target;
    orbit.target_focus = target;
    orbit.radius = Some(radius);
    orbit.target_radius = radius;
    orbit.yaw = Some(yaw);
    orbit.target_yaw = yaw;
    orbit.pitch = Some(pitch);
    orbit.target_pitch = pitch;
    orbit.force_update = true;
}

/// The place `orbit` is at or heading to, or `None` once it has been
/// orbited, panned or zoomed away.
///
/// `PanOrbitCamera` changes its targets only on input, so they keep the
/// values [`orbit_at`] gave them until the camera is moved by hand.
pub fn place_in_view(orbit: &PanOrbitCamera) -> Option<Place> {
    Place::ALL.into_iter().find(|&place| {
        let (focus, radius, yaw, pitch) = orbit_of(place);
        orbit.target_focus == focus
            && orbit.target_radius == radius
            && orbit.target_yaw == yaw
            && orbit.target_pitch == pitch
    })
}

/// Moves the camera straight to `place`.
pub fn move_camera(
    place: Place,
    cameras: &mut Query<(&mut Transform, &mut PanOrbitCamera)>,
) {
    for (mut transform, mut orbit) in cameras {
        *transform = camera_transform(place);
        orbit_at(&mut orbit, place);
    }
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let mut orbit = PanOrbitCamera {
        button_orbit: MouseButton::Right,
        button_pan: MouseButton::Middle,
        ..default()
    };
    orbit_at(&mut orbit, Place::Meadow);
    commands.spawn((
        Camera3d::default(),
        camera_transform(Place::Meadow),
        orbit,
        VoxelWorldCamera::<TuningWorld>::default(),
    ));
    add_lights(&mut commands);
    commands.spawn((
        Mesh3d(meshes.add(Sphere::new(WALKER_RADIUS))),
        MeshMaterial3d(materials.add(StandardMaterial::from(Color::from(WALKER_COLOUR)))),
        Transform::from_xyz(0.0, scene::GROUND as f32 + WALKER_RADIUS, WALK_FROM),
        Walker,
    ));
}

fn follow_ambient(world: Res<TuningWorld>, mut ambient: ResMut<GlobalAmbientLight>) {
    if ambient.brightness != world.tuning.ambient {
        ambient.brightness = world.tuning.ambient;
    }
}

// --- The walker -----------------------------------------------------------------

/// The ball that walks in and out of the cave and dims itself from the light
/// around it.
#[derive(Component)]
struct Walker;

const WALKER_RADIUS: f32 = 0.8;
const WALKER_COLOUR: LinearRgba = LinearRgba::rgb(0.9, 0.35, 0.08);
/// The walk goes along x = 0 between these z, from the meadow to the back of
/// the cave's hall.
const WALK_FROM: f32 = 30.0;
const WALK_TO: f32 = -84.0;
/// Voxels per second.
const WALK_SPEED: f32 = 8.0;

/// The world's voxels, for the walker to find the floor under it.
#[derive(Resource, Default)]
struct WalkerScene(Scene);

/// Walks the ball from the meadow into the cave, past its lanterns, and back.
fn walk(
    time: Res<Time>,
    mut scene: ResMut<WalkerScene>,
    mut walker: Query<&mut Transform, With<Walker>>,
) {
    let length = WALK_FROM - WALK_TO;
    let travelled = (time.elapsed_secs() * WALK_SPEED) % (2.0 * length);
    let z = WALK_FROM
        - if travelled < length {
            travelled
        } else {
            2.0 * length - travelled
        };
    let floor = scene.0.floor(0, z.floor() as i32, 40) as f32;
    for mut transform in &mut walker {
        transform.translation.z = z;
        transform.translation.y = floor + WALKER_RADIUS;
    }
}

/// Dims the ball the way voxels are dimmed: Bevy's ambient light and sun by
/// the sky falloff at its sky light level, and voxel light added through its
/// emissive colour, from the voxel falloff at its voxel light level. With
/// these, the ball goes dark where the rock around it does.
fn dim_walker(
    voxel_world: VoxelWorld<TuningWorld>,
    world: Res<TuningWorld>,
    walker: Query<(&Transform, &MeshMaterial3d<StandardMaterial>), With<Walker>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let look = &world.tuning.light.look;
    for (transform, material) in &walker {
        let Some(levels) = voxel_world.light_at(transform.translation.floor().as_ivec3())
        else {
            continue;
        };
        let Some(mut material) = materials.get_mut(&material.0) else {
            continue;
        };
        let shade = look.sky_falloff.at(levels.sky).max(look.minimum_brightness);
        let glow = look.voxel_falloff.at(levels.voxel) * look.voxel_brightness;
        material.base_color = (WALKER_COLOUR * shade).into();
        material.emissive = WALKER_COLOUR * glow;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_camera_sent_to_a_place_is_there_until_moved_by_hand() {
        for place in Place::ALL {
            let mut orbit = PanOrbitCamera::default();
            orbit_at(&mut orbit, place);
            assert_eq!(place_in_view(&orbit), Some(place));
            orbit.target_yaw += 0.01;
            assert_eq!(place_in_view(&orbit), None);
        }
    }
}
