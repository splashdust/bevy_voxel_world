//! An airship of voxels flies through a lit world while you edit it and the
//! world around it.
//!
//! The airship is a sub-grid: voxels on a grid of its own, which moves
//! freely. It starts as world voxels standing on a dock, on a ledge out of a
//! cliff. Once light has settled it is lifted out of the world, and an
//! autopilot flies it on a loop over open hills, into a cave in the cliff,
//! through a hall inside lit only by lamp voxels, and back out. Faces of the
//! airship show the world's light where they are, so it darkens in the cave
//! and brightens by the hall's lamps, and the camera follows it. Its cabin
//! has a door and a lamp. Under `SubGridSky::Enclosed` the airship's own
//! voxels shade the cabin from the sky, so it is dark inside but for the
//! lamp and the doorway; under `SubGridSky::World` it is as bright as the
//! open sky.
//!
//! It shows lift and stamp in all four directions: the airship lifts out of
//! the world off its dock, and stamps back onto it; a piece lifts off the
//! airship into a sub-grid of its own, and stamps back onto it. Each ends in
//! a handover, so no frame shows the moved voxels twice or not at all.
//!
//! Controls:
//!   - Left click to dig, Shift + left click to place a voxel, in the world,
//!     the airship or a piece: whichever the cursor's ray hits first
//!   - 1, 2 and 3 to place planks, stone or a lamp
//!   - F to send the airship to hover over its dock, and again to fly on;
//!     when docked, to lift it off the dock
//!   - Enter, while it hovers, to stamp it onto the dock
//!   - X to lift the box under the cursor off the airship as a piece
//!   - V to stamp the piece under the cursor, or the newest, back on
//!   - E to switch `SubGridSky` between `Enclosed` and `World`
//!   - Space to pause the autopilot
//!   - Right drag to turn the camera about the airship, the wheel to zoom
//!   - H to hide the help
//!
//! While the airship hovers over its dock, the gizmos draw where its stamp
//! would put it: the outline of its snapped bounds, and a red box at each
//! world voxel a solid voxel of it would land on.
//!
//! Run with
//! `cargo run --release --example airship --features sub_grids,light_propagation`.

use std::sync::Arc;

use bevy::{
    ecs::system::SystemParam,
    input::mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll, MouseScrollUnit},
    light::CascadeShadowConfigBuilder,
    picking::hover::Hovered,
    platform::collections::HashMap,
    prelude::*,
};
use bevy_voxel_world::prelude::*;

mod flight;
mod panel;
mod pieces;
mod world;

use flight::Flight;
use world::{Generator, DARK_PLANKS, LANTERN, STONE};

/// The voxel world, with the sky setting the E key switches.
#[derive(Resource, Clone, Default)]
pub struct AirshipWorld {
    pub sub_grid_sky: SubGridSky,
}

impl VoxelWorldConfig for AirshipWorld {
    type MaterialIndex = u8;
    type ChunkUserBundle = ();

    fn spawning_distance(&self) -> u32 {
        12
    }

    fn chunk_despawn_strategy(&self) -> ChunkDespawnStrategy {
        ChunkDespawnStrategy::FarAway
    }

    fn voxel_lookup_delegate(&self) -> VoxelLookupDelegate<Self::MaterialIndex> {
        Box::new(|_, _, _| {
            let generator = Generator::default();
            // The ground's height depends only on x and z, so it is kept per
            // column.
            let mut heights = HashMap::<(i32, i32), f32>::new();
            Box::new(move |position, _| {
                let height = *heights
                    .entry((position.x, position.z))
                    .or_insert_with(|| generator.ground_height(position.x, position.z));
                match generator.voxel(position, height) {
                    Some(material) => WorldVoxel::Solid(material),
                    None => WorldVoxel::Air,
                }
            })
        })
    }

    // Everything lies between y = 0 and y = 128, with rock below and air
    // above, so neither needs chunks spawned.
    fn chunk_fill_hint(&self, key: ChunkKey) -> FillHint {
        if key.position.y + key.extent() <= 0 {
            FillHint::Uniform(WorldVoxel::Solid(STONE))
        } else if key.position.y >= 4 {
            FillHint::Uniform(WorldVoxel::Air)
        } else {
            FillHint::Unknown
        }
    }

    fn texture_index_mapper(&self) -> Arc<dyn Fn(u8) -> [u32; 3] + Send + Sync> {
        Arc::new(world::texture_indices)
    }

    fn voxel_texture(&self) -> Option<(String, u32)> {
        Some((world::TEXTURE.into(), world::TILE_COUNT))
    }

    fn light_propagation(&self) -> Option<LightPropagation> {
        Some(LightPropagation {
            // Nothing stands at y = 128 or above, so sky light enters every
            // column open up to there.
            sky: Sky::Ceiling { y: 127 },
            sub_grid_sky: self.sub_grid_sky,
            ..default()
        })
    }

    fn emissive_materials(&self) -> Vec<(Self::MaterialIndex, Emission)> {
        vec![(LANTERN, Emission::level(15.0))]
    }
}

const SKY: Color = Color::srgb(0.58, 0.74, 0.92);

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(AirshipPlugin)
        .run();
}

/// Everything the example adds to an app with Bevy's default plugins.
pub struct AirshipPlugin;

impl Plugin for AirshipPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(VoxelWorldPlugin::with_config(AirshipWorld::default()))
            .add_plugins(panel::PanelPlugin)
            .insert_resource(ClearColor(SKY))
            .init_resource::<Flight>()
            .init_resource::<Status>()
            .init_resource::<Brush>()
            .init_resource::<pieces::PieceCount>()
            .add_systems(Startup, setup)
            .add_systems(
                Update,
                (
                    flight::follow_lift_and_stamp,
                    pieces::follow_lifts,
                    flight::first_lift,
                    flight::keys,
                    pieces::lift_key,
                    pieces::send_back_key,
                    keys,
                    edit,
                    flight::fly,
                    pieces::slide,
                    follow_camera,
                    flight::draw_stamp_preview,
                )
                    .chain(),
            );
    }
}

/// The line the panel shows about the last thing that happened.
#[derive(Resource, Default)]
pub struct Status(pub String);

impl Status {
    /// Shows `line` in the panel, and logs it.
    pub fn set(&mut self, line: String) {
        info!("{line}");
        self.0 = line;
    }
}

/// The material a placed voxel gets.
#[derive(Resource)]
pub struct Brush(pub u8);

impl Default for Brush {
    fn default() -> Self {
        Self(DARK_PLANKS)
    }
}

/// The ray from the camera through the cursor, unless the cursor is over the
/// panel.
#[derive(SystemParam)]
pub struct Cursor<'w, 's> {
    window: Query<'w, 's, &'static Window>,
    camera: Query<'w, 's, (&'static Camera, &'static GlobalTransform), With<ChaseCamera>>,
    panel: Query<'w, 's, &'static Hovered, With<panel::Panel>>,
}

impl Cursor<'_, '_> {
    /// The nearest voxel the cursor points at, in the world, the airship or
    /// a piece.
    pub fn hit(&self, voxel_world: &VoxelWorld<AirshipWorld>) -> Option<GridRaycastHit> {
        if self.panel.iter().any(|hovered| hovered.get()) {
            return None;
        }
        let (window, (camera, camera_transform)) =
            (self.window.single().ok()?, self.camera.single().ok()?);
        let ray = camera
            .viewport_to_world(camera_transform, window.cursor_position()?)
            .ok()?;
        voxel_world.raycast_grids(ray, 600.0, &|_| true, &|_, _, _| true)
    }
}

/// E switches `SubGridSky`, and 1 to 3 choose what a placed voxel is.
fn keys(
    keys: Res<ButtonInput<KeyCode>>,
    mut config: ResMut<AirshipWorld>,
    mut brush: ResMut<Brush>,
    mut status: ResMut<Status>,
) {
    if keys.just_pressed(KeyCode::KeyE) {
        config.sub_grid_sky = match config.sub_grid_sky {
            SubGridSky::Enclosed => SubGridSky::World,
            SubGridSky::World => SubGridSky::Enclosed,
        };
        status.set(format!("SubGridSky::{:?}", config.sub_grid_sky));
    }
    for (key, material) in [
        (KeyCode::Digit1, DARK_PLANKS),
        (KeyCode::Digit2, STONE),
        (KeyCode::Digit3, LANTERN),
    ] {
        if keys.just_pressed(key) {
            brush.0 = material;
        }
    }
}

/// How far a dig reaches from the voxel under the cursor.
const DIG_RADIUS: i32 = 1;

/// A left click digs a small ball of voxels where the cursor points, and one
/// with Shift held places a voxel against the face there, in whichever grid
/// the cursor's ray hits.
fn edit(
    buttons: Res<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
    cursor: Cursor,
    brush: Res<Brush>,
    mut voxel_world: VoxelWorld<AirshipWorld>,
) {
    if !buttons.just_pressed(MouseButton::Left) {
        return;
    }
    let Some(hit) = cursor.hit(&voxel_world) else {
        return;
    };
    let place = keys.any_pressed([KeyCode::ShiftLeft, KeyCode::ShiftRight]);
    let Some(mut grid) = voxel_world.grid_mut(hit.grid) else {
        return;
    };
    if place {
        let normal = hit.normal.unwrap_or(IVec3::Y);
        grid.set_voxel(hit.position + normal, WorldVoxel::Solid(brush.0));
    } else {
        let reach = IVec3::splat(DIG_RADIUS);
        grid.set_voxels_in(hit.position - reach, hit.position + reach + 1, |position| {
            let off = position - hit.position;
            (off.length_squared() <= 2).then_some(WorldVoxel::Air)
        });
    }
}

/// The camera, which follows the airship from where its angles put it.
#[derive(Component)]
pub struct ChaseCamera {
    /// Its angle about the vertical, from straight behind the airship, and
    /// above the horizon, in radians.
    pub yaw: f32,
    pub pitch: f32,
    /// How far it wants to be from the airship's centre, and how far it is,
    /// which is closer where rock stands in the way.
    pub distance: f32,
    shown: f32,
}

impl Default for ChaseCamera {
    fn default() -> Self {
        Self {
            yaw: 0.7,
            pitch: 0.3,
            distance: 55.0,
            shown: 55.0,
        }
    }
}

/// Turns and zooms the camera from the mouse, and places it looking at the
/// airship. Where rock stands between the two, it moves in front of the
/// rock, so it stays in the cave with the airship.
fn follow_camera(
    time: Res<Time>,
    buttons: Res<ButtonInput<MouseButton>>,
    motion: Res<AccumulatedMouseMotion>,
    scroll: Res<AccumulatedMouseScroll>,
    flight: Res<Flight>,
    voxel_world: VoxelWorld<AirshipWorld>,
    mut cameras: Query<(&mut Transform, &mut ChaseCamera)>,
) {
    let Ok((mut transform, mut camera)) = cameras.single_mut() else {
        return;
    };
    if buttons.pressed(MouseButton::Right) {
        camera.yaw -= motion.delta.x * 0.005;
        camera.pitch = (camera.pitch + motion.delta.y * 0.005).clamp(-1.2, 1.4);
    }
    let lines = match scroll.unit {
        MouseScrollUnit::Line => scroll.delta.y,
        MouseScrollUnit::Pixel => scroll.delta.y / 40.0,
    };
    camera.distance = (camera.distance * (1.0 - 0.1 * lines)).clamp(12.0, 200.0);

    let focus = flight.pose.position;
    let forward = flight.pose.rotation * Vec3::Z;
    let yaw = forward.x.atan2(forward.z) + camera.yaw;
    let direction = Vec3::new(
        -yaw.sin() * camera.pitch.cos(),
        camera.pitch.sin(),
        -yaw.cos() * camera.pitch.cos(),
    );
    let in_the_way = voxel_world.raycast_grids(
        Ray3d::new(focus, Dir3::new(direction).unwrap_or(Dir3::Y)),
        camera.distance,
        &|grid| grid == Grid::World,
        &|_, _, _| true,
    );
    let wanted = in_the_way.map_or(camera.distance, |hit| (hit.distance - 2.0).max(4.0));
    // In at once in front of the rock, and out slowly once it's passed.
    camera.shown = if wanted < camera.shown {
        wanted
    } else {
        camera.shown + (wanted - camera.shown) * (1.0 - (-2.0 * time.delta_secs()).exp())
    };
    *transform = Transform::from_translation(focus + direction * camera.shown)
        .looking_at(focus, Vec3::Y);
}

fn setup(mut commands: Commands) {
    commands.spawn((
        Camera3d::default(),
        Transform::from_translation(
            flight::Pose::docked().position + Vec3::new(-40.0, 20.0, -30.0),
        ),
        ChaseCamera::default(),
        VoxelWorldCamera::<AirshipWorld>::default(),
    ));
    // With light propagation the sun shines only on faces with direct sky.
    commands.spawn((
        DirectionalLight {
            color: Color::srgb(0.98, 0.95, 0.82),
            shadow_maps_enabled: true,
            ..default()
        },
        Transform::default().looking_at(Vec3::new(-0.4, -1.0, 0.25), Vec3::Y),
        CascadeShadowConfigBuilder {
            maximum_distance: 400.0,
            ..default()
        }
        .build(),
    ));
    // Each face gets a share of the ambient light set by its sky light.
    commands.insert_resource(GlobalAmbientLight {
        color: Color::srgb(0.98, 0.95, 0.82),
        brightness: 600.0,
        affects_lightmapped_meshes: true,
    });
}
