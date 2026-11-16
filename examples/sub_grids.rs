//! Pillars and rods on a flat floor, which you can cut apart, build on and
//! stamp back into the world: a walkthrough of the `sub_grids` feature.
//!
//! A sub-grid is a set of voxels on a grid of its own. It has a `Transform`,
//! so it can move, turn and fall, while the world's voxels stay where they
//! are. This example moves voxels between the world and sub-grids in three
//! ways:
//!
//! - Cutting. The world asks the crate to lift any piece that an edit cuts
//!   loose into a new sub-grid. Cut through a pillar or a rod, and the part
//!   that lost its support falls.
//! - Spawning. R spawns a new, empty sub-grid above the cursor and fills it
//!   with a rod.
//! - Stamping. Enter writes the voxels of the sub-grid under the cursor back
//!   into the world.
//!
//! The crate has no physics of its own. Here avian3d moves the sub-grids,
//! with colliders built from their voxels.
//!
//! Two lamps stand on the floor. A lamp is an emissive voxel. A lamp in the
//! world lights the world and the sub-grids in it. A lamp in a sub-grid
//! lights only that sub-grid.
//!
//! Controls:
//!   - Left click removes the voxel under the cursor, in the world or in a
//!     sub-grid.
//!   - Shift and left click adds a stone voxel against the face under the
//!     cursor.
//!   - L adds a lamp against the face under the cursor.
//!   - R drops a new rod from above the cursor.
//!   - Enter stamps the sub-grid under the cursor into the world.
//!   - Drag with the right mouse button to orbit, with the middle button to
//!     pan, and scroll to zoom.
//!
//! Run with `cargo run --example sub_grids --features
//! sub_grids,dislodged_fragments,light_propagation`.

use std::sync::Arc;

use avian3d::prelude::*;
use bevy::prelude::*;
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
use bevy_voxel_world::{custom_meshing::CHUNK_SIZE_I, prelude::*};

// The materials. `texture_index_mapper` below picks a tile for each.
const FLOOR: u8 = 0;
const STONE: u8 = 1;
const WOOD: u8 = 2;
const LAMP: u8 = 3;

// ---------------------------------------------------------------------------
// Step 1: the world

/// The voxel at `position` when the world starts.
fn scene_voxel(position: IVec3) -> WorldVoxel {
    let IVec3 { x, y, z } = position;

    // A floor one voxel thick, at y = -1. A thin floor keeps its colliders
    // small.
    if y == -1 {
        return WorldVoxel::Solid(FLOOR);
    }

    // Two lamps standing on the floor.
    if position == IVec3::new(-4, 0, 4) || position == IVec3::new(3, 0, -4) {
        return WorldVoxel::Solid(LAMP);
    }

    // Three pillars, each 2 by 2 voxels and 10 tall, with their corner at
    // (`corner_x`, 0, `corner_z`).
    let pillar = |corner_x: i32, corner_z: i32| {
        (corner_x..corner_x + 2).contains(&x)
            && (corner_z..corner_z + 2).contains(&z)
            && (0..10).contains(&y)
    };
    if pillar(-8, 0) || pillar(6, 0) || pillar(-1, -8) {
        return WorldVoxel::Solid(STONE);
    }

    // A rod joining the tops of the first two pillars. Cut it once and both
    // halves still hang on a pillar. Cut it twice and the middle falls.
    if y == 9 && z == 0 && (-6..6).contains(&x) {
        return WorldVoxel::Solid(WOOD);
    }

    // A rod sticking out of the third pillar's side. One cut drops it.
    if y == 6 && z == -7 && (1..8).contains(&x) {
        return WorldVoxel::Solid(WOOD);
    }

    WorldVoxel::Air
}

#[derive(Resource, Clone, Default)]
struct PillarsWorld;

impl VoxelWorldConfig for PillarsWorld {
    type MaterialIndex = u8;
    type ChunkUserBundle = ();

    fn voxel_lookup_delegate(&self) -> VoxelLookupDelegate<u8> {
        Box::new(|_, _, _| Box::new(|position, _| scene_voxel(position)))
    }

    // The scene is small, so a few chunks around the camera hold all of it.
    fn spawning_distance(&self) -> u32 {
        3
    }

    // Keep every chunk near the camera, even those out of view, so a piece
    // that falls out of view still lands on the floor's colliders.
    fn chunk_spawn_strategy(&self) -> ChunkSpawnStrategy {
        ChunkSpawnStrategy::Close
    }

    fn chunk_despawn_strategy(&self) -> ChunkDespawnStrategy {
        ChunkDespawnStrategy::FarAway
    }

    // Tiles of `mountain_house_texture.png`, counted from the top. Each
    // material shows one tile on its top, sides and bottom.
    fn texture_index_mapper(&self) -> Arc<dyn Fn(u8) -> [u32; 3] + Send + Sync> {
        Arc::new(|material| match material {
            FLOOR => [16; 3],
            WOOD => [8; 3],
            LAMP => [13; 3],
            _ => [3; 3],
        })
    }

    fn voxel_texture(&self) -> Option<(String, u32)> {
        Some(("mountain_house_texture.png".into(), 18))
    }

    fn light_propagation(&self) -> Option<LightPropagation> {
        Some(LightPropagation {
            // Nothing in the scene is above y = 32, so sky light reaches every
            // column that is open up to there.
            sky: Sky::Ceiling { y: 32 },
            // Brighter lamps than the default, so they show in the evening.
            look: LightLook {
                voxel_brightness: 5000.0,
                ..default()
            },
            ..default()
        })
    }

    // Lamps give off light, at the brightest level.
    fn emissive_materials(&self) -> Vec<(u8, Emission)> {
        vec![(LAMP, Emission::level(15.0))]
    }

    // After each edit, the crate looks for voxels the edit cut off from the
    // rest of their grid. With `auto_lift`, it lifts each such piece into a
    // new sub-grid. That works in sub-grids too: cutting a falling rod in two
    // lifts the smaller half into a sub-grid of its own.
    //
    // Without `auto_lift`, a game gets a `FragmentDislodged` message for each
    // piece, and can lift it itself with `VoxelWorld::lift`.
    fn detect_dislodged_fragments(&self) -> Option<FragmentDetection<u8>> {
        Some(FragmentDetection {
            auto_lift: true,
            ..default()
        })
    }
}

// ---------------------------------------------------------------------------
// Step 2: the app

/// The sky after sunset.
const EVENING: Color = Color::srgb(0.13, 0.14, 0.27);

fn main() {
    App::new()
        .add_plugins((
            DefaultPlugins.set(ImagePlugin::default_nearest()),
            PanOrbitCameraPlugin,
            PhysicsPlugins::default(),
            VoxelWorldPlugin::with_config(PillarsWorld),
        ))
        .insert_resource(ClearColor(EVENING))
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
                remove_or_add_voxel,
                add_lamp,
                drop_rod,
                stamp_into_world,
                despawn_empty_sub_grids,
            ),
        )
        // The physics systems run right after the voxel world's own systems.
        // That is the frame new chunks first show, so a piece gets its
        // collider in the same frame its voxels appear.
        .add_systems(
            PreUpdate,
            (move_lifted_pieces_to_root, build_colliders)
                .chain()
                .after(VoxelWorldSet::<PillarsWorld>::default()),
        )
        .run();
}

fn setup(mut commands: Commands) {
    let focus = Vec3::new(0.0, 4.0, 0.0);
    commands.spawn((
        Camera3d::default(),
        PanOrbitCamera {
            focus,
            radius: Some(30.0),
            pitch: Some(0.4),
            yaw: Some(0.6),
            button_orbit: MouseButton::Right,
            button_pan: MouseButton::Middle,
            ..default()
        },
        // The world spawns chunks around this camera.
        VoxelWorldCamera::<PillarsWorld>::default(),
    ));

    // A dim sun low in the sky, and a little blue ambient light, so the
    // lamps stand out.
    commands.spawn((
        DirectionalLight {
            color: Color::srgb(1.0, 0.62, 0.42),
            illuminance: 900.0,
            shadow_maps_enabled: true,
            ..default()
        },
        Transform::default().looking_at(Vec3::new(1.0, -0.35, 0.3), Vec3::Y),
    ));
    commands.insert_resource(GlobalAmbientLight {
        color: Color::srgb(0.55, 0.6, 0.95),
        brightness: 380.0,
        affects_lightmapped_meshes: true,
    });
}

// ---------------------------------------------------------------------------
// Step 3: editing any grid

/// How far the cursor reaches, in voxels.
const REACH: f32 = 200.0;

/// The voxel under the cursor, in the world or in a sub-grid.
///
/// `raycast_grids` checks the world and every sub-grid, and returns the
/// nearest hit. The two closures can skip grids or voxels. Here they skip
/// nothing.
fn voxel_under_cursor(
    window: &Window,
    (camera, camera_transform): (&Camera, &GlobalTransform),
    voxel_world: &VoxelWorld<PillarsWorld>,
) -> Option<GridRaycastHit<u8>> {
    let cursor = window.cursor_position()?;
    let ray = camera.viewport_to_world(camera_transform, cursor).ok()?;
    voxel_world.raycast_grids(ray, REACH, &|_| true, &|_, _, _| true)
}

/// A left click removes the voxel under the cursor. With Shift held, it adds
/// a stone voxel against the face under the cursor instead.
fn remove_or_add_voxel(
    buttons: Res<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
    window: Single<&Window>,
    camera: Single<(&Camera, &GlobalTransform)>,
    mut voxel_world: VoxelWorld<PillarsWorld>,
) {
    if !buttons.just_pressed(MouseButton::Left) {
        return;
    }
    let Some(hit) = voxel_under_cursor(&window, *camera, &voxel_world) else {
        return;
    };

    // The hit names its grid: `Grid::World`, or `Grid::SubGrid` with the
    // sub-grid's entity. Its `position` is on that grid. `grid_mut` edits
    // any grid with the same calls, so this code doesn't care which grid it
    // is.
    let Some(mut grid) = voxel_world.grid_mut(hit.grid) else {
        return;
    };

    if !keys.any_pressed([KeyCode::ShiftLeft, KeyCode::ShiftRight]) {
        // Removing a voxel can cut a piece loose. The crate then lifts it
        // into a sub-grid a few frames later (see `detect_dislodged_fragments`
        // in step 1), and step 6 makes it fall.
        grid.set_voxel(hit.position, WorldVoxel::Air);
        return;
    }

    // `normal` points out of the face the ray hit, on the hit's grid. So
    // `position + normal` is the voxel in front of that face, even on a
    // sub-grid that is turned.
    if let Some(normal) = hit.normal {
        grid.set_voxel(hit.position + normal, WorldVoxel::Solid(STONE));
    }
}

/// L adds a lamp against the face under the cursor.
///
/// A lamp added to the world lights the world around it. A lamp added to a
/// sub-grid lights only that sub-grid, and moves with it.
fn add_lamp(
    keys: Res<ButtonInput<KeyCode>>,
    window: Single<&Window>,
    camera: Single<(&Camera, &GlobalTransform)>,
    mut voxel_world: VoxelWorld<PillarsWorld>,
) {
    if !keys.just_pressed(KeyCode::KeyL) {
        return;
    }
    let Some(hit) = voxel_under_cursor(&window, *camera, &voxel_world) else {
        return;
    };
    let (Some(normal), Some(mut grid)) = (hit.normal, voxel_world.grid_mut(hit.grid))
    else {
        return;
    };
    grid.set_voxel(hit.position + normal, WorldVoxel::Solid(LAMP));
}

// ---------------------------------------------------------------------------
// Step 4: spawning a sub-grid

/// R drops a new rod from above the point under the cursor.
fn drop_rod(
    keys: Res<ButtonInput<KeyCode>>,
    window: Single<&Window>,
    camera: Single<(&Camera, &GlobalTransform)>,
    mut voxel_world: VoxelWorld<PillarsWorld>,
) {
    if !keys.just_pressed(KeyCode::KeyR) {
        return;
    }
    let Some(hit) = voxel_under_cursor(&window, *camera, &voxel_world) else {
        return;
    };

    // `spawn_sub_grid` spawns an empty sub-grid as a child of the world's
    // root. The sub-grid's voxel (0, 0, 0) has its corner at the
    // `Transform`'s translation. This one starts 12 voxels up, and tilted,
    // since a sub-grid needn't line up with the world's grid.
    let transform = Transform::from_translation(hit.point + Vec3::Y * 12.0)
        .with_rotation(Quat::from_euler(EulerRot::YXZ, 0.6, 0.0, 0.3));
    let rod = voxel_world.spawn_sub_grid(transform);

    // The new sub-grid can be filled right away, in the same system.
    let Some(mut grid) = voxel_world.grid_mut(Grid::SubGrid(rod)) else {
        return;
    };
    for x in -4..4 {
        grid.set_voxel(IVec3::new(x, 0, 0), WorldVoxel::Solid(WOOD));
    }

    // The rod gets no physics body here. `build_colliders` gives it one
    // along with its first collider, once its voxels show.
}

// ---------------------------------------------------------------------------
// Step 5: stamping a sub-grid into the world

/// Enter stamps the sub-grid under the cursor into the world.
fn stamp_into_world(
    keys: Res<ButtonInput<KeyCode>>,
    window: Single<&Window>,
    camera: Single<(&Camera, &GlobalTransform)>,
    mut voxel_world: VoxelWorld<PillarsWorld>,
    mut commands: Commands,
) {
    if !keys.just_pressed(KeyCode::Enter) {
        return;
    }
    let Some(hit) = voxel_under_cursor(&window, *camera, &voxel_world) else {
        return;
    };
    let Grid::SubGrid(sub_grid) = hit.grid else {
        return;
    };

    // `stamp` writes the sub-grid's voxels into the world, and despawns the
    // sub-grid a few frames later, when the world's chunks show the voxels.
    //
    // The world's voxels can only sit on the world's grid, so the stamp
    // first snaps the sub-grid to the nearest whole voxel and quarter turn.
    // Here the snap shows as a small jump. A game can hide it by moving the
    // sub-grid to the pose `stamp_preview` returns before stamping.
    //
    // `incoming_replaces` is the rule for where both grids hold a voxel: the
    // sub-grid's voxel wins.
    if let Err(error) = voxel_world.stamp(sub_grid, Grid::World, incoming_replaces) {
        warn!("Couldn't stamp the sub-grid: {error}");
        return;
    }

    // The stamp uses the pose the sub-grid has now. Stop its body, so it
    // stays put until the stamp lands.
    commands.entity(sub_grid).insert(RigidBody::Static);
}

// ---------------------------------------------------------------------------
// Step 6: physics

/// Moves each piece cut from a sub-grid out from under that sub-grid.
///
/// A lift spawns the new sub-grid as a child of the grid it came from. For a
/// piece cut from the world, that is the world's root, which is where it
/// should be. A piece cut from a sub-grid would move along with that
/// sub-grid, so it moves to the world's root, keeping its place.
///
/// `Lifted` arrives in the frame the piece's voxels first show in the new
/// sub-grid.
fn move_lifted_pieces_to_root(
    mut lifted: MessageReader<Lifted<PillarsWorld>>,
    parents: Query<&ChildOf>,
    mut commands: Commands,
) {
    for message in lifted.read() {
        let Grid::SubGrid(source) = message.source else {
            continue;
        };
        // Every sub-grid in this example is a child of the world's root.
        if let Ok(&ChildOf(root)) = parents.get(source) {
            commands.entity(message.sub_grid).set_parent_in_place(root);
        }
    }
}

/// Gives each chunk that just showed new voxels a collider made of its solid
/// voxels, in the world and in sub-grids.
///
/// A world chunk becomes a static body. A sub-grid's chunks are children of
/// the sub-grid, so avian adds their colliders to the sub-grid's body. A
/// sub-grid gets a dynamic body along with its first collider: a body with
/// no collider has no mass.
fn build_colliders(
    mut published: MessageReader<ChunkPublished<PillarsWorld>>,
    voxel_world: VoxelWorld<PillarsWorld>,
    bodies: Query<(), With<RigidBody>>,
    mut commands: Commands,
) {
    for message in published.read() {
        let Some(chunk) = voxel_world
            .grid(message.grid)
            .and_then(|grid| grid.get_chunk_data(message.chunk_key))
        else {
            continue;
        };

        // An empty chunk has nothing to collide with.
        let voxels = if chunk.is_empty() {
            Vec::new()
        } else {
            solid_voxels(chunk.key().min_voxel(), |position| {
                chunk.get_voxel_at_world_position(position)
            })
        };
        if voxels.is_empty() {
            commands.entity(message.entity).try_remove::<Collider>();
            continue;
        }
        commands
            .entity(message.entity)
            .try_insert(Collider::voxels(Vec3::ONE, &voxels));

        match message.grid {
            Grid::World => {
                if !bodies.contains(message.entity) {
                    commands
                        .entity(message.entity)
                        .try_insert(RigidBody::Static);
                }
            }
            Grid::SubGrid(sub_grid) => {
                if !bodies.contains(sub_grid) {
                    commands.entity(sub_grid).try_insert(RigidBody::Dynamic);
                }
            }
        }
    }
}

/// The solid voxels of the chunk whose first voxel is `first`, as positions
/// from the chunk entity's origin. `voxel_at` reads the chunk's voxel at a
/// position on its grid.
///
/// A chunk's entity sits one voxel below its first voxel, so its voxels are
/// at 1 to `CHUNK_SIZE` on each axis.
fn solid_voxels(
    first: IVec3,
    voxel_at: impl Fn(IVec3) -> Option<WorldVoxel>,
) -> Vec<IVec3> {
    let mut voxels = Vec::new();
    for z in 0..CHUNK_SIZE_I {
        for y in 0..CHUNK_SIZE_I {
            for x in 0..CHUNK_SIZE_I {
                let offset = IVec3::new(x, y, z);
                if voxel_at(first + offset).is_some_and(|voxel| voxel.is_solid()) {
                    voxels.push(offset + IVec3::ONE);
                }
            }
        }
    }
    voxels
}

// ---------------------------------------------------------------------------
// Step 7: cleaning up

/// Despawns each sub-grid whose last voxel was removed.
///
/// bevy_voxel_world keeps an empty sub-grid until the game despawns it, since a
/// game may want to fill it again.
fn despawn_empty_sub_grids(
    mut edited: MessageReader<SubGridEditsCommitted<PillarsWorld>>,
    mut voxel_world: VoxelWorld<PillarsWorld>,
) {
    for message in edited.read() {
        let empty = voxel_world
            .grid(Grid::SubGrid(message.sub_grid))
            .is_some_and(|grid| grid.bounds().is_none());
        if empty {
            voxel_world.despawn_sub_grid(message.sub_grid);
        }
    }
}
