//! Animates ripples in a pool of water by setting its voxels every frame.
//!
//! Shows how to animate voxels with `set_voxels_in`, which sets a box of voxels in
//! one call. Every voxel in the pool is set each frame, across four chunks, and the
//! chunks where something changed remesh to show it.

use bevy::{light::CascadeShadowConfigBuilder, prelude::*};
use bevy_voxel_world::prelude::*;

// The pool spans the four chunks around the origin, and every voxel of water
// in it is set every frame.
const POOL_RADIUS: f32 = 48.0;
const RIM_WIDTH: f32 = 2.0;
const MAX_HEIGHT: i32 = 16;

// Layers of the default voxel texture
const STONE: u8 = 2;
const WATER: u8 = 3;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(VoxelWorldPlugin::default())
        .add_systems(Startup, (setup, build_basin))
        .add_systems(Update, (animate_water, move_camera))
        .run();
}

fn setup(mut commands: Commands) {
    // Camera
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(100.0, 70.0, 100.0).looking_at(Vec3::ZERO, Vec3::Y),
        // This tells bevy_voxel_world to use this cameras transform to calculate spawning area
        VoxelWorldCamera::<DefaultWorld>::default(),
    ));

    // Sun
    let cascade_shadow_config = CascadeShadowConfigBuilder {
        maximum_distance: 200.0,
        ..default()
    }
    .build();
    commands.spawn((
        DirectionalLight {
            illuminance: 4000.0,
            shadow_maps_enabled: true,
            ..default()
        },
        Transform::from_xyz(0.0, 0.0, 0.0)
            .looking_at(Vec3::new(-0.3, -0.2, 0.5), Vec3::Y),
        cascade_shadow_config,
    ));

    // Ambient light
    commands.insert_resource(GlobalAmbientLight {
        color: Color::WHITE,
        brightness: 300.0,
        affects_lightmapped_meshes: true,
    });
}

// Build the stone basin the water ripples in. Voxels set with `set_voxel` stay
// set, so this only needs to run once.
fn build_basin(mut voxel_world: VoxelWorld<DefaultWorld>) {
    for (x, z, distance) in columns_within(POOL_RADIUS + RIM_WIDTH) {
        voxel_world.set_voxel(IVec3::new(x, 0, z), WorldVoxel::Solid(STONE));
        if distance > POOL_RADIUS {
            for y in 1..=MAX_HEIGHT {
                voxel_world.set_voxel(IVec3::new(x, y, z), WorldVoxel::Solid(STONE));
            }
        }
    }
}

// Fill the pool with water up to the height of ripples spreading out from its
// centre. Each frame sets every voxel in the box around the pool, leaving the
// rim alone, and the chunks where something changed remesh to show the new
// frame.
fn animate_water(mut voxel_world: VoxelWorld<DefaultWorld>, time: Res<Time>) {
    let t = time.elapsed_secs();
    let extent = POOL_RADIUS.ceil() as i32;
    let width = 2 * extent;

    // The water's height in each column of the box, or `None` outside the pool
    let heights: Vec<Option<i32>> = (0..width * width)
        .map(|i| {
            let distance = column_distance(i % width - extent, i / width - extent);
            let wave = (distance * 0.2 - t * 2.0).sin();
            (distance <= POOL_RADIUS)
                .then(|| (MAX_HEIGHT as f32 * (0.5 + 0.4 * wave)).round() as i32)
        })
        .collect();

    let min = IVec3::new(-extent, 1, -extent);
    let max = IVec3::new(extent, MAX_HEIGHT + 1, extent);
    voxel_world.set_voxels_in(min, max, |position| {
        let column = (position.x + extent) + (position.z + extent) * width;
        let height = heights[column as usize]?;
        Some(if position.y <= height {
            WorldVoxel::Solid(WATER)
        } else {
            WorldVoxel::Air
        })
    });
}

// The distance from the pool's centre to the middle of the column at `x`, `z`
fn column_distance(x: i32, z: i32) -> f32 {
    Vec2::new(x as f32 + 0.5, z as f32 + 0.5).length()
}

// The columns within `radius` of the pool's centre, with their distance from it
fn columns_within(radius: f32) -> impl Iterator<Item = (i32, i32, f32)> {
    let extent = radius.ceil() as i32;
    (-extent..extent)
        .flat_map(move |x| (-extent..extent).map(move |z| (x, z)))
        .map(|(x, z)| (x, z, column_distance(x, z)))
        .filter(move |&(_, _, distance)| distance <= radius)
}

// Rotate the camera around the origin
fn move_camera(
    time: Res<Time>,
    mut query: Query<&mut Transform, With<VoxelWorldCamera<DefaultWorld>>>,
) {
    let mut transform = query.single_mut().unwrap();
    let time_seconds = time.elapsed_secs();
    transform.translation.x = 100.0 * (time_seconds * 0.1).sin();
    transform.translation.z = 100.0 * (time_seconds * 0.1).cos();
    transform.look_at(Vec3::ZERO, Vec3::Y);
}
