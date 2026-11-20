//! Coloured light on sub-grids
//!
//! This module checks that the emissive voxels of a sub-grid light it in
//! their own light colour, and never light the world. It checks that
//! `grid(g).light_at` mixes the light of the world with the light of the
//! sub-grid. It also checks that a recolour relights only the sub-grid chunks
//! that hold the material.
//!
//! Sky light is off in every test.

use std::sync::Arc;

use bevy::ecs::system::RunSystemOnce;
use bevy::prelude::*;

use super::sub_grid_light::{grid_light, lone_lamp, spawn_holding};
use crate::prelude::*;
use crate::test::light::edits::{relights_started, settle};
use crate::test::light::mixing::{emissive, BLUE_LAMP, RED_LAMP, STEP};
use crate::test::light::voxel_light::{app, light_at, LampWorld, LAMP, STONE};

/// The position of the lamp of the sub-grid, in the middle of the sub-grid
/// chunk at the origin.
const OWN_LAMP: IVec3 = IVec3::splat(16);

/// Returns a world of air apart from `lamps`, whose emissive materials are
/// the white, red, green and blue lamps of the mixing tests.
fn world(lamps: Vec<(IVec3, u8)>) -> LampWorld {
    LampWorld {
        scene: Arc::new(move |p| {
            lamps
                .iter()
                .find(|(at, _)| *at == p)
                .map_or(WorldVoxel::Air, |(_, material)| {
                    WorldVoxel::Solid(*material)
                })
        }),
        emissive: emissive(),
        ..default()
    }
}

/// Fails unless each component of `color` lies within [`STEP`] of the same
/// component of `expected`.
fn assert_color(color: LinearRgba, expected: [f32; 3], at: IVec3) {
    let read = color.to_f32_array_no_alpha();
    for (read, expected) in read.iter().zip(expected) {
        assert!(
            (read - expected).abs() < STEP,
            "{read:?} at {at}: {expected:?}"
        );
    }
}

#[test]
fn a_sub_grids_red_lamp_lights_it_red_and_never_the_world() {
    let mut app = app(world(Vec::new()));
    settle(&mut app, "the world to settle");
    let entity = spawn_holding(
        &mut app,
        Transform::default(),
        &[(OWN_LAMP, WorldVoxel::Solid(RED_LAMP))],
    );
    settle(&mut app, "the sub-grid's light to settle");

    let reference = lone_lamp(OWN_LAMP);
    let positions: Vec<IVec3> = (1..14).map(|x| OWN_LAMP + IVec3::X * x).collect();
    let read = grid_light(&mut app, entity, positions.clone());
    for (p, levels) in positions.iter().zip(read) {
        let levels = levels.unwrap_or_else(|| panic!("no light at {p}"));
        assert_eq!(
            levels.voxel,
            reference.read(reference.voxel_at(*p)),
            "at {p}"
        );
        assert_color(levels.voxel_color, [1.0, 0.0, 0.0], *p);
        let world = light_at(&mut app, *p).expect("the world's light");
        assert_eq!(world.voxel, 0.0, "the world's light at {p}");
    }
    let lamp = grid_light(&mut app, entity, vec![OWN_LAMP])[0].unwrap();
    assert_eq!(lamp.voxel, 15.0);
    assert_color(lamp.voxel_color, [1.0, 0.0, 0.0], OWN_LAMP);
}

/// Returns the weight that light at `level` carries in a mix of light colours.
///
/// The weight comes from the fixed formula
/// `(0.7^(15 - level) - 0.7^14) / (1 - 0.7^14)`, not from the voxel falloff
/// of the look. A weight below 0 reads as 0.
fn weight(level: f32) -> f32 {
    let floor = 0.7f32.powi(14);
    ((0.7f32.powf(15.0 - level) - floor) / (1.0 - floor)).max(0.0)
}

#[test]
fn a_cabin_with_a_white_lantern_docked_in_blue_world_light_mixes_the_two() {
    let blue = IVec3::new(40, 16, 16);
    let mut app = app(world(vec![(blue, BLUE_LAMP)]));
    settle(&mut app, "the world to settle");
    let entity = spawn_holding(
        &mut app,
        Transform::default(),
        &[(OWN_LAMP, WorldVoxel::Solid(LAMP))],
    );
    settle(&mut app, "the sub-grid's light to settle");

    let own = lone_lamp(OWN_LAMP);
    // Read voxels along x, from a west end beyond the reach of the blue lamp,
    // past the lantern, to an east end beyond the reach of the lantern.
    let positions: Vec<IVec3> = (-12..40)
        .map(|x| IVec3::new(x, 16, 16))
        .filter(|p| *p != OWN_LAMP && *p != blue)
        .collect();
    let read = grid_light(&mut app, entity, positions.clone());
    let (mut mixed, mut white, mut blue_only) = (0, 0, 0);
    for (p, levels) in positions.iter().zip(read) {
        let levels = levels.unwrap_or_else(|| panic!("no light at {p}"));
        let world = light_at(&mut app, *p).expect("the world's light").voxel;
        let lantern = own.read(own.voxel_at(*p));
        assert_eq!(levels.voxel, world.max(lantern), "at {p}");
        // White light weighs into every component. Blue light weighs into
        // blue alone.
        let expected = match (world > 0.0, lantern > 0.0) {
            (false, false) | (false, true) => [1.0, 1.0, 1.0],
            (true, false) => [0.0, 0.0, 1.0],
            (true, true) => {
                let (w, b) = match (weight(lantern), weight(world)) {
                    (0.0, 0.0) => (1.0, 1.0),
                    weights => weights,
                };
                // Red and green take the weight of the lantern, and blue takes
                // the larger of the two weights. Scaling then makes blue 1.
                let top = w.max(b);
                [w / top, w / top, 1.0]
            }
        };
        match (world > 0.0, lantern > 0.0) {
            (true, true) => mixed += 1,
            (false, true) => white += 1,
            (true, false) => blue_only += 1,
            _ => {}
        }
        assert_color(levels.voxel_color, expected, *p);
    }
    assert!(
        mixed > 0 && white > 0 && blue_only > 0,
        "{mixed} {white} {blue_only}"
    );
}

/// Returns how many relights of sub-grid chunks have started.
fn sub_grid_relights(app: &App) -> u64 {
    app.world()
        .resource::<crate::light::LightSettings<LampWorld>>()
        .counters()
        .sub_grid_relights
        .load(std::sync::atomic::Ordering::Relaxed)
}

#[test]
fn recolouring_a_material_relights_only_the_sub_grid_chunks_holding_it() {
    let mut app = app(world(Vec::new()));
    settle(&mut app, "the world to settle");
    // The sub-grid holds the chunk of the lamp and a chunk of stone two chunks
    // east of it. The chunk of stone lies beyond the light box of the chunk of
    // the lamp, so the recolour must leave it alone.
    let entity = spawn_holding(
        &mut app,
        Transform::default(),
        &[
            (OWN_LAMP, WorldVoxel::Solid(RED_LAMP)),
            (OWN_LAMP + IVec3::X * 64, WorldVoxel::Solid(STONE)),
        ],
    );
    settle(&mut app, "the sub-grid's light to settle");
    let (relights, sub_grid) = (relights_started(&app), sub_grid_relights(&app));

    app.world_mut()
        .run_system_once(|mut world: ResMut<LampWorld>| {
            let red = world
                .emissive
                .iter_mut()
                .find(|(material, _)| *material == RED_LAMP)
                .unwrap();
            red.1 = Emission::level(15.0).with_color(LinearRgba::GREEN);
        })
        .unwrap();
    settle(&mut app, "the recolour's light to settle");

    assert_eq!(sub_grid_relights(&app) - sub_grid, 1);
    assert_eq!(relights_started(&app) - relights, 1);
    let near = OWN_LAMP + IVec3::X * 3;
    let read = grid_light(&mut app, entity, vec![near, OWN_LAMP]);
    for (p, levels) in [near, OWN_LAMP].into_iter().zip(read) {
        assert_color(levels.unwrap().voxel_color, [0.0, 1.0, 0.0], p);
    }
}
