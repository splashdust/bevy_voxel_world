//! Tests of coloured light on sub-grids: a sub-grid's own emissive voxels
//! light it in their colour, `grid(g).light_at` mixes the world's light with
//! the sub-grid's own, and a recolour relights the sub-grid chunks holding
//! the material. Sky light is off in all of them.

use std::sync::Arc;

use bevy::ecs::system::RunSystemOnce;
use bevy::prelude::*;

use super::sub_grid_light::{grid_light, lone_lamp, spawn_holding};
use crate::prelude::*;
use crate::test::light::edits::{relights_started, settle};
use crate::test::light::mixing::{emissive, BLUE_LAMP, RED_LAMP, STEP};
use crate::test::light::voxel_light::{app, light_at, LampWorld, LAMP, STONE};

/// A lamp of the sub-grid, in the middle of its chunk at the origin.
const OWN_LAMP: IVec3 = IVec3::splat(16);

/// A world of air, apart from `lamps`, with white, red, green and blue lamps.
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

/// Fails unless `color` is within a step of the packing of `expected`.
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

/// The fixed mixing curve's weight of `level`, from its formula.
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
    // From beyond the blue lamp's reach on the west, past the lantern, to
    // beyond the lantern's reach on the east.
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
        // White weighs into every component, blue into blue alone.
        let expected = match (world > 0.0, lantern > 0.0) {
            (false, false) | (false, true) => [1.0, 1.0, 1.0],
            (true, false) => [0.0, 0.0, 1.0],
            (true, true) => {
                let (w, b) = match (weight(lantern), weight(world)) {
                    (0.0, 0.0) => (1.0, 1.0),
                    weights => weights,
                };
                // Red and green are the lantern's weight, blue the larger
                // weight, scaled so blue is 1.
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

/// How many sub-grid chunk relights have started.
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
    // The lamp's chunk, and a chunk of stone two chunks east, beyond the
    // lamp chunk's light box.
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
