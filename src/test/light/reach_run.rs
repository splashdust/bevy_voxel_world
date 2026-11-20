//! Skipping light colours that can't reach a chunk
//!
//! This module tests `skip_unreached_colors`. With it on, a solve whose light
//! box holds four or more light colours first makes a reach run. The reach
//! run finds the light colours whose light can't reach the chunk, and the
//! solve skips their colour runs. The tests check that turning the setting on
//! or off changes no light anywhere and starts no relight. They also check
//! that the reach run doesn't run with three light colours or in a white
//! world.
//!
//! Sky light is off in every test.

use std::sync::atomic::Ordering;

use bevy::ecs::system::RunSystemOnce;
use bevy::prelude::*;

use super::super::update_until;
use super::mixing::{around_origin, three_primaries_meeting, Scene};
use super::sky_light::{chunk_light, settled, settled_but};
use super::voxel_light::{LampWorld, LAMP};
use crate::light::solver::LightCounters;
use crate::light::LightSettings;
use crate::light_bench::light_stats;
use crate::prelude::*;

/// The first of eight lamp materials, each with a light colour of its own.
const FIRST_LAMP: u8 = 10;

const COLORS: [LinearRgba; 8] = [
    LinearRgba::RED,
    LinearRgba::GREEN,
    LinearRgba::BLUE,
    LinearRgba::rgb(1.0, 1.0, 0.0),
    LinearRgba::rgb(0.0, 1.0, 1.0),
    LinearRgba::rgb(1.0, 0.0, 1.0),
    LinearRgba::rgb(1.0, 0.5, 0.0),
    LinearRgba::WHITE,
];

fn lamp(color: usize) -> u8 {
    FIRST_LAMP + color as u8
}

/// Returns one lamp of each of the first `colors` light colours, among walls
/// that fill the planes x = 0, x = 31, z = 0 and z = 31.
///
/// The walls lie along the faces of the chunk at the origin and close a room
/// inside it. Some lamps stand in the room. The rest stand outside it, within
/// the light box of the chunk, and the walls keep their light out of the
/// chunk.
fn walls_keep_colours_apart(colors: usize) -> Scene {
    let places = [
        IVec3::new(8, 16, 16),
        IVec3::new(24, 16, 16),
        IVec3::new(-4, 16, 16),
        IVec3::new(16, 16, 36),
        IVec3::new(16, 8, 8),
        IVec3::new(36, 16, 16),
        IVec3::new(16, 16, -5),
        IVec3::new(-4, 16, -4),
    ];
    Scene {
        lamps: (0..colors)
            .map(|color| (places[color], lamp(color)))
            .collect(),
        wall: |p| p.x == 0 || p.x == 31 || p.z == 0 || p.z == 31,
        edits: Vec::new(),
    }
}

/// Returns eight lamps of eight light colours in air, in the chunk at the
/// origin and just beside it, so the light of every colour reaches that chunk.
fn every_colour_reaches() -> Scene {
    let places = [
        IVec3::new(4, 4, 4),
        IVec3::new(28, 4, 4),
        IVec3::new(4, 28, 4),
        IVec3::new(4, 4, 28),
        IVec3::new(28, 28, 28),
        IVec3::new(-3, 16, 16),
        IVec3::new(16, 35, 16),
        IVec3::new(16, 16, -3),
    ];
    Scene::in_air(
        places
            .iter()
            .enumerate()
            .map(|(color, place)| (*place, lamp(color)))
            .collect(),
    )
}

fn world(scene: &Scene, shape: LightShape, skip: bool) -> LampWorld {
    let mut world = scene.world(14, shape, LightPrecision::Steps16);
    world.emissive = COLORS
        .iter()
        .enumerate()
        .map(|(color, rgba)| (lamp(color), Emission::level(15.0).with_color(*rgba)))
        .collect();
    world.light.as_mut().unwrap().skip_unreached_colors = skip;
    world
}

fn counter(app: &App, read: fn(&LightCounters) -> u64) -> u64 {
    read(
        app.world()
            .resource::<LightSettings<LampWorld>>()
            .counters(),
    )
}

fn reach_runs(app: &App) -> u64 {
    counter(app, |counters| counters.reach_runs.load(Ordering::Relaxed))
}

fn colors_skipped(app: &App) -> u64 {
    counter(app, |counters| {
        counters.colors_skipped.load(Ordering::Relaxed)
    })
}

/// Returns the light of every voxel of the chunk at the origin and the 26
/// chunks around it, with `None` for a chunk that has no light published.
fn light_around(app: &mut App) -> Vec<Option<Vec<(IVec3, LightLevels)>>> {
    around_origin()
        .into_iter()
        .map(|chunk| chunk_light(app, chunk))
        .collect()
}

/// Returns an app of `world` that has settled, relit through `reset_light`
/// and settled again, so every chunk has solved with its whole light box
/// loaded.
///
/// Without the relight, the walls can keep a chunk from ever solving again
/// once its neighbours arrive.
fn settled_and_relit(world: LampWorld) -> App {
    let mut app = settled(world);
    app.world_mut()
        .run_system_once(|mut voxel_world: VoxelWorld<LampWorld>| {
            voxel_world.reset_light();
        })
        .unwrap();
    app.update();
    update_until(&mut app, "the relight to settle", |app| {
        settled_but(app, None)
    });
    app
}

/// Settles `scene` under each shape, with `skip_unreached_colors` on and off,
/// and fails unless the light around the origin is the same both ways.
///
/// Returns how many light colours the reach runs skipped, summed over the
/// shapes.
fn same_light_either_way(scene: &Scene) -> u64 {
    let mut skipped = 0;
    for shape in [LightShape::Round, LightShape::Diamond] {
        let mut on = settled_and_relit(world(scene, shape, true));
        let mut off = settled_and_relit(world(scene, shape, false));
        assert!(reach_runs(&on) > 0, "{shape:?}");
        assert_eq!(reach_runs(&off), 0, "{shape:?}");
        let light = light_around(&mut on);
        assert!(light.iter().all(Option::is_some), "{shape:?}");
        assert!(light == light_around(&mut off), "{shape:?}");
        skipped += colors_skipped(&on);
    }
    skipped
}

#[test]
fn where_walls_keep_four_colours_apart_skipping_unreached_colours_changes_no_light() {
    assert!(same_light_either_way(&walls_keep_colours_apart(4)) > 0);
}

#[test]
fn where_walls_keep_eight_colours_apart_skipping_unreached_colours_changes_no_light() {
    assert!(same_light_either_way(&walls_keep_colours_apart(8)) > 0);
}

#[test]
fn where_every_colour_reaches_skipping_unreached_colours_changes_no_light() {
    same_light_either_way(&every_colour_reaches());
}

/// Returns the counts of relights and solves, which a change of
/// `skip_unreached_colors` must leave alone.
fn work_done(app: &App) -> [u64; 2] {
    let stats = light_stats::<LampWorld>(app.world()).unwrap();
    [stats.relights, stats.solves]
}

#[test]
fn toggling_skip_unreached_colors_starts_no_relight() {
    for shape in [LightShape::Round, LightShape::Diamond] {
        let mut app = settled(world(&walls_keep_colours_apart(8), shape, true));
        let light = light_around(&mut app);
        for skip in [false, true] {
            let work = work_done(&app);
            app.world_mut()
                .run_system_once(move |mut world: ResMut<LampWorld>| {
                    world.light.as_mut().unwrap().skip_unreached_colors = skip;
                })
                .unwrap();
            for _ in 0..5 {
                app.update();
            }
            assert_eq!(work_done(&app), work, "{shape:?}, skipping {skip}");
            assert!(
                light_around(&mut app) == light,
                "{shape:?}, skipping {skip}"
            );
        }
    }
}

#[test]
fn the_reach_run_doesnt_run_with_three_colours_or_in_a_white_world() {
    let scene = three_primaries_meeting(14);
    let three = settled(scene.world(14, LightShape::Round, LightPrecision::Steps16));
    assert!(three
        .world()
        .resource::<LightSettings<LampWorld>>()
        .task
        .emission
        .is_colored());
    assert_eq!(reach_runs(&three), 0);

    let places = [
        IVec3::new(4, 4, 4),
        IVec3::new(28, 4, 4),
        IVec3::new(4, 28, 4),
        IVec3::new(4, 4, 28),
        IVec3::new(28, 28, 28),
    ];
    let scene = Scene::in_air(places.into_iter().map(|place| (place, LAMP)).collect());
    let mut white = world(&scene, LightShape::Round, true);
    white.emissive = vec![(LAMP, Emission::level(15.0))];
    let white = settled(white);
    assert_eq!(reach_runs(&white), 0);
}
