//! Tests of light as it runs on WASM, where relights run on the main thread
//! within the world's `wasm_relight_budget` each frame and at most one sky
//! probe starts a frame. They turn that mode on in a native app. Each check
//! of relights runs in a white world and in one lit by emissive voxels of
//! several light colours, whose chunks solve once per colour.

use std::sync::atomic::Ordering;
use std::time::Duration;

use bevy::prelude::*;

use super::edits::{
    assert_matches, dig, edited, framed_update, place, reference, set_voxels, settle,
    world,
};
use super::mixing::{assert_on_the_reference, Scene, BLUE_LAMP, GREEN_LAMP, RED_LAMP};
use super::sky_light::{settled_but, sky_light};
use super::voxel_light::{app, LampWorld, STONE};
use crate::light::LightSettings;
use crate::prelude::*;

/// An app of `world` whose light runs as on WASM.
fn main_thread_app(world: LampWorld) -> App {
    let mut app = app(world);
    app.world_mut()
        .resource_mut::<LightSettings<LampWorld>>()
        .main_thread = true;
    app
}

fn counter(app: &App, read: impl Fn(&crate::light::solver::LightCounters) -> u64) -> u64 {
    read(
        app.world()
            .resource::<LightSettings<LampWorld>>()
            .counters(),
    )
}

#[test]
fn a_single_voxel_edits_light_lands_within_2_frames() {
    let mut app = main_thread_app(world());
    settle(&mut app, "the world to stream in");
    for _ in 0..5 {
        framed_update(&mut app);
    }
    let edit = [place(IVec3::new(9, 4, 8))];
    set_voxels(&mut app, &edit);
    // Frame 0, which commits the edit.
    app.update();
    let mut frames = 0;
    while !settled_but(&mut app, None) {
        frames += 1;
        assert!(frames <= 2, "the edit's light hasn't landed by frame 2");
        framed_update(&mut app);
    }
    assert_matches(
        &mut app,
        &[IVec3::ZERO],
        &reference(&[IVec3::ZERO], edited(&edit)),
        "after the edit",
    );
}

#[test]
fn a_big_edit_relights_within_the_budget_each_frame_give_or_take_one_relight() {
    let budget = Duration::from_micros(500);
    let mut app = main_thread_app(LampWorld {
        light: Some(LightPropagation {
            wasm_relight_budget: budget,
            ..sky_light(Sky::default(), 30)
        }),
        ..world()
    });
    settle(&mut app, "the world to stream in");
    let relights = counter(&app, |c| c.relights.load(Ordering::Relaxed));

    // Through the cavern's floor where 8 chunks meet, so the edit relights
    // each of them.
    let centre = IVec3::new(32, 2, 32);
    let bomb: Vec<(IVec3, WorldVoxel)> = (-8..=8)
        .flat_map(|z| {
            (-8..=8).flat_map(move |y| (-8..=8).map(move |x| IVec3::new(x, y, z)))
        })
        .filter(|offset| offset.length_squared() <= 64)
        .map(|offset| dig(centre + offset))
        .collect();
    set_voxels(&mut app, &bomb);
    app.update();
    let mut frames = 1;
    while !settled_but(&mut app, None) {
        frames += 1;
        assert!(frames < 1000, "the bomb's light never settled");
        app.update();
    }

    let started = counter(&app, |c| c.relights.load(Ordering::Relaxed)) - relights;
    let max_frame = counter(&app, |c| c.max_relight_frame_nanos.load(Ordering::Relaxed));
    let longest = counter(&app, |c| c.max_relight_nanos.load(Ordering::Relaxed));
    assert!(started > 1, "the bomb started {started} relights");
    assert!(longest > 0, "no relight ran on the main thread");
    assert!(
        max_frame <= budget.as_nanos() as u64 + longest,
        "relights took {max_frame} ns in a frame, over the budget of {budget:?} \
         and one relight of at most {longest} ns"
    );
    assert_matches(
        &mut app,
        &[IVec3::ZERO],
        &reference(&[IVec3::ZERO], edited(&bomb)),
        "after the bomb",
    );
}

/// Stone below y = 0, with a red, a green and a blue lamp on it 10 voxels
/// from the origin, where 8 chunks meet.
fn three_colours_on_stone() -> Scene {
    Scene {
        lamps: vec![
            (IVec3::new(-10, 0, 0), RED_LAMP),
            (IVec3::new(10, 0, 0), GREEN_LAMP),
            (IVec3::new(0, 0, -10), BLUE_LAMP),
        ],
        wall: |position| position.y < 0,
        edits: Vec::new(),
    }
}

const COLOURED_REACH: (u8, LightShape, LightPrecision) =
    (14, LightShape::Round, LightPrecision::Steps16);

/// An app of `scene` whose light runs as on WASM, with `budget` for relights.
fn coloured_main_thread_app(scene: &Scene, budget: Duration) -> App {
    let (reach, shape, precision) = COLOURED_REACH;
    let mut world = LampWorld {
        spawning_distance: 3,
        ..scene.world(reach, shape, precision)
    };
    world.light = world.light.map(|light| LightPropagation {
        wasm_relight_budget: budget,
        ..light
    });
    main_thread_app(world)
}

#[test]
fn a_single_voxel_edits_coloured_light_lands_within_2_frames() {
    let scene = three_colours_on_stone();
    let mut app = coloured_main_thread_app(&scene, Duration::from_millis(4));
    settle(&mut app, "the world to stream in");
    for _ in 0..5 {
        framed_update(&mut app);
    }
    // A block on the stone between the green and the blue lamp, where their
    // light meets.
    let edit = [(IVec3::new(5, 0, 0), WorldVoxel::Solid(STONE))];
    set_voxels(&mut app, &edit);
    app.update();
    let mut frames = 0;
    while !settled_but(&mut app, None) {
        frames += 1;
        assert!(frames <= 2, "the edit's light hasn't landed by frame 2");
        framed_update(&mut app);
    }
    let (reach, shape, precision) = COLOURED_REACH;
    assert_on_the_reference(&mut app, &scene.edited(&edit), reach, shape, precision);
}

#[test]
fn a_big_edit_in_coloured_light_relights_within_the_budget_each_frame_give_or_take_one_relight(
) {
    let budget = Duration::from_micros(500);
    let scene = three_colours_on_stone();
    let mut app = coloured_main_thread_app(&scene, budget);
    settle(&mut app, "the world to stream in");
    let relights = counter(&app, |c| c.relights.load(Ordering::Relaxed));

    // Into the stone where 8 chunks meet, between the lamps, so the edit
    // relights each of those chunks in each colour.
    let centre = IVec3::ZERO;
    let bomb: Vec<(IVec3, WorldVoxel)> = (-8..=8)
        .flat_map(|z| {
            (-8..=8).flat_map(move |y| (-8..=8).map(move |x| IVec3::new(x, y, z)))
        })
        .filter(|offset| offset.length_squared() <= 64)
        .map(|offset| dig(centre + offset))
        .collect();
    set_voxels(&mut app, &bomb);
    app.update();
    let mut frames = 1;
    while !settled_but(&mut app, None) {
        frames += 1;
        assert!(frames < 1000, "the bomb's light never settled");
        app.update();
    }

    let started = counter(&app, |c| c.relights.load(Ordering::Relaxed)) - relights;
    let max_frame = counter(&app, |c| c.max_relight_frame_nanos.load(Ordering::Relaxed));
    let longest = counter(&app, |c| c.max_relight_nanos.load(Ordering::Relaxed));
    assert!(started > 1, "the bomb started {started} relights");
    assert!(longest > 0, "no relight ran on the main thread");
    assert!(
        max_frame <= budget.as_nanos() as u64 + longest,
        "relights took {max_frame} ns in a frame, over the budget of {budget:?} \
         and one relight of at most {longest} ns"
    );
    let (reach, shape, precision) = COLOURED_REACH;
    assert_on_the_reference(&mut app, &scene.edited(&bomb), reach, shape, precision);
}

#[test]
fn at_most_one_sky_probe_starts_a_frame() {
    let mut app = main_thread_app(LampWorld {
        spawning_distance: 3,
        ..world()
    });
    let mut probes = 0;
    let mut frames = 0;
    while !settled_but(&mut app, None) {
        frames += 1;
        assert!(frames < 5000, "the world never settled");
        app.update();
        let now = counter(&app, |c| c.probes.load(Ordering::Relaxed));
        assert!(
            now - probes <= 1,
            "{} sky probes started in a frame",
            now - probes
        );
        probes = now;
    }
    assert!(probes > 1, "only {probes} sky probes ran");
}
