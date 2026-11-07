//! Tests of a lit world's look: the level curves, and changing the look while
//! the world runs.

use std::sync::Arc;

use bevy::ecs::system::RunSystemOnce;
use bevy::prelude::*;

use super::super::{chunk_materials, chunk_status, update_until, DefaultVoxelMaterial};
use super::voxel_light::{app, LampWorld, LAMP, STONE};
use crate::light::material::LookUniform;
use crate::light_bench::light_stats;
use crate::prelude::*;
use crate::rendering::VoxelWorldMaterial;

#[test]
fn level_curve_at_matches_the_named_curves() {
    let close = |a: f32, b: f32| (a - b).abs() < 1e-5;

    let exponential = LevelCurve::exponential(0.7);
    assert!(close(exponential.at(15.0), 1.0));
    assert!(close(exponential.at(14.0), 0.7));
    assert!(close(exponential.at(13.0), 0.49));
    assert!(close(exponential.at(1.0), 0.7_f32.powi(14)));
    assert_eq!(exponential, LevelCurve::default());

    let linear = LevelCurve::linear();
    assert!(close(linear.at(15.0), 1.0));
    assert!(close(linear.at(5.0), 1.0 / 3.0));
    assert!(close(linear.at(3.0), 0.2));

    let gamma = LevelCurve::gamma(2.2);
    assert!(close(gamma.at(15.0), 1.0));
    // (1/3)^2.2 and (0.2)^2.2.
    assert!(close(gamma.at(5.0), 0.089_190_3));
    assert!(close(gamma.at(3.0), 0.028_991_2));

    let from_fn = LevelCurve::from_fn(|level| level as f32 * 0.01);
    assert!(close(from_fn.at(7.0), 0.07));

    let table: [f32; 16] = std::array::from_fn(|level| (level * 2) as f32);
    let from_table = LevelCurve::from_table(table);
    assert!(close(from_table.at(9.0), 18.0));

    // Between whole levels, it reads between entries, as the shader does.
    assert!(close(linear.at(4.5), 0.3));
    // Above 15 reads 15, and level 0 or below is black.
    assert!(close(linear.at(20.0), 1.0));
    assert_eq!(exponential.at(-1.0), 0.0);
}

#[test]
fn a_level_curves_entry_0_reads_0_whatever_the_table_says() {
    let mut table = [1.0; 16];
    table[0] = 0.5;
    let curve = LevelCurve::from_table(table);
    assert_eq!(curve.at(0.0), 0.0);
    // Between 0 and 1 it reads between black and level 1.
    assert!((curve.at(0.5) - 0.5).abs() < 1e-6);

    let curve = LevelCurve::from_fn(|_| 1.0);
    assert_eq!(curve.at(0.0), 0.0);
}

/// A stone floor at y = 0 with a lamp on it in the chunk at the origin, so
/// that chunk takes a slot.
fn floor_with_lamp(look: LightLook) -> LampWorld {
    LampWorld {
        scene: Arc::new(|position| {
            if position == IVec3::new(16, 1, 16) {
                WorldVoxel::Solid(LAMP)
            } else if position.y == 0 {
                WorldVoxel::Solid(STONE)
            } else {
                WorldVoxel::Air
            }
        }),
        light: Some(LightPropagation {
            sky_fill: None,
            look,
            ..default()
        }),
        ..default()
    }
}

/// An app of `world` run until the chunks at and around the origin show
/// their meshes and the world's light has settled.
fn settled(world: LampWorld) -> App {
    let mut app = app(world);
    let around = [IVec3::ZERO, IVec3::X, IVec3::NEG_X, IVec3::Z, IVec3::NEG_Z];
    update_until(&mut app, "the floor to show", |app| {
        around.iter().all(|position| {
            chunk_status::<LampWorld>(app, *position) == Some(ChunkStatus::Ready)
        }) && !chunk_materials::<LampWorld, LitMaterial>(app).is_empty()
    });
    // Chunks farther out may still be streaming in and relighting their
    // neighbours, so wait until no light work is done for a while.
    let mut quiet_frames = 0;
    let mut last = work_done(&app);
    update_until(&mut app, "the world's light to settle", |app| {
        let now = work_done(app);
        if now == last && light_is_settled(app) && every_chunk_ready(app) {
            quiet_frames += 1;
        } else {
            quiet_frames = 0;
        }
        last = now;
        quiet_frames >= 20
    });
    app
}

type LitMaterial = VoxelWorldMaterial<DefaultVoxelMaterial>;

/// Whether every spawned chunk is ready, so no chunk task is left to land
/// light.
fn every_chunk_ready(app: &mut App) -> bool {
    let keys: Vec<ChunkKey> = app
        .world_mut()
        .query::<&Chunk<LampWorld>>()
        .iter(app.world())
        .map(|chunk| chunk.key())
        .collect();
    app.world_mut()
        .run_system_once(move |voxel_world: VoxelWorld<LampWorld>| {
            keys.iter()
                .all(|key| voxel_world.chunk_state(*key) == Some(ChunkStatus::Ready))
        })
        .unwrap()
}

fn light_is_settled(app: &mut App) -> bool {
    app.world_mut()
        .run_system_once(|voxel_world: VoxelWorld<LampWorld>| {
            voxel_world.light_is_settled()
        })
        .unwrap()
}

/// The look uniform of the material on the world's chunks.
fn chunk_look(app: &mut App) -> LookUniform {
    let handles = chunk_materials::<LampWorld, LitMaterial>(app);
    let materials = app.world().resource::<Assets<LitMaterial>>();
    let looks: Vec<LookUniform> = handles
        .iter()
        .map(|handle| *materials.get(handle).unwrap().extension.look())
        .collect();
    assert!(looks.windows(2).all(|pair| pair[0] == pair[1]));
    looks[0]
}

/// Sets the look the world's config returns, through `ResMut` as a game's
/// system would.
fn set_look(app: &mut App, look: LightLook) {
    app.world_mut()
        .run_system_once(move |mut world: ResMut<LampWorld>| {
            world.light.as_mut().unwrap().look = look.clone();
        })
        .unwrap();
}

/// What a look change mustn't touch: relights, solves and slot writes.
fn work_done(app: &App) -> [u64; 6] {
    let stats = light_stats::<LampWorld>(app.world()).unwrap();
    [
        stats.relights,
        stats.solves,
        stats.slot_allocations,
        stats.plane_writes,
        stats.full_writes,
        stats.evictions,
    ]
}

#[test]
fn changing_any_look_setting_starts_no_relight_and_light_is_settled_stays_true() {
    let mut app = settled(floor_with_lamp(LightLook::default()));
    assert_eq!(chunk_look(&mut app).voxel_brightness, 1000.0);

    let changes: [fn(&mut LightLook); 9] = [
        |look| look.smoothing = LightSmoothing::Stepped,
        |look| look.sky_falloff = LevelCurve::linear(),
        |look| look.voxel_falloff = LevelCurve::gamma(2.2),
        |look| look.sun = SunGate::ByLevel(LevelCurve::linear()),
        |look| look.voxel_brightness = 250.0,
        |look| look.emissive_glow = 2.0,
        |look| look.ao = LightAo::BaseColor,
        |look| look.ao = LightAo::Off,
        |look| look.minimum_brightness = 0.05,
    ];
    let mut look = LightLook::default();
    for (index, change) in changes.iter().enumerate() {
        let before = work_done(&app);
        change(&mut look);
        set_look(&mut app, look.clone());
        app.update();

        let expected = LookUniform::new(&LightPropagation {
            sky_fill: None,
            look: look.clone(),
            ..default()
        });
        assert_eq!(chunk_look(&mut app), expected, "change {index}");
        assert!(light_is_settled(&mut app), "change {index}");
        assert_eq!(work_done(&app), before, "change {index}");
    }

    let uniform = chunk_look(&mut app);
    assert_eq!(uniform.smoothed, 0);
    assert_eq!(uniform.sun_step, 0.0);
    assert_eq!(uniform.voxel_brightness, 250.0);
    assert_eq!(uniform.emissive_glow, 2.0);
    assert_eq!(uniform.ao, 2);
    assert_eq!(uniform.minimum_brightness, 0.05);
    assert!((uniform.sky_falloff[1].y - 5.0 / 15.0).abs() < 1e-6);

    // Frames after the change find nothing more to do.
    let before = work_done(&app);
    for _ in 0..5 {
        app.update();
    }
    assert_eq!(work_done(&app), before);
    assert!(light_is_settled(&mut app));
}

#[test]
fn out_of_range_look_values_clamp() {
    let mut curve = [1.0; 16];
    curve[3] = -1.0;
    curve[4] = f32::NAN;
    let mut sun = [0.5; 16];
    sun[15] = 3.0;
    let out_of_range = LightLook {
        sky_falloff: LevelCurve::from_table(curve),
        voxel_falloff: LevelCurve::from_table(curve),
        sun: SunGate::ByLevel(LevelCurve::from_table(sun)),
        voxel_brightness: -5.0,
        emissive_glow: f32::NAN,
        minimum_brightness: -1.0,
        ..default()
    };
    let check = |uniform: LookUniform| {
        for table in [uniform.sky_falloff, uniform.voxel_falloff] {
            assert_eq!(table[0].w, 0.0);
            assert_eq!(table[1].x, 0.0);
            assert_eq!(table[1].y, 1.0);
        }
        assert_eq!(uniform.sun_falloff[3].w, 1.0);
        assert_eq!(uniform.sun_falloff[3].z, 0.5);
        assert_eq!(uniform.voxel_brightness, 0.0);
        assert_eq!(uniform.emissive_glow, 0.0);
        assert_eq!(uniform.minimum_brightness, 0.0);
    };

    // From the start.
    let mut app = settled(floor_with_lamp(out_of_range.clone()));
    check(chunk_look(&mut app));

    // And changed to while running.
    let mut app = settled(floor_with_lamp(LightLook::default()));
    set_look(&mut app, out_of_range);
    app.update();
    check(chunk_look(&mut app));
}
