//! Tests of the sky record and sky probes: columns a chunk's own task leaves
//! provisional settle once probes look up the positions above it, and what
//! the record remembers, forgets or loses to voxel edits never changes the
//! light that settles. Sky fill is off in all of them.

use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use bevy::ecs::system::RunSystemOnce;
use bevy::prelude::*;

use super::super::{chunk_status, move_cameras, update_until};
use super::reference::Reference;
use super::sky_light::{
    around_origin, chunk_light, levels, light_is_settled, settled, settled_but,
    sky_light, spawned,
};
use super::voxel_light::{app, light_at, Hold, LampWorld, Scene, STONE};
use crate::light::LightSettings;
use crate::prelude::*;

/// Ground below y = -40, and a roof of stone from y = 100 to 104, in the
/// chunk positions at chunk height 3, above where the camera spawns chunks.
/// The roof has a hole 4 voxels wide at x and z from 8 to 12.
fn roof_with_hole(p: IVec3) -> WorldVoxel {
    let roof = (100..104).contains(&p.y);
    let hole = (8..12).contains(&p.x) && (8..12).contains(&p.z);
    if p.y < -40 || roof && !hole {
        WorldVoxel::Solid(STONE)
    } else {
        WorldVoxel::Air
    }
}

/// The ground of [`roof_with_hole`] alone.
fn open_sky(p: IVec3) -> WorldVoxel {
    if p.y < -40 {
        WorldVoxel::Solid(STONE)
    } else {
        WorldVoxel::Air
    }
}

/// A chunk position of the roof, unloaded, over the chunks at x = 1, z = 0.
const ROOF: IVec3 = IVec3::new(1, 3, 0);

/// A voxel under the roof, in a chunk below [`ROOF`].
const UNDER_ROOF: IVec3 = IVec3::new(40, 40, 16);

fn world(scene: fn(IVec3) -> WorldVoxel) -> LampWorld {
    LampWorld {
        scene: Arc::new(scene),
        light: Some(sky_light(Sky::default(), 30)),
        ..default()
    }
}

/// The reference sky light and voxel light of `scene` for the chunks around
/// the origin.
fn reference_of(scene: fn(IVec3) -> WorldVoxel) -> (Reference, Reference) {
    let margin = IVec3::splat(31);
    let (min, max) = (IVec3::splat(-32) - margin, IVec3::splat(64) + margin);
    let sky =
        Reference::sky_light(min, max, 30, LightShape::Round, &Sky::default(), scene);
    let voxel = Reference::voxel_light(min, max, 14, LightShape::Round, scene, |_| None);
    (sky, voxel)
}

fn roof_reference() -> &'static (Reference, Reference) {
    static REFERENCE: OnceLock<(Reference, Reference)> = OnceLock::new();
    REFERENCE.get_or_init(|| reference_of(roof_with_hole))
}

fn open_sky_reference() -> &'static (Reference, Reference) {
    static REFERENCE: OnceLock<(Reference, Reference)> = OnceLock::new();
    REFERENCE.get_or_init(|| reference_of(open_sky))
}

/// The settled light of each chunk around the origin.
fn light_around_origin(app: &mut App) -> HashMap<IVec3, Vec<(IVec3, LightLevels)>> {
    around_origin()
        .into_iter()
        .filter_map(|chunk| Some((chunk, chunk_light(app, chunk)?)))
        .collect()
}

/// Fails unless every chunk around the origin whose whole box is loaded
/// matches the reference.
fn assert_matches(app: &mut App, reference: &(Reference, Reference)) {
    let spawned = spawned(app);
    let checked: Vec<IVec3> = around_origin()
        .into_iter()
        .filter(|chunk| {
            around_origin()
                .iter()
                .all(|offset| spawned.contains(&(*chunk + *offset)))
        })
        .collect();
    assert!(
        checked.contains(&IVec3::ZERO),
        "the origin's box isn't loaded"
    );
    for chunk in checked {
        for (p, levels) in chunk_light(app, chunk).expect("the chunk's light") {
            assert_eq!(
                (levels.sky, levels.voxel),
                (reference.0.sky_at(p) as f32, reference.1.voxel_at(p) as f32),
                "at {p} in {chunk}"
            );
        }
    }
}

/// Fails if any voxel of the chunks around the origin reads brighter than the
/// reference.
fn assert_no_brighter(app: &mut App, reference: &(Reference, Reference)) {
    for (chunk, light) in light_around_origin(app) {
        for (p, levels) in light {
            let sky = reference.0.sky_at(p) as f32;
            assert!(
                levels.sky <= sky,
                "{} at {p} in {chunk} is brighter than {sky}",
                levels.sky
            );
        }
    }
}

/// Fails unless every open voxel with light in the column through `p`, down
/// to the ground, has direct sky exactly when `direct_sky`.
fn assert_column(app: &mut App, p: IVec3, direct_sky: bool) {
    let mut checked = 0;
    for y in -40..128 {
        let p = IVec3::new(p.x, y, p.z);
        if let Some(levels) = light_at(app, p) {
            assert_eq!(levels.direct_sky(), direct_sky, "at {p}");
            checked += 1;
        }
    }
    assert!(checked >= 64, "only {checked} voxels have light");
}

fn counter(app: &App, read: impl Fn(&crate::light::solver::LightCounters) -> u64) -> u64 {
    read(
        app.world()
            .resource::<LightSettings<LampWorld>>()
            .counters(),
    )
}

fn relights_started(app: &App) -> u64 {
    counter(app, |counters| counters.relights.load(Ordering::Relaxed))
}

fn set_voxel(app: &mut App, position: IVec3, voxel: WorldVoxel) {
    app.world_mut()
        .run_system_once(move |mut voxel_world: VoxelWorld<LampWorld>| {
            voxel_world.set_voxel(position, voxel)
        })
        .unwrap();
}

/// Runs `app` until every lookup `hold` holds back has started waiting, and
/// every spawned chunk but `held` is Ready.
fn until_held(app: &mut App, hold: &Hold, held: Option<IVec3>) {
    update_until(app, "the hold to be waited on", |app| {
        let chunks = spawned(app);
        hold.waited.load(Ordering::Acquire) > 0
            && !chunks.is_empty()
            && chunks
                .iter()
                .filter(|chunk| Some(**chunk) != held)
                .all(|chunk| {
                    chunk_status::<LampWorld>(app, *chunk) == Some(ChunkStatus::Ready)
                })
    });
}

#[test]
fn a_chunk_whose_columns_start_provisional_and_are_disproved_ends_with_no_direct_sky_under_rock(
) {
    let _holding = Hold::exclusive();
    let hold = Hold::inside(ROOF);
    let mut app = app(LampWorld {
        hold: Some(hold.clone()),
        ..world(roof_with_hole)
    });
    until_held(&mut app, &hold, None);
    // Too many columns are open for a walk, so they show direct sky until a
    // probe of the roof's position disproves them.
    assert!(levels(&mut app, UNDER_ROOF).direct_sky());

    hold.release();
    update_until(&mut app, "the world to settle", |app| {
        settled_but(app, None)
    });
    assert_column(&mut app, UNDER_ROOF, false);
    // Under the hole, direct sky falls to the ground.
    assert!(levels(&mut app, IVec3::new(9, -40, 9)).direct_sky());
    assert_matches(&mut app, roof_reference());
}

#[test]
fn light_is_settled_stays_false_until_probes_finish() {
    let _holding = Hold::exclusive();
    let hold = Hold::inside(ROOF);
    let mut app = app(LampWorld {
        hold: Some(hold.clone()),
        ..world(roof_with_hole)
    });
    until_held(&mut app, &hold, None);
    for _ in 0..10 {
        app.update();
        assert!(!light_is_settled(&mut app));
    }
    hold.release();
    update_until(&mut app, "light to settle", light_is_settled);
    assert!(!levels(&mut app, UNDER_ROOF).direct_sky());
}

#[test]
fn emptying_the_sky_record_changes_no_settled_light() {
    let mut normal = settled(world(roof_with_hole));
    let expected = light_around_origin(&mut normal);

    // The record is emptied every frame, so nothing a chunk or a probe adds
    // stays in it.
    let mut emptied = app(world(roof_with_hole));
    emptied.add_systems(Last, |settings: Res<LightSettings<LampWorld>>| {
        settings.record.lock().unwrap().forget_all();
    });
    update_until(&mut emptied, "the world to settle", |app| {
        settled_but(app, None)
    });
    assert!(light_around_origin(&mut emptied) == expected);
    assert_matches(&mut emptied, roof_reference());
}

#[test]
fn evicting_under_a_tiny_cap_changes_no_settled_light() {
    let mut normal = settled(world(roof_with_hole));
    let expected = light_around_origin(&mut normal);

    let mut tiny = settled(LampWorld {
        light: Some(LightPropagation {
            sky_record_cap: 0,
            ..sky_light(Sky::default(), 30)
        }),
        ..world(roof_with_hole)
    });
    let evictions = tiny
        .world()
        .resource::<LightSettings<LampWorld>>()
        .record
        .lock()
        .unwrap()
        .evictions;
    assert!(evictions > 0);
    assert!(light_around_origin(&mut tiny) == expected);
}

/// Moves the camera far along x and back, until the chunk at `chunk` has
/// retired and spawned again. Returns the light it first published with.
fn respawn(app: &mut App, chunk: IVec3) -> Vec<(IVec3, LightLevels)> {
    let away = Transform::from_xyz(10.0 + 32.0 * 10.0, 10.0, 10.0)
        .looking_at(Vec3::new(32.0 * 10.0, 0.0, 0.0), Vec3::Y);
    move_cameras::<LampWorld>(app, away);
    update_until(app, "the chunk to retire", |app| {
        !spawned(app).contains(&chunk)
    });
    let back = Transform::from_xyz(10.0, 10.0, 10.0).looking_at(Vec3::ZERO, Vec3::Y);
    move_cameras::<LampWorld>(app, back);
    let deadline = Instant::now() + Duration::from_secs(20);
    loop {
        assert!(Instant::now() < deadline, "timed out waiting for the chunk");
        app.update();
        if let Some(light) = chunk_light(app, chunk) {
            return light;
        }
    }
}

/// The voxels of `light` with direct sky.
fn direct_sky(light: &[(IVec3, LightLevels)]) -> Vec<IVec3> {
    light
        .iter()
        .filter(|(_, levels)| levels.direct_sky())
        .map(|(p, _)| *p)
        .collect()
}

#[test]
fn a_respawned_chunk_reads_the_same_light_on_its_first_publish() {
    // Under the roof's hole, so direct sky reaches only 16 of its columns,
    // which takes every position above it to prove.
    let chunk = IVec3::new(0, 1, 0);
    let mut app = settled(world(roof_with_hole));
    let settled_light = chunk_light(&mut app, chunk).unwrap();
    let expected = direct_sky(&settled_light);
    assert_eq!(expected.len(), 16 * 32);

    let first = respawn(&mut app, chunk);
    assert_eq!(direct_sky(&first), expected);
}

/// Stone below y = 320 with air above, a cave at the origin's chunk from
/// y = 4 to 20, and a 1×1 shaft from the cave's roof up to the surface, 300
/// voxels long.
fn cave_under_tall_shaft(p: IVec3) -> WorldVoxel {
    let cave = (4..28).contains(&p.x) && (4..20).contains(&p.y) && (4..28).contains(&p.z);
    let shaft = p.x == 16 && p.z == 16 && (20..320).contains(&p.y);
    if p.y >= 320 || cave || shaft {
        WorldVoxel::Air
    } else {
        WorldVoxel::Solid(STONE)
    }
}

#[test]
fn a_cave_lit_through_a_tall_shaft_stays_lit_when_the_surface_chunks_retire() {
    let scene: Scene = Arc::new(cave_under_tall_shaft);
    // Above the surface's chunks, fill hints settle the sky with no probes.
    let mut app = app(LampWorld {
        scene,
        light: Some(sky_light(Sky::default(), 30)),
        hints: Some(Arc::new(|position: IVec3| {
            if position.y >= 11 {
                FillHint::Uniform(WorldVoxel::Air)
            } else {
                FillHint::Unknown
            }
        })),
        ..default()
    });
    // The camera spawns at the first update.
    app.update();
    let surface = Transform::from_xyz(10.0, 330.0, 10.0)
        .looking_at(Vec3::new(0.0, 320.0, 0.0), Vec3::Y);
    move_cameras::<LampWorld>(&mut app, surface);
    update_until(&mut app, "the surface to settle", |app| {
        spawned(app).contains(&IVec3::new(0, 10, 0)) && settled_but(app, None)
    });

    let cave = Transform::from_xyz(10.0, 10.0, 10.0).looking_at(Vec3::ZERO, Vec3::Y);
    move_cameras::<LampWorld>(&mut app, cave);
    update_until(&mut app, "the cave to settle", |app| {
        let spawned = spawned(app);
        spawned.contains(&IVec3::ZERO)
            && !spawned.contains(&IVec3::new(0, 10, 0))
            && settled_but(app, None)
    });

    let margin = IVec3::splat(31);
    let reference = Reference::sky_light(
        IVec3::splat(-32) - margin,
        IVec3::splat(64) + margin,
        30,
        LightShape::Round,
        &Sky::default(),
        cave_under_tall_shaft,
    );
    for y in 4..32 {
        assert!(
            levels(&mut app, IVec3::new(16, y, 16)).direct_sky(),
            "at y = {y}"
        );
    }
    for (p, levels) in chunk_light(&mut app, IVec3::ZERO).unwrap() {
        assert_eq!(levels.sky, reference.sky_at(p) as f32, "at {p}");
    }
}

#[test]
fn chunks_retiring_one_at_a_time_leave_no_reading_brighter_than_the_reference() {
    // The chunk two along x keeps its provisional columns while its probe is
    // held, and lights its neighbour with direct sky it doesn't have.
    let _holding = Hold::exclusive();
    let retiring = IVec3::new(2, 0, 0);
    let hold = Hold::inside(IVec3::new(2, 3, 0));
    let mut app = app(LampWorld {
        hold: Some(hold.clone()),
        ..world(roof_with_hole)
    });
    until_held(&mut app, &hold, None);
    assert!(spawned(&mut app).contains(&retiring));
    // The held probe keeps light unsettled, so a few frames go by for the
    // other probes and relights to land.
    for _ in 0..60 {
        app.update();
    }
    assert!(levels(&mut app, IVec3::new(64, 0, 16)).direct_sky());

    let spawned_before = spawned(&mut app).len();
    app.world_mut()
        .resource_mut::<LampWorld>()
        .spawning_distance = 1;
    update_until(&mut app, "the chunk to retire", |app| {
        let spawned = spawned(app);
        spawned.len() < spawned_before && !spawned.contains(&retiring)
    });
    hold.release();
    update_until(&mut app, "the world to settle", |app| {
        settled_but(app, None)
    });
    assert_no_brighter(&mut app, roof_reference());
}

#[test]
fn a_chunk_retiring_with_no_provisional_columns_starts_no_relight() {
    let mut app = settled(world(roof_with_hole));
    for _ in 0..3 {
        app.update();
    }
    let before = relights_started(&app);
    let spawned_before = spawned(&mut app).len();
    app.world_mut()
        .resource_mut::<LampWorld>()
        .spawning_distance = 1;
    update_until(&mut app, "chunks to retire", |app| {
        spawned(app).len() < spawned_before && settled_but(app, None)
    });
    for _ in 0..3 {
        app.update();
    }
    assert_eq!(relights_started(&app), before);
}

#[test]
fn an_edit_made_while_a_probe_is_running_over_its_position_never_leaves_a_stale_mask() {
    let _holding = Hold::exclusive();
    let hold = Hold::inside(ROOF);
    let mut app = app(LampWorld {
        hold: Some(hold.clone()),
        ..world(roof_with_hole)
    });
    until_held(&mut app, &hold, None);
    // A hole dug through the roof while its probe runs.
    let dug = IVec3::new(48, 0, 20);
    for y in 100..104 {
        set_voxel(&mut app, IVec3::new(dug.x, y, dug.z), WorldVoxel::Air);
    }
    for _ in 0..3 {
        app.update();
    }
    hold.release();
    update_until(&mut app, "the world to settle", |app| {
        settled_but(app, None)
    });
    assert!(counter(&app, |c| c.probes_discarded.load(Ordering::Relaxed)) > 0);
    assert_column(&mut app, dug, true);
    assert!(!levels(&mut app, UNDER_ROOF).direct_sky());
}

/// Ground below y = -40, and stone from y = 64 to 128 but for a 1×1 shaft at
/// x = 16, z = 16, so the chunks below it walk the shaft's column.
fn rock_with_a_shaft(p: IVec3) -> WorldVoxel {
    let rock = (64..128).contains(&p.y) && !(p.x == 16 && p.z == 16);
    if p.y < -40 || rock {
        WorldVoxel::Solid(STONE)
    } else {
        WorldVoxel::Air
    }
}

#[test]
fn an_edit_made_while_a_walk_is_running_over_its_position_never_leaves_a_stale_answer() {
    let _holding = Hold::exclusive();
    // The position above the loaded chunks, which their walks read.
    let hold = Hold::inside(IVec3::new(0, 3, 0));
    let mut app = app(LampWorld {
        hold: Some(hold.clone()),
        ..world(rock_with_a_shaft)
    });
    update_until(&mut app, "a walk to be held", |app| {
        hold.waited.load(Ordering::Acquire) > 0 && !spawned(app).is_empty()
    });
    // The shaft is capped while the walk runs.
    set_voxel(&mut app, IVec3::new(16, 110, 16), WorldVoxel::Solid(STONE));
    for _ in 0..3 {
        app.update();
    }
    hold.release();
    update_until(&mut app, "the world to settle", |app| {
        settled_but(app, None)
    });
    for y in 64..96 {
        let p = IVec3::new(16, y, 16);
        assert!(!levels(&mut app, p).direct_sky(), "at {p}");
    }
}

#[test]
fn reset_light_after_changing_what_lookups_return_makes_light_follow_the_new_voxels() {
    let mut app = settled(world(roof_with_hole));
    assert!(!levels(&mut app, UNDER_ROOF).direct_sky());

    // The roof is gone from the lookups. Every loaded chunk is below it, so
    // they keep the same voxels.
    app.world_mut().resource_mut::<LampWorld>().scene = Arc::new(open_sky);
    app.world_mut()
        .run_system_once(|mut voxel_world: VoxelWorld<LampWorld>| {
            voxel_world.reset_light()
        })
        .unwrap();
    assert!(!light_is_settled(&mut app));
    update_until(&mut app, "the world to settle", |app| {
        settled_but(app, None)
    });
    assert!(levels(&mut app, UNDER_ROOF).direct_sky());
    assert_matches(&mut app, open_sky_reference());
}

#[test]
fn a_probe_reads_voxel_edits_at_unloaded_positions() {
    let mut app = app(world(roof_with_hole));
    app.update();
    // A hole dug through the roof before any probe reads it.
    for y in 100..104 {
        set_voxel(&mut app, IVec3::new(48, y, 20), WorldVoxel::Air);
    }
    update_until(&mut app, "the world to settle", |app| {
        settled_but(app, None)
    });
    assert_column(&mut app, IVec3::new(48, 0, 20), true);
    assert!(light_at(&mut app, UNDER_ROOF).is_some_and(|levels| !levels.direct_sky()));
}
