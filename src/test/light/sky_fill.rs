//! Tests of sky fill: sky light reaching past where it falls and spreads,
//! under high floating solids and into caves behind wide openings, read
//! through `light_at`.

use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{Duration, Instant};

use bevy::ecs::system::RunSystemOnce;
use bevy::prelude::*;

use super::super::spawn_camera;
use super::edits::settle;
use super::reference::{fill_reference, fill_reference_in_steps, Reference};
use super::sky_light::{
    chunk_light, levels, light_is_settled, settled_but, shaft_world, spawned,
};
use super::voxel_light::{app, Hold, LampWorld, STONE};
use crate::light::LightSettings;
use crate::prelude::*;

/// Sky light under a sky ceiling at y = 100, with sky fill on at its
/// defaults, and voxel light at the default voxel reach.
///
/// The ceiling keeps the windows of the chunks above the ones spawned short,
/// so their columns are proved by their own walks.
fn filled() -> LightPropagation {
    LightPropagation {
        sky: Sky::Ceiling { y: 100 },
        ..default()
    }
}

/// Stone below y = 0 and air above, with a slab of stone 8 voxels thick at
/// `from` to `from + 8` over the ground, from -40 to 72 along x and z.
fn slab_at(from: i32) -> impl Fn(IVec3) -> WorldVoxel + Send + Sync + 'static {
    move |p: IVec3| {
        let slab = (from..from + 8).contains(&p.y)
            && (-40..72).contains(&p.x)
            && (-40..72).contains(&p.z);
        if p.y < 0 || slab {
            WorldVoxel::Solid(STONE)
        } else {
            WorldVoxel::Air
        }
    }
}

fn slab_world(from: i32) -> LampWorld {
    LampWorld {
        scene: Arc::new(slab_at(from)),
        light: Some(filled()),
        spawning_distance: 3,
        ..default()
    }
}

fn settled(world: LampWorld) -> App {
    let mut app = app(world);
    settle(&mut app, "the world to settle");
    app
}

#[test]
fn the_ground_under_a_high_floating_island_stays_lit() {
    let mut app = settled(slab_world(72));

    // The ground's top face reads the all-air chunk above it, 56 voxels in
    // from the island's edge, past where sky light spreads from direct sky.
    let under = levels(&mut app, IVec3::new(16, 0, 16));
    assert!(under.sky > 0.0, "{under:?} under the island");
    assert!(!under.direct_sky());
}

#[test]
fn the_ground_under_a_low_slab_goes_dark_in_the_middle() {
    let mut app = settled(slab_world(8));

    assert_eq!(levels(&mut app, IVec3::new(16, 0, 16)).sky, 0.0);
    // Near the slab's edge, sky light spreads in.
    assert!(levels(&mut app, IVec3::new(-36, 0, 16)).sky > 0.0);
}

/// Stone, with air above y = 0 west of a cliff at x = -32, and a cave in the
/// cliff: a mouth 24 voxels square and 8 deep, a hall behind it from x = -24
/// to 60, 32 voxels high and wide, and a second hall from x = 64 to 128
/// behind a hole 8 voxels square in the first hall's far wall.
fn cave(p: IVec3) -> WorldVoxel {
    let within = |x: (i32, i32), y: (i32, i32), z: (i32, i32)| {
        (x.0..x.1).contains(&p.x)
            && (y.0..y.1).contains(&p.y)
            && (z.0..z.1).contains(&p.z)
    };
    let outside = p.x < -32 && p.y >= 0;
    let mouth = within((-32, -24), (0, 24), (0, 24));
    let hall = within((-24, 60), (0, 32), (0, 32));
    let hole = within((60, 64), (8, 16), (8, 16));
    let far_hall = within((64, 128), (0, 32), (0, 32));
    if outside || mouth || hall || hole || far_hall {
        WorldVoxel::Air
    } else {
        WorldVoxel::Solid(STONE)
    }
}

#[test]
fn a_cave_behind_a_wide_opening_is_lit_near_its_mouth_and_dark_further_in() {
    let mut app = settled(LampWorld {
        scene: Arc::new(cave),
        light: Some(filled()),
        spawning_distance: 3,
        ..default()
    });

    // 72 voxels in from the mouth, far past where sky light spreads from the
    // direct sky outside.
    let hall = levels(&mut app, IVec3::new(40, 2, 16));
    assert!(hall.sky > 0.0, "{hall:?} in the hall");
    for x in 100..128 {
        let p = IVec3::new(x, 16, 16);
        assert_eq!(levels(&mut app, p).sky, 0.0, "at {p} in the far hall");
    }
}

#[test]
fn walling_up_a_caves_mouth_darkens_its_hall_and_opening_it_again_brings_fill_back() {
    let mut app = settled(LampWorld {
        scene: Arc::new(cave),
        light: Some(filled()),
        spawning_distance: 3,
        ..default()
    });
    let hall = IVec3::new(40, 2, 16);
    let lit = levels(&mut app, hall);

    let wall: Vec<IVec3> = (0..24)
        .flat_map(|z| (0..24).map(move |y| IVec3::new(-32, y, z)))
        .collect();
    let set = |app: &mut App, voxel: WorldVoxel| {
        let wall = wall.clone();
        app.world_mut()
            .run_system_once(move |mut voxel_world: VoxelWorld<LampWorld>| {
                for p in &wall {
                    voxel_world.set_voxel(*p, voxel);
                }
            })
            .unwrap();
    };
    set(&mut app, WorldVoxel::Solid(STONE));
    settle(&mut app, "light to settle with the mouth walled up");
    assert_eq!(levels(&mut app, hall).sky, 0.0);

    set(&mut app, WorldVoxel::Air);
    settle(&mut app, "light to settle with the mouth open");
    assert_eq!(levels(&mut app, hall), lit);
}

#[test]
fn a_1x1_shaft_lets_no_fill_through_and_a_sealed_pocket_stays_black() {
    let mut app = settled(shaft_world(filled()));

    for z in 20..26 {
        for y in 18..24 {
            for x in 20..26 {
                let p = IVec3::new(x, y, z);
                assert_eq!(levels(&mut app, p), LightLevels::new(0, 0), "at {p}");
            }
        }
    }
    // The corridor past the shaft's pool and the lamp's.
    for z in 4..12 {
        for y in 5..13 {
            for x in 42..60 {
                let p = IVec3::new(x, y, z);
                assert_eq!(levels(&mut app, p), LightLevels::new(0, 0), "at {p}");
            }
        }
    }
}

/// The sky light every voxel of `chunks` reads in `app`.
fn sky_of(app: &mut App, chunks: &[IVec3]) -> Vec<(IVec3, f32)> {
    chunks
        .iter()
        .flat_map(|chunk| {
            chunk_light(app, *chunk)
                .unwrap_or_else(|| panic!("{chunk} has no light"))
                .into_iter()
                .map(|(p, levels)| (p, levels.sky))
        })
        .collect()
}

/// Fails unless every voxel of `chunks` in `app`, a world of `scene` with
/// `light`, reads the reference's sky light, with the spawned chunks known
/// to sky fill.
fn assert_matches_reference(
    app: &mut App,
    chunks: &[IVec3],
    scene: impl Fn(IVec3) -> WorldVoxel,
    light: &LightPropagation,
    when: &str,
) {
    let Some(SkyFill::Paths(paths)) = &light.sky_fill else {
        panic!("sky fill isn't on paths");
    };
    let known = spawned(app);
    let steps = match light.precision {
        LightPrecision::Steps16 => 15,
        LightPrecision::Steps256 => 255,
    };
    let reference = fill_reference_in_steps(
        steps,
        chunks,
        &known,
        light.sky_reach as u32,
        light.shape,
        &light.sky,
        paths,
        scene,
    );
    let mut wrong = Vec::new();
    for (p, sky) in sky_of(app, chunks) {
        if sky != reference[&p] as f32 * 15.0 / steps as f32 {
            wrong.push((p, sky, reference[&p]));
        }
    }
    assert!(
        wrong.is_empty(),
        "{} voxels differ from the reference {when}, such as {:?}",
        wrong.len(),
        &wrong[..wrong.len().min(8)]
    );
}

fn set_light(app: &mut App, light: LightPropagation) {
    app.world_mut().resource_mut::<LampWorld>().light = Some(light);
}

/// Streams a world of `scene` and holds the light of `chunks` against the
/// reference, then again after a change of sky reach and one of sky fill.
fn faces_match_the_reference(scene: fn(IVec3) -> WorldVoxel, chunks: &[IVec3]) {
    let mut light = filled();
    let mut app = settled(LampWorld {
        scene: Arc::new(scene),
        light: Some(light.clone()),
        spawning_distance: 3,
        ..default()
    });
    assert_matches_reference(&mut app, chunks, scene, &light, "after streaming");

    light.sky_reach = 16;
    set_light(&mut app, light.clone());
    settle(&mut app, "light to settle after a change of sky reach");
    assert_matches_reference(&mut app, chunks, scene, &light, "at sky reach 16");

    light.sky_fill = Some(SkyFill::Paths(PathFill {
        opening_power: 1.0,
        beside_open_sky: 10.0,
        ..default()
    }));
    set_light(&mut app, light.clone());
    settle(&mut app, "light to settle after a change of sky fill");
    assert_matches_reference(&mut app, chunks, scene, &light, "after the change of fill");
}

/// [`slab_at`] with an island 8 voxels thick whose underside lies on the
/// chunk boundary at y = 64, over two chunks of air on the ground's chunk
/// boundary at y = 0.
fn island_on_boundaries(p: IVec3) -> WorldVoxel {
    slab_at(64)(p)
}

/// [`slab_at`] with a slab whose underside lies on the chunk boundary at
/// y = 32, one chunk of air above the ground.
fn low_slab_on_boundaries(p: IVec3) -> WorldVoxel {
    slab_at(32)(p)
}

#[test]
fn faces_on_chunk_boundaries_under_a_floating_island_match_the_reference() {
    faces_match_the_reference(island_on_boundaries, &[IVec3::ZERO, IVec3::Y]);
}

#[test]
fn faces_on_chunk_boundaries_under_a_low_slab_match_the_reference() {
    faces_match_the_reference(low_slab_on_boundaries, &[IVec3::ZERO]);
}

/// Chunks under and around the island of [`island_on_boundaries`].
const AROUND_ISLAND: [IVec3; 6] = [
    IVec3::ZERO,
    IVec3::Y,
    IVec3::new(1, 1, 0),
    IVec3::new(-1, 0, -1),
    IVec3::new(0, 2, 0),
    IVec3::new(-2, 1, 0),
];

fn island_world() -> LampWorld {
    LampWorld {
        scene: Arc::new(island_on_boundaries),
        light: Some(filled()),
        spawning_distance: 3,
        ..default()
    }
}

#[test]
fn after_streaming_settles_at_256_levels_every_fully_loaded_chunk_matches_the_fill_reference(
) {
    let light = LightPropagation {
        precision: LightPrecision::Steps256,
        ..filled()
    };
    let mut app = settled(LampWorld {
        light: Some(light.clone()),
        ..island_world()
    });
    assert_matches_reference(
        &mut app,
        &AROUND_ISLAND,
        island_on_boundaries,
        &light,
        "at 256 levels",
    );
}

#[test]
fn arrival_order_changes_no_settled_fill_and_no_reading_is_brighter_than_the_reference() {
    let mut normal = settled(island_world());
    let expected = sky_of(&mut normal, &AROUND_ISLAND);

    // The island's chunk arrives last, after the chunks around it have
    // settled.
    let _holding = Hold::exclusive();
    let held = IVec3::new(0, 2, 0);
    let hold = Hold::new(held);
    let mut app = app(LampWorld {
        hold: Some(hold.clone()),
        ..island_world()
    });
    let deadline = Instant::now() + Duration::from_secs(60);
    while !settled_but(&mut app, Some(held)) {
        assert!(
            Instant::now() < deadline,
            "timed out waiting for the stream"
        );
        app.update();
    }
    hold.release();
    settle(&mut app, "the held chunk to land");
    assert!(sky_of(&mut app, &AROUND_ISLAND) == expected);

    let light = filled();
    let Some(SkyFill::Paths(paths)) = &light.sky_fill else {
        unreachable!()
    };
    let reference = fill_reference(
        &AROUND_ISLAND,
        &spawned(&mut app),
        30,
        LightShape::Round,
        &light.sky,
        paths,
        island_on_boundaries,
    );
    for (p, sky) in sky_of(&mut app, &AROUND_ISLAND) {
        assert!(
            sky <= reference[&p] as f32,
            "{sky} at {p} over {}",
            reference[&p]
        );
    }
}

#[test]
fn light_reads_the_same_after_reset_light_rebuilds_the_record_and_the_field() {
    let mut app = settled(island_world());
    let before = sky_of(&mut app, &AROUND_ISLAND);
    assert!(before.iter().any(|(_, sky)| *sky > 0.0 && *sky < 15.0));

    app.world_mut()
        .run_system_once(|mut voxel_world: VoxelWorld<LampWorld>| {
            voxel_world.reset_light()
        })
        .unwrap();
    assert!(!light_is_settled(&mut app));
    settle(&mut app, "light to settle after reset_light");
    assert!(sky_of(&mut app, &AROUND_ISLAND) == before);
}

#[test]
fn a_cell_with_direct_sky_reads_15() {
    let mut app = settled(island_world());

    // The cell at the bottom of the open sky beside the island, at x = -64
    // to -56.
    for z in 0..8 {
        for y in 0..8 {
            for x in -64..-56 {
                let p = IVec3::new(x, y, z);
                let levels = levels(&mut app, p);
                assert_eq!(levels.sky, 15.0, "at {p}");
                assert!(levels.direct_sky());
            }
        }
    }
}

/// The light of `chunks` in a `VoxelWorldPlugin::minimal()` app of `world`
/// and in a shown one, settled.
fn minimal_and_shown(
    world: LampWorld,
    apps: &mut [App; 2],
    chunks: &[IVec3],
    when: &str,
) {
    for app in apps.iter_mut() {
        app.world_mut().insert_resource(world.clone());
        settle(app, when);
    }
    let [minimal, shown] = apps;
    assert!(sky_of(minimal, chunks) == sky_of(shown, chunks), "{when}");
}

#[test]
fn a_minimal_world_reads_the_same_light_as_a_shown_one_through_streaming_fill_and_a_settings_change(
) {
    let world = island_world();
    let mut minimal = App::new();
    minimal
        .add_plugins((
            MinimalPlugins,
            VoxelWorldPlugin::<LampWorld>::minimal_with_config(world.clone()),
        ))
        .add_systems(Startup, spawn_camera::<LampWorld>);
    let mut apps = [minimal, app(world.clone())];
    minimal_and_shown(world.clone(), &mut apps, &AROUND_ISLAND, "after streaming");

    let changed = LampWorld {
        light: Some(LightPropagation {
            sky_fill: Some(SkyFill::Paths(PathFill {
                opening_power: 1.0,
                ..default()
            })),
            ..filled()
        }),
        ..world
    };
    minimal_and_shown(changed, &mut apps, &AROUND_ISLAND, "after a change of fill");
}

#[test]
fn turning_sky_fill_off_and_on_at_runtime_relights_to_match_the_reference() {
    let chunks = [IVec3::ZERO];
    let mut light = filled();
    let mut app = settled(island_world());
    let lit = sky_of(&mut app, &chunks);

    light.sky_fill = None;
    set_light(&mut app, light.clone());
    assert!(
        !light_is_settled(&mut app) || {
            app.update();
            !light_is_settled(&mut app)
        }
    );
    settle(&mut app, "light to settle with fill off");
    let reference = Reference::sky_light(
        IVec3::splat(-30),
        IVec3::splat(62),
        30,
        LightShape::Round,
        &light.sky,
        island_on_boundaries,
    );
    for (p, sky) in sky_of(&mut app, &chunks) {
        assert_eq!(sky, reference.sky_at(p) as f32, "at {p} with fill off");
    }
    assert!(!app
        .world()
        .contains_resource::<crate::light::fill::driver::SkyFillWork<LampWorld>>());

    set_light(&mut app, filled());
    settle(&mut app, "light to settle with fill on again");
    assert!(sky_of(&mut app, &chunks) == lit);
}

#[test]
fn with_sky_fill_off_the_world_keeps_no_field_and_no_cells() {
    let mut app = settled(LampWorld {
        light: Some(LightPropagation {
            sky_fill: None,
            ..filled()
        }),
        ..island_world()
    });
    assert!(!app
        .world()
        .contains_resource::<crate::light::fill::driver::SkyFillWork<LampWorld>>());
    let settings = app.world().resource::<LightSettings<LampWorld>>();
    let counters = settings.counters();
    assert_eq!(counters.fill_seeded.load(Ordering::Relaxed), 0);
    assert_eq!(counters.fill_probes.load(Ordering::Relaxed), 0);
    assert!(settings.record.lock().unwrap().all_cells().is_empty());
    assert!(settings.fill_cells.read().is_empty());
    let _ = levels(&mut app, IVec3::ZERO);
}
