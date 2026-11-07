//! Tests of sky fill: sky light reaching past where it falls and spreads,
//! under high floating solids and into caves behind wide openings, read
//! through `light_at`.
//!
//! The suite every model must pass runs once per model, in the modules
//! `paths` and `openness` at the end.

use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{Duration, Instant};

use bevy::ecs::system::RunSystemOnce;
use bevy::prelude::*;

use super::super::spawn_camera;
use super::edits::settle;
use super::reference::{fill_reference_everywhere, fill_reference_in_steps, Reference};
use super::sky_light::{
    chunk_light, levels, light_is_settled, settled_but, shaft_world, spawned,
};
use super::voxel_light::{app, Hold, LampWorld, STONE};
use crate::light::LightSettings;
use crate::prelude::*;

/// Sky light under a sky ceiling at y = 100, with sky fill on under paths
/// at its defaults, and voxel light at the default voxel reach.
///
/// The ceiling keeps the windows of the chunks above the ones spawned short,
/// so their columns are proved by their own walks.
fn filled() -> LightPropagation {
    LightPropagation {
        sky: Sky::Ceiling { y: 100 },
        sky_fill: Some(paths()),
        ..default()
    }
}

/// [`filled`] with sky fill under `fill`.
fn filled_with(fill: &SkyFill) -> LightPropagation {
    LightPropagation {
        sky_fill: Some(fill.clone()),
        ..filled()
    }
}

/// Paths with its defaults.
fn paths() -> SkyFill {
    SkyFill::Paths(PathFill::default())
}

/// Sky openness with its defaults.
fn openness() -> SkyFill {
    SkyFill::Openness(OpenFill::default())
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
    set_voxels(&mut app, &wall, WorldVoxel::Solid(STONE));
    settle(&mut app, "light to settle with the mouth walled up");
    assert_eq!(levels(&mut app, hall).sky, 0.0);

    set_voxels(&mut app, &wall, WorldVoxel::Air);
    settle(&mut app, "light to settle with the mouth open");
    assert_eq!(levels(&mut app, hall), lit);
}

fn set_voxels(app: &mut App, voxels: &[IVec3], voxel: WorldVoxel) {
    let voxels = voxels.to_vec();
    app.world_mut()
        .run_system_once(move |mut voxel_world: VoxelWorld<LampWorld>| {
            for p in &voxels {
                voxel_world.set_voxel(*p, voxel);
            }
        })
        .unwrap();
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

/// The positions the world's sky fill field knows.
fn known_to_fill(app: &mut App) -> Vec<IVec3> {
    let settings = app.world().resource::<LightSettings<LampWorld>>();
    let known = settings.fill_cells.read().keys().copied().collect();
    known
}

/// The top step of `light`'s precision: 15 at 16 levels, 255 at 256.
fn steps_of(light: &LightPropagation) -> u32 {
    match light.precision {
        LightPrecision::Steps16 => 15,
        LightPrecision::Steps256 => 255,
    }
}

/// The reference's sky light of `chunks` in a world of `scene` with `light`,
/// with the positions `known` known to sky fill, in steps of its precision.
fn reference_of(
    chunks: &[IVec3],
    known: &[IVec3],
    scene: impl Fn(IVec3) -> WorldVoxel,
    light: &LightPropagation,
) -> bevy::platform::collections::HashMap<IVec3, u8> {
    fill_reference_in_steps(
        steps_of(light),
        chunks,
        known,
        light.sky_reach as u32,
        light.shape,
        &light.sky,
        light.sky_fill.as_ref().expect("sky fill is on"),
        scene,
    )
}

/// Fails unless every voxel of `chunks` in `app`, a world of `scene` with
/// `light`, reads the reference's sky light, with the positions the field
/// knows known to the reference.
fn assert_matches_reference(
    app: &mut App,
    chunks: &[IVec3],
    scene: impl Fn(IVec3) -> WorldVoxel,
    light: &LightPropagation,
    when: &str,
) {
    let known = known_to_fill(app);
    let reference = reference_of(chunks, &known, scene, light);
    let mut wrong = Vec::new();
    let steps = steps_of(light);
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

/// Chunks under and around the island of [`island_on_boundaries`].
const AROUND_ISLAND: [IVec3; 6] = [
    IVec3::ZERO,
    IVec3::Y,
    IVec3::new(1, 1, 0),
    IVec3::new(-1, 0, -1),
    IVec3::new(0, 2, 0),
    IVec3::new(-2, 1, 0),
];

fn island_world(fill: &SkyFill) -> LampWorld {
    LampWorld {
        scene: Arc::new(island_on_boundaries),
        light: Some(filled_with(fill)),
        spawning_distance: 3,
        ..default()
    }
}

/// Stone below y = 0 and air above, with an island of stone 8 voxels thick
/// at y = 72 to 80, from -8 to 24 along x and z: high above the ground and
/// narrow, so the ground under it sees most of the sky.
fn high_narrow_island(p: IVec3) -> WorldVoxel {
    let island =
        (72..80).contains(&p.y) && (-8..24).contains(&p.x) && (-8..24).contains(&p.z);
    if p.y < 0 || island {
        WorldVoxel::Solid(STONE)
    } else {
        WorldVoxel::Air
    }
}

fn high_island_world(fill: &SkyFill) -> LampWorld {
    LampWorld {
        scene: Arc::new(high_narrow_island),
        light: Some(filled_with(fill)),
        spawning_distance: 3,
        ..default()
    }
}

/// The middle of the ground under [`high_narrow_island`].
const UNDER_HIGH_ISLAND: IVec3 = IVec3::new(8, 0, 8);

#[test]
fn a_high_floating_island_under_sky_openness_leaves_no_straight_down_shade_where_most_of_the_sky_is_in_view(
) {
    // Paths leaves the ground straight under the island in shade.
    let mut paths = settled(high_island_world(&paths()));
    assert!(levels(&mut paths, UNDER_HIGH_ISLAND).sky < 15.0);

    let mut open = settled(high_island_world(&openness()));
    let levels = levels(&mut open, UNDER_HIGH_ISLAND);
    assert_eq!(levels.sky, 15.0, "{levels:?} under the island");
    assert!(levels.direct_sky());
}

#[test]
fn changing_sky_fill_from_paths_to_openness_at_runtime_settles_on_the_references_light_for_openness(
) {
    let chunks = [IVec3::ZERO, IVec3::Y];
    let mut app = settled(high_island_world(&paths()));
    let light = filled_with(&openness());
    set_light(&mut app, light.clone());
    settle(&mut app, "light to settle under openness");
    assert_matches_reference(
        &mut app,
        &chunks,
        high_narrow_island,
        &light,
        "after changing to openness",
    );
    assert_eq!(levels(&mut app, UNDER_HIGH_ISLAND).sky, 15.0);
}

#[test]
fn a_roof_built_over_ground_under_sky_openness_shades_it_and_taking_it_away_lights_it_again(
) {
    let mut app = settled(high_island_world(&openness()));
    let lit = levels(&mut app, UNDER_HIGH_ISLAND);

    // A roof 64 voxels wide, 16 voxels over the ground.
    let roof: Vec<IVec3> = (-24..40)
        .flat_map(|z| (-24..40).map(move |x| IVec3::new(x, 16, z)))
        .collect();
    set_voxels(&mut app, &roof, WorldVoxel::Solid(STONE));
    settle(&mut app, "light to settle under the roof");
    let shaded = levels(&mut app, UNDER_HIGH_ISLAND);
    assert!(shaded.sky < 15.0, "{shaded:?} under the roof");
    let with_roof = |p: IVec3| {
        if p.y == 16 && (-24..40).contains(&p.x) && (-24..40).contains(&p.z) {
            WorldVoxel::Solid(STONE)
        } else {
            high_narrow_island(p)
        }
    };
    assert_matches_reference(
        &mut app,
        &[IVec3::ZERO],
        with_roof,
        &filled_with(&openness()),
        "under the roof",
    );

    set_voxels(&mut app, &roof, WorldVoxel::Air);
    settle(&mut app, "light to settle with the roof gone");
    assert_eq!(levels(&mut app, UNDER_HIGH_ISLAND), lit);
}

#[test]
fn out_of_range_sky_openness_settings_clamp() {
    let app = settled(high_island_world(&SkyFill::Openness(OpenFill {
        sun_at: 2.0,
        gain: -1.0,
        rays: 0,
        reach: 100_000,
    })));
    let settings = app.world().resource::<LightSettings<LampWorld>>();
    assert_eq!(
        settings.config.sky_fill,
        Some(SkyFill::Openness(OpenFill {
            sun_at: 1.0,
            gain: 0.0,
            rays: 1,
            reach: 512,
        }))
    );
}

// --- Once per model ----------------------------------------------------------

fn after_streaming_settles_at_256_levels_every_fully_loaded_chunk_matches_the_fill_reference(
    fill: SkyFill,
) {
    let light = LightPropagation {
        precision: LightPrecision::Steps256,
        ..filled_with(&fill)
    };
    let mut app = settled(LampWorld {
        light: Some(light.clone()),
        ..island_world(&fill)
    });
    assert_matches_reference(
        &mut app,
        &AROUND_ISLAND,
        island_on_boundaries,
        &light,
        "at 256 levels",
    );
}

fn a_1x1_shaft_lets_no_fill_through_and_a_sealed_pocket_stays_black(fill: SkyFill) {
    let mut app = settled(shaft_world(filled_with(&fill)));

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

/// Streams a world of `scene` and holds the light of `chunks` against the
/// reference, then again after a change of sky reach and one to `changed`.
fn faces_match_the_reference(
    fill: SkyFill,
    changed: SkyFill,
    scene: fn(IVec3) -> WorldVoxel,
    chunks: &[IVec3],
) {
    let mut light = filled_with(&fill);
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

    light.sky_fill = Some(changed);
    set_light(&mut app, light.clone());
    settle(&mut app, "light to settle after a change of sky fill");
    assert_matches_reference(&mut app, chunks, scene, &light, "after the change of fill");
}

/// `fill` with other settings, for a change of sky fill.
fn changed(fill: &SkyFill) -> SkyFill {
    match fill {
        SkyFill::Paths(_) => SkyFill::Paths(PathFill {
            opening_power: 1.0,
            beside_open_sky: 10.0,
            ..default()
        }),
        SkyFill::Openness(_) => SkyFill::Openness(OpenFill {
            sun_at: 0.6,
            gain: 0.5,
            rays: 16,
            reach: 64,
        }),
    }
}

fn faces_on_chunk_boundaries_under_a_floating_island_match_the_reference(fill: SkyFill) {
    let changed = changed(&fill);
    faces_match_the_reference(
        fill,
        changed,
        island_on_boundaries,
        &[IVec3::ZERO, IVec3::Y],
    );
}

fn faces_on_chunk_boundaries_under_a_low_slab_match_the_reference(fill: SkyFill) {
    let changed = changed(&fill);
    faces_match_the_reference(fill, changed, low_slab_on_boundaries, &[IVec3::ZERO]);
}

fn arrival_order_changes_no_settled_fill_and_no_reading_is_brighter_than_the_reference(
    fill: SkyFill,
) {
    let mut normal = settled(island_world(&fill));
    let expected = sky_of(&mut normal, &AROUND_ISLAND);

    // The island's chunk arrives last, after the chunks around it have
    // settled.
    let _holding = Hold::exclusive();
    let held = IVec3::new(0, 2, 0);
    let hold = Hold::new(held);
    let mut app = app(LampWorld {
        hold: Some(hold.clone()),
        ..island_world(&fill)
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

    // Never brighter than a reference knowing every position whose cells
    // could light the chunks: those within sky openness's rays of the 3×3×3
    // around each.
    let wide: Vec<IVec3> = AROUND_ISLAND
        .iter()
        .flat_map(|at| {
            (-5..=5).flat_map(move |z| {
                (-1..=5)
                    .flat_map(move |y| (-5..=5).map(move |x| *at + IVec3::new(x, y, z)))
            })
        })
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();
    let reference = reference_of(
        &AROUND_ISLAND,
        &wide,
        island_on_boundaries,
        &filled_with(&fill),
    );
    for (p, sky) in sky_of(&mut app, &AROUND_ISLAND) {
        assert!(
            sky <= reference[&p] as f32,
            "{sky} at {p} over {}",
            reference[&p]
        );
    }
}

fn light_reads_the_same_after_reset_light_rebuilds_the_record_and_the_field(
    fill: SkyFill,
) {
    let mut app = settled(island_world(&fill));
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

fn a_cell_a_model_marks_as_direct_sky_reads_15(fill: SkyFill) {
    // The cell at the bottom of the open sky beside the island, at x = -64
    // to -56, which every model marks as direct sky.
    let mut app = settled(island_world(&fill));
    // Under openness, a cell straight under a high narrow island too.
    let mut open =
        matches!(fill, SkyFill::Openness(_)).then(|| settled(high_island_world(&fill)));
    let mut cells = vec![(&mut app, IVec3::new(-64, 0, 0))];
    if let Some(open) = open.as_mut() {
        cells.push((open, IVec3::new(8, 0, 8)));
    }
    for (app, min) in cells {
        for z in 0..8 {
            for y in 0..8 {
                for x in 0..8 {
                    let p = min + IVec3::new(x, y, z);
                    let levels = levels(app, p);
                    assert_eq!(levels.sky, 15.0, "at {p}");
                    assert!(levels.direct_sky());
                }
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

fn a_minimal_world_reads_the_same_light_as_a_shown_one_through_streaming_fill_and_a_settings_change(
    fill: SkyFill,
) {
    let world = island_world(&fill);
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
        light: Some(filled_with(&changed(&fill))),
        ..world
    };
    minimal_and_shown(changed, &mut apps, &AROUND_ISLAND, "after a change of fill");
}

fn turning_sky_fill_off_and_on_at_runtime_relights_to_match_the_reference(fill: SkyFill) {
    let chunks = [IVec3::ZERO];
    let mut light = filled_with(&fill);
    let mut app = settled(island_world(&fill));
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

    set_light(&mut app, filled_with(&fill));
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
        ..island_world(&paths())
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

// --- light_at where no face reads -------------------------------------------

/// The sky light `light_at` reads at every voxel of the chunk positions
/// `chunks`.
fn sky_read(app: &mut App, chunks: &[IVec3]) -> Vec<(IVec3, f32)> {
    let chunks = chunks.to_vec();
    app.world_mut()
        .run_system_once(move |voxel_world: VoxelWorld<LampWorld>| {
            chunks
                .iter()
                .flat_map(|chunk| {
                    super::voxel_light::origin_voxels()
                        .map(move |local| *chunk * 32 + local)
                })
                .map(|p| {
                    let levels = voxel_world
                        .light_at(p)
                        .unwrap_or_else(|| panic!("no light at {p}"));
                    (p, levels.sky)
                })
                .collect::<Vec<_>>()
        })
        .unwrap()
}

/// The most `read` differs in `chunks` from the reference's sky light with
/// every chunk taking fill, with `known` known to sky fill.
fn worst_difference(
    read: &[(IVec3, f32)],
    chunks: &[IVec3],
    known: &[IVec3],
    scene: impl Fn(IVec3) -> WorldVoxel,
    light: &LightPropagation,
) -> f32 {
    let reference = fill_reference_everywhere(
        chunks,
        known,
        light.sky_reach as u32,
        light.shape,
        &light.sky,
        light.sky_fill.as_ref().expect("sky fill is on"),
        scene,
    );
    read.iter()
        .map(|(p, sky)| (sky - reference[p] as f32).abs())
        .fold(0.0, f32::max)
}

/// Stone below y = 0 and a slab 8 voxels thick at y = 104, over the chunk of
/// air at (0, 1, 0), which no solid in its padding touches.
fn high_slab(p: IVec3) -> WorldVoxel {
    slab_at(104)(p)
}

const HIGH_SLAB_AIR: IVec3 = IVec3::Y;

fn high_slab_light() -> LightPropagation {
    LightPropagation {
        sky: Sky::Ceiling { y: 140 },
        ..default()
    }
}

fn high_slab_world(hints: Option<super::voxel_light::Hints>) -> LampWorld {
    LampWorld {
        scene: Arc::new(high_slab),
        light: Some(high_slab_light()),
        spawning_distance: 3,
        hints,
        ..default()
    }
}

/// Moves the camera into the middle of the chunk position `chunk`.
fn camera_at(app: &mut App, chunk: IVec3) {
    app.update();
    let middle = (chunk * 32 + 16).as_vec3();
    super::super::move_cameras::<LampWorld>(
        app,
        Transform::from_translation(middle).looking_at(middle + Vec3::NEG_Z, Vec3::Y),
    );
}

#[test]
fn a_chunk_of_air_taking_no_sky_fill_reads_close_to_the_reference_through_changes_of_reach_and_fill(
) {
    let mut light = high_slab_light();
    let mut app = app(high_slab_world(None));
    camera_at(&mut app, HIGH_SLAB_AIR);
    settle(&mut app, "the world to settle");
    assert!(spawned(&mut app).contains(&HIGH_SLAB_AIR));
    let chunks = [HIGH_SLAB_AIR];
    let check = |app: &mut App, light: &LightPropagation, when: &str| {
        let read = sky_read(app, &chunks);
        let known = spawned(app);
        let worst = worst_difference(&read, &chunks, &known, high_slab, light);
        assert!(worst <= 2.0, "off by {worst} {when}");
        assert!(
            read.iter().any(|(_, sky)| *sky > 0.0 && *sky < 15.0),
            "some of the chunk is shaded {when}"
        );
    };
    check(&mut app, &light, "after streaming");

    light.sky_reach = 16;
    set_light(&mut app, light.clone());
    settle(&mut app, "light to settle after a change of sky reach");
    check(&mut app, &light, "at sky reach 16");

    light.sky_fill = Some(SkyFill::Paths(PathFill {
        opening_power: 1.0,
        beside_open_sky: 10.0,
        ..default()
    }));
    set_light(&mut app, light.clone());
    settle(&mut app, "light to settle after a change of sky fill");
    check(&mut app, &light, "after the change of fill");
}

#[test]
fn hinted_air_holding_no_light_entry_reads_close_to_the_reference() {
    let hints: super::voxel_light::Hints = Arc::new(|position: IVec3| {
        if position.y >= 0 && position.y != 3 {
            FillHint::Uniform(WorldVoxel::Air)
        } else {
            FillHint::Unknown
        }
    });
    // The camera starts two chunks away, so the hinted air under the slab
    // never spawns.
    let mut app = super::super::meshing_app();
    app.add_plugins(VoxelWorldPlugin::with_config(high_slab_world(Some(hints))))
        .add_systems(Startup, |mut commands: Commands| {
            commands.spawn((
                Camera3d::default(),
                Transform::from_xyz(80.0, 48.0, 16.0),
                VoxelWorldCamera::<LampWorld>::default(),
            ));
        });
    settle(&mut app, "the world to settle");
    assert!(!spawned(&mut app).contains(&HIGH_SLAB_AIR));
    let entry = app
        .world()
        .resource::<LightSettings<LampWorld>>()
        .view
        .read()
        .contains_key(&HIGH_SLAB_AIR);
    assert!(!entry, "no light entry under the slab");

    let chunks = [HIGH_SLAB_AIR];
    let read = sky_read(&mut app, &chunks);
    let known: Vec<IVec3> = spawned(&mut app)
        .into_iter()
        .chain((-1..=1).flat_map(|x| {
            (-1..=1).flat_map(move |z| (0..3).map(move |y| IVec3::new(x, y, z)))
        }))
        .collect();
    let worst = worst_difference(&read, &chunks, &known, high_slab, &high_slab_light());
    assert!(worst <= 2.0, "off by {worst}");
    assert!(read.iter().any(|(_, sky)| *sky > 0.0 && *sky < 15.0));
}

/// Runs each test of the suite every model of sky fill must pass, once under
/// paths and once under sky openness, each with its defaults.
macro_rules! once_per_model {
    ($($test:ident),* $(,)?) => {
        mod paths {
            $(
                #[test]
                fn $test() {
                    super::$test(super::paths());
                }
            )*
        }

        mod openness {
            $(
                #[test]
                fn $test() {
                    super::$test(super::openness());
                }
            )*
        }
    };
}

once_per_model!(
    after_streaming_settles_at_256_levels_every_fully_loaded_chunk_matches_the_fill_reference,
    a_1x1_shaft_lets_no_fill_through_and_a_sealed_pocket_stays_black,
    faces_on_chunk_boundaries_under_a_floating_island_match_the_reference,
    faces_on_chunk_boundaries_under_a_low_slab_match_the_reference,
    arrival_order_changes_no_settled_fill_and_no_reading_is_brighter_than_the_reference,
    light_reads_the_same_after_reset_light_rebuilds_the_record_and_the_field,
    a_cell_a_model_marks_as_direct_sky_reads_15,
    a_minimal_world_reads_the_same_light_as_a_shown_one_through_streaming_fill_and_a_settings_change,
    turning_sky_fill_off_and_on_at_runtime_relights_to_match_the_reference,
);
