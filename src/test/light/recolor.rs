//! Recolouring an emissive material while the world runs
//!
//! This module changes the light colour of an emissive material through
//! `ResMut` of the world config, as a system of a game would. The tests check
//! that only the chunks holding an emissive voxel of the material, and the 26
//! chunks around each of them, relight. Light must then settle on the
//! reference for the new light colours.
//!
//! Sky light is off in every test, so only voxel light is compared.

use std::sync::Arc;

use bevy::ecs::system::RunSystemOnce;
use bevy::prelude::*;

use super::edits::{relights_started, settle};
use super::reference::Reference;
use super::sky_light::{chunk_light, light_is_settled, settled, spawned};
use super::voxel_light::{app, light_at, lit, Hold, LampWorld};
use crate::prelude::*;

const RED_LAMP: u8 = 6;
const GREEN_LAMP: u8 = 7;

/// The position of the west lamp, in the chunk at the origin. Its light
/// crosses into the chunk east of it.
const WEST: IVec3 = IVec3::new(24, 16, 16);
/// The position of the green east lamp, two chunks east of the origin. Its
/// light and the light of the west lamp both reach the chunk between them.
const EAST: IVec3 = IVec3::new(72, 16, 16);

/// How far each component of a light colour read back may be from the
/// reference.
///
/// The packing holds a component in steps of 1/127, and the extra 1e-5 allows
/// for rounding.
const STEP: f32 = 1.0 / 127.0 + 1e-5;

fn scene(position: IVec3) -> WorldVoxel {
    match position {
        WEST => WorldVoxel::Solid(RED_LAMP),
        EAST => WorldVoxel::Solid(GREEN_LAMP),
        _ => WorldVoxel::Air,
    }
}

fn lamp(color: LinearRgba) -> Emission {
    Emission::level(15.0).with_color(color)
}

/// Returns the world of the scene, with the west lamp giving off `west` and
/// the east lamp green.
fn world(west: LinearRgba) -> LampWorld {
    LampWorld {
        scene: Arc::new(scene),
        light: Some(lit(LightShape::Round)),
        emissive: vec![
            (RED_LAMP, lamp(west)),
            (GREEN_LAMP, lamp(LinearRgba::GREEN)),
        ],
        spawning_distance: 3,
        ..default()
    }
}

/// Sets the light colour of the west lamp through `ResMut` of the config, as
/// a system of a game would.
fn recolor(app: &mut App, color: impl Into<Color> + Send + 'static) {
    let color = color.into();
    app.world_mut()
        .run_system_once(move |mut world: ResMut<LampWorld>| {
            world.emissive[0].1 = lamp(LinearRgba::WHITE).with_color(color);
        })
        .unwrap();
}

/// Returns the chunk position at the origin, where the camera is, and the 26
/// positions around it.
fn around_origin() -> Vec<IVec3> {
    (-1..=1)
        .flat_map(|z| {
            (-1..=1).flat_map(move |y| (-1..=1).map(move |x| IVec3::new(x, y, z)))
        })
        .collect()
}

/// Fails unless every voxel of each spawned chunk whose light box is loaded
/// reads the reference light for the current emissive list of the world.
///
/// The voxel light level must match the reference, and each component of the
/// light colour must lie within [`STEP`] of it. The chunk east of the origin
/// must be among the chunks checked.
fn matches_reference(app: &mut App, when: &str) {
    let emissive = app.world().resource::<LampWorld>().emissive.clone();
    let reference = Reference::colored_voxel_light_in_steps(
        15,
        IVec3::new(-64, -48, -48),
        IVec3::new(128, 80, 80),
        14,
        LightShape::Round,
        scene,
        |material| {
            emissive
                .iter()
                .find(|(emissive, _)| *emissive == material)
                .map(|(_, emission)| {
                    let level = emission.whole_level() as f32;
                    (level, emission.color().to_f32_array_no_alpha())
                })
        },
    );
    let loaded = spawned(app);
    let boxed: Vec<IVec3> = loaded
        .iter()
        .copied()
        .filter(|chunk| {
            around_origin()
                .iter()
                .all(|offset| loaded.contains(&(*chunk + *offset)))
        })
        .collect();
    assert!(
        boxed.contains(&IVec3::X),
        "{when}: the chunk east of the origin isn't boxed"
    );
    for chunk in boxed {
        let light = chunk_light(app, chunk).expect("the chunk's light");
        for (p, levels) in light {
            let level = reference.read(reference.voxel_at(p));
            let color = reference.color_at(p);
            assert!(
                (levels.voxel - level).abs() < 1e-4,
                "{when}: {levels:?} at {p}: {level}"
            );
            let read = levels.voxel_color.to_f32_array_no_alpha();
            let close = read.iter().zip(color).all(|(a, b)| (a - b).abs() <= STEP);
            assert!(close, "{when}: {read:?} at {p}: {color:?}");
        }
    }
}

/// A voxel that the light of the west lamp reaches, and the light of the east
/// lamp doesn't.
const WEST_LIT: IVec3 = IVec3::new(18, 16, 16);

#[test]
fn recolouring_a_material_relights_only_the_chunks_whose_light_box_holds_it() {
    let mut app = settled(world(LinearRgba::RED));
    let relights = relights_started(&app);
    recolor(&mut app, LinearRgba::BLUE);
    settle(&mut app, "light to settle after the recolour");
    // The west lamp is in the chunk at the origin, so that chunk and the 26
    // chunks around it relight. Nothing else relights. The chunk of the green
    // lamp keeps its light, although the chunk between the lamps holds both
    // light colours.
    let boxed = spawned(&mut app)
        .into_iter()
        .filter(|chunk| (*chunk).abs().max_element() <= 1)
        .count() as u64;
    assert_eq!(relights_started(&app) - relights, boxed);
    assert_eq!(
        light_at(&mut app, WEST_LIT).unwrap().voxel_color,
        LinearRgba::BLUE
    );
    matches_reference(&mut app, "after the recolour");
}

#[test]
fn a_recolour_that_scales_to_the_same_light_colour_relights_nothing() {
    let mut app = settled(world(LinearRgba::RED));
    let relights = relights_started(&app);
    recolor(&mut app, Color::srgb(0.5, 0.0, 0.0));
    for _ in 0..5 {
        app.update();
        assert!(light_is_settled(&mut app));
    }
    assert_eq!(relights_started(&app), relights);
}

#[test]
fn old_light_holds_and_light_is_unsettled_until_the_recolours_light_lands() {
    let mut app = settled(world(LinearRgba::RED));
    recolor(&mut app, LinearRgba::BLUE);
    app.update();
    assert!(!light_is_settled(&mut app));
    assert_eq!(
        light_at(&mut app, WEST_LIT).unwrap().voxel_color,
        LinearRgba::RED
    );
    settle(&mut app, "light to settle after the recolour");
    assert!(light_is_settled(&mut app));
    assert_eq!(
        light_at(&mut app, WEST_LIT).unwrap().voxel_color,
        LinearRgba::BLUE
    );
}

#[test]
fn a_chunk_task_solved_under_the_old_colours_relights_as_it_lands() {
    let _holding = Hold::exclusive();
    let hold = Hold::new(IVec3::X);
    let mut app = app(LampWorld {
        hold: Some(hold.clone()),
        ..world(LinearRgba::RED)
    });
    // Wait until everything but the held chunk has settled, and the task of
    // the held chunk waits on a lookup in its padding.
    super::super::update_until(&mut app, "all but the held chunk to settle", |app| {
        super::sky_light::settled_but(app, Some(IVec3::X))
            && hold.waited.load(std::sync::atomic::Ordering::Acquire) > 0
    });
    recolor(&mut app, LinearRgba::BLUE);
    app.update();
    hold.release();
    settle(&mut app, "light to settle after the held chunk lands");
    // The west lamp alone lights this voxel of the held chunk, so the voxel
    // must show the new blue.
    let east_lit = IVec3::new(33, 16, 16);
    let color = light_at(&mut app, east_lit).unwrap().voxel_color;
    assert!(color.blue > 0.9 && color.red < 0.01, "{color:?}");
    matches_reference(&mut app, "after the held chunk lands");
}

#[test]
fn a_world_going_from_white_to_coloured_and_back_settles_on_the_reference() {
    let mut app = app(LampWorld {
        emissive: vec![
            (RED_LAMP, lamp(LinearRgba::WHITE)),
            (GREEN_LAMP, lamp(LinearRgba::WHITE)),
        ],
        ..world(LinearRgba::WHITE)
    });
    settle(&mut app, "the world to stream in");
    matches_reference(&mut app, "all white");
    for (index, color) in [LinearRgba::RED, LinearRgba::WHITE].into_iter().enumerate() {
        app.world_mut()
            .run_system_once(move |mut world: ResMut<LampWorld>| {
                for (_, emission) in &mut world.emissive {
                    *emission = lamp(color);
                }
            })
            .unwrap();
        let relights = relights_started(&app);
        settle(&mut app, "light to settle after the recolour");
        assert!(
            relights_started(&app) > relights,
            "change {index} relit nothing"
        );
        matches_reference(&mut app, &format!("after change {index}"));
    }
}

#[test]
fn a_recolours_landings_count_as_changing_light_colours_alone() {
    use crate::light_bench::light_stats;

    let mut app = settled(world(LinearRgba::RED));
    let before = light_stats::<LampWorld>(app.world()).unwrap();
    recolor(&mut app, LinearRgba::BLUE);
    settle(&mut app, "light to settle after the recolour");
    let after = light_stats::<LampWorld>(app.world()).unwrap();
    assert_eq!(after.landed_changing_levels, before.landed_changing_levels);
    assert!(
        after.landed_changing_colors > before.landed_changing_colors,
        "no landing changed a light colour"
    );
}

#[test]
fn the_fresh_solve_check_finds_a_chunk_holding_another_light_colour() {
    use crate::light::color::{pack, ChunkColor};
    use crate::light::solver::ChunkLight;
    use crate::light::view::in_world;
    use crate::light::LightSettings;
    use crate::light_bench::fresh_solve_check;

    let mut app = settled(world(LinearRgba::RED));
    assert_eq!(fresh_solve_check::<LampWorld>(app.world()).stale, vec![]);
    // The red west lamp alone lights the chunk at the origin. `hold` replaces
    // the light colour that the world view holds for that chunk.
    let hold = |app: &mut App, color: LinearRgba| {
        let settings = app.world().resource::<LightSettings<LampWorld>>();
        let mut view = settings.view.write();
        let chunk = view.chunks.get_mut(&in_world(IVec3::ZERO)).unwrap();
        let light = &chunk.light;
        chunk.light = Arc::new(ChunkLight {
            inputs: light.inputs.clone(),
            sky: light.sky.clone(),
            voxel: light.voxel.clone(),
            color: ChunkColor::One(pack(color)),
            borders: light.borders.clone(),
            precision: light.precision,
        });
    };
    hold(&mut app, LinearRgba::rgb(1.0, 0.5, 0.0));
    assert_eq!(
        fresh_solve_check::<LampWorld>(app.world()).stale,
        vec![(IVec3::ZERO, false)]
    );
    // A colour one step of the packing away from red still counts as the same
    // light colour.
    hold(&mut app, LinearRgba::rgb(1.0, 0.004, 0.0));
    assert_eq!(fresh_solve_check::<LampWorld>(app.world()).stale, vec![]);
}

#[test]
fn a_recolour_counts_as_a_visible_change_of_light() {
    use crate::light_bench::{change_counts, count_changes};

    // A floor lies under the west lamp, so the light of the lamp shows on
    // faces.
    let mut app = app(LampWorld {
        scene: Arc::new(|position: IVec3| {
            if position.y < 12 {
                WorldVoxel::Solid(0)
            } else {
                scene(position)
            }
        }),
        ..world(LinearRgba::RED)
    });
    count_changes::<LampWorld>(app.world_mut());
    super::super::update_until(&mut app, "the world to settle", |app| {
        super::sky_light::settled_but(app, None)
    });
    let before = change_counts::<LampWorld>(app.world()).unwrap();
    recolor(&mut app, LinearRgba::BLUE);
    settle(&mut app, "light to settle after the recolour");
    let after = change_counts::<LampWorld>(app.world()).unwrap();
    let visible = |counts: &crate::light_bench::ChangeCounts| {
        counts.visible_from_arrival
            + counts.visible_from_column
            + counts.visible_from_other
    };
    assert!(
        visible(&after) > visible(&before),
        "the recolour changed no visible light"
    );
    assert!(after.visible_with_shells > before.visible_with_shells);
}
