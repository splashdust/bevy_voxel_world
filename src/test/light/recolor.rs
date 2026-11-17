//! Tests of changing an emissive material's light colour while the world
//! runs, through `ResMut` of its config: only the chunks whose light box
//! holds an emissive voxel of the material relight, and light settles on the
//! reference for the new colours. Sky light is off in all of them.

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

/// A lamp in the chunk at the origin whose light crosses into the chunk east
/// of it.
const WEST: IVec3 = IVec3::new(24, 16, 16);
/// A green lamp in the chunk two east of the origin, whose light meets the
/// west lamp's in the chunk between.
const EAST: IVec3 = IVec3::new(72, 16, 16);

/// How far a light colour read back may be from the reference's: a step of
/// its packing.
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

/// The scene with the west lamp in `west` and the east lamp green.
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

/// Sets the west lamp's light colour through `ResMut`, as a game's system
/// would.
fn recolor(app: &mut App, color: impl Into<Color> + Send + 'static) {
    let color = color.into();
    app.world_mut()
        .run_system_once(move |mut world: ResMut<LampWorld>| {
            world.emissive[0].1 = lamp(LinearRgba::WHITE).with_color(color);
        })
        .unwrap();
}

/// The chunks around the origin, where the camera is.
fn around_origin() -> Vec<IVec3> {
    (-1..=1)
        .flat_map(|z| {
            (-1..=1).flat_map(move |y| (-1..=1).map(move |x| IVec3::new(x, y, z)))
        })
        .collect()
}

/// Fails unless every voxel of each spawned chunk whose light box is loaded
/// reads the reference light for the world's current emissive list: levels
/// exactly, light colours within a step of the packing.
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

/// Where the west lamp's light alone reaches.
const WEST_LIT: IVec3 = IVec3::new(18, 16, 16);

#[test]
fn recolouring_a_material_relights_only_the_chunks_whose_light_box_holds_it() {
    let mut app = settled(world(LinearRgba::RED));
    let relights = relights_started(&app);
    recolor(&mut app, LinearRgba::BLUE);
    settle(&mut app, "light to settle after the recolour");
    // The west lamp is in the chunk at the origin, so the chunks around it
    // relight, and nothing else: not the chunk of the green lamp, though
    // the chunk between the lamps mixes both colours.
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
    // Everything but the held chunk settles, its task waiting on its padding.
    super::super::update_until(&mut app, "all but the held chunk to settle", |app| {
        super::sky_light::settled_but(app, Some(IVec3::X))
            && hold.waited.load(std::sync::atomic::Ordering::Acquire) > 0
    });
    recolor(&mut app, LinearRgba::BLUE);
    app.update();
    hold.release();
    settle(&mut app, "light to settle after the held chunk lands");
    // The east chunk's own voxels lit by the west lamp alone.
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
