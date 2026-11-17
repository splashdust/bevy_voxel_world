//! Tests of light colours mixing where they meet, at LOD level 0, read
//! through `light_at` and held against the coloured reference flood. Sky
//! light is off in all of them.

use std::sync::atomic::Ordering;
use std::sync::Arc;

use bevy::ecs::system::RunSystemOnce;
use bevy::prelude::*;

use super::super::chunk_keys;
use super::reference::Reference;
use super::sky_light::{chunk_light, settled};
use super::voxel_light::{light_at, lit, LampWorld, DIM_LAMP, LAMP, STONE};
use crate::light::color::{pack, voxel_index, ChunkColor, WHITE};
use crate::light::LightSettings;
use crate::light_bench::light_stats;
use crate::prelude::*;

const RED_LAMP: u8 = 6;
const GREEN_LAMP: u8 = 7;
const BLUE_LAMP: u8 = 8;

/// The declared level and light colour of each emissive material: white
/// lamps at 15 and 8, and a red, a green and a blue lamp at 15.
const EMISSION: [(u8, f32, LinearRgba); 5] = [
    (LAMP, 15.0, LinearRgba::WHITE),
    (DIM_LAMP, 8.0, LinearRgba::WHITE),
    (RED_LAMP, 15.0, LinearRgba::RED),
    (GREEN_LAMP, 15.0, LinearRgba::GREEN),
    (BLUE_LAMP, 15.0, LinearRgba::BLUE),
];

/// How far a light colour read back may be from the reference's: a step of
/// its packing.
const STEP: f32 = 1.0 / 127.0 + 1e-5;

fn emissive() -> Vec<(u8, Emission)> {
    EMISSION
        .iter()
        .map(|(material, level, color)| {
            (*material, Emission::level(*level).with_color(*color))
        })
        .collect()
}

/// Lamps in air, with stone wherever `wall` holds.
#[derive(Clone)]
struct Scene {
    lamps: Vec<(IVec3, u8)>,
    wall: fn(IVec3) -> bool,
}

impl Scene {
    fn in_air(lamps: Vec<(IVec3, u8)>) -> Self {
        Self {
            lamps,
            wall: |_| false,
        }
    }

    fn voxel_at(&self, position: IVec3) -> WorldVoxel {
        match self.lamps.iter().find(|(lamp, _)| *lamp == position) {
            Some((_, material)) => WorldVoxel::Solid(*material),
            None if (self.wall)(position) => WorldVoxel::Solid(STONE),
            None => WorldVoxel::Air,
        }
    }

    fn world(
        &self,
        reach: u8,
        shape: LightShape,
        precision: LightPrecision,
    ) -> LampWorld {
        let scene = self.clone();
        LampWorld {
            scene: Arc::new(move |position| scene.voxel_at(position)),
            light: Some(LightPropagation {
                voxel_reach: reach,
                precision,
                ..lit(shape)
            }),
            emissive: emissive(),
            ..default()
        }
    }

    /// The reference light, in a box reaching past the lamps far enough to
    /// hold every voxel their light reaches.
    fn reference(
        &self,
        reach: u8,
        shape: LightShape,
        precision: LightPrecision,
    ) -> Reference {
        let lamps = self.lamps.iter().map(|(lamp, _)| *lamp);
        let low = lamps.clone().reduce(IVec3::min).unwrap();
        let high = lamps.reduce(IVec3::max).unwrap();
        let margin = IVec3::splat(reach as i32 + 2);
        let top = match precision {
            LightPrecision::Steps16 => 15,
            LightPrecision::Steps256 => 255,
        };
        Reference::colored_voxel_light_in_steps(
            top,
            low - margin,
            high + margin + IVec3::ONE,
            reach as u32,
            shape,
            |position| self.voxel_at(position),
            |material| {
                EMISSION
                    .iter()
                    .find(|(emissive, _, _)| *emissive == material)
                    .map(|(_, level, color)| (*level, color.to_f32_array_no_alpha()))
            },
        )
    }
}

fn two_colours_meeting(_reach: u8) -> Scene {
    Scene::in_air(vec![
        (IVec3::new(8, 16, 16), RED_LAMP),
        (IVec3::new(30, 18, 16), GREEN_LAMP),
    ])
}

fn three_primaries_meeting(_reach: u8) -> Scene {
    Scene::in_air(vec![
        (IVec3::new(6, 10, 16), RED_LAMP),
        (IVec3::new(26, 10, 16), GREEN_LAMP),
        (IVec3::new(16, 27, 14), BLUE_LAMP),
    ])
}

fn a_white_pool_ending_inside_a_red_one(_reach: u8) -> Scene {
    Scene::in_air(vec![
        (IVec3::new(16, 16, 16), RED_LAMP),
        (IVec3::new(21, 16, 16), DIM_LAMP),
    ])
}

/// A red lamp west of a wall filling the plane x = -1, and a green lamp east
/// of it, in the chunk at the origin, whose light box holds both.
fn a_colour_walls_keep_from_a_chunk(_reach: u8) -> Scene {
    Scene {
        lamps: vec![
            (IVec3::new(-6, 16, 16), RED_LAMP),
            (IVec3::new(10, 16, 16), GREEN_LAMP),
        ],
        wall: |position| position.x == -1,
    }
}

/// The voxel [`every_colour_at_level_1`] lights at level 1 from each lamp.
const AT_LEVEL_1: IVec3 = IVec3::new(16, 16, 16);

/// A red and a green lamp `reach` either side of [`AT_LEVEL_1`].
fn every_colour_at_level_1(reach: u8) -> Scene {
    let reach = reach as i32;
    Scene::in_air(vec![
        (AT_LEVEL_1 - IVec3::X * reach, RED_LAMP),
        (AT_LEVEL_1 + IVec3::X * reach, GREEN_LAMP),
    ])
}

/// The chunks around the origin, where the camera is.
fn around_origin() -> Vec<IVec3> {
    (-1..=1)
        .flat_map(|z| {
            (-1..=1).flat_map(move |y| (-1..=1).map(move |x| IVec3::new(x, y, z)))
        })
        .collect()
}

/// The chunks around the origin whose whole light box is loaded.
fn boxed(app: &mut App) -> Vec<IVec3> {
    let spawned: Vec<IVec3> = chunk_keys::<LampWorld>(app)
        .into_iter()
        .filter(|key| key.lod_level == 0)
        .map(|key| key.position)
        .collect();
    around_origin()
        .into_iter()
        .filter(|chunk| {
            around_origin()
                .iter()
                .all(|offset| spawned.contains(&(*chunk + *offset)))
        })
        .collect()
}

fn close(a: [f32; 3], b: [f32; 3], within: f32) -> bool {
    a.iter().zip(b).all(|(a, b)| (a - b).abs() <= within)
}

/// Streams `scene` under each reach, shape and precision until it settles,
/// then holds every voxel of each chunk around the origin whose light box is
/// loaded against the reference: levels exactly, light colours within a step
/// of the packing.
fn settles_on_the_reference(scene: fn(u8) -> Scene) {
    for reach in [14, 16] {
        for shape in [LightShape::Round, LightShape::Diamond] {
            for precision in [LightPrecision::Steps16, LightPrecision::Steps256] {
                let scene = scene(reach);
                let mut app = settled(scene.world(reach, shape, precision));
                let reference = scene.reference(reach, shape, precision);
                let checked = boxed(&mut app);
                assert!(checked.contains(&IVec3::ZERO));
                let mut lit = 0;
                for chunk in checked {
                    let light = chunk_light(&mut app, chunk).expect("the chunk's light");
                    for (p, levels) in light {
                        let (level, color) = if reference.holds(p) {
                            (reference.read(reference.voxel_at(p)), reference.color_at(p))
                        } else {
                            (0.0, [1.0; 3])
                        };
                        let at =
                            format!("at {p}, reach {reach}, {shape:?}, {precision:?}");
                        assert!(
                            (levels.voxel - level).abs() < 1e-4,
                            "{levels:?} {at}: {level}"
                        );
                        let read = levels.voxel_color.to_f32_array_no_alpha();
                        assert!(close(read, color, STEP), "{read:?} {at}: {color:?}");
                        lit += (level > 0.0) as usize;
                    }
                }
                assert!(lit > 1000, "only {lit} voxels lit");
            }
        }
    }
}

#[test]
fn where_two_colours_meet_light_settles_on_the_reference() {
    settles_on_the_reference(two_colours_meeting);
}

#[test]
fn where_three_primaries_meet_light_settles_on_the_reference() {
    settles_on_the_reference(three_primaries_meeting);
}

#[test]
fn a_white_pool_ending_inside_a_red_one_settles_on_the_reference() {
    settles_on_the_reference(a_white_pool_ending_inside_a_red_one);
}

#[test]
fn a_colour_walls_keep_from_a_chunk_settles_on_the_reference() {
    settles_on_the_reference(a_colour_walls_keep_from_a_chunk);
}

#[test]
fn where_every_colour_is_at_level_1_light_settles_on_the_reference() {
    settles_on_the_reference(every_colour_at_level_1);
}

#[test]
fn where_every_colour_is_at_level_1_they_mix_evenly() {
    let scene = every_colour_at_level_1(14);
    let mut app = settled(scene.world(14, LightShape::Round, LightPrecision::Steps16));
    let levels = light_at(&mut app, AT_LEVEL_1).unwrap();
    assert_eq!(levels.voxel, 1.0);
    assert_eq!(levels.voxel_color, LinearRgba::rgb(1.0, 1.0, 0.0));
}

fn chunk_color(app: &App, chunk: IVec3) -> ChunkColor {
    app.world()
        .resource::<LightSettings<LampWorld>>()
        .view
        .read()
        .get(&chunk)
        .expect("the chunk's light")
        .light
        .color
        .clone()
}

fn color_runs(app: &App) -> u64 {
    app.world()
        .resource::<LightSettings<LampWorld>>()
        .counters()
        .color_runs
        .load(Ordering::Relaxed)
}

fn default_world(scene: Scene) -> LampWorld {
    scene.world(14, LightShape::Round, LightPrecision::Steps16)
}

#[test]
fn a_light_box_of_one_colour_runs_once_and_stores_one_colour() {
    let scene = Scene::in_air(vec![
        (IVec3::new(8, 16, 16), RED_LAMP),
        (IVec3::new(30, 18, 16), RED_LAMP),
    ]);
    let app = settled(default_world(scene));
    assert_eq!(color_runs(&app), 0);
    assert_eq!(
        chunk_color(&app, IVec3::ZERO),
        ChunkColor::One(pack(LinearRgba::RED))
    );
}

#[test]
fn a_light_box_whose_walls_keep_one_colour_out_stores_the_other() {
    let app = settled(default_world(a_colour_walls_keep_from_a_chunk(14)));
    assert_eq!(
        chunk_color(&app, IVec3::ZERO),
        ChunkColor::One(pack(LinearRgba::GREEN))
    );
    assert_eq!(
        chunk_color(&app, IVec3::NEG_X),
        ChunkColor::One(pack(LinearRgba::RED))
    );
}

#[test]
fn a_white_chunk_in_a_coloured_world_holds_no_light_colour() {
    // A white lamp in the chunk at the origin, and a red one two chunks east.
    // The chunk between holds both in its light box, and neither's light.
    let scene = Scene::in_air(vec![
        (IVec3::new(16, 16, 16), LAMP),
        (IVec3::new(80, 16, 16), RED_LAMP),
    ]);
    let mut app = settled(default_world(scene));
    assert!(light_at(&mut app, IVec3::new(20, 16, 16)).unwrap().voxel > 0.0);
    assert_eq!(chunk_color(&app, IVec3::ZERO), ChunkColor::White);
    assert_eq!(chunk_color(&app, IVec3::X), ChunkColor::White);
}

#[test]
fn a_mixed_chunk_holds_a_packed_light_colour_per_voxel_and_unlit_voxels_read_white() {
    let mut app = settled(default_world(two_colours_meeting(14)));
    let ChunkColor::Mixed(packed) = chunk_color(&app, IVec3::ZERO) else {
        panic!("the chunk at the origin isn't mixed");
    };
    assert_eq!(packed.len(), 32 * 32 * 32);
    // The light box holds both lamps whenever the chunk solves, so it runs
    // once for each.
    assert!(color_runs(&app) >= 2);
    // Too far from either lamp for their light.
    let dark = IVec3::new(16, 16, 0);
    let levels = light_at(&mut app, dark).unwrap();
    assert_eq!(levels.voxel, 0.0);
    assert_eq!(levels.voxel_color, LinearRgba::WHITE);
    assert_eq!(packed[voxel_index(16, 16, 0)], WHITE);
    // Between the lamps the colours mix to yellow-ish, and a lamp reads its
    // own colour.
    let between = light_at(&mut app, IVec3::new(19, 17, 16)).unwrap();
    assert!(between.voxel_color.red > 0.5 && between.voxel_color.green > 0.5);
    assert_eq!(
        light_at(&mut app, IVec3::new(8, 16, 16))
            .unwrap()
            .voxel_color,
        LinearRgba::RED
    );
}

/// What a look change mustn't touch: relights, solves and runs.
fn work_done(app: &App) -> [u64; 3] {
    let stats = light_stats::<LampWorld>(app.world()).unwrap();
    [stats.relights, stats.solves, color_runs(app)]
}

#[test]
fn changing_the_voxel_falloff_changes_no_light_colour_and_starts_no_relight() {
    let mut app = settled(default_world(three_primaries_meeting(14)));
    let before: Vec<_> = around_origin()
        .into_iter()
        .map(|chunk| chunk_light(&mut app, chunk))
        .collect();
    for falloff in [LevelCurve::linear(), LevelCurve::exponential(0.5)] {
        let work = work_done(&app);
        app.world_mut()
            .run_system_once(move |mut world: ResMut<LampWorld>| {
                world.light.as_mut().unwrap().look.voxel_falloff = falloff.clone();
            })
            .unwrap();
        for _ in 0..5 {
            app.update();
        }
        assert_eq!(work_done(&app), work);
        let after: Vec<_> = around_origin()
            .into_iter()
            .map(|chunk| chunk_light(&mut app, chunk))
            .collect();
        assert!(before == after);
    }
}
