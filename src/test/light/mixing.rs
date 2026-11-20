//! Light colours mixing where they meet
//!
//! This module streams scenes of lamps of several light colours at LOD level
//! 0, and reads their light back through `light_at`. Most tests check the
//! light of every chunk whose light box is loaded against the coloured
//! reference flood of [`Reference`]. The others check how the world view
//! stores the light colours of a chunk. The tests of coloured light through
//! arrivals and edits build on its `Scene` and its checks.
//!
//! Sky light is off in every test, so only voxel light is compared.

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

pub(super) const RED_LAMP: u8 = 6;
pub(super) const GREEN_LAMP: u8 = 7;
pub(super) const BLUE_LAMP: u8 = 8;

/// The emissive materials of the scenes, each with its declared level and
/// light colour.
///
/// `LAMP` and `DIM_LAMP` are white, at levels 15 and 8. `RED_LAMP`,
/// `GREEN_LAMP` and `BLUE_LAMP` are all at level 15.
const EMISSION: [(u8, f32, LinearRgba); 5] = [
    (LAMP, 15.0, LinearRgba::WHITE),
    (DIM_LAMP, 8.0, LinearRgba::WHITE),
    (RED_LAMP, 15.0, LinearRgba::RED),
    (GREEN_LAMP, 15.0, LinearRgba::GREEN),
    (BLUE_LAMP, 15.0, LinearRgba::BLUE),
];

/// How far each component of a light colour read back may be from the
/// reference.
///
/// The packing holds a component in steps of 1/127, and the extra 1e-5 allows
/// for rounding.
pub(super) const STEP: f32 = 1.0 / 127.0 + 1e-5;

pub(super) fn emissive() -> Vec<(u8, Emission)> {
    EMISSION
        .iter()
        .map(|(material, level, color)| {
            (*material, Emission::level(*level).with_color(*color))
        })
        .collect()
}

/// A scene of lamps in air, with stone wherever `wall` returns true.
///
/// Each voxel in `edits` replaces whatever the scene holds at its position.
/// A later edit wins over an earlier one at the same position.
#[derive(Clone)]
pub(super) struct Scene {
    pub lamps: Vec<(IVec3, u8)>,
    pub wall: fn(IVec3) -> bool,
    pub edits: Vec<(IVec3, WorldVoxel)>,
}

impl Scene {
    pub fn in_air(lamps: Vec<(IVec3, u8)>) -> Self {
        Self {
            lamps,
            wall: |_| false,
            edits: Vec::new(),
        }
    }

    /// Returns a copy of this scene with `edits` set over it.
    pub fn edited(&self, edits: &[(IVec3, WorldVoxel)]) -> Self {
        let mut scene = self.clone();
        scene.edits.extend_from_slice(edits);
        scene
    }

    pub fn voxel_at(&self, position: IVec3) -> WorldVoxel {
        if let Some((_, voxel)) = self.edits.iter().rev().find(|(p, _)| *p == position) {
            return *voxel;
        }
        match self.lamps.iter().find(|(lamp, _)| *lamp == position) {
            Some((_, material)) => WorldVoxel::Solid(*material),
            None if (self.wall)(position) => WorldVoxel::Solid(STONE),
            None => WorldVoxel::Air,
        }
    }

    pub fn world(
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

    /// Returns the reference light of the scene, in a box that reaches far
    /// enough past the lamps to hold every voxel their light reaches.
    pub fn reference(
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

pub(super) fn two_colours_meeting(_reach: u8) -> Scene {
    Scene::in_air(vec![
        (IVec3::new(8, 16, 16), RED_LAMP),
        (IVec3::new(30, 18, 16), GREEN_LAMP),
    ])
}

pub(super) fn three_primaries_meeting(_reach: u8) -> Scene {
    Scene::in_air(vec![
        (IVec3::new(6, 10, 16), RED_LAMP),
        (IVec3::new(26, 10, 16), GREEN_LAMP),
        (IVec3::new(16, 27, 14), BLUE_LAMP),
    ])
}

pub(super) fn a_white_pool_ending_inside_a_red_one(_reach: u8) -> Scene {
    Scene::in_air(vec![
        (IVec3::new(16, 16, 16), RED_LAMP),
        (IVec3::new(21, 16, 16), DIM_LAMP),
    ])
}

/// Returns a wall filling the plane x = -1, with a red lamp west of it and a
/// green lamp east of it.
///
/// The green lamp is in the chunk at the origin. The light box of that chunk
/// holds both lamps, but the wall keeps the red light out of the chunk.
pub(super) fn a_colour_walls_keep_from_a_chunk(_reach: u8) -> Scene {
    Scene {
        lamps: vec![
            (IVec3::new(-6, 16, 16), RED_LAMP),
            (IVec3::new(10, 16, 16), GREEN_LAMP),
        ],
        wall: |position| position.x == -1,
        edits: Vec::new(),
    }
}

/// The voxel that each lamp of [`every_colour_at_level_1`] lights at level 1.
const AT_LEVEL_1: IVec3 = IVec3::new(16, 16, 16);

/// Returns a red lamp and a green lamp, `reach` voxels from [`AT_LEVEL_1`] on
/// either side of it.
pub(super) fn every_colour_at_level_1(reach: u8) -> Scene {
    let reach = reach as i32;
    Scene::in_air(vec![
        (AT_LEVEL_1 - IVec3::X * reach, RED_LAMP),
        (AT_LEVEL_1 + IVec3::X * reach, GREEN_LAMP),
    ])
}

/// Returns the chunk position at the origin, where the camera is, and the 26
/// positions around it.
pub(super) fn around_origin() -> Vec<IVec3> {
    (-1..=1)
        .flat_map(|z| {
            (-1..=1).flat_map(move |y| (-1..=1).map(move |x| IVec3::new(x, y, z)))
        })
        .collect()
}

/// Returns the spawned level-0 chunks whose whole light box is loaded.
pub(super) fn boxed(app: &mut App) -> Vec<IVec3> {
    let spawned: Vec<IVec3> = chunk_keys::<LampWorld>(app)
        .into_iter()
        .filter(|key| key.lod_level == 0)
        .map(|key| key.position)
        .collect();
    spawned
        .iter()
        .copied()
        .filter(|chunk| {
            around_origin()
                .iter()
                .all(|offset| spawned.contains(&(*chunk + *offset)))
        })
        .collect()
}

pub(super) fn close(a: [f32; 3], b: [f32; 3], within: f32) -> bool {
    a.iter().zip(b).all(|(a, b)| (a - b).abs() <= within)
}

/// Streams `scene` under each voxel reach, shape and precision until it
/// settles, then checks it with [`assert_on_the_reference`].
///
/// The world streams far enough that every chunk around the origin has its
/// whole light box loaded. Those chunks are checked too, so the check covers
/// the corners of the light box of the chunk at the origin.
fn settles_on_the_reference(scene: fn(u8) -> Scene) {
    for reach in [14, 16] {
        for shape in [LightShape::Round, LightShape::Diamond] {
            for precision in [LightPrecision::Steps16, LightPrecision::Steps256] {
                let scene = scene(reach);
                let mut app = settled(LampWorld {
                    spawning_distance: 4,
                    ..scene.world(reach, shape, precision)
                });
                let checked = boxed(&mut app);
                let unchecked: Vec<IVec3> = around_origin()
                    .into_iter()
                    .filter(|chunk| !checked.contains(chunk))
                    .collect();
                assert!(unchecked.is_empty(), "unchecked: {unchecked:?}");
                assert_on_the_reference(&mut app, &scene, reach, shape, precision);
            }
        }
    }
}

/// Fails unless every voxel of each chunk whose light box is loaded reads the
/// reference light of `scene`.
///
/// The voxel light level must match the reference, and each component of the
/// light colour must lie within [`STEP`] of it. A voxel outside the reference
/// box must be unlit and white. More than 1000 voxels must be lit, so an empty
/// scene can't pass.
pub(super) fn assert_on_the_reference(
    app: &mut App,
    scene: &Scene,
    reach: u8,
    shape: LightShape,
    precision: LightPrecision,
) {
    let reference = scene.reference(reach, shape, precision);
    let checked = boxed(app);
    assert!(checked.contains(&IVec3::ZERO));
    let mut lit = 0;
    for chunk in checked {
        let light = chunk_light(app, chunk).expect("the chunk's light");
        for (p, levels) in light {
            let (level, color) = if reference.holds(p) {
                (reference.read(reference.voxel_at(p)), reference.color_at(p))
            } else {
                (0.0, [1.0; 3])
            };
            let at = format!("at {p}, reach {reach}, {shape:?}, {precision:?}");
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
    // A white lamp stands in the chunk at the origin, and a red lamp two chunks
    // east. Red light reaches neither the chunk at the origin nor the chunk
    // between them.
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
    // The light box of the chunk holds both lamps whenever the chunk solves,
    // so every solve makes one colour run for each lamp.
    assert!(color_runs(&app) >= 2);
    // This voxel is too far from either lamp for their light to reach it.
    let dark = IVec3::new(16, 16, 0);
    let levels = light_at(&mut app, dark).unwrap();
    assert_eq!(levels.voxel, 0.0);
    assert_eq!(levels.voxel_color, LinearRgba::WHITE);
    assert_eq!(packed[voxel_index(16, 16, 0)], WHITE);
    // Between the lamps, red and green light mix to a yellow. An emissive voxel
    // reads its own light colour.
    let between = light_at(&mut app, IVec3::new(19, 17, 16)).unwrap();
    assert!(between.voxel_color.red > 0.5 && between.voxel_color.green > 0.5);
    assert_eq!(
        light_at(&mut app, IVec3::new(8, 16, 16))
            .unwrap()
            .voxel_color,
        LinearRgba::RED
    );
}

/// Returns the counts of relights, solves and colour runs, which a look change
/// must leave alone.
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
