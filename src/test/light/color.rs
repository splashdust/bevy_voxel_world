//! Tests of light colour at LOD level 0, read through `light_at`. Sky light is
//! off in all of them.

use std::sync::Arc;

use bevy::color::palettes::css;
use bevy::prelude::*;

use super::edits::{set_voxels, settle};
use super::settings::warned;
use super::sky_light::{chunk_light, settled};
use super::voxel_light::{app, light_at, lit, LampWorld, LAMP};
use crate::light::color::ChunkColor;
use crate::light::LightSettings;
use crate::prelude::*;

const ORANGE_LAMP: u8 = 4;
const BLUE_LAMP: u8 = 5;

/// The middle voxel of the chunk at the origin.
const MIDDLE: IVec3 = IVec3::splat(16);

/// How far a light colour read back may be from the one declared: a step of
/// its packing.
const STEP: f32 = 1.0 / 127.0;

fn orange() -> LinearRgba {
    // `css::ORANGE` scaled so its brightest component is 1.
    let linear = LinearRgba::from(css::ORANGE);
    LinearRgba::rgb(1.0, linear.green / linear.red, linear.blue / linear.red)
}

fn close(a: LinearRgba, b: LinearRgba) -> bool {
    a.to_f32_array_no_alpha()
        .iter()
        .zip(b.to_f32_array_no_alpha())
        .all(|(a, b)| (a - b).abs() <= STEP)
}

/// A world of air with `lamps` in it, each of a material, with a white lamp
/// [`LAMP`], an orange one and a blue one, all at level 15.
fn lamps(lamps: Vec<(IVec3, u8)>) -> LampWorld {
    LampWorld {
        scene: Arc::new(move |position| {
            match lamps.iter().find(|(lamp, _)| *lamp == position) {
                Some((_, material)) => WorldVoxel::Solid(*material),
                None => WorldVoxel::Air,
            }
        }),
        light: Some(lit(LightShape::Round)),
        emissive: vec![
            (LAMP, Emission::level(15.0)),
            (ORANGE_LAMP, Emission::level(15.0).with_color(css::ORANGE)),
            (BLUE_LAMP, Emission::level(15.0).with_color(css::BLUE)),
        ],
        ..default()
    }
}

/// The light of every voxel of the chunk at the origin, once the world has
/// settled.
fn origin_light(world: LampWorld) -> Vec<(IVec3, LightLevels)> {
    let mut app = settled(world);
    chunk_light(&mut app, IVec3::ZERO).expect("the origin chunk's light")
}

#[test]
fn light_at_reads_an_orange_lamps_colour_in_its_pool_and_white_outside_it() {
    let light = origin_light(lamps(vec![(MIDDLE, ORANGE_LAMP)]));
    let (mut lit, mut dark) = (0, 0);
    for (p, levels) in light {
        if p == MIDDLE {
            assert_eq!(levels.voxel, 15.0);
            assert!(close(levels.voxel_color, orange()), "the lamp's own voxel");
        } else if levels.voxel > 0.0 {
            lit += 1;
            assert!(close(levels.voxel_color, orange()), "at {p}: {levels:?}");
        } else {
            dark += 1;
            assert_eq!(levels.voxel_color, LinearRgba::WHITE, "at {p}");
        }
    }
    assert!(lit > 0 && dark > 0);
}

#[test]
fn a_coloured_worlds_levels_match_a_white_worlds() {
    let scenes = [
        vec![(MIDDLE, ORANGE_LAMP)],
        vec![
            (MIDDLE, ORANGE_LAMP),
            (MIDDLE + IVec3::new(6, 2, 0), BLUE_LAMP),
        ],
        vec![(MIDDLE, ORANGE_LAMP), (IVec3::new(40, 16, 16), LAMP)],
        vec![
            (MIDDLE - IVec3::X * 5, ORANGE_LAMP),
            (MIDDLE + IVec3::new(4, 3, 0), BLUE_LAMP),
            (MIDDLE + IVec3::new(1, -5, 2), LAMP),
        ],
    ];
    for scene in scenes {
        let white = scene.iter().map(|(p, _)| (*p, LAMP)).collect();
        let colored = origin_light(lamps(scene.clone()));
        let white = origin_light(lamps(white));
        for ((p, colored), (_, white)) in colored.iter().zip(&white) {
            assert_eq!(
                (colored.sky, colored.voxel),
                (white.sky, white.voxel),
                "at {p} in {scene:?}"
            );
        }
    }
}

#[test]
fn a_world_whose_emissions_are_white_or_grey_holds_no_light_colour() {
    let mut world = lamps(vec![(MIDDLE, LAMP), (MIDDLE + IVec3::X * 40, ORANGE_LAMP)]);
    world.emissive = vec![
        (LAMP, Emission::level(15.0)),
        (
            ORANGE_LAMP,
            Emission::level(12.0).with_color(Color::srgb(0.3, 0.3, 0.3)),
        ),
    ];
    let mut app = settled(world);
    for chunk in [IVec3::ZERO, IVec3::X] {
        let light = chunk_light(&mut app, chunk).unwrap();
        assert!(light.iter().any(|(_, levels)| levels.voxel > 0.0));
        for (p, levels) in light {
            assert_eq!(levels.voxel_color, LinearRgba::WHITE, "at {p}");
        }
    }
    let settings = app.world().resource::<LightSettings<LampWorld>>();
    assert!(!settings.task.emission.is_colored());
    let view = settings.view.read();
    assert!(view
        .chunks
        .values()
        .all(|chunk| chunk.light.color == ChunkColor::White));
}

#[test]
fn half_red_gives_the_same_light_as_red_and_alpha_changes_nothing() {
    let red = |color: Color| {
        let mut world = lamps(vec![(MIDDLE, ORANGE_LAMP)]);
        world.emissive = vec![(ORANGE_LAMP, Emission::level(15.0).with_color(color))];
        origin_light(world)
    };
    let full = red(Color::srgb(1.0, 0.0, 0.0));
    assert_eq!(full, red(Color::srgb(0.5, 0.0, 0.0)));
    assert_eq!(full, red(Color::srgba(1.0, 0.0, 0.0, 0.2)));
    let lit = full.iter().find(|(p, _)| *p == MIDDLE + IVec3::X).unwrap();
    assert_eq!(lit.1.voxel_color, LinearRgba::RED);
}

#[test]
fn black_gives_no_light() {
    let mut world = lamps(vec![(MIDDLE, ORANGE_LAMP)]);
    world.emissive = vec![(ORANGE_LAMP, Emission::level(15.0).with_color(Color::BLACK))];
    for (p, levels) in origin_light(world) {
        assert_eq!(levels.voxel, 0.0, "at {p}");
        assert_eq!(levels.voxel_color, LinearRgba::WHITE, "at {p}");
    }
}

#[test]
fn a_negative_component_reads_as_0_with_one_warning() {
    super::settings::warnings();
    let negative = LinearRgba::rgb(-1.0, 0.5, 0.0);
    let emission = Emission::level(15.0).with_color(negative);
    let again = Emission::level(15.0).with_color(negative);
    assert_eq!(emission, again);
    assert_eq!(
        emission,
        Emission::level(15.0).with_color(LinearRgba::GREEN)
    );
    assert_eq!(warned("negative component"), 1);
}

#[test]
fn at_reach_14_256_levels_read_the_same_colours_as_16() {
    let scene = vec![(MIDDLE, ORANGE_LAMP)];
    let at = |precision| {
        let mut world = lamps(scene.clone());
        world.light = Some(LightPropagation {
            precision,
            ..lit(LightShape::Round)
        });
        origin_light(world)
    };
    let sixteen = at(LightPrecision::Steps16);
    let fine = at(LightPrecision::Steps256);
    for ((p, sixteen), (_, fine)) in sixteen.iter().zip(&fine) {
        assert_eq!(sixteen.voxel_color, fine.voxel_color, "at {p}");
    }
}

#[test]
fn an_edit_recolouring_a_lamp_recolours_its_pool() {
    // A lamp near the chunk's +X face, so its pool reaches the chunk beside.
    let lamp = IVec3::new(28, 16, 16);
    let mut app = settled(lamps(vec![(lamp, LAMP)]));
    let beside = lamp - IVec3::new(5, 0, 0);
    let across = IVec3::new(34, 16, 16);
    assert_eq!(
        light_at(&mut app, beside).unwrap().voxel_color,
        LinearRgba::WHITE
    );

    set_voxels(&mut app, &[(lamp, WorldVoxel::Solid(ORANGE_LAMP))]);
    settle(&mut app, "the recolour to land");
    for p in [beside, across] {
        let levels = light_at(&mut app, p).unwrap();
        assert!(levels.voxel > 0.0);
        assert!(close(levels.voxel_color, orange()), "at {p}: {levels:?}");
    }

    set_voxels(&mut app, &[(lamp, WorldVoxel::Solid(LAMP))]);
    settle(&mut app, "the undo to land");
    for p in [beside, across] {
        assert_eq!(
            light_at(&mut app, p).unwrap().voxel_color,
            LinearRgba::WHITE
        );
    }
}

/// The chunk position the camera stands in for [`light_entry_world`].
const CAMERA_CHUNK: IVec3 = IVec3::new(4, 0, 0);

/// The orange lamp of [`light_entry_world`].
const ENTRY_LAMP: IVec3 = IVec3::new(96, 16, 16);

/// An orange lamp on the -X face of the chunk position west of
/// [`CAMERA_CHUNK`], with every chunk position west of that hinted air, so the
/// lamp touches the hinted air beside it, which holds a light entry.
fn light_entry_world() -> LampWorld {
    LampWorld {
        hints: Some(Arc::new(|position: IVec3| {
            if position.x < 3 {
                FillHint::Uniform(WorldVoxel::Air)
            } else {
                FillHint::Unknown
            }
        })),
        ..lamps(vec![(ENTRY_LAMP, ORANGE_LAMP)])
    }
}

#[test]
fn a_light_entry_in_hinted_air_reads_its_colour_and_hinted_air_without_one_white() {
    let mut app = app(light_entry_world());
    app.update();
    let middle = (CAMERA_CHUNK * 32 + 16).as_vec3();
    super::super::move_cameras::<LampWorld>(
        &mut app,
        Transform::from_translation(middle).looking_at(middle + Vec3::NEG_Z, Vec3::Y),
    );
    settle(&mut app, "the world to stream in");
    let holds_entry = |app: &App, position: IVec3| {
        app.world()
            .resource::<LightSettings<LampWorld>>()
            .view
            .read()
            .get(&position)
            .is_some_and(|chunk| chunk.is_entry())
    };
    assert!(holds_entry(&app, IVec3::new(2, 0, 0)));

    let in_entry = light_at(&mut app, ENTRY_LAMP - IVec3::X * 3).unwrap();
    assert_eq!(in_entry.voxel, 12.0);
    assert!(close(in_entry.voxel_color, orange()), "{in_entry:?}");

    assert!(!holds_entry(&app, IVec3::X));
    let past = light_at(&mut app, IVec3::new(48, 16, 16)).unwrap();
    assert_eq!(past.voxel_color, LinearRgba::WHITE);
}
