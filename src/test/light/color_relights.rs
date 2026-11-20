//! Tests of coloured light through arrivals and edits: in hostile arrival
//! orders, and through edits beside emissive voxels of other colours, light
//! settles on the coloured reference. Sky light is off in all of them.

use std::time::{Duration, Instant};

use std::collections::HashMap;
use std::sync::Arc;

use bevy::ecs::system::RunSystemOnce;
use bevy::prelude::*;

use super::edits::{framed_update, paced_update, relights_started, set_voxels};
use super::mixing::{
    a_colour_walls_keep_from_a_chunk, a_white_pool_ending_inside_a_red_one,
    assert_on_the_reference, boxed, every_colour_at_level_1, three_primaries_meeting,
    two_colours_meeting, Scene, BLUE_LAMP, GREEN_LAMP, RED_LAMP,
};
use super::sky_light::{settled, settled_but};
use super::voxel_light::{app, light_at, Hold, LampWorld, LAMP, STONE};
use crate::chunk_map::ChunkMap;
use crate::grid::GridKey;
use crate::light::solver::ChunkLight;
use crate::light::LightSettings;
use crate::prelude::*;

/// Runs `app` until every chunk but `held` is Ready and light has settled,
/// for a few frames in a row.
fn settle_but(app: &mut App, held: Option<IVec3>, what: &str) {
    let deadline = Instant::now() + Duration::from_secs(60);
    let mut frames_done = 0;
    while frames_done < 3 {
        assert!(Instant::now() < deadline, "timed out waiting: {what}");
        app.update();
        frames_done = if settled_but(app, held) {
            frames_done + 1
        } else {
            0
        };
    }
}

/// Streams `scene`, holding back the chunk task of `held` until everything
/// else has settled, then holds the light around the origin against the
/// reference.
fn settles_with_held(
    scene: &Scene,
    held: IVec3,
    reach: u8,
    shape: LightShape,
    precision: LightPrecision,
) {
    let _holding = Hold::exclusive();
    let hold = Hold::new(held);
    let mut app = app(LampWorld {
        hold: Some(hold.clone()),
        spawning_distance: 3,
        ..scene.world(reach, shape, precision)
    });
    settle_but(&mut app, Some(held), "everything but the held chunk");
    hold.release();
    settle_but(&mut app, None, "the held chunk");
    let checked = boxed(&mut app);
    assert!(checked.contains(&IVec3::X) && checked.contains(&IVec3::NEG_X));
    assert_on_the_reference(&mut app, scene, reach, shape, precision);
}

#[test]
fn in_hostile_arrival_orders_coloured_light_settles_on_the_reference() {
    let scenes: [fn(u8) -> Scene; 5] = [
        two_colours_meeting,
        three_primaries_meeting,
        a_white_pool_ending_inside_a_red_one,
        a_colour_walls_keep_from_a_chunk,
        every_colour_at_level_1,
    ];
    for scene in scenes {
        for held in [IVec3::ZERO, IVec3::X, IVec3::NEG_X] {
            for (reach, shape, precision) in [
                (14, LightShape::Round, LightPrecision::Steps16),
                (16, LightShape::Diamond, LightPrecision::Steps256),
            ] {
                settles_with_held(&scene(reach), held, reach, shape, precision);
            }
        }
    }
}

/// A red lamp in the chunk at the origin, and a green one in the chunk east
/// of it, nearer their shared face, so green is brighter than red on both
/// sides of it.
fn green_on_the_face_and_red_further_in() -> Scene {
    Scene::in_air(vec![
        (IVec3::new(25, 16, 16), RED_LAMP),
        (IVec3::new(35, 16, 16), GREEN_LAMP),
    ])
}

#[test]
fn a_neighbour_lit_green_on_its_face_relights_for_a_dimmer_red_arriving() {
    let scene = green_on_the_face_and_red_further_in();
    // Green outshines red on the face.
    let reference = scene.reference(14, LightShape::Round, LightPrecision::Steps16);
    let face = IVec3::new(32, 16, 16);
    let green = reference.voxel_at(face);
    let red_only = Scene::in_air(vec![(IVec3::new(25, 16, 16), RED_LAMP)]);
    let red = red_only
        .reference(14, LightShape::Round, LightPrecision::Steps16)
        .voxel_at(face);
    assert!(0 < red && red < green, "red {red}, green {green}");
    for shape in [LightShape::Round, LightShape::Diamond] {
        for precision in [LightPrecision::Steps16, LightPrecision::Steps256] {
            settles_with_held(&scene, IVec3::ZERO, 14, shape, precision);
        }
    }
}

/// Stone filling the plane x = 31, the east face of the chunk at the origin,
/// with a red lamp set in it, and a green lamp a voxel east of the face.
fn a_red_lamp_on_the_face_and_green_across_it() -> Scene {
    Scene {
        lamps: vec![
            (IVec3::new(31, 16, 16), RED_LAMP),
            (IVec3::new(33, 16, 16), GREEN_LAMP),
        ],
        wall: |position| position.x == 31,
        edits: Vec::new(),
    }
}

#[test]
fn an_emissive_voxel_on_a_face_lights_the_voxel_across_it_in_a_colour_the_neighbour_lacks(
) {
    let scene = a_red_lamp_on_the_face_and_green_across_it();
    for shape in [LightShape::Round, LightShape::Diamond] {
        for precision in [LightPrecision::Steps16, LightPrecision::Steps256] {
            settles_with_held(&scene, IVec3::ZERO, 14, shape, precision);
        }
    }
}

/// A red lamp beside the east face of the chunk at the origin, a green lamp
/// west of it, a blue one above, and stone below y = 0.
fn lamps_of_three_colours() -> Scene {
    Scene {
        lamps: vec![
            (RED, RED_LAMP),
            (IVec3::new(18, 16, 16), GREEN_LAMP),
            (IVec3::new(24, 24, 16), BLUE_LAMP),
        ],
        wall: |position| position.y < 0,
        edits: Vec::new(),
    }
}

const RED: IVec3 = IVec3::new(29, 16, 16);

fn reach_14() -> (u8, LightShape, LightPrecision) {
    (14, LightShape::Round, LightPrecision::Steps16)
}

/// The world of `scene`, with every emissive material white when `white`.
fn three_colour_world(scene: &Scene, white: bool) -> LampWorld {
    let (reach, shape, precision) = reach_14();
    let mut world = LampWorld {
        spawning_distance: 3,
        ..scene.world(reach, shape, precision)
    };
    if white {
        for (_, emission) in &mut world.emissive {
            *emission = emission.with_color(LinearRgba::WHITE);
        }
    }
    world
}

/// The light the world view holds for each chunk and light entry.
fn view_lights(app: &App) -> HashMap<GridKey, Arc<ChunkLight>> {
    let settings = app.world().resource::<LightSettings<LampWorld>>();
    let view = settings.view.read();
    view.chunks
        .iter()
        .map(|(key, chunk)| (*key, chunk.light.clone()))
        .collect()
}

/// Whether any chunk's light in `now` differs from its light in `was`.
fn light_changed(
    was: &HashMap<GridKey, Arc<ChunkLight>>,
    now: &HashMap<GridKey, Arc<ChunkLight>>,
) -> bool {
    now.iter()
        .any(|(key, light)| was.get(key).is_none_or(|old| !old.same_light(light)))
}

/// Sets `edits` over the world of `scene`, then sets the voxels back, and
/// returns the frame after each commit on which light last changed. Coloured
/// light must settle on the reference each time.
fn frames_to_last_change(
    scene: &Scene,
    edits: &[(IVec3, WorldVoxel)],
    white: bool,
) -> [u32; 2] {
    let (reach, shape, precision) = reach_14();
    let mut app = app(three_colour_world(scene, white));
    settle_but(&mut app, None, "the world to stream in");
    if !white {
        assert_on_the_reference(&mut app, scene, reach, shape, precision);
    }
    let undo: Vec<(IVec3, WorldVoxel)> = edits
        .iter()
        .map(|(p, _)| (*p, scene.voxel_at(*p)))
        .collect();
    let mut taken = [0; 2];
    for (step, (edits, after)) in
        [(edits, scene.edited(edits)), (&undo[..], scene.clone())]
            .into_iter()
            .enumerate()
    {
        for _ in 0..5 {
            framed_update(&mut app);
        }
        let mut lights = view_lights(&app);
        set_voxels(&mut app, edits);
        // Frame 0, which commits the edit.
        app.update();
        let mut frame = 0;
        while !settled_but(&mut app, None) {
            frame += 1;
            assert!(frame <= 30, "{edits:?}'s light hasn't settled");
            framed_update(&mut app);
            let now = view_lights(&app);
            if light_changed(&lights, &now) {
                taken[step] = frame;
            }
            lights = now;
        }
        if !white {
            assert_on_the_reference(&mut app, &after, reach, shape, precision);
        }
    }
    taken
}

/// Sets `edits` over [`lamps_of_three_colours`], and then sets the voxels
/// back. Light must settle on the reference each time, and its last change
/// land within 2 frames wherever the same edit's does in a world whose lamps
/// are all white.
fn edit_and_undo(edits: &[(IVec3, WorldVoxel)]) {
    edit_and_undo_beside(edits, edits);
}

/// As [`edit_and_undo`], comparing with `white_edits` in the white world.
fn edit_and_undo_beside(
    edits: &[(IVec3, WorldVoxel)],
    white_edits: &[(IVec3, WorldVoxel)],
) {
    let scene = lamps_of_three_colours();
    let colored = frames_to_last_change(&scene, edits, false);
    let white = frames_to_last_change(&scene, white_edits, true);
    for (colored, white) in colored.into_iter().zip(white) {
        assert!(
            colored <= white.max(2),
            "{edits:?}'s light last changed on frame {colored}, and {white} in white"
        );
    }
}

#[test]
fn placing_an_emissive_voxel_beside_lamps_of_other_colours_and_digging_it_out_settles_on_the_reference(
) {
    edit_and_undo(&[(IVec3::new(30, 20, 16), WorldVoxel::Solid(LAMP))]);
    edit_and_undo(&[(RED, WorldVoxel::Air)]);
}

#[test]
fn swapping_an_emissive_material_for_one_of_another_colour_settles_on_the_reference() {
    // In a white world the swap changes no light, so it's held to digging
    // the lamp out, which relights the same chunks.
    edit_and_undo_beside(
        &[(RED, WorldVoxel::Solid(BLUE_LAMP))],
        &[(RED, WorldVoxel::Air)],
    );
    let green = IVec3::new(18, 16, 16);
    edit_and_undo_beside(
        &[(green, WorldVoxel::Solid(RED_LAMP))],
        &[(green, WorldVoxel::Air)],
    );
}

#[test]
fn placing_a_block_beside_an_emissive_voxel_settles_on_the_reference() {
    edit_and_undo(&[(RED + IVec3::X, WorldVoxel::Solid(STONE))]);
    edit_and_undo(&[(RED - IVec3::X, WorldVoxel::Solid(STONE))]);
}

#[test]
fn digging_in_a_dark_cave_beside_coloured_light_starts_no_relight() {
    let scene = lamps_of_three_colours();
    let (reach, shape, precision) = reach_14();
    let mut app = settled(scene.world(reach, shape, precision));
    let before = relights_started(&app);
    // A voxel dug deep in the stone, and a stone placed back in it, both out
    // of every lamp's reach.
    let dug = IVec3::new(16, -24, 16);
    let edits = [(dug, WorldVoxel::Air), (dug, WorldVoxel::Solid(STONE))];
    for edit in edits {
        set_voxels(&mut app, &[edit]);
        settle_but(&mut app, None, "the edit to settle");
    }
    assert_eq!(relights_started(&app), before);
}

#[test]
fn after_reset_light_coloured_light_settles_on_the_reference() {
    let scene = lamps_of_three_colours();
    let (reach, shape, precision) = reach_14();
    let mut app = settled(scene.world(reach, shape, precision));
    app.world_mut()
        .run_system_once(|mut voxel_world: VoxelWorld<LampWorld>| {
            voxel_world.reset_light()
        })
        .unwrap();
    settle_but(&mut app, None, "light to settle after reset_light");
    assert_on_the_reference(&mut app, &scene, reach, shape, precision);
}

/// The voxel at `p` in the published chunk holding it.
fn published_voxel(app: &mut App, p: IVec3) -> Option<WorldVoxel> {
    app.world_mut()
        .run_system_once(move |chunk_map: Res<ChunkMap<LampWorld, u8>>| {
            let map = chunk_map.get_read_lock();
            map.get(&ChunkKey::new(p.div_euclid(IVec3::splat(32)), 0))?
                .get_voxel_at_world_position(p)
        })
        .unwrap()
}

#[test]
fn a_swapped_lamps_new_light_colour_lands_with_its_mesh() {
    let _holding = Hold::exclusive();
    let scene = lamps_of_three_colours();
    let (reach, shape, precision) = reach_14();
    let mut app = settled(scene.world(reach, shape, precision));
    let beside = RED - IVec3::Y;
    let old = light_at(&mut app, beside).unwrap();

    // The origin's new mesh waits, so its light lands first.
    let hold = Hold::new(IVec3::ZERO);
    app.world_mut().resource_mut::<LampWorld>().mesh_hold = Some(hold.clone());
    let swap = [(RED, WorldVoxel::Solid(BLUE_LAMP))];
    set_voxels(&mut app, &swap);
    for _ in 0..30 {
        paced_update(&mut app);
        assert_eq!(
            published_voxel(&mut app, RED),
            Some(WorldVoxel::Solid(RED_LAMP))
        );
        assert_eq!(light_at(&mut app, beside), Some(old), "before the mesh");
    }

    hold.release();
    let deadline = Instant::now() + Duration::from_secs(20);
    while published_voxel(&mut app, RED) == Some(WorldVoxel::Solid(RED_LAMP)) {
        assert!(Instant::now() < deadline, "timed out waiting for the mesh");
        app.update();
    }
    // The new mesh and the new light, levels and colour, landed together.
    let after = scene.edited(&swap);
    let reference = after.reference(reach, shape, precision);
    let read = light_at(&mut app, beside).unwrap();
    assert_eq!(read.voxel, reference.read(reference.voxel_at(beside)));
    assert_ne!(read.voxel_color, old.voxel_color);
    settle_but(&mut app, None, "light to settle");
    assert_on_the_reference(&mut app, &after, reach, shape, precision);
}
