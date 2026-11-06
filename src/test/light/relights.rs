//! Tests of relights: light coming right as chunks arrive around a chunk, in
//! any order, and staying no brighter than the truth along the way. Sky light
//! is off in all of them.

use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use bevy::ecs::system::RunSystemOnce;
use bevy::prelude::*;

use super::super::{chunk_keys, chunk_status, update_until};
use super::reference::Reference;
use super::voxel_light::{
    app, caves, caves_reference, chunk_voxel_light, light_at, lit, Hold, LampWorld,
    Scene, LAMP, STONE,
};
use crate::light::LightSettings;
use crate::prelude::*;

/// Voxel reach 14, the default, and 16, the most one `u64` row holds.
const REACHES: [u8; 2] = [14, 16];

/// The chunks around the origin, where the camera is.
fn around_origin() -> Vec<IVec3> {
    (-1..=1)
        .flat_map(|z| {
            (-1..=1).flat_map(move |y| (-1..=1).map(move |x| IVec3::new(x, y, z)))
        })
        .collect()
}

/// The world of [`caves`] at `reach`, holding back the chunk at `held`.
fn caves_world(reach: u8, held: Option<IVec3>) -> (LampWorld, Option<Arc<Hold>>) {
    let hold = held.map(Hold::new);
    let world = LampWorld {
        scene: Arc::new(caves),
        light: Some(LightPropagation {
            voxel_reach: reach,
            ..lit(LightShape::Round)
        }),
        hold: hold.clone(),
        ..default()
    };
    (world, hold)
}

/// The reference light of [`caves`] at `reach` for the chunks around the
/// origin, worked out once per reach.
fn reference(reach: u8) -> &'static Reference {
    static REFERENCES: OnceLock<HashMap<u8, Reference>> = OnceLock::new();
    let references = REFERENCES.get_or_init(|| {
        REACHES
            .iter()
            .map(|&reach| {
                // Far enough past the chunks that light from outside it can't
                // reach them.
                let margin = IVec3::splat(reach as i32 + 1);
                let min = IVec3::splat(-32) - margin;
                let max = IVec3::splat(64) + margin;
                (reach, caves_reference(min, max, reach as u32))
            })
            .collect()
    });
    &references[&reach]
}

/// The level-0 chunks spawned in `app`.
fn spawned(app: &mut App) -> Vec<IVec3> {
    chunk_keys::<LampWorld>(app)
        .into_iter()
        .filter(|key| key.lod_level == 0)
        .map(|key| key.position)
        .collect()
}

fn light_is_settled(app: &mut App) -> bool {
    app.world_mut()
        .run_system_once(|voxel_world: VoxelWorld<LampWorld>| {
            voxel_world.light_is_settled()
        })
        .unwrap()
}

fn relights_started(app: &App) -> u64 {
    app.world()
        .resource::<LightSettings<LampWorld>>()
        .counters()
        .relights
        .load(Ordering::Relaxed)
}

/// Whether every spawned chunk but `held` is Ready and light has settled.
fn settled_but(app: &mut App, held: Option<IVec3>) -> bool {
    let chunks = spawned(app);
    !chunks.is_empty()
        && chunks
            .iter()
            .filter(|chunk| Some(**chunk) != held)
            .all(|chunk| {
                chunk_status::<LampWorld>(app, *chunk) == Some(ChunkStatus::Ready)
            })
        && light_is_settled(app)
}

/// Fails if any voxel of the chunks around the origin reads brighter than the
/// reference.
fn assert_no_brighter(app: &mut App, reference: &Reference, when: &str) {
    for chunk in around_origin() {
        let Some(light) = chunk_voxel_light(app, chunk) else {
            continue;
        };
        for (p, level) in light {
            let truth = reference.voxel_at(p) as f32;
            assert!(
                level <= truth,
                "{level} at {p} is brighter than {truth} {when}"
            );
        }
    }
}

/// Runs `app` until `done` holds for a few frames in a row, checking every
/// frame that no reading is brighter than the reference.
fn run_checked(
    app: &mut App,
    reference: &Reference,
    what: &str,
    mut done: impl FnMut(&mut App) -> bool,
) {
    let deadline = Instant::now() + Duration::from_secs(60);
    let mut frames_done = 0;
    while frames_done < 3 {
        assert!(Instant::now() < deadline, "timed out waiting: {what}");
        app.update();
        assert_no_brighter(app, reference, what);
        frames_done = if done(app) { frames_done + 1 } else { 0 };
    }
}

/// Streams the world of [`caves`] at `reach` until it settles, holding back
/// the chunk at `held` until everything else has settled. Checks every frame
/// that no reading is brighter than the reference, and returns the settled
/// light of each chunk around the origin.
fn stream_caves(
    reach: u8,
    held: Option<IVec3>,
) -> (App, HashMap<IVec3, Vec<(IVec3, f32)>>) {
    let _holding = held.map(|_| Hold::exclusive());
    let reference = reference(reach);
    let (world, hold) = caves_world(reach, held);
    let mut app = app(world);
    run_checked(&mut app, reference, "for all but the held chunk", |app| {
        settled_but(app, held)
    });
    if let Some(hold) = hold {
        hold.release();
        run_checked(&mut app, reference, "for the held chunk", |app| {
            settled_but(app, None)
        });
    }
    let light = around_origin()
        .into_iter()
        .filter_map(|chunk| Some((chunk, chunk_voxel_light(&mut app, chunk)?)))
        .collect();
    (app, light)
}

/// The chunks whose whole box is loaded: they and the 26 chunks around them.
fn boxed(spawned: &[IVec3]) -> Vec<IVec3> {
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

/// Streams with the chunk at `held` arriving last, then checks every chunk
/// around the origin whose whole box is loaded against the reference.
fn settles_on_the_reference(reach: u8, held: Option<IVec3>) {
    let (mut app, light) = stream_caves(reach, held);
    let checked: Vec<IVec3> = boxed(&spawned(&mut app))
        .into_iter()
        .filter(|chunk| around_origin().contains(chunk))
        .collect();
    assert!(
        checked.contains(&IVec3::ZERO),
        "the origin's box isn't loaded"
    );
    let reference = reference(reach);
    for chunk in checked {
        let light = &light[&chunk];
        let lit = light.iter().filter(|(_, level)| *level > 0.0).count();
        assert!(lit > 100, "only {lit} voxels of {chunk} are lit");
        for (p, level) in light {
            assert_eq!(
                *level,
                reference.voxel_at(*p) as f32,
                "at {p} in {chunk}, with {held:?} held, at reach {reach}"
            );
        }
    }
}

#[test]
fn after_streaming_settles_every_chunk_whose_whole_box_is_loaded_matches_the_reference() {
    for reach in REACHES {
        settles_on_the_reference(reach, None);
    }
}

#[test]
fn a_chunk_arriving_after_its_face_neighbours_settles_on_the_reference() {
    for reach in REACHES {
        settles_on_the_reference(reach, Some(IVec3::ZERO));
    }
}

#[test]
fn a_face_neighbour_arriving_last_settles_on_the_reference() {
    for reach in REACHES {
        settles_on_the_reference(reach, Some(IVec3::X));
    }
}

#[test]
fn an_edge_neighbour_arriving_last_settles_on_the_reference() {
    for reach in REACHES {
        settles_on_the_reference(reach, Some(IVec3::new(0, 1, -1)));
    }
}

#[test]
fn a_corner_neighbour_arriving_last_settles_on_the_reference() {
    for reach in REACHES {
        settles_on_the_reference(reach, Some(IVec3::new(-1, 1, 1)));
    }
}

#[test]
fn the_same_chunks_arriving_in_different_orders_settle_on_identical_light() {
    let reach = 16;
    let (_, natural) = stream_caves(reach, None);
    for held in [IVec3::ZERO, IVec3::NEG_Y, IVec3::new(1, 1, 0)] {
        let (_, light) = stream_caves(reach, Some(held));
        assert_eq!(
            light.keys().collect::<std::collections::HashSet<_>>(),
            natural.keys().collect(),
            "other chunks have light with {held} held"
        );
        for (chunk, light) in light {
            assert!(
                light == natural[&chunk],
                "{chunk} settled on other light with {held} held"
            );
        }
    }
}

#[test]
fn chunks_retiring_one_at_a_time_leave_no_reading_brighter_than_the_reference() {
    let reach = 14;
    let reference = reference(reach);
    let (world, _) = caves_world(reach, None);
    let mut app = app(world);
    run_checked(&mut app, reference, "streaming in", |app| {
        settled_but(app, None)
    });

    // The camera moves away a chunk at a time, so the chunks around the
    // origin retire a few at a time, then comes back.
    let path: Vec<f32> = (1..=4)
        .chain((0..4).rev())
        .map(|step| step as f32)
        .collect();
    for step in path {
        let to = Transform::from_xyz(10.0 + 32.0 * step, 10.0, 10.0)
            .looking_at(Vec3::new(32.0 * step, 0.0, 0.0), Vec3::Y);
        super::super::move_cameras::<LampWorld>(&mut app, to);
        run_checked(&mut app, reference, "while chunks retire", |app| {
            settled_but(app, None)
        });
    }
}

#[test]
fn a_chunk_retiring_with_no_provisional_columns_starts_no_relight() {
    let (world, _) = caves_world(14, None);
    let mut app = app(world);
    update_until(&mut app, "the world to settle", |app| {
        settled_but(app, None)
    });
    for _ in 0..3 {
        app.update();
    }
    let before = relights_started(&app);
    let spawned_before = spawned(&mut app).len();

    // Fewer chunks are spawned within a shorter distance, so some retire and
    // none spawn.
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

/// A floor of stone 4 voxels deep under air, with a lamp above it in the chunk
/// beside the origin along x, 3 voxels from their shared face.
fn lamp_beside_the_origin(p: IVec3) -> WorldVoxel {
    if p == IVec3::new(34, 8, 16) {
        WorldVoxel::Solid(LAMP)
    } else if (0..4).contains(&p.y) {
        WorldVoxel::Solid(STONE)
    } else {
        WorldVoxel::Air
    }
}

#[test]
fn a_relit_chunk_stays_ready_and_visible() {
    let _holding = Hold::exclusive();
    let hold = Hold::new(IVec3::X);
    let scene: Scene = Arc::new(lamp_beside_the_origin);
    let mut app = app(LampWorld {
        scene,
        hold: Some(hold.clone()),
        ..default()
    });
    let lit = IVec3::new(31, 8, 16);
    update_until(&mut app, "the origin to be ready", |app| {
        chunk_status::<LampWorld>(app, IVec3::ZERO) == Some(ChunkStatus::Ready)
            && light_at(app, lit).is_some()
    });
    assert_eq!(light_at(&mut app, lit).unwrap().voxel, 0.0);

    hold.release();
    let origin = app
        .world_mut()
        .run_system_once(|voxel_world: VoxelWorld<LampWorld>| {
            voxel_world.chunk_at(IVec3::ZERO)
        })
        .unwrap()
        .unwrap();
    let entity = app
        .world_mut()
        .query::<(Entity, &Chunk<LampWorld>)>()
        .iter(app.world())
        .find(|(_, chunk)| chunk.key() == origin)
        .unwrap()
        .0;
    let deadline = Instant::now() + Duration::from_secs(20);
    while light_at(&mut app, lit).unwrap().voxel == 0.0 {
        assert!(
            Instant::now() < deadline,
            "timed out waiting for the relight"
        );
        app.update();
        assert_eq!(
            chunk_status::<LampWorld>(&mut app, IVec3::ZERO),
            Some(ChunkStatus::Ready)
        );
        let shown = app.world().entity(entity);
        assert!(shown.contains::<Mesh3d>(), "the origin lost its mesh");
        assert_ne!(shown.get::<Visibility>(), Some(&Visibility::Hidden));
    }
    // 3 steps from the lamp, at reach 14.
    assert_eq!(light_at(&mut app, lit).unwrap().voxel, 12.0);
    assert!(relights_started(&app) > 0);
}

#[test]
fn light_is_settled_goes_true_once_arrivals_have_landed() {
    let _holding = Hold::exclusive();
    let hold = Hold::new(IVec3::X);
    let scene: Scene = Arc::new(lamp_beside_the_origin);
    let mut app = app(LampWorld {
        scene,
        hold: Some(hold.clone()),
        ..default()
    });
    let lit = IVec3::new(31, 8, 16);
    update_until(&mut app, "all but the held chunk to settle", |app| {
        settled_but(app, Some(IVec3::X))
    });
    assert!(light_is_settled(&mut app));

    hold.release();
    let mut was_unsettled = false;
    update_until(&mut app, "light to settle again", |app| {
        let settled = light_is_settled(app);
        was_unsettled |= !settled;
        was_unsettled && settled
    });
    assert_eq!(light_at(&mut app, lit).unwrap().voxel, 12.0);
}
