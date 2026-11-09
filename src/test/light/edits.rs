//! Tests of voxel edits: light follows the voxels a game sets, settling on the
//! reference for the edited world, within two frames for a single voxel, and
//! with no relight where the edit touches no light. Sky fill is off in all of
//! them.

use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use bevy::ecs::system::RunSystemOnce;
use bevy::prelude::*;

use super::reference::Reference;
use super::sky_light::{chunk_light, light_is_settled, settled_but, sky_light, spawned};
use super::voxel_light::{app, light_at, Hold, LampWorld, DIM_LAMP, LAMP, STONE};
use crate::chunk_map::ChunkMap;
use crate::light::LightSettings;
use crate::prelude::*;

/// A lamp in the floor of the cavern of [`cavern`].
const FLOOR_LAMP: IVec3 = IVec3::new(-6, 3, 2);

/// Stone below y = 40 and air above, with a cavern 12 voxels high under the
/// origin chunk, from y = 4 to 16 and from -12 to 44 along x and z. A 1×1
/// shaft at x = 8, z = 8 and a roof hole 3 voxels wide at x and z from 20 to
/// 23 open its roof to the sky. A lamp sits in its floor, and a sealed pocket
/// of air lies in the stone above it.
pub(super) fn cavern(p: IVec3) -> WorldVoxel {
    let cavern =
        (-12..44).contains(&p.x) && (4..16).contains(&p.y) && (-12..44).contains(&p.z);
    let shaft = p.x == 8 && p.z == 8 && (16..40).contains(&p.y);
    let hole =
        (20..23).contains(&p.x) && (20..23).contains(&p.z) && (16..40).contains(&p.y);
    if p == FLOOR_LAMP {
        WorldVoxel::Solid(LAMP)
    } else if p.y >= 40 || cavern || shaft || hole || in_pocket(p) {
        WorldVoxel::Air
    } else {
        WorldVoxel::Solid(STONE)
    }
}

/// Whether `p` is in the sealed pocket of [`cavern`], which no light reaches.
fn in_pocket(p: IVec3) -> bool {
    (26..32).contains(&p.x) && (22..30).contains(&p.y) && (2..8).contains(&p.z)
}

/// The voxels of [`cavern`] with `edits` set over them.
pub(super) fn edited(
    edits: &[(IVec3, WorldVoxel)],
) -> impl Fn(IVec3) -> WorldVoxel + use<'_> {
    let edits: HashMap<IVec3, WorldVoxel> = edits.iter().copied().collect();
    move |p| edits.get(&p).copied().unwrap_or_else(|| cavern(p))
}

pub(super) fn world() -> LampWorld {
    LampWorld {
        scene: Arc::new(cavern),
        light: Some(sky_light(Sky::default(), 30)),
        ..default()
    }
}

/// The reference sky light and voxel light of `voxel_at`, in a box reaching
/// far enough past `chunks` that light from outside it can't reach them.
pub(super) fn reference(
    chunks: &[IVec3],
    voxel_at: impl Fn(IVec3) -> WorldVoxel,
) -> (Reference, Reference) {
    let margin = IVec3::splat(31);
    let min = chunks.iter().fold(IVec3::MAX, |min, chunk| min.min(*chunk)) * 32 - margin;
    let max =
        chunks.iter().fold(IVec3::MIN, |max, chunk| max.max(*chunk)) * 32 + 32 + margin;
    let sky =
        Reference::sky_light(min, max, 30, LightShape::Round, &Sky::default(), &voxel_at);
    let voxel =
        Reference::voxel_light(min, max, 14, LightShape::Round, &voxel_at, |material| {
            match material {
                LAMP => Some(15.0),
                DIM_LAMP => Some(8.0),
                _ => None,
            }
        });
    (sky, voxel)
}

/// The reference of the unedited [`cavern`] for the origin chunk.
fn origin_reference() -> &'static (Reference, Reference) {
    static REFERENCE: OnceLock<(Reference, Reference)> = OnceLock::new();
    REFERENCE.get_or_init(|| reference(&[IVec3::ZERO], cavern))
}

pub(super) fn set_voxels(app: &mut App, edits: &[(IVec3, WorldVoxel)]) {
    let edits = edits.to_vec();
    app.world_mut()
        .run_system_once(move |mut voxel_world: VoxelWorld<LampWorld>| {
            for (position, voxel) in &edits {
                voxel_world.set_voxel(*position, *voxel);
            }
        })
        .unwrap();
}

/// Runs `app` until every spawned chunk is Ready and light has settled, for a
/// few frames in a row.
pub(super) fn settle(app: &mut App, what: &str) {
    let deadline = Instant::now() + Duration::from_secs(60);
    let mut frames_done = 0;
    while frames_done < 3 {
        assert!(Instant::now() < deadline, "timed out waiting: {what}");
        app.update();
        frames_done = if settled_but(app, None) {
            frames_done + 1
        } else {
            0
        };
    }
}

/// Fails unless every voxel of `chunks` reads the light of `reference`.
pub(super) fn assert_matches(
    app: &mut App,
    chunks: &[IVec3],
    reference: &(Reference, Reference),
    when: &str,
) {
    let spawned = spawned(app);
    for chunk in chunks {
        let boxed = (-1..=1).all(|z| {
            (-1..=1).all(|y| {
                (-1..=1).all(|x| spawned.contains(&(*chunk + IVec3::new(x, y, z))))
            })
        });
        assert!(boxed, "the box of {chunk} isn't loaded");
        let light =
            chunk_light(app, *chunk).unwrap_or_else(|| panic!("{chunk} has no light"));
        for (p, levels) in light {
            assert_eq!(
                (levels.sky, levels.voxel),
                (
                    reference.0.read(reference.0.sky_at(p)),
                    reference.1.read(reference.1.voxel_at(p))
                ),
                "at {p} in {chunk} {when}"
            );
        }
    }
}

/// Streams [`cavern`], then sets `edits`, and then sets the voxels back. Light
/// in `chunks` must match the reference once settled at each step.
fn edit_and_undo(chunks: &[IVec3], edits: &[(IVec3, WorldVoxel)]) {
    // A chunk beside the origin needs a wider spawning distance to have its
    // whole box loaded.
    let spawning_distance = if chunks == [IVec3::ZERO] { 2 } else { 3 };
    let mut app = app(LampWorld {
        spawning_distance,
        ..world()
    });
    settle(&mut app, "the world to stream in");
    let before = if chunks == [IVec3::ZERO] {
        origin_reference()
    } else {
        &reference(chunks, cavern)
    };
    assert_matches(&mut app, chunks, before, "before the edit");

    set_voxels(&mut app, edits);
    settle(&mut app, "light to settle after the edit");
    let after = reference(chunks, edited(edits));
    assert_matches(&mut app, chunks, &after, "after the edit");

    let undo: Vec<(IVec3, WorldVoxel)> =
        edits.iter().map(|(p, _)| (*p, cavern(*p))).collect();
    set_voxels(&mut app, &undo);
    settle(&mut app, "light to settle after the undo");
    assert_matches(&mut app, chunks, before, "after the undo");
}

pub(super) fn place(p: IVec3) -> (IVec3, WorldVoxel) {
    (p, WorldVoxel::Solid(STONE))
}

pub(super) fn dig(p: IVec3) -> (IVec3, WorldVoxel) {
    (p, WorldVoxel::Air)
}

#[test]
fn edits_where_light_is_match_the_reference() {
    // Lit by sky light from the shaft, and by the lamp.
    edit_and_undo(&[IVec3::ZERO], &[place(IVec3::new(2, 10, 3))]);
    edit_and_undo(&[IVec3::ZERO], &[place(IVec3::new(-4, 5, 2))]);
}

#[test]
fn edits_against_chunk_faces_match_the_reference() {
    let chunks = [IVec3::ZERO, IVec3::X];
    edit_and_undo(&chunks, &[place(IVec3::new(31, 6, 8))]);
    edit_and_undo(&chunks, &[place(IVec3::new(32, 6, 8))]);
    edit_and_undo(&[IVec3::ZERO, IVec3::NEG_Y], &[dig(IVec3::new(10, 0, 10))]);
}

#[test]
fn placing_and_removing_a_torch_matches_the_reference() {
    edit_and_undo(
        &[IVec3::ZERO],
        &[(IVec3::new(16, 10, 28), WorldVoxel::Solid(LAMP))],
    );
    // The lamp in the floor, removed and put back.
    edit_and_undo(&[IVec3::ZERO], &[place(FLOOR_LAMP)]);
}

#[test]
fn digs_match_the_reference() {
    // The cavern's floor, and its roof.
    edit_and_undo(&[IVec3::ZERO], &[dig(IVec3::new(14, 3, 12))]);
    edit_and_undo(&[IVec3::ZERO], &[dig(IVec3::new(14, 16, 12))]);
}

#[test]
fn placing_in_a_shafts_pool_matches_the_reference() {
    edit_and_undo(&[IVec3::ZERO], &[place(IVec3::new(9, 4, 8))]);
    edit_and_undo(&[IVec3::ZERO], &[place(IVec3::new(8, 4, 8))]);
}

#[test]
fn capping_and_uncapping_a_shaft_matches_the_reference() {
    // The shaft's top, in the chunk above the origin.
    edit_and_undo(&[IVec3::ZERO], &[place(IVec3::new(8, 39, 8))]);
}

#[test]
fn capping_and_uncapping_a_roof_hole_matches_the_reference() {
    let cap: Vec<(IVec3, WorldVoxel)> = (20..23)
        .flat_map(|z| (20..23).map(move |x| place(IVec3::new(x, 39, z))))
        .collect();
    edit_and_undo(&[IVec3::ZERO], &cap);
}

#[test]
fn a_radius_8_bomb_and_its_undo_match_the_reference() {
    let centre = IVec3::new(16, 16, 16);
    let bomb: Vec<(IVec3, WorldVoxel)> = (-8..=8)
        .flat_map(|z| {
            (-8..=8).flat_map(move |y| (-8..=8).map(move |x| IVec3::new(x, y, z)))
        })
        .filter(|offset| offset.length_squared() <= 64)
        .map(|offset| dig(centre + offset))
        .collect();
    edit_and_undo(&[IVec3::ZERO], &bomb);
}

/// [`cavern`] with its floor lamp turned to stone and its shaft filled, so
/// regenerating changes a chunk's emissive voxels and occupancy but none of
/// its voxel edits.
fn dark_cavern(p: IVec3) -> WorldVoxel {
    let shaft = p.x == 8 && p.z == 8 && (16..40).contains(&p.y);
    if p == FLOOR_LAMP || shaft {
        WorldVoxel::Solid(STONE)
    } else {
        cavern(p)
    }
}

#[test]
fn a_regenerated_chunk_is_solved_again_and_matches_the_reference() {
    let mut app = app(world());
    settle(&mut app, "the world to stream in");
    let edit = place(IVec3::new(2, 10, 3));
    set_voxels(&mut app, &[edit]);
    settle(&mut app, "light to settle after the edit");
    assert_matches(
        &mut app,
        &[IVec3::ZERO],
        &reference(&[IVec3::ZERO], edited(&[edit])),
        "before regenerating",
    );

    app.world_mut().resource_mut::<LampWorld>().scene = Arc::new(dark_cavern);
    app.world_mut()
        .run_system_once(|mut voxel_world: VoxelWorld<LampWorld>| {
            voxel_world.regenerate_chunks()
        })
        .unwrap();
    settle(&mut app, "light to settle after regenerating");
    let after = reference(&[IVec3::ZERO], |p| {
        if p == edit.0 {
            edit.1
        } else {
            dark_cavern(p)
        }
    });
    assert_matches(&mut app, &[IVec3::ZERO], &after, "after regenerating");
}

pub(super) fn relights_started(app: &App) -> u64 {
    app.world()
        .resource::<LightSettings<LampWorld>>()
        .counters()
        .relights
        .load(Ordering::Relaxed)
}

/// One app update, after waiting as long as a frame at 50 frames a second
/// takes, so tasks started on the frame before have had time to finish.
pub(super) fn paced_update(app: &mut App) {
    std::thread::sleep(Duration::from_millis(20));
    app.update();
}

/// Whether every chunk task, relight and probe running in `app` has finished.
fn tasks_finished(app: &mut App) -> bool {
    use crate::chunk_lifecycle::{ChunkState, ChunkStateOf};
    use crate::light::{fill::driver::SkyFillWork, probe::SkyProbes, relight::LightWork};

    let world = app.world_mut();
    let chunks_finished =
        world
            .query::<&ChunkStateOf<LampWorld>>()
            .iter(world)
            .all(|state| match state {
                ChunkState::Generating { task, .. } => task.is_finished(),
                _ => true,
            });
    chunks_finished
        && world.resource::<LightWork<LampWorld>>().tasks_finished()
        && world
            .get_resource::<SkyProbes<LampWorld>>()
            .is_none_or(|probes| probes.tasks_finished())
        && world
            .get_resource::<SkyFillWork<LampWorld>>()
            .is_none_or(|fill| fill.tasks_finished())
}

/// One app update, once every task running has finished, as a frame long
/// enough for the tasks started on the frame before.
///
/// The task pool is shared by every test in the process, so a fixed wait can
/// end before a task even starts.
pub(super) fn framed_update(app: &mut App) {
    let deadline = Instant::now() + Duration::from_secs(30);
    while !tasks_finished(app) {
        assert!(Instant::now() < deadline, "timed out waiting for tasks");
        std::thread::sleep(Duration::from_millis(1));
    }
    app.update();
}

#[test]
fn a_single_voxel_edits_light_lands_within_2_frames() {
    let mut app = app(world());
    settle(&mut app, "the world to stream in");
    for _ in 0..5 {
        framed_update(&mut app);
    }
    let edit = [place(IVec3::new(9, 4, 8))];
    set_voxels(&mut app, &edit);
    // Frame 0, which commits the edit.
    app.update();
    let mut frames = 0;
    while !settled_but(&mut app, None) {
        frames += 1;
        assert!(frames <= 2, "the edit's light hasn't landed by frame 2");
        framed_update(&mut app);
    }
    assert_matches(
        &mut app,
        &[IVec3::ZERO],
        &reference(&[IVec3::ZERO], edited(&edit)),
        "after the edit",
    );
}

#[test]
fn an_edit_in_a_dark_cave_starts_no_relight() {
    let mut app = app(world());
    settle(&mut app, "the world to stream in");
    let before = relights_started(&app);

    // A stone placed in the sealed pocket, and a voxel of its wall dug.
    let edits = [place(IVec3::new(28, 25, 4)), dig(IVec3::new(25, 25, 4))];
    for edit in edits {
        set_voxels(&mut app, &[edit]);
        settle(&mut app, "the edit to settle");
    }
    assert_eq!(relights_started(&app), before);
    assert_matches(
        &mut app,
        &[IVec3::ZERO],
        &reference(&[IVec3::ZERO], edited(&edits)),
        "after the edits",
    );
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
fn after_placing_a_block_the_old_floor_face_never_reads_the_new_solids_0_for_a_frame() {
    let _holding = Hold::exclusive();
    let mut app = app(world());
    settle(&mut app, "the world to stream in");
    // On the cavern's floor, in sky light from the roof hole.
    let placed = IVec3::new(16, 4, 16);
    let lit = light_at(&mut app, placed).unwrap();
    assert!(lit.sky > 0.0);

    // The origin's new mesh waits, so its light lands first.
    let hold = Hold::new(IVec3::ZERO);
    app.world_mut().resource_mut::<LampWorld>().mesh_hold = Some(hold.clone());
    set_voxels(&mut app, &[place(placed)]);
    for _ in 0..30 {
        paced_update(&mut app);
        assert_eq!(published_voxel(&mut app, placed), Some(WorldVoxel::Air));
        assert_eq!(
            light_at(&mut app, placed),
            Some(lit),
            "the old floor face reads the placed block's light"
        );
    }

    hold.release();
    let deadline = Instant::now() + Duration::from_secs(20);
    while published_voxel(&mut app, placed) == Some(WorldVoxel::Air) {
        assert!(Instant::now() < deadline, "timed out waiting for the mesh");
        app.update();
    }
    // The new mesh and the new light landed in the same frame.
    let above = placed + IVec3::Y;
    let truth = reference(&[IVec3::ZERO], edited(&[place(placed)]));
    assert_eq!(
        light_at(&mut app, above).unwrap().sky,
        truth.0.sky_at(above) as f32
    );
    settle(&mut app, "light to settle");
    assert_matches(&mut app, &[IVec3::ZERO], &truth, "after the edit");
}

#[test]
fn digging_every_frame_keeps_light_moving_and_light_settles_once_digging_stops() {
    let mut app = app(LampWorld {
        spawning_distance: 3,
        ..world()
    });
    settle(&mut app, "the world to stream in");

    // A tunnel along -x from the cavern's wall at y = 4, beside the lamp's
    // side of the cavern, a voxel a frame.
    let tunnel: Vec<(IVec3, WorldVoxel)> =
        (13..33).map(|x| dig(IVec3::new(-x, 4, 2))).collect();
    let first = tunnel[0].0;
    for edit in &tunnel {
        set_voxels(&mut app, &[*edit]);
        paced_update(&mut app);
    }
    // The first voxels dug have their light while digging goes on.
    let read = light_at(&mut app, first);
    let published = published_voxel(&mut app, first);
    assert!(read.unwrap().voxel > 0.0, "{read:?} {published:?}");
    assert!(!light_is_settled(&mut app) || relights_started(&app) > 0);

    settle(&mut app, "light to settle once digging stops");
    let chunks = [IVec3::ZERO, IVec3::NEG_X];
    assert_matches(
        &mut app,
        &chunks,
        &reference(&chunks, edited(&tunnel)),
        "after digging",
    );
}

#[test]
fn light_is_settled_goes_false_on_an_edit_and_true_once_everything_lands() {
    let mut app = app(world());
    settle(&mut app, "the world to stream in");
    assert!(light_is_settled(&mut app));

    let edit = [place(IVec3::new(2, 10, 3))];
    set_voxels(&mut app, &edit);
    assert!(!light_is_settled(&mut app), "settled with an edit waiting");
    app.update();
    assert!(!light_is_settled(&mut app), "settled on the edit's frame");
    settle(&mut app, "light to settle");
    assert!(light_is_settled(&mut app));
    assert_matches(
        &mut app,
        &[IVec3::ZERO],
        &reference(&[IVec3::ZERO], edited(&edit)),
        "after the edit",
    );
}
