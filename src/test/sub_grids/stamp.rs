//! Tests of stamp: a sub-grid's voxels move into another grid at the nearest
//! pose that lines up with it, the move lands as one handover that
//! `Stamped` announces, and the sub-grid despawns.

use std::collections::BTreeSet;
use std::f32::consts::FRAC_PI_2;

use bevy::diagnostic::FrameCount;
use bevy::ecs::system::RunSystemOnce;
use bevy::prelude::*;

use super::lift::{
    all_published_since, assert_poses_match, ground, lift_app, lifted, next_frame,
    padded_chunks, positions, published_since, row, run, sub_grid_holding, until_lifted,
    LiftWorld,
};
use crate::prelude::*;
use crate::test::{record_chunk_messages, recorded, update_until};

/// Each `Stamped` an app read, with the frame it arrived in and whether its
/// sub-grid's entity still existed as it was read.
#[derive(Resource, Default)]
struct StampedLog(Vec<(u32, Stamped<LiftWorld>, bool)>);

fn record_stamped(
    mut log: ResMut<StampedLog>,
    mut stamped: MessageReader<Stamped<LiftWorld>>,
    frame: Res<FrameCount>,
    entities: Query<Entity>,
) {
    let frame = frame.0;
    log.0.extend(
        stamped
            .read()
            .map(|message| (frame, *message, entities.contains(message.sub_grid))),
    );
}

fn stamped(app: &App) -> &[(u32, Stamped<LiftWorld>, bool)] {
    &app.world().resource::<StampedLog>().0
}

/// [`lift_app`], also recording `Stamped` and `ChunkWillDespawn`.
fn stamp_app() -> App {
    let mut app = lift_app();
    app.init_resource::<StampedLog>()
        .add_systems(Update, record_stamped);
    record_chunk_messages::<ChunkWillDespawn<LiftWorld>>(&mut app);
    app
}

/// Runs `app` until it has read a `Stamped` naming `sub_grid`.
fn until_stamped(app: &mut App, sub_grid: Entity) {
    update_until(app, "the stamp to land", |app| {
        stamped(app)
            .iter()
            .any(|(_, message, _)| message.sub_grid == sub_grid)
    });
}

/// The frames, from `since` on, in which each chunk of `grid` despawned.
fn despawned_since(app: &App, since: u32, grid: Grid) -> Vec<([i32; 3], u32)> {
    recorded::<ChunkWillDespawn<LiftWorld>>(app)
        .messages
        .iter()
        .filter(|(frame, message)| *frame >= since && message.grid == grid)
        .map(|(frame, message)| (message.chunk_key.position.to_array(), *frame))
        .collect()
}

/// The one frame all of `frames` happened in.
fn one_frame(frames: &[([i32; 3], u32)]) -> u32 {
    let set: BTreeSet<u32> = frames.iter().map(|(_, frame)| *frame).collect();
    assert_eq!(set.len(), 1, "{frames:?}");
    *set.first().unwrap()
}

/// Runs `app` until each chunk of `grid` at one of `chunks` has published
/// since `since`.
fn until_published(app: &mut App, since: u32, grid: Grid, chunks: &BTreeSet<[i32; 3]>) {
    update_until(app, "the chunks to publish", |app| {
        all_published_since(app, since, grid, chunks)
    });
}

/// What `grid` reads at each of `positions`.
fn read(app: &mut App, grid: Grid, positions: Vec<IVec3>) -> Vec<WorldVoxel> {
    run(app, move |voxel_world| {
        let grid = voxel_world.grid(grid).unwrap();
        positions.iter().map(|p| grid.get_voxel(*p)).collect()
    })
}

/// [`row`] and two voxels off it, one landing on air and one on the hole in
/// the ground under the pose [`preview_pose`] gives.
fn row_and_two() -> Vec<IVec3> {
    let mut voxels = row();
    voxels.extend([IVec3::new(3, 6, 5), IVec3::new(0, 4, 0)]);
    voxels
}

/// Five voxels down and a little off the grid, turned slightly about y.
fn preview_pose() -> Transform {
    Transform::from_translation(Vec3::new(0.2, -5.0, 0.1))
        .with_rotation(Quat::from_rotation_y(0.05))
}

#[test]
fn the_preview_reports_the_snap_and_each_overlap_and_changes_nothing() {
    let mut app = stamp_app();
    let sub_grid = sub_grid_holding(&mut app, preview_pose(), row_and_two());
    let never = app.world_mut().spawn_empty().id();

    let (preview, dead_target, edited, bounds) = run(&mut app, move |voxel_world| {
        (
            voxel_world.stamp_preview(sub_grid, Grid::World),
            voxel_world.stamp_preview(sub_grid, Grid::SubGrid(never)),
            voxel_world.edited_chunks(),
            voxel_world.grid(Grid::SubGrid(sub_grid)).unwrap().bounds(),
        )
    });

    let preview = preview.unwrap();
    assert_eq!(preview.rotation, GridRotation::IDENTITY);
    assert!((preview.angle - 0.05).abs() < 1e-4, "{}", preview.angle);
    // The solids' centre, (3.5, 5.5, 3), sits at about (3.846, 0.5, 2.921)
    // under the pose and at (3.5, 0.5, 3) once snapped.
    assert!(
        preview
            .offset
            .abs_diff_eq(Vec3::new(-0.3455, 0.0, 0.0789), 1e-3),
        "{}",
        preview.offset
    );
    assert_eq!(preview.scale, Vec3::ONE);
    assert!(preview
        .pose
        .translation
        .abs_diff_eq(Vec3::new(0.0, -5.0, 0.0), 1e-4));
    assert!(preview.pose.rotation.abs_diff_eq(Quat::IDENTITY, 1e-5));
    let overlaps: BTreeSet<[i32; 3]> =
        preview.overlaps.iter().map(|p| p.to_array()).collect();
    let expected: BTreeSet<[i32; 3]> = (3..7).map(|x| [x, -1, 5]).collect();
    assert_eq!(overlaps, expected);

    assert_eq!(dead_target, None);
    assert!(edited.is_empty(), "{edited:?}");
    assert_eq!(bounds, Some((IVec3::new(0, 4, 0), IVec3::new(7, 7, 6))));
    app.world_mut().entity_mut(sub_grid).despawn();
    app.update();
    let dead = run(&mut app, move |voxel_world| {
        voxel_world.stamp_preview(sub_grid, Grid::World)
    });
    assert_eq!(dead, None);
}

#[test]
fn a_long_sub_grid_turned_off_an_axis_snaps_about_its_solids_centre() {
    let mut app = stamp_app();
    let long: Vec<IVec3> = (1..41).map(|x| IVec3::new(x, 1, 1)).collect();
    let turned = Transform::from_translation(Vec3::new(0.0, 10.0, 0.0))
        .with_rotation(Quat::from_rotation_y(0.1));
    let sub_grid = sub_grid_holding(&mut app, turned, long);

    let preview = run(&mut app, move |voxel_world| {
        let preview = voxel_world.stamp_preview(sub_grid, Grid::World);
        voxel_world
            .stamp(sub_grid, Grid::World, incoming_replaces)
            .unwrap();
        preview.unwrap()
    });
    until_stamped(&mut app, sub_grid);

    assert_eq!(preview.rotation, GridRotation::IDENTITY);
    // The centre of the solids, (21, 1.5, 1.5), sits at about z = -0.6 under
    // the turned pose, so the body lands two voxels along -z. A snap about
    // the grid origin would leave it at z = 1.
    let landed = read(
        &mut app,
        Grid::World,
        (1..41).map(|x| IVec3::new(x, 11, -1)).collect(),
    );
    assert!(
        landed.iter().all(|v| *v == WorldVoxel::Solid(5)),
        "{landed:?}"
    );
    let about_origin = read(
        &mut app,
        Grid::World,
        (1..41).map(|x| IVec3::new(x, 11, 1)).collect(),
    );
    assert!(about_origin.iter().all(|v| !v.is_solid()));
    let far_end = preview.pose.transform_point(Vec3::new(40.5, 1.5, 1.5));
    assert_eq!(far_end.floor().as_ivec3(), IVec3::new(40, 11, -1));
}

/// How a test moves the sub-grid before stamping it.
#[derive(Clone, Copy)]
enum Before {
    Nothing,
    MoveToPreviewInTheSameSystem,
    MoveToPreviewAFrameEarlier,
}

/// Stamps a turned sub-grid into the world, after doing what `before` says,
/// and returns the world's voxels around where it lands once it has.
fn stamp_after(before: Before) -> Vec<WorldVoxel> {
    let mut app = stamp_app();
    let mut voxels: Vec<IVec3> = (1..6).map(|x| IVec3::new(x, 1, 1)).collect();
    voxels.extend((2..5).map(|y| IVec3::new(1, y, 1)));
    voxels.push(IVec3::new(3, 3, 2));
    let turned = Transform::from_translation(Vec3::new(0.3, 9.6, -0.2))
        .with_rotation(Quat::from_rotation_y(0.08) * Quat::from_rotation_x(-0.06));
    let sub_grid = sub_grid_holding(&mut app, turned, voxels);
    let move_to_preview = |app: &mut App| {
        app.world_mut()
            .run_system_once(
                move |voxel_world: VoxelWorld<LiftWorld>,
                      mut transforms: Query<&mut Transform>| {
                    let preview =
                        voxel_world.stamp_preview(sub_grid, Grid::World).unwrap();
                    *transforms.get_mut(sub_grid).unwrap() = preview.pose;
                },
            )
            .unwrap();
    };
    match before {
        Before::Nothing => {}
        Before::MoveToPreviewInTheSameSystem => {}
        Before::MoveToPreviewAFrameEarlier => {
            move_to_preview(&mut app);
            app.update();
            app.update();
        }
    }
    app.world_mut()
        .run_system_once(
            move |mut voxel_world: VoxelWorld<LiftWorld>,
                  mut transforms: Query<&mut Transform>| {
                if let Before::MoveToPreviewInTheSameSystem = before {
                    let preview =
                        voxel_world.stamp_preview(sub_grid, Grid::World).unwrap();
                    *transforms.get_mut(sub_grid).unwrap() = preview.pose;
                }
                voxel_world
                    .stamp(sub_grid, Grid::World, incoming_replaces)
                    .unwrap();
            },
        )
        .unwrap();
    until_stamped(&mut app, sub_grid);
    read(
        &mut app,
        Grid::World,
        positions(IVec3::new(-3, 6, -4), IVec3::new(9, 16, 6)),
    )
}

#[test]
fn moving_to_the_preview_pose_and_stamping_lands_the_same_voxels() {
    let still = stamp_after(Before::Nothing);
    assert_eq!(
        still.iter().filter(|v| v.is_solid()).count(),
        9,
        "every voxel lands in the box"
    );
    assert_eq!(stamp_after(Before::MoveToPreviewInTheSameSystem), still);
    assert_eq!(stamp_after(Before::MoveToPreviewAFrameEarlier), still);
}

/// A sub-grid turned a quarter about y, which lands its voxel (1, 1, 1) on
/// the ground at (11, -2, 8) and its voxel (1, 4, 1) above it at (11, 1, 8).
fn quarter_turned(app: &mut App) -> Entity {
    let turned = Transform::from_translation(Vec3::new(10.1, -2.95, 9.9))
        .with_rotation(Quat::from_rotation_y(FRAC_PI_2));
    sub_grid_holding(app, turned, vec![IVec3::new(1, 1, 1), IVec3::new(1, 4, 1)])
}

const ON_GROUND: IVec3 = IVec3::new(11, -2, 8);
const ABOVE_GROUND: IVec3 = IVec3::new(11, 1, 8);

#[test]
fn the_rule_sees_each_voxel_and_none_keeps_the_targets_voxel() {
    let mut app = stamp_app();
    let sub_grid = quarter_turned(&mut app);

    let seen = run(&mut app, move |voxel_world| {
        let mut seen = Vec::new();
        voxel_world
            .stamp(sub_grid, Grid::World, |voxel| {
                seen.push(*voxel);
                (!voxel.existing.is_solid()).then_some(voxel.incoming)
            })
            .unwrap();
        seen
    });

    assert_eq!(seen.len(), 2);
    for voxel in &seen {
        assert_eq!(voxel.incoming, WorldVoxel::Solid(5));
        assert_eq!(voxel.rotation.apply(IVec3::X), IVec3::new(0, 0, -1));
        assert_eq!(voxel.rotation.apply(IVec3::Z), IVec3::X);
    }
    let on_ground = seen.iter().find(|v| v.position == ON_GROUND).unwrap();
    assert_eq!(on_ground.existing, ground(ON_GROUND));
    assert!(on_ground.existing.is_solid());
    let above = seen.iter().find(|v| v.position == ABOVE_GROUND).unwrap();
    assert!(!above.existing.is_solid());

    until_stamped(&mut app, sub_grid);
    assert_eq!(
        read(&mut app, Grid::World, vec![ON_GROUND, ABOVE_GROUND]),
        [ground(ON_GROUND), WorldVoxel::Solid(5)]
    );
}

#[test]
fn incoming_replaces_writes_the_incoming_voxel_over_a_solid() {
    let mut app = stamp_app();
    let sub_grid = quarter_turned(&mut app);
    run(&mut app, move |voxel_world| {
        voxel_world
            .stamp(sub_grid, Grid::World, incoming_replaces)
            .unwrap()
    });
    until_stamped(&mut app, sub_grid);
    assert_eq!(
        read(&mut app, Grid::World, vec![ON_GROUND, ABOVE_GROUND]),
        [WorldVoxel::Solid(5), WorldVoxel::Solid(5)]
    );
}

#[test]
fn stamp_refuses_a_dead_sub_grid_a_dead_target_and_a_target_inside_it() {
    let mut app = stamp_app();
    let sub_grid = sub_grid_holding(&mut app, Transform::default(), row());
    let dead = sub_grid_holding(&mut app, Transform::default(), row());
    app.world_mut().entity_mut(dead).despawn();
    let never = app.world_mut().spawn_empty().id();
    // A sub-grid under a plain entity under `sub_grid`, which the crate
    // finds once the pose snapshot has run.
    let holder = app.world_mut().spawn(ChildOf(sub_grid)).id();
    let deep = app.world_mut().spawn(ChildOf(holder)).id();
    run(&mut app, move |voxel_world| {
        voxel_world.insert_sub_grid(deep)
    });
    app.update();

    let (results, previews, child) = run(&mut app, move |voxel_world| {
        // Lifted in this system, so its parent is known from the call.
        let child = voxel_world
            .lift(
                Grid::SubGrid(sub_grid),
                IVec3::new(3, 4, 5),
                IVec3::new(4, 5, 6),
                |_, _| true,
            )
            .unwrap();
        let attempts = [
            (dead, Grid::World),
            (never, Grid::World),
            (sub_grid, Grid::SubGrid(never)),
            (sub_grid, Grid::SubGrid(dead)),
            (sub_grid, Grid::SubGrid(sub_grid)),
            (sub_grid, Grid::SubGrid(child)),
            (sub_grid, Grid::SubGrid(deep)),
        ];
        let previews: Vec<bool> = attempts
            .iter()
            .map(|(s, t)| voxel_world.stamp_preview(*s, *t).is_some())
            .collect();
        let results: Vec<Result<(), StampError>> = attempts
            .iter()
            .map(|(s, t)| voxel_world.stamp(*s, *t, incoming_replaces))
            .collect();
        (results, previews, child)
    });

    use StampError::*;
    assert_eq!(
        results,
        [
            Err(NoSuchSubGrid),
            Err(NoSuchSubGrid),
            Err(NoSuchTarget),
            Err(NoSuchTarget),
            Err(IntoItself),
            Err(IntoItself),
            Err(IntoItself),
        ]
    );
    assert_eq!(previews, [false; 7]);
    until_lifted(&mut app, child);
    app.update();
    let (after_snapshot, child_into_parent) = run(&mut app, move |voxel_world| {
        (
            voxel_world.stamp(sub_grid, Grid::SubGrid(child), incoming_replaces),
            voxel_world
                .stamp_preview(child, Grid::SubGrid(sub_grid))
                .is_some(),
        )
    });
    assert_eq!(after_snapshot, Err(IntoItself));
    assert!(child_into_parent);
    let left = read(&mut app, Grid::SubGrid(sub_grid), row());
    assert_eq!(left.iter().filter(|v| v.is_solid()).count(), 3);
    assert!(stamped(&app).is_empty());
}

#[test]
fn right_after_the_call_each_voxel_reads_from_one_grid() {
    let mut app = stamp_app();
    let above = Transform::from_translation(Vec3::new(0.0, 10.0, 0.0));
    let sub_grid = sub_grid_holding(&mut app, above, row());

    let (sub_grid_reads, bounds, world_reads) = run(&mut app, move |voxel_world| {
        voxel_world
            .stamp(sub_grid, Grid::World, incoming_replaces)
            .unwrap();
        let grid = voxel_world.grid(Grid::SubGrid(sub_grid)).unwrap();
        let sub_grid_reads: Vec<WorldVoxel> =
            row().iter().map(|p| grid.get_voxel(*p)).collect();
        let world_reads: Vec<WorldVoxel> = row()
            .iter()
            .map(|p| voxel_world.get_voxel(*p + IVec3::new(0, 10, 0)))
            .collect();
        (sub_grid_reads, grid.bounds(), world_reads)
    });

    assert!(sub_grid_reads.iter().all(|v| *v == WorldVoxel::Air));
    assert_eq!(bounds, None);
    assert!(world_reads.iter().all(|v| *v == WorldVoxel::Solid(5)));
}

#[test]
fn a_stamp_into_the_world_lands_in_one_frame_and_then_despawns_the_sub_grid() {
    let mut app = stamp_app();
    let since = next_frame(&app);
    let down = Transform::from_translation(Vec3::new(0.0, -5.0, 0.0));
    let sub_grid = sub_grid_holding(&mut app, down, row());
    let sub_grid_chunks = BTreeSet::from([[0, 0, 0]]);
    until_published(&mut app, since, Grid::SubGrid(sub_grid), &sub_grid_chunks);

    let commit = next_frame(&app);
    run(&mut app, move |voxel_world| {
        voxel_world
            .stamp(sub_grid, Grid::World, incoming_replaces)
            .unwrap()
    });
    let landed: Vec<IVec3> = row().iter().map(|p| *p - IVec3::new(0, 5, 0)).collect();
    let world_chunks = padded_chunks(landed[0], landed[3] + 1);
    until_published(&mut app, commit, Grid::World, &world_chunks);

    let mut frames = published_since(&app, commit, Grid::World, &world_chunks);
    let despawned = despawned_since(&app, commit, Grid::SubGrid(sub_grid));
    assert_eq!(despawned.len(), 1, "{despawned:?}");
    frames.extend(despawned);
    let landing = one_frame(&frames);
    assert!(landing > commit, "the slow world takes frames to land");
    assert_eq!(
        stamped(&app),
        [(landing, Stamped::new(sub_grid, Grid::World), true)].as_slice()
    );
    assert!(app.world().get_entity(sub_grid).is_err());
    let reads = read(&mut app, Grid::World, landed);
    assert!(
        reads.iter().all(|v| *v == WorldVoxel::Solid(5)),
        "{reads:?}"
    );
}

#[test]
fn a_stamp_of_an_empty_sub_grid_lands_in_the_next_frame() {
    let mut app = stamp_app();
    let sub_grid = sub_grid_holding(&mut app, Transform::default(), Vec::new());
    let call = next_frame(&app);
    run(&mut app, move |voxel_world| {
        voxel_world
            .stamp(sub_grid, Grid::World, incoming_replaces)
            .unwrap()
    });
    app.update();
    assert_eq!(
        stamped(&app),
        [(call, Stamped::new(sub_grid, Grid::World), true)].as_slice()
    );
    assert!(app.world().get_entity(sub_grid).is_err());
}

/// A sub-grid turned a quarter about y, holding a column of voxels, whose
/// chunks have published.
fn docking_target(app: &mut App) -> Entity {
    let turned = Transform::from_translation(Vec3::new(20.0, 3.0, 0.0))
        .with_rotation(Quat::from_rotation_y(FRAC_PI_2));
    let since = next_frame(app);
    let target =
        sub_grid_holding(app, turned, (1..4).map(|y| IVec3::new(1, y, 1)).collect());
    until_published(
        app,
        since,
        Grid::SubGrid(target),
        &BTreeSet::from([[0, 0, 0]]),
    );
    target
}

#[test]
fn docking_a_sub_grid_into_another_lands_as_one_handover() {
    let mut app = stamp_app();
    let target = docking_target(&mut app);
    // The same quarter turn, four voxels further along the target's x, so
    // the row lands from (4, 4, 5) to (7, 4, 5) in the target.
    let docking = Transform::from_translation(Vec3::new(20.0, 3.0, -1.0))
        .with_rotation(Quat::from_rotation_y(FRAC_PI_2));
    let since = next_frame(&app);
    let sub_grid = sub_grid_holding(&mut app, docking, row());
    until_published(
        &mut app,
        since,
        Grid::SubGrid(sub_grid),
        &BTreeSet::from([[0, 0, 0]]),
    );

    let commit = next_frame(&app);
    run(&mut app, move |voxel_world| {
        voxel_world
            .stamp(sub_grid, Grid::SubGrid(target), incoming_replaces)
            .unwrap()
    });
    until_stamped(&mut app, sub_grid);

    let mut frames = published_since(
        &app,
        commit,
        Grid::SubGrid(target),
        &BTreeSet::from([[0, 0, 0]]),
    );
    assert_eq!(frames.len(), 1, "{frames:?}");
    frames.extend(despawned_since(&app, commit, Grid::SubGrid(sub_grid)));
    assert_eq!(frames.len(), 2, "{frames:?}");
    let landing = one_frame(&frames);
    assert_eq!(stamped(&app)[0].0, landing);
    let docked: Vec<IVec3> = row().iter().map(|p| *p + IVec3::new(1, 0, 0)).collect();
    let reads = read(&mut app, Grid::SubGrid(target), docked);
    assert!(
        reads.iter().all(|v| *v == WorldVoxel::Solid(5)),
        "{reads:?}"
    );
}

#[test]
fn a_stamped_sub_grids_child_sub_grids_move_onto_the_target_keeping_their_pose() {
    let mut app = stamp_app();
    let target = docking_target(&mut app);
    let turned = Transform::from_translation(Vec3::new(2.0, 8.0, 1.0))
        .with_rotation(Quat::from_rotation_y(0.3));
    let sub_grid = sub_grid_holding(&mut app, turned, row());
    let child = run(&mut app, move |voxel_world| {
        voxel_world
            .lift(
                Grid::SubGrid(sub_grid),
                IVec3::new(6, 4, 5),
                IVec3::new(7, 5, 6),
                |_, _| true,
            )
            .unwrap()
    });
    until_lifted(&mut app, child);
    let other = app
        .world_mut()
        .spawn((Transform::default(), ChildOf(sub_grid)))
        .id();
    app.update();
    let was = *app.world().get::<GlobalTransform>(child).unwrap();

    run(&mut app, move |voxel_world| {
        voxel_world
            .stamp(sub_grid, Grid::SubGrid(target), incoming_replaces)
            .unwrap()
    });
    until_stamped(&mut app, sub_grid);
    app.update();

    assert!(app.world().get_entity(sub_grid).is_err());
    assert!(app.world().get_entity(other).is_err());
    assert_eq!(
        app.world().get::<ChildOf>(child).map(ChildOf::parent),
        Some(target)
    );
    assert_poses_match(*app.world().get::<GlobalTransform>(child).unwrap(), was);
    let kept = read(&mut app, Grid::SubGrid(child), vec![IVec3::ONE]);
    assert_eq!(kept, [WorldVoxel::Solid(5)]);
}

#[test]
fn lifting_from_the_world_lands_as_one_handover() {
    let mut app = stamp_app();
    let (min, max) = (IVec3::new(-1, -2, -1), IVec3::new(2, 0, 2));
    let commit = next_frame(&app);
    let sub_grid = run(&mut app, move |voxel_world| {
        voxel_world
            .lift(Grid::World, min, max, |_, _| true)
            .unwrap()
    });
    let world_chunks = padded_chunks(min, max);
    until_lifted(&mut app, sub_grid);

    let mut frames = published_since(&app, commit, Grid::World, &world_chunks);
    frames.extend(published_since(
        &app,
        commit,
        Grid::SubGrid(sub_grid),
        &BTreeSet::from([[0, 0, 0]]),
    ));
    assert_eq!(frames.len(), world_chunks.len() + 1, "{frames:?}");
    assert_eq!(one_frame(&frames), lifted(&app)[0].0);
    let origin = min - 1;
    let solids: Vec<IVec3> = positions(min, max)
        .into_iter()
        .filter(|p| ground(*p).is_solid())
        .collect();
    let expected: Vec<WorldVoxel> = solids.iter().map(|p| ground(*p)).collect();
    let back = read(
        &mut app,
        Grid::SubGrid(sub_grid),
        solids.iter().map(|p| *p - origin).collect(),
    );
    assert_eq!(back, expected);
}

/// A sub-grid holding [`row`] and the same row two voxels along z, five
/// voxels down, whose chunk has published.
fn two_rows(app: &mut App) -> Entity {
    let mut voxels = row();
    voxels.extend(row().iter().map(|p| *p + IVec3::new(0, 0, 2)));
    let since = next_frame(app);
    let down = Transform::from_translation(Vec3::new(0.0, -5.0, 0.0));
    let sub_grid = sub_grid_holding(app, down, voxels);
    until_published(
        app,
        since,
        Grid::SubGrid(sub_grid),
        &BTreeSet::from([[0, 0, 0]]),
    );
    sub_grid
}

/// The box holding the second of [`two_rows`].
fn second_row() -> (IVec3, IVec3) {
    (IVec3::new(3, 4, 7), IVec3::new(7, 5, 8))
}

#[test]
fn lifting_from_a_sub_grid_into_a_new_one_lands_as_one_handover() {
    let mut app = stamp_app();
    let source = two_rows(&mut app);
    let commit = next_frame(&app);
    let (min, max) = second_row();
    let piece = run(&mut app, move |voxel_world| {
        voxel_world
            .lift(Grid::SubGrid(source), min, max, |_, _| true)
            .unwrap()
    });
    until_lifted(&mut app, piece);

    let chunk = BTreeSet::from([[0, 0, 0]]);
    let mut frames = published_since(&app, commit, Grid::SubGrid(source), &chunk);
    frames.extend(published_since(&app, commit, Grid::SubGrid(piece), &chunk));
    assert_eq!(frames.len(), 2, "{frames:?}");
    assert_eq!(one_frame(&frames), lifted(&app)[0].0);
    let back = read(
        &mut app,
        Grid::SubGrid(piece),
        (1..5).map(|x| IVec3::new(x, 1, 1)).collect(),
    );
    assert!(back.iter().all(|v| *v == WorldVoxel::Solid(5)), "{back:?}");
    let left = read(&mut app, Grid::SubGrid(source), positions(min, max));
    assert!(left.iter().all(|v| *v == WorldVoxel::Air));
}

#[test]
fn a_lift_from_a_sub_grid_and_a_stamp_of_the_piece_in_one_system_land_together() {
    let mut app = stamp_app();
    let source = two_rows(&mut app);
    let commit = next_frame(&app);
    let (min, max) = second_row();
    let piece = run(&mut app, move |voxel_world| {
        let piece = voxel_world
            .lift(Grid::SubGrid(source), min, max, |_, _| true)
            .unwrap();
        voxel_world
            .stamp(piece, Grid::World, incoming_replaces)
            .unwrap();
        piece
    });
    until_stamped(&mut app, piece);

    // The second row lands five voxels down, on the ground.
    let landed: Vec<IVec3> = positions(min, max)
        .into_iter()
        .map(|p| p - IVec3::new(0, 5, 0))
        .collect();
    let world_chunks = padded_chunks(landed[0], landed[3] + 1);
    let mut frames = published_since(&app, commit, Grid::World, &world_chunks);
    assert_eq!(frames.len(), world_chunks.len(), "{frames:?}");
    frames.extend(published_since(
        &app,
        commit,
        Grid::SubGrid(source),
        &BTreeSet::from([[0, 0, 0]]),
    ));
    assert_eq!(frames.len(), world_chunks.len() + 1, "{frames:?}");
    let landing = one_frame(&frames);
    assert_eq!(
        lifted(&app),
        [(landing, Lifted::new(Grid::SubGrid(source), piece))].as_slice()
    );
    assert_eq!(
        stamped(&app),
        [(landing, Stamped::new(piece, Grid::World), true)].as_slice()
    );
    assert!(app.world().get_entity(piece).is_err());
    let reads = read(&mut app, Grid::World, landed);
    assert!(
        reads.iter().all(|v| *v == WorldVoxel::Solid(5)),
        "{reads:?}"
    );
}
