//! Tests of raycasts across grids and of the point query, through
//! `VoxelWorld`.

use std::f32::consts::FRAC_PI_2;

use bevy::ecs::system::RunSystemOnce;
use bevy::prelude::*;

use super::{floor_app, run, set, spawn, sub_grid_app, FloorWorld};
use crate::prelude::*;

/// Casts `ray` across every grid of the floor world, with no filter.
fn cast(app: &mut App, ray: Ray3d, max_distance: f32) -> Option<GridRaycastHit> {
    run::<FloorWorld, _>(app, move |voxel_world| {
        voxel_world.raycast_grids(ray, max_distance, &|_| true, &|_, _, _| true)
    })
}

fn ray(origin: Vec3, direction: Vec3) -> Ray3d {
    Ray3d::new(origin, Dir3::new(direction).unwrap())
}

/// Whether `hit` is `expected`, with its point, distance and world normal
/// within rounding of it.
fn is_near(hit: &GridRaycastHit, expected: &GridRaycastHit) -> bool {
    let normals = match (hit.world_normal, expected.world_normal) {
        (Some(a), Some(b)) => a.abs_diff_eq(b, 1e-5),
        (a, b) => a == b,
    };
    hit.grid == expected.grid
        && hit.position == expected.position
        && hit.normal == expected.normal
        && hit.voxel == expected.voxel
        && normals
        && hit.point.abs_diff_eq(expected.point, 1e-4)
        && (hit.distance - expected.distance).abs() < 1e-4
}

#[track_caller]
fn assert_hit(hit: Option<GridRaycastHit>, expected: GridRaycastHit) {
    let hit = hit.expect("a hit");
    assert!(is_near(&hit, &expected), "{hit:?} is not {expected:?}");
}

#[test]
fn a_ray_hits_a_rotated_and_translated_sub_grid_at_the_voxel_its_pose_implies() {
    let mut app = sub_grid_app();
    // Turned a quarter about y, so the grid's +x points along the root's -z
    // and its +z along the root's +x.
    let turned = Transform::from_xyz(5.0, 4.0, 2.0)
        .with_rotation(Quat::from_rotation_y(FRAC_PI_2));
    let entity = spawn(&mut app, turned);
    // Fill the root's boxes x 5..6, y 4..5, z 1..2 and z 0..1.
    set(
        &mut app,
        entity,
        &[IVec3::ZERO, IVec3::X],
        WorldVoxel::Solid(3),
    );
    app.update();

    let along_z = cast(&mut app, ray(Vec3::new(5.5, 4.5, 20.0), Vec3::NEG_Z), 100.0);
    assert_hit(
        along_z,
        GridRaycastHit {
            grid: Grid::SubGrid(entity),
            position: IVec3::ZERO,
            normal: Some(IVec3::NEG_X),
            world_normal: Some(Vec3::Z),
            point: Vec3::new(5.5, 4.5, 2.0),
            distance: 18.0,
            voxel: WorldVoxel::Solid(3),
        },
    );

    let down = cast(&mut app, ray(Vec3::new(5.5, 10.0, 0.5), Vec3::NEG_Y), 100.0);
    assert_hit(
        down,
        GridRaycastHit {
            grid: Grid::SubGrid(entity),
            position: IVec3::X,
            normal: Some(IVec3::Y),
            world_normal: Some(Vec3::Y),
            point: Vec3::new(5.5, 5.0, 0.5),
            distance: 5.0,
            voxel: WorldVoxel::Solid(3),
        },
    );
}

#[test]
fn under_a_scaled_pose_the_distance_is_measured_in_the_roots_frame() {
    let mut app = sub_grid_app();
    // Each voxel is two units wide in the root's frame.
    let scaled = Transform::from_xyz(3.0, 2.0, -6.0)
        .with_rotation(Quat::from_rotation_y(FRAC_PI_2))
        .with_scale(Vec3::splat(2.0));
    let entity = spawn(&mut app, scaled);
    // Fill the root's boxes x 3..5, y 2..4, z -8..-6 and z -10..-8.
    set(
        &mut app,
        entity,
        &[IVec3::ZERO, IVec3::X],
        WorldVoxel::Solid(3),
    );
    app.update();

    // Meets the plane x = 3 before the box, then the face z = -6 four
    // steps along (1, 0, -2), at x = 4.
    let oblique = cast(
        &mut app,
        ray(Vec3::new(0.0, 3.0, 2.0), Vec3::new(1.0, 0.0, -2.0)),
        100.0,
    );
    assert_hit(
        oblique,
        GridRaycastHit {
            grid: Grid::SubGrid(entity),
            position: IVec3::ZERO,
            normal: Some(IVec3::NEG_X),
            world_normal: Some(Vec3::Z),
            point: Vec3::new(4.0, 3.0, -6.0),
            distance: 4.0 * 5.0_f32.sqrt(),
            voxel: WorldVoxel::Solid(3),
        },
    );
}

/// The floor world with three sub-grids at the identity of the root's frame
/// above, in and below the floor, each holding one voxel.
struct Stack {
    app: App,
    /// Holds (6, 2, 6), above the floor.
    above: Entity,
    /// Holds (2, -1, 2), a voxel the floor holds too.
    level: Entity,
    /// Holds (6, -5, 6), inside the floor.
    below: Entity,
}

fn stack() -> Stack {
    let mut app = floor_app();
    let mut one_voxel = |position: IVec3, material| {
        let entity = spawn(&mut app, Transform::IDENTITY);
        set(&mut app, entity, &[position], WorldVoxel::Solid(material));
        entity
    };
    let above = one_voxel(IVec3::new(6, 2, 6), 3);
    let level = one_voxel(IVec3::new(2, -1, 2), 4);
    let below = one_voxel(IVec3::new(6, -5, 6), 5);
    app.update();
    Stack {
        app,
        above,
        level,
        below,
    }
}

/// Straight down onto (6, y, 6), meeting the voxel above the floor at
/// distance 7, the floor at 10 and the voxel inside it at 14.
const DOWN_AT_6: Ray3d = Ray3d {
    origin: Vec3::new(6.5, 10.0, 6.5),
    direction: Dir3::NEG_Y,
};

/// Straight down onto (2, y, 2), meeting the floor at distance 10.
const DOWN_AT_2: Ray3d = Ray3d {
    origin: Vec3::new(2.5, 10.0, 2.5),
    direction: Dir3::NEG_Y,
};

/// Straight down onto (9, y, 9), where only the floor is.
const DOWN_AT_9: Ray3d = Ray3d {
    origin: Vec3::new(9.5, 10.0, 9.5),
    direction: Dir3::NEG_Y,
};

/// The grid and voxel position `ray` hits across every grid, with
/// `grid_filter`.
fn hit_with(
    app: &mut App,
    ray: Ray3d,
    max_distance: f32,
    grid_filter: impl Fn(Grid) -> bool + Send + Sync + 'static,
) -> Option<(Grid, IVec3)> {
    run::<FloorWorld, _>(app, move |voxel_world| {
        voxel_world
            .raycast_grids(ray, max_distance, &grid_filter, &|_, _, _| true)
            .map(|hit| (hit.grid, hit.position))
    })
}

#[test]
fn the_nearest_hit_across_the_world_and_every_sub_grid_wins_and_a_tie_goes_to_the_sub_grid(
) {
    let Stack {
        mut app,
        above,
        level,
        ..
    } = stack();
    let any = |_| true;
    assert_eq!(
        hit_with(&mut app, DOWN_AT_6, 100.0, any),
        Some((Grid::SubGrid(above), IVec3::new(6, 2, 6))),
        "the voxel above the floor",
    );
    assert_eq!(
        hit_with(&mut app, DOWN_AT_9, 100.0, any),
        Some((Grid::World, IVec3::new(9, -1, 9))),
    );
    let tie = cast(&mut app, DOWN_AT_2, 100.0).unwrap();
    assert_eq!(
        (tie.grid, tie.position, tie.distance),
        (Grid::SubGrid(level), IVec3::new(2, -1, 2), 10.0),
        "the sub-grid holding the floor's voxel"
    );
}

#[test]
fn the_world_hit_gives_its_point_normal_and_distance() {
    let mut app = floor_app();
    assert_hit(
        cast(&mut app, DOWN_AT_9, 100.0),
        GridRaycastHit {
            grid: Grid::World,
            position: IVec3::new(9, -1, 9),
            normal: Some(IVec3::Y),
            world_normal: Some(Vec3::Y),
            point: Vec3::new(9.5, 0.0, 9.5),
            distance: 10.0,
            voxel: WorldVoxel::Solid(1),
        },
    );
}

#[test]
fn the_grid_filter_runs_once_per_grid_and_skips_the_grids_it_rejects() {
    let Stack {
        mut app,
        above,
        level,
        below,
    } = stack();

    let asked = run::<FloorWorld, _>(&mut app, |voxel_world| {
        let asked = std::cell::RefCell::new(Vec::new());
        voxel_world.raycast_grids(
            DOWN_AT_6,
            100.0,
            &|grid| {
                asked.borrow_mut().push(grid);
                true
            },
            &|_, _, _| true,
        );
        asked.into_inner()
    });
    assert_eq!(asked.len(), 4, "asked {asked:?}");
    for grid in [
        Grid::World,
        Grid::SubGrid(above),
        Grid::SubGrid(level),
        Grid::SubGrid(below),
    ] {
        assert!(asked.contains(&grid), "{grid:?} is never asked about");
    }

    assert_eq!(
        hit_with(&mut app, DOWN_AT_6, 100.0, move |grid| grid
            != Grid::SubGrid(above)),
        Some((Grid::World, IVec3::new(6, -1, 6))),
    );
    assert_eq!(
        hit_with(&mut app, DOWN_AT_6, 100.0, move |grid| grid
            == Grid::SubGrid(below)),
        Some((Grid::SubGrid(below), IVec3::new(6, -5, 6))),
        "the world is skipped too",
    );
    assert_eq!(
        hit_with(&mut app, DOWN_AT_9, 100.0, |grid| grid != Grid::World),
        None,
    );
}

#[test]
fn the_ray_passes_through_a_voxel_the_voxel_filter_rejects() {
    let Stack {
        mut app,
        above,
        below,
        ..
    } = stack();
    let (seen, hit) = run::<FloorWorld, _>(&mut app, |voxel_world| {
        let seen = std::cell::RefCell::new(Vec::new());
        let hit = voxel_world.raycast_grids(
            DOWN_AT_6,
            100.0,
            &|_| true,
            &|grid, position, voxel| {
                seen.borrow_mut().push((grid, position, voxel));
                // Pass through the sub-grid above the floor, and through the
                // floor's own voxels.
                voxel != WorldVoxel::Solid(3) && grid != Grid::World
            },
        );
        (seen.into_inner(), hit.map(|hit| (hit.grid, hit.position)))
    });
    assert_eq!(hit, Some((Grid::SubGrid(below), IVec3::new(6, -5, 6))));
    assert!(
        seen.contains(&(
            Grid::SubGrid(above),
            IVec3::new(6, 2, 6),
            WorldVoxel::Solid(3)
        )),
        "the filter sees the sub-grid's voxel, in {seen:?}"
    );
    assert!(
        seen.contains(&(Grid::World, IVec3::new(6, -1, 6), WorldVoxel::Solid(1))),
        "the filter sees the floor's voxel, in {seen:?}"
    );
}

#[test]
fn no_hit_beyond_the_maximum_distance_is_returned() {
    let Stack {
        mut app,
        above,
        below,
        ..
    } = stack();
    let any = |_| true;
    assert_eq!(hit_with(&mut app, DOWN_AT_6, 6.9, any), None);
    assert_eq!(
        hit_with(&mut app, DOWN_AT_6, 7.0, any),
        Some((Grid::SubGrid(above), IVec3::new(6, 2, 6))),
    );
    let past_above = move |grid| grid != Grid::SubGrid(above);
    assert_eq!(hit_with(&mut app, DOWN_AT_6, 9.9, past_above), None);
    let only_below = move |grid| grid == Grid::SubGrid(below);
    assert_eq!(hit_with(&mut app, DOWN_AT_6, 13.9, only_below), None);
    assert_eq!(
        hit_with(&mut app, DOWN_AT_6, 14.0, only_below),
        Some((Grid::SubGrid(below), IVec3::new(6, -5, 6))),
    );
}

#[test]
fn the_sendable_raycast_agrees_with_the_raycast_on_another_thread() {
    let Stack { mut app, above, .. } = stack();
    let rays = [
        DOWN_AT_6,
        DOWN_AT_2,
        DOWN_AT_9,
        ray(Vec3::new(20.0, 2.5, 6.5), Vec3::new(-1.0, 0.1, 0.05)),
        ray(Vec3::new(0.0, 8.0, 0.0), Vec3::new(1.0, -1.0, 1.0)),
    ];
    let skip_above = move |grid| grid != Grid::SubGrid(above);
    let (here, cast_fn) = run::<FloorWorld, _>(&mut app, move |voxel_world| {
        let here: Vec<_> = rays
            .iter()
            .flat_map(|ray| {
                [
                    voxel_world.raycast_grids(*ray, 100.0, &|_| true, &|_, _, _| true),
                    voxel_world.raycast_grids(*ray, 100.0, &skip_above, &|_, _, _| true),
                ]
            })
            .collect();
        (here, voxel_world.raycast_grids_fn())
    });
    let there = std::thread::spawn(move || {
        rays.iter()
            .flat_map(|ray| {
                [
                    cast_fn(*ray, 100.0, &|_| true, &|_, _, _| true),
                    cast_fn(*ray, 100.0, &skip_above, &|_, _, _| true),
                ]
            })
            .collect::<Vec<_>>()
    })
    .join()
    .unwrap();
    assert!(here.iter().flatten().count() >= 8, "the rays hit: {here:?}");
    assert_eq!(here, there);
}

#[test]
fn the_world_raycasts_pass_through_a_sub_grid_to_the_world_voxel_behind_it() {
    let Stack { mut app, .. } = stack();
    let (hit, hit_fn) = run::<FloorWorld, _>(&mut app, |voxel_world| {
        (
            voxel_world.raycast(DOWN_AT_6, &|_| true),
            voxel_world.raycast_fn()(DOWN_AT_6, &|_: (Vec3, WorldVoxel)| true),
        )
    });
    let floor = Some(Vec3::new(6.0, -1.0, 6.0));
    assert_eq!(hit.map(|hit| hit.position), floor);
    assert_eq!(hit_fn.map(|hit| hit.position), floor);
}

#[test]
fn the_point_query_finds_a_point_in_a_rotated_hollow_sub_grids_air_and_none_outside_it() {
    let mut app = sub_grid_app();
    let turned = Transform::from_xyz(10.0, 5.0, 10.0)
        .with_rotation(Quat::from_rotation_y(std::f32::consts::FRAC_PI_4));
    let cabin = spawn(&mut app, turned);
    // A 3 by 3 by 3 box of walls around one voxel of air.
    let walls: Vec<IVec3> = (0..27)
        .map(|i| IVec3::new(i % 3, i / 3 % 3, i / 9))
        .filter(|position| *position != IVec3::ONE)
        .collect();
    set(&mut app, cabin, &walls, WorldVoxel::Solid(3));
    // A sub-grid with no solid voxel has no box to hold the point.
    spawn(&mut app, turned);
    app.update();

    let at = |app: &mut App, point: Vec3| {
        run::<FloorWorld, _>(app, move |voxel_world| {
            voxel_world.sub_grids_at_point(point).collect::<Vec<_>>()
        })
    };
    let pose = turned.compute_affine();
    let room = pose.transform_point3(Vec3::splat(1.5));
    assert_eq!(at(&mut app, room), vec![cabin]);
    // Past the middle of the box's -x face, but inside the root-aligned box
    // around the turned one.
    let beside = pose.transform_point3(Vec3::new(-0.5, 1.5, 1.5));
    assert_eq!(at(&mut app, beside), vec![]);
}

#[test]
fn a_ray_cast_in_the_frame_a_transform_changes_hits_the_sub_grid_at_last_frames_pose() {
    let mut app = floor_app();
    let entity = spawn(&mut app, Transform::from_xyz(6.0, 2.0, 6.0));
    set(&mut app, entity, &[IVec3::ZERO], WorldVoxel::Solid(3));
    app.update();

    let moved_to = Vec3::new(20.0, 2.0, 6.0);
    let (old, new) = app
        .world_mut()
        .run_system_once(
            move |mut transforms: Query<&mut Transform>,
                  voxel_world: VoxelWorld<FloorWorld>| {
                transforms.get_mut(entity).unwrap().translation = moved_to;
                let hit = |ray| {
                    voxel_world
                        .raycast_grids(ray, 100.0, &|_| true, &|_, _, _| true)
                        .map(|hit| hit.grid)
                };
                let over_new = Ray3d {
                    origin: moved_to + Vec3::new(0.5, 8.0, 0.5),
                    ..DOWN_AT_6
                };
                (hit(DOWN_AT_6), hit(over_new))
            },
        )
        .unwrap();
    assert_eq!(old, Some(Grid::SubGrid(entity)), "at last frame's pose");
    assert_eq!(new, Some(Grid::World), "not at this frame's");

    app.update();
    let over_new = Ray3d {
        origin: moved_to + Vec3::new(0.5, 8.0, 0.5),
        ..DOWN_AT_6
    };
    assert_eq!(
        hit_with(&mut app, over_new, 100.0, |_| true).map(|(grid, _)| grid),
        Some(Grid::SubGrid(entity)),
        "at the new pose a frame later"
    );
}

#[test]
fn a_sub_grid_spawned_this_frame_is_hit_at_the_pose_it_was_given() {
    let mut app = floor_app();
    let (entity, hit) = run::<FloorWorld, _>(&mut app, |voxel_world| {
        let entity = voxel_world.spawn_sub_grid(Transform::from_xyz(6.0, 2.0, 6.0));
        voxel_world
            .grid_mut(Grid::SubGrid(entity))
            .unwrap()
            .set_voxel(IVec3::ZERO, WorldVoxel::Solid(3));
        let hit = voxel_world.raycast_grids(DOWN_AT_6, 100.0, &|_| true, &|_, _, _| true);
        (
            entity,
            hit.map(|hit| (hit.grid, hit.position, hit.distance)),
        )
    });
    assert_eq!(hit, Some((Grid::SubGrid(entity), IVec3::ZERO, 7.0)));
}
