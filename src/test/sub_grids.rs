//! Tests of sub-grids through `VoxelWorld`: spawning them, reading and
//! editing them, their bounds and pose, and how their voxels stay out of the
//! world's grid.

use bevy::ecs::system::RunSystemOnce;
use bevy::prelude::*;

use super::{
    chunk_data, chunk_status, spawn_camera, test_plugins, update_until, FloorWorld,
};
use crate::chunk_map::ChunkMap;
use crate::grid::{GridId, GridKey};
use crate::prelude::*;
use crate::voxel_edits::VoxelEdits;
use crate::voxel_world_internal::WorldRoot;

mod raycasts;

const OTHER: GridId = GridId::new(1);

/// A voxel the world shows as air, and the other grid holds as solid.
const OTHER_EDIT: IVec3 = IVec3::new(0, 5, 0);

fn edits(app: &App) -> &VoxelEdits<FloorWorld, u8> {
    app.world().resource::<VoxelEdits<FloorWorld, u8>>()
}

/// The other grid's committed edit at [`OTHER_EDIT`], if it has one.
fn other_edit(app: &App) -> Option<WorldVoxel> {
    edits(app).reader_in(OTHER).get(OTHER_EDIT)
}

#[test]
fn another_grids_edits_and_chunks_never_show_through_the_worlds_methods() {
    let mut app = App::new();
    app.add_plugins((test_plugins(), VoxelWorldPlugin::<FloorWorld>::minimal()))
        .add_systems(Startup, spawn_camera::<FloorWorld>);
    update_until(
        &mut app,
        "the chunks at and below the origin to be ready",
        |app| {
            [IVec3::ZERO, IVec3::NEG_Y].into_iter().all(|chunk| {
                chunk_status::<FloorWorld>(app, chunk) == Some(ChunkStatus::Ready)
            })
        },
    );

    // The other grid's chunk at the origin is solid, where the world's is air.
    let below = ChunkKey::new(IVec3::NEG_Y, 0);
    let solid = chunk_data::<FloorWorld>(&mut app, below.position).unwrap();
    let origin = ChunkKey::new(IVec3::ZERO, 0);
    app.world()
        .resource::<ChunkMap<FloorWorld, u8>>()
        .get_map()
        .write()
        .unwrap()
        .insert(
            GridKey::new(OTHER, origin),
            crate::chunk::ChunkData {
                key: origin,
                ..solid
            },
        );
    edits(&app).set_in(OTHER, OTHER_EDIT, WorldVoxel::Solid(2));
    edits(&app).commit_in(OTHER);
    app.update();

    let (voxel, edited, delta, down, across) = app
        .world_mut()
        .run_system_once(|voxel_world: VoxelWorld<FloorWorld>| {
            let cast = |origin, direction| {
                voxel_world
                    .raycast(Ray3d { origin, direction }, &|_| true)
                    .map(|hit| hit.position)
            };
            (
                voxel_world.get_voxel(OTHER_EDIT),
                voxel_world.edited_chunks(),
                voxel_world.full_delta(IVec3::ZERO),
                cast(Vec3::new(1.5, 20.0, 1.5), -Dir3::Y),
                cast(Vec3::new(0.5, 5.5, 20.0), -Dir3::Z),
            )
        })
        .unwrap();
    assert_eq!(voxel, WorldVoxel::Air);
    assert_eq!(edited, vec![]);
    assert!(delta.is_empty() && delta.seq == 0);
    assert_eq!(
        down,
        Some(Vec3::new(1.0, -1.0, 1.0)),
        "hits the world's floor"
    );
    assert_eq!(across, None, "passes the other grid's edit");

    app.world_mut()
        .run_system_once(|mut voxel_world: VoxelWorld<FloorWorld>| {
            voxel_world.clear_edits();
        })
        .unwrap();
    app.update();
    assert_eq!(
        other_edit(&app),
        Some(WorldVoxel::Solid(2)),
        "after a clear"
    );

    app.world_mut()
        .run_system_once(|mut voxel_world: VoxelWorld<FloorWorld>| {
            voxel_world.forget_edits();
        })
        .unwrap();
    app.update();
    assert_eq!(
        other_edit(&app),
        Some(WorldVoxel::Solid(2)),
        "after a forget"
    );
    assert_eq!(
        edits(&app).edited_chunks_in(OTHER),
        vec![(IVec3::ZERO, 1)],
        "the other grid keeps its edit sequence"
    );
}

/// A world with no camera, so it spawns no chunks of its own, whose
/// transforms propagate.
fn sub_grid_app() -> App {
    let mut app = App::new();
    app.add_plugins((
        test_plugins(),
        TransformPlugin,
        VoxelWorldPlugin::<FloorWorld>::minimal(),
    ));
    app.update();
    app
}

/// [`sub_grid_app`] with a camera, run until the world's chunks at and below
/// the origin are ready.
fn floor_app() -> App {
    let mut app = sub_grid_app();
    app.world_mut()
        .run_system_once(spawn_camera::<FloorWorld>)
        .unwrap();
    update_until(
        &mut app,
        "the chunks at and below the origin to be ready",
        |app| {
            [IVec3::ZERO, IVec3::NEG_Y].into_iter().all(|chunk| {
                chunk_status::<FloorWorld>(app, chunk) == Some(ChunkStatus::Ready)
            })
        },
    );
    app
}

/// Runs `f` on the voxel world `C` in a system of its own, whose commands
/// apply when it returns.
fn run<C: VoxelWorldConfig, T: 'static>(
    app: &mut App,
    mut f: impl FnMut(&mut VoxelWorld<C>) -> T + Send + Sync + 'static,
) -> T {
    app.world_mut()
        .run_system_once(move |mut voxel_world: VoxelWorld<C>| f(&mut voxel_world))
        .unwrap()
}

/// The voxel at `position` of the sub-grid `entity`, or `None` if it isn't
/// a live sub-grid.
fn voxel(app: &mut App, entity: Entity, position: IVec3) -> Option<WorldVoxel> {
    run::<FloorWorld, _>(app, move |voxel_world| {
        voxel_world
            .grid(Grid::SubGrid(entity))
            .map(|grid| grid.get_voxel(position))
    })
}

fn bounds(app: &mut App, entity: Entity) -> Option<(IVec3, IVec3)> {
    run::<FloorWorld, _>(app, move |voxel_world| {
        voxel_world.grid(Grid::SubGrid(entity)).unwrap().bounds()
    })
}

fn pose(app: &mut App, entity: Entity) -> GlobalTransform {
    run::<FloorWorld, _>(app, move |voxel_world| {
        voxel_world.grid(Grid::SubGrid(entity)).unwrap().pose()
    })
}

/// Sets `voxel` at each of `positions` in the sub-grid `entity`.
fn set(app: &mut App, entity: Entity, positions: &[IVec3], voxel: WorldVoxel) {
    let positions = positions.to_vec();
    run::<FloorWorld, _>(app, move |voxel_world| {
        let mut grid = voxel_world.grid_mut(Grid::SubGrid(entity)).unwrap();
        for position in &positions {
            grid.set_voxel(*position, voxel);
        }
    });
}

fn spawn(app: &mut App, transform: Transform) -> Entity {
    run::<FloorWorld, _>(app, move |voxel_world| {
        voxel_world.spawn_sub_grid(transform)
    })
}

fn root(app: &mut App) -> Entity {
    app.world_mut()
        .query_filtered::<Entity, With<WorldRoot<FloorWorld>>>()
        .single(app.world())
        .unwrap()
}

/// A voxel of a sub-grid, in the chunk at the origin.
const A: IVec3 = IVec3::new(1, 2, 3);
/// A voxel of a sub-grid, in a chunk away from [`A`]'s.
const B: IVec3 = IVec3::new(-40, 7, 33);

#[test]
fn a_spawned_sub_grid_is_filled_and_read_in_the_system_that_spawns_it_and_after() {
    let mut app = sub_grid_app();
    let transform = Transform::from_xyz(4.0, 5.0, 6.0);

    let (entity, read) = run::<FloorWorld, _>(&mut app, move |voxel_world| {
        let entity = voxel_world.spawn_sub_grid(transform);
        voxel_world
            .grid_mut(Grid::SubGrid(entity))
            .unwrap()
            .set_voxel(A, WorldVoxel::Solid(3));
        let read = voxel_world
            .grid(Grid::SubGrid(entity))
            .unwrap()
            .get_voxel(A);
        (entity, read)
    });
    assert_eq!(read, WorldVoxel::Solid(3));

    let root = root(&mut app);
    let spawned = app.world().entity(entity);
    assert!(spawned.contains::<SubGrid<FloorWorld>>());
    assert!(spawned.contains::<Visibility>());
    assert_eq!(spawned.get::<Transform>(), Some(&transform));
    assert_eq!(spawned.get::<ChildOf>().map(ChildOf::parent), Some(root));

    for frame in 0..2 {
        app.update();
        assert_eq!(
            voxel(&mut app, entity, A),
            Some(WorldVoxel::Solid(3)),
            "{frame}"
        );
    }
}

#[test]
fn an_entity_made_a_sub_grid_keeps_its_parent_and_is_filled_at_once() {
    let mut app = sub_grid_app();
    let parent = app.world_mut().spawn(Transform::default()).id();
    let transform = Transform::from_xyz(1.0, 0.0, 0.0);
    let entity = app.world_mut().spawn((transform, ChildOf(parent))).id();

    let read = run::<FloorWorld, _>(&mut app, move |voxel_world| {
        voxel_world.insert_sub_grid(entity);
        voxel_world
            .grid_mut(Grid::SubGrid(entity))
            .unwrap()
            .set_voxel(A, WorldVoxel::Solid(3));
        voxel_world
            .grid(Grid::SubGrid(entity))
            .unwrap()
            .get_voxel(A)
    });
    assert_eq!(read, WorldVoxel::Solid(3));

    let inserted = app.world().entity(entity);
    assert!(inserted.contains::<SubGrid<FloorWorld>>());
    assert!(inserted.contains::<Visibility>());
    assert_eq!(inserted.get::<Transform>(), Some(&transform));
    assert_eq!(inserted.get::<ChildOf>().map(ChildOf::parent), Some(parent));

    app.update();
    assert_eq!(voxel(&mut app, entity, A), Some(WorldVoxel::Solid(3)));
}

#[test]
fn the_worlds_grid_reads_and_edits_what_the_worlds_own_methods_do() {
    let mut app = floor_app();
    let (above, floor) = (IVec3::new(2, 5, 2), IVec3::new(2, -3, 2));

    let reads = run::<FloorWorld, _>(&mut app, move |voxel_world| {
        let mut world = voxel_world.grid_mut(Grid::World).unwrap();
        world.set_voxel(above, WorldVoxel::Solid(4));
        world.set_voxels_in(A, A + 2, |_| Some(WorldVoxel::Solid(5)));
        world.set_voxel(B, WorldVoxel::Solid(6));
        world.clear_voxels_in(B, B + 1);
        voxel_world.set_voxel(floor, WorldVoxel::Solid(7));

        let world = voxel_world.grid(Grid::World).unwrap();
        let read_fn = world.get_voxel_fn();
        [
            above,
            floor,
            A,
            A + 1,
            B,
            IVec3::new(9, 9, 9),
            IVec3::new(9, -9, 9),
        ]
        .map(|position| {
            (
                world.get_voxel(position),
                read_fn(position),
                voxel_world.get_voxel(position),
            )
        })
    });
    let expected = [
        Some(WorldVoxel::Solid(4)),
        Some(WorldVoxel::Solid(7)),
        Some(WorldVoxel::Solid(5)),
        Some(WorldVoxel::Solid(5)),
        // Cleared: whatever the world shows there.
        None,
        Some(WorldVoxel::Air),
        Some(WorldVoxel::Solid(1)),
    ];
    for ((grid, read_fn, world), expected) in reads.into_iter().zip(expected) {
        assert_eq!((grid, read_fn), (world, world));
        assert!(
            expected.is_none_or(|expected| world == expected),
            "{world:?}"
        );
    }

    app.update();
    let (edited, bounds, pose) = run::<FloorWorld, _>(&mut app, |voxel_world| {
        let world = voxel_world.grid(Grid::World).unwrap();
        let mut edited = voxel_world.edited_chunks();
        edited.sort_by_key(|(chunk, _)| chunk.to_array());
        (edited, world.bounds(), world.pose())
    });
    assert_eq!(edited, vec![(IVec3::NEG_Y, 1), (IVec3::ZERO, 9)]);
    assert_eq!(bounds, None);
    assert_eq!(pose, GlobalTransform::IDENTITY);
}

/// A second world type, whose sub-grids aren't `FloorWorld`'s.
#[derive(Resource, Clone, Default)]
struct OtherWorld;

impl VoxelWorldConfig for OtherWorld {
    type MaterialIndex = u8;
    type ChunkUserBundle = ();
}

#[test]
fn a_grid_is_none_for_a_despawned_sub_grid_any_other_entity_and_another_worlds() {
    let mut app = App::new();
    app.add_plugins((
        test_plugins(),
        TransformPlugin,
        VoxelWorldPlugin::<FloorWorld>::minimal(),
        VoxelWorldPlugin::<OtherWorld>::minimal(),
    ));
    app.update();
    let never = app.world_mut().spawn(Transform::default()).id();
    let other = run::<OtherWorld, _>(&mut app, |voxel_world| {
        voxel_world.spawn_sub_grid(Transform::default())
    });
    let despawned = spawn(&mut app, Transform::default());
    app.update();
    app.world_mut().despawn(despawned);
    app.update();

    for entity in [never, other, despawned] {
        let (read, edit) = run::<FloorWorld, _>(&mut app, move |voxel_world| {
            (
                voxel_world.grid(Grid::SubGrid(entity)).is_none(),
                voxel_world.grid_mut(Grid::SubGrid(entity)).is_none(),
            )
        });
        assert!(read && edit, "{entity}");
    }
    let in_its_world = run::<OtherWorld, _>(&mut app, move |voxel_world| {
        voxel_world.grid(Grid::SubGrid(other)).is_some()
    });
    assert!(in_its_world);
}

#[test]
fn air_unset_and_a_clear_remove_a_sub_grids_voxels_which_then_read_air() {
    let mut app = sub_grid_app();
    let entity = spawn(&mut app, Transform::default());
    let never_set = IVec3::new(7, 7, 7);
    assert_eq!(voxel(&mut app, entity, never_set), Some(WorldVoxel::Air));
    let (cleared, region, this_frame) = (
        IVec3::new(5, 0, 0),
        IVec3::new(-3, -3, -3),
        IVec3::new(0, 9, 0),
    );
    set(
        &mut app,
        entity,
        &[A, B, cleared, region],
        WorldVoxel::Solid(2),
    );
    app.update();

    run::<FloorWorld, _>(&mut app, move |voxel_world| {
        let mut grid = voxel_world.grid_mut(Grid::SubGrid(entity)).unwrap();
        grid.set_voxel(A, WorldVoxel::Air);
        grid.set_voxel(B, WorldVoxel::Unset);
        grid.clear_voxels_in(cleared, cleared + 1);
        grid.set_voxels_in(region, region + 1, |_| Some(WorldVoxel::Unset));
        grid.set_voxel(this_frame, WorldVoxel::Solid(2));
        grid.set_voxel(this_frame, WorldVoxel::Unset);
    });
    let removed = [A, B, cleared, region, this_frame, never_set];
    for frame in 0..2 {
        for position in removed {
            assert_eq!(
                voxel(&mut app, entity, position),
                Some(WorldVoxel::Air),
                "{position} in frame {frame}"
            );
        }
        assert_eq!(bounds(&mut app, entity), None, "frame {frame}");
        app.update();
    }
}

#[test]
fn bounds_follow_a_sub_grids_solids_and_are_none_without_any_and_for_the_world() {
    let mut app = sub_grid_app();
    let entity = spawn(&mut app, Transform::default());
    assert_eq!(bounds(&mut app, entity), None);

    set(&mut app, entity, &[A], WorldVoxel::Solid(1));
    assert_eq!(bounds(&mut app, entity), Some((A, A + 1)));
    set(&mut app, entity, &[B], WorldVoxel::Solid(1));
    let both = Some((A.min(B), A.max(B) + 1));
    assert_eq!(bounds(&mut app, entity), both);
    app.update();
    assert_eq!(bounds(&mut app, entity), both);

    set(&mut app, entity, &[B], WorldVoxel::Air);
    assert_eq!(
        bounds(&mut app, entity),
        Some((A, A + 1)),
        "before the commit"
    );
    app.update();
    assert_eq!(
        bounds(&mut app, entity),
        Some((A, A + 1)),
        "after the commit"
    );

    // A box set across chunks, holding `A`, then cleared down to its edge
    // along x and y at their lowest.
    let (min, max) = (IVec3::new(-2, -2, -2), IVec3::new(2, 3, 4));
    run::<FloorWorld, _>(&mut app, move |voxel_world| {
        voxel_world
            .grid_mut(Grid::SubGrid(entity))
            .unwrap()
            .set_voxels_in(min, max, |_| Some(WorldVoxel::Solid(1)));
    });
    assert_eq!(bounds(&mut app, entity), Some((min, max)));
    app.update();
    assert_eq!(bounds(&mut app, entity), Some((min, max)));
    run::<FloorWorld, _>(&mut app, move |voxel_world| {
        let mut grid = voxel_world.grid_mut(Grid::SubGrid(entity)).unwrap();
        grid.clear_voxels_in(IVec3::new(-1, -2, -2), max);
        grid.clear_voxels_in(IVec3::new(-2, -1, -2), max);
    });
    let edge = Some((min, IVec3::new(-1, -1, 4)));
    assert_eq!(bounds(&mut app, entity), edge);
    app.update();
    assert_eq!(bounds(&mut app, entity), edge);

    // Removing the voxels at the box's minimum corner moves it.
    run::<FloorWorld, _>(&mut app, move |voxel_world| {
        let mut grid = voxel_world.grid_mut(Grid::SubGrid(entity)).unwrap();
        grid.clear_voxels_in(min, IVec3::new(-1, -1, 0));
        grid.set_voxel(IVec3::new(-1, 2, 0), WorldVoxel::Solid(1));
    });
    let left = Some((IVec3::new(-2, -2, 0), IVec3::new(0, 3, 4)));
    assert_eq!(bounds(&mut app, entity), left);
    app.update();
    assert_eq!(bounds(&mut app, entity), left);

    run::<FloorWorld, _>(&mut app, |voxel_world| {
        let world = voxel_world.grid(Grid::World).unwrap().bounds();
        assert_eq!(world, None);
    });
    run::<FloorWorld, _>(&mut app, move |voxel_world| {
        voxel_world
            .grid_mut(Grid::SubGrid(entity))
            .unwrap()
            .clear_voxels_in(IVec3::splat(-100), IVec3::splat(100));
    });
    assert_eq!(bounds(&mut app, entity), None);
    app.update();
    assert_eq!(bounds(&mut app, entity), None);
}

/// Whether `pose` is `expected`, to within rounding.
fn is_near(pose: GlobalTransform, expected: GlobalTransform) -> bool {
    pose.affine().abs_diff_eq(expected.affine(), 1e-4)
}

#[test]
fn a_sub_grids_pose_is_its_last_frames_global_transform_against_the_root() {
    let mut app = sub_grid_app();
    let root = root(&mut app);
    *app.world_mut().get_mut::<Transform>(root).unwrap() =
        Transform::from_xyz(10.0, -4.0, 2.0).with_rotation(Quat::from_rotation_y(0.5));
    let world_pose = run::<FloorWorld, _>(&mut app, |voxel_world| {
        voxel_world.grid(Grid::World).unwrap().pose()
    });
    assert_eq!(world_pose, GlobalTransform::IDENTITY);

    let given = Transform::from_xyz(1.0, 2.0, 3.0)
        .with_rotation(Quat::from_rotation_x(0.3))
        .with_scale(Vec3::new(2.0, 1.0, 0.5));
    // A sub-grid two parents below the root.
    let parent =
        Transform::from_xyz(0.0, 5.0, 0.0).with_rotation(Quat::from_rotation_z(0.4));
    let parent = app.world_mut().spawn((parent, ChildOf(root))).id();
    let child = Transform::from_xyz(3.0, 0.0, 0.0);
    let deep = app.world_mut().spawn((child, ChildOf(parent))).id();
    let (spawned, spawned_pose, deep_pose) =
        run::<FloorWorld, _>(&mut app, move |voxel_world| {
            let spawned = voxel_world.spawn_sub_grid(given);
            voxel_world.insert_sub_grid(deep);
            let pose = |entity| voxel_world.grid(Grid::SubGrid(entity)).unwrap().pose();
            (spawned, pose(spawned), pose(deep))
        });
    assert!(is_near(spawned_pose, given.into()), "{spawned_pose:?}");
    assert_eq!(
        deep_pose,
        GlobalTransform::IDENTITY,
        "before its first frame"
    );

    app.update();
    let parent_transform = *app.world().get::<Transform>(parent).unwrap();
    let through_parent =
        GlobalTransform::from(parent_transform) * GlobalTransform::from(child);
    assert!(is_near(pose(&mut app, spawned), given.into()));
    let read = pose(&mut app, deep);
    assert!(is_near(read, through_parent), "{read:?}");

    // A transform changed this frame shows from the next.
    let moved = Transform::from_xyz(-6.0, 1.0, 0.0);
    *app.world_mut().get_mut::<Transform>(deep).unwrap() = moved;
    assert!(is_near(pose(&mut app, deep), through_parent));
    app.update();
    let read = pose(&mut app, deep);
    let expected = GlobalTransform::from(parent_transform) * GlobalTransform::from(moved);
    assert!(is_near(read, expected), "{read:?}");
}

/// Every edit delta committed so far.
#[derive(Resource, Default)]
struct Committed(Vec<EditDelta<u8>>);

#[test]
fn the_worlds_methods_read_and_edit_only_the_world_where_a_sub_grid_holds_voxels() {
    let mut app = floor_app();
    app.init_resource::<Committed>().add_systems(
        Update,
        |mut committed: ResMut<Committed>,
         mut messages: MessageReader<EditsCommitted<FloorWorld>>| {
            committed
                .0
                .extend(messages.read().map(|message| message.delta.clone()));
        },
    );
    let (above, below) = (IVec3::new(2, 5, 2), IVec3::new(2, -3, 2));

    let (entity, reads) = run::<FloorWorld, _>(&mut app, move |voxel_world| {
        let entity = voxel_world.spawn_sub_grid(Transform::default());
        let mut grid = voxel_world.grid_mut(Grid::SubGrid(entity)).unwrap();
        grid.set_voxel(above, WorldVoxel::Solid(2));
        grid.set_voxel(below, WorldVoxel::Solid(2));
        voxel_world.set_voxel(above, WorldVoxel::Solid(5));
        (
            entity,
            [voxel_world.get_voxel(above), voxel_world.get_voxel(below)],
        )
    });
    assert_eq!(reads, [WorldVoxel::Solid(5), WorldVoxel::Solid(1)]);
    app.update();
    let keeps = |app: &mut App, what: &str| {
        for position in [above, below] {
            assert_eq!(
                voxel(app, entity, position),
                Some(WorldVoxel::Solid(2)),
                "{position} {what}"
            );
        }
    };
    keeps(&mut app, "after the commit");
    let (edited, world) = run::<FloorWorld, _>(&mut app, move |voxel_world| {
        (voxel_world.edited_chunks(), voxel_world.get_voxel(below))
    });
    assert_eq!(edited, vec![(IVec3::ZERO, 1)]);
    assert_eq!(world, WorldVoxel::Solid(1));
    let committed = std::mem::take(&mut app.world_mut().resource_mut::<Committed>().0);
    assert_eq!(committed.len(), 1);
    assert_eq!(
        committed[0].iter().collect::<Vec<_>>(),
        vec![(above, WorldVoxel::Solid(5))]
    );

    run::<FloorWorld, _>(&mut app, move |voxel_world| {
        voxel_world.clear_voxels_in(below, above + 1);
    });
    app.update();
    keeps(&mut app, "after a box of the world is cleared");
    run::<FloorWorld, _>(&mut app, |voxel_world| voxel_world.clear_edits());
    app.update();
    keeps(&mut app, "after clear_edits");
    run::<FloorWorld, _>(&mut app, |voxel_world| voxel_world.forget_edits());
    app.update();
    keeps(&mut app, "after forget_edits");
    let committed = &app.world().resource::<Committed>().0;
    assert!(
        committed
            .iter()
            .all(|delta| delta.iter().all(|(_, voxel)| voxel != WorldVoxel::Solid(2))),
        "{committed:?}"
    );
}

#[test]
fn despawning_a_sub_grid_drops_its_voxels_and_no_later_sub_grid_finds_them() {
    let mut app = sub_grid_app();
    let entity = spawn(&mut app, Transform::default());
    let (min, max) = (IVec3::new(-4, 0, 0), IVec3::new(4, 4, 40));
    run::<FloorWorld, _>(&mut app, move |voxel_world| {
        voxel_world
            .grid_mut(Grid::SubGrid(entity))
            .unwrap()
            .set_voxels_in(min, max, |_| Some(WorldVoxel::Solid(1)));
    });
    app.update();
    // Set in the frame it despawns, so it never commits.
    set(&mut app, entity, &[B], WorldVoxel::Solid(1));
    app.world_mut().despawn(entity);
    app.update();
    assert_eq!(voxel(&mut app, entity, A), None);

    let later = [
        spawn(&mut app, Transform::default()),
        spawn(&mut app, Transform::default()),
    ];
    app.update();
    for entity in later {
        assert_eq!(bounds(&mut app, entity), None);
        for position in [min, max - 1, A, B] {
            assert_eq!(voxel(&mut app, entity, position), Some(WorldVoxel::Air));
        }
    }
}

#[test]
fn an_entity_made_a_sub_grid_again_in_the_frame_its_marker_was_removed_is_a_new_sub_grid()
{
    let mut app = sub_grid_app();
    let entity = spawn(&mut app, Transform::default());
    set(&mut app, entity, &[A], WorldVoxel::Solid(1));
    app.update();
    let old = app
        .world()
        .get::<SubGrid<FloorWorld>>(entity)
        .unwrap()
        .grid();

    app.world_mut()
        .entity_mut(entity)
        .remove::<SubGrid<FloorWorld>>();
    let read = run::<FloorWorld, _>(&mut app, move |voxel_world| {
        voxel_world.insert_sub_grid(entity);
        let mut grid = voxel_world.grid_mut(Grid::SubGrid(entity)).unwrap();
        grid.set_voxel(B, WorldVoxel::Solid(2));
        let grid = voxel_world.grid(Grid::SubGrid(entity)).unwrap();
        (grid.get_voxel(A), grid.get_voxel(B))
    });
    assert_eq!(read, (WorldVoxel::Air, WorldVoxel::Solid(2)));

    for frame in 0..3 {
        app.update();
        assert_eq!(voxel(&mut app, entity, A), Some(WorldVoxel::Air), "{frame}");
        assert_eq!(
            voxel(&mut app, entity, B),
            Some(WorldVoxel::Solid(2)),
            "{frame}"
        );
    }
    let new = app
        .world()
        .get::<SubGrid<FloorWorld>>(entity)
        .map(SubGrid::grid);
    assert!(new.is_some_and(|new| new != old), "{new:?} after {old:?}");
}
