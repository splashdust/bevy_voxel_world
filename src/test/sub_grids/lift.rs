//! Tests of lift: the voxels a game picks move out of a grid into a new
//! sub-grid, which shows them at the pose they had, and the move lands as
//! one handover that `Lifted` announces.

use std::collections::BTreeSet;

use bevy::diagnostic::FrameCount;
use bevy::ecs::system::RunSystemOnce;
use bevy::prelude::*;

use crate::chunk::{CHUNK_SIZE_F, CHUNK_SIZE_I};
use crate::prelude::*;
use crate::test::{
    chunk_status_of, record_chunk_messages, recorded, spawn_camera, test_plugins,
    update_until,
};
use crate::voxel_edits::VoxelEdits;
use crate::voxel_world_internal::WorldRoot;

/// A position below ground the world generates as air.
const HOLE: IVec3 = IVec3::new(0, -1, 0);

/// Ground below y = 0, of material 1 where x is even and 2 where it is odd,
/// with air at [`HOLE`], `Unset` at y = 0 and air above.
pub(super) fn ground(position: IVec3) -> WorldVoxel {
    if position.y < 0 && position != HOLE {
        WorldVoxel::Solid(1 + position.x.rem_euclid(2) as u8)
    } else if position.y == 0 {
        WorldVoxel::Unset
    } else {
        WorldVoxel::Air
    }
}

/// A world of [`ground`]. Once `slow`, it runs two chunk tasks at a time,
/// so a commit touching many chunks takes several frames to publish them.
#[derive(Resource, Clone, Default)]
pub(super) struct LiftWorld {
    pub(super) slow: bool,
}

impl VoxelWorldConfig for LiftWorld {
    type MaterialIndex = u8;
    type ChunkUserBundle = ();

    fn spawning_distance(&self) -> u32 {
        2
    }

    // A test app has no window, so its camera casts no rays: the chunks
    // around the camera are the ones that spawn.
    fn min_despawn_distance(&self) -> u32 {
        2
    }

    fn chunk_despawn_strategy(&self) -> ChunkDespawnStrategy {
        ChunkDespawnStrategy::FarAway
    }

    fn max_active_chunk_threads(&self) -> usize {
        if self.slow {
            2
        } else {
            bevy::tasks::available_parallelism()
        }
    }

    fn voxel_lookup_delegate(&self) -> VoxelLookupDelegate<Self::MaterialIndex> {
        Box::new(|_, _, _| Box::new(|position, _| ground(position)))
    }
}

/// The `Lifted` messages an app read, each with the frame it arrived in.
#[derive(Resource, Default)]
struct LiftedLog(Vec<(u32, Lifted<LiftWorld>)>);

fn record_lifted(
    mut log: ResMut<LiftedLog>,
    mut lifted: MessageReader<Lifted<LiftWorld>>,
    frame: Res<FrameCount>,
) {
    let frame = frame.0;
    log.0.extend(lifted.read().map(|message| (frame, *message)));
}

pub(super) fn lifted(app: &App) -> &[(u32, Lifted<LiftWorld>)] {
    &app.world().resource::<LiftedLog>().0
}

/// The top layer of the ground around the origin, which reaches into eight
/// chunks of the world through their padding.
pub(super) fn slab() -> (IVec3, IVec3) {
    (IVec3::new(-1, -1, -1), IVec3::new(2, 0, 2))
}

/// A voxel of the ground in the chunk below the origin, away from the slab.
pub(super) const APART: IVec3 = IVec3::new(9, -1, 9);

/// The level-0 chunk positions whose padded voxels hold a voxel of the box
/// from `min` inclusive to `max` exclusive: a chunk's padding is one voxel
/// deep.
pub(super) fn padded_chunks(min: IVec3, max: IVec3) -> BTreeSet<[i32; 3]> {
    let chunk = |voxel: IVec3| voxel.div_euclid(IVec3::splat(CHUNK_SIZE_I));
    let (lo, hi) = (chunk(min - 1), chunk(max));
    let mut chunks = BTreeSet::new();
    for x in lo.x..=hi.x {
        for y in lo.y..=hi.y {
            for z in lo.z..=hi.z {
                chunks.insert([x, y, z]);
            }
        }
    }
    chunks
}

/// Runs `f` on the voxel world in a system of its own, whose commands apply
/// when it returns.
pub(super) fn run<T: 'static>(
    app: &mut App,
    mut f: impl FnMut(&mut VoxelWorld<LiftWorld>) -> T + Send + Sync + 'static,
) -> T {
    app.world_mut()
        .run_system_once(move |mut voxel_world: VoxelWorld<LiftWorld>| {
            f(&mut voxel_world)
        })
        .unwrap()
}

/// The frame the next update runs, as the message recorders count it.
pub(super) fn next_frame(app: &App) -> u32 {
    app.world().resource::<FrameCount>().0
}

/// The frames, from `since` on, in which each chunk of `grid` at one of
/// `chunks` published.
pub(super) fn published_since(
    app: &App,
    since: u32,
    grid: Grid,
    chunks: &BTreeSet<[i32; 3]>,
) -> Vec<([i32; 3], u32)> {
    recorded::<ChunkPublished<LiftWorld>>(app)
        .messages
        .iter()
        .filter(|(frame, message)| {
            *frame >= since
                && message.grid == grid
                && chunks.contains(&message.chunk_key.position.to_array())
        })
        .map(|(frame, message)| (message.chunk_key.position.to_array(), *frame))
        .collect()
}

pub(super) fn all_published_since(
    app: &App,
    since: u32,
    grid: Grid,
    chunks: &BTreeSet<[i32; 3]>,
) -> bool {
    let published: BTreeSet<[i32; 3]> = published_since(app, since, grid, chunks)
        .into_iter()
        .map(|(chunk, _)| chunk)
        .collect();
    published == *chunks
}

pub(super) fn edits(app: &App) -> &VoxelEdits<LiftWorld, u8> {
    app.world().resource::<VoxelEdits<LiftWorld, u8>>()
}

/// The id the sub-grid `entity`'s voxel edits are kept under.
pub(super) fn grid_id(app: &App, entity: Entity) -> crate::grid::GridId {
    app.world()
        .get::<SubGrid<LiftWorld>>(entity)
        .expect("a live sub-grid")
        .grid()
}

/// A world app with transforms, a camera and message recorders, run until
/// the world's chunks around the origin are published. Then the world turns
/// slow.
pub(super) fn lift_app() -> App {
    let mut app = App::new();
    app.add_plugins((
        test_plugins(),
        TransformPlugin,
        VoxelWorldPlugin::<LiftWorld>::minimal(),
    ))
    .init_resource::<LiftedLog>()
    .add_systems(Update, record_lifted);
    record_chunk_messages::<ChunkPublished<LiftWorld>>(&mut app);
    app.world_mut()
        .run_system_once(spawn_camera::<LiftWorld>)
        .unwrap();
    let (min, max) = slab();
    let mut around = padded_chunks(min - 2, max + 2);
    around.extend(padded_chunks(APART, APART + 1));
    update_until(&mut app, "the world's chunks around the origin", |app| {
        all_published_since(app, 0, Grid::World, &around)
    });
    app.world_mut().resource_mut::<LiftWorld>().slow = true;
    app
}

/// Runs `app` until it has read a `Lifted` naming `sub_grid`.
pub(super) fn until_lifted(app: &mut App, sub_grid: Entity) {
    update_until(app, "the lift to land", |app| {
        lifted(app)
            .iter()
            .any(|(_, message)| message.sub_grid == sub_grid)
    });
}

#[test]
fn lifting_a_box_moves_only_its_solids_and_leaves_air_edits_in_the_world() {
    let mut app = lift_app();
    let (min, max) = (IVec3::new(-2, -2, -2), IVec3::new(3, 2, 3));
    // The minimum solid is at `min`, so the origin is one voxel below it.
    let origin = IVec3::splat(-3);

    let (entity, seen, read_back, bounds) = run(&mut app, move |voxel_world| {
        let mut seen = Vec::new();
        let entity = voxel_world
            .lift(Grid::World, min, max, |position, voxel| {
                seen.push((position, voxel));
                true
            })
            .unwrap();
        let grid = voxel_world.grid(Grid::SubGrid(entity)).unwrap();
        let read_back: Vec<WorldVoxel> = seen
            .iter()
            .map(|(position, _)| grid.get_voxel(*position - origin))
            .collect();
        (entity, seen, read_back, grid.bounds())
    });

    assert_eq!(seen.len(), 5 * 4 * 5, "the pick sees every position");
    assert!(seen.contains(&(HOLE, WorldVoxel::Air)));
    assert!(seen.iter().any(|(_, voxel)| *voxel == WorldVoxel::Unset));
    for ((position, voxel), back) in seen.iter().zip(&read_back) {
        // A published chunk with no solid voxel reads `Unset` throughout.
        let generated = ground(*position);
        if generated.is_solid() {
            assert_eq!(*voxel, generated, "{position}");
        } else {
            assert!(!voxel.is_solid(), "{position}");
        }
        let solid = if voxel.is_solid() {
            *voxel
        } else {
            WorldVoxel::Air
        };
        assert_eq!(*back, solid, "{position}");
    }
    assert_eq!(bounds, Some((IVec3::ONE, IVec3::new(6, 3, 6))));

    until_lifted(&mut app, entity);
    let grid = grid_id(&app, entity);
    let (world_edits, sub_grid_edits) =
        (edits(&app).reader(), edits(&app).reader_in(grid));
    for (position, voxel) in &seen {
        let local = *position - origin;
        if voxel.is_solid() {
            assert_eq!(
                world_edits.get(*position),
                Some(WorldVoxel::Air),
                "{position}"
            );
            assert_eq!(sub_grid_edits.get(local), Some(*voxel), "{position}");
        } else {
            assert_eq!(world_edits.get(*position), None, "{position}");
            assert_eq!(sub_grid_edits.get(local), None, "{position}");
        }
    }
    let world_reads: Vec<WorldVoxel> = run(&mut app, move |voxel_world| {
        (min.x..max.x)
            .flat_map(|x| (min.z..max.z).map(move |z| IVec3::new(x, -1, z)))
            .map(|position| voxel_world.get_voxel(position))
            .collect()
    });
    assert!(world_reads.iter().all(|voxel| *voxel == WorldVoxel::Air));
}

/// The positions of the box from `min` inclusive to `max` exclusive.
pub(super) fn positions(min: IVec3, max: IVec3) -> Vec<IVec3> {
    let mut positions = Vec::new();
    for z in min.z..max.z {
        for y in min.y..max.y {
            for x in min.x..max.x {
                positions.push(IVec3::new(x, y, z));
            }
        }
    }
    positions
}

/// Lifts every solid of the box from `min` inclusive to `max` exclusive out
/// of `source`, and returns the new sub-grid with what it reads at each
/// solid the source read there before, by the source's position.
fn lift_and_read_back(
    app: &mut App,
    source: Grid,
    min: IVec3,
    max: IVec3,
) -> (Entity, Vec<(IVec3, WorldVoxel, WorldVoxel)>) {
    run(app, move |voxel_world| {
        let mut solids = Vec::new();
        let entity = voxel_world
            .lift(source, min, max, |position, voxel| {
                if voxel.is_solid() {
                    solids.push((position, voxel));
                }
                true
            })
            .unwrap();
        let origin = solids.iter().map(|(p, _)| *p).reduce(IVec3::min).unwrap() - 1;
        let grid = voxel_world.grid(Grid::SubGrid(entity)).unwrap();
        let read = solids
            .iter()
            .map(|(position, voxel)| {
                (*position, *voxel, grid.get_voxel(*position - origin))
            })
            .collect();
        (entity, read)
    })
}

#[test]
fn lifting_where_no_chunk_is_loaded_reads_the_generator() {
    let mut app = lift_app();
    let far = IVec3::new(3000, -2, 3000);
    let (min, max) = (far, far + IVec3::new(3, 4, 3));

    let (entity, read) = lift_and_read_back(&mut app, Grid::World, min, max);

    let expected: Vec<(IVec3, WorldVoxel)> = positions(min, max)
        .into_iter()
        .map(|position| (position, ground(position)))
        .filter(|(_, voxel)| voxel.is_solid())
        .collect();
    assert_eq!(expected.len(), 18);
    let lifted: Vec<(IVec3, WorldVoxel)> =
        read.iter().map(|(p, v, _)| (*p, *v)).collect();
    assert_eq!(lifted, expected);
    for (position, voxel, back) in read {
        assert_eq!(back, voxel, "{position}");
    }
    until_lifted(&mut app, entity);
    let world: Vec<WorldVoxel> = run(&mut app, move |voxel_world| {
        expected
            .iter()
            .map(|(position, _)| voxel_world.get_voxel(*position))
            .collect()
    });
    assert!(world.iter().all(|voxel| *voxel == WorldVoxel::Air));
}

/// Ground below y = 0, solid only where x is odd from y = -40 up, whose
/// chunks beyond the one under the camera stream at level 1, so coarse
/// chunks sample the even columns.
#[derive(Resource, Clone, Default)]
struct StripedWorld;

fn striped(position: IVec3) -> WorldVoxel {
    let ground = position.y < 0 && (position.y < -40 || position.x.rem_euclid(2) == 1);
    if ground {
        WorldVoxel::Solid(1)
    } else {
        WorldVoxel::Air
    }
}

impl VoxelWorldConfig for StripedWorld {
    type MaterialIndex = u8;
    type ChunkUserBundle = ();

    fn spawning_distance(&self) -> u32 {
        3
    }

    fn chunk_spawn_strategy(&self) -> ChunkSpawnStrategy {
        ChunkSpawnStrategy::Close
    }

    fn chunk_despawn_strategy(&self) -> ChunkDespawnStrategy {
        ChunkDespawnStrategy::FarAway
    }

    fn voxel_lookup_delegate(&self) -> VoxelLookupDelegate<Self::MaterialIndex> {
        Box::new(|_, _, _| Box::new(|position, _| striped(position)))
    }

    fn chunk_lod(
        &self,
        chunk_position: IVec3,
        _previous_lod: Option<LodLevel>,
        camera_position: Vec3,
    ) -> LodLevel {
        let camera_chunk = (camera_position / CHUNK_SIZE_F).floor().as_ivec3();
        if chunk_position.distance_squared(camera_chunk) < 4 {
            0
        } else {
            1
        }
    }
}

#[test]
fn lifting_inside_a_coarse_chunk_lifts_the_generators_level_0_voxels() {
    let mut app = App::new();
    app.add_plugins((test_plugins(), VoxelWorldPlugin::<StripedWorld>::minimal()))
        .add_systems(Startup, spawn_camera::<StripedWorld>);
    let coarse = ChunkKey::new(IVec3::new(2, -2, 0), 1);
    update_until(&mut app, "the coarse chunk to be ready", |app| {
        chunk_status_of::<StripedWorld>(app, coarse) == Some(ChunkStatus::Ready)
    });
    let min = coarse.min_voxel() + IVec3::new(4, 44, 4);
    let max = min + IVec3::new(4, 2, 2);
    assert!(positions(min, max).iter().all(|p| p.y < 0 && p.y >= -40));

    let (sampled, lifted) = app
        .world_mut()
        .run_system_once(move |mut voxel_world: VoxelWorld<StripedWorld>| {
            let sampled: Vec<WorldVoxel> = positions(min, max)
                .into_iter()
                .map(|position| voxel_world.get_voxel(position))
                .collect();
            let mut lifted = Vec::new();
            voxel_world
                .lift(Grid::World, min, max, |position, voxel| {
                    lifted.push((position, voxel));
                    true
                })
                .unwrap();
            (sampled, lifted)
        })
        .unwrap();

    let generated: Vec<(IVec3, WorldVoxel)> = positions(min, max)
        .into_iter()
        .map(|position| (position, striped(position)))
        .collect();
    assert_ne!(
        sampled,
        generated
            .iter()
            .map(|(_, voxel)| *voxel)
            .collect::<Vec<_>>(),
        "get_voxel reads the coarse chunk's samples"
    );
    assert_eq!(lifted, generated);
}

/// Spawns a sub-grid at `transform` holding `Solid(5)` at each of `voxels`,
/// and runs `app` until its pose is the one its transform gives.
pub(super) fn sub_grid_holding(
    app: &mut App,
    transform: Transform,
    voxels: Vec<IVec3>,
) -> Entity {
    let entity = run(app, move |voxel_world| {
        let entity = voxel_world.spawn_sub_grid(transform);
        let mut grid = voxel_world.grid_mut(Grid::SubGrid(entity)).unwrap();
        for voxel in &voxels {
            grid.set_voxel(*voxel, WorldVoxel::Solid(5));
        }
        entity
    });
    app.update();
    app.update();
    entity
}

/// A row of four voxels along x, from (3, 4, 5).
pub(super) fn row() -> Vec<IVec3> {
    (3..7).map(|x| IVec3::new(x, 4, 5)).collect()
}

#[test]
fn lifting_from_a_sub_grid_removes_the_lifted_voxels_from_it() {
    let mut app = lift_app();
    let source = sub_grid_holding(&mut app, Transform::default(), row());

    // The box reaches below y = 0, where the world is solid at the same
    // positions, and the sub-grid holds nothing.
    let (entity, read) = lift_and_read_back(
        &mut app,
        Grid::SubGrid(source),
        IVec3::new(5, -3, 5),
        IVec3::new(7, 5, 6),
    );
    assert_eq!(read.len(), 2);
    assert!(read.iter().all(|(_, voxel, back)| back == voxel));
    let source_now = |app: &mut App| {
        run(app, move |voxel_world| {
            let grid = voxel_world.grid(Grid::SubGrid(source)).unwrap();
            let voxels: Vec<WorldVoxel> =
                row().iter().map(|p| grid.get_voxel(*p)).collect();
            (voxels, grid.bounds())
        })
    };
    let expected = (
        vec![
            WorldVoxel::Solid(5),
            WorldVoxel::Solid(5),
            WorldVoxel::Air,
            WorldVoxel::Air,
        ],
        Some((IVec3::new(3, 4, 5), IVec3::new(5, 5, 6))),
    );
    assert_eq!(source_now(&mut app), expected, "at once");
    until_lifted(&mut app, entity);
    assert_eq!(source_now(&mut app), expected, "once landed");
}

#[test]
fn a_sub_grid_lifted_bare_lives_on_with_no_bounds() {
    let mut app = lift_app();
    let source = sub_grid_holding(&mut app, Transform::default(), row());

    let (entity, read) = lift_and_read_back(
        &mut app,
        Grid::SubGrid(source),
        IVec3::ZERO,
        IVec3::splat(10),
    );
    assert_eq!(read.len(), 4);
    until_lifted(&mut app, entity);
    app.update();
    let bounds = run(&mut app, move |voxel_world| {
        voxel_world
            .grid(Grid::SubGrid(source))
            .map(|grid| grid.bounds())
    });
    assert_eq!(bounds, Some(None));
    assert!(app.world().get_entity(source).is_ok());
}

pub(super) fn root(app: &mut App) -> Entity {
    app.world_mut()
        .query_filtered::<Entity, With<WorldRoot<LiftWorld>>>()
        .single(app.world())
        .unwrap()
}

/// The lifted sub-grid's parent and transform.
fn placement(app: &App, entity: Entity) -> (Entity, Transform) {
    (
        app.world().get::<ChildOf>(entity).unwrap().parent(),
        *app.world().get::<Transform>(entity).unwrap(),
    )
}

pub(super) fn assert_poses_match(got: GlobalTransform, expected: GlobalTransform) {
    let (got, expected) = (got.affine(), expected.affine());
    assert!(
        got.abs_diff_eq(expected, 1e-4),
        "{got:?} against {expected:?}"
    );
}

#[test]
fn a_sub_grid_lifted_from_the_world_sits_under_the_root_at_its_origin() {
    let mut app = lift_app();
    let (min, max) = (IVec3::new(4, -3, 2), IVec3::new(7, -1, 4));

    let (entity, pose) = run(&mut app, move |voxel_world| {
        let entity = voxel_world
            .lift(Grid::World, min, max, |_, _| true)
            .unwrap();
        (
            entity,
            voxel_world.grid(Grid::SubGrid(entity)).unwrap().pose(),
        )
    });

    let origin = IVec3::new(3, -4, 1);
    assert_poses_match(pose, GlobalTransform::from_translation(origin.as_vec3()));
    let root = root(&mut app);
    assert_eq!(
        placement(&app, entity),
        (root, Transform::from_translation(origin.as_vec3()))
    );
}

#[test]
fn a_sub_grid_lifted_from_a_turned_sub_grid_shows_its_voxels_where_they_were() {
    let mut app = lift_app();
    let turned = Transform {
        translation: Vec3::new(10.0, 2.0, -4.0),
        rotation: Quat::from_rotation_y(0.7) * Quat::from_rotation_x(0.3),
        scale: Vec3::ONE,
    };
    let source = sub_grid_holding(&mut app, turned, row());

    let (entity, source_pose, pose) = run(&mut app, move |voxel_world| {
        let source_pose = voxel_world.grid(Grid::SubGrid(source)).unwrap().pose();
        let entity = voxel_world
            .lift(
                Grid::SubGrid(source),
                IVec3::new(4, 4, 5),
                IVec3::new(6, 5, 6),
                |_, _| true,
            )
            .unwrap();
        let pose = voxel_world.grid(Grid::SubGrid(entity)).unwrap().pose();
        (entity, source_pose, pose)
    });

    assert_poses_match(source_pose, GlobalTransform::from(turned));
    let origin = IVec3::new(3, 3, 4);
    assert_poses_match(
        pose,
        GlobalTransform::from(turned * Transform::from_translation(origin.as_vec3())),
    );
    let centre = |v: IVec3| v.as_vec3() + 0.5;
    for voxel in [IVec3::new(4, 4, 5), IVec3::new(5, 4, 5)] {
        let was = source_pose.transform_point(centre(voxel));
        let is = pose.transform_point(centre(voxel - origin));
        assert!(was.abs_diff_eq(is, 1e-4), "{voxel}: {was} against {is}");
    }
    assert_eq!(
        placement(&app, entity),
        (source, Transform::from_translation(origin.as_vec3()))
    );
    app.update();
    let snapshot = run(&mut app, move |voxel_world| {
        voxel_world.grid(Grid::SubGrid(entity)).unwrap().pose()
    });
    assert_poses_match(snapshot, pose);
}

/// How many sub-grids of the world are spawned.
fn sub_grid_count(app: &mut App) -> usize {
    app.world_mut()
        .query::<&SubGrid<LiftWorld>>()
        .iter(app.world())
        .count()
}

#[test]
fn a_lift_with_no_solid_spawns_no_sub_grid_and_commits_nothing() {
    let mut app = lift_app();
    let (results, picked_solid) = run(&mut app, |voxel_world| {
        let mut picked_solid = false;
        let air = voxel_world.lift(
            Grid::World,
            IVec3::new(0, 1, 0),
            IVec3::new(4, 5, 4),
            |_, voxel| {
                picked_solid |= voxel.is_solid();
                true
            },
        );
        let none = voxel_world.lift(
            Grid::World,
            IVec3::new(0, -3, 0),
            IVec3::new(4, -1, 4),
            |_, _| false,
        );
        ([air, none], picked_solid)
    });
    assert!(!picked_solid);
    assert_eq!(results, [Err(LiftError::NoSolid), Err(LiftError::NoSolid)]);
    for _ in 0..3 {
        app.update();
    }
    assert_eq!(sub_grid_count(&mut app), 0);
    let edited = run(&mut app, |voxel_world| voxel_world.edited_chunks());
    assert!(edited.is_empty(), "{edited:?}");
    assert!(lifted(&app).is_empty());
}

#[test]
fn a_lift_from_a_dead_sub_grid_finds_no_grid() {
    let mut app = lift_app();
    let source = sub_grid_holding(&mut app, Transform::default(), row());
    app.world_mut().entity_mut(source).despawn();
    app.update();
    let never = app.world_mut().spawn_empty().id();

    let results = run(&mut app, move |voxel_world| {
        [source, never].map(|entity| {
            voxel_world.lift(
                Grid::SubGrid(entity),
                IVec3::ZERO,
                IVec3::splat(10),
                |_, _| true,
            )
        })
    });
    assert_eq!(
        results,
        [Err(LiftError::NoSuchGrid), Err(LiftError::NoSuchGrid)]
    );
}

#[test]
fn every_chunk_a_lift_touches_publishes_in_the_frame_lifted_arrives() {
    let mut app = lift_app();
    let (min, max) = slab();
    let commit = next_frame(&app);
    let entity = run(&mut app, move |voxel_world| {
        voxel_world
            .lift(Grid::World, min, max, |_, _| true)
            .unwrap()
    });
    // The slab's minimum sits at (1, 1, 1), so the sub-grid needs one chunk.
    let world_chunks = padded_chunks(min, max);
    let sub_grid_chunks = BTreeSet::from([[0, 0, 0]]);
    update_until(&mut app, "both grids' chunks to publish", |app| {
        all_published_since(app, commit, Grid::World, &world_chunks)
            && all_published_since(app, commit, Grid::SubGrid(entity), &sub_grid_chunks)
    });
    for _ in 0..3 {
        app.update();
    }

    let mut frames = published_since(&app, commit, Grid::World, &world_chunks);
    frames.extend(published_since(
        &app,
        commit,
        Grid::SubGrid(entity),
        &sub_grid_chunks,
    ));
    assert_eq!(
        frames.len(),
        world_chunks.len() + 1,
        "each chunk published once"
    );
    let landing: BTreeSet<u32> = frames.iter().map(|(_, frame)| *frame).collect();
    assert_eq!(landing.len(), 1, "{frames:?}");
    let landing = *landing.first().unwrap();
    assert!(landing > commit, "the slow world takes frames to land");
    assert_eq!(
        lifted(&app),
        [(landing, Lifted::new(Grid::World, entity))].as_slice()
    );
}

#[test]
fn two_lifts_sharing_a_chunk_each_get_lifted_in_their_landing_frame() {
    let mut app = lift_app();
    let (min, max) = slab();
    let commit = next_frame(&app);
    let first = run(&mut app, move |voxel_world| {
        voxel_world
            .lift(Grid::World, min, max, |_, _| true)
            .unwrap()
    });
    app.update();
    let second = run(&mut app, |voxel_world| {
        voxel_world
            .lift(Grid::World, APART, APART + 1, |_, _| true)
            .unwrap()
    });
    let (mut world_chunks, apart) =
        (padded_chunks(min, max), padded_chunks(APART, APART + 1));
    assert!(world_chunks.contains(&[0, -1, 0]) && apart.contains(&[0, -1, 0]));
    world_chunks.extend(apart);
    update_until(&mut app, "both lifts to land", |app| lifted(app).len() == 2);

    let landing: BTreeSet<u32> =
        published_since(&app, commit, Grid::World, &world_chunks)
            .into_iter()
            .map(|(_, frame)| frame)
            .collect();
    assert_eq!(landing.len(), 1, "{landing:?}");
    let landing = *landing.first().unwrap();
    let mut got = lifted(&app).to_vec();
    got.sort_by_key(|(_, message)| message.sub_grid != first);
    assert_eq!(
        got,
        [
            (landing, Lifted::new(Grid::World, first)),
            (landing, Lifted::new(Grid::World, second))
        ]
    );
}

#[test]
fn a_lifted_sub_grid_despawned_before_landing_gets_no_lifted() {
    let mut app = lift_app();
    let (min, max) = slab();
    let commit = next_frame(&app);
    let entity = run(&mut app, move |voxel_world| {
        voxel_world
            .lift(Grid::World, min, max, |_, _| true)
            .unwrap()
    });
    app.update();
    let world_chunks = padded_chunks(min, max);
    assert!(!all_published_since(
        &app,
        commit,
        Grid::World,
        &world_chunks
    ));
    app.world_mut().entity_mut(entity).despawn();

    update_until(&mut app, "the world's chunks to publish", |app| {
        all_published_since(app, commit, Grid::World, &world_chunks)
    });
    for _ in 0..3 {
        app.update();
    }
    assert!(lifted(&app).is_empty());
}

#[test]
fn a_sub_grid_filled_through_grid_mut_gets_no_lifted() {
    let mut app = lift_app();
    let commit = next_frame(&app);
    let entity = sub_grid_holding(&mut app, Transform::default(), row());
    let chunks = BTreeSet::from([[0, 0, 0]]);
    update_until(&mut app, "the sub-grid's chunk to publish", |app| {
        all_published_since(app, commit, Grid::SubGrid(entity), &chunks)
    });
    for _ in 0..3 {
        app.update();
    }
    assert!(lifted(&app).is_empty());
}

#[test]
fn a_lift_from_a_sub_grid_lifted_in_the_same_system_reads_its_new_voxels() {
    let mut app = lift_app();
    let (min, max) = slab();

    let (first, second, read, left) = run(&mut app, move |voxel_world| {
        let first = voxel_world
            .lift(Grid::World, min, max, |_, _| true)
            .unwrap();
        // The slab sits at y = 1 in the first sub-grid, from x = 1 to 3.
        let mut read = Vec::new();
        let second = voxel_world
            .lift(
                Grid::SubGrid(first),
                IVec3::new(3, 0, 0),
                IVec3::new(4, 3, 5),
                |position, voxel| {
                    read.push((position, voxel));
                    true
                },
            )
            .unwrap();
        let left = voxel_world.grid(Grid::SubGrid(first)).unwrap().bounds();
        (first, second, read, left)
    });

    let solids: Vec<(IVec3, WorldVoxel)> = read
        .into_iter()
        .filter(|(_, voxel)| voxel.is_solid())
        .collect();
    let expected: Vec<(IVec3, WorldVoxel)> = (1..4)
        .map(|z| (IVec3::new(3, 1, z), ground(IVec3::new(1, -1, z - 2))))
        .collect();
    assert_eq!(solids, expected);
    assert_eq!(left, Some((IVec3::ONE, IVec3::new(3, 2, 4))));
    update_until(&mut app, "both lifts to land", |app| lifted(app).len() == 2);
    let messages: BTreeSet<(u32, bool, Entity)> = lifted(&app)
        .iter()
        .map(|(frame, message)| (*frame, message.source == Grid::World, message.sub_grid))
        .collect();
    let landing = lifted(&app)[0].0;
    assert_eq!(
        messages,
        BTreeSet::from([(landing, true, first), (landing, false, second)])
    );
    assert!(lifted(&app)
        .iter()
        .all(|(_, message)| message.source == Grid::World
            || message.source == Grid::SubGrid(first)));
}
