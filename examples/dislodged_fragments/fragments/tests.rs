use std::time::{Duration, Instant};

use avian3d::prelude::*;
use bevy::{ecs::system::RunSystemOnce, prelude::*, time::TimeUpdateStrategy};
use bevy_voxel_world::prelude::*;

use super::{
    outline::outline_segments,
    panel::Setting,
    physics::collider_voxels,
    rest::{ease, Piece},
    FragmentsConfig, PiecesPlugin,
};

#[test]
fn a_single_voxel_is_outlined_by_its_12_edges() {
    let segments =
        outline_segments(IVec3::ZERO, IVec3::ONE, |voxel| voxel == IVec3::ZERO);
    assert_eq!(segments.len(), 12);
}

#[test]
fn a_bar_of_two_voxels_is_outlined_without_the_seam_between_them() {
    // A 2 by 1 by 1 box has 4 edges two voxels long and 8 one voxel long.
    let bar = |voxel: IVec3| voxel == IVec3::ZERO || voxel == IVec3::X;
    let segments = outline_segments(IVec3::ZERO, IVec3::new(2, 1, 1), bar);
    assert_eq!(segments.len(), 4 * 2 + 8);
    assert!(!segments.contains(&(Vec3::new(1.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 0.0))));
}

#[test]
fn two_voxels_touching_at_an_edge_share_that_edge_once() {
    let pair = |voxel: IVec3| voxel == IVec3::ZERO || voxel == IVec3::new(1, 1, 0);
    let segments = outline_segments(IVec3::ZERO, IVec3::new(2, 2, 1), pair);
    assert_eq!(segments.len(), 12 + 12 - 1);
}

#[test]
fn a_collider_holds_the_chunks_own_solid_voxels_and_not_its_padding() {
    let first = IVec3::new(32, -32, 64);
    let solid = WorldVoxel::Solid(0u8);
    // Solid in the padding on every side, and at two of the chunk's corners.
    let voxels = collider_voxels(first, |position| {
        let offset = position - first;
        let in_padding = offset.min_element() == -1 || offset.max_element() == 32;
        let corner = offset == IVec3::ZERO || offset == IVec3::splat(31);
        Some(if in_padding || corner {
            solid
        } else {
            WorldVoxel::Air
        })
    });
    assert_eq!(voxels, vec![IVec3::ONE, IVec3::splat(32)]);
}

#[test]
fn a_full_chunk_gives_a_collider_voxel_at_each_of_its_positions() {
    let voxels = collider_voxels(IVec3::ZERO, |_| Some(WorldVoxel::Solid(0u8)));
    assert_eq!(voxels.len(), 32 * 32 * 32);
}

#[test]
fn easing_starts_at_the_resting_pose_and_ends_at_the_stamped_one() {
    let from =
        Transform::from_xyz(1.0, 2.0, 3.0).with_rotation(Quat::from_rotation_z(0.4));
    let to = Transform::from_xyz(1.0, 2.5, 3.0);
    assert_eq!(ease(from, to, 0.0), from);
    assert_eq!(ease(from, to, 1.0).translation, to.translation);
    assert!(ease(from, to, 1.0).rotation.angle_between(to.rotation) < 1e-6);
    assert_eq!(ease(from, to, 0.5).translation, Vec3::new(1.0, 2.25, 3.0));
    assert_eq!(ease(from, to, 3.0), ease(from, to, 1.0));
}

#[test]
fn the_panel_reads_and_sets_each_detection_setting() {
    let mut detection = FragmentDetection::<u8>::default();
    // The crate's defaults: faces only, a limit of 32,768, no auto-lift.
    assert_eq!(Setting::Connectivity.option(&detection), Some(0));
    assert_eq!(Setting::Limit.option(&detection), Some(1));
    assert_eq!(Setting::AutoLift.option(&detection), Some(1));

    Setting::Connectivity.set(&mut detection, 2);
    Setting::Limit.set(&mut detection, 2);
    Setting::AutoLift.set(&mut detection, 0);
    assert_eq!(detection.connectivity, Connectivity::Corners);
    assert_eq!(detection.limit, 131_072);
    assert!(detection.auto_lift);

    detection.limit = 1_000;
    assert_eq!(Setting::Limit.option(&detection), None);
}

// --- A ledge falling end to end -------------------------------------------------

/// Flat ground below y = 0, a pillar on it, and a ledge on the pillar's top
/// reaching out to one side.
#[derive(Resource, Clone)]
struct LedgeWorld {
    detection: FragmentDetection<u8>,
}

impl Default for LedgeWorld {
    fn default() -> Self {
        Self {
            detection: FragmentDetection {
                auto_lift: true,
                ..default()
            },
        }
    }
}

const STONE: WorldVoxel<u8> = WorldVoxel::Solid(1);

/// The ledge's box, and the plane of it a cut clears, which frees the part
/// beyond it, `FREED` voxels.
const LEDGE_MIN: IVec3 = IVec3::new(-1, 30, -6);
const LEDGE_MAX: IVec3 = IVec3::new(13, 34, 7);
const CUT_X: i32 = 4;
const FREED: u32 = 8 * 4 * 13;

fn ledge_voxel(position: IVec3) -> WorldVoxel<u8> {
    let pillar = position.x.abs() <= 1 && position.z.abs() <= 1 && position.y < 30;
    let ledge = position.cmpge(LEDGE_MIN).all() && position.cmplt(LEDGE_MAX).all();
    if position.y < 0 || pillar || ledge {
        STONE
    } else {
        WorldVoxel::Air
    }
}

impl VoxelWorldConfig for LedgeWorld {
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

    fn voxel_lookup_delegate(&self) -> VoxelLookupDelegate<u8> {
        Box::new(|_, _, _| Box::new(|position, _| ledge_voxel(position)))
    }

    fn detect_dislodged_fragments(&self) -> Option<FragmentDetection<u8>> {
        Some(self.detection.clone())
    }
}

impl FragmentsConfig for LedgeWorld {
    fn detection(&self) -> &FragmentDetection<u8> {
        &self.detection
    }

    fn detection_mut(&mut self) -> &mut FragmentDetection<u8> {
        &mut self.detection
    }
}

/// What the app has told the test, by frame.
#[derive(Resource, Default)]
struct Log {
    frame: u32,
    lifted: Vec<(u32, Lifted<LedgeWorld>)>,
    stamped: Vec<(u32, Stamped<LedgeWorld>)>,
    reports: Vec<FragmentDislodged<LedgeWorld>>,
}

fn record(
    mut log: ResMut<Log>,
    mut lifted: MessageReader<Lifted<LedgeWorld>>,
    mut stamped: MessageReader<Stamped<LedgeWorld>>,
    mut reports: MessageReader<FragmentDislodged<LedgeWorld>>,
) {
    log.frame += 1;
    let frame = log.frame;
    log.lifted
        .extend(lifted.read().map(|message| (frame, *message)));
    log.stamped
        .extend(stamped.read().map(|message| (frame, *message)));
    log.reports.extend(reports.read().cloned());
}

fn ledge_app(auto_lift: bool) -> App {
    let mut app = App::new();
    let mut world = LedgeWorld::default();
    world.detection.auto_lift = auto_lift;
    app.add_plugins((
        MinimalPlugins,
        TransformPlugin,
        bevy::asset::AssetPlugin::default(),
        PhysicsPlugins::default(),
        VoxelWorldPlugin::<LedgeWorld>::minimal(),
        PiecesPlugin::<LedgeWorld>::default(),
    ))
    .insert_resource(world)
    // Each frame steps the simulation by a 60th of a second, however long
    // it takes to run.
    .insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_secs_f32(
        1.0 / 60.0,
    )))
    .init_resource::<Log>()
    .add_systems(
        PreUpdate,
        record.after(VoxelWorldSet::<LedgeWorld>::default()),
    )
    .add_systems(Startup, |mut commands: Commands| {
        commands.spawn((
            Camera3d::default(),
            Transform::from_xyz(4.0, 16.0, 24.0)
                .looking_at(Vec3::new(4.0, 16.0, 0.0), Vec3::Y),
            VoxelWorldCamera::<LedgeWorld>::default(),
        ));
    });
    // Avian sets up some of its resources as the app finishes building.
    app.finish();
    app.cleanup();
    app
}

/// Runs `app` until `done` holds, failing the test after a minute.
fn run_until(app: &mut App, what: &str, mut done: impl FnMut(&mut App) -> bool) {
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        app.update();
        if done(app) {
            return;
        }
        assert!(Instant::now() < deadline, "timed out waiting: {what}");
    }
}

/// The collider of the world chunk holding `voxel`, if it holds `voxel`.
fn world_collider_holds(app: &mut App, voxel: IVec3) -> bool {
    let key = ChunkKey::containing(voxel, 0);
    let mut chunks = app
        .world_mut()
        .query::<(&Chunk<LedgeWorld>, &Collider, &ChildOf)>();
    let world = app.world();
    chunks.iter(world).any(|(chunk, collider, child_of)| {
        chunk.key() == key
            && world
                .get::<SubGrid<LedgeWorld>>(child_of.parent())
                .is_none()
            && collider.shape().as_voxels().is_some_and(|voxels| {
                let local = voxel - key.min_voxel() + IVec3::ONE;
                voxels
                    .voxel_state(local)
                    .is_some_and(|state| !state.is_empty())
            })
    })
}

/// How many of `piece`'s chunks have a collider.
fn piece_colliders(app: &mut App, piece: Entity) -> usize {
    let mut chunks = app.world_mut().query::<(&ChildOf, &Collider)>();
    chunks
        .iter(app.world())
        .filter(|(child_of, _)| child_of.parent() == piece)
        .count()
}

fn cut_ledge(app: &mut App) {
    app.world_mut()
        .run_system_once(|mut voxel_world: VoxelWorld<LedgeWorld>| {
            voxel_world.set_voxels_in(
                IVec3::new(CUT_X, LEDGE_MIN.y, LEDGE_MIN.z),
                IVec3::new(CUT_X + 1, LEDGE_MAX.y, LEDGE_MAX.z),
                |_| Some(WorldVoxel::Air),
            );
        })
        .unwrap();
}

fn ready_ledge_app(auto_lift: bool) -> App {
    let mut app = ledge_app(auto_lift);
    let tip = IVec3::new(12, 31, 0);
    run_until(&mut app, "the ledge's colliders", |app| {
        // The ground under the ledge, on both sides of z = 0.
        world_collider_holds(app, tip)
            && world_collider_holds(app, IVec3::new(12, -1, 20))
            && world_collider_holds(app, IVec3::new(12, -1, -20))
    });
    app
}

fn piece_state(app: &App, piece: Entity) -> Option<Piece> {
    app.world().get::<Piece>(piece).copied()
}

#[test]
fn a_freed_ledge_falls_and_its_colliders_swap_with_the_worlds_in_one_frame() {
    let mut app = ready_ledge_app(true);
    let tip = IVec3::new(12, 31, 0);
    cut_ledge(&mut app);

    // Until the lift lands, the world's collider keeps the ledge's tip.
    run_until(&mut app, "the lift to land", |app| {
        let landed = !app.world().resource::<Log>().lifted.is_empty();
        if !landed {
            assert!(
                world_collider_holds(app, tip),
                "the tip's collider left early"
            );
        }
        landed
    });
    let (_, lifted) = app.world().resource::<Log>().lifted[0];
    let piece = lifted.sub_grid;
    assert_eq!(lifted.source, Grid::World);
    assert!(!world_collider_holds(&mut app, tip));
    assert!(piece_colliders(&mut app, piece) > 0);
    assert_eq!(
        app.world().get::<RigidBody>(piece),
        Some(&RigidBody::Dynamic)
    );
    assert_eq!(piece_state(&app, piece), Some(Piece::Falling));
    let report = &app.world().resource::<Log>().reports[0];
    assert_eq!(report.voxel_count, FREED);
    assert_eq!(report.lifted, Some(piece));

    run_until(&mut app, "the piece to be stamped", |app| {
        !app.world().resource::<Log>().stamped.is_empty()
    });
    assert!(
        app.world().get_entity(piece).is_err(),
        "a stamped piece despawns"
    );

    // The ledge's freed end now lies on the ground, as world voxels, and
    // nothing is left where it hung.
    let (fallen, hanging) = app
        .world_mut()
        .run_system_once(|voxel_world: VoxelWorld<LedgeWorld>| {
            let mut fallen = 0;
            let mut hanging = 0;
            for z in -24..24 {
                for x in -24..24 {
                    for y in 0..40 {
                        let position = IVec3::new(x, y, z);
                        let beside_pillar = x.abs() > 1 || z.abs() > 1;
                        if !voxel_world.get_voxel(position).is_solid() || !beside_pillar {
                            continue;
                        }
                        if y < 16 {
                            fallen += 1;
                        } else if x > CUT_X {
                            hanging += 1;
                        }
                    }
                }
            }
            (fallen, hanging)
        })
        .unwrap();
    assert_eq!(hanging, 0);
    // Snapped into the ground, a few of its voxels can land below y = 0.
    assert!(
        (FREED * 3 / 4..=FREED).contains(&fallen),
        "fallen: {fallen}"
    );
}

#[test]
fn a_falling_piece_cut_in_two_falls_as_two_bodies_under_the_root() {
    let mut app = ready_ledge_app(true);
    cut_ledge(&mut app);
    run_until(&mut app, "the lift to land", |app| {
        !app.world().resource::<Log>().lifted.is_empty()
    });
    let piece = app.world().resource::<Log>().lifted[0].1.sub_grid;
    let root = app.world().get::<ChildOf>(piece).unwrap().parent();

    // Cut off the column at the piece's lowest x, in its own grid.
    app.world_mut()
        .run_system_once(move |mut voxel_world: VoxelWorld<LedgeWorld>| {
            let mut grid = voxel_world.grid_mut(Grid::SubGrid(piece)).unwrap();
            let (min, max) = grid.bounds().unwrap();
            grid.set_voxels_in(
                IVec3::new(min.x + 1, min.y, min.z),
                IVec3::new(min.x + 2, max.y, max.z),
                |_| Some(WorldVoxel::Air),
            );
        })
        .unwrap();
    assert_eq!(piece_state(&app, piece), Some(Piece::Falling));

    run_until(&mut app, "the cut-off column's lift to land", |app| {
        app.world().resource::<Log>().lifted.len() == 2
    });
    let (_, second) = app.world().resource::<Log>().lifted[1];
    assert_eq!(second.source, Grid::SubGrid(piece));
    let column = second.sub_grid;
    assert_eq!(app.world().get::<ChildOf>(column).unwrap().parent(), root);
    assert_eq!(
        app.world().get::<RigidBody>(column),
        Some(&RigidBody::Dynamic)
    );
    assert_eq!(
        app.world().get::<RigidBody>(piece),
        Some(&RigidBody::Dynamic)
    );
    assert!(piece_colliders(&mut app, column) > 0);
}

#[test]
fn without_auto_lift_a_freed_ledge_is_reported_and_stays() {
    let mut app = ready_ledge_app(false);
    cut_ledge(&mut app);
    run_until(&mut app, "the report", |app| {
        !app.world().resource::<Log>().reports.is_empty()
    });
    for _ in 0..30 {
        app.update();
    }
    let log = app.world().resource::<Log>();
    assert_eq!(log.reports[0].voxel_count, FREED);
    assert_eq!(log.reports[0].lifted, None);
    assert!(log.lifted.is_empty());
    let mut sub_grids = app.world_mut().query::<&SubGrid<LedgeWorld>>();
    assert_eq!(sub_grids.iter(app.world()).count(), 0);
    assert!(world_collider_holds(&mut app, IVec3::new(12, 31, 0)));
}
