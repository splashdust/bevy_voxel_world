//! End-to-end tests of dislodged fragments with sub-grids: checks in every
//! grid, the report's grid, and lifting each fragment automatically.

use bevy::ecs::system::RunSystemOnce;

use super::*;

/// A bar of stone in a sub-grid, along x.
const BAR: (IVec3, IVec3) = (IVec3::new(1, 1, 1), IVec3::new(21, 3, 3));
/// A plane of air across the bar, cutting it in two.
const BAR_CUT: (IVec3, IVec3) = (IVec3::new(8, 1, 1), IVec3::new(9, 3, 3));
/// The bar's smaller piece after the cut: 28 voxels.
const BAR_SHORT: (IVec3, IVec3) = (IVec3::new(1, 1, 1), IVec3::new(8, 3, 3));
/// The bar's larger piece after the cut: 48 voxels.
const BAR_LONG: (IVec3, IVec3) = (IVec3::new(9, 1, 1), IVec3::new(21, 3, 3));

/// Runs `f` on the voxel world in a system of its own, whose commands apply
/// when it returns.
fn run<T: 'static>(
    app: &mut App,
    mut f: impl FnMut(&mut VoxelWorld<CliffWorld>) -> T + Send + Sync + 'static,
) -> T {
    app.world_mut()
        .run_system_once(move |mut voxel_world: VoxelWorld<CliffWorld>| {
            f(&mut voxel_world)
        })
        .unwrap()
}

/// Spawns a sub-grid at `translation` holding stone over each box of
/// `boxes`, runs a frame to commit it, and lets any checks land.
fn spawn_filled(app: &mut App, translation: Vec3, boxes: &[(IVec3, IVec3)]) -> Entity {
    let boxes = boxes.to_vec();
    let sub_grid = run(app, move |voxel_world| {
        let sub_grid =
            voxel_world.spawn_sub_grid(Transform::from_translation(translation));
        let mut grid = voxel_world.grid_mut(Grid::SubGrid(sub_grid)).unwrap();
        for &(min, max) in &boxes {
            grid.set_voxels_in(min, max, |_| Some(WorldVoxel::Solid(STONE)));
        }
        sub_grid
    });
    app.update();
    assert!(settle(app).is_empty(), "filling a sub-grid cuts nothing");
    sub_grid
}

/// Sets air over the box `(min, max)` of `grid`, then runs a frame, which
/// commits it.
fn dig_in(app: &mut App, grid: Grid, (min, max): (IVec3, IVec3)) {
    run(app, move |voxel_world| {
        voxel_world
            .grid_mut(grid)
            .unwrap()
            .set_voxels_in(min, max, |_| Some(WorldVoxel::Air))
    });
    app.update();
}

/// The voxel of `grid` at `position`, or `None` for a grid that isn't live.
fn voxel_of(app: &mut App, grid: Grid, position: IVec3) -> Option<WorldVoxel> {
    run(app, move |voxel_world| {
        voxel_world.grid(grid).map(|grid| grid.get_voxel(position))
    })
}

/// How many fragment checks ran again after what they read moved on.
fn reruns(app: &App) -> u64 {
    fragment_stats::<CliffWorld>(app.world()).map_or(0, |stats| stats.reruns)
}

fn detecting_with_limit(limit: u32) -> CliffWorld {
    CliffWorld {
        detection: Some(FragmentDetection { limit, ..default() }),
        ..default()
    }
}

#[test]
fn cutting_a_sub_grid_in_two_dislodges_the_smaller_piece_whatever_the_limit() {
    let mut app = fragments_app(detecting_with_limit(10));
    let sub_grid = spawn_filled(&mut app, Vec3::new(100.0, 50.0, 100.0), &[BAR]);

    dig_in(&mut app, Grid::SubGrid(sub_grid), BAR_CUT);
    let dislodged = settle(&mut app);

    assert_eq!(dislodged.len(), 1, "{dislodged:?}");
    assert_holds_exactly(&dislodged[0], BAR_SHORT);
    assert_eq!(dislodged[0].grid, Grid::SubGrid(sub_grid));
    assert_eq!(dislodged[0].lifted, None);
    for position in [BAR_SHORT.0, BAR_LONG.0, BAR_LONG.1 - 1] {
        assert_eq!(
            voxel_of(&mut app, Grid::SubGrid(sub_grid), position),
            Some(WorldVoxel::Solid(STONE)),
            "{position}"
        );
    }
}

#[test]
fn a_forget_of_the_worlds_edits_leaves_a_sub_grids_check_to_report_its_piece() {
    let mut app = fragments_app(CliffWorld::detecting());
    let sub_grid = spawn_filled(&mut app, Vec3::new(100.0, 50.0, 100.0), &[BAR]);
    // An edit of the world's, which the forget drops, cutting nothing.
    edit(&mut app, |voxel_world| {
        voxel_world.set_voxel(IVec3::new(0, 100, 0), WorldVoxel::Solid(STONE))
    });

    dig_in(&mut app, Grid::SubGrid(sub_grid), BAR_CUT);
    // The cut's check has read its stamps once its floods have counted.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while fragment_stats::<CliffWorld>(app.world())
        .unwrap()
        .voxels_visited
        == 0
    {
        assert!(std::time::Instant::now() < deadline, "the check never ran");
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    // Forgetting edits the world held moves every edit stamp, so the cut's
    // check, landing after it, runs again in a frame with no edits to the
    // sub-grid.
    edit(&mut app, |voxel_world| voxel_world.forget_edits());
    let dislodged = settle(&mut app);

    assert_eq!(dislodged.len(), 1, "{dislodged:?}");
    assert_holds_exactly(&dislodged[0], BAR_SHORT);
    assert_eq!(dislodged[0].grid, Grid::SubGrid(sub_grid));
    assert!(reruns(&app) > 0);
}

/// The `Lifted` messages received so far.
#[derive(Resource, Default)]
struct LiftedLog(Vec<Lifted<CliffWorld>>);

/// [`fragments_app`] of `world`, also recording each `Lifted`.
fn lifting_app(world: CliffWorld) -> App {
    let mut app = fragments_app(world);
    app.init_resource::<LiftedLog>().add_systems(
        Update,
        |mut log: ResMut<LiftedLog>, mut lifted: MessageReader<Lifted<CliffWorld>>| {
            log.0.extend(lifted.read().copied());
        },
    );
    app
}

fn auto_lifting() -> CliffWorld {
    CliffWorld {
        detection: Some(FragmentDetection {
            auto_lift: true,
            ..default()
        }),
        ..default()
    }
}

fn lifted_log(app: &App) -> Vec<Lifted<CliffWorld>> {
    app.world().resource::<LiftedLog>().0.clone()
}

/// Runs frames until a `Lifted` names `sub_grid`, and returns it.
fn wait_for_lifted(app: &mut App, sub_grid: Entity) -> Lifted<CliffWorld> {
    update_until(app, "the lift's handover to land", |app| {
        lifted_log(app)
            .iter()
            .any(|lifted| lifted.sub_grid == sub_grid)
    });
    let log = lifted_log(app);
    let mut named = log.iter().filter(|lifted| lifted.sub_grid == sub_grid);
    let lifted = *named.next().unwrap();
    assert!(named.next().is_none(), "{log:?}");
    lifted
}

/// Asserts that `source` reads air wherever `fragment` holds a voxel, and
/// that `sub_grid` holds exactly the fragment's voxels at the pose they had
/// in `source`, which sits at `source_at`.
fn assert_moved(
    app: &mut App,
    fragment: &FragmentDislodged<CliffWorld>,
    source: Grid,
    source_at: IVec3,
    sub_grid: Entity,
) {
    let fragment = fragment.clone();
    run(app, move |voxel_world| {
        let source = voxel_world.grid(source).unwrap();
        let lifted = voxel_world.grid(Grid::SubGrid(sub_grid)).unwrap();
        let offset = lifted.pose().translation().as_ivec3() - source_at;
        for z in fragment.min.z - 1..=fragment.max.z {
            for y in fragment.min.y - 1..=fragment.max.y {
                for x in fragment.min.x - 1..=fragment.max.x {
                    let position = IVec3::new(x, y, z);
                    let held = lifted.get_voxel(position - offset).is_solid();
                    assert_eq!(held, fragment.contains(position), "{position}");
                    if fragment.contains(position) {
                        assert_eq!(source.get_voxel(position), WorldVoxel::Air);
                    }
                }
            }
        }
    });
}

#[test]
fn a_cut_in_the_world_names_the_world() {
    let mut app = fragments_app(CliffWorld::detecting());

    dig(&mut app, SHELF_ROOT);
    let dislodged = settle(&mut app);

    assert_eq!(dislodged.len(), 1, "{dislodged:?}");
    assert_eq!(dislodged[0].grid, Grid::World);
    assert_eq!(dislodged[0].lifted, None);
}

#[test]
fn lifting_the_voxels_holding_up_a_piece_dislodges_the_piece() {
    let mut app = fragments_app(CliffWorld::detecting());

    run(&mut app, |voxel_world| {
        voxel_world
            .lift(Grid::World, SHELF_ROOT.0, SHELF_ROOT.1, |_, _| true)
            .unwrap()
    });
    app.update();
    let dislodged = settle(&mut app);

    assert_eq!(dislodged.len(), 1, "{dislodged:?}");
    assert_holds_exactly(&dislodged[0], SHELF_TIP);
    assert_eq!(dislodged[0].grid, Grid::World);
}

#[test]
fn auto_lift_lifts_a_piece_cut_under_a_generated_overhang() {
    let mut app = lifting_app(auto_lifting());

    dig(&mut app, SHELF_ROOT);
    let dislodged = settle(&mut app);

    assert_eq!(dislodged.len(), 1, "{dislodged:?}");
    let fragment = &dislodged[0];
    assert_holds_exactly(fragment, SHELF_TIP);
    let sub_grid = fragment.lifted.expect("the fragment was lifted");
    assert!(voxel_of(&mut app, Grid::SubGrid(sub_grid), IVec3::ZERO).is_some());
    let lifted = wait_for_lifted(&mut app, sub_grid);
    assert_eq!(lifted.source, Grid::World);
    assert_moved(&mut app, fragment, Grid::World, IVec3::ZERO, sub_grid);
}

#[test]
fn auto_lift_lifts_a_piece_cut_in_a_sub_grid_out_of_that_sub_grid() {
    let mut app = lifting_app(auto_lifting());
    let at = IVec3::new(100, 50, 100);
    let source = spawn_filled(&mut app, at.as_vec3(), &[CLIFF, SHELF]);

    dig_in(&mut app, Grid::SubGrid(source), SHELF_ROOT);
    let dislodged = settle(&mut app);

    assert_eq!(dislodged.len(), 1, "{dislodged:?}");
    let fragment = &dislodged[0];
    assert_holds_exactly(fragment, SHELF_TIP);
    assert_eq!(fragment.grid, Grid::SubGrid(source));
    let sub_grid = fragment.lifted.expect("the fragment was lifted");
    let lifted = wait_for_lifted(&mut app, sub_grid);
    assert_eq!(lifted.source, Grid::SubGrid(source));
    assert_moved(&mut app, fragment, Grid::SubGrid(source), at, sub_grid);
}

#[test]
fn an_auto_lift_reports_each_fragment_once() {
    let mut app = lifting_app(auto_lifting());

    dig(&mut app, SHELF_ROOT);
    let dislodged = settle(&mut app);
    assert_eq!(dislodged.len(), 1, "{dislodged:?}");
    wait_for_lifted(&mut app, dislodged[0].lifted.unwrap());

    // The lift's own commit, of the air it leaves, checks and finds nothing.
    assert!(settle(&mut app).is_empty());
    assert_eq!(checks_started(&app), 2, "the cut's check and the lift's");
    assert_eq!(lifted_log(&app).len(), 1);
}

#[test]
fn a_sub_grid_despawned_before_its_check_lands_gets_no_report_and_no_lift() {
    let mut app = lifting_app(auto_lifting());
    let source = spawn_filled(&mut app, Vec3::new(100.0, 50.0, 100.0), &[BAR]);
    let checks = checks_started(&app);

    dig_in(&mut app, Grid::SubGrid(source), BAR_CUT);
    assert_eq!(checks_started(&app), checks + 1, "the cut's check started");
    app.world_mut().despawn(source);

    assert!(settle(&mut app).is_empty());
    assert!(lifted_log(&app).is_empty());
    assert_eq!(reruns(&app), 0, "dropped as it landed, not run again");
}

#[test]
fn a_sub_grid_stamped_before_its_check_lands_gets_no_report_and_no_lift() {
    let mut app = lifting_app(auto_lifting());
    let at = IVec3::new(100, 50, 100);
    let source = spawn_filled(&mut app, at.as_vec3(), &[BAR]);
    let checks = checks_started(&app);

    dig_in(&mut app, Grid::SubGrid(source), BAR_CUT);
    assert_eq!(checks_started(&app), checks + 1, "the cut's check started");
    run(&mut app, move |voxel_world| {
        voxel_world
            .stamp(source, Grid::World, incoming_replaces)
            .unwrap()
    });

    assert!(settle(&mut app).is_empty());
    assert!(lifted_log(&app).is_empty());
    assert_eq!(reruns(&app), 0, "dropped as it landed, not run again");
    assert_eq!(
        voxel_of(&mut app, Grid::World, at + BAR_LONG.0),
        Some(WorldVoxel::Solid(STONE)),
        "the stamp landed"
    );
}
