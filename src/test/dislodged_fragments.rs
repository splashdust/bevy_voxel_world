//! End-to-end tests of dislodged fragments in the world: the
//! `FragmentDislodged` messages a game receives after its edits.

use bevy::ecs::system::RunSystemOnce;
use bevy::prelude::*;

use super::{chunk_data, chunk_status, spawn_camera, test_plugins, update_until};
use crate::dislodged_fragments::FragmentChecks;
use crate::grid_bench::fragment_stats;
use crate::prelude::*;

const STONE: u8 = 1;
const ICE: u8 = 2;

/// The cliff's box, from its minimum inclusive to its maximum exclusive.
const CLIFF: (IVec3, IVec3) = (IVec3::new(0, 0, 0), IVec3::new(4, 12, 4));
/// The shelf sticking out of the cliff's side, with air below it.
const SHELF: (IVec3, IVec3) = (IVec3::new(4, 8, 0), IVec3::new(8, 10, 4));
/// The shelf's layer against the cliff, which holds the rest of it up.
const SHELF_ROOT: (IVec3, IVec3) = (IVec3::new(4, 8, 0), IVec3::new(5, 10, 4));
/// The rest of the shelf, which digging out its root cuts free.
const SHELF_TIP: (IVec3, IVec3) = (IVec3::new(5, 8, 0), IVec3::new(8, 10, 4));

fn within((min, max): (IVec3, IVec3), position: IVec3) -> bool {
    position.cmpge(min).all() && position.cmplt(max).all()
}

/// Stone ground below y = 0, with a stone cliff on it and a stone shelf
/// sticking out of the cliff high above the ground: an overhang.
///
/// With `no_cliff`, the shelf floats with nothing under or beside it. With
/// `iced`, the shelf's root is ice but for two stone voxels, at its lowest
/// corners against the cliff.
#[derive(Resource, Clone, Default)]
struct CliffWorld {
    detection: Option<FragmentDetection<u8>>,
    no_cliff: bool,
    iced: bool,
}

impl CliffWorld {
    fn detecting() -> Self {
        Self {
            detection: Some(FragmentDetection::default()),
            ..default()
        }
    }

    fn voxel(&self, position: IVec3) -> WorldVoxel {
        let iced_root = self.iced
            && within(SHELF_ROOT, position)
            && !(position.y == 8 && (position.z == 0 || position.z == 3));
        if position.y < 0 || (!self.no_cliff && within(CLIFF, position)) {
            WorldVoxel::Solid(STONE)
        } else if iced_root {
            WorldVoxel::Solid(ICE)
        } else if within(SHELF, position) {
            WorldVoxel::Solid(STONE)
        } else {
            WorldVoxel::Air
        }
    }
}

impl VoxelWorldConfig for CliffWorld {
    type MaterialIndex = u8;
    type ChunkUserBundle = ();

    fn spawning_distance(&self) -> u32 {
        2
    }

    fn chunk_despawn_strategy(&self) -> ChunkDespawnStrategy {
        ChunkDespawnStrategy::FarAway
    }

    fn voxel_lookup_delegate(&self) -> VoxelLookupDelegate<Self::MaterialIndex> {
        let world = self.clone();
        Box::new(move |_, _, _| {
            let world = world.clone();
            Box::new(move |position, _| world.voxel(position))
        })
    }

    fn detect_dislodged_fragments(&self) -> Option<FragmentDetection<u8>> {
        self.detection.clone()
    }
}

/// Every `FragmentDislodged` received so far.
#[derive(Resource, Default)]
struct Dislodged(Vec<FragmentDislodged<CliffWorld>>);

/// A world of `world` with no camera, so it spawns no chunks and its checks
/// read the world's generator. It records the fragments it receives.
fn fragments_app(world: CliffWorld) -> App {
    let mut app = App::new();
    app.add_plugins((test_plugins(), VoxelWorldPlugin::minimal_with_config(world)))
        .init_resource::<Dislodged>()
        .add_systems(
            Update,
            |mut dislodged: ResMut<Dislodged>,
             mut messages: MessageReader<FragmentDislodged<CliffWorld>>| {
                dislodged.0.extend(messages.read().cloned());
            },
        );
    app.update();
    app
}

/// [`fragments_app`] with a camera, so chunks spawn and publish around the
/// cliff.
fn loaded_fragments_app(world: CliffWorld) -> App {
    let mut app = fragments_app(world);
    app.world_mut()
        .run_system_once(spawn_camera::<CliffWorld>)
        .unwrap();
    update_until(&mut app, "the cliff's chunk to publish", |app| {
        chunk_status::<CliffWorld>(app, IVec3::ZERO) == Some(ChunkStatus::Ready)
    });
    app
}

/// Runs `edit` on the voxel world, then a frame, which commits it.
fn edit(
    app: &mut App,
    mut edit: impl FnMut(&mut VoxelWorld<CliffWorld>) + Send + Sync + 'static,
) {
    app.world_mut()
        .run_system_once(move |mut voxel_world: VoxelWorld<CliffWorld>| {
            edit(&mut voxel_world)
        })
        .unwrap();
    app.update();
}

/// Sets air over the box `(min, max)`.
fn dig(app: &mut App, (min, max): (IVec3, IVec3)) {
    edit(app, move |voxel_world| {
        voxel_world.set_voxels_in(min, max, |_| Some(WorldVoxel::Air))
    });
}

/// Runs frames until no fragment check is running, then a few more, and
/// returns the fragments received since the last call.
fn settle(app: &mut App) -> Vec<FragmentDislodged<CliffWorld>> {
    update_until(app, "the fragment checks to land", |app| {
        app.world()
            .get_resource::<FragmentChecks<CliffWorld>>()
            .is_none_or(|checks| checks.is_idle())
    });
    for _ in 0..3 {
        app.update();
    }
    std::mem::take(&mut app.world_mut().resource_mut::<Dislodged>().0)
}

/// How many fragment checks the world has started.
fn checks_started(app: &App) -> u64 {
    fragment_stats::<CliffWorld>(app.world()).map_or(0, |stats| stats.checks)
}

/// Asserts that `fragment` holds exactly the voxels of `voxels`, with a
/// margin of one voxel around them checked too.
fn assert_holds_exactly(
    fragment: &FragmentDislodged<CliffWorld>,
    voxels: (IVec3, IVec3),
) {
    let (min, max) = voxels;
    assert_eq!((fragment.min, fragment.max), (min, max));
    let size = max - min;
    assert_eq!(fragment.voxel_count, (size.x * size.y * size.z) as u32);
    for z in min.z - 1..=max.z {
        for y in min.y - 1..=max.y {
            for x in min.x - 1..=max.x {
                let position = IVec3::new(x, y, z);
                assert_eq!(
                    fragment.contains(position),
                    within(voxels, position),
                    "{position}"
                );
            }
        }
    }
}

/// Whether any of the app's schedules holds the fragment detection system.
///
/// Systems are told apart by type, since Bevy keeps their names only with
/// its `debug` feature.
fn has_detection_system(app: &App) -> bool {
    let detection = IntoSystem::into_system(FragmentChecks::<CliffWorld>::follow_commit)
        .system_type();
    app.world()
        .resource::<Schedules>()
        .iter()
        .filter_map(|(_, schedule)| schedule.systems().ok())
        .flatten()
        .any(|(_, system)| system.system_type() == detection)
}

#[test]
fn digging_out_the_voxels_holding_up_an_overhang_dislodges_it() {
    let mut app = fragments_app(CliffWorld::detecting());

    dig(&mut app, SHELF_ROOT);
    let dislodged = settle(&mut app);

    assert_eq!(dislodged.len(), 1, "{dislodged:?}");
    assert_holds_exactly(&dislodged[0], SHELF_TIP);
}

#[test]
fn a_dig_into_solid_ground_dislodges_nothing() {
    let mut app = fragments_app(CliffWorld::detecting());

    dig(&mut app, (IVec3::new(20, -5, 20), IVec3::new(21, -4, 21)));

    assert!(settle(&mut app).is_empty());
    assert_eq!(checks_started(&app), 1);
}

#[test]
fn a_piece_larger_than_the_limit_counts_as_attached() {
    let mut app = fragments_app(CliffWorld {
        detection: Some(FragmentDetection {
            limit: 10,
            ..default()
        }),
        ..default()
    });

    dig(&mut app, SHELF_ROOT);

    assert!(settle(&mut app).is_empty());
    assert_eq!(checks_started(&app), 1);
}

#[test]
fn a_support_set_before_the_check_lands_keeps_the_piece_up() {
    let mut app = fragments_app(CliffWorld::detecting());

    dig(&mut app, SHELF_ROOT);
    // A pillar from the ground up to the shelf's underside, set in the frame
    // after the cut.
    edit(&mut app, |voxel_world| {
        voxel_world.set_voxels_in(IVec3::new(6, 0, 1), IVec3::new(7, 8, 2), |_| {
            Some(WorldVoxel::Solid(STONE))
        })
    });

    assert!(settle(&mut app).is_empty());
}

#[test]
fn clearing_the_edits_holding_up_a_piece_of_edits_dislodges_it_before_its_chunk_regenerates(
) {
    let mut app = loaded_fragments_app(CliffWorld::detecting());
    // A slab of edits on a pillar of edits, in the generator's air.
    let slab = (IVec3::new(20, 4, 20), IVec3::new(23, 6, 23));
    let pillar = (IVec3::new(21, 0, 21), IVec3::new(22, 4, 22));
    edit(&mut app, move |voxel_world| {
        for (min, max) in [slab, pillar] {
            voxel_world.set_voxels_in(min, max, |_| Some(WorldVoxel::Solid(STONE)));
        }
    });
    update_until(&mut app, "the pillar to publish", |app| {
        chunk_data::<CliffWorld>(app, IVec3::ZERO).is_some_and(|data| {
            data.get_voxel_at_world_position(IVec3::new(21, 2, 21))
                == Some(WorldVoxel::Solid(STONE))
        })
    });
    assert!(settle(&mut app).is_empty());

    edit(&mut app, move |voxel_world| {
        voxel_world.clear_voxels_in(pillar.0, pillar.1)
    });
    let dislodged = settle(&mut app);

    assert_eq!(dislodged.len(), 1, "{dislodged:?}");
    assert_holds_exactly(&dislodged[0], slab);
}

#[test]
fn an_applied_delta_carrying_a_cut_dislodges_nothing() {
    // The same cut, made in a world that doesn't detect fragments.
    let mut sender = fragments_app(CliffWorld::default());
    let deltas = {
        dig(&mut sender, SHELF_ROOT);
        sender
            .world_mut()
            .run_system_once(|mut messages: MessageReader<EditsCommitted<CliffWorld>>| {
                messages
                    .read()
                    .map(|message| message.delta.clone())
                    .collect::<Vec<_>>()
            })
            .unwrap()
    };
    assert!(!deltas.is_empty());
    let mut app = fragments_app(CliffWorld::detecting());

    edit(&mut app, move |voxel_world| {
        for delta in deltas.clone() {
            voxel_world.apply_delta(delta).unwrap();
        }
    });

    assert!(settle(&mut app).is_empty());
    assert_eq!(checks_started(&app), 0);
}

#[test]
fn forgetting_the_edits_holding_up_a_piece_dislodges_nothing() {
    // The shelf floats, held up by a pillar of edits.
    let mut app = fragments_app(CliffWorld {
        no_cliff: true,
        ..CliffWorld::detecting()
    });
    edit(&mut app, |voxel_world| {
        voxel_world.set_voxels_in(IVec3::new(6, 0, 1), IVec3::new(7, 8, 2), |_| {
            Some(WorldVoxel::Solid(STONE))
        })
    });
    assert!(settle(&mut app).is_empty());

    edit(&mut app, |voxel_world| voxel_world.forget_edits());

    assert!(settle(&mut app).is_empty());
    assert_eq!(checks_started(&app), 0);
}

#[test]
fn regenerating_chunks_without_a_pieces_support_dislodges_nothing() {
    let mut app = loaded_fragments_app(CliffWorld::detecting());

    // The generator now makes no cliff, so the shelf floats.
    app.world_mut().resource_mut::<CliffWorld>().no_cliff = true;
    edit(&mut app, |voxel_world| voxel_world.regenerate_chunks());
    update_until(&mut app, "the cliff's chunk to regenerate", |app| {
        chunk_data::<CliffWorld>(app, IVec3::ZERO).is_some_and(|data| {
            data.get_voxel_at_world_position(IVec3::new(1, 5, 1)) == Some(WorldVoxel::Air)
        })
    });

    assert!(settle(&mut app).is_empty());
    assert_eq!(checks_started(&app), 0);
}

#[test]
fn each_check_reads_the_settings_as_they_are_when_it_starts() {
    let mut app = fragments_app(CliffWorld {
        iced: true,
        ..CliffWorld::detecting()
    });
    let stone_root = |z| (IVec3::new(4, 8, z), IVec3::new(5, 9, z + 1));

    // Ice carries by default, so the iced root still holds the shelf.
    dig(&mut app, stone_root(0));
    assert!(settle(&mut app).is_empty());

    app.world_mut()
        .resource_mut::<CliffWorld>()
        .detection
        .as_mut()
        .unwrap()
        .non_carrying = vec![ICE];
    dig(&mut app, stone_root(3));
    let dislodged = settle(&mut app);

    assert_eq!(dislodged.len(), 1, "{dislodged:?}");
    assert_holds_exactly(&dislodged[0], SHELF_TIP);
}

#[test]
fn a_world_that_does_not_opt_in_has_no_detection_systems() {
    let mut app = fragments_app(CliffWorld::default());

    dig(&mut app, SHELF_ROOT);

    assert!(settle(&mut app).is_empty());
    assert!(!has_detection_system(&app));
    // The same search finds the system in a world that opts in.
    assert!(has_detection_system(
        &fragments_app(CliffWorld::detecting())
    ));
}
