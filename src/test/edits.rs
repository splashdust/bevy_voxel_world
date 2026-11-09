//! Tests of the edit deltas a world's commit reports.

use bevy::ecs::system::RunSystemOnce;
use bevy::prelude::*;

use super::{test_plugins, FloorWorld};
use crate::edits::Voxels;
use crate::prelude::*;

/// Every edit delta committed so far, in commit order.
#[derive(Resource, Default)]
struct Committed(Vec<EditDelta<u8>>);

fn record_committed(
    mut committed: ResMut<Committed>,
    mut messages: MessageReader<EditsCommitted<FloorWorld>>,
) {
    committed
        .0
        .extend(messages.read().map(|message| message.delta.clone()));
}

/// A world with no camera, so it spawns no chunks and only commits edits.
fn edits_app() -> App {
    let mut app = App::new();
    app.add_plugins((test_plugins(), VoxelWorldPlugin::<FloorWorld>::minimal()))
        .init_resource::<Committed>()
        .add_systems(Update, record_committed);
    app.update();
    app
}

/// Runs `edit` on the voxel world, then a frame, and returns the deltas that
/// frame's commit reported, sorted by chunk.
fn commit(
    app: &mut App,
    mut edit: impl FnMut(&mut VoxelWorld<FloorWorld>) + Send + Sync + 'static,
) -> Vec<EditDelta<u8>> {
    app.world_mut()
        .run_system_once(move |mut voxel_world: VoxelWorld<FloorWorld>| {
            edit(&mut voxel_world)
        })
        .unwrap();
    app.update();
    let mut deltas = std::mem::take(&mut app.world_mut().resource_mut::<Committed>().0);
    deltas.sort_by_key(|delta| delta.chunk.to_array());
    deltas
}

#[test]
fn a_commit_reports_one_delta_per_edited_chunk() {
    let mut app = edits_app();

    let deltas = commit(&mut app, |voxel_world| {
        voxel_world.set_voxel(IVec3::new(1, 2, 3), WorldVoxel::Solid(4));
        voxel_world.set_voxel(IVec3::new(-1, 2, 3), WorldVoxel::Unset);
    });

    assert_eq!(
        deltas,
        vec![
            EditDelta {
                chunk: IVec3::new(-1, 0, 0),
                seq: 1,
                full: false,
                // Local (31, 2, 3).
                voxels: Voxels::Sparse(vec![(3167, WorldVoxel::Unset)]),
            },
            EditDelta {
                chunk: IVec3::ZERO,
                seq: 1,
                full: false,
                // Local (1, 2, 3).
                voxels: Voxels::Sparse(vec![(3137, WorldVoxel::Solid(4))]),
            },
        ]
    );
}

#[test]
fn each_delta_continues_from_the_chunks_previous_sequence() {
    let mut app = edits_app();
    let mut previous = 0;

    // Each round sets one more voxel than the last, all but the newest with
    // the value of the round before, so only some of them change.
    for round in 0..4u8 {
        let deltas = commit(&mut app, move |voxel_world| {
            voxel_world.set_voxels_in(
                IVec3::ZERO,
                IVec3::new(round as i32 + 3, 1, 1),
                |position| Some(WorldVoxel::Solid((position.x as u8).min(round))),
            );
        });

        assert_eq!(deltas.len(), 1);
        let delta = &deltas[0];
        assert_eq!(delta.seq - delta.len() as u64, previous, "round {round}");
        previous = delta.seq;
    }
    // Round 0 sets x 0 to 2; round 1 changes x 1 to 3; round 2 x 2 to 4;
    // round 3 x 3 to 5.
    assert_eq!(previous, 3 + 3 + 3 + 3);
}

#[test]
fn a_commit_setting_nothing_new_reports_no_delta() {
    let mut app = edits_app();
    commit(&mut app, |voxel_world| {
        voxel_world.set_voxel(IVec3::ONE, WorldVoxel::Air);
    });

    let deltas = commit(&mut app, |voxel_world| {
        voxel_world.set_voxel(IVec3::ONE, WorldVoxel::Air);
    });

    assert_eq!(deltas, vec![]);
}

/// The delta of a region setting the voxels of the chunk at the origin whose
/// index is below `count`.
fn delta_of_first_voxels(count: i32) -> EditDelta<u8> {
    let mut app = edits_app();
    let mut deltas = commit(&mut app, move |voxel_world| {
        voxel_world.set_voxels_in(IVec3::ZERO, IVec3::splat(32), |position| {
            let index = position.x + 32 * position.y + 1024 * position.z;
            (index < count).then_some(WorldVoxel::Solid((index % 7) as u8))
        });
    });
    assert_eq!(deltas.len(), 1);
    deltas.remove(0)
}

#[test]
fn a_delta_of_fewer_than_2048_voxels_is_sparse() {
    let delta = delta_of_first_voxels(2047);

    let Voxels::Sparse(voxels) = &delta.voxels else {
        panic!("dense delta of {} voxels", delta.len());
    };
    assert_eq!(voxels.len(), 2047);
    assert_eq!(voxels[1000], (1000, WorldVoxel::Solid(6)));
}

#[test]
fn a_delta_of_2048_voxels_or_more_is_dense() {
    let delta = delta_of_first_voxels(2048);

    let Voxels::Dense { mask, voxels } = &delta.voxels else {
        panic!("sparse delta of {} voxels", delta.len());
    };
    assert_eq!(voxels.len(), 2048);
    assert_eq!(mask.len(), 2048);
    assert!(mask.contains(2047));
    assert!(!mask.contains(2048));
    assert_eq!(voxels[1000], WorldVoxel::Solid(6));
    // Index 1000 is local (8, 31, 0).
    assert!(delta
        .iter()
        .any(|voxel| voxel == (IVec3::new(8, 31, 0), WorldVoxel::Solid(6))));
}

#[test]
fn a_delta_names_its_voxels_by_world_position() {
    let mut app = edits_app();
    let below_origin = IVec3::new(-1, -33, 40);

    let deltas = commit(&mut app, move |voxel_world| {
        voxel_world.set_voxel(below_origin, WorldVoxel::Solid(2));
    });

    assert_eq!(deltas[0].chunk, IVec3::new(-1, -2, 1));
    assert_eq!(
        deltas[0].iter().collect::<Vec<_>>(),
        vec![(below_origin, WorldVoxel::Solid(2))]
    );
}
