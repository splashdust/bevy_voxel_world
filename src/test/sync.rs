//! Tests of moving edit deltas from one world to another: a sender world
//! commits voxel edits, and a receiver world of another config type applies
//! the deltas it reports.

use bevy::ecs::system::RunSystemOnce;
use bevy::prelude::*;

use super::{
    chunk_data, chunk_status, move_cameras, spawn_camera, test_plugins, update_until,
    FloorWorld, SecondFloorWorld,
};
use crate::edits::{Mask, Voxels};
use crate::prelude::*;

#[cfg(feature = "serde")]
mod serialise;

/// Every edit delta a world `C` committed so far, in commit order.
#[derive(Resource)]
struct Committed<C>(Vec<EditDelta<u8>>, std::marker::PhantomData<C>);

impl<C> Default for Committed<C> {
    fn default() -> Self {
        Self(Vec::new(), std::marker::PhantomData)
    }
}

fn record_committed<C: VoxelWorldConfig<MaterialIndex = u8>>(
    mut committed: ResMut<Committed<C>>,
    mut messages: MessageReader<EditsCommitted<C>>,
) {
    committed
        .0
        .extend(messages.read().map(|message| message.delta.clone()));
}

type Sender = FloorWorld;
type Receiver = SecondFloorWorld;

/// An app with a sender world that has no camera, so it only commits edits,
/// and a receiver world with a camera at the origin.
fn sync_app() -> App {
    let mut app = App::new();
    app.add_plugins((
        test_plugins(),
        VoxelWorldPlugin::<Sender>::minimal(),
        VoxelWorldPlugin::<Receiver>::minimal(),
    ))
    .init_resource::<Committed<Sender>>()
    .init_resource::<Committed<Receiver>>()
    .add_systems(Startup, spawn_camera::<Receiver>)
    .add_systems(
        Update,
        (record_committed::<Sender>, record_committed::<Receiver>),
    );
    app.update();
    app
}

/// Takes the deltas world `C` committed since the last call, sorted by chunk.
fn take_committed<C: VoxelWorldConfig>(app: &mut App) -> Vec<EditDelta<u8>> {
    let mut deltas =
        std::mem::take(&mut app.world_mut().resource_mut::<Committed<C>>().0);
    deltas.sort_by_key(|delta| delta.chunk.to_array());
    deltas
}

/// Runs `edit` on the sender, then a frame, and returns the deltas that
/// frame's commit reported, sorted by chunk.
fn send(
    app: &mut App,
    mut edit: impl FnMut(&mut VoxelWorld<Sender>) + Send + Sync + 'static,
) -> Vec<EditDelta<u8>> {
    take_committed::<Sender>(app);
    app.world_mut()
        .run_system_once(move |mut voxel_world: VoxelWorld<Sender>| {
            edit(&mut voxel_world)
        })
        .unwrap();
    app.update();
    take_committed::<Sender>(app)
}

/// Applies `deltas` to the receiver in one system, in order, returning what
/// each apply returned. The receiver commits them on the next update.
fn apply(app: &mut App, deltas: Vec<EditDelta<u8>>) -> Vec<Result<(), EditGap>> {
    app.world_mut()
        .run_system_once(move |mut voxel_world: VoxelWorld<Receiver>| {
            deltas
                .iter()
                .map(|delta| voxel_world.apply_delta(delta.clone()))
                .collect::<Vec<_>>()
        })
        .unwrap()
}

fn get_voxel<C: VoxelWorldConfig<MaterialIndex = u8>>(
    app: &mut App,
    position: IVec3,
) -> WorldVoxel {
    app.world_mut()
        .run_system_once(move |voxel_world: VoxelWorld<C>| {
            voxel_world.get_voxel(position)
        })
        .unwrap()
}

fn full_delta<C: VoxelWorldConfig<MaterialIndex = u8>>(
    app: &mut App,
    chunk: IVec3,
) -> EditDelta<u8> {
    app.world_mut()
        .run_system_once(move |voxel_world: VoxelWorld<C>| voxel_world.full_delta(chunk))
        .unwrap()
}

fn edited_chunks<C: VoxelWorldConfig<MaterialIndex = u8>>(
    app: &mut App,
) -> Vec<(IVec3, u64)> {
    let mut chunks = app
        .world_mut()
        .run_system_once(|voxel_world: VoxelWorld<C>| voxel_world.edited_chunks())
        .unwrap();
    chunks.sort_by_key(|(chunk, _)| chunk.to_array());
    chunks
}

/// Waits until the receiver's published chunk data shows `voxel` at
/// `position`.
fn published(app: &mut App, position: IVec3, voxel: WorldVoxel) {
    let chunk = position.div_euclid(IVec3::splat(32));
    update_until(
        app,
        &format!("{voxel:?} at {position} to be published"),
        |app| {
            chunk_data::<Receiver>(app, chunk)
                .and_then(|data| data.get_voxel_at_world_position(position))
                == Some(voxel)
        },
    );
}

#[test]
fn a_receiver_reads_the_voxels_the_sender_set() {
    let mut app = sync_app();
    update_until(&mut app, "the receiver's origin chunk to be ready", |app| {
        chunk_status::<Receiver>(app, IVec3::ZERO) == Some(ChunkStatus::Ready)
    });

    // A few voxels across two chunks, and a whole chunk below the origin.
    let deltas = send(&mut app, |voxel_world| {
        voxel_world.set_voxel(IVec3::new(3, 4, 5), WorldVoxel::Solid(7));
        voxel_world.set_voxel(IVec3::new(-1, 1, 1), WorldVoxel::Solid(8));
        voxel_world.set_voxels_in(IVec3::new(0, -32, 0), IVec3::new(32, 0, 32), |p| {
            Some(WorldVoxel::Solid((p.x % 5) as u8 + 2))
        });
    });
    assert_eq!(deltas.len(), 3);
    assert!(matches!(deltas[1].voxels, Voxels::Dense { .. }));

    assert_eq!(apply(&mut app, deltas.clone()), vec![Ok(()); 3]);
    app.update();

    for position in [
        IVec3::new(3, 4, 5),
        IVec3::new(-1, 1, 1),
        IVec3::new(7, -3, 9),
        IVec3::new(31, -32, 31),
    ] {
        assert_eq!(
            get_voxel::<Receiver>(&mut app, position),
            get_voxel::<Sender>(&mut app, position),
            "at {position}"
        );
    }
    assert_eq!(
        edited_chunks::<Receiver>(&mut app),
        edited_chunks::<Sender>(&mut app)
    );
    // The receiver's own commit reports the same deltas, at the sequences it
    // received.
    assert_eq!(take_committed::<Receiver>(&mut app), deltas);
    published(&mut app, IVec3::new(3, 4, 5), WorldVoxel::Solid(7));
    published(&mut app, IVec3::new(7, -3, 9), WorldVoxel::Solid(4));
}

#[test]
fn a_missed_delta_is_reported_as_a_gap_and_repaired_by_the_full_delta() {
    let mut app = sync_app();
    let first = send(&mut app, |voxel_world| {
        voxel_world.set_voxel(IVec3::new(1, 1, 1), WorldVoxel::Solid(1));
    });
    let missed = send(&mut app, |voxel_world| {
        voxel_world.set_voxel(IVec3::new(2, 1, 1), WorldVoxel::Solid(2));
    });
    let after_the_gap = send(&mut app, |voxel_world| {
        voxel_world.set_voxel(IVec3::new(3, 1, 1), WorldVoxel::Solid(3));
        voxel_world.set_voxel(IVec3::new(4, 1, 1), WorldVoxel::Solid(4));
    });
    assert_eq!(apply(&mut app, first), vec![Ok(())]);
    app.update();

    assert_eq!(
        apply(&mut app, after_the_gap),
        vec![Err(EditGap {
            chunk: IVec3::ZERO,
            local: 1,
            received: 4,
        })]
    );
    app.update();
    assert_ne!(
        get_voxel::<Receiver>(&mut app, IVec3::new(3, 1, 1)),
        WorldVoxel::Solid(3)
    );
    assert_eq!(edited_chunks::<Receiver>(&mut app), vec![(IVec3::ZERO, 1)]);

    let full = full_delta::<Sender>(&mut app, IVec3::ZERO);
    assert!(full.full);
    assert_eq!((full.seq, full.len()), (4, 4));
    assert_eq!(apply(&mut app, vec![full]), vec![Ok(())]);
    app.update();
    for x in 1..=4 {
        assert_eq!(
            get_voxel::<Receiver>(&mut app, IVec3::new(x, 1, 1)),
            WorldVoxel::Solid(x as u8)
        );
    }
    assert_eq!(edited_chunks::<Receiver>(&mut app), vec![(IVec3::ZERO, 4)]);

    // The delta it missed is older than the full delta, so already applied.
    assert_eq!(apply(&mut app, missed), vec![Ok(())]);
}

#[test]
fn an_older_full_delta_is_skipped_and_an_equal_one_changes_nothing() {
    let mut app = sync_app();
    let position = IVec3::new(5, 5, 5);
    send(&mut app, move |voxel_world| {
        voxel_world.set_voxel(position, WorldVoxel::Solid(1));
    });
    let older = full_delta::<Sender>(&mut app, IVec3::ZERO);
    send(&mut app, move |voxel_world| {
        voxel_world.set_voxel(position, WorldVoxel::Solid(2));
    });
    let newer = full_delta::<Sender>(&mut app, IVec3::ZERO);
    assert_eq!(apply(&mut app, vec![newer.clone()]), vec![Ok(())]);
    app.update();
    take_committed::<Receiver>(&mut app);

    assert_eq!(apply(&mut app, vec![older, newer]), vec![Ok(()), Ok(())]);
    app.update();

    assert_eq!(
        get_voxel::<Receiver>(&mut app, position),
        WorldVoxel::Solid(2)
    );
    assert_eq!(edited_chunks::<Receiver>(&mut app), vec![(IVec3::ZERO, 2)]);
    assert_eq!(take_committed::<Receiver>(&mut app), vec![]);
}

#[test]
fn two_deltas_for_one_chunk_in_one_frame_are_not_a_gap() {
    let mut app = sync_app();
    let mut deltas = send(&mut app, |voxel_world| {
        voxel_world.set_voxel(IVec3::new(1, 1, 1), WorldVoxel::Solid(1));
    });
    deltas.extend(send(&mut app, |voxel_world| {
        voxel_world.set_voxel(IVec3::new(2, 1, 1), WorldVoxel::Solid(2));
        voxel_world.set_voxel(IVec3::new(3, 1, 1), WorldVoxel::Solid(3));
    }));

    assert_eq!(apply(&mut app, deltas), vec![Ok(()), Ok(())]);
    app.update();

    assert_eq!(edited_chunks::<Receiver>(&mut app), vec![(IVec3::ZERO, 3)]);
    for x in 1..=3 {
        assert_eq!(
            get_voxel::<Receiver>(&mut app, IVec3::new(x, 1, 1)),
            WorldVoxel::Solid(x as u8)
        );
    }
}

#[test]
fn deltas_setting_a_voxel_and_back_in_one_frame_report_a_delta_that_continues() {
    let mut app = sync_app();
    let position = IVec3::new(1, 1, 1);
    let first = send(&mut app, move |voxel_world| {
        voxel_world.set_voxel(position, WorldVoxel::Solid(1));
    });
    assert_eq!(apply(&mut app, first), vec![Ok(())]);
    app.update();
    take_committed::<Receiver>(&mut app);
    let mut there_and_back = send(&mut app, move |voxel_world| {
        voxel_world.set_voxel(position, WorldVoxel::Solid(2));
    });
    there_and_back.extend(send(&mut app, move |voxel_world| {
        voxel_world.set_voxel(position, WorldVoxel::Solid(1));
    }));

    assert_eq!(apply(&mut app, there_and_back), vec![Ok(()), Ok(())]);
    app.update();

    // The receiver's voxel ends where it was, but its sequence moved from 1
    // to 3, so a world it forwards to must be able to follow.
    let committed = take_committed::<Receiver>(&mut app);
    assert_eq!(committed.len(), 1);
    let delta = &committed[0];
    assert_eq!(delta.seq, 3);
    assert!(
        delta.full || delta.seq - delta.len() as u64 == 1,
        "{delta:?} doesn't continue from sequence 1"
    );
    assert_eq!(
        get_voxel::<Receiver>(&mut app, position),
        WorldVoxel::Solid(1)
    );
}

#[test]
fn a_full_delta_replaces_the_edits_and_sets_before_it_but_not_after() {
    let mut app = sync_app();
    update_until(&mut app, "the receiver's origin chunk to be ready", |app| {
        chunk_status::<Receiver>(app, IVec3::ZERO) == Some(ChunkStatus::Ready)
    });
    let (committed, set_before, carried, set_after) = (
        IVec3::new(1, 2, 1),
        IVec3::new(2, 2, 1),
        IVec3::new(3, 2, 1),
        IVec3::new(4, 2, 1),
    );
    app.world_mut()
        .run_system_once(move |mut voxel_world: VoxelWorld<Receiver>| {
            voxel_world.set_voxel(committed, WorldVoxel::Solid(9));
        })
        .unwrap();
    published(&mut app, committed, WorldVoxel::Solid(9));
    send(&mut app, move |voxel_world| {
        voxel_world.set_voxel(carried, WorldVoxel::Solid(3));
    });
    let full = full_delta::<Sender>(&mut app, IVec3::ZERO);
    take_committed::<Receiver>(&mut app);

    let applied = app
        .world_mut()
        .run_system_once(move |mut voxel_world: VoxelWorld<Receiver>| {
            voxel_world.set_voxel(set_before, WorldVoxel::Solid(9));
            let applied = voxel_world.apply_delta(full.clone());
            voxel_world.set_voxel(set_after, WorldVoxel::Solid(4));
            applied
        })
        .unwrap();
    assert_eq!(applied, Ok(()));
    app.update();

    // The edit the full delta doesn't carry is gone once the chunk
    // regenerates.
    published(&mut app, committed, WorldVoxel::Air);
    assert_eq!(get_voxel::<Receiver>(&mut app, committed), WorldVoxel::Air);
    assert_eq!(get_voxel::<Receiver>(&mut app, set_before), WorldVoxel::Air);
    assert_eq!(
        get_voxel::<Receiver>(&mut app, carried),
        WorldVoxel::Solid(3)
    );
    assert_eq!(
        get_voxel::<Receiver>(&mut app, set_after),
        WorldVoxel::Solid(4)
    );
    // The full delta's sequence, and one more for the set after it.
    assert_eq!(edited_chunks::<Receiver>(&mut app), vec![(IVec3::ZERO, 2)]);
    let reported = take_committed::<Receiver>(&mut app);
    assert_eq!(reported.len(), 1);
    assert!(reported[0].full);
    assert_eq!(
        reported[0].iter().collect::<Vec<_>>(),
        vec![
            (carried, WorldVoxel::Solid(3)),
            (set_after, WorldVoxel::Solid(4))
        ]
    );
}

#[test]
fn a_delta_for_a_chunk_not_spawned_shows_once_it_spawns() {
    let mut app = sync_app();
    let far = IVec3::new(20 * 32 + 3, 2, 4);
    let deltas = send(&mut app, move |voxel_world| {
        voxel_world.set_voxel(far, WorldVoxel::Solid(6));
    });
    assert_eq!(
        chunk_status::<Receiver>(&mut app, IVec3::new(20, 0, 0)),
        None
    );

    assert_eq!(apply(&mut app, deltas), vec![Ok(())]);
    app.update();
    move_cameras::<Receiver>(
        &mut app,
        Transform::from_xyz(far.x as f32 + 10.0, 10.0, 10.0)
            .looking_at(far.as_vec3(), Vec3::Y),
    );

    published(&mut app, far, WorldVoxel::Solid(6));
}

#[test]
fn a_delta_whose_voxels_do_not_fit_a_chunk_is_reported_as_a_gap() {
    let mut app = sync_app();
    let past_the_chunk = EditDelta {
        chunk: IVec3::ZERO,
        seq: 1,
        full: false,
        voxels: Voxels::Sparse(vec![(32768, WorldVoxel::Solid(1))]),
    };
    let mut mask = Mask::default();
    mask.insert(0);
    mask.insert(1);
    let short = EditDelta {
        chunk: IVec3::ZERO,
        seq: 1,
        full: true,
        voxels: Voxels::Dense {
            mask,
            voxels: vec![WorldVoxel::Solid(1)],
        },
    };

    let gap = Err(EditGap {
        chunk: IVec3::ZERO,
        local: 0,
        received: 1,
    });
    assert_eq!(apply(&mut app, vec![past_the_chunk, short]), vec![gap, gap]);
    app.update();
    assert_eq!(edited_chunks::<Receiver>(&mut app), vec![]);
}

#[test]
fn a_delta_past_the_voxel_positions_or_the_highest_sequence_is_reported_as_a_gap() {
    let mut app = sync_app();
    let one_voxel = |chunk: IVec3, seq: u64| EditDelta {
        chunk,
        seq,
        full: true,
        voxels: Voxels::Sparse(vec![(0, WorldVoxel::Solid(1))]),
    };
    let gap = |chunk: IVec3, received: u64| {
        Err(EditGap {
            chunk,
            local: 0,
            received,
        })
    };
    // The chunks holding the lowest and the highest voxel positions, and the
    // ones past them, which hold no `i32` voxel position.
    let lowest = crate::chunk::chunk_containing(IVec3::MIN);
    let highest = crate::chunk::chunk_containing(IVec3::MAX);
    let highest_seq = 1 << 48;

    let results = apply(
        &mut app,
        vec![
            one_voxel(lowest, 1),
            one_voxel(highest, 1),
            one_voxel(lowest - IVec3::X, 1),
            one_voxel(highest + IVec3::Z, 1),
            one_voxel(IVec3::ZERO, highest_seq),
            one_voxel(IVec3::Y, highest_seq + 1),
            one_voxel(IVec3::Y, u64::MAX),
        ],
    );
    assert_eq!(
        results,
        vec![
            Ok(()),
            Ok(()),
            gap(lowest - IVec3::X, 1),
            gap(highest + IVec3::Z, 1),
            Ok(()),
            gap(IVec3::Y, highest_seq + 1),
            gap(IVec3::Y, u64::MAX),
        ]
    );
    app.update();
    assert_eq!(
        edited_chunks::<Receiver>(&mut app),
        vec![(lowest, 1), (IVec3::ZERO, highest_seq), (highest, 1)]
    );
    assert_eq!(
        get_voxel::<Receiver>(&mut app, IVec3::MIN),
        WorldVoxel::Solid(1)
    );
}

#[test]
fn the_export_methods_read_committed_edits_only() {
    let mut app = sync_app();
    assert_eq!(
        full_delta::<Sender>(&mut app, IVec3::ZERO),
        EditDelta {
            chunk: IVec3::ZERO,
            seq: 0,
            full: true,
            voxels: Voxels::Sparse(vec![]),
        }
    );

    // A whole chunk of edits, then one voxel set but not yet committed.
    send(&mut app, |voxel_world| {
        voxel_world.set_voxels_in(IVec3::ZERO, IVec3::splat(32), |_| {
            Some(WorldVoxel::Solid(1))
        });
    });
    app.world_mut()
        .run_system_once(|mut voxel_world: VoxelWorld<Sender>| {
            voxel_world.set_voxel(IVec3::new(0, 0, 40), WorldVoxel::Solid(2));
        })
        .unwrap();

    let full = full_delta::<Sender>(&mut app, IVec3::ZERO);
    assert!(matches!(full.voxels, Voxels::Dense { .. }));
    assert_eq!((full.seq, full.len()), (32768, 32768));
    assert_eq!(
        edited_chunks::<Sender>(&mut app),
        vec![(IVec3::ZERO, 32768)]
    );
}

/// Waits until the receiver's origin chunk is Ready.
fn origin_ready(app: &mut App) {
    update_until(app, "the receiver's origin chunk to be ready", |app| {
        chunk_status::<Receiver>(app, IVec3::ZERO) == Some(ChunkStatus::Ready)
    });
}

/// Runs `edit` on the receiver, in one system.
fn on_receiver<T: Send + 'static>(
    app: &mut App,
    mut edit: impl FnMut(&mut VoxelWorld<Receiver>) -> T + Send + Sync + 'static,
) -> T {
    app.world_mut()
        .run_system_once(move |mut voxel_world: VoxelWorld<Receiver>| {
            edit(&mut voxel_world)
        })
        .unwrap()
}

#[test]
fn a_clear_reaches_a_receiver_as_an_empty_full_delta() {
    let mut app = sync_app();
    origin_ready(&mut app);
    // A voxel placed in the air and one dug out of the floor.
    let (placed, dug) = (IVec3::new(3, 2, 3), IVec3::new(5, -2, 5));
    let deltas = send(&mut app, move |voxel_world| {
        voxel_world.set_voxel(placed, WorldVoxel::Solid(5));
        voxel_world.set_voxel(dug, WorldVoxel::Air);
        voxel_world.set_voxel(IVec3::new(6, 2, 6), WorldVoxel::Solid(6));
    });
    assert_eq!(deltas.len(), 2);
    assert_eq!(apply(&mut app, deltas), vec![Ok(()); 2]);
    app.update();
    published(&mut app, placed, WorldVoxel::Solid(5));
    published(&mut app, dug, WorldVoxel::Air);

    let cleared = send(&mut app, |voxel_world| {
        voxel_world.clear_voxels_in(IVec3::splat(-32), IVec3::splat(32));
    });

    // Both chunks moved on by the edits they lost, and carry none.
    let empty = |chunk: IVec3, seq| EditDelta {
        chunk,
        seq,
        full: true,
        voxels: Voxels::Sparse(vec![]),
    };
    assert_eq!(cleared, vec![empty(IVec3::NEG_Y, 2), empty(IVec3::ZERO, 4)]);
    assert_eq!(
        edited_chunks::<Sender>(&mut app),
        vec![(IVec3::NEG_Y, 2), (IVec3::ZERO, 4)]
    );

    assert_eq!(apply(&mut app, cleared), vec![Ok(()); 2]);
    app.update();
    published(&mut app, placed, WorldVoxel::Air);
    published(&mut app, dug, WorldVoxel::Solid(1));
    assert_eq!(get_voxel::<Receiver>(&mut app, placed), WorldVoxel::Air);
    assert_eq!(get_voxel::<Receiver>(&mut app, dug), WorldVoxel::Solid(1));
    assert_eq!(
        edited_chunks::<Receiver>(&mut app),
        vec![(IVec3::NEG_Y, 2), (IVec3::ZERO, 4)]
    );
}

#[test]
fn a_partial_clear_sends_the_edits_left_at_the_new_sequence() {
    let mut app = sync_app();
    send(&mut app, |voxel_world| {
        for x in 1..=4 {
            voxel_world.set_voxel(IVec3::new(x, 1, 1), WorldVoxel::Solid(x as u8));
        }
    });

    let cleared = send(&mut app, |voxel_world| {
        voxel_world.clear_voxels_in(IVec3::new(1, 0, 0), IVec3::new(3, 32, 32));
    });

    assert_eq!(cleared.len(), 1);
    assert!(cleared[0].full);
    assert_eq!(cleared[0].seq, 6);
    let mut left: Vec<_> = cleared[0].iter().collect();
    left.sort_by_key(|(position, _)| position.x);
    assert_eq!(
        left,
        vec![
            (IVec3::new(3, 1, 1), WorldVoxel::Solid(3)),
            (IVec3::new(4, 1, 1), WorldVoxel::Solid(4)),
        ]
    );
    // A box holding no edits clears nothing.
    let nothing = send(&mut app, |voxel_world| {
        voxel_world.clear_voxels_in(IVec3::new(0, 10, 0), IVec3::new(32, 20, 32));
        voxel_world.clear_voxels_in(IVec3::new(64, 0, 0), IVec3::new(96, 32, 32));
    });
    assert_eq!(nothing, vec![]);
}

#[test]
fn a_clear_drops_the_voxels_set_before_it_in_its_frame_but_not_after() {
    let mut app = sync_app();
    origin_ready(&mut app);
    let (committed, set_before, set_after, outside) = (
        IVec3::new(1, 2, 1),
        IVec3::new(2, 2, 1),
        IVec3::new(3, 2, 1),
        IVec3::new(20, 2, 1),
    );
    on_receiver(&mut app, move |voxel_world| {
        voxel_world.set_voxel(committed, WorldVoxel::Solid(9));
        voxel_world.set_voxel(outside, WorldVoxel::Solid(9));
    });
    published(&mut app, committed, WorldVoxel::Solid(9));
    take_committed::<Receiver>(&mut app);

    let read_back = on_receiver(&mut app, move |voxel_world| {
        voxel_world.set_voxel(set_before, WorldVoxel::Solid(2));
        voxel_world.clear_voxels_in(IVec3::ZERO, IVec3::new(10, 10, 10));
        voxel_world.set_voxel(set_after, WorldVoxel::Solid(3));
        // A cleared voxel reads what its chunk last published.
        (
            voxel_world.get_voxel(committed),
            voxel_world.get_voxel(set_before),
        )
    });
    assert_eq!(read_back, (WorldVoxel::Solid(9), WorldVoxel::Air));
    app.update();

    published(&mut app, committed, WorldVoxel::Air);
    published(&mut app, set_after, WorldVoxel::Solid(3));
    assert_eq!(get_voxel::<Receiver>(&mut app, set_before), WorldVoxel::Air);
    assert_eq!(
        get_voxel::<Receiver>(&mut app, outside),
        WorldVoxel::Solid(9)
    );
    // Two edits, then one removed and one set.
    assert_eq!(edited_chunks::<Receiver>(&mut app), vec![(IVec3::ZERO, 4)]);
    let reported = take_committed::<Receiver>(&mut app);
    assert_eq!(reported.len(), 1);
    let mut carried: Vec<_> = reported[0].iter().collect();
    carried.sort_by_key(|(position, _)| position.x);
    assert_eq!(
        carried,
        vec![
            (set_after, WorldVoxel::Solid(3)),
            (outside, WorldVoxel::Solid(9))
        ]
    );
}

#[test]
fn clearing_every_edit_leaves_each_edited_chunk_empty() {
    let mut app = sync_app();
    send(&mut app, |voxel_world| {
        voxel_world.set_voxel(IVec3::new(1, 1, 1), WorldVoxel::Solid(1));
        voxel_world.set_voxels_in(IVec3::new(0, 0, 384), IVec3::new(32, 32, 416), |_| {
            Some(WorldVoxel::Solid(2))
        });
    });

    let cleared = send(&mut app, |voxel_world| voxel_world.clear_edits());

    assert!(cleared.iter().all(|delta| delta.full && delta.is_empty()));
    assert_eq!(
        edited_chunks::<Sender>(&mut app),
        vec![(IVec3::ZERO, 2), (IVec3::new(0, 0, 12), 65536)]
    );
}

/// The full delta of every chunk the receiver has edits in, as a save holds
/// them.
fn save(app: &mut App) -> Vec<EditDelta<u8>> {
    edited_chunks::<Receiver>(app)
        .into_iter()
        .map(|(chunk, _)| full_delta::<Receiver>(app, chunk))
        .collect()
}

#[test]
fn forgetting_then_applying_a_save_restores_its_sequences() {
    let mut app = sync_app();
    origin_ready(&mut app);
    let (kept, later) = (IVec3::new(1, 1, 1), IVec3::new(-3, 1, 1));
    on_receiver(&mut app, move |voxel_world| {
        voxel_world.set_voxel(kept, WorldVoxel::Solid(4));
    });
    app.update();
    let saved = save(&mut app);
    let saved_chunks = edited_chunks::<Receiver>(&mut app);
    assert_eq!(saved_chunks, vec![(IVec3::ZERO, 1)]);

    // More edits after the save, in the saved chunk and in another one.
    on_receiver(&mut app, move |voxel_world| {
        voxel_world.set_voxel(kept, WorldVoxel::Solid(5));
        voxel_world.set_voxel(kept + IVec3::X, WorldVoxel::Solid(5));
        voxel_world.set_voxel(later, WorldVoxel::Solid(6));
    });
    published(&mut app, later, WorldVoxel::Solid(6));
    take_committed::<Receiver>(&mut app);

    let applied = on_receiver(&mut app, move |voxel_world| {
        voxel_world.forget_edits();
        saved
            .iter()
            .map(|delta| voxel_world.apply_delta(delta.clone()))
            .collect::<Vec<_>>()
    });
    assert_eq!(applied, vec![Ok(())]);
    app.update();

    assert_eq!(edited_chunks::<Receiver>(&mut app), saved_chunks);
    published(&mut app, later, WorldVoxel::Air);
    published(&mut app, kept + IVec3::X, WorldVoxel::Air);
    assert_eq!(get_voxel::<Receiver>(&mut app, kept), WorldVoxel::Solid(4));
}

#[test]
fn forgetting_edits_reports_nothing_and_leaves_no_sequence() {
    let mut app = sync_app();
    origin_ready(&mut app);
    let position = IVec3::new(2, -1, 2);
    on_receiver(&mut app, move |voxel_world| {
        voxel_world.set_voxel(position, WorldVoxel::Air);
    });
    published(&mut app, position, WorldVoxel::Air);
    take_committed::<Receiver>(&mut app);

    on_receiver(&mut app, |voxel_world| voxel_world.forget_edits());
    app.update();

    assert_eq!(take_committed::<Receiver>(&mut app), vec![]);
    assert_eq!(edited_chunks::<Receiver>(&mut app), vec![]);
    assert_eq!(full_delta::<Receiver>(&mut app, IVec3::NEG_Y).seq, 0);
    published(&mut app, position, WorldVoxel::Solid(1));
    // The next incremental delta for the chunk now reads as a gap.
    send(&mut app, move |voxel_world| {
        voxel_world.set_voxel(position, WorldVoxel::Air);
        voxel_world.set_voxel(position + IVec3::X, WorldVoxel::Air);
    });
    let after = send(&mut app, move |voxel_world| {
        voxel_world.set_voxel(position + IVec3::Z, WorldVoxel::Air);
    });
    assert_eq!(
        apply(&mut app, after),
        vec![Err(EditGap {
            chunk: IVec3::NEG_Y,
            local: 0,
            received: 3,
        })]
    );
}
