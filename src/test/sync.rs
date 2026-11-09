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
