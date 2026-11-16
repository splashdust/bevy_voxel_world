//! Tests of moving a sub-grid's edit deltas from one world to another: a
//! sender world commits a sub-grid's voxel edits, and a receiver world of
//! another config type applies them to a sub-grid of its own.

use std::collections::BTreeSet;
use std::marker::PhantomData;

use bevy::diagnostic::FrameCount;
use bevy::ecs::system::RunSystemOnce;
use bevy::prelude::*;

use super::super::{
    floor_voxel, record_chunk_messages, recorded, spawn_camera, test_plugins,
    update_until, FloorWorld,
};
use crate::chunk::CHUNK_SIZE_I;
use crate::edits::Voxels;
use crate::prelude::*;

type Sender = FloorWorld;

/// A world with solid ground below y = 0. Once `slow`, it runs two chunk
/// tasks at a time, so a commit touching many chunks takes several frames to
/// publish them.
#[derive(Resource, Clone, Default)]
struct Receiver {
    slow: bool,
}

impl VoxelWorldConfig for Receiver {
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
        Box::new(|_, _, _| Box::new(|pos, _| floor_voxel(pos)))
    }
}

/// The deltas one commit of a world reported: the world's, and each
/// sub-grid's with the sub-grid, each sorted by chunk.
#[derive(Debug, Default)]
struct Commit {
    world: Vec<EditDelta<u8>>,
    sub_grids: Vec<(Entity, EditDelta<u8>)>,
}

impl Commit {
    /// The sub-grids' deltas, without their sub-grids.
    fn sub_grid_deltas(&self) -> Vec<EditDelta<u8>> {
        self.sub_grids
            .iter()
            .map(|(_, delta)| delta.clone())
            .collect()
    }
}

/// Every delta a world `C` committed since it was last taken.
#[derive(Resource)]
struct Sent<C>(Commit, PhantomData<C>);

impl<C> Default for Sent<C> {
    fn default() -> Self {
        Self(Commit::default(), PhantomData)
    }
}

fn record<C: VoxelWorldConfig<MaterialIndex = u8>>(
    mut sent: ResMut<Sent<C>>,
    mut world: MessageReader<EditsCommitted<C>>,
    mut sub_grids: MessageReader<SubGridEditsCommitted<C>>,
) {
    let sent = &mut sent.0;
    sent.world
        .extend(world.read().map(|message| message.delta.clone()));
    sent.sub_grids.extend(
        sub_grids
            .read()
            .map(|message| (message.sub_grid, message.delta.clone())),
    );
}

/// An app with a sender world that has no camera, so it only commits edits,
/// and a receiver world with a camera at the origin, whose chunk messages are
/// recorded.
fn sync_app() -> App {
    let mut app = App::new();
    app.add_plugins((
        test_plugins(),
        TransformPlugin,
        VoxelWorldPlugin::<Sender>::minimal(),
        VoxelWorldPlugin::<Receiver>::minimal(),
    ))
    .init_resource::<Sent<Sender>>()
    .init_resource::<Sent<Receiver>>()
    .add_systems(Update, (record::<Sender>, record::<Receiver>));
    record_chunk_messages::<ChunkPublished<Receiver>>(&mut app);
    record_chunk_messages::<ChunkWillDespawn<Receiver>>(&mut app);
    app.world_mut()
        .run_system_once(spawn_camera::<Receiver>)
        .unwrap();
    app.update();
    app
}

/// Runs `f` on world `C` in a system of its own, whose commands apply when it
/// returns.
fn on<C: VoxelWorldConfig, T: 'static>(
    app: &mut App,
    mut f: impl FnMut(&mut VoxelWorld<C>) -> T + Send + Sync + 'static,
) -> T {
    app.world_mut()
        .run_system_once(move |mut voxel_world: VoxelWorld<C>| f(&mut voxel_world))
        .unwrap()
}

/// Takes the deltas world `C` committed since the last call.
fn take<C: VoxelWorldConfig>(app: &mut App) -> Commit {
    let mut commit = std::mem::take(&mut app.world_mut().resource_mut::<Sent<C>>().0);
    commit.world.sort_by_key(|delta| delta.chunk.to_array());
    commit
        .sub_grids
        .sort_by_key(|(_, delta)| delta.chunk.to_array());
    commit
}

/// Runs `edit` on the sender, then a frame, and returns what that frame's
/// commit reported.
fn send<T: 'static>(
    app: &mut App,
    edit: impl FnMut(&mut VoxelWorld<Sender>) -> T + Send + Sync + 'static,
) -> (T, Commit) {
    take::<Sender>(app);
    let made = on(app, edit);
    app.update();
    (made, take::<Sender>(app))
}

/// Applies `commit`'s deltas to the receiver in one system, the sub-grids'
/// to `sub_grid`, returning what each apply returned in order, the world's
/// first. The receiver commits them on the next update.
fn apply(app: &mut App, sub_grid: Entity, commit: &Commit) -> Vec<Result<(), EditGap>> {
    let (world, sub_grids) = (commit.world.clone(), commit.sub_grid_deltas());
    on(app, move |voxel_world: &mut VoxelWorld<Receiver>| {
        let mut applied: Vec<_> = world
            .iter()
            .map(|delta| voxel_world.apply_delta(delta.clone()))
            .collect();
        let mut grid = voxel_world.grid_mut(Grid::SubGrid(sub_grid)).unwrap();
        applied.extend(
            sub_grids
                .iter()
                .map(|delta| grid.apply_delta(delta.clone())),
        );
        applied
    })
}

/// Spawns a sub-grid on the receiver, at the identity pose.
fn receiver_sub_grid(app: &mut App) -> Entity {
    on(app, |voxel_world: &mut VoxelWorld<Receiver>| {
        voxel_world.spawn_sub_grid(Transform::default())
    })
}

fn get_voxel<C: VoxelWorldConfig<MaterialIndex = u8>>(
    app: &mut App,
    grid: Grid,
    position: IVec3,
) -> WorldVoxel {
    on(app, move |voxel_world: &mut VoxelWorld<C>| {
        voxel_world.grid(grid).unwrap().get_voxel(position)
    })
}

/// `grid`'s edited chunks with their sequences, sorted by chunk.
fn edited_chunks<C: VoxelWorldConfig<MaterialIndex = u8>>(
    app: &mut App,
    grid: Grid,
) -> Vec<(IVec3, u64)> {
    let mut chunks = on(app, move |voxel_world: &mut VoxelWorld<C>| {
        voxel_world.grid(grid).unwrap().edited_chunks()
    });
    chunks.sort_by_key(|(chunk, _)| chunk.to_array());
    chunks
}

/// The full delta of each of `grid`'s edited chunks, sorted by chunk: what a
/// join or a save sends.
fn full_deltas<C: VoxelWorldConfig<MaterialIndex = u8>>(
    app: &mut App,
    grid: Grid,
) -> Vec<EditDelta<u8>> {
    let chunks = edited_chunks::<C>(app, grid);
    on(app, move |voxel_world: &mut VoxelWorld<C>| {
        let grid = voxel_world.grid(grid).unwrap();
        chunks
            .iter()
            .map(|(chunk, _)| grid.full_delta(*chunk))
            .collect()
    })
}

/// Spawns a sub-grid on the sender holding `voxel` at each of `positions`,
/// and returns it with what the commit reported.
fn send_sub_grid(
    app: &mut App,
    positions: Vec<IVec3>,
    voxel: WorldVoxel,
) -> (Entity, Commit) {
    send(app, move |voxel_world| {
        let entity = voxel_world.spawn_sub_grid(Transform::default());
        let mut grid = voxel_world.grid_mut(Grid::SubGrid(entity)).unwrap();
        for position in &positions {
            grid.set_voxel(*position, voxel);
        }
        entity
    })
}

/// Sets `voxel` at each of `positions` in the sender's sub-grid `entity`,
/// and returns what the commit reported.
fn send_set(
    app: &mut App,
    entity: Entity,
    positions: Vec<IVec3>,
    voxel: WorldVoxel,
) -> Commit {
    send(app, move |voxel_world| {
        let mut grid = voxel_world.grid_mut(Grid::SubGrid(entity)).unwrap();
        for position in &positions {
            grid.set_voxel(*position, voxel);
        }
    })
    .1
}

/// `count` voxels in a row along x from the origin, in one chunk.
fn row(count: i32) -> Vec<IVec3> {
    (0..count).map(|x| IVec3::new(x, 3, 3)).collect()
}

#[test]
fn a_commit_writes_a_message_for_each_sub_grid_chunk_whose_edits_changed() {
    let mut app = sync_app();
    let (_, world_only) = send(&mut app, |voxel_world| {
        voxel_world.set_voxel(IVec3::new(1, 1, 1), WorldVoxel::Solid(4));
    });
    assert_eq!(world_only.world.len(), 1);
    assert!(world_only.sub_grids.is_empty(), "a world with no sub-grid");

    let (sent_by, commit) = send(&mut app, |voxel_world| {
        let entity = voxel_world.spawn_sub_grid(Transform::default());
        let mut grid = voxel_world.grid_mut(Grid::SubGrid(entity)).unwrap();
        grid.set_voxel(IVec3::new(1, 1, 1), WorldVoxel::Solid(2));
        grid.set_voxel(IVec3::new(2, 1, 1), WorldVoxel::Solid(3));
        grid.set_voxel(IVec3::new(40, 1, 1), WorldVoxel::Solid(2));
        voxel_world.set_voxel(IVec3::new(-5, 1, 1), WorldVoxel::Solid(5));
        entity
    });
    assert_eq!(
        commit
            .world
            .iter()
            .map(|delta| delta.chunk)
            .collect::<Vec<_>>(),
        vec![IVec3::new(-1, 0, 0)],
        "the world's message carries only the world's chunk"
    );
    assert_eq!(
        commit
            .sub_grids
            .iter()
            .map(|(entity, delta)| (*entity, delta.chunk, delta.seq, delta.full))
            .collect::<Vec<_>>(),
        vec![
            (sent_by, IVec3::ZERO, 2, false),
            (sent_by, IVec3::new(1, 0, 0), 1, false)
        ]
    );

    let receiver = receiver_sub_grid(&mut app);
    assert_eq!(apply(&mut app, receiver, &commit), vec![Ok(()); 3]);
    app.update();
    let received = take::<Receiver>(&mut app);
    assert_eq!(received.world, commit.world);
    assert_eq!(
        received.sub_grids,
        commit
            .sub_grid_deltas()
            .into_iter()
            .map(|delta| (receiver, delta))
            .collect::<Vec<_>>(),
        "a receiver's commit names its own sub-grid"
    );
    for position in [IVec3::new(2, 1, 1), IVec3::new(40, 1, 1)] {
        assert_eq!(
            get_voxel::<Receiver>(&mut app, Grid::SubGrid(receiver), position),
            get_voxel::<Sender>(&mut app, Grid::SubGrid(sent_by), position),
        );
    }
}

#[test]
fn a_voxel_removed_from_a_sub_grid_travels_as_air() {
    let mut app = sync_app();
    let (sent_by, filled) = send_sub_grid(&mut app, row(3), WorldVoxel::Solid(2));
    let receiver = receiver_sub_grid(&mut app);
    assert_eq!(apply(&mut app, receiver, &filled), vec![Ok(())]);
    app.update();

    // Unset removes a voxel as air does, and travels as air too.
    let removed = send_set(&mut app, sent_by, vec![row(3)[1]], WorldVoxel::Unset);
    let delta = &removed.sub_grids[0].1;
    assert_eq!((delta.seq, delta.full), (4, false));
    assert_eq!(
        delta.voxels,
        Voxels::Sparse(vec![(
            crate::edits::voxel_index(row(3)[1]) as u16,
            WorldVoxel::Air
        )])
    );

    assert_eq!(apply(&mut app, receiver, &removed), vec![Ok(())]);
    app.update();
    assert_eq!(
        get_voxel::<Receiver>(&mut app, Grid::SubGrid(receiver), row(3)[1]),
        WorldVoxel::Air
    );
    assert_eq!(
        full_deltas::<Receiver>(&mut app, Grid::SubGrid(receiver)),
        full_deltas::<Sender>(&mut app, Grid::SubGrid(sent_by))
    );
}

#[test]
fn an_applied_unset_reads_as_air_and_never_reaches_a_sub_grids_delta() {
    let mut app = sync_app();
    let receiver = receiver_sub_grid(&mut app);
    let unset = IVec3::new(5, 3, 3);
    let delta = EditDelta::new(
        IVec3::ZERO,
        3,
        false,
        vec![
            (
                crate::edits::voxel_index(row(2)[0]) as u16,
                WorldVoxel::Solid(2),
            ),
            (
                crate::edits::voxel_index(row(2)[1]) as u16,
                WorldVoxel::Solid(2),
            ),
            (crate::edits::voxel_index(unset) as u16, WorldVoxel::Unset),
        ],
    );
    let commit = Commit {
        world: Vec::new(),
        sub_grids: vec![(Entity::PLACEHOLDER, delta)],
    };
    assert_eq!(apply(&mut app, receiver, &commit), vec![Ok(())]);
    assert_eq!(
        get_voxel::<Receiver>(&mut app, Grid::SubGrid(receiver), unset),
        WorldVoxel::Air
    );
    app.update();
    let received = take::<Receiver>(&mut app);
    for (_, delta) in &received.sub_grids {
        assert!(
            delta.iter().all(|(_, voxel)| voxel != WorldVoxel::Unset),
            "{delta:?}"
        );
    }
    for delta in full_deltas::<Receiver>(&mut app, Grid::SubGrid(receiver)) {
        assert!(delta.iter().all(|(_, voxel)| voxel.is_solid()), "{delta:?}");
        assert_eq!(delta.len(), 2);
    }
}

#[test]
fn a_chunk_left_with_fewer_solids_than_its_commit_changed_sends_its_full_delta() {
    let mut app = sync_app();
    let (sent_by, _) = send_sub_grid(&mut app, row(10), WorldVoxel::Solid(2));

    let few = send_set(&mut app, sent_by, row(10)[..2].to_vec(), WorldVoxel::Air);
    let delta = &few.sub_grids[0].1;
    assert_eq!(
        (delta.seq, delta.full, delta.len()),
        (12, false, 2),
        "8 left"
    );

    let most = send_set(&mut app, sent_by, row(10)[2..8].to_vec(), WorldVoxel::Air);
    let delta = &most.sub_grids[0].1;
    assert_eq!((delta.seq, delta.full), (18, true), "2 left of 6 changed");
    assert_eq!(
        delta.iter().collect::<Vec<_>>(),
        row(10)[8..]
            .iter()
            .map(|position| (*position, WorldVoxel::Solid(2)))
            .collect::<Vec<_>>()
    );

    let all = send_set(&mut app, sent_by, row(10)[8..].to_vec(), WorldVoxel::Air);
    let delta = &all.sub_grids[0].1;
    assert_eq!(
        (delta.seq, delta.full, delta.len()),
        (20, true, 0),
        "emptied"
    );
}

#[test]
fn an_emptied_chunk_stays_listed_and_heals_a_receiver_that_missed_its_delta() {
    let mut app = sync_app();
    let (sent_by, filled) = send_sub_grid(&mut app, row(2), WorldVoxel::Solid(2));
    let receiver = receiver_sub_grid(&mut app);
    assert_eq!(apply(&mut app, receiver, &filled), vec![Ok(())]);
    app.update();

    let _missed = send_set(&mut app, sent_by, row(2), WorldVoxel::Air);
    assert_eq!(
        edited_chunks::<Sender>(&mut app, Grid::SubGrid(sent_by)),
        vec![(IVec3::ZERO, 4)],
        "an emptied chunk keeps its sequence"
    );

    let next = send_set(&mut app, sent_by, vec![row(3)[2]], WorldVoxel::Solid(3));
    assert_eq!(
        apply(&mut app, receiver, &next),
        vec![Err(EditGap {
            chunk: IVec3::ZERO,
            local: 2,
            received: 5
        })]
    );
    // A heartbeat shows the receiver behind, so it asks for the full delta.
    let healed = Commit {
        world: Vec::new(),
        sub_grids: full_deltas::<Sender>(&mut app, Grid::SubGrid(sent_by))
            .into_iter()
            .map(|delta| (sent_by, delta))
            .collect(),
    };
    assert_eq!(apply(&mut app, receiver, &healed), vec![Ok(())]);
    app.update();
    for position in row(3) {
        assert_eq!(
            get_voxel::<Receiver>(&mut app, Grid::SubGrid(receiver), position),
            get_voxel::<Sender>(&mut app, Grid::SubGrid(sent_by), position),
            "at {position}"
        );
    }
    assert_eq!(
        edited_chunks::<Receiver>(&mut app, Grid::SubGrid(receiver)),
        edited_chunks::<Sender>(&mut app, Grid::SubGrid(sent_by)),
    );
}

#[test]
fn a_join_of_every_edited_chunks_full_delta_reproduces_the_sub_grid_and_continues() {
    let mut app = sync_app();
    let (sent_by, _) = send_sub_grid(
        &mut app,
        vec![
            IVec3::new(1, 1, 1),
            IVec3::new(40, 1, 1),
            IVec3::new(41, 1, 1),
            IVec3::new(1, -20, 1),
        ],
        WorldVoxel::Solid(2),
    );
    // One chunk emptied, which a join carries as an empty full delta.
    send_set(
        &mut app,
        sent_by,
        vec![IVec3::new(1, -20, 1)],
        WorldVoxel::Air,
    );

    let join = Commit {
        world: Vec::new(),
        sub_grids: full_deltas::<Sender>(&mut app, Grid::SubGrid(sent_by))
            .into_iter()
            .map(|delta| (sent_by, delta))
            .collect(),
    };
    assert_eq!(join.sub_grids.len(), 3);
    let receiver = receiver_sub_grid(&mut app);
    assert_eq!(apply(&mut app, receiver, &join), vec![Ok(()); 3]);
    app.update();
    assert_eq!(
        full_deltas::<Receiver>(&mut app, Grid::SubGrid(receiver)),
        full_deltas::<Sender>(&mut app, Grid::SubGrid(sent_by))
    );

    let later = send_set(
        &mut app,
        sent_by,
        vec![IVec3::new(2, -20, 1), IVec3::new(40, 1, 1)],
        WorldVoxel::Solid(3),
    );
    assert!(later.sub_grids.iter().all(|(_, delta)| !delta.full));
    assert_eq!(apply(&mut app, receiver, &later), vec![Ok(()); 2]);
    app.update();
    assert_eq!(
        full_deltas::<Receiver>(&mut app, Grid::SubGrid(receiver)),
        full_deltas::<Sender>(&mut app, Grid::SubGrid(sent_by))
    );
}

/// The top layer of the ground around the origin, which reaches into eight
/// chunks of either grid through their padding.
fn slab() -> Vec<IVec3> {
    let mut voxels = Vec::new();
    for x in -1..=1 {
        for z in -1..=1 {
            voxels.push(IVec3::new(x, -1, z));
        }
    }
    voxels
}

/// The level-0 chunk positions whose padded voxels hold one of `voxels`: a
/// chunk's padding is one voxel deep.
fn padded_chunks(voxels: &[IVec3]) -> BTreeSet<[i32; 3]> {
    let mut chunks = BTreeSet::new();
    for voxel in voxels {
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    let reach = *voxel + IVec3::new(dx, dy, dz);
                    chunks
                        .insert(reach.div_euclid(IVec3::splat(CHUNK_SIZE_I)).to_array());
                }
            }
        }
    }
    chunks
}

/// The frame the next update runs, as the message recorders count it.
fn next_frame(app: &App) -> u32 {
    app.world().resource::<FrameCount>().0
}

/// The frames, from `since` on, in which each of the receiver's chunks of
/// `grid` at one of `chunks` published, by chunk.
fn published_since(
    app: &App,
    since: u32,
    grid: Grid,
    chunks: &BTreeSet<[i32; 3]>,
) -> Vec<([i32; 3], u32)> {
    recorded::<ChunkPublished<Receiver>>(app)
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

/// Whether each of the receiver's chunks of `grid` at one of `chunks` has
/// published since `since`.
fn all_published_since(
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

/// A sync app whose receiver has published its world chunks around the slab,
/// and then turns slow.
fn settled_app() -> App {
    let mut app = sync_app();
    let chunks = padded_chunks(&slab());
    update_until(&mut app, "the receiver's chunks around the slab", |app| {
        all_published_since(app, 0, Grid::World, &chunks)
    });
    app.world_mut().resource_mut::<Receiver>().slow = true;
    app
}

/// Lifts the slab on the sender, as a lift would: a new sub-grid holding it
/// as `Solid(2)`, and air in the world. Returns the sub-grid and the commit.
fn send_lift(app: &mut App) -> (Entity, Commit) {
    send(app, |voxel_world| {
        let entity = voxel_world.spawn_sub_grid(Transform::default());
        let mut grid = voxel_world.grid_mut(Grid::SubGrid(entity)).unwrap();
        for position in slab() {
            grid.set_voxel(position, WorldVoxel::Solid(2));
        }
        for position in slab() {
            voxel_world.set_voxel(position, WorldVoxel::Air);
        }
        entity
    })
}

/// Stamps the sender's sub-grid `entity` back into the world, as a stamp
/// would: its voxels removed, and the slab set to `Solid(3)` in the world.
fn send_stamp(app: &mut App, entity: Entity) -> Commit {
    send(app, move |voxel_world| {
        voxel_world
            .grid_mut(Grid::SubGrid(entity))
            .unwrap()
            .clear_voxels_in(IVec3::new(-1, -1, -1), IVec3::new(2, 0, 2));
        for position in slab() {
            voxel_world.set_voxel(position, WorldVoxel::Solid(3));
        }
    })
    .1
}

/// Spawns a sub-grid on the receiver and applies `lift`'s parts to it and the
/// world, all in one system. Returns the sub-grid.
fn receive_lift(app: &mut App, lift: &Commit) -> Entity {
    let (world, sub_grid) = (lift.world.clone(), lift.sub_grid_deltas());
    on(app, move |voxel_world: &mut VoxelWorld<Receiver>| {
        let entity = voxel_world.spawn_sub_grid(Transform::default());
        let mut grid = voxel_world.grid_mut(Grid::SubGrid(entity)).unwrap();
        for delta in &sub_grid {
            grid.apply_delta(delta.clone()).unwrap();
        }
        for delta in &world {
            voxel_world.apply_delta(delta.clone()).unwrap();
        }
        entity
    })
}

#[test]
fn a_receiver_applying_a_lifts_parts_in_one_system_lands_them_as_one_handover() {
    let mut app = settled_app();
    let (_, lift) = send_lift(&mut app);
    assert_eq!((lift.world.len(), lift.sub_grids.len()), (4, 4));

    let commit = next_frame(&app);
    let receiver = receive_lift(&mut app, &lift);
    let chunks = padded_chunks(&slab());
    update_until(
        &mut app,
        "the receiver's sub-grid chunks to publish",
        |app| all_published_since(app, commit, Grid::SubGrid(receiver), &chunks),
    );

    let mut frames = published_since(&app, commit, Grid::SubGrid(receiver), &chunks);
    frames.extend(published_since(&app, commit, Grid::World, &chunks));
    assert_eq!(frames.len(), 2 * chunks.len(), "each chunk published once");
    let landing: BTreeSet<u32> = frames.iter().map(|(_, frame)| *frame).collect();
    assert_eq!(landing.len(), 1, "{frames:?}");
}

#[test]
fn a_receiver_applying_a_stamps_parts_in_one_system_lands_them_as_one_handover() {
    let mut app = settled_app();
    let (sent_by, lift) = send_lift(&mut app);
    let receiver = receive_lift(&mut app, &lift);
    let chunks = padded_chunks(&slab());
    update_until(
        &mut app,
        "the receiver's sub-grid chunks to publish",
        |app| all_published_since(app, 0, Grid::SubGrid(receiver), &chunks),
    );

    let stamp = send_stamp(&mut app, sent_by);
    assert_eq!(stamp.sub_grids.len(), 4);
    for (_, delta) in &stamp.sub_grids {
        assert!(delta.full && delta.is_empty(), "{delta:?}");
    }
    let commit = next_frame(&app);
    assert_eq!(apply(&mut app, receiver, &stamp), vec![Ok(()); 8]);
    update_until(&mut app, "the receiver's world chunks to publish", |app| {
        all_published_since(app, commit, Grid::World, &chunks)
    });

    let world: BTreeSet<u32> = published_since(&app, commit, Grid::World, &chunks)
        .into_iter()
        .map(|(_, frame)| frame)
        .collect();
    assert_eq!(world.len(), 1, "{world:?}");
    let despawned: Vec<u32> = recorded::<ChunkWillDespawn<Receiver>>(&app)
        .messages
        .iter()
        .filter(|(_, message)| message.grid == Grid::SubGrid(receiver))
        .map(|(frame, _)| *frame)
        .collect();
    assert_eq!(despawned.len(), chunks.len());
    assert_eq!(
        despawned.into_iter().collect::<BTreeSet<_>>(),
        world,
        "the sub-grid's chunks despawn in the frame the world's publish"
    );
}

/// How many times the full deltas of the sender's world and its sub-grid
/// `entity` hold a solid voxel at `position`.
fn solids_held(app: &mut App, entity: Entity, position: IVec3) -> usize {
    [Grid::World, Grid::SubGrid(entity)]
        .into_iter()
        .flat_map(|grid| full_deltas::<Sender>(app, grid))
        .flat_map(|delta| delta.iter().collect::<Vec<_>>())
        .filter(|(at, voxel)| *at == position && voxel.is_solid())
        .count()
}

#[test]
fn a_commit_emptying_a_sub_grid_into_the_world_leaves_each_voxel_in_one_grids_full_deltas(
) {
    let mut app = sync_app();
    let (sent_by, _) = send_lift(&mut app);
    for position in slab() {
        assert_eq!(solids_held(&mut app, sent_by, position), 1, "lifted");
    }
    send_stamp(&mut app, sent_by);
    for position in slab() {
        assert_eq!(solids_held(&mut app, sent_by, position), 1, "stamped");
        assert_eq!(
            get_voxel::<Sender>(&mut app, Grid::World, position),
            WorldVoxel::Solid(3)
        );
    }
    assert!(full_deltas::<Sender>(&mut app, Grid::SubGrid(sent_by))
        .iter()
        .all(EditDelta::is_empty));
}

/// A sub-grid as a save holds it: its pose and the full delta of each of its
/// edited chunks.
#[derive(Clone, Debug, PartialEq)]
struct SavedSubGrid {
    pose: Transform,
    chunks: Vec<EditDelta<u8>>,
}

fn save_sub_grid(app: &mut App, entity: Entity) -> SavedSubGrid {
    let pose = on(app, move |voxel_world: &mut VoxelWorld<Sender>| {
        voxel_world
            .grid(Grid::SubGrid(entity))
            .unwrap()
            .pose()
            .compute_transform()
    });
    SavedSubGrid {
        pose,
        chunks: full_deltas::<Sender>(app, Grid::SubGrid(entity)),
    }
}

/// Loads a save into the receiver in one system, in the documented order:
/// forget the world's edits, despawn every sub-grid, apply the world's full
/// deltas, then spawn each saved sub-grid at its pose and apply its full
/// deltas. Returns the loaded sub-grids in the save's order.
fn load(
    app: &mut App,
    world: Vec<EditDelta<u8>>,
    saved: Vec<SavedSubGrid>,
) -> Vec<Entity> {
    app.world_mut()
        .run_system_once(
            move |mut commands: Commands,
                  sub_grids: Query<Entity, With<SubGrid<Receiver>>>,
                  mut voxel_world: VoxelWorld<Receiver>| {
                voxel_world.forget_edits();
                for entity in &sub_grids {
                    commands.entity(entity).despawn();
                }
                for delta in &world {
                    voxel_world.apply_delta(delta.clone()).unwrap();
                }
                saved
                    .iter()
                    .map(|sub_grid| {
                        let entity = voxel_world.spawn_sub_grid(sub_grid.pose);
                        let mut grid =
                            voxel_world.grid_mut(Grid::SubGrid(entity)).unwrap();
                        for delta in &sub_grid.chunks {
                            grid.apply_delta(delta.clone()).unwrap();
                        }
                        entity
                    })
                    .collect::<Vec<_>>()
            },
        )
        .unwrap()
}

#[test]
fn the_documented_load_order_reproduces_the_saved_sub_grids_and_leaves_no_stale_one() {
    let mut app = sync_app();
    // The receiver has edits and a sub-grid of its own before it loads.
    let stale = on(&mut app, |voxel_world: &mut VoxelWorld<Receiver>| {
        voxel_world.set_voxel(IVec3::new(7, 7, 7), WorldVoxel::Solid(9));
        let entity = voxel_world.spawn_sub_grid(Transform::default());
        voxel_world
            .grid_mut(Grid::SubGrid(entity))
            .unwrap()
            .set_voxel(IVec3::new(1, 1, 1), WorldVoxel::Solid(9));
        entity
    });
    app.update();

    send(&mut app, |voxel_world| {
        voxel_world.set_voxel(IVec3::new(3, 4, 5), WorldVoxel::Solid(4));
    });
    let (holding, _) = send(&mut app, |voxel_world| {
        let entity = voxel_world.spawn_sub_grid(Transform::from_xyz(5.0, 2.0, 0.0));
        let mut grid = voxel_world.grid_mut(Grid::SubGrid(entity)).unwrap();
        for position in [
            IVec3::new(1, 1, 1),
            IVec3::new(2, 1, 1),
            IVec3::new(40, 1, 1),
        ] {
            grid.set_voxel(position, WorldVoxel::Solid(2));
        }
        entity
    });
    // One chunk emptied, which saves as an empty full delta.
    send_set(
        &mut app,
        holding,
        vec![IVec3::new(40, 1, 1)],
        WorldVoxel::Air,
    );
    let (empty, _) = send(&mut app, |voxel_world| {
        voxel_world.spawn_sub_grid(Transform::from_xyz(-3.0, 0.0, 8.0))
    });
    app.update();

    let world = full_deltas::<Sender>(&mut app, Grid::World);
    let saved = vec![
        save_sub_grid(&mut app, holding),
        save_sub_grid(&mut app, empty),
    ];
    assert_eq!(saved[0].chunks.len(), 2);
    assert!(
        saved[1].chunks.is_empty(),
        "an empty sub-grid saves as its pose"
    );

    let loaded = load(&mut app, world.clone(), saved.clone());
    app.update();
    app.update();

    let mut live: Vec<Entity> = app
        .world_mut()
        .query_filtered::<Entity, With<SubGrid<Receiver>>>()
        .iter(app.world())
        .collect();
    live.sort();
    let mut expected = loaded.clone();
    expected.sort();
    assert_eq!(live, expected, "only the loaded sub-grids live");
    assert!(on(
        &mut app,
        move |voxel_world: &mut VoxelWorld<Receiver>| {
            voxel_world.grid(Grid::SubGrid(stale)).is_none()
        }
    ));
    assert_eq!(full_deltas::<Receiver>(&mut app, Grid::World), world);
    for (entity, saved) in loaded.into_iter().zip(&saved) {
        let reloaded = SavedSubGrid {
            pose: on(&mut app, move |voxel_world: &mut VoxelWorld<Receiver>| {
                voxel_world
                    .grid(Grid::SubGrid(entity))
                    .unwrap()
                    .pose()
                    .compute_transform()
            }),
            chunks: full_deltas::<Receiver>(&mut app, Grid::SubGrid(entity)),
        };
        assert_eq!(&reloaded, saved, "with the saved edit sequences");
    }
}
