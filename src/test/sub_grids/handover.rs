//! Tests of handovers: voxels moving between grids publish in one frame in
//! every grid, never showing twice or not at all.

use std::collections::BTreeSet;

use bevy::diagnostic::FrameCount;
use bevy::ecs::system::RunSystemOnce;
use bevy::prelude::*;

use crate::chunk::{Chunk, CHUNK_SIZE_I};
use crate::grid_bench::{handover_stats, HandoverStats};
use crate::mesh_cache::MeshRef;
use crate::prelude::*;
use crate::test::{
    floor_voxel, meshing_app, record_chunk_messages, recorded, spawn_camera,
    test_plugins, update_until,
};

/// A world with solid ground below y = 0, as `FloorWorld`. Once `slow`, it
/// runs two chunk tasks at a time, so a commit touching many chunks takes
/// several frames to publish them.
#[derive(Resource, Clone, Default)]
struct SlowFloor {
    slow: bool,
}

impl VoxelWorldConfig for SlowFloor {
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

/// A voxel of the ground in the chunk below the origin, away from the slab.
const APART: IVec3 = IVec3::new(9, -1, 9);

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

fn level_0(position: [i32; 3]) -> ChunkKey {
    ChunkKey::new(IVec3::from_array(position), 0)
}

/// Runs `f` on the voxel world in a system of its own, whose commands apply
/// when it returns.
fn run<T: 'static>(
    app: &mut App,
    mut f: impl FnMut(&mut VoxelWorld<SlowFloor>) -> T + Send + Sync + 'static,
) -> T {
    app.world_mut()
        .run_system_once(move |mut voxel_world: VoxelWorld<SlowFloor>| {
            f(&mut voxel_world)
        })
        .unwrap()
}

/// The frame the next update runs, as the message recorders count it.
fn next_frame(app: &App) -> u32 {
    app.world().resource::<FrameCount>().0
}

/// The voxel the world's published chunk data shows at `position`.
fn world_shows(app: &mut App, position: IVec3) -> Option<WorldVoxel> {
    let key = ChunkKey::new(position.div_euclid(IVec3::splat(CHUNK_SIZE_I)), 0);
    run(app, move |voxel_world| {
        voxel_world
            .get_chunk_data(key)
            .and_then(|data| data.get_voxel_at_world_position(position))
    })
}

/// The voxel the sub-grid `entity`'s published chunk data shows at
/// `position`, if it has a chunk published there.
fn sub_grid_shows(app: &mut App, entity: Entity, position: IVec3) -> Option<WorldVoxel> {
    let key = ChunkKey::new(position.div_euclid(IVec3::splat(CHUNK_SIZE_I)), 0);
    run(app, move |voxel_world| {
        voxel_world
            .grid(Grid::SubGrid(entity))
            .and_then(|grid| grid.get_chunk_data(key))
            .and_then(|data| data.get_voxel_at_world_position(position))
    })
}

/// The frames, from `since` on, in which each chunk of `grid` at one of
/// `chunks` published, by chunk.
fn published_since(
    app: &App,
    since: u32,
    grid: Grid,
    chunks: &BTreeSet<[i32; 3]>,
) -> Vec<([i32; 3], u32)> {
    recorded::<ChunkPublished<SlowFloor>>(app)
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

/// Whether every chunk of `grid` at one of `chunks` has published since
/// `since`.
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

fn stats(app: &App) -> HandoverStats {
    handover_stats::<SlowFloor>(app.world()).unwrap()
}

/// Gives `app`, holding the world's plugin, a camera, records the chunk
/// messages, and runs it until the world's chunks the slab reaches are
/// published. Then the world turns slow.
fn settle(mut app: App) -> App {
    record_chunk_messages::<ChunkPublished<SlowFloor>>(&mut app);
    record_chunk_messages::<ChunkWillDespawn<SlowFloor>>(&mut app);
    app.world_mut()
        .run_system_once(spawn_camera::<SlowFloor>)
        .unwrap();
    let world_chunks = padded_chunks(&slab());
    update_until(&mut app, "the world's chunks around the slab", |app| {
        all_published_since(app, 0, Grid::World, &world_chunks)
    });
    app.world_mut().resource_mut::<SlowFloor>().slow = true;
    app
}

/// A world without meshes, settled around the slab.
fn floor_app() -> App {
    let mut app = App::new();
    app.add_plugins((
        test_plugins(),
        TransformPlugin,
        VoxelWorldPlugin::<SlowFloor>::minimal(),
    ));
    settle(app)
}

/// Spawns a sub-grid holding `voxel` at each of `positions`, in the system
/// that sets the world's voxels there to air, as a lift does. Returns the
/// sub-grid and the frame its commit runs.
fn fill_from_world(
    app: &mut App,
    positions: Vec<IVec3>,
    voxel: WorldVoxel,
) -> (Entity, u32) {
    let frame = next_frame(app);
    let entity = run(app, move |voxel_world| {
        let entity = voxel_world.spawn_sub_grid(Transform::default());
        let mut grid = voxel_world.grid_mut(Grid::SubGrid(entity)).unwrap();
        for position in &positions {
            grid.set_voxel(*position, voxel);
        }
        for position in &positions {
            voxel_world.set_voxel(*position, WorldVoxel::Air);
        }
        entity
    });
    (entity, frame)
}

/// The mesh the chunk of `parent` at `key` shows, if it has one.
fn mesh_of(app: &mut App, parent: Entity, key: ChunkKey) -> Option<Handle<Mesh>> {
    let mut chunks = app
        .world_mut()
        .query::<(&Chunk<SlowFloor>, Option<&MeshRef>, &ChildOf)>();
    chunks
        .iter(app.world())
        .find(|(chunk, _, child_of)| chunk.key() == key && child_of.parent() == parent)
        .and_then(|(_, mesh, _)| mesh.map(|mesh| (*mesh.0).clone()))
}

fn root(app: &mut App) -> Entity {
    app.world_mut()
        .query_filtered::<Entity, With<crate::voxel_world_internal::WorldRoot<SlowFloor>>>()
        .single(app.world())
        .unwrap()
}

#[test]
fn a_commit_filling_a_sub_grid_holds_every_chunk_it_touches_and_publishes_them_in_one_frame(
) {
    let mut app = meshing_app();
    app.add_plugins(VoxelWorldPlugin::with_config(SlowFloor::default()));
    let mut app = settle(app);
    let root = root(&mut app);
    let chunks = padded_chunks(&slab());
    let below = level_0([0, -1, 0]);
    update_until(
        &mut app,
        "the world's chunk below the origin to get a mesh",
        |app| mesh_of(app, root, below).is_some(),
    );
    let world_mesh = mesh_of(&mut app, root, below);
    let before = stats(&app);

    let (entity, commit) = fill_from_world(&mut app, slab(), WorldVoxel::Solid(2));
    update_until(&mut app, "the sub-grid's chunks to publish", |app| {
        let landed = all_published_since(app, commit, Grid::SubGrid(entity), &chunks);
        if !landed {
            for position in slab() {
                assert_eq!(world_shows(app, position), Some(WorldVoxel::Solid(1)));
                assert_eq!(sub_grid_shows(app, entity, position), None);
            }
            assert_eq!(
                mesh_of(app, root, below),
                world_mesh,
                "the world's old mesh"
            );
            for chunk in &chunks {
                assert_eq!(mesh_of(app, entity, level_0(*chunk)), None, "{chunk:?}");
            }
        }
        landed
    });

    let mut frames: Vec<([i32; 3], u32)> =
        published_since(&app, commit, Grid::SubGrid(entity), &chunks);
    frames.extend(published_since(&app, commit, Grid::World, &chunks));
    assert_eq!(frames.len(), 2 * chunks.len(), "each chunk published once");
    let landing: BTreeSet<u32> = frames.iter().map(|(_, frame)| *frame).collect();
    assert_eq!(landing.len(), 1, "{frames:?}");
    for position in slab() {
        assert_eq!(world_shows(&mut app, position), Some(WorldVoxel::Air));
        assert_eq!(
            sub_grid_shows(&mut app, entity, position),
            Some(WorldVoxel::Solid(2))
        );
    }
    assert_ne!(mesh_of(&mut app, root, below), world_mesh);
    assert!(mesh_of(&mut app, entity, below).is_some());
    let after = stats(&app);
    assert_eq!(
        (
            after.handovers - before.handovers,
            after.chunks - before.chunks
        ),
        (1, 2 * chunks.len() as u64)
    );
}

/// A world settled around the slab with a sub-grid holding the slab's
/// voxels as `Solid(2)`, published, where the world has air.
fn floor_app_with_slab_lifted() -> (App, Entity) {
    let mut app = floor_app();
    let (entity, commit) = fill_from_world(&mut app, slab(), WorldVoxel::Solid(2));
    let chunks = padded_chunks(&slab());
    update_until(&mut app, "the sub-grid's chunks to publish", |app| {
        all_published_since(app, commit, Grid::SubGrid(entity), &chunks)
    });
    (app, entity)
}

/// The chunk entities of the sub-grid `entity`.
fn chunks_of(app: &mut App, entity: Entity) -> Vec<Entity> {
    let mut chunks = app
        .world_mut()
        .query::<(Entity, &Chunk<SlowFloor>, &ChildOf)>();
    chunks
        .iter(app.world())
        .filter(|(_, _, child_of)| child_of.parent() == entity)
        .map(|(chunk, ..)| chunk)
        .collect()
}

/// Empties the sub-grid `entity` and sets the slab's voxels to `Solid(3)` in
/// the world, as a stamp does, in one system, and despawns the sub-grid
/// through `despawn_sub_grid` there too if `despawn`. Returns the frame the
/// commit runs.
fn stamp_slab_back(app: &mut App, entity: Entity, despawn: bool) -> u32 {
    let frame = next_frame(app);
    run(app, move |voxel_world| {
        voxel_world
            .grid_mut(Grid::SubGrid(entity))
            .unwrap()
            .clear_voxels_in(IVec3::new(-1, -1, -1), IVec3::new(2, 0, 2));
        for position in slab() {
            voxel_world.set_voxel(position, WorldVoxel::Solid(3));
        }
        if despawn {
            voxel_world.despawn_sub_grid(entity);
        }
    });
    frame
}

#[test]
fn emptying_a_sub_grid_into_the_world_despawns_its_chunks_in_the_frame_the_worlds_publish(
) {
    let (mut app, entity) = floor_app_with_slab_lifted();
    let sub_grid_chunks = chunks_of(&mut app, entity);
    assert_eq!(sub_grid_chunks.len(), 8);
    let world_chunks = padded_chunks(&slab());

    let commit = stamp_slab_back(&mut app, entity, false);
    update_until(&mut app, "the world's chunks to publish", |app| {
        let landed = all_published_since(app, commit, Grid::World, &world_chunks);
        let alive = sub_grid_chunks
            .iter()
            .filter(|chunk| app.world().get_entity(**chunk).is_ok())
            .count();
        let world = slab()
            .into_iter()
            .filter(|position| world_shows(app, *position) == Some(WorldVoxel::Solid(3)))
            .count();
        let sub_grid = slab()
            .into_iter()
            .filter(|position| {
                sub_grid_shows(app, entity, *position) == Some(WorldVoxel::Solid(2))
            })
            .count();
        if landed {
            assert_eq!((alive, world, sub_grid), (0, 9, 0));
        } else {
            assert_eq!((alive, world, sub_grid), (8, 0, 9));
        }
        landed
    });
    let landing = published_since(&app, commit, Grid::World, &world_chunks)[0].1;
    let despawned: BTreeSet<u32> = recorded::<ChunkWillDespawn<SlowFloor>>(&app)
        .messages
        .iter()
        .filter(|(_, message)| message.grid == Grid::SubGrid(entity))
        .map(|(frame, _)| *frame)
        .collect();
    assert_eq!(despawned, BTreeSet::from([landing]));
    assert!(
        app.world().get_entity(entity).is_ok(),
        "the sub-grid lives on"
    );
}

#[test]
fn two_fills_in_consecutive_frames_sharing_a_world_chunk_land_in_the_same_frame() {
    let mut app = floor_app();
    let (first, commit) = fill_from_world(&mut app, slab(), WorldVoxel::Solid(2));
    app.update();
    let (second, _) = fill_from_world(&mut app, vec![APART], WorldVoxel::Solid(2));
    let (first_chunks, second_chunks) = (padded_chunks(&slab()), padded_chunks(&[APART]));
    assert!(!first_chunks.is_disjoint(&second_chunks));
    update_until(&mut app, "both sub-grids' chunks to publish", |app| {
        all_published_since(app, commit, Grid::SubGrid(first), &first_chunks)
            && all_published_since(app, commit, Grid::SubGrid(second), &second_chunks)
    });

    let mut world_chunks = first_chunks.clone();
    world_chunks.extend(second_chunks.iter().copied());
    let mut frames = published_since(&app, commit, Grid::World, &world_chunks);
    frames.extend(published_since(
        &app,
        commit,
        Grid::SubGrid(first),
        &first_chunks,
    ));
    frames.extend(published_since(
        &app,
        commit,
        Grid::SubGrid(second),
        &second_chunks,
    ));
    let landing: BTreeSet<u32> = frames.iter().map(|(_, frame)| *frame).collect();
    assert_eq!(landing.len(), 1, "{frames:?}");
}

#[test]
fn a_voxel_set_every_frame_in_a_held_chunk_does_not_hold_the_handover() {
    let mut app = floor_app();
    let (entity, commit) = fill_from_world(&mut app, slab(), WorldVoxel::Solid(2));
    let chunks = padded_chunks(&slab());
    let mut material = 4;
    update_until(&mut app, "the handover to land", |app| {
        material = 9 - material;
        let voxel = WorldVoxel::Solid(material);
        run(app, move |voxel_world| voxel_world.set_voxel(APART, voxel));
        all_published_since(app, commit, Grid::SubGrid(entity), &chunks)
    });
    for position in slab() {
        assert_eq!(world_shows(&mut app, position), Some(WorldVoxel::Air));
    }
}

#[test]
fn despawning_a_sub_grid_mid_handover_lets_the_worlds_chunks_land() {
    let mut app = floor_app();
    let (entity, commit) = fill_from_world(&mut app, slab(), WorldVoxel::Solid(2));
    let chunks = padded_chunks(&slab());
    app.update();
    assert!(!all_published_since(&app, commit, Grid::World, &chunks));
    app.world_mut().entity_mut(entity).despawn();

    update_until(&mut app, "the world's chunks to publish", |app| {
        all_published_since(app, commit, Grid::World, &chunks)
    });
    for position in slab() {
        assert_eq!(world_shows(&mut app, position), Some(WorldVoxel::Air));
    }
    assert!(published_since(&app, commit, Grid::SubGrid(entity), &chunks).is_empty());
}

/// Runs `app` until the world's chunks the slab reaches have published
/// since `commit`, checking that the sub-grid `entity` lives until the
/// frame they publish in and is gone after it.
fn until_landed_with_sub_grid_despawned(app: &mut App, entity: Entity, commit: u32) {
    let chunks = padded_chunks(&slab());
    update_until(app, "the world's chunks to publish", |app| {
        let landed = all_published_since(app, commit, Grid::World, &chunks);
        assert_eq!(app.world().get_entity(entity).is_err(), landed);
        landed
    });
}

#[test]
fn despawn_sub_grid_called_with_its_emptying_edits_despawns_it_as_its_handover_lands() {
    let (mut app, entity) = floor_app_with_slab_lifted();
    let commit = stamp_slab_back(&mut app, entity, true);
    until_landed_with_sub_grid_despawned(&mut app, entity, commit);
}

#[test]
fn despawn_sub_grid_waits_for_a_pending_handover_holding_its_chunks() {
    let mut app = floor_app();
    let (entity, commit) = fill_from_world(&mut app, slab(), WorldVoxel::Solid(2));
    app.update();
    run(&mut app, move |voxel_world| {
        voxel_world.despawn_sub_grid(entity)
    });
    assert!(app.world().get_entity(entity).is_ok());
    until_landed_with_sub_grid_despawned(&mut app, entity, commit);
}

#[test]
fn despawn_sub_grid_despawns_at_once_when_no_handover_holds_its_chunks() {
    let (mut app, entity) = floor_app_with_slab_lifted();
    run(&mut app, move |voxel_world| {
        voxel_world.despawn_sub_grid(entity)
    });
    assert!(app.world().get_entity(entity).is_err());
}

#[test]
fn a_commit_that_fills_or_empties_no_sub_grid_lands_no_handover() {
    let (mut app, entity) = floor_app_with_slab_lifted();
    let before = stats(&app);
    let commit = next_frame(&app);
    run(&mut app, move |voxel_world| {
        voxel_world
            .grid_mut(Grid::SubGrid(entity))
            .unwrap()
            .set_voxel(APART + IVec3::Y * 4, WorldVoxel::Solid(2));
        voxel_world.set_voxel(APART, WorldVoxel::Air);
    });
    let (below, origin) = (BTreeSet::from([[0, -1, 0]]), BTreeSet::from([[0, 0, 0]]));
    update_until(&mut app, "the edited chunks to publish", |app| {
        all_published_since(app, commit, Grid::World, &below)
            && all_published_since(app, commit, Grid::SubGrid(entity), &origin)
    });
    assert_eq!(stats(&app), before);
}
