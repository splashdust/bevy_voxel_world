//! Tests of sub-grids' chunks: where a commit spawns and retires them, the
//! chunk messages they fire, reading them through their grid, and what of
//! the world they leave alone.

use std::collections::BTreeSet;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use bevy::ecs::system::RunSystemOnce;
use bevy::prelude::*;

use crate::chunk::{Chunk, ChunkData, VoxelArray};
use crate::chunk_lifecycle::ChunkStateOf;
use crate::mesh_cache::MeshRef;
use crate::prelude::*;
use crate::test::{
    chunk_status, meshing_app, move_cameras, record_chunk_messages, recorded,
    spawn_camera, test_plugins, update_until, FloorWorld,
};
use crate::voxel_world::{
    ChunkEvent, ChunkEventType, Published, WillDespawn, WillRemesh, WillSpawn, WillUpdate,
};

/// A world with no camera, so it streams no chunks of its own, whose
/// transforms propagate.
fn app_of<C: VoxelWorldConfig>() -> App {
    let mut app = App::new();
    app.add_plugins((
        test_plugins(),
        TransformPlugin,
        VoxelWorldPlugin::<C>::minimal(),
    ));
    app.update();
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

/// Spawns a sub-grid at `transform` holding `voxel` at each of `positions`.
fn spawn_filled<C: VoxelWorldConfig<MaterialIndex = u8>>(
    app: &mut App,
    transform: Transform,
    positions: &[IVec3],
    voxel: WorldVoxel,
) -> Entity {
    let positions = positions.to_vec();
    run::<C, _>(app, move |voxel_world| {
        let entity = voxel_world.spawn_sub_grid(transform);
        let mut grid = voxel_world.grid_mut(Grid::SubGrid(entity)).unwrap();
        for position in &positions {
            grid.set_voxel(*position, voxel);
        }
        entity
    })
}

/// Sets `voxel` at each of `positions` in the sub-grid `entity`.
fn set<C: VoxelWorldConfig<MaterialIndex = u8>>(
    app: &mut App,
    entity: Entity,
    positions: &[IVec3],
    voxel: WorldVoxel,
) {
    let positions = positions.to_vec();
    run::<C, _>(app, move |voxel_world| {
        let mut grid = voxel_world.grid_mut(Grid::SubGrid(entity)).unwrap();
        for position in &positions {
            grid.set_voxel(*position, voxel);
        }
    });
}

fn level_0(position: [i32; 3]) -> ChunkKey {
    ChunkKey::new(IVec3::from_array(position), 0)
}

/// The chunk state of the sub-grid `entity`'s chunk at `key`.
fn state<C: VoxelWorldConfig>(
    app: &mut App,
    entity: Entity,
    key: ChunkKey,
) -> Option<ChunkStatus> {
    run::<C, _>(app, move |voxel_world| {
        voxel_world
            .grid(Grid::SubGrid(entity))
            .and_then(|grid| grid.chunk_state(key))
    })
}

/// The published data of the sub-grid `entity`'s chunk at `key`.
fn data<C: VoxelWorldConfig>(
    app: &mut App,
    entity: Entity,
    key: ChunkKey,
) -> Option<ChunkData<C::MaterialIndex>> {
    run::<C, _>(app, move |voxel_world| {
        voxel_world
            .grid(Grid::SubGrid(entity))
            .and_then(|grid| grid.get_chunk_data(key))
    })
}

/// The chunk entities that are children of `entity` and aren't Retiring,
/// by chunk position. Checks that each is at LOD level 0 and placed by its
/// chunk key, as world chunks are under the root.
fn chunks_of<C: VoxelWorldConfig>(app: &mut App, entity: Entity) -> BTreeSet<[i32; 3]> {
    let mut chunks = app
        .world_mut()
        .query::<(&Chunk<C>, &ChunkStateOf<C>, &ChildOf, &Transform)>();
    let mut positions = BTreeSet::new();
    for (chunk, state, parent, transform) in chunks.iter(app.world()) {
        if parent.parent() != entity || state.status() == ChunkStatus::Retiring {
            continue;
        }
        assert_eq!(chunk.lod_level(), 0, "{}", chunk.key());
        assert_eq!(
            transform.translation,
            chunk.key().world_position() - 1.0,
            "{}",
            chunk.key()
        );
        positions.insert(chunk.position().to_array());
    }
    positions
}

/// Runs `app` until every chunk of the sub-grid `entity` at `keys` is Ready.
fn until_ready<C: VoxelWorldConfig>(app: &mut App, entity: Entity, keys: &[ChunkKey]) {
    update_until(app, "the sub-grid's chunks to be ready", |app| {
        keys.iter()
            .all(|key| state::<C>(app, entity, *key) == Some(ChunkStatus::Ready))
    });
}

#[test]
fn after_a_commit_a_sub_grid_has_a_chunk_wherever_its_padded_voxels_hold_a_solid() {
    let mut app = app_of::<FloorWorld>();
    let transform = Transform::from_xyz(3.0, -2.0, 7.0);
    let solid = WorldVoxel::Solid(1);
    let inside =
        spawn_filled::<FloorWorld>(&mut app, transform, &[IVec3::new(5, 6, 7)], solid);
    let face =
        spawn_filled::<FloorWorld>(&mut app, transform, &[IVec3::new(0, 6, 7)], solid);
    let corner =
        spawn_filled::<FloorWorld>(&mut app, transform, &[IVec3::splat(31)], solid);
    app.update();

    assert_eq!(
        chunks_of::<FloorWorld>(&mut app, inside),
        BTreeSet::from([[0, 0, 0]])
    );
    assert_eq!(
        chunks_of::<FloorWorld>(&mut app, face),
        BTreeSet::from([[-1, 0, 0], [0, 0, 0]])
    );
    let eight: BTreeSet<[i32; 3]> =
        (0..8).map(|i| [i & 1, (i >> 1) & 1, i >> 2]).collect();
    assert_eq!(chunks_of::<FloorWorld>(&mut app, corner), eight);
}

#[test]
fn edits_past_a_sub_grids_chunks_add_chunks_and_a_chunk_gone_all_air_retires() {
    let mut app = app_of::<FloorWorld>();
    let (a, b) = (IVec3::new(5, 6, 7), IVec3::new(40, 6, 7));
    let solid = WorldVoxel::Solid(1);
    let entity = spawn_filled::<FloorWorld>(&mut app, Transform::default(), &[a], solid);
    app.update();
    assert_eq!(
        chunks_of::<FloorWorld>(&mut app, entity),
        BTreeSet::from([[0, 0, 0]])
    );

    set::<FloorWorld>(&mut app, entity, &[b], solid);
    app.update();
    assert_eq!(
        chunks_of::<FloorWorld>(&mut app, entity),
        BTreeSet::from([[0, 0, 0], [1, 0, 0]])
    );

    set::<FloorWorld>(&mut app, entity, &[b], WorldVoxel::Air);
    app.update();
    assert_eq!(
        state::<FloorWorld>(&mut app, entity, level_0([1, 0, 0])),
        Some(ChunkStatus::Retiring)
    );
    assert_eq!(
        chunks_of::<FloorWorld>(&mut app, entity),
        BTreeSet::from([[0, 0, 0]])
    );

    // Needed again while it is still Retiring: it spawns again once it has
    // despawned.
    set::<FloorWorld>(&mut app, entity, &[b], solid);
    update_until(&mut app, "the chunk to spawn again", |app| {
        chunks_of::<FloorWorld>(app, entity) == BTreeSet::from([[0, 0, 0], [1, 0, 0]])
    });
    until_ready::<FloorWorld>(&mut app, entity, &[level_0([1, 0, 0])]);
    let published = data::<FloorWorld>(&mut app, entity, level_0([1, 0, 0])).unwrap();
    assert_eq!(published.get_voxel_at_world_position(b), Some(solid));

    set::<FloorWorld>(&mut app, entity, &[a, b], WorldVoxel::Unset);
    update_until(&mut app, "every chunk to despawn", |app| {
        [[0, 0, 0], [1, 0, 0]]
            .into_iter()
            .all(|key| state::<FloorWorld>(app, entity, level_0(key)).is_none())
    });
    assert!(chunks_of::<FloorWorld>(&mut app, entity).is_empty());
    assert!(
        app.world().get_entity(entity).is_ok(),
        "the sub-grid lives on"
    );
    let bounds = run::<FloorWorld, _>(&mut app, move |voxel_world| {
        voxel_world
            .grid(Grid::SubGrid(entity))
            .map(|grid| grid.bounds())
    });
    assert_eq!(bounds, Some(None));
}

/// The grid and chunk key of every recorded message `M` about `entity`.
fn about<M: ChunkEventType, C: VoxelWorldConfig>(
    app: &App,
    entity: Entity,
) -> Vec<(Grid, ChunkKey)> {
    recorded::<ChunkEvent<C, M>>(app)
        .messages
        .iter()
        .filter(|(_, message)| message.entity == entity)
        .map(|(_, message)| (message.grid, message.chunk_key))
        .collect()
}

/// Every recorded message `M` about a chunk other than `entity` names the
/// world.
fn others_name_the_world<M: ChunkEventType, C: VoxelWorldConfig>(
    app: &App,
    entity: Entity,
) -> usize {
    let others: Vec<_> = recorded::<ChunkEvent<C, M>>(app)
        .messages
        .iter()
        .filter(|(_, message)| message.entity != entity)
        .collect();
    assert!(others
        .iter()
        .all(|(_, message)| message.grid == Grid::World));
    others.len()
}

fn record_all<C: VoxelWorldConfig>(app: &mut App) {
    record_chunk_messages::<ChunkWillSpawn<C>>(app);
    record_chunk_messages::<ChunkPublished<C>>(app);
    record_chunk_messages::<ChunkWillDespawn<C>>(app);
    record_chunk_messages::<ChunkWillRemesh<C>>(app);
    record_chunk_messages::<ChunkWillUpdate<C>>(app);
}

/// [`app_of`] with a camera, recording every chunk message, run until the
/// world's chunks at and below the origin are ready.
fn floor_app() -> App {
    let mut app = app_of::<FloorWorld>();
    record_all::<FloorWorld>(&mut app);
    app.world_mut()
        .run_system_once(spawn_camera::<FloorWorld>)
        .unwrap();
    update_until(
        &mut app,
        "the world's chunks at the origin to be ready",
        |app| {
            [IVec3::ZERO, IVec3::NEG_Y].into_iter().all(|chunk| {
                chunk_status::<FloorWorld>(app, chunk) == Some(ChunkStatus::Ready)
            })
        },
    );
    app
}

/// The entity of the sub-grid `entity`'s chunk at `key`.
fn chunk_entity<C: VoxelWorldConfig>(
    app: &mut App,
    entity: Entity,
    key: ChunkKey,
) -> Entity {
    let mut chunks = app.world_mut().query::<(Entity, &Chunk<C>, &ChildOf)>();
    chunks
        .iter(app.world())
        .find(|(_, chunk, parent)| parent.parent() == entity && chunk.key() == key)
        .map(|(chunk, ..)| chunk)
        .unwrap()
}

#[test]
fn the_chunk_messages_name_a_sub_grids_chunks_by_their_grid_and_the_worlds_by_the_world()
{
    let mut app = floor_app();
    let origin = level_0([0, 0, 0]);
    let solid = WorldVoxel::Solid(1);
    let entity = spawn_filled::<FloorWorld>(
        &mut app,
        Transform::from_xyz(0.5, 40.0, 0.0),
        &[IVec3::new(5, 6, 7)],
        solid,
    );
    until_ready::<FloorWorld>(&mut app, entity, &[origin]);
    let chunk = chunk_entity::<FloorWorld>(&mut app, entity, origin);
    // A second edit updates the chunk, and removing both retires it.
    set::<FloorWorld>(&mut app, entity, &[IVec3::new(9, 9, 9)], solid);
    app.update();
    until_ready::<FloorWorld>(&mut app, entity, &[origin]);
    set::<FloorWorld>(
        &mut app,
        entity,
        &[IVec3::new(5, 6, 7), IVec3::new(9, 9, 9)],
        WorldVoxel::Air,
    );
    update_until(&mut app, "the chunk to despawn", |app| {
        state::<FloorWorld>(app, entity, origin).is_none()
    });

    let ours = (Grid::SubGrid(entity), origin);
    let spawned = about::<WillSpawn, FloorWorld>(&app, chunk);
    assert_eq!(spawned, vec![ours]);
    let published = about::<Published, FloorWorld>(&app, chunk);
    assert!(published.len() >= 2 && published.iter().all(|m| *m == ours));
    let remeshed = about::<WillRemesh, FloorWorld>(&app, chunk);
    assert!(remeshed.len() >= 2 && remeshed.iter().all(|m| *m == ours));
    let updated = about::<WillUpdate, FloorWorld>(&app, chunk);
    assert!(!updated.is_empty() && updated.iter().all(|m| *m == ours));
    assert_eq!(about::<WillDespawn, FloorWorld>(&app, chunk), vec![ours]);

    assert!(others_name_the_world::<WillSpawn, FloorWorld>(&app, chunk) > 0);
    assert!(others_name_the_world::<Published, FloorWorld>(&app, chunk) > 0);
    assert!(others_name_the_world::<WillRemesh, FloorWorld>(&app, chunk) > 0);
    others_name_the_world::<WillUpdate, FloorWorld>(&app, chunk);
    others_name_the_world::<WillDespawn, FloorWorld>(&app, chunk);
    // The world's chunk at the origin was never told about the sub-grid.
    let world_origin = recorded::<ChunkPublished<FloorWorld>>(&app)
        .keys()
        .filter(|(_, key)| *key == origin)
        .count();
    assert_eq!(world_origin, 1 + published.len());
}

#[test]
fn a_grids_chunk_reads_through_its_grid_where_the_world_has_a_chunk_at_its_key() {
    let mut app = floor_app();
    let (origin, below) = (level_0([0, 0, 0]), level_0([0, -1, 0]));
    let a = IVec3::new(5, 6, 7);
    let entity = spawn_filled::<FloorWorld>(
        &mut app,
        Transform::default(),
        &[a],
        WorldVoxel::Solid(3),
    );
    until_ready::<FloorWorld>(&mut app, entity, &[origin]);

    let read = run::<FloorWorld, _>(&mut app, move |voxel_world| {
        let grid = voxel_world.grid(Grid::SubGrid(entity)).unwrap();
        let world = voxel_world.grid(Grid::World).unwrap();
        let voxel_in = |data: Option<ChunkData<u8>>| {
            data.and_then(|data| data.get_voxel_at_world_position(a))
        };
        (
            voxel_in(grid.get_chunk_data(origin)),
            voxel_in(voxel_world.get_chunk_data(origin)),
            voxel_in(world.get_chunk_data(origin)),
            grid.chunk_state(below),
            voxel_world.chunk_state(below),
            world.chunk_state(below),
            grid.get_chunk_data(below).is_some(),
        )
    });
    assert_eq!(
        read,
        (
            Some(WorldVoxel::Solid(3)),
            Some(WorldVoxel::Air),
            Some(WorldVoxel::Air),
            None,
            Some(ChunkStatus::Ready),
            Some(ChunkStatus::Ready),
            false,
        )
    );
}

/// The voxels of the sub-grid chunks a collider system rebuilt on
/// `ChunkPublished`, read at [`COLLIDER_VOXEL`].
#[derive(Resource, Default)]
struct Colliders(Vec<(Grid, Option<WorldVoxel>)>);

const COLLIDER_VOXEL: IVec3 = IVec3::new(5, 6, 7);

/// Rebuilds a collider for each published chunk, of any grid, from its
/// voxels, as a physics integration would.
fn rebuild_colliders(
    mut published: MessageReader<ChunkPublished<FloorWorld>>,
    voxel_world: VoxelWorld<FloorWorld>,
    mut colliders: ResMut<Colliders>,
) {
    for message in published.read() {
        let Some(data) = voxel_world
            .grid(message.grid)
            .and_then(|grid| grid.get_chunk_data(message.chunk_key))
        else {
            continue;
        };
        if message.chunk_key == level_0([0, 0, 0]) {
            colliders.0.push((
                message.grid,
                data.get_voxel_at_world_position(COLLIDER_VOXEL),
            ));
        }
    }
}

#[test]
fn a_collider_rebuilt_on_chunk_published_reads_each_grids_own_voxels() {
    let mut app = app_of::<FloorWorld>();
    app.init_resource::<Colliders>()
        .add_systems(Update, rebuild_colliders);
    app.world_mut()
        .run_system_once(spawn_camera::<FloorWorld>)
        .unwrap();
    let entity = spawn_filled::<FloorWorld>(
        &mut app,
        Transform::default(),
        &[COLLIDER_VOXEL],
        WorldVoxel::Solid(2),
    );
    update_until(
        &mut app,
        "both grids' chunks at the origin to publish",
        |app| {
            let colliders = &app.world().resource::<Colliders>().0;
            colliders.iter().any(|(grid, _)| *grid == Grid::World)
                && colliders
                    .iter()
                    .any(|(grid, _)| *grid == Grid::SubGrid(entity))
        },
    );
    for (grid, voxel) in &app.world().resource::<Colliders>().0 {
        let expected = match grid {
            Grid::World => WorldVoxel::Air,
            Grid::SubGrid(_) => WorldVoxel::Solid(2),
        };
        assert_eq!(*voxel, Some(expected), "{grid:?}");
    }
}

static LOOKUPS: AtomicUsize = AtomicUsize::new(0);
static HINTS: AtomicUsize = AtomicUsize::new(0);

/// A world that counts every call to its voxel lookup delegate, the lookups
/// it returns, and its fill hint.
#[derive(Resource, Clone, Default)]
struct CountingWorld;

impl VoxelWorldConfig for CountingWorld {
    type MaterialIndex = u8;
    type ChunkUserBundle = ();

    fn voxel_lookup_delegate(&self) -> VoxelLookupDelegate<Self::MaterialIndex> {
        LOOKUPS.fetch_add(1, Ordering::Relaxed);
        Box::new(|_, _, _| {
            LOOKUPS.fetch_add(1, Ordering::Relaxed);
            Box::new(|_, _| {
                LOOKUPS.fetch_add(1, Ordering::Relaxed);
                WorldVoxel::Air
            })
        })
    }

    fn chunk_fill_hint(&self, _key: ChunkKey) -> FillHint<Self::MaterialIndex> {
        HINTS.fetch_add(1, Ordering::Relaxed);
        FillHint::Unknown
    }
}

#[test]
fn filling_a_sub_grid_never_calls_the_worlds_voxel_lookup_or_fill_hint() {
    let mut app = app_of::<CountingWorld>();
    let entity = run::<CountingWorld, _>(&mut app, |voxel_world| {
        let entity = voxel_world.spawn_sub_grid(Transform::from_xyz(1.0, 2.0, 3.0));
        voxel_world
            .grid_mut(Grid::SubGrid(entity))
            .unwrap()
            .set_voxels_in(IVec3::new(-3, 0, 0), IVec3::new(40, 4, 4), |_| {
                Some(WorldVoxel::Solid(1))
            });
        entity
    });
    let keys: Vec<ChunkKey> = [[-1, 0, 0], [0, 0, 0], [1, 0, 0], [-1, -1, 0], [0, 0, -1]]
        .map(level_0)
        .to_vec();
    until_ready::<CountingWorld>(&mut app, entity, &keys);
    // The fill lands as one handover, so a chunk can be Ready before the
    // last of the fill's chunks has finished and the data publishes.
    update_until(&mut app, "the fill to publish", |app| {
        data::<CountingWorld>(app, entity, level_0([1, 0, 0])).is_some()
    });
    let published = data::<CountingWorld>(&mut app, entity, level_0([1, 0, 0])).unwrap();
    assert_eq!(
        published.get_voxel_at_world_position(IVec3::new(39, 3, 3)),
        Some(WorldVoxel::Solid(1))
    );
    assert_eq!(LOOKUPS.load(Ordering::Relaxed), 0, "voxel lookups");
    assert_eq!(HINTS.load(Ordering::Relaxed), 0, "fill hints");
}

#[test]
fn moving_the_camera_and_regenerating_leave_a_sub_grids_chunks_alone() {
    let mut app = floor_app();
    let origin = level_0([0, 0, 0]);
    let entity = spawn_filled::<FloorWorld>(
        &mut app,
        Transform::default(),
        &[IVec3::new(5, 6, 7)],
        WorldVoxel::Solid(1),
    );
    until_ready::<FloorWorld>(&mut app, entity, &[origin]);
    let chunk = chunk_entity::<FloorWorld>(&mut app, entity, origin);

    move_cameras::<FloorWorld>(&mut app, Transform::from_xyz(5000.0, 10.0, 10.0));
    update_until(
        &mut app,
        "the world's chunk at the origin to despawn",
        |app| chunk_status::<FloorWorld>(app, IVec3::ZERO).is_none(),
    );
    assert_eq!(
        state::<FloorWorld>(&mut app, entity, origin),
        Some(ChunkStatus::Ready)
    );

    let published_before = about::<Published, FloorWorld>(&app, chunk).len();
    let updates_before = recorded::<ChunkWillUpdate<FloorWorld>>(&app).messages.len();
    run::<FloorWorld, _>(&mut app, |voxel_world| voxel_world.regenerate_chunks());
    for _ in 0..10 {
        app.update();
    }
    let updates = &recorded::<ChunkWillUpdate<FloorWorld>>(&app).messages;
    assert!(
        updates.len() > updates_before,
        "the world's chunks regenerate"
    );
    assert!(about::<WillUpdate, FloorWorld>(&app, chunk).is_empty());
    assert_eq!(
        about::<Published, FloorWorld>(&app, chunk).len(),
        published_before
    );
    assert_eq!(
        state::<FloorWorld>(&mut app, entity, origin),
        Some(ChunkStatus::Ready)
    );
}

/// The voxel at [`ONE_VOXEL`] is solid and every other is air.
const ONE_VOXEL: IVec3 = IVec3::new(5, 6, 7);

fn one_voxel(position: IVec3) -> WorldVoxel {
    if position == ONE_VOXEL {
        WorldVoxel::Solid(1)
    } else {
        WorldVoxel::Air
    }
}

/// The chunk position and padded voxels of each chunk the
/// [`RecordingMesherWorld`]'s meshing delegate meshed.
static MESHED: Mutex<Vec<(IVec3, VoxelArray<u8>)>> = Mutex::new(Vec::new());

/// A world of one solid voxel, whose meshing delegate records what it meshes.
#[derive(Resource, Clone, Default)]
struct RecordingMesherWorld;

impl VoxelWorldConfig for RecordingMesherWorld {
    type MaterialIndex = u8;
    type ChunkUserBundle = ();

    fn spawning_distance(&self) -> u32 {
        2
    }

    fn voxel_lookup_delegate(&self) -> VoxelLookupDelegate<Self::MaterialIndex> {
        Box::new(|_, _, _| Box::new(|position, _| one_voxel(position)))
    }

    fn chunk_meshing_delegate(
        &self,
    ) -> ChunkMeshingDelegate<Self::MaterialIndex, Self::ChunkUserBundle> {
        Some(Box::new(|position, _, _, _, _| {
            Box::new(move |voxels, data_shape, mesh_shape, mapper| {
                MESHED.lock().unwrap().push((position, Arc::clone(&voxels)));
                let mesh = crate::meshing::generate_chunk_mesh_for_shape(
                    voxels, position, data_shape, mesh_shape, mapper,
                );
                (mesh, None)
            })
        }))
    }
}

/// The padded voxels of each chunk at the origin meshed since this was last
/// called.
fn meshed_at_origin() -> Vec<VoxelArray<u8>> {
    MESHED
        .lock()
        .unwrap()
        .drain(..)
        .filter(|(position, _)| *position == IVec3::ZERO)
        .map(|(_, voxels)| voxels)
        .collect()
}

#[test]
fn a_meshing_delegate_meshes_a_sub_grid_chunk_as_a_world_chunk_with_its_voxels() {
    let mut app = app_of::<RecordingMesherWorld>();
    app.world_mut()
        .run_system_once(spawn_camera::<RecordingMesherWorld>)
        .unwrap();
    update_until(
        &mut app,
        "the world's chunk at the origin to be ready",
        |app| {
            chunk_status::<RecordingMesherWorld>(app, IVec3::ZERO)
                == Some(ChunkStatus::Ready)
        },
    );
    let world = meshed_at_origin();
    assert_eq!(world.len(), 1);

    let entity = spawn_filled::<RecordingMesherWorld>(
        &mut app,
        Transform::from_xyz(-20.0, 3.0, 0.0),
        &[ONE_VOXEL],
        WorldVoxel::Solid(1),
    );
    until_ready::<RecordingMesherWorld>(&mut app, entity, &[level_0([0, 0, 0])]);
    let sub_grid = meshed_at_origin();
    assert_eq!(sub_grid.len(), 1);
    assert_eq!(sub_grid[0], world[0]);
}

/// A world of one solid voxel, meshed by the crate's mesher.
#[derive(Resource, Clone, Default)]
struct OneVoxelWorld;

impl VoxelWorldConfig for OneVoxelWorld {
    type MaterialIndex = u8;
    type ChunkUserBundle = ();

    fn spawning_distance(&self) -> u32 {
        2
    }

    fn voxel_lookup_delegate(&self) -> VoxelLookupDelegate<Self::MaterialIndex> {
        Box::new(|_, _, _| Box::new(|position, _| one_voxel(position)))
    }
}

/// The mesh of the chunk at the origin whose parent is `parent`, or of the
/// world's chunk there for `None`, once it has one.
fn origin_mesh(app: &mut App, parent: Option<Entity>) -> Option<Handle<Mesh>> {
    let mut chunks = app
        .world_mut()
        .query::<(&Chunk<OneVoxelWorld>, &MeshRef, Option<&ChildOf>)>();
    let root = app
        .world_mut()
        .query_filtered::<Entity, With<crate::voxel_world_internal::WorldRoot<OneVoxelWorld>>>()
        .single(app.world())
        .ok();
    chunks
        .iter(app.world())
        .find(|(chunk, _, child_of)| {
            let of = child_of.map(ChildOf::parent);
            chunk.key() == level_0([0, 0, 0])
                && (of == parent || (parent.is_none() && of == root))
        })
        .map(|(_, mesh, _)| (*mesh.0).clone())
}

#[test]
fn identical_voxels_in_two_sub_grids_and_the_world_share_one_cached_mesh() {
    let mut app = meshing_app();
    app.add_plugins(VoxelWorldPlugin::with_config(OneVoxelWorld));
    app.update();
    app.world_mut()
        .run_system_once(spawn_camera::<OneVoxelWorld>)
        .unwrap();
    update_until(
        &mut app,
        "the world's chunk at the origin to get a mesh",
        |app| origin_mesh(app, None).is_some(),
    );
    let world = origin_mesh(&mut app, None).unwrap();

    let mut meshes = Vec::new();
    for x in [10.0, 20.0] {
        let entity = spawn_filled::<OneVoxelWorld>(
            &mut app,
            Transform::from_xyz(x, 0.0, 0.0),
            &[ONE_VOXEL],
            WorldVoxel::Solid(1),
        );
        update_until(&mut app, "the sub-grid's chunk to get a mesh", |app| {
            origin_mesh(app, Some(entity)).is_some()
        });
        meshes.push(origin_mesh(&mut app, Some(entity)).unwrap());
    }
    assert_eq!(meshes, vec![world.clone(), world]);
}

#[test]
fn despawning_a_sub_grid_despawns_its_chunks_and_a_sub_grid_given_its_id_finds_none() {
    let mut app = app_of::<FloorWorld>();
    let keys: Vec<ChunkKey> = [[0, 0, 0], [1, 0, 0]].map(level_0).to_vec();
    let solids = [IVec3::new(5, 6, 7), IVec3::new(40, 6, 7)];
    let entity = spawn_filled::<FloorWorld>(
        &mut app,
        Transform::default(),
        &solids,
        WorldVoxel::Solid(1),
    );
    until_ready::<FloorWorld>(&mut app, entity, &keys);
    let chunks: Vec<Entity> = keys
        .iter()
        .map(|key| chunk_entity::<FloorWorld>(&mut app, entity, *key))
        .collect();
    let id = app
        .world()
        .get::<SubGrid<FloorWorld>>(entity)
        .unwrap()
        .grid();

    app.world_mut().entity_mut(entity).despawn();
    for _ in 0..3 {
        app.update();
    }
    for chunk in chunks {
        assert!(app.world().get_entity(chunk).is_err());
    }

    let later =
        spawn_filled::<FloorWorld>(&mut app, Transform::default(), &[], WorldVoxel::Air);
    app.update();
    assert_eq!(
        app.world()
            .get::<SubGrid<FloorWorld>>(later)
            .unwrap()
            .grid(),
        id,
        "the later sub-grid is given the despawned one's id"
    );
    for key in &keys {
        assert_eq!(state::<FloorWorld>(&mut app, later, *key), None);
        assert!(data::<FloorWorld>(&mut app, later, *key).is_none());
    }
}

#[test]
fn removing_the_sub_grid_marker_despawns_its_chunks_and_keeps_its_entity() {
    let mut app = app_of::<FloorWorld>();
    let origin = level_0([0, 0, 0]);
    let entity = spawn_filled::<FloorWorld>(
        &mut app,
        Transform::default(),
        &[IVec3::new(5, 6, 7)],
        WorldVoxel::Solid(1),
    );
    until_ready::<FloorWorld>(&mut app, entity, &[origin]);
    let chunk = chunk_entity::<FloorWorld>(&mut app, entity, origin);

    app.world_mut()
        .entity_mut(entity)
        .remove::<SubGrid<FloorWorld>>();
    app.update();
    app.update();
    assert!(app.world().get_entity(chunk).is_err());
    assert!(app.world().get_entity(entity).is_ok());
    assert!(chunks_of::<FloorWorld>(&mut app, entity).is_empty());
}

/// Fills a sub-grid with chunks at two positions, ends it with `end`, and
/// checks that each of its chunks fired one `ChunkWillDespawn`, naming the
/// sub-grid, in the frame after `end`.
fn ending_fires_will_despawn_for_each_chunk(end: impl FnOnce(&mut App, Entity)) {
    let mut app = app_of::<FloorWorld>();
    record_chunk_messages::<ChunkWillDespawn<FloorWorld>>(&mut app);
    let keys: Vec<ChunkKey> = [[0, 0, 0], [1, 0, 0]].map(level_0).to_vec();
    let entity = spawn_filled::<FloorWorld>(
        &mut app,
        Transform::default(),
        &[IVec3::new(5, 6, 7), IVec3::new(40, 6, 7)],
        WorldVoxel::Solid(1),
    );
    until_ready::<FloorWorld>(&mut app, entity, &keys);
    let chunks: std::collections::HashSet<(ChunkKey, Entity)> = keys
        .iter()
        .map(|key| (*key, chunk_entity::<FloorWorld>(&mut app, entity, *key)))
        .collect();
    let frame = app.world().resource::<bevy::diagnostic::FrameCount>().0;

    end(&mut app, entity);
    for _ in 0..3 {
        app.update();
    }
    let despawned: Vec<(u32, ChunkKey, Entity)> =
        recorded::<ChunkWillDespawn<FloorWorld>>(&app)
            .messages
            .iter()
            .filter(|(_, message)| message.grid == Grid::SubGrid(entity))
            .map(|(frame, message)| (*frame, message.chunk_key, message.entity))
            .collect();
    assert_eq!(
        despawned
            .iter()
            .map(|(_, key, chunk)| (*key, *chunk))
            .collect::<std::collections::HashSet<_>>(),
        chunks
    );
    assert_eq!(despawned.len(), chunks.len(), "{despawned:?}");
    assert!(
        despawned.iter().all(|(at, ..)| *at == frame),
        "{despawned:?} after {frame}"
    );
}

#[test]
fn despawning_a_sub_grid_fires_will_despawn_for_each_of_its_chunks() {
    ending_fires_will_despawn_for_each_chunk(|app, entity| {
        app.world_mut().entity_mut(entity).despawn();
    });
}

#[test]
fn removing_the_sub_grid_marker_fires_will_despawn_for_each_of_its_chunks() {
    ending_fires_will_despawn_for_each_chunk(|app, entity| {
        app.world_mut()
            .entity_mut(entity)
            .remove::<SubGrid<FloorWorld>>();
    });
}

#[test]
fn despawn_sub_grid_fires_will_despawn_for_each_of_its_chunks() {
    ending_fires_will_despawn_for_each_chunk(|app, entity| {
        run::<FloorWorld, _>(app, move |voxel_world| {
            voxel_world.despawn_sub_grid(entity)
        });
    });
}
