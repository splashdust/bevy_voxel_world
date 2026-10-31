use bevy::ecs::system::RunSystemOnce;
use bevy::mesh::VertexAttributeValues;
use bevy::prelude::*;
use std::collections::HashSet;
use std::sync::{Arc, Mutex};

use crate::configuration::VoxelWorldConfig;
use crate::meshing::generate_chunk_mesh_for_shape;
use crate::prelude::*;
use crate::voxel_traversal::voxel_line_traversal;
use crate::{
    chunk::{Chunk, ChunkData, CHUNK_SIZE_F, CHUNK_SIZE_I},
    prelude::VoxelWorldCamera,
    voxel_world::*,
};
use ndshape::{RuntimeShape, Shape};
use rand::{rngs::StdRng, RngExt, SeedableRng};

fn _test_setup_app() -> App {
    let mut app = App::new();
    app.add_plugins((MinimalPlugins, VoxelWorldPlugin::<DefaultWorld>::minimal()));
    app.add_systems(Startup, |mut commands: Commands| {
        commands.spawn((
            Camera3d::default(),
            Transform::from_xyz(10.0, 10.0, 10.0).looking_at(Vec3::ZERO, Vec3::Y),
            VoxelWorldCamera::<DefaultWorld>::default(),
        ));
    });

    app
}

#[test]
fn chunks_spawn_with_multiple_voxel_world_cameras() {
    let mut app = App::new();
    app.add_plugins((MinimalPlugins, VoxelWorldPlugin::<DefaultWorld>::minimal()));
    app.add_systems(Startup, |mut commands: Commands| {
        let first_transform =
            Transform::from_xyz(10.0, 10.0, 10.0).looking_at(Vec3::ZERO, Vec3::Y);
        commands.spawn((
            Camera::default(),
            Camera3d::default(),
            first_transform,
            GlobalTransform::from(first_transform),
            VoxelWorldCamera::<DefaultWorld>::default(),
        ));

        let second_transform =
            Transform::from_xyz(1000.0, 10.0, 10.0).looking_at(Vec3::ZERO, Vec3::Y);
        commands.spawn((
            Camera::default(),
            Camera3d::default(),
            second_transform,
            GlobalTransform::from(second_transform),
            VoxelWorldCamera::<DefaultWorld>::default(),
        ));
    });

    app.update();

    for chunk_pos in [IVec3::ZERO, IVec3::new(1000, 10, 10) / CHUNK_SIZE_I] {
        assert!(
            chunk_status::<DefaultWorld>(&mut app, chunk_pos).is_some(),
            "no chunk spawned at {chunk_pos}"
        );
    }
}

#[test]
fn can_set_get_voxels() {
    let mut app = _test_setup_app();

    // Set and get some voxels
    app.add_systems(Update, |mut voxel_world: VoxelWorld<DefaultWorld>| {
        let positions = vec![
            IVec3::new(0, 100, 0),
            IVec3::new(0, 0, 0),
            IVec3::new(1, 0, 1),
            IVec3::new(1, 1, 1),
            IVec3::new(100, 200, 300),
            IVec3::new(-1, 0, -1),
            IVec3::new(0, -1, 0),
            IVec3::new(-100, -200, -300),
        ];

        let test_voxel = WorldVoxel::Solid(1);

        for pos in positions {
            voxel_world.set_voxel(pos, test_voxel);
            assert_eq!(voxel_world.get_voxel(pos), test_voxel)
        }
    });

    app.update();
}

#[test]
fn set_voxel_can_be_found_by_2d_coordinate() {
    let mut app = _test_setup_app();

    // Set up vector och positions to test
    let positions = vec![
        IVec3::new(0, 5, 0),
        IVec3::new(1, 7, 0),
        IVec3::new(2, 10, 0),
        IVec3::new(3, 5, 10),
        IVec3::new(4, 7, 10),
        IVec3::new(5, 10, 10),
        IVec3::new(-6, 5, -10),
        IVec3::new(-7, 7, -10),
        IVec3::new(-10, 10, -10),
    ];

    let make_pos = positions.clone();

    app.add_systems(Update, move |mut voxel_world: VoxelWorld<DefaultWorld>| {
        let test_voxel = WorldVoxel::Solid(1);

        for pos in make_pos.clone() {
            voxel_world.set_voxel(pos, test_voxel);
        }
    });

    // Raycasts only cover published chunks.
    update_until(&mut app, "the edited chunks to be ready", |app| {
        positions.iter().all(|pos| {
            let (chunk_pos, _) = get_chunk_voxel_position(*pos);
            chunk_status::<DefaultWorld>(app, chunk_pos) == Some(ChunkStatus::Ready)
        })
    });

    let check_pos = positions.clone();

    app.add_systems(
        Update,
        move |voxel_world: crate::prelude::VoxelWorld<DefaultWorld>| {
            let test_voxel = crate::voxel::WorldVoxel::Solid(1);

            for pos in check_pos.clone() {
                let origin = Vec3::new(pos.x as f32 + 0.5, 256.0, pos.z as f32 + 0.5);
                let ray = Ray3d::new(origin, Dir3::NEG_Y);

                let result = voxel_world
                    .raycast(ray, &|(_, voxel)| matches!(voxel, WorldVoxel::Solid(_)));

                assert!(result.is_some(), "expected to hit solid voxel at {:?}", pos);

                let result = result.unwrap();
                assert_eq!(result.voxel_pos(), pos);
                assert_eq!(result.voxel, test_voxel);
            }
        },
    );

    app.update();
}

/// A world with solid ground below y = 0 and air above it.
#[derive(Resource, Clone, Default)]
struct FloorWorld;

impl VoxelWorldConfig for FloorWorld {
    type MaterialIndex = u8;
    type ChunkUserBundle = ();

    fn spawning_distance(&self) -> u32 {
        2
    }

    // Chunks out of view stay, so a still camera never retires a chunk.
    fn chunk_despawn_strategy(&self) -> ChunkDespawnStrategy {
        ChunkDespawnStrategy::FarAway
    }

    fn voxel_lookup_delegate(&self) -> VoxelLookupDelegate<Self::MaterialIndex> {
        Box::new(|_, _, _| Box::new(|pos, _| floor_voxel(pos)))
    }
}

fn floor_voxel(position: IVec3) -> WorldVoxel {
    if position.y < 0 {
        WorldVoxel::Solid(1)
    } else {
        WorldVoxel::Air
    }
}

fn spawn_camera<C: VoxelWorldConfig>(mut commands: Commands) {
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(10.0, 10.0, 10.0).looking_at(Vec3::ZERO, Vec3::Y),
        VoxelWorldCamera::<C>::default(),
    ));
}

/// Runs `app` at least once and until `done` holds, failing the test if it takes
/// too long.
fn update_until(app: &mut App, what: &str, mut done: impl FnMut(&mut App) -> bool) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    app.update();
    while !done(app) {
        assert!(
            std::time::Instant::now() < deadline,
            "timed out waiting: {what}"
        );
        app.update();
    }
}

fn move_cameras<C: VoxelWorldConfig>(app: &mut App, to: Transform) {
    let mut cameras = app
        .world_mut()
        .query_filtered::<(&mut Transform, &mut GlobalTransform), With<VoxelWorldCamera<C>>>();
    for (mut transform, mut global) in cameras.iter_mut(app.world_mut()) {
        *transform = to;
        *global = GlobalTransform::from(to);
    }
}

fn chunk_status<C: VoxelWorldConfig>(
    app: &mut App,
    chunk_pos: IVec3,
) -> Option<ChunkStatus> {
    app.world_mut()
        .run_system_once(move |voxel_world: VoxelWorld<C>| {
            voxel_world.chunk_state(chunk_pos)
        })
        .unwrap()
}

fn chunk_data<C: VoxelWorldConfig>(
    app: &mut App,
    chunk_pos: IVec3,
) -> Option<ChunkData<C::MaterialIndex>> {
    app.world_mut()
        .run_system_once(move |voxel_world: VoxelWorld<C>| {
            voxel_world.get_chunk_data(chunk_pos)
        })
        .unwrap()
}

#[derive(Resource, Default)]
struct Published(Vec<IVec3>);

#[test]
fn minimal_world_publishes_generated_voxels() {
    let mut app = App::new();
    app.add_plugins((MinimalPlugins, VoxelWorldPlugin::<FloorWorld>::minimal()))
        .add_systems(Startup, spawn_camera::<FloorWorld>);

    let below_ground = IVec3::new(1, -3, 1);
    update_until(&mut app, "generated voxel to be readable", |app| {
        app.world_mut()
            .run_system_once(move |voxel_world: VoxelWorld<FloorWorld>| {
                voxel_world.get_voxel(below_ground)
            })
            .unwrap()
            == WorldVoxel::Solid(1)
    });
}

#[test]
fn published_chunks_are_readable_by_systems_after_the_voxel_world_set() {
    let mut app = App::new();
    app.add_plugins((MinimalPlugins, VoxelWorldPlugin::<FloorWorld>::minimal()))
        .init_resource::<Published>()
        .add_systems(Startup, spawn_camera::<FloorWorld>)
        .add_systems(
            PreUpdate,
            (|mut published: ResMut<Published>,
              mut ev: MessageReader<ChunkPublished<FloorWorld>>,
              voxel_world: VoxelWorld<FloorWorld>| {
                for msg in ev.read() {
                    let chunk_data = voxel_world.get_chunk_data(msg.chunk_key);
                    assert!(
                        chunk_data.is_some_and(|data| data.has_generated()),
                        "chunk {} was announced before its data was readable",
                        msg.chunk_key
                    );
                    published.0.push(msg.chunk_key);
                }
            })
            .after(VoxelWorldSet::<FloorWorld>::default()),
        );

    update_until(&mut app, "a chunk to be published", |app| {
        app.world().resource::<Published>().0.len() > 5
    });
}

#[test]
fn a_chunk_is_ready_once_it_is_published() {
    let mut app = App::new();
    app.add_plugins((MinimalPlugins, VoxelWorldPlugin::<FloorWorld>::minimal()))
        .add_systems(Startup, spawn_camera::<FloorWorld>);

    update_until(&mut app, "the chunk at the origin to be ready", |app| {
        chunk_status::<FloorWorld>(app, IVec3::ZERO) == Some(ChunkStatus::Ready)
    });
    assert!(chunk_data::<FloorWorld>(&mut app, IVec3::ZERO)
        .is_some_and(|data| data.has_generated()));
}

/// A world that never starts a chunk task.
#[derive(Resource, Clone, Default)]
struct PausedWorld;

impl VoxelWorldConfig for PausedWorld {
    type MaterialIndex = u8;
    type ChunkUserBundle = ();

    fn spawning_distance(&self) -> u32 {
        2
    }

    fn max_active_chunk_threads(&self) -> usize {
        0
    }
}

#[test]
fn a_chunk_has_no_data_until_it_is_published() {
    let mut app = App::new();
    app.add_plugins((MinimalPlugins, VoxelWorldPlugin::<PausedWorld>::minimal()))
        .add_systems(Startup, spawn_camera::<PausedWorld>);

    for _ in 0..3 {
        app.update();
    }

    assert_eq!(
        chunk_status::<PausedWorld>(&mut app, IVec3::ZERO),
        Some(ChunkStatus::Pending)
    );
    assert!(chunk_data::<PausedWorld>(&mut app, IVec3::ZERO).is_none());
}

#[derive(Resource, Default)]
struct Retired(Vec<IVec3>);

#[test]
fn a_retiring_chunk_can_be_queried_in_the_frame_it_retires() {
    let mut app = App::new();
    app.add_plugins((MinimalPlugins, VoxelWorldPlugin::<FloorWorld>::minimal()))
        .init_resource::<Retired>()
        .add_systems(Startup, spawn_camera::<FloorWorld>)
        .add_systems(
            Update,
            |mut retired: ResMut<Retired>,
             mut ev: MessageReader<ChunkWillDespawn<FloorWorld>>,
             chunks: Query<&Chunk<FloorWorld>>,
             voxel_world: VoxelWorld<FloorWorld>| {
                for msg in ev.read() {
                    let chunk = chunks.get(msg.entity);
                    assert!(
                        chunk.is_ok_and(|chunk| chunk.position() == msg.chunk_key),
                        "retiring chunk {} was already despawned",
                        msg.chunk_key
                    );
                    assert_eq!(
                        voxel_world.chunk_state(msg.chunk_key),
                        Some(ChunkStatus::Retiring)
                    );
                    retired.0.push(msg.chunk_key);
                }
            },
        );

    update_until(&mut app, "the chunk at the origin to be ready", |app| {
        chunk_status::<FloorWorld>(app, IVec3::ZERO) == Some(ChunkStatus::Ready)
    });

    move_cameras::<FloorWorld>(&mut app, Transform::from_xyz(10_000.0, 10.0, 10.0));

    update_until(&mut app, "the chunk at the origin to retire", |app| {
        app.world().resource::<Retired>().0.contains(&IVec3::ZERO)
    });
    app.update();
    assert_eq!(chunk_status::<FloorWorld>(&mut app, IVec3::ZERO), None);
}

/// A world whose task slots can be closed while it runs.
#[derive(Resource, Clone)]
struct ThrottledWorld {
    task_slots: usize,
}

impl Default for ThrottledWorld {
    fn default() -> Self {
        Self {
            task_slots: usize::MAX,
        }
    }
}

impl VoxelWorldConfig for ThrottledWorld {
    type MaterialIndex = u8;
    type ChunkUserBundle = ();

    fn spawning_distance(&self) -> u32 {
        2
    }

    fn max_active_chunk_threads(&self) -> usize {
        self.task_slots
    }

    fn voxel_lookup_delegate(&self) -> VoxelLookupDelegate<Self::MaterialIndex> {
        FloorWorld.voxel_lookup_delegate()
    }
}

#[test]
fn the_world_survives_despawning_every_published_chunk() {
    let mut app = App::new();
    app.add_plugins((
        MinimalPlugins,
        VoxelWorldPlugin::<ThrottledWorld>::minimal(),
    ))
    .add_systems(Startup, spawn_camera::<ThrottledWorld>);

    update_until(&mut app, "the chunk at the origin to be ready", |app| {
        chunk_status::<ThrottledWorld>(app, IVec3::ZERO) == Some(ChunkStatus::Ready)
    });

    // No new chunk publishes, so every published chunk is gone once these retire.
    app.world_mut().resource_mut::<ThrottledWorld>().task_slots = 0;
    move_cameras::<ThrottledWorld>(&mut app, Transform::from_xyz(10_000.0, 10.0, 10.0));
    for _ in 0..3 {
        app.update();
    }

    assert_eq!(chunk_status::<ThrottledWorld>(&mut app, IVec3::ZERO), None);
    assert!(chunk_data::<ThrottledWorld>(&mut app, IVec3::ZERO).is_none());
}

/// A world whose voxel texture never loads.
#[derive(Resource, Clone, Default)]
struct MissingTextureWorld;

impl VoxelWorldConfig for MissingTextureWorld {
    type MaterialIndex = u8;
    type ChunkUserBundle = ();

    fn spawning_distance(&self) -> u32 {
        2
    }

    fn voxel_lookup_delegate(&self) -> VoxelLookupDelegate<Self::MaterialIndex> {
        FloorWorld.voxel_lookup_delegate()
    }

    fn voxel_texture(&self) -> Option<(String, u32)> {
        Some(("does_not_exist.png".into(), 4))
    }
}

#[test]
fn chunks_publish_while_the_voxel_texture_is_loading() {
    let mut app = App::new();
    app.add_plugins((
        MinimalPlugins,
        bevy::asset::AssetPlugin::default(),
        bevy::image::ImagePlugin::default(),
        bevy::mesh::MeshPlugin,
        TransformPlugin,
        bevy::camera::CameraPlugin,
    ))
    // Normally registered by RenderPlugin; the voxel material loads a shader.
    .init_asset::<bevy::shader::Shader>()
    .add_plugins(VoxelWorldPlugin::with_config(MissingTextureWorld))
    .add_systems(Startup, spawn_camera::<MissingTextureWorld>);

    update_until(&mut app, "a chunk mesh to be published", |app| {
        app.world_mut()
            .query::<&crate::mesh_cache::MeshRef>()
            .iter(app.world())
            .next()
            .is_some()
    });

    let shown = app
        .world_mut()
        .query_filtered::<(), With<Mesh3d>>()
        .iter(app.world())
        .count();
    assert_eq!(shown, 0, "meshes were shown before the texture loaded");
}

#[test]
fn chunk_will_remesh_event_after_set_voxel() {
    let mut app = _test_setup_app();

    // An edit to a generating chunk remeshes only after its running task.
    update_until(&mut app, "the chunk at the origin to be ready", |app| {
        chunk_status::<DefaultWorld>(app, IVec3::ZERO) == Some(ChunkStatus::Ready)
    });

    app.add_systems(Update, |mut voxel_world: VoxelWorld<DefaultWorld>| {
        voxel_world.set_voxel(IVec3::new(0, 0, 0), WorldVoxel::Solid(1));
    });

    app.update();

    app.add_systems(
        Update,
        |mut ev_chunk_will_remesh: MessageReader<ChunkWillRemesh<DefaultWorld>>| {
            let count = ev_chunk_will_remesh.read().count();
            assert!(count > 0)
        },
    );

    app.update();
}

#[test]
fn chunk_will_despawn_event() {
    let mut app = _test_setup_app();

    // move camera to simulate chunks going out of view
    app.add_systems(
        Update,
        |mut query: Query<&mut GlobalTransform, With<VoxelWorldCamera<DefaultWorld>>>| {
            for mut transform in query.iter_mut() {
                // Not sure why, but when running tests, bevy won't update the GlobalTransform
                // when a Transform has changed, so as a workaround we set it here directly.
                let tf = Transform::from_xyz(1000.0, 1000.0, 1000.0);
                *transform = GlobalTransform::from(tf);
            }
        },
    );

    app.update();

    app.add_systems(
        Update,
        |mut ev_chunk_will_despawn: MessageReader<ChunkWillDespawn<DefaultWorld>>| {
            let count = ev_chunk_will_despawn.read().count();
            assert!(count > 0)
        },
    );

    app.update();
}

#[test]
fn chunk_will_update_event() {
    let mut app = _test_setup_app();

    app.add_systems(Update, |mut voxel_world: VoxelWorld<DefaultWorld>| {
        voxel_world.set_voxel(IVec3::new(0, 0, 0), WorldVoxel::Solid(1));
    });

    app.update();

    app.add_systems(
        Update,
        |mut ev_chunk_will_update: MessageReader<ChunkWillUpdate<DefaultWorld>>| {
            let count = ev_chunk_will_update.read().count();
            assert!(count > 0)
        },
    );

    app.update();
}

#[test]
fn affected_chunk_positions_include_padding_neighbors() {
    let affected: HashSet<_> = get_affected_chunk_positions(IVec3::new(0, 0, 0))
        .into_iter()
        .collect();
    let expected = HashSet::from([
        IVec3::new(0, 0, 0),
        IVec3::new(0, 0, -1),
        IVec3::new(0, -1, 0),
        IVec3::new(0, -1, -1),
        IVec3::new(-1, 0, 0),
        IVec3::new(-1, 0, -1),
        IVec3::new(-1, -1, 0),
        IVec3::new(-1, -1, -1),
    ]);

    assert_eq!(affected, expected);

    assert_eq!(
        get_affected_chunk_positions(IVec3::new(1, 1, 1)),
        vec![IVec3::new(0, 0, 0)]
    );

    let affected: HashSet<_> =
        get_affected_chunk_positions(IVec3::new(CHUNK_SIZE_I - 1, 1, 1))
            .into_iter()
            .collect();
    let expected = HashSet::from([IVec3::new(0, 0, 0), IVec3::new(1, 0, 0)]);

    assert_eq!(affected, expected);
}

#[test]
fn set_voxel_on_chunk_boundary_marks_padding_neighbors_for_remesh() {
    let expected = HashSet::from([
        IVec3::new(0, 0, 0),
        IVec3::new(0, 0, -1),
        IVec3::new(0, -1, 0),
        IVec3::new(0, -1, -1),
        IVec3::new(-1, 0, 0),
        IVec3::new(-1, 0, -1),
        IVec3::new(-1, -1, 0),
        IVec3::new(-1, -1, -1),
    ]);

    let mut app = _test_setup_app();
    update_until(
        &mut app,
        "the chunks around the origin to be ready",
        |app| {
            expected.iter().all(|chunk_pos| {
                chunk_status::<DefaultWorld>(app, *chunk_pos) == Some(ChunkStatus::Ready)
            })
        },
    );

    app.world_mut()
        .resource_mut::<Messages<ChunkWillRemesh<DefaultWorld>>>()
        .clear();

    let remeshed_chunks = Arc::new(Mutex::new(Vec::new()));
    let remeshed_chunks_writer = remeshed_chunks.clone();
    app.add_systems(
        Update,
        move |mut events: MessageReader<ChunkWillRemesh<DefaultWorld>>| {
            remeshed_chunks_writer
                .lock()
                .unwrap()
                .extend(events.read().map(|event| event.chunk_key));
        },
    );

    app.add_systems(
        Update,
        |mut voxel_world: VoxelWorld<DefaultWorld>, mut did_set: Local<bool>| {
            if *did_set {
                return;
            }

            voxel_world.set_voxel(IVec3::new(0, 0, 0), WorldVoxel::Solid(1));
            *did_set = true;
        },
    );

    app.update();
    app.update();
    app.update();

    let remeshed_chunks: HashSet<_> =
        remeshed_chunks.lock().unwrap().iter().copied().collect();

    assert!(
        expected.is_subset(&remeshed_chunks),
        "expected {expected:?} to be remeshed, got {remeshed_chunks:?}"
    );
}

/// Voxel lookups by the chunk tasks of the chunk at the origin, in `ReuseWorld`.
static REUSE_LOOKUPS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// A `FloorWorld` that regenerates chunks from their previous data, and counts
/// the voxel lookups of the chunk at the origin.
#[derive(Resource, Clone, Default)]
struct ReuseWorld;

impl VoxelWorldConfig for ReuseWorld {
    type MaterialIndex = u8;
    type ChunkUserBundle = ();

    fn spawning_distance(&self) -> u32 {
        1
    }

    fn chunk_regenerate_strategy(&self) -> ChunkRegenerateStrategy {
        ChunkRegenerateStrategy::Reuse
    }

    fn voxel_lookup_delegate(&self) -> VoxelLookupDelegate<Self::MaterialIndex> {
        Box::new(|chunk_pos, _, _| {
            Box::new(move |pos, _| {
                if chunk_pos == IVec3::ZERO {
                    REUSE_LOOKUPS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
                floor_voxel(pos)
            })
        })
    }
}

#[test]
fn a_remeshed_chunk_reuses_its_previous_data_when_configured() {
    let mut app = App::new();
    app.add_plugins((MinimalPlugins, VoxelWorldPlugin::<ReuseWorld>::minimal()))
        .init_resource::<Announced>()
        .add_systems(Startup, spawn_camera::<ReuseWorld>)
        .add_systems(Update, record_announcements::<ReuseWorld>);

    update_until(&mut app, "the chunk at the origin to be ready", |app| {
        chunk_status::<ReuseWorld>(app, IVec3::ZERO) == Some(ChunkStatus::Ready)
    });
    let lookups = REUSE_LOOKUPS.load(std::sync::atomic::Ordering::Relaxed);
    assert!(
        lookups > 0,
        "the first task should look up the chunk's voxels"
    );

    let edited = IVec3::new(16, 16, 16);
    app.world_mut()
        .run_system_once(move |mut voxel_world: VoxelWorld<ReuseWorld>| {
            voxel_world.set_voxel(edited, WorldVoxel::Solid(9));
        })
        .unwrap();
    update_until(&mut app, "the edited chunk to be published again", |app| {
        Announced::count(&app.world().resource::<Announced>().published, IVec3::ZERO) == 2
    });

    assert_eq!(
        REUSE_LOOKUPS.load(std::sync::atomic::Ordering::Relaxed),
        lookups,
        "the remesh should reuse the previous voxels instead of looking them up"
    );
    let data = chunk_data::<ReuseWorld>(&mut app, IVec3::ZERO).unwrap();
    assert_eq!(
        data.get_voxel_at_world_position(edited),
        Some(WorldVoxel::Solid(9))
    );
    assert_eq!(
        data.get_voxel_at_world_position(IVec3::new(16, -1, 16)),
        Some(WorldVoxel::Solid(1))
    );
}

#[test]
fn generate_chunk_mesh_has_constant_size_for_random_shapes() {
    type Mat = <DefaultWorld as VoxelWorldConfig>::MaterialIndex;
    let mut rng = StdRng::seed_from_u64(0xDEADBEEF);

    for _ in 0..8 {
        let data_shape = {
            let dim = |rng: &mut StdRng| rng.random_range(3..=66);
            UVec3::new(dim(&mut rng), dim(&mut rng), dim(&mut rng))
        };
        let mesh_shape = UVec3::new(
            rng.random_range(3..=data_shape.x),
            rng.random_range(3..=data_shape.y),
            rng.random_range(3..=data_shape.z),
        );
        let data_runtime_shape = RuntimeShape::<u32, 3>::new(data_shape.to_array());
        let mut voxels =
            vec![WorldVoxel::<Mat>::Unset; data_runtime_shape.size() as usize];

        for x in 1..data_shape.x - 1 {
            for y in 1..data_shape.y - 1 {
                for z in 1..data_shape.z - 1 {
                    let idx = data_runtime_shape.linearize([x, y, z]) as usize;
                    voxels[idx] = WorldVoxel::Solid(1);
                }
            }
        }

        let texture_index_mapper: TextureIndexMapperFn<Mat> = Arc::new(|_| [0, 0, 0]);
        let mesh = generate_chunk_mesh_for_shape(
            Arc::from(voxels),
            IVec3::ZERO,
            data_shape,
            mesh_shape,
            texture_index_mapper,
        );

        let positions = match mesh.attribute(Mesh::ATTRIBUTE_POSITION) {
            Some(VertexAttributeValues::Float32x3(values)) => values,
            other => panic!("unexpected position attribute layout: {:?}", other),
        };
        assert!(
            !positions.is_empty(),
            "expected generated mesh to contain vertices"
        );

        let mut min = Vec3::splat(f32::INFINITY);
        let mut max = Vec3::splat(f32::NEG_INFINITY);
        for p in positions {
            let p = Vec3::from_array(*p);
            min = min.min(p);
            max = max.max(p);
        }

        let extent = max - min;
        for (axis, len) in ["x", "y", "z"].into_iter().zip(extent.to_array()) {
            assert!(
                (len - CHUNK_SIZE_F).abs() < 0.001,
                "mesh extent along {} axis should be {} but was {} (data_shape={:?}, mesh_shape={:?})",
                axis,
                CHUNK_SIZE_F,
                len,
                data_shape,
                mesh_shape
            );
        }
    }
}

#[test]
fn raycast_finds_voxel() {
    let mut app = App::new();
    app.add_plugins((MinimalPlugins, VoxelWorldPlugin::<FloorWorld>::minimal()))
        .add_systems(Startup, spawn_camera::<FloorWorld>);

    update_until(&mut app, "the chunk at the origin to be ready", |app| {
        chunk_status::<FloorWorld>(app, IVec3::ZERO) == Some(ChunkStatus::Ready)
    });
    app.world_mut()
        .run_system_once(|mut voxel_world: VoxelWorld<FloorWorld>| {
            voxel_world.set_voxel(IVec3::new(0, 0, -1), WorldVoxel::Solid(2));
            voxel_world.set_voxel(IVec3::new(0, 0, 0), WorldVoxel::Solid(2));
        })
        .unwrap();

    let result = app
        .world_mut()
        .run_system_once(|voxel_world: VoxelWorld<FloorWorld>| {
            let ray = Ray3d {
                origin: Vec3::new(0.5, 0.5, 70.0),
                direction: -Dir3::Z,
            };
            voxel_world.raycast(ray, &|(_pos, _vox)| true)
        })
        .unwrap();

    assert_eq!(
        result,
        Some(VoxelRaycastResult {
            position: Vec3::ZERO,
            normal: Some(Vec3::new(0.0, 0.0, 1.0)),
            voxel: WorldVoxel::Solid(2),
        })
    );
}

struct VisitVoxelTestState<'a> {
    test_name: &'a str,
    expected_path: &'a [IVec3],
    expected_face: Option<VoxelFace>,
    path_step_index: usize,
    traversal_time_out: Timer,
}

impl<'a> VisitVoxelTestState<'a> {
    fn new(
        test_name: &'a str,
        expected_path: &'a [IVec3],
        expected_face: Option<VoxelFace>,
    ) -> Self {
        VisitVoxelTestState {
            test_name,
            expected_path,
            expected_face,
            path_step_index: 0,
            traversal_time_out: Timer::from_seconds(1., TimerMode::Once),
        }
    }
}

fn visit_voxel_check(
    test_state: &mut VisitVoxelTestState,
    voxel_coords: IVec3,
    time: f32,
    face: VoxelFace,
) -> bool {
    // println!(
    //     "Traversed {:?} at {} through {:?}",
    //     voxel_coords, time, face
    // );

    assert!(
        0. <= time,
        "{}: Time must always be >= 0",
        test_state.test_name
    );
    assert!(
        time <= 1.,
        "{}: Time must always be <= 1",
        test_state.test_name
    );
    assert!(
        !test_state.traversal_time_out.is_finished(),
        "{}: Infinite loop detected (bc. such a simple trace should be much faster than 1s)",
        test_state.test_name
    );
    assert!(
        test_state.path_step_index == 0
            || test_state.expected_face.unwrap_or(face) == face,
        "{}: Expected entering through {:?}",
        test_state.test_name,
        test_state.expected_face
    );
    assert!(
        test_state.path_step_index < test_state.expected_path.len(),
        "{}: Expected path is not same length",
        test_state.test_name
    );
    assert_eq!(
        voxel_coords, test_state.expected_path[test_state.path_step_index],
        "{}: Found unexpected step in path",
        test_state.test_name
    );
    test_state.path_step_index += 1;

    true
}

#[test]
fn voxel_line_traversal_along_cartesian_axes() {
    let start = Vec3::splat(VOXEL_SIZE / 2.);

    {
        let end = Vec3::new(
            2. * VOXEL_SIZE + VOXEL_SIZE / 2.,
            VOXEL_SIZE / 2.,
            VOXEL_SIZE / 2.,
        );
        let expected_path = [
            IVec3::new(0, 0, 0),
            IVec3::new(1, 0, 0),
            IVec3::new(2, 0, 0),
        ];

        let mut test_state = VisitVoxelTestState::new(
            "Aligned with cartesian X",
            &expected_path,
            Some(VoxelFace::Left),
        );
        voxel_line_traversal(start, end, |voxel_coords, time, face| {
            visit_voxel_check(&mut test_state, voxel_coords, time, face)
        });
        assert_eq!(
            test_state.path_step_index,
            expected_path.len(),
            "{}: Expected end voxel reached",
            test_state.test_name
        );
    }

    {
        let end = Vec3::new(
            VOXEL_SIZE / 2.,
            2. * VOXEL_SIZE + VOXEL_SIZE / 2.,
            VOXEL_SIZE / 2.,
        );
        let expected_path = [
            IVec3::new(0, 0, 0),
            IVec3::new(0, 1, 0),
            IVec3::new(0, 2, 0),
        ];

        let mut test_state = VisitVoxelTestState::new(
            "Aligned with cartesian Y",
            &expected_path,
            Some(VoxelFace::Bottom),
        );
        voxel_line_traversal(start, end, |voxel_coords, time, face| {
            visit_voxel_check(&mut test_state, voxel_coords, time, face)
        });
        assert_eq!(
            test_state.path_step_index,
            expected_path.len(),
            "{}: Expected end voxel reached",
            test_state.test_name
        );
    }

    {
        let end = Vec3::new(
            VOXEL_SIZE / 2.,
            VOXEL_SIZE / 2.,
            2. * VOXEL_SIZE + VOXEL_SIZE / 2.,
        );
        let expected_path = [
            IVec3::new(0, 0, 0),
            IVec3::new(0, 0, 1),
            IVec3::new(0, 0, 2),
        ];

        let mut test_state = VisitVoxelTestState::new(
            "Aligned with cartesian Z",
            &expected_path,
            Some(VoxelFace::Back),
        );
        voxel_line_traversal(start, end, |voxel_coords, time, face| {
            visit_voxel_check(&mut test_state, voxel_coords, time, face)
        });
        assert_eq!(
            test_state.path_step_index,
            expected_path.len(),
            "{}: Expected end voxel reached",
            test_state.test_name
        );
    }
}

#[test]
fn voxel_line_traversal_ending_on_voxel_boundary() {
    let start = Vec3::new(-5. * VOXEL_SIZE, VOXEL_SIZE / 2., 1.9815);
    let end = Vec3::new(0., 0., 50. * VOXEL_SIZE);
    let expected_path = [
        IVec3::new(-5, 0, 1),
        IVec3::new(-5, 0, 2),
        IVec3::new(-5, 0, 3),
        IVec3::new(-5, 0, 4),
        IVec3::new(-5, 0, 5),
        IVec3::new(-5, 0, 6),
        IVec3::new(-5, 0, 7),
        IVec3::new(-5, 0, 8),
        IVec3::new(-5, 0, 9),
        IVec3::new(-5, 0, 10),
        IVec3::new(-5, 0, 11),
        IVec3::new(-4, 0, 11),
        IVec3::new(-4, 0, 12),
        IVec3::new(-4, 0, 13),
        IVec3::new(-4, 0, 14),
        IVec3::new(-4, 0, 15),
        IVec3::new(-4, 0, 16),
        IVec3::new(-4, 0, 17),
        IVec3::new(-4, 0, 18),
        IVec3::new(-4, 0, 19),
        IVec3::new(-4, 0, 20),
        IVec3::new(-4, 0, 21),
        IVec3::new(-3, 0, 21),
        IVec3::new(-3, 0, 22),
        IVec3::new(-3, 0, 23),
        IVec3::new(-3, 0, 24),
        IVec3::new(-3, 0, 25),
        IVec3::new(-3, 0, 26),
        IVec3::new(-3, 0, 27),
        IVec3::new(-3, 0, 28),
        IVec3::new(-3, 0, 29),
        IVec3::new(-3, 0, 30),
        IVec3::new(-2, 0, 30),
        IVec3::new(-2, 0, 31),
        IVec3::new(-2, 0, 32),
        IVec3::new(-2, 0, 33),
        IVec3::new(-2, 0, 34),
        IVec3::new(-2, 0, 35),
        IVec3::new(-2, 0, 36),
        IVec3::new(-2, 0, 37),
        IVec3::new(-2, 0, 38),
        IVec3::new(-2, 0, 39),
        IVec3::new(-2, 0, 40),
        IVec3::new(-1, 0, 40),
        IVec3::new(-1, 0, 41),
        IVec3::new(-1, 0, 42),
        IVec3::new(-1, 0, 43),
        IVec3::new(-1, 0, 44),
        IVec3::new(-1, 0, 45),
        IVec3::new(-1, 0, 46),
        IVec3::new(-1, 0, 47),
        IVec3::new(-1, 0, 48),
        IVec3::new(-1, 0, 49),
        IVec3::new(-1, 0, 50),
        IVec3::new(0, 0, 50),
    ];

    let mut test_state =
        VisitVoxelTestState::new("Ending on voxel boundary", &expected_path, None);
    voxel_line_traversal(start, end, |voxel_coords, time, face| {
        visit_voxel_check(&mut test_state, voxel_coords, time, face)
    });
    assert_eq!(
        test_state.path_step_index,
        expected_path.len(),
        "{}: Expected end voxel reached",
        test_state.test_name
    );
}

#[test]
fn chunk_data_has_voxels_at_negative_coordinates() {
    let mut app = App::new();
    app.add_plugins((MinimalPlugins, VoxelWorldPlugin::<FloorWorld>::minimal()))
        .add_systems(Startup, spawn_camera::<FloorWorld>);
    let chunk_pos = IVec3::splat(-1);

    update_until(&mut app, "the chunk at (-1, -1, -1) to be ready", |app| {
        chunk_status::<FloorWorld>(app, chunk_pos) == Some(ChunkStatus::Ready)
    });

    let data = chunk_data::<FloorWorld>(&mut app, chunk_pos).unwrap();
    assert!(data.has_voxel(IVec3::new(-5, -1, -5), WorldVoxel::Solid(1)));
}

#[test]
fn can_get_chunk_data() {
    let mut app = _test_setup_app();

    app.add_systems(Update, |mut voxel_world: VoxelWorld<DefaultWorld>| {
        voxel_world.set_voxel(IVec3::new(0, 0, 0), WorldVoxel::Solid(1));
    });

    update_until(&mut app, "chunk data at the origin", |app| {
        chunk_data::<DefaultWorld>(app, IVec3::ZERO).is_some()
    });
}

#[test]
fn set_voxel_same_value_does_not_trigger_remesh() {
    let mut app = _test_setup_app();

    // Set a voxel and let the write fully process through the pipeline.
    // The system will keep writing Solid(1) every frame.
    app.add_systems(Update, |mut voxel_world: VoxelWorld<DefaultWorld>| {
        voxel_world.set_voxel(IVec3::new(0, 0, 0), WorldVoxel::Solid(1));
    });

    // Run enough frames so the initial write is flushed into modified_voxels
    // and subsequent same-value writes are skipped by the guard.
    for _ in 0..5 {
        app.update();
    }

    // Add a reader with a frame counter. On the first run, drain any stale
    // messages from the initial write. On subsequent runs, assert that no
    // new messages appear (guard prevents redundant remeshes).
    app.add_systems(
        Update,
        |mut ev: MessageReader<ChunkWillUpdate<DefaultWorld>>, mut frame: Local<u32>| {
            *frame += 1;
            if *frame > 1 {
                let count = ev.read().count();
                assert_eq!(
                    count, 0,
                    "expected no ChunkWillUpdate when setting voxel to same value"
                );
            } else {
                // Drain stale messages from the initial set_voxel
                ev.read().count();
            }
        },
    );

    app.update(); // reader frame 1 — drains stale messages
    app.update(); // reader frame 2 — asserts no new messages
    app.update(); // reader frame 3 — asserts again for good measure
}

#[test]
fn set_voxel_different_value_triggers_remesh() {
    let mut app = _test_setup_app();

    // Set initial voxel
    app.add_systems(Update, |mut voxel_world: VoxelWorld<DefaultWorld>| {
        voxel_world.set_voxel(IVec3::new(0, 0, 0), WorldVoxel::Solid(1));
    });
    app.update();
    app.update();

    // Now change to a different value — should produce a ChunkWillUpdate event
    app.add_systems(Update, |mut voxel_world: VoxelWorld<DefaultWorld>| {
        voxel_world.set_voxel(IVec3::new(0, 0, 0), WorldVoxel::Solid(2));
    });

    app.update(); // writes Solid(2) to buffer
    app.update(); // flushes — value differs, should trigger update

    app.add_systems(
        Update,
        |mut ev_chunk_will_update: MessageReader<ChunkWillUpdate<DefaultWorld>>| {
            let count = ev_chunk_will_update.read().count();
            assert!(
                count > 0,
                "expected ChunkWillUpdate when setting voxel to different value"
            );
        },
    );

    app.update();
}

/// How long the first voxel lookup of `SlowLookupWorld` holds up its chunk task.
const SLOW_LOOKUP_HOLD: std::time::Duration = std::time::Duration::from_millis(1500);
static SLOW_LOOKUP_UNTIL: std::sync::OnceLock<std::time::Instant> =
    std::sync::OnceLock::new();

/// A world whose chunk tasks are all held up for `SLOW_LOOKUP_HOLD` after the
/// first voxel lookup, as if generation were very expensive.
#[derive(Resource, Clone, Default)]
struct SlowLookupWorld;

impl VoxelWorldConfig for SlowLookupWorld {
    type MaterialIndex = u8;
    type ChunkUserBundle = ();

    fn spawning_distance(&self) -> u32 {
        2
    }

    fn voxel_lookup_delegate(&self) -> VoxelLookupDelegate<Self::MaterialIndex> {
        Box::new(|_, _, _| {
            Box::new(|_, _| {
                let until = *SLOW_LOOKUP_UNTIL
                    .get_or_init(|| std::time::Instant::now() + SLOW_LOOKUP_HOLD);
                std::thread::sleep(
                    until.saturating_duration_since(std::time::Instant::now()),
                );
                WorldVoxel::Air
            })
        })
    }
}

#[test]
fn frames_without_voxel_edits_do_not_wait_for_chunk_tasks() {
    let mut app = App::new();
    app.add_plugins((
        MinimalPlugins,
        VoxelWorldPlugin::<SlowLookupWorld>::minimal(),
    ));
    app.add_systems(Startup, |mut commands: Commands| {
        commands.spawn((
            Camera3d::default(),
            Transform::from_xyz(10.0, 10.0, 10.0).looking_at(Vec3::ZERO, Vec3::Y),
            VoxelWorldCamera::<SlowLookupWorld>::default(),
        ));
    });

    update_until(&mut app, "a chunk task to start", |_| {
        SLOW_LOOKUP_UNTIL.get().is_some()
    });

    // Chunk tasks are now generating; no frame should have to wait for them.
    let until = *SLOW_LOOKUP_UNTIL.get().unwrap();
    while std::time::Instant::now() < until {
        let start = std::time::Instant::now();
        app.update();
        assert!(
            start.elapsed() < SLOW_LOOKUP_HOLD / 3,
            "frame took {:?} while chunk tasks were generating",
            start.elapsed()
        );
    }
}

/// A world whose chunk tasks each take at least 20 ms, much longer than a frame.
#[derive(Resource, Clone, Default)]
struct SlowTaskWorld;

impl VoxelWorldConfig for SlowTaskWorld {
    type MaterialIndex = u8;
    type ChunkUserBundle = ();

    fn spawning_distance(&self) -> u32 {
        1
    }

    fn voxel_lookup_delegate(&self) -> VoxelLookupDelegate<Self::MaterialIndex> {
        Box::new(|_, _, _| {
            let mut slept = false;
            Box::new(move |pos, _| {
                if !slept {
                    std::thread::sleep(std::time::Duration::from_millis(20));
                    slept = true;
                }
                floor_voxel(pos)
            })
        })
    }
}

#[derive(Resource)]
struct KeepEditing(bool);

#[test]
fn a_chunk_edited_every_frame_still_publishes() {
    const EDITED: IVec3 = IVec3::new(16, 16, 16);
    let mut app = App::new();
    app.add_plugins((MinimalPlugins, VoxelWorldPlugin::<SlowTaskWorld>::minimal()))
        .insert_resource(KeepEditing(true))
        .add_systems(Startup, spawn_camera::<SlowTaskWorld>)
        .add_systems(
            Update,
            |mut voxel_world: VoxelWorld<SlowTaskWorld>,
             keep_editing: Res<KeepEditing>,
             mut frame: Local<u8>| {
                if keep_editing.0 {
                    *frame = frame.wrapping_add(1);
                    voxel_world.set_voxel(EDITED, WorldVoxel::Solid(1 + *frame % 2));
                }
            },
        );

    update_until(&mut app, "the edited chunk to publish", |app| {
        chunk_data::<SlowTaskWorld>(app, IVec3::ZERO).is_some()
    });

    app.world_mut().resource_mut::<KeepEditing>().0 = false;
    let last_edit =
        app.world_mut()
            .run_system_once(|voxel_world: VoxelWorld<SlowTaskWorld>| {
                voxel_world.get_voxel(EDITED)
            });
    let last_edit = last_edit.unwrap();
    update_until(&mut app, "the last edit to publish", |app| {
        chunk_status::<SlowTaskWorld>(app, IVec3::ZERO) == Some(ChunkStatus::Ready)
            && chunk_data::<SlowTaskWorld>(app, IVec3::ZERO)
                .and_then(|data| data.get_voxel_at_world_position(EDITED))
                == Some(last_edit)
    });
}

#[test]
fn an_edited_chunk_gets_a_task_slot_before_waiting_spawns() {
    let mut app = App::new();
    app.add_plugins((
        MinimalPlugins,
        VoxelWorldPlugin::<ThrottledWorld>::minimal(),
    ))
    .insert_resource(ThrottledWorld { task_slots: 1 })
    .add_systems(Startup, spawn_camera::<ThrottledWorld>);

    let statuses = |app: &mut App| -> Vec<(IVec3, Option<ChunkStatus>)> {
        let positions: Vec<IVec3> = app
            .world_mut()
            .query::<&Chunk<ThrottledWorld>>()
            .iter(app.world())
            .map(|chunk| chunk.position())
            .collect();
        positions
            .into_iter()
            .map(|pos| (pos, chunk_status::<ThrottledWorld>(app, pos)))
            .collect()
    };
    update_until(&mut app, "a chunk to be ready while others wait", |app| {
        let statuses = statuses(app);
        statuses.iter().any(|(_, s)| *s == Some(ChunkStatus::Ready))
            && statuses
                .iter()
                .any(|(_, s)| *s == Some(ChunkStatus::Pending))
    });

    // Close the slots and let running tasks finish, so the queue stands still.
    app.world_mut().resource_mut::<ThrottledWorld>().task_slots = 0;
    update_until(&mut app, "running tasks to finish", |app| {
        statuses(app)
            .iter()
            .all(|(_, s)| *s != Some(ChunkStatus::Generating))
    });
    let statuses_before = statuses(&mut app);
    let edited = statuses_before
        .iter()
        .find(|(_, s)| *s == Some(ChunkStatus::Ready))
        .unwrap()
        .0;
    let waiting: Vec<IVec3> = statuses_before
        .iter()
        .filter(|(_, s)| *s == Some(ChunkStatus::Pending))
        .map(|(pos, _)| *pos)
        .collect();
    assert!(!waiting.is_empty());

    app.world_mut()
        .run_system_once(move |mut voxel_world: VoxelWorld<ThrottledWorld>| {
            voxel_world.set_voxel(edited * CHUNK_SIZE_I + 16, WorldVoxel::Solid(7));
        })
        .unwrap();
    app.update();
    assert_eq!(
        chunk_status::<ThrottledWorld>(&mut app, edited),
        Some(ChunkStatus::Pending)
    );

    app.world_mut().resource_mut::<ThrottledWorld>().task_slots = 1;
    app.update();
    assert_ne!(
        chunk_status::<ThrottledWorld>(&mut app, edited),
        Some(ChunkStatus::Pending),
        "the edited chunk should have taken the free slot"
    );
    for pos in waiting {
        assert_eq!(
            chunk_status::<ThrottledWorld>(&mut app, pos),
            Some(ChunkStatus::Pending),
            "chunk {pos} took the slot before the edited chunk"
        );
    }
}

/// How long the first voxel lookup of `SlowEditedWorld` holds up its chunk task.
const SLOW_EDITED_HOLD: std::time::Duration = std::time::Duration::from_millis(1500);
static SLOW_EDITED_UNTIL: std::sync::OnceLock<std::time::Instant> =
    std::sync::OnceLock::new();

/// Like `SlowLookupWorld`, for a test of its own, since the hold starts once.
#[derive(Resource, Clone, Default)]
struct SlowEditedWorld;

impl VoxelWorldConfig for SlowEditedWorld {
    type MaterialIndex = u8;
    type ChunkUserBundle = ();

    fn spawning_distance(&self) -> u32 {
        2
    }

    fn voxel_lookup_delegate(&self) -> VoxelLookupDelegate<Self::MaterialIndex> {
        Box::new(|_, _, _| {
            Box::new(|_, _| {
                let until = *SLOW_EDITED_UNTIL
                    .get_or_init(|| std::time::Instant::now() + SLOW_EDITED_HOLD);
                std::thread::sleep(
                    until.saturating_duration_since(std::time::Instant::now()),
                );
                WorldVoxel::Air
            })
        })
    }
}

#[test]
fn frames_with_voxel_edits_do_not_wait_for_chunk_tasks() {
    let mut app = App::new();
    app.add_plugins((
        MinimalPlugins,
        VoxelWorldPlugin::<SlowEditedWorld>::minimal(),
    ))
    .add_systems(Startup, spawn_camera::<SlowEditedWorld>)
    .add_systems(
        Update,
        |mut voxel_world: VoxelWorld<SlowEditedWorld>, mut frame: Local<u8>| {
            *frame = frame.wrapping_add(1);
            voxel_world
                .set_voxel(IVec3::new(16, 16, 16), WorldVoxel::Solid(1 + *frame % 2));
        },
    );

    update_until(&mut app, "a chunk task to start", |_| {
        SLOW_EDITED_UNTIL.get().is_some()
    });

    // Chunk tasks are now generating; no frame should have to wait for them.
    let until = *SLOW_EDITED_UNTIL.get().unwrap();
    while std::time::Instant::now() < until {
        let start = std::time::Instant::now();
        app.update();
        assert!(
            start.elapsed() < SLOW_EDITED_HOLD / 3,
            "frame took {:?} while chunk tasks were generating",
            start.elapsed()
        );
    }
}

#[test]
fn a_published_chunk_includes_the_edits_in_its_padding() {
    let mut app = App::new();
    app.add_plugins((MinimalPlugins, VoxelWorldPlugin::<FloorWorld>::minimal()))
        .add_systems(Startup, spawn_camera::<FloorWorld>);

    update_until(&mut app, "the chunk at the origin to be ready", |app| {
        chunk_status::<FloorWorld>(app, IVec3::ZERO) == Some(ChunkStatus::Ready)
    });
    // Inside the chunk at x = -1, and in the padding of the chunk at the origin.
    let edited = IVec3::new(-1, 0, 0);
    app.world_mut()
        .run_system_once(move |mut voxel_world: VoxelWorld<FloorWorld>| {
            voxel_world.set_voxel(edited, WorldVoxel::Solid(3));
        })
        .unwrap();

    update_until(&mut app, "the edit in the origin chunk's padding", |app| {
        chunk_data::<FloorWorld>(app, IVec3::ZERO)
            .and_then(|data| data.get_voxel_at_world_position(edited))
            == Some(WorldVoxel::Solid(3))
    });
}

/// The chunk positions of each `ChunkWillSpawn` and `ChunkPublished` message.
#[derive(Resource, Default)]
struct Announced {
    will_spawn: Vec<IVec3>,
    published: Vec<IVec3>,
}

impl Announced {
    fn count(list: &[IVec3], chunk_pos: IVec3) -> usize {
        list.iter().filter(|pos| **pos == chunk_pos).count()
    }
}

fn record_announcements<C: VoxelWorldConfig>(
    mut announced: ResMut<Announced>,
    mut will_spawn: MessageReader<ChunkWillSpawn<C>>,
    mut published: MessageReader<ChunkPublished<C>>,
) {
    announced
        .will_spawn
        .extend(will_spawn.read().map(|msg| msg.chunk_key));
    announced
        .published
        .extend(published.read().map(|msg| msg.chunk_key));
}

#[test]
fn a_remeshed_chunk_is_published_again_but_spawns_once() {
    let mut app = App::new();
    app.add_plugins((MinimalPlugins, VoxelWorldPlugin::<FloorWorld>::minimal()))
        .init_resource::<Announced>()
        .add_systems(Startup, spawn_camera::<FloorWorld>)
        .add_systems(Update, record_announcements::<FloorWorld>);

    update_until(&mut app, "the chunk at the origin to be ready", |app| {
        chunk_status::<FloorWorld>(app, IVec3::ZERO) == Some(ChunkStatus::Ready)
    });
    app.world_mut()
        .run_system_once(|mut voxel_world: VoxelWorld<FloorWorld>| {
            voxel_world.set_voxel(IVec3::new(16, 16, 16), WorldVoxel::Solid(7));
        })
        .unwrap();
    update_until(&mut app, "the edited chunk to be published again", |app| {
        Announced::count(&app.world().resource::<Announced>().published, IVec3::ZERO) == 2
    });

    let announced = app.world().resource::<Announced>();
    assert_eq!(Announced::count(&announced.will_spawn, IVec3::ZERO), 1);
}

#[test]
fn a_chunk_spawned_again_after_despawning_announces_its_spawn_again() {
    let mut app = App::new();
    app.add_plugins((
        MinimalPlugins,
        VoxelWorldPlugin::<ThrottledWorld>::minimal(),
    ))
    .init_resource::<Announced>()
    .add_systems(Startup, spawn_camera::<ThrottledWorld>)
    .add_systems(Update, record_announcements::<ThrottledWorld>);
    let ready = |app: &mut App| {
        chunk_status::<ThrottledWorld>(app, IVec3::ZERO) == Some(ChunkStatus::Ready)
    };

    update_until(&mut app, "the chunk at the origin to be ready", ready);
    let home = Transform::from_xyz(10.0, 10.0, 10.0).looking_at(Vec3::ZERO, Vec3::Y);
    move_cameras::<ThrottledWorld>(&mut app, Transform::from_xyz(10_000.0, 10.0, 10.0));
    update_until(&mut app, "the chunk at the origin to despawn", |app| {
        chunk_status::<ThrottledWorld>(app, IVec3::ZERO).is_none()
    });
    move_cameras::<ThrottledWorld>(&mut app, home);
    update_until(&mut app, "the chunk at the origin to be ready again", ready);

    let announced = app.world().resource::<Announced>();
    assert_eq!(Announced::count(&announced.will_spawn, IVec3::ZERO), 2);
}
