use bevy::ecs::schedule::{LogLevel, ScheduleBuildSettings};
use bevy::ecs::system::RunSystemOnce;
use bevy::mesh::{Indices, VertexAttributeValues};
use bevy::prelude::*;
use std::collections::{BTreeMap, HashSet};
use std::sync::{Arc, Mutex};

use crate::configuration::VoxelWorldConfig;
use crate::meshing::{
    binary_chunk_mesh, block_mesh_chunk_mesh, generate_chunk_mesh,
    generate_chunk_mesh_for_shape,
};
use crate::plugin::VoxelWorldMaterialHandle;
use crate::prelude::*;
use crate::voxel_material::{prepare_voxel_texture, ATTRIBUTE_TEX_INDEX};
use crate::voxel_traversal::voxel_line_traversal;
use crate::{
    chunk::{Chunk, ChunkData, CHUNK_SIZE_F, CHUNK_SIZE_I, PADDED_CHUNK_SIZE},
    chunk_lifecycle::{ChunkIndex, ChunkStateOf},
    chunk_map::ChunkMap,
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

/// The chunk state of the level-0 chunk at `chunk_pos`.
fn chunk_status<C: VoxelWorldConfig>(
    app: &mut App,
    chunk_pos: IVec3,
) -> Option<ChunkStatus> {
    chunk_status_of::<C>(app, ChunkKey::new(chunk_pos, 0))
}

/// The chunk state of the chunk with the given key.
fn chunk_status_of<C: VoxelWorldConfig>(
    app: &mut App,
    key: ChunkKey,
) -> Option<ChunkStatus> {
    app.world_mut()
        .run_system_once(move |voxel_world: VoxelWorld<C>| voxel_world.chunk_state(key))
        .unwrap()
}

/// The published data of the level-0 chunk at `chunk_pos`.
fn chunk_data<C: VoxelWorldConfig>(
    app: &mut App,
    chunk_pos: IVec3,
) -> Option<ChunkData<C::MaterialIndex>> {
    app.world_mut()
        .run_system_once(move |voxel_world: VoxelWorld<C>| {
            voxel_world.get_chunk_data(ChunkKey::new(chunk_pos, 0))
        })
        .unwrap()
}

#[test]
fn a_system_can_hold_res_mut_of_the_config_and_voxel_world_together() {
    let mut app = App::new();
    app.add_plugins((MinimalPlugins, VoxelWorldPlugin::<FloorWorld>::minimal()))
        .add_systems(Startup, spawn_camera::<FloorWorld>)
        .add_systems(
            Update,
            |mut world: ResMut<FloorWorld>, voxel_world: VoxelWorld<FloorWorld>| {
                world.set_changed();
                voxel_world.get_voxel(IVec3::ZERO);
            },
        );

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

/// A world with no chunks, whose every chunk is hinted to hold `hinted`.
#[derive(Resource, Clone, Default)]
struct HintOnlyWorld {
    hinted: u8,
}

impl VoxelWorldConfig for HintOnlyWorld {
    type MaterialIndex = u8;
    type ChunkUserBundle = ();

    fn chunk_fill_hint(&self, _key: ChunkKey) -> FillHint<Self::MaterialIndex> {
        FillHint::Uniform(WorldVoxel::Solid(self.hinted))
    }
}

#[test]
fn a_startup_system_reads_a_config_inserted_after_the_plugin() {
    #[derive(Resource)]
    struct ReadAtStartup(WorldVoxel);

    let mut app = App::new();
    app.add_plugins((MinimalPlugins, VoxelWorldPlugin::<HintOnlyWorld>::minimal()))
        .insert_resource(HintOnlyWorld { hinted: 7 })
        .add_systems(
            Startup,
            |voxel_world: VoxelWorld<HintOnlyWorld>, mut commands: Commands| {
                commands
                    .insert_resource(ReadAtStartup(voxel_world.get_voxel(IVec3::ZERO)));
            },
        );
    app.update();

    assert_eq!(
        app.world().resource::<ReadAtStartup>().0,
        WorldVoxel::Solid(7)
    );
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
                    published.0.push(msg.chunk_key.position);
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
                        chunk.is_ok_and(|chunk| chunk.key() == msg.chunk_key),
                        "retiring chunk {} was already despawned",
                        msg.chunk_key
                    );
                    assert_eq!(
                        voxel_world.chunk_state(msg.chunk_key),
                        Some(ChunkStatus::Retiring)
                    );
                    retired.0.push(msg.chunk_key.position);
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

/// The expected place of a chunk in the order chunk tasks start in.
///
/// Chunks within `protected_distance` chunks of the camera come first. The
/// rest follow largest on screen first, ranked by the distance to the chunk's
/// centre divided by its extent.
fn start_order(key: ChunkKey, camera: Vec3, protected_distance: i32) -> (bool, f32) {
    let camera_chunk = (camera / CHUNK_SIZE_F).floor().as_ivec3();
    let nearest = camera_chunk.clamp(key.position, key.position + key.extent() - 1);
    (
        nearest.distance_squared(camera_chunk) > protected_distance.pow(2),
        key.centre().distance(camera) / key.voxel_extent() as f32,
    )
}

/// Runs a world that generates one chunk at a time until every chunk is ready,
/// and returns the chunks in the order they were published, which is the order
/// their tasks started, and the camera's position.
fn chunks_in_publish_order<C: VoxelWorldConfig>(app: &mut App) -> (Vec<ChunkKey>, Vec3) {
    app.init_resource::<LifecycleLog>()
        .add_systems(Last, record_lifecycle::<C>);
    update_until(app, "every chunk to be ready", |app| {
        let keys: Vec<ChunkKey> = app
            .world_mut()
            .query::<&Chunk<C>>()
            .iter(app.world())
            .map(|chunk| chunk.key())
            .collect();
        !keys.is_empty()
            && keys
                .into_iter()
                .all(|key| chunk_status_of::<C>(app, key) == Some(ChunkStatus::Ready))
    });

    let published = app
        .world()
        .resource::<LifecycleLog>()
        .published
        .iter()
        .map(|(_, key)| *key)
        .collect();
    let camera = app
        .world_mut()
        .query_filtered::<&GlobalTransform, With<VoxelWorldCamera<C>>>()
        .single(app.world())
        .unwrap()
        .translation();
    (published, camera)
}

#[test]
fn chunks_are_generated_around_the_camera_first_then_nearest_first() {
    let mut app = App::new();
    app.add_plugins((
        MinimalPlugins,
        VoxelWorldPlugin::<ThrottledWorld>::minimal(),
    ))
    .insert_resource(ThrottledWorld { task_slots: 1 })
    .add_systems(Startup, spawn_camera::<ThrottledWorld>);

    let (published, camera) = chunks_in_publish_order::<ThrottledWorld>(&mut app);

    let order: Vec<(bool, f32)> = published
        .iter()
        .map(|&key| start_order(key, camera, 1))
        .collect();
    assert!(order.len() > 10, "only {} chunks published", order.len());
    assert!(
        order.windows(2).all(|pair| pair[0] <= pair[1]),
        "chunks published out of order: {order:?}"
    );
}

/// Solid ground below y = 0, streamed all around the camera to three chunks
/// away, at level 0 next to the camera and at level 1 beyond, one chunk task at
/// a time.
#[derive(Resource, Clone, Default)]
struct OneTaskLevelsWorld;

impl VoxelWorldConfig for OneTaskLevelsWorld {
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

    fn max_active_chunk_threads(&self) -> usize {
        1
    }

    fn voxel_lookup_delegate(&self) -> VoxelLookupDelegate<Self::MaterialIndex> {
        Box::new(|_, _, _| Box::new(|pos, _| floor_voxel(pos)))
    }

    fn chunk_lod(
        &self,
        chunk_position: IVec3,
        _previous_lod: Option<LodLevel>,
        camera_position: Vec3,
    ) -> LodLevel {
        let camera_chunk = (camera_position / CHUNK_SIZE_F).floor().as_ivec3();
        if chunk_position.distance_squared(camera_chunk) < 2 {
            0
        } else {
            1
        }
    }
}

#[test]
fn coarse_chunks_are_generated_alongside_the_fine_chunks_as_large_on_screen() {
    let mut app = App::new();
    app.add_plugins((
        MinimalPlugins,
        VoxelWorldPlugin::<OneTaskLevelsWorld>::minimal(),
    ))
    .add_systems(Startup, spawn_camera::<OneTaskLevelsWorld>);

    let (published, camera) = chunks_in_publish_order::<OneTaskLevelsWorld>(&mut app);

    let first_coarse = published.iter().position(|key| key.lod_level == 1);
    let last_fine = published.iter().rposition(|key| key.lod_level == 0);
    assert!(
        matches!((first_coarse, last_fine), (Some(coarse), Some(fine)) if coarse < fine),
        "no coarse chunk was generated before a fine one: {published:?}"
    );
    let order: Vec<(bool, f32)> = published
        .iter()
        .map(|&key| start_order(key, camera, 1))
        .collect();
    assert!(
        order.windows(2).all(|pair| pair[0] <= pair[1]),
        "chunks published out of order: {order:?}"
    );
}

/// A world that spawns every chunk within two chunks of the camera, in view or not.
#[derive(Resource, Clone, Default)]
struct AllAroundWorld;

impl VoxelWorldConfig for AllAroundWorld {
    type MaterialIndex = u8;
    type ChunkUserBundle = ();

    fn spawning_distance(&self) -> u32 {
        2
    }

    fn chunk_spawn_strategy(&self) -> ChunkSpawnStrategy {
        ChunkSpawnStrategy::Close
    }

    fn chunk_despawn_strategy(&self) -> ChunkDespawnStrategy {
        ChunkDespawnStrategy::FarAway
    }

    fn voxel_lookup_delegate(&self) -> VoxelLookupDelegate<Self::MaterialIndex> {
        FloorWorld.voxel_lookup_delegate()
    }
}

#[test]
fn every_edited_chunk_gets_a_task_in_the_same_frame() {
    let mut app = App::new();
    app.add_plugins((
        MinimalPlugins,
        VoxelWorldPlugin::<AllAroundWorld>::minimal(),
    ))
    .add_systems(Startup, spawn_camera::<AllAroundWorld>);

    let positions = |app: &mut App| -> Vec<IVec3> {
        app.world_mut()
            .query::<&Chunk<AllAroundWorld>>()
            .iter(app.world())
            .map(|chunk| chunk.position())
            .collect()
    };
    update_until(&mut app, "every chunk to be ready", |app| {
        let positions = positions(app);
        !positions.is_empty()
            && positions.into_iter().all(|pos| {
                chunk_status::<AllAroundWorld>(app, pos) == Some(ChunkStatus::Ready)
            })
    });

    // Far more chunks than the task pool finishes in an idle frame.
    let edited = positions(&mut app);
    assert!(edited.len() > 30, "only {} chunks", edited.len());
    let to_edit = edited.clone();
    app.world_mut()
        .run_system_once(move |mut voxel_world: VoxelWorld<AllAroundWorld>| {
            for pos in &to_edit {
                voxel_world.set_voxel(pos * CHUNK_SIZE_I + 16, WorldVoxel::Solid(7));
            }
        })
        .unwrap();
    app.update();

    for pos in edited {
        assert_ne!(
            chunk_status::<AllAroundWorld>(&mut app, pos),
            Some(ChunkStatus::Pending),
            "edited chunk {pos} is still waiting for a task"
        );
    }
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
    let mut app = meshing_app();
    app.add_plugins(VoxelWorldPlugin::with_config(MissingTextureWorld))
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

/// The material of worlds that use the built-in voxel material.
type DefaultVoxelMaterial = bevy::pbr::ExtendedMaterial<
    StandardMaterial,
    crate::voxel_material::StandardVoxelMaterial,
>;

/// A world with the example voxel texture.
#[derive(Resource, Clone, Default)]
struct TexturedWorld;

impl VoxelWorldConfig for TexturedWorld {
    type MaterialIndex = u8;
    type ChunkUserBundle = ();

    fn spawning_distance(&self) -> u32 {
        2
    }

    fn voxel_lookup_delegate(&self) -> VoxelLookupDelegate<Self::MaterialIndex> {
        FloorWorld.voxel_lookup_delegate()
    }

    fn voxel_texture(&self) -> Option<(String, u32)> {
        Some(("example_voxel_texture.png".into(), 4))
    }
}

/// A second world like `FloorWorld`, for tests that need two.
#[derive(Resource, Clone, Default)]
struct SecondFloorWorld;

impl VoxelWorldConfig for SecondFloorWorld {
    type MaterialIndex = u8;
    type ChunkUserBundle = ();

    fn spawning_distance(&self) -> u32 {
        2
    }

    fn voxel_lookup_delegate(&self) -> VoxelLookupDelegate<Self::MaterialIndex> {
        FloorWorld.voxel_lookup_delegate()
    }
}

/// A world whose custom material is created and added by the test.
#[derive(Resource, Clone, Default)]
struct HandAddedMaterialWorld;

impl VoxelWorldConfig for HandAddedMaterialWorld {
    type MaterialIndex = u8;
    type ChunkUserBundle = ();

    fn spawning_distance(&self) -> u32 {
        2
    }

    fn voxel_lookup_delegate(&self) -> VoxelLookupDelegate<Self::MaterialIndex> {
        FloorWorld.voxel_lookup_delegate()
    }

    fn init_custom_materials(&self) -> bool {
        false
    }
}

/// An app with what chunk meshes and voxel materials need, but no renderer.
fn meshing_app() -> App {
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
    .init_asset::<bevy::shader::Shader>();
    app
}

/// The material handles on the chunks of world `C`.
fn chunk_materials<C: VoxelWorldConfig, M: Material>(app: &mut App) -> Vec<Handle<M>> {
    app.world_mut()
        .query_filtered::<&MeshMaterial3d<M>, With<Chunk<C>>>()
        .iter(app.world())
        .map(|material| material.0.clone())
        .collect()
}

fn has_chunk_mesh<C: VoxelWorldConfig>(app: &mut App) -> bool {
    app.world_mut()
        .query_filtered::<(), (With<crate::mesh_cache::MeshRef>, With<Chunk<C>>)>()
        .iter(app.world())
        .next()
        .is_some()
}

/// Runs `app` until the chunks of world `Other` have material `M`, and checks
/// that the chunk meshes of `MissingTextureWorld` have not been shown.
fn assert_only_missing_texture_world_waits<Other: VoxelWorldConfig, M: Material>(
    app: &mut App,
) {
    update_until(app, "the other world's chunks to get materials", |app| {
        !chunk_materials::<Other, M>(app).is_empty()
            && has_chunk_mesh::<MissingTextureWorld>(app)
    });

    let shown = app
        .world_mut()
        .query_filtered::<(), (With<Mesh3d>, With<Chunk<MissingTextureWorld>>)>()
        .iter(app.world())
        .count();
    assert_eq!(shown, 0, "meshes were shown before the texture loaded");
}

#[test]
fn worlds_with_different_voxel_textures_get_different_materials() {
    let mut app = meshing_app();
    // Normally registered by RenderPlugin.
    app.register_asset_loader(bevy::image::ImageLoader::new(
        bevy::image::CompressedImageFormats::empty(),
    ))
    .add_plugins(VoxelWorldPlugin::with_config(TexturedWorld))
    .add_plugins(VoxelWorldPlugin::with_config(FloorWorld))
    .add_systems(
        Startup,
        (spawn_camera::<TexturedWorld>, spawn_camera::<FloorWorld>),
    );

    update_until(&mut app, "both worlds' chunks to get materials", |app| {
        !chunk_materials::<TexturedWorld, DefaultVoxelMaterial>(app).is_empty()
            && !chunk_materials::<FloorWorld, DefaultVoxelMaterial>(app).is_empty()
    });

    let textured = chunk_materials::<TexturedWorld, DefaultVoxelMaterial>(&mut app);
    let built_in = chunk_materials::<FloorWorld, DefaultVoxelMaterial>(&mut app);
    assert!(
        textured.iter().all(|material| !built_in.contains(material)),
        "the two worlds' chunks share a material"
    );

    let example_texture = app
        .world()
        .resource::<AssetServer>()
        .get_handle::<Image>("example_voxel_texture.png")
        .expect("the example texture should have been requested");
    let materials = app.world().resource::<Assets<DefaultVoxelMaterial>>();
    for material in &textured {
        assert_eq!(
            materials.get(material).unwrap().extension.voxels_texture,
            example_texture,
            "a chunk of the textured world has another world's texture"
        );
    }
}

#[test]
fn a_world_does_not_wait_for_the_voxel_texture_of_a_world_built_after_it() {
    let mut app = meshing_app();
    app.add_plugins(VoxelWorldPlugin::with_config(FloorWorld))
        .add_plugins(VoxelWorldPlugin::with_config(MissingTextureWorld))
        .add_systems(
            Startup,
            (
                spawn_camera::<FloorWorld>,
                spawn_camera::<MissingTextureWorld>,
            ),
        );

    assert_only_missing_texture_world_waits::<FloorWorld, DefaultVoxelMaterial>(&mut app);
}

#[test]
fn a_world_waits_for_its_voxel_texture_when_a_world_built_after_it_is_ready() {
    let mut app = meshing_app();
    app.add_plugins(VoxelWorldPlugin::with_config(MissingTextureWorld))
        .add_plugins(VoxelWorldPlugin::with_config(FloorWorld))
        .add_systems(
            Startup,
            (
                spawn_camera::<MissingTextureWorld>,
                spawn_camera::<FloorWorld>,
            ),
        );

    assert_only_missing_texture_world_waits::<FloorWorld, DefaultVoxelMaterial>(&mut app);
}

#[test]
fn a_world_waits_for_its_voxel_texture_when_a_world_built_after_it_has_a_custom_material()
{
    let mut app = meshing_app();
    app.init_asset::<StandardMaterial>()
        .add_plugins(VoxelWorldPlugin::with_config(MissingTextureWorld))
        .add_plugins(
            VoxelWorldPlugin::with_config(FloorWorld)
                .with_material(StandardMaterial::default()),
        )
        .add_systems(
            Startup,
            (
                spawn_camera::<MissingTextureWorld>,
                spawn_camera::<FloorWorld>,
            ),
        );

    assert_only_missing_texture_world_waits::<FloorWorld, StandardMaterial>(&mut app);
}

#[test]
fn worlds_sharing_a_custom_material_type_keep_their_own_materials() {
    let red = Color::srgb(1.0, 0.0, 0.0);
    let blue = Color::srgb(0.0, 0.0, 1.0);

    let mut app = meshing_app();
    app.init_asset::<StandardMaterial>()
        .add_plugins(VoxelWorldPlugin::with_config(FloorWorld).with_material(
            StandardMaterial {
                base_color: red,
                ..default()
            },
        ))
        .add_plugins(
            VoxelWorldPlugin::with_config(SecondFloorWorld).with_material(
                StandardMaterial {
                    base_color: blue,
                    ..default()
                },
            ),
        )
        .add_systems(
            Startup,
            (spawn_camera::<FloorWorld>, spawn_camera::<SecondFloorWorld>),
        );

    update_until(&mut app, "both worlds' chunks to get materials", |app| {
        !chunk_materials::<FloorWorld, StandardMaterial>(app).is_empty()
            && !chunk_materials::<SecondFloorWorld, StandardMaterial>(app).is_empty()
    });

    let first = chunk_materials::<FloorWorld, StandardMaterial>(&mut app);
    let second = chunk_materials::<SecondFloorWorld, StandardMaterial>(&mut app);
    let materials = app.world().resource::<Assets<StandardMaterial>>();
    assert!(first
        .iter()
        .all(|material| materials.get(material).unwrap().base_color == red));
    assert!(second
        .iter()
        .all(|material| materials.get(material).unwrap().base_color == blue));
}

#[test]
fn the_plugin_adds_the_material_handle_of_a_custom_material() {
    let mut app = meshing_app();
    app.init_asset::<StandardMaterial>().add_plugins(
        VoxelWorldPlugin::with_config(FloorWorld)
            .with_material(StandardMaterial::default()),
    );

    let handle = &app
        .world()
        .resource::<VoxelWorldMaterialHandle<StandardMaterial>>()
        .handle;
    let materials = app.world().resource::<Assets<StandardMaterial>>();
    assert!(materials.get(handle).is_some());
}

#[test]
fn a_hand_added_material_handle_reaches_worlds_that_do_not_init_custom_materials() {
    let mut app = meshing_app();
    app.init_asset::<StandardMaterial>()
        .add_plugins(
            VoxelWorldPlugin::with_config(HandAddedMaterialWorld)
                .with_material(StandardMaterial::default()),
        )
        .add_systems(Startup, spawn_camera::<HandAddedMaterialWorld>);

    update_until(&mut app, "a chunk mesh to be published", |app| {
        has_chunk_mesh::<HandAddedMaterialWorld>(app)
    });
    assert!(
        chunk_materials::<HandAddedMaterialWorld, StandardMaterial>(&mut app).is_empty(),
        "a chunk got a material before one was added"
    );

    let handle = app
        .world_mut()
        .resource_mut::<Assets<StandardMaterial>>()
        .add(StandardMaterial::default());
    app.insert_resource(VoxelWorldMaterialHandle {
        handle: handle.clone(),
    });

    update_until(&mut app, "the chunks to get the added material", |app| {
        !chunk_materials::<HandAddedMaterialWorld, StandardMaterial>(app).is_empty()
    });
    assert!(
        chunk_materials::<HandAddedMaterialWorld, StandardMaterial>(&mut app)
            .iter()
            .all(|material| *material == handle)
    );
}

/// A second world whose custom material is created and added by the test.
#[derive(Resource, Clone, Default)]
struct SecondHandAddedMaterialWorld;

impl VoxelWorldConfig for SecondHandAddedMaterialWorld {
    type MaterialIndex = u8;
    type ChunkUserBundle = ();

    fn spawning_distance(&self) -> u32 {
        2
    }

    fn voxel_lookup_delegate(&self) -> VoxelLookupDelegate<Self::MaterialIndex> {
        FloorWorld.voxel_lookup_delegate()
    }

    fn init_custom_materials(&self) -> bool {
        false
    }
}

/// The one material handle on the chunks of world `C`, once they all have it.
fn the_chunk_material<C: VoxelWorldConfig, M: Material>(app: &mut App) -> Handle<M> {
    update_until(app, "the chunks to get a material", |app| {
        !chunk_materials::<C, M>(app).is_empty()
    });
    let materials = chunk_materials::<C, M>(app);
    assert!(
        materials.iter().all(|material| *material == materials[0]),
        "the chunks of one world have different materials"
    );
    materials[0].clone()
}

#[test]
fn a_default_worlds_chunks_get_the_default_material() {
    let mut app = meshing_app();
    app.add_plugins(VoxelWorldPlugin::with_config(FloorWorld))
        .add_systems(Startup, spawn_camera::<FloorWorld>);

    let material = the_chunk_material::<FloorWorld, DefaultVoxelMaterial>(&mut app);

    let materials = app.world().resource::<Assets<DefaultVoxelMaterial>>();
    assert_eq!(materials.len(), 1);
    assert_eq!(
        materials.get(&material).unwrap().base.perceptual_roughness,
        0.95
    );
}

#[test]
fn a_custom_material_worlds_chunks_get_its_material() {
    let red = Color::srgb(1.0, 0.0, 0.0);
    let mut app = meshing_app();
    app.init_asset::<StandardMaterial>()
        .add_plugins(VoxelWorldPlugin::with_config(FloorWorld).with_material(
            StandardMaterial {
                base_color: red,
                ..default()
            },
        ))
        .add_systems(Startup, spawn_camera::<FloorWorld>);

    let material = the_chunk_material::<FloorWorld, StandardMaterial>(&mut app);

    let materials = app.world().resource::<Assets<StandardMaterial>>();
    assert_eq!(materials.get(&material).unwrap().base_color, red);
    assert!(
        app.world()
            .get_resource::<Assets<DefaultVoxelMaterial>>()
            .is_none_or(|materials| materials.is_empty()),
        "a custom material world made a default material"
    );
}

#[test]
fn a_hand_added_material_is_shared_by_worlds_that_name_it() {
    let mut app = meshing_app();
    app.init_asset::<StandardMaterial>()
        .add_plugins(
            VoxelWorldPlugin::with_config(HandAddedMaterialWorld)
                .with_material(StandardMaterial::default()),
        )
        .add_plugins(
            VoxelWorldPlugin::with_config(SecondHandAddedMaterialWorld)
                .with_material(StandardMaterial::default()),
        )
        .add_systems(
            Startup,
            (
                spawn_camera::<HandAddedMaterialWorld>,
                spawn_camera::<SecondHandAddedMaterialWorld>,
            ),
        );
    assert!(
        app.world()
            .resource::<Assets<StandardMaterial>>()
            .is_empty(),
        "the plugin made a material for a world that doesn't init one"
    );

    let handle = app
        .world_mut()
        .resource_mut::<Assets<StandardMaterial>>()
        .add(StandardMaterial::default());
    app.insert_resource(VoxelWorldMaterialHandle {
        handle: handle.clone(),
    });

    assert_eq!(
        the_chunk_material::<HandAddedMaterialWorld, StandardMaterial>(&mut app),
        handle
    );
    assert_eq!(
        the_chunk_material::<SecondHandAddedMaterialWorld, StandardMaterial>(&mut app),
        handle
    );
}

#[test]
fn two_worlds_each_get_their_own_material() {
    let mut app = meshing_app();
    app.add_plugins(VoxelWorldPlugin::with_config(FloorWorld))
        .add_plugins(VoxelWorldPlugin::with_config(SecondFloorWorld))
        .add_systems(
            Startup,
            (spawn_camera::<FloorWorld>, spawn_camera::<SecondFloorWorld>),
        );

    let first = the_chunk_material::<FloorWorld, DefaultVoxelMaterial>(&mut app);
    let second = the_chunk_material::<SecondFloorWorld, DefaultVoxelMaterial>(&mut app);
    assert_ne!(first, second);
}

#[test]
fn a_minimal_world_with_a_custom_material_assigns_no_material() {
    let mut app = meshing_app();
    app.init_asset::<StandardMaterial>()
        .add_plugins(
            VoxelWorldPlugin::<FloorWorld>::minimal()
                .with_material(StandardMaterial::default()),
        )
        .add_systems(Startup, spawn_camera::<FloorWorld>);

    update_until(&mut app, "a chunk to be published", |app| {
        !app.world()
            .resource::<ChunkMap<FloorWorld, u8>>()
            .get_read_lock()
            .is_empty()
    });

    assert!(app
        .world()
        .resource::<Assets<StandardMaterial>>()
        .is_empty());
    assert!(app
        .world()
        .get_resource::<VoxelWorldMaterialHandle<StandardMaterial>>()
        .is_none());
    assert!(chunk_materials::<FloorWorld, StandardMaterial>(&mut app).is_empty());
}

#[test]
fn a_voxel_texture_shared_by_two_worlds_can_be_prepared_twice() {
    let mut image = Image::new_fill(
        bevy::render::render_resource::Extent3d {
            width: 2,
            height: 8,
            depth_or_array_layers: 1,
        },
        bevy::render::render_resource::TextureDimension::D2,
        &[0, 0, 0, 255],
        bevy::render::render_resource::TextureFormat::Rgba8UnormSrgb,
        bevy::asset::RenderAssetUsages::default(),
    );

    prepare_voxel_texture(&mut image, 4).unwrap();
    prepare_voxel_texture(&mut image, 4).unwrap();

    assert_eq!(image.texture_descriptor.size.depth_or_array_layers, 4);
    assert_eq!(image.height(), 2);
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

/// The voxel that `set_origin_voxel` sets at the origin every frame.
#[derive(Resource)]
struct OriginVoxel(WorldVoxel);

fn set_origin_voxel(mut voxel_world: VoxelWorld<DefaultWorld>, voxel: Res<OriginVoxel>) {
    voxel_world.set_voxel(IVec3::ZERO, voxel.0);
}

/// The chunks that `ChunkWillUpdate` was fired for.
#[derive(Resource, Default)]
struct Updated(Vec<IVec3>);

fn record_updates(
    mut updated: ResMut<Updated>,
    mut ev: MessageReader<ChunkWillUpdate<DefaultWorld>>,
) {
    updated
        .0
        .extend(ev.read().map(|msg| msg.chunk_key.position));
}

/// A `DefaultWorld` app that sets `voxel` at the origin every frame once the
/// chunk there is ready, and records the chunks `ChunkWillUpdate` fires for.
fn app_setting_the_origin_voxel(voxel: WorldVoxel) -> App {
    let mut app = _test_setup_app();
    update_until(&mut app, "the chunk at the origin to be ready", |app| {
        chunk_status::<DefaultWorld>(app, IVec3::ZERO) == Some(ChunkStatus::Ready)
    });
    app.insert_resource(OriginVoxel(voxel))
        .init_resource::<Updated>()
        .add_systems(Update, (set_origin_voxel, record_updates));
    app
}

fn origin_updated(app: &mut App) -> bool {
    app.world().resource::<Updated>().0.contains(&IVec3::ZERO)
}

#[test]
fn chunk_will_update_event() {
    let mut app = app_setting_the_origin_voxel(WorldVoxel::Solid(1));
    update_until(
        &mut app,
        "the chunk at the origin to update",
        origin_updated,
    );
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
                .extend(events.read().map(|event| event.chunk_key.position));
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

    update_until(&mut app, "the chunks around the origin to remesh", |_| {
        let remeshed_chunks: HashSet<_> =
            remeshed_chunks.lock().unwrap().iter().copied().collect();
        expected.is_subset(&remeshed_chunks)
    });
}

/// Voxel lookups by the chunk tasks of the chunk at the origin, in `ReuseWorld`.
static REUSE_LOOKUPS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// A `FloorWorld` that counts the voxel lookups of the chunk at the origin.
#[derive(Resource, Clone, Default)]
struct ReuseWorld;

impl VoxelWorldConfig for ReuseWorld {
    type MaterialIndex = u8;
    type ChunkUserBundle = ();

    fn spawning_distance(&self) -> u32 {
        1
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
fn a_remeshed_chunk_reuses_its_previous_data() {
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

/// A world solid all through, so every chunk is full of one material.
#[derive(Resource, Clone, Default)]
struct SolidWorld;

impl VoxelWorldConfig for SolidWorld {
    type MaterialIndex = u8;
    type ChunkUserBundle = ();

    fn spawning_distance(&self) -> u32 {
        1
    }

    fn chunk_despawn_strategy(&self) -> ChunkDespawnStrategy {
        ChunkDespawnStrategy::FarAway
    }

    fn voxel_lookup_delegate(&self) -> VoxelLookupDelegate<Self::MaterialIndex> {
        Box::new(|_, _, _| Box::new(|_, _| WorldVoxel::Solid(1)))
    }
}

#[test]
fn a_full_chunk_keeps_an_edited_voxel_of_another_material() {
    let mut app = App::new();
    app.add_plugins((MinimalPlugins, VoxelWorldPlugin::<SolidWorld>::minimal()))
        .init_resource::<Announced>()
        .add_systems(Startup, spawn_camera::<SolidWorld>)
        .add_systems(Update, record_announcements::<SolidWorld>);

    update_until(&mut app, "the chunk at the origin to be ready", |app| {
        chunk_status::<SolidWorld>(app, IVec3::ZERO) == Some(ChunkStatus::Ready)
    });
    let edited = IVec3::new(16, 16, 16);
    app.world_mut()
        .run_system_once(move |mut voxel_world: VoxelWorld<SolidWorld>| {
            voxel_world.set_voxel(edited, WorldVoxel::Solid(2));
        })
        .unwrap();
    update_until(&mut app, "the edited chunk to be published again", |app| {
        Announced::count(&app.world().resource::<Announced>().published, IVec3::ZERO) == 2
    });

    let data = chunk_data::<SolidWorld>(&mut app, IVec3::ZERO).unwrap();
    assert_eq!(
        data.get_voxel_at_world_position(edited),
        Some(WorldVoxel::Solid(2))
    );
    assert_eq!(
        data.get_voxel_at_world_position(IVec3::new(16, 15, 16)),
        Some(WorldVoxel::Solid(1))
    );
}

#[test]
fn chunk_at_finds_the_chunk_covering_a_voxel() {
    let mut app = App::new();
    app.add_plugins((MinimalPlugins, VoxelWorldPlugin::<FloorWorld>::minimal()))
        .add_systems(Startup, spawn_camera::<FloorWorld>);
    update_until(&mut app, "the chunk at the origin to be ready", |app| {
        chunk_status::<FloorWorld>(app, IVec3::ZERO) == Some(ChunkStatus::Ready)
    });

    let found = app
        .world_mut()
        .run_system_once(|voxel_world: VoxelWorld<FloorWorld>| {
            [
                voxel_world.chunk_at(IVec3::new(5, 3, 31)),
                voxel_world.chunk_at(IVec3::new(-1, 0, 0)),
                voxel_world.chunk_at(IVec3::splat(10_000)),
            ]
        })
        .unwrap();
    assert_eq!(found[0], Some(ChunkKey::new(IVec3::ZERO, 0)));
    assert_eq!(found[1], Some(ChunkKey::new(IVec3::new(-1, 0, 0), 0)));
    assert_eq!(found[2], None);
}

/// Solid ground below y = 0, streamed at level 0 within two chunks of the
/// camera and at level 1 beyond, all around the camera.
#[derive(Resource, Clone, Default)]
struct LevelsWorld;

impl VoxelWorldConfig for LevelsWorld {
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

    fn voxel_lookup_delegate(&self) -> VoxelLookupDelegate<Self::MaterialIndex> {
        Box::new(|_, _, _| Box::new(|pos, _| floor_voxel(pos)))
    }

    fn chunk_lod(
        &self,
        chunk_position: IVec3,
        _previous_lod: Option<LodLevel>,
        camera_position: Vec3,
    ) -> LodLevel {
        let camera_chunk = (camera_position / CHUNK_SIZE_F).floor().as_ivec3();
        if chunk_position.distance_squared(camera_chunk) < 4 {
            0
        } else {
            1
        }
    }
}

/// A level-1 chunk two chunks from the camera of `levels_world_app`, below
/// ground, and a voxel in it at one of its samples.
const COARSE: ChunkKey = ChunkKey {
    position: IVec3::new(2, -2, 0),
    lod_level: 1,
};
const COARSE_SAMPLE: IVec3 = IVec3::new(68, -2, 6);

/// A `LevelsWorld` app with the camera in the chunk at the origin, run until
/// the chunk `COARSE` is ready.
fn levels_world_app() -> App {
    let mut app = App::new();
    app.add_plugins((MinimalPlugins, VoxelWorldPlugin::<LevelsWorld>::minimal()))
        .add_systems(Startup, spawn_camera::<LevelsWorld>);
    update_until(&mut app, "the coarse chunk to be ready", |app| {
        chunk_status_of::<LevelsWorld>(app, COARSE) == Some(ChunkStatus::Ready)
    });
    app
}

#[test]
fn chunks_stream_at_the_level_chunk_lod_wants_and_answer_for_every_voxel() {
    let mut app = levels_world_app();

    let (near, far, voxels) = app
        .world_mut()
        .run_system_once(|voxel_world: VoxelWorld<LevelsWorld>| {
            (
                voxel_world.chunk_at(IVec3::new(5, -2, 5)),
                voxel_world.chunk_at(COARSE_SAMPLE),
                [
                    voxel_world.get_voxel(COARSE_SAMPLE),
                    // Between two data voxels: answered by the one below it.
                    voxel_world.get_voxel(IVec3::new(69, -1, 7)),
                    voxel_world.get_voxel(IVec3::new(68, -64, 6)),
                ],
            )
        })
        .unwrap();
    assert_eq!(near, Some(ChunkKey::new(IVec3::new(0, -1, 0), 0)));
    assert_eq!(far, Some(COARSE));
    assert_eq!(voxels, [WorldVoxel::Solid(1); 3]);
}

/// The chunk keys `ChunkWillUpdate` was fired for.
#[derive(Resource, Default)]
struct UpdatedKeys(Vec<ChunkKey>);

fn record_updated_keys<C: VoxelWorldConfig>(
    mut updated: ResMut<UpdatedKeys>,
    mut ev: MessageReader<ChunkWillUpdate<C>>,
) {
    updated.0.extend(ev.read().map(|msg| msg.chunk_key));
}

#[test]
fn an_edit_updates_the_coarse_chunk_covering_it() {
    let mut app = levels_world_app();
    app.init_resource::<UpdatedKeys>()
        .add_systems(Update, record_updated_keys::<LevelsWorld>);
    app.world_mut()
        .run_system_once(|mut voxel_world: VoxelWorld<LevelsWorld>| {
            voxel_world.set_voxel(COARSE_SAMPLE, WorldVoxel::Solid(7));
        })
        .unwrap();

    update_until(&mut app, "the coarse chunk to show the edit", |app| {
        app.world_mut()
            .run_system_once(|voxel_world: VoxelWorld<LevelsWorld>| {
                voxel_world
                    .get_chunk_data(COARSE)
                    .and_then(|data| data.get_voxel_at_world_position(COARSE_SAMPLE))
            })
            .unwrap()
            == Some(WorldVoxel::Solid(7))
    });
    assert!(app.world().resource::<UpdatedKeys>().0.contains(&COARSE));
}

/// The frame of each `ChunkPublished` and `ChunkWillDespawn`.
#[derive(Resource, Default)]
struct LifecycleLog {
    frame: u32,
    published: Vec<(u32, ChunkKey)>,
    despawned: Vec<(u32, ChunkKey)>,
}

fn record_lifecycle<C: VoxelWorldConfig>(
    mut log: ResMut<LifecycleLog>,
    mut published: MessageReader<ChunkPublished<C>>,
    mut despawned: MessageReader<ChunkWillDespawn<C>>,
) {
    log.frame += 1;
    let frame = log.frame;
    let keys: Vec<_> = published.read().map(|msg| (frame, msg.chunk_key)).collect();
    log.published.extend(keys);
    let keys: Vec<_> = despawned.read().map(|msg| (frame, msg.chunk_key)).collect();
    log.despawned.extend(keys);
}

#[test]
fn a_coarse_chunk_retires_once_the_finer_chunks_replacing_it_are_published() {
    let mut app = levels_world_app();
    app.init_resource::<LifecycleLog>()
        .add_systems(Update, record_lifecycle::<LevelsWorld>);

    // Into the coarse chunk, which then wants level 0.
    move_cameras::<LevelsWorld>(
        &mut app,
        Transform::from_translation((COARSE.position.as_vec3() + 0.5) * CHUNK_SIZE_F),
    );
    update_until(&mut app, "the coarse chunk to be gone", |app| {
        chunk_status_of::<LevelsWorld>(app, COARSE).is_none()
    });

    let log = app.world().resource::<LifecycleLog>();
    let (retired_at, _) = log
        .despawned
        .iter()
        .find(|(_, key)| *key == COARSE)
        .expect("the coarse chunk was retired");
    for child in COARSE.children() {
        let (published_at, _) = log
            .published
            .iter()
            .find(|(_, key)| *key == child)
            .unwrap_or_else(|| panic!("{child} was published"));
        // Chunk data is published before the streaming policy runs in a frame,
        // so the retirement can follow a publish within the same frame.
        assert!(
            published_at <= retired_at,
            "{child} published in frame {published_at}, the coarse chunk retired in {retired_at}"
        );
    }
}

/// `FloorWorld` with every chunk's fill hinted: the floor is level, so a chunk
/// row at or above y = 0 is air and a row below it solid. Streams all around the
/// camera to four chunks away.
#[derive(Resource, Clone, Default)]
struct HintedFloorWorld;

impl VoxelWorldConfig for HintedFloorWorld {
    type MaterialIndex = u8;
    type ChunkUserBundle = ();

    fn spawning_distance(&self) -> u32 {
        4
    }

    fn chunk_spawn_strategy(&self) -> ChunkSpawnStrategy {
        ChunkSpawnStrategy::Close
    }

    fn chunk_despawn_strategy(&self) -> ChunkDespawnStrategy {
        ChunkDespawnStrategy::FarAway
    }

    fn voxel_lookup_delegate(&self) -> VoxelLookupDelegate<Self::MaterialIndex> {
        FloorWorld.voxel_lookup_delegate()
    }

    fn chunk_fill_hint(&self, key: ChunkKey) -> FillHint<Self::MaterialIndex> {
        if key.position.y >= 0 {
            FillHint::Uniform(WorldVoxel::Air)
        } else if key.position.y + key.extent() <= 0 {
            FillHint::Uniform(WorldVoxel::Solid(1))
        } else {
            FillHint::Unknown
        }
    }
}

/// The keys of the level-0 chunks within one chunk of the chunk at the origin:
/// a camera's protected radius there.
fn protected_around_origin() -> HashSet<ChunkKey> {
    let mut keys = HashSet::new();
    for x in -1..=1 {
        for y in -1..=1 {
            for z in -1..=1 {
                let offset = IVec3::new(x, y, z);
                if offset.length_squared() <= 1 {
                    keys.insert(ChunkKey::new(offset, 0));
                }
            }
        }
    }
    keys
}

fn chunk_keys<C: VoxelWorldConfig>(app: &mut App) -> HashSet<ChunkKey> {
    app.world_mut()
        .query::<&Chunk<C>>()
        .iter(app.world())
        .map(|chunk| chunk.key())
        .collect()
}

#[test]
fn hinted_chunks_are_not_spawned_and_get_voxel_answers_from_their_hints() {
    let mut app = App::new();
    app.add_plugins((
        MinimalPlugins,
        VoxelWorldPlugin::<HintedFloorWorld>::minimal(),
    ))
    .add_systems(Startup, spawn_camera::<HintedFloorWorld>);

    // The camera is in the chunk at the origin, whose protected radius spawns
    // whatever the hints say.
    update_until(&mut app, "the chunk below the camera to be ready", |app| {
        chunk_status::<HintedFloorWorld>(app, IVec3::new(0, -1, 0))
            == Some(ChunkStatus::Ready)
    });
    for _ in 0..5 {
        app.update();
    }

    assert_eq!(
        chunk_keys::<HintedFloorWorld>(&mut app),
        protected_around_origin()
    );
    let (air, rock, covering) = app
        .world_mut()
        .run_system_once(|voxel_world: VoxelWorld<HintedFloorWorld>| {
            (
                voxel_world.get_voxel(IVec3::new(100, 5, 100)),
                voxel_world.get_voxel(IVec3::new(100, -5, 100)),
                voxel_world.chunk_at(IVec3::new(100, -5, 100)),
            )
        })
        .unwrap();
    assert_eq!(air, WorldVoxel::Air);
    assert_eq!(rock, WorldVoxel::Solid(1));
    assert_eq!(covering, None, "no chunk was spawned there");
    assert_eq!(
        app.world()
            .resource::<ChunkIndex<HintedFloorWorld>>()
            .hint_mismatches,
        0
    );
}

#[test]
fn the_raycast_hits_hinted_ground_where_no_chunk_exists() {
    let mut app = App::new();
    app.add_plugins((
        MinimalPlugins,
        VoxelWorldPlugin::<HintedFloorWorld>::minimal(),
    ))
    // Two cameras six chunks apart, so that the published chunks' bounds, which
    // the raycast is traced through, span hinted chunks between them.
    .add_systems(Startup, |mut commands: Commands| {
        for chunk_z in [0.0, 6.0] {
            let transform =
                Transform::from_xyz(10.0, 10.0, 10.0 + chunk_z * CHUNK_SIZE_F);
            commands.spawn((
                Camera3d::default(),
                transform,
                GlobalTransform::from(transform),
                VoxelWorldCamera::<HintedFloorWorld>::default(),
            ));
        }
    });
    update_until(
        &mut app,
        "the chunks below both cameras to be ready",
        |app| {
            [IVec3::new(0, -1, 0), IVec3::new(0, -1, 6)]
                .into_iter()
                .all(|chunk_pos| {
                    chunk_status::<HintedFloorWorld>(app, chunk_pos)
                        == Some(ChunkStatus::Ready)
                })
        },
    );

    let (result, covering) = app
        .world_mut()
        .run_system_once(|voxel_world: VoxelWorld<HintedFloorWorld>| {
            let ray = Ray3d {
                origin: Vec3::new(16.5, 16.5, 100.5),
                direction: -Dir3::Y,
            };
            (
                voxel_world.raycast(ray, &|_| true),
                voxel_world.chunk_at(IVec3::new(16, -1, 100)),
            )
        })
        .unwrap();

    assert_eq!(covering, None, "no chunk was spawned there");
    assert_eq!(
        result,
        Some(VoxelRaycastResult {
            position: Vec3::new(16.0, -1.0, 100.0),
            normal: Some(Vec3::Y),
            voxel: WorldVoxel::Solid(1),
        })
    );
}

/// `FloorWorld` hinted air everywhere, which is wrong below the floor.
#[derive(Resource, Clone, Default)]
struct MisHintedWorld;

impl VoxelWorldConfig for MisHintedWorld {
    type MaterialIndex = u8;
    type ChunkUserBundle = ();

    fn spawning_distance(&self) -> u32 {
        2
    }

    fn chunk_despawn_strategy(&self) -> ChunkDespawnStrategy {
        ChunkDespawnStrategy::FarAway
    }

    fn voxel_lookup_delegate(&self) -> VoxelLookupDelegate<Self::MaterialIndex> {
        FloorWorld.voxel_lookup_delegate()
    }

    fn chunk_fill_hint(&self, _key: ChunkKey) -> FillHint<Self::MaterialIndex> {
        FillHint::Uniform(WorldVoxel::Air)
    }
}

#[test]
fn a_hinted_chunk_in_the_protected_radius_is_generated_and_a_wrong_hint_is_noticed() {
    let mut app = App::new();
    app.add_plugins((
        MinimalPlugins,
        VoxelWorldPlugin::<MisHintedWorld>::minimal(),
    ))
    .add_systems(Startup, spawn_camera::<MisHintedWorld>);
    let below = IVec3::new(0, -1, 0);
    update_until(&mut app, "the chunk below the camera to be ready", |app| {
        chunk_status::<MisHintedWorld>(app, below) == Some(ChunkStatus::Ready)
    });

    // The hint never fills the chunk's data: it was generated, and is solid.
    let voxel = app
        .world_mut()
        .run_system_once(|voxel_world: VoxelWorld<MisHintedWorld>| {
            voxel_world.get_voxel(IVec3::new(5, -5, 5))
        })
        .unwrap();
    assert_eq!(voxel, WorldVoxel::Solid(1));
    assert!(chunk_data::<MisHintedWorld>(&mut app, below)
        .is_some_and(|data| data.has_generated()));
    if cfg!(debug_assertions) {
        assert!(
            app.world()
                .resource::<ChunkIndex<MisHintedWorld>>()
                .hint_mismatches
                > 0,
            "the wrong hint went unnoticed"
        );
    }
}

#[test]
fn a_chunk_is_found_by_its_exact_key_only() {
    let mut app = App::new();
    app.add_plugins((MinimalPlugins, VoxelWorldPlugin::<FloorWorld>::minimal()))
        .add_systems(Startup, spawn_camera::<FloorWorld>);
    update_until(&mut app, "the chunk at the origin to be ready", |app| {
        chunk_status::<FloorWorld>(app, IVec3::ZERO) == Some(ChunkStatus::Ready)
    });

    // The level-1 chunk covering the origin has the same position.
    let coarser = ChunkKey::new(IVec3::ZERO, 1);
    let (state, has_data) = app
        .world_mut()
        .run_system_once(move |voxel_world: VoxelWorld<FloorWorld>| {
            (
                voxel_world.chunk_state(coarser),
                voxel_world.get_chunk_data(coarser).is_some(),
            )
        })
        .unwrap();
    assert_eq!(state, None);
    assert!(!has_data);
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

/// A quad of a chunk mesh.
///
/// Corners are snapped to the voxel grid. The other float attributes, except
/// texture coordinates, are stored as their bits so they compare exactly.
#[derive(Clone, Copy, Debug)]
struct MeshQuad {
    /// Its indices, counted from its first vertex.
    indices: [u32; 6],
    corners: [IVec3; 4],
    normals: [[u32; 3]; 4],
    tex_coords: [Vec2; 4],
    materials: [[u32; 3]; 4],
    colors: [[u32; 4]; 4],
}

impl MeshQuad {
    /// A key that is different for every face of a chunk.
    fn face_key(&self) -> ([u32; 3], [[i32; 3]; 4]) {
        (
            self.normals[0],
            self.corners.map(|corner| corner.to_array()),
        )
    }

    /// Splits the quad into one quad per voxel face it covers, each with the
    /// attributes the quad has there.
    ///
    /// The quad's ambient occlusion must not change along an axis on which it
    /// covers more than one face.
    fn faces(&self) -> Vec<MeshQuad> {
        let [c0, c1, c2, c3] = self.corners;
        let [uv0, uv1, uv2, uv3] = self.tex_coords;
        assert!(
            c3 == c1 + c2 - c0 && (uv3 - (uv1 + uv2 - uv0)).abs().max_element() < 1e-3,
            "not a rectangle: {self:?}"
        );
        let (width, height) =
            ((c1 - c0).abs().element_sum(), (c2 - c0).abs().element_sum());
        let (u, v) = ((c1 - c0) / width, (c2 - c0) / height);
        let [ao0, ao1, ao2, ao3] = self.colors;
        assert!(
            width == 1 || (ao0 == ao1 && ao2 == ao3),
            "ambient occlusion changes along a merged axis: {self:?}"
        );
        assert!(
            height == 1 || (ao0 == ao2 && ao1 == ao3),
            "ambient occlusion changes along a merged axis: {self:?}"
        );
        let tex_coord = |a: i32, b: i32| {
            uv0 + (uv1 - uv0) * a as f32 / width as f32
                + (uv2 - uv0) * b as f32 / height as f32
        };
        let corner_offsets = [(0, 0), (1, 0), (0, 1), (1, 1)];
        let mut faces = Vec::new();
        for b in 0..height {
            for a in 0..width {
                faces.push(MeshQuad {
                    corners: corner_offsets.map(|(i, j)| c0 + u * (a + i) + v * (b + j)),
                    tex_coords: corner_offsets.map(|(i, j)| tex_coord(a + i, b + j)),
                    ..*self
                });
            }
        }
        faces
    }
}

/// The quads of `mesh`, a chunk mesh of the padded `shape`.
fn mesh_quads(mesh: &Mesh, shape: UVec3) -> Vec<MeshQuad> {
    let (
        Some(VertexAttributeValues::Float32x3(positions)),
        Some(VertexAttributeValues::Float32x3(normals)),
        Some(VertexAttributeValues::Float32x2(tex_coords)),
        Some(VertexAttributeValues::Uint32x3(materials)),
        Some(VertexAttributeValues::Float32x4(colors)),
        Some(Indices::U32(indices)),
    ) = (
        mesh.attribute(Mesh::ATTRIBUTE_POSITION),
        mesh.attribute(Mesh::ATTRIBUTE_NORMAL),
        mesh.attribute(Mesh::ATTRIBUTE_UV_0),
        mesh.attribute(ATTRIBUTE_TEX_INDEX),
        mesh.attribute(Mesh::ATTRIBUTE_COLOR),
        mesh.indices(),
    )
    else {
        panic!("unexpected chunk mesh layout");
    };
    assert_eq!(positions.len() * 6, indices.len() * 4);
    let voxel_size = CHUNK_SIZE_F / (shape - 2).max(UVec3::ONE).as_vec3();

    indices
        .as_chunks::<6>()
        .0
        .iter()
        .map(|quad_indices| {
            let first = *quad_indices.iter().min().unwrap();
            let vertex = |corner: usize| first as usize + corner;
            MeshQuad {
                indices: quad_indices.map(|index| index - first),
                corners: std::array::from_fn(|corner| {
                    let position = Vec3::from(positions[vertex(corner)]);
                    ((position - 1.0) / voxel_size + 1.0).round().as_ivec3()
                }),
                normals: std::array::from_fn(|corner| {
                    normals[vertex(corner)].map(f32::to_bits)
                }),
                tex_coords: std::array::from_fn(|corner| {
                    Vec2::from(tex_coords[vertex(corner)])
                }),
                materials: std::array::from_fn(|corner| materials[vertex(corner)]),
                colors: std::array::from_fn(|corner| {
                    colors[vertex(corner)].map(f32::to_bits)
                }),
            }
        })
        .collect()
}

/// Voxels of a few solid boxes of random materials on solid ground, with a
/// sprinkling of single voxels.
fn blocky_voxels(rng: &mut StdRng, shape: UVec3) -> Vec<WorldVoxel<u8>> {
    let runtime_shape = RuntimeShape::<u32, 3>::new(shape.to_array());
    let ground = rng.random_range(0..shape.y);
    let boxes: Vec<_> = (0..rng.random_range(1..8))
        .map(|_| {
            let mut corner = || {
                UVec3::new(
                    rng.random_range(0..shape.x),
                    rng.random_range(0..shape.y),
                    rng.random_range(0..shape.z),
                )
            };
            let (a, b) = (corner(), corner());
            (
                a.min(b),
                a.max(b),
                WorldVoxel::Solid(rng.random_range(1..4)),
            )
        })
        .collect();

    (0..runtime_shape.size())
        .map(|index| {
            let position = UVec3::from(runtime_shape.delinearize(index));
            let in_box = boxes.iter().rev().find(|(min, max, _)| {
                position.cmpge(*min).all() && position.cmple(*max).all()
            });
            if rng.random_bool(0.01) {
                WorldVoxel::Solid(rng.random_range(1..4))
            } else if let Some((_, _, voxel)) = in_box {
                *voxel
            } else if position.y < ground {
                WorldVoxel::Solid(1)
            } else {
                WorldVoxel::Air
            }
        })
        .collect()
}

/// Voxels each solid with the chance `solid_chance`, of a random material, and
/// otherwise air or unset.
fn noisy_voxels(
    rng: &mut StdRng,
    shape: UVec3,
    solid_chance: f64,
) -> Vec<WorldVoxel<u8>> {
    (0..shape.element_product())
        .map(|_| {
            if rng.random_bool(solid_chance) {
                WorldVoxel::Solid(rng.random_range(1..4))
            } else if rng.random_bool(0.5) {
                WorldVoxel::Air
            } else {
                WorldVoxel::Unset
            }
        })
        .collect()
}

#[test]
fn binary_meshing_covers_block_mesh_faces_with_quads_that_look_the_same() {
    let mut rng = StdRng::seed_from_u64(0x5EED);
    let texture_index_mapper: TextureIndexMapperFn<u8> =
        Arc::new(|material| [material as u32, material as u32 + 4, material as u32 + 8]);
    let (mut quad_count, mut face_count) = (0, 0);

    for round in 0..60 {
        let shape = match round {
            0 | 1 => UVec3::splat(34),
            2 | 3 => UVec3::splat(18),
            _ => {
                let dim = |rng: &mut StdRng| rng.random_range(3..=64);
                UVec3::new(dim(&mut rng), dim(&mut rng), dim(&mut rng))
            }
        };
        let voxels = if round % 2 == 0 {
            blocky_voxels(&mut rng, shape)
        } else {
            noisy_voxels(&mut rng, shape, [0.1, 0.5, 0.9][round % 3])
        };
        let runtime_shape = RuntimeShape::<u32, 3>::new(shape.to_array());
        let quads = mesh_quads(
            &binary_chunk_mesh(&voxels, &runtime_shape, texture_index_mapper.clone()),
            shape,
        );
        let faces = mesh_quads(
            &block_mesh_chunk_mesh(&voxels, &runtime_shape, texture_index_mapper.clone()),
            shape,
        );
        quad_count += quads.len();
        face_count += faces.len();

        let mut covered = BTreeMap::new();
        for face in quads.iter().flat_map(MeshQuad::faces) {
            assert!(
                covered.insert(face.face_key(), face).is_none(),
                "shape {shape}: two quads cover {face:?}"
            );
        }
        assert_eq!(covered.len(), faces.len(), "shape {shape}");
        let whole_voxels = (CHUNK_SIZE_F / (shape - 2).as_vec3()).fract() == Vec3::ZERO;
        for face in &faces {
            let Some(covering) = covered.get(&face.face_key()) else {
                panic!("shape {shape}: no quad covers {face:?}");
            };
            assert_eq!(
                (covering.indices, covering.normals),
                (face.indices, face.normals),
                "shape {shape}"
            );
            assert_eq!(
                (covering.materials, covering.colors),
                (face.materials, face.colors),
                "shape {shape}"
            );
            // The texture repeats, so it looks the same moved by whole tiles.
            let offset = covering.tex_coords[0] - face.tex_coords[0];
            for (covering, face) in covering.tex_coords.iter().zip(face.tex_coords) {
                assert!((covering - face - offset).abs().max_element() < 1e-3);
            }
            assert!(
                !whole_voxels || (offset - offset.round()).abs().max_element() < 1e-3
            );
        }
    }
    assert!(
        quad_count < face_count,
        "the voxels should have faces to merge"
    );
}

#[test]
fn flat_ground_meshes_as_one_quad() {
    let shape = RuntimeShape::<u32, 3>::new([PADDED_CHUNK_SIZE; 3]);
    let voxels: Vec<WorldVoxel<u8>> = (0..shape.size())
        .map(|index| match shape.delinearize(index) {
            [_, y, _] if y <= 4 => WorldVoxel::Solid(1),
            _ => WorldVoxel::Air,
        })
        .collect();

    let mesh = generate_chunk_mesh(Arc::from(voxels), IVec3::ZERO, Arc::new(|_| [0; 3]));

    assert_eq!(mesh.count_vertices(), 4);
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

/// A flat floor that keeps its chunks around two far apart cameras.
#[derive(Resource, Clone, Default)]
struct WideFloorWorld;

impl VoxelWorldConfig for WideFloorWorld {
    type MaterialIndex = u8;
    type ChunkUserBundle = ();

    fn spawning_distance(&self) -> u32 {
        2000
    }

    fn min_despawn_distance(&self) -> u32 {
        3
    }

    fn voxel_lookup_delegate(&self) -> VoxelLookupDelegate<Self::MaterialIndex> {
        Box::new(|_, _, _| Box::new(|pos, _| floor_voxel(pos)))
    }
}

#[test]
fn raycast_stays_on_its_ray_when_published_chunks_span_a_wide_area() {
    const FAR_X: f32 = 60_000.0;

    let mut app = App::new();
    app.add_plugins((
        MinimalPlugins,
        VoxelWorldPlugin::<WideFloorWorld>::minimal(),
    ))
    .add_systems(Startup, |mut commands: Commands| {
        for x in [10.0, FAR_X] {
            let transform = Transform::from_xyz(x, 10.0, 10.0);
            commands.spawn((
                Camera3d::default(),
                transform,
                GlobalTransform::from(transform),
                VoxelWorldCamera::<WideFloorWorld>::default(),
            ));
        }
    });
    let far_chunk = IVec3::new((FAR_X / CHUNK_SIZE_F).floor() as i32, -1, 0);
    update_until(
        &mut app,
        "the chunks the rays reach and the chunk below the far camera to be ready",
        |app| {
            [
                IVec3::new(0, -1, 0),
                IVec3::new(-1, -1, 0),
                IVec3::new(0, -1, -1),
                IVec3::new(-1, -1, -1),
                far_chunk,
            ]
            .into_iter()
            .all(|chunk_pos| {
                chunk_status::<WideFloorWorld>(app, chunk_pos) == Some(ChunkStatus::Ready)
            })
        },
    );

    let errors = app
        .world_mut()
        .run_system_once(|voxel_world: VoxelWorld<WideFloorWorld>| {
            (0..200)
                .map(|i| {
                    let angle = i as f32 * 0.0314;
                    let direction =
                        Vec3::new(angle.cos() * 0.5, -1.0, angle.sin() * 0.5).normalize();
                    let origin = Vec3::new(0.5, 40.5, 0.5);
                    // Where the ray meets the top of the floor.
                    let expected = origin + direction * (origin.y / -direction.y);
                    let ray = Ray3d {
                        origin,
                        direction: Dir3::new(direction).unwrap(),
                    };
                    let hit = voxel_world
                        .raycast(ray, &|_| true)
                        .expect("the ray points at the floor");
                    (hit.position.xz() + Vec2::splat(0.5)).distance(expected.xz())
                })
                .collect::<Vec<f32>>()
        })
        .unwrap();

    // The hit voxel's centre is at most half a voxel's diagonal from the ray.
    for error in errors {
        assert!(error < 1.0, "the hit is {error} voxels off the ray");
    }
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
    let mut app = app_setting_the_origin_voxel(WorldVoxel::Solid(1));
    update_until(
        &mut app,
        "the first edit to update the chunk",
        origin_updated,
    );

    app.world_mut().resource_mut::<Updated>().0.clear();
    for _ in 0..3 {
        app.update();
    }
    assert!(
        app.world().resource::<Updated>().0.is_empty(),
        "expected no ChunkWillUpdate when setting voxel to same value"
    );
}

#[test]
fn set_voxel_different_value_triggers_remesh() {
    let mut app = app_setting_the_origin_voxel(WorldVoxel::Solid(1));
    update_until(
        &mut app,
        "the first edit to update the chunk",
        origin_updated,
    );

    app.world_mut().resource_mut::<Updated>().0.clear();
    app.insert_resource(OriginVoxel(WorldVoxel::Solid(2)));
    update_until(
        &mut app,
        "the second edit to update the chunk",
        origin_updated,
    );
}

#[test]
fn set_voxel_and_back_does_not_trigger_remesh() {
    let mut app = app_setting_the_origin_voxel(WorldVoxel::Solid(1));
    update_until(
        &mut app,
        "the first edit to update the chunk",
        origin_updated,
    );

    app.world_mut()
        .run_system_once(|mut voxel_world: VoxelWorld<DefaultWorld>| {
            voxel_world.set_voxel(IVec3::ZERO, WorldVoxel::Solid(2));
            voxel_world.set_voxel(IVec3::ZERO, WorldVoxel::Solid(1));
        })
        .unwrap();
    app.world_mut().resource_mut::<Updated>().0.clear();
    for _ in 0..3 {
        app.update();
    }
    assert!(
        app.world().resource::<Updated>().0.is_empty(),
        "expected no ChunkWillUpdate when setting voxel to another value and back"
    );
}

#[test]
fn get_voxel_returns_the_last_voxel_set() {
    let mut app = _test_setup_app();
    // Creates the voxel world's resources.
    app.update();

    let voxels = app
        .world_mut()
        .run_system_once(|mut voxel_world: VoxelWorld<DefaultWorld>| {
            voxel_world.set_voxel(IVec3::ZERO, WorldVoxel::Solid(1));
            voxel_world.set_voxel(IVec3::ZERO, WorldVoxel::Solid(2));
            [
                voxel_world.get_voxel(IVec3::ZERO),
                voxel_world.get_voxel_fn()(IVec3::ZERO),
            ]
        })
        .unwrap();
    assert_eq!(voxels, [WorldVoxel::Solid(2); 2]);
}

#[test]
fn systems_using_voxel_world_can_run_in_parallel() {
    let mut app = _test_setup_app();
    // Creates the voxel world's resources.
    app.update();

    let mut schedule = Schedule::default();
    schedule.set_build_settings(ScheduleBuildSettings {
        ambiguity_detection: LogLevel::Error,
        ..default()
    });
    schedule.add_systems((
        |mut voxel_world: VoxelWorld<DefaultWorld>| {
            voxel_world.set_voxel(IVec3::NEG_X, WorldVoxel::Solid(1));
        },
        |mut voxel_world: VoxelWorld<DefaultWorld>| {
            voxel_world.set_voxel(IVec3::X, WorldVoxel::Solid(2));
        },
    ));
    schedule
        .initialize(app.world_mut())
        .expect("systems using VoxelWorld should not conflict");
    schedule.run(app.world_mut());

    let voxels = app
        .world_mut()
        .run_system_once(|voxel_world: VoxelWorld<DefaultWorld>| {
            [
                voxel_world.get_voxel(IVec3::NEG_X),
                voxel_world.get_voxel(IVec3::X),
            ]
        })
        .unwrap();
    assert_eq!(voxels, [WorldVoxel::Solid(1), WorldVoxel::Solid(2)]);
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

/// Like `SlowLookupWorld`, but for a separate test. Each hold starts only once,
/// from its own static, so two tests can't share one world.
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
        .extend(will_spawn.read().map(|msg| msg.chunk_key.position));
    announced
        .published
        .extend(published.read().map(|msg| msg.chunk_key.position));
}

/// A world with room for one chunk task at a time, whose chunk tasks each take
/// at least 100 ms.
#[derive(Resource, Clone, Default)]
struct OneSlowTaskWorld;

impl VoxelWorldConfig for OneSlowTaskWorld {
    type MaterialIndex = u8;
    type ChunkUserBundle = ();

    fn max_active_chunk_threads(&self) -> usize {
        1
    }

    fn voxel_lookup_delegate(&self) -> VoxelLookupDelegate<Self::MaterialIndex> {
        Box::new(|_, _, _| {
            let mut slept = false;
            Box::new(move |_, _| {
                if !slept {
                    std::thread::sleep(std::time::Duration::from_millis(100));
                    slept = true;
                }
                WorldVoxel::Air
            })
        })
    }
}

#[test]
fn despawning_a_generating_chunk_frees_its_task_slot_and_lets_it_respawn() {
    let mut app = App::new();
    app.add_plugins((
        MinimalPlugins,
        VoxelWorldPlugin::<OneSlowTaskWorld>::minimal(),
    ))
    .add_systems(Startup, spawn_camera::<OneSlowTaskWorld>);

    let mut running = None;
    update_until(&mut app, "a chunk task to start", |app| {
        let world = app.world_mut();
        running = world
            .query::<(
                Entity,
                &Chunk<OneSlowTaskWorld>,
                &ChunkStateOf<OneSlowTaskWorld>,
            )>()
            .iter(world)
            .find(|(_, _, state)| state.status() == ChunkStatus::Generating)
            .map(|(entity, chunk, _)| (entity, chunk.key()));
        running.is_some()
    });
    let (entity, key) = running.unwrap();

    app.world_mut().despawn(entity);

    update_until(&mut app, "the despawned chunk to publish again", |app| {
        chunk_status::<OneSlowTaskWorld>(app, key.position) == Some(ChunkStatus::Ready)
    });
    let data = chunk_data::<OneSlowTaskWorld>(&mut app, key.position)
        .expect("the respawned chunk is published");
    assert_ne!(data.get_entity(), entity);
}

#[test]
fn despawning_a_ready_chunk_removes_its_data_and_lets_it_respawn() {
    let mut app = App::new();
    app.add_plugins((MinimalPlugins, VoxelWorldPlugin::<FloorWorld>::minimal()))
        .init_resource::<Announced>()
        .add_systems(Startup, spawn_camera::<FloorWorld>)
        .add_systems(Update, record_announcements::<FloorWorld>);

    update_until(&mut app, "the chunk at the origin to be ready", |app| {
        chunk_status::<FloorWorld>(app, IVec3::ZERO) == Some(ChunkStatus::Ready)
    });
    let first_entity = chunk_data::<FloorWorld>(&mut app, IVec3::ZERO)
        .expect("the chunk is published")
        .get_entity();

    app.world_mut().despawn(first_entity);

    update_until(&mut app, "the chunk to publish again", |app| {
        chunk_status::<FloorWorld>(app, IVec3::ZERO) == Some(ChunkStatus::Ready)
            && chunk_data::<FloorWorld>(app, IVec3::ZERO)
                .is_some_and(|data| data.get_entity() != first_entity)
    });
    assert_eq!(
        Announced::count(&app.world().resource::<Announced>().will_spawn, IVec3::ZERO),
        2,
        "each of the two chunks announces its first publish"
    );
}

#[test]
fn a_respawned_chunk_keeps_its_data_when_the_old_chunk_is_removed_late() {
    let mut app = App::new();
    app.add_plugins((MinimalPlugins, VoxelWorldPlugin::<FloorWorld>::minimal()))
        .init_resource::<Announced>()
        .add_systems(Startup, spawn_camera::<FloorWorld>)
        .add_systems(Update, record_announcements::<FloorWorld>);

    update_until(&mut app, "the chunk at the origin to be ready", |app| {
        chunk_status::<FloorWorld>(app, IVec3::ZERO) == Some(ChunkStatus::Ready)
    });
    let first_entity = chunk_data::<FloorWorld>(&mut app, IVec3::ZERO)
        .expect("the chunk is published")
        .get_entity();

    // A reader on another thread would hold this lock the same way. While it is
    // held the chunk map can't be written, so its buffers wait.
    let chunk_map = app.world().resource::<ChunkMap<FloorWorld, u8>>().get_map();
    let read_guard = chunk_map.read().unwrap();

    move_cameras::<FloorWorld>(&mut app, Transform::from_xyz(90_000.0, 10.0, 10.0));
    update_until(&mut app, "the chunk at the origin to despawn", |app| {
        chunk_status::<FloorWorld>(app, IVec3::ZERO).is_none()
    });
    move_cameras::<FloorWorld>(&mut app, Transform::from_xyz(10.0, 10.0, 10.0));
    update_until(
        &mut app,
        "the chunk at the origin to be ready again",
        |app| chunk_status::<FloorWorld>(app, IVec3::ZERO) == Some(ChunkStatus::Ready),
    );

    drop(read_guard);
    for _ in 0..3 {
        app.update();
    }

    let data = chunk_data::<FloorWorld>(&mut app, IVec3::ZERO)
        .expect("the new chunk's data outlasts the old chunk's removal");
    assert_ne!(data.get_entity(), first_entity);
    assert_eq!(
        Announced::count(&app.world().resource::<Announced>().will_spawn, IVec3::ZERO),
        2,
        "each of the two chunks announces its first publish"
    );
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

/// A world whose chunks sample every other voxel at level 0, so their padding
/// is two voxels deep, on a floor below y = 0.
#[derive(Resource, Clone, Default)]
struct SparseDataWorld;

impl VoxelWorldConfig for SparseDataWorld {
    type MaterialIndex = u8;
    type ChunkUserBundle = ();

    fn spawning_distance(&self) -> u32 {
        2
    }

    fn voxel_lookup_delegate(&self) -> VoxelLookupDelegate<Self::MaterialIndex> {
        Box::new(|_, _, _| Box::new(|pos, _| floor_voxel(pos)))
    }

    fn chunk_data_shape(&self, _lod_level: LodLevel) -> UVec3 {
        padded_chunk_shape_uniform(CHUNK_SIZE_I as u32 / 2)
    }
}

#[test]
fn an_edit_in_a_neighbours_padding_two_voxels_deep_regenerates_the_chunk() {
    let mut app = App::new();
    app.add_plugins((
        MinimalPlugins,
        VoxelWorldPlugin::<SparseDataWorld>::minimal(),
    ))
    .add_systems(Startup, spawn_camera::<SparseDataWorld>);
    update_until(&mut app, "the chunks on either side to be ready", |app| {
        [IVec3::ZERO, IVec3::new(-1, 0, 0)]
            .into_iter()
            .all(|chunk_pos| {
                chunk_status::<SparseDataWorld>(app, chunk_pos)
                    == Some(ChunkStatus::Ready)
            })
    });

    // Chunk 0's first data voxel along x sits two voxels into chunk -1.
    let edited = IVec3::new(-2, 8, 8);
    app.world_mut()
        .run_system_once(move |mut voxel_world: VoxelWorld<SparseDataWorld>| {
            voxel_world.set_voxel(edited, WorldVoxel::Solid(7));
        })
        .unwrap();

    let padding_voxel = UVec3::new(0, 5, 5);
    update_until(&mut app, "chunk 0's padding to show the edit", |app| {
        app.world_mut()
            .run_system_once(move |voxel_world: VoxelWorld<SparseDataWorld>| {
                voxel_world
                    .get_chunk_data(ChunkKey::new(IVec3::ZERO, 0))
                    .is_some_and(|data| {
                        data.get_voxel(padding_voxel) == WorldVoxel::Solid(7)
                    })
            })
            .unwrap()
    });
}

#[test]
fn a_coarse_chunk_with_sparse_data_shows_an_edit_past_the_next_level_0_chunk() {
    let key = ChunkKey::new(IVec3::ZERO, 2);
    let shape = UVec3::splat(4);
    let edits = crate::voxel_edits::VoxelEdits::<DefaultWorld, u8>::default();
    edits.set(IVec3::new(-64, 0, 0), WorldVoxel::Solid(1));
    edits.commit();

    let mut task = crate::chunk::ChunkTask::<DefaultWorld, u8>::new(
        Entity::PLACEHOLDER,
        key,
        shape,
        shape,
        edits.chunk_edits(key, crate::chunk::padding_depth(shape, key.lod_level)),
        FillHint::Unknown,
    );
    task.generate(|_, _| WorldVoxel::Air, None);

    assert_eq!(
        task.chunk_data.get_voxel(UVec3::new(0, 1, 1)),
        WorldVoxel::Solid(1)
    );
}

/// A chunk task for the level-0 chunk at the origin with the data `shape`.
fn task_with_shape(shape: UVec3) -> crate::chunk::ChunkTask<DefaultWorld, u8> {
    let key = ChunkKey::new(IVec3::ZERO, 0);
    let edits = crate::voxel_edits::VoxelEdits::<DefaultWorld, u8>::default();
    crate::chunk::ChunkTask::new(
        Entity::PLACEHOLDER,
        key,
        shape,
        shape,
        edits.chunk_edits(key, crate::chunk::padding_depth(shape, key.lod_level)),
        FillHint::Unknown,
    )
}

#[test]
fn a_densely_sampled_chunk_reads_its_low_padding_from_the_neighbouring_chunk() {
    let shape = UVec3::splat(66);
    let mut task = task_with_shape(shape);
    task.generate(
        |pos, _| {
            if pos.x >= 0 {
                WorldVoxel::Solid(1)
            } else {
                WorldVoxel::Air
            }
        },
        None,
    );

    assert_eq!(
        task.chunk_data.get_voxel(UVec3::new(0, 30, 30)),
        WorldVoxel::Air
    );
    assert!(!task.chunk_data.is_full());

    let mesh = generate_chunk_mesh_for_shape(
        task.chunk_data.voxels_arc().unwrap(),
        IVec3::ZERO,
        shape,
        shape,
        Arc::new(|_| [0, 0, 0]),
    );
    let Some(VertexAttributeValues::Float32x3(normals)) =
        mesh.attribute(Mesh::ATTRIBUTE_NORMAL)
    else {
        panic!("normals missing");
    };
    assert!(
        normals.iter().any(|normal| normal[0] < -0.5),
        "the solid voxels at x = 0 have a face towards the air at x < 0"
    );
}

#[test]
fn a_chunk_with_uneven_spacing_samples_and_answers_at_rounded_down_positions() {
    // 32 voxels over 10 data voxels is 3.2 voxels apart.
    let mut task = task_with_shape(UVec3::splat(12));
    let sampled = Arc::new(Mutex::new(std::collections::BTreeSet::new()));
    let sampled_in_task = sampled.clone();
    task.generate(
        move |pos, _| {
            sampled_in_task.lock().unwrap().insert(pos.x);
            WorldVoxel::Solid((pos.x + 10) as u8)
        },
        None,
    );

    let sampled = sampled.lock().unwrap();
    assert_eq!(
        sampled.iter().copied().collect::<Vec<_>>(),
        vec![-4, 0, 3, 6, 9, 12, 16, 19, 22, 25, 28, 32]
    );

    // Every voxel in the padded data is answered by the last data voxel at or
    // before it.
    for x in -4..=32 {
        let covering = *sampled.range(..=x).next_back().unwrap();
        assert_eq!(
            task.chunk_data
                .get_voxel_at_world_position(IVec3::new(x, 0, 0)),
            Some(WorldVoxel::Solid((covering + 10) as u8)),
            "voxel {x}"
        );
    }
    assert_eq!(
        task.chunk_data
            .get_voxel_at_world_position(IVec3::new(-5, 0, 0)),
        None
    );
    assert!(
        -sampled.first().unwrap() <= crate::chunk::padding_depth(UVec3::splat(12), 0).x,
        "the padding depth covers the deepest sample"
    );
}

mod light;
