use bevy::prelude::*;

use crate::chunk_map::ChunkMapUpdateBuffer;
use crate::mesh_cache::MeshCacheInsertBuffer;
use crate::prelude::*;
use crate::{
    chunk::{ChunkData, FillType},
    prelude::VoxelWorldCamera,
    voxel_world::*,
};

fn _test_setup_app() -> App {
    let mut app = App::new();
    app.add_plugins((MinimalPlugins, VoxelWorldPlugin::minimal()));
    app.add_systems(Startup, |mut commands: Commands| {
        commands.spawn((
            Camera3dBundle {
                transform: Transform::from_xyz(10.0, 10.0, 10.0).looking_at(Vec3::ZERO, Vec3::Y),
                ..default()
            },
            VoxelWorldCamera,
        ));
    });

    app
}

#[test]
fn can_set_get_voxels() {
    let mut app = _test_setup_app();

    // Set and get some voxels
    app.add_systems(Update, |mut voxel_world: VoxelWorld| {
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

    app.add_systems(Update, move |mut voxel_world: VoxelWorld| {
        let test_voxel = WorldVoxel::Solid(1);

        for pos in make_pos.clone() {
            voxel_world.set_voxel(pos, test_voxel);
        }
    });

    app.update();

    let check_pos = positions.clone();

    app.add_systems(Update, move |voxel_world: crate::prelude::VoxelWorld| {
        let test_voxel = crate::voxel::WorldVoxel::Solid(1);

        for pos in check_pos.clone() {
            assert_eq!(
                voxel_world.get_surface_voxel_at_2d_pos(Vec2::new(pos.x as f32, pos.z as f32)),
                Some((pos, test_voxel))
            )
        }
    });

    app.update();
}

// ChunkWillSpawn event now fires from the mesh spawning system, which cannot run in tests.
#[ignore]
#[test]
fn chunk_will_spawn_events() {
    let mut app = _test_setup_app();

    app.add_systems(
        Update,
        |mut ev_chunk_will_spawn: EventReader<ChunkWillSpawn>| {
            let spawn_count = ev_chunk_will_spawn.read().count();
            assert!(spawn_count > 0);
        },
    );

    app.update();
}

#[test]
fn chunk_will_remesh_event_after_set_voxel() {
    let mut app = _test_setup_app();

    // Run the world a 100 cycles
    for _ in 0..100 {
        app.update();
    }

    app.add_systems(Update, |mut voxel_world: VoxelWorld| {
        voxel_world.set_voxel(IVec3::new(0, 0, 0), WorldVoxel::Solid(1));
    });

    app.update();

    app.add_systems(
        Update,
        |mut ev_chunk_will_remesh: EventReader<ChunkWillRemesh>| {
            let count = ev_chunk_will_remesh.read().count();
            assert_eq!(count, 1)
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
        |mut query: Query<&mut GlobalTransform, With<VoxelWorldCamera>>| {
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
        |mut ev_chunk_will_despawn: EventReader<ChunkWillDespawn>| {
            let count = ev_chunk_will_despawn.read().count();
            assert!(count > 0)
        },
    );

    app.update();
}

#[test]
fn raycast_finds_voxel() {
    let mut app = _test_setup_app();

    // Set up vector och positions to test
    let positions = vec![IVec3::new(0, 0, -1), IVec3::new(0, 0, 0)];

    let make_pos = positions.clone();

    app.add_systems(
        Startup,
        move |mut voxel_world: VoxelWorld,
              buffers: (ResMut<ChunkMapUpdateBuffer>, ResMut<MeshCacheInsertBuffer>)| {
            let test_voxel = crate::voxel::WorldVoxel::Solid(1);

            for pos in make_pos.clone() {
                voxel_world.set_voxel(pos, test_voxel);
            }

            let (mut chunk_map_update_buffer, _) = buffers;

            chunk_map_update_buffer.push((
                IVec3::new(0, 0, 0),
                ChunkData {
                    position: IVec3::new(0, 0, 0),
                    voxels: Some(std::sync::Arc::new([WorldVoxel::Unset; 39304])),
                    voxels_hash: 0,
                    is_full: false,
                    is_empty: false,
                    fill_type: FillType::Mixed,
                    entity: Entity::PLACEHOLDER,
                },
                ChunkWillSpawn {
                    chunk_key: IVec3::new(0, 0, 0),
                    entity: Entity::PLACEHOLDER,
                },
            ));
        },
    );

    app.update();

    app.add_systems(Update, move |voxel_world_raycast: VoxelWorldRaycast| {
        let test_voxel = crate::voxel::WorldVoxel::Solid(1);

        let ray = Ray3d {
            origin: Vec3::new(0.5, 0.5, 70.0),
            direction: -Direction3d::Z,
        };

        let Some(result) = voxel_world_raycast.raycast(ray, &|(_pos, _vox)| true) else {
            panic!("No voxel found")
        };

        assert_eq!(
            result,
            VoxelRaycastResult {
                position: Vec3::ZERO,
                normal: Vec3::new(0.0, 0.0, 1.0),
                voxel: test_voxel
            }
        )
    });

    app.update();
}
