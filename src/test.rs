use bevy::prelude::*;
use std::sync::Arc;

use crate::chunk_map::ChunkMapUpdateBuffer;
use crate::mesh_cache::MeshCacheInsertBuffer;
use crate::prelude::*;
use crate::voxel_traversal::voxel_line_traversal;
use crate::{
    chunk::{ChunkData, ChunkTask, FillType, PADDED_CHUNK_SIZE},
    prelude::VoxelWorldCamera,
    voxel_world_internal::ModifiedVoxels,
    voxel_world::*,
};
use ndshape::{RuntimeShape, Shape};

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

    app.update();

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

// ChunkWillSpawn event now fires from the mesh spawning system, which cannot run in tests.
#[ignore]
#[test]
fn chunk_will_spawn_events() {
    let mut app = _test_setup_app();

    app.add_systems(
        Update,
        |mut ev_chunk_will_spawn: MessageReader<ChunkWillSpawn<DefaultWorld>>| {
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
fn chunk_generate_reuses_previous_data_when_configured() {
    type Mat = <DefaultWorld as VoxelWorldConfig>::MaterialIndex;

    let entity = Entity::from_raw_u32(1).unwrap();
    let position = IVec3::ZERO;
    let data_shape = UVec3::splat(PADDED_CHUNK_SIZE);
    let mesh_shape = data_shape;

    let data_shape_runtime = RuntimeShape::<u32, 3>::new(data_shape.to_array());
    let mut voxel_payload =
        vec![WorldVoxel::Solid(5); data_shape_runtime.size() as usize];
    let highlighted_block = [1, 1, 1];
    let highlighted_world_pos = IVec3::new(0, 0, 0);
    voxel_payload[data_shape_runtime.linearize(highlighted_block) as usize] =
        WorldVoxel::Solid(9);
    let voxel_payload: Arc<[WorldVoxel<Mat>]> = Arc::from(voxel_payload);

    let modified_voxels = ModifiedVoxels::<DefaultWorld, Mat>::default();
    let mut chunk_task = ChunkTask::<DefaultWorld, Mat>::new(
        entity,
        position,
        0,
        data_shape,
        mesh_shape,
        modified_voxels,
    );

    let mut previous_data = ChunkData::<Mat>::with_entity(entity);
    previous_data.position = position;
    previous_data.data_shape = data_shape;
    previous_data.mesh_shape = mesh_shape;
    previous_data.has_generated = true;
    previous_data.is_full = true;
    previous_data.is_empty = false;
    previous_data.fill_type = FillType::Mixed;
    previous_data.voxels = Some(voxel_payload.clone());
    previous_data.generate_hash();

    chunk_task.generate(
        |_pos, _previous| -> WorldVoxel<Mat> {
            panic!("voxel lookup should not run when reusing previous data");
        },
        Some(previous_data),
        ChunkRegenerateStrategy::Reuse,
    );

    assert!(chunk_task.chunk_data.voxels.is_some());
    assert!(chunk_task.is_full());
    match chunk_task.chunk_data.fill_type {
        FillType::Mixed => {}
        ref other => panic!("expected mixed fill type, got {:?}", other),
    }

    let voxel = chunk_task
        .chunk_data
        .get_voxel_at_world_position(highlighted_world_pos)
        .expect("voxel should be addressable");
    assert_eq!(voxel, WorldVoxel::Solid(9));
}

#[test]
fn raycast_finds_voxel() {
    let mut app = _test_setup_app();

    // Set up vector och positions to test
    let positions = vec![IVec3::new(0, 0, -1), IVec3::new(0, 0, 0)];

    let make_pos = positions.clone();

    app.add_systems(
        Startup,
        move |mut voxel_world: VoxelWorld<DefaultWorld>,
              buffers: (
            ResMut<
                ChunkMapUpdateBuffer<
                    DefaultWorld,
                    <DefaultWorld as VoxelWorldConfig>::MaterialIndex,
                >,
            >,
            ResMut<MeshCacheInsertBuffer<DefaultWorld>>,
        )| {
            let test_voxel = crate::voxel::WorldVoxel::Solid(1);

            for pos in make_pos.clone() {
                voxel_world.set_voxel(pos, test_voxel);
            }

            let (mut chunk_map_update_buffer, _) = buffers;

            chunk_map_update_buffer.push((
                IVec3::new(0, 0, 0),
                ChunkData {
                    position: IVec3::new(0, 0, 0),
                    lod_level: 0,
                    voxels: Some(std::sync::Arc::new([WorldVoxel::Unset; 39304])),
                    voxels_hash: 0,
                    is_full: false,
                    is_empty: false,
                    fill_type: FillType::Mixed,
                    entity: Entity::PLACEHOLDER,
                    has_generated: false,
                    data_shape: UVec3::splat(PADDED_CHUNK_SIZE),
                    mesh_shape: UVec3::splat(PADDED_CHUNK_SIZE),
                },
                ChunkWillSpawn::<DefaultWorld>::new(
                    IVec3::new(0, 0, 0),
                    Entity::PLACEHOLDER,
                ),
            ));
        },
    );

    app.update();

    app.add_systems(Update, move |voxel_world: VoxelWorld<DefaultWorld>| {
        let test_voxel = crate::voxel::WorldVoxel::Solid(1);

        let ray = Ray3d {
            origin: Vec3::new(0.5, 0.5, 70.0),
            direction: -Dir3::Z,
        };

        let Some(result) = voxel_world.raycast(ray, &|(_pos, _vox)| true) else {
            panic!("No voxel found")
        };

        assert_eq!(
            result,
            VoxelRaycastResult {
                position: Vec3::ZERO,
                normal: Some(Vec3::new(0.0, 0.0, 1.0)),
                voxel: test_voxel,
            }
        )
    });

    app.update();
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
fn can_get_chunk_data() {
    let mut app = _test_setup_app();

    app.add_systems(Update, |mut voxel_world: VoxelWorld<DefaultWorld>| {
        voxel_world.set_voxel(IVec3::new(0, 0, 0), WorldVoxel::Solid(1));
    });

    app.update();

    app.add_systems(Update, |voxel_world: VoxelWorld<DefaultWorld>| {
        let chunk_data = voxel_world.get_chunk_data(IVec3::ZERO);
        assert!(chunk_data.is_some());
    });

    app.update();
}
