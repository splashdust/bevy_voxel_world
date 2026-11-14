//! Tests that another grid's chunks and voxel edits stay out of the world's
//! grid.

use bevy::ecs::system::RunSystemOnce;
use bevy::prelude::*;

use super::{
    chunk_data, chunk_status, spawn_camera, test_plugins, update_until, FloorWorld,
};
use crate::chunk_map::ChunkMap;
use crate::grid::{GridId, GridKey};
use crate::prelude::*;
use crate::voxel_edits::VoxelEdits;

const OTHER: GridId = GridId::new(1);

/// A voxel the world shows as air, and the other grid holds as solid.
const OTHER_EDIT: IVec3 = IVec3::new(0, 5, 0);

fn edits(app: &App) -> &VoxelEdits<FloorWorld, u8> {
    app.world().resource::<VoxelEdits<FloorWorld, u8>>()
}

/// The other grid's committed edit at [`OTHER_EDIT`], if it has one.
fn other_edit(app: &App) -> Option<WorldVoxel> {
    edits(app).reader_in(OTHER).get(OTHER_EDIT)
}

#[test]
fn another_grids_edits_and_chunks_never_show_through_the_worlds_methods() {
    let mut app = App::new();
    app.add_plugins((test_plugins(), VoxelWorldPlugin::<FloorWorld>::minimal()))
        .add_systems(Startup, spawn_camera::<FloorWorld>);
    update_until(
        &mut app,
        "the chunks at and below the origin to be ready",
        |app| {
            [IVec3::ZERO, IVec3::NEG_Y].into_iter().all(|chunk| {
                chunk_status::<FloorWorld>(app, chunk) == Some(ChunkStatus::Ready)
            })
        },
    );

    // The other grid's chunk at the origin is solid, where the world's is air.
    let below = ChunkKey::new(IVec3::NEG_Y, 0);
    let solid = chunk_data::<FloorWorld>(&mut app, below.position).unwrap();
    let origin = ChunkKey::new(IVec3::ZERO, 0);
    app.world()
        .resource::<ChunkMap<FloorWorld, u8>>()
        .get_map()
        .write()
        .unwrap()
        .insert(
            GridKey::new(OTHER, origin),
            crate::chunk::ChunkData {
                key: origin,
                ..solid
            },
        );
    edits(&app).set_in(OTHER, OTHER_EDIT, WorldVoxel::Solid(2));
    edits(&app).commit_in(OTHER);
    app.update();

    let (voxel, edited, delta, down, across) = app
        .world_mut()
        .run_system_once(|voxel_world: VoxelWorld<FloorWorld>| {
            let cast = |origin, direction| {
                voxel_world
                    .raycast(Ray3d { origin, direction }, &|_| true)
                    .map(|hit| hit.position)
            };
            (
                voxel_world.get_voxel(OTHER_EDIT),
                voxel_world.edited_chunks(),
                voxel_world.full_delta(IVec3::ZERO),
                cast(Vec3::new(1.5, 20.0, 1.5), -Dir3::Y),
                cast(Vec3::new(0.5, 5.5, 20.0), -Dir3::Z),
            )
        })
        .unwrap();
    assert_eq!(voxel, WorldVoxel::Air);
    assert_eq!(edited, vec![]);
    assert!(delta.is_empty() && delta.seq == 0);
    assert_eq!(
        down,
        Some(Vec3::new(1.0, -1.0, 1.0)),
        "hits the world's floor"
    );
    assert_eq!(across, None, "passes the other grid's edit");

    app.world_mut()
        .run_system_once(|mut voxel_world: VoxelWorld<FloorWorld>| {
            voxel_world.clear_edits();
        })
        .unwrap();
    app.update();
    assert_eq!(
        other_edit(&app),
        Some(WorldVoxel::Solid(2)),
        "after a clear"
    );

    app.world_mut()
        .run_system_once(|mut voxel_world: VoxelWorld<FloorWorld>| {
            voxel_world.forget_edits();
        })
        .unwrap();
    app.update();
    assert_eq!(
        other_edit(&app),
        Some(WorldVoxel::Solid(2)),
        "after a forget"
    );
    assert_eq!(
        edits(&app).edited_chunks_in(OTHER),
        vec![(IVec3::ZERO, 1)],
        "the other grid keeps its edit sequence"
    );
}
