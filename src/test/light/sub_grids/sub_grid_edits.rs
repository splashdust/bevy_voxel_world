//! Tests that a sub-grid's voxel edits and chunks leave the world's light
//! alone.

use bevy::ecs::system::RunSystemOnce;
use bevy::prelude::*;

use crate::prelude::*;
use crate::test::light::edits::{settle, world};
use crate::test::light::voxel_light::{app, light_at, LampWorld, LAMP, STONE};
use crate::test::update_until;

#[test]
fn filling_a_sub_grid_inside_loaded_world_chunks_leaves_the_worlds_light_alone() {
    let mut app = app(world());
    settle(&mut app, "the cavern to settle");
    // In the cavern's air, under its shaft and its roof hole, in its stone,
    // and in the open air above it.
    let places = [
        IVec3::new(2, 5, 2),
        IVec3::new(8, 10, 8),
        IVec3::new(21, 8, 21),
        IVec3::new(2, 2, 2),
        IVec3::new(5, 44, 5),
    ];
    let before = places.map(|place| light_at(&mut app, place));
    assert!(before.iter().all(Option::is_some), "{before:?}");

    let entity = app
        .world_mut()
        .run_system_once(move |mut voxel_world: VoxelWorld<LampWorld>| {
            let entity = voxel_world.spawn_sub_grid(Transform::default());
            let mut grid = voxel_world.grid_mut(Grid::SubGrid(entity)).unwrap();
            for (i, place) in places.into_iter().enumerate() {
                let material = if i % 2 == 0 { LAMP } else { STONE };
                grid.set_voxel(place, WorldVoxel::Solid(material));
            }
            // A slab roofing the cavern's shaft, and lamps across its floor.
            grid.set_voxels_in(IVec3::new(4, 15, 4), IVec3::new(12, 16, 12), |_| {
                Some(WorldVoxel::Solid(STONE))
            });
            grid.set_voxels_in(IVec3::new(0, 4, 0), IVec3::new(32, 5, 32), |_| {
                Some(WorldVoxel::Solid(LAMP))
            });
            entity
        })
        .unwrap();
    // The sub-grid's chunks share their keys with the world's loaded chunks.
    let keys: Vec<ChunkKey> = places
        .iter()
        .map(|place| ChunkKey::containing(*place, 0))
        .collect();
    update_until(&mut app, "the sub-grid's chunks to be ready", |app| {
        let keys = keys.clone();
        app.world_mut()
            .run_system_once(move |voxel_world: VoxelWorld<LampWorld>| {
                let grid = voxel_world.grid(Grid::SubGrid(entity)).unwrap();
                keys.iter().all(|key| {
                    grid.chunk_state(*key) == Some(ChunkStatus::Ready)
                        && voxel_world.chunk_state(*key) == Some(ChunkStatus::Ready)
                })
            })
            .unwrap()
    });
    settle(&mut app, "light to settle after the sub-grid is filled");

    let after = places.map(|place| light_at(&mut app, place));
    assert_eq!(after, before);

    // Moved, turned and edited, with lamps of its own lit, it still leaves
    // the world's light alone.
    app.world_mut().entity_mut(entity).insert(
        Transform::from_xyz(3.0, 2.0, -4.0).with_rotation(Quat::from_rotation_y(0.7)),
    );
    app.world_mut()
        .run_system_once(move |mut voxel_world: VoxelWorld<LampWorld>| {
            let mut grid = voxel_world.grid_mut(Grid::SubGrid(entity)).unwrap();
            grid.set_voxel(IVec3::new(8, 8, 8), WorldVoxel::Solid(LAMP));
            grid.set_voxel(places[1], WorldVoxel::Air);
        })
        .unwrap();
    settle(
        &mut app,
        "light to settle after the sub-grid moved and was edited",
    );

    let moved = places.map(|place| light_at(&mut app, place));
    assert_eq!(moved, before);
}
