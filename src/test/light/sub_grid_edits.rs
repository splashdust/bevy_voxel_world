//! Tests that a sub-grid's voxel edits leave the world's light alone.

use bevy::ecs::system::RunSystemOnce;
use bevy::prelude::*;

use super::edits::{settle, world};
use super::voxel_light::{app, light_at, LampWorld, LAMP, STONE};
use crate::prelude::*;

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

    app.world_mut()
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
        })
        .unwrap();
    settle(&mut app, "light to settle after the sub-grid is filled");

    let after = places.map(|place| light_at(&mut app, place));
    assert_eq!(after, before);
}
