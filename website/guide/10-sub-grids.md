---
title: Sub-grids
feature: sub_grids
api:
  - SubGrid
  - Grid
  - GridVoxels
  - GridVoxelsMut
  - GridRotation
  - StampVoxel
  - StampPreview
  - Stamped
  - StampError
  - incoming_replaces
  - Lifted
  - LiftError
---

A sub-grid is a set of voxels on a grid of its own, with its own transform, that moves freely against the world's grid: a ship, a vehicle, a falling piece of rock. It sits behind the opt-in `sub_grids` feature.

```bash
$ cargo add bevy_voxel_world --features sub_grids
```

Sub-grids can be created either from scratch, by creating an empty sub-grid and editing its voxels, or by 'lifting' a portion of the world out in to a sub-grid. En example of the latter would be an overhang being broken of a cliff, or a tree branching being severed and falling to the ground (see the `light_mountain_house` example)

In a lit world, sug-grids are lit by their own lights, the sky, and lights from the regular world. However, lights in sub-grids does not light the regular world or other sub-grids. Regular Bevy lights can be added to achive that instead.

Main ways to interact with sub-grids are:

- `VoxelWorld::spawn_sub_grid` creaes an empty sub-grid.
- `VoxelWorld::grid` and `VoxelWorld::grid_mut` returns an access object with `set_voxel` methods, so the grid can be edited the same way as the regular world.
- `VoxelWorld::lift` grabs a range of voxels from the world and make a sub-grid from them, leaving air behind in the parent world.
- `VoxelWorld::stamp` writes a sub-grid's voxels back into any other grid at the nearest pose that lines up with it.
- `VoxelWorld::raycast_grids` casts a ray through every grid, including the regular world.

A sub-grid has no generator, so every voxel it holds is a voxel edit. It has chunks of its own, built by the world's chunk tasks, mesher and material, and its chunks are children of its entity, so they move with it.

The [`sub_grids` example](https://github.com/splashdust/bevy_voxel_world/blob/main/examples/sub_grids.rs) walks through and demos the feature.

```bash
$ cargo run -r --example sub_grids --features sub_grids,dislodged_fragments,light_propagation
```
