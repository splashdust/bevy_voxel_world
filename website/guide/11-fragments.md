---
title: Dislodged fragments
feature: dislodged_fragments
api:
  - FragmentDetection
  - Connectivity
  - FragmentDislodged
---

Another off-by-default feature called `dislodged_fragments` finds the voxels an edit cuts off from the rest of their grid. A world opts in by returning `FragmentDetection` settings from `VoxelWorldConfig::detect_dislodged_fragments`.

When a dislodged fragment is detected, a `FragmentDislodged<C>` message is sent, with its bounding box, its voxel count and `contains`. The largest piece of a cut keeps its place (typically that's the world itself).

When sub-grids are also enabled, `FragmentDetection::auto_lift` can be set to automatically turn dislodged fragments into sub-grids.

The [`sub_grids` example](https://github.com/splashdust/bevy_voxel_world/blob/main/examples/sub_grids.rs) lifts each piece you cut off a pillar or a rod, and the [`light_mountain_house` example](https://github.com/splashdust/bevy_voxel_world/blob/main/examples/light_mountain_house.rs) fells a tree when you cut through its trunk.

```bash
$ cargo run -r --example sub_grids --features sub_grids,dislodged_fragments,light_propagation
$ cargo run -r --example light_mountain_house --features light_propagation,sub_grids,dislodged_fragments
```
