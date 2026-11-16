---
title: Generating voxels
api:
  - VoxelLookupDelegate
  - VoxelLookupFn
  - FillHint
---

A world asks its config for a voxel lookup function per chunk, and calls it for each voxel the chunk holds. A fill hint can say what a chunk holds before it is generated.
