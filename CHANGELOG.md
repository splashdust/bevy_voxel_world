# Changelog

## 0.2.1

- Move voxel data to `ChunkMap` instead of the `Chunk` component. This makes `get_voxel()` much faster, because we don't need to `collect()` all the `Chunk`s to find the correct voxel data.

## 0.2.0

- Rewrite spawning system\
   The old system would just spawn a cubic volume of chunks with a fixed height of 2 chunks. The new system spawns a spherical volume instead, and uses a combination of ray casting and flood fill to minimize the amount of work needed to find unspawned chunks within the volume.
- Add `ChunkDespawnStrategy` config option\
   Configure wheter chunks should despawned when not in view, or only when outside of `spawning_distance`

## 0.1.0

Initial release
