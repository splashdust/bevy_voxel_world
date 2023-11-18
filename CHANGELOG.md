# Changelog

## 0.3.3

- Defer `ChunkWillSpawn` event until buffers are applied. Fixes issues caused by the event getting fired before the chunk data can actually be looked up.
- Fix texture bleeding issue on negative Y faces (#10)

## 0.3.2

- Performance improvements:
  - More granular resources for more parallelization opportunities.
  - Pre-calculate hash for voxels array for faster mesh cache lookups
  - Buffered iserts/updates and removes for ChunkMap, to reduce time spent waiting to aquire RwLock.

## 0.3.1

- Add lookup map for mesh handles. This allows `bevy_voxel_world` to re-use mesh handles for identical chunks and thereby utilising Bevy's automatic instancing while also avoiding redundant meshing.
- Change the default chunk discovery algorith to only use ray cating. This uses less CPU than the previous flood fill method, and also works with larger spawn distances. The flood fill method can still be used by setting `ChunkSpawnStrategy::Close`
- Add various config options for tuning spawning behaviour for different needs.

## 0.3.0

- Update to Bevy 0.12

## 0.2.2

- Fix an issue where filled underground chunks would never get meshed, even if they were modified
- `ChunkWillSpawn` event now only fire when actually meshed chunks spawn, preventing massive spam of this event.

## 0.2.1

- Move voxel data to `ChunkMap` instead of the `Chunk` component. This makes `get_voxel()` much faster, because we don't need to `collect()` all the `Chunk`s to find the correct voxel data.

## 0.2.0

- Rewrite spawning system\
   The old system would just spawn a cubic volume of chunks with a fixed height of 2 chunks. The new system spawns a spherical volume instead, and uses a combination of ray casting and flood fill to minimize the amount of work needed to find unspawned chunks within the volume.
- Add `ChunkDespawnStrategy` config option\
   Configure wheter chunks should despawned when not in view, or only when outside of `spawning_distance`

## 0.1.0

Initial release
