# bevy_voxel_world

A Bevy plugin that streams a chunked voxel world around one or more cameras, generating, meshing and despawning chunks as they come into and out of range.

## Language

### Chunks

**Chunk**:
A cubic region of the voxel world, addressed by its chunk key, that is generated, meshed and despawned as a unit. A chunk above LOD level 0 spans twice the width of a chunk one level below it and contains eight of them.

**Chunk key**:
The address of a chunk: its position and its LOD level. Two chunks at different LOD levels can share a position.
_Avoid_: chunk position (only half of the address), chunk id

**LOD level**:
How coarsely a chunk samples the voxel world. Level 0 samples every voxel; each level above it samples half as often along each axis and spans twice the width.
_Avoid_: stride, resolution, detail level

**Chunk lifecycle**:
The sequence of states a chunk moves through from being requested to being despawned, and the rules for moving between them.
_Avoid_: chunk pipeline, remesh flow

**Chunk state**:
Where a single chunk currently is in the chunk lifecycle; a chunk is in exactly one chunk state at a time.
_Avoid_: chunk flags, chunk markers

**Pending**:
The chunk state of a chunk that exists in the world but is waiting for a chunk task to start.

**Generating**:
The chunk state of a chunk whose chunk task is running.

**Ready**:
The chunk state of a chunk whose latest chunk task has been published; a Ready chunk may have no mesh (empty, solid, or nothing visible).

**Retiring**:
The chunk state of a chunk that has been chosen for despawning and will be removed on the next frame.
_Avoid_: despawning, dead

**Chunk task**:
The off-main-thread work that generates a chunk's voxel data and builds its mesh.
_Avoid_: chunk thread

**Superseded task**:
A chunk task whose inputs went stale while it ran, because of a voxel edit; it still finishes and is published, and a fresh chunk task follows it.
_Avoid_: cancelled task

**Publish**:
To make a finished chunk task's voxel data readable through the voxel world, independent of whether a mesh is ever shown.

### Voxels

**Voxel edit**:
A voxel set through the voxel world. It replaces the generated voxel at its position for as long as the world exists, across chunk despawns and respawns.
_Avoid_: modified voxel, voxel write

### Streaming

**Streaming policy**:
The rules that decide, from the cameras' positions and views, which chunks should exist, at which LOD level, and which should be retired.
_Avoid_: spawn policy, chunk spawner

**Protected radius**:
The distance around each camera within which chunks are spawned and never retired, whether or not they are in view.
_Avoid_: min despawn distance

**Split**:
Replacing a chunk with the eight chunks one LOD level below it. They spawn first; the chunk retires once they are Ready.
_Avoid_: refine, subdivide

**Merge**:
Replacing eight chunks with the chunk one LOD level above them. It spawns first; they retire once it is Ready.
_Avoid_: coarsen, collapse

**Fill hint**:
A world's declaration that a chunk is uniformly one voxel, or that it doesn't know. An optional accelerator for the streaming policy; the world works fully without it.
_Avoid_: air hint, solid hint
