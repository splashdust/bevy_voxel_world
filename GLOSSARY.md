# bevy_voxel_world

A Bevy plugin that streams a chunked voxel world around one or more cameras, generating, meshing and despawning chunks as they come into and out of range.

## Language

### Chunks

**Chunk**:
A cubic region of the voxel world, addressed by its chunk position, that is generated, meshed and despawned as a unit.

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
A chunk task whose inputs went stale while it ran, because of a voxel edit or LOD change; it still finishes and is published, and a fresh chunk task follows it.
_Avoid_: cancelled task

**Publish**:
To make a finished chunk task's voxel data readable through the voxel world, independent of whether a mesh is ever shown.
