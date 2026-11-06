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

**Coarse chunk**:
A chunk above LOD level 0. Its light is plausible rather than exact, and changes only when its chunk task reruns.

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

### Light

**Light propagation**:
How light spreads from its sources through a world's non-solid voxels, so that a voxel with no path to a source is pitch black, however close it is to the surface. A world opts in to it.
_Avoid_: occlusion, lighting (which also covers Bevy's lights)

**Light level**:
How much light has reached a voxel through light propagation, on a scale from 0 (pitch black) to 15, whatever the world's light precision. A voxel has one for sky light and one for voxel light.
_Avoid_: lightness, brightness

**Sky light**:
Light that enters a world from the open sky, falling straight down. It doesn't depend on where the sun is, so the time of day changes how bright it is without moving it.
_Avoid_: sunlight (the world's directional light, which is a separate thing)

**Direct sky**:
What a voxel has when sky light reaches it at full strength: when it falls straight down onto it with no solid in the way, or, where a world uses sky openness, when the voxel sees most of the sky.
_Avoid_: skylit, exposed

**Sky fill**:
Sky light that goes beyond direct sky into open space, so the ground under something high stays lit while a narrow opening lets none of it through. A world chooses how it's worked out: by the best path from the sky through openings, or by sky openness.
_Avoid_: ambient fill, bounce light

**Sky openness**:
How much of the sky a place sees, from none of it to all of it, as one way of working out sky fill. A place that sees most of the sky has direct sky.
_Avoid_: sky view factor, ambient occlusion

**Sky ceiling**:
One height a world declares, above which nothing casts a shadow. A voxel with no solid above it up to the sky ceiling has direct sky.
_Avoid_: world top, height limit

**Sky clearance**:
How many chunks straight above a chunk must hold no solid in a column for sky light to enter that column. It's for worlds with no known top, and a sealed space taller than this lights up on its floor.
_Avoid_: sky height

**Falloff**:
How a light level becomes a brightness when rendered. It's part of how a world chooses to look, and level 0 stays pitch black.
_Avoid_: attenuation (a light level dropping as it spreads, which falloff doesn't change)

**Light reach**:
How many voxels light spreads from a source before its light level reaches zero. Sky light and voxel light each have their own, and a world chooses them. It doesn't apply to direct sky, which falls any distance.
_Avoid_: radius, range

**Light precision**:
How finely a channel divides the scale of light levels, which a world chooses. 16 steps holds whole levels and 256 steps holds fractions of one. A format is named by what it holds: 4+4 is 16 steps each for sky light and voxel light, and 8+8 is 256 each.
_Avoid_: bit depth, light quality

**Provisional light**:
Light a chunk shows before everything that lights it is known: whether sky reaches its columns, or the chunks beside it. It can change later with no voxel changing.
_Avoid_: temporary light, fallback light

**Sky probe**:
Generating the voxels of a chunk position only to learn which of its columns hold a solid. No chunk spawns from it.
_Avoid_: ghost chunk, hidden chunk

**Sky record**:
What a world remembers about which columns hold a solid at the chunk positions it has seen. Forgetting any of it costs only time, never different light.
_Avoid_: heightmap, top map

**Relight**:
Working out a chunk's light again after something it depends on changed, without regenerating its voxels. The chunk stays visible throughout.
_Avoid_: light update, re-solve

**Relight ripple**:
A relight that starts at the chunks whose own voxels or direct sky changed, and goes on to a neighbour only where light reached their shared face differently.
_Avoid_: wave, cascade, flood

**Occupancy**:
Which voxels of a chunk are solid, and so block light, and nothing else about them.
_Avoid_: solid mask, opacity

**Solver**:
The one way a world works out a chunk's light: from the occupancy, direct sky, emissive voxels and sky fill within light reach of the chunk, with no light from neighbouring chunks, so the order chunks arrive in can't change it. One run of it is a solve, whether a chunk's first or a relight.
_Avoid_: propagator, light engine

**Kernel**:
The part of the solver that spreads light from its sources through open voxels, for one channel at a time. Sky light and voxel light, first solves and relights all use the same one.
_Avoid_: solver (which also gathers the kernel's inputs), flood fill

**Voxel light**:
Light emitted by emissive voxels.
_Avoid_: block light, artificial light

**Emissive voxel**:
A voxel whose material a world declares to be a source of voxel light.
_Avoid_: light block, torch

**Slot**:
Room for one chunk's light in the light texture of its world: a light level for each of the chunk's voxels and for each voxel just past its faces. A world has a fixed number of slots, and a chunk takes one only when its faces see more than one light level in a channel.
_Avoid_: page, atlas entry

**Tag**:
The value a chunk's mesh carries to tell the shader where its light is: which slot holds it or, for a chunk with no slot, the one light level per channel that its faces all see. Above LOD level 0 it also holds the chunk's LOD level.
_Avoid_: mesh tag (the Bevy component that carries it), light index
