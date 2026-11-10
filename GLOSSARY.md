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
A chunk above LOD level 0. Its light is plausible rather than exact, and changes only when its chunk task reruns. A level-0 chunk with a non-default `chunk_data_shape` takes the same light path.

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
A chunk task whose inputs went stale while it ran, because of a voxel edit or a clear; it still finishes and is published, and a fresh chunk task follows it.
_Avoid_: cancelled task

**Regenerate**:
To run a chunk's task again from the generator alone, keeping none of the chunk's previous voxel data. A chunk regenerates when its voxel edits are cleared or forgotten, or when the world is told its generator changed.
_Avoid_: invalidate, refresh, rebuild (meshing a chunk again from the voxels it has)

**Publish**:
To make a finished chunk task's voxel data readable through the voxel world, independent of whether a mesh is ever shown.

### Voxels

**Voxel edit**:
A voxel set through the voxel world. It replaces the generated voxel at its position until it is cleared or forgotten, or the world is dropped, across chunk despawns and respawns.
_Avoid_: modified voxel, voxel write

**Edit delta**:
The voxel edits of one level-0 chunk that changed in one commit, with their new values and the chunk's edit sequence after it. The unit in which voxel edits travel between worlds and to disk.
_Avoid_: edit event, change set (the crate-internal commit output it is built from), snapshot

**Full delta**:
An edit delta carrying every voxel edit a chunk currently has. Applying it replaces the chunk's voxel edits with the ones it carries, so an empty one clears the chunk.
_Avoid_: chunk snapshot (a chunk's voxels, not its edits), bucket dump

**Edit sequence**:
A chunk's count of how many of its voxel edits have ever changed in a commit. It only moves forward, and an edit delta states the chunk's edit sequence after it, so a receiver can tell whether the delta continues from what it has.
_Avoid_: edit counter, version, revision

**Clear**:
Removing a voxel edit, so its position shows the generated voxel again. A chunk's clear reaches other worlds as a full delta, so the chunk's edit sequence keeps moving forward.
_Avoid_: revert, delete edit, reset (the light's operation)

**Forget**:
A world dropping a chunk's voxel edits together with its edit sequence, as if it had never received them, so the next edit delta it receives reads as a gap. The receiver's way to drop edits before being sent them again; never a server's decision about the world.
_Avoid_: clear (which keeps the edit sequence), prune, evict

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

**Paths**:
The default way of working out sky fill: a level for each connected open region of 8-voxel cells, passed between regions through the openings between them, losing more through an opening that fills less of the region behind it.
_Avoid_: flood fill (the kernel's spreading, which is per voxel)

**Sky fill field**:
What a world keeps to work out sky fill: a level for each open 8-voxel cell of the chunk positions it has seen, kept current on the main thread under the world's model of sky fill. It counts against the sky record's cap.
_Avoid_: fill map, sky field

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

**Fill probe**:
A sky probe that sky fill asks for, at the lowest priority, at a chunk position whose voxels could still raise a level in a chunk that takes sky fill.
_Avoid_: fill lookup

**Sky record**:
What a world remembers about which columns hold a solid at the chunk positions it has seen. Forgetting any of it costs only time, never different light.
_Avoid_: heightmap, top map

**Hinted air**:
A level-0 chunk position that a fill hint of air keeps from spawning. Light reads it as open voxels with its own direct sky, not as a neighbour that isn't loaded.
_Avoid_: empty chunk, skipped chunk

**Light entry**:
Light a world keeps for a chunk position that a fill hint keeps from spawning as air, where a loaded chunk's solid touches it, so the faces beside it read the right light. It has no voxels, entity or mesh, and goes once no solid touches it.
_Avoid_: ghost chunk, phantom chunk

**Relight**:
Working out a chunk's light again after something it depends on changed, without regenerating its voxels. The chunk stays visible throughout.
_Avoid_: light update, re-solve

**Relight ripple**:
A relight that starts at the chunks whose own voxels or direct sky changed, and goes on to a neighbour only where light reached their shared face differently.
_Avoid_: wave, cascade, flood

**Settings relight**:
The relight a change of light settings starts: every chunk and light entry with an open voxel touching a solid, nearest the camera first, after chunk tasks. Old light shows until the new light lands.
_Avoid_: full relight, rebuild

**World view**:
What a lit world shares with the tasks that solve light: each loaded level-0 chunk's light and what its neighbours' solves read of it, its occupancy and emissive voxels. Tasks read it off the main thread, the way chunk tasks read published chunks.
_Avoid_: light map, snapshot

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

**Shell**:
The layer one voxel deep around a chunk's voxels in its slot, holding copies of its face neighbours' border planes, so a face on the chunk's boundary reads the light in front of it. It's only for display and never feeds back into light.
_Avoid_: padding (the voxels a chunk's data holds past its faces)

**Look**:
How light shows on chunk faces: the falloffs, the sun gate, smoothing, the glow of emissive voxels, ambient occlusion and brightness. A change to it shows on the next frame, with no relight.
_Avoid_: style, light settings (which include what relights)

**Sun gate**:
Where Bevy's directional lights shine on a lit world's chunks. By default only on direct sky, because Bevy's shadow map holds only spawned chunks and would let the sun into caves through rock that hasn't spawned.
_Avoid_: sun mask, shadow gate
