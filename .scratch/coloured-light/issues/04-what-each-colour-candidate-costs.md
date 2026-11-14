---
title: What each colour candidate costs
labels: [wayfinder:task]
status: open
assignee:
blocked_by: [01, 02]
---

## Question

What does each candidate representation from [Coloured voxel light in other engines](01-coloured-voxel-light-in-other-engines.md) and [Cheap multi-component propagation and storage](02-cheap-multi-component-propagation-and-storage.md) cost over white light?

[How coloured light looks](03-how-coloured-light-looks.md) left two candidates to measure. One is a run per light colour, mixed by the max per component, which is the default look. The other is per-component levels, which stay in play only for tinted translucency. Sum and the blends are out, so leave them unmeasured.

Measure on the cave reference scene, the bench's worlds and a scene of many lamps. For each candidate, at 16 and 256 levels and at voxel reach 14 and 32, record:

- solve time per chunk for a light box holding only white lamps, one light colour, and several light colours;
- an edit's relight and landing against the gate of 2 frames and at most 1 ms of the main thread per frame;
- CPU bytes per voxel in the world view;
- GPU bytes per slot;
- whether a chunk with no slot can still carry its colour in its tag.

Check, for each, whether a white world's `chunk_lifecycle` A/B against main would hold.

This is a task: there's nothing to decide, only numbers the representation's decision waits on. Make the probe worktree of main described in the map's Notes if it doesn't exist yet. Numbers go in `baseline/`, probe code in `.scratch/probes/`.
