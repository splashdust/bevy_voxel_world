---
title: What each colour candidate costs
labels: [wayfinder:task]
status: closed
assignee: Joacim Magnusson
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

## Resolution

Measured, nothing decided. The numbers, scenes and method are in [colour-costs.md](../baseline/colour-costs.md), with the microbenchmark's [CSV](../baseline/colour_cost.csv) and the lifecycle runs in `../baseline/lifecycle/`. Probes: [colour_cost.rs](../../probes/colour_cost.rs) times every chunk of three scenes with the crate's kernel, and [colour_cost_sim.patch](../../probes/colour_cost_sim.patch) makes main's solver pay what each candidate would, with light kept white, so `chunk_lifecycle` times it on the real pool.

Candidates measured: a run per light colour (**groups**), stored as today's planes where one colour reaches the chunk and otherwise either mixed into three 8-bit components (**mix**) or kept as runs for the shader (**keep**); and **per-component** levels, three runs. Palettes of white, one colour (orange), three and eight.

- **Solve per chunk** (median, reach 14 → 32, 16 levels; 256 within 5%):
  - White, under both: 40–48 µs → 284–332 µs, today's run.
  - Groups: one colour costs what white does. Three colours 106–217 µs → 857–1,119; eight 193–542 → 2,127–3,106. Cost grows with distinct light colours with a lamp in the box, including colours whose light walls stop: in the caves with eight colours, the median chunk runs all eight and one reaches it. Mixing adds 15–189 µs at reach 14 and up to 471 at 32.
  - Per-component: three runs, about 3× white whatever the palette: 123–143 µs → 878–1,015. A box of one colour, such as orange, pays the 3× too; only white and greys run once.
- **Edits** (`edits_lamps`, a probe scenario: `edits_lit` on the lamp world with each edit beside a lamp):
  - White on main already misses the frame gate beside lamps: 4 of 24 edits at reach 14, all at one site and in 4 frames, and 20 of 24 at reach 32, in 3 to 5. Its main thread stays under 1 ms.
  - Per-component lands every edit in the frame white's does. Groups land eight edits a frame later at reach 14 (2 frames against 1, still within the gate), and relight more chunks per edit (9 against 8, and 15 against 10 at reach 32), since a change in any colour's arrival at a face ripples.
  - The main thread's worst frame grows by about 100 µs at 16 levels and 150–200 µs at 256 under both, and stays under 1 ms except at reach 32 and 256 levels: 1,250 µs for groups and 1,178 per-component, against white's 706. It was simulated as per-component's bit planes, so it's an upper bound for groups mixed into bytes.
  - A relight costs 2–3× white's per-component, and 5–8× for eight groups.
- **Streaming** (`caves`, sky light at reach 30 included):
  - At reach 14, per-component and three groups hold both gates: per-component solves a chunk, sky light included, in 419 µs against 288, and settles in 4.6 s against 4.1.
  - Eight groups miss the solves-per-arrival gate in every run, 5.55 against 5.5, at both precisions: relights grow 13% while the ones that changed light stay as white's, so slower solves let more neighbours arrive mid-solve.
  - At reach 32 white misses that gate too (5.58). Eight groups reach 6.39 and settle in 32 s against white's 6.
- **CPU bytes per voxel of voxel light** (sky light stays 0.5 and 1): today 0.5 at 16 levels and 1 at 256. Groups with one colour in the chunk: the same, plus an index per chunk. Groups mixed: 3 at both. Groups kept: 0.5 K and K. Per-component: 1.5 and 3. The share of lamp-lit chunks paying the wider form: groups 35–44% in the caves, 69–74% in many_lamps, 92% in the lamp world; per-component 51–92%.
- **GPU bytes per slot** (34³ texels; today 38.4 KiB at 16 levels and 76.8 at 256): per-component 76.8 and 153.5 KiB; groups mixed 153.5 at both (76.8 at 16 levels with 4-bit components, which round the mix); groups kept 76.8 / 153.5 KiB for up to 3 runs, 153.5 / 307 for up to 7. Groups' one-colour chunks keep today's slot.
- **Shell writes on the main thread**, per slot: today 29–59 µs. Groups mixed, kept as a byte per voxel: 35–44. Per-component kept as bit planes: 44–78 µs at 16 levels and 60–181 at 256, up to 3.2× today's. How the world view holds colour sets landing's main-thread cost.
- **Tags of chunks with no slot**: a colour index fits at both precisions (20 bits free at 16 levels, 12 at 256, not counting bits 28–30). Three components fit at 16 levels (3 × 4 bits) but not at 256 (24 bits against 20, or 23 with bits 28–30): 6-bit components or an index would. Kept runs don't fit and must show one colour. Slotless coloured chunks are mostly evicted chunks and coarse chunks that gave their slot away, which show one level today.
- **White world's `chunk_lifecycle` A/B against main: would hold under both**, provided the colour path is skipped by a per-world or per-chunk flag, not found box by box: a white box runs once under both and stores, packs, tags and lands exactly as today. Grouping lamps by colour box by box, done naively, costs 0.2–7 µs a box, up to 5% of a white solve in many_lamps. An `EmissiveVoxel` can carry a 13-bit colour index in its 4 bytes. The shader stays today's only if the colour path is a shader def. No A/B was run: with the flag, a white world runs main's code.
- **Found on the way:**
  - **White beside lamps already misses gates the Notes hold a coloured world to.** The edit gate and `caves`' were set on a world without lamps and at reach 14. Beside lamps white misses the frame gate, and at reach 32 the solves-per-arrival gate too. A coloured gate needs white's own numbers in the same scenario as its baseline. Why an edit beside one lamp takes 4 frames wasn't looked into.
  - **`stream_in` ends early when solves are slow.** Its 30 quiet frames passed with 446 chunks ready under three groups, so the first edit runs went on a world still streaming. A coloured bench scenario has to stream until light settles, as `caves` does.
  - **Groups pay for colours walls stop.** With eight colours, the median cave chunk runs 8 groups and only 1 colour's light reaches it.
