# What each colour candidate costs: measurements

For [What each colour candidate costs](../issues/04-what-each-colour-candidate-costs.md). Measured on 2026-11-14 on a 16-core Linux desktop (Fedora, kernel 7.2), load average 5–7 from other work, against main at 87a8748 (Release 0.19.3) in the probe worktree, bench profile (one codegen unit). Numbers here are for comparing candidates on one machine; they never go in public docs.

## The candidates as measured

- **White**: one run of the kernel with every lamp, as today.
- **Groups**: a run per light colour with a lamp in the box (the default look). Where one run reaches the chunk, it keeps today's planes and a colour index. Where two or more do, two ways to store them:
  - **mix**: on the CPU, each component the max over runs of colour × brightness under the default voxel falloff (black at level 1), stored as three 8-bit components;
  - **keep**: each run's planes as they are, mixed in the shader.
- **Per-component**: a run per component, each lamp seeding a component at the whole level whose brightness is its colour times its own, as the look probe seeded it. Components with the same seeds run once, so white is one run.

Palettes: every lamp white; every lamp orange (1, 0.5, 0.15); three colours; eight colours. A lamp's colour is a hash of its position.

## Scenes

- **caves**: the caves bench's scene, the reference scene's 8×4×8 chunks around `caves`' camera in the deep cavern, with its lamps (one in 32 exposed floor voxels and the torch columns): 2,791 lamps, 178 of 256 chunks with a lamp in reach at reach 14, 233 at 32.
- **lamp_world**: `chunk_lifecycle`'s `LampWorld`, the sine terrain with a lamp on one in 64 surface voxels, 8×2×8 chunks: 1,600 lamps, every chunk with a lamp in reach.
- **many_lamps**: the caves scene with a lamp also on one in 4 exposed floor voxels: 34,119 lamps, 229 of 256 chunks with a lamp in reach at 14, all at 32.

The sine world of `edits_lit`, `stream_in_lit` and the on gate has no lamps, so voxel light never runs there under any candidate.

## Probes

- [colour_cost.rs](../../probes/colour_cost.rs): the microbenchmark. Cuts a kernel box around every chunk of a scene and solves its voxel light with the crate's kernel through `light_bench::KernelBox`, best of 5 per chunk, single-threaded. Built by [colour_cost_build.sh](../../probes/colour_cost_build.sh), which patches two accessors into `light_bench` for the build ([colour_cost_bench.patch](../../probes/colour_cost_bench.patch)). Output: [colour_cost.csv](colour_cost.csv) (p50, p90 and max per scene, reach, precision and palette) and [colour_cost.txt](colour_cost.txt).
- [colour_cost_sim.patch](../../probes/colour_cost_sim.patch): an in-engine cost simulation, built by [colour_cost_sim_build.sh](../../probes/colour_cost_sim_build.sh) and run by [colour_cost_lifecycle.sh](../../probes/colour_cost_lifecycle.sh). See "In the engine".

The probe's slot packer and shell writer for today's format time within 2% of the crate's own `slot_bytes` and `shell_planes` on the lamp world (17.7 against 17.9 µs, 47.4 against 47.4 µs), and within 15% on the caves' shells, so the wider formats are timed by the same method.

## Solving a chunk's voxel light

Per chunk with a lamp in reach, the kernel runs and, for groups, the mixing, in µs: median, with the 90th percentile in brackets. Groups' "one" is a box of orange lamps only. 256 levels are within 5% of 16 levels throughout (see the CSV), so only 16 levels are shown.

| Scene, reach | White | Groups: one | three | eight | Per-component: one | three | eight |
|---|---|---|---|---|---|---|---|
| caves, 14 | 45 (48) | 45 (48) | 106 (188) | 216 (504) | 132 (143) | 131 (143) | 131 (145) |
| caves, 32 | 285 (354) | 285 (352) | 857 (1,078) | 2,178 (2,922) | 898 (1,056) | 900 (1,052) | 896 (1,054) |
| lamp_world, 14 | 48 (48) | 48 (48) | 217 (275) | 542 (686) | 141 (163) | 142 (144) | 143 (159) |
| lamp_world, 32 | 332 (353) | 330 (352) | 1,119 (1,280) | 3,106 (3,450) | 1,015 (1,049) | 989 (1,060) | 984 (1,058) |
| many_lamps, 14 | 40 (50) | 39 (50) | 145 (259) | 193 (471) | 127 (150) | 123 (161) | 125 (166) |
| many_lamps, 32 | 284 (358) | 284 (358) | 912 (1,208) | 2,127 (2,757) | 914 (1,066) | 901 (1,074) | 878 (1,075) |

- **A run costs what a white run costs**, whatever its lamps: a box's groups cost its runs times today's run, and per-component three times it.
- **Groups pay for every light colour with a lamp in the box**, even where walls stop its light short of the chunk. In the caves with eight colours, the median chunk runs 8 groups and only one colour's light reaches it.
- **Mixing** two or more runs into three 8-bit components, with a plain scalar loop and a table per colour, costs 15–35 µs a chunk in the caves and 77–189 µs in the lamp world at reach 14, 160–471 µs at reach 32. It's included in the groups columns. A faster mix (SIMD table lookups) wasn't tried.
- **White costs the same under both**: one run. Grouping a box's lamps by colour, done naively, costs 0.2–0.5 µs a box in the caves and lamp world, and 1.9 µs at reach 14 and 7 µs at 32 in many_lamps, up to 5% of a white solve. A flag that a chunk or world has no coloured lamp makes it free.
- The maxima in the CSV are noisy under the machine's background load; the medians and 90th percentiles held between a trial run and the full run within 2%.

## Chunks where colours mix

The share of chunks with a lamp in reach that would store more than today's planes:

| Scene, reach | Groups, two or more colours reach the chunk (three / eight colours) | More than three colours reach it (eight) | Per-component, components differ (any palette but white) |
|---|---|---|---|
| caves, 14 | 43% / 44% | 39% | 65% |
| caves, 32 | 35% / 35% | 33% | 51% |
| lamp_world, 14 and 32 | 92% / 92% | 92% | 92% |
| many_lamps, 14 | 69% / 69% | 33% | 70% |
| many_lamps, 32 | 74% / 74% | 32% | 75% |

Under groups a box of one light colour stores today's planes; under per-component it doesn't, since orange's components reach different distances. The rest of the chunks are lit by no lamp, or by light that doesn't reach them.

## CPU bytes per voxel in the world view

Voxel light only; sky light stays 0.5 bytes a voxel at 16 levels and 1 at 256 under every candidate.

| Candidate | Per chunk with voxel light, 16 levels | 256 levels |
|---|---|---|
| Today, and groups where one colour reaches the chunk | 0.5 | 1 |
| Groups, mix (three 8-bit components) | 3 | 3 |
| Groups, keep (K runs) | 0.5 K | K |
| Per-component | 1.5 | 3 |

A chunk with one colour also keeps a colour index, a few bytes a chunk. Averaged over the chunks with a lamp in reach, three colours (eight in brackets), 16 levels:

| Scene, reach | Today | Groups, mix | Groups, keep | Per-component |
|---|---|---|---|---|
| caves, 14 | 0.32 | 1.39 (1.42) | 0.73 (1.57) | 0.97 |
| lamp_world, 14 | 0.46 | 2.77 (2.77) | 1.38 (3.59) | 1.38 |
| many_lamps, 14 | 0.35 | 2.09 (2.09) | 1.04 (1.60) | 1.05 |

At 256 levels, today and per-component double and groups' mix doesn't. For reference, `caves` on main holds 47.5 MiB of light on the CPU at its peak, 0.49 bytes of sky and voxel light per voxel of chunks with both solid and open voxels.

## GPU bytes per slot

A slot is 34³ = 39,304 texels. 343 slots fit WebGL2's guaranteed 3D texture side of 256.

| Candidate | 16 levels | 256 levels |
|---|---|---|
| Today, and groups where one colour reaches the chunk | `R8Uint`, 1 B: 38.4 KiB | `Rg8Uint`, 2 B: 76.8 KiB |
| Groups, mix: sky and three 8-bit components | `Rgba8Uint`, 4 B: 153.5 KiB | `Rgba8Uint`, 4 B: 153.5 KiB |
| Groups, mix with 4-bit components | `R16Uint`, 2 B: 76.8 KiB | — |
| Groups, keep, up to 3 runs | `R16Uint`, 2 B: 76.8 KiB | `Rgba8Uint`, 4 B: 153.5 KiB |
| Groups, keep, up to 7 runs | `Rgba8Uint`, 4 B: 153.5 KiB | `Rgba16Uint`, 8 B: 307 KiB |
| Per-component | `R16Uint`, 2 B: 76.8 KiB | `Rgba8Uint`, 4 B: 153.5 KiB |

- Keep also needs each slot's run colours, up to K indices a slot.
- 4-bit components round the mix to whole levels per component, which brings back per-component levels' early fade of a weak component in mixed chunks. The look probe stored them at 12 bits and didn't judge this.
- With a second texture of colour slots, only chunks where colours mix take a wide slot, and a white world makes none.
- `caves` on main peaks at 1,024 slots, 38.4 MiB. If its mixed share of chunks took 4-byte slots instead, that would be about 1,024 × (0.57 × 38.4 + 0.43 × 153.5) KiB ≈ 88 MiB at three colours.

## Packing a slot and writing its shell

Packing runs in the solve, in the task. The shell is written when light lands, on the main thread: six border planes read voxel by voxel from the neighbours' light, then copied in. µs per slot, medians over the chunks that store the format, all scenes and both reaches:

| Format | Pack, 16 levels | Pack, 256 levels | Shell, 16 levels | Shell, 256 levels |
|---|---|---|---|---|
| Today (crate's own) | 18 | 36 | 29–59 | 29–59 |
| Groups, mix (from bytes) | 36–37 | 36–37 | 35–44 | 35–44 |
| Per-component (from planes) | 48–59 | 192–194 | 44–78 | 60–181 |

- How the world view keeps colour sets the main thread's cost. Mixed components kept as a byte per voxel read about as cheaply as today's one channel of bit planes, so their shells cost about what today's do: 35–44 µs against 29–59. Per-component levels kept as bit planes read every bit of three components for each border voxel: up to 3.2× today's shell at 256 levels in the lamp world.
- Per-component packing at 256 levels converts 24 bit planes with the crate's table method; a bit transpose would cost less, but wasn't tried.

## Tags of chunks with no slot

Bits free in a slotless tag: 8–27 at 16 levels (20 bits) and 16–27 at 256 (12 bits), leaving 28–30 clear as the route does today, or 3 more taking those.

| What a slotless chunk must carry | 16 levels | 256 levels |
|---|---|---|
| Groups, one colour: a colour index into a per-world table | Fits: 20 bits | Fits: 12 bits, 4,096 colours |
| Three components: per-component, or groups mixed | Fits: 3 × 4 bits in bits 4–15 | Doesn't fit: 24 bits against 20 (the voxel field's 8 and 12 free), 23 with 28–30. It fits as 6-bit components, or as an index into a per-world table of colours |
| Groups, keep: K levels and K colours | Doesn't fit for K ≥ 2; it has to show one mixed colour, as above | Same |

A slotless chunk seeing coloured voxel light is mostly one that gave its slot away: an evicted chunk shows the most common level its faces see, and a coarse chunk the brightest a tenth of them see. Both would show one colour.

## In the engine

[colour_cost_sim.patch](../../probes/colour_cost_sim.patch) makes main's solver pay what a candidate would, under `COLOUR_SIM`, while light stays white and exact, so `chunk_lifecycle` times it on the real pool with the real relight ripple:

- `groups3` and `groups8`: lamps take one of 3 or 8 colours by a hash of where they sit in the box. A run per colour; where two or more reach the chunk, the mix into three 8-bit components and a 4-byte pack, both thrown away. The white light is the max over the runs, and the face hashes mix every run's, so a change in any colour's arrival ripples, as it would.
- `components`: three runs, every lamp orange (levels 15, 13 and 10), and a pack of three components, thrown away. The white light is the red run.
- Landing reads each lit neighbour's border plane twice, for the two more components. That's per-component's cost kept as bit planes; groups mixed into bytes read about as cheaply as today (see above), so for groups the main thread is an upper bound.

It also adds `edits_lamps`: `edits_lit` on the lamp world, each edit beside a lamp (a block placed on the ground beside it, the lamp dug out, and each undone), with `edits_lit`'s gate. It streams until light settles, as `caves` does: `stream_in`'s rule of 30 quiet frames ends a stream early when solves are slow, which first ran the edits on a world still streaming. `LAMP_REACH` and `LAMP_LEVELS=256` set the lamp world's and the cave world's voxel light. Five runs of `edits_lamps` and three of `caves` per row, each metric the median over the runs; all rows in one round, so compare within a column. Raw output is in [lifecycle/](lifecycle/).

### Edits beside lamps

Frames to land (median, max), edits missing the gate of 2 frames out of 24, the main thread in an edit's worst frame (median and max over the edits), relights per edit (median) and µs per relight:

| Voxel reach, levels | Candidate | Frames | Missed | Main thread µs | Relights | µs per relight |
|---|---|---|---|---|---|---|
| 14, 16 | white | 2, 4 | 4 | 253, 519 | 8 | 190 |
| | groups, 3 | 2, 4 | 4 | 353, 712 | 9 | 549 |
| | groups, 8 | 2, 4 | 4 | 359, 557 | 9 | 1,025 |
| | per-component | 2, 4 | 4 | 347, 811 | 8 | 378 |
| 14, 256 | white | 2, 4 | 4 | 279, 475 | 8 | 181 |
| | groups, 8 | 2, 4 | 4 | 441, 778 | 9 | 1,230 |
| | per-component | 2, 4 | 4 | 423, 791 | 8 | 517 |
| 32, 16 | white | 3, 5 | 20 | 233, 416 | 10 | 513 |
| | groups, 8 | 3, 5 | 20 | 357, 798 | 15 | 4,063 |
| | per-component | 3, 5 | 20 | 362, 790 | 10 | 1,477 |
| 32, 256 | white | 3, 5 | 20 | 291, 706 | 10 | 574 |
| | groups, 8 | 4, 5 | 20 | 501, 1,250 | 15 | 4,703 |
| | per-component | 3, 5 | 20 | 470, 1,178 | 10 | 1,653 |

- **White already misses the frame gate beside lamps.** At reach 14, the four edits at one site, a lamp 2 voxels from one chunk face and 10 from another, land in 4 frames in every run, and the rest in 1 or 2. At reach 32, 20 of 24 edits take 3 to 5 frames. `edits_lit`, with no lamps, lands every edit in 1 frame. The cause wasn't looked into; that each step of the relight ripple to a chunk across an edge costs a frame is a guess.
- **Per-component adds no frames**: every edit lands in the frame white's does, bar one edit in one run at reach 32 at each precision.
- **Groups add frames**: eight edits that land in 1 frame under white take 2 at reach 14, still within the gate, and at reach 32 a few take one more than white in some runs. Groups also relight more chunks per edit, 9 against 8 at reach 14 and 15 against 10 at 32, since a change in any colour's arrival at a face ripples.
- **The main thread** rises by about 100 µs at reach 14 and 16 levels and 150–200 µs at 256, under both. The worst frames stay under 1 ms except at reach 32 and 256 levels (1,250 µs under groups, 1,178 per-component, against white's 706). Single frames of 2–4 ms turned up in individual runs of every row, white included, and are noise.
- **A relight costs** 2–3× white's per-component and 5–8× white's for eight groups. Per-component's slower relights land no edit later. Groups' later edits come with their extra relights; whether the ripple or the slower solves delay them wasn't separated.

### Streaming into the caves

`caves` streams 4,169 chunk arrivals in every row. Its gates: solves per arrival at most 5.5, visible changes per shown chunk at most 2.4.

| Voxel reach, levels | Candidate | Solve µs per chunk | Relight µs | Solves per arrival | Visible per shown | Streamed | Settled |
|---|---|---|---|---|---|---|---|
| 14, 16 | white | 288 | 319 | 5.00 | 2.10 | 2.8 s | 4.1 s |
| | groups, 3 | 554 | 546 | 5.24 | 2.12 | 4.7 s | 5.6 s |
| | groups, 8 | 861 | 834 | **5.55** | 2.12 | 7.3 s | 8.1 s |
| | per-component | 419 | 414 | 5.03 | 2.12 | 3.9 s | 4.6 s |
| 14, 256 | white | 219 | 280 | 5.02 | 2.12 | 2.7 s | 3.6 s |
| | groups, 8 | 988 | 983 | **5.61** | 2.14 | 8.3 s | 9.6 s |
| | per-component | 484 | 494 | 5.05 | 2.14 | 4.4 s | 5.1 s |
| 32, 16 | white | 573 | 553 | **5.58** | 2.27 | 5.1 s | 6.1 s |
| | groups, 8 | 3,762 | 3,498 | **6.39** | 2.28 | 29.9 s | 32.2 s |
| | per-component | 1,354 | 1,262 | **5.71** | 2.27 | 10.9 s | 12.1 s |

- A solve here is sky and voxel light together, at sky reach 30, so colour's share is diluted.
- **Eight groups miss the solves-per-arrival gate** at reach 14, in all three runs. The relights that changed light stay as white's (7,081 against 6,969) while relights grow 13%: slower solves let more neighbours arrive while a chunk solves, and its light has to be worked out again. Per-component holds the gate.
- **At reach 32 white misses it too** (5.58), and both candidates add to it.
- Groups at reach 32 take 30 s to stream what white streams in 5, within the bench's 120 s cap.

For reference, on the same build at reach 14 and 16 levels, white `stream_in_lamps` took 1,385 ms against 1,152 unlit, and `edits_lit` landed every edit in 1 frame with at most 155 µs of main thread.

## Follow-up: a colour index carried with one run

Measured after the ticket closed, for a question it raised: can one run carry the colour along, instead of a run per colour? Probe [colour_label.rs](../../probes/colour_label.rs), `colour_cost.rs` with one more candidate, built by `colour_label_build.sh` with [colour_label.patch](../../probes/colour_label.patch) on the kernel. Output in [colour_label.csv](colour_label.csv) and [colour_label.txt](colour_label.txt), 16 levels only.

The **carried index** is one run of every lamp, as white. A voxel reached in a step takes the colour index of a voxel that reached it, held as a bit plane per bit of the index: 2 planes for three colours, 3 for eight. Each voxel ends with the colour of the brightest light reaching it, so pools meet at a hard seam: the look "How coloured light looks" ruled out. The levels come out as white's, checked chunk by chunk. Every lit voxel's index names a colour whose own run reaches it at white's level: none was wrong in any scene.

This first version carries the index in loops of its own after each pass, row by row, skipping rows with no newly reached voxel. They aren't vectorised as the kernel's passes are, and it wasn't tuned.

Solve, p50 µs per chunk with a lamp in reach, 16 levels:

| Scene, reach | White | Carried, 3 colours | Carried, 8 colours | Groups' runs, 3 | Groups' runs, 8 | Per-component |
|---|---|---|---|---|---|---|
| caves, 14 | 44 | 107 | 108 | 103 | 217 | 128 |
| lamp_world, 14 | 48 | 161 | 169 | 205 | 520 | 144 |
| many_lamps, 14 | 38 | 117 | 121 | 143 | 190 | 119 |
| caves, 32 | 284 | 612 | 630 | 841 | 2,186 | 901 |
| lamp_world, 32 | 333 | 969 | 1,033 | 1,158 | 3,183 | 1,018 |
| many_lamps, 32 | 288 | 803 | 836 | 922 | 2,153 | 888 |

- The carried index costs 2.2–3.5× white, and barely more for eight colours than for three: the carry loops' overhead, not the planes, sets its cost.
- That is 1.6–3.5× less than eight groups' runs, before groups mix, and about what three groups or per-component cost. With eight colours it is within 17% of per-component either way, except 30% less in the caves at reach 32.

## Follow-up: a hue blended within one run

Measured after the carried index, for the next question: can one run, as white, give each voxel a hue blended from its neighbours by level? Probe [colour_hue.rs](../../probes/colour_hue.rs), `colour_label.rs` with one more candidate and a hue score, built by `colour_hue_build.sh` with [colour_hue.patch](../../probes/colour_hue.patch) on the kernel (the carried index's patch plus the blend). Output in [colour_hue.csv](colour_hue.csv) and [colour_hue.txt](colour_hue.txt), 16 levels only.

The **blended hue** is one run of every lamp, as white. A voxel reached in a step takes the mean of the hues (8-bit red, green and blue) of the voxels that reached it in that step. Those voxels all hold the same level, one step brighter, so weighting them by level changes nothing. Dimmer neighbours have no hue yet when a voxel is reached. The levels come out as white's. Wherever one colour alone is the brightest, the hue is exactly that colour's: none was wrong in any scene.

Like the carried index, it works in loops of its own after each pass, untuned. Unlike it, it can't work on a row's bits at once: it visits each newly reached voxel, reads up to six neighbours' hues and divides.

Solve, p50 µs per chunk with a lamp in reach, eight colours, 16 levels (three colours cost the same for both one-run candidates):

| Scene, reach | White | Carried index | Blended hue | Per-component | Groups' runs |
|---|---|---|---|---|---|
| caves, 14 | 45 | 95 | 134 | 127 | 220 |
| lamp_world, 14 | 48 | 162 | 265 | 144 | 527 |
| many_lamps, 14 | 38 | 111 | 162 | 120 | 190 |
| caves, 32 | 279 | 627 | 670 | 886 | 2,167 |
| lamp_world, 32 | 325 | 1,037 | 1,465 | 989 | 3,125 |
| many_lamps, 32 | 285 | 837 | 1,009 | 870 | 2,136 |

- The blended hue costs 2.4–5.5× white, and 7–64% more than the carried index.
- Against per-component it ranges from 24% less (caves, reach 32) to 1.8× more (lamp_world, reach 14).
- It is 1.2–3.2× less than eight groups' runs, before groups mix.
- The carried index's reach-14 times here are 4–12% under the first measurement's: run-to-run spread.

Hue error against the default look (a run per colour, mixed by the max per component), over lit chunk voxels above level 1, eight colours. A voxel's error is the largest difference over the components once both colours are scaled to a brightest component of 1:

| Scene, reach | Index: mean | Index: over 0.1 | Blended: mean | Blended: over 0.1 | Per-component: mean | Per-component: over 0.1 |
|---|---|---|---|---|---|---|
| caves, 14 | 0.27 | 57% | 0.24 | 57% | 0.15 | 58% |
| lamp_world, 14 | 0.44 | 85% | 0.38 | 83% | 0.17 | 63% |
| many_lamps, 14 | 0.62 | 95% | 0.55 | 95% | 0.23 | 95% |
| caves, 32 | 0.24 | 44% | 0.22 | 43% | 0.18 | 64% |
| lamp_world, 32 | 0.61 | 97% | 0.56 | 96% | 0.11 | 55% |
| many_lamps, 32 | 0.72 | 99% | 0.65 | 99% | 0.20 | 97% |

- Blending cuts the hard seam's mean error by 6–13%, and the share of voxels off by more than 0.1 by at most 2 points (up to 4 with three colours).
- Within one run a voxel's reachers disagree only where two pools' fronts meet at the same distance. A dimmer colour's light never reaches the voxels where it loses, so it can't tint them.
- Per-component is scored only for reference, since "How coloured light looks" ruled it out on looks. Its error has another cause, most likely the hue drift "Coloured voxel light in other engines" describes, as the weaker components run out first.
