# Cheap multi-component propagation and storage

Research for [Cheap multi-component propagation and storage](../issues/02-cheap-multi-component-propagation-and-storage.md), 2026-11-14.

Nothing here was measured. Costs marked **reasoned** are worked out from the code and the sources, with the arithmetic shown. [What each colour candidate costs](../issues/04-what-each-colour-candidate-costs.md) measures them. "Component" here means the red, green or blue part of voxel light. Whether the word *channel* covers it is still open in [the pending glossary](../glossary.md).

Versions read: the crate as checked out on `feature/sub-grids` (light code as on main), Bevy 0.20.1, and wgpu 30.0.1 (`Cargo.lock`).

## Summary

- **Bit dilation can't spread three components for much less than three runs.** Today's rows are already SIMD-wide, so packing components into lanes saves only shared setup. The real savings come from runs that never happen: a component or light colour with no seed, components with identical seeds (a white box always runs once), and grouping lamps by light colour (a box holding one light colour runs once). Two cheap kernel changes help coloured light most: skipping the empty leading rounds of a dim component, and keeping each run inside the bounding box of what it has reached.
- **What it costs depends on how a colour rides on the kernel.** Under a *per-component flood*, each component is seeded at the lamp's level times that component, so a weak component reaches less far. That is at most 3 runs a box, but a single orange lamp already needs per-component storage. Under *constant-hue groups*, one run per light colour and each component is the run's level times the colour. A one-colour chunk then stores today's planes plus a colour index, while K colours cost K runs. Which of the two looks right is for [How coloured light looks](../issues/03-how-coloured-light-looks.md).
- **A queue flood fill over packed words (SWAR)** costs in proportion to lit voxels, not to the box. It handles all three components in one pass. It beats three bit-dilation runs only when a box holds one or two lamps in open space, it needs 6 to 8 times the box memory, and it has to reproduce the Round shape's passes exactly so that white stays white.
- **CPU storage**: today a voxel holds 1 byte at 16 levels and 2 at 256. One colour per chunk adds nothing per voxel. Full per-component planes take 2 or 4 bytes. Per-group level planes take 0.5 + 0.5K or 1 + K bytes. A brightest-colour index takes 1 + log₂P/8 or 2 + log₂P/8 bytes. Palettes and run lengths do badly on light's gradients.
- **Textures**: every integer format the ticket names, and `Rgb10a2Uint`, works with `textureLoad` on native and on WebGL2 with no wgpu feature. There's no 3-byte texel format. The largest 3D texture is set per device, not per format: 2048 a side by default and 256 guaranteed on WebGL2, which holds 343 slots. A second light texture fits on WebGL2, but wgpu's GL backend caps a shader stage at 16 sampled textures, and WebGL2 has no storage buffers.
- **The tag**: a slotless tag has 14 free bits (16–29). That's enough for three 4-bit components at 16 levels, but not for three 8-bit ones at 256. A light colour index into a per-world table of declared light colours fits at both precisions, and index 0 for white keeps a white chunk's tag bit-for-bit as today. The sub-grids tag map copies the tag, so it carries whatever the tag carries.

## 1. Today's light, which every cost is measured against

### The kernel

- **The box.** A chunk's light comes from a box of `32 + 2 × reach` voxels on a side. Each row runs along z, stored as one `u64` up to reach 16 and as two `u64`s side by side up to reach 32 (`src/light/kernel.rs:4-10`, `:51-55`, `:127-134`).
- **A round.** Each round ORs every reached row with its neighbours and keeps the open voxels. Round `d` works on a box one voxel smaller on every side than round `d - 1` (`src/light/kernel.rs:17-20`, `:381-388`). Under `LightShape::Round`, even rounds take three passes, along x, then y, then z (`src/light/kernel.rs:384-388`, `:643-700`). Odd rounds, and every round under `Diamond`, take one face step (`src/light/kernel.rs:447-470`).
- **Levels.** A voxel's level is the scale's level for the round that first reaches it, ORed into one bit plane per level bit (`src/light/kernel.rs:399-410`). A seed joins in the round after its `start`, which is the first distance whose level is no brighter than the lamp's (`src/light/kernel.rs:367-379`, `src/light/levels.rs:251-263`). So a voxel holds the level of its least distance from a source, with each source's start counted in.
- **Stopping.** A run stops once a round adds nothing and every seed and sky fill seed has joined (`src/light/kernel.rs:416-419`). A channel with no seed does no work at all (`src/light/solver.rs:12`, `:547-550`).
- **The crate's own expectation** is that coloured voxel light is more runs of the same kernel (`src/light/kernel.rs:22-23`).

**What one run costs (reasoned).** Round `r` touches `(side − 2r)²` rows in each pass.

| | Reach 14 (side 60, `u64` rows) | Reach 32 (side 96, two-word rows) |
|---|---|---|
| Row passes, `Diamond` | Σ (60 − 2r)² for r = 1..14 = 29,260 | Σ (96 − 2r)² for r = 1..32 = 137,920 |
| Row passes, `Round` (even rounds ×3) | 15,260 + 3 × 14,000 = 57,260 | 70,976 + 3 × 66,944 = 271,808 |
| Per round on the chunk's 1,024 columns | fresh rows, up to 4 plane ORs, 6 face hashes | fresh rows, up to 8 plane ORs, 6 face hashes |
| Setup: open rows from 3 chunk columns each | 60² × 3 = 10,800 column reads | 96² × 3 = 27,648 column reads |

The setup is repeated for every run: `spread_in` fills the open rows on each call (`src/light/solver.rs:704-707`). The box's scratch rows are kept per thread (`src/light/solver.rs:528-533`). The bit box is four row arrays: 4 × 3,600 × 8 B = 115 KB at reach 14, and 4 × 9,216 × 16 B = 590 KB at reach 32 (`src/light/kernel.rs:238-256`).

### Storage, slot, tag and shader

- **CPU.** Each channel is either one level for the whole chunk, or one bit plane per level bit: 16 KiB at 16 levels and 32 KiB at 256 (`src/light/solver.rs:64-71`). With sky light and voxel light that is 1 and 2 bytes per voxel, which the precision setting documents (`src/light/config.rs:289-291`, `:688-694`).
- **A slot** is 34³ = 39,304 texels (`src/light/slot.rs:36-38`). At 16 levels a texel is `R8Uint`, sky light in the low nibble and voxel light in the high one. At 256 levels it's `Rg8Uint` (`src/light/slot.rs:46-61`, `:66-80`).
- **A lamp's own cell** holds the top level in its voxel half, which no open voxel holds, and the lamp's level in its sky half (`src/light/slot.rs:95-122`, `src/shaders/light.wesl:130-133`).
- **The slot rule.** A chunk needs a slot when its faces see more than one level in a channel, or when it has a lamp touching open air (`src/light/slot.rs:165-206`).
- **The texture** is made only when the first chunk takes a slot (`src/light/route.rs:42-44`).
- **The tag.** Bit 31 says the chunk has a slot. With one, bits 0–23 are the slot index and a coarse chunk's LOD level sits in bits 24–27. Without one, bits 0–15 hold the levels, and bits 28–30 are always clear (`src/light/route.rs:18-24`, `:50-52`, `:120-134`). Sub-grids takes bit 30 to mark a sub-grid's chunk (`.scratch/sub-grids/spec.md:425`).
- **The shader** reads the slot with `textureLoad` on a `texture_3d<u32>` at binding 200, and the look from a uniform at 201 (`src/shaders/light.wesl:81-85`, `:117-135`). Its `LightLevels` already carries voxel light as a `vec3` (`src/shaders/light.wesl:98-104`, `:305`, `:326`).
- **Face hashes** take 96 B a chunk, one hash per channel per face, and relights compare them only for equality (`src/light/faces.rs:71-98`).

## 2. Two ways a colour can ride on the kernel

The kernel spreads a distance, not a colour, so a lamp's colour has to be applied either before the spread or after it. The two choices cost very differently.

- **Per-component flood.** Each component is its own spread, seeded at the lamp's level times that component. Where lamps overlap, each component takes the per-component maximum. A component weaker than the lamp's level starts later and reaches less far, so hue shifts as light fades. Cubyz works this way: three 8-bit components per voxel, a saturating subtraction per component at each step, and a per-component `@max` ([Cubyz `lighting.zig`](https://github.com/PixelGuys/Cubyz/blob/master/src/renderer/lighting.zig), lines 19–36, 122–124 and 133–135). So does the 4-bit RGB scheme 0fps describes ([0fps, "Voxel lighting"](https://0fps.net/2018/02/21/voxel-lighting/), Recap and Word level parallelism).
- **Constant-hue groups.** The box's lamps are grouped by light colour, and one spread runs per group. A component is then the group's level times the colour, so a lamp keeps its hue as it fades and reaches exactly as far as its level says. Groups mix where they overlap: by a per-component maximum, by sum, or with the brightest group winning.

Under the per-component flood, a white lamp's three spreads are identical. Under constant-hue groups, white is one group with colour (1, 1, 1). Either way a white-only box runs once and gives today's levels. Whether the per-component flood's shorter reach for weak components breaks the Notes' "colour never changes how far light reaches" is a question for [How coloured light looks](../issues/03-how-coloured-light-looks.md) and [The representation of coloured voxel light](../issues/05-the-representation-of-coloured-voxel-light.md). This file prices both.

**Kernel runs per box (reasoned):**

| Box holds | Per-component flood | Constant-hue groups |
|---|---|---|
| White lamps only | 1 (R = G = B, deduplicated) | 1 |
| One saturated colour (red) | 1 (G and B have no seeds) | 1 |
| One mixed colour (orange, 1 : 0.5 : 0) | 2 (R, then G seeded later) | 1 |
| White and red | 2 (R; G = B) | 2 |
| K distinct colours | at most 3 | K |

## 3. Spreading

### 3.1 Ways under three times one component's cost

| Mechanism | What it saves | Exists today? | Family |
|---|---|---|---|
| **Skip a component or group with no seed** | the whole run | yes, per channel (`src/light/solver.rs:547-550`) | both |
| **Deduplicate components with identical seeds** (same voxels, same starts) | whole runs: white, greys and yellow's R = G | no; comparing the sorted seed lists costs O(seeds) | per-component |
| **Group lamps by light colour** | a one-colour box runs once, whatever the colour | no; `EmissiveVoxel` has no colour (`src/light/inputs.rs:103-108`) | constant-hue |
| **One run writes three sets of planes** | the second and third runs for one colour group: each round's fresh voxels go into R, G and B planes with three level tables (level × c_R, c_G, c_B) | no; the kernel writes one level per round (`src/light/kernel.rs:399-410`) | constant-hue |
| **Share setup across runs** | the open rows, the seed scan and the scratch: roughly 5–15 % of a run (**reasoned**: 10,800 column reads against about 72k–130k operations a run at reach 14, and 27,648 against about 300k–570k at reach 32, counting §1's row passes plus 1,024 columns × 3–9 operations a round) | no; each run refills open rows (`src/light/solver.rs:704-707`) | both |
| **Stop early** | rounds after light stops spreading | yes, per run (`src/light/kernel.rs:416-419`) | both |
| **Skip empty leading rounds** | rounds before the first seed joins, which today sweep the whole shrinking box for nothing | no: rounds run from 1 whatever the seeds' starts (`src/light/kernel.rs:367-372`) | both, mostly per-component |
| **Keep each run inside the bounding box of reached rows** | rows no light has reached yet | no; each round sweeps the whole shrinking box (`src/light/kernel.rs:381-388`) | both |
| **Components in SIMD lanes** (`[u64; 4]`, or two components in one two-word row) | shared `open` loads and loop control only | no | per-component |

**Skipping empty leading rounds (reasoned).** At reach 14 and 16 levels, level(d) = 15 − d, so a lamp at level 8 starts at distance 7 and joins in round 8 (`src/light/levels.rs:132-147`, `:255-263`). Rounds 1–7 sweep 58² + 56² + … + 46² = 19,040 of a `Diamond` run's 29,260 row passes with nothing reached, about 65 %. This is exactly the shape of a per-component flood's weak component, such as orange's green. Skipping them is exact: with nothing reached and no seed joining, a round adds no fresh voxels and no hashes.

**Keeping a run inside its reached bounding box (reasoned).** For one lamp, round r's reached rows lie within a square 2r + 1 rows wide, clipped to the shrinking box. That's Σ (2r + 1)² = 4,494 row passes at reach 14, against 29,260 for the whole box (6.5× fewer), and 47,904 against 137,920 at reach 32 (2.9× fewer). Lamps spread across the box widen the bounding box until it's the whole box again, so this pays most for a few coloured lamps among many white ones. It would speed up white light too, so it's a kernel change in its own right, not a colour feature.

**Why lanes don't buy much (reasoned).** The one-word rows are zipped so that LLVM vectorises them along x (`src/light/kernel.rs:453`), and the two-word rows already fill one SIMD register (`src/light/kernel.rs:127-131`). The shifts and ORs scale with the number of components either way. Only the shared `open` loads, `z_mask` and loop control are saved. Per-round work on the chunk (fresh rows, plane ORs, face hashes) is per component in every scheme.

**Merging groups (reasoned).** With K > 1 constant-hue groups, each group's planes have to be mixed per voxel. A bit-sliced maximum of two 4-bit values over a 32-voxel column costs a few dozen bitwise operations. Over 1,024 columns and 3 components that's roughly 1,024 × 3 × 40 ≈ 120k operations per extra group at 16 levels, about twice that at 256. That's a fraction of a run (§1).

**Relights.** A per-component flood or K groups need per-component or per-group face comparisons in `could_reach_sooner` (`src/light/faces.rs:109-146`). The hashes can stay at 96 B by mixing the component or group into the voxel channel's hash, since relights compare them only for equality (`src/light/faces.rs:86-91`). Relights are in the map's fog.

### 3.2 Queue flood fill over packed words (SWAR)

- **The technique.** Several small values packed in one machine word, processed with ordinary full-word instructions, using guard bits so that carries and borrows stay inside each field ([Lamport, "Multiple byte processing with full-word instructions", CACM 1975](https://www.microsoft.com/en-us/research/uploads/prod/2016/12/Multiple-Byte-Processing-with-Full-Word-Instructions.pdf), [DOI 10.1145/360933.360994](https://doi.org/10.1145/360933.360994)).
- **For light.** 0fps applies it to 4-bit RGB light: split the even and odd fields with a `0x0f0f0f0f` mask to make room for carries, compare per field by a guarded subtraction, select the larger fields with the result, and decrement every field at once with a clamp at zero ([0fps, "Voxel lighting"](https://0fps.net/2018/02/21/voxel-lighting/), Word level parallelism). One queue entry then carries all three components.
- **Cubyz** carries `[3]u8` per queue entry in one queue, takes the per-component `@max`, and keeps per-component flags for removal ([`lighting.zig`](https://github.com/PixelGuys/Cubyz/blob/master/src/renderer/lighting.zig), lines 66–71 and 122–124).

**What it would have to match here (reasoned).**

- **Distances, not levels.** The crate's levels aren't one per step: at 256 levels a step is about 8 levels, rounded (`src/light/levels.rs:112-118`). So a SWAR fill would carry each component's distance as a per-field *min*, and map it through `LevelScale` at the end. Distance runs 0 to reach + 1 ≤ 33, which is 6 bits plus a guard bit, so three components fit in 21 bits of a `u32`.
- **The Round shape.** On even rounds, light moves along x, then y, then z, each pass gated by open voxels (`src/light/kernel.rs:643-700`). A queue fill must run round by round, with separate pass frontiers on even rounds, or a white lamp in a coloured box would land on different levels than it does today. Lamps' own voxels are open only in the round they join (`src/light/kernel.rs:371-397`).

**Cost at reach 14 and 32 (reasoned).** A queue fill's work is the voxels it lights times the neighbours it checks: 6 for a face step, more for the three-pass rounds. It's the same for one component or three.

| | Reach 14 | Reach 32 |
|---|---|---|
| Voxels one lamp lights, `Diamond` (\|x\|+\|y\|+\|z\| ≤ n) | (2n+1)(2n²+2n+3)/3 = 4,089 | 45,825 |
| Voxels one lamp lights, `Round` (≈ ball, (4/3)πn³) | ≈ 11,500 | ≈ 137,000 |
| Voxels in the whole box (dense lamps or lava) | 60³ = 216,000 | 96³ = 884,736 |
| Three bit-dilation runs, `Diamond`, row passes of 60 or 96 voxels each | 3 × 29,260 = 87,780 | 3 × 137,920 = 413,760 |
| Box memory at 4 B a voxel, against the bit box's 115 KB / 590 KB | 864 KB | 3.5 MB |

If a row pass and a neighbour check cost about the same, which flatters the queue (its accesses are scattered and branchy), the two break even at about 87,780 / 6 ≈ 15,000 lit voxels at reach 14 and 413,760 / 6 ≈ 69,000 at reach 32. That's one or two lamps in open space. **Reasoned conclusion:** a SWAR queue fill wins for a box with a lamp or two of colour and loses for many lamps, open caves or emissive fields. The bounding-box run (§3.1) closes much of the gap for the few-lamp case while keeping today's kernel and its exact metric. No source found compares the two for multi-component light.

## 4. Storage on the CPU

Per voxel, sky light included, against today's 1 byte at 16 levels and 2 at 256 (`src/light/config.rs:289-291`). One byte per voxel is 32 KiB a chunk.

| Form | 16 levels | 256 levels | Holds mixing? | Notes |
|---|---|---|---|---|
| **Today (white)** | 1 | 2 | n/a | `ChannelLight::Uniform` is 1 byte a chunk (`src/light/solver.rs:64-71`) |
| **One light colour per chunk**: today's planes plus a colour index | 1 (+1–4 B a chunk) | 2 (+1–4 B a chunk) | no colour varies inside the chunk | constant-hue only. A per-component flood's orange lamp changes hue with distance, so it doesn't fit |
| **Per-group level planes** (K groups reach the chunk, colours per chunk) | 0.5 + 0.5K: 1, 1.5, 2 for K = 1, 2, 3 | 1 + K: 2, 3, 4 | yes, worked out on read | constant-hue; beats a palette of mixed triples for K ≤ 3, since the K levels *are* the factored palette |
| **Per-component planes** (sky + R, G, B) | 2 | 4 | yes | either family; for K > 3 groups, or any per-component flood with mixed colours |
| **Deduplicated component sets** (n distinct of R, G, B) | 0.5 + 0.5n: 1, 1.5, 2 | 1 + n: 2, 3, 4 | yes | per-component; white + red stores R and one shared G = B |
| **Level + brightest colour's index** (P colours in the chunk, as index planes) | 1 + ⌈log₂P⌉/8: 1.125 (P = 2), 1.25 (4), 1.5 (16) | 2.125, 2.25, 2.5 | no: brightest wins, with a seam where two colours meet at equal level | constant-hue with brightest-wins mixing; the index is into a table of declared colours, so it's exact |
| **Level + per-voxel chroma** (e.g. RGB 4-4-4) | 1 + 1.5 = 2.5 | 2 + 1.5 = 3.5 | yes, lossy | worse than per-component planes at 16 levels |
| **Palette of mixed (R, G, B) triples** | up to 1 + plane bits; two overlapping lamps make up to 16² = 256 triples, so 8 index bits against 12 | up to 34² ≈ 1,100 triples, 11 bits against 24 | yes | needs a hash per voxel to build; Cubyz stores light palette-compressed this way ([`lighting.zig`](https://github.com/PixelGuys/Cubyz/blob/master/src/renderer/lighting.zig), line 47) |
| **Sparse columns**: a 1,024-bit mask (128 B) plus extra planes only for columns whose colour isn't the chunk's | 128 B + 16 B per extra plane per column | 128 B + 32 B per extra plane per column | yes | pays where colour varies; columns are already the planes' unit (`src/light/kernel.rs:47-49`) |
| **Run lengths along z** | depends on runs | depends on runs | yes | **reasoned:** voxel light changes level at almost every voxel within reach, so runs are short. It also loses the O(1) reads that `at`, `slot_bytes` and `across` rely on (`src/light/solver.rs:74-85`, `src/light/slot.rs:301-358`, `:433-477`) |

**Whether a chunk has one colour (reasoned).** It has to be decided from what reaches the chunk's faces, not from the box's lamp list. Faces read both the chunk's own voxels and its shell, the neighbours' border planes (`src/light/route.rs:26-31`). A red-only chunk beside a blue-only one would otherwise show its shell's blue light as red, a hard colour seam. The existing `FacesSee`/`Seen` join over own voxels and across planes is the natural place for it (`src/light/slot.rs:165-206`, `:433-477`), with "one colour or level 0" added beside "one level".

## 5. Textures

### 5.1 Integer formats for `textureLoad`

All of these are guaranteed `TEXTURE_BINDING` by WebGPU, need no wgpu feature, and are sampled on wgpu's GL backend, so they work under WebGL2. Integer formats are never filterable, which `textureLoad` doesn't need. Sources: wgpu's guaranteed features table ([`format.rs`](https://docs.rs/wgpu-types/30.0.1/src/wgpu_types/texture/format.rs.html#914), lines 914–994), its required features ([line 818](https://docs.rs/wgpu-types/30.0.1/src/wgpu_types/texture/format.rs.html#818)), its block sizes ([line 1215](https://docs.rs/wgpu-types/30.0.1/src/wgpu_types/texture/format.rs.html#1215)), the GL backend's capabilities ([`gles/adapter.rs`](https://docs.rs/wgpu-hal/30.0.1/src/wgpu_hal/gles/adapter.rs.html#1173), lines 1173 and 1245–1288, where `renderable` includes `SAMPLED`), and OpenGL ES 3.0's `glTexImage3D` internal formats ([Khronos reference page](https://registry.khronos.org/OpenGL-Refpages/es3.0/html/glTexImage3D.xhtml), Table 2).

| wgpu format | Bytes a texel | GLES internal format | Could hold |
|---|---|---|---|
| `R8Uint` | 1 | `R8UI` | today's 16 levels (sky + voxel nibbles); or G + B nibbles in a colour texture |
| `Rg8Uint` | 2 | `RG8UI` | today's 256 levels; 16 levels with RGB (sky 4 + 3 × 4); or G + B at 256 in a colour texture |
| `R16Uint` | 2 | `R16UI` | 16 levels with RGB; or sky 4 + level 4 + an 8-bit colour index |
| `Rgba8Uint` | 4 | `RGBA8UI` | 256 levels with RGB (sky + R + G + B); all-is-cubes uses it for RGB light plus a status byte ([`light_texture.rs`](https://github.com/kpreid/all-is-cubes/blob/main/all-is-cubes-gpu/src/light_texture.rs), [`light/data.rs`](https://docs.rs/all-is-cubes/latest/src/all_is_cubes/space/light/data.rs.html), lines 51–58 and 155–163) |
| `Rg16Uint` | 4 | `RG16UI` | 256 levels with RGB packed as two halves |
| `R32Uint` | 4 | `R32UI` | 256 levels with RGB in one word |
| `Rgb10a2Uint` | 4 | `RGB10_A2UI` | three 10-bit fields and one 2-bit field; no gain over `Rgba8Uint` here |
| `Rg32Uint`, `Rgba16Uint` | 8 | `RG32UI`, `RGBA16UI` | more than any candidate needs |
| `Rgba32Uint` | 16 | `RGBA32UI` | the sub-grids tag map's entries (`.scratch/sub-grids/spec.md:418`) |

GLES has `RGB8UI` with 3 bytes a texel ([`glTexImage3D`](https://registry.khronos.org/OpenGL-Refpages/es3.0/html/glTexImage3D.xhtml), Table 2), but wgpu has no 3-byte format ([`block_copy_size`](https://docs.rs/wgpu-types/30.0.1/src/wgpu_types/texture/format.rs.html#1215)). So 24 bits of RGB at 256 levels take 4 bytes, either with sky in the fourth byte or split 2 + 2 across two textures.

### 5.2 The largest 3D texture

- **Per device, not per format.** wgpu's `max_texture_dimension_3d` caps each side of any 3D texture, whatever its format: 2048 by default and 256 in the downlevel and WebGL2 limits ([`limits.rs`](https://docs.rs/wgpu-types/30.0.1/src/wgpu_types/limits.rs.html#147), lines 147–151, 425, 561 and 582). WebGPU's default is also 2048 ([WebGPU, Limits](https://www.w3.org/TR/webgpu/#limits)). OpenGL ES 3.0 guarantees 256 ([`glGet`](https://registry.khronos.org/OpenGL-Refpages/es3.0/html/glGet.xhtml)), and wgpu's GL backend reads the real `MAX_3D_TEXTURE_SIZE` ([`gles/adapter.rs`](https://docs.rs/wgpu-hal/30.0.1/src/wgpu_hal/gles/adapter.rs.html#738), lines 738 and 773). By default, Bevy asks for the adapter's own limits, so a browser offering more than 256 gets it ([`renderer/mod.rs`](https://github.com/bevyengine/bevy/blob/v0.20.1/crates/bevy_render/src/renderer/mod.rs#L322-L332)).
- **What fits.** Slots per side are ⌊side / 34⌋: 7³ = 343 slots at 256, as the crate's test and docs say (`src/light/route.rs:1222-1230`, `src/light/config.rs:304-313`), and 60³ = 216,000 at 2048. The crate clamps to the device (`src/light/route.rs:146-160`, `:324-328`).
- **Memory per slot (reasoned, 39,304 texels).**

| Bytes a texel | Per slot | 1,024 slots (the default) | 343 slots (the WebGL2 minimum) |
|---|---|---|---|
| 1 | 38.4 KiB | 38.4 MiB | 12.9 MiB |
| 2 | 76.8 KiB | 76.8 MiB | 25.7 MiB |
| 4 | 153.5 KiB | 153.5 MiB | 51.4 MiB |

A whole-slot upload is 39,304 × that many bytes, and a shell plane is 1,024 × that many (`src/light/route.rs:1120-1178`).

### 5.3 A second light texture, and WebGL2

- **Sampled textures are capped at 16 per shader stage** under WebGL2's limits ([`limits.rs`](https://docs.rs/wgpu-types/30.0.1/src/wgpu_types/limits.rs.html#589), line 589). wgpu's GL backend hard-codes 16, whatever the hardware ([`gles/mod.rs`](https://docs.rs/wgpu-hal/30.0.1/src/wgpu_hal/gles/mod.rs.html#135), line 135, and [`gles/adapter.rs`](https://docs.rs/wgpu-hal/30.0.1/src/wgpu_hal/gles/adapter.rs.html#784), line 784). GLES guarantees 16 fragment texture units ([`glGet`](https://registry.khronos.org/OpenGL-Refpages/es3.0/html/glGet.xhtml)).
- **What already takes units.** The crate binds the voxel texture array and the light texture (`src/voxel_material.rs:99-100`, `src/shaders/light.wesl:81-82`), and sub-grids adds the tag map texture (`.scratch/sub-grids/spec.md:418`). Bevy's view bindings add a point shadow map, a directional shadow map, the tonemapping LUT, and environment maps and prepass textures when those are on ([`mesh_view_bindings.wesl`](https://github.com/bevyengine/bevy/blob/v0.20.1/crates/bevy_pbr/src/render/mesh_view_bindings.wesl#L11-L130)). A colour texture is one more unit. It's possible, but its headroom depends on which Bevy features a game turns on. That count wasn't checked here.
- **No storage buffers.** WebGL2 has none (`max_storage_buffers_per_shader_stage: 0`, [`limits.rs`](https://docs.rs/wgpu-types/30.0.1/src/wgpu_types/limits.rs.html#591), line 591). A per-world colour table must therefore be a uniform or a texture. A uniform binding holds 16 KiB ([`limits.rs`](https://docs.rs/wgpu-types/30.0.1/src/wgpu_types/limits.rs.html#597), line 597, and `GL_MAX_UNIFORM_BLOCK_SIZE` in [`glGet`](https://registry.khronos.org/OpenGL-Refpages/es3.0/html/glGet.xhtml)). That's 4,096 packed RGB colours, and the table could ride in the look's existing uniform at binding 201. The shader already indexes that uniform dynamically on WebGL2, in `falloff` (`src/shaders/light.wesl:340-349`).
- **Nothing makes a second texture impossible.** Its costs on WebGL2 are one texture unit, a second `textureLoad` per cell for chunks that use it, and its upload bytes. Made lazily, as today's texture is (`src/light/route.rs:42-44`), a white world never makes it.

**Two GPU layouts (reasoned).**

- **One wider texture** in an RGB format for the whole world: 2 B a texel at 16 levels, 4 B at 256. One `textureLoad` per cell, but every slot pays double, including white chunks in a world with colour on.
- **Today's texture plus a colour texture of the same format family**, taken only by chunks whose faces see more than one colour. At 16 levels it's `R8Uint` holding G and B, at 256 `Rg8Uint` holding G and B, with the main slot's voxel half holding R, which for white is today's level. A white chunk pays nothing, and a coloured chunk pays 2 B or 4 B a texel and two loads per cell. A lamp's own cell can keep today's marker in the main slot (voxel half at the top level) with its R in the sky half, which a solid never uses, and its G and B in the colour slot. The tag must say where the colour slot is (§6.3).
- **Shader cost.** A face reads 1 cell on the grid, 4 when smoothed, and 8 off the grid (`src/shaders/light.wesl:162-188`, `:246-278`). Blending runs once per channel per read: twice today (sky and voxel), and four times with R, G and B. That's unless the zero-as-wall mask is worked out once from the brightest component and shared (`src/shaders/light.wesl:141-157`, `:210-242`).
- **Keeping white's shader cost.** A branch on a tag bit is uniform across a draw, since the tag is per mesh instance. A shader def set only in worlds with colour on, beside today's `VOXEL_LIGHT` (`src/shaders/light.wesl:4-6`), would keep a white world's shader exactly as it is today.

## 6. The tag

### 6.1 What Bevy gives

- **One `u32` per mesh instance.** `MeshTag` is a `u32` that Bevy copies into the mesh uniform's `tag` field ([`components.rs`](https://github.com/bevyengine/bevy/blob/v0.20.1/crates/bevy_mesh/src/components.rs#L181-L208), [`mesh_types.wesl`](https://github.com/bevyengine/bevy/blob/v0.20.1/crates/bevy_pbr/src/render/mesh_types.wesl#L22-L23)). Bevy has no second per-instance slot for it.
- **`GpuComponentArrayBuffer` can't carry colour here.** Bevy 0.20's mechanism for per-instance component data takes the tag over as an index into its buffer ([`components.rs`](https://github.com/bevyengine/bevy/blob/v0.20.1/crates/bevy_mesh/src/components.rs#L188-L192)). That buffer is a storage buffer ([`gpu_component_array_buffer.rs`](https://github.com/bevyengine/bevy/blob/v0.20.1/crates/bevy_render/src/gpu_component_array_buffer.rs#L36-L59)), so it would collide with the crate's tag and doesn't exist on WebGL2.
- **Meshes can't carry it either.** Meshes are shared through the mesh cache, so vertices can't hold light (`src/light/route.rs:4-8`).

### 6.2 Bits a chunk without a slot needs

A slotless tag uses bits 0–15 for levels and keeps bit 31 for "has a slot" and bit 30 for sub-grids, which leaves bits 16–29: **14 free bits**.

| Carrying colour as | 16 levels | 256 levels |
|---|---|---|
| Three components (sky + R, G, B) | 4 + 12 = 16 bits. Keep sky and voxel/R in bits 0–7 as today and G, B in 8 more bits, with one flag bit for "coloured" so that a white chunk's tag stays bit-identical. **Fits** | 8 + 24 = 32 bits. G and B need 16 bits and only 14 are free. **Doesn't fit** without dropping G and B to 7 bits each, which costs white exactness unless flagged |
| Level + a light colour index into a per-world table | up to 14 bits of index | up to 14 bits of index (16,384 colours) |
| Level + chroma (e.g. RGB 5-5-4) | 14 bits, lossy | 14 bits, lossy |

**Which index fits best (reasoned).** Constant-hue groups give a one-colour chunk exactly one declared light colour. An index into a per-world table of declared colours is then exact, and index 0 for white keeps a white chunk's tag bit for bit as today. A per-component flood's mixed or fading colours aren't table entries, so it needs the per-component form, which doesn't fit at 256 levels.

**How much it matters (reasoned).** A chunk goes without a slot only when every face sees one level per channel (`src/light/slot.rs:203-205`). Voxel light drops a level at almost every step within reach, so a slotless chunk nearly always shows voxel level 0, where colour doesn't matter. The slotless chunks with non-zero voxel light are mostly fallbacks:

- a chunk whose slots ran out, showing its most common level (`src/light/route.rs:45-48`);
- a coarse chunk that gave its slot away, showing its brightest level (`src/light/route.rs:53-57`).

Those already approximate, so lossy colour there costs little, as long as white stays exact. A coloured chunk that must be exact can take a slot instead.

### 6.3 With a slot, and other carriers

- **Spare bits with a slot.** A tag with a slot has bits 28–29 spare. Its slot index needs ⌈log₂ slots⌉ bits: 10 for the default 1,024, and 18 for the 216,000 slots a 2048-a-side texture holds. **Reasoned:** capping the index at 18 bits frees bits 18–23. A cap of 2¹⁸ slots is about 9.6 GiB of `R8Uint`, beyond any real budget. Those 6 + 2 bits can hold a light colour index for a one-colour slot holder, or a colour-slot index for a coloured one.
- **Naming the colour slot.** The colour texture can mirror the main texture's shape and reuse the slot index, which costs full-capacity memory once a world has any colour. Or it can hold fewer colour slots, named from those freed bits, with the main slots of coloured chunks drawn from a matching range.
- **Other carriers:**
  - a per-world colour table indexed from the tag, as a uniform: works on WebGL2 (§5.3);
  - a separate material per colour: splits batches, so it's costly;
  - vertex data and `GpuComponentArrayBuffer`: ruled out (§6.1).

### 6.4 The sub-grids tag map's entries

Each entry is one `Rgba32Uint` texel of (x, y, z, tag), 16 bytes, holding the chunk's tag "in its usual encoding", and the shader does one lookup per sub-grid fragment (`.scratch/sub-grids/spec.md:418`, `:427-430`, `:431`). So any colour bits the tag gains reach sub-grid faces at no cost. If a coloured world chunk needs more than the tag holds, such as a separate colour-slot index or a colour, the key can be packed. **Reasoned:** with chunk coordinates bounded to ±2²⁰ (±33 million voxels), x, y and z fit 21 bits each in 64 bits, which frees a whole `u32` per entry. The other choice is two texels per entry, 32 B.

## 7. Cheapest combinations

### A world where most chunks see no coloured light (reasoned)

- **Spread.** Today's kernel, one run per white box, unchanged. A box with coloured lamps runs once per light colour under constant-hue groups, or once per distinct component under a per-component flood. It shares the open rows and skips empty leading rounds, and the bounding-box run keeps a lone coloured lamp's cost to a fraction of a box.
- **CPU.** White chunks keep `ChannelLight` as it is. A one-colour chunk keeps today's planes plus a colour index (constant-hue). Only chunks where colours mix keep per-group or per-component planes.
- **GPU.** Today's `R8Uint`/`Rg8Uint` texture is unchanged. A one-colour chunk names its colour in the tag: a light colour index into a per-world table in the look's uniform, 0 meaning white. Chunks where colours mix take a colour slot in a second texture made only when the first one needs it, `R8Uint` at 16 levels and `Rg8Uint` at 256.
- **White.** A white world makes no colour texture and fills no table, and its tags are bit-identical. A colour shader def keeps its shader as it is. That meets "a chunk pays for colour only where coloured light is" on CPU, GPU memory and uploads, and in the shader for white chunks.

### A world where many chunks see coloured light (reasoned)

- **Spread.** K grows, and constant-hue groups pay K runs plus merges. A per-component flood is capped at 3 runs, so with many colours it's the cheaper spread. Either way the bounding-box run and shared setup help. A SWAR queue fill is cheaper only where boxes hold a lamp or two in open space.
- **CPU.** Per-component planes at 2 or 4 bytes a voxel, deduplicated where components match.
- **GPU.** One wider texture, `Rg8Uint` or `R16Uint` at 16 levels and `Rgba8Uint` or `R32Uint` at 256, saves the second `textureLoad` per cell once most slot holders are coloured. It costs every slot 2× the memory. Which layout wins depends on the share of coloured slots, a per-world choice for the representation ticket.

### Left for [What each colour candidate costs](../issues/04-what-each-colour-candidate-costs.md) to measure

- the real cost of a row pass against a SWAR neighbour check;
- how much skipping empty rounds and the bounding-box run save on the cave scene and the many-lamps scene;
- how often boxes hold one, two or three-plus light colours;
- the bit-sliced merge's cost;
- the shader cost of the second load.

## 8. What wasn't found

- **No source compares bit dilation with a queue flood fill for multi-component light.** The kernel's reference reports bitwise fills far faster than a stack fill on single-channel masks, and nothing on several channels ([Atrufulgium, "SIMD flood fill"](https://atrufulgium.net/2024/08/21/simd-flood-fill)).
- **Seed of Andromeda's own article** on flood fill lighting couldn't be fetched, since the site is gone and the archive refused automated requests. Its 2-byte layout (4 bits each of R, G, B and sky) is cited from 0fps.
- **Vintage Story's light layout.** Its `lightHsv` is hue, saturation and value, but its bit layout isn't public. That's for [Coloured voxel light in other engines](../issues/01-coloured-voxel-light-in-other-engines.md).
- **WebGL2 texture units.** How many units Bevy 0.20's view bindings take on WebGL2 under common feature sets wasn't counted.
- **The WebGPU spec's format table** (§26.1.1) didn't come through the fetch tool. Bytes per texel and guaranteed usages are cited from wgpu-types, whose table is documented as the WebGPU spec's guarantees.

## Sources

- The crate: `src/light/kernel.rs`, `solver.rs`, `slot.rs`, `route.rs`, `faces.rs`, `levels.rs`, `inputs.rs`, `config.rs`, `src/voxel_material.rs`, `src/shaders/light.wesl`; `.scratch/sub-grids/spec.md`.
- Bevy 0.20.1:
  - [`bevy_mesh/src/components.rs`](https://github.com/bevyengine/bevy/blob/v0.20.1/crates/bevy_mesh/src/components.rs#L181-L208)
  - [`bevy_pbr/src/render/mesh_types.wesl`](https://github.com/bevyengine/bevy/blob/v0.20.1/crates/bevy_pbr/src/render/mesh_types.wesl#L22-L23)
  - [`bevy_pbr/src/render/mesh_view_bindings.wesl`](https://github.com/bevyengine/bevy/blob/v0.20.1/crates/bevy_pbr/src/render/mesh_view_bindings.wesl#L11-L130)
  - [`bevy_render/src/gpu_component_array_buffer.rs`](https://github.com/bevyengine/bevy/blob/v0.20.1/crates/bevy_render/src/gpu_component_array_buffer.rs#L36-L59)
  - [`bevy_render/src/settings.rs`](https://github.com/bevyengine/bevy/blob/v0.20.1/crates/bevy_render/src/settings.rs#L90-L110)
  - [`bevy_render/src/renderer/mod.rs`](https://github.com/bevyengine/bevy/blob/v0.20.1/crates/bevy_render/src/renderer/mod.rs#L322-L332)
- wgpu 30.0.1:
  - [`wgpu-types/src/limits.rs`](https://docs.rs/wgpu-types/30.0.1/src/wgpu_types/limits.rs.html#557)
  - [`wgpu-types/src/texture/format.rs`](https://docs.rs/wgpu-types/30.0.1/src/wgpu_types/texture/format.rs.html#914)
  - [`wgpu-hal/src/gles/adapter.rs`](https://docs.rs/wgpu-hal/30.0.1/src/wgpu_hal/gles/adapter.rs.html#1173)
  - [`wgpu-hal/src/gles/mod.rs`](https://docs.rs/wgpu-hal/30.0.1/src/wgpu_hal/gles/mod.rs.html#135)
- Specs:
  - [WebGPU](https://www.w3.org/TR/webgpu/#limits)
  - [OpenGL ES 3.0 `glTexImage3D`](https://registry.khronos.org/OpenGL-Refpages/es3.0/html/glTexImage3D.xhtml)
  - [OpenGL ES 3.0 `glGet`](https://registry.khronos.org/OpenGL-Refpages/es3.0/html/glGet.xhtml)
- Techniques and engines:
  - [Lamport 1975](https://www.microsoft.com/en-us/research/uploads/prod/2016/12/Multiple-Byte-Processing-with-Full-Word-Instructions.pdf)
  - [0fps, "Voxel lighting"](https://0fps.net/2018/02/21/voxel-lighting/)
  - [Cubyz `lighting.zig`](https://github.com/PixelGuys/Cubyz/blob/master/src/renderer/lighting.zig)
  - [all-is-cubes `light_texture.rs`](https://github.com/kpreid/all-is-cubes/blob/main/all-is-cubes-gpu/src/light_texture.rs) and [`light/data.rs`](https://docs.rs/all-is-cubes/latest/src/all_is_cubes/space/light/data.rs.html)
  - [Atrufulgium, "SIMD flood fill"](https://atrufulgium.net/2024/08/21/simd-flood-fill)
