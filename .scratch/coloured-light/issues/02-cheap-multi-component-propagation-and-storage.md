---
title: Cheap multi-component propagation and storage
labels: [wayfinder:research]
status: closed
assignee: Joacim Magnusson
blocked_by: []
---

## Question

What are the cheapest known ways to spread, store and upload three colour components of voxel light, given how this crate's light works today?

Read the crate's light first. The kernel is in `src/light/kernel.rs`, the solver in `src/light/solver.rs`, slots and the tag in `src/light/slot.rs` and `src/light/route.rs`, the face hashes in `src/light/faces.rs`, and the shader read in `src/shaders/light.wesl`. Read the sub-grids tag map's design in `.scratch/sub-grids/spec.md`. Then find, from primary sources (papers, engine and library source, the wgpu, WebGPU, WebGL2 and OpenGL ES 3.0 specs, Bevy's source):

- **Spreading:** ways to run three components through a bit-dilation flood fill for less than three times one component's cost. Candidates include sharing the open mask and the shrinking box across components, skipping a component with no source in the box, grouping lamps by light colour so that a box holding only one light colour runs once, and stopping a component early once its rounds add nothing. Also alternatives to bit dilation for colour, such as a queue flood fill over packed nibbles that takes a per-component max of a whole word at once (SWAR), with what each costs at reach 14 and 32.
- **Storage on the CPU:** ways to hold colour only where it's present. Candidates include one light colour for a whole chunk, colour planes kept only for chunks whose box holds more than one light colour, palettes, and run-length or sparse forms. Say what each costs per voxel at 16 and 256 levels against today's 1 and 2 bytes.
- **Textures:** which integer 3D texture formats wgpu offers for `textureLoad` on native and on WebGL2 (such as `R8Uint`, `Rg8Uint`, `R16Uint`, `Rgba8Uint`, `Rg16Uint`, `R32Uint`), their bytes per texel, the largest 3D texture each allows, and anything that makes a second light texture or binding expensive or impossible on WebGL2.
- **The tag:** today's `MeshTag` bit layout, how many bits a chunk with no slot would need to carry a light colour at 16 and at 256 levels, what else could carry it (such as a small per-world buffer of light colours indexed from the tag), and what the sub-grids tag map's `Rgba32Uint` entries could hold.

End with the combinations that look cheapest for a world where most chunks see no coloured light, and for one where many do. Findings go in [research/cheap-multi-component-propagation-and-storage.md](../research/cheap-multi-component-propagation-and-storage.md).

## Resolution

Findings: [research/cheap-multi-component-propagation-and-storage.md](../research/cheap-multi-component-propagation-and-storage.md). Nothing was measured; reasoned costs show their arithmetic.

- **Spreading:** bit dilation gains little from packing components, since its rows are already SIMD-wide. The savings are runs that never happen: no seed, identical components (white runs once) and lamps grouped by light colour (a one-colour box runs once). Skipping empty leading rounds and a reached bounding box make dim or lone coloured lamps cheap. A SWAR queue fill wins only for a lamp or two per box.
- **The look sets the cost:** a per-component flood caps at 3 runs but needs per-component storage for any mixed colour. Constant-hue groups take one run per light colour and let a one-colour chunk keep today's planes plus a colour index.
- **CPU per voxel** against today's 1 and 2 bytes: one colour per chunk adds nothing. Per-component planes take 2 and 4 bytes, and per-group levels 0.5 + 0.5K and 1 + K.
- **Textures and tag:**
  - Every integer format named works with `textureLoad` on native and WebGL2. There's no 3-byte format, and the 3D size limit is per device: 256 guaranteed on WebGL2, which holds 343 slots.
  - A second texture fits within WebGL2's 16 sampled textures per stage.
  - A slotless tag has 14 free bits: three 4-bit components fit at 16 levels but not three 8-bit ones at 256. A light colour index into a per-world table fits both, and 0 means white.
- **Cheapest:** in a mostly white world, constant-hue groups, a colour index in the tag, and a lazily made second texture only for chunks where colours mix. In a heavily coloured world, a per-component flood with per-component planes and one wider texture.
