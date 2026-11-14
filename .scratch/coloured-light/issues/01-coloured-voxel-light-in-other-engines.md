---
title: Coloured voxel light in other engines
labels: [wayfinder:research]
status: closed
assignee: Joacim Magnusson
blocked_by: []
---

## Question

How do shipped voxel engines and mods represent, propagate and render coloured block light, and what does each cost and get wrong?

For each engine or mod found, from its source or its authors' own writing where possible:

- **Storage:** bits per voxel, how they're split between sky light and the colour of block light, and whether colour is per component, a hue or a palette entry.
- **Propagation:** whether each component floods on its own as a max, whether one intensity floods and colour rides along with it, or something else; and how its cost compares with the engine's white light, if it had one.
- **Mixing:** what happens where lamps of different colours overlap: blending, the brighter lamp winning per voxel, or averaging.
- **Hue as light fades:** whether a lamp's colour holds as it dims, or shifts as a weaker component runs out first, and what the falloff curve does to that.
- **Banding:** how coarse colour looks at the engine's precision.
- **Glow:** what colour a lamp's own faces show.
- **Light through tinted translucent blocks:** whether stained glass or water tints block light, sky light or both; whether that's done per component (open for red, closed for green and blue) or another way; and how partial dimming, such as light losing levels through water, is handled.
- **Artifacts, rewrites and reported costs:** what players or authors complained about, and what was changed and why.

Leads, not a limit: all-is-cubes' RGB light, Vintage Story's block light, Hytale, Cubyz, Terraria and Starbound's coloured tile light, Seed of Andromeda, and coloured-light mods and forks of Minecraft and Luanti.

End with what fits this crate. Its kernel spreads one component at a time by bit dilation over rows of bits, with a reach of up to 32 and 16 or 256 levels, and its light reaches the shader through a texture of slots and a `MeshTag` (see `src/light/kernel.rs`, `src/light/slot.rs`, `src/light/route.rs` and `src/shaders/light.wesl`). For each family of representation the findings support, say:

- whether a white lamp would look and read exactly as today;
- whether a chunk with no coloured lamp near it could pay nothing extra;
- whether a later tint fits it without a rewrite;
- how it would take a cap on distinct light colours, if it needs one.

Findings go in [research/coloured-voxel-light-in-other-engines.md](../research/coloured-voxel-light-in-other-engines.md).

## Resolution

Findings: [research/coloured-voxel-light-in-other-engines.md](../research/coloured-voxel-light-in-other-engines.md).

- **Five families ship.**
  - Per-component levels, each flooding as a max: Cubyz, Seed of Andromeda, three Minecraft mods, and Hytale per secondary docs.
  - Proportional components: Starbound.
  - One level plus a hue and saturation: Vintage Story.
  - A level per light kind, each kind one colour per world: ClassiCube, Jamango.
  - Additive fields: all-is-cubes, Photon, Terraria.
- **Per-component levels are the norm, and their known flaw is hue drift.** A weaker component runs out first. Under a linear falloff that shows well before the light is dim. Under an exponential one, like this crate's default, it's confined to the last faint voxels. They're the only flood-fill family that tints per component, and every engine that tints through glass or water does it that way.
- **In this crate's kernel, proportional components, level plus colour, and light kinds are all a run per light colour.** A run records only when each voxel is first reached, so colour can't ride along with one run without a hard seam.
  - A box with one light colour runs once, as today, so white is exact and pays nothing.
  - Hue holds and reach never changes.
  - Cost grows with distinct light colours per box, which is where a cap behind a type parameter would go.
  - A tint would mean seeding a new colour's run where light leaves a tinted voxel. That's possible, but not shown.
- **Per-component levels here are three runs.** White is exact and a box with no coloured lamp pays nothing, and a tint is a per-component open mask. But colour changes each component's reach, which the settled notes rule out, and a lamp's hue depends on the falloff curve.
- **Additive fields are out.** White wouldn't look as today, and they settle over frames.
- **Not found:** Seed of Andromeda's coloured-light post, Vintage Story's closed engine, and Hytale's and Terraria's internals rest on secondary sources or inference.
