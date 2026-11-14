---
title: Coloured light
labels: [wayfinder:map]
status: open
---

# Coloured light

## Destination

A spec at `spec.md` in this directory, with no decisions left in it, for voxel light that carries a light colour in `bevy_voxel_world`. It's written with `/to-spec` from the closed tickets, then split into work with `/to-tickets` into a folder of its own. No crate code is written on the way, only throwaway probes. The spec treats sub-grids as merged and covers them. Implementation starts once sub-grids is on main, as one PR with staged commits.

## Notes

- **Settled at charting** (two grilling rounds on 2026-11-14; not tickets):
  - **Only voxel light gets colour.** Each emissive material's light carries a light colour. Sky light stays one level per voxel, and its colour stays whatever Bevy's ambient light, environment map and sun give it.
  - **The path to translucency.** Tinted translucent voxels are the next feature after this one. They're out of scope here, but every candidate is judged on whether a tint can be added later without a rewrite, and sky light's format must not rule out colour, since sky light under tinted glass becomes coloured. A run of the kernel can pass through a tinted voxel with a level drop, as a delay, so a tint can be a drop per colour component, or a hue change where light leaves the tinted voxel: see [The representation of coloured voxel light](issues/05-the-representation-of-coloured-voxel-light.md). The spec gets a short "path to translucency" section, as the light spec had one for ray tracing.
  - **White pays what it pays today.** A white world costs exactly what it does on main: CPU, CPU memory, GPU memory and shader work. The `chunk_lifecycle` A/B against main must hold. Colour is a runtime choice, never a cargo feature, unless a cargo feature turns out to be a massive win. Whether it's a per-world setting or automatic is in the fog.
  - **A coloured world keeps every light gate.** A single edit's light lands within 2 frames, using at most 1 ms of the main thread in any frame; `edits_lit`'s and `caves`' `gate_failures` hold; the lit `chunk_lifecycle` gate holds.
  - **A chunk pays for colour only where coloured light is.** A chunk with no coloured lamp in its light box solves, stores and uploads as in a white world. Any remaining extra cost is minimised and described by what it grows with.
  - **White is judged by eye too** (amended on 2026-11-14 after the research tickets, replacing "white stays white"). White light needn't look exactly as it does today. The prototype judges it as it judges any light colour, and a change is fine if it looks good. Only white's cost binds, as above.
  - **A coloured lamp's weaker components may reach less far** (amended on the same day), as under per-component levels, if the prototype's look accepts it. `voxel_reach` still sets how far a lamp's light reaches.
  - **Full coverage, except perhaps WASM.** Colour works wherever white voxel light does:
    - level-0 chunks and coarse chunks;
    - light entries in hinted air;
    - edits and every kind of relight;
    - custom meshes, through the eight-voxel read, with nothing per mesher;
    - sub-grids' own lamps;
    - `light_at`;
    - both precisions.

    WASM is the one exception allowed: if there's a lot to gain, it may be limited to white light or to part of the coverage.
  - **Recolouring may relight.** Changing a lamp's colour at runtime may cost a settings relight, as changing its level does today. A representation that recolours more cheaply earns points but isn't required.
  - **A lamp's colour is declared on `Emission`,** next to its level, as any RGB colour, white by default. `Emission::level` keeps working unchanged.
  - **The look is a prototype's call.** Mixing where lamps of different colours overlap, hue as light fades, colour banding and the glow's colour are all judged by eye with candidates as dials. If probes can't tell two apart, both go behind a seam. Nothing may read as a bug, such as a hard colour seam that looks like a glitch.
  - **A cap on distinct light colours** is only worth it for a big win, and then it goes behind a type parameter, as `MaterialIndex` is, so the game developer decides what to optimise for.
- **Base:** light propagation as it is on main, and sub-grids as [its spec](../sub-grids/spec.md) describes it. [The light propagation map](../light-propagation/map.md) and [its spec](../light-propagation/spec.md) say why white light is as it is. Read them rather than deciding those questions again.
- **Language:** use the Light section of `GLOSSARY.md`. `GLOSSARY.md` doesn't change until sub-grids is merged, so new and amended terms go in [the pending glossary](glossary.md), as entries ready to paste.
- **Skills:** grilling tickets call `grilling` and `domain-modeling`. Research tickets call `research`. Prototype tickets call `prototype`. A ticket placing a seam or shaping the public API also calls `codebase-design`, and one that drafts doc comments calls `comment-writing`.
- **Standing preferences:**
  - Content is general voxels, never heightmap terrain.
  - Nothing is handled per mesher.
  - Look and performance choices are the game developer's: ship a good default and a setting.
  - When probes can't tell two mechanisms apart, put both behind a seam and pick the default later.
  - Shoot islands and overhangs where a mechanism engages, and rule out any candidate that invents structure.
  - A prototype's reaction message opens with how the mechanism works and why it breaks, then the dials.
  - No ADRs.
  - Public docs say what a setting's cost grows with, never measured numbers.
  - WASM must run, but its limits never set a default.
- **Measuring:** `[profile.bench] codegen-units = 1` is in `Cargo.toml` on main, so A/Bs against main need no env var. Frame-time metrics move about 7% between builds of identical source, so trust only alternating A/B runs, and profile before guessing at a failed gate. Measured numbers go in `baseline/` here and never in public docs.
- **Probes:** this checkout carries the sub-grids work in progress, so nothing outside `.scratch/` changes in it, and probes are never copied into its `examples/`. Probe code lives in `.scratch/probes/`. It runs from one git worktree of main at `.scratch/coloured-light/probe-tree`, made by the first ticket that needs it, and removed when the map closes; its `target/` takes 30 to 60 GB. The old probes there were written against light before it shipped, so check one against the shipped code before reusing it. Subagent briefs forbid `until ! pgrep` wait loops, which match themselves and hang.
- **Spec and tickets:** writing the spec is the last ticket, once the fog has cleared. `/to-tickets` writes to a folder of its own, never this map's `issues/`.

### How this tracker works

This map is tracked in local files, which are git-ignored. Nothing here gets committed or branched.

- **Tickets** are the files in `issues/`, and the filename without its extension is the ticket's id. Refer to tickets by title, with the link wrapped around it.
- **Type** is the `wayfinder:<type>` entry in the ticket's `labels` frontmatter.
- **Claim** a ticket by setting `assignee: Joacim Magnusson` before doing any work. Empty means unclaimed.
- **Blocking** is the `blocked_by` frontmatter, a list of ticket ids. A ticket is unblocked once all of those are `status: closed`.
- **The frontier** is every ticket that is open, unblocked and unclaimed. Work it in filename order.
- **Resolve** a ticket by adding a `## Resolution` section, setting `status: closed`, and adding one line to Decisions so far below.
- **Assets** are linked from the ticket: research findings in `research/`, measurements in `baseline/`, probes in `.scratch/probes/`.

## Decisions so far

<!-- the index: one line per closed ticket, then follow the link for the detail the ticket holds -->
- [Coloured voxel light in other engines](issues/01-coloured-voxel-light-in-other-engines.md): engines ship per-component levels (most common, hue drifts as the weaker component runs out, the only flood-fill family that tints per component), proportional components, level plus hue, a level per light kind, or additive fields; in this kernel all but the first and last are a run per light colour, which keeps white exact and free, hue and reach unchanged, and costs grow with distinct colours per box; per-component runs keep white free but change each component's reach; additive fields are out
- [Cheap multi-component propagation and storage](issues/02-cheap-multi-component-propagation-and-storage.md): three components cost under three runs only by skipping runs (no seed, identical components, lamps grouped by light colour); one colour per chunk adds nothing per voxel, per-component planes take 2 or 4 bytes; all integer formats work on WebGL2; a light colour index (0 = white) fits a slotless tag at both precisions
- [How coloured light looks](issues/03-how-coloured-light-looks.md): the default look is a run per light colour mixed by the max per component (hue and reach hold, white's own rule); sum, both blends and per-component levels are out on looks, and whether per-component stays for tinting is the representation's call; glow keeps the face's own colour, with a setting to tint it by light colour; the default voxel falloff fades to black at level 1, so a pool's edge doesn't show as a hue step inside another colour's pool; 16 and 256 levels look the same at reach 14
- [What each colour candidate costs](issues/04-what-each-colour-candidate-costs.md): measured, nothing decided. White costs today's under both if a flag skips the colour path. A run per light colour costs a run per colour with a lamp in the box, walls or not (eight colours 5–11× white, missing `caves`' solves-per-arrival gate), and keeps today's planes where one colour reaches a chunk. Per-component costs a steady 3× and holds the gates at reach 14, but pays for one-colour boxes too, and its bit planes cost landing up to 3× today's shell reads at 256 levels. White beside lamps already misses the edit frame gate on main

## Not yet specified

<!-- in-scope fog that can't be ticketed yet; graduates as the frontier advances -->

- **How a world opts in to colour:** a per-world setting beside `LightPrecision`, or automatic, so a world becomes coloured where it declares a coloured lamp. This hangs on whether the chosen representation can be free in a white world without a setting.
- **The route to the shader:** the slot's texel format, or a second texture of colour slots taken only by chunks with coloured light. Also how a chunk with no slot carries colour in its tag (a colour index fits at both precisions; three components fit at 16 levels but not at 256, where 6-bit components or an index would), how a lamp's own cell marks it as a lamp and gives its colour, and what the sub-grids tag map's entries hold.
- **The API:** how `Emission` takes a colour, what `light_at` and `LightLevels` return for coloured light and whether that breaks them, and what the look gains: colour's brightness against white, the setting that tints a lamp's glow by its light colour, and how `LevelCurve` names the fading curve that becomes the voxel falloff's default.
- **Relights:** face hashes and the relight ripple with more than one component, edit relights, and the settings relight when a light colour changes. A run per light colour ripples further, since a change in any run's arrivals at a face relights the neighbour: see [What each colour candidate costs](issues/04-what-each-colour-candidate-costs.md).
- **Coarse chunks:** colour from a coarse chunk's own cells, the slot it takes, and the brightest-level fallback when it gives its slot away.
- **Sub-grids:** a sub-grid's own coloured lamps max'd on top of the world's light, per component or otherwise; enclosure stays sky light only.
- **Precision:** the CPU bytes per voxel that colour costs at 16 and at 256 levels, now measured for each candidate (see [What each colour candidate costs](issues/04-what-each-colour-candidate-costs.md)); what stays open is which the world view keeps.
- **WASM's limits:** whether WASM gets the full coverage, part of it or white only.
- **Gates and examples:** a coloured bench scenario beside `edits_lit` and `caves`, and where `light_tuning` shows colour. White on main already misses the edit frame gate beside lamps, and `caves`' solves per arrival at reach 32 ([What each colour candidate costs](issues/04-what-each-colour-candidate-costs.md)), so what a coloured world's gates hold against is open; white's own numbers in the same scenario are one answer. A coloured scenario streams until light settles, as `caves` does.
- **The path to translucency:** the short section the spec holds, once the representation is known.

## Out of scope

<!-- work ruled beyond the destination; never graduates -->

- **Tinted translucent voxels (stained glass, coloured water):** the next feature after this one. `WorldVoxel` has no translucency, and adding it touches meshing. This effort only keeps a path open for it: see Notes.
- **Coloured sky light:** sky light stays one level per voxel. It gains colour only through tinted voxels, which come with translucency.
- **Bevy light entities as propagation sources:** `PointLight` and `SpotLight` stay dynamic and don't feed propagation, as in the light spec.
- **Ray traced lighting:** still kept possible, as in the light spec, but not part of this effort.
- **Lighting entities other than chunks:** `light_at` returns the colour, and a game tints its own entities from it, as it dims them today.
