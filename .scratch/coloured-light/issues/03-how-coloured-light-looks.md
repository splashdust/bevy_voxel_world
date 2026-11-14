---
title: How coloured light looks
labels: [wayfinder:prototype]
status: closed
assignee: Joacim Magnusson
blocked_by: [01, 02]
---

## Question

Which representations of coloured voxel light look right, and what's the default look?

Build a fly-around probe on a new coloured reference scene that has:

- lamps of different light colours whose light overlaps, in a cave, under a floating island and under an overhang, where the mechanism engages;
- a white lamp beside a copy of today's white light, so white can be judged against today's; it may change if it looks good;
- an orange lamp, or another colour mixing two components, in a long corridor, to show hue as light fades.

Give it dials for:

- each candidate representation from [Coloured voxel light in other engines](01-coloured-voxel-light-in-other-engines.md) and [Cheap multi-component propagation and storage](02-cheap-multi-component-propagation-and-storage.md);
- 16 and 256 levels;
- `voxel_falloff`, today's default and a linear curve;
- the glow's colour: the light colour, the material's own colour, or both.

Judge by eye:

- mixing where lamps overlap;
- hue as light fades;
- colour banding;
- the glow;
- that nothing reads as a bug.

Decide the default look, and which candidates stay behind a seam because the probe can't tell them apart. The probe runs from the probe worktree of main described in the map's Notes.

## Resolution

Decided with the probe described under Comments. Shots are in `../prototype/shots/` and contact sheets in `../prototype/sheets/`.

- **The default look is a run per light colour, mixed by the max per component.**
  - Each component is the max over runs of colour times brightness.
  - Hue holds to the last voxel, and colour never changes reach.
  - Overlaps blend continuously, through yellow, cyan and purple.
  - It's white's own rule, the brighter value wins per component, so a red and a green lamp meet as a yellow no brighter than either.
  - The one thing that read as a bug was where a pool's reach ends, and that comes from the falloff curve, not the mixing rule. See the next point.
- **The default voxel falloff fades to black at level 1:** `(0.7^(15 - level) - 0.7^14) / (1 - 0.7^14)`, in place of `0.7^(15 - level)`.
  - **Why:** under the old curve, a pool's last level is 0.7% of full brightness, and the next voxel is 0. Against black, or under a brighter white pool, the step can't be seen. Inside a saturated colour's pool it can, because that colour's weak components are as dim as the step. A white pool ending inside a red one halved the red pool's green and blue within a voxel, so the red suddenly read deeper (green to red went from 0.25 to 0.13). Every mixing rule shows the same step, and red alone holds its hue.
  - **What it costs:** under the new curve, green to red slides from 0.20 to 0.13 over about four voxels. Each level loses 0.7% of full brightness: level 2 keeps 30% of its old brightness, level 3 51% and level 4 66%. So the last few voxels of every pool, white included, are dimmer. That changed views lit at full strength by at most 5/255 per pixel, but surfaces lit only by a pool's tail go darker. It's a look change, with no cost in solving, memory or shader work.
  - **Scope:** only the voxel falloff's default changes. Sky light isn't coloured and adds to voxel light rather than mixing with it per component, so the sky falloff keeps `0.7^(15 - level)`.
  - **For the docs:** a voxel falloff whose level 1 isn't black, such as `LevelCurve::linear`, ends each pool in a step. Where that step crosses another colour's pool, it shows as a hue step.
- **Out on looks:**
  - **Sum:** overlaps come out brighter, but lamps of different colours add up where two white lamps don't, so white and red wash a room pink. It's dropped, not offered as a setting.
  - **Blend by brightness and blend by level:** averaging the colours at the brightest level dims the overlaps, leaving a dark band between differently coloured lamps that reads as a shadow. Blend by brightness also bakes the falloff into the stored colour.
  - **Per-component levels:**
    - Under the default curve, saturated lamps look as they do under max.
    - Mixed colours get a faint rim of their stronger component where their pool ends: orange rims red, and teal rims blue. It shows in the light-only view or at 4× voxel brightness.
    - Under `LevelCurve::linear` it halos badly, and a falloff change recolours every lamp.

    It's out as a look. Whether it survives behind a seam for tinted translucency is for [The representation of coloured voxel light](05-the-representation-of-coloured-voxel-light.md) to decide.
- **Nothing stays behind a seam for the look.** The probe told every pair of candidates apart, except the two blends under the default curve, which are equal by construction. Both blends are out.
- **Glow:** a lamp's own faces keep today's glow by default, the face's own colour, so white is unchanged and the game's lamp texture decides. A setting tints the glow by the light colour (texture times light colour). The light colour alone loses the texture, and it's out.
- **White:** a white lamp looks as it does today under every candidate. Only the glow setting could change it, and its default doesn't.
- **16 and 256 levels:**
  - At voxel reach 14 they're identical, since both fall a level per step and lamps seed whole levels.
  - At reach 32, smoothing hides the difference.

  So the look doesn't pick a precision, and colour works at both.
- **For the shader:** components blend with one weight per cell, and a cell is left out when its brightest component is 0, as today's blend leaves out a 0. A zero test per component would let a component that's missing from one cell take its neighbour's value at full weight.
- **Walls:** light doesn't pass a wall one voxel thick. A lamp set in such a wall also shines out of its back face, as any lamp does. That's what lit the stages' outsides until their back walls were made two thick, and today's white light does the same.
- **Day:** coloured pools under the island and the overhang fade into sky light, as white ones do.

## Comments

**2026-11-14, probe built, awaiting the verdict.**

- Probe: [colour_look.rs](../../probes/colour_look.rs) and [colour_look.wesl](../../probes/colour_look.wesl), built by [colour_look_build.sh](../../probes/colour_look_build.sh) in the probe worktree (`probe-tree`, made by this ticket). Run `target/release/examples/colour_look_01` from `probe-tree`; `shots out=<dir>` shoots the fixed list.
- How it works: the crate renders the world as on main, including sky light and today's white voxel light. The probe solves each candidate's coloured voxel light with the crate's own kernel (`light_bench::KernelBox`, one box per chunk as the solver does) into a 3D texture, and its material swaps that in for voxel light. The middle stage always shows the crate's own light, as today's copy.
- Check: under "today", the probe's white matches the crate's voxel light everywhere except about 18 of 1.2M sampled pixels on silhouette edges, where the probe reads the fragment position without centroid sampling.
- Shots: `../prototype/shots/`, and contact sheets in `../prototype/sheets/`, made by [colour_look_sheet.py](../../probes/colour_look_sheet.py).

**2026-11-14, walls.** Light seemed to bleed through one-voxel walls on the third stage. Two-voxel back walls (`../prototype/walls/before_after.png`) showed it was the lamps' back faces, not a leak.

**2026-11-14, the deeper red.** On the third stage's outer wall, the red deepened with distance from the lamp. It was the white lamp's pool ending in a step, not tonemapping: four tonemappers gave the same profile, and the red alone held its hue. A curve black at level 1 turned the step into a fade. The shots are in `../prototype/seam/` (`seam_before_after.png`, plus white compared under both curves in `white/`) and in `../prototype/tonemap/`. The probe has this curve on F, a tonemapping dial on T, and the arguments `thin` (one-voxel back walls) and `nowhite` (no white lamp on the third stage).
