---
title: The representation of coloured voxel light
labels: [wayfinder:grilling]
status: open
assignee:
blocked_by: [03, 04]
---

## Question

How does voxel light hold its light colour, and how does a chunk pay for colour only where coloured light is?

Using the look from [How coloured light looks](03-how-coloured-light-looks.md) and the numbers from [What each colour candidate costs](04-what-each-colour-candidate-costs.md), decide:

- **The default representation:** how voxel light holds the default look from [How coloured light looks](03-how-coloured-light-looks.md), a run per light colour mixed by the max per component, and whether per-component levels stay behind a seam for tinted translucency, though they're out as a look.
- **A chunk with no coloured lamp in its light box:** how it's told apart from a coloured one, and how it keeps white light's cost.
- **A cap on distinct light colours:** whether the default needs one. If any representation does and is offered, it goes behind a type parameter, as `MaterialIndex` is.
- **The path to translucency:** how a later tint fits the chosen representation. A run holds no value per voxel to modify, only the round in which light first reached each voxel. A run can still pass through a tinted voxel with a modifier, as a delay: light reaching it in round `d` leaves it in round `d + k`, which is a drop of `k` levels. Under an exponential falloff, such as the default `0.7^(15 - level)`, that's multiplying brightness by `0.7^k`, which is what a tint does. Under a linear falloff it isn't. How the delay fits depends on the representation:
  - **Per-component levels:** each component's run gets its own delay, or is blocked. So orange glass is red passing freely, green dropping two levels and blue blocked. Any tint then passes through without a run ending, and nothing waits on another run.
  - **A run per light colour:** a delay dims a run, but a run has one colour, so it can't change hue inside itself. It also can't tell which path reached a voxel first. A hue change therefore needs light to move into another colour's run at the glass, with these costs:
    - runs then wait on each other, in order of how many panes the light has crossed;
    - tints make light colours nobody declared, so a cap or palette must cover them;
    - two arrivals in the same round, one through the glass and one around it, need a rule for which colour wins.

  Light reaches a voxel both through a tinted voxel and around it, and each component keeps the brighter of the two. So how strongly a pane tints what's behind it is the smaller of its drop and what the detour around it costs. That's strongest right behind the pane and fades with distance, as a single solid's shadow does under voxel light. A wider pane tints further, and a window filling an opening tints all the light through it. At 16 levels the tint comes in whole levels.

  Either way, direct sky isn't a run: a column either has it or doesn't, and it falls with no loss. Glass in a column means the column's answer must carry a level and a tint below the glass. Even then, the components the glass dims come in from the open columns beside it at one level less, so the whole column below shows about a one-level tint at any height, however far the ground is. That's the tint's version of the straight-down shade a floating solid casts today. A tint that's stronger near the glass would need something measured by rays, as sky openness is.
- **The word *channel*:** whether it stays sky light and voxel light only, with a new term for colour components. Record the answer in [the pending glossary](../glossary.md).

Then graduate the fog this answer makes specifiable into tickets.
