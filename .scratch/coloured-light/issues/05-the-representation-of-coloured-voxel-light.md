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
  - **One run can't carry the default look.** Measured after [What each colour candidate costs](04-what-each-colour-candidate-costs.md) closed, in the follow-ups at the end of [colour-costs.md](../baseline/colour-costs.md). One run is cheap because its front stops at voxels it has already lit. So a dimmer colour's light never reaches the voxels where a brighter one wins, and can't tint them. Mixing as the default look does needs a front per colour, which is a run per light colour, or a flood that re-enters lit voxels and compares levels voxel by voxel, which gives up the kernel's rows of bits. Two one-run versions of the family "one level plus a colour" from [Coloured voxel light in other engines](01-coloured-voxel-light-in-other-engines.md) were measured, each with eight colours and 16 levels:
    - **A carried index:** one run as white, each voxel taking the colour index of a voxel that lit it, in 2 bit planes for three colours and 3 for eight. It costs 2.2–3.5× white whatever the number of colours, about what per-component costs. It gives the brightest light's colour, with a hard seam where pools meet, which [How coloured light looks](03-how-coloured-light-looks.md) ruled out.
    - **A blended hue:** one run as white, each voxel taking the mean colour of the voxels that lit it. Those all hold the same level, one step brighter, so weighting them by level changes nothing. It costs 2.4–5.5× white, from 24% less to 1.8× more than per-component, since it works voxel by voxel. Against the default look it cuts the hard seam's colour error by only 6–13%: colours mix only where two fronts meet at the same distance.
  - **Storage as a level and a colour:** with each light colour scaled to a brightest component of 1, the default look's brightest component is white's level. So where colours mix, a chunk could keep today's planes and add a hue per voxel instead of holding three components. Where one colour reaches it, today's planes and the chunk's colour index already hold it. That saves storage, not solving: the mixed hue still comes from the runs per light colour.
- **A chunk with no coloured lamp in its light box:** how it's told apart from a coloured one, and how it keeps white light's cost.
  - **Colours whose light can't reach the chunk:** a run per light colour pays for every colour with a lamp in the box, even where walls stop its light short of the chunk. In the numbers from [What each colour candidate costs](04-what-each-colour-candidate-costs.md), with eight colours at reach 14 and 32, the median `caves` chunk runs 8 colours and 1 reaches it, and the median `many_lamps` chunk runs 5–8 and 2 reach it. In `lamp_world`'s open terrain every colour reaches. Skipping the runs that can't reach would cut the caves' cost, and a chunk whose coloured lamps all stop short could keep white's cost. Neither what a run that doesn't reach costs, nor a test for which colours reach before running them, has been measured or designed.
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
