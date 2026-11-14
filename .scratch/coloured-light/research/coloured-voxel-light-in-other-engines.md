# Coloured voxel light in other engines

Findings for [Coloured voxel light in other engines](../issues/01-coloured-voxel-light-in-other-engines.md), gathered on 2026-11-14 from engine and mod source, the authors' own posts and issue threads. GitHub links are pinned to the commit read. Where a fact rests on a community wiki, a mod page or decompiled code, it says so.

Other engines call voxel light "block light" or "lamp light". This file says voxel light throughout, a lamp is an emissive voxel, and a component is one of red, green and blue. "Channel" keeps its meaning in `GLOSSARY.md`: sky light or voxel light.

## Short answer

- Shipped engines and mods store coloured light in five ways: **per-component levels** that each flood as a max (Cubyz, Seed of Andromeda, three Minecraft mods, Hytale); **proportional components**, where the brightest component sets the loss and the others follow (Starbound); **one level plus a colour** (Vintage Story's hue and saturation); **a level per light kind**, each kind with one colour for the whole world (ClassiCube, Jamango); and **additive fields** worked out iteratively (all-is-cubes, Photon, Terraria).
- Per-component levels are by far the most common. The authors who wrote about them (Cubyz's, Luanti's, and Seed of Andromeda's by way of Luanti) all name the same flaw: a weaker component runs out first, so a lamp's colour drifts towards its strongest component at the edge of its light. Only colours whose components are each 0 or equal to the brightest keep their hue. Cubyz has an open issue on it with no fix. They're also the only flood-fill family where a tint works per component for free.
- In this crate's kernel a run records only the round each voxel is first reached. So colour can't ride along with one run. Proportional components, one level plus a colour, and light kinds all come out as one design here: **a run per light colour**. A box with one light colour, white or not, runs once as today, and colour never changes how far light reaches or how it falls off. Per-component levels come out as three runs, and colour changes each component's reach, which goes against the map's settled notes.
- Additive fields don't fit: white would no longer look as it does today, and they settle over frames.

## The five families

| Family | Engines | Stored per voxel | How it spreads | Where colours overlap | As the light fades |
|---|---|---|---|---|---|
| Per-component levels | Cubyz, Seed of Andromeda, Colored Lights Core, Colorful Lighting, Canvas (open PR), Hytale (secondary) | 4 to 8 bits per component, sometimes beside the engine's white level | Each component loses a fixed step per voxel and takes the max of its neighbours, on its own | The brighter value wins per component, so red and green make yellow | A weaker component runs out first, so the colour drifts towards the strongest one |
| Proportional components | Starbound | Three floats | The brightest component loses a fixed step, and the others lose the same share of themselves | The brighter value wins per component | The hue holds, by design |
| One level plus a colour | Vintage Story | A 5-bit level and a hue and saturation | One level floods, and the colour comes with it | Helpers blend hue and saturation weighted by level, and keep the brighter level | The hue holds exactly: only the level falls |
| A level per light kind | ClassiCube, Jamango | A level for each kind | Each kind floods on its own, as white light does | Kinds' colours are screen-blended into a lookup table | The hue holds: each kind's colour is fixed |
| Additive fields | all-is-cubes, Photon, Terraria | RGB values, 8-bit log to 32-bit float | Repeated averaging or ray casting, settling over updates or frames | Light adds | The hue holds, apart from quantisation near black |

## Engine by engine

### Baselines with white light: Minecraft and Luanti

- **Minecraft** stores 4 bits of voxel light and 4 bits of sky light per block, as two 2048-byte arrays per 16³ section, and leaves the voxel light array out of a section no light reaches ([Minecraft Wiki, "Chunk format"](https://minecraft.wiki/w/Chunk_format), a community wiki). The three Minecraft mods below build colour on top of this.
- **Luanti** keeps light in `param1`: light with the sun in the low 4 bits and without it in the high 4 bits ([Lua API docs](https://github.com/luanti-org/luanti/blob/a10dc8dbff3b0af8295ae55df44f65b1d9ee92c7/doc/lua_api.md#L1569-L1571)). Coloured light has been requested since 2015 and isn't implemented ([Luanti, "Colored lighting"](https://github.com/luanti-org/luanti/issues/2670)). What its developers wrote is under [Luanti's discussion](#luantis-discussion).

### Per-component levels

#### Cubyz

The richest open implementation, in Zig.

- **Storage:** voxel light and sky light are each three 8-bit components in a packed `u32` per voxel, with a byte of padding ([`lighting.zig`](https://github.com/PixelGuys/Cubyz/blob/b23c693d0054bce8ad85c7f7dab3941ecb17e4df/src/renderer/lighting.zig#L19-L35)). Each chunk keeps one store for voxel light and one for sky light, palette-compressed, so its size grows with how many distinct values the chunk holds ([`lighting.zig`](https://github.com/PixelGuys/Cubyz/blob/b23c693d0054bce8ad85c7f7dab3941ecb17e4df/src/renderer/lighting.zig#L47), [`chunk_meshing.zig`](https://github.com/PixelGuys/Cubyz/blob/b23c693d0054bce8ad85c7f7dab3941ecb17e4df/src/renderer/chunk_meshing.zig#L603-L606)). Sky light is coloured too, so glass can tint it.
- **Spread:** a queue flood fill takes the max of each component separately ([`lighting.zig`](https://github.com/PixelGuys/Cubyz/blob/b23c693d0054bce8ad85c7f7dab3941ecb17e4df/src/renderer/lighting.zig#L121-L124)), and each step takes 8 from every component ([`lighting.zig`](https://github.com/PixelGuys/Cubyz/blob/b23c693d0054bce8ad85c7f7dab3941ecb17e4df/src/renderer/lighting.zig#L133-L135)), so a full component reaches 31 voxels. All three components travel in one queue entry, so there's no separate white path to compare with.
- **Lamps:** a block declares `emittedLight` as `0xRRGGBB` ([`blocks.zig`](https://github.com/PixelGuys/Cubyz/blob/b23c693d0054bce8ad85c7f7dab3941ecb17e4df/src/blocks.zig#L178-L179)). The torch is `0xa58d73`, so its red reaches about 20 voxels and its blue about 14 ([`torch.zig.zon`](https://github.com/PixelGuys/Cubyz/blob/b23c693d0054bce8ad85c7f7dab3941ecb17e4df/assets/cubyz/blocks/torch.zig.zon#L8)).
- **To the GPU:** the mesher packs sky and voxel light at 5 bits per component, 30 bits for each corner of a face, into a shared buffer ([`lighting.zig`](https://github.com/PixelGuys/Cubyz/blob/b23c693d0054bce8ad85c7f7dab3941ecb17e4df/src/renderer/lighting.zig#L464-L476)). The vertex shader combines them per component as the square root of the sum of their squares ([`chunk_vertex.vert`](https://github.com/PixelGuys/Cubyz/blob/b23c693d0054bce8ad85c7f7dab3941ecb17e4df/assets/cubyz/shaders/chunks/chunk_vertex.vert#L83-L93)). So brightness is linear in level.
- **Glow:** a separate emission texture, shown at up to four times the light ([`chunk_fragment.frag`](https://github.com/PixelGuys/Cubyz/blob/b23c693d0054bce8ad85c7f7dab3941ecb17e4df/assets/cubyz/shaders/chunks/chunk_fragment.frag#L93)). It isn't tied to the light's colour.
- **Through translucent blocks:** each block has `absorbedLight`, also `0xRRGGBB`, taken from each component per voxel crossed ([`lighting.zig`](https://github.com/PixelGuys/Cubyz/blob/b23c693d0054bce8ad85c7f7dab3941ecb17e4df/src/renderer/lighting.zig#L82-L106)). Water takes 9, 5 and 1, so light under water turns blue with depth ([`water.zig.zon`](https://github.com/PixelGuys/Cubyz/blob/b23c693d0054bce8ad85c7f7dab3941ecb17e4df/assets/cubyz/blocks/water.zig.zon#L10)). Blue glass takes 220 red, 146 green and 60 blue ([`glass/blue.zig.zon`](https://github.com/PixelGuys/Cubyz/blob/b23c693d0054bce8ad85c7f7dab3941ecb17e4df/assets/cubyz/blocks/glass/blue.zig.zon#L2)). Partial dimming and tint are the same mechanism.
- **What its authors say** ([Cubyz, hue shifting at the edge of lights](https://github.com/PixelGuys/Cubyz/issues/896), open since January 2025):
  - The maintainer holds that flood fill only works if the components are treated independently, and floated a fourth component to widen the colours that keep their hue.
  - A contributor showed orange at (31, 15, 0) fading to (16, 0, 0), which can't be told from dark red.
  - The maintainer turned down an extra bit per component: light would no longer fit one `u32`, and he had measured that even one more memory read in the vertex shader costs a lot.
  - Tone-mapping was turned down because a faint red and a far orange are the same stored value.
  - A contributor found no averaging rule that settles in a flood fill. Averaging two colours makes a lamp feed its own light back.
- **Other complaints:** a linear palette search in the light stores was slow ([Cubyz issue](https://github.com/PixelGuys/Cubyz/issues/318)).

#### Seed of Andromeda

Its public repository is MIT-licensed, but the light engine survives only commented out, under a note asking whether to keep it ([`VoxelLightEngine.cpp`](https://github.com/RegrowthStudios/SoACode-Public/blob/c3ddd69355b534d5e70e2e6d0c489b4e93ab1ffe/SoA/VoxelLightEngine.cpp#L7)).

- **Storage:** voxel light is 16 bits, 5 bits each for red, green and blue ([`VoxelBits.h`](https://github.com/RegrowthStudios/SoACode-Public/blob/c3ddd69355b534d5e70e2e6d0c489b4e93ab1ffe/SoA/VoxelBits.h#L18-L23)). Sky light is a separate level up to 31 ([`ChunkMesher.cpp`](https://github.com/RegrowthStudios/SoACode-Public/blob/c3ddd69355b534d5e70e2e6d0c489b4e93ab1ffe/SoA/ChunkMesher.cpp#L26-L28)). The 0fps post below says Seed of Andromeda used 4 bits each for red, green, blue and sky, which doesn't match this code.
- **Spread:** a queue flood fill that takes the max per component and takes 1 from each component per step ([`VoxelLightEngine.cpp`](https://github.com/RegrowthStudios/SoACode-Public/blob/c3ddd69355b534d5e70e2e6d0c489b4e93ab1ffe/SoA/VoxelLightEngine.cpp#L521-L566)).
- **Through translucent blocks:** each block's `colorFilter` scales each component's level as light enters it ([`VoxelLightEngine.cpp`](https://github.com/RegrowthStudios/SoACode-Public/blob/c3ddd69355b534d5e70e2e6d0c489b4e93ab1ffe/SoA/VoxelLightEngine.cpp#L541-L546)).
- **Falloff:** brightness was `0.95^(31 - level) - 0.2` ([`ChunkMesher.cpp`](https://github.com/RegrowthStudios/SoACode-Public/blob/c3ddd69355b534d5e70e2e6d0c489b4e93ab1ffe/SoA/ChunkMesher.cpp#L26-L28), [`ChunkMesher.cpp`](https://github.com/RegrowthStudios/SoACode-Public/blob/c3ddd69355b534d5e70e2e6d0c489b4e93ab1ffe/SoA/ChunkMesher.cpp#L1282-L1285)), nearly exponential. See [hue as light fades](#hue-as-light-fades) for why that matters.
- **The author's post on coloured light,** "Fast flood fill lighting in a blocky voxel game, part 2" on seedofandromeda.com, couldn't be read: the site is offline and the web archive couldn't be fetched from here. A Luanti developer's summary of it is the only account found (secondary): only seven colours keep their hue under this scheme ([Luanti, "Coloured lighting"](https://github.com/luanti-org/luanti/issues/5202)).

#### Colored Lights Core (Minecraft 1.7.10 core mod)

A Forge core mod that replaced the light engine ([README](https://github.com/CptSpaceToaster/CptsModdingLight/blob/d524c02a1de6ac08f4bde39eae004f2c1f6c9a4e/README.md)).

- **Storage:** Minecraft's own 4-bit level stays, and each section gains three more 4-bit arrays for red, green and blue ([`ChunkStorageRGB.java`](https://github.com/CptSpaceToaster/CptsModdingLight/blob/d524c02a1de6ac08f4bde39eae004f2c1f6c9a4e/src/main/java/coloredlightscore/server/ChunkStorageRGB.java#L46-L48)). A lamp declares 4 bits per component, and its white level is its brightest component ([`CLApi.java`](https://github.com/CptSpaceToaster/CptsModdingLight/blob/d524c02a1de6ac08f4bde39eae004f2c1f6c9a4e/src/main/java/coloredlightscore/src/api/CLApi.java#L56-L80)). An uncoloured lamp's level is copied into all three components, so it stays white ([`CLWorldHelper.java`](https://github.com/CptSpaceToaster/CptsModdingLight/blob/d524c02a1de6ac08f4bde39eae004f2c1f6c9a4e/src/main/java/coloredlightscore/src/helper/CLWorldHelper.java#L106)).
- **Spread:** all four values travel in one 20-bit word with a guard bit above each, so a single subtract and mask compares every component at once ([`CLWorldHelper.java`](https://github.com/CptSpaceToaster/CptsModdingLight/blob/d524c02a1de6ac08f4bde39eae004f2c1f6c9a4e/src/main/java/coloredlightscore/src/helper/CLWorldHelper.java#L223-L226)). This is the packed-word trick that [0fps](https://0fps.net/2018/02/21/voxel-lighting/) describes for 4-bit components.
- **Through translucent blocks:** per-component opacity stayed a to-do, and Minecraft's own opacity was used ([`CLWorldHelper.java`](https://github.com/CptSpaceToaster/CptsModdingLight/blob/d524c02a1de6ac08f4bde39eae004f2c1f6c9a4e/src/main/java/coloredlightscore/src/helper/CLWorldHelper.java#L139-L160)).

#### Colorful Lighting (Minecraft 1.21 client mod)

A client-side mod over Minecraft's own light, now archived in favour of a Sodium-compatible fork ([README](https://github.com/erykczy/colorful-lighting/blob/5068a5b50d6b2e7d96f0185cb631873a2ea0a0ee/README.md)).

- **Storage:** 12 bits per voxel, 4 per component, in an array a section allocates only when coloured light first reaches it ([`ColoredLightSection.java`](https://github.com/erykczy/colorful-lighting/blob/5068a5b50d6b2e7d96f0185cb631873a2ea0a0ee/src/main/java/me/erykczy/colorfullighting/common/ColoredLightSection.java#L9-L40)). Sky light stays Minecraft's.
- **Lamps:** a lamp's colour is scaled by its Minecraft level over 15 ([`Config.java`](https://github.com/erykczy/colorful-lighting/blob/5068a5b50d6b2e7d96f0185cb631873a2ea0a0ee/src/main/java/me/erykczy/colorfullighting/common/Config.java#L31-L41)).
- **Spread:** a queue flood fill on a background thread, the max per component ([`ColoredLightEngine.java`](https://github.com/erykczy/colorful-lighting/blob/5068a5b50d6b2e7d96f0185cb631873a2ea0a0ee/src/main/java/me/erykczy/colorfullighting/common/ColoredLightEngine.java#L370-L375)). Each step takes at least 1 from every component, or the block's own light blocking ([`ColoredLightEngine.java`](https://github.com/erykczy/colorful-lighting/blob/5068a5b50d6b2e7d96f0185cb631873a2ea0a0ee/src/main/java/me/erykczy/colorfullighting/common/ColoredLightEngine.java#L387-L394)). It runs beside Minecraft's own white light, so it's a cost on top of it.
- **Through translucent blocks:** a block's filter caps each component rather than scaling it, so red glass lets red through at full level and stops green and blue ([`ColoredLightEngine.java`](https://github.com/erykczy/colorful-lighting/blob/5068a5b50d6b2e7d96f0185cb631873a2ea0a0ee/src/main/java/me/erykczy/colorfullighting/common/ColoredLightEngine.java#L388-L394)). It tints voxel light only.

#### Canvas's coloured flood fill (open pull request)

A pull request to the Canvas renderer mod, open since June 2023 ([Canvas, "Colored flood fill lighting"](https://github.com/vram-guild/canvas/pull/428)).

- **Storage:** 16 bits per voxel, 4 per component plus 4 flag bits that mark lamps and occluders ([`LightOp.java`](https://github.com/spiralhalo/canvas/blob/4cae0d24c7dab4743027f2a93633b4937e01218b/src/main/java/grondag/canvas/light/color/LightOp.java#L24-L27)). It's uploaded to a sparse `RGBA4` texture ([`LightDataTexture.java`](https://github.com/spiralhalo/canvas/blob/4cae0d24c7dab4743027f2a93633b4937e01218b/src/main/java/grondag/canvas/light/color/LightDataTexture.java#L36-L40)).
- **Lamps:** a lamp's components are its level times its colour, rounded to 0 to 15 ([`FloodFillBlockLight.java`](https://github.com/spiralhalo/canvas/blob/4cae0d24c7dab4743027f2a93633b4937e01218b/src/main/java/grondag/canvas/light/api/impl/FloodFillBlockLight.java#L52-L55)).
- **Spread:** the max per component, and 1 from each component per step ([`LightOp.java`](https://github.com/spiralhalo/canvas/blob/4cae0d24c7dab4743027f2a93633b4937e01218b/src/main/java/grondag/canvas/light/color/LightOp.java#L91-L109)). It runs on the render thread when a region is rebuilt.
- **Through translucent blocks:** no per-component tint was found in its light code.
- **Note:** the PR warns that coloured light reads as a different brightness than white at the same level.

#### Hytale (secondary)

Hytale's server source is shared only with selected creators ([The Escapist](https://www.escapistmagazine.com/news-hytale-releases-server-source-code-assets-shared-source-programme/)). Everything here is from an unofficial site that documents behaviour visible in the decompiled server ([Hytale Server Docs (Unofficial), "Lighting System"](https://hytale-docs.pages.dev/modding/content/lighting)):

- **Storage:** four 4-bit values per voxel: red, green, blue and sky light, kept in an octree-like buffer per section.
- **Spread:** a flood fill per section, then from neighbouring sections. Each step takes at least 1 from every value, sky light included.
- **Through translucent blocks:** semi-transparent and cut-out blocks take 1 more from every value. No tint is described.
- **Mixing:** a voxel that has both a block's light and a fluid's takes the max per component.

Nothing was found on how the client renders it.

#### Luanti's discussion

Luanti never built coloured light, but its developers put numbers on it ([Luanti, "Coloured lighting"](https://github.com/luanti-org/luanti/issues/5202), [Luanti, "Colored lighting"](https://github.com/luanti-org/luanti/issues/2670)):

- **Hue:** with per-component levels only seven colours keep their hue (white, red, yellow, green, cyan, blue and magenta), citing Seed of Andromeda's post. Another developer's example: a light green at (10, 14, 10) is a faint pure green 10 nodes out.
- **Cost:** three components would make a node 6 bytes instead of 4, wasted on nodes light can't enter, and a white lamp would be worked out three times over.
- **Translucency:** realistic stained glass needs coloured sky light.
- **Why it wasn't done:** a maintainer judged that it needs a new light system and a new map format for an effect most scenes wouldn't use.
- **Alternatives raised:** shader point lights were floated instead, and turned down for cost in a forward renderer once about 20 lamps are in view.

### Proportional components: Starbound

Read from OpenStarbound, which is derived from Starbound 1.4.4's leaked source ([README](https://github.com/OpenStarbound/OpenStarbound/blob/aa42287f3e603d27bad00e037962e577b57edd9f/README.md)). The code below is in its first import, so it isn't the fork's own work.

- **Storage:** three floats per cell, worked out each frame for the screen plus a border as wide as light spreads ([`StarCellularLightArray.hpp`](https://github.com/OpenStarbound/OpenStarbound/blob/aa42287f3e603d27bad00e037962e577b57edd9f/source/base/StarCellularLightArray.hpp#L38-L48), [`borderCells`](https://github.com/OpenStarbound/OpenStarbound/blob/aa42287f3e603d27bad00e037962e577b57edd9f/source/base/StarCellularLightArray.hpp#L222-L225)). The same template has a scalar form for white light.
- **Spread:** the drop per cell is divided by the brightest component and then taken from each component in proportion to its size. The code's own comment says this keeps colour ratios, so hues don't drift as light spreads ([`StarCellularLightArray.hpp`](https://github.com/OpenStarbound/OpenStarbound/blob/aa42287f3e603d27bad00e037962e577b57edd9f/source/base/StarCellularLightArray.hpp#L21-L36), [`spread` and `subtract`](https://github.com/OpenStarbound/OpenStarbound/blob/aa42287f3e603d27bad00e037962e577b57edd9f/source/base/StarCellularLightArray.hpp#L164-L190)). So a lamp's reach is set by its brightest component.
- **Mixing:** spread light takes the max per component ([`max`](https://github.com/OpenStarbound/OpenStarbound/blob/aa42287f3e603d27bad00e037962e577b57edd9f/source/base/StarCellularLightArray.hpp#L204-L206)). Point lights can add instead (`pointAdditive`).
- **Through translucent blocks:** obstacles dim light by one scalar, with no tint.
- **Cost:** OpenStarbound notes that brighter caps widen the border that must be worked out, and that spread light's cap costs more than point light's ([`lighting.md`](https://github.com/OpenStarbound/OpenStarbound/blob/aa42287f3e603d27bad00e037962e577b57edd9f/doc/json/openstarbound/lighting.md)).

### One level plus a colour: Vintage Story

The light engine is closed. The public API shows the stored form.

- **Storage:** light is a `ushort` per voxel: sky level in bits 0 to 4, voxel light level in bits 5 to 9, and a 6-bit hue in bits 10 to 15 ([`ColorUtil.cs`](https://github.com/anegostudios/vsapi/blob/63d33f70018c606fb0f70ead900fe90625fbd7b2/Math/ColorUtil.cs#L1065-L1076)). A saturation is stored beside it, and 0 means white ([`IWorldChunk.cs`](https://github.com/anegostudios/vsapi/blob/63d33f70018c606fb0f70ead900fe90625fbd7b2/Common/API/IWorldChunk.cs#L171-L175), [`ColorUtil.cs`](https://github.com/anegostudios/vsapi/blob/63d33f70018c606fb0f70ead900fe90625fbd7b2/Math/ColorUtil.cs#L1070-L1073)).
- **Lamps:** a lamp declares hue, saturation and brightness ([`Collectible.cs`](https://github.com/anegostudios/vsapi/blob/63d33f70018c606fb0f70ead900fe90625fbd7b2/Common/Collectible/Collectible.cs#L262-L265)). The lead developer's chart of every value has 64 hues, 8 saturations and 32 brightnesses, 16,384 in all ([VS Light Wheel](http://tyron.at/vs/vslightwheel.html)). The 2016 post that launched coloured light promised seamless blending of 16k colours ([Vintage Story, "New plants and colored lights"](https://www.vintagestory.at/blog.html/news/new-plants-and-colored-lights-r35/)).
- **Rendering:** hue and saturation turn into RGB with the level as HSV's value, through a brightness table ([`ColorUtil.cs`](https://github.com/anegostudios/vsapi/blob/63d33f70018c606fb0f70ead900fe90625fbd7b2/Math/ColorUtil.cs#L1032-L1105)). Only the level falls with distance, so a lamp's hue and saturation hold exactly as it fades.
- **Mixing:** two helpers exist. `MergeLightHSV` averages hue and saturation weighted by level and keeps the brighter level. `ColorCombineHSV` does the same blend in RGB and converts back ([`ColorUtil.cs`](https://github.com/anegostudios/vsapi/blob/63d33f70018c606fb0f70ead900fe90625fbd7b2/Math/ColorUtil.cs#L450-L499)). Which one the closed engine uses for voxel light couldn't be confirmed. A mod page says light from several held items has its hues blended (secondary: [Unmixed Glow](https://mods.vintagestory.at/unmixedglow)). By my arithmetic, averaging the hue number itself, as `MergeLightHSV` does, runs from red to blue through green, which the RGB helper avoids.
- **Through translucent blocks:** light absorption is one scalar per block, and above 32 it blocks all light ([`Block.cs`](https://github.com/anegostudios/vsapi/blob/63d33f70018c606fb0f70ead900fe90625fbd7b2/Common/Collectible/Block/Block.cs#L113-L116)). The base game doesn't seem to tint light through glass: a mod adds tinted sunlight through a shadow pass (secondary: [Transmittance](https://mods.vintagestory.at/transmittance)). Players report that a lantern's light colour follows the glass fitted to it (secondary: [forum](https://www.vintagestory.at/forums/topic/12055-what-happened-to-colored-lights)).
- **Glow:** an 8-bit glow level per vertex, separate from the light's hue ([`VertexFlags.cs`](https://github.com/anegostudios/vsapi/blob/63d33f70018c606fb0f70ead900fe90625fbd7b2/Common/Collectible/Block/VertexFlags.cs#L125-L130)).

### A level per light kind: ClassiCube and Jamango

- **ClassiCube's fancy lighting** stores one byte per voxel: a 4-bit lamp level and a 4-bit lava level ([`Lighting.h`](https://github.com/ClassiCube/ClassiCube/blob/bbd38b9c7ecda4f4fd0f945d413fa9497ed79766/src/Lighting.h#L32-L38)).
  - **Colours:** each kind has one colour for the whole world, set by the environment ([`FancyLighting.c`](https://github.com/ClassiCube/ClassiCube/blob/bbd38b9c7ecda4f4fd0f945d413fa9497ed79766/src/FancyLighting.c#L36-L77)).
  - **Mixing:** a 16 × 16 table per face shade screen-blends the two kinds' colours, at each pair of levels, with the sunlit or shadow colour. Each kind falls off on a cosine curve.
  - **Cost:** a chunk with no light holds no light array ([`FancyLighting.c`](https://github.com/ClassiCube/ClassiCube/blob/bbd38b9c7ecda4f4fd0f945d413fa9497ed79766/src/FancyLighting.c#L500-L505)).
  - **Sky light:** only whether a voxel is above the highest solid.
- **Jamango** gives every lamp in a world one colour, which the world's creator picks. Its authors say they traded multi-colour light for speed and smooth frame rates ([Jamango, "How we rebuilt Jamango's lighting"](https://jamango.io/blog/how-we-rebuilt-jamangos-lighting-from-the-ground-up), July 2025). They give no storage details.

### Additive fields: all-is-cubes, Photon and Terraria

- **all-is-cubes** (Rust):
  - **Storage:** RGB per cube, 8-bit log-encoded per component (each step about 7% brighter than the last), plus a status byte: 4 bytes in all ([`data.rs`](https://github.com/kpreid/all-is-cubes/blob/b6786418d0e5c397c083f5b4c331e2bdd0abe5b2/all-is-cubes/src/space/light/data.rs#L48-L69), [size test](https://github.com/kpreid/all-is-cubes/blob/b6786418d0e5c397c083f5b4c331e2bdd0abe5b2/all-is-cubes/src/space/light/data.rs#L465)). It goes to the GPU as an `Rgba8Uint` 3D texture that the fragment shader reads by position ([`light_texture.rs`](https://github.com/kpreid/all-is-cubes/blob/b6786418d0e5c397c083f5b4c331e2bdd0abe5b2/all-is-cubes-gpu/src/light_texture.rs#L192)). This crate's route already cites that texture.
  - **Spread:** each cube's light is the weighted average of a fixed chart of rays. A ray picks up a struck face's emission plus its colour times the light stored in front of it, so light bounces off coloured surfaces and takes their colour ([`updater.rs`](https://github.com/kpreid/all-is-cubes/blob/b6786418d0e5c397c083f5b4c331e2bdd0abe5b2/all-is-cubes/src/space/light/updater.rs#L826-L833), [`finish`](https://github.com/kpreid/all-is-cubes/blob/b6786418d0e5c397c083f5b4c331e2bdd0abe5b2/all-is-cubes/src/space/light/updater.rs#L936-L948)). It updates from a queue on a time budget and converges over many updates. Differences of one step are left unqueued, since chasing them could loop forever ([`updater.rs`](https://github.com/kpreid/all-is-cubes/blob/b6786418d0e5c397c083f5b4c331e2bdd0abe5b2/all-is-cubes/src/space/light/updater.rs#L352-L358)).
  - **Through translucent blocks:** a partly transparent block dims a ray by one scalar alpha, so transmitted light isn't tinted per component. Only reflected light takes colour ([`updater.rs`](https://github.com/kpreid/all-is-cubes/blob/b6786418d0e5c397c083f5b4c331e2bdd0abe5b2/all-is-cubes/src/space/light/updater.rs#L860-L887)).
  - **The author's view:** a note says to measure later whether coloured light is worth its compute and storage, perhaps moving to a luma-and-chroma encoding ([`data.rs`](https://github.com/kpreid/all-is-cubes/blob/b6786418d0e5c397c083f5b4c331e2bdd0abe5b2/all-is-cubes/src/space/light/data.rs#L59-L62)).
- **Photon** (a Minecraft shader pack, under Iris):
  - **Storage:** a 64³ to 256³ volume of block ids around the camera, and two `rgba16f` light volumes it swaps between ([`shaders.properties`](https://github.com/sixthsurge/photon/blob/2a3d502c17d2e310412c717be47241f014cba7fc/shaders/shaders.properties#L430-L450)).
  - **Spread:** each frame, every voxel becomes the average of itself and its six neighbours, times its tint, plus its emission ([`floodfill.glsl`](https://github.com/sixthsurge/photon/blob/2a3d502c17d2e310412c717be47241f014cba7fc/shaders/include/lighting/lpv/floodfill.glsl#L89-L113)). Light adds, a tinted block scales each component, and the field settles over frames. Light ends at the volume's edge.
- **Terraria:**
  - **Storage:** tModLoader's patches against the decompiled game show a `Vector3` per tile ([`LightMap.cs.patch`](https://github.com/tModLoader/tModLoader/blob/2f3104abe840dc377b27836b6bd4bce879f7133c/patches/tModLoader/Terraria/Graphics/Light/LightMap.cs.patch)).
  - **Through water:** light decays per component through water ([`LightingEngine.cs.patch`](https://github.com/tModLoader/tModLoader/blob/2f3104abe840dc377b27836b6bd4bce879f7133c/patches/tModLoader/Terraria/Graphics/Light/LightingEngine.cs.patch)).
  - **Cost:** the community wiki rates coloured mode as the most demanding and an all-white mode on the same engine as cheaper. Coloured light arrived in 1.1 (secondary: [Terraria Wiki, "Lighting mode"](https://terraria.wiki.gg/wiki/Lighting_mode)).
  - **Not found:** how its colours mix, from a primary source.

## Across the engines

### Hue as light fades

- **Per-component levels:** the hue drifts once a weaker component runs out. How far it drifts first depends on the falloff curve:
  - **Linear curve** (Cubyz, where brightness is level over 31): the ratio between components changes at every step, so orange drifts to red long before it's dim. That's Cubyz's complaint.
  - **Exponential curve** (Seed of Andromeda's, and this crate's default, below): one level less multiplies every component by the same factor, so the ratio holds until the weaker component reaches 0. Then the colour snaps to the stronger component for the last few voxels.

  This is arithmetic on the engines' formulas, not a claim their authors make.
- **Proportional components** (Starbound) and **one level plus a colour** (Vintage Story): the hue holds by construction.
- **Light kinds** (ClassiCube): each kind's colour is fixed.
- **Additive fields:** the hue holds until quantisation near black. all-is-cubes quantises each component on its own log scale.

### Banding

- **Per-component levels at 4 bits:** a lamp's colour rounds to 16 values per component, and a dim lamp has fewer, since its components can't exceed its level (Canvas, Colorful Lighting, Colored Lights Core).
- **Cubyz:** stores 8 bits but shows 5.
- **Vintage Story:** quantises hue to 64 and saturation to 8, and brightness bands as white light does.
- **Floats and log-encoded fields:** don't band visibly.
- **What wasn't found:** any author's complaint about colour banding as such. The complaints are all about hue drift.

### Glow

- **Separate from the light colour:** in the engines whose glow was checked, a lamp's own faces glow from something other than the propagated light. Cubyz uses an emission texture, Vintage Story a per-vertex glow level, and ClassiCube full-bright blocks. None of them ties the glow to the lamp's light colour.
- **Left to shaders:** Canvas marks a lamp's cell with a flag bit and leaves its look to shaders.
- **Not checked:** the glow in the other engines.

### Light through tinted translucent voxels

| Engine | Voxel light | Sky light | How |
|---|---|---|---|
| Cubyz | Yes | Yes | Subtracts per component per voxel, which also does partial dimming (water) |
| Seed of Andromeda | Yes | Not seen | Scales each component's level |
| Colorful Lighting | Yes | No | Caps each component |
| Photon | Yes | Not checked | Multiplies each component |
| Terraria | Through water | Not found | Per-component decay |
| all-is-cubes | Dimmed, not tinted | Dimmed, not tinted | Scalar alpha; colour only from reflection |
| Vintage Story, Starbound, Hytale, Canvas, Colored Lights Core | No | No | Scalar opacity |

Every engine that tints does it per component on per-component storage. No engine that stores proportional components or one level plus a colour tints.

### Artifacts, rewrites and reported costs

- **Hue drift at the edge of light:** Cubyz, Luanti and Seed of Andromeda all report it, by way of Luanti's summary for the last.
- **Bits per component capped by upload cost:** Cubyz.
- **Storage growth:** half as much again per node in Luanti, and three extra 4-bit arrays per section in Colored Lights Core.
- **Doing white light's work three times:** Luanti's estimate.
- **A coloured mode that costs more than white:** Terraria (secondary).
- **One colour per world, chosen for frame rate:** Jamango.
- **Colour engines that ended up dormant or archived:** the light engine in Seed of Andromeda's public code is commented out; Colorful Lighting was archived for a fork; Canvas's PR is still open.
- **No numbers:** none of the authors gave measured costs of colour against white.

## What fits this crate

### How light works here today

- **What a run records:** the kernel spreads one channel by bit dilation and records, for each voxel, the round it's first reached. Its level is `scale.level(round)`, and a lamp joins in the round its level says ([`kernel.rs`](../../../src/light/kernel.rs), [`levels.rs`](../../../src/light/levels.rs)). So a run already takes the max over lamps of level minus distance, and it holds no other value per voxel. A second lamp's front that arrives later leaves no trace.
- **What a solve reads:** only voxels. It never reads a neighbour's light ([`solver.rs`](../../../src/light/solver.rs)), so anything worked out from a box is a pure function of it, with no question of settling.
- **The level scale:** at 16 levels and reach 14 a level falls by one per step, and the default voxel falloff is `0.7^(15 - level)` ([`config.rs`](../../../src/light/config.rs)).
- **Slots and tags:** a slot holds sky light and voxel light per texel, as one `R8Uint` byte at 16 levels and `Rg8Uint` at 256. A lamp's own cell holds the top level as a marker and the lamp's level in its sky half. A tag with no slot holds one level per channel in bits 0 to 15 ([`slot.rs`](../../../src/light/slot.rs), [`route.rs`](../../../src/light/route.rs)).
- **The shader:** it already carries voxel light as a `vec3` and applies `voxel_falloff` per component, with the glow multiplied by the face's diffuse colour ([`light.wesl`](../../../src/shaders/light.wesl)).
- **The light spec:** it asks that more voxel light channels be more runs of the same kernel, and that `light_at`'s `voxel` stay the brightest ([spec](../../light-propagation/spec.md)).

### Per-component levels here

Three runs of voxel light, one per component. Each lamp's seed starts at its component's level.

- **Hue:** the falloff curve maps levels to brightness, so a component holding half the brightness has to start about 2 levels lower under the default curve, and about 7 or 8 lower under `LevelCurve::linear`.
  - **Default curve:** the ratio holds until the weaker component ends about 2 steps short. The stronger component is then at about 1% of its peak or less, so the drift is confined to a faint fringe.
  - **Linear curve:** the ratio slides at every step, which is the Cubyz complaint.
  - **Changing the curve:** the levels encode the colour through the curve, so a look change (which today needs no relight) shifts every lamp's hue unless coloured chunks relight. A table from `LevelCurve::from_table` may have no inverse at all.
- **Banding:** at 16 levels under the default curve a component's share comes only in powers of 0.7 (1, 0.7, 0.49, 0.34, and so on). At 256 levels the steps are about 2%.
- **Mixing:** the brighter value wins per component, as white light's does today. It's continuous, with no new kind of seam.
- **Reach:** colour changes each component's reach, which the map's notes rule out.

### A run per light colour here

Proportional components, one level plus a colour, and light kinds all reduce to this in a bit-dilation kernel. Group a box's lamps by light colour, run the kernel once per group, keep the max of the groups' levels as today's level, and work out a colour from the groups' levels.

- **Hue and reach:** each group's reach and falloff are its lamps' levels, so the hue holds exactly as light fades and colour never changes how far light reaches.
- **Mixing:** this needs a rule, and the prototype in [How coloured light looks](../issues/03-how-coloured-light-looks.md) is where it belongs. The candidates the engines suggest:
  - the max of colour times brightness per component, as in Starbound;
  - a blend weighted by brightness, as in Vintage Story's RGB helper.

  Both are continuous. Two choices read as bugs. Letting the brighter colour win per voxel puts a hard seam where the two levels are equal. Averaging hue numbers sends red and blue through green.
- **Cost of the mixing rule:** a rule weighted by brightness bakes the falloff curve into the stored colour, so a falloff change would relight coloured chunks. A rule weighted by level difference wouldn't.
- **What a single run can't do:** carrying colour along with one run gives the hard seam, for the reason above.

### Additive fields here

- **White:** two white lamps would add, so white wouldn't look as it does today.
- **Reach:** reach and the level scale stop meaning what they do.
- **Timing:** the field settles over frames, which doesn't fit landing a single edit's light within 2 frames.
- **Verdict:** ruled out.

### Against the four criteria

| | Per-component levels | A run per light colour |
|---|---|---|
| **A white lamp looks and reads as today** | Yes. Three equal components through the existing `vec3` path give today's image, and `light_at` reads the brightest. | Yes. A box of white lamps runs once, as today. Where white and coloured lamps meet, the level is still the max, so `light_at` is unchanged. Only the colour there is new. |
| **A chunk with no coloured lamp near it pays nothing** | Yes, if no lamp in the box is coloured: one run, stored and uploaded as today. Any coloured lamp in the box costs three runs and three components per voxel, however many colours there are. The solver already gathers each box's emissive voxels, so the check is a scan of a list it has. | Yes, and more: a box with one light colour of any hue runs once and needs only one colour for the whole chunk, in the tag or a table. Cost grows with the number of distinct light colours in the box. |
| **A later tint fits without a rewrite** | Best of the two. A tinted voxel that's open for red and closed for green and blue is a per-component open mask, the way Cubyz, Seed of Andromeda and Colorful Lighting tint. Partial dimming, such as light losing levels through water, has no counterpart in bit dilation yet: every open voxel costs one round. Sky light under tinted glass would become three runs. | Possible, not shown. Light leaving a tinted voxel becomes a new light colour, so the untinted colour's run would hand its arrival rounds at the tinted voxels to the new colour's run, much as sky fill's seeds join a run at a set round. Groups multiply with tints crossed, and partial dimming is as open as on the left. |
| **A cap on distinct light colours** | Not needed. Colour precision is tied to level precision and to the falloff curve instead. | Bounds the runs per box and turns the stored colour into an index. Behind a type parameter like `MaterialIndex`, as the map's notes ask, a world with one light colour is the cap of 1 that ClassiCube and Jamango ship. Without a cap, a chunk seeing more than one light colour stores a colour value per voxel. |

Both families could write the same stored form: the brightest level, plus a colour wherever a chunk sees more than one. So a box could take whichever path suits it, for example per-component runs where tinted voxels are. That's an option for [The representation of coloured voxel light](../issues/05-the-representation-of-coloured-voxel-light.md), not something any engine ships.

### What this leaves for later tickets

- **For [Cheap multi-component propagation and storage](../issues/02-cheap-multi-component-propagation-and-storage.md):**
  - the cost of three component runs against N colour runs in a box;
  - what per-voxel colour costs beside today's 1 and 2 bytes;
  - whether a packed-word max with guard bits, as in Colored Lights Core and 0fps, ever beats three bit-dilation runs.
- **For [How coloured light looks](../issues/03-how-coloured-light-looks.md):**
  - the mixing rule;
  - the glow's colour: the light colour, the face's own colour (as today, and in the engines whose glow was checked), or both;
  - whether a per-component fringe at 1% shows at all under the default curve.
- **For [The representation of coloured voxel light](../issues/05-the-representation-of-coloured-voxel-light.md):**
  - how a lamp's own cell gives its colour;
  - what sub-grids' lamps max per component;
  - whether "channel" grows to cover components.

## Not found, or resting on secondary sources

- **Seed of Andromeda's post on coloured light:** couldn't be read. Its claim comes through a Luanti developer's summary.
- **Vintage Story:** its light engine is closed. How it mixes and spreads hue is inferred from public API helpers and a mod page, and the lack of tint through glass from a mod page.
- **Hytale:** everything rests on unofficial docs from the decompiled server. Nothing was found on its client's rendering.
- **Terraria:** rests on tModLoader's patches and the community wiki. Its mixing rule and update algorithm weren't found.
- **Minecraft's stored light layout:** from the community wiki.
- **Reported costs:** no engine reported a measured cost of colour against white.
