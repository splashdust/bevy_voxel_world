// Propagated light for chunk faces
//
// This module reads the light of a lit world for chunk faces, and applies it
// to the PBR input of Bevy. Import it as `bevy_voxel_world::light`. The
// material of a lit world holds the light texture of the world at binding 200
// and the look of the world at binding 201. The vertex shaders of the crate
// output the local position and local normal of each face in every world,
// and the tag of the chunk under the `VOXEL_LIGHT` shader def. A fragment
// shader reads the light of a face with `sample_face_light`, and applies it
// to the PBR input with `apply_voxel_light`, the one place light meets the
// PBR calculation. A shader without PBR can use `sky_brightness` and
// `voxel_brightness` instead.
//
// Light is read from the position and normal of a face alone, so it reaches
// any mesh. A cube face reads the voxel in front of it, and a face off the
// voxel grid reads the 8 voxels around the point in front of it (see
// `read_front`). A custom vertex shader outputs what the vertex shaders of
// the crate do. In every world it outputs the centroid-sampled local position
// at location 9 and the local normal at location 10. Under `VOXEL_LIGHT` it
// outputs the flat tag at location 11, and under `SUB_GRIDS` the
// centroid-sampled position in the grid of the world at location 12. The
// docs of the `rendering` module of the crate list every output and its
// space.
//
// The tag is the `MeshTag` of the chunk. With bit 31 set, bits 0-23 name the
// slot of the chunk in the light texture. A slot is 34 texels a side and
// holds the voxels of the chunk in its padded local coordinates, with texel
// rows along z. Without bit 31, the low bits hold the one sky light level and
// the one voxel light level that every face of the chunk sees.
//
// The `steps_256` of the look says how levels are stored:
//
// - At 16 levels a texel is one `R8Uint` byte, with sky light in its low
//   nibble and voxel light in its high one. A tag with no slot holds them the
//   same way in bits 0-7.
// - At 256 levels a texel is `Rg8Uint`, with sky light in red and voxel light
//   in green. A tag with no slot holds sky light in bits 0-7 and voxel light
//   in bits 8-15.
//
// Levels at 256 are read on the 0 to 15 scale, in steps of 1/17. The cell of
// an emissive voxel holds the top level, 15 or 255, in its voxel light half,
// and the level the voxel gives off in its sky light half.
//
// A chunk above LOD level 0 with a slot holds its LOD level L in bits 24-27.
// The cells of its slot are `1 << L` voxels wide, so its local position is
// divided by that width, keeping the corner of its first cell at (1, 1, 1).
//
// Bit 30 marks the chunk of a sub-grid, with or without a slot. A tag with a
// slot sets bit 29 when the entry of the slot in the colour texture holds a
// light colour other than white, or names a colour slot. A tag with no slot
// holds its light colour in bits 16-29. Every read of a field masks the field
// to its width, so a tag reads the same slot, LOD level and levels whatever
// its colour bits and sub-grid bit hold.
//
// A world with an emissive material whose light colour isn't white sets the
// `LIGHT_COLOR` shader def, and its material holds the colour texture of the
// world at binding 203. The colour texture holds one `R16Uint` texel per
// slot, the entry of that slot, at the place of the slot in the grid of
// slots. Past the entries along z lie the colour slots. An entry holds the
// packed light colour of its chunk, or names the colour slot of the chunk. A
// colour slot holds a packed light colour per cell, laid out as a slot is.
//
// Only under `LIGHT_COLOR` does the shader read colours, in `color_entry` and
// `cell_color`, and mix them, in `mix_color` and `blend_color`. Without it
// every colour is white, so the shader of a white world reads and mixes no
// colours. In a coloured world, the extra cost of a face depends on its
// chunk:
//
// - A chunk with no slot, or with a slot and a tag that says white, costs a
//   little arithmetic on its tag.
// - A chunk whose entry holds its light colour costs one more texture read.
// - A chunk with a colour slot costs that read, and one more read for each
//   cell whose voxel light the face reads.
//
// With the `sub_grids` feature of the crate, a lit world sets the `SUB_GRIDS`
// shader def, and its material holds the tag map of the world at binding
// 202. The tag map is a hash table from a level-0 world chunk position to the
// tag the world shows there, for the world chunks around its sub-grids.
// `sample_face_light` lights a face of a sub-grid with the light of the world
// where the face is, read through the tag map by `world_light`, as well as
// with the light of the sub-grid itself. It reads the position of the face in
// the grid of the world from the world position Bevy gives it, so it assumes
// that the `WorldRoot` of the world is at the identity. A shader that still
// calls `sample_light` lights the faces of a sub-grid from the light of the
// sub-grid alone: they don't darken in a cave or brighten beside a lamp of
// the world. Each fragment of a face of a sub-grid costs one lookup in the
// tag map, and one more `read_front` with its cell reads. A face of a world
// chunk costs nothing more.
//
// References:
//
// - `BlockMeshPart` in Terasology, which lights a vertex from 8 points around
//   it, so a vertex anywhere gets light:
//   https://github.com/MovingBlocks/Terasology/blob/develop/engine/src/main/java/org/terasology/engine/world/block/shapes/BlockMeshPart.java
// - The lighting of Cubyz, which blends the light of the 8 cells around a
//   vertex off the grid:
//   https://github.com/PixelGuys/Cubyz/blob/master/src/renderer/lighting.zig
// - The smooth lighting of Luanti, which averages the cells around a corner,
//   leaving out cells that don't carry light:
//   https://github.com/luanti-org/luanti/blob/master/src/client/content_mapblock.cpp
// - The light propagation of tesserae, where sky light scales the occlusion
//   terms of Bevy and `directional_shadow_factor`, and voxel light adds
//   through the emissive output:
//   https://github.com/mkappworks-dev/tesserae/pull/3

import bevy_pbr::render::{
    pbr_types::PbrInput,
    mesh_view_bindings as view_bindings,
    mesh_view_types,
    mesh_types::MESH_FLAGS_SHADOW_RECEIVER_BIT,
    pbr_functions::{calculate_diffuse_color, calculate_F0_dielectric},
    pbr_lighting as lighting,
    pbr_lighting::{LAYER_BASE, LAYER_CLEARCOAT},
};

// How light looks, from the world's `LightLook`. Each table is the brightness
// of levels 0 to 15, with level 0 at 0.
struct LightLook {
    sky_falloff: array<vec4<f32>, 4>,
    voxel_falloff: array<vec4<f32>, 4>,
    sun_falloff: array<vec4<f32>, 4>,
    // Above 0, the sun shines fully from this sky level up and not below it,
    // in place of `sun_falloff`.
    sun_step: f32,
    voxel_brightness: f32,
    emissive_glow: f32,
    minimum_brightness: f32,
    smoothed: u32,
    ao: u32,
    // 1 at 256 levels, 0 at 16.
    steps_256: u32,
    // 1 when a sub-grid's enclosure caps the world's voxel light its faces
    // show, as it caps the world's sky: see `sample_face_light`.
    enclosure_caps_voxel: u32,
    // 1 when the faces of an emissive voxel glow tinted by its light colour:
    // see `glow_color`.
    tinted_glow: u32,
}

@group(constants::MATERIAL_BIND_GROUP) @binding(200)
var light_texture: texture_3d<u32>;

@group(constants::MATERIAL_BIND_GROUP) @binding(201)
var<uniform> light_look: LightLook;

// The world's tag map, under `SUB_GRIDS`: see `tag_map_tag`.
@if(SUB_GRIDS)
@group(constants::MATERIAL_BIND_GROUP) @binding(202)
var tag_map: texture_2d<u32>;

// The colour texture of the world, under `LIGHT_COLOR`, which `color_entry`
// and `cell_color` read.
@if(LIGHT_COLOR)
@group(constants::MATERIAL_BIND_GROUP) @binding(203)
var light_colors: texture_3d<u32>;

const HAS_SLOT: u32 = 0x80000000u;
const SLOT_SIDE: u32 = 34u;

// Bit 30 of a tag: the chunk is a sub-grid's.
const SUB_GRID: u32 = 0x40000000u;

// Bit 29 of a tag with a slot: the entry of the slot holds a light colour
// other than white, or names a colour slot.
const COLORED: u32 = 0x20000000u;

// The lowest of the tag bits that hold the light colour of a chunk with no
// slot.
const COLOR_SHIFT: u32 = 16u;

// The tag of an empty entry of the tag map.
const EMPTY: u32 = 0xFFFFFFFFu;

// How many voxels a level-0 chunk spans on a side, as a shift.
const CHUNK_SHIFT: u32 = 5u;

// The look's `ao`, as `LightAo`: 0 is `IndirectLight`.
const AO_BASE_COLOR: u32 = 1u;
const AO_OFF: u32 = 2u;

const SKY_TABLE: u32 = 0u;
const VOXEL_TABLE: u32 = 1u;
const SUN_TABLE: u32 = 2u;

// The light a face sees.
//
// `sky` and `voxel` are the levels, from 0 to 15, of the light in front of
// the face. `voxel_color` is the light colour of that voxel light, with its
// brightest component 1. `own_voxel` is the level of the emissive voxel
// behind the face, or 0 if the voxel behind it isn't emissive.
// `own_voxel_color` is the light colour of that emissive voxel. Without
// `LIGHT_COLOR`, both colours are white.
struct LightLevels {
    sky: f32,
    voxel: f32,
    own_voxel: f32,
    voxel_color: vec3<f32>,
    own_voxel_color: vec3<f32>,
}

// Returns the light colour that `packed` holds, packed as `light::color` in
// the crate packs it, with `bits` bits a component.
//
// The 2 bits above the components name the brightest component, from 0 for
// red to 2 for blue, and that component reads 1. The next `bits` bits, and
// then the lowest `bits` bits, say how far each of the other two components,
// in order, falls short of 1, in steps of 1 / `steps`. A value of 3 in the
// top bits names no colour and reads white. An entry packs a colour in 7 bits
// a component with steps of 1/127, and a tag with no slot in 6 bits with
// steps of 1/63.
fn unpack_color(packed: u32, bits: u32, steps: f32) -> vec3<f32> {
    let top = (packed >> (2u * bits)) & 3u;
    let mask = (1u << bits) - 1u;
    let first = 1.0 - f32((packed >> bits) & mask) / steps;
    let second = 1.0 - f32(packed & mask) / steps;
    if top == 0u {
        return vec3<f32>(1.0, first, second);
    } else if top == 1u {
        return vec3<f32>(first, 1.0, second);
    } else if top == 2u {
        return vec3<f32>(first, second, 1.0);
    }
    return vec3<f32>(1.0);
}

// Returns the one light colour that a chunk shows on every face, from its
// tag:
//
// - For a tag with no slot, it is the colour in bits 16-29, cut to 6 bits a
//   component.
// - For a tag with a slot, it is white, or the colour in the entry of the
//   slot when bit 29 is set.
// - Without `LIGHT_COLOR`, it is white.
fn chunk_color(tag: u32) -> vec3<f32> {
    @if(LIGHT_COLOR) {
        if (tag & HAS_SLOT) == 0u {
            return unpack_color((tag >> COLOR_SHIFT) & 0x3FFFu, 6u, 63.0);
        }
        if (tag & COLORED) == 0u {
            return vec3<f32>(1.0);
        }
        // An entry with the spare value 3 in its top bits names a colour
        // slot, not a colour, and `unpack_color` reads it as white.
        return unpack_color(color_entry(tag), 7u, 127.0);
    }
    @else {
        return vec3<f32>(1.0);
    }
}

// Returns the entry of the slot that `tag` names: a packed light colour, or
// the name of a colour slot. The entry is the texel of the colour texture at
// the place of the slot in the grid of slots. Until the colour texture is
// made, the material binds a placeholder of one white texel, and every entry
// reads that texel.
@if(LIGHT_COLOR)
fn color_entry(tag: u32) -> u32 {
    let slot = vec3<u32>(slot_origin(tag)) / SLOT_SIDE;
    let entry = min(slot, textureDimensions(light_colors) - vec3<u32>(1u));
    return textureLoad(light_colors, entry, 0).r;
}

// Where the light colours of the cells of a slot come from. When `slotted`
// is 1, they come from the colour slot whose rows begin at `origin`.
// Otherwise every cell has `color`, the one light colour of the chunk.
struct CellColors {
    origin: vec3<i32>,
    slotted: u32,
    color: vec3<f32>,
}

// Returns where the light colours of the cells of the slot that `tag` names
// come from. Only a chunk whose entry names a colour slot reads colours cell
// by cell. Every other chunk shows its one light colour on every face.
fn cell_colors(tag: u32) -> CellColors {
    @if(LIGHT_COLOR) {
        if (tag & HAS_SLOT) != 0u && (tag & COLORED) != 0u {
            let entry = color_entry(tag);
            if (entry >> 14u) == 3u {
                let slots = textureDimensions(light_texture) / SLOT_SIDE;
                let origin = color_slot_place(entry & 0x3FFFu) * SLOT_SIDE
                    + vec3<u32>(0u, 0u, slots.z);
                return CellColors(vec3<i32>(origin), 1u, vec3<f32>(1.0));
            }
            return CellColors(vec3<i32>(0), 0u, unpack_color(entry, 7u, 127.0));
        }
    }
    return CellColors(vec3<i32>(0), 0u, chunk_color(tag));
}

// Returns where the colour slot `index` lies among the colour slots of the
// colour texture, counted in colour slots along each axis. The bits of the
// index are dealt to x, y and z in turn, as `color_slot_place` in the crate
// deals them, so index 5, `0b101`, lies at (1, 0, 1). An index holds 14
// bits.
@if(LIGHT_COLOR)
fn color_slot_place(index: u32) -> vec3<u32> {
    var place = vec3<u32>(0u);
    for (var bit = 0u; bit < 14u; bit++) {
        place[bit % 3u] |= ((index >> bit) & 1u) << (bit / 3u);
    }
    return place;
}

// Returns the light colour of the cell at `cell`, in the padded local
// coordinates of the chunk. The texels of a colour slot are laid out as the
// texels of a slot are. A cell with no voxel light reads white.
@if(LIGHT_COLOR)
fn cell_color(colors: CellColors, cell: vec3<i32>) -> vec3<f32> {
    if colors.slotted == 0u {
        return colors.color;
    }
    let inside = clamp(cell, vec3<i32>(0), vec3<i32>(i32(SLOT_SIDE) - 1));
    let packed = textureLoad(light_colors, colors.origin + inside.zxy, 0).r;
    return unpack_color(packed, 7u, 127.0);
}

// Light colours mixing where they meet.
//
// `mix_color`, `mixed_color` and `mixing_weight` mirror `mix_weighted` and
// `mixing_weight` in `light::color` of the crate, so a change to that rule
// changes them too. Each component is the max over the colours of colour
// times weight, scaled back so the brightest component is 1. Where every
// weight is 0, each colour weighs 1 instead.
//
// The weight of a colour is its `mixing_weight` at its level, times the share
// it is given. `weighted` holds what the colours times their weights come to
// so far. `even` holds what the colours times their shares come to.
//
// A face that blends the cells in front of it adds their colours with
// `blend_color` instead, which sums in place of the max, and gives each cell
// its blend weight as its share. The max would show the colour of each cell
// across most of its face and step at the edges of the cells, since the
// blend weight of a cell outweighs those of its neighbours until their edge.
// Scaling the sum back still keeps red and green cells from averaging to a
// dark yellow.
struct ColorMix {
    weighted: vec3<f32>,
    even: vec3<f32>,
}

// The value of `0.7^(15 - level)` at level 1, which is 0.7 to the 14th.
// `mixing_weight` subtracts it so that level 1 weighs 0.
const MIXING_FLOOR: f32 = 0.0067822307;

// Returns the weight of a light colour at `level` where colours meet, under
// the fixed mixing curve `(0.7^(15 - level) - 0.7^14) / (1 - 0.7^14)`. The
// weight is 1 at level 15, about 0.70 at level 14, and 0 at level 1 and
// below, whatever the voxel falloff of the look.
@if(LIGHT_COLOR)
fn mixing_weight(level: f32) -> f32 {
    return max((pow(0.7, 15.0 - level) - MIXING_FLOOR) / (1.0 - MIXING_FLOOR), 0.0);
}

// Returns `mix` with the light colour `color` added at `level`, with the
// share `share`.
@if(LIGHT_COLOR)
fn mix_color(mix: ColorMix, color: vec3<f32>, level: f32, share: f32) -> ColorMix {
    return ColorMix(
        max(mix.weighted, color * (share * mixing_weight(level))),
        max(mix.even, color * share),
    );
}

// Returns `mix` with the light colour `color` of a cell blended in at
// `level`, with the blend weight `share` of the cell as its share.
@if(LIGHT_COLOR)
fn blend_color(mix: ColorMix, color: vec3<f32>, level: f32, share: f32) -> ColorMix {
    return ColorMix(
        mix.weighted + color * (share * mixing_weight(level)),
        mix.even + color * share,
    );
}

// Returns the light colour that the colours added to `mix` mix to, or white
// if none was added.
@if(LIGHT_COLOR)
fn mixed_color(mix: ColorMix) -> vec3<f32> {
    var mixed = mix.weighted;
    if max(mixed.x, max(mixed.y, mixed.z)) <= 0.0 {
        mixed = mix.even;
    }
    let brightest = max(mixed.x, max(mixed.y, mixed.z));
    if brightest <= 0.0 {
        return vec3<f32>(1.0);
    }
    return mixed / brightest;
}

// The texel of the slot `tag` names where its rows begin.
fn slot_origin(tag: u32) -> vec3<i32> {
    let slot = tag & 0xFFFFFFu;
    let slots = textureDimensions(light_texture) / SLOT_SIDE;
    let at = vec3<u32>(slot % slots.x, (slot / slots.x) % slots.y, slot / (slots.x * slots.y));
    return vec3<i32>(at * SLOT_SIDE);
}

// The light of the slot's cell at `cell`, in the chunk's padded local
// coordinates: sky light, voxel light, and the lamp's level if the cell is a
// lamp's.
fn load_cell(origin: vec3<i32>, cell: vec3<i32>) -> vec3<f32> {
    let inside = clamp(cell, vec3<i32>(0), vec3<i32>(i32(SLOT_SIDE) - 1));
    let texel = textureLoad(light_texture, origin + inside.zxy, 0);
    var sky = texel.r & 15u;
    var voxel = texel.r >> 4u;
    var top = 15u;
    var per_level = 1.0;
    if light_look.steps_256 == 1u {
        sky = texel.r;
        voxel = texel.g;
        top = 255u;
        per_level = 17.0;
    }
    if voxel == top {
        let lamp = f32(sky) / per_level;
        return vec3<f32>(0.0, lamp, lamp);
    }
    return vec3<f32>(f32(sky), f32(voxel), 0.0) / per_level;
}

// The light in front of a face. `sky` and `voxel` are levels from 0 to 15.
// `color` is the light colour of the voxel light, with its brightest
// component 1.
struct FrontLight {
    sky: f32,
    voxel: f32,
    color: vec3<f32>,
}

// Returns the blend weights of the cell in front of a face and of its
// neighbours in the plane of the face. `levels` holds the sky light or the
// voxel light of the front cell, the cell along u, the cell along v and the
// diagonal cell, in that order. `du` and `dv` are how far the point read lies
// from the middle of the front cell along u and v.
//
// A level of zero reads as a wall and weighs 0. The diagonal also weighs 0
// when both cells beside it are zero, so no light passes through a corner
// between two solids.
fn blend_weights(levels: vec4<f32>, du: f32, dv: f32) -> vec4<f32> {
    let w_front = step(0.01, levels.x) * (1.0 - du) * (1.0 - dv);
    let w_u = step(0.01, levels.y) * du * (1.0 - dv);
    let w_v = step(0.01, levels.z) * (1.0 - du) * dv;
    var w_d = 0.0;
    if levels.y > 0.01 || levels.z > 0.01 {
        w_d = step(0.01, levels.w) * du * dv;
    }
    return vec4<f32>(w_front, w_u, w_v, w_d);
}

// Returns the weighted mean of `levels` under `weights` from
// `blend_weights`, or 0 if every weight is 0.
fn blend(levels: vec4<f32>, weights: vec4<f32>) -> f32 {
    let weight = weights.x + (weights.y + weights.z) + weights.w;
    if weight <= 0.0 {
        return 0.0;
    }
    let total = levels.x * weights.x + (levels.y * weights.y + levels.z * weights.z)
        + levels.w * weights.w;
    return total / weight;
}

// Returns the sky light and voxel light at `front`, the point half a voxel in
// front of a face on the grid, blended over the voxel there and its three
// nearest neighbours in the plane of the face. In a chunk with a colour slot,
// the light colour blends over the cells whose voxel light counts, each
// weighing its blend weight times its `mixing_weight`.
fn smoothed_read(
    origin: vec3<i32>,
    colors: CellColors,
    front: vec3<f32>,
    axis: vec3<f32>,
) -> FrontLight {
    let cell = vec3<i32>(floor(front));
    let f = fract(front) - vec3<f32>(0.5);
    var u = vec3<i32>(1, 0, 0);
    var v = vec3<i32>(0, 0, 1);
    var fu = f.x;
    var fv = f.z;
    if axis.x > 0.5 {
        u = vec3<i32>(0, 1, 0);
        fu = f.y;
    } else if axis.z > 0.5 {
        v = vec3<i32>(0, 1, 0);
        fv = f.y;
    }
    let su = u * i32(sign(fu) + select(0.0, 1.0, fu == 0.0));
    let sv = v * i32(sign(fv) + select(0.0, 1.0, fv == 0.0));
    var cells = array<vec3<i32>, 4>(cell, cell + su, cell + sv, cell + su + sv);
    let l0 = load_cell(origin, cells[0]);
    let l1 = load_cell(origin, cells[1]);
    let l2 = load_cell(origin, cells[2]);
    let l3 = load_cell(origin, cells[3]);
    let du = abs(fu);
    let dv = abs(fv);
    let sky = vec4<f32>(l0.x, l1.x, l2.x, l3.x);
    let voxel = vec4<f32>(l0.y, l1.y, l2.y, l3.y);
    let voxel_weights = blend_weights(voxel, du, dv);
    var color = colors.color;
    @if(LIGHT_COLOR) {
        if colors.slotted == 1u {
            var mix = ColorMix(vec3<f32>(0.0), vec3<f32>(0.0));
            for (var i = 0u; i < 4u; i++) {
                if voxel_weights[i] > 0.0 {
                    let cell_light = cell_color(colors, cells[i]);
                    mix = blend_color(mix, cell_light, voxel[i], voxel_weights[i]);
                }
            }
            color = mixed_color(mix);
        }
    }
    return FrontLight(
        blend(sky, blend_weights(sky, du, dv)),
        blend(voxel, voxel_weights),
        color,
    );
}

// Returns which of the 8 voxels of a 2x2x2 block in `mask` are joined to
// those in `start` through voxels in `mask`, by faces. Bit `i` is the voxel
// at `block_voxel(i)` of the block.
fn joined_to(mask: u32, start: u32) -> u32 {
    var reached = start;
    // No voxel of the block is more than 3 steps from another.
    for (var i = 0; i < 3; i++) {
        let next = ((reached & 0x55u) << 1u) | ((reached & 0xAAu) >> 1u)
            | ((reached & 0x33u) << 2u) | ((reached & 0xCCu) >> 2u)
            | ((reached & 0x0Fu) << 4u) | ((reached & 0xF0u) >> 4u);
        reached |= next & mask;
    }
    return reached;
}

// Returns the voxel that bit `i` names in a 2x2x2 block,
// `(i & 1, (i >> 1) & 1, (i >> 2) & 1)`. Bit 5 names (1, 0, 1).
fn block_voxel(i: u32) -> vec3<i32> {
    return vec3<i32>(vec3<u32>(i & 1u, (i >> 1u) & 1u, (i >> 2u) & 1u));
}

// Returns, as bits, which voxels of a 2x2x2 block count toward its sky light
// or voxel light, from the level and weight of each voxel.
//
// A level of zero reads as a wall and is left out, and so is a weight of 0.
// A voxel counts only if it is joined through counted voxels to `nearest`,
// the bit of the voxel of the most weight. That voxel joins its neighbours
// even when its level is zero.
fn counted_voxels(levels: array<f32, 8>, weights: array<f32, 8>, nearest: u32) -> u32 {
    // Copies, since a function's arguments can't be indexed by a variable.
    var l = levels;
    var w = weights;
    var lit = 0u;
    for (var i = 0u; i < 8u; i++) {
        if l[i] > 0.01 && w[i] > 0.0 {
            lit |= 1u << i;
        }
    }
    return joined_to(lit | nearest, nearest) & lit;
}

// Returns the voxel of the most weight among those in `counted`, as a bit
// index.
fn nearest_counted(weights: array<f32, 8>, counted: u32) -> u32 {
    var w = weights;
    var best = 0.0;
    var nearest = 0u;
    for (var i = 0u; i < 8u; i++) {
        if (counted & (1u << i)) != 0u && w[i] > best {
            best = w[i];
            nearest = i;
        }
    }
    return nearest;
}

// Returns the sky light or voxel light of a 2x2x2 block of voxels, from the
// voxels in `counted` (see `counted_voxels`). Smoothed light takes the
// weighted mean of their levels. Stepped light takes the level of the
// nearest of them.
fn blend_eight(levels: array<f32, 8>, weights: array<f32, 8>, counted: u32) -> f32 {
    if counted == 0u {
        return 0.0;
    }
    var l = levels;
    var w = weights;
    if light_look.smoothed == 0u {
        return l[nearest_counted(weights, counted)];
    }
    var total = 0.0;
    var weight = 0.0;
    for (var i = 0u; i < 8u; i++) {
        if (counted & (1u << i)) != 0u {
            total += l[i] * w[i];
            weight += w[i];
        }
    }
    return total / weight;
}

// Returns the light colour of the voxel light of a 2x2x2 block whose first
// voxel is `base`, from the voxels in `counted`. Smoothed light mixes their
// colours, each weighing its weight times its `mixing_weight`. Stepped light
// takes the colour of the nearest of them. With no voxel counted, the colour
// is white.
@if(LIGHT_COLOR)
fn blend_eight_colors(
    colors: CellColors,
    base: vec3<i32>,
    levels: array<f32, 8>,
    weights: array<f32, 8>,
    counted: u32,
) -> vec3<f32> {
    if counted == 0u {
        return vec3<f32>(1.0);
    }
    if light_look.smoothed == 0u {
        return cell_color(colors, base + block_voxel(nearest_counted(weights, counted)));
    }
    var l = levels;
    var w = weights;
    var mix = ColorMix(vec3<f32>(0.0), vec3<f32>(0.0));
    for (var i = 0u; i < 8u; i++) {
        if (counted & (1u << i)) != 0u {
            mix = blend_color(mix, cell_color(colors, base + block_voxel(i)), l[i], w[i]);
        }
    }
    return mixed_color(mix);
}

// The sky and voxel light at `front`, read from the 8 voxels around it,
// weighted by their distance from it, for a face off the grid.
fn eight_read(origin: vec3<i32>, colors: CellColors, front: vec3<f32>) -> FrontLight {
    let q = front - vec3<f32>(0.5);
    let base = floor(q);
    var t = q - base;
    // A point within rounding of a voxel's centre plane reads only the voxels
    // on that plane.
    let near = round(t);
    t = select(t, near, abs(t - near) < vec3<f32>(0.001));
    var weights: array<f32, 8>;
    var sky: array<f32, 8>;
    var voxel: array<f32, 8>;
    var nearest = 0u;
    var most = -1.0;
    for (var i = 0u; i < 8u; i++) {
        let along = mix(vec3<f32>(1.0) - t, t, vec3<f32>(block_voxel(i)));
        let w = along.x * along.y * along.z;
        weights[i] = w;
        if w > most {
            most = w;
            nearest = 1u << i;
        }
        if w > 0.0 {
            let cell = load_cell(origin, vec3<i32>(base) + block_voxel(i));
            sky[i] = cell.x;
            voxel[i] = cell.y;
        }
    }
    let voxel_counted = counted_voxels(voxel, weights, nearest);
    var color = colors.color;
    @if(LIGHT_COLOR) {
        if colors.slotted == 1u {
            color = blend_eight_colors(colors, vec3<i32>(base), voxel, weights, voxel_counted);
        }
    }
    return FrontLight(
        blend_eight(sky, weights, counted_voxels(sky, weights, nearest)),
        blend_eight(voxel, weights, voxel_counted),
        color,
    );
}

// Returns the one sky light level and voxel light level that every face of a
// chunk sees, from the tag of the chunk when it names no slot.
fn tag_levels(tag: u32) -> LightLevels {
    var sky = f32(tag & 15u);
    var voxel = f32((tag >> 4u) & 15u);
    if light_look.steps_256 == 1u {
        sky = f32(tag & 255u) / 17.0;
        voxel = f32((tag >> 8u) & 255u) / 17.0;
    }
    return LightLevels(sky, voxel, 0.0, chunk_color(tag), vec3<f32>(1.0));
}

// How many voxels wide the cells of the slot `tag` names are: `1 << L` for a
// chunk at LOD level L.
fn cell_width(tag: u32) -> f32 {
    return f32(1u << ((tag >> 24u) & 15u));
}

// `normal` divided by its length, with a floor, since a mesher's normals can
// interpolate through zero, as across the top of a wall one voxel thick.
fn unit_normal(normal: vec3<f32>) -> vec3<f32> {
    return normal / max(length(normal), 1e-4);
}

// Returns the sky light and voxel light in front of a face at `local`, in the
// cells of the slot whose rows begin at `origin`, with `n` its unit normal.
//
// A face on the grid has its normal on an axis and its position on a plane
// between cells. It reads the cell in front of it, at `local + n / 2`, and
// with smoothed light also the three nearest neighbours of that cell in the
// plane of the face. Any other face, such as a slope, may cut into a solid
// voxel, so the cell at `local + n / 2` may be solid. Such a face reads the 8
// cells around that point instead, leaving out those with no light and those
// not joined to the nearest of the 8. On a cube face that would read the same
// as the read on the grid, which costs less.
//
// The light colour comes from `colors`. A chunk with a colour slot blends the
// colours of the cells whose voxel light it reads, and with stepped light
// takes the colour of the nearest one. Any other chunk reads no colours, and
// its one light colour holds for every face.
fn read_front(
    origin: vec3<i32>,
    colors: CellColors,
    local: vec3<f32>,
    n: vec3<f32>,
) -> FrontLight {
    let front = local + n * 0.5;
    let axis = abs(n);
    let along = dot(local, step(vec3<f32>(0.999), axis));
    let on_grid = max(axis.x, max(axis.y, axis.z)) >= 0.999
        && abs(along - round(along)) <= 0.01;
    if !on_grid {
        return eight_read(origin, colors, front);
    } else if light_look.smoothed == 1u {
        return smoothed_read(origin, colors, front, axis);
    }
    let cell = vec3<i32>(floor(front));
    let levels = load_cell(origin, cell);
    var color = colors.color;
    @if(LIGHT_COLOR) {
        if colors.slotted == 1u && levels.y > 0.0 {
            color = cell_color(colors, cell);
        }
    }
    return FrontLight(levels.x, levels.y, color);
}

// Returns the light a chunk face sees, from the tag of the chunk and the
// position and normal of the face in the padded local coordinates of the
// chunk.
//
// A tag with no slot gives its levels. Otherwise the face reads the cells of
// the slot in front of it, as `read_front` says, and glows if the cell behind
// it is emissive. A chunk above LOD level 0 reads its cells, `1 << L` voxels
// wide, after dividing the offset of the face from the first voxel of the
// chunk by that width.
//
// The light colour of the emissive voxel behind the face is read where its
// level is, at the cell behind the face. It comes from the colour slot of the
// chunk if the chunk has one. Otherwise it is the one light colour of the
// chunk, which counts the colours of its emissive voxels. Only `glow_color`
// uses this colour, so the colour slot is read only when the `tinted_glow` of
// the look is on.
fn sample_light(tag: u32, chunk_local: vec3<f32>, normal: vec3<f32>) -> LightLevels {
    if (tag & HAS_SLOT) == 0u {
        return tag_levels(tag);
    }
    let local = (chunk_local - vec3<f32>(1.0)) / cell_width(tag) + vec3<f32>(1.0);
    let origin = slot_origin(tag);
    let colors = cell_colors(tag);
    let n = unit_normal(normal);
    let behind_cell = vec3<i32>(floor(local - n * 0.5));
    let behind = load_cell(origin, behind_cell);
    let front = read_front(origin, colors, local, n);
    var own_color = colors.color;
    @if(LIGHT_COLOR) {
        if colors.slotted == 1u && behind.z > 0.0 && light_look.tinted_glow == 1u {
            own_color = cell_color(colors, behind_cell);
        }
    }
    return LightLevels(front.sky, front.voxel, behind.z, front.color, own_color);
}

// The tag map's hash of the level-0 world chunk position `position`, the
// entry its lookup starts from. It matches the crate's own: every step is a
// wrapping `u32` multiply, xor or shift.
fn tag_map_hash(position: vec3<i32>) -> u32 {
    let p = bitcast<vec3<u32>>(position);
    var hash = (p.x * 0x9E3779B1u) ^ (p.y * 0x85EBCA77u) ^ (p.z * 0xC2B2AE3Du);
    hash ^= hash >> 16u;
    hash *= 0x85EBCA6Bu;
    hash ^= hash >> 13u;
    hash *= 0xC2B2AE35u;
    hash ^= hash >> 16u;
    return hash;
}

// The tag the tag map holds for the level-0 world chunk at `position`, or
// `EMPTY` if it holds none.
//
// It follows the crate's lookup exactly: from the hash masked by the
// capacity, a power of two, step to the next entry, wrapping at the end,
// until the position or an empty entry. Entry `i` is texel
// `(i % width, i / width)`.
@if(SUB_GRIDS)
fn tag_map_tag(position: vec3<i32>) -> u32 {
    let size = textureDimensions(tag_map);
    let capacity = size.x * size.y;
    let mask = capacity - 1u;
    let key = bitcast<vec3<u32>>(position);
    var index = tag_map_hash(position) & mask;
    // The table always keeps empty entries, so the bound is never reached.
    for (var probed = 0u; probed < capacity; probed++) {
        let entry = textureLoad(tag_map, vec2<u32>(index % size.x, index / size.x), 0);
        if entry.w == EMPTY {
            break;
        }
        if all(entry.xyz == key) {
            return entry.w;
        }
        index = (index + 1u) & mask;
    }
    return EMPTY;
}

// The world's light at a face of a sub-grid, from the face's position and
// normal in the world's grid, which assumes the world's root at the
// identity.
//
// It reads as `sample_light` would for a world face there: on the grid, the
// cell in front of the face, and off it the 8 cells around the point in
// front of it. It looks up one world chunk in the tag map, the one holding
// the point half a voxel in front of the face, whose voxel is the nearest of
// the cells read, and reads every cell from that chunk's slot, whose shell
// holds its neighbours' cells. A chunk above LOD level 0 reads its cells, `1 << L`
// voxels wide, from its own minimum corner. A position the tag map doesn't
// hold reads open sky with no lamps. A face never glows from the world's
// lamps.
@if(SUB_GRIDS)
fn world_light(world_position: vec3<f32>, world_normal: vec3<f32>) -> LightLevels {
    let n = unit_normal(world_normal);
    let nearest = vec3<i32>(floor(world_position + n * 0.5));
    let chunk = nearest >> vec3<u32>(CHUNK_SHIFT);
    let tag = tag_map_tag(chunk);
    if tag == EMPTY {
        return LightLevels(15.0, 0.0, 0.0, vec3<f32>(1.0), vec3<f32>(1.0));
    }
    if (tag & HAS_SLOT) == 0u {
        return tag_levels(tag);
    }
    // A coarse chunk's minimum corner is a multiple of its extent in
    // level-0 chunks, `1 << L`.
    let level = vec3<u32>((tag >> 24u) & 15u);
    let corner = (chunk >> level) << level;
    let offset = world_position - vec3<f32>(corner) * f32(1u << CHUNK_SHIFT);
    let local = offset / cell_width(tag) + vec3<f32>(1.0);
    let front = read_front(slot_origin(tag), cell_colors(tag), local, n);
    return LightLevels(front.sky, front.voxel, 0.0, front.color, vec3<f32>(1.0));
}

// Returns the light that any chunk face shows, on a world chunk or on a
// sub-grid: the light a fragment shader of a lit world reads. It takes the
// tag of the chunk, the position and normal of the face in the padded local
// coordinates of the chunk, and the position and normal of the face in the
// grid of the world.
//
// A face of a world chunk reads as `sample_light` reads it, and the position
// and normal in the grid of the world go unused. Under the `SUB_GRIDS` shader
// def, a face of a sub-grid chunk, whose tag has bit 30 set, also reads the
// light of the world with `world_light`. The sky light of the sub-grid itself
// is its enclosure, so the smaller of the two sky levels caps the sky of the
// world by how closed in the face is. When `light_look.enclosure_caps_voxel`
// is set, the voxel light of the world is capped the same way, so the lamps
// of the world don't shine through the walls of a closed cabin. The face then
// shows the larger of that voxel light and the voxel light of the sub-grid.
// Its light colour is the colour the two mix to, each weighing its
// `mixing_weight` at its level. The glow of the face comes from the sub-grid
// alone.
fn sample_face_light(
    tag: u32,
    local_position: vec3<f32>,
    local_normal: vec3<f32>,
    world_position: vec3<f32>,
    world_normal: vec3<f32>,
) -> LightLevels {
    let own = sample_light(tag, local_position, local_normal);
    @if(SUB_GRIDS) {
        if (tag & SUB_GRID) != 0u {
            let world = world_light(world_position, world_normal);
            var world_voxel = world.voxel;
            if light_look.enclosure_caps_voxel == 1u {
                world_voxel = min(world_voxel, own.sky);
            }
            var color = world.voxel_color;
            @if(LIGHT_COLOR) {
                var mix = ColorMix(vec3<f32>(0.0), vec3<f32>(0.0));
                if world_voxel > 0.0 {
                    mix = mix_color(mix, world.voxel_color, world_voxel, 1.0);
                }
                if own.voxel > 0.0 {
                    mix = mix_color(mix, own.voxel_color, own.voxel, 1.0);
                }
                color = mixed_color(mix);
            }
            return LightLevels(
                min(world.sky, own.sky),
                max(world_voxel, own.voxel),
                own.own_voxel,
                color,
                own.own_voxel_color,
            );
        }
    }
    return own;
}

// The brightness of `level` in one of the look's tables, between entries for
// a level between whole levels.
fn falloff(table: u32, level: f32) -> f32 {
    if level <= 0.0 {
        return 0.0;
    }
    let l = min(level, 15.0);
    let low = u32(floor(l));
    let high = min(low + 1u, 15u);
    var a: f32;
    var b: f32;
    if table == SKY_TABLE {
        a = light_look.sky_falloff[low / 4u][low % 4u];
        b = light_look.sky_falloff[high / 4u][high % 4u];
    } else if table == VOXEL_TABLE {
        a = light_look.voxel_falloff[low / 4u][low % 4u];
        b = light_look.voxel_falloff[high / 4u][high % 4u];
    } else {
        a = light_look.sun_falloff[low / 4u][low % 4u];
        b = light_look.sun_falloff[high / 4u][high % 4u];
    }
    return mix(a, b, l - floor(l));
}

fn voxel_falloff(level: f32) -> f32 {
    return falloff(VOXEL_TABLE, level);
}

// Returns the light colour that tints the glow of a face: the light colour
// of its emissive voxel when the `tinted_glow` of the look is on, and white
// otherwise.
fn glow_color(levels: LightLevels) -> vec3<f32> {
    return select(vec3<f32>(1.0), levels.own_voxel_color, light_look.tinted_glow == 1u);
}

// How much of Bevy's directional lights reaches a face at sky light `level`,
// from 0 to 1. By default only direct sky gets the sun: the face's level must
// reach the step halfway between direct sky and the level beside it.
fn sun_brightness(level: f32) -> f32 {
    if light_look.sun_step > 0.0 {
        return select(0.0, 1.0, level >= light_look.sun_step);
    }
    return falloff(SUN_TABLE, level);
}

// How much of Bevy's ambient light reaches a face at sky light `level`, from 0
// to 1.
fn sky_brightness(level: f32) -> f32 {
    return max(falloff(SKY_TABLE, level), light_look.minimum_brightness);
}

// Returns the voxel light a face shows, as light to multiply by its diffuse
// colour: the light around it in its light colour, and its glow if the face
// belongs to an emissive voxel. It includes the exposure of the view.
fn voxel_brightness(levels: LightLevels) -> vec3<f32> {
    let around = voxel_falloff(levels.voxel) * levels.voxel_color;
    let glow = light_look.emissive_glow * voxel_falloff(levels.own_voxel) * glow_color(levels);
    return (around + glow) * light_look.voxel_brightness * view_bindings::view.exposure;
}

// Applies the light of a face to the PBR input of Bevy. Light is never baked
// into the base colour.
//
// Sky light scales the ambient light and environment map of Bevy through the
// occlusion terms, and the directional lights of Bevy through
// `directional_shadow_factor`. Bevy applies that factor only to a directional
// light with a shadow map, so a shader calling this must also call
// `remove_unshadowed_sun` on the colour that the lighting of Bevy returns.
// Voxel light, in its light colour, and the glow of an emissive voxel add
// through the emissive output, in the forward pass only, so no other pass
// counts them twice. The glow takes the light colour of the emissive voxel
// only when the `tinted_glow` of the look is on.
//
// `face_ao` is the ambient occlusion of the face, from 0 to 1, applied as the
// `ao` of the look says. By default it darkens ambient light and voxel light,
// and leaves the base colour and the sun alone. With `AO_BASE_COLOR` it
// darkens the base colour instead, and with `AO_OFF` it's ignored. The glow
// of an emissive voxel is never darkened.
fn apply_voxel_light(pbr_input: ptr<function, PbrInput>, levels: LightLevels, face_ao: f32) {
    var ao = face_ao;
    if light_look.ao == AO_BASE_COLOR {
        let base = (*pbr_input).material.base_color;
        (*pbr_input).material.base_color = vec4<f32>(base.rgb * face_ao, base.a);
        ao = 1.0;
    } else if light_look.ao == AO_OFF {
        ao = 1.0;
    }
    let sky = sky_brightness(levels.sky);
    (*pbr_input).diffuse_occlusion *= sky * ao;
    (*pbr_input).specular_occlusion *= sky * ao;
    (*pbr_input).directional_shadow_factor *= sun_brightness(levels.sky);

    @if(!PREPASS_PIPELINE) {
        let material = (*pbr_input).material;
        let diffuse = calculate_diffuse_color(
            material.base_color.rgb,
            material.metallic,
            material.specular_transmission,
            material.diffuse_transmission,
        );
        let around = voxel_falloff(levels.voxel) * ao * levels.voxel_color;
        let glow = light_look.emissive_glow * voxel_falloff(levels.own_voxel) * glow_color(levels);
        let exposure = view_bindings::view.exposure;
        let light = (around + glow) * light_look.voxel_brightness * exposure;
        // Bevy multiplies emissive light by the exposure as far as the
        // material's emissive alpha says, and this light already has it.
        let weight = mix(1.0, exposure, material.emissive.a);
        (*pbr_input).material.emissive += vec4<f32>(light * diffuse / weight, 0.0);
    }
}

// Removes from `color`, the colour Bevy's lighting returned for `pbr_input`,
// the part of each directional light that `directional_shadow_factor` should
// have dimmed and Bevy didn't. Call it on the result of `apply_pbr_lighting`
// after `apply_voxel_light`.
//
// It exists because Bevy multiplies a directional light by
// `directional_shadow_factor` only for a light with a shadow map, on a mesh
// that receives shadows. Without it, a sun with no shadow map would light
// sealed caves that the sun gate keeps dark. For each other directional light
// it subtracts `(1 - factor) × exposure × directional_light(..)`, the light
// Bevy added and the factor would have taken away.
//
// It rebuilds Bevy's lighting input as `apply_pbr_lighting` in Bevy's
// `pbr_functions` builds it, so it has to follow that function through Bevy
// upgrades. A one-line fix upstream, applying the factor to every directional
// light, would retire it.
@if(!PREPASS_PIPELINE)
fn remove_unshadowed_sun(pbr_input: PbrInput, color: vec4<f32>) -> vec4<f32> {
    let factor = pbr_input.directional_shadow_factor;
    let n_directional_lights = view_bindings::lights.n_directional_lights;
    if factor >= 1.0 || n_directional_lights == 0u {
        return color;
    }
    let receives_shadows = (pbr_input.flags & MESH_FLAGS_SHADOW_RECEIVER_BIT) != 0u;

    let material = pbr_input.material;
    let base_color = material.base_color.rgb;
    let perceptual_roughness = material.perceptual_roughness;
    let NdotV = max(dot(pbr_input.N, pbr_input.V), 0.0001);
    var input: lighting::LightingInput;
    input.layers[LAYER_BASE].NdotV = NdotV;
    input.layers[LAYER_BASE].N = pbr_input.N;
    input.layers[LAYER_BASE].R = reflect(-pbr_input.V, pbr_input.N);
    input.layers[LAYER_BASE].perceptual_roughness = perceptual_roughness;
    input.layers[LAYER_BASE].roughness =
        lighting::perceptualRoughnessToRoughness(perceptual_roughness);
    input.P = pbr_input.world_position.xyz;
    input.V = pbr_input.V;
    input.diffuse_color = calculate_diffuse_color(
        base_color,
        material.metallic,
        material.specular_transmission,
        material.diffuse_transmission,
    );
    input.metallic = material.metallic;
    input.F0_dielectric = calculate_F0_dielectric(material.reflectance);
    input.F0_metallic = base_color;
    input.F_ab = lighting::F_AB(perceptual_roughness, NdotV);
    @if(STANDARD_MATERIAL_CLEARCOAT) {
        let clearcoat_N = pbr_input.clearcoat_N;
        let clearcoat_NdotV = max(dot(clearcoat_N, pbr_input.V), 0.0001);
        let clearcoat_roughness = material.clearcoat_perceptual_roughness;
        input.layers[LAYER_CLEARCOAT].NdotV = clearcoat_NdotV;
        input.layers[LAYER_CLEARCOAT].N = clearcoat_N;
        input.layers[LAYER_CLEARCOAT].R = reflect(-pbr_input.V, clearcoat_N);
        input.layers[LAYER_CLEARCOAT].perceptual_roughness = clearcoat_roughness;
        input.layers[LAYER_CLEARCOAT].roughness =
            lighting::perceptualRoughnessToRoughness(clearcoat_roughness);
        input.clearcoat_strength = material.clearcoat;
    }
    @if(STANDARD_MATERIAL_ANISOTROPY) {
        input.anisotropy = pbr_input.anisotropy_strength;
        input.Ta = pbr_input.anisotropy_T;
        input.Ba = pbr_input.anisotropy_B;
    }

    var added = vec3<f32>(0.0);
    for (var i: u32 = 0u; i < n_directional_lights; i = i + 1u) {
        let flags = view_bindings::lights.directional_lights[i].flags;
        let has_shadow_map =
            (flags & mesh_view_types::DIRECTIONAL_LIGHT_FLAGS_SHADOWS_ENABLED_BIT) != 0u;
        if receives_shadows && has_shadow_map {
            continue;
        }
        @if(LIGHTMAP)
        let enable_diffuse = (flags
            & mesh_view_types::DIRECTIONAL_LIGHT_FLAGS_AFFECTS_LIGHTMAPPED_MESH_DIFFUSE_BIT) != 0u;
        @else
        let enable_diffuse = true;
        added += lighting::directional_light(i, &input, enable_diffuse);
    }
    let removed = (1.0 - factor) * view_bindings::view.exposure * added;
    return vec4<f32>(max(color.rgb - removed, vec3<f32>(0.0)), color.a);
}
