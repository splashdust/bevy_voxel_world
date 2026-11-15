// Propagated light for chunk faces. Import it as `bevy_voxel_world::light`.
//
// A lit world's material holds the world's light texture and its look, at
// bindings 200 and 201. The crate's vertex shaders output each face's local
// position and local normal in every world, and the chunk's tag under the
// `VOXEL_LIGHT` shader def. A fragment shader reads a face's light with
// `sample_face_light` and applies it to Bevy's PBR input with
// `apply_voxel_light`, the one place light meets the PBR calculation. A
// shader without PBR can use `sky_brightness` and `voxel_brightness` instead.
//
// Light is read from a face's position and normal alone, so it reaches any
// mesh: a cube face reads the voxel in front of it, and a face off the voxel
// grid the 8 voxels around the point in front of it (see `read_front`).
// A custom vertex shader outputs what the crate's do: the centroid-sampled
// local position at location 9 and the local normal at 10 in every world,
// the flat tag at 11 under `VOXEL_LIGHT`, and the centroid-sampled position
// in the world's grid at 12 under `SUB_GRIDS`. The crate's `rendering` module
// docs list every output and its space.
//
// The tag is the chunk's `MeshTag`. With bit 31 set, bits 0-23 name the
// chunk's slot in the texture: 34 texels a side, the chunk's voxels in its
// padded local coordinates, with texel rows along z. Without it, the low bits
// hold the one level of each channel every face of the chunk sees.
//
// The look's `steps_256` says how levels are stored. At 16 levels a texel is
// one `R8Uint` byte, sky light in its low nibble and voxel light in its high
// one, and a tag with no slot holds them the same way in bits 0-7. At 256
// levels a texel is `Rg8Uint`, sky light in red and voxel light in green, and
// a tag with no slot holds sky light in bits 0-7 and voxel light in bits 8-15.
// Levels at 256 are read on the 0 to 15 scale, in steps of 1/17. A lamp's own
// cell holds the top level, 15 or 255, in its voxel half and its declared
// level in its sky half.
//
// A chunk above LOD level 0 with a slot holds its LOD level L in bits 24-27.
// Its slot's cells are `1 << L` voxels wide, so its local position is divided
// by that width, keeping the corner of its first cell at (1, 1, 1).
//
// Bit 30 marks a sub-grid's chunk, with or without a slot, and bits 28 and 29
// are clear. Every read of a field masks it to its width, so a sub-grid's tag
// reads the same slot, LOD level and levels it would without bit 30.
//
// With the crate's `sub_grids` feature, a lit world sets the `SUB_GRIDS`
// shader def, and its material holds the world's tag map at binding 202: a
// hash table from a level-0 world chunk position to the tag the world shows
// there, for the world chunks around its sub-grids. `sample_face_light`
// lights a sub-grid's face with the world's light where the face is, read
// through the tag map by `world_light`, as well as with the sub-grid's own.
// It reads the face's position in the world's grid from Bevy's world
// position, so it assumes the world's `WorldRoot` is at the identity. A
// shader that still calls `sample_light` lights a sub-grid's faces from the
// sub-grid's own light alone: they don't darken in a cave or brighten by a
// lamp of the world. Each fragment of a sub-grid's face costs one lookup in
// the tag map and one more read's cells; a world chunk's face costs nothing
// more.
//
// References:
//
// - Terasology's `BlockMeshPart`, which lights a vertex from 8 points around
//   it, so a vertex anywhere gets light:
//   https://github.com/MovingBlocks/Terasology/blob/develop/engine/src/main/java/org/terasology/engine/world/block/shapes/BlockMeshPart.java
// - Cubyz's lighting, which blends the light of the 8 cells around a vertex
//   off the grid:
//   https://github.com/PixelGuys/Cubyz/blob/master/src/renderer/lighting.zig
// - Luanti's smooth lighting, which averages the cells around a corner,
//   leaving out cells that don't carry light:
//   https://github.com/luanti-org/luanti/blob/master/src/client/content_mapblock.cpp
// - tesserae's light propagation, where sky light scales Bevy's occlusion
//   terms and `directional_shadow_factor`, and voxel light adds through the
//   emissive channel:
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
}

@group(constants::MATERIAL_BIND_GROUP) @binding(200)
var light_texture: texture_3d<u32>;

@group(constants::MATERIAL_BIND_GROUP) @binding(201)
var<uniform> light_look: LightLook;

// The world's tag map, under `SUB_GRIDS`: see `tag_map_tag`.
@if(SUB_GRIDS)
@group(constants::MATERIAL_BIND_GROUP) @binding(202)
var tag_map: texture_2d<u32>;

const HAS_SLOT: u32 = 0x80000000u;
const SLOT_SIDE: u32 = 34u;

// Bit 30 of a tag: the chunk is a sub-grid's.
const SUB_GRID: u32 = 0x40000000u;

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

// The light a face sees, as levels from 0 to 15. `own_voxel` is the level of
// the lamp behind the face, or 0 if the voxel behind it isn't a lamp.
struct LightLevels {
    sky: f32,
    voxel: vec3<f32>,
    own_voxel: vec3<f32>,
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

// Blends one channel of the cell in front of a face with its neighbours in the
// face's plane, `du` and `dv` along each. A zero reads as a wall and is left
// out, and the diagonal is left out when both cells beside it are, so no light
// passes through a corner between two solids.
fn blend(front: f32, side_u: f32, side_v: f32, diagonal: f32, du: f32, dv: f32) -> f32 {
    var total = front * (1.0 - du) * (1.0 - dv);
    var weight = step(0.01, front) * (1.0 - du) * (1.0 - dv);
    let w_u = step(0.01, side_u) * du * (1.0 - dv);
    let w_v = step(0.01, side_v) * (1.0 - du) * dv;
    total += side_u * w_u + side_v * w_v;
    weight += w_u + w_v;
    if side_u > 0.01 || side_v > 0.01 {
        let w_d = step(0.01, diagonal) * du * dv;
        total += diagonal * w_d;
        weight += w_d;
    }
    if weight <= 0.0 {
        return 0.0;
    }
    return total / weight;
}

// The sky and voxel light at `front`, the middle of the voxel in front of a
// face on the grid, blended with its three nearest neighbours in the face's
// plane.
fn smoothed_read(origin: vec3<i32>, front: vec3<f32>, axis: vec3<f32>) -> vec2<f32> {
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
    let l0 = load_cell(origin, cell);
    let l1 = load_cell(origin, cell + su);
    let l2 = load_cell(origin, cell + sv);
    let l3 = load_cell(origin, cell + su + sv);
    let du = abs(fu);
    let dv = abs(fv);
    return vec2<f32>(
        blend(l0.x, l1.x, l2.x, l3.x, du, dv),
        blend(l0.y, l1.y, l2.y, l3.y, du, dv),
    );
}

// Which of the 8 voxels of a 2x2x2 block in `mask` are joined to those in
// `start` through voxels in `mask`, by faces. Bit i is the voxel at
// (i & 1, (i >> 1) & 1, (i >> 2) & 1) of the block.
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

// One channel of a 2x2x2 block of voxels, from each voxel's level and weight.
// A zero reads as a wall and is left out, and a voxel counts only if it's
// joined through counted voxels to `nearest`, the voxel of the most weight,
// which joins its neighbours even when it reads zero. Smoothed light takes the
// weighted mean of the counted voxels, and stepped light the nearest of them.
fn blend_eight(levels: array<f32, 8>, weights: array<f32, 8>, nearest: u32) -> f32 {
    // Copies, since a function's arguments can't be indexed by a variable.
    var l = levels;
    var w = weights;
    var lit = 0u;
    for (var i = 0u; i < 8u; i++) {
        if l[i] > 0.01 && w[i] > 0.0 {
            lit |= 1u << i;
        }
    }
    let counted = joined_to(lit | nearest, nearest) & lit;
    if counted == 0u {
        return 0.0;
    }
    var total = 0.0;
    var weight = 0.0;
    var best = 0.0;
    var best_level = 0.0;
    for (var i = 0u; i < 8u; i++) {
        if (counted & (1u << i)) != 0u {
            total += l[i] * w[i];
            weight += w[i];
            if w[i] > best {
                best = w[i];
                best_level = l[i];
            }
        }
    }
    if light_look.smoothed == 0u {
        return best_level;
    }
    return total / weight;
}

// The sky and voxel light at `front`, read from the 8 voxels around it,
// weighted by their distance from it, for a face off the grid.
fn eight_read(origin: vec3<i32>, front: vec3<f32>) -> vec2<f32> {
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
        let o = vec3<u32>(i & 1u, (i >> 1u) & 1u, (i >> 2u) & 1u);
        let along = mix(vec3<f32>(1.0) - t, t, vec3<f32>(o));
        let w = along.x * along.y * along.z;
        weights[i] = w;
        if w > most {
            most = w;
            nearest = 1u << i;
        }
        if w > 0.0 {
            let cell = load_cell(origin, vec3<i32>(base) + vec3<i32>(o));
            sky[i] = cell.x;
            voxel[i] = cell.y;
        }
    }
    return vec2<f32>(
        blend_eight(sky, weights, nearest),
        blend_eight(voxel, weights, nearest),
    );
}

// The one level of each channel that every face of a chunk sees, from the
// chunk's tag when it names no slot.
fn tag_levels(tag: u32) -> LightLevels {
    var sky = f32(tag & 15u);
    var voxel = f32((tag >> 4u) & 15u);
    if light_look.steps_256 == 1u {
        sky = f32(tag & 255u) / 17.0;
        voxel = f32((tag >> 8u) & 255u) / 17.0;
    }
    return LightLevels(sky, vec3<f32>(voxel), vec3<f32>(0.0));
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

// The sky and voxel light in front of a face at `local`, in the cells of the
// slot whose rows begin at `origin`, with `n` its unit normal.
//
// A face on the grid, with its normal on an axis and its position on a plane
// between cells, reads the cell in front of it, at `local + n / 2`, and with
// smoothed light its three nearest neighbours in the face's plane. Any other
// face, such as a slope, may cut into a solid voxel, so the cell at
// `local + n / 2` may be solid. It reads the 8 cells around that point
// instead, leaving out those with no light and those not joined to the
// nearest of the 8. On a cube face that would read the same as the read on
// the grid, which costs less.
fn read_front(origin: vec3<i32>, local: vec3<f32>, n: vec3<f32>) -> vec2<f32> {
    let front = local + n * 0.5;
    let axis = abs(n);
    let along = dot(local, step(vec3<f32>(0.999), axis));
    let on_grid = max(axis.x, max(axis.y, axis.z)) >= 0.999
        && abs(along - round(along)) <= 0.01;
    if !on_grid {
        return eight_read(origin, front);
    } else if light_look.smoothed == 1u {
        return smoothed_read(origin, front, axis);
    }
    return load_cell(origin, vec3<i32>(floor(front))).xy;
}

// The light a chunk face sees, from the chunk's tag and the face's position
// and normal in the chunk's padded local coordinates.
//
// A tag with no slot gives its levels. Otherwise the face reads the slot's
// cells in front of it, as `read_front` says, and glows if the cell behind
// it is a lamp's. A chunk above LOD level 0 reads its cells, `1 << L` voxels
// wide, after dividing the face's offset from the chunk's first voxel by
// that width.
fn sample_light(tag: u32, chunk_local: vec3<f32>, normal: vec3<f32>) -> LightLevels {
    if (tag & HAS_SLOT) == 0u {
        return tag_levels(tag);
    }
    let local = (chunk_local - vec3<f32>(1.0)) / cell_width(tag) + vec3<f32>(1.0);
    let origin = slot_origin(tag);
    let n = unit_normal(normal);
    let behind = load_cell(origin, vec3<i32>(floor(local - n * 0.5)));
    let levels = read_front(origin, local, n);
    return LightLevels(levels.x, vec3<f32>(levels.y), vec3<f32>(behind.z));
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
        return LightLevels(15.0, vec3<f32>(0.0), vec3<f32>(0.0));
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
    let levels = read_front(slot_origin(tag), local, n);
    return LightLevels(levels.x, vec3<f32>(levels.y), vec3<f32>(0.0));
}

// The light any chunk face shows, a world chunk's or a sub-grid's: the light
// a fragment shader of a lit world reads. It takes the chunk's tag, the
// face's position and normal in the chunk's padded local coordinates, and
// its position and normal in the world's grid.
//
// A world chunk's face reads as `sample_light` reads it, and the world's
// position and normal are unused. Under the `SUB_GRIDS` shader def, a
// sub-grid chunk's face, whose tag has bit 30 set, also reads the world's
// light with `world_light`. It shows the smaller of the world's sky light
// and its own, which caps the world's sky by the sub-grid's enclosure, and
// the larger of the world's voxel light and its own lamps'. Its glow is its
// own.
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
            return LightLevels(
                min(world.sky, own.sky),
                max(world.voxel, own.voxel),
                own.own_voxel,
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

fn voxel_falloff(levels: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        falloff(VOXEL_TABLE, levels.x),
        falloff(VOXEL_TABLE, levels.y),
        falloff(VOXEL_TABLE, levels.z),
    );
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

// The voxel light a face shows, as light to multiply by its diffuse colour:
// the light around it, and its glow if it's a lamp's face. Includes the view's
// exposure.
fn voxel_brightness(levels: LightLevels) -> vec3<f32> {
    let around = voxel_falloff(levels.voxel);
    let glow = light_look.emissive_glow * voxel_falloff(levels.own_voxel);
    return (around + glow) * light_look.voxel_brightness * view_bindings::view.exposure;
}

// Applies a face's light to Bevy's PBR input. Light is never baked into the
// base colour.
//
// Sky light scales Bevy's ambient light and environment map through the
// occlusion terms, and its directional lights through
// `directional_shadow_factor`. Bevy applies that factor only to a directional
// light with a shadow map, so a shader calling this must also call
// `remove_unshadowed_sun` on the colour Bevy's lighting returns. Voxel light
// and a lamp's glow add through the
// emissive channel, in the forward pass only, so no other pass counts them
// twice.
//
// `ao` is the face's ambient occlusion, from 0 to 1, applied as the look's
// `ao` says: by default it darkens ambient light and voxel light, and leaves
// the base colour and the sun alone. With `AO_BASE_COLOR` it darkens the base
// colour instead, and with `AO_OFF` it's ignored. A lamp's glow is never
// darkened.
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
        let around = voxel_falloff(levels.voxel) * ao;
        let glow = light_look.emissive_glow * voxel_falloff(levels.own_voxel);
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
