// Propagated light for chunk faces. Import it as `bevy_voxel_world::light`.
//
// A lit world's material holds the world's light texture and its look, at
// bindings 200 and 201, and the crate's vertex shader outputs each face's local
// position, local normal and tag under the `VOXEL_LIGHT` shader def. A fragment
// shader reads a face's light with `sample_light` and applies it to Bevy's PBR
// input with `apply_voxel_light`, the one place light meets the PBR
// calculation. A shader without PBR can use `sky_brightness` and
// `voxel_brightness` instead.
//
// Light is read from a face's position and normal alone, so it reaches any
// mesh: a cube face reads the voxel in front of it, and a face off the voxel
// grid the 8 voxels around the point in front of it (see `sample_light`).
// A custom vertex shader outputs what the crate's do: the centroid-sampled
// local position at location 9, the local normal at 10 and the flat tag at 11.
//
// The tag is the chunk's `MeshTag`. With bit 31 set, bits 0-23 name the
// chunk's slot in the texture: 34 texels a side, the chunk's voxels in its
// padded local coordinates, with texel rows along z. Without it, bits 0-7 hold
// the one level of each channel every face of the chunk sees. Each texel holds
// sky light in its low nibble and voxel light in its high one. A lamp's own
// cell holds 15 in its voxel half and its declared level in its sky half.

import bevy_pbr::render::{
    pbr_types::PbrInput,
    mesh_view_bindings as view_bindings,
    pbr_functions::calculate_diffuse_color,
};

// How light looks, from the world's `LightLook`. Each table is the brightness
// of levels 0 to 15, with level 0 at 0.
struct LightLook {
    sky_falloff: array<vec4<f32>, 4>,
    voxel_falloff: array<vec4<f32>, 4>,
    sun_falloff: array<vec4<f32>, 4>,
    voxel_brightness: f32,
    emissive_glow: f32,
    minimum_brightness: f32,
    smoothed: u32,
    ao: u32,
}

@group(constants::MATERIAL_BIND_GROUP) @binding(200)
var light_texture: texture_3d<u32>;

@group(constants::MATERIAL_BIND_GROUP) @binding(201)
var<uniform> light_look: LightLook;

const HAS_SLOT: u32 = 0x80000000u;
const SLOT_SIDE: u32 = 34u;
const TOP_LEVEL: u32 = 15u;

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
    let texel = textureLoad(light_texture, origin + inside.zxy, 0).r;
    let sky = texel & 15u;
    let voxel = texel >> 4u;
    if voxel == TOP_LEVEL {
        return vec3<f32>(0.0, f32(sky), f32(sky));
    }
    return vec3<f32>(f32(sky), f32(voxel), 0.0);
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

// The light a chunk face sees, from the chunk's tag and the face's position
// and normal in the chunk's padded local coordinates.
//
// A face on the grid, with its normal on an axis and its position on a plane
// between voxels, reads the voxel in front of it, at `local + normal / 2`,
// and with smoothed light its three nearest neighbours in the face's plane.
// Any other face, such as a slope, may cut into a solid voxel, so the voxel
// at `local + normal / 2` may be solid. It reads the 8 voxels around that
// point instead, leaving out those with no light and those not joined to the
// nearest of the 8. On a cube face that would read the same as the read on
// the grid, which costs less.
//
// The normal is divided by its length, with a floor, since a mesher's normals
// can interpolate through zero, as across the top of a wall one voxel thick.
fn sample_light(tag: u32, local: vec3<f32>, normal: vec3<f32>) -> LightLevels {
    if (tag & HAS_SLOT) == 0u {
        let voxel = f32((tag >> 4u) & 15u);
        return LightLevels(f32(tag & 15u), vec3<f32>(voxel), vec3<f32>(0.0));
    }
    let origin = slot_origin(tag);
    let n = normal / max(length(normal), 1e-4);
    let behind = load_cell(origin, vec3<i32>(floor(local - n * 0.5)));
    let front = local + n * 0.5;

    let axis = abs(n);
    let along = dot(local, step(vec3<f32>(0.999), axis));
    let on_grid = max(axis.x, max(axis.y, axis.z)) >= 0.999
        && abs(along - round(along)) <= 0.01;
    var levels: vec2<f32>;
    if !on_grid {
        levels = eight_read(origin, front);
    } else if light_look.smoothed == 1u {
        levels = smoothed_read(origin, front, axis);
    } else {
        levels = load_cell(origin, vec3<i32>(floor(front))).xy;
    }
    return LightLevels(levels.x, vec3<f32>(levels.y), vec3<f32>(behind.z));
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
// `directional_shadow_factor`. Voxel light and a lamp's glow add through the
// emissive channel, in the forward pass only, so no other pass counts them
// twice. `ao` is the face's ambient occlusion, from 0 to 1: it darkens ambient
// light and voxel light, and leaves the base colour and the sun alone.
fn apply_voxel_light(pbr_input: ptr<function, PbrInput>, levels: LightLevels, ao: f32) {
    let sky = sky_brightness(levels.sky);
    (*pbr_input).diffuse_occlusion *= sky * ao;
    (*pbr_input).specular_occlusion *= sky * ao;
    (*pbr_input).directional_shadow_factor *= falloff(SUN_TABLE, levels.sky);

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
