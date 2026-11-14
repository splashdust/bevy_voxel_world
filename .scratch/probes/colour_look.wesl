// Throwaway: the shader of the coloured-light look probe
// (.scratch/coloured-light/issues/03-how-coloured-light-looks.md).
//
// The crate's voxel shader with one change: voxel light comes from the probe's
// texture (binding 300), a texel per world voxel in the box `params.box_min`
// to `+ box_size`, holding red, green and blue levels times 4096 and a lamp's
// own level in alpha. Sky light, and everything in the "today" box, comes from
// the crate's own `sample_light`.

import bevy_voxel_world::voxel_io::{VoxelVertex, VoxelVertexOutput};
import bevy_pbr::render::{
    pbr_fragment::pbr_input_from_standard_material,
    pbr_functions::{alpha_discard, apply_pbr_lighting, main_pass_post_lighting_processing},
    mesh_functions,
    view_transformations::position_world_to_clip,
    forward_io::{VertexOutput, FragmentOutput},
    mesh_view_bindings as view_bindings,
};
import bevy_voxel_world::light::{
    LightLevels, sample_light, apply_voxel_light, remove_unshadowed_sun, voxel_falloff,
};

@group(constants::MATERIAL_BIND_GROUP) @binding(100)
var mat_array_texture: texture_2d_array<f32>;

@group(constants::MATERIAL_BIND_GROUP) @binding(101)
var mat_array_texture_sampler: sampler;

@group(constants::MATERIAL_BIND_GROUP) @binding(300)
var colour_texture: texture_3d<u32>;

struct Params {
    box_min: vec4<i32>,
    box_size: vec4<i32>,
    today_min: vec4<i32>,
    today_max: vec4<i32>,
    candidate: u32,
    glow: u32,
    view: u32,
    smoothed: u32,
    emissive_glow: f32,
    voxel_brightness: f32,
    pad0: f32,
    pad1: f32,
}

@group(constants::MATERIAL_BIND_GROUP) @binding(301)
var<uniform> params: Params;

@vertex
fn vertex(vertex: VoxelVertex) -> VoxelVertexOutput {
    var out: VoxelVertexOutput;
    var model = mesh_functions::get_world_from_local(vertex.instance_index);
    out.world_normal = mesh_functions::mesh_normal_local_to_world(vertex.normal, vertex.instance_index);
    out.world_position = mesh_functions::mesh_position_local_to_world(model, vec4<f32>(vertex.position, 1.0));
    out.position = position_world_to_clip(out.world_position.xyz);
    @if(VERTEX_UVS)
    out.uv = vertex.uv;
    out.color = vertex.color;
    @if(VERTEX_OUTPUT_INSTANCE_INDEX)
    out.instance_index = vertex.instance_index;
    out.tex_idx = vertex.tex_idx;
    out.local_position = vertex.position;
    out.local_normal = vertex.normal;
    out.light_tag = mesh_functions::get_tag(vertex.instance_index);
    return out;
}

// Red, green and blue levels, and a lamp's own level in w, of the world voxel
// at `cell`.
fn coloured_cell(cell: vec3<i32>) -> vec4<f32> {
    let p = cell - params.box_min.xyz;
    if any(p < vec3<i32>(0)) || any(p >= params.box_size.xyz) {
        return vec4<f32>(0.0);
    }
    return vec4<f32>(textureLoad(colour_texture, p, 0)) / 4096.0;
}

fn brightest(l: vec3<f32>) -> f32 {
    return max(l.x, max(l.y, l.z));
}

// The crate's `blend`, on three components at once, leaving out a cell whose
// brightest component is 0.
fn blend3(front: vec3<f32>, side_u: vec3<f32>, side_v: vec3<f32>, diagonal: vec3<f32>, du: f32, dv: f32) -> vec3<f32> {
    var total = front * (1.0 - du) * (1.0 - dv);
    var weight = step(0.01, brightest(front)) * (1.0 - du) * (1.0 - dv);
    let w_u = step(0.01, brightest(side_u)) * du * (1.0 - dv);
    let w_v = step(0.01, brightest(side_v)) * (1.0 - du) * dv;
    total += side_u * w_u + side_v * w_v;
    weight += w_u + w_v;
    if brightest(side_u) > 0.01 || brightest(side_v) > 0.01 {
        let w_d = step(0.01, brightest(diagonal)) * du * dv;
        total += diagonal * w_d;
        weight += w_d;
    }
    if weight <= 0.0 {
        return vec3<f32>(0.0);
    }
    return total / weight;
}

// The crate's read for a face on the grid, from world coordinates.
fn coloured_read(position: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    let front = position + n * 0.5;
    let cell = vec3<i32>(floor(front));
    if params.smoothed == 0u {
        return coloured_cell(cell).xyz;
    }
    let axis = abs(n);
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
    let l0 = coloured_cell(cell).xyz;
    let l1 = coloured_cell(cell + su).xyz;
    let l2 = coloured_cell(cell + sv).xyz;
    let l3 = coloured_cell(cell + su + sv).xyz;
    return blend3(l0, l1, l2, l3, abs(fu), abs(fv));
}

@fragment
fn fragment(
    in: VoxelVertexOutput,
    @builtin(front_facing) is_front: bool,
) -> FragmentOutput {
    var standard_in: VertexOutput;
    standard_in.position = in.position;
    standard_in.world_normal = in.world_normal;
    standard_in.world_position = in.world_position;
    standard_in.uv = in.uv;
    standard_in.color = in.color;
    standard_in.instance_index = in.instance_index;
    var pbr_input = pbr_input_from_standard_material(standard_in, is_front);

    var tex_face = 0;
    if in.world_normal.y == 0.0 {
        tex_face = 1;
    } else if in.world_normal.y < 0.0 {
        tex_face = 2;
    }
    pbr_input.material.base_color = textureSample(
        mat_array_texture, mat_array_texture_sampler, in.uv, in.tex_idx[tex_face]);

    let today = sample_light(in.light_tag, in.local_position, in.local_normal);
    var levels = today;
    let n = in.world_normal / max(length(in.world_normal), 1e-4);
    let position = in.world_position.xyz;
    let behind = vec3<i32>(floor(position - n * 0.5));
    let own = coloured_cell(behind);
    let in_today = all(behind >= params.today_min.xyz) && all(behind < params.today_max.xyz);

    if params.view == 2u {
        // Check: the probe's white (the "today" candidate's texture) against
        // the crate's voxel level, red where they differ.
        var out: FragmentOutput;
        let mine = brightest(coloured_read(position, n));
        let theirs = today.voxel.x;
        if own.w > 0.0 {
            out.color = vec4<f32>(1.0, 1.0, 0.0, 1.0);
        } else if abs(mine - theirs) > 0.05 {
            out.color = vec4<f32>(1.0, 0.0, 0.0, 1.0);
        } else {
            out.color = vec4<f32>(vec3<f32>(theirs / 15.0), 1.0);
        }
        return out;
    }
    if params.view == 1u {
        pbr_input.material.base_color = vec4<f32>(0.75, 0.75, 0.75, 1.0);
    }

    var glow = vec3<f32>(0.0);
    if params.candidate != 0u && !in_today {
        levels.voxel = coloured_read(position, n);
        levels.own_voxel = vec3<f32>(0.0);
        if own.w > 0.0 {
            if params.glow == 0u {
                levels.own_voxel = vec3<f32>(own.w);
            } else if params.glow == 1u {
                glow = params.emissive_glow * voxel_falloff(own.xyz);
            } else {
                levels.own_voxel = own.xyz;
            }
        }
    }

    apply_voxel_light(&pbr_input, levels, in.color.r);
    let exposure = view_bindings::view.exposure;
    let weight = mix(1.0, exposure, pbr_input.material.emissive.a);
    pbr_input.material.emissive += vec4<f32>(glow * params.voxel_brightness * exposure / weight, 0.0);

    var out: FragmentOutput;
    out.color = apply_pbr_lighting(pbr_input);
    out.color = remove_unshadowed_sun(pbr_input, out.color);
    out.color = main_pass_post_lighting_processing(pbr_input, out.color);
    return out;
}
