// The crate's PBR shader for a lit world whose material is plain
// `StandardMaterial`: Bevy's PBR with the world's light applied.
//
// Its vertex shader reads Bevy's own vertex layout, so it draws any mesh,
// including one without the crate's texture index, such as a marching cubes
// mesh. It outputs Bevy's usual values, the local position and normal at
// locations 9 and 10 that every crate vertex shader outputs, the tag at 11,
// and the position in the world's grid at 12 under `SUB_GRIDS`, as the
// crate's `rendering` module docs list them. It only ever runs in a lit
// world, so it writes the tag without testing `VOXEL_LIGHT`, and the
// `bevy_voxel_world::light` module reads them all.
//
// The fragment shader is Bevy's forward PBR, with the light
// `sample_face_light` reads applied by `apply_voxel_light` before Bevy's
// lighting and `remove_unshadowed_sun` after it. Vertex colours tint the base
// colour as in Bevy's own shader. They belong to the mesher, so they
// are not read as ambient occlusion, and the look's AO setting doesn't apply.
//
// Chunk meshes are never skinned or morphed, so neither is handled here.

import bevy_pbr::render::{
    forward_io::{Vertex, VertexOutput, FragmentOutput, decompress_vertex},
    mesh_functions,
    view_transformations::position_world_to_clip,
    pbr_fragment::pbr_input_from_standard_material,
    pbr_functions::{alpha_discard, apply_pbr_lighting, main_pass_post_lighting_processing},
    pbr_types::STANDARD_MATERIAL_FLAGS_UNLIT_BIT,
};
import bevy_voxel_world::light::{sample_face_light, apply_voxel_light, remove_unshadowed_sun};

struct LitVertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) world_position: vec4<f32>,
    @location(1) world_normal: vec3<f32>,
    @if(VERTEX_UVS_A)
    @location(2) uv: vec2<f32>,
    @if(VERTEX_UVS_B)
    @location(3) uv_b: vec2<f32>,
    @if(VERTEX_TANGENTS)
    @location(4) world_tangent: vec4<f32>,
    @if(VERTEX_COLORS)
    @location(5) color: vec4<f32>,
    @if(VERTEX_OUTPUT_INSTANCE_INDEX)
    @location(6) @interpolate(flat) instance_index: u32,

    // The face's position and normal in the chunk's padded local coordinates,
    // which stay with the chunk however it moves, and the chunk's tag. The
    // position is centroid-sampled because with MSAA a pixel centre can lie
    // outside its triangle, and the position there would read the light of a
    // voxel beyond the face.
    @location(9) @interpolate(perspective, centroid) local_position: vec3<f32>,
    @location(10) local_normal: vec3<f32>,
    @location(11) @interpolate(flat) light_tag: u32,
    // The face's position in the world's grid, centroid-sampled for the same
    // reason, under `SUB_GRIDS`.
    @if(SUB_GRIDS)
    @location(12) @interpolate(perspective, centroid) world_grid_position: vec3<f32>,
}

@vertex
fn vertex(vertex_in: Vertex) -> LitVertexOutput {
    let vertex = decompress_vertex(vertex_in, vertex_in.instance_index);
    var out: LitVertexOutput;
    let world_from_local = mesh_functions::get_world_from_local(vertex_in.instance_index);

    out.world_normal = mesh_functions::mesh_normal_local_to_world(
        vertex.normal, vertex_in.instance_index);
    out.world_position = mesh_functions::mesh_position_local_to_world(
        world_from_local, vec4<f32>(vertex.position, 1.0));
    out.position = position_world_to_clip(out.world_position.xyz);

@if(VERTEX_UVS_A)
    out.uv = vertex.uv;
@if(VERTEX_UVS_B)
    out.uv_b = vertex.uv_b;
@if(VERTEX_TANGENTS)
    out.world_tangent = mesh_functions::mesh_tangent_local_to_world(
        world_from_local, vertex.tangent, vertex_in.instance_index);
@if(VERTEX_COLORS)
    out.color = vertex.color;
@if(VERTEX_OUTPUT_INSTANCE_INDEX)
    out.instance_index = vertex_in.instance_index;

    out.local_position = vertex.position;
    out.local_normal = vertex.normal;
    out.light_tag = mesh_functions::get_tag(vertex_in.instance_index);
    @if(SUB_GRIDS)
    out.world_grid_position = out.world_position.xyz;
    return out;
}

@fragment
fn fragment(
    in: LitVertexOutput,
    @builtin(front_facing) is_front: bool,
) -> FragmentOutput {
    var standard_in: VertexOutput;
    standard_in.position = in.position;
    standard_in.world_position = in.world_position;
    standard_in.world_normal = in.world_normal;
@if(VERTEX_UVS_A)
    standard_in.uv = in.uv;
@if(VERTEX_UVS_B)
    standard_in.uv_b = in.uv_b;
@if(VERTEX_TANGENTS)
    standard_in.world_tangent = in.world_tangent;
@if(VERTEX_COLORS)
    standard_in.color = in.color;
@if(VERTEX_OUTPUT_INSTANCE_INDEX)
    standard_in.instance_index = in.instance_index;

    var pbr_input = pbr_input_from_standard_material(standard_in, is_front);
    pbr_input.material.base_color = alpha_discard(
        pbr_input.material.flags,
        pbr_input.material.alpha_cutoff,
        pbr_input.material.base_color
    );

    @if(SUB_GRIDS)
    let grid_position = in.world_grid_position;
    @if(!SUB_GRIDS)
    let grid_position = in.world_position.xyz;
    var out: FragmentOutput;
    if (pbr_input.material.flags & STANDARD_MATERIAL_FLAGS_UNLIT_BIT) == 0u {
        apply_voxel_light(
            &pbr_input,
            sample_face_light(
                in.light_tag,
                in.local_position,
                in.local_normal,
                grid_position,
                in.world_normal,
            ),
            1.0,
        );
        out.color = apply_pbr_lighting(pbr_input);
        out.color = remove_unshadowed_sun(pbr_input, out.color);
    } else {
        out.color = pbr_input.material.base_color;
    }
    out.color = main_pass_post_lighting_processing(pbr_input, out.color);
    return out;
}
