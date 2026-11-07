import bevy_voxel_world::voxel_io::{VoxelVertex, VoxelVertexOutput};
import bevy_pbr::render::{
    pbr_fragment::pbr_input_from_standard_material,
    pbr_functions::alpha_discard,
    mesh_functions,
    view_transformations::position_world_to_clip,
};

@if(VOXEL_LIGHT)
import bevy_voxel_world::light::{sample_light, apply_voxel_light};
@if(VOXEL_LIGHT && !PREPASS_PIPELINE)
import bevy_voxel_world::light::remove_unshadowed_sun;

@if(PREPASS_PIPELINE)
import bevy_pbr::{
    prepass::io::{VertexOutput, FragmentOutput},
    deferred::functions::deferred_output,
};
@else
import bevy_pbr::render::{
    forward_io::{VertexOutput, FragmentOutput},
    pbr_functions::{apply_pbr_lighting, main_pass_post_lighting_processing},
};

@group(constants::MATERIAL_BIND_GROUP) @binding(100)
var mat_array_texture: texture_2d_array<f32>;

@group(constants::MATERIAL_BIND_GROUP) @binding(101)
var mat_array_texture_sampler: sampler;

@vertex
fn vertex(vertex: VoxelVertex) -> VoxelVertexOutput {
    var out: VoxelVertexOutput;
    var model =  mesh_functions::get_world_from_local(vertex.instance_index);

    out.world_normal = mesh_functions::mesh_normal_local_to_world(
        vertex.normal, vertex.instance_index);

    out.world_position = mesh_functions::mesh_position_local_to_world(
        model, vec4<f32>(vertex.position, 1.0));

    out.position = position_world_to_clip(out.world_position.xyz);

@if(VERTEX_UVS)
    out.uv = vertex.uv;

@if(VERTEX_TANGENTS)
    out.world_tangent = mesh_functions::mesh_tangent_local_to_world(
        model,
        vertex.tangent,
        vertex.instance_index
    );

    out.color = vertex.color;

@if(VERTEX_OUTPUT_INSTANCE_INDEX)
    out.instance_index = vertex.instance_index;

    out.tex_idx = vertex.tex_idx;

    @if(VOXEL_LIGHT) {
        out.local_position = vertex.position;
        out.local_normal = vertex.normal;
        out.light_tag = mesh_functions::get_tag(vertex.instance_index);
    }

    return out;
}

@fragment
fn fragment(
    in: VoxelVertexOutput,
    @builtin(front_facing) is_front: bool,
)  -> FragmentOutput {
    var standard_in: VertexOutput;
    standard_in.position = in.position;
    standard_in.world_normal = in.world_normal;
    standard_in.world_position = in.world_position;
    standard_in.uv = in.uv;
    standard_in.color = in.color;
    standard_in.instance_index = in.instance_index;
    var pbr_input = pbr_input_from_standard_material(standard_in, is_front);

    var tex_face = 0;

    // determine texture index based on normal
    if in.world_normal.y == 0.0 {
        tex_face = 1;
    } else if in.world_normal.y < 0.0 {
        tex_face = 2;
    }

    pbr_input.material.base_color = textureSample(mat_array_texture, mat_array_texture_sampler, in.uv, in.tex_idx[tex_face]);
    // The vertex colour holds the mesher's ambient occlusion. With light, it
    // darkens ambient and voxel light instead of the base colour.
    @if(!VOXEL_LIGHT)
    pbr_input.material.base_color = pbr_input.material.base_color * in.color;

    pbr_input.material.base_color = alpha_discard(
        pbr_input.material.flags,
        pbr_input.material.alpha_cutoff,
        pbr_input.material.base_color
    );

@if(PREPASS_PIPELINE)
    let out = deferred_output(in, pbr_input);
@if(VOXEL_LIGHT && !PREPASS_PIPELINE)
    apply_voxel_light(
        &pbr_input,
        sample_light(in.light_tag, in.local_position, in.local_normal),
        in.color.r,
    );

@if(!PREPASS_PIPELINE)
    var out: FragmentOutput;
@if(!PREPASS_PIPELINE)
    out.color = apply_pbr_lighting(pbr_input);
@if(VOXEL_LIGHT && !PREPASS_PIPELINE)
    out.color = remove_unshadowed_sun(pbr_input, out.color);
@if(!PREPASS_PIPELINE)
    out.color = main_pass_post_lighting_processing(pbr_input, out.color);

    return out;
}
