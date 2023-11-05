#import bevy_pbr::{
    pbr_fragment::pbr_input_from_standard_material,
    pbr_functions::alpha_discard,
    mesh_functions,
    view_transformations::position_world_to_clip
}
#import bevy_render::instance_index::get_instance_index

#ifdef PREPASS_PIPELINE
#import bevy_pbr::{
    prepass_io::{VertexOutput, FragmentOutput},
    pbr_deferred_functions::deferred_output,
}
#else
#import bevy_pbr::{
    forward_io::{VertexOutput, FragmentOutput},
    pbr_functions::{apply_pbr_lighting, main_pass_post_lighting_processing},
}
#endif

@group(1) @binding(100)
var mat_array_texture: texture_2d_array<f32>;

@group(1) @binding(101)
var mat_array_texture_sampler: sampler;

struct Vertex {
    @builtin(instance_index) instance_index: u32,
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
#ifdef VERTEX_TANGENTS
    @location(4) tangent: vec4<f32>,
#endif
    @location(5) color: vec4<f32>,
    @location(8) tex_idx: vec3<u32>
};

struct CustomVertexOutput {
    // This is `clip position` when the struct is used as a vertex stage output
    // and `frag coord` when used as a fragment stage input
    @builtin(position) position: vec4<f32>,
    @location(0) world_position: vec4<f32>,
    @location(1) world_normal: vec3<f32>,
#ifdef VERTEX_UVS
    @location(2) uv: vec2<f32>,
#endif
#ifdef VERTEX_TANGENTS
    @location(3) world_tangent: vec4<f32>,
#endif
#ifdef VERTEX_COLORS
    @location(4) color: vec4<f32>,
#endif
#ifdef VERTEX_OUTPUT_INSTANCE_INDEX
    @location(5) @interpolate(flat) instance_index: u32,
#endif

    @location(6) tex_idx: vec3<u32>,
}

@vertex
fn vertex(vertex: Vertex) -> CustomVertexOutput {
    var out: CustomVertexOutput;
    var model =  mesh_functions::get_model_matrix(vertex.instance_index);

    out.world_normal = mesh_functions::mesh_normal_local_to_world(
        vertex.normal, get_instance_index(vertex.instance_index));

    out.world_position = mesh_functions::mesh_position_local_to_world(
        model, vec4<f32>(vertex.position, 1.0));

    out.position = position_world_to_clip(out.world_position.xyz);
        
#ifdef VERTEX_UVS
    out.uv = vertex.uv;
#endif

#ifdef VERTEX_TANGENTS
    out.world_tangent = mesh_functions::mesh_tangent_local_to_world(
        model,
        vertex.tangent,
        get_instance_index(vertex.instance_index)
    );
#endif

    out.color = vertex.color;

#ifdef VERTEX_OUTPUT_INSTANCE_INDEX
    out.instance_index = get_instance_index(vertex.instance_index);
#endif

    out.tex_idx = vertex.tex_idx;

    return out;
}

@fragment
fn fragment(
    in: CustomVertexOutput,
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
    if (in.world_normal.y == 0.0) {
        tex_face = 1;
    } else if (in.world_normal.y == -1.0) {
        tex_face = 2;
    }

    pbr_input.material.base_color = textureSample(mat_array_texture, mat_array_texture_sampler, in.uv, in.tex_idx[tex_face]);
    pbr_input.material.base_color = pbr_input.material.base_color * in.color;

    pbr_input.material.base_color = alpha_discard(pbr_input.material, pbr_input.material.base_color);

    #ifdef PREPASS_PIPELINE
    // in deferred mode we can't modify anything after that, as lighting is run in a separate fullscreen shader.
    let out = deferred_output(in, pbr_input);
#else
    var out: FragmentOutput;
    // apply lighting
    out.color = apply_pbr_lighting(pbr_input);

    // we can optionally modify the lit color before post-processing is applied
    //out.color = vec4<f32>(vec4<u32>(out.color * f32(my_extended_material.quantize_steps))) / f32(my_extended_material.quantize_steps);

    // apply in-shader post processing (fog, alpha-premultiply, and also tonemapping, debanding if the camera is non-hdr)
    // note this does not include fullscreen postprocessing effects like bloom.
    out.color = main_pass_post_lighting_processing(pbr_input, out.color);

    // we can optionally modify the final result here
    out.color = out.color * 2.0;
#endif

    return out;   
}