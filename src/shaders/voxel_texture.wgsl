#import bevy_pbr::mesh_view_bindings view, fog
#import bevy_pbr::mesh_view_types FOG_MODE_OFF
#import bevy_pbr::mesh_bindings mesh
#import bevy_pbr::mesh_functions as mesh_functions
#import bevy_core_pipeline::tonemapping tone_mapping
#import bevy_pbr::pbr_functions as fns

@group(1) @binding(0)
var mat_array_texture: texture_2d_array<f32>;

@group(1) @binding(1)
var mat_array_texture_sampler: sampler;

struct Vertex {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
#ifdef VERTEX_TANGENTS
    @location(3) tangent: vec4<f32>,
#endif
    @location(4) color: vec4<f32>,
    @location(5) tex_idx: vec3<u32>
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec4<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
#ifdef VERTEX_TANGENTS
    @location(3) world_tangent: vec4<f32>,
#endif
    @location(4) color: vec4<f32>,
    @location(5) tex_idx: vec3<u32>,
};

@vertex
fn vertex(vertex: Vertex) -> VertexOutput {
    var out: VertexOutput;
    var model = mesh.model;

    out.world_normal = mesh_functions::mesh_normal_local_to_world(vertex.normal);
    out.world_position = mesh_functions::mesh_position_local_to_world(model, vec4<f32>(vertex.position, 1.0));
    out.clip_position = mesh_functions::mesh_position_world_to_clip(out.world_position);
    out.uv = vertex.uv;
    out.color = vertex.color;
    out.tex_idx = vertex.tex_idx;

#ifdef VERTEX_TANGENTS
    out.world_tangent = mesh_functions::mesh_tangent_local_to_world(model, vertex.tangent);
#endif

    return out;
}

@fragment
fn fragment(
    @builtin(front_facing) is_front: bool,
    mesh: VertexOutput,
) -> @location(0) vec4<f32> {
    var tex_face = 0;

    // determine texture index based on normal
    if (mesh.world_normal.y == 0.0) {
        tex_face = 1;
    } else if (mesh.world_normal.y == -1.0) {
        tex_face = 2;
    }

    var pbr_input: fns::PbrInput = fns::pbr_input_new();
    var output_color: vec4<f32> = pbr_input.material.base_color;

    pbr_input.material.base_color = textureSample(mat_array_texture, mat_array_texture_sampler, mesh.uv, mesh.tex_idx[tex_face]);
    pbr_input.material.base_color = pbr_input.material.base_color * mesh.color;

    //pbr_input.flags = mesh.flags;
    pbr_input.frag_coord = mesh.clip_position;
    pbr_input.world_position = mesh.world_position;
    pbr_input.material.reflectance = 0.05;
    pbr_input.material.metallic = 0.05;
    pbr_input.material.perceptual_roughness = 1.0;
    pbr_input.is_orthographic = view.projection[3].w == 1.0;
    
    pbr_input.world_normal = fns::prepare_world_normal(
        mesh.world_normal,
        false, // double sided
        is_front,
    );

    pbr_input.N = fns::apply_normal_mapping(
        pbr_input.material.flags,
        mesh.world_normal,
#ifdef VERTEX_TANGENTS
        mesh.world_tangent,
#endif
        mesh.uv,
        view.mip_bias,
    );
    pbr_input.V = fns::calculate_view(mesh.world_position, pbr_input.is_orthographic);
    
    output_color = fns::pbr(pbr_input);

    if (fog.mode != FOG_MODE_OFF) {
        output_color = fns::apply_fog(fog, output_color, mesh.world_position.xyz, view.world_position.xyz);
    }

    return tone_mapping(output_color, view.color_grading);
}