// The vertex input and output of voxel chunk meshes, at the shader locations
// `vertex_layout()` maps the mesh attributes to. Import it as
// `bevy_voxel_world::voxel_io`.

struct VoxelVertex {
    @builtin(instance_index) instance_index: u32,
    @if(VERTEX_POSITIONS)
    @location(0) position: vec3<f32>,
    @if(VERTEX_NORMALS)
    @location(1) normal: vec3<f32>,
    @if(VERTEX_UVS)
    @location(2) uv: vec2<f32>,
    @if(VERTEX_UVS_B)
    @location(3) uv_b: vec2<f32>,
    @if(VERTEX_TANGENTS)
    @location(4) tangent: vec4<f32>,
    @if(VERTEX_COLORS)
    @location(5) color: vec4<f32>,
    @if(MORPH_TARGETS)
    @builtin(vertex_index) index: u32,

    // The three texture indices from the world's `texture_index_mapper`.
    @location(8) tex_idx: vec3<u32>
};

struct VoxelVertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) world_position: vec4<f32>,
    @location(1) world_normal: vec3<f32>,
    @if(VERTEX_UVS)
    @location(2) uv: vec2<f32>,
    @if(VERTEX_UVS_B)
    @location(3) uv_b: vec2<f32>,
    @if(VERTEX_TANGENTS)
    @location(4) world_tangent: vec4<f32>,
    @if(VERTEX_COLORS)
    @location(5) color: vec4<f32>,
    @if(VERTEX_OUTPUT_INSTANCE_INDEX)
    @location(6) @interpolate(flat) instance_index: u32,

    @location(8) @interpolate(flat) tex_idx: vec3<u32>,

    // What the `bevy_voxel_world::light` module reads, in a lit world: the
    // face's position and normal in the chunk's padded local coordinates, and
    // the chunk's tag. The position is centroid-sampled because with MSAA a
    // pixel centre can lie outside its triangle, and the position there would
    // read the light of a voxel beyond the face.
    @if(VOXEL_LIGHT)
    @location(9) @interpolate(perspective, centroid) local_position: vec3<f32>,
    @if(VOXEL_LIGHT)
    @location(10) local_normal: vec3<f32>,
    @if(VOXEL_LIGHT)
    @location(11) @interpolate(flat) light_tag: u32,
}
