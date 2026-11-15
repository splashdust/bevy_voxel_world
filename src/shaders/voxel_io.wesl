// The vertex input and output of voxel chunk meshes, at the shader locations
// `vertex_layout()` maps the mesh attributes to. Import it as
// `bevy_voxel_world::voxel_io`. The crate's `rendering` module docs say what
// space each output is in and which outputs a custom vertex shader must write.

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

    // The face's position and normal in the chunk's padded local coordinates,
    // in every world. They stay with the chunk however it moves, so the
    // default fragment shader picks the face's texture from the normal, and
    // in a lit world the `bevy_voxel_world::light` module reads both. The
    // position is centroid-sampled because with MSAA a pixel centre can lie
    // outside its triangle, and the position there would read the light of a
    // voxel beyond the face.
    @location(9) @interpolate(perspective, centroid) local_position: vec3<f32>,
    @location(10) local_normal: vec3<f32>,
    // The chunk's tag, for the `bevy_voxel_world::light` module.
    @if(VOXEL_LIGHT)
    @location(11) @interpolate(flat) light_tag: u32,
    // The face's position in the world's grid, `world_position` centroid-
    // sampled as `local_position` is, for the `bevy_voxel_world::light`
    // module to read the world's light at a sub-grid's face.
    @if(SUB_GRIDS)
    @location(12) @interpolate(perspective, centroid) world_grid_position: vec3<f32>,
}
