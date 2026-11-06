import bevy_pbr::render::forward_io::FragmentOutput;
import bevy_voxel_world::voxel_io::VoxelVertexOutput;

@fragment
fn fragment(
    in: VoxelVertexOutput,
) -> FragmentOutput {
    var out: FragmentOutput;

    // tex_idx will be the array produced by the texture_index_mapper
    // You can use it to send three unsigned integers to the shader
    // based on the voxel type.
    let tex_idx = in.tex_idx;
    if (tex_idx[0] == 1u) {
        out.color = vec4<f32>(1.0, 0.1, 0.1, 1.0);
    }
    if (tex_idx[0] == 2u) {
        out.color = vec4<f32>(0.1, 1.0, 0.1, 1.0);
    }
    if (tex_idx[0] == 3u) {
        out.color = vec4<f32>(0.1, 0.1, 1.0, 1.0);
    }

    // Multiply by the vertex color to get amient occlusion
    out.color = out.color * in.color;

    return out;
}
