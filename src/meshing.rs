use std::sync::Arc;

use block_mesh::{
    visible_block_faces, OrientedBlockFace, UnitQuadBuffer, Voxel, VoxelVisibility,
    RIGHT_HANDED_Y_UP_CONFIG,
};

use block_mesh::ilattice::glam::Vec3 as BMVec3;

use bevy::{
    asset::RenderAssetUsages,
    mesh::{Indices, VertexAttributeValues},
    prelude::*,
    render::render_resource::PrimitiveTopology,
};
use ndshape::{RuntimeShape, Shape};

use crate::{
    chunk::{VoxelArray, CHUNK_SIZE_F, PADDED_CHUNK_SIZE},
    prelude::TextureIndexMapperFn,
    voxel::WorldVoxel,
    voxel_material::ATTRIBUTE_TEX_INDEX,
};

/// Generate a mesh for the given chunks, or None of the chunk is empty
pub fn generate_chunk_mesh<I: PartialEq + Copy>(
    voxels: VoxelArray<I>,
    _pos: IVec3,
    texture_index_mapper: TextureIndexMapperFn<I>,
) -> Mesh {
    generate_chunk_mesh_for_shape(
        voxels,
        _pos,
        UVec3::splat(PADDED_CHUNK_SIZE),
        UVec3::splat(PADDED_CHUNK_SIZE),
        texture_index_mapper,
    )
}

/// Generate a mesh for the given voxels using an arbitrary padded chunk shape.
/// If `mesh_padded_shape` differs from `data_padded_shape`, downsample
pub fn generate_chunk_mesh_for_shape<I: PartialEq + Copy>(
    voxels: Arc<[WorldVoxel<I>]>,
    _pos: IVec3,
    data_padded_shape: UVec3,
    mesh_padded_shape: UVec3,
    texture_index_mapper: TextureIndexMapperFn<I>,
) -> Mesh {
    let faces = RIGHT_HANDED_Y_UP_CONFIG.faces;
    let mut buffer = UnitQuadBuffer::new();

    let data_shape = RuntimeShape::<u32, 3>::new(data_padded_shape.to_array());
    let mesh_shape = RuntimeShape::<u32, 3>::new(mesh_padded_shape.to_array());

    let voxels_for_mesh: Arc<[WorldVoxel<I>]> = if data_padded_shape != mesh_padded_shape
    {
        let coarse = resample_voxels_nearest(voxels.as_ref(), &data_shape, &mesh_shape);
        Arc::<[WorldVoxel<I>]>::from(coarse)
    } else {
        voxels.clone()
    };

    let max = [
        mesh_padded_shape.x.saturating_sub(1),
        mesh_padded_shape.y.saturating_sub(1),
        mesh_padded_shape.z.saturating_sub(1),
    ];

    visible_block_faces(
        voxels_for_mesh.as_ref(),
        &mesh_shape,
        [0; 3],
        max,
        &faces,
        &mut buffer,
    );

    mesh_from_quads_for_shape(
        buffer,
        faces,
        voxels_for_mesh.as_ref(),
        texture_index_mapper,
        &mesh_shape,
    )
}

/// Create a Bevy Mesh from a block_mesh::UnitQuadBuffer
pub fn mesh_from_quads<I: PartialEq + Copy>(
    quads: UnitQuadBuffer,
    faces: [OrientedBlockFace; 6],
    voxels: VoxelArray<I>,
    texture_index_mapper: Arc<dyn Fn(I) -> [u32; 3] + Send + Sync>,
) -> Mesh {
    let shape = RuntimeShape::<u32, 3>::new([PADDED_CHUNK_SIZE; 3]);
    mesh_from_quads_for_shape(quads, faces, voxels.as_ref(), texture_index_mapper, &shape)
}

fn mesh_from_quads_for_shape<I: PartialEq + Copy>(
    quads: UnitQuadBuffer,
    faces: [OrientedBlockFace; 6],
    voxels: &[WorldVoxel<I>],
    texture_index_mapper: Arc<dyn Fn(I) -> [u32; 3] + Send + Sync>,
    shape: &RuntimeShape<u32, 3>,
) -> Mesh {
    let num_indices = quads.num_quads() * 6;
    let num_vertices = quads.num_quads() * 4;

    let mut indices = Vec::with_capacity(num_indices);
    let mut positions = Vec::with_capacity(num_vertices);
    let mut normals = Vec::with_capacity(num_vertices);
    let mut tex_coords = Vec::with_capacity(num_vertices);
    let mut material_types = Vec::with_capacity(num_vertices);
    let mut aos = Vec::with_capacity(num_vertices);

    let voxel_size = voxel_size_from_shape(shape);

    for (group, face) in quads.groups.into_iter().zip(faces.into_iter()) {
        for quad in group.into_iter() {
            let quad = Into::<block_mesh::geometry::UnorientedQuad>::into(quad);
            let normal = IVec3::from([
                face.signed_normal().x,
                face.signed_normal().y,
                face.signed_normal().z,
            ]);

            let ao = face_aos(&quad.minimum, &normal, voxels, shape);
            aos.extend_from_slice(&ao);

            // TODO: Fix AO anisotropy
            indices.extend_from_slice(&face.quad_mesh_indices(positions.len() as u32));

            let corners = face.quad_corners(&quad);

            positions.extend_from_slice(
                &corners.map(|c| {
                    let corner = c.as_vec3();
                    let adjusted =
                        voxel_size * (corner - BMVec3::splat(1.0)) + BMVec3::splat(1.0);
                    adjusted.to_array()
                }),
            );

            normals.extend_from_slice(&face.quad_mesh_normals());

            let u_direction = corners[1].as_vec3() - corners[0].as_vec3();
            let v_direction = corners[2].as_vec3() - corners[0].as_vec3();
            let u_scale = voxel_size.dot(u_direction) / quad.width.max(1) as f32;
            let v_scale = voxel_size.dot(v_direction) / quad.height.max(1) as f32;

            let scaled_tex_coords = face
                .tex_coords(RIGHT_HANDED_Y_UP_CONFIG.u_flip_face, true, &quad)
                .map(|[u, v]| [u * u_scale, v * v_scale]);
            tex_coords.extend_from_slice(&scaled_tex_coords);

            let voxel_index = shape.linearize(quad.minimum) as usize;
            let material_type = match voxels[voxel_index] {
                WorldVoxel::Solid(mt) => texture_index_mapper(mt),
                _ => [0, 0, 0],
            };
            material_types.extend(std::iter::repeat_n(material_type, 4));
        }
    }

    let mut render_mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );

    render_mesh.insert_attribute(
        Mesh::ATTRIBUTE_POSITION,
        VertexAttributeValues::Float32x3(positions.clone()),
    );
    render_mesh.insert_attribute(
        Mesh::ATTRIBUTE_NORMAL,
        VertexAttributeValues::Float32x3(normals),
    );
    render_mesh.insert_attribute(
        Mesh::ATTRIBUTE_UV_0,
        VertexAttributeValues::Float32x2(tex_coords),
    );
    render_mesh.insert_attribute(
        ATTRIBUTE_TEX_INDEX,
        VertexAttributeValues::Uint32x3(material_types),
    );

    // Apply ambient occlusion values
    {
        let colors: Vec<[f32; 4]> = positions
            .iter()
            .enumerate()
            .map(|(i, _)| match aos[i] {
                0 => [0.1, 0.1, 0.1, 1.0],
                1 => [0.3, 0.3, 0.3, 1.0],
                2 => [0.5, 0.5, 0.5, 1.0],
                3 => [1.0, 1.0, 1.0, 1.0],
                _ => [1.0, 1.0, 1.0, 1.0],
            })
            .collect();
        render_mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
    }

    render_mesh.insert_indices(Indices::U32(indices.clone()));

    render_mesh
}

#[inline]
fn map_nearest_1d(mesh_i: u32, mesh_dim: u32, data_dim: u32) -> u32 {
    let mesh_inner = (mesh_dim.saturating_sub(2)).max(1);
    let data_inner = (data_dim.saturating_sub(2)).max(1);

    if mesh_inner == data_inner {
        return mesh_i;
    }

    // Keep padded bounds aligned so that resampling never erodes the outermost voxels.
    if mesh_i == 0 {
        return 0;
    }
    if mesh_i >= mesh_dim - 1 {
        return data_dim - 1;
    }

    let mesh_steps = (mesh_inner - 1).max(1) as f32;
    let data_steps = (data_inner - 1).max(1) as f32;
    let ratio = data_steps / mesh_steps;
    let inner_idx = (mesh_i - 1) as f32;
    let mapped = (inner_idx * ratio).round();

    (mapped as u32 + 1).min(data_dim - 1)
}

fn resample_voxels_nearest<I: PartialEq + Copy>(
    data_voxels: &[WorldVoxel<I>],
    data_shape: &RuntimeShape<u32, 3>,
    mesh_shape: &RuntimeShape<u32, 3>,
) -> Vec<WorldVoxel<I>>
where
    WorldVoxel<I>: Clone,
{
    let [mx, my, mz] = mesh_shape.as_array();
    let [dx, dy, dz] = data_shape.as_array();

    let mut out = Vec::with_capacity(mesh_shape.size() as usize);

    for lin in 0..mesh_shape.size() {
        let [ix, iy, iz] = mesh_shape.delinearize(lin);
        let sx = map_nearest_1d(ix, mx, dx);
        let sy = map_nearest_1d(iy, my, dy);
        let sz = map_nearest_1d(iz, mz, dz);

        let src_lin = data_shape.linearize([sx, sy, sz]);
        out.push(data_voxels[src_lin as usize]);
    }

    out
}

fn voxel_size_from_shape(shape: &RuntimeShape<u32, 3>) -> BMVec3 {
    let [ex, ey, ez] = shape.as_array();
    let ix = (ex.saturating_sub(2)).max(1);
    let iy = (ey.saturating_sub(2)).max(1);
    let iz = (ez.saturating_sub(2)).max(1);

    BMVec3::new(
        CHUNK_SIZE_F / ix as f32,
        CHUNK_SIZE_F / iy as f32,
        CHUNK_SIZE_F / iz as f32,
    )
}

fn ao_value(side1: bool, corner: bool, side2: bool) -> u32 {
    match (side1, corner, side2) {
        (true, _, true) => 0,
        (true, true, false) | (false, true, true) => 1,
        (false, false, false) => 3,
        _ => 2,
    }
}

fn side_aos<I: PartialEq>(neighbours: [WorldVoxel<I>; 8]) -> [u32; 4] {
    let ns = [
        neighbours[0].get_visibility() == VoxelVisibility::Opaque,
        neighbours[1].get_visibility() == VoxelVisibility::Opaque,
        neighbours[2].get_visibility() == VoxelVisibility::Opaque,
        neighbours[3].get_visibility() == VoxelVisibility::Opaque,
        neighbours[4].get_visibility() == VoxelVisibility::Opaque,
        neighbours[5].get_visibility() == VoxelVisibility::Opaque,
        neighbours[6].get_visibility() == VoxelVisibility::Opaque,
        neighbours[7].get_visibility() == VoxelVisibility::Opaque,
    ];

    [
        ao_value(ns[0], ns[1], ns[2]),
        ao_value(ns[2], ns[3], ns[4]),
        ao_value(ns[6], ns[7], ns[0]),
        ao_value(ns[4], ns[5], ns[6]),
    ]
}

fn face_aos<I: PartialEq + Copy>(
    voxel_pos: &[u32; 3],
    face_normal: &IVec3,
    voxels: &[WorldVoxel<I>],
    shape: &RuntimeShape<u32, 3>,
) -> [u32; 4] {
    let [x, y, z] = *voxel_pos;

    match *face_normal {
        IVec3::NEG_X => side_aos([
            voxels[shape.linearize([x - 1, y, z - 1]) as usize],
            voxels[shape.linearize([x - 1, y - 1, z - 1]) as usize],
            voxels[shape.linearize([x - 1, y - 1, z]) as usize],
            voxels[shape.linearize([x - 1, y - 1, z + 1]) as usize],
            voxels[shape.linearize([x - 1, y, z + 1]) as usize],
            voxels[shape.linearize([x - 1, y + 1, z + 1]) as usize],
            voxels[shape.linearize([x - 1, y + 1, z]) as usize],
            voxels[shape.linearize([x - 1, y + 1, z - 1]) as usize],
        ]),
        IVec3::X => side_aos([
            voxels[shape.linearize([x + 1, y, z - 1]) as usize],
            voxels[shape.linearize([x + 1, y - 1, z - 1]) as usize],
            voxels[shape.linearize([x + 1, y - 1, z]) as usize],
            voxels[shape.linearize([x + 1, y - 1, z + 1]) as usize],
            voxels[shape.linearize([x + 1, y, z + 1]) as usize],
            voxels[shape.linearize([x + 1, y + 1, z + 1]) as usize],
            voxels[shape.linearize([x + 1, y + 1, z]) as usize],
            voxels[shape.linearize([x + 1, y + 1, z - 1]) as usize],
        ]),
        IVec3::NEG_Y => side_aos([
            voxels[shape.linearize([x, y - 1, z - 1]) as usize],
            voxels[shape.linearize([x - 1, y - 1, z - 1]) as usize],
            voxels[shape.linearize([x - 1, y - 1, z]) as usize],
            voxels[shape.linearize([x - 1, y - 1, z + 1]) as usize],
            voxels[shape.linearize([x, y - 1, z + 1]) as usize],
            voxels[shape.linearize([x + 1, y - 1, z + 1]) as usize],
            voxels[shape.linearize([x + 1, y - 1, z]) as usize],
            voxels[shape.linearize([x + 1, y - 1, z - 1]) as usize],
        ]),
        IVec3::Y => side_aos([
            voxels[shape.linearize([x, y + 1, z - 1]) as usize],
            voxels[shape.linearize([x - 1, y + 1, z - 1]) as usize],
            voxels[shape.linearize([x - 1, y + 1, z]) as usize],
            voxels[shape.linearize([x - 1, y + 1, z + 1]) as usize],
            voxels[shape.linearize([x, y + 1, z + 1]) as usize],
            voxels[shape.linearize([x + 1, y + 1, z + 1]) as usize],
            voxels[shape.linearize([x + 1, y + 1, z]) as usize],
            voxels[shape.linearize([x + 1, y + 1, z - 1]) as usize],
        ]),
        IVec3::NEG_Z => side_aos([
            voxels[shape.linearize([x - 1, y, z - 1]) as usize],
            voxels[shape.linearize([x - 1, y - 1, z - 1]) as usize],
            voxels[shape.linearize([x, y - 1, z - 1]) as usize],
            voxels[shape.linearize([x + 1, y - 1, z - 1]) as usize],
            voxels[shape.linearize([x + 1, y, z - 1]) as usize],
            voxels[shape.linearize([x + 1, y + 1, z - 1]) as usize],
            voxels[shape.linearize([x, y + 1, z - 1]) as usize],
            voxels[shape.linearize([x - 1, y + 1, z - 1]) as usize],
        ]),
        IVec3::Z => side_aos([
            voxels[shape.linearize([x - 1, y, z + 1]) as usize],
            voxels[shape.linearize([x - 1, y - 1, z + 1]) as usize],
            voxels[shape.linearize([x, y - 1, z + 1]) as usize],
            voxels[shape.linearize([x + 1, y - 1, z + 1]) as usize],
            voxels[shape.linearize([x + 1, y, z + 1]) as usize],
            voxels[shape.linearize([x + 1, y + 1, z + 1]) as usize],
            voxels[shape.linearize([x, y + 1, z + 1]) as usize],
            voxels[shape.linearize([x - 1, y + 1, z + 1]) as usize],
        ]),
        _ => unreachable!(),
    }
}
