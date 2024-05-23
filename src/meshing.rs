use std::sync::Arc;

use block_mesh::{
    visible_block_faces, OrientedBlockFace, UnitQuadBuffer, Voxel, VoxelVisibility,
    RIGHT_HANDED_Y_UP_CONFIG,
};

use bevy::{
    prelude::*,
    render::{
        mesh::{Indices, VertexAttributeValues},
        render_asset::RenderAssetUsages,
        render_resource::PrimitiveTopology,
    },
};
use ndshape::ConstShape;

use crate::{
    chunk::{PaddedChunkShape, CHUNK_SIZE_U},
    voxel::WorldVoxel,
    voxel_material::ATTRIBUTE_TEX_INDEX,
};

type VoxelArray = Arc<[WorldVoxel; PaddedChunkShape::SIZE as usize]>;

/// Generate a mesh for the given chunks, or None of the chunk is empty
pub fn generate_chunk_mesh(
    voxels: VoxelArray,
    _pos: IVec3,
    texture_index_mapper: Arc<dyn Fn(u8) -> [u32; 3] + Send + Sync>,
) -> Mesh {
    let faces = RIGHT_HANDED_Y_UP_CONFIG.faces;
    let mut buffer = UnitQuadBuffer::new();

    visible_block_faces(
        &*voxels,
        &PaddedChunkShape {},
        [0; 3],
        [CHUNK_SIZE_U + 1; 3],
        &faces,
        &mut buffer,
    );

    mesh_from_quads(buffer, faces, voxels, texture_index_mapper)
}

/// Convert a QuadBuffer into a Bevy Mesh
fn mesh_from_quads(
    quads: UnitQuadBuffer,
    faces: [OrientedBlockFace; 6],
    voxels: VoxelArray,
    texture_index_mapper: Arc<dyn Fn(u8) -> [u32; 3] + Send + Sync>,
) -> Mesh {
    let num_indices = quads.num_quads() * 6;
    let num_vertices = quads.num_quads() * 4;

    let mut indices = Vec::with_capacity(num_indices);
    let mut positions = Vec::with_capacity(num_vertices);
    let mut normals = Vec::with_capacity(num_vertices);
    let mut tex_coords = Vec::with_capacity(num_vertices);
    let mut material_types = Vec::with_capacity(num_vertices);
    let mut aos = Vec::with_capacity(num_vertices);

    for (group, face) in quads.groups.into_iter().zip(faces.into_iter()) {
        for quad in group.into_iter() {
            let normal = IVec3::from([
                face.signed_normal().x,
                face.signed_normal().y,
                face.signed_normal().z,
            ]);

            let ao = face_aos(&quad.minimum, &normal, &voxels);
            aos.extend_from_slice(&ao);

            // TODO: Fix AO anisotropy
            indices.extend_from_slice(&face.quad_mesh_indices(positions.len() as u32));

            positions.extend_from_slice(&face.quad_mesh_positions(&quad.into(), 1.0));

            normals.extend_from_slice(&face.quad_mesh_normals());

            tex_coords.extend_from_slice(&face.tex_coords(
                RIGHT_HANDED_Y_UP_CONFIG.u_flip_face,
                true,
                &quad.into(),
            ));

            let voxel_index = PaddedChunkShape::linearize(quad.minimum) as usize;
            let material_type = match voxels[voxel_index] {
                WorldVoxel::Solid(mt) => texture_index_mapper(mt),
                _ => [0, 0, 0],
            };
            material_types.extend(std::iter::repeat(material_type).take(4));
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

fn ao_value(side1: bool, corner: bool, side2: bool) -> u32 {
    match (side1, corner, side2) {
        (true, _, true) => 0,
        (true, true, false) | (false, true, true) => 1,
        (false, false, false) => 3,
        _ => 2,
    }
}

fn side_aos(neighbours: [WorldVoxel; 8]) -> [u32; 4] {
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

fn face_aos(voxel_pos: &[u32; 3], face_normal: &IVec3, voxels: &VoxelArray) -> [u32; 4] {
    let [x, y, z] = *voxel_pos;

    match *face_normal {
        IVec3::NEG_X => side_aos([
            voxels[PaddedChunkShape::linearize([x - 1, y, z - 1]) as usize],
            voxels[PaddedChunkShape::linearize([x - 1, y - 1, z - 1]) as usize],
            voxels[PaddedChunkShape::linearize([x - 1, y - 1, z]) as usize],
            voxels[PaddedChunkShape::linearize([x - 1, y - 1, z + 1]) as usize],
            voxels[PaddedChunkShape::linearize([x - 1, y, z + 1]) as usize],
            voxels[PaddedChunkShape::linearize([x - 1, y + 1, z + 1]) as usize],
            voxels[PaddedChunkShape::linearize([x - 1, y + 1, z]) as usize],
            voxels[PaddedChunkShape::linearize([x - 1, y + 1, z - 1]) as usize],
        ]),
        IVec3::X => side_aos([
            voxels[PaddedChunkShape::linearize([x + 1, y, z - 1]) as usize],
            voxels[PaddedChunkShape::linearize([x + 1, y - 1, z - 1]) as usize],
            voxels[PaddedChunkShape::linearize([x + 1, y - 1, z]) as usize],
            voxels[PaddedChunkShape::linearize([x + 1, y - 1, z + 1]) as usize],
            voxels[PaddedChunkShape::linearize([x + 1, y, z + 1]) as usize],
            voxels[PaddedChunkShape::linearize([x + 1, y + 1, z + 1]) as usize],
            voxels[PaddedChunkShape::linearize([x + 1, y + 1, z]) as usize],
            voxels[PaddedChunkShape::linearize([x + 1, y + 1, z - 1]) as usize],
        ]),
        IVec3::NEG_Y => side_aos([
            voxels[PaddedChunkShape::linearize([x, y - 1, z - 1]) as usize],
            voxels[PaddedChunkShape::linearize([x - 1, y - 1, z - 1]) as usize],
            voxels[PaddedChunkShape::linearize([x - 1, y - 1, z]) as usize],
            voxels[PaddedChunkShape::linearize([x - 1, y - 1, z + 1]) as usize],
            voxels[PaddedChunkShape::linearize([x, y - 1, z + 1]) as usize],
            voxels[PaddedChunkShape::linearize([x + 1, y - 1, z + 1]) as usize],
            voxels[PaddedChunkShape::linearize([x + 1, y - 1, z]) as usize],
            voxels[PaddedChunkShape::linearize([x + 1, y - 1, z - 1]) as usize],
        ]),
        IVec3::Y => side_aos([
            voxels[PaddedChunkShape::linearize([x, y + 1, z - 1]) as usize],
            voxels[PaddedChunkShape::linearize([x - 1, y + 1, z - 1]) as usize],
            voxels[PaddedChunkShape::linearize([x - 1, y + 1, z]) as usize],
            voxels[PaddedChunkShape::linearize([x - 1, y + 1, z + 1]) as usize],
            voxels[PaddedChunkShape::linearize([x, y + 1, z + 1]) as usize],
            voxels[PaddedChunkShape::linearize([x + 1, y + 1, z + 1]) as usize],
            voxels[PaddedChunkShape::linearize([x + 1, y + 1, z]) as usize],
            voxels[PaddedChunkShape::linearize([x + 1, y + 1, z - 1]) as usize],
        ]),
        IVec3::NEG_Z => side_aos([
            voxels[PaddedChunkShape::linearize([x - 1, y, z - 1]) as usize],
            voxels[PaddedChunkShape::linearize([x - 1, y - 1, z - 1]) as usize],
            voxels[PaddedChunkShape::linearize([x, y - 1, z - 1]) as usize],
            voxels[PaddedChunkShape::linearize([x + 1, y - 1, z - 1]) as usize],
            voxels[PaddedChunkShape::linearize([x + 1, y, z - 1]) as usize],
            voxels[PaddedChunkShape::linearize([x + 1, y + 1, z - 1]) as usize],
            voxels[PaddedChunkShape::linearize([x, y + 1, z - 1]) as usize],
            voxels[PaddedChunkShape::linearize([x - 1, y + 1, z - 1]) as usize],
        ]),
        IVec3::Z => side_aos([
            voxels[PaddedChunkShape::linearize([x - 1, y, z + 1]) as usize],
            voxels[PaddedChunkShape::linearize([x - 1, y - 1, z + 1]) as usize],
            voxels[PaddedChunkShape::linearize([x, y - 1, z + 1]) as usize],
            voxels[PaddedChunkShape::linearize([x + 1, y - 1, z + 1]) as usize],
            voxels[PaddedChunkShape::linearize([x + 1, y, z + 1]) as usize],
            voxels[PaddedChunkShape::linearize([x + 1, y + 1, z + 1]) as usize],
            voxels[PaddedChunkShape::linearize([x, y + 1, z + 1]) as usize],
            voxels[PaddedChunkShape::linearize([x - 1, y + 1, z + 1]) as usize],
        ]),
        _ => unreachable!(),
    }
}
