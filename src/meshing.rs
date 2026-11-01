use std::sync::Arc;

use block_mesh::{
    geometry::UnorientedQuad, visible_block_faces, OrientedBlockFace, UnitQuadBuffer,
    Voxel, VoxelVisibility, RIGHT_HANDED_Y_UP_CONFIG,
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
    let data_shape = RuntimeShape::<u32, 3>::new(data_padded_shape.to_array());
    let mesh_shape = RuntimeShape::<u32, 3>::new(mesh_padded_shape.to_array());

    let resampled;
    let voxels_for_mesh: &[WorldVoxel<I>] = if data_padded_shape != mesh_padded_shape {
        resampled = resample_voxels_nearest(voxels.as_ref(), &data_shape, &mesh_shape);
        &resampled
    } else {
        &voxels
    };

    if mesh_padded_shape.max_element() <= MAX_BINARY_MESH_EXTENT {
        binary_chunk_mesh(voxels_for_mesh, &mesh_shape, texture_index_mapper)
    } else {
        block_mesh_chunk_mesh(voxels_for_mesh, &mesh_shape, texture_index_mapper)
    }
}

/// The most voxels along an axis that [`binary_chunk_mesh`] meshes. It keeps
/// each column of voxels as the bits of a `u64`.
const MAX_BINARY_MESH_EXTENT: u32 = u64::BITS;

/// Meshes `voxels` with one quad per visible face, finding the faces with bitwise
/// operations on columns of voxels. Makes the same quads as
/// [`block_mesh_chunk_mesh`]. `shape` is at most [`MAX_BINARY_MESH_EXTENT`]
/// voxels along each axis.
pub(crate) fn binary_chunk_mesh<I: PartialEq + Copy>(
    voxels: &[WorldVoxel<I>],
    shape: &RuntimeShape<u32, 3>,
    texture_index_mapper: TextureIndexMapperFn<I>,
) -> Mesh {
    let faces = RIGHT_HANDED_Y_UP_CONFIG.faces;
    let [size_x, size_y, size_z] = shape.as_array().map(|extent| extent as usize);
    // Faces are made only for the voxels inside the padding.
    if size_x < 3 || size_y < 3 || size_z < 3 {
        return MeshBuffers::with_capacity(0, shape).into_mesh();
    }

    // Bit z of the column at x + y * size_x is set where voxel (x, y, z) is opaque.
    let mut columns = vec![0u64; size_x * size_y];
    for (z, layer) in voxels.chunks_exact(size_x * size_y).enumerate() {
        for (column, voxel) in columns.iter_mut().zip(layer) {
            *column |= ((voxel.get_visibility() == VoxelVisibility::Opaque) as u64) << z;
        }
    }
    let interior_z = ((1u64 << (size_z - 1)) - 1) & !1;
    let column_index = |x: usize, y: usize| x + y * size_x;
    // The visible faces of the voxels in the column at (x, y), as bits, in the
    // order of `faces`.
    let face_bits = |x: usize, y: usize| {
        let column = columns[column_index(x, y)];
        let opaque = column & interior_z;
        [
            opaque & !columns[column_index(x - 1, y)],
            opaque & !columns[column_index(x, y - 1)],
            opaque & !(column << 1),
            opaque & !columns[column_index(x + 1, y)],
            opaque & !columns[column_index(x, y + 1)],
            opaque & !(column >> 1),
        ]
    };

    let mut quad_count = 0;
    for y in 1..size_y - 1 {
        for x in 1..size_x - 1 {
            quad_count += face_bits(x, y)
                .iter()
                .map(|bits| bits.count_ones() as usize)
                .sum::<usize>();
        }
    }

    let normals = faces.map(|face| IVec3::from(face.signed_normal().to_array()));
    let opaque =
        |[x, y, z]: [u32; 3]| columns[column_index(x as usize, y as usize)] >> z & 1 == 1;
    let mut buffers = MeshBuffers::with_capacity(quad_count, shape);
    for y in 1..size_y - 1 {
        for x in 1..size_x - 1 {
            for ((face, normal), mut bits) in
                faces.iter().zip(&normals).zip(face_bits(x, y))
            {
                while bits != 0 {
                    let minimum = [x as u32, y as u32, bits.trailing_zeros()];
                    bits &= bits - 1;
                    let material_type = match voxels[shape.linearize(minimum) as usize] {
                        WorldVoxel::Solid(mt) => texture_index_mapper(mt),
                        _ => [0, 0, 0],
                    };
                    buffers.push_quad(
                        face,
                        &UnorientedQuad {
                            minimum,
                            width: 1,
                            height: 1,
                        },
                        face_aos(&minimum, normal, opaque),
                        material_type,
                    );
                }
            }
        }
    }
    buffers.into_mesh()
}

/// Meshes `voxels` with block-mesh, one quad per visible face.
pub(crate) fn block_mesh_chunk_mesh<I: PartialEq + Copy>(
    voxels: &[WorldVoxel<I>],
    shape: &RuntimeShape<u32, 3>,
    texture_index_mapper: TextureIndexMapperFn<I>,
) -> Mesh {
    let faces = RIGHT_HANDED_Y_UP_CONFIG.faces;
    let mut buffer = UnitQuadBuffer::new();
    let max = shape.as_array().map(|extent| extent.saturating_sub(1));

    visible_block_faces(voxels, shape, [0; 3], max, &faces, &mut buffer);

    mesh_from_quads_for_shape(buffer, faces, voxels, texture_index_mapper, shape)
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
    let mut buffers = MeshBuffers::with_capacity(quads.num_quads(), shape);
    let opaque = |position: [u32; 3]| {
        voxels[shape.linearize(position) as usize].get_visibility()
            == VoxelVisibility::Opaque
    };

    for (group, face) in quads.groups.into_iter().zip(faces) {
        let normal = IVec3::from(face.signed_normal().to_array());
        for quad in group.into_iter() {
            let quad = Into::<UnorientedQuad>::into(quad);
            let voxel_index = shape.linearize(quad.minimum) as usize;
            let material_type = match voxels[voxel_index] {
                WorldVoxel::Solid(mt) => texture_index_mapper(mt),
                _ => [0, 0, 0],
            };
            buffers.push_quad(
                &face,
                &quad,
                face_aos(&quad.minimum, &normal, opaque),
                material_type,
            );
        }
    }

    buffers.into_mesh()
}

/// The vertex attributes and indices of a chunk mesh, added a quad at a time.
struct MeshBuffers {
    voxel_size: BMVec3,
    indices: Vec<u32>,
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    tex_coords: Vec<[f32; 2]>,
    material_types: Vec<[u32; 3]>,
    colors: Vec<[f32; 4]>,
}

impl MeshBuffers {
    /// Buffers with room for `quad_count` quads of a chunk with the padded `shape`.
    fn with_capacity(quad_count: usize, shape: &RuntimeShape<u32, 3>) -> Self {
        let vertex_count = quad_count * 4;
        Self {
            voxel_size: voxel_size_from_shape(shape),
            indices: Vec::with_capacity(quad_count * 6),
            positions: Vec::with_capacity(vertex_count),
            normals: Vec::with_capacity(vertex_count),
            tex_coords: Vec::with_capacity(vertex_count),
            material_types: Vec::with_capacity(vertex_count),
            colors: Vec::with_capacity(vertex_count),
        }
    }

    /// Adds `quad`, facing `face`, with the ambient occlusion `ao` at its corners
    /// and the texture indices `material_type`.
    fn push_quad(
        &mut self,
        face: &OrientedBlockFace,
        quad: &UnorientedQuad,
        ao: [u32; 4],
        material_type: [u32; 3],
    ) {
        // TODO: Fix AO anisotropy
        self.indices
            .extend_from_slice(&face.quad_mesh_indices(self.positions.len() as u32));

        let corners = face.quad_corners(quad);
        let voxel_size = self.voxel_size;

        self.positions.extend_from_slice(&corners.map(|c| {
            let corner = c.as_vec3();
            let adjusted =
                voxel_size * (corner - BMVec3::splat(1.0)) + BMVec3::splat(1.0);
            adjusted.to_array()
        }));

        self.normals.extend_from_slice(&face.quad_mesh_normals());

        let u_direction = corners[1].as_vec3() - corners[0].as_vec3();
        let v_direction = corners[2].as_vec3() - corners[0].as_vec3();
        let u_scale = voxel_size.dot(u_direction) / quad.width.max(1) as f32;
        let v_scale = voxel_size.dot(v_direction) / quad.height.max(1) as f32;

        let scaled_tex_coords = face
            .tex_coords(RIGHT_HANDED_Y_UP_CONFIG.u_flip_face, true, quad)
            .map(|[u, v]| [u * u_scale, v * v_scale]);
        self.tex_coords.extend_from_slice(&scaled_tex_coords);

        self.material_types
            .extend(std::iter::repeat_n(material_type, 4));
        self.colors.extend(ao.map(ao_color));
    }

    fn into_mesh(self) -> Mesh {
        let mut render_mesh = Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::default(),
        );

        render_mesh.insert_attribute(
            Mesh::ATTRIBUTE_POSITION,
            VertexAttributeValues::Float32x3(self.positions),
        );
        render_mesh.insert_attribute(
            Mesh::ATTRIBUTE_NORMAL,
            VertexAttributeValues::Float32x3(self.normals),
        );
        render_mesh.insert_attribute(
            Mesh::ATTRIBUTE_UV_0,
            VertexAttributeValues::Float32x2(self.tex_coords),
        );
        render_mesh.insert_attribute(
            ATTRIBUTE_TEX_INDEX,
            VertexAttributeValues::Uint32x3(self.material_types),
        );
        // Ambient occlusion
        render_mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, self.colors);
        render_mesh.insert_indices(Indices::U32(self.indices));

        render_mesh
    }
}

/// The vertex color of a corner with the ambient occlusion `ao`.
fn ao_color(ao: u32) -> [f32; 4] {
    match ao {
        0 => [0.1, 0.1, 0.1, 1.0],
        1 => [0.3, 0.3, 0.3, 1.0],
        2 => [0.5, 0.5, 0.5, 1.0],
        _ => [1.0, 1.0, 1.0, 1.0],
    }
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

fn side_aos(ns: [bool; 8]) -> [u32; 4] {
    [
        ao_value(ns[0], ns[1], ns[2]),
        ao_value(ns[2], ns[3], ns[4]),
        ao_value(ns[6], ns[7], ns[0]),
        ao_value(ns[4], ns[5], ns[6]),
    ]
}

/// The ambient occlusion at the corners of the face of the voxel at `voxel_pos`
/// facing `face_normal`, where `opaque` says whether the voxel at a position is
/// opaque.
fn face_aos(
    voxel_pos: &[u32; 3],
    face_normal: &IVec3,
    opaque: impl Fn([u32; 3]) -> bool,
) -> [u32; 4] {
    let [x, y, z] = *voxel_pos;

    let neighbours = match *face_normal {
        IVec3::NEG_X => [
            [x - 1, y, z - 1],
            [x - 1, y - 1, z - 1],
            [x - 1, y - 1, z],
            [x - 1, y - 1, z + 1],
            [x - 1, y, z + 1],
            [x - 1, y + 1, z + 1],
            [x - 1, y + 1, z],
            [x - 1, y + 1, z - 1],
        ],
        IVec3::X => [
            [x + 1, y, z - 1],
            [x + 1, y - 1, z - 1],
            [x + 1, y - 1, z],
            [x + 1, y - 1, z + 1],
            [x + 1, y, z + 1],
            [x + 1, y + 1, z + 1],
            [x + 1, y + 1, z],
            [x + 1, y + 1, z - 1],
        ],
        IVec3::NEG_Y => [
            [x, y - 1, z - 1],
            [x - 1, y - 1, z - 1],
            [x - 1, y - 1, z],
            [x - 1, y - 1, z + 1],
            [x, y - 1, z + 1],
            [x + 1, y - 1, z + 1],
            [x + 1, y - 1, z],
            [x + 1, y - 1, z - 1],
        ],
        IVec3::Y => [
            [x, y + 1, z - 1],
            [x - 1, y + 1, z - 1],
            [x - 1, y + 1, z],
            [x - 1, y + 1, z + 1],
            [x, y + 1, z + 1],
            [x + 1, y + 1, z + 1],
            [x + 1, y + 1, z],
            [x + 1, y + 1, z - 1],
        ],
        IVec3::NEG_Z => [
            [x - 1, y, z - 1],
            [x - 1, y - 1, z - 1],
            [x, y - 1, z - 1],
            [x + 1, y - 1, z - 1],
            [x + 1, y, z - 1],
            [x + 1, y + 1, z - 1],
            [x, y + 1, z - 1],
            [x - 1, y + 1, z - 1],
        ],
        IVec3::Z => [
            [x - 1, y, z + 1],
            [x - 1, y - 1, z + 1],
            [x, y - 1, z + 1],
            [x + 1, y - 1, z + 1],
            [x + 1, y, z + 1],
            [x + 1, y + 1, z + 1],
            [x, y + 1, z + 1],
            [x - 1, y + 1, z + 1],
        ],
        _ => unreachable!(),
    };

    side_aos([
        opaque(neighbours[0]),
        opaque(neighbours[1]),
        opaque(neighbours[2]),
        opaque(neighbours[3]),
        opaque(neighbours[4]),
        opaque(neighbours[5]),
        opaque(neighbours[6]),
        opaque(neighbours[7]),
    ])
}
