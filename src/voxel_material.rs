use bevy::{
    pbr::{MaterialPipeline, MaterialPipelineKey},
    prelude::*,
    reflect::{TypePath, TypeUuid},
    render::{
        mesh::{MeshVertexAttribute, MeshVertexBufferLayout},
        render_resource::{
            AsBindGroup, RenderPipelineDescriptor, ShaderRef,
            SpecializedMeshPipelineError, VertexFormat,
        },
    },
};

#[derive(Resource)]
pub(crate) struct VoxelTextureMaterialHandle(pub Handle<VoxelTextureMaterial>);

#[derive(Resource)]
pub(crate) struct TextureLayers(pub u32);

pub(crate) const VOXEL_TEXTURE_SHADER_HANDLE: HandleUntyped =
    HandleUntyped::weak_from_u64(Shader::TYPE_UUID, 3612826716914925351);

pub(crate) const ATTRIBUTE_TEX_INDEX: MeshVertexAttribute =
    MeshVertexAttribute::new("TextureIndex", 989640910, VertexFormat::Uint32x3);

#[derive(AsBindGroup, Debug, Clone, TypeUuid, TypePath)]
#[uuid = "303bc6fc-605d-45b9-9fe5-b3fae5a566b7"]
pub(crate) struct VoxelTextureMaterial {
    #[texture(0, dimension = "2d_array")]
    #[sampler(1)]
    pub voxels_texture: Handle<Image>,
}

impl Material for VoxelTextureMaterial {
    fn fragment_shader() -> ShaderRef {
        VOXEL_TEXTURE_SHADER_HANDLE.typed().into()
    }

    fn vertex_shader() -> ShaderRef {
        VOXEL_TEXTURE_SHADER_HANDLE.typed().into()
    }

    fn specialize(
        _pipeline: &MaterialPipeline<Self>,
        descriptor: &mut RenderPipelineDescriptor,
        layout: &MeshVertexBufferLayout,
        _key: MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        let vertex_layout = layout.get_layout(&[
            Mesh::ATTRIBUTE_POSITION.at_shader_location(0),
            Mesh::ATTRIBUTE_NORMAL.at_shader_location(1),
            Mesh::ATTRIBUTE_UV_0.at_shader_location(2),
            //Mesh::ATTRIBUTE_TANGENT.at_shader_location(3),
            Mesh::ATTRIBUTE_COLOR.at_shader_location(4),
            ATTRIBUTE_TEX_INDEX.at_shader_location(5),
        ])?;
        descriptor.vertex.buffers = vec![vertex_layout];
        Ok(())
    }
}
