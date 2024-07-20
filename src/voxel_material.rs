use bevy::{
    pbr::{MaterialExtension, MaterialExtensionKey, MaterialExtensionPipeline},
    prelude::*,
    reflect::TypePath,
    render::{
        mesh::{MeshVertexAttribute, MeshVertexBufferLayoutRef, VertexAttributeDescriptor},
        render_resource::{
            AsBindGroup, RenderPipelineDescriptor, ShaderRef, SpecializedMeshPipelineError,
            VertexFormat,
        },
    },
};

/// Keeps track of the loading status of the image used for the voxel texture
#[derive(Resource)]
pub(crate) struct LoadingTexture {
    pub is_loaded: bool,
    pub handle: Handle<Image>,
}

#[derive(Resource)]
pub(crate) struct TextureLayers(pub u32);

pub const VOXEL_TEXTURE_SHADER_HANDLE: Handle<Shader> = Handle::weak_from_u128(6998301138411443008);

pub(crate) const ATTRIBUTE_TEX_INDEX: MeshVertexAttribute =
    MeshVertexAttribute::new("TextureIndex", 989640910, VertexFormat::Uint32x3);

pub fn vertex_layout() -> Vec<VertexAttributeDescriptor> {
    vec![
        Mesh::ATTRIBUTE_POSITION.at_shader_location(0),
        Mesh::ATTRIBUTE_NORMAL.at_shader_location(1),
        Mesh::ATTRIBUTE_UV_0.at_shader_location(2),
        //Mesh::ATTRIBUTE_TANGENT.at_shader_location(4),
        Mesh::ATTRIBUTE_COLOR.at_shader_location(5),
        Mesh::ATTRIBUTE_COLOR.at_shader_location(7),
        //Mesh::ATTRIBUTE_JOINT_INDEX.at_shader_location(6),
        //Mesh::ATTRIBUTE_JOINT_WEIGHT.at_shader_location(7),
        ATTRIBUTE_TEX_INDEX.at_shader_location(8),
    ]
}
#[derive(Asset, AsBindGroup, Debug, Clone, TypePath)]
pub(crate) struct StandardVoxelMaterial {
    #[texture(100, dimension = "2d_array")]
    #[sampler(101)]
    pub voxels_texture: Handle<Image>,
}

impl MaterialExtension for StandardVoxelMaterial {
    fn fragment_shader() -> ShaderRef {
        VOXEL_TEXTURE_SHADER_HANDLE.into()
    }

    fn vertex_shader() -> ShaderRef {
        VOXEL_TEXTURE_SHADER_HANDLE.into()
    }

    fn specialize(
        _pipeline: &MaterialExtensionPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        layout: &MeshVertexBufferLayoutRef,
        _key: MaterialExtensionKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        let vertex_layout = layout.0.get_layout(&vertex_layout())?;
        descriptor.vertex.buffers = vec![vertex_layout];
        Ok(())
    }
}

pub(crate) fn prepare_texture(
    asset_server: Res<AssetServer>,
    texture_layers: Res<TextureLayers>,
    mut loading_texture: ResMut<LoadingTexture>,
    mut images: ResMut<Assets<Image>>,
) {
    if loading_texture.is_loaded
        || asset_server.get_load_state(loading_texture.handle.clone().id())
            != Some(bevy::asset::LoadState::Loaded)
    {
        return;
    }
    loading_texture.is_loaded = true;

    let image = images.get_mut(&loading_texture.handle).unwrap();
    image.reinterpret_stacked_2d_as_array(texture_layers.0);
}
