use std::marker::PhantomData;

use bevy::image::{
    CompressedImageFormats, ImageAddressMode, ImageSampler, ImageType,
    TextureReinterpretationError,
};
use bevy::{
    asset::{uuid_handle, RenderAssetUsages},
    mesh::{MeshVertexAttribute, MeshVertexBufferLayoutRef, VertexAttributeDescriptor},
    pbr::{
        ExtendedMaterial, MaterialExtension, MaterialExtensionKey,
        MaterialExtensionPipeline,
    },
    prelude::*,
    reflect::TypePath,
    render::render_resource::{
        AsBindGroup, RenderPipelineDescriptor, SpecializedMeshPipelineError,
        TextureViewDescriptor, TextureViewDimension, VertexFormat,
    },
    shader::{Shader, ShaderDefVal, ShaderRef},
};

use crate::configuration::VoxelWorldConfig;

/// Keeps track of the loading status of the image used for the voxel texture of
/// world `C`. Only worlds with the default material have one.
#[derive(Resource)]
pub(crate) struct LoadingTexture<C> {
    pub is_loaded: bool,
    pub handle: Handle<Image>,
    pub layers: u32,
    _world: PhantomData<C>,
}

impl<C> LoadingTexture<C> {
    pub fn new(is_loaded: bool, handle: Handle<Image>, layers: u32) -> Self {
        Self {
            is_loaded,
            handle,
            layers,
            _world: PhantomData,
        }
    }
}

pub const VOXEL_TEXTURE_SHADER_HANDLE: Handle<Shader> =
    uuid_handle!("df1398dc-56ad-4cd7-9bc2-7678cab2f144");

/// The shader module with the vertex input and output of chunk meshes, imported
/// as `bevy_voxel_world::voxel_io`.
///
/// Its `VoxelVertex` reads the attributes at the locations `vertex_layout`
/// gives them, and its `VoxelVertexOutput` is what the crate's vertex shader
/// returns. A custom material that keeps the crate's vertex shader can take a
/// `VoxelVertexOutput` in its fragment shader.
pub(crate) const VOXEL_IO_SHADER_HANDLE: Handle<Shader> =
    uuid_handle!("6f0e7d43-1b8a-4c52-9d3e-a2b4c6d8e0f1");

/// Adds the crate's shaders to `shaders`.
pub(crate) fn add_shaders(shaders: &mut Assets<Shader>) {
    // An embedded path gives the module its import path.
    let voxel_io = Shader::from_wesl(
        include_str!("shaders/voxel_io.wesl"),
        "embedded://bevy_voxel_world/voxel_io.wesl",
    );
    shaders
        .insert(VOXEL_IO_SHADER_HANDLE.id(), voxel_io)
        .unwrap();
}

pub const ATTRIBUTE_TEX_INDEX: MeshVertexAttribute =
    MeshVertexAttribute::new("TextureIndex", 989640910, VertexFormat::Uint32x3);

pub fn vertex_layout() -> Vec<VertexAttributeDescriptor> {
    vec![
        Mesh::ATTRIBUTE_POSITION.at_shader_location(0),
        Mesh::ATTRIBUTE_NORMAL.at_shader_location(1),
        Mesh::ATTRIBUTE_UV_0.at_shader_location(2),
        //Mesh::ATTRIBUTE_TANGENT.at_shader_location(4),
        Mesh::ATTRIBUTE_COLOR.at_shader_location(5),
        //Mesh::ATTRIBUTE_JOINT_INDEX.at_shader_location(6),
        //Mesh::ATTRIBUTE_JOINT_WEIGHT.at_shader_location(7),
        ATTRIBUTE_TEX_INDEX.at_shader_location(8),
    ]
}
#[derive(Asset, AsBindGroup, Debug, Clone, Default, TypePath)]
pub(crate) struct StandardVoxelMaterial {
    #[texture(100, dimension = "2d_array")]
    #[sampler(101)]
    pub voxels_texture: Handle<Image>,
}

impl Material for StandardVoxelMaterial {
    fn enable_prepass() -> bool {
        false
    }
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
        if descriptor
            .vertex
            .shader_defs
            .contains(&ShaderDefVal::Bool("PREPASS_PIPELINE".into(), true))
        {
            return Ok(());
        }

        let vertex_layout = layout.0.get_layout(&vertex_layout())?;
        descriptor.vertex.buffers = vec![vertex_layout];
        Ok(())
    }
}

/// The material of worlds that don't call `with_material`.
pub(crate) type DefaultVoxelMaterial =
    ExtendedMaterial<StandardMaterial, StandardVoxelMaterial>;

/// Adds the default material of world `C` and returns its handle.
///
/// The material samples the world's voxel texture, or the built-in texture if
/// the world has none. Chunks wait for a voxel texture loaded from a file until
/// `prepare_texture` has made it an array texture.
pub(crate) fn add_default_material<C: VoxelWorldConfig>(
    app: &mut App,
    config: &C,
) -> Handle<DefaultVoxelMaterial> {
    if app
        .get_added_plugins::<MaterialPlugin<DefaultVoxelMaterial>>()
        .is_empty()
    {
        app.add_plugins(MaterialPlugin::<DefaultVoxelMaterial>::default());
    }

    let loading_texture: LoadingTexture<C> = match config.voxel_texture() {
        Some((img_path, layers)) => {
            let asset_server = app.world().resource::<AssetServer>();
            LoadingTexture::new(false, asset_server.load(img_path), layers)
        }
        None => {
            let mut image = Image::from_buffer(
                include_bytes!("shaders/default_texture.png"),
                ImageType::MimeType("image/png"),
                CompressedImageFormats::default(),
                false,
                ImageSampler::Default,
                RenderAssetUsages::default(),
            )
            .unwrap();
            prepare_voxel_texture(&mut image, 4)
                .expect("default voxel texture should be a valid 4-layer array texture");
            let handle = app.world_mut().resource_mut::<Assets<Image>>().add(image);
            LoadingTexture::new(true, handle, 0)
        }
    };

    let handle = app
        .world_mut()
        .resource_mut::<Assets<DefaultVoxelMaterial>>()
        .add(ExtendedMaterial {
            base: StandardMaterial {
                reflectance: 0.05,
                metallic: 0.05,
                perceptual_roughness: 0.95,
                ..default()
            },
            extension: StandardVoxelMaterial {
                voxels_texture: loading_texture.handle.clone(),
            },
        });

    app.insert_resource(loading_texture);
    app.add_systems(Update, prepare_texture::<C>);
    handle
}

pub(crate) fn prepare_texture<C: VoxelWorldConfig>(
    asset_server: Res<AssetServer>,
    mut loading_texture: ResMut<LoadingTexture<C>>,
    mut images: ResMut<Assets<Image>>,
) {
    if loading_texture.is_loaded
        || !matches!(
            asset_server.get_load_state(loading_texture.handle.clone().id()),
            Some(bevy::asset::LoadState::Loaded)
        )
    {
        return;
    }

    let mut image = images.get_mut(&loading_texture.handle).unwrap();
    if let Err(err) = prepare_voxel_texture(&mut image, loading_texture.layers) {
        warn_once!("Failed to prepare voxel texture as a texture array: {err}");
    }

    loading_texture.is_loaded = true;
}

pub(crate) fn prepare_voxel_texture(
    image: &mut Image,
    texture_layers: u32,
) -> Result<(), TextureReinterpretationError> {
    set_repeat_sampler(image);
    // Worlds with the same texture path share one image, so another world may
    // already have made it an array texture.
    let is_array = texture_layers > 1
        && image.texture_descriptor.size.depth_or_array_layers == texture_layers;
    if !is_array {
        image.reinterpret_stacked_2d_as_array(texture_layers)?;
    }
    image.texture_view_descriptor = Some(TextureViewDescriptor {
        dimension: Some(TextureViewDimension::D2Array),
        ..default()
    });

    Ok(())
}

pub(crate) fn set_repeat_sampler(image: &mut Image) {
    let descriptor = image.sampler.get_or_init_descriptor();
    descriptor.address_mode_u = ImageAddressMode::Repeat;
    descriptor.address_mode_v = ImageAddressMode::Repeat;
    descriptor.address_mode_w = ImageAddressMode::Repeat;
}
