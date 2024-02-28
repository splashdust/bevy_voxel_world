use bevy::{
    asset::load_internal_asset,
    pbr::ExtendedMaterial,
    prelude::*,
    render::{
        render_asset::RenderAssetUsages,
        texture::{CompressedImageFormats, ImageSampler, ImageType},
    },
};

use crate::{
    configuration::VoxelWorldConfiguration,
    voxel_material::{
        prepare_texture, LoadingTexture, StandardVoxelMaterial, TextureLayers,
        VOXEL_TEXTURE_SHADER_HANDLE,
    },
    voxel_world::*,
    voxel_world_internal::{
        assign_material, despawn_retired_chunks, flush_chunk_map_buffers, flush_mesh_cache_buffers,
        flush_voxel_write_buffer, remesh_dirty_chunks, retire_chunks, setup_internals,
        spawn_chunks, spawn_meshes,
    },
};

#[derive(Resource)]
pub struct VoxelWorldMaterialHandle<M: Material> {
    pub handle: Handle<M>,
}

pub struct VoxelWorldMaterialPlugin<M: Material> {
    _marker: std::marker::PhantomData<M>,
}

impl<M: Material> Plugin for VoxelWorldMaterialPlugin<M> {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, assign_material::<M>);
    }
}

impl<M: Material> Default for VoxelWorldMaterialPlugin<M> {
    fn default() -> Self {
        Self {
            _marker: std::marker::PhantomData,
        }
    }
}

pub struct VoxelWorldPlugin {
    spawn_meshes: bool,
    voxel_texture: String,
    texture_layers: u32,
    use_custom_material: bool,
}

impl VoxelWorldPlugin {
    pub fn minimal() -> Self {
        Self {
            spawn_meshes: false,
            voxel_texture: "".to_string(),
            texture_layers: 0,
            use_custom_material: false,
        }
    }

    pub fn with_voxel_texture(mut self, texture: &str, layers: u32) -> Self {
        self.voxel_texture = texture.to_string();
        self.texture_layers = layers;
        self
    }

    pub fn without_default_material(mut self) -> Self {
        self.use_custom_material = true;
        self
    }
}

impl Default for VoxelWorldPlugin {
    fn default() -> Self {
        Self {
            spawn_meshes: true,
            voxel_texture: "".to_string(),
            texture_layers: 0,
            use_custom_material: false,
        }
    }
}

impl Plugin for VoxelWorldPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<VoxelWorldConfiguration>()
            .add_systems(PreStartup, setup_internals)
            .add_systems(
                PreUpdate,
                (
                    ((spawn_chunks, retire_chunks).chain(), remesh_dirty_chunks).chain(),
                    (
                        flush_voxel_write_buffer,
                        despawn_retired_chunks,
                        (flush_chunk_map_buffers, flush_mesh_cache_buffers),
                    )
                        .chain(),
                ),
            )
            .add_event::<ChunkWillSpawn>()
            .add_event::<ChunkWillDespawn>()
            .add_event::<ChunkWillRemesh>();

        // Spawning of meshes is optional, mainly to simplify testing.
        // This makes voxel_world work with a MinimalPlugins setup.
        if self.spawn_meshes {
            load_internal_asset!(
                app,
                VOXEL_TEXTURE_SHADER_HANDLE,
                "shaders/voxel_texture.wgsl",
                Shader::from_wgsl
            );

            app.add_systems(Update, spawn_meshes);
        }

        if !self.use_custom_material && self.spawn_meshes {
            app.add_plugins(MaterialPlugin::<
                ExtendedMaterial<StandardMaterial, StandardVoxelMaterial>,
            >::default());

            let mut preloaded_texture = true;

            // Use built-in default texture if no texture is specified.
            let image_handle = if self.voxel_texture.is_empty() {
                let mut image = Image::from_buffer(
                    include_bytes!("shaders/default_texture.png"),
                    ImageType::MimeType("image/png"),
                    CompressedImageFormats::default(),
                    false,
                    ImageSampler::Default,
                    RenderAssetUsages::default(),
                )
                .unwrap();
                image.reinterpret_stacked_2d_as_array(4);
                let mut image_assets = app.world.resource_mut::<Assets<Image>>();
                image_assets.add(image)
            } else {
                let asset_server = app.world.get_resource::<AssetServer>().unwrap();
                preloaded_texture = false;
                asset_server.load(self.voxel_texture.clone())
            };

            let mut material_assets = app
                .world
                .resource_mut::<Assets<ExtendedMaterial<StandardMaterial, StandardVoxelMaterial>>>(
                );

            let mat_handle = material_assets.add(ExtendedMaterial {
                base: StandardMaterial {
                    reflectance: 0.05,
                    metallic: 0.05,
                    perceptual_roughness: 0.95,
                    ..default()
                },
                extension: StandardVoxelMaterial {
                    voxels_texture: image_handle.clone(),
                },
            });

            app.insert_resource(LoadingTexture {
                is_loaded: preloaded_texture,
                handle: image_handle,
            });
            app.insert_resource(VoxelWorldMaterialHandle { handle: mat_handle });
            app.insert_resource(TextureLayers(self.texture_layers));

            app.add_systems(Update, prepare_texture);
            app.add_plugins(VoxelWorldMaterialPlugin::<
                ExtendedMaterial<StandardMaterial, StandardVoxelMaterial>,
            >::default());
        }

        if self.use_custom_material {
            app.insert_resource(LoadingTexture {
                is_loaded: true,
                handle: Handle::default(),
            });
        }
    }
}
