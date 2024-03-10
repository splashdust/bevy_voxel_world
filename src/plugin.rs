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
    configuration::{DefaultWorld, VoxelWorldConfig},
    voxel_material::{
        prepare_texture, LoadingTexture, StandardVoxelMaterial, TextureLayers,
        VOXEL_TEXTURE_SHADER_HANDLE,
    },
    voxel_world::*,
    voxel_world_internal::{assign_material, Internals},
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

/// The main plugin for the voxel world. This plugin sets up the voxel world and its dependencies.
/// The type parameter `C` is used to differentiate between different voxel worlds with different configs.
pub struct VoxelWorldPlugin<C: VoxelWorldConfig = DefaultWorld> {
    spawn_meshes: bool,
    use_custom_material: bool,
    config: C,
}

impl<C> VoxelWorldPlugin<C>
where
    C: VoxelWorldConfig,
{
    pub fn with_config(config: C) -> Self {
        Self {
            config,
            spawn_meshes: true,
            use_custom_material: false,
        }
    }

    pub fn minimal() -> Self {
        Self {
            spawn_meshes: false,
            use_custom_material: false,
            config: C::default(),
        }
    }

    pub fn without_default_material(mut self) -> Self {
        self.use_custom_material = true;
        self
    }
}

impl Default for VoxelWorldPlugin<DefaultWorld> {
    fn default() -> Self {
        Self {
            spawn_meshes: true,
            use_custom_material: false,
            config: DefaultWorld::default(),
        }
    }
}

impl<C> Plugin for VoxelWorldPlugin<C>
where
    C: VoxelWorldConfig,
{
    fn build(&self, app: &mut App) {
        app.init_resource::<C>()
            .add_systems(PreStartup, Internals::<C>::setup)
            .add_systems(
                PreUpdate,
                (
                    (
                        (Internals::<C>::spawn_chunks, Internals::<C>::retire_chunks).chain(),
                        Internals::<C>::remesh_dirty_chunks,
                    )
                        .chain(),
                    (
                        Internals::<C>::flush_voxel_write_buffer,
                        Internals::<C>::despawn_retired_chunks,
                        (
                            Internals::<C>::flush_chunk_map_buffers,
                            Internals::<C>::flush_mesh_cache_buffers,
                        ),
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

            app.add_systems(Update, Internals::<C>::spawn_meshes);
        }

        if !self.use_custom_material && self.spawn_meshes {
            let mat_plugins = app.get_added_plugins::<MaterialPlugin::<
                ExtendedMaterial<StandardMaterial, StandardVoxelMaterial>>>();

            if mat_plugins.is_empty() {
                app.add_plugins(MaterialPlugin::<
                    ExtendedMaterial<StandardMaterial, StandardVoxelMaterial>,
                >::default());
            }

            let mut preloaded_texture = true;
            let texture_conf = self.config.voxel_texture();
            let mut texture_layers = 0;

            // Use built-in default texture if no texture is specified.
            let image_handle = if texture_conf.is_none() {
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
                let (img_path, layers) = texture_conf.unwrap();
                texture_layers = layers;
                let asset_server = app.world.get_resource::<AssetServer>().unwrap();
                preloaded_texture = false;
                asset_server.load(img_path)
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
            app.insert_resource(TextureLayers(texture_layers));

            app.insert_resource(self.config.clone());

            app.add_systems(Update, prepare_texture);

            let voxel_mat_plugins =
                app.get_added_plugins::<VoxelWorldMaterialPlugin<
                    ExtendedMaterial<StandardMaterial, StandardVoxelMaterial>,
                >>();

            if voxel_mat_plugins.is_empty() {
                app.add_plugins(VoxelWorldMaterialPlugin::<
                    ExtendedMaterial<StandardMaterial, StandardVoxelMaterial>,
                >::default());
            }
        }

        if self.use_custom_material {
            app.insert_resource(LoadingTexture {
                is_loaded: true,
                handle: Handle::default(),
            });
        }
    }
}
