use bevy::{
    app::Plugins,
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
    voxel_world_internal::Internals,
};

#[derive(Resource)]
pub struct VoxelWorldMaterialHandle<M: Material> {
    pub handle: Handle<M>,
}

/// The main plugin for the voxel world. This plugin sets up the voxel world and its dependencies.
/// The type parameter `C` is used to differentiate between different voxel worlds with different configs.
pub struct VoxelWorldPlugin<C, M = StandardMaterial>
where
    C: VoxelWorldConfig,
    M: Material,
{
    spawn_meshes: bool,
    use_custom_material: bool,
    config: C,
    material: M,
}

impl<C> VoxelWorldPlugin<C, StandardMaterial>
where
    C: VoxelWorldConfig,
{
    pub fn with_config(config: C) -> Self {
        Self {
            config,
            spawn_meshes: true,
            use_custom_material: false,
            material: StandardMaterial::default(),
        }
    }

    pub fn minimal() -> Self {
        Self {
            spawn_meshes: false,
            use_custom_material: false,
            config: C::default(),
            material: StandardMaterial::default(),
        }
    }
}

impl<C, M> VoxelWorldPlugin<C, M>
where
    C: VoxelWorldConfig,
    M: Material,
{
    /// Use this to tell `bevy_voxel_world` to use a custom material. This can be any material that
    /// implements `bevy::pbr::Material`. You can use this to create custom shaders for your voxel
    /// world. You can set this up like any other material in Bevy.
    ///
    /// `bevy_voxel_world` will add the material as an asset, so you can query for it later using
    /// `Res<Assets<MyCustomVoxelMaterialType>>`.
    pub fn with_material<CustomMaterial: Material>(
        self,
        material: CustomMaterial,
    ) -> VoxelWorldPlugin<C, CustomMaterial> {
        VoxelWorldPlugin {
            spawn_meshes: self.spawn_meshes,
            use_custom_material: true,
            config: self.config,
            material,
        }
    }
}

impl Default for VoxelWorldPlugin<DefaultWorld, StandardMaterial> {
    fn default() -> Self {
        Self {
            spawn_meshes: true,
            use_custom_material: false,
            config: DefaultWorld::default(),
            material: StandardMaterial::default(),
        }
    }
}

impl<C, M> Plugin for VoxelWorldPlugin<C, M>
where
    C: VoxelWorldConfig,
    M: Material,
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

            app.add_systems(
                Update,
                Internals::<C>::assign_material::<
                    ExtendedMaterial<StandardMaterial, StandardVoxelMaterial>,
                >,
            );
        }

        if self.use_custom_material {
            let mut custom_material_assets = app.world.resource_mut::<Assets<M>>();

            let handle = custom_material_assets.add(self.material.clone());
            app.insert_resource(VoxelWorldMaterialHandle { handle });

            app.insert_resource(LoadingTexture {
                is_loaded: true,
                handle: Handle::default(),
            });

            app.add_systems(Update, Internals::<C>::assign_material::<M>);
        }
    }
}
