use bevy::{asset::load_internal_asset, prelude::*};

use crate::{
    configuration::VoxelWorldConfiguration,
    voxel_material::{
        TextureLayers, VoxelTextureMaterial, VoxelTextureMaterialHandle,
        VOXEL_TEXTURE_SHADER_HANDLE,
    },
    voxel_world::*,
};

pub struct VoxelWorldPlugin {
    spawn_meshes: bool,
    voxel_texture: String,
    texture_layers: u32,
}

impl VoxelWorldPlugin {
    pub fn minimal() -> Self {
        Self {
            spawn_meshes: false,
            voxel_texture: "".to_string(),
            texture_layers: 0,
        }
    }

    pub fn with_voxel_texture(mut self, texture: &str, layers: u32) -> Self {
        self.voxel_texture = texture.to_string();
        self.texture_layers = layers;
        self
    }
}

impl Default for VoxelWorldPlugin {
    fn default() -> Self {
        Self {
            spawn_meshes: true,
            voxel_texture: "".to_string(),
            texture_layers: 0,
        }
    }
}

impl Plugin for VoxelWorldPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<VoxelWorldConfiguration>()
            .init_resource::<ChunkMap>()
            .init_resource::<ModifiedVoxels>()
            .add_systems(First, (spawn_chunks_in_view, retire_chunks_out_of_view))
            .add_systems(PostUpdate, remesh_dirty_chunks)
            .add_systems(Last, despawn_retired_chunks)
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

            app.add_plugins(MaterialPlugin::<VoxelTextureMaterial>::default());

            let mut preloaded_texture = true;

            // Use built-in default texture if no texture is specified.
            let image_handle = if self.voxel_texture.is_empty() {
                let mut image = Image::from_buffer(
                    include_bytes!("shaders/default_texture.png"),
                    bevy::render::texture::ImageType::MimeType("image/png"),
                    bevy::render::texture::CompressedImageFormats::default(),
                    false,
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

            let mut material_assets = app.world.resource_mut::<Assets<VoxelTextureMaterial>>();

            let mat_handle = material_assets.add(VoxelTextureMaterial {
                voxels_texture: image_handle.clone(),
            });

            app.insert_resource(LoadingTexture {
                is_loaded: preloaded_texture,
                handle: image_handle,
            });
            app.insert_resource(VoxelTextureMaterialHandle(mat_handle));
            app.insert_resource(TextureLayers(self.texture_layers));

            app.add_systems(Update, (prepare_texture, spawn_chunk_meshes));
        }
    }
}

fn prepare_texture(
    asset_server: Res<AssetServer>,
    texture_layers: Res<TextureLayers>,
    mut loading_texture: ResMut<LoadingTexture>,
    mut images: ResMut<Assets<Image>>,
) {
    if loading_texture.is_loaded
        || asset_server.get_load_state(loading_texture.handle.clone())
            != bevy::asset::LoadState::Loaded
    {
        return;
    }
    loading_texture.is_loaded = true;

    let image = images.get_mut(&loading_texture.handle).unwrap();
    image.reinterpret_stacked_2d_as_array(texture_layers.0);
}

fn spawn_chunks_in_view(mut voxel_world: VoxelWorldInternal) {
    voxel_world.spawn_chunks();
}

fn remesh_dirty_chunks(mut voxel_world: VoxelWorldInternal) {
    voxel_world.remesh_dirty_chunks();
}

fn spawn_chunk_meshes(mut mesh_spawner: VoxelWorldMeshSpawner) {
    mesh_spawner.spawn_meshes();
}

fn retire_chunks_out_of_view(mut voxel_world: VoxelWorldInternal) {
    voxel_world.retire_chunks();
}

fn despawn_retired_chunks(mut voxel_world: VoxelWorldInternal) {
    voxel_world.despawn_retired_chunks();
}

// -------- TESTS --------
#[cfg(test)]
mod tests {
    use bevy::prelude::*;

    use super::VoxelWorldPlugin;
    use crate::{prelude::VoxelWorldCamera, voxel_world::*};

    fn _test_setup_app() -> App {
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, VoxelWorldPlugin::minimal()));

        app
    }

    #[test]
    fn can_set_get_voxels() {
        let mut app = _test_setup_app();

        // Set and get some voxels
        app.add_systems(Startup, |mut voxel_world: crate::prelude::VoxelWorld| {
            let positions = vec![
                IVec3::new(0, 100, 0),
                IVec3::new(0, 0, 0),
                IVec3::new(1, 0, 1),
                IVec3::new(1, 1, 1),
                IVec3::new(100, 200, 300),
                IVec3::new(-1, 0, -1),
                IVec3::new(0, -1, 0),
                IVec3::new(-100, -200, -300),
            ];

            let test_voxel = crate::voxel::WorldVoxel::Solid(1);

            for pos in positions {
                voxel_world.set_voxel(pos, test_voxel);
                assert_eq!(voxel_world.get_voxel(pos), test_voxel)
            }
        });

        app.update();
    }

    #[test]
    fn set_voxel_can_be_found_by_2d_coordinate() {
        let mut app = _test_setup_app();

        // Set up vector och positions to test
        let positions = vec![
            IVec3::new(0, 5, 0),
            IVec3::new(0, 7, 0),
            IVec3::new(0, 10, 0),
            IVec3::new(10, 5, 10),
            IVec3::new(10, 7, 10),
            IVec3::new(10, 10, 10),
            IVec3::new(-10, 5, -10),
            IVec3::new(-10, 7, -10),
            IVec3::new(-10, 10, -10),
        ];

        let make_pos = positions.clone();

        app.add_systems(
            Startup,
            move |mut voxel_world: crate::prelude::VoxelWorld| {
                let test_voxel = crate::voxel::WorldVoxel::Solid(1);

                for pos in make_pos.clone() {
                    voxel_world.set_voxel(pos, test_voxel);
                }
            },
        );

        app.update();

        let check_pos = positions.clone();

        app.add_systems(Startup, move |voxel_world: crate::prelude::VoxelWorld| {
            let test_voxel = crate::voxel::WorldVoxel::Solid(1);

            for pos in check_pos.clone() {
                assert_eq!(
                    voxel_world.get_surface_voxel_at_2d_pos(Vec2::new(pos.x as f32, pos.z as f32)),
                    Some((pos, test_voxel))
                )
            }
        });

        app.update();
    }

    #[test]
    fn chunk_will_spawn_events() {
        let mut app = _test_setup_app();

        app.add_systems(
            Update,
            |mut ev_chunk_will_spawn: EventReader<ChunkWillSpawn>| {
                let spawn_count = ev_chunk_will_spawn.iter().count();
                assert!(spawn_count > 0);
            },
        );

        app.update();
    }

    #[test]
    fn chunk_will_remesh_event_after_set_voxel() {
        let mut app = _test_setup_app();

        // Run the world a 100 cycles
        for _ in 0..100 {
            app.update();
        }

        app.add_systems(Update, |mut voxel_world: crate::prelude::VoxelWorld| {
            voxel_world.set_voxel(IVec3::new(0, 0, 0), crate::voxel::WorldVoxel::Solid(1));
        });

        app.update();

        app.add_systems(
            Update,
            |mut ev_chunk_will_remesh: EventReader<ChunkWillRemesh>| {
                let count = ev_chunk_will_remesh.iter().count();
                assert_eq!(count, 1)
            },
        );

        app.update();
    }

    #[test]
    fn chunk_will_despawn_event() {
        let mut app = _test_setup_app();

        // Setup dummy camera
        app.add_systems(Startup, |mut commands: Commands| {
            commands
                .spawn(Transform::default())
                .insert(VoxelWorldCamera);
        });

        app.update();

        // move camera to simulate chunks going out of view
        app.add_systems(
            Update,
            |mut query: Query<&mut Transform, With<VoxelWorldCamera>>| {
                for mut transform in query.iter_mut() {
                    transform.translation.x += 100.0;
                }
            },
        );

        app.update();

        app.add_systems(
            Update,
            |mut ev_chunk_will_despawn: EventReader<ChunkWillDespawn>| {
                let count = ev_chunk_will_despawn.iter().count();
                assert!(count > 0)
            },
        );

        app.update();
    }
}
