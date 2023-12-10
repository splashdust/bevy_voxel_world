use bevy::{
    asset::load_internal_asset,
    pbr::ExtendedMaterial,
    prelude::*,
    render::texture::{CompressedImageFormats, ImageSampler, ImageType},
};

use crate::{
    configuration::VoxelWorldConfiguration,
    voxel_material::{
        prepare_texture, LoadingTexture, StandardVoxelMaterial, StandardVoxelMaterialHandle,
        TextureLayers, VOXEL_TEXTURE_SHADER_HANDLE,
    },
    voxel_world::*,
    voxel_world_internal::{
        despawn_retired_chunks, flush_chunk_map_buffers, flush_mesh_cache_buffers,
        flush_voxel_write_buffer, remesh_dirty_chunks, retire_chunks, setup_internals,
        spawn_chunks, spawn_meshes,
    },
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
            .add_systems(PreStartup, setup_internals)
            .add_systems(
                Update,
                ((spawn_chunks, retire_chunks).chain(), remesh_dirty_chunks).chain(),
            )
            .add_systems(
                PostUpdate,
                (
                    flush_voxel_write_buffer,
                    despawn_retired_chunks,
                    (flush_chunk_map_buffers, flush_mesh_cache_buffers),
                )
                    .chain(),
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
            app.insert_resource(StandardVoxelMaterialHandle(mat_handle));
            app.insert_resource(TextureLayers(self.texture_layers));

            app.add_systems(Update, spawn_meshes);
            app.add_systems(Update, prepare_texture);
        }
    }
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
        app.add_systems(Startup, |mut commands: Commands| {
            commands.spawn((
                Camera3dBundle {
                    transform: Transform::from_xyz(10.0, 10.0, 10.0)
                        .looking_at(Vec3::ZERO, Vec3::Y),
                    ..default()
                },
                VoxelWorldCamera,
            ));
        });

        app
    }

    #[test]
    fn can_set_get_voxels() {
        let mut app = _test_setup_app();

        // Set and get some voxels
        app.add_systems(Update, |mut voxel_world: crate::prelude::VoxelWorld| {
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
            IVec3::new(1, 7, 0),
            IVec3::new(2, 10, 0),
            IVec3::new(3, 5, 10),
            IVec3::new(4, 7, 10),
            IVec3::new(5, 10, 10),
            IVec3::new(-6, 5, -10),
            IVec3::new(-7, 7, -10),
            IVec3::new(-10, 10, -10),
        ];

        let make_pos = positions.clone();

        app.add_systems(
            Update,
            move |mut voxel_world: crate::prelude::VoxelWorld| {
                let test_voxel = crate::voxel::WorldVoxel::Solid(1);

                for pos in make_pos.clone() {
                    voxel_world.set_voxel(pos, test_voxel);
                }
            },
        );

        app.update();

        let check_pos = positions.clone();

        app.add_systems(Update, move |voxel_world: crate::prelude::VoxelWorld| {
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

    // ChunkWillSpawn event now fires from the mesh spawning system, which cannot run in tests.
    #[ignore]
    #[test]
    fn chunk_will_spawn_events() {
        let mut app = _test_setup_app();

        app.add_systems(
            Update,
            |mut ev_chunk_will_spawn: EventReader<ChunkWillSpawn>| {
                let spawn_count = ev_chunk_will_spawn.read().count();
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
        app.update(); // Need two cycles for the write buffer to flush

        app.add_systems(
            Update,
            |mut ev_chunk_will_remesh: EventReader<ChunkWillRemesh>| {
                let count = ev_chunk_will_remesh.read().count();
                assert_eq!(count, 1)
            },
        );

        app.update();
    }

    #[test]
    fn chunk_will_despawn_event() {
        let mut app = _test_setup_app();

        // move camera to simulate chunks going out of view
        app.add_systems(
            Update,
            |mut query: Query<&mut GlobalTransform, With<VoxelWorldCamera>>| {
                for mut transform in query.iter_mut() {
                    // Not sure why, but when running tests, bevy won't update the GlobalTransform
                    // when a Transform has changed, so as a workaround we set it here directly.
                    let tf = Transform::from_xyz(1000.0, 1000.0, 1000.0);
                    *transform = GlobalTransform::from(tf);
                }
            },
        );

        app.update();

        app.add_systems(
            Update,
            |mut ev_chunk_will_despawn: EventReader<ChunkWillDespawn>| {
                let count = ev_chunk_will_despawn.read().count();
                assert!(count > 0)
            },
        );

        app.update();
    }
}
