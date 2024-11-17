use bevy::{
    pbr::{MaterialPipeline, MaterialPipelineKey},
    prelude::*,
    render::{
        mesh::MeshVertexBufferLayoutRef,
        render_resource::{
            AsBindGroup, RenderPipelineDescriptor, ShaderRef, SpecializedMeshPipelineError,
        },
    },
};
use bevy_voxel_world::{
    prelude::*,
    rendering::{vertex_layout, VOXEL_TEXTURE_SHADER_HANDLE},
};
use std::sync::Arc;

// Declare materials as consts for convenience
const RED: u8 = 0;
const GREEN: u8 = 1;
const BLUE: u8 = 2;

#[derive(Resource, Clone, Default)]
struct MyMainWorld;

impl VoxelWorldConfig for MyMainWorld {
    type MaterialIndex = u8;
    fn texture_index_mapper(&self) -> Arc<dyn Fn(Self::MaterialIndex) -> [u32; 3] + Send + Sync> {
        Arc::new(|vox_mat: u8| match vox_mat {
            RED => [1, 1, 1],
            GREEN => [2, 2, 2],
            BLUE | _ => [3, 3, 3],
        })
    }
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        //
        // First we need to register the material with Bevy. This needs to be done before we add the
        // `VoxelWorldPlugin` so that the plugin can find the material.
        .add_plugins(MaterialPlugin::<CustomVoxelMaterial>::default())
        //
        // Then we can tell `bevy_voxel_world` to use that material when adding the plugin.
        // bevy_voxel_world will add the material as an asset, so you can query for it later using
        // `Res<Assets<CustomVoxelMaterial>>`.
        .add_plugins(
            VoxelWorldPlugin::with_config(MyMainWorld)
                .with_material(CustomVoxelMaterial { _unused: 0 }),
        )
        //
        .add_systems(Startup, (setup, create_voxel_scene))
        .run();
}

fn setup(mut commands: Commands) {
    // Camera
    commands.spawn((
        Camera3dBundle {
            transform: Transform::from_xyz(5.0, 5.0, 5.0).looking_at(Vec3::ZERO, Vec3::Y),
            ..default()
        },
        VoxelWorldCamera::<MyMainWorld>::default(),
    ));
}

fn create_voxel_scene(mut voxel_world: VoxelWorld<MyMainWorld>) {
    // 20 by 20 floor
    for x in -10..10 {
        for z in -10..10 {
            voxel_world.set_voxel(IVec3::new(x, -1, z), WorldVoxel::Solid(BLUE));
        }
    }

    voxel_world.set_voxel(IVec3::new(0, 0, 0), WorldVoxel::Solid(RED));
    voxel_world.set_voxel(IVec3::new(1, 0, 0), WorldVoxel::Solid(RED));
    voxel_world.set_voxel(IVec3::new(0, 0, 1), WorldVoxel::Solid(RED));
    voxel_world.set_voxel(IVec3::new(0, 0, -1), WorldVoxel::Solid(RED));
    voxel_world.set_voxel(IVec3::new(-1, 0, 0), WorldVoxel::Solid(GREEN));
    voxel_world.set_voxel(IVec3::new(-2, 0, 0), WorldVoxel::Solid(GREEN));
    voxel_world.set_voxel(IVec3::new(-1, 1, 0), WorldVoxel::Solid(RED));
    voxel_world.set_voxel(IVec3::new(-2, 1, 0), WorldVoxel::Solid(RED));
    voxel_world.set_voxel(IVec3::new(0, 1, 0), WorldVoxel::Solid(RED));
}

// This is the custom material. You can set this up like any other material in Bevy.
#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
struct CustomVoxelMaterial {
    // We're not using any uniforms in this example
    _unused: u32,
}

impl Material for CustomVoxelMaterial {
    fn vertex_shader() -> ShaderRef {
        // You can use the default shader from bevy_voxel_world for the vertex shader for simplicity
        VOXEL_TEXTURE_SHADER_HANDLE.into()
    }

    fn fragment_shader() -> ShaderRef {
        "custom_material.wgsl".into()
    }

    fn specialize(
        _pipeline: &MaterialPipeline<Self>,
        descriptor: &mut RenderPipelineDescriptor,
        layout: &MeshVertexBufferLayoutRef,
        _key: MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        // Use `vertex_layout()` from `bevy_voxel_world` to get the correct vertex layout
        let vertex_layout = layout.0.get_layout(&vertex_layout())?;
        descriptor.vertex.buffers = vec![vertex_layout];
        Ok(())
    }
}
