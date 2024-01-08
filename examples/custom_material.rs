use bevy::{
    pbr::{MaterialPipeline, MaterialPipelineKey},
    prelude::*,
    render::{
        mesh::MeshVertexBufferLayout,
        render_resource::{
            AsBindGroup, RenderPipelineDescriptor, ShaderRef, SpecializedMeshPipelineError,
        },
    },
};
use bevy_flycam::prelude::*;
use bevy_voxel_world::{
    prelude::*,
    rendering::{
        vertex_layout, VoxelWorldMaterialHandle, VoxelWorldMaterialPlugin,
        VOXEL_TEXTURE_SHADER_HANDLE,
    },
};
use std::sync::Arc;

// Declare materials as consts for convenience
const RED: u8 = 0;
const GREEN: u8 = 1;
const BLUE: u8 = 2;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        //
        // We can tell `bevy_voxel_world` to skip setting up the default material, so that we can use our own
        .add_plugins(VoxelWorldPlugin::default().without_default_material())
        //
        // We also need to tell `bevy_voxel_world` which material to assign.
        // This can be any Bevy material, including ExtendedMaterial.
        .add_plugins(VoxelWorldMaterialPlugin::<CustomVoxelMaterial>::default())
        //
        // Don't forget to Register the material with Bevy too.
        .add_plugins(MaterialPlugin::<CustomVoxelMaterial>::default())
        //
        .add_plugins(NoCameraPlayerPlugin)
        .add_systems(Startup, (setup, create_voxel_scene))
        .run();
}

fn setup(mut commands: Commands, mut materials: ResMut<Assets<CustomVoxelMaterial>>) {
    // Register our custom material
    let handle = materials.add(CustomVoxelMaterial { _unused: 0 });

    // This resource is used to find the correct material handle
    commands.insert_resource(VoxelWorldMaterialHandle { handle });

    commands.insert_resource(VoxelWorldConfiguration {
        // The arrays produces here can be read in the shader
        texture_index_mapper: Arc::new(|vox_mat: u8| match vox_mat {
            RED => [1, 1, 1],
            GREEN => [2, 2, 2],
            BLUE | _ => [3, 3, 3],
        }),
        ..Default::default()
    });

    // Camera
    commands.spawn((
        Camera3dBundle {
            transform: Transform::from_xyz(5.0, 5.0, 5.0).looking_at(Vec3::ZERO, Vec3::Y),
            ..default()
        },
        VoxelWorldCamera,
        FlyCam,
    ));
}

fn create_voxel_scene(mut voxel_world: VoxelWorld) {
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
        layout: &MeshVertexBufferLayout,
        _key: MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        // Use `vertex_layout()` from `bevy_voxel_world` to get the correct vertex layout
        let vertex_layout = layout.get_layout(&vertex_layout())?;
        descriptor.vertex.buffers = vec![vertex_layout];
        Ok(())
    }
}
