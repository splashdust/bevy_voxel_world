use std::sync::Arc;

use bevy::{
    pbr::{CascadeShadowConfigBuilder, MaterialPipeline, MaterialPipelineKey},
    platform::collections::HashMap,
    prelude::*,
    render::{
        mesh::MeshVertexBufferLayoutRef,
        render_resource::{
            AsBindGroup, RenderPipelineDescriptor, ShaderDefVal, ShaderRef,
            SpecializedMeshPipelineError,
        },
    },
};
use bevy_voxel_world::{
    prelude::*,
    rendering::{vertex_layout, VOXEL_TEXTURE_SHADER_HANDLE},
};
use noise::{HybridMulti, NoiseFn, Perlin};

const RED: u8 = 0;
const GREEN: u8 = 1;
const BLUE: u8 = 2;

// This is the main world configuration. In this example, the main world is the procedural terrain.
#[derive(Resource, Clone, Default)]
struct MainWorld;

impl VoxelWorldConfig for MainWorld {
    type MaterialIndex = u8;
    type ChunkUserBundle = ();

    fn max_spawning_distance(&self) -> u32 {
        10
    }

    fn voxel_lookup_delegate(&self) -> VoxelLookupDelegate<Self::MaterialIndex> {
        Box::new(move |_chunk_pos| get_voxel_fn())
    }

    fn texture_index_mapper(
        &self,
    ) -> Arc<dyn Fn(Self::MaterialIndex) -> [u32; 3] + Send + Sync> {
        Arc::new(|mat| match mat {
            0 => [0, 0, 0],
            1 => [1, 1, 1],
            2 => [2, 2, 2],
            3 => [3, 3, 3],
            _ => [0, 0, 0],
        })
    }
}

// This is the second world configuration. In this example, the second world is using a custom material.
#[derive(Resource, Clone, Default)]
struct SecondWorld;

impl VoxelWorldConfig for SecondWorld {
    type MaterialIndex = u8;
    type ChunkUserBundle = ();

    fn texture_index_mapper(&self) -> Arc<dyn Fn(u8) -> [u32; 3] + Send + Sync> {
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
        .add_plugins(MaterialPlugin::<CustomVoxelMaterial>::default())
        .add_plugins(VoxelWorldPlugin::with_config(MainWorld)) // Add the main world
        .add_plugins(
            // Add the second world with a custom material
            VoxelWorldPlugin::with_config(SecondWorld)
                .with_material(CustomVoxelMaterial { _unused: 0 }),
        )
        .add_systems(Startup, setup)
        .run();
}

fn setup(mut commands: Commands, mut second_world: VoxelWorld<SecondWorld>) {
    // --- Just scene setup below ---

    // camera
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(-10.0, 10.0, -10.0).looking_at(Vec3::Y * 4.0, Vec3::Y),
        // This tells bevy_voxel_world to use this cameras transform to calculate spawning area
        VoxelWorldCamera::<MainWorld>::default(),
        VoxelWorldCamera::<SecondWorld>::default(),
    ));

    // Sun
    let cascade_shadow_config = CascadeShadowConfigBuilder { ..default() }.build();
    commands.spawn((
        DirectionalLight {
            color: Color::srgb(0.98, 0.95, 0.82),
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(0.0, 0.0, 0.0)
            .looking_at(Vec3::new(-0.15, -0.1, 0.15), Vec3::Y),
        cascade_shadow_config,
    ));

    // Ambient light, same color as sun
    commands.insert_resource(AmbientLight {
        color: Color::srgb(0.98, 0.95, 0.82),
        brightness: 100.0,
        affects_lightmapped_meshes: true,
    });

    // Set some voxels in the second world
    second_world.set_voxel(IVec3::new(0, 2, 0), WorldVoxel::Solid(RED));
    second_world.set_voxel(IVec3::new(1, 2, 0), WorldVoxel::Solid(RED));
    second_world.set_voxel(IVec3::new(0, 2, 1), WorldVoxel::Solid(RED));
    second_world.set_voxel(IVec3::new(0, 2, -1), WorldVoxel::Solid(RED));
    second_world.set_voxel(IVec3::new(-1, 2, 0), WorldVoxel::Solid(GREEN));
    second_world.set_voxel(IVec3::new(-2, 2, 0), WorldVoxel::Solid(GREEN));
    second_world.set_voxel(IVec3::new(-1, 3, 0), WorldVoxel::Solid(RED));
    second_world.set_voxel(IVec3::new(-2, 3, 0), WorldVoxel::Solid(RED));
    second_world.set_voxel(IVec3::new(0, 3, 0), WorldVoxel::Solid(RED));
}

fn get_voxel_fn() -> Box<dyn FnMut(IVec3) -> WorldVoxel + Send + Sync> {
    // Set up some noise to use as the terrain height map
    let mut noise = HybridMulti::<Perlin>::new(1234);
    noise.octaves = 4;
    noise.frequency = 1.0;
    noise.lacunarity = 2.2;
    noise.persistence = 0.5;

    // We use this to cache the noise value for each y column so we only need
    // to calculate it once per x/z coordinate
    let mut cache = HashMap::<(i32, i32), f64>::new();

    // Then we return this boxed closure that captures the noise and the cache
    // This will get sent off to a separate thread for meshing by bevy_voxel_world
    Box::new(move |pos: IVec3| {
        // Sea level
        if pos.y < 1 {
            return WorldVoxel::Solid(3);
        }

        let [x, y, z] = pos.as_dvec3().to_array();

        let sample = match cache.get(&(pos.x, pos.z)) {
            Some(sample) => *sample,
            None => {
                let sample = noise.get([x / 700.0, z / 700.0]) * 5.0;
                cache.insert((pos.x, pos.z), sample);
                sample
            }
        };

        // If y is less than the noise sample, we will set the voxel to solid
        let is_surface = y < sample;
        let is_sub_surface = y < sample - 1.0;

        if is_surface && !is_sub_surface {
            // Solid voxel of material type 0
            WorldVoxel::Solid(0)
        } else if is_sub_surface {
            // Solid voxel of material type 1
            WorldVoxel::Solid(1)
        } else {
            WorldVoxel::Air
        }
    })
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
        if descriptor
            .vertex
            .shader_defs
            .contains(&ShaderDefVal::Bool("PREPASS_PIPELINE".into(), true))
        {
            return Ok(());
        }

        // Use `vertex_layout()` from `bevy_voxel_world` to get the correct vertex layout
        let vertex_layout = layout.0.get_layout(&vertex_layout())?;
        descriptor.vertex.buffers = vec![vertex_layout];
        Ok(())
    }
}
