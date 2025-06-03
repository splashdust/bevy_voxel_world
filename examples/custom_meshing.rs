use std::sync::Arc;

use bevy::{
    color::palettes::css::*,
    pbr::{
        wireframe::{WireframeConfig, WireframePlugin},
        CascadeShadowConfigBuilder,
    },
    platform::collections::HashMap,
    prelude::*,
    render::{
        mesh::{Indices, VertexAttributeValues},
        render_asset::RenderAssetUsages,
        render_resource::{PrimitiveTopology, WgpuFeatures},
        settings::{RenderCreation, WgpuSettings},
        RenderPlugin,
    },
};

use bevy_voxel_world::{
    custom_meshing::{PaddedChunkShape, VoxelArray, CHUNK_SIZE_U},
    prelude::*,
    rendering::ATTRIBUTE_TEX_INDEX,
};

use block_mesh::RIGHT_HANDED_Y_UP_CONFIG;
use ndshape::ConstShape;
use noise::{HybridMulti, NoiseFn, Perlin};

#[derive(Resource, Clone, Default)]
struct MainWorld;

impl VoxelWorldConfig for MainWorld {
    type MaterialIndex = u8;

    // If you want to add a custom component bundle to the spawned chunk entity from the meshing
    // function, you can define its type here. Otherwise, set it to `()`.
    type ChunkUserBundle = ();

    fn max_spawning_distance(&self) -> u32 {
        25
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

    // A custom meshing delegate can be added via the config implementation
    //
    // In this example we use the greedy meshing algorithm from the block_mesh crate
    // instead of the default simple meshing.
    //
    // The closure returned here is executed on a thread in the task pool, so it's OK to block
    // for as long as needed.
    fn chunk_meshing_delegate(
        &self,
    ) -> ChunkMeshingDelegate<Self::MaterialIndex, Self::ChunkUserBundle> {
        Some(Box::new(|pos: IVec3| {
            // If necessary, we can caputure data here based on the chunk position
            // and move it into the closure below.
            Box::new(
                // The array of voxels for the chunk
                |voxels: VoxelArray<Self::MaterialIndex>,
                 // A reference to the texture index mapper function as defined in the config
                 texture_index_mapper: TextureIndexMapperFn<Self::MaterialIndex>| {
                    let faces = block_mesh::RIGHT_HANDED_Y_UP_CONFIG.faces;
                    let mut buffer = block_mesh::GreedyQuadsBuffer::new(voxels.len());

                    // Call the greedy meshing algorithm from the block_mesh crate
                    block_mesh::greedy_quads(
                        &*voxels,
                        &PaddedChunkShape {},
                        [0; 3],
                        [CHUNK_SIZE_U + 1; 3],
                        &faces,
                        &mut buffer,
                    );

                    let num_indices = buffer.quads.num_quads() * 6;
                    let num_vertices = buffer.quads.num_quads() * 4;
                    let mut indices = Vec::with_capacity(num_indices);
                    let mut positions = Vec::with_capacity(num_vertices);
                    let mut normals = Vec::with_capacity(num_vertices);
                    let mut tex_coords = Vec::with_capacity(num_vertices);
                    let mut material_types = Vec::with_capacity(num_vertices);

                    for (group, face) in buffer.quads.groups.into_iter().zip(faces.into_iter()) {
                        for quad in group.into_iter() {
                            let normal = IVec3::from([
                                face.signed_normal().x,
                                face.signed_normal().y,
                                face.signed_normal().z,
                            ]);

                            indices
                                .extend_from_slice(&face.quad_mesh_indices(positions.len() as u32));
                            positions.extend_from_slice(&face.quad_mesh_positions(&quad, 1.0));
                            normals.extend_from_slice(&face.quad_mesh_normals());

                            tex_coords.extend_from_slice(&face.tex_coords(
                                RIGHT_HANDED_Y_UP_CONFIG.u_flip_face,
                                true,
                                &quad.into(),
                            ));

                            let voxel_index = PaddedChunkShape::linearize(quad.minimum) as usize;
                            let material_type = match voxels[voxel_index] {
                                // Here we call the texture index mapper function to get the texture index
                                // for the material type of the voxel
                                WorldVoxel::Solid(mt) => texture_index_mapper(mt),
                                _ => [1, 1, 1],
                            };
                            material_types.extend(std::iter::repeat(material_type).take(4));
                        }
                    }

                    let mut render_mesh = Mesh::new(
                        PrimitiveTopology::TriangleList,
                        RenderAssetUsages::RENDER_WORLD,
                    );
                    render_mesh.insert_attribute(
                        Mesh::ATTRIBUTE_POSITION,
                        VertexAttributeValues::Float32x3(positions),
                    );
                    render_mesh.insert_attribute(
                        Mesh::ATTRIBUTE_NORMAL,
                        VertexAttributeValues::Float32x3(normals),
                    );
                    render_mesh.insert_attribute(
                        Mesh::ATTRIBUTE_UV_0,
                        VertexAttributeValues::Float32x2(vec![[0.0; 2]; num_vertices]),
                    );
                    render_mesh.insert_attribute(
                        ATTRIBUTE_TEX_INDEX,
                        VertexAttributeValues::Uint32x3(material_types),
                    );
                    render_mesh
                        .insert_attribute(Mesh::ATTRIBUTE_COLOR, vec![[1.0; 4]; num_vertices]);
                    render_mesh.insert_indices(Indices::U32(indices.clone()));

                    // The second value in this tuple is an optional component bundle.
                    // If you want to generate some custom data for the chunk, like a nav mesh,
                    // you can put it here in a regular Bevy component. This will then get added
                    // to the spawned Chunk entity.
                    // The type of this bundle is defined in the `ChunkUserBundle` associated type.
                    (render_mesh, None)
                },
            )
        }))
    }
}

fn main() {
    App::new()
        .add_plugins((
            DefaultPlugins.set(RenderPlugin {
                render_creation: RenderCreation::Automatic(WgpuSettings {
                    features: WgpuFeatures::POLYGON_MODE_LINE,
                    ..default()
                }),
                ..default()
            }),
            WireframePlugin::default(),
        ))
        .add_plugins(VoxelWorldPlugin::with_config(MainWorld))
        .insert_resource(WireframeConfig {
            global: true,
            default_color: WHITE.into(),
        })
        .add_systems(Startup, setup)
        .add_systems(Update, move_camera)
        .run();
}

fn setup(mut commands: Commands) {
    // camera
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(-50.0, 50.0, -50.0).looking_at(Vec3::ZERO, Vec3::Y),
        // This tells bevy_voxel_world to use this cameras transform to calculate spawning area
        VoxelWorldCamera::<MainWorld>::default(),
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
}

fn get_voxel_fn() -> Box<dyn FnMut(IVec3) -> WorldVoxel + Send + Sync> {
    // Set up some noise to use as the terrain height map
    let mut noise = HybridMulti::<Perlin>::new(1234);
    noise.octaves = 5;
    noise.frequency = 1.1;
    noise.lacunarity = 2.8;
    noise.persistence = 0.4;

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

        // If y is less than the noise sample, we will set the voxel to solid
        let is_ground = y < match cache.get(&(pos.x, pos.z)) {
            Some(sample) => *sample,
            None => {
                let sample = noise.get([x / 4000.0, z / 4000.0]) * 20.0;
                cache.insert((pos.x, pos.z), sample);
                sample
            }
        };

        if is_ground {
            // Solid voxel of material type 0
            WorldVoxel::Solid(0)
        } else {
            WorldVoxel::Air
        }
    })
}

fn move_camera(
    time: Res<Time>,
    mut cam_transform: Query<&mut Transform, With<VoxelWorldCamera<MainWorld>>>,
) {
    let mut transform = cam_transform.single_mut().unwrap();
    transform.translation.x += time.delta_secs() * 5.0;
    transform.translation.z += time.delta_secs() * 10.0;
}
