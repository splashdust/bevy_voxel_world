//! Meshes chunks with marching cubes instead of cubes, through a custom
//! meshing delegate. The marching cubes implementation comes from the
//! `transvoxel` crate; this example only turns a chunk's voxels into the
//! density field it samples, and its triangles into a Bevy mesh.
//!
//! Controls:
//!   - Drag with the right mouse button to orbit, scroll to zoom
//!   - Left click to dig a hole where the cursor points
//!   - Shift + left click to add a mound
//!
//! With the `light_propagation` feature the world is lit, and its plain
//! `StandardMaterial` gets the crate's lit PBR shader, which reads light on a
//! mesh that doesn't lie on the voxel grid:
//! `cargo run --example marching_cubes --features light_propagation`.

use std::sync::Arc;

use bevy::{
    light::CascadeShadowConfigBuilder, mesh::Indices, platform::collections::HashMap,
    prelude::*, render::render_resource::PrimitiveTopology,
};
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
use bevy_voxel_world::{
    custom_meshing::{PaddedChunkShape, VoxelArray, CHUNK_SIZE_F, CHUNK_SIZE_U},
    prelude::*,
};
use ndshape::ConstShape;
use noise::{HybridMulti, NoiseFn, Perlin};
use transvoxel::{
    prelude::*,
    structs::{grid_point::GridPoint, vertex_index::VertexIndex},
    traits::mesh_builder::MeshBuilder,
};

const GRASS: u8 = 0;
const DIRT: u8 = 1;
const STONE: u8 = 2;
const SAND: u8 = 3;

/// The number of voxels along each axis of a chunk's voxel array, padding
/// included: one voxel from each neighbouring chunk on every side.
const PADDED_SIZE: u32 = CHUNK_SIZE_U + 2;

#[derive(Resource, Clone, Default)]
struct MainWorld;

impl VoxelWorldConfig for MainWorld {
    type MaterialIndex = u8;
    type ChunkUserBundle = ();

    fn spawning_distance(&self) -> u32 {
        25
    }

    fn voxel_lookup_delegate(&self) -> VoxelLookupDelegate<Self::MaterialIndex> {
        Box::new(move |_chunk_pos, _lod, _previous| get_voxel_fn())
    }

    // The texture index mapper is only used by the default material, which
    // this example replaces with a plain `StandardMaterial` and vertex colours.
    fn texture_index_mapper(
        &self,
    ) -> Arc<dyn Fn(Self::MaterialIndex) -> [u32; 3] + Send + Sync> {
        Arc::new(|_| [0, 0, 0])
    }

    // The meshing delegate is called on a thread in the task pool with the
    // chunk's voxels, and returns the chunk's mesh. See `custom_meshing.rs` for
    // the full shape of the delegate; this example doesn't use LOD, so the
    // shapes it receives are always the default padded chunk shape.
    // Opting in is all a world with plain `StandardMaterial` needs to show
    // light.
    #[cfg(feature = "light_propagation")]
    fn light_propagation(&self) -> Option<LightPropagation> {
        Some(LightPropagation::default())
    }

    fn chunk_meshing_delegate(
        &self,
    ) -> ChunkMeshingDelegate<Self::MaterialIndex, Self::ChunkUserBundle> {
        Some(Box::new(
            |_pos, _lod, _data_shape, _mesh_shape, _previous| {
                Box::new(
                    |voxels: VoxelArray<u8>,
                     _data_shape,
                     _mesh_shape,
                     _texture_index_mapper| {
                        (marching_cubes_mesh(&voxels), None)
                    },
                )
            },
        ))
    }
}

/// Runs marching cubes over the voxels of one chunk.
///
/// Marching cubes wants a scalar density sampled on a grid, and places the
/// surface where the density crosses a threshold. The voxels are solid or
/// not, so the density at each corner of the voxel grid is the fraction of
/// the eight voxels around that corner that are solid, with the surface
/// halfway. Averaging over the voxels rounds the surface off, and since the
/// voxel array is padded with one voxel from every neighbouring chunk, the
/// corners on a chunk's border get the same density in both chunks, so the
/// surface meets without cracks.
fn marching_cubes_mesh(voxels: &[WorldVoxel<u8>]) -> Mesh {
    // The chunk's voxels are in local coordinates where voxel `i` of the
    // padded array spans `i..i + 1`, so the chunk itself spans
    // `1..CHUNK_SIZE + 1`: a block of CHUNK_SIZE cells, with CHUNK_SIZE + 1
    // sample points along each axis, starting at 1.
    let block = Block::new([1.0; 3], CHUNK_SIZE_F, CHUNK_SIZE_U as usize);

    // A closure sampling the density field at a point. The block samples it at
    // whole coordinates, and also one step outside the block to estimate
    // gradients, which is clamped to the voxels we have.
    let field = |x: f32, y: f32, z: f32| sample_corner(voxels, [x, y, z]);

    let builder = extract_from_field(
        &field,
        // Sample each grid point once, instead of every time a cell touches it.
        FieldCaching::CacheCentralBlockOnly,
        block,
        // Transition cells are for stitching blocks of different resolution.
        // Without them this is plain marching cubes.
        TransitionSide::none(),
        // The density above which a point is inside the surface.
        0.5,
        BevyMeshBuilder::default(),
    );

    builder.build()
}

/// What the density field knows about one corner of the voxel grid. Any
/// type with a density works as transvoxel's voxel data, and the rest is
/// handed back to the mesh builder.
#[derive(Clone, Copy, Default)]
struct Corner {
    /// The fraction of the eight voxels around the corner that are solid.
    density: f32,
    /// The gradient of the density at the corner, towards more solid voxels.
    /// Computed from the same eight voxels as the density so it is the same in
    /// neighbouring chunks, unlike transvoxel's own estimate, which needs
    /// voxels past the padding on a chunk's border.
    gradient: Vec3,
    /// The most common material among the solid voxels around the corner.
    material: u8,
}

impl VoxelData for Corner {
    type Density = f32;

    fn density(&self) -> f32 {
        self.density
    }
}

fn sample_corner(voxels: &[WorldVoxel<u8>], position: [f32; 3]) -> Corner {
    let corner = IVec3::from(position.map(|c| c.round() as i32));

    let mut solid_count = 0.0;
    let mut gradient = Vec3::ZERO;
    let mut material_counts = [0u8; 4];
    // The eight voxels around the corner: one step towards the negative side
    // and the voxel at the corner itself, along each axis.
    for offset in [IVec3::ZERO, IVec3::X, IVec3::Y, IVec3::Z, IVec3::ONE]
        .into_iter()
        .chain([
            IVec3::new(1, 1, 0),
            IVec3::new(1, 0, 1),
            IVec3::new(0, 1, 1),
        ])
    {
        let voxel = (corner - IVec3::ONE + offset)
            .clamp(IVec3::ZERO, IVec3::splat(PADDED_SIZE as i32 - 1));
        let index = PaddedChunkShape::linearize(voxel.as_uvec3().to_array()) as usize;
        if let WorldVoxel::Solid(material) = voxels[index] {
            solid_count += 1.0;
            // A solid voxel on the positive side of the corner pulls the
            // gradient that way.
            gradient += offset.as_vec3() * 2.0 - Vec3::ONE;
            material_counts[material as usize % material_counts.len()] += 1;
        }
    }

    let material = (0..material_counts.len())
        .max_by_key(|&material| material_counts[material])
        .unwrap_or(0) as u8;

    Corner {
        density: solid_count / 8.0,
        gradient,
        material,
    }
}

/// Collects the vertices and triangles transvoxel produces into Bevy mesh
/// attributes.
#[derive(Default)]
struct BevyMeshBuilder {
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    colors: Vec<[f32; 4]>,
    indices: Vec<u32>,
}

impl BevyMeshBuilder {
    fn build(self) -> Mesh {
        Mesh::new(
            PrimitiveTopology::TriangleList,
            bevy::asset::RenderAssetUsages::RENDER_WORLD,
        )
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, self.positions)
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, self.normals)
        .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, self.colors)
        .with_inserted_indices(Indices::U32(self.indices))
    }
}

impl MeshBuilder<Corner, f32> for BevyMeshBuilder {
    /// Called for each vertex, which sits on the edge between two grid points
    /// where the density crosses the threshold.
    fn add_vertex_between(
        &mut self,
        a: GridPoint<Corner, f32>,
        b: GridPoint<Corner, f32>,
        toward_b: f32,
    ) -> VertexIndex {
        let position = a.position.interpolate_toward(&b.position, toward_b);
        // The normal points out of the surface, away from the solid voxels.
        let normal = -a.voxel_data.gradient.lerp(b.voxel_data.gradient, toward_b);
        // Colour by the material of whichever end is inside the surface.
        let inside = if a.voxel_data.density > b.voxel_data.density {
            a
        } else {
            b
        };

        self.positions.push([position.x, position.y, position.z]);
        self.normals.push(normal.normalize_or(Vec3::Y).to_array());
        self.colors.push(material_color(inside.voxel_data.material));
        VertexIndex(self.positions.len() - 1)
    }

    fn add_triangle(&mut self, a: VertexIndex, b: VertexIndex, c: VertexIndex) {
        self.indices.extend([a.0 as u32, b.0 as u32, c.0 as u32]);
    }
}

fn material_color(material: u8) -> [f32; 4] {
    match material {
        GRASS => [0.15, 0.7, 0.2, 1.0],
        DIRT => [0.55, 0.2, 0.18, 1.0],
        STONE => [0.5, 0.5, 0.52, 1.0],
        _ => [0.85, 0.78, 0.55, 1.0],
    }
}

fn main() {
    App::new()
        .add_plugins((DefaultPlugins, PanOrbitCameraPlugin))
        // Marching cubes doesn't produce the per-face texture coordinates the
        // default voxel material expects, so render the chunks with a standard
        // material instead. Bevy tints it with the mesh's vertex colours.
        .add_plugins(VoxelWorldPlugin::with_config(MainWorld).with_material(
            StandardMaterial {
                base_color: Color::WHITE,
                perceptual_roughness: 0.9,
                ..default()
            },
        ))
        .add_systems(Startup, setup)
        .add_systems(Update, (edit_terrain, move_camera))
        .run();
}

fn setup(mut commands: Commands) {
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(-200.0, 300.0, -200.0)
            .looking_at(Vec3::ZERO + Vec3::Y * 150.0, Vec3::Y),
        PanOrbitCamera {
            button_orbit: MouseButton::Right,
            button_pan: MouseButton::Middle,
            target_radius: 90.0,
            ..default()
        },
        // This tells bevy_voxel_world to use this cameras transform to calculate spawning area
        VoxelWorldCamera::<MainWorld>::default(),
    ));

    // Sun
    let cascade_shadow_config = CascadeShadowConfigBuilder {
        maximum_distance: 700.0,
        ..default()
    }
    .build();
    commands.spawn((
        DirectionalLight {
            color: Color::srgb(0.98, 0.95, 0.82),
            shadow_maps_enabled: true,
            ..default()
        },
        Transform::from_xyz(0.0, 0.0, 0.0)
            .looking_at(Vec3::new(-0.15, -0.1, 0.15), Vec3::Y),
        cascade_shadow_config,
    ));

    // Ambient light, same color as sun. Propagated light dims it in shade, so a
    // lit world wants more of it.
    commands.insert_resource(GlobalAmbientLight {
        color: Color::srgb(0.98, 0.95, 0.82),
        brightness: if cfg!(feature = "light_propagation") {
            600.0
        } else {
            100.0
        },
        affects_lightmapped_meshes: true,
    });
}

/// Left click digs a hole and shift + left click adds a mound where the
/// cursor points. Edits go through the usual voxel API; the custom mesher
/// picks them up like any other change.
fn edit_terrain(
    buttons: Res<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
    window: Query<&Window>,
    camera: Query<(&Camera, &GlobalTransform), With<VoxelWorldCamera<MainWorld>>>,
    mut voxel_world: VoxelWorld<MainWorld>,
) {
    if !buttons.just_pressed(MouseButton::Left) {
        return;
    }
    let (Ok(window), Ok((camera, camera_transform))) = (window.single(), camera.single())
    else {
        return;
    };
    let Some(cursor) = window.cursor_position() else {
        return;
    };
    let Ok(ray) = camera.viewport_to_world(camera_transform, cursor) else {
        return;
    };
    let Some(hit) = voxel_world.raycast(ray, &|_| true) else {
        return;
    };

    let add = keys.any_pressed([KeyCode::ShiftLeft, KeyCode::ShiftRight]);
    let center = hit.voxel_pos();
    let radius = 16;
    for x in -radius..=radius {
        for y in -radius..=radius {
            for z in -radius..=radius {
                let offset = IVec3::new(x, y, z);
                if offset.length_squared() <= radius * radius {
                    let voxel = if add {
                        WorldVoxel::Solid(DIRT)
                    } else {
                        WorldVoxel::Air
                    };
                    voxel_world.set_voxel(center + offset, voxel);
                }
            }
        }
    }
}

/// Hilly terrain with some overhangs, which marching cubes renders as smooth
/// slopes and arches.
fn get_voxel_fn() -> Box<dyn FnMut(IVec3, Option<WorldVoxel>) -> WorldVoxel + Send + Sync>
{
    let mut height_noise = HybridMulti::<Perlin>::new(1234);
    height_noise.octaves = 4;
    height_noise.frequency = 1.1;
    height_noise.lacunarity = 2.8;
    height_noise.persistence = 0.4;
    let cave_noise = Perlin::new(42);

    // The height only depends on x and z, so cache it per column.
    let mut height_cache = HashMap::<(i32, i32), f64>::new();

    Box::new(move |pos: IVec3, _previous| {
        if pos.y < -8 {
            return WorldVoxel::Solid(STONE);
        }

        let [x, y, z] = pos.as_dvec3().to_array();
        let height = *height_cache
            .entry((pos.x, pos.z))
            .or_insert_with(|| height_noise.get([x / 400.0, z / 400.0]) * 24.0);

        // Carve the ground with 3D noise so there are overhangs and caves.
        let carve = cave_noise.get([x / 18.0, y / 18.0, z / 18.0]) * 6.0;
        let depth = height - y - carve;

        if depth < 0.0 {
            WorldVoxel::Air
        } else if y < 1.0 {
            WorldVoxel::Solid(SAND)
        } else if depth < 1.0 {
            WorldVoxel::Solid(GRASS)
        } else if depth < 4.0 {
            WorldVoxel::Solid(STONE)
        } else {
            WorldVoxel::Solid(DIRT)
        }
    })
}

fn move_camera(
    time: Res<Time>,
    mut cam_transform: Query<&mut Transform, With<VoxelWorldCamera<MainWorld>>>,
) {
    if let Ok(mut transform) = cam_transform.single_mut() {
        transform.translation.x += time.delta_secs() * 12.0;
        transform.translation.z += time.delta_secs() * 24.0;
    }
}
