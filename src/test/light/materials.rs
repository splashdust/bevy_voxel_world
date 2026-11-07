//! Tests of the material a lit world's chunks get when the world brings its
//! own: the crate wraps it per lit world, and any mesh, cubes or not, gets the
//! crate's tag.

use bevy::asset::RenderAssetUsages;
use bevy::ecs::system::RunSystemOnce;
use bevy::mesh::{Indices, MeshTag, PrimitiveTopology};
use bevy::prelude::*;
use bevy::render::render_resource::AsBindGroup;

use super::super::{chunk_materials, meshing_app, spawn_camera, update_until};
use crate::custom_meshing::VoxelArray;
use crate::light::route::LightRoute;
use crate::prelude::*;
use crate::rendering::{VoxelWorldMaterial, VoxelWorldMaterialHandle};

const STONE: u8 = 1;
const LAMP: u8 = 2;

/// A lamp on the floor of the chunk at the origin.
const LAMP_AT: IVec3 = IVec3::new(16, 1, 16);

/// The chunks beside the origin along x and z, which hold only floor that the
/// lamp's light doesn't reach.
const FLOOR_ONLY: [IVec3; 4] = [IVec3::X, IVec3::NEG_X, IVec3::Z, IVec3::NEG_Z];

const HAS_SLOT: u32 = 1 << 31;

/// A material of a world's own, with a colour the tests can tell apart.
#[derive(Asset, AsBindGroup, TypePath, Debug, Clone)]
struct TintMaterial {
    #[uniform(0)]
    tint: LinearRgba,
}

impl Material for TintMaterial {
    fn enable_prepass() -> bool {
        false
    }
}

/// A stone floor at y = 0 through every chunk, with a lamp on it in the chunk
/// at the origin. `N` tells worlds of the same shape apart.
#[derive(Resource, Clone)]
struct LampFloor<const N: u8 = 0> {
    light: Option<LightPropagation>,
    init_custom_materials: bool,
    /// Meshes each chunk as one sloped triangle instead of cubes.
    sloped_mesh: bool,
}

impl<const N: u8> Default for LampFloor<N> {
    fn default() -> Self {
        Self {
            light: Some(LightPropagation {
                sky_fill: None,
                ..default()
            }),
            init_custom_materials: true,
            sloped_mesh: false,
        }
    }
}

impl<const N: u8> VoxelWorldConfig for LampFloor<N> {
    type MaterialIndex = u8;
    type ChunkUserBundle = ();

    fn spawning_distance(&self) -> u32 {
        2
    }

    fn chunk_spawn_strategy(&self) -> ChunkSpawnStrategy {
        ChunkSpawnStrategy::Close
    }

    fn voxel_lookup_delegate(&self) -> VoxelLookupDelegate<Self::MaterialIndex> {
        Box::new(|_, _, _| {
            Box::new(|position, _| {
                if position == LAMP_AT {
                    WorldVoxel::Solid(LAMP)
                } else if position.y == 0 {
                    WorldVoxel::Solid(STONE)
                } else {
                    WorldVoxel::Air
                }
            })
        })
    }

    fn chunk_meshing_delegate(
        &self,
    ) -> ChunkMeshingDelegate<Self::MaterialIndex, Self::ChunkUserBundle> {
        if !self.sloped_mesh {
            return None;
        }
        Some(Box::new(|_, _, _, _, _| {
            Box::new(|_voxels: VoxelArray<u8>, _, _, _| (sloped_triangle(), None))
        }))
    }

    fn light_propagation(&self) -> Option<LightPropagation> {
        self.light.clone()
    }

    fn emissive_materials(&self) -> Vec<(Self::MaterialIndex, Emission)> {
        vec![(LAMP, Emission::level(15.0))]
    }

    fn init_custom_materials(&self) -> bool {
        self.init_custom_materials
    }
}

/// One triangle sloping across a chunk's floor, as a marching cubes mesher
/// draws ground: no face of it lies on the voxel grid.
fn sloped_triangle() -> Mesh {
    let normal = Vec3::new(0.3, 1.0, 0.2).normalize().to_array();
    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    )
    .with_inserted_attribute(
        Mesh::ATTRIBUTE_POSITION,
        vec![[1.0, 2.3, 1.0], [33.0, 1.6, 1.0], [1.0, 1.9, 33.0]],
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, vec![normal; 3])
    .with_inserted_indices(Indices::U32(vec![0, 1, 2]))
}

/// An app with the material plugin of `TintMaterial` and `StandardMaterial`'s
/// assets, as a world bringing its own material has.
fn custom_material_app() -> App {
    let mut app = meshing_app();
    app.init_asset::<TintMaterial>()
        .init_asset::<StandardMaterial>();
    app
}

/// The entities of world `C`'s chunks showing a mesh.
fn shown_chunks<C: VoxelWorldConfig>(app: &mut App) -> Vec<Entity> {
    app.world_mut()
        .query_filtered::<Entity, (With<Chunk<C>>, With<Mesh3d>)>()
        .iter(app.world())
        .collect()
}

fn chunk_entity<C: VoxelWorldConfig>(app: &mut App, position: IVec3) -> Option<Entity> {
    app.world_mut()
        .query::<(Entity, &Chunk<C>)>()
        .iter(app.world())
        .find(|(_, chunk)| chunk.key() == ChunkKey::new(position, 0))
        .map(|(entity, _)| entity)
}

/// Runs `app` until world `C`'s chunks at and around the origin show their
/// meshes.
fn run_until_floor_shown<C: VoxelWorldConfig>(app: &mut App) {
    update_until(app, "the floor chunks to be shown", |app| {
        std::iter::once(IVec3::ZERO)
            .chain(FLOOR_ONLY)
            .all(|position| {
                chunk_entity::<C>(app, position)
                    .is_some_and(|entity| app.world().get::<Mesh3d>(entity).is_some())
            })
    });
}

fn light_at<C: VoxelWorldConfig>(app: &mut App, position: IVec3) -> LightLevels {
    app.world_mut()
        .run_system_once(move |voxel_world: VoxelWorld<C>| voxel_world.light_at(position))
        .unwrap()
        .unwrap_or_else(|| panic!("no light at {position}"))
}

/// Asserts that every shown chunk of world `C` carries the crate's tag and
/// the wrapped `M`, and none carries `M` itself.
fn assert_wrapped<C: VoxelWorldConfig, M: Material>(app: &mut App) {
    let chunks = shown_chunks::<C>(app);
    assert!(!chunks.is_empty());
    for entity in chunks {
        let world = app.world();
        assert!(world.get::<MeshTag>(entity).is_some(), "a chunk has no tag");
        assert!(world
            .get::<MeshMaterial3d<VoxelWorldMaterial<M>>>(entity)
            .is_some());
        assert!(world.get::<MeshMaterial3d<M>>(entity).is_none());
    }
}

#[test]
fn a_lit_world_with_a_custom_material_carries_the_wrapped_material_and_the_crates_tag() {
    let red = LinearRgba::RED;
    let mut app = custom_material_app();
    app.add_plugins(
        VoxelWorldPlugin::with_config(LampFloor::<0>::default())
            .with_material(TintMaterial { tint: red }),
    )
    .add_systems(Startup, spawn_camera::<LampFloor>);
    run_until_floor_shown::<LampFloor>(&mut app);

    assert_wrapped::<LampFloor, TintMaterial>(&mut app);

    // The world's own material is reached through the wrapped handle.
    let handle = app
        .world()
        .resource::<VoxelWorldMaterialHandle<VoxelWorldMaterial<TintMaterial>>>()
        .handle
        .clone();
    let materials = app
        .world()
        .resource::<Assets<VoxelWorldMaterial<TintMaterial>>>();
    assert_eq!(materials.len(), 1);
    assert_eq!(materials.get(&handle).unwrap().base.tint, red);
    assert!(app.world().resource::<Assets<TintMaterial>>().is_empty());
    assert!(
        chunk_materials::<LampFloor, VoxelWorldMaterial<TintMaterial>>(&mut app)
            .iter()
            .all(|material| *material == handle)
    );
}

#[test]
fn a_lit_world_with_plain_standard_material_carries_it_wrapped() {
    let mut app = custom_material_app();
    app.add_plugins(
        VoxelWorldPlugin::with_config(LampFloor::<0>::default())
            .with_material(StandardMaterial::default()),
    )
    .add_systems(Startup, spawn_camera::<LampFloor>);
    run_until_floor_shown::<LampFloor>(&mut app);

    assert_wrapped::<LampFloor, StandardMaterial>(&mut app);
}

#[test]
fn an_unshown_or_unlit_world_keeps_its_custom_material_unwrapped() {
    for light in [
        None,
        Some(LightPropagation {
            shown: false,
            sky_fill: None,
            ..default()
        }),
    ] {
        let mut app = custom_material_app();
        app.add_plugins(
            VoxelWorldPlugin::with_config(LampFloor::<0> {
                light: light.clone(),
                ..default()
            })
            .with_material(TintMaterial {
                tint: LinearRgba::RED,
            }),
        )
        .add_systems(Startup, spawn_camera::<LampFloor>);
        run_until_floor_shown::<LampFloor>(&mut app);

        for entity in shown_chunks::<LampFloor>(&mut app) {
            let world = app.world();
            assert!(world.get::<MeshTag>(entity).is_none());
            assert!(world.get::<MeshMaterial3d<TintMaterial>>(entity).is_some());
        }
        assert!(!app
            .world()
            .contains_resource::<Assets<VoxelWorldMaterial<TintMaterial>>>());
        assert!(!app.world().contains_resource::<LightRoute<LampFloor>>());
    }
}

#[test]
fn hand_added_materials_get_a_copy_per_lit_world() {
    let red = LinearRgba::RED;
    let not_init = LampFloor::<0> {
        init_custom_materials: false,
        ..default()
    };
    let mut app = custom_material_app();
    app.add_plugins(
        VoxelWorldPlugin::with_config(not_init.clone())
            .with_material(TintMaterial { tint: red }),
    )
    .add_plugins(
        VoxelWorldPlugin::with_config(LampFloor::<1> {
            init_custom_materials: false,
            ..default()
        })
        .with_material(TintMaterial { tint: red }),
    )
    .add_systems(
        Startup,
        (spawn_camera::<LampFloor>, spawn_camera::<LampFloor<1>>),
    );
    assert!(
        !app.world()
            .contains_resource::<Assets<VoxelWorldMaterial<TintMaterial>>>()
            || app
                .world()
                .resource::<Assets<VoxelWorldMaterial<TintMaterial>>>()
                .is_empty(),
        "the plugin made a material for a world that doesn't init one"
    );

    let shared = app
        .world_mut()
        .resource_mut::<Assets<TintMaterial>>()
        .add(TintMaterial { tint: red });
    app.insert_resource(VoxelWorldMaterialHandle { handle: shared });
    run_until_floor_shown::<LampFloor>(&mut app);
    run_until_floor_shown::<LampFloor<1>>(&mut app);

    assert_wrapped::<LampFloor, TintMaterial>(&mut app);
    assert_wrapped::<LampFloor<1>, TintMaterial>(&mut app);
    let first = chunk_materials::<LampFloor, VoxelWorldMaterial<TintMaterial>>(&mut app);
    let second =
        chunk_materials::<LampFloor<1>, VoxelWorldMaterial<TintMaterial>>(&mut app);
    assert!(first.iter().all(|material| *material == first[0]));
    assert!(second.iter().all(|material| *material == second[0]));
    assert_ne!(first[0], second[0], "two lit worlds share one wrapped copy");
    let materials = app
        .world()
        .resource::<Assets<VoxelWorldMaterial<TintMaterial>>>();
    for wrapped in [&first[0], &second[0]] {
        assert_eq!(materials.get(wrapped).unwrap().base.tint, red);
    }

    // The wrapped handle reaches one of the copies, for later edits.
    let handle = &app
        .world()
        .resource::<VoxelWorldMaterialHandle<VoxelWorldMaterial<TintMaterial>>>()
        .handle;
    assert!(*handle == first[0] || *handle == second[0]);
}

#[test]
fn a_look_change_reaches_the_copy_of_a_hand_added_material_in_its_own_world_alone() {
    let tint = TintMaterial {
        tint: LinearRgba::RED,
    };
    let mut app = custom_material_app();
    app.add_plugins(
        VoxelWorldPlugin::with_config(LampFloor::<0> {
            init_custom_materials: false,
            ..default()
        })
        .with_material(tint.clone()),
    )
    .add_plugins(
        VoxelWorldPlugin::with_config(LampFloor::<1> {
            init_custom_materials: false,
            ..default()
        })
        .with_material(tint.clone()),
    )
    .add_systems(
        Startup,
        (spawn_camera::<LampFloor>, spawn_camera::<LampFloor<1>>),
    );
    let shared = app
        .world_mut()
        .resource_mut::<Assets<TintMaterial>>()
        .add(tint);
    app.insert_resource(VoxelWorldMaterialHandle { handle: shared });
    run_until_floor_shown::<LampFloor>(&mut app);
    run_until_floor_shown::<LampFloor<1>>(&mut app);
    let brightness = |app: &mut App, copy: &Handle<VoxelWorldMaterial<TintMaterial>>| {
        app.world()
            .resource::<Assets<VoxelWorldMaterial<TintMaterial>>>()
            .get(copy)
            .unwrap()
            .extension
            .look()
            .voxel_brightness
    };
    let first = chunk_materials::<LampFloor, VoxelWorldMaterial<TintMaterial>>(&mut app)
        [0]
    .clone();
    let second =
        chunk_materials::<LampFloor<1>, VoxelWorldMaterial<TintMaterial>>(&mut app)[0]
            .clone();
    let default_brightness = brightness(&mut app, &first);

    app.world_mut()
        .resource_mut::<LampFloor>()
        .light
        .as_mut()
        .unwrap()
        .look
        .voxel_brightness = 250.0;
    app.update();

    assert_eq!(brightness(&mut app, &first), 250.0);
    assert_eq!(brightness(&mut app, &second), default_brightness);
}

#[test]
fn a_minimal_lit_world_with_a_custom_material_wraps_nothing() {
    let mut app = custom_material_app();
    app.add_plugins(VoxelWorldPlugin::<LampFloor>::minimal().with_material(
        TintMaterial {
            tint: LinearRgba::RED,
        },
    ))
    .add_systems(Startup, spawn_camera::<LampFloor>);

    update_until(&mut app, "the lamp's light to land", |app| {
        app.world_mut()
            .run_system_once(|voxel_world: VoxelWorld<LampFloor>| {
                voxel_world.light_at(LAMP_AT + IVec3::Y)
            })
            .unwrap()
            .is_some_and(|light| light.voxel > 0.0)
    });

    assert!(!app
        .world()
        .contains_resource::<Assets<VoxelWorldMaterial<TintMaterial>>>());
    assert!(!app
        .world()
        .contains_resource::<VoxelWorldMaterialHandle<VoxelWorldMaterial<TintMaterial>>>(
        ));
    assert!(!app.world().contains_resource::<LightRoute<LampFloor>>());
    let tagged = app
        .world_mut()
        .query_filtered::<(), (With<Chunk<LampFloor>>, With<MeshTag>)>()
        .iter(app.world())
        .count();
    assert_eq!(tagged, 0);
}

#[test]
fn a_chunk_whose_mesh_has_no_cube_face_still_gets_light() {
    let mut app = custom_material_app();
    app.add_plugins(
        VoxelWorldPlugin::with_config(LampFloor::<0> {
            sloped_mesh: true,
            ..default()
        })
        .with_material(StandardMaterial::default()),
    )
    .add_systems(Startup, spawn_camera::<LampFloor>);
    run_until_floor_shown::<LampFloor>(&mut app);
    update_until(&mut app, "the lamp's light to land", |app| {
        light_at::<LampFloor>(app, LAMP_AT + IVec3::Y).voxel > 0.0
    });
    app.update();

    assert_wrapped::<LampFloor, StandardMaterial>(&mut app);

    // The lamp's light varies across the chunk at the origin, so it takes a
    // slot.
    let tag_of = |app: &mut App, position: IVec3| {
        let entity = chunk_entity::<LampFloor>(app, position).unwrap();
        app.world().get::<MeshTag>(entity).unwrap().value
    };
    let tag = tag_of(&mut app, IVec3::ZERO);
    assert_ne!(tag & HAS_SLOT, 0, "tag {tag:#x}");

    // The floor beside it sees one level per channel, which its tag carries.
    for position in FLOOR_ONLY {
        let levels = light_at::<LampFloor>(&mut app, position * 32 + IVec3::new(5, 1, 5));
        assert_eq!(
            tag_of(&mut app, position),
            levels.sky as u32 | (levels.voxel as u32) << 4,
            "the chunk at {position}"
        );
    }
}
