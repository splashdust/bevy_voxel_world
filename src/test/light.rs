//! Tests of light propagation. The tests of a world without light run in both
//! builds, with the `light_propagation` feature and without it.

use std::any::TypeId;
use std::collections::HashSet;

use bevy::prelude::*;

use super::{meshing_app, spawn_camera, update_until, DefaultVoxelMaterial, FloorWorld};
use crate::prelude::*;

#[cfg(feature = "light_propagation")]
mod reference;
#[cfg(feature = "light_propagation")]
mod voxel_light;

/// The types of the components on each chunk of world `C` that has a mesh
/// shown.
fn shown_chunk_components<C: VoxelWorldConfig>(app: &mut App) -> Vec<HashSet<TypeId>> {
    let entities: Vec<Entity> = app
        .world_mut()
        .query_filtered::<Entity, (With<Chunk<C>>, With<Mesh3d>)>()
        .iter(app.world())
        .collect();
    entities
        .into_iter()
        .map(|entity| {
            app.world()
                .inspect_entity(entity)
                .unwrap()
                .map(|(_, info)| info.type_id().unwrap())
                .collect()
        })
        .collect()
}

#[test]
fn a_world_returning_no_light_config_spawns_the_same_chunks_with_the_same_components_and_material_as_today(
) {
    let mut app = meshing_app();
    app.add_plugins(VoxelWorldPlugin::with_config(FloorWorld))
        .add_systems(Startup, spawn_camera::<FloorWorld>);

    // Streaming is done once the same chunks have been Ready for a few frames.
    let mut settled = (0, 0);
    update_until(&mut app, "the world to finish streaming", |app| {
        let keys = super::chunk_keys::<FloorWorld>(app);
        let ready = !keys.is_empty()
            && keys.iter().all(|key| {
                super::chunk_status_of::<FloorWorld>(app, *key)
                    == Some(ChunkStatus::Ready)
            });
        settled = match ready {
            true if settled.0 == keys.len() => (keys.len(), settled.1 + 1),
            true => (keys.len(), 0),
            false => (0, 0),
        };
        settled.1 >= 10
    });

    let keys = super::chunk_keys::<FloorWorld>(&mut app);
    let shown = app
        .world_mut()
        .query_filtered::<(), (With<Chunk<FloorWorld>>, With<Mesh3d>)>()
        .iter(app.world())
        .count();
    assert_eq!(
        (keys.len(), shown),
        (7, 1),
        "the world spawned or showed other chunks than before"
    );

    let today = HashSet::from([
        TypeId::of::<Chunk<FloorWorld>>(),
        TypeId::of::<crate::chunk_lifecycle::ChunkStateOf<FloorWorld>>(),
        TypeId::of::<ChildOf>(),
        TypeId::of::<Transform>(),
        TypeId::of::<GlobalTransform>(),
        TypeId::of::<bevy::transform::components::TransformTreeChanged>(),
        TypeId::of::<Visibility>(),
        TypeId::of::<InheritedVisibility>(),
        TypeId::of::<ViewVisibility>(),
        TypeId::of::<bevy::camera::visibility::VisibilityClass>(),
        TypeId::of::<bevy::camera::primitives::Aabb>(),
        TypeId::of::<Mesh3d>(),
        TypeId::of::<MeshMaterial3d<DefaultVoxelMaterial>>(),
        TypeId::of::<crate::mesh_cache::MeshRef>(),
    ]);
    for components in shown_chunk_components::<FloorWorld>(&mut app) {
        assert_eq!(
            (
                components.difference(&today).count(),
                today.difference(&components).count()
            ),
            (0, 0),
            "a shown chunk has other components than before (added, missing)"
        );
    }

    let materials = super::chunk_materials::<FloorWorld, DefaultVoxelMaterial>(&mut app);
    assert!(materials.windows(2).all(|pair| pair[0] == pair[1]));

    #[cfg(feature = "light_propagation")]
    assert!(
        !app.world()
            .contains_resource::<crate::light::LightSettings<FloorWorld>>(),
        "a world returning no light config keeps light settings"
    );
}

#[cfg(feature = "light_propagation")]
mod with_feature {
    use std::time::Duration;

    use bevy::prelude::*;

    use super::super::{
        chunk_keys, chunk_status, chunk_status_of, floor_voxel, meshing_app,
        spawn_camera, update_until, FloorWorld,
    };
    use crate::light::LightSettings;
    use crate::prelude::*;

    #[test]
    fn light_config_defaults_match_the_documented_defaults() {
        let config = LightPropagation::default();
        assert!(config.shown);
        assert_eq!(config.sky, Sky::Clearance { chunks: 16 });
        assert_eq!(config.sky_reach, 30);
        assert_eq!(config.voxel_reach, 14);
        assert_eq!(config.shape, LightShape::Round);
        assert_eq!(
            config.sky_fill,
            Some(SkyFill::Paths(PathFill {
                beside_open_sky: 12.0,
                whole_face_loss: 2.0,
                opening_power: 2.0,
            }))
        );
        assert_eq!(config.precision, LightPrecision::Steps16);
        assert_eq!(config.slot_capacity, 1024);
        assert_eq!(config.sky_record_cap, 16 * 1024 * 1024);
        assert_eq!(config.wasm_relight_budget, Duration::from_millis(4));

        let look = config.look;
        assert_eq!(look.smoothing, LightSmoothing::Smoothed);
        assert_eq!(look.sun, SunGate::DirectSkyOnly);
        assert_eq!(look.voxel_brightness, 1000.0);
        assert_eq!(look.emissive_glow, 6.0);
        assert_eq!(look.ao, LightAo::IndirectLight);
        assert_eq!(look.minimum_brightness, 0.0);
        for curve in [&look.sky_falloff, &look.voxel_falloff] {
            assert_eq!(curve.brightness[15], 1.0);
            assert!((curve.brightness[14] - 0.7).abs() < 1e-6);
            assert!((curve.brightness[13] - 0.49).abs() < 1e-6);
        }

        assert_eq!(
            OpenFill::default(),
            OpenFill {
                sun_at: 0.8,
                gain: 1.0,
                rays: 32,
                reach: 128,
            }
        );
        // Struct update syntax works on the config structs.
        let _ = LightPropagation {
            sky_reach: 16,
            look: LightLook {
                emissive_glow: 2.0,
                ..default()
            },
            ..default()
        };
    }

    /// A floor world with a light config whose reaches, slot capacity and
    /// emission levels are out of range.
    #[derive(Resource, Clone, Default)]
    struct OutOfRangeWorld;

    impl VoxelWorldConfig for OutOfRangeWorld {
        type MaterialIndex = u8;
        type ChunkUserBundle = ();

        fn spawning_distance(&self) -> u32 {
            2
        }

        fn voxel_lookup_delegate(&self) -> VoxelLookupDelegate<Self::MaterialIndex> {
            Box::new(|_, _, _| Box::new(|pos, _| floor_voxel(pos)))
        }

        fn light_propagation(&self) -> Option<LightPropagation> {
            Some(LightPropagation {
                sky_reach: 0,
                voxel_reach: 200,
                slot_capacity: 0,
                ..default()
            })
        }

        fn emissive_materials(&self) -> Vec<(Self::MaterialIndex, Emission)> {
            vec![(1, Emission::level(99.0)), (2, Emission::level(-1.0))]
        }
    }

    #[test]
    fn an_out_of_range_reach_clamps_and_doesnt_panic() {
        let mut app = App::new();
        app.add_plugins((
            MinimalPlugins,
            VoxelWorldPlugin::<OutOfRangeWorld>::minimal(),
        ))
        .add_systems(Startup, spawn_camera::<OutOfRangeWorld>);

        update_until(&mut app, "the origin chunk to be ready", |app| {
            chunk_status::<OutOfRangeWorld>(app, IVec3::ZERO) == Some(ChunkStatus::Ready)
        });

        let settings = app.world().resource::<LightSettings<OutOfRangeWorld>>();
        assert_eq!(settings.config.sky_reach, 1);
        assert_eq!(settings.config.voxel_reach, 32);
        assert_eq!(settings.config.slot_capacity, 1);
        assert_eq!(
            settings.emissive_materials,
            vec![(1, Emission::level(15.0)), (2, Emission::level(0.0))]
        );

        // The world's own value is left as it returned it.
        let config = app.world().resource::<OutOfRangeWorld>();
        assert_eq!(config.light_propagation().unwrap().sky_reach, 0);
    }

    /// A floor world with the default light config and one emissive material.
    #[derive(Resource, Clone, Default)]
    struct LitFloorWorld;

    impl VoxelWorldConfig for LitFloorWorld {
        type MaterialIndex = u8;
        type ChunkUserBundle = ();

        fn spawning_distance(&self) -> u32 {
            2
        }

        fn voxel_lookup_delegate(&self) -> VoxelLookupDelegate<Self::MaterialIndex> {
            FloorWorld.voxel_lookup_delegate()
        }

        fn light_propagation(&self) -> Option<LightPropagation> {
            Some(LightPropagation::default())
        }

        fn emissive_materials(&self) -> Vec<(Self::MaterialIndex, Emission)> {
            vec![(2, Emission::level(15.0))]
        }
    }

    #[test]
    fn a_world_can_return_a_light_config_and_an_emissive_list_and_still_stream() {
        let mut app = meshing_app();
        app.add_plugins(VoxelWorldPlugin::with_config(LitFloorWorld))
            .add_systems(Startup, spawn_camera::<LitFloorWorld>);

        // The same chunks the unlit floor world streams and shows.
        update_until(&mut app, "the ground chunk to be shown", |app| {
            let keys = chunk_keys::<LitFloorWorld>(app);
            let shown = app
                .world_mut()
                .query_filtered::<(), (With<Chunk<LitFloorWorld>>, With<Mesh3d>)>()
                .iter(app.world())
                .count();
            keys.len() == 7
                && shown == 1
                && keys.iter().all(|key| {
                    chunk_status_of::<LitFloorWorld>(app, *key)
                        == Some(ChunkStatus::Ready)
                })
        });
        assert!(app
            .world()
            .contains_resource::<LightSettings<LitFloorWorld>>());
    }
}
