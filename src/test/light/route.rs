//! Tests of the route light takes to a lit world's chunks: the wrapped default
//! material, and the crate's `MeshTag` naming a chunk's slot or its levels.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use bevy::ecs::system::RunSystemOnce;
use bevy::mesh::MeshTag;
use bevy::prelude::*;

use super::super::{
    chunk_status, meshing_app, spawn_camera, update_until, DefaultVoxelMaterial,
};
use super::voxel_light::Hold;
use crate::light::route::LightRoute;
use crate::prelude::*;
use crate::rendering::VoxelWorldMaterial;

const STONE: u8 = 1;
const LAMP: u8 = 2;
const DIM_LAMP: u8 = 3;

/// A lamp on the floor of the chunk at the origin.
const LAMP_AT: IVec3 = IVec3::new(16, 1, 16);
/// A dim lamp on the same floor, inside the bright lamp's pool.
const DIM_LAMP_AT: IVec3 = IVec3::new(20, 1, 16);

/// A stone floor at y = 0 through every chunk, with two lamps on it in the
/// chunk at the origin.
#[derive(Resource, Clone)]
struct FloorLampWorld {
    light: Option<LightPropagation>,
    /// Holds back the voxel lookups of the chunk at `HELD` until set.
    release: Option<Arc<AtomicBool>>,
    /// Puts a lamp in the held chunk, beside the chunk at the origin.
    edge_lamp: bool,
}

/// The lamp `FloorLampWorld::edge_lamp` puts in the held chunk.
const EDGE_LAMP_AT: IVec3 = IVec3::new(-2, 1, 16);

/// The chunk whose task waits for `FloorLampWorld::release`.
const HELD: IVec3 = IVec3::NEG_X;

impl Default for FloorLampWorld {
    fn default() -> Self {
        Self {
            light: Some(LightPropagation {
                sky_fill: None,
                ..default()
            }),
            release: None,
            edge_lamp: false,
        }
    }
}

impl VoxelWorldConfig for FloorLampWorld {
    type MaterialIndex = u8;
    type ChunkUserBundle = ();

    fn spawning_distance(&self) -> u32 {
        2
    }

    fn chunk_spawn_strategy(&self) -> ChunkSpawnStrategy {
        ChunkSpawnStrategy::Close
    }

    fn voxel_lookup_delegate(&self) -> VoxelLookupDelegate<Self::MaterialIndex> {
        let release = self.release.clone();
        let edge_lamp = self.edge_lamp;
        Box::new(move |chunk, _, _| {
            let release = release.clone().filter(|_| chunk == HELD);
            Box::new(move |position, _| {
                if let Some(release) = &release {
                    let deadline = Instant::now() + Duration::from_secs(20);
                    while !release.load(Ordering::Acquire) && Instant::now() < deadline {
                        std::thread::sleep(Duration::from_millis(1));
                    }
                }
                if position == LAMP_AT || (edge_lamp && position == EDGE_LAMP_AT) {
                    WorldVoxel::Solid(LAMP)
                } else if position == DIM_LAMP_AT {
                    WorldVoxel::Solid(DIM_LAMP)
                } else if position.y == 0 {
                    WorldVoxel::Solid(STONE)
                } else {
                    WorldVoxel::Air
                }
            })
        })
    }

    fn light_propagation(&self) -> Option<LightPropagation> {
        self.light.clone()
    }

    fn emissive_materials(&self) -> Vec<(Self::MaterialIndex, Emission)> {
        vec![
            (LAMP, Emission::level(15.0)),
            (DIM_LAMP, Emission::level(8.0)),
        ]
    }
}

/// The chunks beside the origin along x and z, which hold only floor.
const FLOOR_ONLY: [IVec3; 4] = [IVec3::X, IVec3::NEG_X, IVec3::Z, IVec3::NEG_Z];

/// An app of `world`, run until the floor chunks at and around the origin
/// show their meshes.
fn floor_shown(world: FloorLampWorld) -> App {
    let mut app = meshing_app();
    app.add_plugins(VoxelWorldPlugin::with_config(world))
        .add_systems(Startup, spawn_camera::<FloorLampWorld>);
    run_until_shown(&mut app, std::iter::once(IVec3::ZERO).chain(FLOOR_ONLY));
    app
}

/// Runs `app` until the chunks at `positions` show their meshes.
fn run_until_shown(app: &mut App, positions: impl Iterator<Item = IVec3> + Clone) {
    update_until(app, "the floor chunks to be shown", |app| {
        positions.clone().all(|position| {
            chunk_status::<FloorLampWorld>(app, position) == Some(ChunkStatus::Ready)
                && chunk_entity(app, position)
                    .is_some_and(|entity| app.world().get::<Mesh3d>(entity).is_some())
        })
    });
}

fn chunk_entity(app: &mut App, position: IVec3) -> Option<Entity> {
    app.world_mut()
        .query::<(Entity, &Chunk<FloorLampWorld>)>()
        .iter(app.world())
        .find(|(_, chunk)| chunk.key() == ChunkKey::new(position, 0))
        .map(|(entity, _)| entity)
}

/// The tag of the chunk at `position`.
fn tag_of(app: &mut App, position: IVec3) -> u32 {
    let entity = chunk_entity(app, position).expect("the chunk exists");
    app.world()
        .get::<MeshTag>(entity)
        .unwrap_or_else(|| panic!("the chunk at {position} has no tag"))
        .value
}

/// The entities of the chunks showing a mesh.
fn shown_chunks(app: &mut App) -> Vec<Entity> {
    app.world_mut()
        .query_filtered::<Entity, (With<Chunk<FloorLampWorld>>, With<Mesh3d>)>()
        .iter(app.world())
        .collect()
}

fn light_at(app: &mut App, position: IVec3) -> LightLevels {
    app.world_mut()
        .run_system_once(move |voxel_world: VoxelWorld<FloorLampWorld>| {
            voxel_world.light_at(position)
        })
        .unwrap()
        .unwrap_or_else(|| panic!("no light at {position}"))
}

const HAS_SLOT: u32 = 1 << 31;

#[test]
fn a_lit_worlds_chunks_carry_the_crates_tag_and_the_wrapped_material() {
    let mut app = floor_shown(FloorLampWorld::default());

    let chunks = shown_chunks(&mut app);
    assert!(!chunks.is_empty());
    for entity in chunks {
        let world = app.world();
        assert!(world.get::<MeshTag>(entity).is_some(), "a chunk has no tag");
        assert!(world
            .get::<MeshMaterial3d<VoxelWorldMaterial<DefaultVoxelMaterial>>>(entity)
            .is_some());
        assert!(world
            .get::<MeshMaterial3d<DefaultVoxelMaterial>>(entity)
            .is_none());
    }
}

#[test]
fn an_unlit_worlds_chunks_carry_neither() {
    let mut app = floor_shown(FloorLampWorld {
        light: None,
        ..default()
    });

    for entity in shown_chunks(&mut app) {
        let world = app.world();
        assert!(world.get::<MeshTag>(entity).is_none());
        assert!(world
            .get::<MeshMaterial3d<DefaultVoxelMaterial>>(entity)
            .is_some());
    }
    assert!(!app
        .world()
        .contains_resource::<LightRoute<FloorLampWorld>>());
}

#[test]
fn shown_false_wraps_nothing_and_tags_nothing() {
    let mut app = floor_shown(FloorLampWorld {
        light: Some(LightPropagation {
            shown: false,
            sky_fill: None,
            ..default()
        }),
        ..default()
    });

    for entity in shown_chunks(&mut app) {
        let world = app.world();
        assert!(world.get::<MeshTag>(entity).is_none());
        assert!(world
            .get::<MeshMaterial3d<DefaultVoxelMaterial>>(entity)
            .is_some());
    }
    assert!(!app
        .world()
        .contains_resource::<LightRoute<FloorLampWorld>>());
    assert!(
        !app.world()
            .contains_resource::<Assets<VoxelWorldMaterial<DefaultVoxelMaterial>>>()
            || app
                .world()
                .resource::<Assets<VoxelWorldMaterial<DefaultVoxelMaterial>>>()
                .is_empty()
    );

    // It reads the same light as a shown world.
    let mut shown = floor_shown(FloorLampWorld::default());
    for position in [
        LAMP_AT,
        DIM_LAMP_AT,
        LAMP_AT + IVec3::Y,
        IVec3::new(9, 1, 16),
    ] {
        assert_eq!(
            light_at(&mut app, position),
            light_at(&mut shown, position),
            "at {position}"
        );
    }
}

#[test]
fn tag_bits_28_to_30_stay_clear() {
    let mut app = floor_shown(FloorLampWorld::default());

    for entity in shown_chunks(&mut app) {
        let tag = app.world().get::<MeshTag>(entity).unwrap().value;
        assert_eq!(tag & (0b111 << 28), 0, "tag {tag:#x}");
    }
}

#[test]
fn a_chunks_tag_names_its_slot_or_its_levels_when_it_has_none() {
    let mut app = floor_shown(FloorLampWorld::default());

    // The lamps' light varies across the chunk at the origin, so it has a slot.
    let tag = tag_of(&mut app, IVec3::ZERO);
    assert_ne!(tag & HAS_SLOT, 0, "tag {tag:#x}");
    let slot = tag & 0xFF_FFFF;
    assert!(slot < 1024);

    // Its slot's bytes hold sky light in the low nibble and voxel light in the
    // high one, and each lamp's own cell holds 15 in its voxel half and its
    // level in its sky half. The slot is the chunk and a shell one voxel deep,
    // with rows along z.
    let bytes = app
        .world()
        .resource::<LightRoute<FloorLampWorld>>()
        .queued_upload(slot)
        .expect("the slot's bytes are queued")
        .to_vec();
    let byte_at = |voxel: IVec3| {
        let cell = voxel + IVec3::ONE;
        bytes[(cell.z + cell.x * 34 + cell.y * 34 * 34) as usize]
    };
    assert_eq!(byte_at(LAMP_AT), 0xF0 | 15);
    assert_eq!(byte_at(DIM_LAMP_AT), 0xF0 | 8);
    for voxel in [
        LAMP_AT + IVec3::Y,
        IVec3::new(9, 1, 16),
        IVec3::new(2, 20, 30),
    ] {
        let levels = light_at(&mut app, voxel);
        assert_eq!(
            byte_at(voxel),
            levels.sky as u8 | (levels.voxel as u8) << 4,
            "at {voxel}"
        );
    }
    assert_eq!(
        byte_at(IVec3::new(9, 0, 16)),
        0,
        "a solid voxel holds no light"
    );

    // The floor beside it sees one level per channel, which its tag carries.
    for position in FLOOR_ONLY {
        let tag = tag_of(&mut app, position);
        let levels = light_at(&mut app, position * 32 + IVec3::new(5, 1, 5));
        assert_eq!(
            tag,
            levels.sky as u32 | (levels.voxel as u32) << 4,
            "the chunk at {position}"
        );
    }
}

#[test]
fn a_lit_worlds_chunks_still_share_cached_meshes() {
    let _holding = Hold::exclusive();
    // The held chunk's task finds the mesh of the same floor in the cache.
    let release = Arc::new(AtomicBool::new(false));
    let mut app = meshing_app();
    app.add_plugins(VoxelWorldPlugin::with_config(FloorLampWorld {
        release: Some(release.clone()),
        ..default()
    }))
    .add_systems(Startup, spawn_camera::<FloorLampWorld>);
    run_until_shown(&mut app, [IVec3::X].into_iter());
    release.store(true, Ordering::Release);
    run_until_shown(&mut app, [HELD].into_iter());

    // Floor chunks meshed at the same time can each make a mesh, but the held
    // chunk takes one of theirs.
    let held = chunk_entity(&mut app, HELD).unwrap();
    let held_mesh = app.world().get::<Mesh3d>(held).unwrap().0.id();
    let others: Vec<AssetId<Mesh>> = shown_chunks(&mut app)
        .into_iter()
        .filter(|entity| *entity != held)
        .map(|entity| app.world().get::<Mesh3d>(entity).unwrap().0.id())
        .collect();
    assert!(
        others.contains(&held_mesh),
        "the held floor chunk has a mesh of its own"
    );
    for position in [IVec3::X, HELD] {
        tag_of(&mut app, position);
    }
}

#[test]
fn a_relight_updates_a_shown_chunks_slot() {
    let _holding = Hold::exclusive();
    // The chunk at the origin shows before the lamp beside it arrives, and
    // relights when it does.
    let release = Arc::new(AtomicBool::new(false));
    let mut app = meshing_app();
    app.add_plugins(VoxelWorldPlugin::with_config(FloorLampWorld {
        release: Some(release.clone()),
        edge_lamp: true,
        ..default()
    }))
    .add_systems(Startup, spawn_camera::<FloorLampWorld>);
    run_until_shown(&mut app, [IVec3::ZERO].into_iter());
    let beside_edge_lamp = IVec3::new(0, 1, 16);
    assert_eq!(light_at(&mut app, beside_edge_lamp).voxel, 0.0);

    release.store(true, Ordering::Release);
    run_until_shown(&mut app, [HELD].into_iter());
    update_until(&mut app, "the edge lamp's light to land", |app| {
        light_at(app, beside_edge_lamp).voxel > 0.0
    });
    app.update();

    let tag = tag_of(&mut app, IVec3::ZERO);
    assert_ne!(tag & HAS_SLOT, 0, "tag {tag:#x}");
    let levels = light_at(&mut app, beside_edge_lamp);
    let bytes = app
        .world()
        .resource::<LightRoute<FloorLampWorld>>()
        .queued_upload(tag & 0xFF_FFFF)
        .expect("the slot's bytes are queued")
        .to_vec();
    let cell = beside_edge_lamp + IVec3::ONE;
    assert_eq!(
        bytes[(cell.z + cell.x * 34 + cell.y * 34 * 34) as usize],
        levels.sky as u8 | (levels.voxel as u8) << 4
    );
}
