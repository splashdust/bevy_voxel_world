//! Tests of the tag map's upkeep: which world chunk positions it holds
//! around a lit world's sub-grids, what each answers, and when its entries
//! are written.

use std::collections::HashSet;

use bevy::ecs::system::RunSystemOnce;
use bevy::mesh::MeshTag;
use bevy::prelude::*;
use bevy::render::render_resource::TextureFormat;

use crate::grid_bench::{tag_map_stats, TagMapStats};
use crate::light::route::{LightRoute, HAS_SLOT};
use crate::prelude::*;
use crate::rendering::VoxelWorldMaterial;
use crate::test::{meshing_app, spawn_camera, update_until};
use crate::voxel_material::DefaultVoxelMaterial;

const STONE: u8 = 1;
const LAMP: u8 = 2;

/// The tag of a position with nothing known, at 16 levels: sky 15 and
/// voxel 0.
const NOTHING_KNOWN: u32 = 15;

/// A stone floor at y = 0 through every chunk, lit or not.
#[derive(Resource, Clone)]
struct FloorWorld {
    light: Option<LightPropagation>,
}

impl Default for FloorWorld {
    fn default() -> Self {
        Self {
            light: Some(LightPropagation {
                sky_fill: None,
                ..default()
            }),
        }
    }
}

impl VoxelWorldConfig for FloorWorld {
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
                if position.y == 0 {
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
        vec![(LAMP, Emission::level(15.0))]
    }
}

/// Floor chunks beside the camera, which show their meshes.
const FLOOR: [IVec3; 3] = [IVec3::ZERO, IVec3::X, IVec3::Z];

/// Where the system [`move_sub_grids`] puts each sub-grid named, in the next
/// frame's `Update`.
#[derive(Resource, Default)]
struct Moves(Vec<(Entity, Transform)>);

fn move_sub_grids(mut moves: ResMut<Moves>, mut transforms: Query<&mut Transform>) {
    for (entity, to) in moves.0.drain(..) {
        *transforms.get_mut(entity).unwrap() = to;
    }
}

/// An app of `world`, run until the floor chunks show their meshes and
/// light has settled.
fn app(world: FloorWorld) -> App {
    let mut app = meshing_app();
    app.add_plugins(VoxelWorldPlugin::with_config(world))
        .init_resource::<Moves>()
        .add_systems(Startup, spawn_camera::<FloorWorld>)
        .add_systems(Update, move_sub_grids);
    settle(&mut app);
    app
}

/// Runs `app` until the floor chunks show their meshes and light has
/// settled, for a few frames in a row.
fn settle(app: &mut App) {
    let mut frames_done = 0;
    update_until(app, "the floor to show and light to settle", |app| {
        let settled = FLOOR
            .iter()
            .all(|position| shown_tag(app, *position).is_some())
            && app
                .world_mut()
                .run_system_once(|voxel_world: VoxelWorld<FloorWorld>| {
                    voxel_world.light_is_settled()
                })
                .unwrap();
        frames_done = if settled { frames_done + 1 } else { 0 };
        frames_done >= 3
    });
}

/// The `MeshTag` of the world chunk at `position`, if it shows its mesh.
fn shown_tag(app: &mut App, position: IVec3) -> Option<u32> {
    app.world_mut()
        .query_filtered::<(&Chunk<FloorWorld>, &MeshTag), With<Mesh3d>>()
        .iter(app.world())
        .find(|(chunk, _)| {
            chunk.grid_key().is_world() && chunk.key() == ChunkKey::new(position, 0)
        })
        .map(|(_, tag)| tag.value)
}

/// Spawns a sub-grid at `transform` holding stone at `voxels`.
fn spawn_sub_grid(app: &mut App, transform: Transform, voxels: &[IVec3]) -> Entity {
    let voxels = voxels.to_vec();
    app.world_mut()
        .run_system_once(move |mut voxel_world: VoxelWorld<FloorWorld>| {
            let entity = voxel_world.spawn_sub_grid(transform);
            let mut grid = voxel_world.grid_mut(Grid::SubGrid(entity)).unwrap();
            for voxel in &voxels {
                grid.set_voxel(*voxel, WorldVoxel::Solid(STONE));
            }
            entity
        })
        .unwrap()
}

fn route(app: &App) -> &LightRoute<FloorWorld> {
    app.world().resource::<LightRoute<FloorWorld>>()
}

/// The positions the tag map holds, if there is one.
fn held(app: &App) -> Option<HashSet<IVec3>> {
    Some(route(app).tag_map()?.positions().collect())
}

/// The tag the tag map holds at `position`.
fn answer(app: &App, position: IVec3) -> Option<u32> {
    route(app).tag_map()?.answer(position)
}

/// The entries whose writes are queued, as the render world would take them
/// after this frame: each position and tag written, leaving out entries
/// written empty.
fn take_writes(app: &mut App) -> Vec<(IVec3, u32)> {
    app.world_mut()
        .resource_mut::<LightRoute<FloorWorld>>()
        .take_tag_map_writes()
        .map(|writes| writes.entries().flatten().collect())
        .unwrap_or_default()
}

fn stats(app: &App) -> TagMapStats {
    tag_map_stats::<FloorWorld>(app.world()).expect("a tag map")
}

/// The chunk positions in the box from `min` to `max`, both inclusive.
fn positions(min: IVec3, max: IVec3) -> HashSet<IVec3> {
    let mut positions = HashSet::new();
    for x in min.x..=max.x {
        for y in min.y..=max.y {
            for z in min.z..=max.z {
                positions.insert(IVec3::new(x, y, z));
            }
        }
    }
    positions
}

/// Two stone voxels at opposite corners of a 3-voxel box.
const CORNERS: [IVec3; 2] = [IVec3::ZERO, IVec3::splat(2)];

#[test]
fn a_lit_world_makes_its_tag_map_in_the_frame_its_first_sub_grid_exists() {
    let mut app = app(FloorWorld::default());
    for _ in 0..3 {
        app.update();
    }
    assert!(route(&app).tag_map().is_none());
    assert!(tag_map_stats::<FloorWorld>(app.world()).is_none());

    spawn_sub_grid(&mut app, Transform::from_xyz(10.0, 5.0, 10.0), &CORNERS);
    app.update();

    assert_eq!(held(&app), Some(HashSet::from([IVec3::ZERO])));
    assert_eq!(stats(&app).full_rewrites, 1);
}

#[test]
fn a_world_without_light_makes_no_tag_map_with_sub_grids_spawned() {
    let mut app = meshing_app();
    app.add_plugins(VoxelWorldPlugin::with_config(FloorWorld { light: None }))
        .add_systems(Startup, spawn_camera::<FloorWorld>);
    app.update();
    spawn_sub_grid(&mut app, Transform::from_xyz(10.0, 5.0, 10.0), &CORNERS);
    for _ in 0..5 {
        app.update();
    }
    assert!(app
        .world()
        .get_resource::<LightRoute<FloorWorld>>()
        .is_none());
    assert!(tag_map_stats::<FloorWorld>(app.world()).is_none());
}

#[test]
fn the_tag_map_follows_a_sub_grids_padded_world_bounds_as_it_moves() {
    let mut app = app(FloorWorld::default());
    let entity = spawn_sub_grid(&mut app, Transform::from_xyz(10.0, 5.0, 10.0), &CORNERS);
    app.update();
    assert_eq!(held(&app), Some(HashSet::from([IVec3::ZERO])));

    // A quarter turn about x sends the box's y along -z, so its padded
    // bounds run from 29 to 34 on every axis.
    let turned = Transform::from_xyz(30.0, 33.0, 30.0)
        .with_rotation(Quat::from_rotation_x(std::f32::consts::FRAC_PI_2));
    app.world_mut()
        .resource_mut::<Moves>()
        .0
        .push((entity, turned));
    app.update();
    assert_eq!(held(&app), Some(positions(IVec3::ZERO, IVec3::ONE)));

    // An empty sub-grid adds nothing.
    spawn_sub_grid(&mut app, Transform::from_xyz(-100.0, 5.0, 10.0), &[]);
    app.update();
    assert_eq!(held(&app), Some(positions(IVec3::ZERO, IVec3::ONE)));
}

#[test]
fn each_position_answers_the_tag_its_world_chunk_shows_or_nothing_known() {
    let mut app = app(FloorWorld::default());
    // Over the floor chunks and the air above them, which shows no mesh.
    let turned = Transform::from_xyz(30.0, 33.0, 30.0)
        .with_rotation(Quat::from_rotation_x(std::f32::consts::FRAC_PI_2));
    spawn_sub_grid(&mut app, turned, &CORNERS);
    // Past the spawning distance.
    spawn_sub_grid(&mut app, Transform::from_xyz(330.0, 5.0, 10.0), &CORNERS);
    app.update();

    let held = held(&app).unwrap();
    assert!(held.contains(&IVec3::new(10, 0, 0)), "{held:?}");
    for position in held {
        let expected = shown_tag(&mut app, position).unwrap_or(NOTHING_KNOWN);
        assert_eq!(answer(&app, position), Some(expected), "{position}");
    }
    for position in FLOOR {
        assert!(shown_tag(&mut app, position).is_some());
    }
    assert_eq!(answer(&app, IVec3::new(10, 0, 0)), Some(NOTHING_KNOWN));
}

#[test]
fn an_entry_changes_in_the_frame_its_chunks_tag_changes() {
    let mut app = app(FloorWorld::default());
    // Over the floor chunk at x = 1, whose faces all see one level.
    spawn_sub_grid(&mut app, Transform::from_xyz(42.0, 3.0, 10.0), &CORNERS);
    app.update();
    take_writes(&mut app);
    let before = shown_tag(&mut app, IVec3::X).unwrap();
    assert_eq!(before & HAS_SLOT, 0);
    assert_eq!(answer(&app, IVec3::X), Some(before));

    // A lamp on the floor beside the sub-grid makes the chunk's faces see
    // more than one level, so it takes a slot.
    app.world_mut()
        .run_system_once(|mut voxel_world: VoxelWorld<FloorWorld>| {
            voxel_world.set_voxel(IVec3::new(48, 1, 16), WorldVoxel::Solid(LAMP));
        })
        .unwrap();
    let mut last = before;
    update_until(&mut app, "the floor chunk to take a slot", |app| {
        let tag = shown_tag(app, IVec3::X).unwrap();
        let writes = take_writes(app);
        if tag != last {
            assert!(
                writes.contains(&(IVec3::X, tag)),
                "{tag:#x} not written: {writes:?}"
            );
            last = tag;
        }
        assert_eq!(answer(app, IVec3::X), Some(tag));
        tag & HAS_SLOT != 0
    });
}

#[test]
fn nothing_is_written_while_a_sub_grid_stays_still() {
    let mut app = app(FloorWorld::default());
    let turned = Transform::from_xyz(30.0, 33.0, 30.0)
        .with_rotation(Quat::from_rotation_x(std::f32::consts::FRAC_PI_2));
    spawn_sub_grid(&mut app, turned, &CORNERS);
    // The sub-grid's own chunks have no light, so the world's settles.
    settle(&mut app);
    take_writes(&mut app);
    let written = stats(&app).entries_written;
    assert!(written > 0);
    for _ in 0..20 {
        app.update();
    }
    assert_eq!(stats(&app).entries_written, written);
    assert_eq!(take_writes(&mut app), []);
}

#[test]
fn a_full_table_grows_into_a_new_texture_and_writes_every_entry() {
    let mut app = app(FloorWorld::default());
    // Padded, each box runs from -1 to 102: chunks -1 to 3 on each axis,
    // 125 of them.
    let spread = [IVec3::ZERO, IVec3::splat(100)];
    spawn_sub_grid(&mut app, Transform::IDENTITY, &spread);
    app.update();
    let first = held(&app).unwrap();
    assert_eq!(first.len(), 125);
    let answers: Vec<(IVec3, Option<u32>)> = first
        .iter()
        .map(|position| (*position, answer(&app, *position)))
        .collect();
    take_writes(&mut app);
    let texture = route(&app).tag_map().unwrap().texture().clone();
    let rewrites = stats(&app).full_rewrites;

    spawn_sub_grid(&mut app, Transform::from_xyz(320.0, 0.0, 0.0), &spread);
    app.update();

    let held = held(&app).unwrap();
    assert_eq!(held.len(), 250);
    assert_eq!(stats(&app).full_rewrites, rewrites + 1);
    assert_ne!(route(&app).tag_map().unwrap().texture(), &texture);
    let written: HashSet<IVec3> = take_writes(&mut app)
        .into_iter()
        .map(|(position, _)| position)
        .collect();
    assert_eq!(written, held);
    for (position, answered) in answers {
        assert_eq!(answer(&app, position), answered, "{position}");
    }
}

/// The texture the world's material binds as its tag map.
fn bound_tag_map(app: &App) -> Handle<Image> {
    let materials = app
        .world()
        .resource::<Assets<VoxelWorldMaterial<DefaultVoxelMaterial>>>();
    assert_eq!(materials.len(), 1);
    let (_, material) = materials.iter().next().unwrap();
    material.extension.tag_map().clone()
}

#[test]
fn the_worlds_material_binds_a_tag_map_where_every_lookup_misses_until_its_first_sub_grid(
) {
    let mut app = app(FloorWorld::default());
    let placeholder = bound_tag_map(&app);
    let images = app.world().resource::<Assets<Image>>();
    let image = images.get(&placeholder).unwrap();
    // A table of one entry, which is empty, so a lookup starts and stops
    // there.
    assert_eq!(image.texture_descriptor.format, TextureFormat::Rgba32Uint);
    assert_eq!(image.width() * image.height(), 1);
    let tag = &image.data.as_ref().unwrap()[12..16];
    assert_eq!(tag, u32::MAX.to_le_bytes());

    spawn_sub_grid(&mut app, Transform::from_xyz(10.0, 5.0, 10.0), &CORNERS);
    app.update();
    let texture = route(&app).tag_map().unwrap().texture().clone();
    assert_eq!(bound_tag_map(&app), texture);
}

#[test]
fn the_worlds_material_binds_each_texture_its_tag_map_grows_into() {
    let mut app = app(FloorWorld::default());
    let spread = [IVec3::ZERO, IVec3::splat(100)];
    spawn_sub_grid(&mut app, Transform::IDENTITY, &spread);
    app.update();
    let first = route(&app).tag_map().unwrap().texture().id();
    assert_eq!(bound_tag_map(&app).id(), first);

    // A second box of 125 positions fills the table past half.
    spawn_sub_grid(&mut app, Transform::from_xyz(320.0, 0.0, 0.0), &spread);
    app.update();
    let grown = route(&app).tag_map().unwrap().texture().id();
    assert_ne!(grown, first);
    assert_eq!(bound_tag_map(&app).id(), grown);

    // Nothing binds the first texture any more, so it's dropped.
    app.update();
    app.update();
    assert!(app.world().resource::<Assets<Image>>().get(first).is_none());
}
