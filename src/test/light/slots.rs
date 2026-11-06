//! Tests of which chunks of a lit world take a slot in its light texture, what
//! a slot's shell holds, and what happens when slots run out.

use std::sync::Arc;

use bevy::ecs::system::RunSystemOnce;
use bevy::mesh::MeshTag;
use bevy::prelude::*;
use bevy::render::render_resource::TextureDimension;

use super::super::{chunk_status, update_until};
use super::voxel_light::{app, light_at, lit, Hold, LampWorld, LAMP, STONE};
use crate::light::route::LightRoute;
use crate::prelude::*;

const HAS_SLOT: u32 = 1 << 31;

/// A material emitting light at level 1, which lights no voxel around it.
const FAINT_LAMP: u8 = 4;

fn chunk_entity(app: &mut App, position: IVec3) -> Option<Entity> {
    app.world_mut()
        .query::<(Entity, &Chunk<LampWorld>)>()
        .iter(app.world())
        .find(|(_, chunk)| chunk.key() == ChunkKey::new(position, 0))
        .map(|(entity, _)| entity)
}

/// The tag of the chunk at `position`, if it shows its mesh.
fn tag_of(app: &mut App, position: IVec3) -> Option<u32> {
    let entity = chunk_entity(app, position)?;
    app.world().get::<MeshTag>(entity).map(|tag| tag.value)
}

fn has_slot(app: &mut App, position: IVec3) -> bool {
    tag_of(app, position).is_some_and(|tag| tag & HAS_SLOT != 0)
}

fn light_is_settled(app: &mut App) -> bool {
    app.world_mut()
        .run_system_once(|voxel_world: VoxelWorld<LampWorld>| {
            voxel_world.light_is_settled()
        })
        .unwrap()
}

/// Runs `app` until the chunks at `positions` show their meshes and light has
/// settled.
fn run_until_shown(app: &mut App, positions: &[IVec3]) {
    update_until(app, "the chunks to show and light to settle", |app| {
        positions.iter().all(|position| {
            chunk_status::<LampWorld>(app, *position) == Some(ChunkStatus::Ready)
                && tag_of(app, *position).is_some()
        }) && light_is_settled(app)
    });
}

/// Runs `app` until `done` holds and light has settled.
fn run_until(app: &mut App, what: &str, mut done: impl FnMut(&mut App) -> bool) {
    update_until(app, what, |app| light_is_settled(app) && done(app));
}

/// The bytes queued for the slot of the chunk at `position`, which has one.
fn slot_bytes(app: &mut App, position: IVec3) -> Vec<u8> {
    let tag = tag_of(app, position).expect("the chunk shows its mesh");
    assert_ne!(tag & HAS_SLOT, 0, "the chunk at {position} has no slot");
    app.world()
        .resource::<LightRoute<LampWorld>>()
        .queued_upload(tag & 0xFF_FFFF)
        .expect("the slot's bytes are queued")
        .to_vec()
}

/// The byte of a slot at the cell of the chunk at `chunk` holding the voxel
/// at `voxel`, which may be in its shell.
fn byte_at(bytes: &[u8], chunk: IVec3, voxel: IVec3) -> u8 {
    let cell = voxel - chunk * 32 + IVec3::ONE;
    assert!(cell.cmpge(IVec3::ZERO).all() && cell.cmplt(IVec3::splat(34)).all());
    bytes[(cell.z + cell.x * 34 + cell.y * 34 * 34) as usize]
}

fn texel(levels: LightLevels) -> u8 {
    levels.sky as u8 | (levels.voxel as u8) << 4
}

/// Ground below y = 1 everywhere, lit only by lamps, spreading in diamonds so
/// that levels drop exactly one per face step.
fn ground(lamps: Vec<(IVec3, u8)>) -> LampWorld {
    LampWorld {
        scene: Arc::new(move |p| {
            if let Some((_, material)) = lamps.iter().find(|(lamp, _)| *lamp == p) {
                WorldVoxel::Solid(*material)
            } else if p.y <= 0 {
                WorldVoxel::Solid(STONE)
            } else {
                WorldVoxel::Air
            }
        }),
        light: Some(lit(LightShape::Diamond)),
        emissive: vec![
            (LAMP, Emission::level(15.0)),
            (FAINT_LAMP, Emission::level(1.0)),
        ],
        ..default()
    }
}

/// The light textures the world's route has allocated: 3D images bigger than
/// one texel.
fn light_textures(app: &App) -> Vec<UVec3> {
    app.world()
        .resource::<Assets<Image>>()
        .iter()
        .filter(|(_, image)| image.texture_descriptor.dimension == TextureDimension::D3)
        .map(|(_, image)| {
            let size = image.texture_descriptor.size;
            UVec3::new(size.width, size.height, size.depth_or_array_layers)
        })
        .filter(|size| *size != UVec3::ONE)
        .collect()
}

#[test]
fn the_slot_rule_counts_voxels_that_touch_a_solid_only_at_an_edge_or_corner() {
    // One solid voxel in the chunk at the origin. A lamp in the chunk beyond
    // its corner lights the voxel touching it at that corner at level 1, and
    // the voxels touching it at an edge or a face at 0.
    let solid = IVec3::splat(28);
    let corner = solid + IVec3::ONE;
    let one_voxel = |lamp: IVec3| LampWorld {
        scene: Arc::new(move |p| {
            if p == solid {
                WorldVoxel::Solid(STONE)
            } else if p == lamp {
                WorldVoxel::Solid(LAMP)
            } else {
                WorldVoxel::Air
            }
        }),
        light: Some(lit(LightShape::Diamond)),
        ..default()
    };
    let in_reach = IVec3::new(33, 33, 35);
    let mut near = app(one_voxel(in_reach));
    run_until_shown(&mut near, &[IVec3::ZERO, IVec3::ONE]);
    assert_eq!(light_at(&mut near, corner).unwrap().voxel, 1.0);
    assert_eq!(
        light_at(&mut near, solid + IVec3::new(1, 1, 0))
            .unwrap()
            .voxel,
        0.0
    );
    assert_eq!(light_at(&mut near, solid + IVec3::X).unwrap().voxel, 0.0);
    assert!(has_slot(&mut near, IVec3::ZERO));

    // One voxel farther, the lamp lights none of them, so every voxel touching
    // the solid sees one level, though the chunk's light varies elsewhere.
    let mut farther = app(one_voxel(in_reach + IVec3::Z));
    run_until_shown(&mut farther, &[IVec3::ZERO, IVec3::ONE]);
    assert_eq!(light_at(&mut farther, corner).unwrap().voxel, 0.0);
    assert!(light_at(&mut farther, IVec3::splat(31)).unwrap().voxel > 0.0);
    assert_eq!(tag_of(&mut farther, IVec3::ZERO), Some(0));
}

#[test]
fn an_exposed_emissive_voxel_forces_a_slot() {
    // A lamp at level 1 lights nothing around it, so every face sees level 0.
    let lamp = IVec3::new(16, 1, 16);
    let mut app = app(ground(vec![(lamp, FAINT_LAMP)]));
    run_until_shown(&mut app, &[IVec3::ZERO]);
    assert_eq!(light_at(&mut app, lamp + IVec3::Y).unwrap().voxel, 0.0);

    let bytes = slot_bytes(&mut app, IVec3::ZERO);
    assert_eq!(byte_at(&bytes, IVec3::ZERO, lamp), 0xF0 | 1);
}

#[test]
fn an_unloaded_neighbour_never_forces_a_slot() {
    let _holding = Hold::exclusive();
    // Ground under the sky: every face of the chunk at the origin sees sky
    // light 15, and its neighbour along x isn't loaded yet.
    let hold = Hold::new(IVec3::X);
    let mut world = ground(Vec::new());
    world.light = Some(LightPropagation {
        sky_fill: None,
        ..default()
    });
    world.hold = Some(hold.clone());
    let mut app = app(world);
    run_until_shown(&mut app, &[IVec3::ZERO]);
    let tag = tag_of(&mut app, IVec3::ZERO).unwrap();
    assert_eq!(tag & HAS_SLOT, 0, "tag {tag:#x}");

    hold.release();
    run_until_shown(&mut app, &[IVec3::ZERO, IVec3::X]);
    assert_eq!(tag_of(&mut app, IVec3::ZERO), Some(tag));
}

#[test]
fn nothing_is_allocated_while_every_chunks_faces_see_one_level() {
    let mut flat = app(ground(Vec::new()));
    run_until_shown(&mut flat, &[IVec3::ZERO, IVec3::X, IVec3::NEG_Z]);
    assert!(light_textures(&flat).is_empty());

    let mut lit = app(ground(vec![(IVec3::new(16, 1, 16), LAMP)]));
    run_until_shown(&mut lit, &[IVec3::ZERO]);
    assert_eq!(light_textures(&lit), vec![UVec3::new(544, 272, 272)]);
}

#[test]
fn a_flat_chunk_takes_a_slot_when_a_neighbours_plane_stops_matching_and_gives_it_back_when_its_faces_see_one_level(
) {
    // A lamp in the chunk at the origin lights its border beside the chunk
    // along x at level 1 in one voxel, and that chunk's own voxels at 0.
    let lamp = IVec3::new(17, 1, 16);
    let beside = IVec3::X;
    let mut app = app(ground(Vec::new()));
    run_until_shown(&mut app, &[IVec3::ZERO, beside]);
    assert!(!has_slot(&mut app, beside));

    set_voxel(&mut app, lamp, WorldVoxel::Solid(LAMP));
    run_until(&mut app, "the chunk beside to take a slot", |app| {
        has_slot(app, beside)
    });
    assert_eq!(
        light_at(&mut app, IVec3::new(31, 1, 16)).unwrap().voxel,
        1.0
    );
    assert_eq!(
        light_at(&mut app, IVec3::new(32, 1, 16)).unwrap().voxel,
        0.0
    );

    set_voxel(&mut app, lamp, WorldVoxel::Air);
    run_until(&mut app, "the chunk beside to give its slot back", |app| {
        tag_of(app, beside) == Some(0)
    });
}

#[test]
fn faces_on_chunk_boundaries_read_their_neighbours_light_through_the_shell() {
    let lamp = IVec3::new(17, 1, 16);
    let beside = IVec3::X;
    let mut app = app(ground(vec![(lamp, LAMP)]));
    run_until_shown(&mut app, &[IVec3::ZERO, beside]);

    let bytes = slot_bytes(&mut app, beside);
    for voxel in [
        IVec3::new(31, 1, 16),
        IVec3::new(31, 1, 15),
        IVec3::new(31, 2, 16),
    ] {
        let levels = light_at(&mut app, voxel).unwrap();
        assert_eq!(byte_at(&bytes, beside, voxel), texel(levels), "at {voxel}");
    }
    assert_eq!(byte_at(&bytes, beside, IVec3::new(31, 1, 16)), 0x10);
    // Edges and corners of the shell stay dark.
    assert_eq!(byte_at(&bytes, beside, IVec3::new(31, 1, 31 + 1)), 0);
    assert_eq!(byte_at(&bytes, beside, IVec3::new(31, 32, 32)), 0);

    // The lamp's own chunk reads its neighbour's border the same way, here
    // the ground of the chunk below.
    let bytes = slot_bytes(&mut app, IVec3::ZERO);
    assert_eq!(byte_at(&bytes, IVec3::ZERO, IVec3::new(16, -1, 16)), 0);
    assert_eq!(byte_at(&bytes, IVec3::ZERO, lamp), 0xF0 | 15);
}

#[test]
fn when_slots_run_out_the_farthest_holder_falls_back_to_flat_light_and_a_warning_is_logged_once(
) {
    let _holding = Hold::exclusive();
    // One slot. The lamp in the far chunk takes it first, while the chunk at
    // the origin, where the camera is, waits.
    let near = IVec3::splat(16);
    let far = IVec3::new(80, 16, 16);
    let hold = Hold::new(IVec3::ZERO);
    let mut world = LampWorld {
        scene: Arc::new(move |p| {
            if p == near || p == far {
                WorldVoxel::Solid(LAMP)
            } else {
                WorldVoxel::Air
            }
        }),
        light: Some(LightPropagation {
            slot_capacity: 1,
            ..lit(LightShape::Diamond)
        }),
        hold: Some(hold.clone()),
        ..default()
    };
    world.spawning_distance = 3;
    let mut app = app(world);
    let far_chunk = IVec3::new(2, 0, 0);
    run_until_shown(&mut app, &[far_chunk]);
    assert!(has_slot(&mut app, far_chunk));
    assert!(!app
        .world()
        .resource::<LightRoute<LampWorld>>()
        .warned_full());

    hold.release();
    run_until_shown(&mut app, &[IVec3::ZERO, far_chunk]);
    assert!(has_slot(&mut app, IVec3::ZERO));
    // The voxels touching the far lamp see 14 at its faces, 13 at its edges
    // and 12 at its corners, so 13 is the most common.
    assert_eq!(tag_of(&mut app, far_chunk), Some(13 << 4));
    assert!(app
        .world()
        .resource::<LightRoute<LampWorld>>()
        .warned_full());
}

fn set_voxel(app: &mut App, position: IVec3, voxel: WorldVoxel) {
    app.world_mut()
        .run_system_once(move |mut voxel_world: VoxelWorld<LampWorld>| {
            voxel_world.set_voxel(position, voxel)
        })
        .unwrap();
}
