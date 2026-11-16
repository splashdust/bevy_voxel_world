//! Tests of a sub-grid chunk's tag and slot: bit 30 marks it, its own light
//! takes a slot under the world's slot rule and budget, and its shell holds
//! its own grid's light.

use std::sync::Arc;
use std::time::{Duration, Instant};

use bevy::ecs::system::RunSystemOnce;
use bevy::mesh::MeshTag;
use bevy::prelude::*;

use crate::light::route::{LightRoute, HAS_SLOT};
use crate::prelude::*;
use crate::test::light::edits::settle;
use crate::test::light::reference::Reference;
use crate::test::light::voxel_light::{app, lit, LampWorld, LAMP, STONE};

/// The tag bit of a sub-grid's chunk.
const SUB_GRID: u32 = 1 << 30;

fn light_is_settled(app: &mut App) -> bool {
    app.world_mut()
        .run_system_once(|voxel_world: VoxelWorld<LampWorld>| {
            voxel_world.light_is_settled()
        })
        .unwrap()
}

/// Spawns a sub-grid at `transform` holding each of `voxels`.
fn spawn_holding(
    app: &mut App,
    transform: Transform,
    voxels: &[(IVec3, WorldVoxel)],
) -> Entity {
    let voxels = voxels.to_vec();
    app.world_mut()
        .run_system_once(move |mut voxel_world: VoxelWorld<LampWorld>| {
            let entity = voxel_world.spawn_sub_grid(transform);
            let mut grid = voxel_world.grid_mut(Grid::SubGrid(entity)).unwrap();
            for (position, voxel) in &voxels {
                grid.set_voxel(*position, *voxel);
            }
            entity
        })
        .unwrap()
}

/// Sets `voxel` at `position` in the sub-grid `entity`.
fn set_in(app: &mut App, entity: Entity, position: IVec3, voxel: WorldVoxel) {
    app.world_mut()
        .run_system_once(move |mut voxel_world: VoxelWorld<LampWorld>| {
            let mut grid = voxel_world.grid_mut(Grid::SubGrid(entity)).unwrap();
            grid.set_voxel(position, voxel);
        })
        .unwrap();
}

/// Each chunk showing its mesh, with its grid's entity (`None` for the
/// world), its chunk position on that grid, and its tag.
fn tags(app: &mut App) -> Vec<(Option<Entity>, IVec3, u32)> {
    app.world_mut()
        .query::<(&Chunk<LampWorld>, &MeshTag, Option<&ChildOf>)>()
        .iter(app.world())
        .map(|(chunk, tag, child_of)| {
            let key = chunk.grid_key();
            let grid = (!key.is_world()).then(|| child_of.unwrap().parent());
            (grid, key.position(), tag.value)
        })
        .collect()
}

/// The tag of the chunk at `position` on the grid of `grid`, if it shows its
/// mesh.
fn tag_of(app: &mut App, grid: Option<Entity>, position: IVec3) -> Option<u32> {
    tags(app)
        .into_iter()
        .find(|(on, at, _)| *on == grid && *at == position)
        .map(|(_, _, tag)| tag)
}

/// Runs `app` until, for three frames in a row, light has settled and every
/// chunk of the sub-grid `entity` at `positions` shows its mesh.
fn settle_shown(app: &mut App, entity: Entity, positions: &[IVec3]) {
    let deadline = Instant::now() + Duration::from_secs(60);
    let mut frames_done = 0;
    while frames_done < 3 {
        assert!(
            Instant::now() < deadline,
            "timed out waiting for light to settle and {positions:?} to show"
        );
        app.update();
        let shown = positions
            .iter()
            .all(|position| tag_of(app, Some(entity), *position).is_some());
        frames_done = match light_is_settled(app) && shown {
            true => frames_done + 1,
            false => 0,
        };
    }
}

/// The bytes queued for `slot`.
fn queued(app: &App, slot: u32) -> Vec<u8> {
    app.world()
        .resource::<LightRoute<LampWorld>>()
        .queued_upload(slot)
        .expect("the slot's bytes are queued")
        .to_vec()
}

/// The byte of a slot at its cell `cell`, in the chunk's padded local
/// coordinates.
fn byte_at(bytes: &[u8], cell: IVec3) -> u8 {
    bytes[(cell.z + cell.x * 34 + cell.y * 34 * 34) as usize]
}

/// A world of air with lamps at `lamps`, lit at `precision`, its sky as
/// `sky` says.
fn world_of(lamps: Vec<IVec3>, precision: LightPrecision, sky: Sky) -> LampWorld {
    LampWorld {
        scene: Arc::new(move |position| match lamps.contains(&position) {
            true => WorldVoxel::Solid(LAMP),
            false => WorldVoxel::Air,
        }),
        light: Some(LightPropagation {
            precision,
            sky,
            ..lit(LightShape::Round)
        }),
        ..default()
    }
}

#[test]
fn bit_30_marks_every_sub_grid_chunks_tag_and_no_world_chunks() {
    for precision in [LightPrecision::Steps16, LightPrecision::Steps256] {
        // The world's lamp gives the chunk at the origin a slot, and lights
        // nothing in the chunks around it, so a stone in the chunk above shows
        // a mesh with no slot.
        let mut world = world_of(vec![IVec3::splat(16)], precision, Sky::Off);
        let lamps = world.scene.clone();
        world.scene =
            Arc::new(move |position| match position == IVec3::new(16, 56, 16) {
                true => WorldVoxel::Solid(STONE),
                false => lamps(position),
            });
        let mut app = app(world);
        settle(&mut app, "the world to settle");
        // The lamp gives the sub-grid's chunk at the origin a slot, and the
        // stone three chunks along z is a chunk of its own, far from it.
        let entity = spawn_holding(
            &mut app,
            Transform::from_xyz(3.0, 0.0, 0.0),
            &[
                (IVec3::splat(16), WorldVoxel::Solid(LAMP)),
                (IVec3::new(16, 16, 112), WorldVoxel::Solid(STONE)),
            ],
        );
        settle_shown(&mut app, entity, &[IVec3::ZERO, IVec3::new(0, 0, 3)]);

        let tags = tags(&mut app);
        let mut seen = [[false; 2]; 2];
        for (grid, position, tag) in tags {
            let sub_grid = grid.is_some();
            assert_eq!(
                tag & SUB_GRID != 0,
                sub_grid,
                "{position} on {grid:?}: {tag:#x} at {precision:?}"
            );
            seen[sub_grid as usize][(tag & HAS_SLOT != 0) as usize] = true;
        }
        assert_eq!(
            seen, [[true; 2]; 2],
            "with slots and without at {precision:?}"
        );
    }
}

#[test]
fn a_sub_grid_chunk_far_from_its_lamps_carries_its_levels_in_its_tag() {
    for (precision, levels) in [
        (LightPrecision::Steps16, 15),
        (LightPrecision::Steps256, 255),
    ] {
        let mut app = app(world_of(Vec::new(), precision, Sky::default()));
        settle(&mut app, "the world to settle");
        // The stone's chunk has no chunk of its own grid beside it, so no
        // face reads a neighbour.
        let stone = IVec3::new(0, 0, 3);
        let entity = spawn_holding(
            &mut app,
            Transform::default(),
            &[
                (IVec3::splat(16), WorldVoxel::Solid(LAMP)),
                (IVec3::new(16, 16, 112), WorldVoxel::Solid(STONE)),
            ],
        );
        settle_shown(&mut app, entity, &[IVec3::ZERO, stone]);

        // Sky 15 and voxel 0.
        assert_eq!(
            tag_of(&mut app, Some(entity), stone),
            Some(SUB_GRID | levels),
            "at {precision:?}"
        );
        let lit = tag_of(&mut app, Some(entity), IVec3::ZERO).unwrap();
        assert_eq!(lit & (HAS_SLOT | SUB_GRID), HAS_SLOT | SUB_GRID, "{lit:#x}");
    }
}

/// The reference voxel light of a lamp at `lamp` alone in air.
fn lone_lamp(lamp: IVec3) -> Reference {
    Reference::voxel_light(
        lamp - 31,
        lamp + 32,
        14,
        LightShape::Round,
        |p| match p == lamp {
            true => WorldVoxel::Solid(LAMP),
            false => WorldVoxel::Air,
        },
        |material| (material == LAMP).then_some(15.0),
    )
}

#[test]
fn a_sub_grid_chunks_shell_holds_its_own_grids_neighbours_and_nothing_else() {
    // World lamps light the world's chunks on both sides of the origin's,
    // up to their borders with it.
    let mut app = app(world_of(
        vec![IVec3::new(-3, 16, 16), IVec3::new(36, 16, 16)],
        LightPrecision::Steps16,
        Sky::Off,
    ));
    settle(&mut app, "the world to settle");
    // The lamp beside the border between the sub-grid's chunks at x = 0 and
    // x = 1 gives it those two chunks alone.
    let lamp = IVec3::new(31, 16, 16);
    let ours = spawn_holding(
        &mut app,
        Transform::default(),
        &[(lamp, WorldVoxel::Solid(LAMP))],
    );
    // Another sub-grid in the same place has chunks at x = -1 and x = 1,
    // each lit up to its border with the chunk at the origin.
    let theirs = spawn_holding(
        &mut app,
        Transform::default(),
        &[
            (IVec3::new(-2, 16, 16), WorldVoxel::Solid(LAMP)),
            (IVec3::new(40, 16, 16), WorldVoxel::Solid(LAMP)),
        ],
    );
    // Our chunk at x = 1 holds no face of its own, so it shows no mesh, but
    // its light lands.
    settle_shown(&mut app, ours, &[IVec3::ZERO]);
    settle_shown(&mut app, theirs, &[IVec3::NEG_X, IVec3::X]);

    let tag = tag_of(&mut app, Some(ours), IVec3::ZERO).unwrap();
    assert_ne!(tag & HAS_SLOT, 0, "{tag:#x}");
    let bytes = queued(&app, tag & 0xFF_FFFF);
    let reference = lone_lamp(lamp);
    let mut lit_cells = 0;
    for y in 1..33 {
        for z in 1..33 {
            // Toward x = 1, the plane of our chunk there, as its own light
            // falls off from the lamp.
            let level =
                reference.read(reference.voxel_at(IVec3::new(32, y - 1, z - 1))) as u8;
            lit_cells += (level > 0) as usize;
            assert_eq!(
                byte_at(&bytes, IVec3::new(33, y, z)),
                level << 4,
                "toward x = 1 at y {y}, z {z}"
            );
            // Toward x = -1, where we have no chunk, nothing.
            assert_eq!(
                byte_at(&bytes, IVec3::new(0, y, z)),
                0,
                "toward x = -1 at y {y}, z {z}"
            );
        }
    }
    assert!(lit_cells > 0);

    // The world's and the other sub-grid's planes at those faces aren't dark,
    // so a shell copying either would show it.
    let world_beside = app
        .world_mut()
        .run_system_once(|voxel_world: VoxelWorld<LampWorld>| {
            [IVec3::new(-1, 16, 16), IVec3::new(32, 16, 16)]
                .map(|p| voxel_world.light_at(p).unwrap().voxel)
        })
        .unwrap();
    assert!(
        world_beside.iter().all(|level| *level > 0.0),
        "{world_beside:?}"
    );
    for (position, x) in [(IVec3::NEG_X, 31), (IVec3::X, 0)] {
        let tag = tag_of(&mut app, Some(theirs), position).unwrap();
        assert_ne!(tag & HAS_SLOT, 0, "{tag:#x}");
        let bytes = queued(&app, tag & 0xFF_FFFF);
        assert!(byte_at(&bytes, IVec3::new(x + 1, 17, 17)) >> 4 > 0);
    }
}

#[test]
fn a_sub_grid_moved_near_the_camera_takes_a_farther_world_chunks_slot() {
    // The world's lamp lights the chunk at x = 1 alone, which takes the one
    // slot. The camera is in the chunk at the origin.
    let world_lamp = IVec3::new(48, 16, 16);
    let mut world = world_of(vec![world_lamp], LightPrecision::Steps16, Sky::Off);
    world.light.as_mut().unwrap().slot_capacity = 1;
    let mut app = app(world);
    settle(&mut app, "the world to settle");
    assert_ne!(tag_of(&mut app, None, IVec3::X).unwrap() & HAS_SLOT, 0);

    // The sub-grid's lamp is ten chunks along x, far past the world's chunk,
    // so its chunk there waits for a slot.
    let lamp = IVec3::new(336, 16, 16);
    let chunk = IVec3::new(10, 0, 0);
    let entity = spawn_holding(
        &mut app,
        Transform::default(),
        &[(lamp, WorldVoxel::Solid(LAMP))],
    );
    settle_shown(&mut app, entity, &[chunk]);
    assert_eq!(tag_of(&mut app, Some(entity), chunk).unwrap() & HAS_SLOT, 0);
    assert_ne!(tag_of(&mut app, None, IVec3::X).unwrap() & HAS_SLOT, 0);

    // Moved so that the chunk's middle is at the camera, then relit by an
    // edit, it takes the world chunk's slot.
    app.world_mut()
        .entity_mut(entity)
        .insert(Transform::from_xyz(-326.0, -6.0, -6.0));
    settle_shown(&mut app, entity, &[chunk]);
    set_in(
        &mut app,
        entity,
        lamp + IVec3::new(4, 0, 0),
        WorldVoxel::Solid(STONE),
    );
    settle_shown(&mut app, entity, &[chunk]);
    let tag = tag_of(&mut app, Some(entity), chunk).unwrap();
    assert_ne!(tag & HAS_SLOT, 0, "{tag:#x}");
    assert_eq!(tag_of(&mut app, None, IVec3::X).unwrap() & HAS_SLOT, 0);
}
