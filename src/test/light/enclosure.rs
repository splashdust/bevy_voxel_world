//! Tests of a sub-grid's enclosure: the sky its faces show, capped where its
//! voxels close it in, read through `grid(g).light_at`.

use std::collections::HashSet;

use bevy::ecs::system::RunSystemOnce;
use bevy::prelude::*;

use super::edits::settle;
use super::reference::Reference;
use super::sky_light::{light_is_settled, sky_light};
use super::voxel_light::{app, LampWorld, LAMP, STONE};
use crate::prelude::*;

/// Runs `f` on the voxel world in a system of its own, whose commands apply
/// when it returns.
fn run<T: 'static>(
    app: &mut App,
    mut f: impl FnMut(&mut VoxelWorld<LampWorld>) -> T + Send + Sync + 'static,
) -> T {
    app.world_mut()
        .run_system_once(move |mut voxel_world: VoxelWorld<LampWorld>| {
            f(&mut voxel_world)
        })
        .unwrap()
}

/// Spawns a sub-grid at `transform` holding stone at each of `solids` and a
/// lamp at each of `lamps`.
fn spawn_holding(
    app: &mut App,
    transform: Transform,
    solids: &HashSet<IVec3>,
    lamps: &[IVec3],
) -> Entity {
    let solids: Vec<IVec3> = solids.iter().copied().collect();
    let lamps = lamps.to_vec();
    run(app, move |voxel_world| {
        let entity = voxel_world.spawn_sub_grid(transform);
        let mut grid = voxel_world.grid_mut(Grid::SubGrid(entity)).unwrap();
        for position in &solids {
            grid.set_voxel(*position, WorldVoxel::Solid(STONE));
        }
        for position in &lamps {
            grid.set_voxel(*position, WorldVoxel::Solid(LAMP));
        }
        entity
    })
}

/// Sets each of `voxels` in the sub-grid `entity`.
fn set_in(app: &mut App, entity: Entity, voxels: Vec<(IVec3, WorldVoxel)>) {
    run(app, move |voxel_world| {
        let mut grid = voxel_world.grid_mut(Grid::SubGrid(entity)).unwrap();
        for (position, voxel) in &voxels {
            grid.set_voxel(*position, *voxel);
        }
    });
}

/// What `grid(Grid::SubGrid(entity)).light_at` reads at each of
/// `positions`, in one system.
fn grid_light(app: &mut App, entity: Entity, positions: &[IVec3]) -> Vec<LightLevels> {
    let positions = positions.to_vec();
    run(app, move |voxel_world| {
        let grid = voxel_world.grid(Grid::SubGrid(entity)).unwrap();
        positions
            .iter()
            .map(|p| {
                grid.light_at(*p)
                    .unwrap_or_else(|| panic!("no light at {p}"))
            })
            .collect()
    })
}

/// The sky `grid_light` reads at each of `positions`.
fn grid_sky(app: &mut App, entity: Entity, positions: &[IVec3]) -> Vec<f32> {
    grid_light(app, entity, positions)
        .into_iter()
        .map(|levels| levels.sky)
        .collect()
}

/// Runs `app` until light has settled for 3 frames in a row. Unlike
/// [`settle`], it doesn't wait for a world chunk at each sub-grid chunk's
/// position.
fn settle_light(app: &mut App, what: &str) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
    let mut frames_done = 0;
    while frames_done < 3 {
        assert!(
            std::time::Instant::now() < deadline,
            "timed out waiting: {what}"
        );
        app.update();
        frames_done = if light_is_settled(app) {
            frames_done + 1
        } else {
            0
        };
    }
}

/// How many solves and relights the world has started.
fn solves_and_relights(app: &App) -> (u64, u64) {
    let counters = app
        .world()
        .resource::<crate::light::LightSettings<LampWorld>>()
        .counters();
    (
        counters.solves.load(std::sync::atomic::Ordering::Relaxed),
        counters.relights.load(std::sync::atomic::Ordering::Relaxed),
    )
}

/// The chunk positions of the sub-grid `entity`'s chunks.
fn chunks_of(app: &mut App, entity: Entity) -> HashSet<IVec3> {
    let mut chunks = app.world_mut().query::<(&Chunk<LampWorld>, &ChildOf)>();
    chunks
        .iter(app.world())
        .filter(|(_, parent)| parent.parent() == entity)
        .map(|(chunk, _)| chunk.position())
        .collect()
}

/// The voxels of the box from `min` to `max`, both inclusive.
fn voxels_in(min: IVec3, max: IVec3) -> Vec<IVec3> {
    (min.z..=max.z)
        .flat_map(|z| {
            (min.y..=max.y)
                .flat_map(move |y| (min.x..=max.x).map(move |x| IVec3::new(x, y, z)))
        })
        .collect()
}

/// The walls of the hollow box from `min` to `max`, both inclusive, one
/// voxel thick.
fn hollow(min: IVec3, max: IVec3) -> HashSet<IVec3> {
    voxels_in(min, max)
        .into_iter()
        .filter(|p| p.cmpeq(min).any() || p.cmpeq(max).any())
        .collect()
}

/// A cabin with walls from `min` to `max`, whose roof overhangs its walls by
/// two voxels across x, with a ledge under the eaves on the +X side, and a
/// beam across its inside below the roof.
fn cabin(min: IVec3, max: IVec3) -> HashSet<IVec3> {
    let mut solids = hollow(min, max);
    for x in min.x - 2..=max.x + 2 {
        for z in min.z..=max.z {
            solids.insert(IVec3::new(x, max.y, z));
        }
    }
    for z in min.z..=max.z {
        solids.insert(IVec3::new(max.x + 1, max.y - 3, z));
        solids.insert(IVec3::new((min.x + max.x) / 2, max.y - 2, z));
    }
    solids
}

/// The voxels strictly inside the walls from `min` to `max`.
fn inside(min: IVec3, max: IVec3) -> Vec<IVec3> {
    voxels_in(min + 1, max - 1)
}

/// Whether the open voxel `p` has direct sky among `solids`: whether any of
/// the six axis lines from it meets none of them. Worked out voxel by voxel.
fn has_direct_sky(solids: &HashSet<IVec3>, p: IVec3) -> bool {
    let (low, high) = solids
        .iter()
        .fold((IVec3::MAX, IVec3::MIN), |(low, high), s| {
            (low.min(*s), high.max(*s))
        });
    [
        IVec3::X,
        IVec3::NEG_X,
        IVec3::Y,
        IVec3::NEG_Y,
        IVec3::Z,
        IVec3::NEG_Z,
    ]
    .into_iter()
    .any(|step| {
        let mut q = p + step;
        while q.cmpge(low).all() && q.cmple(high).all() {
            if solids.contains(&q) {
                return false;
            }
            q += step;
        }
        true
    })
}

/// A world of air with sky light on, so the world's sky is 15 everywhere,
/// with sky light reaching `sky_reach`.
fn open_sky(sky_reach: u8) -> LampWorld {
    LampWorld {
        light: Some(sky_light(Sky::default(), sky_reach)),
        ..default()
    }
}

/// Sets world `C`'s `sub_grid_sky` to `sky`.
fn set_sub_grid_sky(app: &mut App, sky: SubGridSky) {
    app.world_mut()
        .run_system_once(move |mut world: ResMut<LampWorld>| {
            world.light.as_mut().unwrap().sub_grid_sky = sky;
        })
        .unwrap();
}

#[test]
fn enclosed_is_the_default() {
    assert_eq!(
        LightPropagation::default().sub_grid_sky,
        SubGridSky::Enclosed
    );
}

#[test]
fn a_closed_cabin_is_dark_inside_under_enclosed_and_reads_the_worlds_sky_under_world() {
    for sub_grid_sky in [SubGridSky::Enclosed, SubGridSky::World] {
        let mut world = open_sky(12);
        world.light.as_mut().unwrap().sub_grid_sky = sub_grid_sky;
        let mut app = app(world);
        settle(&mut app, "the world to settle");
        let (min, max) = (IVec3::new(0, 0, 0), IVec3::new(12, 9, 10));
        let lamp = IVec3::new(3, 1, 3);
        let entity = spawn_holding(
            &mut app,
            Transform::from_xyz(-6.0, 4.0, -5.0),
            &cabin(min, max),
            &[lamp],
        );
        settle_light(&mut app, "the cabin's light to settle");

        let positions: Vec<IVec3> = inside(min, max)
            .into_iter()
            .filter(|p| *p != lamp && p.y != max.y - 2)
            .collect();
        let read = grid_light(&mut app, entity, &positions);
        let sky = match sub_grid_sky {
            SubGridSky::Enclosed => 0.0,
            SubGridSky::World => 15.0,
        };
        for (p, levels) in positions.iter().zip(&read) {
            assert_eq!(levels.sky, sky, "sky at {p} under {sub_grid_sky:?}");
        }
        let beside = grid_light(&mut app, entity, &[lamp + IVec3::X]);
        assert_eq!(beside[0].voxel, 14.0, "the lamp under {sub_grid_sky:?}");
        // Outside, the world's sky shows under both.
        let out = grid_sky(&mut app, entity, &[max + IVec3::new(4, 2, 0)]);
        assert_eq!(out, vec![15.0]);
    }
}

/// A room 40 long and wide with a doorway in its +X wall, a window in its -X
/// wall and a hole in its roof, none near its corner at the origin.
fn doorway_room() -> (IVec3, IVec3, HashSet<IVec3>) {
    let (min, max) = (IVec3::new(0, 0, 0), IVec3::new(41, 11, 41));
    let mut solids = cabin(min, max);
    for y in 1..=4 {
        for z in 18..=21 {
            solids.remove(&IVec3::new(max.x, y, z));
        }
    }
    solids.remove(&IVec3::new(min.x, 6, 34));
    solids.remove(&IVec3::new(30, max.y, 30));
    (min, max, solids)
}

#[test]
fn sky_falls_away_inside_a_doorway_the_way_the_worlds_sky_does() {
    let sky_reach = 12;
    let mut app = app(open_sky(sky_reach));
    settle(&mut app, "the world to settle");
    let (min, max, solids) = doorway_room();
    let entity = spawn_holding(
        &mut app,
        Transform::from_xyz(-20.0, 0.0, -20.0),
        &solids,
        &[],
    );
    settle_light(&mut app, "the room's light to settle");

    let margin = IVec3::splat(sky_reach as i32 + 2);
    let reference = Reference::sky_from(
        min - margin,
        max + margin + 1,
        sky_reach as u32,
        LightShape::Round,
        |p| {
            if solids.contains(&p) {
                WorldVoxel::Solid(STONE)
            } else {
                WorldVoxel::Air
            }
        },
        |p| has_direct_sky(&solids, p),
    );
    let positions = inside(min, max);
    let read = grid_sky(&mut app, entity, &positions);
    let (mut direct, mut dark) = (0, 0);
    for (p, sky) in positions.iter().zip(read) {
        let expected = if solids.contains(p) {
            sky
        } else {
            reference.read(reference.sky_at(*p))
        };
        assert_eq!(sky, expected, "sky at {p}");
        if !solids.contains(p) && has_direct_sky(&solids, *p) {
            direct += 1;
            assert_eq!(sky, 15.0, "direct sky at {p}");
        }
        if expected == 0.0 && !solids.contains(p) {
            dark += 1;
        }
    }
    assert!(direct > 0, "the doorway lets direct sky in");
    assert!(dark > 0, "the corner is beyond sky reach of the openings");
    let corner = grid_sky(&mut app, entity, &[IVec3::new(2, 2, 2)]);
    assert_eq!(corner, vec![0.0]);
}

#[test]
fn turning_a_cabin_any_way_up_changes_nothing_inside_and_solves_nothing() {
    let mut app = app(open_sky(12));
    settle(&mut app, "the world to settle");
    let (min, max) = (IVec3::ZERO, IVec3::new(16, 9, 12));
    let mut solids = cabin(min, max);
    for y in 1..=5 {
        solids.remove(&IVec3::new(max.x, y, 6));
        solids.remove(&IVec3::new(max.x, y, 7));
    }
    solids.remove(&IVec3::new(5, 5, min.z));
    let centre = Vec3::new(8.0, 5.0, 6.0);
    let entity =
        spawn_holding(&mut app, Transform::from_translation(-centre), &solids, &[]);
    settle_light(&mut app, "the cabin's light to settle");
    let positions: Vec<IVec3> = inside(min, max)
        .into_iter()
        .filter(|p| !solids.contains(p))
        .collect();
    let before = grid_sky(&mut app, entity, &positions);
    assert!(before.contains(&15.0), "the doorway lets direct sky in");
    assert!(
        before.iter().any(|sky| *sky < 15.0),
        "sky falls away inside"
    );
    let counted = solves_and_relights(&app);

    let turns = [
        Quat::from_rotation_y(std::f32::consts::FRAC_PI_2),
        Quat::from_rotation_x(std::f32::consts::PI),
        Quat::from_rotation_z(std::f32::consts::FRAC_PI_2),
        Quat::from_euler(EulerRot::XYZ, 0.4, 1.1, -2.3),
    ];
    for turn in turns {
        // Turned about its middle, so it stays in the open sky.
        let pose = Transform::from_rotation(turn) * Transform::from_translation(-centre);
        app.world_mut().entity_mut(entity).insert(pose);
        for _ in 0..3 {
            app.update();
        }
        assert!(light_is_settled(&mut app));
        assert_eq!(grid_sky(&mut app, entity, &positions), before, "{turn}");
    }
    assert_eq!(solves_and_relights(&app), counted);
}

#[test]
fn a_hull_longer_than_a_solve_box_and_a_hangar_round_a_chunk_of_air_read_enclosed() {
    let mut app = app(open_sky(12));
    settle(&mut app, "the world to settle");
    // Along x across five chunks, far longer than a solve's box.
    let (min, max) = (IVec3::new(-60, 0, 0), IVec3::new(90, 9, 9));
    let hull = spawn_holding(
        &mut app,
        Transform::from_xyz(0.0, 10.0, 0.0),
        &cabin(min, max),
        &[],
    );
    // So wide inside that no solid lies in the padding of its chunk at
    // (1, 1, 1), where it has no chunk.
    let (hangar_min, hangar_max) = (IVec3::splat(-1), IVec3::splat(96));
    let hangar = spawn_holding(
        &mut app,
        Transform::from_xyz(-33.0, -33.0, -33.0),
        &hollow(hangar_min, hangar_max),
        &[],
    );
    settle_light(&mut app, "the hull's and hangar's light to settle");
    assert!(!chunks_of(&mut app, hangar).contains(&IVec3::ONE));

    let middle: Vec<IVec3> = voxels_in(IVec3::new(0, 1, 1), IVec3::new(30, 7, 8))
        .into_iter()
        .filter(|p| p.y != max.y - 2)
        .collect();
    assert!(grid_sky(&mut app, hull, &middle)
        .iter()
        .all(|sky| *sky == 0.0));
    let hollow_chunk = voxels_in(IVec3::splat(32), IVec3::splat(63));
    assert!(grid_sky(&mut app, hangar, &hollow_chunk)
        .iter()
        .all(|sky| *sky == 0.0));
    // Beside the -Z wall, in the sub-grid's chunk at (1, 1, 0), whose lines
    // along z cross the chunk it has none at.
    let near_wall = voxels_in(IVec3::new(40, 40, 0), IVec3::new(60, 60, 3));
    assert!(grid_sky(&mut app, hangar, &near_wall)
        .iter()
        .all(|sky| *sky == 0.0));
}

#[test]
fn a_hole_in_a_long_hulls_end_lights_its_line_far_past_the_edits_reach() {
    let mut app = app(open_sky(12));
    settle(&mut app, "the world to settle");
    let (min, max) = (IVec3::new(-60, 0, 0), IVec3::new(90, 9, 9));
    let mut solids = cabin(min, max);
    // A shed on the roof, closed but for a window, which encloses its
    // inside on every axis.
    let (shed_min, shed_max) = (IVec3::new(20, max.y, 0), IVec3::new(30, 16, 9));
    solids.extend(hollow(shed_min, shed_max));
    solids.remove(&IVec3::new(22, 13, shed_max.z));
    let entity =
        spawn_holding(&mut app, Transform::from_xyz(0.0, 10.0, 0.0), &solids, &[]);
    settle_light(&mut app, "the hull's light to settle");

    let on_line = IVec3::new(60, 4, 4);
    let beside_line = IVec3::new(60, 4, 6);
    assert_eq!(
        grid_sky(&mut app, entity, &[on_line, beside_line]),
        vec![0.0, 0.0]
    );

    set_in(
        &mut app,
        entity,
        vec![(IVec3::new(min.x, 4, 4), WorldVoxel::Air)],
    );
    assert!(!light_is_settled(&mut app));
    settle_light(&mut app, "the hole's light to settle");
    let read = grid_sky(&mut app, entity, &[on_line, beside_line]);
    assert_eq!(read[0], 15.0, "direct sky along the hole's line");
    assert!(read[1] > 0.0 && read[1] < 15.0, "sky spreads from the line");

    // Every line through the shed's inside was already blocked.
    let block = IVec3::new(25, 12, 4);
    let reach = 14;
    let within: usize = chunks_of(&mut app, entity)
        .into_iter()
        .filter(|chunk| {
            let low = *chunk * 32 - reach;
            let high = *chunk * 32 + 31 + reach;
            block.cmpge(low).all() && block.cmple(high).all()
        })
        .count();
    let (_, relights) = solves_and_relights(&app);
    set_in(&mut app, entity, vec![(block, WorldVoxel::Solid(STONE))]);
    settle_light(&mut app, "the block's light to settle");
    let (_, after) = solves_and_relights(&app);
    assert_eq!(after - relights, within as u64);
}

#[test]
fn switching_sub_grid_sky_darkens_and_brightens_a_cabin_and_relights_no_world_chunk() {
    let mut world = open_sky(12);
    world.light.as_mut().unwrap().sub_grid_sky = SubGridSky::World;
    let mut app = app(world);
    settle(&mut app, "the world to settle");
    let (min, max) = (IVec3::ZERO, IVec3::new(12, 9, 10));
    let entity = spawn_holding(&mut app, Transform::default(), &cabin(min, max), &[]);
    settle_light(&mut app, "the cabin's light to settle");
    let positions: Vec<IVec3> = inside(min, max)
        .into_iter()
        .filter(|p| p.y != max.y - 2)
        .collect();
    let chunks = chunks_of(&mut app, entity).len() as u64;
    assert!(grid_sky(&mut app, entity, &positions)
        .iter()
        .all(|sky| *sky == 15.0));

    for (sky, expected) in [(SubGridSky::Enclosed, 0.0), (SubGridSky::World, 15.0)] {
        let (_, relights) = solves_and_relights(&app);
        set_sub_grid_sky(&mut app, sky);
        app.update();
        assert!(!light_is_settled(&mut app), "after switching to {sky:?}");
        settle_light(&mut app, "light to settle after the switch");
        let (_, after) = solves_and_relights(&app);
        assert_eq!(after - relights, chunks, "only the cabin's chunks relight");
        assert!(
            grid_sky(&mut app, entity, &positions)
                .iter()
                .all(|read| *read == expected),
            "inside under {sky:?}"
        );
    }
}

#[test]
fn with_sky_light_off_a_cabin_reads_no_sky_inside_or_out() {
    let mut world = open_sky(12);
    world.light.as_mut().unwrap().sky = Sky::Off;
    let mut app = app(world);
    settle(&mut app, "the world to settle");
    let (min, max, solids) = doorway_room();
    let entity = spawn_holding(
        &mut app,
        Transform::from_xyz(-20.0, 0.0, -20.0),
        &solids,
        &[],
    );
    settle_light(&mut app, "the room's light to settle");
    let mut positions = inside(min, max);
    positions.push(max + IVec3::new(3, 1, 0));
    positions.push(IVec3::new(max.x - 1, 2, 19));
    assert!(grid_sky(&mut app, entity, &positions)
        .iter()
        .all(|sky| *sky == 0.0));
}
