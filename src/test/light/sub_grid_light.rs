//! Tests of a sub-grid's own light: its lamps light it, read through
//! `grid(g).light_at`, on top of the world's light at its pose.

use bevy::ecs::system::RunSystemOnce;
use bevy::prelude::*;

use super::edits::settle;
use super::reference::Reference;
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

/// Spawns a sub-grid at `transform` holding each of `voxels`.
fn spawn_holding(
    app: &mut App,
    transform: Transform,
    voxels: &[(IVec3, WorldVoxel)],
) -> Entity {
    let voxels = voxels.to_vec();
    run(app, move |voxel_world| {
        let entity = voxel_world.spawn_sub_grid(transform);
        let mut grid = voxel_world.grid_mut(Grid::SubGrid(entity)).unwrap();
        for (position, voxel) in &voxels {
            grid.set_voxel(*position, *voxel);
        }
        entity
    })
}

/// What `grid(Grid::SubGrid(entity)).light_at` reads at each of
/// `positions`, in one system.
fn grid_light(
    app: &mut App,
    entity: Entity,
    positions: Vec<IVec3>,
) -> Vec<Option<LightLevels>> {
    run(app, move |voxel_world| {
        let grid = voxel_world.grid(Grid::SubGrid(entity)).unwrap();
        positions.iter().map(|p| grid.light_at(*p)).collect()
    })
}

/// The voxels of the box from `min` to `max`, exclusive.
fn voxels_in(min: IVec3, max: IVec3) -> Vec<IVec3> {
    (min.z..max.z)
        .flat_map(|z| {
            (min.y..max.y)
                .flat_map(move |y| (min.x..max.x).map(move |x| IVec3::new(x, y, z)))
        })
        .collect()
}

/// The reference voxel light of a lamp at `lamp` alone in air, at the
/// world's default reach and the round shape.
fn lone_lamp(lamp: IVec3) -> Reference {
    Reference::voxel_light(
        lamp - 31,
        lamp + 32,
        14,
        LightShape::Round,
        |p| {
            if p == lamp {
                WorldVoxel::Solid(LAMP)
            } else {
                WorldVoxel::Air
            }
        },
        |material| (material == LAMP).then_some(15.0),
    )
}

#[test]
fn a_sub_grids_lamp_lights_it_across_a_border_between_its_chunks() {
    let mut app = app(LampWorld::default());
    settle(&mut app, "the world to settle");
    // Beside the border between the sub-grid's chunks at x = 0 and x = 1.
    let lamp = IVec3::new(31, 16, 16);
    let entity = spawn_holding(
        &mut app,
        Transform::default(),
        &[(lamp, WorldVoxel::Solid(LAMP))],
    );
    settle(&mut app, "the sub-grid's light to settle");

    let reference = lone_lamp(lamp);
    let positions = voxels_in(lamp - 15, lamp + 16);
    let read = grid_light(&mut app, entity, positions.clone());
    for (p, levels) in positions.into_iter().zip(read) {
        let levels = levels.unwrap_or_else(|| panic!("no light at {p}"));
        assert_eq!(levels.sky, 0.0, "at {p}");
        assert_eq!(
            levels.voxel,
            reference.read(reference.voxel_at(p)),
            "at {p}"
        );
    }
}

#[test]
fn a_lamps_light_reaches_a_sub_grid_chunk_through_a_chunk_position_it_lacks() {
    let mut app = app(LampWorld::default());
    settle(&mut app, "the world to settle");
    // The lamp needs only the chunk at the origin, and the stone only the
    // chunk at (1, 1, 0), which touches it along an edge. Light gets from
    // one to the other only through (1, 0, 0) or (0, 1, 0), where the
    // sub-grid has no chunk.
    let lamp = IVec3::new(30, 30, 16);
    let stone = IVec3::new(33, 33, 16);
    let entity = spawn_holding(
        &mut app,
        Transform::default(),
        &[
            (lamp, WorldVoxel::Solid(LAMP)),
            (stone, WorldVoxel::Solid(STONE)),
        ],
    );
    settle(&mut app, "the sub-grid's light to settle");

    let reference = Reference::voxel_light(
        lamp - 31,
        lamp + 32,
        14,
        LightShape::Round,
        |p| match p {
            p if p == lamp => WorldVoxel::Solid(LAMP),
            p if p == stone => WorldVoxel::Solid(STONE),
            _ => WorldVoxel::Air,
        },
        |material| (material == LAMP).then_some(15.0),
    );
    let positions: Vec<IVec3> = voxels_in(IVec3::new(32, 32, 14), IVec3::new(36, 36, 19))
        .into_iter()
        .filter(|p| *p != stone)
        .collect();
    let read = grid_light(&mut app, entity, positions.clone());
    for (p, levels) in positions.into_iter().zip(read) {
        let expected = reference.read(reference.voxel_at(p));
        assert!(expected > 0.0, "the lamp reaches {p}");
        assert_eq!(levels.map(|levels| levels.voxel), Some(expected), "at {p}");
    }
}

/// The voxel light `reference` holds at `p`, on the 0 to 15 scale.
fn own_light(reference: &Reference, p: IVec3) -> f32 {
    reference.read(reference.voxel_at(p))
}

#[test]
fn a_turned_sub_grid_reads_the_worlds_light_where_it_is_with_its_lamps_on_top() {
    use super::edits::{cavern, world};

    let mut app = app(world());
    settle(&mut app, "the world to settle");
    // A quarter turn about y takes the sub-grid's x to the world's -z. The
    // voxel at the sub-grid's origin lands at (27, 23, 6), in the sealed
    // pocket of air in the cavern's roof, which no light reaches.
    let pose = Transform::from_xyz(27.0, 23.0, 7.0)
        .with_rotation(Quat::from_rotation_y(std::f32::consts::FRAC_PI_2));
    let lamp = IVec3::new(2, 2, 2);
    let entity = spawn_holding(&mut app, pose, &[(lamp, WorldVoxel::Solid(LAMP))]);
    settle(&mut app, "the sub-grid's light to settle");

    let reference = lone_lamp(lamp);
    let positions = voxels_in(IVec3::ZERO, IVec3::splat(8));
    let (read, world_read, world_forwarded) = run(&mut app, {
        let positions = positions.clone();
        move |voxel_world| {
            let grid = voxel_world.grid(Grid::SubGrid(entity)).unwrap();
            let pose = grid.pose();
            let world = voxel_world.grid(Grid::World).unwrap();
            let mut read = Vec::new();
            let mut world_read = Vec::new();
            let mut forwarded = Vec::new();
            for p in &positions {
                let at = pose.transform_point(p.as_vec3() + 0.5).floor().as_ivec3();
                read.push(grid.light_at(*p));
                world_read.push((at, voxel_world.light_at(at)));
                forwarded.push(world.light_at(at) == voxel_world.light_at(at));
            }
            (read, world_read, forwarded)
        }
    });
    assert!(world_forwarded.into_iter().all(|same| same));
    let mut in_pocket = 0;
    for ((p, levels), (at, world)) in positions.into_iter().zip(read).zip(world_read) {
        let world = world.unwrap_or_else(|| panic!("no world light at {at}"));
        let levels = levels.unwrap_or_else(|| panic!("no light at {p}"));
        assert_eq!(levels.sky, world.sky, "sky at {p}, over {at}");
        assert_eq!(
            levels.voxel,
            world.voxel.max(own_light(&reference, p)),
            "voxel light at {p}, over {at}"
        );
        if cavern(at) == WorldVoxel::Air {
            in_pocket += 1;
            assert_eq!(levels.sky, 0.0, "dark in the pocket at {p}, over {at}");
        }
    }
    assert!(in_pocket > 0, "some of the sub-grid is in the pocket");

    // Out in the open sky above the world's stone.
    app.world_mut()
        .entity_mut(entity)
        .insert(pose.with_translation(Vec3::new(27.0, 50.0, 7.0)));
    app.update();
    let read = grid_light(&mut app, entity, vec![lamp + IVec3::Y]);
    let levels = read[0].expect("light in the open");
    assert_eq!(levels.sky, 15.0);
    assert_eq!(levels.voxel, own_light(&reference, lamp + IVec3::Y));
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

#[test]
fn moving_and_turning_a_lit_sub_grid_solves_nothing() {
    let mut app = app(LampWorld::default());
    settle(&mut app, "the world to settle");
    let lamp = IVec3::new(31, 16, 16);
    let entity = spawn_holding(
        &mut app,
        Transform::default(),
        &[
            (lamp, WorldVoxel::Solid(LAMP)),
            (IVec3::new(40, 16, 16), WorldVoxel::Solid(STONE)),
        ],
    );
    settle(&mut app, "the sub-grid's light to settle");
    let before = solves_and_relights(&app);

    for frame in 0..40 {
        let t = frame as f32 * 0.1;
        app.world_mut().entity_mut(entity).insert(
            Transform::from_xyz(t * 3.0, -t, t * 2.0).with_rotation(Quat::from_euler(
                EulerRot::XYZ,
                t,
                t * 0.5,
                -t,
            )),
        );
        app.update();
        assert!(
            super::sky_light::light_is_settled(&mut app),
            "settled on frame {frame}"
        );
    }
    assert_eq!(solves_and_relights(&app), before);
}

#[test]
fn an_edit_at_one_end_of_a_long_sub_grid_relights_only_the_chunks_within_reach() {
    let mut app = app(LampWorld::default());
    settle(&mut app, "the world to settle");
    // A bar across five chunks, from x = -2 to 2, where the world has
    // chunks too, with a lamp near its far end.
    let mut voxels: Vec<(IVec3, WorldVoxel)> = (-60..68)
        .map(|x| (IVec3::new(x, 16, 16), WorldVoxel::Solid(STONE)))
        .collect();
    let far_lamp = IVec3::new(60, 17, 16);
    voxels.push((far_lamp, WorldVoxel::Solid(LAMP)));
    let entity = spawn_holding(&mut app, Transform::default(), &voxels);
    settle(&mut app, "the sub-grid's light to settle");
    let far: Vec<IVec3> = voxels_in(IVec3::new(-32, 10, 10), IVec3::new(68, 24, 24));
    let far_before = grid_light(&mut app, entity, far.clone());
    let (_, relights) = solves_and_relights(&app);

    // In the middle of the bar's first chunk, more than the reach of 14
    // from its faces: only that chunk's solve box holds it.
    let lamp = IVec3::new(-48, 17, 16);
    run(&mut app, move |voxel_world| {
        let mut grid = voxel_world.grid_mut(Grid::SubGrid(entity)).unwrap();
        grid.set_voxel(lamp, WorldVoxel::Solid(LAMP));
    });
    settle(&mut app, "the edit's light to settle");

    let (_, relights_after) = solves_and_relights(&app);
    assert_eq!(
        relights_after - relights,
        1,
        "only the edited chunk relights"
    );
    let beside = grid_light(&mut app, entity, vec![lamp + IVec3::Y]);
    assert_eq!(beside[0].map(|levels| levels.voxel), Some(14.0));
    assert_eq!(grid_light(&mut app, entity, far), far_before);
}

#[test]
fn light_is_settled_waits_for_a_sub_grids_light() {
    use super::sky_light::light_is_settled;

    let mut app = app(LampWorld::default());
    settle(&mut app, "the world to settle");
    let lamp = IVec3::new(31, 16, 16);
    let lit = |levels: Vec<Option<LightLevels>>| {
        levels
            .into_iter()
            .all(|levels| levels.is_some_and(|levels| levels.voxel == 14.0))
    };

    // A new sub-grid, whose lamp lights two of its chunks.
    let entity = spawn_holding(
        &mut app,
        Transform::default(),
        &[(lamp, WorldVoxel::Solid(LAMP))],
    );
    assert!(!light_is_settled(&mut app), "with its voxels to commit");
    let mut frames = 0;
    while frames == 0 || !light_is_settled(&mut app) {
        assert!(frames < 600, "the new sub-grid's light never settled");
        app.update();
        frames += 1;
    }
    let beside = vec![lamp + IVec3::X, lamp - IVec3::X];
    assert!(lit(grid_light(&mut app, entity, beside)));

    // An edit moving the lamp.
    let moved = lamp + IVec3::new(0, 4, 0);
    run(&mut app, move |voxel_world| {
        let mut grid = voxel_world.grid_mut(Grid::SubGrid(entity)).unwrap();
        grid.set_voxel(lamp, WorldVoxel::Air);
        grid.set_voxel(moved, WorldVoxel::Solid(LAMP));
    });
    assert!(!light_is_settled(&mut app), "with the edit to commit");
    let mut frames = 0;
    while frames == 0 || !light_is_settled(&mut app) {
        assert!(frames < 600, "the edit's light never settled");
        app.update();
        frames += 1;
    }
    let beside = vec![moved + IVec3::X, moved - IVec3::X];
    assert!(lit(grid_light(&mut app, entity, beside)));

    // Edited every frame, it never settles.
    for frame in 0..20 {
        let voxel = if frame % 2 == 0 {
            WorldVoxel::Solid(STONE)
        } else {
            WorldVoxel::Air
        };
        run(&mut app, move |voxel_world| {
            let mut grid = voxel_world.grid_mut(Grid::SubGrid(entity)).unwrap();
            grid.set_voxel(IVec3::new(20, 16, 16), voxel);
        });
        app.update();
        assert!(!light_is_settled(&mut app), "on frame {frame}");
    }
}

#[test]
fn a_settings_relight_relights_a_sub_grids_chunks() {
    let mut app = app(LampWorld::default());
    settle(&mut app, "the world to settle");
    let lamp = IVec3::new(31, 16, 16);
    let entity = spawn_holding(
        &mut app,
        Transform::from_xyz(5.0, 0.0, 3.0),
        &[(lamp, WorldVoxel::Solid(LAMP))],
    );
    settle(&mut app, "the sub-grid's light to settle");
    let near = lamp + IVec3::new(3, 0, 0);
    let far = lamp + IVec3::new(6, 0, 0);
    let before = grid_light(&mut app, entity, vec![near, far]);
    assert!(before.iter().all(|levels| levels.unwrap().voxel > 0.0));

    app.world_mut()
        .run_system_once(|mut world: ResMut<LampWorld>| {
            world.light.as_mut().unwrap().voxel_reach = 4;
        })
        .unwrap();
    settle(&mut app, "light to settle after the change");

    let after = grid_light(&mut app, entity, vec![near, far]);
    assert!(
        after[0].unwrap().voxel > 0.0,
        "the lamp still reaches {near}"
    );
    assert_eq!(after[1].unwrap().voxel, 0.0, "the lamp reaches past 4");
}

#[test]
fn a_sub_grid_given_a_despawned_ones_id_reads_none_of_its_lamp_light() {
    let mut app = app(LampWorld::default());
    settle(&mut app, "the world to settle");
    let lamp = IVec3::new(16, 16, 16);
    let first = spawn_holding(
        &mut app,
        Transform::default(),
        &[(lamp, WorldVoxel::Solid(LAMP))],
    );
    settle(&mut app, "the first sub-grid's light to settle");
    let id = app.world().get::<SubGrid<LampWorld>>(first).unwrap().grid();

    app.world_mut().entity_mut(first).despawn();
    for _ in 0..3 {
        app.update();
    }
    // Stone where the lamp was, in the same chunk.
    let later = spawn_holding(
        &mut app,
        Transform::default(),
        &[(lamp, WorldVoxel::Solid(STONE))],
    );
    assert_eq!(
        app.world().get::<SubGrid<LampWorld>>(later).unwrap().grid(),
        id,
        "the later sub-grid is given the despawned one's id"
    );
    settle(&mut app, "the later sub-grid's light to settle");

    let positions = voxels_in(lamp - 3, lamp + 4);
    let read = grid_light(&mut app, later, positions.clone());
    for (p, levels) in positions.into_iter().zip(read) {
        assert_eq!(levels.map(|levels| levels.voxel), Some(0.0), "at {p}");
    }
}

#[test]
fn a_sub_grid_chunk_gone_all_air_leaves_its_light() {
    let mut app = app(LampWorld::default());
    settle(&mut app, "the world to settle");
    // The lamp needs only the chunk at the origin and the stone only the one
    // beside it, which the lamp's light reaches into.
    let lamp = IVec3::new(28, 16, 16);
    let stone = IVec3::new(40, 16, 16);
    let entity = spawn_holding(
        &mut app,
        Transform::default(),
        &[
            (lamp, WorldVoxel::Solid(LAMP)),
            (stone, WorldVoxel::Solid(STONE)),
        ],
    );
    settle(&mut app, "the sub-grid's light to settle");
    let lit = vec![IVec3::new(34, 16, 16), IVec3::new(26, 16, 16)];
    let before = grid_light(&mut app, entity, lit.clone());
    assert!(before.iter().all(|levels| levels.unwrap().voxel > 0.0));

    run(&mut app, move |voxel_world| {
        let mut grid = voxel_world.grid_mut(Grid::SubGrid(entity)).unwrap();
        grid.set_voxel(stone, WorldVoxel::Air);
    });
    settle(&mut app, "light to settle after the stone is gone");

    let after = grid_light(&mut app, entity, lit);
    assert_eq!(
        after[0].map(|levels| levels.voxel),
        Some(0.0),
        "the chunk gone all air holds no light"
    );
    assert_eq!(after[1], before[1], "the lamp's chunk keeps its light");
}

#[test]
fn the_bench_stats_count_a_sub_grids_relights_apart_from_the_worlds() {
    use crate::light_bench::light_stats;
    let stats = |app: &App| light_stats::<LampWorld>(app.world()).unwrap();
    let mut app = app(LampWorld::default());
    settle(&mut app, "the world to settle");

    let before = stats(&app);
    run(&mut app, |voxel_world| {
        voxel_world.set_voxel(IVec3::new(8, 16, 16), WorldVoxel::Solid(LAMP));
    });
    settle(&mut app, "the world's lamp to settle");
    let lamp_in_world = stats(&app);
    assert!(lamp_in_world.relights > before.relights);
    assert_eq!(lamp_in_world.sub_grid_relights, before.sub_grid_relights);

    // One voxel in the middle of a chunk gives the sub-grid one chunk, whose
    // first solve is a relight.
    spawn_holding(
        &mut app,
        Transform::from_xyz(0.0, 40.0, 0.0),
        &[(IVec3::new(16, 16, 16), WorldVoxel::Solid(LAMP))],
    );
    settle(&mut app, "the sub-grid's light to settle");
    let after = stats(&app);
    assert_eq!(after.sub_grid_relights - lamp_in_world.sub_grid_relights, 1);
    assert_eq!(after.relights - lamp_in_world.relights, 1);
    assert_eq!(after.sub_grid_chunk_solves, 0);
}

#[test]
fn a_sub_grid_edited_voxel_by_voxel_reads_the_light_of_one_built_with_its_voxels() {
    let mut app = app(LampWorld::default());
    settle(&mut app, "the world to settle");
    let (first, second, stone) = (
        IVec3::new(10, 16, 16),
        IVec3::new(22, 16, 16),
        IVec3::new(16, 16, 16),
    );
    let edited = spawn_holding(
        &mut app,
        Transform::default(),
        &[
            (first, WorldVoxel::Solid(LAMP)),
            (stone, WorldVoxel::Solid(STONE)),
            (stone + IVec3::Y, WorldVoxel::Solid(STONE)),
        ],
    );
    settle(&mut app, "the sub-grid's light to settle");
    // A lamp turned to stone, a stone taken out and a lamp put in, in one
    // commit.
    run(&mut app, move |voxel_world| {
        let mut grid = voxel_world.grid_mut(Grid::SubGrid(edited)).unwrap();
        grid.set_voxel(first, WorldVoxel::Solid(STONE));
        grid.set_voxel(stone + IVec3::Y, WorldVoxel::Air);
        grid.set_voxel(second, WorldVoxel::Solid(LAMP));
    });
    let built = spawn_holding(
        &mut app,
        Transform::from_xyz(0.0, 64.0, 0.0),
        &[
            (first, WorldVoxel::Solid(STONE)),
            (stone, WorldVoxel::Solid(STONE)),
            (second, WorldVoxel::Solid(LAMP)),
        ],
    );
    settle(&mut app, "both sub-grids' light to settle");

    let own = |app: &App, entity: Entity, p: IVec3| {
        let grid = app
            .world()
            .get::<SubGrid<LampWorld>>(entity)
            .unwrap()
            .grid();
        app.world()
            .resource::<crate::light::LightSettings<LampWorld>>()
            .own_light_at(grid, p)
            .map(|levels| levels.voxel)
    };
    let positions = voxels_in(IVec3::new(0, 8, 8), IVec3::new(32, 24, 24));
    for p in positions {
        assert_eq!(own(&app, edited, p), own(&app, built, p), "at {p}");
    }
    assert_eq!(own(&app, edited, second + IVec3::X), Some(14.0));
}

#[test]
fn stamping_a_lit_sub_grid_relights_none_of_its_chunks() {
    use crate::light_bench::light_stats;
    let stats = |app: &App| light_stats::<LampWorld>(app.world()).unwrap();
    let mut app = app(LampWorld::default());
    settle(&mut app, "the world to settle");
    let entity = spawn_holding(
        &mut app,
        Transform::from_xyz(0.0, 8.0, 0.0),
        &[
            (IVec3::new(16, 16, 16), WorldVoxel::Solid(LAMP)),
            (IVec3::new(40, 16, 16), WorldVoxel::Solid(STONE)),
        ],
    );
    settle(&mut app, "the sub-grid's light to settle");
    let before = stats(&app);

    run(&mut app, move |voxel_world| {
        voxel_world
            .stamp(entity, Grid::World, incoming_replaces)
            .unwrap();
    });
    settle(&mut app, "the stamp's light to settle");
    assert!(app.world().get_entity(entity).is_err(), "the stamp landed");
    let after = stats(&app);
    assert!(after.relights > before.relights, "the world relights");
    assert_eq!(after.sub_grid_relights, before.sub_grid_relights);
}
