//! Tests of what the tag map answers in hinted air: a light entry's most
//! common level where one exists, and otherwise the column answer
//! `light_at` gives, which lands from a task.

use std::collections::HashMap;
use std::sync::Arc;

use bevy::ecs::system::RunSystemOnce;
use bevy::prelude::*;

use super::edits::settle;
use super::sky_light::sky_light;
use super::voxel_light::{app, light_at, LampWorld, STONE};
use crate::light::route::LightRoute;
use crate::light::LightSettings;
use crate::prelude::*;
use crate::test::update_until;

/// The chunk position of an island 16 voxels thick, two chunk positions
/// above the hinted air under it, within the spawning distance.
const ISLAND: IVec3 = IVec3::new(2, 2, 0);

/// An island like [`ISLAND`] beyond the spawning distance.
const FAR_ISLAND: IVec3 = IVec3::new(4, 2, 0);

/// Hinted air under [`ISLAND`], which no loaded solid touches.
const UNDER: IVec3 = IVec3::new(2, 0, 0);

/// Hinted air right under [`ISLAND`], which holds a light entry.
const BELOW_ISLAND: IVec3 = IVec3::new(2, 1, 0);

/// Hinted air under open sky.
const OPEN: IVec3 = IVec3::new(-2, 0, 0);

/// Hinted air under [`FAR_ISLAND`], beyond the spawning distance.
const FAR_UNDER: IVec3 = IVec3::new(4, 0, 0);

/// Air, apart from the islands at [`ISLAND`] and [`FAR_ISLAND`]: stone
/// filling each one's chunk position from y = 64 to 80.
fn islands(p: IVec3) -> WorldVoxel {
    let chunk = p.div_euclid(IVec3::splat(32));
    let island = (chunk == ISLAND || chunk == FAR_ISLAND) && (64..80).contains(&p.y);
    if island {
        WorldVoxel::Solid(STONE)
    } else {
        WorldVoxel::Air
    }
}

/// A sky-lit world of [`islands`] whose fill hints say every chunk position
/// but the islands' is air. The camera sits in the chunk at the origin, so
/// only chunk positions next to it spawn as air.
fn island_world() -> LampWorld {
    LampWorld {
        scene: Arc::new(islands),
        light: Some(sky_light(Sky::default(), 30)),
        hints: Some(Arc::new(|position: IVec3| {
            if position == ISLAND || position == FAR_ISLAND {
                FillHint::Unknown
            } else {
                FillHint::Uniform(WorldVoxel::Air)
            }
        })),
        spawning_distance: 3,
        ..default()
    }
}

/// An app of [`island_world`], run until the island has streamed in and
/// light has settled.
fn island_app() -> App {
    let mut app = app(island_world());
    settle(&mut app, "the island to stream in");
    app
}

/// Spawns a sub-grid holding two stone voxels in the chunk position
/// `chunk`, whose padded box lies inside it.
fn spawn_body(app: &mut App, chunk: IVec3) -> Entity {
    let at = (chunk * 32 + 10).as_vec3();
    app.world_mut()
        .run_system_once(move |mut voxel_world: VoxelWorld<LampWorld>| {
            let entity = voxel_world.spawn_sub_grid(Transform::from_translation(at));
            let mut grid = voxel_world.grid_mut(Grid::SubGrid(entity)).unwrap();
            for voxel in [IVec3::ZERO, IVec3::splat(2)] {
                grid.set_voxel(voxel, WorldVoxel::Solid(STONE));
            }
            entity
        })
        .unwrap()
}

fn route(app: &App) -> &LightRoute<LampWorld> {
    app.world().resource::<LightRoute<LampWorld>>()
}

/// The tag the tag map holds at `position`.
fn answer(app: &App, position: IVec3) -> Option<u32> {
    route(app).tag_map()?.answer(position)
}

/// The tag of a chunk with no slot showing `levels`, at 16 levels.
fn tag_of(levels: LightLevels) -> u32 {
    levels.sky as u32 | (levels.voxel as u32) << 4
}

/// The tag of open sky with no lamps: sky 15 and voxel 0.
const OPEN_SKY: u32 = 15;

/// What `light_at` reads at the middle voxel of the chunk position `chunk`.
fn middle_light(app: &mut App, chunk: IVec3) -> LightLevels {
    light_at(app, chunk * 32 + 16).expect("light in hinted air")
}

/// Whether the world view holds a light entry at the chunk position
/// `position`.
fn holds_entry(app: &App, position: IVec3) -> bool {
    app.world()
        .resource::<LightSettings<LampWorld>>()
        .view
        .read()
        .get(&position)
        .is_some_and(|chunk| chunk.is_entry())
}

/// How many light entries the world view holds.
fn entries(app: &App) -> usize {
    app.world()
        .resource::<LightSettings<LampWorld>>()
        .view
        .read()
        .world_chunks()
        .filter(|(_, chunk)| chunk.is_entry())
        .count()
}

#[test]
fn hinted_air_answers_its_column_once_the_answer_lands() {
    let mut app = island_app();
    for chunk in [UNDER, OPEN] {
        assert!(!holds_entry(&app, chunk), "{chunk}");
    }
    spawn_body(&mut app, UNDER);
    spawn_body(&mut app, OPEN);
    app.update();
    // No answer has landed in the frame the bodies enter.
    assert_eq!(answer(&app, UNDER), Some(OPEN_SKY));
    assert_eq!(answer(&app, OPEN), Some(OPEN_SKY));

    update_until(&mut app, "the answer under the island", |app| {
        answer(app, UNDER) != Some(OPEN_SKY)
    });
    assert_eq!(middle_light(&mut app, UNDER), LightLevels::new(0, 0));
    assert_eq!(answer(&app, UNDER), Some(tag_of(LightLevels::new(0, 0))));
    for _ in 0..5 {
        app.update();
    }
    assert_eq!(middle_light(&mut app, OPEN), LightLevels::new(15, 0));
    assert_eq!(answer(&app, OPEN), Some(OPEN_SKY));
}

#[test]
fn hinted_air_holding_a_light_entry_answers_its_most_common_level() {
    let mut app = island_app();
    assert!(holds_entry(&app, BELOW_ISLAND));
    spawn_body(&mut app, BELOW_ISLAND);
    app.update();

    // How many voxels of the entry, all open, read each pair of levels
    // through `light_at`.
    let read = app
        .world_mut()
        .run_system_once(|voxel_world: VoxelWorld<LampWorld>| {
            let mut counts: HashMap<(u32, u32), usize> = HashMap::new();
            for local in super::voxel_light::origin_voxels() {
                let levels = voxel_world.light_at(BELOW_ISLAND * 32 + local).unwrap();
                *counts
                    .entry((levels.sky as u32, levels.voxel as u32))
                    .or_default() += 1;
            }
            counts
        })
        .unwrap();
    let most_common = |channel: fn(&(u32, u32)) -> u32| {
        let mut by_level: HashMap<u32, usize> = HashMap::new();
        for (levels, count) in &read {
            *by_level.entry(channel(levels)).or_default() += count;
        }
        by_level
            .into_iter()
            .max_by_key(|(level, count)| (*count, *level))
            .unwrap()
            .0
    };
    let (sky, voxel) = (most_common(|l| l.0), most_common(|l| l.1));
    // Sky light reaches in under the island's edges, so the entry isn't one
    // level throughout.
    assert!(read.len() > 1, "{read:?}");
    assert_eq!(answer(&app, BELOW_ISLAND), Some(sky | voxel << 4));
}

#[test]
fn a_body_entering_hinted_air_makes_no_light_entry_and_takes_no_slot() {
    let mut app = island_app();
    let (before, allocations) = (entries(&app), route(&app).counters.allocations);
    let body = spawn_body(&mut app, UNDER);
    update_until(&mut app, "the answer under the island", |app| {
        answer(app, UNDER) != Some(OPEN_SKY)
    });
    // Across a chunk boundary into the hinted air beside it.
    app.world_mut()
        .run_system_once(move |mut transforms: Query<&mut Transform>| {
            transforms.get_mut(body).unwrap().translation = Vec3::new(95.0, 10.0, 10.0);
        })
        .unwrap();
    for _ in 0..10 {
        app.update();
    }
    assert!(route(&app).tag_map().unwrap().holds(UNDER + IVec3::X));
    assert_eq!(entries(&app), before);
    assert!(!holds_entry(&app, UNDER) && !holds_entry(&app, UNDER + IVec3::X));
    assert_eq!(route(&app).counters.allocations, allocations);
}

#[test]
fn a_shaft_dug_through_the_island_opens_the_sky_under_it_once_the_answer_lands() {
    let mut app = island_app();
    spawn_body(&mut app, UNDER);
    update_until(&mut app, "the answer under the island", |app| {
        answer(app, UNDER) == Some(tag_of(LightLevels::new(0, 0)))
    });

    // Through the island above the middle voxel of the hinted air.
    let middle = UNDER * 32 + 16;
    app.world_mut()
        .run_system_once(move |mut voxel_world: VoxelWorld<LampWorld>| {
            for y in 64..80 {
                voxel_world.set_voxel(IVec3::new(middle.x, y, middle.z), WorldVoxel::Air);
            }
        })
        .unwrap();
    update_until(&mut app, "the answer under the shaft", |app| {
        answer(app, UNDER) == Some(OPEN_SKY)
    });
    assert_eq!(middle_light(&mut app, UNDER), LightLevels::new(15, 0));
}

#[test]
fn hinted_air_beyond_the_spawning_distance_answers_open_sky() {
    let mut app = island_app();
    spawn_body(&mut app, FAR_UNDER);
    for _ in 0..10 {
        app.update();
    }
    assert_eq!(answer(&app, FAR_UNDER), Some(OPEN_SKY));
    // A walk would find the island above it.
    assert_eq!(middle_light(&mut app, FAR_UNDER), LightLevels::new(0, 0));
}
