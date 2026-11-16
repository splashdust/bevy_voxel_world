//! Tests of the slots that world chunks of air take while the tag map holds
//! them: a sub-grid hanging in a lamp-lit hall reads the hall air's light
//! voxel by voxel, made from the light the world view already stores.

use std::sync::atomic::Ordering;
use std::sync::Arc;

use bevy::ecs::system::RunSystemOnce;
use bevy::prelude::*;

use crate::light::route::{tag_levels, LightRoute, HAS_SLOT};
use crate::light::slot::{cell_index, texel_at, texel_levels};
use crate::light::LightSettings;
use crate::prelude::*;
use crate::test::light::edits::settle;
use crate::test::light::sky_light::chunk_light;
use crate::test::light::voxel_light::{app, LampWorld, LAMP, STONE};

/// The hall's air, from its minimum corner inclusive to its maximum corner
/// exclusive. It holds the chunks of air at (0, 0, 0) and (1, 0, 0), whose
/// padding holds no solid.
const HALL: (IVec3, IVec3) = (IVec3::new(-8, -8, -8), IVec3::new(72, 40, 40));

/// The hall's walls, 8 voxels thick around its air.
const WALLS: (IVec3, IVec3) = (IVec3::new(-16, -16, -16), IVec3::new(80, 48, 48));

/// Lamps in the hall's floor, under each chunk of air.
const LAMPS: [IVec3; 2] = [IVec3::new(16, -9, 16), IVec3::new(48, -9, 16)];

/// The chunks of air in the hall.
const HALL_AIR: [IVec3; 2] = [IVec3::ZERO, IVec3::X];

/// A chunk of open sky above the hall.
const SKY: IVec3 = IVec3::new(0, 2, 0);

fn within((min, max): (IVec3, IVec3), p: IVec3) -> bool {
    p.cmpge(min).all() && p.cmplt(max).all()
}

/// A closed hall of stone floating in air, lit by lamps in its floor. With
/// `mouth`, its west wall is open to the air outside.
fn hall(mouth: bool) -> impl Fn(IVec3) -> WorldVoxel + Send + Sync + 'static {
    move |p: IVec3| {
        let open_wall = mouth && p.x < HALL.0.x && within((WALLS.0, HALL.1), p);
        if LAMPS.contains(&p) {
            WorldVoxel::Solid(LAMP)
        } else if within(WALLS, p) && !within(HALL, p) && !open_wall {
            WorldVoxel::Solid(STONE)
        } else {
            WorldVoxel::Air
        }
    }
}

/// An app of the hall under `light`, run until light has settled.
fn settled_hall(mouth: bool, light: LightPropagation) -> App {
    let mut app = app(LampWorld {
        scene: Arc::new(hall(mouth)),
        light: Some(light),
        spawning_distance: 3,
        ..default()
    });
    app.init_resource::<Moves>()
        .add_systems(Update, move_sub_grids);
    settle(&mut app, "the hall to settle");
    app
}

/// Sky light and voxel light, with no sky fill.
fn unfilled() -> LightPropagation {
    LightPropagation {
        sky_fill: None,
        ..default()
    }
}

/// Where the system [`move_sub_grids`] puts each sub-grid named, in the next
/// frame's `Update`.
#[derive(Resource, Default)]
struct Moves(Vec<(Entity, Transform)>);

fn move_sub_grids(mut moves: ResMut<Moves>, mut transforms: Query<&mut Transform>) {
    for (entity, to) in moves.0.drain(..) {
        *transforms.get_mut(entity).unwrap() = to;
    }
}

/// Moves the sub-grid `entity` to `to` in the next frame's `Update`.
fn move_to(app: &mut App, entity: Entity, to: Transform) {
    app.world_mut().resource_mut::<Moves>().0.push((entity, to));
}

/// Two stone voxels at opposite corners of a 3-voxel box.
const CORNERS: [IVec3; 2] = [IVec3::ZERO, IVec3::splat(2)];

/// Where a sub-grid of [`CORNERS`] touches both chunks of the hall's air
/// and nothing else.
fn across_the_hall() -> Transform {
    Transform::from_xyz(30.0, 14.0, 14.0)
}

/// Where a sub-grid of [`CORNERS`] touches only the chunk [`SKY`].
fn in_the_sky() -> Transform {
    Transform::from_xyz(10.0, 80.0, 10.0)
}

/// Spawns a sub-grid at `transform` holding stone at [`CORNERS`].
fn spawn_sub_grid(app: &mut App, transform: Transform) -> Entity {
    app.world_mut()
        .run_system_once(move |mut voxel_world: VoxelWorld<LampWorld>| {
            let entity = voxel_world.spawn_sub_grid(transform);
            let mut grid = voxel_world.grid_mut(Grid::SubGrid(entity)).unwrap();
            for voxel in CORNERS {
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

/// The light `light_at` gives at every voxel of the level-0 chunk at
/// `chunk`, by its index in the chunk, x first.
fn light_of_chunk(app: &mut App, chunk: IVec3) -> Vec<LightLevels> {
    app.world_mut()
        .run_system_once(move |voxel_world: VoxelWorld<LampWorld>| {
            crate::test::light::voxel_light::origin_voxels()
                .map(|local| {
                    voxel_world
                        .light_at(chunk * 32 + local)
                        .unwrap_or_else(|| panic!("no light at {}", chunk * 32 + local))
                })
                .collect::<Vec<_>>()
        })
        .unwrap()
}

/// The light the slot of the chunk of air at `chunk`, which the tag map
/// holds, shows at each of its voxels, from the bytes queued for it. Fails
/// unless it holds a slot.
fn slot_light(app: &App, chunk: IVec3) -> Vec<LightLevels> {
    let tag = answer(app, chunk).expect("the tag map holds the chunk");
    assert_ne!(tag & HAS_SLOT, 0, "{chunk} holds no slot: {tag:#x}");
    let bytes = route(app)
        .queued_upload(tag & 0xFF_FFFF)
        .expect("the slot's bytes are queued");
    let precision = LightPropagation::default().precision;
    crate::test::light::voxel_light::origin_voxels()
        .map(|local| {
            let index = cell_index(local.as_uvec3() + UVec3::ONE);
            let (sky, voxel) = texel_at(bytes, index, precision);
            texel_levels(sky, voxel, precision)
        })
        .collect()
}

/// Fails unless the slot of the chunk of air at `chunk` shows, at every
/// voxel, the light `light_at` gives there.
fn assert_slot_matches_light_at(app: &mut App, chunk: IVec3, when: &str) {
    let shown = slot_light(app, chunk);
    let read = light_of_chunk(app, chunk);
    for (index, local) in crate::test::light::voxel_light::origin_voxels().enumerate() {
        assert_eq!(
            shown[index],
            read[index],
            "at {} in {chunk}, {when}",
            chunk * 32 + local
        );
    }
}

/// The solves and relights light has run.
fn solves_and_relights(app: &App) -> (u64, u64) {
    let counters = app
        .world()
        .resource::<LightSettings<LampWorld>>()
        .counters();
    (
        counters.solves.load(Ordering::Relaxed),
        counters.relights.load(Ordering::Relaxed),
    )
}

#[test]
fn a_sub_grid_in_a_lamp_lit_hall_gives_the_halls_air_slots_and_open_sky_none() {
    let mut app = settled_hall(false, unfilled());
    for chunk in HALL_AIR {
        assert_eq!(answer(&app, chunk), None);
    }

    spawn_sub_grid(&mut app, across_the_hall());
    spawn_sub_grid(&mut app, in_the_sky());
    app.update();

    for chunk in HALL_AIR {
        assert_slot_matches_light_at(&mut app, chunk, "with a sub-grid in the hall");
    }
    let open_sky = tag_levels(15, 0, LightPropagation::default().precision);
    assert_eq!(answer(&app, SKY), Some(open_sky));
    assert!(light_of_chunk(&mut app, SKY)
        .iter()
        .all(|levels| *levels == LightLevels::new(15, 0)));
}

#[test]
fn a_sub_grid_entering_a_hall_starts_no_solve_and_no_relight() {
    let mut app = settled_hall(false, unfilled());
    let entity = spawn_sub_grid(&mut app, in_the_sky());
    settle(&mut app, "the sub-grid's own light to settle");
    let before = solves_and_relights(&app);

    move_to(&mut app, entity, across_the_hall());
    for _ in 0..10 {
        app.update();
    }

    assert_ne!(answer(&app, IVec3::ZERO).unwrap() & HAS_SLOT, 0);
    assert_eq!(solves_and_relights(&app), before);
}

#[test]
fn an_air_chunk_with_no_solid_in_its_padding_shows_the_brighter_of_its_light_and_sky_fill(
) {
    let mut app = settled_hall(
        true,
        LightPropagation {
            sky: Sky::Ceiling { y: 100 },
            ..default()
        },
    );

    spawn_sub_grid(&mut app, across_the_hall());
    app.update();

    // Deep in the hall, past where sky light spreads from the mouth, the
    // sky fill field gives more than the stored light.
    let far = IVec3::X;
    let stored = chunk_light(&mut app, far).expect("the chunk's light");
    let read = light_of_chunk(&mut app, far);
    assert!(
        stored
            .iter()
            .zip(&read)
            .any(|((_, stored), read)| read.sky > stored.sky),
        "sky fill adds nothing in {far}"
    );
    assert_slot_matches_light_at(&mut app, far, "with sky fill");
}

#[test]
fn an_air_chunks_slot_follows_a_relight_of_its_light() {
    let mut app = settled_hall(false, unfilled());
    spawn_sub_grid(&mut app, across_the_hall());
    app.update();
    let before = light_of_chunk(&mut app, IVec3::ZERO);

    // A lamp on the hall's back wall.
    app.world_mut()
        .run_system_once(|mut voxel_world: VoxelWorld<LampWorld>| {
            voxel_world.set_voxel(IVec3::new(16, 16, -9), WorldVoxel::Solid(LAMP));
        })
        .unwrap();
    settle(&mut app, "light to settle with the new lamp");

    assert_ne!(light_of_chunk(&mut app, IVec3::ZERO), before);
    assert_slot_matches_light_at(&mut app, IVec3::ZERO, "after the relight");
}

#[test]
fn an_air_chunks_slot_is_freed_as_its_entry_leaves_when_the_sub_grid_moves_away() {
    let mut app = settled_hall(false, unfilled());
    let entity = spawn_sub_grid(&mut app, in_the_sky());
    settle(&mut app, "the sub-grid's own light to settle");
    let held_before = route(&app).slots_held();
    move_to(&mut app, entity, across_the_hall());
    app.update();
    assert_eq!(route(&app).slots_held(), held_before + HALL_AIR.len());

    move_to(&mut app, entity, in_the_sky());
    app.update();

    for chunk in HALL_AIR {
        assert_eq!(answer(&app, chunk), None);
    }
    assert!(answer(&app, SKY).is_some());
    assert_eq!(route(&app).slots_held(), held_before);
}

#[test]
fn the_bench_stats_count_the_slots_the_halls_air_takes_while_a_sub_grid_is_near() {
    let air_slots = |app: &App| {
        crate::grid_bench::tag_map_stats::<LampWorld>(app.world())
            .expect("a tag map")
            .air_slots
    };
    let mut app = settled_hall(false, unfilled());
    let entity = spawn_sub_grid(&mut app, in_the_sky());
    settle(&mut app, "the sub-grid's own light to settle");
    assert_eq!(air_slots(&app), 0);

    move_to(&mut app, entity, across_the_hall());
    app.update();
    assert_eq!(air_slots(&app), HALL_AIR.len() as u64);

    move_to(&mut app, entity, in_the_sky());
    app.update();
    assert_eq!(air_slots(&app), 0);
}
