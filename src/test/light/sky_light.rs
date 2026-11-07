//! Tests of sky light at LOD level 0: direct sky from the sky rule, the
//! column answers chunks work out, and light spreading from direct sky, read
//! through `light_at`. Sky fill is off in all of them.

use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use bevy::ecs::system::RunSystemOnce;
use bevy::prelude::*;

use super::super::{chunk_keys, chunk_status, update_until};
use super::reference::Reference;
use super::voxel_light::{app, light_at, Hold, LampWorld, Scene, LAMP, STONE};
use crate::chunk_map::ChunkMap;
use crate::light::LightSettings;
use crate::prelude::*;

/// Sky light under `sky` at `sky_reach`, with sky fill off, and voxel light at
/// the default voxel reach.
fn sky_light(sky: Sky, sky_reach: u8) -> LightPropagation {
    LightPropagation {
        sky,
        sky_reach,
        sky_fill: None,
        ..default()
    }
}

/// A lamp in the far end of the corridor of [`shaft_and_caves`].
const CORRIDOR_LAMP: IVec3 = IVec3::new(-24, 4, 8);

/// Stone below y = 40 and air above, with a 1×1 shaft from the surface at
/// x = 8, z = 8 down into a corridor 8 voxels high and wide that runs along x
/// from -28 to 60, a lamp on the corridor's floor far from the shaft, and a
/// sealed pocket of air beside it.
///
/// A second shaft rises from the corridor at x = 24, z = 6 into the chunk
/// above and ends under stone at y = 36. Its column is open in the origin
/// chunk's padding, so only a walk proves it has no direct sky.
fn shaft_and_caves(p: IVec3) -> WorldVoxel {
    let shaft = p.x == 8 && p.z == 8 && (12..40).contains(&p.y);
    let dead_end = p.x == 24 && p.z == 6 && (12..36).contains(&p.y);
    let corridor =
        (-28..60).contains(&p.x) && (5..13).contains(&p.y) && (4..12).contains(&p.z);
    let pocket =
        (20..26).contains(&p.x) && (18..24).contains(&p.y) && (20..26).contains(&p.z);
    if p == CORRIDOR_LAMP {
        WorldVoxel::Solid(LAMP)
    } else if p.y >= 40 || shaft || dead_end || corridor || pocket {
        WorldVoxel::Air
    } else {
        WorldVoxel::Solid(STONE)
    }
}

/// The world of [`shaft_and_caves`] with `light`.
fn shaft_world(light: LightPropagation) -> LampWorld {
    LampWorld {
        scene: Arc::new(shaft_and_caves),
        light: Some(light),
        ..default()
    }
}

fn light_is_settled(app: &mut App) -> bool {
    app.world_mut()
        .run_system_once(|voxel_world: VoxelWorld<LampWorld>| {
            voxel_world.light_is_settled()
        })
        .unwrap()
}

/// The level-0 chunks spawned in `app`.
fn spawned(app: &mut App) -> Vec<IVec3> {
    chunk_keys::<LampWorld>(app)
        .into_iter()
        .filter(|key| key.lod_level == 0)
        .map(|key| key.position)
        .collect()
}

/// Whether every spawned chunk but `held` is Ready and light has settled.
fn settled_but(app: &mut App, held: Option<IVec3>) -> bool {
    let chunks = spawned(app);
    !chunks.is_empty()
        && chunks
            .iter()
            .filter(|chunk| Some(**chunk) != held)
            .all(|chunk| {
                chunk_status::<LampWorld>(app, *chunk) == Some(ChunkStatus::Ready)
            })
        && light_is_settled(app)
}

/// An app of `world` run until every chunk is Ready and light has settled.
fn settled(world: LampWorld) -> App {
    let mut app = app(world);
    update_until(&mut app, "the world to settle", |app| {
        settled_but(app, None)
    });
    app
}

fn levels(app: &mut App, position: IVec3) -> LightLevels {
    light_at(app, position).unwrap_or_else(|| panic!("no light at {position}"))
}

/// The light of every voxel of the level-0 chunk at `chunk`, or `None` if it
/// has none published.
fn chunk_light(app: &mut App, chunk: IVec3) -> Option<Vec<(IVec3, LightLevels)>> {
    app.world_mut()
        .run_system_once(
            move |settings: Res<LightSettings<LampWorld>>,
                  chunk_map: Res<ChunkMap<LampWorld, u8>>| {
                let map = chunk_map.get_read_lock();
                let data = map.get(&ChunkKey::new(chunk, 0))?;
                let light = settings.view.read().get(&chunk)?.light.clone();
                super::voxel_light::origin_voxels()
                    .map(|local| {
                        let p = chunk * 32 + local;
                        let voxel = data.get_voxel_at_world_position(p)?;
                        let levels = light.levels_at(
                            local.as_uvec3(),
                            voxel,
                            &settings.task.emission,
                        );
                        Some((p, levels))
                    })
                    .collect::<Option<Vec<_>>>()
            },
        )
        .unwrap()
}

fn walk_lookups(app: &App) -> u64 {
    app.world()
        .resource::<LightSettings<LampWorld>>()
        .counters()
        .walks
        .lookups
        .load(Ordering::Relaxed)
}

#[test]
fn a_1x1_shaft_lights_its_column_and_a_pool_at_its_foot_that_fades_to_black() {
    let mut app = settled(shaft_world(sky_light(Sky::default(), 30)));

    // Direct sky falls down the shaft to the corridor's floor.
    for y in 5..40 {
        let p = IVec3::new(8, y, 8);
        let levels = levels(&mut app, p);
        assert_eq!(levels.sky, 15.0, "at {p}");
        assert!(levels.direct_sky(), "at {p}");
    }
    // Beside it in the corridor, light fades with distance from the shaft.
    let mut last = 15.0;
    for x in 9..=38 {
        let sky = levels(&mut app, IVec3::new(x, 6, 8)).sky;
        assert!(sky > 0.0 && sky <= last, "{sky} at x = {x} after {last}");
        assert!(sky < 15.0, "at x = {x}");
        last = sky;
    }
    assert_eq!(levels(&mut app, IVec3::new(9, 6, 8)).sky, 14.0);
    assert!(last <= 2.0, "{last} at the pool's edge");
    // Past sky reach the corridor is black.
    for x in 42..60 {
        assert_eq!(levels(&mut app, IVec3::new(x, 6, 8)).sky, 0.0, "at x = {x}");
    }
    // The surface has direct sky.
    assert!(levels(&mut app, IVec3::new(20, 40, 3)).direct_sky());
}

#[test]
fn a_sealed_pocket_and_a_cavern_beyond_its_pool_read_0_in_both_channels() {
    let mut app = settled(shaft_world(sky_light(Sky::default(), 30)));

    for z in 20..26 {
        for y in 18..24 {
            for x in 20..26 {
                let p = IVec3::new(x, y, z);
                assert_eq!(levels(&mut app, p), LightLevels::new(0, 0), "at {p}");
            }
        }
    }
    // The corridor past the shaft's pool and the lamp's.
    for z in 4..12 {
        for y in 5..13 {
            for x in 42..60 {
                let p = IVec3::new(x, y, z);
                assert_eq!(levels(&mut app, p), LightLevels::new(0, 0), "at {p}");
            }
        }
    }
}

#[test]
fn sky_off_leaves_voxel_light_working() {
    let mut app = settled(shaft_world(sky_light(Sky::Off, 30)));

    let beside_lamp = levels(&mut app, CORRIDOR_LAMP + IVec3::Y);
    assert_eq!(beside_lamp, LightLevels::new(0, 14));
    for p in [
        IVec3::new(8, 30, 8),
        IVec3::new(8, 6, 8),
        IVec3::new(20, 40, 3),
    ] {
        assert_eq!(levels(&mut app, p).sky, 0.0, "at {p}");
    }
}

/// A slab of stone at y = 32 to 36, wide enough to cover every spawned chunk,
/// over air down to stone below y = -28.
fn slab_over_ground(p: IVec3) -> WorldVoxel {
    if (32..36).contains(&p.y) || p.y < -28 {
        WorldVoxel::Solid(STONE)
    } else {
        WorldVoxel::Air
    }
}

#[test]
fn a_shadow_ends_at_the_sky_clearance() {
    let mut app = settled(LampWorld {
        scene: Arc::new(slab_over_ground),
        light: Some(sky_light(Sky::Clearance { chunks: 1 }, 30)),
        ..default()
    });

    // The chunk right under the slab's chunk is in its shadow, and the chunk
    // below that, whose one chunk of clearance holds only air, isn't: the
    // shadow ends in a flat line on the chunk boundary at y = 0.
    for x in [2, 16, 30] {
        for z in [2, 16, 30] {
            for y in 0..32 {
                let p = IVec3::new(x, y, z);
                assert!(!levels(&mut app, p).direct_sky(), "at {p}");
            }
            for y in -28..0 {
                let p = IVec3::new(x, y, z);
                assert!(levels(&mut app, p).direct_sky(), "at {p}");
            }
        }
    }
}

/// Stone below y = 64, with a sealed room of air inside it 12 voxels wide and
/// 68 voxels high, from y = -28 to 40, much taller than a sky clearance of 1
/// chunk.
fn tall_sealed_room(p: IVec3) -> WorldVoxel {
    let room =
        (10..22).contains(&p.x) && (-28..40).contains(&p.y) && (10..22).contains(&p.z);
    if p.y >= 64 || room {
        WorldVoxel::Air
    } else {
        WorldVoxel::Solid(STONE)
    }
}

#[test]
fn a_sky_ceiling_is_exact_for_a_sealed_space_taller_than_the_clearance() {
    let room_voxels = || {
        (10..22).flat_map(|z| {
            (-28..40).flat_map(move |y| (10..22).map(move |x| IVec3::new(x, y, z)))
        })
    };
    let ceiling = Sky::Ceiling { y: 40 };
    let mut app = settled(LampWorld {
        scene: Arc::new(tall_sealed_room),
        light: Some(sky_light(ceiling.clone(), 30)),
        ..default()
    });
    let reference = Reference::sky_light(
        IVec3::new(0, -32, 0),
        IVec3::new(32, 64, 32),
        30,
        LightShape::Round,
        &ceiling,
        tall_sealed_room,
    );
    for p in room_voxels() {
        assert_eq!(reference.sky_at(p), 0);
        assert_eq!(levels(&mut app, p).sky, 0.0, "at {p}");
    }

    // A clearance of 1 chunk lets sky light onto the room's floor, as the sky
    // rule says.
    let mut app = settled(LampWorld {
        scene: Arc::new(tall_sealed_room),
        light: Some(sky_light(Sky::Clearance { chunks: 1 }, 30)),
        ..default()
    });
    assert!(levels(&mut app, IVec3::new(16, -28, 16)).direct_sky());
}

#[test]
fn a_fill_hint_settles_columns_with_no_lookups_and_never_changes_light() {
    let unhinted_world = shaft_world(sky_light(Sky::default(), 30));
    // Every chunk position above the surface's chunk is air.
    let hinted_world = LampWorld {
        hints: Some(Arc::new(|position: IVec3| {
            if position.y >= 2 {
                FillHint::Uniform(WorldVoxel::Air)
            } else {
                FillHint::Unknown
            }
        })),
        ..unhinted_world.clone()
    };
    let mut unhinted = settled(unhinted_world);
    let mut hinted = settled(hinted_world);

    // The origin's walk up its shaft reads the surface's chunk and stops at
    // the hinted air above it.
    assert!(walk_lookups(&hinted) > 0);
    assert!(walk_lookups(&hinted) * 4 < walk_lookups(&unhinted));
    for chunk in [IVec3::ZERO, IVec3::Y, IVec3::NEG_Y, IVec3::X] {
        assert_eq!(
            chunk_light(&mut hinted, chunk),
            chunk_light(&mut unhinted, chunk),
            "{chunk} has other light with fill hints"
        );
    }

    // A fill hint above every column settles them all with no lookups.
    let mut all_hinted = settled(LampWorld {
        hints: Some(Arc::new(|position: IVec3| {
            if position.y >= 1 {
                FillHint::Uniform(WorldVoxel::Air)
            } else {
                FillHint::Unknown
            }
        })),
        scene: Arc::new(|p: IVec3| {
            if p.y < 16 {
                WorldVoxel::Solid(STONE)
            } else {
                WorldVoxel::Air
            }
        }),
        ..shaft_world(sky_light(Sky::default(), 30))
    });
    assert_eq!(walk_lookups(&all_hinted), 0);
    assert!(levels(&mut all_hinted, IVec3::new(5, 16, 5)).direct_sky());
}

// --- Ground truth ---------------------------------------------------------------

/// The chunks around the origin, where the camera is.
fn around_origin() -> Vec<IVec3> {
    (-1..=1)
        .flat_map(|z| {
            (-1..=1).flat_map(move |y| (-1..=1).map(move |x| IVec3::new(x, y, z)))
        })
        .collect()
}

/// The reference light, sky and voxel, of [`shaft_and_caves`] at a sky reach
/// and the default voxel reach, for the chunks around the origin, worked out
/// once per sky reach.
fn reference(sky_reach: u8) -> &'static (Reference, Reference) {
    static REFERENCES: OnceLock<HashMap<u8, (Reference, Reference)>> = OnceLock::new();
    let references = REFERENCES.get_or_init(|| {
        [30, 14]
            .into_iter()
            .map(|reach| {
                // Far enough past the chunks that light from outside it can't
                // reach them.
                let margin = IVec3::splat(reach as i32 + 1);
                let (min, max) = (IVec3::splat(-32) - margin, IVec3::splat(64) + margin);
                let sky = Reference::sky_light(
                    min,
                    max,
                    reach as u32,
                    LightShape::Round,
                    &Sky::default(),
                    shaft_and_caves,
                );
                let voxel = Reference::voxel_light(
                    min,
                    max,
                    14,
                    LightShape::Round,
                    shaft_and_caves,
                    |material| (material == LAMP).then_some(15.0),
                );
                (reach, (sky, voxel))
            })
            .collect()
    });
    &references[&sky_reach]
}

/// Fails if any voxel of the chunks around the origin reads brighter than the
/// reference.
fn assert_no_brighter(app: &mut App, reference: &(Reference, Reference), when: &str) {
    for chunk in around_origin() {
        let Some(light) = chunk_light(app, chunk) else {
            continue;
        };
        for (p, levels) in light {
            let (sky, voxel) =
                (reference.0.sky_at(p) as f32, reference.1.voxel_at(p) as f32);
            assert!(
                levels.sky <= sky && levels.voxel <= voxel,
                "{levels:?} at {p} is brighter than sky {sky} and voxel {voxel} {when}"
            );
        }
    }
}

/// Runs `app` until `done` holds for a few frames in a row, checking every
/// frame that no reading is brighter than the reference.
fn run_checked(
    app: &mut App,
    reference: &(Reference, Reference),
    what: &str,
    mut done: impl FnMut(&mut App) -> bool,
) {
    let deadline = Instant::now() + Duration::from_secs(60);
    let mut frames_done = 0;
    while frames_done < 3 {
        assert!(Instant::now() < deadline, "timed out waiting: {what}");
        app.update();
        assert_no_brighter(app, reference, what);
        frames_done = if done(app) { frames_done + 1 } else { 0 };
    }
}

/// Streams the world of [`shaft_and_caves`] at `sky_reach`, holding back the
/// chunk at `held` until everything else has settled, then checks every chunk
/// around the origin whose whole box is loaded against the reference.
fn settles_on_the_reference(sky_reach: u8, held: Option<IVec3>) {
    let _holding = held.map(|_| Hold::exclusive());
    let reference = reference(sky_reach);
    let hold = held.map(Hold::new);
    let mut app = app(LampWorld {
        hold: hold.clone(),
        ..shaft_world(sky_light(Sky::default(), sky_reach))
    });
    run_checked(&mut app, reference, "for all but the held chunk", |app| {
        settled_but(app, held)
    });
    if let Some(hold) = hold {
        hold.release();
        run_checked(&mut app, reference, "for the held chunk", |app| {
            settled_but(app, None)
        });
    }

    let spawned = spawned(&mut app);
    let checked: Vec<IVec3> = around_origin()
        .into_iter()
        .filter(|chunk| {
            around_origin()
                .iter()
                .all(|offset| spawned.contains(&(*chunk + *offset)))
        })
        .collect();
    assert!(
        checked.contains(&IVec3::ZERO),
        "the origin's box isn't loaded"
    );
    for chunk in checked {
        let light = chunk_light(&mut app, chunk).expect("the chunk's light");
        for (p, levels) in light {
            assert_eq!(
                (levels.sky, levels.voxel),
                (reference.0.sky_at(p) as f32, reference.1.voxel_at(p) as f32),
                "at {p} in {chunk}, with {held:?} held, at sky reach {sky_reach}"
            );
        }
    }
}

#[test]
fn after_streaming_settles_every_chunk_whose_whole_box_is_loaded_matches_the_reference() {
    settles_on_the_reference(30, None);
    settles_on_the_reference(14, None);
}

#[test]
fn in_hostile_arrival_orders_light_settles_on_the_reference() {
    // The shaft's chunk, the surface's chunk above it, and a chunk the
    // corridor runs through.
    for held in [IVec3::ZERO, IVec3::Y, IVec3::X] {
        settles_on_the_reference(30, Some(held));
    }
    settles_on_the_reference(14, Some(IVec3::Y));
}

#[test]
fn light_at_reads_sky_light_and_direct_sky_exactly_at_15() {
    let scene: Scene = Arc::new(shaft_and_caves);
    let mut app = settled(LampWorld {
        scene,
        ..shaft_world(sky_light(Sky::default(), 30))
    });
    for p in [
        IVec3::new(8, 20, 8),
        IVec3::new(9, 6, 8),
        IVec3::new(30, 6, 8),
        IVec3::new(50, 6, 8),
        IVec3::new(3, 45, 3),
    ] {
        let levels = levels(&mut app, p);
        assert_eq!(levels.direct_sky(), levels.sky == 15.0, "at {p}");
    }
    // A solid voxel reads 0 in sky light.
    assert_eq!(levels(&mut app, IVec3::new(8, 4, 8)).sky, 0.0);
}
