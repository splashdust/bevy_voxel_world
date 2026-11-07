//! Tests of hinted air: chunk positions a `Uniform(Air)` fill hint keeps from
//! spawning, which light reads as known air, read through `light_at`.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use bevy::ecs::system::RunSystemOnce;
use bevy::prelude::*;

use super::edits::{set_voxels, settle};
use super::reference::Reference;
use super::sky_light::sky_light;
use super::voxel_light::{app, light_at, Hints, Hold, LampWorld, LAMP, STONE};
use crate::prelude::*;

/// Where a world's fill hints say a chunk position is air.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Hinting {
    /// Every chunk position above the highest solid of its footprint.
    AboveHighestSolid,
    /// Every chunk position whose voxels are all air.
    EveryAllAirPosition,
}

const HINTINGS: [Hinting; 2] = [Hinting::AboveHighestSolid, Hinting::EveryAllAirPosition];

/// Stone below y = 0, and an island of stone filling the chunk position
/// (2, 1, 0) from y = 32 to 40, so its underside lies on a chunk boundary
/// over a chunk position of air.
fn island(p: IVec3) -> WorldVoxel {
    let island =
        (64..96).contains(&p.x) && (32..40).contains(&p.y) && (0..32).contains(&p.z);
    if p.y < 0 || island {
        WorldVoxel::Solid(STONE)
    } else {
        WorldVoxel::Air
    }
}

/// The fill hints of [`island`] under `hinting`.
fn island_hints(hinting: Hinting) -> Hints {
    Arc::new(move |position: IVec3| {
        let footprint_top = if position.xz() == IVec2::new(2, 0) {
            1
        } else {
            -1
        };
        let air = match hinting {
            Hinting::AboveHighestSolid => position.y > footprint_top,
            Hinting::EveryAllAirPosition => {
                position.y >= 0 && position != IVec3::new(2, 1, 0)
            }
        };
        if air {
            FillHint::Uniform(WorldVoxel::Air)
        } else {
            FillHint::Unknown
        }
    })
}

fn island_world(hinting: Hinting) -> LampWorld {
    LampWorld {
        scene: Arc::new(island),
        light: Some(sky_light(Sky::default(), 30)),
        hints: Some(island_hints(hinting)),
        spawning_distance: 3,
        ..default()
    }
}

/// The reference light, sky and voxel, of [`island`] around the chunk
/// positions it's checked in.
fn island_reference() -> &'static (Reference, Reference) {
    static REFERENCE: OnceLock<(Reference, Reference)> = OnceLock::new();
    REFERENCE.get_or_init(|| {
        let margin = IVec3::splat(31);
        let (min, max) = (
            IVec3::new(0, -32, -32) - margin,
            IVec3::new(128, 96, 64) + margin,
        );
        (
            Reference::sky_light(
                min,
                max,
                30,
                LightShape::Round,
                &Sky::default(),
                island,
            ),
            Reference::voxel_light(min, max, 14, LightShape::Round, island, |_| None),
        )
    })
}

/// What `light_at` reads at every voxel of the chunk positions `chunks`.
fn light_in(app: &mut App, chunks: &[IVec3]) -> Vec<(IVec3, Option<LightLevels>)> {
    let chunks = chunks.to_vec();
    app.world_mut()
        .run_system_once(move |voxel_world: VoxelWorld<LampWorld>| {
            chunks
                .iter()
                .flat_map(|chunk| {
                    super::voxel_light::origin_voxels()
                        .map(move |local| *chunk * 32 + local)
                })
                .map(|p| (p, voxel_world.light_at(p)))
                .collect::<Vec<_>>()
        })
        .unwrap()
}

/// Fails if any voxel of the chunk positions `chunks` reads brighter than
/// `reference` through `light_at`.
fn assert_no_brighter(
    app: &mut App,
    chunks: &[IVec3],
    reference: &(Reference, Reference),
    when: &str,
) {
    for (p, got) in light_in(app, chunks) {
        let Some(levels) = got else {
            continue;
        };
        let (sky, voxel) = (reference.0.sky_at(p) as f32, reference.1.voxel_at(p) as f32);
        assert!(
            levels.sky <= sky && levels.voxel <= voxel,
            "{levels:?} at {p} is brighter than sky {sky} and voxel {voxel} {when}"
        );
    }
}

/// Fails unless every voxel of the chunk positions `chunks` reads the light of
/// `reference` through `light_at`.
fn assert_light_matches(
    app: &mut App,
    chunks: &[IVec3],
    reference: &(Reference, Reference),
    when: &str,
) {
    let mut wrong = Vec::new();
    for (p, got) in light_in(app, chunks) {
        let want = (reference.0.sky_at(p) as f32, reference.1.voxel_at(p) as f32);
        let got = got.map(|levels| (levels.sky, levels.voxel));
        if got != Some(want) {
            wrong.push((p, got, want));
        }
    }
    assert!(
        wrong.is_empty(),
        "{} voxels read other light than the reference {when}, such as {:?}",
        wrong.len(),
        &wrong[..wrong.len().min(6)]
    );
}

/// Under the island, beside it and in the hinted air around it.
const AROUND_ISLAND: [IVec3; 4] = [
    IVec3::new(2, 0, 0),
    IVec3::new(1, 1, 0),
    IVec3::new(2, 1, 0),
    IVec3::new(1, 0, 0),
];

#[test]
fn light_crossing_hinted_air_and_faces_beside_it_match_the_reference() {
    let mut light: HashMap<Hinting, Vec<Option<LightLevels>>> = HashMap::new();
    for hinting in HINTINGS {
        let mut app = app(island_world(hinting));
        settle(&mut app, "the island to stream in");
        assert_light_matches(
            &mut app,
            &AROUND_ISLAND,
            island_reference(),
            &format!("hinting {hinting:?}"),
        );
        let read = light_in(&mut app, &AROUND_ISLAND);
        light.insert(
            hinting,
            read.into_iter().map(|(_, levels)| levels).collect(),
        );
    }
    assert_eq!(
        light[&Hinting::AboveHighestSolid],
        light[&Hinting::EveryAllAirPosition]
    );
}

/// The reference light of [`island`] with `edits` over it.
fn edited_island_reference(edits: &[(IVec3, WorldVoxel)]) -> (Reference, Reference) {
    let edits: HashMap<IVec3, WorldVoxel> = edits.iter().copied().collect();
    let scene = move |p: IVec3| edits.get(&p).copied().unwrap_or_else(|| island(p));
    let margin = IVec3::splat(31);
    let (min, max) = (
        IVec3::new(0, -32, -32) - margin,
        IVec3::new(128, 96, 64) + margin,
    );
    (
        Reference::sky_light(min, max, 30, LightShape::Round, &Sky::default(), &scene),
        Reference::voxel_light(min, max, 14, LightShape::Round, &scene, |material| {
            (material == LAMP).then_some(15.0)
        }),
    )
}

#[test]
fn voxel_edits_in_and_beside_hinted_air_settle_on_the_reference() {
    let edits = [
        // A lamp in the hinted air east of the island, which never spawns.
        (IVec3::new(100, 36, 16), WorldVoxel::Solid(LAMP)),
        // A lamp in the hinted air under the island.
        (IVec3::new(70, 4, 10), WorldVoxel::Solid(LAMP)),
        // A block on the ground beside the hinted air under the island.
        (IVec3::new(63, 0, 20), WorldVoxel::Solid(STONE)),
        // A hole through the island.
        (IVec3::new(80, 32, 16), WorldVoxel::Air),
        (IVec3::new(80, 33, 16), WorldVoxel::Air),
    ];
    let undo: Vec<(IVec3, WorldVoxel)> =
        edits.iter().map(|(p, _)| (*p, island(*p))).collect();
    let edited = edited_island_reference(&edits);
    let checked: Vec<IVec3> = AROUND_ISLAND
        .iter()
        .copied()
        .chain([IVec3::new(3, 1, 0)])
        .collect();
    for hinting in HINTINGS {
        let mut app = app(island_world(hinting));
        settle(&mut app, "the island to stream in");
        set_voxels(&mut app, &edits);
        settle(&mut app, "light to settle after the edits");
        assert_light_matches(
            &mut app,
            &checked,
            &edited,
            &format!("after the edits, hinting {hinting:?}"),
        );
        set_voxels(&mut app, &undo);
        settle(&mut app, "light to settle after the undo");
        assert_light_matches(
            &mut app,
            &checked,
            island_reference(),
            &format!("after the undo, hinting {hinting:?}"),
        );
    }
}

/// Runs `app` until every spawned chunk but `held` is Ready and light has
/// settled, checking every frame that no voxel of `chunks` reads brighter than
/// `reference`.
fn settle_checked(
    app: &mut App,
    chunks: &[IVec3],
    reference: &(Reference, Reference),
    held: Option<IVec3>,
    what: &str,
) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
    let mut frames_done = 0;
    while frames_done < 3 {
        assert!(
            std::time::Instant::now() < deadline,
            "timed out waiting: {what}"
        );
        app.update();
        assert_no_brighter(app, chunks, reference, what);
        frames_done = if super::sky_light::settled_but(app, held) {
            frames_done + 1
        } else {
            0
        };
    }
}

#[test]
fn light_with_hinted_air_is_never_brighter_than_the_reference_in_hostile_orders_and_retirements(
) {
    let _holding = Hold::exclusive();
    let reference = island_reference();
    // The island's chunk, and the chunk beside the hinted air under it.
    let island_chunk = IVec3::new(2, 1, 0);
    for held in [island_chunk, IVec3::new(1, 0, 0)] {
        for hinting in HINTINGS {
            let when = format!("with {held} held, hinting {hinting:?}");
            let hold = Hold::new(held);
            let mut app = app(LampWorld {
                hold: Some(hold.clone()),
                ..island_world(hinting)
            });
            // Hinted air under the island shows provisional direct sky until
            // the island's chunk publishes or a sky probe reads it, as a chunk
            // there that skipped its walk would: no lookups prove its columns.
            // Streaming is checked frame by frame only where no hinted air
            // lies under the island, and at rest otherwise.
            let checked: &[IVec3] = if hinting == Hinting::EveryAllAirPosition {
                &[]
            } else {
                &AROUND_ISLAND
            };
            settle_checked(&mut app, checked, reference, Some(held), &when);
            hold.release();
            settle_checked(&mut app, checked, reference, None, &when);
            assert_light_matches(&mut app, &AROUND_ISLAND, reference, &when);

            // The island's side of the world retires a chunk at a time.
            for x in [-1, -2, -4] {
                camera_at(&mut app, IVec3::new(x, 0, 0));
                settle_checked(
                    &mut app,
                    &AROUND_ISLAND,
                    reference,
                    None,
                    &format!("{when}, with the camera at x = {x}"),
                );
            }
        }
    }
}

#[test]
fn a_settings_change_relights_light_entries() {
    let mut app = app(island_world(Hinting::EveryAllAirPosition));
    settle(&mut app, "the island to stream in");
    app.world_mut().resource_mut::<LampWorld>().light =
        Some(sky_light(Sky::default(), 16));
    settle(&mut app, "light to settle at sky reach 16");
    let margin = IVec3::splat(31);
    let (min, max) = (
        IVec3::new(0, -32, -32) - margin,
        IVec3::new(128, 96, 64) + margin,
    );
    let reference = (
        Reference::sky_light(min, max, 16, LightShape::Round, &Sky::default(), island),
        Reference::voxel_light(min, max, 14, LightShape::Round, island, |_| None),
    );
    assert_light_matches(&mut app, &AROUND_ISLAND, &reference, "at sky reach 16");
}

/// Stone below y = 0, under a slab of stone in the chunk position (2, 6, 0),
/// far above, which shades the ground under it.
fn ceiling(p: IVec3) -> WorldVoxel {
    let slab =
        (64..96).contains(&p.x) && (200..208).contains(&p.y) && (0..32).contains(&p.z);
    if p.y < 0 || slab {
        WorldVoxel::Solid(STONE)
    } else {
        WorldVoxel::Air
    }
}

/// The slab's chunk position, which holds the only solid above the ground.
const SLAB: IVec3 = IVec3::new(2, 6, 0);

#[test]
fn hinted_air_whose_direct_sky_is_provisional_settles_on_the_reference() {
    let _holding = Hold::exclusive();
    let hold = Hold::inside(SLAB);
    let mut app = app(LampWorld {
        scene: Arc::new(ceiling),
        light: Some(sky_light(Sky::default(), 30)),
        hints: Some(Arc::new(|position: IVec3| {
            if position.y >= 0 && position != SLAB {
                FillHint::Uniform(WorldVoxel::Air)
            } else {
                FillHint::Unknown
            }
        })),
        spawning_distance: 3,
        hold: Some(hold.clone()),
        ..default()
    });
    super::super::update_until(&mut app, "a sky probe to read the slab", |app| {
        hold.waited.load(std::sync::atomic::Ordering::Acquire) > 0
            && super::sky_light::spawned(app).contains(&IVec3::new(2, -1, 0))
    });
    for _ in 0..20 {
        app.update();
    }
    // Under the slab, nothing has proved the columns of the hinted air yet.
    assert!(!spawned_at(&mut app, IVec3::new(2, 0, 0)));
    assert_eq!(
        light_at(&mut app, IVec3::new(80, 0, 16)).map(|levels| levels.sky),
        Some(15.0),
        "on the ground under the slab before the probe"
    );

    hold.release();
    settle(&mut app, "the probe to disprove the columns");
    let margin = IVec3::splat(31);
    let (min, max) = (
        IVec3::new(0, -32, -32) - margin,
        IVec3::new(128, 64, 64) + margin,
    );
    let reference = (
        Reference::sky_light(min, max, 30, LightShape::Round, &Sky::default(), ceiling),
        Reference::voxel_light(min, max, 14, LightShape::Round, ceiling, |_| None),
    );
    assert_light_matches(
        &mut app,
        &[IVec3::new(2, 0, 0), IVec3::new(1, 0, 0)],
        &reference,
        "once the probe has read the slab",
    );
}

/// Air everywhere but a roof of stone in the chunk position (1, 0, 0) whose
/// east edge touches the chunk position (2, 0, 0), and slabs of stone far
/// above the two chunk positions (2, 0, 0) and (2, 0, 1), which shade them.
fn roof_under_slabs(p: IVec3) -> WorldVoxel {
    let roof =
        (40..64).contains(&p.x) && (24..28).contains(&p.y) && (4..31).contains(&p.z);
    let slabs =
        (64..96).contains(&p.x) && (200..208).contains(&p.y) && (0..64).contains(&p.z);
    if roof || slabs {
        WorldVoxel::Solid(STONE)
    } else {
        WorldVoxel::Air
    }
}

#[test]
fn a_column_disproved_in_hinted_air_relights_the_light_around_it() {
    let _holding = Hold::exclusive();
    // The slab over the hinted air at (2, 0, 1), which touches no loaded
    // solid and so holds no light entry.
    let slab = IVec3::new(2, 6, 1);
    let hold = Hold::inside(slab);
    let mut app = app(LampWorld {
        scene: Arc::new(roof_under_slabs),
        light: Some(sky_light(Sky::default(), 30)),
        hints: Some(Arc::new(|position: IVec3| {
            let solid = [IVec3::X, IVec3::new(2, 6, 0), IVec3::new(2, 6, 1)];
            if solid.contains(&position) {
                FillHint::Unknown
            } else {
                FillHint::Uniform(WorldVoxel::Air)
            }
        })),
        hold: Some(hold.clone()),
        ..default()
    });
    super::super::update_until(&mut app, "a sky probe to read the slab", |_| {
        hold.waited.load(std::sync::atomic::Ordering::Acquire) > 0
    });
    // Everything else lands meanwhile, reading the provisional direct sky of
    // the hinted air under the held slab.
    for _ in 0..40 {
        app.update();
    }

    hold.release();
    settle(&mut app, "the probe to disprove the columns");
    let margin = IVec3::splat(31);
    let (min, max) = (
        IVec3::new(0, -32, -32) - margin,
        IVec3::new(128, 32, 64) + margin,
    );
    let reference = (
        Reference::sky_light(
            min,
            max,
            30,
            LightShape::Round,
            &Sky::default(),
            roof_under_slabs,
        ),
        Reference::voxel_light(min, max, 14, LightShape::Round, roof_under_slabs, |_| {
            None
        }),
    );
    assert_light_matches(
        &mut app,
        &[IVec3::new(2, 0, 0), IVec3::X],
        &reference,
        "once the probe has read the slab",
    );
}

// --- Light entries ----------------------------------------------------------

/// A wall of stone 4 voxels thick from x = 96, the west side of the chunk
/// positions at x = 3, in a world whose chunk positions west of it are hinted
/// air.
fn wall(p: IVec3) -> WorldVoxel {
    if (96..100).contains(&p.x) {
        WorldVoxel::Solid(STONE)
    } else {
        WorldVoxel::Air
    }
}

fn wall_world() -> LampWorld {
    LampWorld {
        scene: Arc::new(wall),
        light: Some(sky_light(Sky::default(), 30)),
        hints: Some(Arc::new(|position: IVec3| {
            if position.x < 3 {
                FillHint::Uniform(WorldVoxel::Air)
            } else {
                FillHint::Unknown
            }
        })),
        spawning_distance: 2,
        ..default()
    }
}

/// Moves the camera to the middle of the chunk at `chunk`, once it has
/// spawned.
fn camera_at(app: &mut App, chunk: IVec3) {
    app.update();
    let middle = (chunk * 32 + 16).as_vec3();
    super::super::move_cameras::<LampWorld>(
        app,
        Transform::from_translation(middle).looking_at(middle + Vec3::NEG_Z, Vec3::Y),
    );
}

/// The chunk position holding the wall.
const WALL: IVec3 = IVec3::new(3, 0, 0);

/// The hinted air west of the wall.
const WEST_OF_WALL: IVec3 = IVec3::new(2, 0, 0);

/// Beside the wall, in the hinted air west of it.
const BESIDE_WALL: IVec3 = IVec3::new(95, 16, 16);

fn spawned_at(app: &mut App, chunk: IVec3) -> bool {
    super::sky_light::spawned(app).contains(&chunk)
}

#[test]
fn a_light_entry_is_made_where_a_loaded_solid_touches_hinted_air_and_dropped_when_none_does(
) {
    let mut app = app(wall_world());
    camera_at(&mut app, IVec3::new(4, 0, 0));
    settle(&mut app, "the wall to stream in");
    // The hinted air beside the wall never spawned.
    assert!(spawned_at(&mut app, WALL));
    assert!(!spawned_at(&mut app, WEST_OF_WALL));
    assert_eq!(
        light_at(&mut app, BESIDE_WALL),
        Some(LightLevels::new(15, 0)),
        "beside the wall"
    );
    // Hinted air no loaded solid touches holds no light entry, and reads
    // the walk on request.
    assert!(!holds_entry(&app, WEST_OF_WALL - IVec3::X));
    assert_eq!(
        light_at(&mut app, BESIDE_WALL - IVec3::X * 32),
        Some(LightLevels::new(15, 0))
    );

    camera_at(&mut app, IVec3::new(8, 0, 0));
    settle(&mut app, "the wall to retire");
    assert!(!spawned_at(&mut app, WALL));
    assert!(!holds_entry(&app, WEST_OF_WALL), "with the wall retired");
    assert_eq!(
        light_at(&mut app, BESIDE_WALL),
        Some(LightLevels::new(15, 0)),
        "with the wall retired"
    );
}

/// Whether the world view holds a light entry at the chunk position
/// `position`.
fn holds_entry(app: &App, position: IVec3) -> bool {
    app.world()
        .resource::<crate::light::LightSettings<LampWorld>>()
        .view
        .read()
        .get(&position)
        .is_some_and(|chunk| chunk.is_entry())
}

#[test]
fn a_chunk_spawned_in_hinted_air_leaves_a_light_entry_with_its_light_as_it_retires() {
    let mut app = app(wall_world());
    // The hinted air west of the wall spawns in the camera's protected radius.
    camera_at(&mut app, WEST_OF_WALL);
    settle(&mut app, "the wall to stream in");
    assert!(spawned_at(&mut app, WEST_OF_WALL));
    let before = light_at(&mut app, BESIDE_WALL);
    assert_eq!(before, Some(LightLevels::new(15, 0)));

    camera_at(&mut app, IVec3::new(5, 0, 0));
    settle(&mut app, "the chunk in hinted air to retire");
    assert!(!spawned_at(&mut app, WEST_OF_WALL));
    assert!(spawned_at(&mut app, WALL));
    assert_eq!(light_at(&mut app, BESIDE_WALL), before);
}
