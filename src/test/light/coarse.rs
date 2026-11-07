//! Tests of light in coarse chunks: chunks above LOD level 0, and level-0
//! chunks with a data shape other than the default.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use bevy::ecs::system::RunSystemOnce;
use bevy::mesh::MeshTag;
use bevy::prelude::*;

use super::super::{
    chunk_status_of, meshing_app, move_cameras, spawn_camera, update_until,
};
use super::voxel_light::Scene;
use crate::chunk::CHUNK_SIZE_F;
use crate::prelude::*;

const HAS_SLOT: u32 = 1 << 31;
const STONE: u8 = 1;

/// A lit world of `scene`, at level 0 within `near` chunks of the camera
/// and at level 1 beyond, whose level-0 chunks hold `data` data voxels a
/// side.
#[derive(Resource, Clone)]
struct CoarseWorld {
    scene: Scene,
    near: i32,
    data: u32,
    hints: Option<KeyHints>,
}

/// A world's fill hint for each chunk key, at every level.
type KeyHints = Arc<dyn Fn(ChunkKey) -> FillHint + Send + Sync>;

impl Default for CoarseWorld {
    fn default() -> Self {
        Self {
            scene: Arc::new(slab_and_shaft),
            near: 2,
            data: 32,
            hints: None,
        }
    }
}

impl VoxelWorldConfig for CoarseWorld {
    type MaterialIndex = u8;
    type ChunkUserBundle = ();

    fn spawning_distance(&self) -> u32 {
        3
    }

    fn chunk_spawn_strategy(&self) -> ChunkSpawnStrategy {
        ChunkSpawnStrategy::Close
    }

    fn chunk_despawn_strategy(&self) -> ChunkDespawnStrategy {
        ChunkDespawnStrategy::FarAway
    }

    fn voxel_lookup_delegate(&self) -> VoxelLookupDelegate<Self::MaterialIndex> {
        let scene = self.scene.clone();
        Box::new(move |_, _, _| {
            let scene = scene.clone();
            Box::new(move |position, _| scene(position))
        })
    }

    fn chunk_fill_hint(&self, key: ChunkKey) -> FillHint {
        self.hints
            .as_ref()
            .map_or(FillHint::Unknown, |hints| hints(key))
    }

    fn chunk_lod(
        &self,
        chunk_position: IVec3,
        _previous_lod: Option<LodLevel>,
        camera_position: Vec3,
    ) -> LodLevel {
        let camera_chunk = (camera_position / CHUNK_SIZE_F).floor().as_ivec3();
        if chunk_position.distance_squared(camera_chunk) < self.near * self.near {
            0
        } else {
            1
        }
    }

    fn chunk_data_shape(&self, lod_level: LodLevel) -> UVec3 {
        if lod_level == 0 {
            padded_chunk_shape_uniform(self.data)
        } else {
            padded_chunk_shape_uniform(32)
        }
    }

    fn light_propagation(&self) -> Option<LightPropagation> {
        Some(LightPropagation {
            sky_fill: None,
            ..default()
        })
    }
}

/// Ground below y = 0 with a shaft 8 voxels wide down to y = -20, under a
/// slab floating at y = 10 that shades part of a level-1 chunk. The camera of
/// [`app`] is at level 0 within two chunks of the origin.
fn slab_and_shaft(p: IVec3) -> WorldVoxel {
    let slab =
        (64..96).contains(&p.x) && (10..12).contains(&p.y) && (0..64).contains(&p.z);
    let shaft =
        (104..112).contains(&p.x) && (-20..0).contains(&p.y) && (24..32).contains(&p.z);
    if slab || (p.y < 0 && !shaft) {
        WorldVoxel::Solid(STONE)
    } else {
        WorldVoxel::Air
    }
}

const SHADED: IVec3 = IVec3::new(80, 4, 30);
const SUNLIT: IVec3 = IVec3::new(110, 4, 30);
const IN_SHAFT: IVec3 = IVec3::new(106, -10, 26);

/// The level-1 chunks above and below y = 0 that the slab and the shaft lie
/// in.
const ABOVE: ChunkKey = ChunkKey {
    position: IVec3::new(2, 0, 0),
    lod_level: 1,
};
const BELOW: ChunkKey = ChunkKey {
    position: IVec3::new(2, -2, 0),
    lod_level: 1,
};

fn app(scene: Scene, data: u32) -> App {
    let mut app = meshing_app();
    app.add_plugins(VoxelWorldPlugin::with_config(CoarseWorld {
        scene,
        near: 2,
        data,
        hints: None,
    }))
    .add_systems(Startup, spawn_camera::<CoarseWorld>);
    app
}

fn light_at(app: &mut App, position: IVec3) -> Option<LightLevels> {
    app.world_mut()
        .run_system_once(move |voxel_world: VoxelWorld<CoarseWorld>| {
            voxel_world.light_at(position)
        })
        .unwrap()
}

fn sky_at(app: &mut App, position: IVec3) -> f32 {
    light_at(app, position)
        .unwrap_or_else(|| panic!("no light at {position}"))
        .sky
}

fn tag_of(app: &mut App, key: ChunkKey) -> Option<u32> {
    app.world_mut()
        .query::<(&Chunk<CoarseWorld>, &MeshTag)>()
        .iter(app.world())
        .find(|(chunk, _)| chunk.key() == key)
        .map(|(_, tag)| tag.value)
}

fn coarse_lit(app: &mut App, positions: &[IVec3]) -> bool {
    positions
        .iter()
        .all(|position| light_at(app, *position).is_some())
}

/// The slab-and-shaft world, run until both level-1 chunks show their meshes
/// and their light has landed.
fn slab_app() -> App {
    let mut app = app(Arc::new(slab_and_shaft), 32);
    update_until(&mut app, "the coarse chunks to show their light", |app| {
        tag_of(app, ABOVE).is_some()
            && tag_of(app, BELOW).is_some()
            && coarse_lit(app, &[SHADED, SUNLIT, IN_SHAFT])
    });
    app
}

#[test]
fn a_coarse_chunks_tag_carries_its_lod_level_beside_its_slot() {
    let mut app = slab_app();
    update_until(&mut app, "the shaded coarse chunk to take a slot", |app| {
        tag_of(app, ABOVE).is_some_and(|tag| tag & HAS_SLOT != 0)
    });

    let tag = tag_of(&mut app, ABOVE).unwrap();
    assert_eq!(tag >> 24 & 15, 1, "the LOD level in bits 24-27 of {tag:#x}");
    assert_eq!(tag & 0x7000_0000, 0, "bits 28-30 of {tag:#x}");
}

#[test]
fn light_at_above_level_0_reads_a_slot_cell_and_a_tag_only_chunk() {
    let mut app = slab_app();

    let shaded = sky_at(&mut app, SHADED);
    assert!(
        shaded > 0.0 && shaded < 15.0,
        "under the slab sky light comes from its sides, {shaded}"
    );
    assert_eq!(sky_at(&mut app, SUNLIT), 15.0);
    assert_eq!(
        sky_at(&mut app, IN_SHAFT),
        15.0,
        "sky light falls down the shaft"
    );

    // The flat ground on the other side of the camera: every face sees
    // direct sky, so the chunk shows its tag alone and reads its level.
    let flat = ChunkKey::new(IVec3::new(-2, -2, 2), 1);
    update_until(&mut app, "the flat coarse chunk to show", |app| {
        tag_of(app, flat).is_some()
    });
    assert_eq!(tag_of(&mut app, flat), Some(15), "no slot, sky 15, voxel 0");
    let in_flat = IVec3::new(-50, -1, 80);
    assert_eq!(light_at(&mut app, in_flat), Some(LightLevels::new(15, 0)));
}

#[test]
fn reading_across_a_split_stays_plausible() {
    let mut app = slab_app();
    let coarse = [SHADED, SUNLIT].map(|position| sky_at(&mut app, position));

    // Into the shaded chunk, which splits into level-0 chunks.
    move_cameras::<CoarseWorld>(&mut app, Transform::from_xyz(80.0, 20.0, 30.0));
    update_until(&mut app, "the shaded chunk to split and settle", |app| {
        chunk_status_of::<CoarseWorld>(app, ABOVE).is_none()
            && coarse_lit(app, &[SHADED, SUNLIT])
            && light_is_settled(app)
    });
    let fine = [SHADED, SUNLIT].map(|position| sky_at(&mut app, position));
    for (coarse, fine) in coarse.iter().zip(&fine) {
        assert!(
            (coarse - fine).abs() <= 3.0,
            "coarse {coarse:?} and fine {fine:?} differ by more than the coarse rule allows"
        );
    }
}

fn light_is_settled(app: &mut App) -> bool {
    app.world_mut()
        .run_system_once(|voxel_world: VoxelWorld<CoarseWorld>| {
            voxel_world.light_is_settled()
        })
        .unwrap()
}

#[test]
fn an_edit_outside_a_coarse_chunks_padding_changes_nothing_there_until_its_task_runs_again(
) {
    let mut app = slab_app();
    assert_eq!(sky_at(&mut app, IN_SHAFT), 15.0);

    // A cap far above the shaft, in the chunk above, outside the shaft's
    // chunk and its padding.
    let cap = IVec3::new(96, 40, 16)..IVec3::new(120, 42, 40);
    app.world_mut()
        .run_system_once(move |mut voxel_world: VoxelWorld<CoarseWorld>| {
            for x in cap.start.x..cap.end.x {
                for y in cap.start.y..cap.end.y {
                    for z in cap.start.z..cap.end.z {
                        voxel_world
                            .set_voxel(IVec3::new(x, y, z), WorldVoxel::Solid(STONE));
                    }
                }
            }
        })
        .unwrap();
    let capped = IVec3::new(106, 40, 26);
    update_until(&mut app, "the chunk above to show the cap", |app| {
        app.world_mut()
            .run_system_once(move |voxel_world: VoxelWorld<CoarseWorld>| {
                voxel_world
                    .get_chunk_data(ABOVE)
                    .and_then(|data| data.get_voxel_at_world_position(capped))
            })
            .unwrap()
            == Some(WorldVoxel::Solid(STONE))
    });
    for _ in 0..10 {
        app.update();
    }
    assert_eq!(
        sky_at(&mut app, IN_SHAFT),
        15.0,
        "the shaft's chunk didn't run again"
    );

    // An edit inside the shaft's chunk runs its task again, whose walk finds
    // the cap.
    app.world_mut()
        .run_system_once(|mut voxel_world: VoxelWorld<CoarseWorld>| {
            voxel_world.set_voxel(IVec3::new(70, -40, 10), WorldVoxel::Air);
        })
        .unwrap();
    update_until(&mut app, "the shaft to lose its direct sky", |app| {
        light_at(app, IN_SHAFT).is_some_and(|light| light.sky < 15.0)
    });
}

/// Ground below y = 0 with a roof over part of the chunk at the origin.
fn roofed(p: IVec3) -> WorldVoxel {
    let roof = (0..16).contains(&p.x) && (8..10).contains(&p.y) && (0..32).contains(&p.z);
    if roof || p.y < 0 {
        WorldVoxel::Solid(STONE)
    } else {
        WorldVoxel::Air
    }
}

#[test]
fn level_0_chunks_with_other_data_shapes_get_plausible_light() {
    for data in [16, 24, 64] {
        let mut app = app(Arc::new(roofed), data);
        let (under_roof, open) = (IVec3::new(4, 2, 16), IVec3::new(24, 2, 16));
        update_until(&mut app, "the origin chunk's light to land", |app| {
            coarse_lit(app, &[under_roof, open])
        });

        assert_eq!(sky_at(&mut app, open), 15.0, "{data} data voxels: open sky");
        let shaded = sky_at(&mut app, under_roof);
        assert!(
            shaded > 0.0 && shaded < 15.0,
            "{data} data voxels: under the roof, lit from its side, read {shaded}"
        );
    }
}

// --- Where a coarse cell is solid, or nothing renders --------------------------

#[test]
fn a_solid_coarse_cell_reads_its_brightest_open_neighbour_or_0_when_buried() {
    let mut app = slab_app();
    // The slab's top cell, a level-1 cell 2 voxels wide, under open sky.
    let in_slab = IVec3::new(80, 10, 30);
    let mut brightest = LightLevels::new(0, 0);
    for offset in [
        IVec3::X,
        IVec3::NEG_X,
        IVec3::Y,
        IVec3::NEG_Y,
        IVec3::Z,
        IVec3::NEG_Z,
    ] {
        let neighbour = in_slab + offset * 2;
        if slab_and_shaft(neighbour).is_solid() {
            continue;
        }
        let levels = light_at(&mut app, neighbour).unwrap();
        brightest.sky = brightest.sky.max(levels.sky);
        brightest.voxel = brightest.voxel.max(levels.voxel);
    }
    assert_eq!(brightest.sky, 15.0, "the cell above the slab sees the sky");
    assert_eq!(light_at(&mut app, in_slab), Some(brightest));

    // Deep in the ground every neighbour is solid.
    assert_eq!(
        light_at(&mut app, IVec3::new(80, -40, 40)),
        Some(LightLevels::new(0, 0))
    );
}

/// Ground below y = -64, and a roof 8 voxels thick at y = 80 over x 64..96,
/// z 0..32, far above the level-1 chunk at (2, 0, 0), which holds air alone.
fn high_roof(p: IVec3) -> WorldVoxel {
    let roof =
        (64..96).contains(&p.x) && (80..88).contains(&p.y) && (0..32).contains(&p.z);
    if roof || p.y < -64 {
        WorldVoxel::Solid(STONE)
    } else {
        WorldVoxel::Air
    }
}

/// Air hints for every chunk of [`high_roof`] holding air alone, at every
/// level.
fn high_roof_hints(key: ChunkKey) -> FillHint {
    let (min, max) = (key.min_voxel(), key.min_voxel() + key.voxel_extent());
    let misses_roof = max.x <= 64
        || min.x >= 96
        || max.y <= 80
        || min.y >= 88
        || max.z <= 0
        || min.z >= 32;
    if min.y >= -64 && misses_roof {
        FillHint::Uniform(WorldVoxel::Air)
    } else {
        FillHint::Unknown
    }
}

const UNDER_ROOF: IVec3 = IVec3::new(80, 30, 16);
const OPEN_COLUMN: IVec3 = IVec3::new(80, 30, 48);

fn high_roof_app(hints: Option<KeyHints>) -> App {
    let mut app = meshing_app();
    app.add_plugins(VoxelWorldPlugin::with_config(CoarseWorld {
        scene: Arc::new(high_roof),
        hints,
        ..default()
    }))
    .add_systems(Startup, spawn_camera::<CoarseWorld>);
    app
}

fn request_walks(app: &App) -> u64 {
    crate::light::bench::light_stats::<CoarseWorld>(app.world())
        .unwrap()
        .request_walks
}

#[test]
fn a_coarse_chunk_with_no_mesh_walks_the_column_once() {
    let mut app = high_roof_app(None);
    let empty = ChunkKey::new(IVec3::new(2, 0, 0), 1);
    update_until(&mut app, "the empty coarse chunk to publish", |app| {
        app.world_mut()
            .run_system_once(move |voxel_world: VoxelWorld<CoarseWorld>| {
                voxel_world.get_chunk_data(empty).is_some()
            })
            .unwrap()
    });

    let walks = request_walks(&app);
    assert_eq!(light_at(&mut app, UNDER_ROOF), Some(LightLevels::new(0, 0)));
    assert_eq!(
        light_at(&mut app, OPEN_COLUMN),
        Some(LightLevels::new(15, 0))
    );
    assert_eq!(request_walks(&app), walks + 2);
    assert_eq!(
        light_at(&mut app, OPEN_COLUMN),
        Some(LightLevels::new(15, 0))
    );
    assert_eq!(request_walks(&app), walks + 2, "the answer is kept");
}

#[test]
fn a_coarse_chunk_of_solid_reads_0_with_no_walk() {
    // Level-0 chunks of 16 data voxels a side take the coarse path, and the
    // one below the origin is solid through its padding.
    let mut app = app(
        Arc::new(|p: IVec3| {
            if p.y < 2 {
                WorldVoxel::Solid(STONE)
            } else {
                WorldVoxel::Air
            }
        }),
        16,
    );
    let buried = IVec3::new(5, -10, 5);
    update_until(&mut app, "the solid chunk to publish", |app| {
        light_at(app, buried).is_some()
    });
    let walks = request_walks(&app);
    assert_eq!(light_at(&mut app, buried), Some(LightLevels::new(0, 0)));
    assert_eq!(request_walks(&app), walks);
}

#[test]
fn hinted_air_of_a_lod_world_walks_the_column_and_follows_an_edit_above() {
    let mut app = high_roof_app(Some(Arc::new(high_roof_hints)));
    for _ in 0..10 {
        app.update();
    }
    let spawned = super::super::chunk_keys::<CoarseWorld>(&mut app);
    for position in [UNDER_ROOF, OPEN_COLUMN] {
        assert!(
            spawned
                .iter()
                .all(|key| ChunkKey::containing(position, key.lod_level) != *key),
            "no chunk spawned at {position}"
        );
    }
    assert_eq!(light_at(&mut app, UNDER_ROOF), Some(LightLevels::new(0, 0)));
    assert_eq!(
        light_at(&mut app, OPEN_COLUMN),
        Some(LightLevels::new(15, 0))
    );

    // A voxel set in the hinted air above a column already read.
    app.world_mut()
        .run_system_once(|mut voxel_world: VoxelWorld<CoarseWorld>| {
            voxel_world.set_voxel(OPEN_COLUMN + IVec3::Y * 40, WorldVoxel::Solid(STONE));
        })
        .unwrap();
    assert_eq!(
        light_at(&mut app, OPEN_COLUMN),
        Some(LightLevels::new(0, 0))
    );
    app.update();
    assert_eq!(
        light_at(&mut app, OPEN_COLUMN),
        Some(LightLevels::new(0, 0))
    );

    // Below the ground's hints nothing says what is there.
    assert_eq!(light_at(&mut app, IVec3::new(80, -1000, 16)), None);
}

#[test]
fn a_walk_on_request_in_a_world_with_slow_lookups_runs_once_per_voxel() {
    // Lookups take 20 µs each once the world has streamed in.
    static SLOW: AtomicBool = AtomicBool::new(false);
    let scene: Scene = Arc::new(|p| {
        if SLOW.load(Ordering::Relaxed) {
            std::thread::sleep(Duration::from_micros(20));
        }
        high_roof(p)
    });
    let mut app = meshing_app();
    app.add_plugins(VoxelWorldPlugin::with_config(CoarseWorld {
        scene,
        ..default()
    }))
    .add_systems(Startup, spawn_camera::<CoarseWorld>);
    let empty = ChunkKey::new(IVec3::new(2, 0, 0), 1);
    update_until(&mut app, "the empty coarse chunk to publish", |app| {
        app.world_mut()
            .run_system_once(move |voxel_world: VoxelWorld<CoarseWorld>| {
                voxel_world.get_chunk_data(empty).is_some()
            })
            .unwrap()
    });
    SLOW.store(true, Ordering::Relaxed);

    let lookups = |app: &App| {
        crate::light::bench::light_stats::<CoarseWorld>(app.world())
            .unwrap()
            .request_lookups
    };
    let before = lookups(&app);
    let started = Instant::now();
    assert_eq!(
        light_at(&mut app, OPEN_COLUMN),
        Some(LightLevels::new(15, 0))
    );
    let first = started.elapsed();
    let made = lookups(&app) - before;
    let started = Instant::now();
    assert_eq!(
        light_at(&mut app, OPEN_COLUMN),
        Some(LightLevels::new(15, 0))
    );
    let again = started.elapsed();
    SLOW.store(false, Ordering::Relaxed);

    let stats = crate::light::bench::light_stats::<CoarseWorld>(app.world()).unwrap();
    println!(
        "walk on request at level 1, lookups of 20 µs: {made} lookups, first reading \
         {first:?}, again {again:?}, walk cache {} B",
        stats.walk_cache_bytes
    );
    assert!(made > 0 && made <= 64, "{made} lookups");
    assert_eq!(
        lookups(&app),
        before + made,
        "the second reading walks nothing"
    );
}
