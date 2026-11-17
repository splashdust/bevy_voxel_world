//! Tests of light in coarse chunks: chunks above LOD level 0, and level-0
//! chunks with a data shape other than the default.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use bevy::ecs::system::RunSystemOnce;
use bevy::mesh::MeshTag;
use bevy::prelude::*;

use super::super::{
    chunk_keys, chunk_status_of, meshing_app, move_cameras, spawn_camera, update_until,
};
use super::voxel_light::{Hold, Scene, LAMP};
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
    light: LightPropagation,
    emissive: Vec<(u8, Emission)>,
    /// Holds back the padding lookups of the chunk task at its chunk
    /// position, at any level.
    hold: Option<Arc<Hold>>,
    /// The `min_despawn_distance`, within which chunks spawn whatever their
    /// fill hint.
    protected: u32,
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
            light: LightPropagation {
                sky_fill: None,
                ..default()
            },
            emissive: Vec::new(),
            hold: None,
            protected: 1,
        }
    }
}

impl VoxelWorldConfig for CoarseWorld {
    type MaterialIndex = u8;
    type ChunkUserBundle = ();

    fn spawning_distance(&self) -> u32 {
        3
    }

    fn min_despawn_distance(&self) -> u32 {
        self.protected
    }

    fn chunk_spawn_strategy(&self) -> ChunkSpawnStrategy {
        ChunkSpawnStrategy::Close
    }

    fn chunk_despawn_strategy(&self) -> ChunkDespawnStrategy {
        ChunkDespawnStrategy::FarAway
    }

    fn voxel_lookup_delegate(&self) -> VoxelLookupDelegate<Self::MaterialIndex> {
        let scene = self.scene.clone();
        let hold = self.hold.clone();
        Box::new(move |chunk, _, _| {
            let scene = scene.clone();
            let hold = hold.clone().filter(|hold| hold.chunk == chunk);
            Box::new(move |position, _| {
                // A chunk task generates its padding first.
                let inside = position.div_euclid(IVec3::splat(32)) == chunk;
                if let Some(hold) = hold.as_ref().filter(|_| !inside) {
                    hold.wait();
                }
                scene(position)
            })
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
        Some(self.light.clone())
    }

    fn emissive_materials(&self) -> Vec<(Self::MaterialIndex, Emission)> {
        self.emissive.clone()
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
        // The test app casts no spawning rays, and the flood fill that finds
        // chunks to spawn stops at chunks already spawned, so a level-1 chunk
        // two chunks out respawns after a merge only when this distance
        // holds it.
        protected: 2,
        ..default()
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
fn reading_across_a_split_and_a_merge_stays_plausible() {
    let mut app = slab_app();
    let coarse = [SHADED, SUNLIT].map(|position| sky_at(&mut app, position));
    let assert_plausible = |coarse: &[f32; 2], fine: &[f32; 2]| {
        for (coarse, fine) in coarse.iter().zip(fine) {
            assert!(
                (coarse - fine).abs() <= 3.0,
                "coarse {coarse:?} and fine {fine:?} differ by more than the coarse rule allows"
            );
        }
    };

    // Into the shaded chunk, which splits into level-0 chunks.
    move_cameras::<CoarseWorld>(&mut app, Transform::from_xyz(80.0, 20.0, 30.0));
    update_until(&mut app, "the shaded chunk to split and settle", |app| {
        chunk_status_of::<CoarseWorld>(app, ABOVE).is_none()
            && coarse_lit(app, &[SHADED, SUNLIT])
            && light_is_settled(app)
    });
    let fine = [SHADED, SUNLIT].map(|position| sky_at(&mut app, position));
    assert_plausible(&coarse, &fine);

    // Back to where the camera started, so the level-0 chunks merge into
    // the shaded chunk again.
    move_cameras::<CoarseWorld>(&mut app, Transform::from_xyz(10.0, 10.0, 10.0));
    update_until(
        &mut app,
        "the shaded chunk to merge back and settle",
        |app| {
            chunk_status_of::<CoarseWorld>(app, ABOVE).is_some()
                && tag_of(app, ABOVE).is_some()
                && coarse_lit(app, &[SHADED, SUNLIT])
                && light_is_settled(app)
        },
    );
    let merged = [SHADED, SUNLIT].map(|position| sky_at(&mut app, position));
    assert_plausible(&merged, &fine);
}

#[test]
fn a_change_of_precision_solves_coarse_chunks_again_at_the_new_precision() {
    let mut app = slab_app();
    update_until(&mut app, "the shaded coarse chunk to take a slot", |app| {
        tag_of(app, ABOVE).is_some_and(|tag| tag & HAS_SLOT != 0)
    });
    // Under the slab, where sky light falls off from its sides.
    let under_slab: Vec<IVec3> = (64..96).map(|x| IVec3::new(x, 4, 30)).collect();
    let fractions = |app: &mut App| {
        under_slab
            .iter()
            .filter(|position| sky_at(app, **position).fract() != 0.0)
            .count()
    };
    assert_eq!(fractions(&mut app), 0, "16 levels read whole levels");

    app.world_mut()
        .resource_mut::<CoarseWorld>()
        .light
        .precision = LightPrecision::Steps256;
    update_until(&mut app, "light to settle at 256 levels", |app| {
        light_is_settled(app) && tag_of(app, ABOVE).is_some_and(|tag| tag & HAS_SLOT != 0)
    });
    assert!(
        fractions(&mut app) > 0,
        "the shaded coarse chunk still shows its light at 16 levels"
    );
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
    let spawned = chunk_keys::<CoarseWorld>(&mut app);
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

// --- A change of emission ----------------------------------------------------

/// [`slab_and_shaft`] with a block of lamps floating in the sunlit part of
/// [`ABOVE`], away from its padding, so [`BELOW`] holds no lamp.
fn slab_and_lamp(p: IVec3) -> WorldVoxel {
    let lamp =
        (112..116).contains(&p.x) && (20..24).contains(&p.y) && (40..44).contains(&p.z);
    if lamp {
        WorldVoxel::Solid(LAMP)
    } else {
        slab_and_shaft(p)
    }
}

/// Beside the lamps of [`slab_and_lamp`], in [`ABOVE`].
const BY_LAMP: IVec3 = IVec3::new(114, 26, 42);
/// Under the slab, in [`ABOVE`], out of the lamps' reach.
const FAR_FROM_LAMP: IVec3 = IVec3::new(70, 4, 4);
/// In the shaft, in [`BELOW`].
const SHAFT: IVec3 = IVec3::new(106, -10, 26);

fn emission_app(world: CoarseWorld) -> App {
    let mut app = meshing_app();
    app.add_plugins(VoxelWorldPlugin::with_config(CoarseWorld {
        protected: 2,
        ..world
    }))
    .add_systems(Startup, spawn_camera::<CoarseWorld>);
    app
}

fn coarse_solves(app: &App) -> u64 {
    crate::light::bench::light_stats::<CoarseWorld>(app.world())
        .unwrap()
        .coarse_solves
}

/// Runs `app` until streaming is done and its light is settled: every chunk
/// is Ready and no chunk has spawned or retired for 20 frames in a row, and
/// light is settled. It counts frames and waits on chunk states, never on
/// time, so a busy machine only makes it take longer.
fn until_still(app: &mut App) {
    let mut still = 0;
    let mut keys = chunk_keys::<CoarseWorld>(app);
    update_until(app, "the world to stream in and settle", |app| {
        let now = chunk_keys::<CoarseWorld>(app);
        let ready = now.iter().all(|key| {
            chunk_status_of::<CoarseWorld>(app, *key) == Some(ChunkStatus::Ready)
        });
        still = if ready && now == keys { still + 1 } else { 0 };
        keys = now;
        still >= 20 && light_is_settled(app)
    });
}

fn set_emissive(app: &mut App, emissive: Vec<(u8, Emission)>) {
    app.world_mut().resource_mut::<CoarseWorld>().emissive = emissive;
}

fn lamp_world(level: f32) -> CoarseWorld {
    CoarseWorld {
        scene: Arc::new(slab_and_lamp),
        emissive: vec![(LAMP, Emission::level(level))],
        ..default()
    }
}

fn voxel_at(app: &mut App, position: IVec3) -> f32 {
    light_at(app, position)
        .unwrap_or_else(|| panic!("no light at {position}"))
        .voxel
}

#[test]
fn raising_a_lamps_level_solves_again_the_coarse_chunk_holding_it_alone() {
    let mut app = emission_app(lamp_world(6.0));
    until_still(&mut app);
    let old = voxel_at(&mut app, BY_LAMP);
    let below = light_at(&mut app, SHAFT);
    let far = light_at(&mut app, FAR_FROM_LAMP);
    let solves = coarse_solves(&app);

    set_emissive(&mut app, vec![(LAMP, Emission::level(15.0))]);
    app.update();
    assert!(
        !light_is_settled(&mut app),
        "the coarse solve is waited for"
    );
    let mut new = old;
    update_until(&mut app, "the coarse chunk's new light to land", |app| {
        new = voxel_at(app, BY_LAMP);
        assert!(
            new >= old,
            "old light shows until new light lands, read {new} after {old}"
        );
        new > old && light_is_settled(app)
    });
    for _ in 0..20 {
        app.update();
    }
    assert_eq!(
        coarse_solves(&app),
        solves + 1,
        "only the coarse chunk holding the lamp solved again"
    );
    assert_eq!(light_at(&mut app, SHAFT), below);
    assert_eq!(light_at(&mut app, FAR_FROM_LAMP), far);
}

#[test]
fn adding_or_dropping_an_emissive_material_solves_again_the_coarse_chunks_holding_it() {
    let mut app = emission_app(CoarseWorld {
        scene: Arc::new(slab_and_lamp),
        ..default()
    });
    until_still(&mut app);
    assert_eq!(voxel_at(&mut app, BY_LAMP), 0.0);
    let solves = coarse_solves(&app);

    set_emissive(&mut app, vec![(LAMP, Emission::level(15.0))]);
    update_until(&mut app, "the added lamp's light to land", |app| {
        voxel_at(app, BY_LAMP) > 0.0 && light_is_settled(app)
    });
    assert_eq!(coarse_solves(&app), solves + 1);

    set_emissive(&mut app, Vec::new());
    update_until(&mut app, "the dropped lamp's light to go", |app| {
        voxel_at(app, BY_LAMP) == 0.0 && light_is_settled(app)
    });
    assert_eq!(coarse_solves(&app), solves + 2);
}

#[test]
fn a_coarse_chunk_task_solved_under_the_old_emissive_list_solves_again_as_it_lands() {
    let _exclusive = Hold::exclusive();
    let hold = Hold::new(ABOVE.position);
    let mut app = emission_app(CoarseWorld {
        hold: Some(hold.clone()),
        ..lamp_world(6.0)
    });
    update_until(&mut app, "the lamps' coarse chunk task to be held", |_| {
        hold.waited.load(Ordering::Acquire) > 0
    });
    // Into the light the held task solves at level 6.
    set_emissive(&mut app, vec![(LAMP, Emission::level(15.0))]);
    for _ in 0..5 {
        app.update();
    }
    hold.release();
    until_still(&mut app);

    let mut fresh = emission_app(lamp_world(15.0));
    until_still(&mut fresh);
    assert_eq!(
        voxel_at(&mut app, BY_LAMP),
        voxel_at(&mut fresh, BY_LAMP),
        "the light of level 15"
    );
}

#[test]
fn a_level_0_chunk_with_another_data_shape_follows_a_change_of_emission() {
    let lamp = |p: IVec3| {
        if (14..18).contains(&p.x) && (8..12).contains(&p.y) && (14..18).contains(&p.z) {
            WorldVoxel::Solid(LAMP)
        } else {
            roofed(p)
        }
    };
    let mut app = emission_app(CoarseWorld {
        scene: Arc::new(lamp),
        data: 16,
        emissive: vec![(LAMP, Emission::level(6.0))],
        ..default()
    });
    until_still(&mut app);
    let by_lamp = IVec3::new(16, 14, 16);
    let old = voxel_at(&mut app, by_lamp);

    set_emissive(&mut app, vec![(LAMP, Emission::level(15.0))]);
    update_until(&mut app, "the new light to land", |app| {
        voxel_at(app, by_lamp) > old && light_is_settled(app)
    });
}

#[test]
fn a_change_of_reach_shape_sky_or_sky_fill_leaves_coarse_chunks_as_they_are() {
    let mut app = emission_app(lamp_world(15.0));
    until_still(&mut app);
    let changes: [fn(&mut LightPropagation); 4] = [
        |light| light.voxel_reach = 8,
        |light| light.shape = LightShape::Diamond,
        |light| light.sky_fill = Some(default()),
        |light| light.sky = Sky::Off,
    ];
    for change in changes {
        let (solves, by_lamp) = (coarse_solves(&app), light_at(&mut app, BY_LAMP));
        change(&mut app.world_mut().resource_mut::<CoarseWorld>().light);
        update_until(&mut app, "the change to settle", |app| {
            light_is_settled(app)
        });
        for _ in 0..10 {
            app.update();
        }
        assert_eq!(coarse_solves(&app), solves);
        assert_eq!(light_at(&mut app, BY_LAMP), by_lamp);
    }
}

// --- Coloured light ----------------------------------------------------------

/// A material emitting blue light.
const BLUE_LAMP: u8 = 9;

fn red(level: f32) -> Emission {
    Emission::level(level).with_color(LinearRgba::RED)
}

fn blue(level: f32) -> Emission {
    Emission::level(level).with_color(LinearRgba::BLUE)
}

/// [`slab_and_lamp`] with a block of blue lamps floating just above the slab,
/// 40 voxels from the red ones, in [`ABOVE`] too.
fn two_lamps(p: IVec3) -> WorldVoxel {
    let blue =
        (72..76).contains(&p.x) && (14..18).contains(&p.y) && (40..44).contains(&p.z);
    if blue {
        WorldVoxel::Solid(BLUE_LAMP)
    } else {
        slab_and_lamp(p)
    }
}

/// Above the blue lamps of [`two_lamps`].
const BY_BLUE: IVec3 = IVec3::new(74, 20, 42);

fn two_colour_world() -> CoarseWorld {
    CoarseWorld {
        scene: Arc::new(two_lamps),
        emissive: vec![(LAMP, red(15.0)), (BLUE_LAMP, blue(15.0))],
        ..default()
    }
}

/// Runs `app` until [`ABOVE`] shows its mesh and its light, and then as
/// [`until_still`] does, so a busy machine's slow streaming can't end the
/// wait before it shows.
fn until_coarse_shown(app: &mut App) {
    update_until(app, "the lamps' coarse chunk to show its light", |app| {
        tag_of(app, ABOVE).is_some() && coarse_lit(app, &[BY_LAMP])
    });
    until_still(app);
}

fn color_at(app: &mut App, position: IVec3) -> LinearRgba {
    light_at(app, position)
        .unwrap_or_else(|| panic!("no light at {position}"))
        .voxel_color
}

#[test]
fn a_coarse_chunk_with_lamps_of_two_colours_reads_its_colours_cell_by_cell() {
    let mut app = emission_app(two_colour_world());
    until_coarse_shown(&mut app);
    assert!(voxel_at(&mut app, BY_LAMP) > 0.0);
    assert_eq!(color_at(&mut app, BY_LAMP), LinearRgba::RED);
    assert!(voxel_at(&mut app, BY_BLUE) > 0.0);
    assert_eq!(color_at(&mut app, BY_BLUE), LinearRgba::BLUE);
    let tag = tag_of(&mut app, ABOVE).unwrap();
    assert_eq!(tag & 0xF000_0000, HAS_SLOT | 1 << 29, "tag {tag:#x}");

    // The slab's top cell under the blue lamps, a level-1 cell 2 voxels wide.
    let in_slab = IVec3::new(74, 10, 42);
    let above = light_at(&mut app, in_slab + IVec3::Y * 2).unwrap();
    assert!(above.voxel > 0.0);
    let solid = light_at(&mut app, in_slab).unwrap();
    assert!(solid.voxel >= above.voxel);
    assert_eq!(solid.voxel_color, LinearRgba::BLUE);
    // A lamp's own cell reads its own colour.
    assert_eq!(color_at(&mut app, IVec3::new(114, 22, 42)), LinearRgba::RED);
}

/// Ground below y = 8, a roof near the origin that shades a level-0 chunk,
/// and in [`ABOVE`] a block of bright red lamps beside a wide field of dim
/// blue ones, whose pool lights more of the ground than the red one does.
fn red_beside_blue(p: IVec3) -> WorldVoxel {
    let roof =
        (8..16).contains(&p.x) && (12..14).contains(&p.y) && (8..16).contains(&p.z);
    let red =
        (100..104).contains(&p.x) && (8..10).contains(&p.y) && (28..32).contains(&p.z);
    let blue = (64..92).contains(&p.x) && p.y == 8 && p.x % 4 == 0 && p.z % 4 == 0;
    if red {
        WorldVoxel::Solid(LAMP)
    } else if blue {
        WorldVoxel::Solid(BLUE_LAMP)
    } else if roof || p.y < 8 {
        WorldVoxel::Solid(STONE)
    } else {
        WorldVoxel::Air
    }
}

#[test]
fn a_coarse_chunk_that_gives_its_slot_up_shows_the_colour_of_its_brightest_tenth() {
    // One slot, which a nearer level-0 chunk takes.
    let mut app = emission_app(CoarseWorld {
        scene: Arc::new(red_beside_blue),
        emissive: vec![(LAMP, red(15.0)), (BLUE_LAMP, blue(3.0))],
        light: LightPropagation {
            sky_fill: None,
            slot_capacity: 1,
            ..default()
        },
        ..default()
    });
    until_coarse_shown(&mut app);
    let tag = tag_of(&mut app, ABOVE).unwrap();
    assert_eq!(tag & HAS_SLOT, 0, "tag {tag:#x}");
    let red_cut =
        crate::light::color::tag_cut(crate::light::color::pack(LinearRgba::RED));
    assert_eq!(
        tag >> 16 & 0x3FFF,
        red_cut,
        "red, not blue or magenta: {tag:#x}"
    );
    assert!(tag >> 4 & 15 > 3, "brighter than the blue pool: {tag:#x}");
    // Its tag's colour, in every cell.
    let in_blue = IVec3::new(70, 8, 10);
    assert!(voxel_at(&mut app, in_blue) > 0.0);
    assert_eq!(color_at(&mut app, in_blue), LinearRgba::RED);
}

#[test]
fn recolouring_a_lamp_solves_again_the_coarse_chunk_holding_it_alone() {
    let mut app = emission_app(two_colour_world());
    until_coarse_shown(&mut app);
    let below = light_at(&mut app, SHAFT);
    let far = light_at(&mut app, FAR_FROM_LAMP);
    let solves = coarse_solves(&app);

    let green = Emission::level(15.0).with_color(LinearRgba::GREEN);
    set_emissive(&mut app, vec![(LAMP, green), (BLUE_LAMP, blue(15.0))]);
    app.update();
    assert!(
        !light_is_settled(&mut app),
        "the coarse solve is waited for"
    );
    update_until(&mut app, "the coarse chunk's new colour to land", |app| {
        let color = color_at(app, BY_LAMP);
        assert!(
            color == LinearRgba::RED || color == LinearRgba::GREEN,
            "old light shows until new light lands, read {color:?}"
        );
        color == LinearRgba::GREEN && light_is_settled(app)
    });
    for _ in 0..20 {
        app.update();
    }
    assert_eq!(coarse_solves(&app), solves + 1);
    assert_eq!(color_at(&mut app, BY_BLUE), LinearRgba::BLUE);
    assert_eq!(light_at(&mut app, SHAFT), below);
    assert_eq!(light_at(&mut app, FAR_FROM_LAMP), far);

    // The same light colour once scaled.
    let dim_green = Emission::level(15.0).with_color(LinearRgba::rgb(0.0, 0.5, 0.0));
    set_emissive(&mut app, vec![(LAMP, dim_green), (BLUE_LAMP, blue(15.0))]);
    for _ in 0..20 {
        app.update();
    }
    assert!(light_is_settled(&mut app));
    assert_eq!(coarse_solves(&app), solves + 1);
}

#[test]
fn colouring_a_white_worlds_lamp_solves_again_the_coarse_chunk_holding_it() {
    let mut app = emission_app(lamp_world(15.0));
    until_coarse_shown(&mut app);
    assert_eq!(color_at(&mut app, BY_LAMP), LinearRgba::WHITE);
    let solves = coarse_solves(&app);

    set_emissive(&mut app, vec![(LAMP, red(15.0))]);
    update_until(&mut app, "the coarse chunk's colour to land", |app| {
        color_at(app, BY_LAMP) == LinearRgba::RED && light_is_settled(app)
    });
    for _ in 0..20 {
        app.update();
    }
    assert_eq!(coarse_solves(&app), solves + 1);
}

#[test]
fn a_level_0_chunk_with_another_data_shape_shows_coloured_light() {
    let lamp = |p: IVec3| {
        if (14..18).contains(&p.x) && (8..12).contains(&p.y) && (14..18).contains(&p.z) {
            WorldVoxel::Solid(LAMP)
        } else {
            roofed(p)
        }
    };
    let mut app = emission_app(CoarseWorld {
        scene: Arc::new(lamp),
        data: 16,
        emissive: vec![(LAMP, red(15.0))],
        ..default()
    });
    until_still(&mut app);
    let by_lamp = IVec3::new(16, 14, 16);
    assert!(voxel_at(&mut app, by_lamp) > 0.0);
    assert_eq!(color_at(&mut app, by_lamp), LinearRgba::RED);
}

/// The colour textures the world's route has made: 3D `R16Uint` images
/// bigger than one texel.
fn color_textures(app: &App) -> usize {
    use bevy::render::render_resource::TextureFormat;
    app.world()
        .resource::<Assets<Image>>()
        .iter()
        .filter(|(_, image)| image.texture_descriptor.format == TextureFormat::R16Uint)
        .filter(|(_, image)| image.texture_descriptor.size.width > 1)
        .count()
}

#[test]
fn a_white_lit_coarse_chunk_shows_white_and_makes_no_colour_texture() {
    let mut app = emission_app(CoarseWorld {
        emissive: vec![
            (LAMP, Emission::level(15.0)),
            (
                BLUE_LAMP,
                Emission::level(15.0).with_color(LinearRgba::BLUE),
            ),
        ],
        ..lamp_world(15.0)
    });
    until_coarse_shown(&mut app);
    let tag = tag_of(&mut app, ABOVE).unwrap();
    assert_eq!(tag & 0xF000_0000, HAS_SLOT, "tag {tag:#x}");
    assert_eq!(color_at(&mut app, BY_LAMP), LinearRgba::WHITE);
    assert_eq!(color_textures(&app), 0);
    let settings = app
        .world()
        .resource::<crate::light::LightSettings<CoarseWorld>>();
    assert_eq!(settings.counters().color_runs.load(Ordering::Relaxed), 0);
}

#[test]
fn a_change_of_precision_replaces_the_colour_texture_coarse_chunks_use() {
    let mut app = emission_app(two_colour_world());
    until_coarse_shown(&mut app);
    let color_texture = |app: &App| {
        app.world()
            .resource::<crate::light::route::LightRoute<CoarseWorld>>()
            .color_texture()
    };
    let before = color_texture(&app).id();
    assert_eq!(color_textures(&app), 1);

    app.world_mut()
        .resource_mut::<CoarseWorld>()
        .light
        .precision = LightPrecision::Steps256;
    update_until(&mut app, "the colour texture to be replaced", |app| {
        let now = color_texture(app);
        now.id() != before
            && now
                != app
                    .world()
                    .resource::<crate::light::route::LightRoute<CoarseWorld>>()
                    .color_placeholder
            && light_is_settled(app)
    });
    let tag = tag_of(&mut app, ABOVE).unwrap();
    assert_eq!(tag & 0xF000_0000, HAS_SLOT | 1 << 29, "tag {tag:#x}");
    assert_eq!(color_at(&mut app, BY_BLUE), LinearRgba::BLUE);
}

#[test]
fn coarse_copy_bytes_count_the_copies_of_colour_slots() {
    let copies = |world: CoarseWorld| {
        let mut app = emission_app(world);
        until_coarse_shown(&mut app);
        crate::light::bench::light_stats::<CoarseWorld>(app.world())
            .unwrap()
            .coarse_copy_bytes
    };
    let white = copies(CoarseWorld {
        emissive: vec![
            (LAMP, Emission::level(15.0)),
            (BLUE_LAMP, Emission::level(15.0)),
        ],
        ..two_colour_world()
    });
    let colored = copies(two_colour_world());
    // One coarse chunk holds a colour slot: 34³ texels of 2 bytes.
    assert_eq!(colored, white + 34 * 34 * 34 * 2);
}
