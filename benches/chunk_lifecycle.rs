//! Chunk lifecycle benchmarks.
//!
//! Runs scripted scenarios against a headless App (no window, no GPU) that still
//! exercises the full chunk lifecycle: spawn, generate, mesh, publish, retire, despawn.
//!
//! Record a run:
//!     cargo bench --bench chunk_lifecycle -- run <label>
//! Compare recorded runs (exits non-zero if a gated metric regressed); each side
//! may be a comma-separated list of labels, combined by their median:
//!     cargo bench --bench chunk_lifecycle -- compare <before>[,<before>…] <after>[,<after>…]
//!
//! Background load on a desktop drifts by more than the gates over minutes, so
//! comparisons alternate the two builds instead of reusing an old recording:
//!     cargo bench --bench chunk_lifecycle -- save-binary baseline   # on the baseline commit
//!     cargo bench --bench chunk_lifecycle -- ab baseline            # on the candidate
//! `ab` runs the saved baseline binary and this build alternately, `AB_ROUNDS`
//! times each, then compares them. Build both sides with one codegen unit:
//!     CARGO_PROFILE_BENCH_CODEGEN_UNITS=1 cargo bench --bench chunk_lifecycle -- …
//! The voxel world's generic code is compiled inside this crate, and how the
//! default sixteen units group it has moved flat terrain's generation
//! throughput by 15% between builds that differ only in code the bench never
//! runs; one unit makes such builds identical.
//!
//! Results and saved binaries are written to `target/bench/chunk_lifecycle/`.
//!
//! To see what a scenario's slow frames are doing, run it once under Tracy. Start
//! the Tracy profiler (or `tracy-capture -o <file>.tracy`) first, then:
//!     cargo bench --bench chunk_lifecycle --features trace_tracy -- profile <scenario>
//! The Tracy version must match the protocol of the `tracy-client` crate Bevy uses;
//! see the version table in the `tracy-client` README.

use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use bevy::app::{TaskPoolOptions, TaskPoolPlugin, TaskPoolThreadAssignmentPolicy};
use bevy::camera::{CameraProjection, RenderTargetInfo};
use bevy::ecs::system::SystemState;
use bevy::prelude::*;
use bevy_voxel_world::prelude::*;

#[cfg(feature = "light_propagation")]
#[path = "common/light_scene.rs"]
mod light_scene;

#[cfg(feature = "light_propagation")]
use light_scene::{SceneTable, SCENE_TOP};

#[cfg(feature = "light_propagation")]
#[path = "common/fill_worlds.rs"]
mod fill_worlds;

#[cfg(feature = "light_propagation")]
use fill_worlds::FILL_SCENES;

/// Pool sizes are pinned so the machine's core count can't skew comparisons, and
/// kept, with the main thread, below the M1 Max's 8 performance cores so
/// background work has room to run beside the benchmark rather than on top of it.
const ASYNC_COMPUTE_THREADS: usize = 3;
const COMPUTE_THREADS: usize = 2;
const IO_THREADS: usize = 1;
/// One-minute load average above which a run is flagged as likely noisy.
const NOISY_LOAD: f64 = 8.0;
const AB_ROUNDS: usize = 3;
const SPAWNING_DISTANCE: i32 = 10;
/// Chunks within this distance of the camera are the near ones, whose time to
/// become Ready says how soon the player sees ground.
const NEAR_DISTANCE: i32 = 4;
/// How many frames in a row without any lifecycle change end `stream_in`.
const SETTLE_FRAMES: u32 = 30;
const MAX_FRAMES: u32 = 5000;

/// For scenarios whose run-to-run spread sits close to the gates.
const MANY_RUNS: usize = 9;
const RUNS: usize = 5;
/// `caves` samples its scene once per process and streams thousands of
/// chunks at sky reach 30 each run.
#[cfg(feature = "light_propagation")]
const CAVE_RUNS: usize = 3;
/// Runs of each world for comparing the models of sky fill.
#[cfg(feature = "light_propagation")]
const FILL_RUNS: usize = 3;
const IDLE_FRAMES: usize = 2000;
/// Frames a lit world is left at rest to check that its light starts no work.
#[cfg(feature = "light_propagation")]
const IDLE_LIGHT_FRAMES: usize = 200;
const FLY_FRAMES: usize = 1200;
/// Camera speed in voxels per second of wall-clock time, just under the speed
/// `stream_in` can keep up with.
const FLY_SPEED: f32 = 180.0;
const CAMERA_START: Vec3 = Vec3::new(16.0, 30.0, 16.0);
const STORM_FRAMES: u32 = 300;
/// Far enough that no chunk around the start stays wanted after the jump.
const TELEPORT: Vec3 = Vec3::new(1000.0, 0.0, 0.0);
/// Frames slower than these count as stutters.
const STUTTER_MS: [u32; 2] = [4, 8];

/// Set by `profile`: apps also send their traces to Tracy.
static PROFILING: AtomicBool = AtomicBool::new(false);

const SINE: u8 = 0;
const FLAT: u8 = 1;

#[derive(Resource, Clone, Default)]
struct BenchWorld<const TERRAIN: u8, const IN_VIEW: bool>;

type SineWorld = BenchWorld<SINE, false>;
type FlatWorld = BenchWorld<FLAT, false>;
type InViewWorld = BenchWorld<SINE, true>;

impl<const TERRAIN: u8, const IN_VIEW: bool> VoxelWorldConfig
    for BenchWorld<TERRAIN, IN_VIEW>
{
    type MaterialIndex = u8;
    type ChunkUserBundle = ();

    fn spawning_distance(&self) -> u32 {
        SPAWNING_DISTANCE as u32
    }

    fn chunk_spawn_strategy(&self) -> ChunkSpawnStrategy {
        if IN_VIEW {
            ChunkSpawnStrategy::CloseAndInView
        } else {
            ChunkSpawnStrategy::Close
        }
    }

    fn chunk_despawn_strategy(&self) -> ChunkDespawnStrategy {
        if IN_VIEW {
            ChunkDespawnStrategy::FarAwayOrOutOfView
        } else {
            ChunkDespawnStrategy::FarAway
        }
    }

    fn voxel_lookup_delegate(&self) -> VoxelLookupDelegate<Self::MaterialIndex> {
        Box::new(|_, _, _| {
            Box::new(|pos: IVec3, _| {
                let height = match TERRAIN {
                    FLAT => 0,
                    _ => {
                        (8.0 * (pos.x as f32 / 23.0).sin()
                            + 8.0 * (pos.z as f32 / 17.0).cos())
                            as i32
                    }
                };
                if pos.y < height {
                    WorldVoxel::Solid(if pos.y < height - 3 { 1 } else { 0 })
                } else {
                    WorldVoxel::Air
                }
            })
        })
    }
}

/// The sine terrain with a lamp on one in 64 surface voxels, lit with the
/// default light settings when `LIT`. Its scenarios report what solving light
/// in chunk tasks costs, against the same world unlit, and gate nothing.
#[cfg(feature = "light_propagation")]
#[derive(Resource, Clone, Default)]
struct LampWorld<const LIT: bool>;

#[cfg(feature = "light_propagation")]
const LAMP: u8 = 2;

#[cfg(feature = "light_propagation")]
impl<const LIT: bool> VoxelWorldConfig for LampWorld<LIT> {
    type MaterialIndex = u8;
    type ChunkUserBundle = ();

    fn spawning_distance(&self) -> u32 {
        SPAWNING_DISTANCE as u32
    }

    fn chunk_spawn_strategy(&self) -> ChunkSpawnStrategy {
        ChunkSpawnStrategy::Close
    }

    fn chunk_despawn_strategy(&self) -> ChunkDespawnStrategy {
        ChunkDespawnStrategy::FarAway
    }

    fn voxel_lookup_delegate(&self) -> VoxelLookupDelegate<Self::MaterialIndex> {
        Box::new(|_, _, _| {
            Box::new(|pos: IVec3, _| {
                let height = (8.0 * (pos.x as f32 / 23.0).sin()
                    + 8.0 * (pos.z as f32 / 17.0).cos())
                    as i32;
                let hash = (pos.x.wrapping_mul(73_856_093)
                    ^ pos.z.wrapping_mul(19_349_663)) as u32;
                if pos.y == height - 1 && hash.is_multiple_of(64) {
                    WorldVoxel::Solid(LAMP)
                } else if pos.y < height {
                    WorldVoxel::Solid(if pos.y < height - 3 { 1 } else { 0 })
                } else {
                    WorldVoxel::Air
                }
            })
        })
    }

    fn light_propagation(&self) -> Option<LightPropagation> {
        LIT.then(LightPropagation::default)
    }

    fn emissive_materials(&self) -> Vec<(Self::MaterialIndex, Emission)> {
        vec![(LAMP, Emission::level(15.0))]
    }
}

/// The world of `stream_in`, or of `stream_in_flat` with `FLAT`, lit with the
/// default light settings. Its scenarios report what light costs on the gated
/// worlds, against `light-baseline`'s runs of them, and gate nothing yet.
#[cfg(feature = "light_propagation")]
#[derive(Resource, Clone, Default)]
struct LitBenchWorld<const TERRAIN: u8>;

#[cfg(feature = "light_propagation")]
impl<const TERRAIN: u8> VoxelWorldConfig for LitBenchWorld<TERRAIN> {
    type MaterialIndex = u8;
    type ChunkUserBundle = ();

    fn spawning_distance(&self) -> u32 {
        SPAWNING_DISTANCE as u32
    }

    fn chunk_spawn_strategy(&self) -> ChunkSpawnStrategy {
        ChunkSpawnStrategy::Close
    }

    fn chunk_despawn_strategy(&self) -> ChunkDespawnStrategy {
        ChunkDespawnStrategy::FarAway
    }

    fn voxel_lookup_delegate(&self) -> VoxelLookupDelegate<Self::MaterialIndex> {
        BenchWorld::<TERRAIN, false>.voxel_lookup_delegate()
    }

    fn light_propagation(&self) -> Option<LightPropagation> {
        Some(LightPropagation::default())
    }
}

/// The sine terrain with floating islands over it, lit with the default
/// light settings: slabs of stone 8 voxels thick and 64 wide, from y = 64 to
/// 72, every 128 voxels along x and z. The ground under them gets its light
/// from sky fill, which neither bench world shows.
///
/// With `HINTS` at [`ABOVE_HIGHEST_SOLID`] or [`EVERY_ALL_AIR_POSITION`] its
/// fill hints say where chunk positions are air, and the streaming policy
/// spawns none of those outside the protected radius.
#[cfg(feature = "light_propagation")]
#[derive(Resource, Clone, Default)]
struct IslandWorld<const HINTS: u8 = NO_HINTS>;

#[cfg(feature = "light_propagation")]
const NO_HINTS: u8 = 0;
/// Air hints at every chunk position above the highest solid of its
/// footprint.
#[cfg(feature = "light_propagation")]
const ABOVE_HIGHEST_SOLID: u8 = 1;
/// Air hints at every chunk position whose voxels are all air, under the
/// islands too.
#[cfg(feature = "light_propagation")]
const EVERY_ALL_AIR_POSITION: u8 = 2;

#[cfg(feature = "light_propagation")]
impl<const HINTS: u8> VoxelWorldConfig for IslandWorld<HINTS> {
    type MaterialIndex = u8;
    type ChunkUserBundle = ();

    fn spawning_distance(&self) -> u32 {
        SPAWNING_DISTANCE as u32
    }

    fn chunk_spawn_strategy(&self) -> ChunkSpawnStrategy {
        ChunkSpawnStrategy::Close
    }

    fn chunk_despawn_strategy(&self) -> ChunkDespawnStrategy {
        ChunkDespawnStrategy::FarAway
    }

    fn voxel_lookup_delegate(&self) -> VoxelLookupDelegate<Self::MaterialIndex> {
        let terrain = SineWorld::default().voxel_lookup_delegate();
        Box::new(move |position, lod, previous| {
            let mut terrain = terrain(position, lod, previous);
            Box::new(move |pos: IVec3, previous| {
                let island = (64..72).contains(&pos.y)
                    && pos.x.rem_euclid(128) < 64
                    && pos.z.rem_euclid(128) < 64;
                if island {
                    WorldVoxel::Solid(1)
                } else {
                    terrain(pos, previous)
                }
            })
        })
    }

    fn chunk_fill_hint(&self, key: ChunkKey) -> FillHint<Self::MaterialIndex> {
        // The terrain lies within 16 voxels of y = 0, and the islands fill
        // the chunk positions at y = 2 of their footprints.
        let position = key.position;
        let under_island = key.lod_level == 0
            && position.x.rem_euclid(4) < 2
            && position.z.rem_euclid(4) < 2;
        let air = match HINTS {
            ABOVE_HIGHEST_SOLID if under_island => position.y >= 3,
            ABOVE_HIGHEST_SOLID => position.y >= 1,
            EVERY_ALL_AIR_POSITION => {
                position.y >= 1 && !(under_island && position.y == 2)
            }
            _ => false,
        };
        if air && key.lod_level == 0 {
            FillHint::Uniform(WorldVoxel::Air)
        } else {
            FillHint::Unknown
        }
    }

    fn light_propagation(&self) -> Option<LightPropagation> {
        Some(LightPropagation::default())
    }
}

/// One of the worlds for comparing the models of sky fill, the one in
/// `FILL_SCENES` at index `SCENE`, lit with the default light settings but
/// with sky fill under paths when `MODEL` is [`PATHS`] and under sky openness
/// when it's [`OPENNESS`], each at its defaults. With `CEILING`, sky light
/// enters below a sky ceiling at y = 192, over every solid of the islands,
/// in place of the sky clearance.
#[cfg(feature = "light_propagation")]
#[derive(Resource, Clone, Default)]
struct FillWorld<const SCENE: usize, const MODEL: u8, const CEILING: bool = false>;

#[cfg(feature = "light_propagation")]
const PATHS: u8 = 0;
#[cfg(feature = "light_propagation")]
const OPENNESS: u8 = 1;

#[cfg(feature = "light_propagation")]
impl<const SCENE: usize, const MODEL: u8, const CEILING: bool> VoxelWorldConfig
    for FillWorld<SCENE, MODEL, CEILING>
{
    type MaterialIndex = u8;
    type ChunkUserBundle = ();

    fn spawning_distance(&self) -> u32 {
        SPAWNING_DISTANCE as u32
    }

    fn chunk_spawn_strategy(&self) -> ChunkSpawnStrategy {
        ChunkSpawnStrategy::Close
    }

    fn chunk_despawn_strategy(&self) -> ChunkDespawnStrategy {
        ChunkDespawnStrategy::FarAway
    }

    fn voxel_lookup_delegate(&self) -> VoxelLookupDelegate<Self::MaterialIndex> {
        Box::new(|_, _, _| {
            Box::new(|pos: IVec3, _| {
                if FILL_SCENES[SCENE].solid(pos) {
                    WorldVoxel::Solid(1)
                } else {
                    WorldVoxel::Air
                }
            })
        })
    }

    fn light_propagation(&self) -> Option<LightPropagation> {
        let fill = match MODEL {
            PATHS => SkyFill::Paths(PathFill::default()),
            _ => SkyFill::Openness(OpenFill::default()),
        };
        Some(LightPropagation {
            sky: if CEILING {
                Sky::Ceiling { y: 192 }
            } else {
                Sky::default()
            },
            sky_fill: Some(fill),
            ..default()
        })
    }
}

/// The world of `stream_in`, lit, with light settings that `settings_lit`
/// changes while it runs.
#[cfg(feature = "light_propagation")]
#[derive(Resource, Clone, Default)]
struct SettingsWorld {
    light: LightPropagation,
}

#[cfg(feature = "light_propagation")]
impl VoxelWorldConfig for SettingsWorld {
    type MaterialIndex = u8;
    type ChunkUserBundle = ();

    fn spawning_distance(&self) -> u32 {
        SPAWNING_DISTANCE as u32
    }

    fn chunk_spawn_strategy(&self) -> ChunkSpawnStrategy {
        ChunkSpawnStrategy::Close
    }

    fn chunk_despawn_strategy(&self) -> ChunkDespawnStrategy {
        ChunkDespawnStrategy::FarAway
    }

    fn voxel_lookup_delegate(&self) -> VoxelLookupDelegate<Self::MaterialIndex> {
        BenchWorld::<SINE, false>.voxel_lookup_delegate()
    }

    fn light_propagation(&self) -> Option<LightPropagation> {
        Some(self.light.clone())
    }
}

/// The sine terrain streamed out to `LOD_SPAWNING_DISTANCE` chunks in four
/// levels of detail, lit with the default light settings, with fill hints
/// for the rock below and the air above it when `HINTED`. Its scenarios
/// report what coarse chunks' light costs, and gate nothing yet.
#[cfg(feature = "light_propagation")]
#[derive(Resource, Clone, Default)]
struct LodWorld<const HINTED: bool>;

#[cfg(feature = "light_propagation")]
const LOD_SPAWNING_DISTANCE: u32 = 48;

#[cfg(feature = "light_propagation")]
impl<const HINTED: bool> VoxelWorldConfig for LodWorld<HINTED> {
    type MaterialIndex = u8;
    type ChunkUserBundle = ();

    fn spawning_distance(&self) -> u32 {
        LOD_SPAWNING_DISTANCE
    }

    fn chunk_spawn_strategy(&self) -> ChunkSpawnStrategy {
        ChunkSpawnStrategy::Close
    }

    fn chunk_despawn_strategy(&self) -> ChunkDespawnStrategy {
        ChunkDespawnStrategy::FarAway
    }

    fn voxel_lookup_delegate(&self) -> VoxelLookupDelegate<Self::MaterialIndex> {
        SineWorld::default().voxel_lookup_delegate()
    }

    fn chunk_lod(
        &self,
        chunk_position: IVec3,
        _previous_lod: Option<LodLevel>,
        camera_position: Vec3,
    ) -> LodLevel {
        let camera_chunk = (camera_position / 32.0).floor();
        match chunk_position.as_vec3().distance(camera_chunk) {
            distance if distance < 4.0 => 0,
            distance if distance < 10.0 => 1,
            distance if distance < 24.0 => 2,
            _ => 3,
        }
    }

    fn chunk_fill_hint(&self, key: ChunkKey) -> FillHint<Self::MaterialIndex> {
        // The terrain lies within 16 voxels of y = 0.
        if !HINTED {
            FillHint::Unknown
        } else if key.position.y + key.extent() <= -1 {
            FillHint::Uniform(WorldVoxel::Solid(1))
        } else if key.position.y >= 1 {
            FillHint::Uniform(WorldVoxel::Air)
        } else {
            FillHint::Unknown
        }
    }

    fn light_propagation(&self) -> Option<LightPropagation> {
        Some(LightPropagation::default())
    }
}

/// The light reference scene around the foot of its deep shaft, lit with the
/// default light settings: caves, shafts and a deep cavern under hills, with
/// emissive voxels on one in 32 exposed floors and on the floors of its torch
/// columns. Its voxels are read from [`cave_table`].
#[cfg(feature = "light_propagation")]
#[derive(Resource, Clone, Default)]
struct CaveWorld;

/// Where `caves` starts its camera: at the foot of the reference scene's
/// deep shaft, in the deep cavern, looking at the shaft.
#[cfg(feature = "light_propagation")]
const CAVE_CAMERA: CameraStart = CameraStart {
    at: Vec3::new(34.0, -196.0, 34.0),
    target: Vec3::new(0.5, -210.0, 0.5),
};

/// The reference scene sampled around `CAVE_CAMERA`: every chunk within
/// `SPAWNING_DISTANCE` of the camera's chunk and the voxels padding them,
/// from below the lowest up to the height above which the scene is air, so
/// the columns walks and sky probes read are in it too. Sampled once per
/// process, before the first `caves` run.
#[cfg(feature = "light_propagation")]
fn cave_table() -> Arc<SceneTable> {
    static TABLE: std::sync::OnceLock<Arc<SceneTable>> = std::sync::OnceLock::new();
    TABLE
        .get_or_init(|| {
            let started = Instant::now();
            let camera = (CAVE_CAMERA.at / 32.0).floor().as_ivec3();
            let reach = IVec3::splat(SPAWNING_DISTANCE + 1);
            let min = (camera - reach) * 32;
            let max = ((camera + reach + IVec3::ONE) * 32).with_y(SCENE_TOP);
            let table = SceneTable::new(min, max, 8);
            eprintln!(
                "sampled the reference scene from {min} to {max} in {:.1?}",
                started.elapsed()
            );
            Arc::new(table)
        })
        .clone()
}

#[cfg(feature = "light_propagation")]
impl VoxelWorldConfig for CaveWorld {
    type MaterialIndex = u8;
    type ChunkUserBundle = ();

    fn spawning_distance(&self) -> u32 {
        SPAWNING_DISTANCE as u32
    }

    fn chunk_spawn_strategy(&self) -> ChunkSpawnStrategy {
        ChunkSpawnStrategy::Close
    }

    fn chunk_despawn_strategy(&self) -> ChunkDespawnStrategy {
        ChunkDespawnStrategy::FarAway
    }

    fn voxel_lookup_delegate(&self) -> VoxelLookupDelegate<Self::MaterialIndex> {
        let table = cave_table();
        Box::new(move |_, _, _| {
            let table = table.clone();
            Box::new(move |pos: IVec3, _| {
                if table.open(pos) {
                    WorldVoxel::Air
                } else if table.emissive(pos) {
                    WorldVoxel::Solid(LAMP)
                } else {
                    WorldVoxel::Solid(1)
                }
            })
        })
    }

    fn light_propagation(&self) -> Option<LightPropagation> {
        Some(LightPropagation::default())
    }

    fn emissive_materials(&self) -> Vec<(Self::MaterialIndex, Emission)> {
        vec![(LAMP, Emission::level(15.0))]
    }
}

/// The world of the `noise_terrain_lod` example: a noise terrain streamed
/// out to 500 chunks in six levels of detail, with fill hints for the rock
/// below it and the air above it, lit with the default light settings when
/// `LIT`. Its camera starts where the example's does and stays there.
#[cfg(feature = "light_propagation")]
#[derive(Resource, Clone)]
struct TerrainLodWorld<const LIT: bool> {
    noise: Arc<noise::HybridMulti<noise::Perlin>>,
}

#[cfg(feature = "light_propagation")]
impl<const LIT: bool> Default for TerrainLodWorld<LIT> {
    fn default() -> Self {
        let mut noise = noise::HybridMulti::<noise::Perlin>::new(1234);
        noise.octaves = 5;
        noise.frequency = 1.1;
        noise.lacunarity = 2.8;
        noise.persistence = 0.4;
        Self {
            noise: Arc::new(noise),
        }
    }
}

/// The camera of the `noise_terrain_lod` example.
#[cfg(feature = "light_propagation")]
const TERRAIN_LOD_CAMERA: CameraStart = CameraStart {
    at: Vec3::new(-200.0, 260.0, -200.0),
    target: Vec3::new(0.0, 170.0, 0.0),
};

/// What fills the chunk at `key` of `TerrainLodWorld` when no terrain
/// crosses it, as in the `noise_terrain_lod` example.
#[cfg(feature = "light_propagation")]
fn terrain_lod_fill(key: ChunkKey) -> FillHint<u8> {
    if key.position.y + key.extent() <= -1 {
        FillHint::Uniform(WorldVoxel::Solid(3))
    } else if key.position.y > 5 {
        FillHint::Uniform(WorldVoxel::Air)
    } else {
        FillHint::Unknown
    }
}

#[cfg(feature = "light_propagation")]
impl<const LIT: bool> VoxelWorldConfig for TerrainLodWorld<LIT> {
    type MaterialIndex = u8;
    type ChunkUserBundle = ();

    fn spawning_distance(&self) -> u32 {
        500
    }

    fn min_despawn_distance(&self) -> u32 {
        1
    }

    fn voxel_lookup_delegate(&self) -> VoxelLookupDelegate<Self::MaterialIndex> {
        use noise::NoiseFn;

        let chunk_noise = Arc::clone(&self.noise);
        Box::new(move |chunk_pos, lod_level, _| {
            if let FillHint::Uniform(voxel) =
                terrain_lod_fill(ChunkKey::new(chunk_pos, lod_level))
            {
                return Box::new(move |_, _| voxel);
            }
            let chunk_min = chunk_pos * 32;
            let chunk_max = chunk_min + IVec3::splat(32 << lod_level);
            let skirt_enabled = lod_level == 1;
            let noise = Arc::clone(&chunk_noise);
            let mut cache = HashMap::<(i32, i32), f64>::new();
            Box::new(move |pos: IVec3, _| {
                if skirt_enabled
                    && (pos.cmplt(chunk_min).any() || pos.cmpge(chunk_max).any())
                {
                    return WorldVoxel::Unset;
                }
                if pos.y < 0 {
                    return WorldVoxel::Solid(3);
                }
                let sample = *cache.entry((pos.x, pos.z)).or_insert_with(|| {
                    noise.get([pos.x as f64 / 1000.0, pos.z as f64 / 1000.0]) * 50.0
                });
                if (pos.y as f64) < sample {
                    WorldVoxel::Solid(0)
                } else {
                    WorldVoxel::Air
                }
            })
        })
    }

    fn chunk_lod(
        &self,
        chunk_position: IVec3,
        _previous_lod: Option<LodLevel>,
        camera_position: Vec3,
    ) -> LodLevel {
        let camera_chunk = (camera_position / 32.0).floor();
        match chunk_position.as_vec3().distance(camera_chunk) {
            distance if distance < 30.0 => 0,
            distance if distance < 70.0 => 1,
            distance if distance < 150.0 => 2,
            distance if distance < 200.0 => 3,
            distance if distance < 300.0 => 4,
            _ => 5,
        }
    }

    fn chunk_fill_hint(&self, key: ChunkKey) -> FillHint<Self::MaterialIndex> {
        terrain_lod_fill(key)
    }

    fn attach_chunks_to_root(&self) -> bool {
        false
    }

    fn light_propagation(&self) -> Option<LightPropagation> {
        LIT.then(LightPropagation::default)
    }
}

// --- Lifecycle observation -------------------------------------------------

/// The message that marks a chunk's voxel data as published.
type PublishMessage<C> = ChunkPublished<C>;

#[derive(Resource, Default)]
struct Tracker {
    frame: u32,
    publishes: u64,
    /// Chunks whose data has been published, and that have not been retired since.
    ready: HashMap<IVec3, Entity>,
    /// Frame of the latest lifecycle change of any kind.
    last_change: u32,
}

fn track<C: VoxelWorldConfig>(
    mut tracker: ResMut<Tracker>,
    mut published: MessageReader<PublishMessage<C>>,
    mut despawned: MessageReader<ChunkWillDespawn<C>>,
    mut remeshed: MessageReader<ChunkWillRemesh<C>>,
) {
    tracker.frame += 1;
    let frame = tracker.frame;
    let mut changed = remeshed.read().count() > 0;
    for msg in published.read() {
        tracker.publishes += 1;
        tracker.ready.insert(msg.chunk_key.position, msg.entity);
        changed = true;
    }
    for msg in despawned.read() {
        tracker.ready.remove(&msg.chunk_key.position);
        changed = true;
    }
    if changed {
        tracker.last_change = frame;
    }
}

/// When the camera started flying along +X. Paced by wall-clock time, not per
/// frame, so a faster frame loop doesn't also mean a faster camera.
#[derive(Resource, Default)]
struct Flight(Option<Instant>);

fn move_camera<C: VoxelWorldConfig>(
    flight: Res<Flight>,
    mut cameras: Query<(&mut Transform, &mut GlobalTransform), With<VoxelWorldCamera<C>>>,
) {
    let Some(start) = flight.0 else {
        return;
    };
    for (mut transform, mut global) in &mut cameras {
        transform.translation.x =
            CAMERA_START.x + FLY_SPEED * start.elapsed().as_secs_f32();
        *global = GlobalTransform::from(*transform);
    }
}

fn teleport_camera<C: VoxelWorldConfig>(app: &mut App, offset: Vec3) {
    let world = app.world_mut();
    let mut cameras =
        world.query_filtered::<(&mut Transform, &mut GlobalTransform), With<VoxelWorldCamera<C>>>();
    for (mut transform, mut global) in cameras.iter_mut(world) {
        transform.translation += offset;
        *global = GlobalTransform::from(*transform);
    }
}

/// Voxel edits issued every frame while active.
#[derive(Resource, Default)]
struct EditStorm {
    active: bool,
    tick: u32,
}

fn storm_positions() -> impl Iterator<Item = IVec3> {
    // 8x8 voxels straddling the chunk corner at the origin: touches 4 chunks
    // directly and their padding neighbours.
    (-4..4).flat_map(|x| (-4..4).map(move |z| IVec3::new(x, 0, z)))
}

/// A distinct material per tick, never used by the terrain, so neither terrain nor
/// a stale publish of an earlier tick can pass for the final edit.
fn storm_value(tick: u32) -> WorldVoxel<u8> {
    WorldVoxel::Solid(10 + (tick % 200) as u8)
}

fn edit_storm<C: VoxelWorldConfig<MaterialIndex = u8>>(
    mut storm: ResMut<EditStorm>,
    mut voxel_world: VoxelWorld<C>,
) {
    if !storm.active {
        return;
    }
    storm.tick += 1;
    let value = storm_value(storm.tick);
    for pos in storm_positions() {
        voxel_world.set_voxel(pos, value);
    }
}

// --- App -------------------------------------------------------------------

/// Asks the scheduler to keep the calling thread on a performance core. Without
/// this, macOS moves busy default-priority threads to the efficiency cores.
fn prefer_performance_cores() {
    #[cfg(target_os = "macos")]
    // SAFETY: only changes the calling thread's QoS class.
    unsafe {
        libc::pthread_set_qos_class_self_np(
            libc::qos_class_t::QOS_CLASS_USER_INTERACTIVE,
            0,
        );
    }
}

fn pool(threads: usize) -> TaskPoolThreadAssignmentPolicy {
    TaskPoolThreadAssignmentPolicy {
        min_threads: threads,
        max_threads: threads,
        percent: 1.0,
        on_thread_spawn: Some(Arc::new(prefer_performance_cores)),
        on_thread_destroy: None,
    }
}

fn build_app<C: VoxelWorldConfig<MaterialIndex = u8>>(config: C) -> App {
    let mut app = App::new();
    let threads = IO_THREADS + ASYNC_COMPUTE_THREADS + COMPUTE_THREADS;
    let task_pool_options = TaskPoolOptions {
        min_total_threads: threads,
        max_total_threads: threads,
        io: pool(IO_THREADS),
        async_compute: pool(ASYNC_COMPUTE_THREADS),
        compute: pool(COMPUTE_THREADS),
    };
    app.add_plugins(MinimalPlugins.set(TaskPoolPlugin { task_pool_options }))
        .add_plugins((
            bevy::asset::AssetPlugin::default(),
            bevy::image::ImagePlugin::default(),
            bevy::mesh::MeshPlugin,
            TransformPlugin,
            bevy::camera::CameraPlugin,
        ))
        // Normally registered by RenderPlugin; the voxel material loads a shader.
        .init_asset::<bevy::shader::Shader>()
        .add_plugins(VoxelWorldPlugin::with_config(config))
        .init_resource::<Tracker>()
        .init_resource::<Flight>()
        .init_resource::<EditStorm>()
        .add_systems(Startup, spawn_camera::<C>)
        .add_systems(First, move_camera::<C>)
        .add_systems(Update, edit_storm::<C>)
        .add_systems(Last, track::<C>);
    if PROFILING.load(Ordering::Relaxed) {
        add_profiling(&mut app);
    }
    app.finish();
    app.cleanup();
    app
}

#[cfg(feature = "trace_tracy")]
fn add_profiling(app: &mut App) {
    // Rendering normally marks frame boundaries for Tracy; there is no renderer here.
    fn frame_mark() {
        bevy::log::info!(message = "finished frame", tracy.frame_mark = true);
    }
    app.add_plugins(bevy::log::LogPlugin::default())
        .add_systems(Last, frame_mark);
}

#[cfg(not(feature = "trace_tracy"))]
fn add_profiling(_: &mut App) {}

/// Where a scenario's camera starts and what it looks at, in place of
/// `CAMERA_START` looking along the diagonal of the sine terrain.
#[derive(Resource, Clone, Copy)]
struct CameraStart {
    at: Vec3,
    target: Vec3,
}

fn spawn_camera<C: VoxelWorldConfig>(
    mut commands: Commands,
    start: Option<Res<CameraStart>>,
) {
    // Positive coordinates throughout, so chunk rounding conventions agree.
    let start = start.map_or(
        CameraStart {
            at: CAMERA_START,
            target: Vec3::new(116.0, 0.0, 116.0),
        },
        |start| *start,
    );
    let transform =
        Transform::from_translation(start.at).looking_at(start.target, Vec3::Y);
    let projection = PerspectiveProjection {
        aspect_ratio: 1280.0 / 720.0,
        ..default()
    };
    // Without a render target nothing computes these, and viewport rays degenerate.
    let mut camera = Camera::default();
    camera.computed.target_info = Some(RenderTargetInfo {
        physical_size: UVec2::new(1280, 720),
        scale_factor: 1.0,
    });
    camera.computed.clip_from_view = projection.get_clip_from_view();
    commands.spawn((
        camera,
        Camera3d::default(),
        Projection::Perspective(projection),
        transform,
        GlobalTransform::from(transform),
        VoxelWorldCamera::<C>::default(),
    ));
}

fn chunk_count<C: VoxelWorldConfig>(app: &mut App) -> usize {
    let world = app.world_mut();
    world.query::<&Chunk<C>>().iter(world).count()
}

/// Fraction of chunks within spawning distance of the camera that are Ready.
fn coverage<C: VoxelWorldConfig>(app: &mut App, offsets: &[IVec3]) -> f64 {
    let world = app.world_mut();
    let camera = world
        .query_filtered::<&GlobalTransform, With<VoxelWorldCamera<C>>>()
        .single(world)
        .unwrap()
        .translation();
    let camera_chunk = (camera / 32.0).floor().as_ivec3();
    let tracker = world.resource::<Tracker>();
    let ready = offsets
        .iter()
        .filter(|offset| {
            tracker
                .ready
                .get(&(camera_chunk + **offset))
                .is_some_and(|entity| world.get_entity(*entity).is_ok())
        })
        .count();
    ready as f64 / offsets.len() as f64
}

fn wanted_offsets() -> Vec<IVec3> {
    offsets_within(SPAWNING_DISTANCE)
}

fn near_offsets() -> Vec<IVec3> {
    offsets_within(NEAR_DISTANCE)
}

fn offsets_within(r: i32) -> Vec<IVec3> {
    let mut offsets = Vec::new();
    for x in -r..=r {
        for y in -r..=r {
            for z in -r..=r {
                let offset = IVec3::new(x, y, z);
                if offset.length_squared() <= r * r {
                    offsets.push(offset);
                }
            }
        }
    }
    offsets
}

// --- Scenarios -------------------------------------------------------------

type Metrics = BTreeMap<String, f64>;

/// A scenario that takes the prefix of its metrics' names.
#[cfg(feature = "light_propagation")]
type Scenario = fn(&str) -> Metrics;

struct StreamIn {
    time_ms: f64,
    /// Time until every chunk within `NEAR_DISTANCE` of the camera was Ready, or
    /// `time_ms` if that never happened.
    near_time_ms: f64,
    frames: u32,
    /// Duration of each frame up to the last lifecycle change.
    frame_times: Vec<f64>,
}

/// Runs frames until the lifecycle has been quiet for `SETTLE_FRAMES`.
fn stream_in<C: VoxelWorldConfig>(app: &mut App) -> StreamIn {
    let start = Instant::now();
    let near = near_offsets();
    let mut near_time_ms = None;
    let mut frame_times = Vec::new();
    let mut last_count = chunk_count::<C>(app);
    let mut last_change = (0, 0.0);
    for frame in 1..=MAX_FRAMES {
        let frame_start = Instant::now();
        app.update();
        frame_times.push(frame_start.elapsed().as_secs_f64() * 1000.0);
        if near_time_ms.is_none() && coverage::<C>(app, &near) >= 1.0 {
            near_time_ms = Some(start.elapsed().as_secs_f64() * 1000.0);
        }
        let count = chunk_count::<C>(app);
        let tracker = app.world().resource::<Tracker>();
        if count != last_count || tracker.last_change == tracker.frame {
            last_change = (frame, start.elapsed().as_secs_f64() * 1000.0);
        }
        last_count = count;
        if count > 0 && frame - last_change.0 >= SETTLE_FRAMES {
            break;
        }
    }
    frame_times.truncate(last_change.0 as usize);
    StreamIn {
        frames: last_change.0,
        time_ms: last_change.1,
        near_time_ms: near_time_ms.unwrap_or(last_change.1),
        frame_times,
    }
}

fn timed_frames(
    app: &mut App,
    frames: usize,
    mut after: impl FnMut(&mut App),
) -> Vec<f64> {
    let mut times = Vec::with_capacity(frames);
    for _ in 0..frames {
        let start = Instant::now();
        app.update();
        times.push(start.elapsed().as_secs_f64() * 1000.0);
        after(app);
    }
    times
}

fn percentile(values: &[f64], p: f64) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[((sorted.len() - 1) as f64 * p).round() as usize]
}

fn frame_metrics(prefix: &str, times: &[f64], metrics: &mut Metrics) {
    for (name, p) in [("p50_ms", 0.5), ("p95_ms", 0.95), ("p99_ms", 0.99)] {
        metrics.insert(format!("{prefix}.{name}"), percentile(times, p));
    }
    // p99 is the 12th-slowest of 1200 frames, so a short burst of long frames
    // barely moves it. The worst-1% mean, the maximum and the stutter counts
    // below reflect every slow frame.
    let mut slowest = times.to_vec();
    slowest.sort_by(|a, b| b.total_cmp(a));
    let worst = &slowest[..times.len().div_ceil(100)];
    metrics.insert(
        format!("{prefix}.worst_1pct_ms"),
        worst.iter().sum::<f64>() / worst.len() as f64,
    );
    metrics.insert(format!("{prefix}.max_ms"), slowest[0]);
    for limit in STUTTER_MS {
        let count = times.iter().filter(|t| **t > limit as f64).count();
        metrics.insert(format!("{prefix}.over_{limit}ms"), count as f64);
    }
}

fn scenario_stream_in<C: VoxelWorldConfig<MaterialIndex = u8> + Default>(
    prefix: &str,
) -> Metrics {
    let mut app = build_app(C::default());
    let result = stream_in::<C>(&mut app);
    let ready = app.world().resource::<Tracker>().ready.len();
    let publishes = app.world().resource::<Tracker>().publishes;
    let mut metrics = Metrics::new();
    metrics.insert(format!("{prefix}.time_ms"), result.time_ms);
    metrics.insert(format!("{prefix}.near_time_ms"), result.near_time_ms);
    metrics.insert(format!("{prefix}.frames"), result.frames as f64);
    metrics.insert(format!("{prefix}.ready_chunks"), ready as f64);
    metrics.insert(format!("{prefix}.publishes"), publishes as f64);
    metrics.insert(
        format!("{prefix}.chunks_per_s"),
        ready as f64 / (result.time_ms / 1000.0),
    );
    metrics.insert(
        format!("{prefix}.coverage"),
        coverage::<C>(&mut app, &wanted_offsets()),
    );
    frame_metrics(prefix, &result.frame_times, &mut metrics);
    metrics
}

#[cfg(feature = "light_propagation")]
fn light_is_settled<C: VoxelWorldConfig>(app: &mut App) -> bool {
    let mut state = SystemState::<VoxelWorld<C>>::new(app.world_mut());
    state.get(app.world()).unwrap().light_is_settled()
}

/// Runs frames after a stream until the world's light is settled, for at
/// most a minute. Returns how long that took, in milliseconds.
#[cfg(feature = "light_propagation")]
fn settle_light<C: VoxelWorldConfig>(app: &mut App) -> f64 {
    let start = Instant::now();
    while start.elapsed() < std::time::Duration::from_secs(60) {
        if light_is_settled::<C>(app) {
            break;
        }
        app.update();
    }
    assert!(light_is_settled::<C>(app), "light didn't settle");
    start.elapsed().as_secs_f64() * 1000.0
}

/// How a stream into a lit world went, up to its light settling.
#[cfg(feature = "light_propagation")]
struct LitStreamIn {
    /// As `stream_in`'s.
    stream: StreamIn,
    /// Time until the chunk lifecycle was quiet and light settled, whichever
    /// came later.
    settled_ms: f64,
    /// Time until the lifecycle was quiet and no relight was queued, running
    /// or waiting to land, whichever came later: `settled_ms` without the
    /// sky probes and fill probes that run on after the stream.
    relit_ms: f64,
    /// Whether the wall-clock cap ended the run before light settled.
    capped: bool,
}

/// Runs frames until the lifecycle has been quiet and light settled for
/// `SETTLE_FRAMES`, or for at most `cap` of wall-clock time. Relights and
/// sky and fill probes count, and so do the remeshes they cause.
///
/// Calls `at_quiet` once, on the first frame the lifecycle has been quiet for
/// `SETTLE_FRAMES`, where `stream_in` would have stopped.
#[cfg(feature = "light_propagation")]
fn stream_in_settled<C: VoxelWorldConfig>(
    app: &mut App,
    cap: std::time::Duration,
    mut at_quiet: impl FnMut(&mut App),
) -> LitStreamIn {
    let start = Instant::now();
    let near = near_offsets();
    let mut settled_state = SystemState::<VoxelWorld<C>>::new(app.world_mut());
    let mut near_time_ms = None;
    let mut frame_times = Vec::new();
    let mut last_count = chunk_count::<C>(app);
    let mut last_change = (0, 0.0);
    // The frame light last settled on, and the time at its end.
    let mut settled_at: Option<(u32, f64)> = None;
    let mut relit_at: Option<f64> = None;
    let mut frame = 0;
    let mut was_quiet = false;
    let capped = loop {
        frame += 1;
        let frame_start = Instant::now();
        app.update();
        frame_times.push(frame_start.elapsed().as_secs_f64() * 1000.0);
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        if near_time_ms.is_none() && coverage::<C>(app, &near) >= 1.0 {
            near_time_ms = Some(elapsed_ms);
        }
        let count = chunk_count::<C>(app);
        let tracker = app.world().resource::<Tracker>();
        if count != last_count || tracker.last_change == tracker.frame {
            last_change = (frame, elapsed_ms);
        }
        last_count = count;
        let settled = settled_state.get(app.world()).unwrap().light_is_settled();
        let relit = bevy_voxel_world::light_bench::relights_are_settled::<C>(app.world());
        relit_at = match (relit, relit_at) {
            (false, _) => None,
            (true, None) => Some(elapsed_ms),
            (true, at) => at,
        };
        settled_at = match (settled, settled_at) {
            (false, _) => None,
            (true, None) => Some((frame, elapsed_ms)),
            (true, at) => at,
        };
        let quiet = count > 0 && frame - last_change.0 >= SETTLE_FRAMES;
        if quiet && !was_quiet {
            was_quiet = true;
            at_quiet(app);
        }
        if quiet && settled_at.is_some_and(|(at, _)| frame - at >= SETTLE_FRAMES) {
            break false;
        }
        if start.elapsed() >= cap {
            break true;
        }
    };
    let settled_ms = settled_at.map_or(f64::NAN, |(_, ms)| ms.max(last_change.1));
    let last_frame = last_change.0.max(settled_at.map_or(frame, |(at, _)| at));
    frame_times.truncate(last_frame as usize);
    LitStreamIn {
        stream: StreamIn {
            frames: last_frame,
            time_ms: last_change.1,
            near_time_ms: near_time_ms.unwrap_or(last_change.1),
            frame_times,
        },
        settled_ms,
        relit_ms: relit_at.map_or(f64::NAN, |ms| ms.max(last_change.1)),
        capped,
    }
}

/// `stream_in` on a lit world, with what working out its direct sky and
/// solving its light cost in chunk tasks, what sky fill cost, and how long
/// sky probes and fill probes take to bring its light to rest after the
/// stream. Light at rest starts no work.
#[cfg(feature = "light_propagation")]
fn scenario_stream_in_lit<C: VoxelWorldConfig<MaterialIndex = u8> + Default>(
    prefix: &str,
) -> Metrics {
    use bevy_voxel_world::light_bench::light_stats;

    let mut app = build_app(C::default());
    let mut streamed = None;
    let lit =
        stream_in_settled::<C>(&mut app, std::time::Duration::from_secs(120), |app| {
            streamed = light_stats::<C>(app.world());
        });
    let result = &lit.stream;
    let streamed = streamed.expect("a lit world");
    settle_light::<C>(&mut app);
    let rest_ms = (lit.settled_ms - result.time_ms).max(0.0);
    let stats = light_stats::<C>(app.world()).expect("a lit world");
    for _ in 0..IDLE_LIGHT_FRAMES {
        app.update();
    }
    let idle = light_stats::<C>(app.world()).expect("a lit world");
    let mut metrics = Metrics::new();
    metrics.insert(format!("{prefix}.rest_ms"), rest_ms);
    metrics.insert(format!("{prefix}.probes_in_stream"), streamed.probes as f64);
    metrics.insert(format!("{prefix}.probes"), stats.probes as f64);
    metrics.insert(
        format!("{prefix}.probe_lookups"),
        stats.probe_lookups as f64,
    );
    metrics.insert(
        format!("{prefix}.probes_discarded"),
        stats.probes_discarded as f64,
    );
    metrics.insert(
        format!("{prefix}.record_kib"),
        stats.record_bytes as f64 / 1024.0,
    );
    metrics.insert(
        format!("{prefix}.field_kib"),
        stats.field_bytes as f64 / 1024.0,
    );
    metrics.insert(
        format!("{prefix}.idle_probes"),
        (idle.probes - stats.probes) as f64,
    );
    metrics.insert(
        format!("{prefix}.idle_relights"),
        (idle.relights - stats.relights) as f64,
    );
    metrics.insert(format!("{prefix}.time_ms"), result.time_ms);
    metrics.insert(format!("{prefix}.settled_ms"), lit.settled_ms);
    metrics.insert(format!("{prefix}.relit_ms"), lit.relit_ms);
    metrics.insert(format!("{prefix}.near_time_ms"), result.near_time_ms);
    metrics.insert(format!("{prefix}.frames"), result.frames as f64);
    let solves = stats.solves.max(1) as f64;
    metrics.insert(format!("{prefix}.solves"), stats.solves as f64);
    metrics.insert(
        format!("{prefix}.pool_solve_us_per_chunk"),
        stats.solve_ms * 1000.0 / solves,
    );
    metrics.insert(
        format!("{prefix}.column_us_per_chunk"),
        stats.column_ms * 1000.0 / solves,
    );
    metrics.insert(format!("{prefix}.walks"), stats.walks as f64);
    metrics.insert(format!("{prefix}.walk_lookups"), stats.walk_lookups as f64);
    metrics.insert(format!("{prefix}.relights"), stats.relights as f64);
    // Hinted air's cost: relight CPU for light entries against all of it,
    // and what the light entries and hinted air keep at rest.
    metrics.insert(format!("{prefix}.relight_ms"), stats.relight_ms);
    metrics.insert(
        format!("{prefix}.entry_relights"),
        stats.entry_relights as f64,
    );
    metrics.insert(format!("{prefix}.entry_relight_ms"), stats.entry_relight_ms);
    metrics.insert(format!("{prefix}.entries"), stats.entries as f64);
    metrics.insert(
        format!("{prefix}.entry_kib"),
        stats.entry_bytes as f64 / 1024.0,
    );
    metrics.insert(
        format!("{prefix}.air_positions"),
        stats.air_positions as f64,
    );
    metrics.insert(
        format!("{prefix}.publishes"),
        app.world().resource::<Tracker>().publishes as f64,
    );
    metrics.insert(
        format!("{prefix}.main_thread_total_ms"),
        stats.main_thread_ms,
    );
    // Sky fill's gates on the bench's worlds: no seeded solves, no relights
    // from fill, and field updates of at most 2 µs per chunk during the
    // stream. Checked by hand; nothing compares them.
    let chunks = app.world().resource::<Tracker>().publishes.max(1) as f64;
    metrics.insert(format!("{prefix}.fill_seeded"), stats.fill_seeded as f64);
    metrics.insert(
        format!("{prefix}.fill_seed_us_per_seeded"),
        stats.fill_seed_ms * 1000.0 / stats.fill_seeded.max(1) as f64,
    );
    metrics.insert(
        format!("{prefix}.fill_relights"),
        stats.fill_relights as f64,
    );
    metrics.insert(
        format!("{prefix}.fill_relights_skipped"),
        stats.fill_relights_skipped as f64,
    );
    metrics.insert(
        format!("{prefix}.fill_update_us_per_chunk_in_stream"),
        streamed.fill_update_ms * 1000.0 / chunks,
    );
    metrics.insert(
        format!("{prefix}.fill_update_us_per_chunk"),
        stats.fill_update_ms * 1000.0 / chunks,
    );
    metrics.insert(
        format!("{prefix}.fill_main_thread_us_per_chunk_in_stream"),
        streamed.fill_main_thread_ms * 1000.0 / chunks,
    );
    metrics.insert(format!("{prefix}.fill_probes"), stats.fill_probes as f64);
    metrics.insert(
        format!("{prefix}.idle_fill_probes"),
        (idle.fill_probes - stats.fill_probes) as f64,
    );
    metrics
}

/// `stream_in` on a world with levels of detail, with what coarse chunks'
/// light cost: their solves in chunk tasks, their walks' lookups, the slots
/// they took and the copies of their slots they keep.
///
/// The cost gate for a coarse chunk's solve, walk included, is 250 µs a
/// chunk in `coarse_solve_us_per_chunk`, set from the first runs in real
/// chunk tasks at about 150 µs. It's checked by hand; nothing compares it.
#[cfg(feature = "light_propagation")]
fn scenario_stream_in_lod<C: VoxelWorldConfig<MaterialIndex = u8> + Default>(
    prefix: &str,
) -> Metrics {
    use bevy_voxel_world::light_bench::light_stats;

    let mut app = build_app(C::default());
    let result = stream_in::<C>(&mut app);
    let stats = light_stats::<C>(app.world()).expect("a lit world");
    let coarse = stats.coarse_solves.max(1) as f64;
    let mut metrics = Metrics::new();
    metrics.insert(format!("{prefix}.time_ms"), result.time_ms);
    metrics.insert(
        format!("{prefix}.chunks"),
        chunk_count::<C>(&mut app) as f64,
    );
    metrics.insert(format!("{prefix}.solves"), stats.solves as f64);
    metrics.insert(
        format!("{prefix}.coarse_solves"),
        stats.coarse_solves as f64,
    );
    metrics.insert(
        format!("{prefix}.coarse_solve_us_per_chunk"),
        stats.coarse_solve_ms * 1000.0 / coarse,
    );
    metrics.insert(
        format!("{prefix}.coarse_lookups_per_chunk"),
        stats.coarse_lookups as f64 / coarse,
    );
    metrics.insert(
        format!("{prefix}.slot_allocations"),
        stats.slot_allocations as f64,
    );
    metrics.insert(format!("{prefix}.evictions"), stats.evictions as f64);
    metrics.insert(
        format!("{prefix}.coarse_copy_mib"),
        stats.coarse_copy_bytes as f64 / (1024.0 * 1024.0),
    );
    frame_metrics(prefix, &result.frame_times, &mut metrics);
    metrics
}

/// The wall-clock cap on one run of `caves`.
#[cfg(feature = "light_propagation")]
const CAVE_CAP: std::time::Duration = std::time::Duration::from_secs(120);

/// The bytes of GPU a slot takes: 34³ voxels of one byte.
#[cfg(feature = "light_propagation")]
const SLOT_BYTES: f64 = 39_304.0;

/// The thresholds of `caves`; see `scenario_caves`. Each is the most a metric
/// may reach. The two counts sit about 20% above a run on an idle machine
/// (4.55 solves per arrival, 1.93 visible changes per shown chunk); the costs
/// have no threshold.
#[cfg(feature = "light_propagation")]
const CAVE_GATES: [(&str, f64); 4] = [
    ("solves_per_arrival", CAVE_SOLVES_PER_ARRIVAL),
    ("visible_per_shown", CAVE_VISIBLE_PER_SHOWN),
    ("pool_solve_us_per_chunk", CAVE_SOLVE_US),
    ("relight_us_per_relight", CAVE_RELIGHT_US),
];
#[cfg(feature = "light_propagation")]
const CAVE_SOLVES_PER_ARRIVAL: f64 = 5.5;
#[cfg(feature = "light_propagation")]
const CAVE_VISIBLE_PER_SHOWN: f64 = 2.4;
#[cfg(feature = "light_propagation")]
const CAVE_SOLVE_US: f64 = f64::INFINITY;
#[cfg(feature = "light_propagation")]
const CAVE_RELIGHT_US: f64 = f64::INFINITY;

/// A stream into the reference scene's caves at the default light settings,
/// sky reach 30 with round spread, ending once light is settled. Reports how
/// often light changes after chunks first show it, and why:
///
/// - `solves_per_arrival`: solves in chunk tasks and relights, per chunk
///   whose light landed;
/// - `changed_batches_per_arrival`: frames after a chunk's arrival in which
///   its light changed, per arrival;
/// - `visible_per_shown`: frames after a chunk's arrival in which a relight
///   changed light its own faces show, per chunk with any such voxel, as the
///   probes counted them; `visible_with_shells_per_shown` also counts frames
///   in which only light in front of its faces from a neighbour changed;
/// - `visible_from_*_per_shown`: the frames of either, by cause: a
///   neighbour arriving, a column found out by sky probes, a shell filling
///   in in front of the chunk's solid voxels, and anything else. A frame
///   with changes of several causes counts for the first.
///
/// It also reports what the solves cost in chunk tasks, emissive scan
/// included, and the slots and memory light took at their peak.
///
/// Thresholds, checked here and reported as `gate_failures`, are in
/// `CAVE_GATES`.
#[cfg(feature = "light_propagation")]
fn scenario_caves() -> Metrics {
    use bevy_voxel_world::light_bench::{change_counts, count_changes, light_stats};

    type World = CaveWorld;
    let prefix = "caves";
    cave_table();
    let mut app = build_app(CaveWorld);
    app.insert_resource(CAVE_CAMERA);
    count_changes::<World>(app.world_mut());
    let result = stream_in_settled::<World>(&mut app, CAVE_CAP, |_| {});
    let stats = light_stats::<World>(app.world()).expect("a lit world");
    let counts = change_counts::<World>(app.world()).expect("counted");
    for _ in 0..IDLE_LIGHT_FRAMES {
        app.update();
    }
    let idle = light_stats::<World>(app.world()).expect("a lit world");

    let mut metrics = Metrics::new();
    let mut put = |name: &str, value: f64| {
        metrics.insert(format!("{prefix}.{name}"), value);
    };
    let arrivals = counts.arrivals.max(1) as f64;
    let shown = counts.shown.max(1) as f64;
    put("time_ms", result.stream.time_ms);
    put("settled_ms", result.settled_ms);
    put("relit_ms", result.relit_ms);
    put(
        "rest_ms",
        (result.settled_ms - result.stream.time_ms).max(0.0),
    );
    put("capped", result.capped as u8 as f64);
    put("arrivals", counts.arrivals as f64);
    put("shown", counts.shown as f64);
    put("solves", stats.solves as f64);
    put("relights", stats.relights as f64);
    // Every solve, a chunk task's or a relight's; relights stopped before
    // solving aren't solves.
    put("solves_per_arrival", stats.solves as f64 / arrivals);
    put("changed_relights", counts.changed_relights as f64);
    put(
        "changed_batches_per_arrival",
        counts.changed_batches as f64 / arrivals,
    );
    // A chunk's own relights, which the probes counted, and the same with
    // its shell filling in.
    let visible = counts.visible_from_arrival
        + counts.visible_from_column
        + counts.visible_from_other;
    put("visible_per_shown", visible as f64 / shown);
    put(
        "visible_with_shells_per_shown",
        counts.visible_with_shells as f64 / shown,
    );
    put(
        "visible_from_arrival_per_shown",
        counts.visible_from_arrival as f64 / shown,
    );
    put(
        "visible_from_column_per_shown",
        counts.visible_from_column as f64 / shown,
    );
    put(
        "visible_from_shell_per_shown",
        counts.visible_from_shell as f64 / shown,
    );
    put(
        "visible_from_other_per_shown",
        counts.visible_from_other as f64 / shown,
    );
    put("emissive_scans", stats.emissive_scans as f64);
    put(
        "pool_solve_us_per_chunk",
        stats.solve_ms * 1000.0 / stats.solves.max(1) as f64,
    );
    put(
        "relight_us_per_relight",
        stats.relight_ms * 1000.0 / stats.relights.max(1) as f64,
    );
    put("main_thread_total_ms", stats.main_thread_ms);
    put("main_thread_max_frame_ms", stats.max_frame_ms);
    put("probes", stats.probes as f64);
    put("fill_seeded", stats.fill_seeded as f64);
    put("fill_relights", stats.fill_relights as f64);
    put("fill_probes", stats.fill_probes as f64);
    put("idle_relights", (idle.relights - stats.relights) as f64);
    put("idle_probes", (idle.probes - stats.probes) as f64);
    put(
        "idle_fill_probes",
        (idle.fill_probes - stats.fill_probes) as f64,
    );
    put("slot_allocations", stats.slot_allocations as f64);
    put("peak_slots", stats.peak_slots as f64);
    put("evictions", stats.evictions as f64);
    put("starved", stats.starved as f64);
    put(
        "gpu_light_mib_at_peak",
        stats.peak_slots as f64 * SLOT_BYTES / (1024.0 * 1024.0),
    );
    put(
        "cpu_light_mib",
        stats.light_bytes as f64 / (1024.0 * 1024.0),
    );
    put(
        "cpu_light_bytes_per_mixed_voxel",
        stats.light_bytes as f64 / (stats.mixed_chunks.max(1) as f64 * 32768.0),
    );
    put("record_kib", stats.record_bytes as f64 / 1024.0);
    frame_metrics(prefix, &result.stream.frame_times, &mut metrics);

    let failures: Vec<String> = CAVE_GATES
        .iter()
        .filter(|(name, most)| metrics[&format!("{prefix}.{name}")] > *most)
        .map(|(name, most)| format!("{name} > {most}"))
        .collect();
    if !failures.is_empty() || result.capped {
        eprintln!(
            "{prefix}: missed its thresholds: {failures:?}{}",
            if result.capped { " (capped)" } else { "" }
        );
    }
    metrics.insert(
        format!("{prefix}.gate_failures"),
        (failures.len() + result.capped as usize) as f64,
    );
    metrics
}

/// How many frames `lod_scale` times at rest.
#[cfg(feature = "light_propagation")]
const SCALE_REST_FRAMES: usize = 300;

/// The world of the `noise_terrain_lod` example, streamed from the
/// example's camera, unlit and then lit at the default light settings.
/// Reports the chunks streamed, what coarse chunks' solves cost, and frame
/// times at rest for both, so light's cost per frame at rest shows as the
/// difference, with the relights, probes and main thread light took at rest.
///
/// It runs only when named, as it streams tens of thousands of chunks.
#[cfg(feature = "light_propagation")]
fn scenario_lod_scale() -> Metrics {
    use bevy_voxel_world::light_bench::light_stats;

    let prefix = "lod_scale";
    let cap = std::time::Duration::from_secs(600);
    let mut metrics = Metrics::new();

    let mut unlit = build_app(TerrainLodWorld::<false>::default());
    unlit.insert_resource(TERRAIN_LOD_CAMERA);
    let streamed = stream_in_settled::<TerrainLodWorld<false>>(&mut unlit, cap, |_| {});
    metrics.insert(format!("{prefix}.unlit_time_ms"), streamed.stream.time_ms);
    metrics.insert(
        format!("{prefix}.unlit_chunks"),
        chunk_count::<TerrainLodWorld<false>>(&mut unlit) as f64,
    );
    let times = timed_frames(&mut unlit, SCALE_REST_FRAMES, |_| {});
    frame_metrics(&format!("{prefix}.unlit_rest"), &times, &mut metrics);
    drop(unlit);

    type Lit = TerrainLodWorld<true>;
    let mut lit = build_app(Lit::default());
    lit.insert_resource(TERRAIN_LOD_CAMERA);
    let result = stream_in_settled::<Lit>(&mut lit, cap, |_| {});
    let stats = light_stats::<Lit>(lit.world()).expect("a lit world");
    let mut frame_ms = Vec::with_capacity(SCALE_REST_FRAMES);
    let mut last = stats.main_thread_ms;
    let times = timed_frames(&mut lit, SCALE_REST_FRAMES, |app| {
        let now = light_stats::<Lit>(app.world()).unwrap().main_thread_ms;
        frame_ms.push(now - last);
        last = now;
    });
    let idle = light_stats::<Lit>(lit.world()).expect("a lit world");
    let mut put = |name: &str, value: f64| {
        metrics.insert(format!("{prefix}.{name}"), value);
    };
    put("lit_time_ms", result.stream.time_ms);
    put("lit_settled_ms", result.settled_ms);
    put("lit_capped", result.capped as u8 as f64);
    put("lit_chunks", chunk_count::<Lit>(&mut lit) as f64);
    put("solves", stats.solves as f64);
    put("coarse_solves", stats.coarse_solves as f64);
    put(
        "coarse_solve_us_per_chunk",
        stats.coarse_solve_ms * 1000.0 / stats.coarse_solves.max(1) as f64,
    );
    put(
        "coarse_lookups_per_chunk",
        stats.coarse_lookups as f64 / stats.coarse_solves.max(1) as f64,
    );
    put(
        "pool_solve_us_per_chunk",
        stats.solve_ms * 1000.0 / stats.solves.max(1) as f64,
    );
    put("relights", stats.relights as f64);
    put("slot_allocations", stats.slot_allocations as f64);
    put("peak_slots", stats.peak_slots as f64);
    put("evictions", stats.evictions as f64);
    put("starved", stats.starved as f64);
    put(
        "coarse_copy_mib",
        stats.coarse_copy_bytes as f64 / (1024.0 * 1024.0),
    );
    put(
        "cpu_light_mib",
        stats.light_bytes as f64 / (1024.0 * 1024.0),
    );
    put(
        "gpu_light_mib_at_peak",
        stats.peak_slots as f64 * SLOT_BYTES / (1024.0 * 1024.0),
    );
    put("rest_relights", (idle.relights - stats.relights) as f64);
    put("rest_probes", (idle.probes - stats.probes) as f64);
    put(
        "rest_fill_probes",
        (idle.fill_probes - stats.fill_probes) as f64,
    );
    put(
        "rest_light_main_thread_us_per_frame",
        frame_ms.iter().sum::<f64>() * 1000.0 / frame_ms.len() as f64,
    );
    frame_metrics(&format!("{prefix}.lit_rest"), &times, &mut metrics);
    metrics
}

/// How long a frame of `edits_lit` takes at least, as at 60 frames a second,
/// so relights started on one frame have had a frame's time to run.
#[cfg(feature = "light_propagation")]
const PACED_FRAME: std::time::Duration = std::time::Duration::from_micros(16_667);

/// The gate on a single voxel edit's light on native: it lands within this
/// many frames after the edit's frame...
#[cfg(feature = "light_propagation")]
const EDIT_FRAMES_GATE: u32 = 2;
/// ...and relighting takes at most this much main thread in any frame.
#[cfg(feature = "light_propagation")]
const EDIT_MAIN_THREAD_GATE_US: f64 = 1000.0;

/// The height of the sine terrain's surface at `x`, `z`: the first voxel of
/// air above it.
#[cfg(feature = "light_propagation")]
fn sine_height(x: i32, z: i32) -> i32 {
    (8.0 * (x as f32 / 23.0).sin() + 8.0 * (z as f32 / 17.0).cos()) as i32
}

/// Single voxel edits on the lit sine terrain once its light is at rest:
/// placing a block on its surface and digging one from it, each followed by
/// its undo, mid-chunk and against chunk faces, then a radius-8 bomb and its
/// undo. Reports each edit's frames to land, counted from the edit's frame
/// with frames paced at 60 a second, and the most main thread relighting took
/// in any frame.
///
/// Gate, checked here and reported as `gate_failures`: a single voxel edit
/// lands within 2 frames and relighting takes at most 1 ms of main thread in
/// any frame. The bomb has no gate.
#[cfg(feature = "light_propagation")]
fn scenario_edits_lit() -> Metrics {
    use bevy_voxel_world::light_bench::light_stats;

    type Lit = LitBenchWorld<SINE>;
    let prefix = "edits_lit";
    let mut app = build_app(Lit::default());
    stream_in::<Lit>(&mut app);
    settle_light::<Lit>(&mut app);

    let main_ms = |app: &App| light_stats::<Lit>(app.world()).unwrap().main_thread_ms;
    let relights = |app: &App| light_stats::<Lit>(app.world()).unwrap().relights;
    let set = |app: &mut App, edits: &[(IVec3, WorldVoxel<u8>)]| {
        let mut state = SystemState::<VoxelWorld<Lit>>::new(app.world_mut());
        let mut voxel_world = state.get_mut(app.world_mut()).unwrap();
        for (position, voxel) in edits {
            voxel_world.set_voxel(*position, *voxel);
        }
    };
    // Runs paced frames from the edit's frame until light settles. Returns
    // the frames after the edit's, and the most main thread in one frame, in
    // microseconds.
    let run_edit = |app: &mut App, edits: &[(IVec3, WorldVoxel<u8>)]| -> (u32, f64) {
        set(app, edits);
        let mut frames = 0;
        let mut worst_us: f64 = 0.0;
        loop {
            let start = Instant::now();
            let before = main_ms(app);
            app.update();
            worst_us = worst_us.max((main_ms(app) - before) * 1000.0);
            if light_is_settled::<Lit>(app) {
                break;
            }
            frames += 1;
            assert!(frames < MAX_FRAMES, "the edit's light didn't land");
            std::thread::sleep(PACED_FRAME.saturating_sub(start.elapsed()));
        }
        // A frame's rest, as a game's frame would give the edit's frame.
        std::thread::sleep(PACED_FRAME);
        (frames, worst_us)
    };

    let mut sites = Vec::new();
    for (x, z) in [(16, 16), (48, 20), (80, 72), (31, 40), (32, 40), (100, 63)] {
        let height = sine_height(x, z);
        let surface = IVec3::new(x, height - 1, z);
        let above = IVec3::new(x, height, z);
        sites.push(vec![(above, WorldVoxel::Solid(0))]);
        sites.push(vec![(above, WorldVoxel::Air)]);
        sites.push(vec![(surface, WorldVoxel::Air)]);
        sites.push(vec![(surface, WorldVoxel::Solid(0))]);
    }
    let mut frames = Vec::new();
    let mut worst = Vec::new();
    let mut tasks = Vec::new();
    for edit in &sites {
        let started = relights(&app);
        let (edit_frames, edit_worst) = run_edit(&mut app, edit);
        frames.push(edit_frames as f64);
        worst.push(edit_worst);
        tasks.push((relights(&app) - started) as f64);
    }
    let failures = frames
        .iter()
        .zip(&worst)
        .filter(|(frames, worst)| {
            **frames > EDIT_FRAMES_GATE as f64 || **worst > EDIT_MAIN_THREAD_GATE_US
        })
        .count();
    if failures > 0 {
        eprintln!(
            "{prefix}: {failures} of {} single voxel edits missed the gate \
             (frames {frames:?}, main thread µs {worst:?})",
            sites.len()
        );
    }

    let centre = IVec3::new(48, sine_height(48, 48) - 2, 48);
    let bomb: Vec<IVec3> = (-8..=8)
        .flat_map(|z| {
            (-8..=8).flat_map(move |y| (-8..=8).map(move |x| IVec3::new(x, y, z)))
        })
        .filter(|offset| offset.length_squared() <= 64)
        .map(|offset| centre + offset)
        .collect();
    let dug: Vec<(IVec3, WorldVoxel<u8>)> =
        bomb.iter().map(|p| (*p, WorldVoxel::Air)).collect();
    let (bomb_frames, bomb_worst) = run_edit(&mut app, &dug);
    let undo: Vec<(IVec3, WorldVoxel<u8>)> = bomb
        .iter()
        .map(|p| {
            let voxel = if p.y < sine_height(p.x, p.z) {
                WorldVoxel::Solid(if p.y < sine_height(p.x, p.z) - 3 {
                    1
                } else {
                    0
                })
            } else {
                WorldVoxel::Air
            };
            (*p, voxel)
        })
        .collect();
    let (undo_frames, undo_worst) = run_edit(&mut app, &undo);

    let mut metrics = Metrics::new();
    let max = |values: &[f64]| values.iter().copied().fold(0.0, f64::max);
    metrics.insert(format!("{prefix}.edits"), sites.len() as f64);
    metrics.insert(format!("{prefix}.max_frames_to_land"), max(&frames));
    metrics.insert(
        format!("{prefix}.p50_frames_to_land"),
        percentile(&frames, 0.5),
    );
    metrics.insert(format!("{prefix}.max_main_thread_us"), max(&worst));
    metrics.insert(
        format!("{prefix}.p50_main_thread_us"),
        percentile(&worst, 0.5),
    );
    metrics.insert(format!("{prefix}.max_relights"), max(&tasks));
    metrics.insert(format!("{prefix}.p50_relights"), percentile(&tasks, 0.5));
    metrics.insert(format!("{prefix}.gate_failures"), failures as f64);
    metrics.insert(format!("{prefix}.bomb_frames_to_land"), bomb_frames as f64);
    metrics.insert(format!("{prefix}.bomb_main_thread_us"), bomb_worst);
    metrics.insert(
        format!("{prefix}.bomb_undo_frames_to_land"),
        undo_frames as f64,
    );
    metrics.insert(format!("{prefix}.bomb_undo_main_thread_us"), undo_worst);
    metrics
}

/// Changes of light settings on the lit sine terrain once its light is at
/// rest, each made through the config resource, as a game's system would:
/// sky reach, voxel reach, a sky ceiling in place of the clearance, sky light
/// off, and back on. Reports for each how long light took to settle, in
/// frames run as fast as they go, how many relights it started, and the most
/// main thread light took in one frame.
#[cfg(feature = "light_propagation")]
fn scenario_settings_lit() -> Metrics {
    use bevy_voxel_world::light_bench::light_stats;

    let prefix = "settings_lit";
    let mut app = build_app(SettingsWorld::default());
    stream_in::<SettingsWorld>(&mut app);
    settle_light::<SettingsWorld>(&mut app);

    type Change = (&'static str, fn(&mut LightPropagation));
    let changes: [Change; 5] = [
        ("sky_reach", |light| light.sky_reach = 16),
        ("voxel_reach", |light| light.voxel_reach = 8),
        ("sky_ceiling", |light| light.sky = Sky::Ceiling { y: 64 }),
        ("sky_off", |light| light.sky = Sky::Off),
        ("sky_on", |light| light.sky = Sky::default()),
    ];
    let mut metrics = Metrics::new();
    for (name, change) in changes {
        let stats = |app: &App| light_stats::<SettingsWorld>(app.world()).unwrap();
        let before = stats(&app);
        change(&mut app.world_mut().resource_mut::<SettingsWorld>().light);
        let start = Instant::now();
        let mut frames = 0;
        let mut worst_ms: f64 = 0.0;
        loop {
            let main = stats(&app).main_thread_ms;
            app.update();
            worst_ms = worst_ms.max(stats(&app).main_thread_ms - main);
            frames += 1;
            if light_is_settled::<SettingsWorld>(&mut app) {
                break;
            }
            assert!(frames < MAX_FRAMES, "light didn't settle after {name}");
        }
        let settle_ms = start.elapsed().as_secs_f64() * 1000.0;
        let after = stats(&app);
        metrics.insert(format!("{prefix}.{name}.settle_ms"), settle_ms);
        metrics.insert(format!("{prefix}.{name}.frames"), frames as f64);
        metrics.insert(
            format!("{prefix}.{name}.relights"),
            (after.relights - before.relights) as f64,
        );
        metrics.insert(
            format!("{prefix}.{name}.probes"),
            (after.probes - before.probes) as f64,
        );
        metrics.insert(format!("{prefix}.{name}.max_main_thread_ms"), worst_ms);
    }
    metrics.insert(
        format!("{prefix}.chunks"),
        chunk_count::<SettingsWorld>(&mut app) as f64,
    );
    metrics
}

/// `stream_in` on the world with lamps, lit, with what solving its light
/// cost: in chunk tasks and relight tasks on the pool, against the same chunks
/// solved one after another on one thread, and on the main thread landing
/// light and starting relights.
#[cfg(feature = "light_propagation")]
fn scenario_stream_in_lamps() -> Metrics {
    use bevy_voxel_world::light_bench::{light_stats, solve_published_serially};

    type Lit = LampWorld<true>;
    let prefix = "stream_in_lamps";
    let mut metrics = scenario_stream_in::<Lit>(prefix);
    let mut app = build_app(Lit::default());
    stream_in::<Lit>(&mut app);
    let rest_ms = settle_light::<Lit>(&mut app);
    metrics.insert(format!("{prefix}.rest_ms"), rest_ms);
    let stats = light_stats::<Lit>(app.world()).expect("a lit world");
    let (serial_solves, serial_ms) = solve_published_serially::<Lit>(app.world());
    metrics.insert(format!("{prefix}.solves"), stats.solves as f64);
    metrics.insert(
        format!("{prefix}.emissive_scans"),
        stats.emissive_scans as f64,
    );
    metrics.insert(
        format!("{prefix}.pool_solve_us_per_chunk"),
        stats.solve_ms * 1000.0 / stats.solves.max(1) as f64,
    );
    metrics.insert(
        format!("{prefix}.serial_solve_us_per_chunk"),
        serial_ms * 1000.0 / serial_solves.max(1) as f64,
    );
    metrics.insert(format!("{prefix}.snapshot_total_ms"), stats.snapshot_ms);
    metrics.insert(format!("{prefix}.relights"), stats.relights as f64);
    metrics.insert(
        format!("{prefix}.relights_per_solve"),
        stats.relights as f64 / stats.solves.max(1) as f64,
    );
    metrics.insert(
        format!("{prefix}.main_thread_total_ms"),
        stats.main_thread_ms,
    );
    metrics.insert(
        format!("{prefix}.main_thread_max_frame_ms"),
        stats.max_frame_ms,
    );
    for (name, count) in [
        ("slot_allocations", stats.slot_allocations),
        ("late_allocations", stats.late_allocations),
        ("slots_given_back", stats.slots_given_back),
        ("evictions", stats.evictions),
        ("plane_writes", stats.plane_writes),
        ("full_writes", stats.full_writes),
    ] {
        metrics.insert(format!("{prefix}.{name}"), count as f64);
    }
    metrics
}

/// Every chunk around the camera becomes unwanted, and a whole new area wanted, on
/// the same frame.
fn scenario_teleport() -> Metrics {
    let mut app = build_app(SineWorld::default());
    stream_in::<SineWorld>(&mut app);
    teleport_camera::<SineWorld>(&mut app, TELEPORT);
    let result = stream_in::<SineWorld>(&mut app);
    let mut metrics = Metrics::new();
    metrics.insert("teleport.time_ms".into(), result.time_ms);
    metrics.insert("teleport.near_time_ms".into(), result.near_time_ms);
    metrics.insert("teleport.frames".into(), result.frames as f64);
    metrics.insert(
        "teleport.coverage".into(),
        coverage::<SineWorld>(&mut app, &wanted_offsets()),
    );
    frame_metrics("teleport", &result.frame_times, &mut metrics);
    metrics
}

fn scenario_idle() -> Metrics {
    let mut app = build_app(SineWorld::default());
    stream_in::<SineWorld>(&mut app);
    let times = timed_frames(&mut app, IDLE_FRAMES, |_| {});
    let mut metrics = Metrics::new();
    frame_metrics("idle", &times, &mut metrics);
    metrics
}

fn scenario_fly_through<C: VoxelWorldConfig<MaterialIndex = u8> + Default>(
    prefix: &str,
) -> Metrics {
    let mut app = build_app(C::default());
    stream_in::<C>(&mut app);
    app.insert_resource(Flight(Some(Instant::now())));
    let offsets = wanted_offsets();
    let mut coverage_sum = 0.0;
    let times = timed_frames(&mut app, FLY_FRAMES, |app| {
        coverage_sum += coverage::<C>(app, &offsets);
    });
    let mut metrics = Metrics::new();
    frame_metrics(prefix, &times, &mut metrics);
    metrics.insert(
        format!("{prefix}.coverage"),
        coverage_sum / FLY_FRAMES as f64,
    );
    metrics
}

fn edits_published(app: &mut App, expected: WorldVoxel<u8>) -> bool {
    let mut state = SystemState::<VoxelWorld<SineWorld>>::new(app.world_mut());
    let voxel_world = state.get_mut(app.world_mut()).unwrap();
    storm_positions().all(|pos| {
        let (chunk_pos, _) = get_chunk_voxel_position(pos);
        voxel_world
            .get_chunk_data(ChunkKey::new(chunk_pos, 0))
            .and_then(|data| data.get_voxel_at_world_position(pos))
            == Some(expected)
    })
}

fn scenario_edit_storm() -> Metrics {
    let mut app = build_app(SineWorld::default());
    stream_in::<SineWorld>(&mut app);
    app.world_mut().resource_mut::<EditStorm>().active = true;
    let times = timed_frames(&mut app, STORM_FRAMES as usize, |_| {});
    let mut storm = app.world_mut().resource_mut::<EditStorm>();
    storm.active = false;
    let expected = storm_value(storm.tick);

    // Gate on wall-clock time: frames here are far shorter than in a rendering
    // app, so a frame count mostly reflects how fast the frame loop spins.
    let start = Instant::now();
    let mut frames_to_publish = MAX_FRAMES;
    for frame in 1..=MAX_FRAMES {
        app.update();
        if edits_published(&mut app, expected) {
            frames_to_publish = frame;
            break;
        }
    }
    let mut metrics = Metrics::new();
    frame_metrics("edit_storm", &times, &mut metrics);
    metrics.insert(
        "edit_storm.time_to_publish_ms".into(),
        start.elapsed().as_secs_f64() * 1000.0,
    );
    metrics.insert(
        "edit_storm.frames_to_publish".into(),
        frames_to_publish as f64,
    );
    metrics
}

/// Combines repeated samples of the same metrics into one, key by key.
fn combine(samples: &[Metrics], pick: impl Fn(&str, &[f64]) -> f64) -> Metrics {
    samples[0]
        .keys()
        .map(|key| {
            let values: Vec<f64> = samples.iter().map(|s| s[key]).collect();
            (key.clone(), pick(key, &values))
        })
        .collect()
}

/// Repeated runs within one process. Interference only ever slows a run down, so
/// whole-run totals take their best run; everything else, already summarised over
/// many frames, takes the median.
fn run_repeatedly(runs: usize, scenario: impl Fn() -> Metrics) -> Metrics {
    let samples: Vec<Metrics> = (0..runs).map(|_| scenario()).collect();
    combine(&samples, |key, values| {
        match key.rsplit('.').next().unwrap() {
            "time_ms" | "near_time_ms" | "settled_ms" | "relit_ms" => {
                percentile(values, 0.0)
            }
            "chunks_per_s" => percentile(values, 1.0),
            _ => percentile(values, 0.5),
        }
    })
}

fn load_average() -> Option<f64> {
    #[cfg(unix)]
    {
        let mut load = [0.0f64; 1];
        // SAFETY: writes at most one value into `load`.
        if unsafe { libc::getloadavg(load.as_mut_ptr(), 1) } == 1 {
            return Some(load[0]);
        }
    }
    None
}

fn warn_if_noisy() {
    if let Some(load) = load_average().filter(|load| *load > NOISY_LOAD) {
        eprintln!(
            "warning: load average is {load:.1} (> {NOISY_LOAD}); results are likely noisy"
        );
    }
}

/// Runs every scenario, or only those named in `only` (comma-separated). When
/// profiling, each runs once, without a warm-up.
fn run_all(only: Option<&str>) -> Metrics {
    let profiling = PROFILING.load(Ordering::Relaxed);
    if !profiling {
        warn_if_noisy();
        // Unrecorded: the first scenario after a build or idle period runs measurably off.
        eprintln!("warm-up...");
        scenario_stream_in::<SineWorld>("warm_up");
    }

    let mut metrics = Metrics::new();
    let named = |name: &str| only.is_some_and(|only| only.split(',').any(|s| s == name));
    let mut step = |name: &str, runs: usize, scenario: &dyn Fn() -> Metrics| {
        // The worlds for comparing the models of sky fill run only when
        // named, alone or all together as `fill_worlds`.
        let wanted = if name.starts_with("fill_") {
            named(name) || named("fill_worlds")
        } else {
            only.is_none() || named(name)
        };
        if !wanted {
            return;
        }
        let runs = if profiling { 1 } else { runs };
        eprintln!("{name} ({runs} runs)...");
        metrics.extend(run_repeatedly(runs, scenario));
    };
    step("stream_in", MANY_RUNS, &|| {
        scenario_stream_in::<SineWorld>("stream_in")
    });
    step("stream_in_flat", MANY_RUNS, &|| {
        scenario_stream_in::<FlatWorld>("stream_in_flat")
    });
    step("idle", MANY_RUNS, &scenario_idle);
    step("fly_through", MANY_RUNS, &|| {
        scenario_fly_through::<SineWorld>("fly_through")
    });
    step("teleport", MANY_RUNS, &scenario_teleport);
    step("edit_storm", RUNS, &scenario_edit_storm);
    step("in_view_fly_through", RUNS, &|| {
        scenario_fly_through::<InViewWorld>("in_view_fly_through")
    });
    #[cfg(feature = "light_propagation")]
    step("stream_in_lamps", RUNS, &scenario_stream_in_lamps);
    #[cfg(feature = "light_propagation")]
    step("stream_in_lamps_unlit", RUNS, &|| {
        scenario_stream_in::<LampWorld<false>>("stream_in_lamps_unlit")
    });
    #[cfg(feature = "light_propagation")]
    step("stream_in_lit", RUNS, &|| {
        scenario_stream_in_lit::<LitBenchWorld<SINE>>("stream_in_lit")
    });
    #[cfg(feature = "light_propagation")]
    step("stream_in_flat_lit", RUNS, &|| {
        scenario_stream_in_lit::<LitBenchWorld<FLAT>>("stream_in_flat_lit")
    });
    #[cfg(feature = "light_propagation")]
    step("stream_in_islands", RUNS, &|| {
        scenario_stream_in_lit::<IslandWorld>("stream_in_islands")
    });
    #[cfg(feature = "light_propagation")]
    step("stream_in_islands_hinted_above", RUNS, &|| {
        scenario_stream_in_lit::<IslandWorld<ABOVE_HIGHEST_SOLID>>(
            "stream_in_islands_hinted_above",
        )
    });
    #[cfg(feature = "light_propagation")]
    step("stream_in_islands_hinted_air", RUNS, &|| {
        scenario_stream_in_lit::<IslandWorld<EVERY_ALL_AIR_POSITION>>(
            "stream_in_islands_hinted_air",
        )
    });
    #[cfg(feature = "light_propagation")]
    {
        let fill_worlds: [(&str, Scenario); 12] = [
            (
                "fill_islands_paths",
                scenario_stream_in_lit::<FillWorld<0, PATHS>>,
            ),
            (
                "fill_islands_openness",
                scenario_stream_in_lit::<FillWorld<0, OPENNESS>>,
            ),
            (
                "fill_islands_ceiling_paths",
                scenario_stream_in_lit::<FillWorld<0, PATHS, true>>,
            ),
            (
                "fill_islands_ceiling_openness",
                scenario_stream_in_lit::<FillWorld<0, OPENNESS, true>>,
            ),
            (
                "fill_arches_paths",
                scenario_stream_in_lit::<FillWorld<1, PATHS>>,
            ),
            (
                "fill_arches_openness",
                scenario_stream_in_lit::<FillWorld<1, OPENNESS>>,
            ),
            (
                "fill_caves_paths",
                scenario_stream_in_lit::<FillWorld<2, PATHS>>,
            ),
            (
                "fill_caves_openness",
                scenario_stream_in_lit::<FillWorld<2, OPENNESS>>,
            ),
            (
                "fill_pebbles_paths",
                scenario_stream_in_lit::<FillWorld<3, PATHS>>,
            ),
            (
                "fill_pebbles_openness",
                scenario_stream_in_lit::<FillWorld<3, OPENNESS>>,
            ),
            (
                "fill_interiors_paths",
                scenario_stream_in_lit::<FillWorld<4, PATHS>>,
            ),
            (
                "fill_interiors_openness",
                scenario_stream_in_lit::<FillWorld<4, OPENNESS>>,
            ),
        ];
        for (name, scenario) in fill_worlds {
            step(name, FILL_RUNS, &|| scenario(name));
        }
    }
    #[cfg(feature = "light_propagation")]
    step("stream_in_lod", RUNS, &|| {
        scenario_stream_in_lod::<LodWorld<false>>("stream_in_lod")
    });
    #[cfg(feature = "light_propagation")]
    step("stream_in_lod_hinted", RUNS, &|| {
        scenario_stream_in_lod::<LodWorld<true>>("stream_in_lod_hinted")
    });
    #[cfg(feature = "light_propagation")]
    step("caves", CAVE_RUNS, &scenario_caves);
    #[cfg(feature = "light_propagation")]
    if only.is_some_and(|only| only.split(',').any(|s| s == "lod_scale")) {
        step("lod_scale", 1, &scenario_lod_scale);
    }
    #[cfg(feature = "light_propagation")]
    step("edits_lit", RUNS, &scenario_edits_lit);
    #[cfg(feature = "light_propagation")]
    step("settings_lit", RUNS, &scenario_settings_lit);
    metrics
}

// --- Gating ----------------------------------------------------------------

enum Gate {
    /// Lower is better: a regression when the relative increase exceeds the limit.
    Lower(f64),
    /// Like `Lower`, for small counts: a regression also needs an increase of
    /// more than `min`.
    LowerCount {
        limit: f64,
        min: f64,
    },
    /// Higher is better: a regression when the relative decrease exceeds the limit.
    Higher(f64),
    ReportOnly,
}

fn gate(key: &str) -> Gate {
    if key.starts_with("in_view_fly_through.")
        || key.starts_with("stream_in_lamps")
        || key.starts_with("stream_in_lit")
        || key.starts_with("stream_in_flat_lit")
        || key.starts_with("stream_in_islands")
        || key.starts_with("edits_lit")
        || key.starts_with("settings_lit")
        || key.starts_with("caves")
        || key.starts_with("lod_scale")
    {
        return Gate::ReportOnly;
    }
    match key.rsplit('.').next().unwrap() {
        "p50_ms" | "time_ms" => Gate::Lower(0.05),
        "p99_ms" | "time_to_publish_ms" | "near_time_ms" => Gate::Lower(0.10),
        // Stutter metrics rest on a handful of frames; with identical code they
        // still move by up to about 10% (worst 1%) and 5 frames (counts).
        "worst_1pct_ms" => Gate::Lower(0.15),
        suffix if suffix.starts_with("over_") => Gate::LowerCount {
            limit: 0.20,
            min: 5.0,
        },
        "chunks_per_s" | "coverage" => Gate::Higher(0.05),
        _ => Gate::ReportOnly,
    }
}

fn compare(before: &Metrics, after: &Metrics) -> bool {
    let mut ok = true;
    println!(
        "{:<38} {:>12} {:>12} {:>9}  gate",
        "metric", "before", "after", "change"
    );
    for (key, b) in before {
        let Some(a) = after.get(key) else { continue };
        let change = match (*b, *a) {
            (0.0, 0.0) => 0.0,
            (0.0, _) => f64::INFINITY,
            _ => (a - b) / b,
        };
        let verdict = match gate(key) {
            Gate::Lower(limit) if change > limit => "REGRESSED",
            Gate::LowerCount { limit, min } if change > limit && a - b > min => {
                "REGRESSED"
            }
            Gate::Higher(limit) if -change > limit => "REGRESSED",
            Gate::ReportOnly => "report",
            _ => "ok",
        };
        ok &= verdict != "REGRESSED";
        println!(
            "{key:<38} {b:>12.3} {a:>12.3} {:>+8.1}%  {verdict}",
            change * 100.0
        );
    }
    for (unlit, lit, gated) in ON_GATE {
        let (Some(b), Some(a)) = (before.get(unlit), after.get(lit)) else {
            continue;
        };
        let change = (a - b) / b;
        let verdict = if !gated {
            "report"
        } else if change > ON_GATE_LIMIT || a.is_nan() {
            "REGRESSED"
        } else {
            "ok"
        };
        ok &= verdict != "REGRESSED";
        println!(
            "{:<38} {b:>12.3} {a:>12.3} {:>+8.1}%  on gate {verdict}",
            format!("{lit} vs {unlit}"),
            change * 100.0
        );
    }
    ok
}

/// The on gate: a lit stream against the same stream unlit in the before
/// build, `light-baseline` for the gate, as (unlit, lit, gated). The gate
/// binds the time until relights alone have settled. The time until
/// `light_is_settled`, which adds the sky probes and fill probes running on
/// after the stream, is only reported.
const ON_GATE: [(&str, &str, bool); 4] = [
    ("stream_in.time_ms", "stream_in_lit.relit_ms", true),
    (
        "stream_in_flat.time_ms",
        "stream_in_flat_lit.relit_ms",
        true,
    ),
    ("stream_in.time_ms", "stream_in_lit.settled_ms", false),
    (
        "stream_in_flat.time_ms",
        "stream_in_flat_lit.settled_ms",
        false,
    ),
];
/// How much longer the on gate lets a lit stream take.
const ON_GATE_LIMIT: f64 = 0.25;

// --- Storage ---------------------------------------------------------------

fn bench_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/bench/chunk_lifecycle")
}

fn results_path(label: &str) -> PathBuf {
    bench_dir().join(format!("{label}.json"))
}

fn binary_path(name: &str) -> PathBuf {
    bench_dir().join(format!("{name}-bin"))
}

fn save(label: &str, metrics: &Metrics) {
    let path = results_path(label);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let body: Vec<String> = metrics
        .iter()
        .map(|(key, value)| format!("  \"{key}\": {value}"))
        .collect();
    std::fs::write(&path, format!("{{\n{}\n}}\n", body.join(",\n"))).unwrap();
    eprintln!("wrote {}", path.display());
}

/// Reads the flat `{"key": number}` files written by `save`.
fn load(label: &str) -> Metrics {
    let path = results_path(label);
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("can't read {}: {e}", path.display()));
    text.trim()
        .trim_start_matches('{')
        .trim_end_matches('}')
        .split(',')
        .filter(|entry| !entry.trim().is_empty())
        .map(|entry| {
            let (key, value) = entry.split_once(':').unwrap();
            (
                key.trim().trim_matches('"').to_string(),
                value.trim().parse().unwrap(),
            )
        })
        .collect()
}

/// Loads a comma-separated list of labels and takes the median of each metric. Now
/// and then a whole process runs well off the rest, so separate processes take the
/// median, not the best, which would let one fast outlier decide.
fn load_all(labels: &str) -> Metrics {
    let samples: Vec<Metrics> = labels.split(',').map(load).collect();
    combine(&samples, |_, values| percentile(values, 0.5))
}

fn save_binary(name: &str) {
    let path = binary_path(name);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::copy(std::env::current_exe().unwrap(), &path).unwrap();
    eprintln!("wrote {}", path.display());
}

fn run_binary(binary: &PathBuf, label: &str) {
    eprintln!("== {} run {label}", binary.display());
    let status = Command::new(binary).args(["run", label]).status().unwrap();
    assert!(status.success(), "{} failed: {status}", binary.display());
}

/// Runs the saved `baseline` binary and this build alternately, then compares them.
/// The order flips every round (ABBAAB…) so drift over the session hits both sides
/// alike.
fn ab(baseline: &str) -> bool {
    let baseline = binary_path(baseline);
    let this = std::env::current_exe().unwrap();
    let mut before = Vec::new();
    let mut after = Vec::new();
    for round in 1..=AB_ROUNDS {
        let before_label = format!("ab-before-{round}");
        let after_label = format!("ab-after-{round}");
        if round % 2 == 1 {
            run_binary(&baseline, &before_label);
            run_binary(&this, &after_label);
        } else {
            run_binary(&this, &after_label);
            run_binary(&baseline, &before_label);
        }
        before.push(before_label);
        after.push(after_label);
    }
    compare(&load_all(&before.join(",")), &load_all(&after.join(",")))
}

fn main() {
    prefer_performance_cores();
    // Cargo passes `--bench` to custom harnesses.
    let args: Vec<String> = std::env::args()
        .skip(1)
        .filter(|a| a != "--bench")
        .collect();
    match args
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>()
        .as_slice()
    {
        ["run", label, rest @ ..] if rest.len() <= 1 => {
            let metrics = run_all(rest.first().copied());
            for (key, value) in &metrics {
                println!("{key:<38} {value:>12.3}");
            }
            save(label, &metrics);
        }
        ["compare", before, after] => {
            if !compare(&load_all(before), &load_all(after)) {
                std::process::exit(1);
            }
        }
        ["profile", scenario] => {
            if !cfg!(feature = "trace_tracy") {
                eprintln!("profile needs a Tracy build: add `--features trace_tracy`");
                std::process::exit(2);
            }
            PROFILING.store(true, Ordering::Relaxed);
            for (key, value) in &run_all(Some(scenario)) {
                println!("{key:<38} {value:>12.3}");
            }
        }
        ["save-binary", name] => save_binary(name),
        ["ab", baseline] => {
            if !ab(baseline) {
                std::process::exit(1);
            }
        }
        [] => {}
        _ => {
            eprintln!(
                "usage: chunk_lifecycle run <label> [<scenario>,…] | compare <before> <after> \
                 | save-binary <name> | ab <baseline> | profile <scenario>"
            );
            std::process::exit(2);
        }
    }
}
