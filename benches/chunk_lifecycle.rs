//! Chunk lifecycle benchmarks.
//!
//! Runs scripted scenarios against a headless App (no window, no GPU) that still
//! exercises the full chunk lifecycle: spawn, generate, mesh, publish, retire, despawn.
//!
//! Record a run of every scenario, or only of those listed:
//!     cargo bench --bench chunk_lifecycle -- run <label> [<scenario>,…]
//! Compare recorded runs (exits non-zero if a gated metric regressed); each side
//! may be a comma-separated list of labels, combined by their median:
//!     cargo bench --bench chunk_lifecycle -- compare <before>[,<before>…] <after>[,<after>…]
//! A metric found on one side only prints as `new` (after only) or `gone`
//! (before only) and gates nothing, so a scenario one build lacks still shows.
//! Ratio lines, listed in `RATIO_LINES`, print the ratio of two metrics within
//! the before build and within the after build, and the change between them;
//! they gate nothing either.
//!
//! Background load on a desktop drifts by more than the gates over minutes, so
//! comparisons alternate the two builds instead of reusing an old recording:
//!     cargo bench --bench chunk_lifecycle -- save-binary baseline   # on the baseline commit
//!     cargo bench --bench chunk_lifecycle -- ab baseline [<scenario>,…]   # on the candidate
//! `ab` runs the saved baseline binary and this build alternately, `AB_ROUNDS`
//! times each, then compares them. With a list, both builds run only the
//! listed scenarios, and one the baseline doesn't know prints as `new`. A
//! listed run is a quick check; the run without a list, over every scenario,
//! is the binding one.
//!
//! Scenarios behind a cargo feature run only in a build with it, so pick the
//! features of an A/B by what the change touches. A change touching sub-grids
//! or fragment detection runs its A/B with
//! `--features sub_grids,dislodged_fragments`, adding `light_propagation`
//! when it touches light.
//!
//! Results and saved binaries are written to `target/bench/chunk_lifecycle/`
//! of the checkout the binary was built in, which it finds through
//! `CARGO_MANIFEST_DIR`. A saved baseline binary keeps writing there when `ab`
//! runs it from another checkout or worktree, so make that checkout's
//! `target/bench/chunk_lifecycle` a link to the baseline's folder before
//! running `ab` from it.
//!
//! `[profile.bench]` in Cargo.toml builds both sides with one codegen unit. The
//! voxel world's generic code is compiled inside this crate, and how the
//! default sixteen units group it has moved flat terrain's generation
//! throughput by more than the gates between builds that differ only in code
//! the bench never runs; one unit makes such builds identical. A baseline from
//! a commit older than that setting needs it from the environment:
//!     CARGO_PROFILE_BENCH_CODEGEN_UNITS=1 cargo bench --bench chunk_lifecycle -- save-binary baseline
//!
//! To see what a scenario's slow frames are doing, run it once under Tracy. Start
//! the Tracy profiler (or `tracy-capture -o <file>.tracy`) first, then:
//!     cargo bench --bench chunk_lifecycle --features trace_tracy -- profile <scenario>
//! The Tracy version must match the protocol of the `tracy-client` crate Bevy uses;
//! see the version table in the `tracy-client` README.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use bevy::app::{TaskPoolOptions, TaskPoolPlugin, TaskPoolThreadAssignmentPolicy};
use bevy::camera::{CameraProjection, RenderTargetInfo};
use bevy::ecs::system::SystemState;
use bevy::prelude::*;
use bevy_voxel_world::prelude::*;

#[cfg(any(feature = "light_propagation", feature = "dislodged_fragments"))]
#[path = "common/light_scene.rs"]
mod light_scene;

#[cfg(any(feature = "light_propagation", feature = "dislodged_fragments"))]
use light_scene::{SceneTable, SCENE_TOP};

#[cfg(feature = "dislodged_fragments")]
#[path = "common/fragment_sites.rs"]
mod fragment_sites;

#[cfg(feature = "light_propagation")]
#[path = "common/fill_worlds.rs"]
mod fill_worlds;

#[cfg(feature = "light_propagation")]
use fill_worlds::FILL_SCENES;

#[cfg(feature = "sub_grids")]
#[path = "common/vehicle_hull.rs"]
mod vehicle_hull;

#[cfg(feature = "sub_grids")]
#[path = "common/lift_pieces.rs"]
mod lift_pieces;

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
///
/// With `LAMPS`, the material `LAMP` glows at level 15. The terrain holds
/// none of it, so only voxels a scenario sets with it are lamps.
#[cfg(feature = "light_propagation")]
#[derive(Resource, Clone, Default)]
struct LitBenchWorld<const TERRAIN: u8, const LAMPS: bool = false>;

#[cfg(feature = "light_propagation")]
impl<const TERRAIN: u8, const LAMPS: bool> VoxelWorldConfig
    for LitBenchWorld<TERRAIN, LAMPS>
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
        BenchWorld::<TERRAIN, false>.voxel_lookup_delegate()
    }

    fn light_propagation(&self) -> Option<LightPropagation> {
        Some(LightPropagation::default())
    }

    fn emissive_materials(&self) -> Vec<(Self::MaterialIndex, Emission)> {
        if LAMPS {
            vec![(LAMP, Emission::level(15.0))]
        } else {
            Vec::new()
        }
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

/// The light reference scene around `FRAGMENT_CAMERA`, unlit, with every
/// solid voxel of one material. With `detect`, it detects dislodged
/// fragments with the default settings and only reports them, or with
/// `auto_lift` lifts each of them too. Its voxels are read from
/// [`fragment_table`].
#[cfg(feature = "dislodged_fragments")]
#[derive(Resource, Clone, Default)]
struct FragmentWorld {
    detect: bool,
    #[cfg(feature = "sub_grids")]
    auto_lift: bool,
}

/// Where `fragments` starts its camera: beside the reference scene's slab,
/// which sticks out of its mesa, looking up at the slab's edge.
#[cfg(feature = "dislodged_fragments")]
const FRAGMENT_CAMERA: CameraStart = CameraStart {
    at: Vec3::new(0.5, 50.0, 420.0),
    target: Vec3::new(0.5, 78.0, 370.0),
};

/// The reference scene sampled around `FRAGMENT_CAMERA`, as [`cave_table`]
/// samples it around `CAVE_CAMERA`. Sampled once per process, before the
/// first `fragments` run.
#[cfg(feature = "dislodged_fragments")]
fn fragment_table() -> Arc<SceneTable> {
    static TABLE: std::sync::OnceLock<Arc<SceneTable>> = std::sync::OnceLock::new();
    TABLE
        .get_or_init(|| {
            let started = Instant::now();
            let camera = (FRAGMENT_CAMERA.at / 32.0).floor().as_ivec3();
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

/// Where `fragments` digs and cuts, found in [`fragment_table`] once per
/// process.
#[cfg(feature = "dislodged_fragments")]
fn fragment_sites() -> &'static fragment_sites::Sites {
    static SITES: std::sync::OnceLock<fragment_sites::Sites> = std::sync::OnceLock::new();
    SITES.get_or_init(|| {
        let started = Instant::now();
        let sites = fragment_sites::find(&fragment_table(), FRAGMENT_CAMERA.at);
        let pieces: Vec<u32> = sites.cuts.iter().map(|cut| cut.piece).collect();
        eprintln!(
            "found {} digs, cuts around pieces of {pieces:?} voxels and a limit cut \
             at {} in {:.1?}",
            sites.digs.len(),
            sites.limit_cut.min,
            started.elapsed()
        );
        sites
    })
}

#[cfg(feature = "dislodged_fragments")]
impl VoxelWorldConfig for FragmentWorld {
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
        let table = fragment_table();
        Box::new(move |_, _, _| {
            let table = table.clone();
            Box::new(move |pos: IVec3, _| {
                if table.open(pos) {
                    WorldVoxel::Air
                } else {
                    WorldVoxel::Solid(1)
                }
            })
        })
    }

    fn detect_dislodged_fragments(&self) -> Option<FragmentDetection<u8>> {
        self.detect.then(|| FragmentDetection {
            #[cfg(feature = "sub_grids")]
            auto_lift: self.auto_lift,
            ..default()
        })
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
#[cfg(any(feature = "light_propagation", feature = "sub_grids"))]
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

/// Runs of `fragments`, each of which streams the reference scene twice.
#[cfg(feature = "dislodged_fragments")]
const FRAGMENT_RUNS: usize = 3;

/// The most frames `fragments` waits for a cut's check to land.
#[cfg(feature = "dislodged_fragments")]
const CUT_FRAME_CAP: u32 = 600;

/// Frames in a row with no change to a world's fragment counters, once a
/// check has finished, after which `fragments` takes its checks to have
/// landed.
#[cfg(feature = "dislodged_fragments")]
const QUIET_CHECK_FRAMES: u32 = 10;

/// The threshold of `fragments`: the most voxels a check of a dig that cuts
/// nothing may visit on average, about 20% above its first run. The digs
/// and the order a check floods in are fixed, so machine load doesn't move
/// it.
#[cfg(feature = "dislodged_fragments")]
const FRAGMENT_DIG_VISITED_PER_CHECK: f64 = 22.0;

/// Every dislodged fragment received so far, as its box's lowest corner,
/// its box's highest corner, exclusive, and its voxel count.
#[cfg(feature = "dislodged_fragments")]
#[derive(Resource, Default)]
struct Dislodged(Vec<(IVec3, IVec3, u32)>);

#[cfg(feature = "dislodged_fragments")]
fn record_dislodged<C: VoxelWorldConfig>(
    mut dislodged: ResMut<Dislodged>,
    mut messages: MessageReader<FragmentDislodged<C>>,
) {
    dislodged.0.extend(
        messages
            .read()
            .map(|fragment| (fragment.min, fragment.max, fragment.voxel_count)),
    );
}

/// An app of `world`, recording the fragments it reports, streamed in
/// around `FRAGMENT_CAMERA`.
#[cfg(feature = "dislodged_fragments")]
fn fragments_app(world: FragmentWorld) -> App {
    let mut app = build_app(world);
    app.insert_resource(FRAGMENT_CAMERA)
        .init_resource::<Dislodged>()
        .add_systems(Update, record_dislodged::<FragmentWorld>);
    #[cfg(feature = "sub_grids")]
    app.init_resource::<FragmentLifts>()
        .add_systems(Update, record_fragment_lifts::<FragmentWorld>);
    stream_in::<FragmentWorld>(&mut app);
    app
}

/// The fragment counters of `app`'s world, all zero in a world that
/// doesn't detect dislodged fragments.
#[cfg(feature = "dislodged_fragments")]
fn fragment_counters(app: &App) -> bevy_voxel_world::grid_bench::FragmentStats {
    bevy_voxel_world::grid_bench::fragment_stats::<FragmentWorld>(app.world())
        .unwrap_or_default()
}

/// What the counters counted from `before` to `after`.
#[cfg(feature = "dislodged_fragments")]
fn counted_since(
    after: bevy_voxel_world::grid_bench::FragmentStats,
    before: bevy_voxel_world::grid_bench::FragmentStats,
) -> bevy_voxel_world::grid_bench::FragmentStats {
    bevy_voxel_world::grid_bench::FragmentStats {
        checks: after.checks - before.checks,
        voxels_visited: after.voxels_visited - before.voxels_visited,
        reruns: after.reruns - before.reruns,
        task_micros: after.task_micros - before.task_micros,
    }
}

/// Sets the voxels at `positions` to air, to commit in the next frame.
#[cfg(feature = "dislodged_fragments")]
fn dig_voxels(app: &mut App, positions: &[IVec3]) {
    let mut state = SystemState::<VoxelWorld<FragmentWorld>>::new(app.world_mut());
    let mut voxel_world = state.get_mut(app.world_mut()).unwrap();
    for position in positions {
        voxel_world.set_voxel(*position, WorldVoxel::Air);
    }
}

/// Runs frames until the fragment counters have not changed for
/// `QUIET_CHECK_FRAMES` in a row, or for at most `CUT_FRAME_CAP` frames.
#[cfg(feature = "dislodged_fragments")]
fn settle_fragment_checks(app: &mut App) {
    let mut last = fragment_counters(app);
    let mut quiet = 0;
    for _ in 0..CUT_FRAME_CAP {
        app.update();
        let now = fragment_counters(app);
        quiet = if now == last { quiet + 1 } else { 0 };
        last = now;
        if quiet >= QUIET_CHECK_FRAMES {
            return;
        }
    }
}

/// Part 1 of `fragments`: digs at `digs`, one a frame, then waits for the
/// checks to land. Reports, as `name`, the frames' metrics, the checks run,
/// the voxels visited and task µs per check, the reruns, and how many digs
/// wrote a `FragmentDislodged` within a voxel of the fragment's box.
#[cfg(feature = "dislodged_fragments")]
fn fragment_digs(app: &mut App, name: &str, digs: &[IVec3], metrics: &mut Metrics) {
    let before = fragment_counters(app);
    let seen = app.world().resource::<Dislodged>().0.len();
    dig_voxels(app, &digs[..1]);
    let mut next = 1;
    let times = timed_frames(app, digs.len(), |app| {
        if let Some(position) = digs.get(next) {
            dig_voxels(app, &[*position]);
            next += 1;
        }
    });
    settle_fragment_checks(app);
    let counted = counted_since(fragment_counters(app), before);
    let written = &app.world().resource::<Dislodged>().0[seen..];
    let dislodging = digs
        .iter()
        .filter(|dig| {
            written.iter().any(|(min, max, _)| {
                dig.cmpge(*min - 1).all() && dig.cmplt(*max + 1).all()
            })
        })
        .count();

    frame_metrics(name, &times, metrics);
    let checks = counted.checks.max(1) as f64;
    let mut put = |metric: &str, value: f64| {
        metrics.insert(format!("{name}.{metric}"), value);
    };
    put("checks", counted.checks as f64);
    put("reruns", counted.reruns as f64);
    put(
        "voxels_visited_per_check",
        counted.voxels_visited as f64 / checks,
    );
    put("task_us_per_check", counted.task_micros as f64 / checks);
    put("dislodging_digs", dislodging as f64);
}

/// What one cut of `fragments` did.
#[cfg(feature = "dislodged_fragments")]
struct CutReport {
    /// Frames from the cut's frame to the one its check landed in.
    frames: u32,
    counted: bevy_voxel_world::grid_bench::FragmentStats,
    /// The voxel count of each fragment it reported.
    dislodged: Vec<u32>,
    /// Whether its check hadn't landed after `CUT_FRAME_CAP` frames.
    capped: bool,
}

/// Makes `cut` and runs frames until its check lands.
///
/// A check that reports a fragment has landed in the frame the report
/// arrives. One that reports none is taken to have landed in the frame
/// after the counters last changed, once they show it finished and stay
/// unchanged for `QUIET_CHECK_FRAMES`.
#[cfg(feature = "dislodged_fragments")]
fn fragment_cut(app: &mut App, cut: &fragment_sites::Cut) -> CutReport {
    fragment_cut_by(app, |app| dig_voxels(app, &cut.dug))
}

/// Cuts with `dig`, which sets voxels to commit in the next frame, and runs
/// frames until the cut's check lands, as [`fragment_cut`] does.
#[cfg(feature = "dislodged_fragments")]
fn fragment_cut_by(app: &mut App, dig: impl FnOnce(&mut App)) -> CutReport {
    let before = fragment_counters(app);
    let seen = app.world().resource::<Dislodged>().0.len();
    dig(app);
    let mut last = before;
    let mut changed_at = 0;
    for frame in 1..=CUT_FRAME_CAP {
        app.update();
        let now = fragment_counters(app);
        let written = &app.world().resource::<Dislodged>().0[seen..];
        if !written.is_empty() {
            return CutReport {
                frames: frame,
                counted: counted_since(now, before),
                dislodged: written.iter().map(|(_, _, count)| *count).collect(),
                capped: false,
            };
        }
        if now != last {
            last = now;
            changed_at = frame;
        }
        let finished = now.voxels_visited > before.voxels_visited;
        if finished && frame - changed_at >= QUIET_CHECK_FRAMES {
            return CutReport {
                frames: changed_at + 1,
                counted: counted_since(now, before),
                dislodged: Vec::new(),
                capped: false,
            };
        }
    }
    CutReport {
        frames: CUT_FRAME_CAP,
        counted: counted_since(fragment_counters(app), before),
        dislodged: Vec::new(),
        capped: true,
    }
}

/// Part 2 of `fragments`: makes each cut of `sites` in turn, waiting for its
/// check to land, then the limit cut. Reports each cut as `cut_NN` and their
/// sum as `cuts`: frames to land, the check's task µs, voxels visited and
/// reruns, and the voxels dislodged. `cuts_without_report` counts the cuts
/// that wrote no `FragmentDislodged`, and `limit_cut` reports the same as a
/// cut. Returns how many cuts were capped.
#[cfg(feature = "dislodged_fragments")]
fn fragment_cuts(
    app: &mut App,
    prefix: &str,
    sites: &fragment_sites::Sites,
    metrics: &mut Metrics,
) -> usize {
    let mut put = |name: &str, report: &CutReport| {
        let mut put = |metric: &str, value: f64| {
            metrics.insert(format!("{prefix}.{name}.{metric}"), value);
        };
        put("frames", report.frames as f64);
        put("task_us", report.counted.task_micros as f64);
        put("voxels_visited", report.counted.voxels_visited as f64);
        put("reruns", report.counted.reruns as f64);
        put(
            "dislodged_voxels",
            report.dislodged.iter().sum::<u32>() as f64,
        );
    };
    let mut total = CutReport {
        frames: 0,
        counted: default(),
        dislodged: Vec::new(),
        capped: false,
    };
    let mut without_report = 0;
    let mut capped = 0;
    for (i, cut) in sites.cuts.iter().enumerate() {
        let report = fragment_cut(app, cut);
        if report.dislodged != [cut.piece] {
            eprintln!(
                "{prefix}: the cut at {} dislodged {:?} voxels, not its piece of {}",
                cut.min, report.dislodged, cut.piece
            );
        }
        put(&format!("cut_{:02}", i + 1), &report);
        without_report += report.dislodged.is_empty() as usize;
        capped += report.capped as usize;
        total.frames += report.frames;
        total.counted.task_micros += report.counted.task_micros;
        total.counted.voxels_visited += report.counted.voxels_visited;
        total.counted.reruns += report.counted.reruns;
        total.dislodged.extend(report.dislodged);
    }
    put("cuts", &total);
    let limit = fragment_cut(app, &sites.limit_cut);
    if !limit.dislodged.is_empty() {
        eprintln!(
            "{prefix}: the limit cut dislodged {:?} voxels",
            limit.dislodged
        );
    }
    put("limit_cut", &limit);
    capped += limit.capped as usize;
    metrics.insert(
        format!("{prefix}.cuts_without_report"),
        without_report as f64,
    );
    capped
}

/// The sub-grid each `FragmentDislodged` received so far was lifted into,
/// in the order they arrived, and the sub-grid of each `Lifted`.
#[cfg(all(feature = "dislodged_fragments", feature = "sub_grids"))]
#[derive(Resource, Default)]
struct FragmentLifts {
    lifted: Vec<Option<Entity>>,
    landed: Vec<Entity>,
}

#[cfg(all(feature = "dislodged_fragments", feature = "sub_grids"))]
fn record_fragment_lifts<C: VoxelWorldConfig>(
    mut lifts: ResMut<FragmentLifts>,
    mut dislodged: MessageReader<FragmentDislodged<C>>,
    mut landed: MessageReader<Lifted<C>>,
) {
    let lifts = &mut *lifts;
    lifts
        .lifted
        .extend(dislodged.read().map(|fragment| fragment.lifted));
    lifts
        .landed
        .extend(landed.read().map(|lifted| lifted.sub_grid));
}

/// What one cut of `fragments` made with auto-lift on did.
#[cfg(all(feature = "dislodged_fragments", feature = "sub_grids"))]
struct LiftCutReport {
    /// Frames from the cut's frame to the one its check landed in.
    frames_to_report: u32,
    /// Frames from the cut's frame to the one the handover of the lifts of
    /// its fragments landed in.
    frames: u32,
    /// The voxel count of each fragment it reported.
    dislodged: Vec<u32>,
    /// Whether each fragment it reported was lifted and its `Lifted`
    /// arrived.
    landed: bool,
    /// Whether that hadn't happened after `CUT_FRAME_CAP` frames.
    capped: bool,
}

/// Makes `cut` in a world that lifts each fragment, and runs frames until
/// the fragments it reports have all been lifted and their `Lifted` has
/// arrived.
#[cfg(all(feature = "dislodged_fragments", feature = "sub_grids"))]
fn fragment_lift_cut(app: &mut App, cut: &fragment_sites::Cut) -> LiftCutReport {
    let seen = app.world().resource::<Dislodged>().0.len();
    let seen_lifts = app.world().resource::<FragmentLifts>().lifted.len();
    dig_voxels(app, &cut.dug);
    let mut report = LiftCutReport {
        frames_to_report: CUT_FRAME_CAP,
        frames: CUT_FRAME_CAP,
        dislodged: Vec::new(),
        landed: false,
        capped: true,
    };
    for frame in 1..=CUT_FRAME_CAP {
        app.update();
        let written = &app.world().resource::<Dislodged>().0[seen..];
        if written.is_empty() {
            continue;
        }
        if report.dislodged.is_empty() {
            report.frames_to_report = frame;
            report.dislodged = written.iter().map(|(_, _, count)| *count).collect();
        }
        let lifts = app.world().resource::<FragmentLifts>();
        let lifted = &lifts.lifted[seen_lifts..];
        let landed = lifted
            .iter()
            .all(|sub_grid| sub_grid.is_some_and(|s| lifts.landed.contains(&s)));
        if landed || lifted.contains(&None) {
            report.frames = frame;
            report.landed = landed;
            report.capped = false;
            break;
        }
    }
    report
}

/// Part 3 of `fragments`, with `sub_grids`: sets the voxels part 2's cuts
/// dug back to what they were, turns `auto_lift` on, and makes the same
/// cuts again, one at a time. Reports each as `lift_cut_NN`, and their sum
/// as `lift_cuts`: frames to the `FragmentDislodged`, frames to its lift's
/// handover landing and the voxels dislodged, with the handovers that
/// landed and their chunks per handover. `lifts_without_landing` counts the
/// cuts whose fragments weren't all lifted, or whose `Lifted` never came.
/// Turns `auto_lift` off again after. Returns how many cuts were capped.
#[cfg(all(feature = "dislodged_fragments", feature = "sub_grids"))]
fn fragment_lift_cuts(
    app: &mut App,
    prefix: &str,
    sites: &fragment_sites::Sites,
    metrics: &mut Metrics,
) -> usize {
    {
        let mut state = SystemState::<VoxelWorld<FragmentWorld>>::new(app.world_mut());
        let mut voxel_world = state.get_mut(app.world_mut()).unwrap();
        for position in sites.cuts.iter().flat_map(|cut| &cut.dug) {
            voxel_world.set_voxel(*position, WorldVoxel::Solid(1));
        }
    }
    settle_fragment_checks(app);
    app.world_mut().resource_mut::<FragmentWorld>().auto_lift = true;

    let handovers = |app: &App| {
        bevy_voxel_world::grid_bench::handover_stats::<FragmentWorld>(app.world())
            .unwrap_or_default()
    };
    let before = handovers(app);
    let mut put = |name: &str, metric: &str, value: f64| {
        metrics.insert(format!("{prefix}.{name}.{metric}"), value);
    };
    let (mut frames_to_report, mut frames, mut dislodged) = (0, 0, 0);
    let (mut without_landing, mut capped) = (0, 0);
    for (i, cut) in sites.cuts.iter().enumerate() {
        let report = fragment_lift_cut(app, cut);
        if report.dislodged != [cut.piece] {
            eprintln!(
                "{prefix}: the lifted cut at {} dislodged {:?} voxels, not its piece of {}",
                cut.min, report.dislodged, cut.piece
            );
        }
        let name = format!("lift_cut_{:02}", i + 1);
        let voxels = report.dislodged.iter().sum::<u32>();
        put(&name, "frames_to_report", report.frames_to_report as f64);
        put(&name, "frames", report.frames as f64);
        put(&name, "dislodged_voxels", voxels as f64);
        frames_to_report += report.frames_to_report;
        frames += report.frames;
        dislodged += voxels;
        without_landing += !report.landed as usize;
        capped += report.capped as usize;
    }
    let landed = handovers(app);
    let handovers = landed.handovers - before.handovers;
    let chunks = landed.chunks - before.chunks;
    put("lift_cuts", "frames_to_report", frames_to_report as f64);
    put("lift_cuts", "frames", frames as f64);
    put("lift_cuts", "dislodged_voxels", dislodged as f64);
    put("lift_cuts", "handovers", handovers as f64);
    put(
        "lift_cuts",
        "chunks_per_handover",
        chunks as f64 / handovers.max(1) as f64,
    );
    metrics.insert(
        format!("{prefix}.lifts_without_landing"),
        without_landing as f64,
    );

    app.world_mut().resource_mut::<FragmentWorld>().auto_lift = false;
    settle_fragment_checks(app);
    capped
}

/// Where part 4 of `fragments` spawns the vehicle hull: in the air above
/// the reference scene, near `FRAGMENT_CAMERA`.
#[cfg(all(feature = "dislodged_fragments", feature = "sub_grids"))]
const FRAGMENT_HULL_AT: Vec3 = Vec3::new(-48.0, 200.0, 200.0);

/// The plane of air part 4 of `fragments` cuts the hull with, at this z in
/// the hull's sub-grid: across the envelope, ahead of the gondola.
#[cfg(all(feature = "dislodged_fragments", feature = "sub_grids"))]
const HULL_CUT_Z: i32 = 64;

/// Part 4 of `fragments`, with `sub_grids`: builds the vehicle hull in a
/// sub-grid of the world, lets its chunks settle, and cuts it in two with a
/// plane of air across a whole cross-section of the box holding it, at
/// `HULL_CUT_Z`. A sub-grid is checked with no limit, so the check floods
/// until the smaller half runs out. Reports `hull_cut` as part 2 reports a
/// cut, with the hull's solid voxels. Returns whether the cut was capped.
#[cfg(all(feature = "dislodged_fragments", feature = "sub_grids"))]
fn fragment_hull_cut(app: &mut App, prefix: &str, metrics: &mut Metrics) -> usize {
    let (hull, built) = spawn_hull::<FragmentWorld>(
        app,
        Transform::from_translation(FRAGMENT_HULL_AT),
        None,
    );
    let ready = settle_hull::<FragmentWorld>(app, hull);
    if ready != built.chunks.len() {
        eprintln!(
            "{prefix}: {ready} hull chunks are Ready, but the hull should have {}",
            built.chunks.len()
        );
    }
    settle_fragment_checks(app);
    let (min, max) = (vehicle_hull::MIN, vehicle_hull::MAX);
    let mut piece = 0;
    for z in min.z..HULL_CUT_Z {
        for y in min.y..max.y {
            for x in min.x..max.x {
                piece +=
                    vehicle_hull::hull_voxel(IVec3::new(x, y, z), None).is_some() as u32;
            }
        }
    }

    let report = fragment_cut_by(app, |app| {
        let mut state = SystemState::<VoxelWorld<FragmentWorld>>::new(app.world_mut());
        let mut voxel_world = state.get_mut(app.world_mut()).unwrap();
        voxel_world
            .grid_mut(Grid::SubGrid(hull))
            .unwrap()
            .set_voxels_in(min.with_z(HULL_CUT_Z), max.with_z(HULL_CUT_Z + 1), |_| {
                Some(WorldVoxel::Air)
            });
    });
    if report.dislodged != [piece] {
        eprintln!(
            "{prefix}: the hull cut dislodged {:?} voxels, not its smaller half of {piece}",
            report.dislodged
        );
    }
    let mut put = |metric: &str, value: f64| {
        metrics.insert(format!("{prefix}.hull_cut.{metric}"), value);
    };
    put("solid_voxels", built.solid_voxels as f64);
    put("frames", report.frames as f64);
    put("task_us", report.counted.task_micros as f64);
    put("voxels_visited", report.counted.voxels_visited as f64);
    put("reruns", report.counted.reruns as f64);
    put(
        "dislodged_voxels",
        report.dislodged.iter().sum::<u32>() as f64,
    );
    report.capped as usize
}

/// What detecting dislodged fragments costs in the light reference scene,
/// unlit, in a world that detects them with the default settings and only
/// reports them. `fragment_sites` finds where it digs and cuts.
///
/// 1. 200 digs that cut nothing, into slopes and cave walls, one a frame.
///    `digs` reports their frames, the checks run, voxels visited and task
///    µs per check, reruns, and the digs that wrote a `FragmentDislodged`.
///    `digs_not_opted_in` reports the same digs in a fresh app of the same
///    world not opted in, and `RATIO_LINES` prints the frame time of one
///    against the other.
/// 2. A dozen cuts, each around a piece of a few hundred to a few thousand
///    voxels with air under it, as in a cave ceiling, an overhang or an
///    arch, then a cut around more voxels than a check floods by default;
///    see `fragment_cuts`.
/// 3. With `sub_grids`, the same dozen cuts again with auto-lift on, in the
///    same app, timed to their lifts' handovers landing; see
///    `fragment_lift_cuts`.
/// 4. With `sub_grids`, the `vehicle` hull built in a sub-grid and cut in
///    two, which a check floods with no limit; see `fragment_hull_cut`.
///
/// Threshold, checked here and reported as `gate_failures`: voxels visited
/// per check in `digs` at most `FRAGMENT_DIG_VISITED_PER_CHECK`. A cut whose
/// check, or whose lifts' handover, hasn't landed within `CUT_FRAME_CAP`
/// frames also counts. Timings have no threshold.
#[cfg(feature = "dislodged_fragments")]
fn scenario_fragments() -> Metrics {
    let prefix = "fragments";
    let sites = fragment_sites();
    let mut metrics = Metrics::new();

    let mut app = fragments_app(FragmentWorld {
        detect: true,
        #[cfg(feature = "sub_grids")]
        auto_lift: false,
    });
    fragment_digs(
        &mut app,
        &format!("{prefix}.digs"),
        &sites.digs,
        &mut metrics,
    );
    #[allow(unused_mut)]
    let mut capped = fragment_cuts(&mut app, prefix, sites, &mut metrics);
    #[cfg(feature = "sub_grids")]
    {
        capped += fragment_lift_cuts(&mut app, prefix, sites, &mut metrics);
        capped += fragment_hull_cut(&mut app, prefix, &mut metrics);
    }
    drop(app);

    let mut app = fragments_app(FragmentWorld::default());
    fragment_digs(
        &mut app,
        &format!("{prefix}.digs_not_opted_in"),
        &sites.digs,
        &mut metrics,
    );
    drop(app);

    let visited = metrics[&format!("{prefix}.digs.voxels_visited_per_check")];
    let mut failures = capped;
    if visited > FRAGMENT_DIG_VISITED_PER_CHECK {
        eprintln!(
            "{prefix}: voxels visited per check {visited} > {FRAGMENT_DIG_VISITED_PER_CHECK}"
        );
        failures += 1;
    }
    if capped > 0 {
        eprintln!("{prefix}: {capped} cuts didn't land within {CUT_FRAME_CAP} frames");
    }
    metrics.insert(format!("{prefix}.gate_failures"), failures as f64);
    metrics
}

/// Frames in each of `vehicle`'s phases: upright flight, tumbling, and
/// tumbling while edited. The flight is one lap of its circle.
#[cfg(feature = "sub_grids")]
const VEHICLE_PHASE_FRAMES: u32 = 600;

/// The centre of the circle the hull's centre flies around, high enough
/// above the sine terrain that the tumbling hull never reaches it.
#[cfg(feature = "sub_grids")]
const VEHICLE_ORBIT_CENTRE: Vec3 = Vec3::new(16.0, 220.0, 16.0);

#[cfg(feature = "sub_grids")]
const VEHICLE_ORBIT_RADIUS: f32 = 160.0;

/// How far the tumbling hull turns each frame about its x, y and z axes,
/// in radians.
#[cfg(feature = "sub_grids")]
const VEHICLE_TUMBLE_RATES: Vec3 = Vec3::new(0.011, 0.019, 0.027);

/// The first of the materials `vehicle` sets at the hull's edit sites, a
/// different one at each visit to a site. The hull uses none of them, so a
/// publish showing one can only come from its edit.
#[cfg(feature = "sub_grids")]
const VEHICLE_EDIT: u8 = 10;

/// The most frames `vehicle` runs after its edited phase, still tumbling,
/// for its last edits to publish.
#[cfg(feature = "sub_grids")]
const VEHICLE_DRAIN_FRAMES: u32 = 120;

/// The hull a scenario moves, and the frame of its motion.
#[cfg(feature = "sub_grids")]
#[derive(Resource, Default)]
struct VehicleMotion {
    hull: Option<Entity>,
    tumbling: bool,
    frame: u32,
}

/// The hull's transform at `frame` of its flight, or of its tumbling when
/// `tumbling`.
///
/// In flight the hull stays upright and circles `VEHICLE_ORBIT_CENTRE`,
/// facing along its path. Tumbling, it turns about its centre at the point
/// where the flight starts and ends.
#[cfg(feature = "sub_grids")]
fn vehicle_pose(tumbling: bool, frame: u32) -> Transform {
    let (centre, rotation) = if tumbling {
        let angles = VEHICLE_TUMBLE_RATES * frame as f32;
        (
            VEHICLE_ORBIT_CENTRE + VEHICLE_ORBIT_RADIUS * Vec3::X,
            Quat::from_euler(EulerRot::XYZ, angles.x, angles.y, angles.z),
        )
    } else {
        let angle = std::f32::consts::TAU * frame as f32 / VEHICLE_PHASE_FRAMES as f32;
        (
            VEHICLE_ORBIT_CENTRE
                + VEHICLE_ORBIT_RADIUS * Vec3::new(angle.cos(), 0.0, angle.sin()),
            Quat::from_rotation_y(-angle),
        )
    };
    Transform {
        translation: centre - rotation * vehicle_hull::CENTRE,
        rotation,
        scale: Vec3::ONE,
    }
}

/// Sets the hull's transform from the frame of its motion, then moves on a
/// frame. Driven per frame, not by wall-clock time, so every run shows the
/// same poses.
#[cfg(feature = "sub_grids")]
fn move_vehicle(
    mut motion: ResMut<VehicleMotion>,
    mut transforms: Query<&mut Transform>,
) {
    let Some(hull) = motion.hull else {
        return;
    };
    if let Ok(mut transform) = transforms.get_mut(hull) {
        *transform = vehicle_pose(motion.tumbling, motion.frame);
    }
    motion.frame += 1;
}

/// The chunk positions of the hull's chunks published since the scenario
/// last took them.
#[cfg(feature = "sub_grids")]
#[derive(Resource, Default)]
struct HullPublishes {
    hull: Option<Entity>,
    chunks: Vec<IVec3>,
}

#[cfg(feature = "sub_grids")]
fn record_hull_publishes<C: VoxelWorldConfig>(
    mut publishes: ResMut<HullPublishes>,
    mut messages: MessageReader<PublishMessage<C>>,
) {
    let Some(hull) = publishes.hull else {
        messages.read().for_each(drop);
        return;
    };
    let published: Vec<IVec3> = messages
        .read()
        .filter(|message| message.grid == Grid::SubGrid(hull))
        .map(|message| message.chunk_key.position)
        .collect();
    publishes.chunks.extend(published);
}

/// Spawns a sub-grid at `transform` and builds the vehicle hull in it, with
/// lamp voxels of the material `lamp` when given. Returns the sub-grid's
/// entity and what the hull holds. The hull commits in the next frame.
#[cfg(feature = "sub_grids")]
fn spawn_hull<C: VoxelWorldConfig<MaterialIndex = u8>>(
    app: &mut App,
    transform: Transform,
    lamp: Option<u8>,
) -> (Entity, vehicle_hull::Hull) {
    use bevy::ecs::system::RunSystemOnce;
    app.world_mut()
        .run_system_once(move |mut voxel_world: VoxelWorld<C>| {
            let hull = voxel_world.spawn_sub_grid(transform);
            let mut grid = voxel_world.grid_mut(Grid::SubGrid(hull)).unwrap();
            (hull, vehicle_hull::build(&mut grid, lamp))
        })
        .unwrap()
}

/// How many chunks the sub-grid `hull` has, and how many of them are Ready.
#[cfg(feature = "sub_grids")]
fn hull_chunks<C: VoxelWorldConfig>(app: &mut App, hull: Entity) -> (usize, usize) {
    let world = app.world_mut();
    let keys: Vec<ChunkKey> = world
        .query::<(&Chunk<C>, &ChildOf)>()
        .iter(world)
        .filter(|(_, parent)| parent.parent() == hull)
        .map(|(chunk, _)| chunk.key())
        .collect();
    let mut state = SystemState::<VoxelWorld<C>>::new(world);
    let voxel_world = state.get(world).unwrap();
    let grid = voxel_world.grid(Grid::SubGrid(hull)).unwrap();
    let ready = keys
        .iter()
        .filter(|key| grid.chunk_state(**key) == Some(ChunkStatus::Ready))
        .count();
    (keys.len(), ready)
}

/// Runs frames until every chunk of the sub-grid `hull` has been Ready for
/// `SETTLE_FRAMES` frames in a row, or for at most `MAX_FRAMES`, and
/// returns how many of its chunks are Ready.
#[cfg(feature = "sub_grids")]
fn settle_hull<C: VoxelWorldConfig>(app: &mut App, hull: Entity) -> usize {
    let mut settled = 0;
    for _ in 0..MAX_FRAMES {
        app.update();
        let (chunks, ready) = hull_chunks::<C>(app, hull);
        settled = if chunks > 0 && ready == chunks {
            settled + 1
        } else {
            0
        };
        if settled >= SETTLE_FRAMES {
            break;
        }
    }
    hull_chunks::<C>(app, hull).1
}

/// A voxel `vehicle` set in the hull that hasn't published yet.
#[cfg(feature = "sub_grids")]
struct PendingEdit {
    position: IVec3,
    voxel: WorldVoxel<u8>,
    set_at: Instant,
    frames: u32,
}

/// Adds what the vehicle scenarios need to an app of world `C`: the hull's
/// motion and the record of its chunks' publishes.
#[cfg(feature = "sub_grids")]
fn add_vehicle<C: VoxelWorldConfig>(app: &mut App) {
    app.init_resource::<VehicleMotion>()
        .init_resource::<HullPublishes>()
        .add_systems(Update, move_vehicle)
        .add_systems(Last, record_hull_publishes::<C>);
}

/// Builds the vehicle hull in a new sub-grid at the start of its flight,
/// with lamp voxels of the material `lamp` when given, and runs frames until
/// its chunks are Ready. Reports its solid voxels and Ready chunks under
/// `prefix`, and returns the sub-grid's entity.
///
/// Prints a warning when the Ready chunks differ from `vehicle_hull`'s
/// prediction.
#[cfg(feature = "sub_grids")]
fn build_vehicle<C: VoxelWorldConfig<MaterialIndex = u8>>(
    app: &mut App,
    prefix: &str,
    lamp: Option<u8>,
    metrics: &mut Metrics,
) -> Entity {
    let (hull, built) = spawn_hull::<C>(app, vehicle_pose(false, 0), lamp);
    let ready = settle_hull::<C>(app, hull);
    if ready != built.chunks.len() {
        eprintln!(
            "{prefix}: {ready} hull chunks are Ready, but the hull should have {}",
            built.chunks.len()
        );
    }
    metrics.insert(format!("{prefix}.solid_voxels"), built.solid_voxels as f64);
    metrics.insert(format!("{prefix}.ready_chunks"), ready as f64);
    hull
}

/// Starts the motion of `hull` from its first frame, in flight or
/// tumbling. With no hull, the hull stays where it is.
#[cfg(feature = "sub_grids")]
fn set_motion(app: &mut App, hull: Option<Entity>, tumbling: bool) {
    *app.world_mut().resource_mut::<VehicleMotion>() = VehicleMotion {
        hull,
        tumbling,
        frame: 0,
    };
}

/// Edits the sub-grid `hull` every frame for `VEHICLE_PHASE_FRAMES` frames
/// while it keeps its motion, and reports under `name` the frames' metrics
/// and how long the edits took to publish. Calls `after` after each frame,
/// outside the frame's time.
///
/// The edits visit `vehicle_hull::edit_sites`, one in each hull chunk, in
/// turn, setting a material from `VEHICLE_EDIT` up on even frames and
/// digging on odd ones, so each site alternates between the two. An edit
/// has published on the first frame its chunk's `ChunkPublished` arrives
/// with data holding it or a later edit at its site: a chunk task that
/// starts after several edits publishes them all at once. A dug site reads
/// air whichever dig it shows, so air counts as the earliest dig waiting.
///
/// Reports the median and worst frames and time from setting an edit to its
/// publish. An edit that hasn't published `VEHICLE_DRAIN_FRAMES` after the
/// phase, which keeps the motion with no edits, counts as unpublished.
#[cfg(feature = "sub_grids")]
fn edit_vehicle_every_frame<C: VoxelWorldConfig<MaterialIndex = u8>>(
    app: &mut App,
    hull: Entity,
    name: &str,
    metrics: &mut Metrics,
    mut after: impl FnMut(&mut App),
) {
    let sites = vehicle_hull::edit_sites();
    let site_in_chunk: HashMap<IVec3, IVec3> = sites
        .iter()
        .map(|site| (vehicle_hull::chunk_of(*site), *site))
        .collect();
    *app.world_mut().resource_mut::<HullPublishes>() = HullPublishes {
        hull: Some(hull),
        chunks: Vec::new(),
    };
    let mut state = SystemState::<VoxelWorld<C>>::new(app.world_mut());
    let mut pending: Vec<PendingEdit> = Vec::new();
    let mut frames_to_publish = Vec::new();
    let mut times_to_publish = Vec::new();
    let mut times = Vec::new();
    for frame in 0..VEHICLE_PHASE_FRAMES + VEHICLE_DRAIN_FRAMES {
        if frame < VEHICLE_PHASE_FRAMES {
            let position = sites[frame as usize % sites.len()];
            let visit = frame as usize / sites.len();
            let voxel = if frame % 2 == 0 {
                WorldVoxel::Solid(VEHICLE_EDIT + (visit % 200) as u8)
            } else {
                WorldVoxel::Air
            };
            let mut voxel_world = state.get_mut(app.world_mut()).unwrap();
            voxel_world
                .grid_mut(Grid::SubGrid(hull))
                .unwrap()
                .set_voxel(position, voxel);
            state.apply(app.world_mut());
            pending.push(PendingEdit {
                position,
                voxel,
                set_at: Instant::now(),
                frames: 0,
            });
        } else if pending.is_empty() {
            break;
        }

        let start = Instant::now();
        app.update();
        if frame < VEHICLE_PHASE_FRAMES {
            times.push(start.elapsed().as_secs_f64() * 1000.0);
        }
        after(app);

        let mut published =
            std::mem::take(&mut app.world_mut().resource_mut::<HullPublishes>().chunks);
        published.sort_by_key(|chunk| chunk.to_array());
        published.dedup();
        let voxel_world = state.get(app.world()).unwrap();
        let grid = voxel_world.grid(Grid::SubGrid(hull)).unwrap();
        let mut landed = vec![false; pending.len()];
        for chunk in published {
            let Some(site) = site_in_chunk.get(&chunk) else {
                continue;
            };
            let Some(shown) = grid
                .get_chunk_data(ChunkKey::new(chunk, 0))
                .and_then(|data| data.get_voxel_at_world_position(*site))
            else {
                continue;
            };
            // The site's waiting edits, oldest first: the data holds the
            // first one that matches what it shows, and every one before.
            let waiting: Vec<usize> = (0..pending.len())
                .filter(|index| pending[*index].position == *site)
                .collect();
            if let Some(shown) = waiting
                .iter()
                .position(|index| pending[*index].voxel == shown)
            {
                for index in &waiting[..=shown] {
                    landed[*index] = true;
                }
            }
        }
        let mut landed = landed.into_iter();
        pending.retain_mut(|edit| {
            edit.frames += 1;
            let landed = landed.next().unwrap();
            if landed {
                frames_to_publish.push(edit.frames as f64);
                times_to_publish.push(edit.set_at.elapsed().as_secs_f64() * 1000.0);
            }
            !landed
        });
    }
    let unpublished = pending.len();

    frame_metrics(name, &times, metrics);
    metrics.insert(format!("{name}.edits"), VEHICLE_PHASE_FRAMES as f64);
    metrics.insert(format!("{name}.sites"), sites.len() as f64);
    metrics.insert(format!("{name}.unpublished_edits"), unpublished as f64);
    if frames_to_publish.is_empty() {
        eprintln!("{name}: no edit published");
        frames_to_publish.push(f64::INFINITY);
        times_to_publish.push(f64::INFINITY);
    }
    metrics.insert(
        format!("{name}.frames_to_publish"),
        percentile(&frames_to_publish, 0.5),
    );
    metrics.insert(
        format!("{name}.max_frames_to_publish"),
        percentile(&frames_to_publish, 1.0),
    );
    metrics.insert(
        format!("{name}.time_to_publish_ms"),
        percentile(&times_to_publish, 0.5),
    );
    metrics.insert(
        format!("{name}.max_time_to_publish_ms"),
        percentile(&times_to_publish, 1.0),
    );
}

/// A large, editable sub-grid that moves: the vehicle hull, built by voxel
/// edits before timing starts, flies upright for `VEHICLE_PHASE_FRAMES`
/// over the streamed sine world, tumbles on all three axes for as many, and
/// tumbles for as many again while one voxel of it is set or dug every
/// frame, as `edit_vehicle_every_frame` says. The camera stays still, so
/// the world's chunks stay put.
///
/// Reports the hull's solid voxels and its Ready level-0 chunks, frame
/// metrics for each phase, and the median and worst frames and time from
/// setting an edit to its publish. Timings have no threshold.
#[cfg(feature = "sub_grids")]
fn scenario_vehicle() -> Metrics {
    let prefix = "vehicle";
    let mut app = build_app(SineWorld::default());
    add_vehicle::<SineWorld>(&mut app);
    stream_in::<SineWorld>(&mut app);

    let mut metrics = Metrics::new();
    let hull = build_vehicle::<SineWorld>(&mut app, prefix, None, &mut metrics);

    set_motion(&mut app, Some(hull), false);
    let times = timed_frames(&mut app, VEHICLE_PHASE_FRAMES as usize, |_| {});
    frame_metrics(&format!("{prefix}.flight"), &times, &mut metrics);

    set_motion(&mut app, Some(hull), true);
    let times = timed_frames(&mut app, VEHICLE_PHASE_FRAMES as usize, |_| {});
    frame_metrics(&format!("{prefix}.tumble"), &times, &mut metrics);

    edit_vehicle_every_frame::<SineWorld>(
        &mut app,
        hull,
        &format!("{prefix}.edited"),
        &mut metrics,
        |_| {},
    );
    metrics
}

/// Frames `vehicle_lit` hovers before it flies.
#[cfg(all(feature = "light_propagation", feature = "sub_grids"))]
const VEHICLE_HOVER_FRAMES: u32 = 60;

/// Light work in a lit world over one frame or a span of frames, as
/// `LightWatch` records it.
#[cfg(all(feature = "light_propagation", feature = "sub_grids"))]
#[derive(Clone, Copy, Default)]
struct LightFrame {
    /// Whether `light_is_settled` held after the frame.
    settled: bool,
    /// Solves of every grid, in chunk tasks and relights.
    solves: u64,
    /// Relights of every grid started.
    relights: u64,
    /// Of those, the relights of sub-grid chunks.
    sub_grid_relights: u64,
    /// Sub-grid chunks solved in their chunk tasks.
    sub_grid_chunk_solves: u64,
    /// Main thread spent landing light and starting relights, in
    /// microseconds: in a frame, or the most in any frame of a span.
    main_thread_us: f64,
}

#[cfg(all(feature = "light_propagation", feature = "sub_grids"))]
impl LightFrame {
    /// The relights started for the world's own chunks.
    fn world_relights(&self) -> u64 {
        self.relights - self.sub_grid_relights
    }

    /// The sub-grid solves and relights started, each a light solve of a
    /// sub-grid chunk.
    fn sub_grid_work(&self) -> u64 {
        self.sub_grid_relights + self.sub_grid_chunk_solves
    }
}

/// The light work of a lit world `C`, frame by frame, in named phases.
///
/// A scenario starts each phase with `start` and calls `frame` after each
/// frame it runs, outside the frame's time. A phase runs from its start to
/// the next phase's. The counts come from `light_stats`, so work a chunk
/// task or relight task does off the main thread shows in the frame it
/// finished by.
#[cfg(all(feature = "light_propagation", feature = "sub_grids"))]
struct LightWatch<C> {
    frames: Vec<LightFrame>,
    /// Each phase's name and its first frame's index in `frames`.
    phases: Vec<(&'static str, usize)>,
    last: bevy_voxel_world::light_bench::LightStats,
    world: std::marker::PhantomData<C>,
}

#[cfg(all(feature = "light_propagation", feature = "sub_grids"))]
impl<C: VoxelWorldConfig> LightWatch<C> {
    fn new(app: &App) -> Self {
        Self {
            frames: Vec::new(),
            phases: Vec::new(),
            last: Self::stats(app),
            world: std::marker::PhantomData,
        }
    }

    fn stats(app: &App) -> bevy_voxel_world::light_bench::LightStats {
        bevy_voxel_world::light_bench::light_stats::<C>(app.world()).unwrap()
    }

    /// Starts the phase `name`, ending the one before.
    fn start(&mut self, name: &'static str) {
        self.phases.push((name, self.frames.len()));
    }

    /// Records the frame just run.
    fn frame(&mut self, app: &mut App) {
        let now = Self::stats(app);
        let last = self.last;
        self.frames.push(LightFrame {
            settled: light_is_settled::<C>(app),
            solves: now.solves - last.solves,
            relights: now.relights - last.relights,
            sub_grid_relights: now.sub_grid_relights - last.sub_grid_relights,
            sub_grid_chunk_solves: now.sub_grid_chunk_solves - last.sub_grid_chunk_solves,
            main_thread_us: (now.main_thread_ms - last.main_thread_ms) * 1000.0,
        });
        self.last = now;
    }

    /// Runs and records frames until light is settled, or for at most
    /// `MAX_FRAMES`.
    fn settle(&mut self, app: &mut App) {
        for _ in 0..MAX_FRAMES {
            if light_is_settled::<C>(app) {
                return;
            }
            app.update();
            self.frame(app);
        }
        eprintln!("light didn't settle in {MAX_FRAMES} frames");
    }

    /// The index in `frames` of the phase `name`'s first frame, and of the
    /// next phase's.
    fn phase(&self, name: &str) -> (usize, usize) {
        let index = self
            .phases
            .iter()
            .position(|(phase, _)| *phase == name)
            .unwrap();
        let end = self
            .phases
            .get(index + 1)
            .map_or(self.frames.len(), |(_, first)| *first);
        (self.phases[index].1, end)
    }

    /// The light work of the frames from `first` up to `end`, exclusive.
    fn sum(&self, first: usize, end: usize) -> LightFrame {
        let frames = &self.frames[first..end];
        LightFrame {
            settled: frames.last().is_some_and(|frame| frame.settled),
            solves: frames.iter().map(|frame| frame.solves).sum(),
            relights: frames.iter().map(|frame| frame.relights).sum(),
            sub_grid_relights: frames.iter().map(|frame| frame.sub_grid_relights).sum(),
            sub_grid_chunk_solves: frames
                .iter()
                .map(|frame| frame.sub_grid_chunk_solves)
                .sum(),
            main_thread_us: frames
                .iter()
                .map(|frame| frame.main_thread_us)
                .fold(0.0, f64::max),
        }
    }

    /// The light work of the phase `name`.
    fn in_phase(&self, name: &str) -> LightFrame {
        let (first, end) = self.phase(name);
        self.sum(first, end)
    }

    /// The frames from the start of the phase `name` until light was
    /// settled, counting the frame it settled on, which may come in a later
    /// phase, and the light work of those frames. Light that never settled
    /// counts every frame recorded, with a warning.
    fn until_settled(&self, name: &str) -> (usize, LightFrame) {
        let (first, _) = self.phase(name);
        let end = match self.frames[first..].iter().position(|frame| frame.settled) {
            Some(settled) => first + settled + 1,
            None => {
                eprintln!("{name}: light didn't settle");
                self.frames.len()
            }
        };
        (end - first, self.sum(first, end))
    }

    /// Reports the phase `name`'s frames until light settled and their light
    /// work under `{prefix}.{name}`. Returns that light work.
    fn report_until_settled(
        &self,
        prefix: &str,
        name: &str,
        metrics: &mut Metrics,
    ) -> LightFrame {
        let (frames, work) = self.until_settled(name);
        let name = format!("{prefix}.{name}");
        metrics.insert(format!("{name}.frames_to_settled"), frames as f64);
        metrics.insert(format!("{name}.solves"), work.solves as f64);
        metrics.insert(format!("{name}.relights"), work.relights as f64);
        metrics.insert(
            format!("{name}.world_relights"),
            work.world_relights() as f64,
        );
        metrics.insert(
            format!("{name}.sub_grid_relights"),
            work.sub_grid_relights as f64,
        );
        metrics.insert(
            format!("{name}.sub_grid_chunk_solves"),
            work.sub_grid_chunk_solves as f64,
        );
        metrics.insert(format!("{name}.max_main_thread_us"), work.main_thread_us);
        work
    }

    /// Reports the sub-grid light work of the phase `name`, in which none
    /// should start, under `{prefix}.{name}`. Returns how much started.
    fn report_still(&self, prefix: &str, name: &str, metrics: &mut Metrics) -> u64 {
        let work = self.in_phase(name);
        let name = format!("{prefix}.{name}");
        metrics.insert(
            format!("{name}.sub_grid_relights"),
            work.sub_grid_relights as f64,
        );
        metrics.insert(
            format!("{name}.sub_grid_chunk_solves"),
            work.sub_grid_chunk_solves as f64,
        );
        metrics.insert(format!("{name}.max_main_thread_us"), work.main_thread_us);
        work.sub_grid_work()
    }
}

#[cfg(all(feature = "light_propagation", feature = "sub_grids"))]
impl<C: VoxelWorldConfig> BurstWatch for LightWatch<C> {
    /// Lets light settle before every phase but `moving`, so each phase's
    /// light work is its own, and starts the phase. The pieces start moving
    /// right after the lift lands, as a game's would.
    fn phase(&mut self, app: &mut App, name: &'static str) {
        if name != "moving" {
            self.settle(app);
        }
        self.start(name);
    }

    fn frame(&mut self, app: &mut App) {
        LightWatch::frame(self, app);
    }
}

/// The edits `vehicle_lit` makes to the hull one at a time, in order, with
/// frames paced at 60 a second: `edits_lit`'s edits, on the hull. At each
/// site, a block set on the surface and its undo, then the surface voxel
/// dug and its undo.
///
/// The sites are on the envelope's top and the cabin's floor, in the middle
/// of a chunk and against chunk faces. Each dig leaves the wall under it
/// closed, so the hull stays enclosed.
#[cfg(all(feature = "light_propagation", feature = "sub_grids"))]
fn paced_hull_edits() -> Vec<(IVec3, WorldVoxel<u8>)> {
    let lamp = Some(LAMP);
    let first_solid_down = |from: IVec3| {
        (0..=from.y)
            .map(|down| from - IVec3::Y * down)
            .find(|position| vehicle_hull::hull_voxel(*position, lamp).is_some())
            .unwrap()
    };
    let top = vehicle_hull::MAX.y - 1;
    // The cabin's inside reaches up to y 34; x 44 misses the lamps.
    let cabin = 33;
    let surfaces = [
        first_solid_down(IVec3::new(48, top, 112)),
        first_solid_down(IVec3::new(31, top, 112)),
        first_solid_down(IVec3::new(48, top, 128)),
        first_solid_down(IVec3::new(44, cabin, 116)),
        first_solid_down(IVec3::new(44, cabin, 127)),
        first_solid_down(IVec3::new(44, cabin, 128)),
    ];
    let mut edits = Vec::new();
    for surface in surfaces {
        let material = vehicle_hull::hull_voxel(surface, lamp).unwrap();
        let above = surface + IVec3::Y;
        edits.push((above, WorldVoxel::Solid(material)));
        edits.push((above, WorldVoxel::Air));
        edits.push((surface, WorldVoxel::Air));
        edits.push((surface, WorldVoxel::Solid(material)));
    }
    edits
}

/// The vehicle hull with lamps in its cabin, in the lit sine world: built
/// by voxel edits with its light settled before timing starts, it hovers
/// for `VEHICLE_HOVER_FRAMES`, then flies and tumbles as in `vehicle`. Still
/// tumbling, it takes `paced_hull_edits` one at a time with frames paced
/// at 60 a second, as `edits_lit` does, and then an edit every frame as in
/// `vehicle`.
///
/// Reports frame metrics for each phase, and for each phase with no edits
/// the sub-grid solves and relights it started and the most main thread
/// light took in a frame. For the paced edits, each one's frames to land,
/// counted from the edit's frame until light is settled, its frames and
/// time to publish, counting the edit's frame, and the most main thread
/// light took in any of its frames. For the edits every frame, what
/// `edit_vehicle_every_frame` reports, and the relights and most main
/// thread in a frame from the phase's start until light settled.
///
/// Gate, checked here and reported as `gate_failures`: no sub-grid solve or
/// relight starts while the hull hovers, flies or tumbles with no edit, and
/// each paced edit lands within 2 frames with at most 1 ms of main thread
/// relighting in any frame. Each solve or relight and each edit missing its
/// gate counts once. The edits every frame have no gate.
#[cfg(all(feature = "light_propagation", feature = "sub_grids"))]
fn scenario_vehicle_lit() -> Metrics {
    type Lit = LitBenchWorld<SINE, true>;
    let prefix = "vehicle_lit";
    let mut app = build_app(Lit::default());
    add_vehicle::<Lit>(&mut app);
    stream_in::<Lit>(&mut app);
    settle_light::<Lit>(&mut app);

    let mut metrics = Metrics::new();
    let hull = build_vehicle::<Lit>(&mut app, prefix, Some(LAMP), &mut metrics);
    settle_light::<Lit>(&mut app);
    let mut watch = LightWatch::<Lit>::new(&app);

    let mut still = 0;
    for (name, motion, frames) in [
        ("hover", None, VEHICLE_HOVER_FRAMES),
        ("flight", Some(false), VEHICLE_PHASE_FRAMES),
        ("tumble", Some(true), VEHICLE_PHASE_FRAMES),
    ] {
        set_motion(&mut app, motion.map(|_| hull), motion == Some(true));
        watch.start(name);
        let times = timed_frames(&mut app, frames as usize, |app| watch.frame(app));
        frame_metrics(&format!("{prefix}.{name}"), &times, &mut metrics);
        still += watch.report_still(prefix, name, &mut metrics);
    }
    if still > 0 {
        eprintln!("{prefix}: {still} sub-grid solves and relights while the hull moved");
    }

    // Still tumbling.
    watch.start("paced");
    *app.world_mut().resource_mut::<HullPublishes>() = HullPublishes {
        hull: Some(hull),
        chunks: Vec::new(),
    };
    let mut state = SystemState::<VoxelWorld<Lit>>::new(app.world_mut());
    let edits = paced_hull_edits();
    let mut times = Vec::new();
    let mut frames_to_land = Vec::new();
    let mut frames_to_publish = Vec::new();
    let mut times_to_publish = Vec::new();
    let mut worst_us = Vec::new();
    let mut unpublished = 0;
    for (position, voxel) in &edits {
        let mut voxel_world = state.get_mut(app.world_mut()).unwrap();
        voxel_world
            .grid_mut(Grid::SubGrid(hull))
            .unwrap()
            .set_voxel(*position, *voxel);
        state.apply(app.world_mut());
        let set_at = Instant::now();
        let first = watch.frames.len();
        let chunk = vehicle_hull::chunk_of(*position);
        let mut landed = None;
        let mut published = None;
        for frame in 1..=MAX_FRAMES {
            let start = Instant::now();
            app.update();
            times.push(start.elapsed().as_secs_f64() * 1000.0);
            watch.frame(&mut app);
            let chunks = std::mem::take(
                &mut app.world_mut().resource_mut::<HullPublishes>().chunks,
            );
            if published.is_none() && chunks.contains(&chunk) {
                let voxel_world = state.get(app.world()).unwrap();
                let shown = voxel_world
                    .grid(Grid::SubGrid(hull))
                    .unwrap()
                    .get_chunk_data(ChunkKey::new(chunk, 0))
                    .and_then(|data| data.get_voxel_at_world_position(*position));
                if shown == Some(*voxel) {
                    published = Some((frame, set_at.elapsed().as_secs_f64() * 1000.0));
                }
            }
            if landed.is_none() && watch.frames.last().unwrap().settled {
                landed = Some(frame - 1);
            }
            if landed.is_some() && published.is_some() {
                break;
            }
            std::thread::sleep(PACED_FRAME.saturating_sub(start.elapsed()));
        }
        // A frame's rest, as a game's frame would give the edit's frame.
        std::thread::sleep(PACED_FRAME);
        frames_to_land.push(landed.map_or(f64::INFINITY, |frames| frames as f64));
        match published {
            Some((frames, ms)) => {
                frames_to_publish.push(frames as f64);
                times_to_publish.push(ms);
            }
            None => unpublished += 1,
        }
        worst_us.push(watch.sum(first, watch.frames.len()).main_thread_us);
    }
    let missed = frames_to_land
        .iter()
        .zip(&worst_us)
        .filter(|(frames, worst)| {
            **frames > EDIT_FRAMES_GATE as f64 || **worst > EDIT_MAIN_THREAD_GATE_US
        })
        .count();
    if missed > 0 {
        eprintln!(
            "{prefix}: {missed} of {} paced edits missed the gate \
             (frames {frames_to_land:?}, main thread µs {worst_us:?})",
            edits.len()
        );
    }
    let paced = format!("{prefix}.paced");
    frame_metrics(&paced, &times, &mut metrics);
    let max = |values: &[f64]| values.iter().copied().fold(0.0, f64::max);
    metrics.insert(format!("{paced}.edits"), edits.len() as f64);
    metrics.insert(format!("{paced}.max_frames_to_land"), max(&frames_to_land));
    metrics.insert(
        format!("{paced}.p50_frames_to_land"),
        percentile(&frames_to_land, 0.5),
    );
    metrics.insert(format!("{paced}.max_main_thread_us"), max(&worst_us));
    metrics.insert(
        format!("{paced}.p50_main_thread_us"),
        percentile(&worst_us, 0.5),
    );
    metrics.insert(format!("{paced}.unpublished_edits"), unpublished as f64);
    if frames_to_publish.is_empty() {
        frames_to_publish.push(f64::INFINITY);
        times_to_publish.push(f64::INFINITY);
    }
    metrics.insert(
        format!("{paced}.frames_to_publish"),
        percentile(&frames_to_publish, 0.5),
    );
    metrics.insert(
        format!("{paced}.max_frames_to_publish"),
        max(&frames_to_publish),
    );
    metrics.insert(
        format!("{paced}.time_to_publish_ms"),
        percentile(&times_to_publish, 0.5),
    );
    metrics.insert(
        format!("{paced}.max_time_to_publish_ms"),
        max(&times_to_publish),
    );

    watch.start("edited");
    edit_vehicle_every_frame::<Lit>(
        &mut app,
        hull,
        &format!("{prefix}.edited"),
        &mut metrics,
        |app| watch.frame(app),
    );
    watch.settle(&mut app);
    watch.report_until_settled(prefix, "edited", &mut metrics);

    metrics.insert(
        format!("{prefix}.gate_failures"),
        (still as usize + missed) as f64,
    );
    metrics
}

/// Frames `lift_burst`'s pieces move between their lift and their stamp.
#[cfg(feature = "sub_grids")]
const BURST_MOVE_FRAMES: u32 = 300;

/// How far a moving piece rises, in voxels, at the middle of its path.
#[cfg(feature = "sub_grids")]
const BURST_RISE: f32 = 24.0;

/// How far a moving piece turns, in radians, at the middle of its path.
#[cfg(feature = "sub_grids")]
const BURST_TURN: f32 = 0.6;

/// How far a piece ends its path from its start pose: turned by
/// `BURST_END_TURN` radians, about 3 degrees, and moved by
/// `BURST_END_DRIFT` voxels. The stamp snaps both away, so the piece lands
/// where it was lifted from.
#[cfg(feature = "sub_grids")]
const BURST_END_TURN: f32 = 0.05;
#[cfg(feature = "sub_grids")]
const BURST_END_DRIFT: Vec3 = Vec3::new(0.3, 0.2, -0.25);

/// The sub-grids whose `Lifted` and `Stamped` have arrived since a burst
/// phase started.
#[cfg(feature = "sub_grids")]
#[derive(Resource, Default)]
struct BurstLanded {
    lifted: Vec<Entity>,
    stamped: Vec<Entity>,
}

#[cfg(feature = "sub_grids")]
fn record_burst_landed<C: VoxelWorldConfig>(
    mut landed: ResMut<BurstLanded>,
    mut lifted: MessageReader<Lifted<C>>,
    mut stamped: MessageReader<Stamped<C>>,
) {
    landed
        .lifted
        .extend(lifted.read().map(|message| message.sub_grid));
    landed
        .stamped
        .extend(stamped.read().map(|message| message.sub_grid));
}

/// A piece lifted out of the world, as it was when its lift landed.
#[cfg(feature = "sub_grids")]
#[derive(Clone, Copy)]
struct LiftedPiece {
    sub_grid: Entity,
    /// The sub-grid's translation under the world's root when lifted.
    start: Vec3,
    /// The centre of the box holding the sub-grid's solid voxels, on its
    /// own grid, which the piece turns about.
    pivot: Vec3,
    /// The sub-grid's level-0 chunks.
    chunks: usize,
}

/// How a burst of lifts or stamps landed, from the frame of the calls to
/// the frame the last `Lifted` or `Stamped` arrived.
#[cfg(feature = "sub_grids")]
struct BurstLanding {
    frames: u32,
    time_ms: f64,
    worst_frame_ms: f64,
    /// Handovers that landed in the span, and the chunks they held.
    handovers: u64,
    chunks: u64,
    errors: usize,
}

#[cfg(feature = "sub_grids")]
impl BurstLanding {
    fn report(&self, prefix: &str, metrics: &mut Metrics) {
        metrics.insert(format!("{prefix}.frames"), self.frames as f64);
        metrics.insert(format!("{prefix}.time_ms"), self.time_ms);
        metrics.insert(format!("{prefix}.worst_frame_ms"), self.worst_frame_ms);
        metrics.insert(format!("{prefix}.handovers"), self.handovers as f64);
        metrics.insert(
            format!("{prefix}.chunks_per_handover"),
            if self.handovers == 0 {
                0.0
            } else {
                self.chunks as f64 / self.handovers as f64
            },
        );
        metrics.insert(format!("{prefix}.errors"), self.errors as f64);
    }
}

/// What a burst scenario watches beside what `lift_burst_metrics` times:
/// told as each phase starts, before its calls, and after each frame it
/// runs, outside the frame's time and the phase's.
///
/// The phases are `lift`, `moving`, `stamp`, `relift` and `despawn`.
#[cfg(feature = "sub_grids")]
trait BurstWatch {
    fn phase(&mut self, _app: &mut App, _name: &'static str) {}

    fn frame(&mut self, _app: &mut App) {}
}

/// Watches nothing.
#[cfg(feature = "sub_grids")]
impl BurstWatch for () {}

/// Runs `calls`, which returns how many sub-grids should land and how many
/// calls failed, then runs frames until `landed` counts that many, or for
/// at most `MAX_FRAMES`. The first frame's time includes the calls'.
#[cfg(feature = "sub_grids")]
fn land_burst<C: VoxelWorldConfig>(
    app: &mut App,
    calls: impl FnOnce(&mut App) -> (usize, usize),
    landed: impl Fn(&BurstLanded) -> usize,
    watch: &mut dyn BurstWatch,
) -> BurstLanding {
    let stats = |app: &App| {
        bevy_voxel_world::grid_bench::handover_stats::<C>(app.world()).unwrap()
    };
    *app.world_mut().resource_mut::<BurstLanded>() = BurstLanded::default();
    let before = stats(app);
    let start = Instant::now();
    let (expected, errors) = calls(app);
    let mut frame_start = start;
    let mut frames = 0;
    let mut worst_frame_ms: f64 = 0.0;
    let mut watched = std::time::Duration::ZERO;
    while frames < MAX_FRAMES {
        app.update();
        frames += 1;
        worst_frame_ms = worst_frame_ms.max(frame_start.elapsed().as_secs_f64() * 1000.0);
        let watch_start = Instant::now();
        watch.frame(app);
        watched += watch_start.elapsed();
        frame_start = Instant::now();
        if landed(app.world().resource::<BurstLanded>()) >= expected {
            break;
        }
    }
    let time_ms = (start.elapsed() - watched).as_secs_f64() * 1000.0;
    let landed = landed(app.world().resource::<BurstLanded>());
    if landed < expected {
        eprintln!("{landed} of {expected} sub-grids landed in {MAX_FRAMES} frames");
    }
    let after = stats(app);
    BurstLanding {
        frames,
        time_ms,
        worst_frame_ms,
        handovers: after.handovers - before.handovers,
        chunks: after.chunks - before.chunks,
        errors,
    }
}

/// Lifts every piece out of the world in one frame, each with every solid
/// voxel in its box, and runs frames until every lift has landed. Returns
/// the pieces that lifted, in the order of `pieces`, and how the burst
/// landed.
#[cfg(feature = "sub_grids")]
fn lift_burst<C: VoxelWorldConfig<MaterialIndex = u8>>(
    app: &mut App,
    pieces: &[lift_pieces::Piece],
    watch: &mut dyn BurstWatch,
) -> (Vec<LiftedPiece>, BurstLanding) {
    use bevy::ecs::system::RunSystemOnce;
    let boxes: Vec<(IVec3, IVec3)> =
        pieces.iter().map(|piece| (piece.min, piece.max)).collect();
    let mut sub_grids = Vec::new();
    let landing = land_burst::<C>(
        app,
        |app| {
            let results = app
                .world_mut()
                .run_system_once(move |mut voxel_world: VoxelWorld<C>| {
                    boxes
                        .iter()
                        .map(|(min, max)| {
                            voxel_world.lift(Grid::World, *min, *max, |_, voxel| {
                                matches!(voxel, WorldVoxel::Solid(_))
                            })
                        })
                        .collect::<Vec<_>>()
                })
                .unwrap();
            sub_grids = results.iter().filter_map(|result| result.ok()).collect();
            (sub_grids.len(), results.len() - sub_grids.len())
        },
        |landed| landed.lifted.len(),
        watch,
    );

    let world = app.world_mut();
    let mut chunks: HashMap<Entity, usize> = HashMap::new();
    for (_, parent) in world.query::<(&Chunk<C>, &ChildOf)>().iter(world) {
        *chunks.entry(parent.parent()).or_default() += 1;
    }
    let mut state = SystemState::<VoxelWorld<C>>::new(world);
    let voxel_world = state.get(world).unwrap();
    let pivots: Vec<Vec3> = sub_grids
        .iter()
        .map(|sub_grid| {
            let (min, max) = voxel_world
                .grid(Grid::SubGrid(*sub_grid))
                .and_then(|grid| grid.bounds())
                .unwrap_or_default();
            (min + max).as_vec3() / 2.0
        })
        .collect();
    let lifted = sub_grids
        .iter()
        .zip(pivots)
        .map(|(sub_grid, pivot)| LiftedPiece {
            sub_grid: *sub_grid,
            start: world.get::<Transform>(*sub_grid).unwrap().translation,
            pivot,
            chunks: chunks.get(sub_grid).copied().unwrap_or(0),
        })
        .collect();
    (lifted, landing)
}

/// The transform of the `index`th lifted piece at `frame` of its path,
/// from 1 to `BURST_MOVE_FRAMES`, under the world's root.
///
/// The piece rises by up to `BURST_RISE` and turns by up to `BURST_TURN`
/// about its pivot, on an axis of its own, and comes back down, ending
/// `BURST_END_TURN` and `BURST_END_DRIFT` from its start pose.
#[cfg(feature = "sub_grids")]
fn burst_pose(index: usize, piece: &LiftedPiece, frame: u32) -> Transform {
    let t = frame as f32 / BURST_MOVE_FRAMES as f32;
    let arc = (std::f32::consts::PI * t).sin();
    let axis = Vec3::new((index as f32).sin(), 2.0, (index as f32).cos()).normalize();
    let rotation = Quat::from_axis_angle(axis, BURST_TURN * arc + BURST_END_TURN * t);
    let pivot =
        piece.start + piece.pivot + BURST_RISE * arc * Vec3::Y + BURST_END_DRIFT * t;
    Transform {
        translation: pivot - rotation * piece.pivot,
        rotation,
        scale: Vec3::ONE,
    }
}

/// Moves every lifted piece along its path for `BURST_MOVE_FRAMES` frames,
/// setting its transform before each frame, and returns the frames' times.
#[cfg(feature = "sub_grids")]
fn move_burst(
    app: &mut App,
    pieces: &[LiftedPiece],
    watch: &mut dyn BurstWatch,
) -> Vec<f64> {
    let mut times = Vec::with_capacity(BURST_MOVE_FRAMES as usize);
    for frame in 1..=BURST_MOVE_FRAMES {
        let start = Instant::now();
        for (index, piece) in pieces.iter().enumerate() {
            if let Some(mut transform) =
                app.world_mut().get_mut::<Transform>(piece.sub_grid)
            {
                *transform = burst_pose(index, piece, frame);
            }
        }
        app.update();
        times.push(start.elapsed().as_secs_f64() * 1000.0);
        watch.frame(app);
    }
    times
}

/// Stamps every lifted piece back into the world in one frame, writing
/// every voxel it holds, and runs frames until every stamp has landed.
#[cfg(feature = "sub_grids")]
fn stamp_burst<C: VoxelWorldConfig<MaterialIndex = u8>>(
    app: &mut App,
    pieces: &[LiftedPiece],
    watch: &mut dyn BurstWatch,
) -> BurstLanding {
    use bevy::ecs::system::RunSystemOnce;
    let sub_grids: Vec<Entity> = pieces.iter().map(|piece| piece.sub_grid).collect();
    land_burst::<C>(
        app,
        |app| {
            let errors = app
                .world_mut()
                .run_system_once(move |mut voxel_world: VoxelWorld<C>| {
                    sub_grids
                        .iter()
                        .filter(|sub_grid| {
                            voxel_world
                                .stamp(**sub_grid, Grid::World, incoming_replaces)
                                .is_err()
                        })
                        .count()
                })
                .unwrap();
            (pieces.len() - errors, errors)
        },
        |landed| landed.stamped.len(),
        watch,
    )
}

/// Despawns every lifted piece in one frame and returns that frame's time,
/// the calls included.
#[cfg(feature = "sub_grids")]
fn despawn_burst<C: VoxelWorldConfig>(
    app: &mut App,
    pieces: &[LiftedPiece],
    watch: &mut dyn BurstWatch,
) -> f64 {
    use bevy::ecs::system::RunSystemOnce;
    let sub_grids: Vec<Entity> = pieces.iter().map(|piece| piece.sub_grid).collect();
    let start = Instant::now();
    app.world_mut()
        .run_system_once(move |mut voxel_world: VoxelWorld<C>| {
            for sub_grid in &sub_grids {
                voxel_world.despawn_sub_grid(*sub_grid);
            }
        })
        .unwrap();
    app.update();
    let time_ms = start.elapsed().as_secs_f64() * 1000.0;
    watch.frame(app);
    time_ms
}

/// A burst of lifts and stamps in a streamed world with its camera still:
/// every piece of `lift_pieces` lifted in one frame, moved every frame for
/// `BURST_MOVE_FRAMES`, stamped back in one frame, lifted again, and
/// despawned in one frame. The app must hold `BurstLanded` and run
/// `record_burst_landed` for the world `C`.
///
/// Reports how many pieces lifted into sub-grids of one chunk, of 2 to 4,
/// and of more, and how many cross a chunk corner of the world. For the
/// first lift, the stamp and the second lift (`relift`), it reports frames
/// and time from the calls to the last sub-grid landing, the worst frame
/// in that span, the handovers that landed and their chunks per handover,
/// and the calls that returned an error. Pieces lifted in one frame share
/// world chunks and fold into fewer handovers, which the handover count
/// shows. Then frame metrics while the pieces move, and the time of the
/// frame that despawns the second lift's pieces.
///
/// Each piece's path ends near its start pose, so the stamp snaps it back
/// to where it was lifted from and the second lift takes the same voxels.
/// A sub-grid whose chunk count differs from its piece's, or a second lift
/// that starts somewhere else, prints a warning.
///
/// `watch` is told of each phase and frame. The time it takes counts in no
/// timing reported here.
#[cfg(feature = "sub_grids")]
fn lift_burst_metrics<C: VoxelWorldConfig<MaterialIndex = u8>>(
    app: &mut App,
    prefix: &str,
    watch: &mut dyn BurstWatch,
) -> Metrics {
    let pieces = lift_pieces::pieces(sine_height);
    let mut metrics = Metrics::new();

    watch.phase(app, "lift");
    let (lifted, landing) = lift_burst::<C>(app, &pieces, watch);
    landing.report(&format!("{prefix}.lift"), &mut metrics);
    let mut counts = [0; 3];
    for (index, (piece, sub_grid)) in pieces.iter().zip(&lifted).enumerate() {
        if sub_grid.chunks != piece.chunks {
            eprintln!(
                "{prefix}: piece {index} has {} chunks, but should have {}",
                sub_grid.chunks, piece.chunks
            );
        }
        counts[match sub_grid.chunks {
            0 | 1 => 0,
            2..=4 => 1,
            _ => 2,
        }] += 1;
    }
    metrics.insert(format!("{prefix}.pieces"), lifted.len() as f64);
    for (name, count) in [
        "pieces_1_chunk",
        "pieces_2_to_4_chunks",
        "pieces_over_4_chunks",
    ]
    .into_iter()
    .zip(counts)
    {
        metrics.insert(format!("{prefix}.{name}"), count as f64);
    }
    metrics.insert(
        format!("{prefix}.pieces_on_chunk_corners"),
        pieces
            .iter()
            .filter(|piece| piece.crosses_chunk_corner())
            .count() as f64,
    );
    metrics.insert(
        format!("{prefix}.sub_grid_chunks"),
        lifted.iter().map(|piece| piece.chunks).sum::<usize>() as f64,
    );

    watch.phase(app, "moving");
    let times = move_burst(app, &lifted, watch);
    frame_metrics(&format!("{prefix}.moving"), &times, &mut metrics);

    watch.phase(app, "stamp");
    let landing = stamp_burst::<C>(app, &lifted, watch);
    landing.report(&format!("{prefix}.stamp"), &mut metrics);

    watch.phase(app, "relift");
    let (relifted, landing) = lift_burst::<C>(app, &pieces, watch);
    landing.report(&format!("{prefix}.relift"), &mut metrics);
    let moved = lifted
        .iter()
        .zip(&relifted)
        .filter(|(first, second)| {
            first.start != second.start || first.chunks != second.chunks
        })
        .count();
    if moved > 0 || relifted.len() != lifted.len() {
        eprintln!(
            "{prefix}: {} pieces lifted again, {moved} of them not as first lifted",
            relifted.len()
        );
    }

    watch.phase(app, "despawn");
    metrics.insert(
        format!("{prefix}.despawn_frame_ms"),
        despawn_burst::<C>(app, &relifted, watch),
    );
    metrics
}

/// `lift_burst_metrics` in the unlit sine world.
#[cfg(feature = "sub_grids")]
fn scenario_lift_burst() -> Metrics {
    let mut app = build_app(SineWorld::default());
    app.init_resource::<BurstLanded>()
        .add_systems(Last, record_burst_landed::<SineWorld>);
    stream_in::<SineWorld>(&mut app);
    lift_burst_metrics::<SineWorld>(&mut app, "lift_burst", &mut ())
}

/// `lift_burst_metrics` in the lit sine world, once its light is at rest,
/// with the light work of each phase, then the same voxels dug in a fresh
/// app of the world, for comparing a lift with the digs it stands for.
///
/// Light settles before every phase but `moving`, which starts right after
/// the lift lands. For `lift`, `stamp`, `relift` and `despawn`, it reports
/// the frames from the calls until light is settled, counting the frame it
/// settles on, and in those frames the solves, the relights of the world's
/// chunks and of sub-grid chunks, sub-grid chunks solved in their chunk
/// tasks, and the most main thread light took in a frame. For `moving`, the
/// sub-grid solves and relights and the most main thread in a frame while
/// the pieces move. For `digs`, the same as for `lift`. Ratio lines compare
/// the lift with the digs.
///
/// Gate, checked here and reported as `gate_failures`, each solve or relight
/// against it counting once: no sub-grid solve or relight starts while the
/// pieces move, counted from the frame after every lift has landed; each
/// lift solves each new sub-grid chunk once, its first solve being a
/// relight; and the first lift starts no more relights of the world's
/// chunks than the digs.
#[cfg(all(feature = "light_propagation", feature = "sub_grids"))]
fn scenario_lift_burst_lit() -> Metrics {
    use bevy::ecs::system::RunSystemOnce;

    type Lit = LitBenchWorld<SINE>;
    let prefix = "lift_burst_lit";
    let mut app = build_app(Lit::default());
    app.init_resource::<BurstLanded>()
        .add_systems(Last, record_burst_landed::<Lit>);
    stream_in::<Lit>(&mut app);
    settle_light::<Lit>(&mut app);
    let mut watch = LightWatch::<Lit>::new(&app);
    let mut metrics = lift_burst_metrics::<Lit>(&mut app, prefix, &mut watch);
    watch.settle(&mut app);
    drop(app);

    let new_chunks = metrics[&format!("{prefix}.sub_grid_chunks")] as u64;
    let mut failures = watch.report_still(prefix, "moving", &mut metrics);
    let mut lift = LightFrame::default();
    for name in ["lift", "stamp", "relift", "despawn"] {
        let work = watch.report_until_settled(prefix, name, &mut metrics);
        if name == "lift" || name == "relift" {
            let again = work.sub_grid_work().saturating_sub(new_chunks);
            if again > 0 {
                eprintln!("{prefix}: {name} solved sub-grid chunks {again} more times than it has");
            }
            failures += again;
        }
        if name == "lift" {
            lift = work;
        }
    }

    let mut app = build_app(Lit::default());
    stream_in::<Lit>(&mut app);
    settle_light::<Lit>(&mut app);
    let mut digs = LightWatch::<Lit>::new(&app);
    digs.start("digs");
    let solids: Vec<IVec3> = lift_pieces::pieces(sine_height)
        .iter()
        .flat_map(|piece| {
            let (min, max) = (piece.min, piece.max);
            (min.z..max.z).flat_map(move |z| {
                (min.y..max.y)
                    .flat_map(move |y| (min.x..max.x).map(move |x| IVec3::new(x, y, z)))
            })
        })
        .filter(|position| position.y < sine_height(position.x, position.z))
        .collect();
    app.world_mut()
        .run_system_once(move |mut voxel_world: VoxelWorld<Lit>| {
            for position in &solids {
                voxel_world.set_voxel(*position, WorldVoxel::Air);
            }
        })
        .unwrap();
    app.update();
    digs.frame(&mut app);
    digs.settle(&mut app);
    let dug = digs.report_until_settled(prefix, "digs", &mut metrics);

    let beyond = lift.world_relights().saturating_sub(dug.world_relights());
    if beyond > 0 {
        eprintln!(
            "{prefix}: the lift started {} world relights, {beyond} more than the digs",
            lift.world_relights()
        );
    }
    if failures > 0 {
        eprintln!("{prefix}: {failures} sub-grid solves and relights missed the gate");
    }
    metrics.insert(
        format!("{prefix}.gate_failures"),
        (failures + beyond) as f64,
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
///
/// `gate_failures` and `errors` take the worst run: a scenario's promise must
/// hold in every run, and the median would read 0 for a promise broken in
/// fewer than half.
fn run_repeatedly(runs: usize, scenario: impl Fn() -> Metrics) -> Metrics {
    let samples: Vec<Metrics> = (0..runs).map(|_| scenario()).collect();
    combine(&samples, |key, values| {
        match key.rsplit('.').next().unwrap() {
            "time_ms" | "near_time_ms" | "settled_ms" | "relit_ms" => {
                percentile(values, 0.0)
            }
            "chunks_per_s" | "gate_failures" | "errors" => percentile(values, 1.0),
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
    #[cfg(feature = "dislodged_fragments")]
    step("fragments", FRAGMENT_RUNS, &scenario_fragments);
    #[cfg(feature = "sub_grids")]
    step("vehicle", RUNS, &scenario_vehicle);
    #[cfg(feature = "sub_grids")]
    step("lift_burst", RUNS, &scenario_lift_burst);
    #[cfg(all(feature = "light_propagation", feature = "sub_grids"))]
    step("vehicle_lit", RUNS, &scenario_vehicle_lit);
    #[cfg(all(feature = "light_propagation", feature = "sub_grids"))]
    step("lift_burst_lit", RUNS, &scenario_lift_burst_lit);
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
        || key.starts_with("fragments")
        || key.starts_with("vehicle")
        || key.starts_with("lift_burst")
        || key.starts_with("vehicle_lit")
        || key.starts_with("lift_burst_lit")
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

/// The relative change from `b` to `a`, with no change between two zeros.
fn relative_change(b: f64, a: f64) -> f64 {
    match (b, a) {
        (0.0, 0.0) => 0.0,
        (0.0, _) => f64::INFINITY,
        _ => (a - b) / b,
    }
}

/// Prints every metric of either side, the on gate and the ratio lines, and
/// returns whether no gated metric regressed.
///
/// A metric on one side only prints as `new` (after only) or `gone` (before
/// only) and gates nothing, so a scenario one build lacks still shows.
fn compare(before: &Metrics, after: &Metrics) -> bool {
    let mut ok = true;
    println!(
        "{:<38} {:>12} {:>12} {:>9}  gate",
        "metric", "before", "after", "change"
    );
    let keys: BTreeSet<&String> = before.keys().chain(after.keys()).collect();
    for key in keys {
        let (b, a) = match (before.get(key), after.get(key)) {
            (Some(b), Some(a)) => (b, a),
            (None, Some(a)) => {
                println!("{key:<38} {:>12} {a:>12.3} {:>9}  report", "-", "new");
                continue;
            }
            (Some(b), None) => {
                println!("{key:<38} {b:>12.3} {:>12} {:>9}  report", "-", "gone");
                continue;
            }
            (None, None) => unreachable!(),
        };
        let change = relative_change(*b, *a);
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
    for (top, bottom) in RATIO_LINES {
        let ratio = |metrics: &Metrics| Some(metrics.get(top)? / metrics.get(bottom)?);
        let (b, a) = (ratio(before), ratio(after));
        let cell =
            |ratio: Option<f64>| ratio.map_or("-".to_string(), |r| format!("{r:.3}"));
        let change = match (b, a) {
            (None, None) => continue,
            (Some(b), Some(a)) => format!("{:+.1}%", relative_change(b, a) * 100.0),
            _ => String::new(),
        };
        println!(
            "{:<38} {:>12} {:>12} {change:>9}  ratio report",
            format!("{top} / {bottom}"),
            cell(b),
            cell(a),
        );
    }
    ok
}

/// Ratio lines: two metrics of one build, as (numerator, denominator).
/// `compare` prints the ratio within the before build and within the after
/// build, and the change between them, and gates nothing.
///
/// A side lacking either metric prints `-`, so a ratio between scenarios only
/// the after build runs still shows. A pair neither side has prints nothing.
const RATIO_LINES: [(&str, &str); 4] = [
    (
        "fragments.digs.p50_ms",
        "fragments.digs_not_opted_in.p50_ms",
    ),
    (
        "lift_burst_lit.lift.frames_to_settled",
        "lift_burst_lit.digs.frames_to_settled",
    ),
    (
        "lift_burst_lit.lift.world_relights",
        "lift_burst_lit.digs.world_relights",
    ),
    (
        "lift_burst_lit.lift.max_main_thread_us",
        "lift_burst_lit.digs.max_main_thread_us",
    ),
];

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

/// Runs `binary` as `run <label>`, passing `only` on as its scenario list.
fn run_binary(binary: &Path, label: &str, only: Option<&str>) {
    eprintln!("== {} run {label}", binary.display());
    let status = Command::new(binary)
        .args(["run", label])
        .args(only)
        .status()
        .unwrap();
    assert!(status.success(), "{} failed: {status}", binary.display());
}

/// Runs the saved `baseline` binary and this build alternately, then compares them.
/// The order flips every round (ABBAAB…) so drift over the session hits both sides
/// alike.
///
/// With `only`, a comma-separated list, both builds run only the listed
/// scenarios. A scenario the baseline doesn't know records nothing there and
/// prints as `new`.
fn ab(baseline: &str, only: Option<&str>) -> bool {
    let baseline = binary_path(baseline);
    let this = std::env::current_exe().unwrap();
    let mut before = Vec::new();
    let mut after = Vec::new();
    for round in 1..=AB_ROUNDS {
        let before_label = format!("ab-before-{round}");
        let after_label = format!("ab-after-{round}");
        if round % 2 == 1 {
            run_binary(&baseline, &before_label, only);
            run_binary(&this, &after_label, only);
        } else {
            run_binary(&this, &after_label, only);
            run_binary(&baseline, &before_label, only);
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
        ["ab", baseline, rest @ ..] if rest.len() <= 1 => {
            if !ab(baseline, rest.first().copied()) {
                std::process::exit(1);
            }
        }
        [] => {}
        _ => {
            eprintln!(
                "usage: chunk_lifecycle run <label> [<scenario>,…] | compare <before> <after> \
                 | save-binary <name> | ab <baseline> [<scenario>,…] | profile <scenario>\n\
                 \n\
                 `ab` with a scenario list runs only those scenarios in both builds, as a quick \
                 check; the run without a list is the binding one. `compare` and `ab` print a \
                 metric found in one build only as `new` (after only) or `gone` (before only), \
                 and print the ratio lines in `RATIO_LINES`; neither kind gates anything."
            );
            std::process::exit(2);
        }
    }
}
