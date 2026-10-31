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
//! times each, then compares them.
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
/// Frames without any lifecycle change before stream-in counts as converged.
const SETTLE_FRAMES: u32 = 30;
const MAX_FRAMES: u32 = 5000;

/// For scenarios whose run-to-run spread sits close to the gates.
const MANY_RUNS: usize = 9;
const RUNS: usize = 5;
const IDLE_FRAMES: usize = 2000;
const FLY_FRAMES: usize = 1200;
/// Voxels per second of wall-clock time; just under what stream-in sustains.
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
        tracker.ready.insert(msg.chunk_key, msg.entity);
        changed = true;
    }
    for msg in despawned.read() {
        tracker.ready.remove(&msg.chunk_key);
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

fn spawn_camera<C: VoxelWorldConfig>(mut commands: Commands) {
    // Positive coordinates throughout, so chunk rounding conventions agree.
    let transform = Transform::from_translation(CAMERA_START)
        .looking_at(Vec3::new(116.0, 0.0, 116.0), Vec3::Y);
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
    let r = SPAWNING_DISTANCE;
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

struct StreamIn {
    time_ms: f64,
    frames: u32,
    /// Duration of each frame up to the last lifecycle change.
    frame_times: Vec<f64>,
}

/// Runs frames until the lifecycle has been quiet for `SETTLE_FRAMES`.
fn stream_in<C: VoxelWorldConfig>(app: &mut App) -> StreamIn {
    let start = Instant::now();
    let mut frame_times = Vec::new();
    let mut last_count = chunk_count::<C>(app);
    let mut last_change = (0, 0.0);
    for frame in 1..=MAX_FRAMES {
        let frame_start = Instant::now();
        app.update();
        frame_times.push(frame_start.elapsed().as_secs_f64() * 1000.0);
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
    // barely moves it; these see every one of them.
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

/// Every chunk around the camera becomes unwanted, and a whole new area wanted, on
/// the same frame.
fn scenario_teleport() -> Metrics {
    let mut app = build_app(SineWorld::default());
    stream_in::<SineWorld>(&mut app);
    teleport_camera::<SineWorld>(&mut app, TELEPORT);
    let result = stream_in::<SineWorld>(&mut app);
    let mut metrics = Metrics::new();
    metrics.insert("teleport.time_ms".into(), result.time_ms);
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
            .get_chunk_data(chunk_pos)
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
            "time_ms" => percentile(values, 0.0),
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
    let mut step = |name: &str, runs: usize, scenario: &dyn Fn() -> Metrics| {
        if only.is_some_and(|only| !only.split(',').any(|s| s == name)) {
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
    metrics
}

// --- Gating ----------------------------------------------------------------

enum Gate {
    /// Higher is worse; regression when the relative increase exceeds the limit.
    Lower(f64),
    /// Like `Lower`, for small counts: also needs an increase of more than `min`.
    LowerCount {
        limit: f64,
        min: f64,
    },
    /// Lower is worse; regression when the relative decrease exceeds the limit.
    Higher(f64),
    ReportOnly,
}

fn gate(key: &str) -> Gate {
    if key.starts_with("in_view_fly_through.") {
        return Gate::ReportOnly;
    }
    match key.rsplit('.').next().unwrap() {
        "p50_ms" | "time_ms" => Gate::Lower(0.05),
        "p99_ms" | "time_to_publish_ms" => Gate::Lower(0.10),
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
    ok
}

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
