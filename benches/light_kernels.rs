//! Light propagation's micro-benchmarks: the kernel, filling a slot and the
//! emissive scan.
//!
//! These guard hot paths against innocent changes that make them much slower;
//! the kernel once went from 27 to 49 µs on one. They time the crate's code
//! through its hidden `light_bench` module.
//!
//! Record a run, compare runs, and alternate two builds as `chunk_lifecycle`
//! does, building both sides with one codegen unit:
//!     CARGO_PROFILE_BENCH_CODEGEN_UNITS=1 cargo bench --features light_propagation \
//!         --bench light_kernels -- run <label>
//!     … -- compare <before>[,<before>…] <after>[,<after>…]
//!     … -- save-binary <name>     # on the baseline commit
//!     … -- ab <name>              # on the candidate
//!     … -- seed                   # seeding sky fill alone
//!     … -- field                  # sky fill's field alone, both models
//!
//! The kernel runs on boxes cut from the light reference scene: caves, shafts,
//! a sealed room and a floating island, with emissive voxels scattered on
//! exposed floors. It spreads voxel light from those, and sky light from the
//! voxels with direct sky under the default sky clearance, at reaches that
//! take one-`u64` rows and two-`u64` rows, in both shapes. Filling a slot
//! turns the kernel's result on those boxes into bytes and copies six
//! neighbours' border planes, at both slot formats. The emissive scan runs on a chunk of layered
//! terrain and on
//! chunks of random materials, with emissive lists of several lengths.
//!
//! Sky fill's field is built under both models over the floating island, the
//! mesa and the ground around them, and worked out again after an edit there.
//! Sky openness is held against the 46–59 µs of CPU per shaded position its
//! probe took.

use std::collections::BTreeMap;
use std::hint::black_box;
use std::path::PathBuf;
use std::process::Command;
use std::sync::Arc;
use std::time::Instant;

use bevy::prelude::*;
use bevy_voxel_world::light_bench::{
    emissive_scan, FillField, KernelBox, SeedBox, SlotBox,
};
use bevy_voxel_world::prelude::{LightShape, WorldVoxel};

const AB_ROUNDS: usize = 3;
const REPEATS: usize = 5;

#[path = "common/light_scene.rs"]
mod light_scene;

use light_scene::{Scene, SCENE_TOP};

/// The scene's voxels, emissive voxels and voxels with direct sky around a
/// chunk, out to 32 voxels past its faces, enough for any reach the kernel
/// takes.
struct Dump {
    open: Vec<bool>,
    lamps: Vec<IVec3>,
    direct: Vec<bool>,
}

const DUMP_REACH: i32 = 32;
const DUMP_SIDE: i32 = 32 + 2 * DUMP_REACH;

/// How many chunks above a chunk a column must be clear in to have direct
/// sky, the default sky clearance.
const SKY_CLEARANCE: i32 = 16;

impl Dump {
    fn of(scene: &mut Scene, chunk: IVec3) -> Self {
        let origin = chunk * 32 - IVec3::splat(DUMP_REACH);
        let count = (DUMP_SIDE * DUMP_SIDE * DUMP_SIDE) as usize;
        let mut open = Vec::with_capacity(count);
        let mut lamps = Vec::new();
        for z in 0..DUMP_SIDE {
            for y in 0..DUMP_SIDE {
                for x in 0..DUMP_SIDE {
                    let local = IVec3::new(x, y, z);
                    let is_open = scene.open(origin + local);
                    if !is_open && scene.emissive(origin + local) {
                        lamps.push(local);
                    }
                    open.push(is_open);
                }
            }
        }
        // A voxel has direct sky when its column holds no solid up to the
        // top of the chunks its sky clearance covers.
        let mut direct = vec![false; count];
        for z in 0..DUMP_SIDE {
            for x in 0..DUMP_SIDE {
                let mut solid_from = i32::MAX;
                for y in (origin.y..SCENE_TOP.max(origin.y + DUMP_SIDE)).rev() {
                    let p = IVec3::new(origin.x + x, y, origin.z + z);
                    let local = p - origin;
                    let inside = local.y < DUMP_SIDE;
                    let index =
                        (local.x + local.y * DUMP_SIDE + local.z * DUMP_SIDE * DUMP_SIDE)
                            as usize;
                    let is_open = if inside { open[index] } else { scene.open(p) };
                    if !is_open {
                        solid_from = y;
                    } else if inside {
                        let top = (y.div_euclid(32) + 1 + SKY_CLEARANCE) * 32 - 1;
                        direct[index] = solid_from > top;
                    }
                }
            }
        }
        Self {
            open,
            lamps,
            direct,
        }
    }

    fn index(p: IVec3) -> usize {
        (p.x + p.y * DUMP_SIDE + p.z * DUMP_SIDE * DUMP_SIDE) as usize
    }

    /// The kernel box of sky light of `reach` around the chunk.
    fn sky_box(&self, reach: u32) -> KernelBox {
        let offset = IVec3::splat(DUMP_REACH - reach as i32);
        KernelBox::with_direct_sky(
            reach,
            |p| self.open[Self::index(p + offset)],
            |p| self.direct[Self::index(p + offset)],
        )
    }

    /// The kernel box of `reach` around the chunk.
    fn kernel_box(&self, reach: u32) -> KernelBox {
        let offset = IVec3::splat(DUMP_REACH - reach as i32);
        let side = 32 + 2 * reach as i32;
        let inside =
            |p: IVec3| p.cmpge(IVec3::ZERO).all() && p.cmplt(IVec3::splat(side)).all();
        let lamps: Vec<(IVec3, u8)> = self
            .lamps
            .iter()
            .map(|lamp| (*lamp - offset, 15))
            .filter(|(lamp, _)| inside(*lamp))
            .collect();
        KernelBox::new(
            reach,
            |p| self.open_at(p - IVec3::splat(reach as i32)),
            &lamps,
        )
    }

    /// Whether the voxel at `p`, relative to the chunk, is open. Voxels past
    /// the dump are open.
    fn open_at(&self, p: IVec3) -> bool {
        let p = p + IVec3::splat(DUMP_REACH);
        if p.cmplt(IVec3::ZERO).any() || p.cmpge(IVec3::splat(DUMP_SIDE)).any() {
            return true;
        }
        self.open[(p.x + p.y * DUMP_SIDE + p.z * DUMP_SIDE * DUMP_SIDE) as usize]
    }
}

/// Chunks of the scene with emissive voxels in reach: around the deep
/// cavern's torches, the torch room, the reference cavern and the open ground.
fn dumps() -> Vec<Dump> {
    let mut scene = Scene::new();
    let chunks = [
        IVec3::new(2, -7, 0),
        IVec3::new(-3, -7, 0),
        IVec3::new(0, -7, 2),
        IVec3::new(0, -7, -3),
        IVec3::new(9, -2, -11),
        IVec3::new(10, -2, -10),
        IVec3::new(-10, -1, 0),
        IVec3::new(-11, -1, 1),
        IVec3::new(4, 0, 4),
        IVec3::new(-4, 1, -4),
        IVec3::new(6, 0, -2),
        IVec3::new(-2, 1, 11),
    ];
    chunks
        .iter()
        .map(|chunk| Dump::of(&mut scene, *chunk))
        .filter(|dump| !dump.lamps.is_empty())
        .collect()
}

// --- Scenarios ---------------------------------------------------------------

type Metrics = BTreeMap<String, f64>;

/// The best of `REPEATS` timings of `work`, in microseconds.
fn best_us(mut work: impl FnMut()) -> f64 {
    (0..REPEATS)
        .map(|_| {
            let start = Instant::now();
            work();
            start.elapsed().as_secs_f64() * 1e6
        })
        .fold(f64::MAX, f64::min)
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

/// Cuts a kernel box of a reach from a dump.
type MakeBox = fn(&Dump, u32) -> KernelBox;

fn kernel_metrics(dumps: &[Dump], metrics: &mut Metrics) {
    // Voxel light at the default reach and the most a one-`u64` row holds,
    // and at 30 on two-`u64` rows; sky light at the default reach of 30 and
    // at 16.
    let runs: [(&str, u32, MakeBox); 5] = [
        ("kernel", 14, Dump::kernel_box),
        ("kernel", 16, Dump::kernel_box),
        ("kernel", 30, Dump::kernel_box),
        ("sky", 16, Dump::sky_box),
        ("sky", 30, Dump::sky_box),
    ];
    for (kind, reach, make) in runs {
        let mut boxes: Vec<KernelBox> =
            dumps.iter().map(|dump| make(dump, reach)).collect();
        for (shape, name) in [
            (LightShape::Diamond, "diamond"),
            (LightShape::Round, "round"),
        ] {
            let mut times: Vec<f64> = boxes
                .iter_mut()
                .map(|kernel_box| {
                    best_us(|| {
                        kernel_box.spread(shape);
                        black_box(&*kernel_box);
                    })
                })
                .collect();
            metrics.insert(
                format!("{kind}.reach{reach}.{name}.p50_us"),
                median(&mut times),
            );
        }
    }
}

const SIDE: usize = 34;

/// A chunk of voxels with its padding: `terrain` is ground in layers of one
/// material under a wavy surface, and otherwise every voxel is solid in a
/// random one of `materials`. Material 0 is the one emissive material.
fn scan_chunk(terrain: bool, materials: u8) -> Arc<[WorldVoxel<u8>]> {
    let mut seed = 0x9E37_79B9u32;
    let mut random = move || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        seed
    };
    (0..SIDE * SIDE * SIDE)
        .map(|i| {
            let (x, y, z) = (i % SIDE, i / SIDE % SIDE, i / (SIDE * SIDE));
            if terrain {
                let height =
                    17.0 + 6.0 * ((x as f32 * 0.3).sin() + (z as f32 * 0.2).cos());
                if (y as f32) < height {
                    WorldVoxel::Solid(((y / 6) as u8).min(materials - 1))
                } else {
                    WorldVoxel::Air
                }
            } else {
                WorldVoxel::Solid((random() % materials as u32) as u8)
            }
        })
        .collect()
}

fn scan_metrics(metrics: &mut Metrics) {
    for (terrain, materials, name) in [
        (true, 8, "terrain8"),
        (false, 8, "noise8"),
        (false, 64, "noise64"),
    ] {
        let voxels = scan_chunk(terrain, materials);
        for length in [0usize, 1, 4, 16, 64] {
            // The emissive material is last, so a list is searched to its end.
            let list: Vec<(u8, u8)> = (0..length)
                .map(|i| (if i + 1 == length { 0 } else { 200 + i as u8 }, 15))
                .collect();
            let mut times: Vec<f64> = (0..9)
                .map(|_| {
                    best_us(|| {
                        black_box(emissive_scan(&voxels, &list));
                    })
                })
                .collect();
            metrics.insert(
                format!("scan.{name}.list{length}.p50_us"),
                median(&mut times),
            );
        }
    }
}

/// Filling a slot at landing: the solver's bit planes to bytes with the
/// table, and the six shell planes from the neighbours' light, at 16 levels
/// (`slot.*`) and at 256 (`slot256.*`).
fn slot_metrics(dumps: &[Dump], metrics: &mut Metrics) {
    slot_metrics_at(dumps, metrics, "slot", false);
    slot_metrics_at(dumps, metrics, "slot256", true);
}

fn slot_metrics_at(dumps: &[Dump], metrics: &mut Metrics, name: &str, steps_256: bool) {
    let slots: Vec<SlotBox> = dumps
        .iter()
        .map(|dump| {
            let mut kernel_box = dump.kernel_box(14);
            if steps_256 {
                kernel_box = kernel_box.at_256_levels();
            }
            kernel_box.spread(LightShape::Round);
            SlotBox::new(&kernel_box, |p| dump.open_at(p))
        })
        .collect();
    let mut times: Vec<f64> = slots
        .iter()
        .map(|slot| {
            best_us(|| {
                black_box(slot.slot_bytes());
            })
        })
        .collect();
    metrics.insert(format!("{name}.bytes.p50_us"), median(&mut times));
    let mut times: Vec<f64> = slots
        .iter()
        .map(|slot| {
            let mut bytes = slot.slot_bytes();
            best_us(|| {
                slot.shell_planes(&mut bytes);
                black_box(&bytes);
            })
        })
        .collect();
    metrics.insert(format!("{name}.shell_planes.p50_us"), median(&mut times));
}

/// The chunk positions sky fill's field learns, from the first corner to the
/// second, both included: the floating island, the mesa and the slab beside
/// it, and the ground under them.
const FILL_FROM: IVec3 = IVec3::new(-16, -1, 5);
const FILL_TO: IVec3 = IVec3::new(2, 7, 17);

/// Chunks lit by sky fill: on and over the ground under the floating island,
/// under the slab, and against the island's underside.
const FILL_CHUNKS: [IVec3; 10] = [
    IVec3::new(-11, 0, 11),
    IVec3::new(-11, 1, 11),
    IVec3::new(-10, 1, 10),
    IVec3::new(-12, 1, 12),
    IVec3::new(-10, 0, 12),
    IVec3::new(-11, 4, 11),
    IVec3::new(0, 1, 11),
    IVec3::new(-1, 2, 11),
    IVec3::new(0, 2, 12),
    IVec3::new(1, 1, 12),
];

/// Each learned position with its cell masks, `open` and `direct`, worked
/// out from the scene's voxels.
fn fill_masks(scene: &mut Scene) -> Vec<(IVec3, u64, u64)> {
    let low = FILL_FROM * 32;
    let high = (FILL_TO + IVec3::ONE) * 32;
    let width = (high.x - low.x) as usize;
    // The highest solid voxel in each column, from the top of the scene down
    // to the bottom of the learned positions.
    let mut top = vec![i32::MIN; width * (high.z - low.z) as usize];
    for z in low.z..high.z {
        for x in low.x..high.x {
            let column = (x - low.x) as usize + (z - low.z) as usize * width;
            top[column] = (low.y..SCENE_TOP)
                .rev()
                .find(|&y| !scene.open(IVec3::new(x, y, z)))
                .unwrap_or(i32::MIN);
        }
    }
    let mut masks = Vec::new();
    for cz in FILL_FROM.z..=FILL_TO.z {
        for cy in FILL_FROM.y..=FILL_TO.y {
            for cx in FILL_FROM.x..=FILL_TO.x {
                let origin = IVec3::new(cx, cy, cz) * 32;
                let (mut open, mut direct) = (0u64, 0u64);
                for cell in 0..64 {
                    let corner =
                        origin + IVec3::new(cell & 3, cell >> 2 & 3, cell >> 4) * 8;
                    let highest = (0..64)
                        .map(|i| {
                            let (x, z) = (corner.x + i % 8, corner.z + i / 8);
                            top[(x - low.x) as usize + (z - low.z) as usize * width]
                        })
                        .max()
                        .unwrap();
                    let is_open = highest < corner.y
                        || (0..512).all(|i| {
                            scene.open(corner + IVec3::new(i % 8, i / 8 % 8, i / 64))
                        });
                    if is_open {
                        open |= 1 << cell;
                        if highest < corner.y {
                            direct |= 1 << cell;
                        }
                    }
                }
                masks.push((IVec3::new(cx, cy, cz), open, direct));
            }
        }
    }
    masks
}

/// The field of sky fill's default model over the learned positions.
fn fill_field(scene: &mut Scene) -> FillField {
    let mut field = FillField::default();
    for (position, open, direct) in fill_masks(scene) {
        field.learn(position, open, direct);
    }
    field.settle();
    field
}

/// Updating sky fill's field under each model: learning every position at
/// once, per shaded position (one with an open cell that isn't direct sky),
/// and working out the field again after a cell on the ground under the
/// floating island fills in and opens again, as a voxel edit there would.
fn field_metrics(metrics: &mut Metrics) {
    let masks = fill_masks(&mut Scene::new());
    let shaded = masks
        .iter()
        .filter(|(_, open, direct)| open & !direct != 0)
        .count();
    metrics.insert("field.shaded_positions".into(), shaded as f64);
    // The first chunk lit by fill with a shaded cell.
    let (edited, open, direct) = *FILL_CHUNKS
        .iter()
        .filter_map(|chunk| masks.iter().find(|(position, ..)| position == chunk))
        .find(|(_, open, direct)| open & !direct != 0)
        .unwrap();
    // The lowest open cell that isn't direct sky.
    let cell = (open & !direct).trailing_zeros();
    for name in ["paths", "openness"] {
        let model = || match name {
            "paths" => FillField::default(),
            _ => FillField::openness(),
        };
        let mut build = f64::INFINITY;
        let mut edit = f64::INFINITY;
        for _ in 0..REPEATS {
            let mut field = model();
            for (position, open, direct) in &masks {
                field.learn(*position, *open, *direct);
            }
            let start = Instant::now();
            black_box(field.settle());
            build = build.min(start.elapsed().as_secs_f64() * 1e6);

            let start = Instant::now();
            field.learn(edited, open & !(1 << cell), direct);
            black_box(field.settle());
            field.learn(edited, open, direct);
            black_box(field.settle());
            edit = edit.min(start.elapsed().as_secs_f64() * 1e6 / 2.0);
        }
        metrics.insert(format!("field.{name}.build_ms"), build / 1000.0);
        metrics.insert(
            format!("field.{name}.us_per_shaded_position"),
            build / shaded as f64,
        );
        metrics.insert(format!("field.{name}.edit_us"), edit);
    }
}

/// Seeding sky fill with the default model, on chunks it lights, at the
/// default sky reach of 30 and at 16. The gate is 100 µs per chunk at 30.
fn seed_metrics(metrics: &mut Metrics) {
    let mut scene = Scene::new();
    let field = fill_field(&mut scene);
    for reach in [16, 30] {
        let side = 32 + 2 * reach as i32;
        let boxes: Vec<SeedBox> = FILL_CHUNKS
            .iter()
            .map(|chunk| {
                let origin = *chunk * 32 - IVec3::splat(reach as i32);
                let mut open = vec![false; (side * side * side) as usize];
                for (i, open) in open.iter_mut().enumerate() {
                    let i = i as i32;
                    let p = IVec3::new(i % side, i / side % side, i / (side * side));
                    *open = scene.open(origin + p);
                }
                SeedBox::new(&field, *chunk, reach, |p| {
                    open[(p.x + p.y * side + p.z * side * side) as usize]
                })
            })
            .filter(|seed_box| seed_box.seed() > 0)
            .collect();
        metrics.insert(format!("seed.reach{reach}.chunks"), boxes.len() as f64);
        let mut times: Vec<f64> = boxes
            .iter()
            .map(|seed_box| {
                best_us(|| {
                    black_box(seed_box.seed());
                })
            })
            .collect();
        let most = times.iter().copied().fold(0.0, f64::max);
        metrics.insert(format!("seed.reach{reach}.p50_us"), median(&mut times));
        metrics.insert(format!("seed.reach{reach}.max_us"), most);
    }
}

fn run_all() -> Metrics {
    let mut metrics = Metrics::new();
    eprintln!("seeding sky fill...");
    seed_metrics(&mut metrics);
    eprintln!("sky fill's field...");
    field_metrics(&mut metrics);
    eprintln!("cutting boxes from the reference scene...");
    let dumps = dumps();
    metrics.insert("kernel.boxes".into(), dumps.len() as f64);
    eprintln!("kernel on {} boxes...", dumps.len());
    kernel_metrics(&dumps, &mut metrics);
    eprintln!("slots...");
    slot_metrics(&dumps, &mut metrics);
    eprintln!("emissive scan...");
    scan_metrics(&mut metrics);
    metrics
}

// --- Gating and storage, as in chunk_lifecycle -----------------------------

/// A timing more than this much slower is a regression.
const GATE: f64 = 0.15;

fn compare(before: &Metrics, after: &Metrics) -> bool {
    let mut ok = true;
    println!(
        "{:<38} {:>10} {:>10} {:>9}  gate",
        "metric", "before", "after", "change"
    );
    for (key, b) in before {
        let Some(a) = after.get(key) else { continue };
        let change = if *b == 0.0 { 0.0 } else { (a - b) / b };
        let verdict = if !key.ends_with("_us") {
            "report"
        } else if change > GATE {
            "REGRESSED"
        } else {
            "ok"
        };
        ok &= verdict != "REGRESSED";
        println!(
            "{key:<38} {b:>10.2} {a:>10.2} {:>+8.1}%  {verdict}",
            change * 100.0
        );
    }
    ok
}

fn bench_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/bench/light_kernels")
}

fn save(label: &str, metrics: &Metrics) {
    let path = bench_dir().join(format!("{label}.txt"));
    std::fs::create_dir_all(bench_dir()).unwrap();
    let body: String = metrics
        .iter()
        .map(|(key, value)| format!("{key} {value}\n"))
        .collect();
    std::fs::write(&path, body).unwrap();
    eprintln!("wrote {}", path.display());
}

fn load(label: &str) -> Metrics {
    let path = bench_dir().join(format!("{label}.txt"));
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("can't read {}: {e}", path.display()))
        .lines()
        .filter_map(|line| {
            let (key, value) = line.split_once(' ')?;
            Some((key.to_string(), value.parse().ok()?))
        })
        .collect()
}

/// The median of each metric over a comma-separated list of labels.
fn load_all(labels: &str) -> Metrics {
    let samples: Vec<Metrics> = labels.split(',').map(load).collect();
    samples[0]
        .keys()
        .map(|key| {
            let mut values: Vec<f64> = samples
                .iter()
                .filter_map(|sample| sample.get(key).copied())
                .collect();
            (key.clone(), median(&mut values))
        })
        .collect()
}

fn binary_path(name: &str) -> PathBuf {
    bench_dir().join(format!("{name}-bin"))
}

fn run_binary(binary: &PathBuf, label: &str) {
    let status = Command::new(binary).args(["run", label]).status().unwrap();
    assert!(status.success(), "{} failed: {status}", binary.display());
}

/// Runs the saved binary and this build alternately, then compares them.
fn ab(baseline: &str) -> bool {
    let baseline = binary_path(baseline);
    let this = std::env::current_exe().unwrap();
    let (mut before, mut after) = (Vec::new(), Vec::new());
    for round in 1..=AB_ROUNDS {
        let (b, a) = (format!("ab-before-{round}"), format!("ab-after-{round}"));
        if round % 2 == 1 {
            run_binary(&baseline, &b);
            run_binary(&this, &a);
        } else {
            run_binary(&this, &a);
            run_binary(&baseline, &b);
        }
        before.push(b);
        after.push(a);
    }
    compare(&load_all(&before.join(",")), &load_all(&after.join(",")))
}

fn main() {
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
        ["run", label] => {
            let metrics = run_all();
            for (key, value) in &metrics {
                println!("{key:<38} {value:>10.2}");
            }
            save(label, &metrics);
        }
        ["seed"] => {
            let mut metrics = Metrics::new();
            seed_metrics(&mut metrics);
            for (key, value) in &metrics {
                println!("{key:<38} {value:>10.2}");
            }
        }
        ["field"] => {
            let mut metrics = Metrics::new();
            field_metrics(&mut metrics);
            for (key, value) in &metrics {
                println!("{key:<38} {value:>10.2}");
            }
        }
        ["compare", before, after] => {
            if !compare(&load_all(before), &load_all(after)) {
                std::process::exit(1);
            }
        }
        ["save-binary", name] => {
            std::fs::create_dir_all(bench_dir()).unwrap();
            std::fs::copy(std::env::current_exe().unwrap(), binary_path(name)).unwrap();
        }
        ["ab", baseline] => {
            if !ab(baseline) {
                std::process::exit(1);
            }
        }
        [] => {}
        _ => {
            eprintln!(
                "usage: light_kernels run <label> | seed | field \
                 | compare <before> <after> | save-binary <name> | ab <baseline>"
            );
            std::process::exit(2);
        }
    }
}
