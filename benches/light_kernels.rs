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

use std::collections::{BTreeMap, HashMap};
use std::hint::black_box;
use std::path::PathBuf;
use std::process::Command;
use std::sync::Arc;
use std::time::Instant;

use bevy::prelude::*;
use bevy_voxel_world::light_bench::{emissive_scan, KernelBox, SlotBox};
use bevy_voxel_world::prelude::{LightShape, WorldVoxel};
use noise::{NoiseFn, Perlin};

const AB_ROUNDS: usize = 3;
const REPEATS: usize = 5;

// --- The reference scene -----------------------------------------------------

const CAVE_FLOOR: i32 = -480;

/// A box, minimum inclusive and maximum exclusive.
#[derive(Clone, Copy)]
struct Bounds([i32; 3], [i32; 3]);

impl Bounds {
    fn contains(&self, p: IVec3) -> bool {
        (0..3).all(|axis| p[axis] >= self.0[axis] && p[axis] < self.1[axis])
    }
}

struct Ellipsoid([f64; 3], [f64; 3]);

impl Ellipsoid {
    fn contains(&self, p: IVec3) -> bool {
        (0..3)
            .map(|axis| ((p[axis] as f64 + 0.5 - self.0[axis]) / self.1[axis]).powi(2))
            .sum::<f64>()
            < 1.0
    }
}

/// A disc of ground forced to one height: x, z, flat radius, blend radius,
/// height.
type Plateau = (f64, f64, f64, f64, f64);

const PLATEAUS: [Plateau; 6] = [
    (0.5, 0.5, 24.0, 190.0, 160.0),
    (-319.5, 0.5, 20.0, 120.0, 120.0),
    (320.5, 0.5, 64.0, 150.0, 40.0),
    (0.0, 360.0, 90.0, 170.0, 32.0),
    (-340.0, 360.0, 90.0, 170.0, 30.0),
    (-320.0, 128.0, 16.0, 56.0, 48.0),
];
const DEEP_CAVERN: Ellipsoid = Ellipsoid([0.5, -176.0, 0.5], [110.0, 36.0, 110.0]);
const REF_CAVERN: Ellipsoid = Ellipsoid([-319.5, -10.0, 0.5], [70.0, 30.0, 70.0]);
const POCKET: Bounds = Bounds([-326, 40, 122], [-314, 46, 134]);
const TORCH_ROOM: Bounds = Bounds([300, -40, -340], [340, -20, -300]);
const MESA: Bounds = Bounds([-48, 32, 290], [48, 84, 340]);
const SLAB: Bounds = Bounds([-48, 72, 340], [48, 84, 404]);
const ISLAND: Bounds = Bounds([-390, 150, 310], [-290, 176, 410]);
const QUIET: [Bounds; 7] = [
    Bounds([-126, -228, -126], [126, -124, 126]),
    Bounds([-16, -140, -16], [16, 168, 16]),
    Bounds([-406, -56, -86], [-234, 36, 86]),
    Bounds([-336, 20, -16], [-304, 128, 16]),
    Bounds([-342, 24, 112], [-298, 60, 144]),
    Bounds([256, -304, -64], [384, 48, 64]),
    Bounds([284, -56, -356], [356, -4, -284]),
];
const TORCH_COLUMNS: [(i32, i32); 5] =
    [(70, 0), (-70, 0), (0, 70), (0, -70), (320, -320)];

/// The light reference scene, with an emissive voxel on one in 32 exposed
/// floor voxels.
struct Scene {
    hills: Perlin,
    region: Perlin,
    big: Perlin,
    worm_a: Perlin,
    worm_b: Perlin,
    surfaces: HashMap<(i32, i32), i32>,
}

impl Scene {
    fn new() -> Self {
        Self {
            hills: Perlin::new(11),
            region: Perlin::new(19),
            big: Perlin::new(23),
            worm_a: Perlin::new(37),
            worm_b: Perlin::new(41),
            surfaces: HashMap::new(),
        }
    }

    fn surface(&mut self, x: i32, z: i32) -> i32 {
        if let Some(height) = self.surfaces.get(&(x, z)) {
            return *height;
        }
        let (fx, fz) = (x as f64, z as f64);
        let mut height = 32.0
            + self.hills.get([fx / 160.0, fz / 160.0]) * 18.0
            + self.hills.get([fx / 47.0, fz / 47.0]) * 4.0;
        for (px, pz, flat, blend, level) in PLATEAUS {
            let t = (((fx - px).hypot(fz - pz) - flat) / (blend - flat)).clamp(0.0, 1.0);
            let t = t * t * (3.0 - 2.0 * t);
            height = level + (height - level) * t;
        }
        let height = height.floor() as i32;
        self.surfaces.insert((x, z), height);
        height
    }

    fn noise_cave(&self, p: IVec3) -> bool {
        let (x, y, z) = (p.x as f64, p.y as f64, p.z as f64);
        if self.region.get([x / 220.0, y / 160.0, z / 220.0]) < -0.05 {
            return false;
        }
        if self.big.get([x / 72.0, y / 48.0, z / 72.0]) > 0.42 {
            return true;
        }
        let at = [x / 36.0, y / 28.0, z / 36.0];
        self.worm_a.get(at).abs() < 0.04 && self.worm_b.get(at).abs() < 0.04
    }

    fn placed_hollow(&self, p: IVec3) -> bool {
        (p.x == 0 && p.z == 0 && p.y >= -141)
            || (p.x == -320 && p.z == 0 && p.y >= 19)
            || (p.x == 320 && p.z == 0 && p.y >= 32)
            || DEEP_CAVERN.contains(p)
            || REF_CAVERN.contains(p)
            || POCKET.contains(p)
            || TORCH_ROOM.contains(p)
            || (p.y >= -288 && p.y < 32 && (p.x - 320).pow(2) + p.z.pow(2) < 48 * 48)
    }

    fn floating_solid(&self, p: IVec3) -> bool {
        if MESA.contains(p) || SLAB.contains(p) {
            return true;
        }
        ISLAND.contains(p)
            && p.y as f64
                >= 160.0 + 8.0 * self.hills.get([p.x as f64 / 16.0, p.z as f64 / 16.0])
    }

    fn open(&mut self, p: IVec3) -> bool {
        let surface = self.surface(p.x, p.z);
        if p.y >= surface {
            return !self.floating_solid(p);
        }
        self.placed_hollow(p)
            || (p.y >= CAVE_FLOOR
                && !QUIET.iter().any(|quiet| quiet.contains(p))
                && self.noise_cave(p))
    }

    /// Whether the voxel at `p`, which must be solid, is emissive.
    fn emissive(&mut self, p: IVec3) -> bool {
        if !self.open(p + IVec3::Y) {
            return false;
        }
        TORCH_COLUMNS.contains(&(p.x, p.z))
            || (hash3(p).is_multiple_of(32)
                && !QUIET.iter().any(|quiet| quiet.contains(p)))
    }
}

fn hash3(p: IVec3) -> u32 {
    let mut h = (p.x as u32).wrapping_mul(0x8da6_b343)
        ^ (p.y as u32).wrapping_mul(0xd816_3841)
        ^ (p.z as u32).wrapping_mul(0xcb1a_b31f);
    h ^= h >> 13;
    h = h.wrapping_mul(0x5bd1_e995);
    h ^ (h >> 15)
}

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
/// Above this height the scene holds no solid voxel.
const SCENE_TOP: i32 = 200;

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

fn run_all() -> Metrics {
    let mut metrics = Metrics::new();
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
                "usage: light_kernels run <label> | compare <before> <after> \
                 | save-binary <name> | ab <baseline>"
            );
            std::process::exit(2);
        }
    }
}
