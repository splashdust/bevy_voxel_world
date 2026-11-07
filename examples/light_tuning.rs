//! Every light setting on a dial: turn one, see what it changes and what it
//! costs, and print the config as Rust to paste into your own world.
//!
//! Up and Down pick a setting and Left and Right turn it. Each dial changes
//! one value. Beside each one the panel says when a change shows (on the
//! next frame, after a relight, with a new light texture, as a limit, or
//! only at plugin build) and what it costs, and under the list how long the
//! last change took to settle. P prints the config as Rust, with only what
//! differs from the default, ready to paste into a world's
//! `VoxelWorldConfig`. Keys 1 to 5 move the camera, and the right mouse
//! button orbits it.
//!
//! The ball walking in and out of the tunnel isn't a chunk, so propagated
//! light doesn't shade it. It dims itself from `VoxelWorld::light_at` at its
//! position, through the look's `LevelCurve::at`, so it goes dark in the
//! tunnel as the rock around it does instead of standing out fully lit.
//!
//! Run with `cargo run --example light_tuning --features light_propagation`.
//! On WASM the world starts with a sky reach of 16, as a WASM application
//! should.
//!
//! With the argument `measure`, it renders offscreen instead, turns a few
//! dials one step and back once the world has settled, and prints how many
//! frames and milliseconds each change took to settle, at 60 frames a second.

use bevy::{
    camera::RenderTarget, light::CascadeShadowConfigBuilder, platform::time::Instant,
    prelude::*, render::render_resource::TextureFormat, window::ExitCondition,
};
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
use bevy_voxel_world::prelude::*;
use noise::{NoiseFn, Perlin};

/// Writes a line of output: to standard output on native, and to the
/// browser's console on WASM.
macro_rules! report {
    ($($line:tt)*) => {{
        #[cfg(not(target_arch = "wasm32"))]
        println!($($line)*);
        #[cfg(target_arch = "wasm32")]
        info!($($line)*);
    }};
}

// --- The scene ---------------------------------------------------------------

const GRASS: u8 = 0;
const DIRT: u8 = 1;
const STONE: u8 = 2;
const LAMP: u8 = 3;
const DIM_LAMP: u8 = 4;

/// The lowest open voxel of the flat ground around the hill.
const GROUND: i32 = 24;
const HILL_TOP: f64 = 72.0;
const HILL_RADIUS: f64 = 80.0;

/// A box, minimum inclusive and maximum exclusive.
struct Bounds {
    min: IVec3,
    max: IVec3,
}

impl Bounds {
    const fn new(min: [i32; 3], max: [i32; 3]) -> Self {
        Self {
            min: IVec3::from_array(min),
            max: IVec3::from_array(max),
        }
    }

    fn contains(&self, position: IVec3) -> bool {
        position.cmpge(self.min).all() && position.cmplt(self.max).all()
    }
}

/// A tunnel through the hill, long enough that its middle is out of sky
/// light's reach from both ends.
const TUNNEL: Bounds = Bounds::new([-90, GROUND, -4], [90, GROUND + 7, 4]);
/// Lamps in the tunnel's walls.
const LAMP_AT: IVec3 = IVec3::new(20, GROUND + 2, -5);
const DIM_LAMP_AT: IVec3 = IVec3::new(-20, GROUND + 2, 4);

/// An overhang: a slab sticking out of a block.
const BLOCK: Bounds = Bounds::new([-40, -64, 120], [40, 50, 140]);
const SLAB: Bounds = Bounds::new([-40, 44, 90], [40, 50, 120]);

/// A floating island, high enough above the ground that sky light spreading
/// from around it can't reach the middle of the ground below.
const ISLAND_CENTRE: Vec3 = Vec3::new(0.0, 104.0, -160.0);
const ISLAND_RADII: Vec3 = Vec3::new(36.0, 5.0, 36.0);

fn smoothstep(t: f64) -> f64 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// What voxel is where: a pure function of position, as sky light needs.
struct Scene {
    hills: Perlin,
}

impl Scene {
    /// The lowest open voxel of the column at `x`, `z`.
    fn ground(&self, x: i32, z: i32) -> i32 {
        let (fx, fz) = (x as f64, z as f64);
        let distance = fx.hypot(fz);
        let hill = (HILL_TOP - GROUND as f64) * smoothstep(1.0 - distance / HILL_RADIUS);
        let far = smoothstep((distance - 110.0) / 80.0);
        let rolling = self.hills.get([fx / 90.0, fz / 90.0]) * 10.0 * far;
        (GROUND as f64 + hill + rolling).floor() as i32
    }

    fn voxel(&self, position: IVec3) -> WorldVoxel {
        if position == LAMP_AT {
            return WorldVoxel::Solid(LAMP);
        }
        if position == DIM_LAMP_AT {
            return WorldVoxel::Solid(DIM_LAMP);
        }
        let ground = self.ground(position.x, position.z);
        let island = (position.as_vec3() + 0.5 - ISLAND_CENTRE) / ISLAND_RADII;
        let solid = if position.y >= ground {
            island.length_squared() < 1.0
                || BLOCK.contains(position)
                || SLAB.contains(position)
        } else {
            !TUNNEL.contains(position)
        };
        if !solid {
            return WorldVoxel::Air;
        }
        WorldVoxel::Solid(if position.y + 1 >= ground {
            GRASS
        } else if position.y + 4 >= ground {
            DIRT
        } else {
            STONE
        })
    }
}

// --- The dials ---------------------------------------------------------------

/// Which model of sky fill the world uses, if any. Each model keeps its own
/// settings in [`Tuning`], so switching back and forth doesn't lose them.
#[derive(Clone, Copy, PartialEq, Debug)]
enum FillModel {
    Off,
    Paths,
    Openness,
}

/// Every value a dial turns.
#[derive(Clone, PartialEq)]
struct Tuning {
    /// The light settings, apart from `sky_fill`, which comes from the three
    /// fields after it.
    light: LightPropagation,
    fill: FillModel,
    paths: PathFill,
    openness: OpenFill,
    /// The levels the two lamps' materials are declared at.
    lamp: f32,
    dim_lamp: f32,
    /// The app's `GlobalAmbientLight`, which isn't a light setting.
    ambient: f32,
}

impl Default for Tuning {
    fn default() -> Self {
        Self {
            light: LightPropagation {
                sky_reach: if cfg!(target_arch = "wasm32") { 16 } else { 30 },
                ..default()
            },
            fill: FillModel::Paths,
            paths: PathFill::default(),
            openness: OpenFill::default(),
            lamp: 15.0,
            dim_lamp: 8.0,
            ambient: 600.0,
        }
    }
}

impl Tuning {
    fn light(&self) -> LightPropagation {
        LightPropagation {
            sky_fill: match self.fill {
                FillModel::Off => None,
                FillModel::Paths => Some(SkyFill::Paths(self.paths.clone())),
                FillModel::Openness => Some(SkyFill::Openness(self.openness.clone())),
            },
            ..self.light.clone()
        }
    }

    fn emissive_materials(&self) -> Vec<(u8, Emission)> {
        vec![
            (LAMP, Emission::level(self.lamp)),
            (DIM_LAMP, Emission::level(self.dim_lamp)),
        ]
    }
}

/// When a change to a dial's value shows, as each field's documentation says.
#[derive(Clone, Copy)]
enum Applies {
    ReadOnce,
    Relights,
    ReplacesTexture,
    NextFrame,
    Limit,
    App,
}

impl Applies {
    fn label(self) -> &'static str {
        match self {
            Applies::ReadOnce => "read once",
            Applies::Relights => "relights",
            Applies::ReplacesTexture => "new texture",
            Applies::NextFrame => "next frame",
            Applies::Limit => "limit",
            Applies::App => "the app's",
        }
    }
}

/// One setting: how to show it and how to turn it one step.
struct Dial {
    name: &'static str,
    applies: Applies,
    cost: &'static str,
    value: fn(&Tuning) -> String,
    turn: fn(&mut Tuning, i32),
}

/// The option `step` places from `current` in `options`, stopping at either
/// end. A value not among the options counts as the first.
fn pick<T: PartialEq + Clone>(options: &[T], current: &T, step: i32) -> T {
    let at = options
        .iter()
        .position(|option| option == current)
        .unwrap_or(0) as i32;
    options[(at + step).clamp(0, options.len() as i32 - 1) as usize].clone()
}

/// The level curves the curve dials pick from, each with the Rust that makes
/// it.
fn curves() -> [(&'static str, LevelCurve); 5] {
    [
        ("LevelCurve::exponential(0.6)", LevelCurve::exponential(0.6)),
        ("LevelCurve::exponential(0.7)", LevelCurve::exponential(0.7)),
        ("LevelCurve::exponential(0.8)", LevelCurve::exponential(0.8)),
        ("LevelCurve::gamma(2.2)", LevelCurve::gamma(2.2)),
        ("LevelCurve::linear()", LevelCurve::linear()),
    ]
}

/// The Rust that makes `curve`, which must be one of [`curves`].
fn curve_rust(curve: &LevelCurve) -> &'static str {
    curves()
        .into_iter()
        .find(|(_, preset)| preset == curve)
        .map(|(rust, _)| rust)
        .expect("every curve a dial picks is one of `curves()`")
}

fn turn_curve(curve: &mut LevelCurve, step: i32) {
    let presets = curves().map(|(_, curve)| curve);
    *curve = pick(&presets, curve, step);
}

fn suns() -> Vec<SunGate> {
    std::iter::once(SunGate::DirectSkyOnly)
        .chain(curves().map(|(_, curve)| SunGate::ByLevel(curve)))
        .collect()
}

const SKIES: [Sky; 10] = [
    Sky::Clearance { chunks: 1 },
    Sky::Clearance { chunks: 2 },
    Sky::Clearance { chunks: 4 },
    Sky::Clearance { chunks: 8 },
    Sky::Clearance { chunks: 16 },
    Sky::Clearance { chunks: 32 },
    Sky::Ceiling { y: 64 },
    Sky::Ceiling { y: 128 },
    Sky::Ceiling { y: 256 },
    Sky::Off,
];

const MIB: usize = 1024 * 1024;

/// GPU memory of the light texture when every slot is taken.
fn texture_mib(light: &LightPropagation) -> f32 {
    let bytes_per_slot = match light.precision {
        LightPrecision::Steps16 => 39_304.0,
        LightPrecision::Steps256 => 78_608.0,
    };
    light.slot_capacity as f32 * bytes_per_slot / MIB as f32
}

fn dials() -> Vec<Dial> {
    vec![
        Dial {
            name: "shown",
            applies: Applies::ReadOnce,
            cost: "changing it here only warns; it counts once pasted",
            value: |t| t.light.shown.to_string(),
            turn: |t, _| t.light.shown = !t.light.shown,
        },
        Dial {
            name: "sky",
            applies: Applies::Relights,
            cost: "Off to on starts an empty sky record: caves flash until probed",
            value: |t| format!("{:?}", t.light.sky),
            turn: |t, step| t.light.sky = pick(&SKIES, &t.light.sky, step),
        },
        Dial {
            name: "sky_reach",
            applies: Applies::Relights,
            cost: "past 16 rows take two words: dearer sky solves",
            value: |t| t.light.sky_reach.to_string(),
            turn: |t, step| {
                t.light.sky_reach = (t.light.sky_reach as i32 + step).clamp(1, 32) as u8
            },
        },
        Dial {
            name: "voxel_reach",
            applies: Applies::Relights,
            cost: "past 16 rows take two words: dearer voxel solves",
            value: |t| t.light.voxel_reach.to_string(),
            turn: |t, step| {
                t.light.voxel_reach =
                    (t.light.voxel_reach as i32 + step).clamp(1, 32) as u8
            },
        },
        Dial {
            name: "shape",
            applies: Applies::Relights,
            cost: "same cost either way",
            value: |t| format!("{:?}", t.light.shape),
            turn: |t, step| {
                t.light.shape = pick(
                    &[LightShape::Round, LightShape::Diamond],
                    &t.light.shape,
                    step,
                )
            },
        },
        Dial {
            name: "sky_fill",
            applies: Applies::Relights,
            cost: "openness: 15-150x paths' field updates, seconds to settle",
            value: |t| format!("{:?}", t.fill),
            turn: |t, step| {
                t.fill = pick(
                    &[FillModel::Off, FillModel::Paths, FillModel::Openness],
                    &t.fill,
                    step,
                )
            },
        },
        Dial {
            name: "  paths: beside_open_sky",
            applies: Applies::Relights,
            cost: "chunks taking sky fill",
            value: |t| format!("{:?}", t.paths.beside_open_sky),
            turn: |t, step| {
                t.paths.beside_open_sky =
                    (t.paths.beside_open_sky + step as f32).clamp(0.0, 15.0)
            },
        },
        Dial {
            name: "  paths: whole_face_loss",
            applies: Applies::Relights,
            cost: "lower carries light further; field updates dearer",
            value: |t| format!("{:?}", t.paths.whole_face_loss),
            turn: |t, step| {
                t.paths.whole_face_loss = pick(
                    &[0.5, 1.0, 1.5, 2.0, 3.0, 4.0],
                    &t.paths.whole_face_loss,
                    step,
                )
            },
        },
        Dial {
            name: "  paths: opening_power",
            applies: Applies::Relights,
            cost: "lower first for brighter caves; no dearer",
            value: |t| format!("{:?}", t.paths.opening_power),
            turn: |t, step| {
                t.paths.opening_power = pick(
                    &[0.0, 0.5, 1.0, 1.5, 2.0, 3.0],
                    &t.paths.opening_power,
                    step,
                )
            },
        },
        Dial {
            name: "  openness: sun_at",
            applies: Applies::Relights,
            cost: "chunks taking sky fill",
            value: |t| format!("{:?}", t.openness.sun_at),
            turn: |t, step| {
                t.openness.sun_at =
                    pick(&[0.5, 0.6, 0.7, 0.8, 0.9, 1.0], &t.openness.sun_at, step)
            },
        },
        Dial {
            name: "  openness: gain",
            applies: Applies::Relights,
            cost: "chunks taking sky fill",
            value: |t| format!("{:?}", t.openness.gain),
            turn: |t, step| {
                t.openness.gain =
                    pick(&[0.5, 0.75, 1.0, 1.5, 2.0], &t.openness.gain, step)
            },
        },
        Dial {
            name: "  openness: rays",
            applies: Applies::Relights,
            cost: "a cell costs as much as its rays",
            value: |t| t.openness.rays.to_string(),
            turn: |t, step| {
                t.openness.rays = pick(&[8, 16, 32, 64, 128], &t.openness.rays, step)
            },
        },
        Dial {
            name: "  openness: reach",
            applies: Applies::Relights,
            cost: "an edit casts again every cell this far below and beside it",
            value: |t| t.openness.reach.to_string(),
            turn: |t, step| {
                t.openness.reach = pick(&[32, 64, 128, 256, 512], &t.openness.reach, step)
            },
        },
        Dial {
            name: "precision",
            applies: Applies::ReplacesTexture,
            cost: "256 levels: 2 bytes per voxel on the CPU, twice the GPU memory",
            value: |t| {
                format!(
                    "{:?} ({:.0} MiB GPU)",
                    t.light.precision,
                    texture_mib(&t.light)
                )
            },
            turn: |t, step| {
                t.light.precision = pick(
                    &[LightPrecision::Steps16, LightPrecision::Steps256],
                    &t.light.precision,
                    step,
                )
            },
        },
        Dial {
            name: "slot_capacity",
            applies: Applies::ReplacesTexture,
            cost: "too few: far chunks lose shadows and glow, with a warning",
            value: |t| {
                format!(
                    "{} ({:.0} MiB GPU)",
                    t.light.slot_capacity,
                    texture_mib(&t.light)
                )
            },
            turn: |t, step| {
                t.light.slot_capacity = pick(
                    &[64, 128, 256, 343, 512, 1024, 2048, 4096],
                    &t.light.slot_capacity,
                    step,
                )
            },
        },
        Dial {
            name: "look.smoothing",
            applies: Applies::NextFrame,
            cost: "one uniform write",
            value: |t| format!("{:?}", t.light.look.smoothing),
            turn: |t, step| {
                t.light.look.smoothing = pick(
                    &[LightSmoothing::Smoothed, LightSmoothing::Stepped],
                    &t.light.look.smoothing,
                    step,
                )
            },
        },
        Dial {
            name: "look.sky_falloff",
            applies: Applies::NextFrame,
            cost: "one uniform write",
            value: |t| curve_rust(&t.light.look.sky_falloff).into(),
            turn: |t, step| turn_curve(&mut t.light.look.sky_falloff, step),
        },
        Dial {
            name: "look.voxel_falloff",
            applies: Applies::NextFrame,
            cost: "one uniform write",
            value: |t| curve_rust(&t.light.look.voxel_falloff).into(),
            turn: |t, step| turn_curve(&mut t.light.look.voxel_falloff, step),
        },
        Dial {
            name: "look.sun",
            applies: Applies::NextFrame,
            cost: "past direct sky the sun can reach caves rock doesn't shadow",
            value: |t| sun_rust(&t.light.look.sun),
            turn: |t, step| t.light.look.sun = pick(&suns(), &t.light.look.sun, step),
        },
        Dial {
            name: "look.voxel_brightness",
            applies: Applies::NextFrame,
            cost: "one uniform write",
            value: |t| format!("{:?}", t.light.look.voxel_brightness),
            turn: |t, step| {
                t.light.look.voxel_brightness = pick(
                    &[250.0, 500.0, 1000.0, 2000.0, 4000.0],
                    &t.light.look.voxel_brightness,
                    step,
                )
            },
        },
        Dial {
            name: "look.emissive_glow",
            applies: Applies::NextFrame,
            cost: "one uniform write",
            value: |t| format!("{:?}", t.light.look.emissive_glow),
            turn: |t, step| {
                t.light.look.emissive_glow = pick(
                    &[0.0, 2.0, 4.0, 6.0, 9.0, 12.0],
                    &t.light.look.emissive_glow,
                    step,
                )
            },
        },
        Dial {
            name: "look.ao",
            applies: Applies::NextFrame,
            cost: "one uniform write",
            value: |t| format!("{:?}", t.light.look.ao),
            turn: |t, step| {
                t.light.look.ao = pick(
                    &[LightAo::IndirectLight, LightAo::BaseColor, LightAo::Off],
                    &t.light.look.ao,
                    step,
                )
            },
        },
        Dial {
            name: "look.minimum_brightness",
            applies: Applies::NextFrame,
            cost: "above 0, no face is exactly black",
            value: |t| format!("{:?}", t.light.look.minimum_brightness),
            turn: |t, step| {
                t.light.look.minimum_brightness = pick(
                    &[0.0, 0.01, 0.02, 0.05, 0.1, 0.2],
                    &t.light.look.minimum_brightness,
                    step,
                )
            },
        },
        Dial {
            name: "sky_record_cap",
            applies: Applies::Limit,
            cost: "past it, chunks look up the voxels above them again",
            value: |t| format!("{} MiB", t.light.sky_record_cap / MIB),
            turn: |t, step| {
                let mib = pick(
                    &[1, 2, 4, 8, 16, 32, 64],
                    &(t.light.sky_record_cap / MIB),
                    step,
                );
                t.light.sky_record_cap = mib * MIB;
            },
        },
        Dial {
            name: "wasm_relight_budget",
            applies: Applies::Limit,
            cost: "WASM only: frame time spent relighting",
            value: |t| format!("{:?}", t.light.wasm_relight_budget),
            turn: |t, step| {
                let ms = pick(
                    &[1, 2, 4, 8, 16, 32],
                    &(t.light.wasm_relight_budget.as_millis() as u64),
                    step,
                );
                t.light.wasm_relight_budget = std::time::Duration::from_millis(ms);
            },
        },
        Dial {
            name: "emissive: lamp",
            applies: Applies::Relights,
            cost: "chunks with solids find their lamps again, 40-100 µs each",
            value: |t| format!("level {:?}", t.lamp),
            turn: |t, step| t.lamp = (t.lamp + step as f32).clamp(0.0, 15.0),
        },
        Dial {
            name: "emissive: dim lamp",
            applies: Applies::Relights,
            cost: "a lamp at 8 reaches about half as far as one at 15",
            value: |t| format!("level {:?}", t.dim_lamp),
            turn: |t, step| t.dim_lamp = (t.dim_lamp + step as f32).clamp(0.0, 15.0),
        },
        Dial {
            name: "GlobalAmbientLight",
            applies: Applies::App,
            cost: "not a light setting; caves stay black at any value",
            value: |t| format!("{:?}", t.ambient),
            turn: |t, step| {
                t.ambient = pick(
                    &[0.0, 100.0, 300.0, 600.0, 1200.0, 2400.0, 5000.0],
                    &t.ambient,
                    step,
                )
            },
        },
    ]
}

// --- Printing the config as Rust ---------------------------------------------

/// Each line of `text` indented by four spaces.
fn indent(text: &str) -> String {
    text.lines().map(|line| format!("    {line}\n")).collect()
}

/// A struct literal of `name` with `fields`, and the rest from its default.
fn literal(name: &str, fields: Vec<String>) -> String {
    if fields.is_empty() {
        return format!("{name}::default()");
    }
    let body: String = fields
        .iter()
        .map(|field| indent(&format!("{field},")))
        .collect();
    format!("{name} {{\n{body}    ..default()\n}}")
}

/// `name: value` for each field of `$value` that differs from `$default`,
/// with the value written by `$rust`.
macro_rules! differing {
    ($fields:ident, $value:expr, $default:expr, $($field:ident => $rust:expr),* $(,)?) => {
        $(
            if $value.$field != $default.$field {
                let rust: fn(&_) -> String = $rust;
                $fields.push(format!("{}: {}", stringify!($field), rust(&$value.$field)));
            }
        )*
    };
}

fn debug(value: &impl std::fmt::Debug) -> String {
    format!("{value:?}")
}

fn sun_rust(sun: &SunGate) -> String {
    match sun {
        SunGate::ByLevel(curve) => format!("SunGate::ByLevel({})", curve_rust(curve)),
        direct => format!("SunGate::{direct:?}"),
    }
}

fn duration_rust(duration: &std::time::Duration) -> String {
    if duration.subsec_micros().is_multiple_of(1000) {
        format!("std::time::Duration::from_millis({})", duration.as_millis())
    } else {
        format!("std::time::Duration::from_micros({})", duration.as_micros())
    }
}

fn bytes_rust(bytes: &usize) -> String {
    if bytes.is_multiple_of(MIB) {
        format!("{} * 1024 * 1024", bytes / MIB)
    } else {
        bytes.to_string()
    }
}

fn sky_fill_rust(fill: &Option<SkyFill>) -> String {
    match fill {
        None => "None".into(),
        Some(SkyFill::Paths(paths)) => {
            let default = PathFill::default();
            let mut fields = Vec::new();
            differing!(fields, paths, default,
                beside_open_sky => debug,
                whole_face_loss => debug,
                opening_power => debug,
            );
            format!("Some(SkyFill::Paths({}))", literal("PathFill", fields))
        }
        Some(SkyFill::Openness(openness)) => {
            let default = OpenFill::default();
            let mut fields = Vec::new();
            differing!(fields, openness, default,
                sun_at => debug,
                gain => debug,
                rays => debug,
                reach => debug,
            );
            format!("Some(SkyFill::Openness({}))", literal("OpenFill", fields))
        }
    }
}

fn look_rust(look: &LightLook) -> String {
    let default = LightLook::default();
    let mut fields = Vec::new();
    differing!(fields, look, default,
        smoothing => |smoothing| format!("LightSmoothing::{smoothing:?}"),
        sky_falloff => |curve| curve_rust(curve).into(),
        voxel_falloff => |curve| curve_rust(curve).into(),
        sun => sun_rust,
        voxel_brightness => debug,
        emissive_glow => debug,
        ao => |ao| format!("LightAo::{ao:?}"),
        minimum_brightness => debug,
    );
    literal("LightLook", fields)
}

/// `light` as a Rust expression, naming only the fields that differ from
/// the default. It compiles where the crate's prelude and Bevy's are in
/// scope.
fn light_rust(light: &LightPropagation) -> String {
    let default = LightPropagation::default();
    let mut fields = Vec::new();
    differing!(fields, light, default,
        shown => debug,
        sky => |sky| format!("Sky::{sky:?}"),
        sky_reach => debug,
        voxel_reach => debug,
        shape => |shape| format!("LightShape::{shape:?}"),
        sky_fill => sky_fill_rust,
        precision => |precision| format!("LightPrecision::{precision:?}"),
        slot_capacity => debug,
        look => look_rust,
        sky_record_cap => bytes_rust,
        wasm_relight_budget => duration_rust,
    );
    literal("LightPropagation", fields)
}

/// The two config methods a world pastes into its `VoxelWorldConfig`.
fn config_rust(tuning: &Tuning) -> String {
    let lamps = [(LAMP, tuning.lamp), (DIM_LAMP, tuning.dim_lamp)]
        .map(|(material, level)| format!("({material}, Emission::level({level:?}))"))
        .join(", ");
    format!(
        "fn light_propagation(&self) -> Option<LightPropagation> {{\n{}}}\n\n\
         fn emissive_materials(&self) -> Vec<(u8, Emission)> {{\n    vec![{lamps}]\n}}",
        indent(&format!("Some({})", light_rust(&tuning.light()))),
    )
}

// --- The world ---------------------------------------------------------------

#[derive(Resource, Clone, Default)]
struct TuningWorld {
    tuning: Tuning,
}

impl VoxelWorldConfig for TuningWorld {
    type MaterialIndex = u8;
    type ChunkUserBundle = ();

    fn voxel_lookup_delegate(&self) -> VoxelLookupDelegate<Self::MaterialIndex> {
        Box::new(|_, _, _| {
            let scene = Scene {
                hills: Perlin::new(7),
            };
            Box::new(move |position, _| scene.voxel(position))
        })
    }

    fn texture_index_mapper(
        &self,
    ) -> std::sync::Arc<dyn Fn(Self::MaterialIndex) -> [u32; 3] + Send + Sync> {
        std::sync::Arc::new(|material| match material {
            GRASS => [0, 1, 1],
            DIRT => [1, 1, 1],
            STONE => [2, 2, 2],
            _ => [3, 3, 3],
        })
    }

    fn light_propagation(&self) -> Option<LightPropagation> {
        Some(self.tuning.light())
    }

    fn emissive_materials(&self) -> Vec<(Self::MaterialIndex, Emission)> {
        self.tuning.emissive_materials()
    }
}

// --- The app -----------------------------------------------------------------

const CAMERAS: [([f32; 3], [f32; 3]); 5] = [
    // The tunnel's mouth.
    ([-110.0, 34.0, 6.0], [0.0, 27.0, 0.0]),
    // The lamps in the tunnel.
    ([-4.0, 28.0, 0.0], [20.0, 26.0, -4.0]),
    // The ground under the floating island.
    ([-70.0, 40.0, -160.0], [0.0, 26.0, -160.0]),
    // Under the overhang.
    ([0.0, 34.0, 70.0], [0.0, 28.0, 120.0]),
    // Everything from above.
    ([-170.0, 150.0, -60.0], [0.0, 40.0, 0.0]),
];

fn camera_transform(index: usize) -> Transform {
    let (position, look_at) = CAMERAS[index];
    Transform::from_translation(Vec3::from_array(position))
        .looking_at(Vec3::from_array(look_at), Vec3::Y)
}

/// The ball that walks in and out of the tunnel and dims itself from the
/// light around it.
#[derive(Component)]
struct Walker;

const WALKER_RADIUS: f32 = 0.8;
const WALKER_COLOUR: LinearRgba = LinearRgba::rgb(0.9, 0.35, 0.08);

/// Which dial is picked, and the last change's time to settle.
#[derive(Resource, Default)]
struct Panel {
    picked: usize,
    changed: Option<Change>,
}

/// A change of a dial, and when its light settled.
struct Change {
    dial: usize,
    at: Instant,
    frames: u32,
    settled: Option<(f32, u32)>,
}

#[derive(Component)]
struct PanelText;

fn main() {
    #[cfg(not(target_arch = "wasm32"))]
    if std::env::args().nth(1).as_deref() == Some("measure") {
        return measure();
    }
    #[cfg(target_arch = "wasm32")]
    let plugins = DefaultPlugins.set(WindowPlugin {
        primary_window: Some(Window {
            canvas: Some("#bevy".into()),
            ..default()
        }),
        ..default()
    });
    #[cfg(not(target_arch = "wasm32"))]
    let plugins = DefaultPlugins;
    App::new()
        .add_plugins((plugins, PanOrbitCameraPlugin))
        .add_plugins(VoxelWorldPlugin::with_config(TuningWorld::default()))
        .init_resource::<Panel>()
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
                turn_dials,
                time_settling,
                show_panel,
                switch_camera,
                walk,
                dim_walker,
                follow_ambient,
            )
                .chain(),
        )
        .run();
}

fn add_lights(commands: &mut Commands) {
    commands.spawn((
        DirectionalLight {
            color: Color::srgb(0.98, 0.95, 0.82),
            shadow_maps_enabled: true,
            ..default()
        },
        Transform::default().looking_at(Vec3::new(-0.15, -0.1, 0.15), Vec3::Y),
        CascadeShadowConfigBuilder::default().build(),
    ));
    // A lit world wants about 600: propagated light scales it down in shade,
    // and caves stay black at any value.
    commands.insert_resource(GlobalAmbientLight {
        color: Color::srgb(0.98, 0.95, 0.82),
        brightness: Tuning::default().ambient,
        affects_lightmapped_meshes: true,
    });
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    commands.spawn((
        Camera3d::default(),
        camera_transform(0),
        PanOrbitCamera {
            focus: Vec3::from_array(CAMERAS[0].1),
            target_focus: Vec3::from_array(CAMERAS[0].1),
            button_orbit: MouseButton::Right,
            button_pan: MouseButton::Middle,
            ..default()
        },
        VoxelWorldCamera::<TuningWorld>::default(),
    ));
    add_lights(&mut commands);
    commands.spawn((
        Mesh3d(meshes.add(Sphere::new(WALKER_RADIUS))),
        MeshMaterial3d(materials.add(StandardMaterial::from(Color::from(WALKER_COLOUR)))),
        Transform::from_xyz(0.0, GROUND as f32 + WALKER_RADIUS, 0.0),
        Walker,
    ));
    commands.spawn((
        Text::default(),
        TextFont {
            font_size: FontSize::Px(13.0),
            ..default()
        },
        TextColor(Color::WHITE),
        BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.6)),
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(8.0),
            left: Val::Px(8.0),
            padding: UiRect::all(Val::Px(6.0)),
            ..default()
        },
        PanelText,
    ));
}

/// Up and Down pick a dial, Left and Right turn it, and P prints the config.
fn turn_dials(
    keys: Res<ButtonInput<KeyCode>>,
    mut panel: ResMut<Panel>,
    mut world: ResMut<TuningWorld>,
) {
    let dials = dials();
    if keys.just_pressed(KeyCode::ArrowDown) {
        panel.picked = (panel.picked + 1) % dials.len();
    }
    if keys.just_pressed(KeyCode::ArrowUp) {
        panel.picked = (panel.picked + dials.len() - 1) % dials.len();
    }
    if keys.just_pressed(KeyCode::KeyP) {
        report!(
            "// In your world's `impl VoxelWorldConfig`:\n{}",
            config_rust(&world.tuning)
        );
    }
    let step = keys.just_pressed(KeyCode::ArrowRight) as i32
        - keys.just_pressed(KeyCode::ArrowLeft) as i32;
    if step != 0 {
        let mut tuning = world.tuning.clone();
        (dials[panel.picked].turn)(&mut tuning, step);
        // Writing only on a real change keeps the config resource unchanged
        // at the ends of a dial, so the crate does nothing.
        if tuning != world.tuning {
            world.tuning = tuning;
            panel.changed = Some(Change {
                dial: panel.picked,
                at: Instant::now(),
                frames: 0,
                settled: None,
            });
        }
    }
}

/// Times the last change until the world's light has settled.
fn time_settling(mut panel: ResMut<Panel>, voxel_world: VoxelWorld<TuningWorld>) {
    let Some(change) = panel.changed.as_mut() else {
        return;
    };
    if change.settled.is_some() {
        return;
    }
    change.frames += 1;
    if voxel_world.light_is_settled() {
        let ms = change.at.elapsed().as_secs_f32() * 1000.0;
        change.settled = Some((ms, change.frames));
    }
}

fn show_panel(
    panel: Res<Panel>,
    world: Res<TuningWorld>,
    mut text: Query<&mut Text, With<PanelText>>,
) {
    let Ok(mut text) = text.single_mut() else {
        return;
    };
    let dials = dials();
    let mut lines = vec![
        "Up/Down pick, Left/Right turn, P prints the config, 1-5 cameras".to_string(),
        String::new(),
    ];
    for (index, dial) in dials.iter().enumerate() {
        let marker = if index == panel.picked { ">" } else { " " };
        lines.push(format!(
            "{marker} {}: {}  [{}] {}",
            dial.name,
            (dial.value)(&world.tuning),
            dial.applies.label(),
            dial.cost
        ));
    }
    lines.push(String::new());
    lines.push(match &panel.changed {
        None => "Turn a dial to time how long its light takes to settle.".into(),
        Some(change) => match change.settled {
            None => format!(
                "{}: settling, {} frames so far",
                dials[change.dial].name.trim(),
                change.frames
            ),
            Some((ms, frames)) => format!(
                "{}: settled in {ms:.0} ms, {frames} frames",
                dials[change.dial].name.trim()
            ),
        },
    });
    let joined = lines.join("\n");
    if text.0 != joined {
        text.0 = joined;
    }
}

fn switch_camera(
    keys: Res<ButtonInput<KeyCode>>,
    mut cameras: Query<(&mut Transform, &mut PanOrbitCamera)>,
) {
    let digits = [
        KeyCode::Digit1,
        KeyCode::Digit2,
        KeyCode::Digit3,
        KeyCode::Digit4,
        KeyCode::Digit5,
    ];
    for (index, key) in digits.iter().enumerate() {
        if !keys.just_pressed(*key) {
            continue;
        }
        for (mut transform, mut orbit) in &mut cameras {
            *transform = camera_transform(index);
            let (position, look_at) = CAMERAS[index];
            let offset = Vec3::from_array(position) - Vec3::from_array(look_at);
            orbit.target_focus = Vec3::from_array(look_at);
            orbit.target_radius = offset.length();
            orbit.target_yaw = offset.x.atan2(offset.z);
            orbit.target_pitch = (offset.y / offset.length()).asin();
        }
    }
}

/// Walks the ball from the open ground into the tunnel, past both lamps, and
/// back.
fn walk(time: Res<Time>, mut walker: Query<&mut Transform, With<Walker>>) {
    const FROM: f32 = -110.0;
    const TO: f32 = 40.0;
    const SPEED: f32 = 8.0;
    let length = TO - FROM;
    let travelled = (time.elapsed_secs() * SPEED) % (2.0 * length);
    let x = FROM
        + if travelled < length {
            travelled
        } else {
            2.0 * length - travelled
        };
    for mut transform in &mut walker {
        transform.translation.x = x;
    }
}

/// Dims the ball the way voxels are dimmed: Bevy's ambient light and sun by
/// the sky falloff at its sky light level, and voxel light added through its
/// emissive colour, from the voxel falloff at its voxel light level. With
/// these, the ball goes dark where the rock around it does.
fn dim_walker(
    voxel_world: VoxelWorld<TuningWorld>,
    world: Res<TuningWorld>,
    walker: Query<(&Transform, &MeshMaterial3d<StandardMaterial>), With<Walker>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let look = &world.tuning.light.look;
    for (transform, material) in &walker {
        let Some(levels) = voxel_world.light_at(transform.translation.floor().as_ivec3())
        else {
            continue;
        };
        let Some(mut material) = materials.get_mut(&material.0) else {
            continue;
        };
        let shade = look.sky_falloff.at(levels.sky).max(look.minimum_brightness);
        let glow = look.voxel_falloff.at(levels.voxel) * look.voxel_brightness;
        material.base_color = (WALKER_COLOUR * shade).into();
        material.emissive = WALKER_COLOUR * glow;
    }
}

fn follow_ambient(world: Res<TuningWorld>, mut ambient: ResMut<GlobalAmbientLight>) {
    if ambient.brightness != world.tuning.ambient {
        ambient.brightness = world.tuning.ambient;
    }
}

// --- Measuring ---------------------------------------------------------------

/// The dials the `measure` mode turns one step and back, by name.
#[cfg(not(target_arch = "wasm32"))]
const MEASURED: [&str; 10] = [
    "sky",
    "sky_reach",
    "voxel_reach",
    "shape",
    "  paths: opening_power",
    "  paths: whole_face_loss",
    "precision",
    "slot_capacity",
    "look.voxel_brightness",
    "emissive: lamp",
];

/// Frames the world must stay settled, with every chunk Ready, before a
/// measured change is over.
#[cfg(not(target_arch = "wasm32"))]
const QUIET_FRAMES: u32 = 30;

#[cfg(not(target_arch = "wasm32"))]
#[derive(Resource, Default)]
struct Measuring {
    frame: u32,
    /// The last frame on which light hadn't settled or a chunk wasn't Ready.
    last_busy: u32,
    /// How many changes have been made: each dial in `MEASURED` one step and
    /// then back.
    changes: usize,
    /// The frame and time the change being measured was made.
    started: Option<(u32, Instant)>,
    /// The time of the last busy frame.
    last_busy_at: Option<Instant>,
    /// The values before a dial was turned, to turn it back to.
    before: Option<Tuning>,
}

#[cfg(not(target_arch = "wasm32"))]
fn measure() {
    App::new()
        .add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: None,
                    exit_condition: ExitCondition::DontExit,
                    ..default()
                })
                .disable::<bevy::winit::WinitPlugin>()
                .add(bevy::app::ScheduleRunnerPlugin::run_loop(
                    std::time::Duration::from_secs_f64(1.0 / 60.0),
                )),
        )
        .add_plugins(VoxelWorldPlugin::with_config(TuningWorld::default()))
        .init_resource::<Measuring>()
        .add_systems(Startup, setup_measure)
        .add_systems(Update, turn_and_measure)
        .run();
}

#[cfg(not(target_arch = "wasm32"))]
fn setup_measure(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
    let image = Image::new_target_texture(1280, 720, TextureFormat::Rgba8UnormSrgb, None);
    commands.spawn((
        Camera3d::default(),
        RenderTarget::Image(images.add(image).into()),
        camera_transform(0),
        VoxelWorldCamera::<TuningWorld>::default(),
    ));
    add_lights(&mut commands);
}

/// Waits for the world to settle, then turns each measured dial one step and
/// back, reporting how long each change took to settle.
#[cfg(not(target_arch = "wasm32"))]
fn turn_and_measure(
    mut measuring: ResMut<Measuring>,
    mut world: ResMut<TuningWorld>,
    chunks: Query<&Chunk<TuningWorld>>,
    voxel_world: VoxelWorld<TuningWorld>,
    mut exit: MessageWriter<AppExit>,
) {
    measuring.frame += 1;
    let frame = measuring.frame;
    let busy = !voxel_world.light_is_settled()
        || chunks.iter().any(|chunk| {
            voxel_world.chunk_state(chunk.key()) != Some(ChunkStatus::Ready)
        });
    if busy {
        measuring.last_busy = frame;
        measuring.last_busy_at = Some(Instant::now());
    }
    if frame - measuring.last_busy < QUIET_FRAMES {
        return;
    }
    let dials = dials();
    let dial_of = |change: usize| {
        dials
            .iter()
            .find(|dial| dial.name == MEASURED[change / 2])
            .expect("every measured dial is one of `dials()`")
    };
    if let Some((started, at)) = measuring.started.take() {
        let dial = dial_of(measuring.changes - 1);
        let (frames, ms) = match measuring.last_busy_at {
            Some(busy_at) if measuring.last_busy >= started => (
                measuring.last_busy + 1 - started,
                (busy_at - at).as_secs_f64() * 1000.0,
            ),
            _ => (1, 0.0),
        };
        report!(
            "{} to {}: settled in {frames} frames, {ms:.0} ms, with {} chunks",
            dial.name.trim(),
            (dial.value)(&world.tuning),
            chunks.iter().count()
        );
    } else if measuring.changes == 0 {
        report!(
            "measure: the world settled after {frame} frames, with {} chunks",
            chunks.iter().count()
        );
    }
    if measuring.changes == MEASURED.len() * 2 {
        exit.write(AppExit::Success);
        return;
    }
    if let Some(before) = measuring.before.take() {
        world.tuning = before;
    } else {
        // A dial at its top end turns down instead.
        let turn = dial_of(measuring.changes).turn;
        let mut tuning = world.tuning.clone();
        turn(&mut tuning, 1);
        if tuning == world.tuning {
            turn(&mut tuning, -1);
        }
        measuring.before = Some(std::mem::replace(&mut world.tuning, tuning));
    }
    measuring.changes += 1;
    measuring.started = Some((frame, Instant::now()));
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An expression and its source text.
    macro_rules! code {
        ($expression:expr) => {
            ($expression, stringify!($expression))
        };
    }

    fn without_spaces(text: &str) -> String {
        text.chars().filter(|c| !c.is_whitespace()).collect()
    }

    #[test]
    fn the_default_config_prints_as_its_default() {
        assert_eq!(
            light_rust(&LightPropagation::default()),
            "LightPropagation::default()"
        );
    }

    #[test]
    fn a_config_prints_as_the_rust_that_builds_it() {
        let (light, rust) = code!(LightPropagation {
            shown: false,
            sky: Sky::Ceiling { y: 128 },
            sky_reach: 16,
            shape: LightShape::Diamond,
            sky_fill: Some(SkyFill::Openness(OpenFill {
                sun_at: 0.9,
                rays: 16,
                ..default()
            })),
            precision: LightPrecision::Steps256,
            slot_capacity: 343,
            look: LightLook {
                smoothing: LightSmoothing::Stepped,
                sky_falloff: LevelCurve::gamma(2.2),
                sun: SunGate::ByLevel(LevelCurve::linear()),
                voxel_brightness: 2000.0,
                ao: LightAo::BaseColor,
                minimum_brightness: 0.05,
                ..default()
            },
            sky_record_cap: 4 * 1024 * 1024,
            wasm_relight_budget: std::time::Duration::from_millis(8),
            ..default()
        });
        assert_eq!(without_spaces(&light_rust(&light)), without_spaces(rust));
    }

    #[test]
    fn sky_fill_prints_each_model_and_off() {
        for (light, rust) in [
            code!(LightPropagation {
                sky_fill: None,
                ..default()
            }),
            code!(LightPropagation {
                sky_fill: Some(SkyFill::Paths(PathFill {
                    opening_power: 1.0,
                    ..default()
                })),
                ..default()
            }),
            code!(LightPropagation {
                sky_fill: Some(SkyFill::Openness(OpenFill::default())),
                ..default()
            }),
        ] {
            assert_eq!(without_spaces(&light_rust(&light)), without_spaces(rust));
        }
    }

    #[test]
    fn every_dial_turns_to_a_config_that_prints() {
        let mut tuning = Tuning::default();
        for dial in dials() {
            (dial.turn)(&mut tuning, 1);
            (dial.turn)(&mut tuning, -2);
            assert!(!light_rust(&tuning.light()).is_empty());
            assert!(!(dial.value)(&tuning).is_empty());
        }
    }
}
