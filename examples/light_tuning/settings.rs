//! Every light setting the panel offers: what it does, where to see it, and
//! how its control reads and writes the values being tuned.

use std::sync::LazyLock;

use bevy::prelude::*;
use bevy_voxel_world::prelude::*;

use crate::scene::{Place, CRYSTAL, LANTERN};

/// Which model of sky fill the world uses, if any. Each model keeps its own
/// settings in [`Tuning`], so switching back and forth doesn't lose them.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum FillModel {
    Off,
    Paths,
    Openness,
}

/// Every value the panel tunes.
#[derive(Clone, PartialEq)]
pub struct Tuning {
    /// The light settings, apart from `sky_fill`, which comes from the three
    /// fields after it.
    pub light: LightPropagation,
    pub fill: FillModel,
    pub paths: PathFill,
    pub openness: OpenFill,
    /// The levels the lanterns and crystals are declared at.
    pub lantern: f32,
    pub crystal: f32,
    /// The brightness of the app's `GlobalAmbientLight`, which isn't a light
    /// setting.
    pub ambient: f32,
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
            lantern: 15.0,
            crystal: 13.0,
            ambient: 600.0,
        }
    }
}

impl Tuning {
    /// The light settings, with `sky_fill` made from the chosen model.
    pub fn light(&self) -> LightPropagation {
        LightPropagation {
            sky_fill: match self.fill {
                FillModel::Off => None,
                FillModel::Paths => Some(SkyFill::Paths(self.paths.clone())),
                FillModel::Openness => Some(SkyFill::Openness(self.openness.clone())),
            },
            ..self.light.clone()
        }
    }

    pub fn emissive_materials(&self) -> Vec<(u8, Emission)> {
        vec![
            (LANTERN, Emission::level(self.lantern)),
            (CRYSTAL, Emission::level(self.crystal)),
        ]
    }
}

/// The panel's tabs, each a group of settings about one thing.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Tab {
    #[default]
    Sky,
    Lamps,
    Shading,
    Memory,
}

impl Tab {
    pub const ALL: [Tab; 4] = [Tab::Sky, Tab::Lamps, Tab::Shading, Tab::Memory];

    pub fn label(self) -> &'static str {
        match self {
            Tab::Sky => "Sky",
            Tab::Lamps => "Lamps",
            Tab::Shading => "Shading",
            Tab::Memory => "Memory",
        }
    }

    /// One line under the tabs saying what the tab's settings are about.
    pub fn subtitle(self) -> &'static str {
        match self {
            Tab::Sky => "Daylight: where it enters and how far it spreads.",
            Tab::Lamps => "Voxel light, from emissive voxels.",
            Tab::Shading => "How light looks on faces. Shows at once.",
            Tab::Memory => "What light costs, and its limits.",
        }
    }
}

/// When a change to a setting shows, as each field's documentation says.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Applies {
    /// Only the shader reads it, so it shows on the next frame.
    Instant,
    /// Every chunk with a solid near its open voxels works its light out again.
    Relights,
    /// As a relight, and the light texture is made again in a new format.
    NewTexture,
    /// It caps what the world may use, and changes nothing on screen.
    Limit,
    /// It belongs to the app, not to light propagation.
    App,
}

impl Applies {
    pub fn label(self) -> &'static str {
        match self {
            Applies::Instant => "Instant",
            Applies::Relights => "Relights",
            Applies::NewTexture => "New texture",
            Applies::Limit => "Limit",
            Applies::App => "Bevy setting",
        }
    }

    pub fn colour(self) -> Color {
        match self {
            Applies::Instant => Color::srgb(0.22, 0.55, 0.32),
            Applies::Relights => Color::srgb(0.66, 0.45, 0.12),
            Applies::NewTexture => Color::srgb(0.70, 0.27, 0.22),
            Applies::Limit => Color::srgb(0.35, 0.38, 0.45),
            Applies::App => Color::srgb(0.25, 0.42, 0.68),
        }
    }
}

/// How a setting is turned in the panel.
pub enum Control {
    /// A number between `min` and `max` in steps of `step`, shown with
    /// `decimals` decimal places.
    Slider {
        min: f32,
        max: f32,
        step: f32,
        decimals: i32,
        get: fn(&Tuning) -> f32,
        set: fn(&mut Tuning, f32),
    },
    /// One of `options`, by index. Two or three options show as buttons side
    /// by side, and more as a drop-down list.
    Choice {
        options: &'static [&'static str],
        get: fn(&Tuning) -> usize,
        set: fn(&mut Tuning, usize),
    },
}

/// One setting in the panel.
pub struct Setting {
    /// The name in the panel.
    pub label: &'static str,
    /// The field it sets, as written in Rust.
    pub rust: &'static str,
    pub tab: Tab,
    pub applies: Applies,
    /// What it does, in a sentence or two.
    pub about: &'static str,
    /// Where in the world it shows best.
    pub place: Place,
    pub control: Control,
    /// Whether the setting means anything with the other values as they are:
    /// a sky fill model's settings show only while that model is chosen.
    pub shown: fn(&Tuning) -> bool,
}

/// A setting's value, comparable between two [`Tuning`]s.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Value {
    Number(f32),
    Option(usize),
}

impl Setting {
    pub fn value(&self, tuning: &Tuning) -> Value {
        match &self.control {
            Control::Slider { get, .. } => Value::Number(get(tuning)),
            Control::Choice { get, .. } => Value::Option(get(tuning)),
        }
    }

    pub fn set(&self, tuning: &mut Tuning, value: Value) {
        match (&self.control, value) {
            (Control::Slider { set, .. }, Value::Number(number)) => set(tuning, number),
            (Control::Choice { set, .. }, Value::Option(option)) => set(tuning, option),
            _ => {}
        }
    }

    /// Whether `tuning` holds something other than the default for this
    /// setting.
    pub fn changed(&self, tuning: &Tuning) -> bool {
        self.value(tuning) != self.value(&Tuning::default())
    }

    /// The value as the panel writes it.
    pub fn show(&self, tuning: &Tuning) -> String {
        match &self.control {
            Control::Slider { get, decimals, .. } => {
                format!("{:.*}", *decimals as usize, get(tuning))
            }
            Control::Choice { options, get, .. } => options[get(tuning)].to_string(),
        }
    }

    /// Turns the setting one step up, or down when it is at the top.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn nudge(&self, tuning: &mut Tuning) {
        match &self.control {
            Control::Slider {
                min,
                max,
                step,
                get,
                set,
                ..
            } => {
                let value = get(tuning);
                let up = value + step;
                set(
                    tuning,
                    if up <= *max {
                        up
                    } else {
                        (value - step).max(*min)
                    },
                );
            }
            Control::Choice {
                options, get, set, ..
            } => {
                let at = get(tuning);
                set(
                    tuning,
                    if at + 1 < options.len() {
                        at + 1
                    } else {
                        at - 1
                    },
                );
            }
        }
    }
}

/// The index of `value` in `options`, or 0 if it isn't there.
fn index_of<T: PartialEq>(options: &[T], value: &T) -> usize {
    options
        .iter()
        .position(|option| option == value)
        .unwrap_or(0)
}

pub const SKIES: [Sky; 10] = [
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

const SKY_LABELS: [&str; 10] = [
    "Clearance: 1 chunk",
    "Clearance: 2 chunks",
    "Clearance: 4 chunks",
    "Clearance: 8 chunks",
    "Clearance: 16 chunks",
    "Clearance: 32 chunks",
    "Ceiling: y = 64",
    "Ceiling: y = 128",
    "Ceiling: y = 256",
    "Off",
];

/// The level curves the curve settings pick from, each with the Rust that
/// makes it.
pub fn curves() -> [(&'static str, LevelCurve); 5] {
    [
        ("LevelCurve::exponential(0.6)", LevelCurve::exponential(0.6)),
        ("LevelCurve::exponential(0.7)", LevelCurve::exponential(0.7)),
        ("LevelCurve::exponential(0.8)", LevelCurve::exponential(0.8)),
        ("LevelCurve::gamma(2.2)", LevelCurve::gamma(2.2)),
        ("LevelCurve::linear()", LevelCurve::linear()),
    ]
}

const CURVE_LABELS: [&str; 5] = [
    "x 0.6 per level",
    "x 0.7 per level",
    "x 0.8 per level",
    "Gamma 2.2",
    "Linear",
];

/// The index in [`curves`] of the preset `curve` matches, if one does.
///
/// Curves match when every level's brightness is within a millionth, not
/// when they are equal: the same curve built in two places can differ in the
/// last bit, since the compiler may work out `powf` at compile time in one
/// and at run time in the other.
pub fn preset_of(curve: &LevelCurve) -> Option<usize> {
    curves().iter().position(|(_, preset)| {
        (0..=15)
            .all(|level| (preset.at(level as f32) - curve.at(level as f32)).abs() < 1e-6)
    })
}

fn curve(index: usize) -> LevelCurve {
    curves()[index].1.clone()
}

const SUN_LABELS: [&str; 6] = [
    "Direct sky only",
    "By level, x 0.6",
    "By level, x 0.7",
    "By level, x 0.8",
    "By level, gamma 2.2",
    "By level, linear",
];

fn sun_index(sun: &SunGate) -> usize {
    match sun {
        SunGate::DirectSkyOnly => 0,
        SunGate::ByLevel(curve) => preset_of(curve).map_or(0, |index| index + 1),
    }
}

fn sun(index: usize) -> SunGate {
    match index {
        0 => SunGate::DirectSkyOnly,
        index => SunGate::ByLevel(curve(index - 1)),
    }
}

const FILL_MODELS: [FillModel; 3] =
    [FillModel::Off, FillModel::Paths, FillModel::Openness];
const SHAPES: [LightShape; 2] = [LightShape::Round, LightShape::Diamond];
const SMOOTHINGS: [LightSmoothing; 2] =
    [LightSmoothing::Smoothed, LightSmoothing::Stepped];
const AOS: [LightAo; 3] = [LightAo::IndirectLight, LightAo::BaseColor, LightAo::Off];
const PRECISIONS: [LightPrecision; 2] =
    [LightPrecision::Steps16, LightPrecision::Steps256];
const RAYS: [u32; 5] = [8, 16, 32, 64, 128];
const RAY_REACHES: [u32; 5] = [32, 64, 128, 256, 512];
const SLOT_CAPACITIES: [u32; 8] = [64, 128, 256, 343, 512, 1024, 2048, 4096];
const MIB: usize = 1024 * 1024;
const RECORD_CAPS_MIB: [usize; 7] = [1, 2, 4, 8, 16, 32, 64];
const BUDGETS_MS: [u64; 6] = [1, 2, 4, 8, 16, 32];

/// GPU memory of the light texture when every slot is taken, in MiB.
pub fn texture_mib(light: &LightPropagation) -> f32 {
    let bytes_per_slot = match light.precision {
        LightPrecision::Steps16 => 39_304.0,
        LightPrecision::Steps256 => 78_608.0,
    };
    light.slot_capacity as f32 * bytes_per_slot / MIB as f32
}

fn always(_: &Tuning) -> bool {
    true
}

fn under_paths(tuning: &Tuning) -> bool {
    tuning.fill == FillModel::Paths
}

fn under_openness(tuning: &Tuning) -> bool {
    tuning.fill == FillModel::Openness
}

/// Every setting, in the order the panel lists them.
pub static SETTINGS: LazyLock<Vec<Setting>> = LazyLock::new(|| {
    vec![
        // --- Sky ---------------------------------------------------------------
        Setting {
            label: "Sky",
            rust: "sky",
            tab: Tab::Sky,
            applies: Applies::Relights,
            about: "Where sky light enters a column: when nothing solid is within \
                    so many chunks above it (clearance) or up to a fixed height \
                    (ceiling), or never. A shadow ends about the clearance below \
                    what casts it.",
            place: Place::Island,
            control: Control::Choice {
                options: &SKY_LABELS,
                get: |t| index_of(&SKIES, &t.light.sky),
                set: |t, index| t.light.sky = SKIES[index].clone(),
            },
            shown: always,
        },
        Setting {
            label: "Sky reach",
            rust: "sky_reach",
            tab: Tab::Sky,
            applies: Applies::Relights,
            about: "How many voxels sky light spreads sideways and down past where \
                    it falls, fading as it goes. Past 16 every solve costs more.",
            place: Place::Well,
            control: Control::Slider {
                min: 1.0,
                max: 32.0,
                step: 1.0,
                decimals: 0,
                get: |t| t.light.sky_reach as f32,
                set: |t, value| t.light.sky_reach = value as u8,
            },
            shown: always,
        },
        Setting {
            label: "Shape",
            rust: "shape",
            tab: Tab::Sky,
            applies: Applies::Relights,
            about: "The shape light spreads in, sky and voxel light alike: nearly \
                    round pools, or diamonds that lose a level per face step.",
            place: Place::Chamber,
            control: Control::Choice {
                options: &["Round", "Diamond"],
                get: |t| index_of(&SHAPES, &t.light.shape),
                set: |t, index| t.light.shape = SHAPES[index],
            },
            shown: always,
        },
        Setting {
            label: "Sky fill",
            rust: "sky_fill",
            tab: Tab::Sky,
            applies: Applies::Relights,
            about: "Sky light that reaches further than it spreads: down the \
                    cave's hall, under the floating island. Paths passes light \
                    through openings; openness lights a place by how much sky \
                    it sees, and takes seconds to settle.",
            place: Place::Cave,
            control: Control::Choice {
                options: &["Off", "Paths", "Openness"],
                get: |t| index_of(&FILL_MODELS, &t.fill),
                set: |t, index| t.fill = FILL_MODELS[index],
            },
            shown: always,
        },
        Setting {
            label: "Open-sky level",
            rust: "PathFill::beside_open_sky",
            tab: Tab::Sky,
            applies: Applies::Relights,
            about: "The light level of open space beside the open sky, which sky \
                    fill passes on from there. Lower darkens everything under \
                    cover.",
            place: Place::Cave,
            control: Control::Slider {
                min: 0.0,
                max: 15.0,
                step: 1.0,
                decimals: 0,
                get: |t| t.paths.beside_open_sky,
                set: |t, value| t.paths.beside_open_sky = value,
            },
            shown: under_paths,
        },
        Setting {
            label: "Loss per face",
            rust: "PathFill::whole_face_loss",
            tab: Tab::Sky,
            applies: Applies::Relights,
            about: "Levels lost through a whole open face of 8-voxel cells, and \
                    more through narrower openings: the cave's hall darkens a \
                    step per chunk. Lower carries fill deeper, and each change \
                    of the world then reaches further.",
            place: Place::Cave,
            control: Control::Slider {
                min: 0.5,
                max: 4.0,
                step: 0.5,
                decimals: 1,
                get: |t| t.paths.whole_face_loss,
                set: |t, value| t.paths.whole_face_loss = value,
            },
            shown: under_paths,
        },
        Setting {
            label: "Narrow loss",
            rust: "PathFill::opening_power",
            tab: Tab::Sky,
            applies: Applies::Relights,
            about: "How much more an opening loses when it is narrower than the \
                    space behind it, as each chunk boundary across the cave's \
                    hall is. Lower carries fill down the hall, and at 0 \
                    through the passage, at no extra cost.",
            place: Place::Cave,
            control: Control::Slider {
                min: 0.0,
                max: 3.0,
                step: 0.25,
                decimals: 2,
                get: |t| t.paths.opening_power,
                set: |t, value| t.paths.opening_power = value,
            },
            shown: under_paths,
        },
        Setting {
            label: "Sun at",
            rust: "OpenFill::sun_at",
            tab: Tab::Sky,
            applies: Applies::Relights,
            about: "The share of the sky a place must see to count as direct sky \
                    and get the sun. Lower lets the sun under the island's rim.",
            place: Place::Island,
            control: Control::Slider {
                min: 0.5,
                max: 1.0,
                step: 0.05,
                decimals: 2,
                get: |t| t.openness.sun_at,
                set: |t, value| t.openness.sun_at = value,
            },
            shown: under_openness,
        },
        Setting {
            label: "Gain",
            rust: "OpenFill::gain",
            tab: Tab::Sky,
            applies: Applies::Relights,
            about: "How bright a place is for the share of the sky it sees: the \
                    cave's hall sees it only through the mouth.",
            place: Place::Cave,
            control: Control::Slider {
                min: 0.25,
                max: 2.0,
                step: 0.25,
                decimals: 2,
                get: |t| t.openness.gain,
                set: |t, value| t.openness.gain = value,
            },
            shown: under_openness,
        },
        Setting {
            label: "Rays",
            rust: "OpenFill::rays",
            tab: Tab::Sky,
            applies: Applies::Relights,
            about: "How many rays each 8-voxel cell casts at the sky. Fewer \
                    leave the cave's hall blotchy; twice the rays, twice the \
                    time.",
            place: Place::Cave,
            control: Control::Choice {
                options: &["8", "16", "32", "64", "128"],
                get: |t| index_of(&RAYS, &t.openness.rays),
                set: |t, index| t.openness.rays = RAYS[index],
            },
            shown: under_openness,
        },
        Setting {
            label: "Ray reach",
            rust: "OpenFill::reach",
            tab: Tab::Sky,
            applies: Applies::Relights,
            about: "How many voxels a ray goes looking for the sky. At 32, rays \
                    from the middle of the ground under the island stop short \
                    of the sky past its edge. Longer makes every voxel edit \
                    cast again the rays of more cells.",
            place: Place::Island,
            control: Control::Choice {
                options: &["32", "64", "128", "256", "512"],
                get: |t| index_of(&RAY_REACHES, &t.openness.reach),
                set: |t, index| t.openness.reach = RAY_REACHES[index],
            },
            shown: under_openness,
        },
        // --- Lamps -------------------------------------------------------------
        Setting {
            label: "Lamp reach",
            rust: "voxel_reach",
            tab: Tab::Lamps,
            applies: Applies::Relights,
            about: "How many voxels light spreads from a lamp at level 15. Past \
                    16 every solve costs more.",
            place: Place::Cave,
            control: Control::Slider {
                min: 1.0,
                max: 32.0,
                step: 1.0,
                decimals: 0,
                get: |t| t.light.voxel_reach as f32,
                set: |t, value| t.light.voxel_reach = value as u8,
            },
            shown: always,
        },
        Setting {
            label: "Lantern level",
            rust: "emissive_materials",
            tab: Tab::Lamps,
            applies: Applies::Relights,
            about: "The level the lanterns give off. A lamp at 8 lights about half \
                    as far as one at 15.",
            place: Place::Cave,
            control: Control::Slider {
                min: 0.0,
                max: 15.0,
                step: 1.0,
                decimals: 0,
                get: |t| t.lantern,
                set: |t, value| t.lantern = value,
            },
            shown: always,
        },
        Setting {
            label: "Crystal level",
            rust: "emissive_materials",
            tab: Tab::Lamps,
            applies: Applies::Relights,
            about: "The level the crystals in the deep chamber give off.",
            place: Place::Chamber,
            control: Control::Slider {
                min: 0.0,
                max: 15.0,
                step: 1.0,
                decimals: 0,
                get: |t| t.crystal,
                set: |t, value| t.crystal = value,
            },
            shown: always,
        },
        Setting {
            label: "Brightness",
            rust: "look.voxel_brightness",
            tab: Tab::Lamps,
            applies: Applies::Instant,
            about: "How bright voxel light is at level 15, multiplied by the \
                    camera's exposure.",
            place: Place::House,
            control: Control::Slider {
                min: 0.0,
                max: 4000.0,
                step: 250.0,
                decimals: 0,
                get: |t| t.light.look.voxel_brightness,
                set: |t, value| t.light.look.voxel_brightness = value,
            },
            shown: always,
        },
        Setting {
            label: "Glow",
            rust: "look.emissive_glow",
            tab: Tab::Lamps,
            applies: Applies::Instant,
            about: "How brightly a lamp's own faces glow, as a multiple of voxel \
                    light at level 15.",
            place: Place::Chamber,
            control: Control::Slider {
                min: 0.0,
                max: 12.0,
                step: 1.0,
                decimals: 0,
                get: |t| t.light.look.emissive_glow,
                set: |t, value| t.light.look.emissive_glow = value,
            },
            shown: always,
        },
        Setting {
            label: "Falloff",
            rust: "look.voxel_falloff",
            tab: Tab::Lamps,
            applies: Applies::Instant,
            about: "How quickly voxel light dims from one level to the next.",
            place: Place::Cave,
            control: Control::Choice {
                options: &CURVE_LABELS,
                get: |t| preset_of(&t.light.look.voxel_falloff).unwrap_or(0),
                set: |t, index| t.light.look.voxel_falloff = curve(index),
            },
            shown: always,
        },
        // --- Shading -----------------------------------------------------------
        Setting {
            label: "Smoothing",
            rust: "look.smoothing",
            tab: Tab::Shading,
            applies: Applies::Instant,
            about: "Blend light across a face, or show each voxel's level as a \
                    flat step for a blocky look.",
            place: Place::House,
            control: Control::Choice {
                options: &["Smooth", "Stepped"],
                get: |t| index_of(&SMOOTHINGS, &t.light.look.smoothing),
                set: |t, index| t.light.look.smoothing = SMOOTHINGS[index],
            },
            shown: always,
        },
        Setting {
            label: "Sky falloff",
            rust: "look.sky_falloff",
            tab: Tab::Shading,
            applies: Applies::Instant,
            about: "How quickly sky light dims from one level to the next, as the \
                    share of Bevy's ambient light a face gets.",
            place: Place::Cave,
            control: Control::Choice {
                options: &CURVE_LABELS,
                get: |t| preset_of(&t.light.look.sky_falloff).unwrap_or(0),
                set: |t, index| t.light.look.sky_falloff = curve(index),
            },
            shown: always,
        },
        Setting {
            label: "Sun",
            rust: "look.sun",
            tab: Tab::Shading,
            applies: Applies::Instant,
            about: "Where the sun shines: only on direct sky, so it stops at the \
                    arch's edge whatever its angle, or scaled by sky light so it \
                    fades into shade. By level can let it into caves.",
            place: Place::Arch,
            control: Control::Choice {
                options: &SUN_LABELS,
                get: |t| sun_index(&t.light.look.sun),
                set: |t, index| t.light.look.sun = sun(index),
            },
            shown: always,
        },
        Setting {
            label: "Occlusion",
            rust: "look.ao",
            tab: Tab::Shading,
            applies: Applies::Instant,
            about: "What darkens in the mesh's corners: ambient and voxel light \
                    only, every light including the sun, or nothing.",
            place: Place::Arch,
            control: Control::Choice {
                options: &["Indirect", "All", "Off"],
                get: |t| index_of(&AOS, &t.light.look.ao),
                set: |t, index| t.light.look.ao = AOS[index],
            },
            shown: always,
        },
        Setting {
            label: "Minimum",
            rust: "look.minimum_brightness",
            tab: Tab::Shading,
            applies: Applies::Instant,
            about: "The least share of ambient light any face gets, so places no \
                    light reaches needn't be pitch black.",
            place: Place::Chamber,
            control: Control::Slider {
                min: 0.0,
                max: 0.2,
                step: 0.01,
                decimals: 2,
                get: |t| t.light.look.minimum_brightness,
                set: |t, value| t.light.look.minimum_brightness = value,
            },
            shown: always,
        },
        Setting {
            label: "Ambient",
            rust: "GlobalAmbientLight::brightness",
            tab: Tab::Shading,
            applies: Applies::App,
            about: "Bevy's ambient light, which sky light scales down in shade. A \
                    lit world wants about 600; caves stay black at any value.",
            place: Place::Meadow,
            control: Control::Slider {
                min: 0.0,
                max: 3000.0,
                step: 100.0,
                decimals: 0,
                get: |t| t.ambient,
                set: |t, value| t.ambient = value,
            },
            shown: always,
        },
        // --- Memory ------------------------------------------------------------
        Setting {
            label: "Precision",
            rust: "precision",
            tab: Tab::Memory,
            applies: Applies::NewTexture,
            about: "Light levels kept per voxel. 256 smooths the faint terraces \
                    of 16, and doubles the memory on the CPU and the GPU.",
            place: Place::Island,
            control: Control::Choice {
                options: &["16 levels", "256 levels"],
                get: |t| index_of(&PRECISIONS, &t.light.precision),
                set: |t, index| t.light.precision = PRECISIONS[index],
            },
            shown: always,
        },
        Setting {
            label: "Slots",
            rust: "slot_capacity",
            tab: Tab::Memory,
            applies: Applies::NewTexture,
            about: "How many chunks the GPU light texture holds. With too few, far \
                    chunks look flat and a warning is logged.",
            place: Place::Overview,
            control: Control::Choice {
                options: &["64", "128", "256", "343", "512", "1024", "2048", "4096"],
                get: |t| index_of(&SLOT_CAPACITIES, &t.light.slot_capacity),
                set: |t, index| t.light.slot_capacity = SLOT_CAPACITIES[index],
            },
            shown: always,
        },
        Setting {
            label: "Sky record",
            rust: "sky_record_cap",
            tab: Tab::Memory,
            applies: Applies::Limit,
            about: "Memory for remembering which columns hold a solid. Past it, \
                    chunks look up the voxels above them again: slower, never \
                    different light.",
            place: Place::Overview,
            control: Control::Choice {
                options: &[
                    "1 MiB", "2 MiB", "4 MiB", "8 MiB", "16 MiB", "32 MiB", "64 MiB",
                ],
                get: |t| index_of(&RECORD_CAPS_MIB, &(t.light.sky_record_cap / MIB)),
                set: |t, index| t.light.sky_record_cap = RECORD_CAPS_MIB[index] * MIB,
            },
            shown: always,
        },
        Setting {
            label: "WASM budget",
            rust: "wasm_relight_budget",
            tab: Tab::Memory,
            applies: Applies::Limit,
            about: "Main-thread time per frame spent relighting in the browser. \
                    Ignored on native, where relights run on the task pool.",
            place: Place::Overview,
            control: Control::Choice {
                options: &["1 ms", "2 ms", "4 ms", "8 ms", "16 ms", "32 ms"],
                get: |t| {
                    let ms = t.light.wasm_relight_budget.as_millis() as u64;
                    index_of(&BUDGETS_MS, &ms)
                },
                set: |t, index| {
                    t.light.wasm_relight_budget =
                        std::time::Duration::from_millis(BUDGETS_MS[index])
                },
            },
            shown: always,
        },
    ]
});

/// The index in [`SETTINGS`] of the setting with this label.
#[cfg(not(target_arch = "wasm32"))]
pub fn setting_named(label: &str) -> usize {
    SETTINGS
        .iter()
        .position(|setting| setting.label == label)
        .unwrap_or_else(|| panic!("no setting is labelled {label:?}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_choice_reads_back_each_option_it_sets() {
        for setting in SETTINGS.iter() {
            if let Control::Choice { options, get, set } = &setting.control {
                for index in 0..options.len() {
                    let mut tuning = Tuning::default();
                    set(&mut tuning, index);
                    assert_eq!(get(&tuning), index, "{}", setting.label);
                }
            }
        }
    }

    #[test]
    fn every_default_is_one_the_panel_can_show() {
        let tuning = Tuning::default();
        for setting in SETTINGS.iter() {
            match &setting.control {
                Control::Slider { min, max, get, .. } => {
                    let value = get(&tuning);
                    assert!(*min <= value && value <= *max, "{}", setting.label);
                }
                Control::Choice { get, set, .. } => {
                    let mut round_trip = tuning.clone();
                    set(&mut round_trip, get(&tuning));
                    assert!(round_trip == tuning, "{}", setting.label);
                }
            }
        }
    }

    #[test]
    fn the_default_look_has_a_preset_for_every_curve() {
        let look = LightLook::default();
        assert!(preset_of(&look.sky_falloff).is_some());
        assert!(preset_of(&look.voxel_falloff).is_some());
    }

    #[test]
    fn every_nudge_changes_the_value() {
        for setting in SETTINGS.iter() {
            let mut tuning = Tuning::default();
            setting.nudge(&mut tuning);
            assert!(setting.changed(&tuning), "{}", setting.label);
        }
    }
}
