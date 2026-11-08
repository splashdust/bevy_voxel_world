//! Pixel checks of light propagation, rendered offscreen in a lit world of
//! hills, caves and sealed rooms, with lamps underground.
//!
//! It needs a GPU, so `cargo test` skips it (`test = false`) and a maintainer
//! runs it by hand:
//!
//! ```text
//! cargo test --release --features light_propagation --test light_render
//! ```
//!
//! Arguments go after `--`. With none, or with `check`, it renders fixed
//! cameras offscreen and checks the pictures:
//!
//! - With sky light off, a lamp declared below the top level still glows, less
//!   than one at the top level.
//! - With sky light off, a sun with no shadow map leaves the lamp room as it
//!   was with the shadow map.
//! - In a sealed pocket with no lamp in reach, with sky light on and sky fill
//!   off, every pixel is exactly black, with no tonemapping, with and without
//!   the sun's shadow map, and still after a change to 256 levels at runtime.
//!   It stays exactly black with sky fill on under sky openness.
//! - In the worlds `chunk_lifecycle` compares the models of sky fill on,
//!   under paths and then under sky openness, a shot under each floating
//!   island height, at overhangs and an arch, into cave mouths and inside a
//!   building is taken for a person to judge, and each shot of a cave or a
//!   room sealed from the sky is exactly black.
//! - Each change of the look shows on the frame after it's made, and the glow
//!   still scales with each lamp's declared level.
//! - A sealed pocket stays exactly black at minimum brightness 0 under every
//!   kind of level curve, with entry 0 set above 0.
//! - Stepped light renders the lamp room and the overhang, as smoothed light
//!   does.
//! - A slot given to another chunk the frame after it's freed never shows the
//!   other chunk's light: two lamps in rooms carved deep in the rock switch on
//!   and off in turn, so their chunks take and give back the same slot, while
//!   every frame of the dark room beside one of them stays black.
//!
//! The lamp room's checks run with sky light off because the room spans a
//! chunk boundary with more open columns than a chunk's walk can settle, so
//! its lower chunk shows direct sky until a sky probe disproves it.
//!
//! It exits with a failure if a check fails. With a directory after `check`,
//! it also saves each picture there. `check-sky-off`, `check-sky-on`,
//! `check-sky-openness`, `check-fill-paths` and `check-fill-openness` each run
//! one part of the checks.
//!
//! In the browser, `check` can't start processes of its own, so each part
//! runs from its own page load: `?check-sky-off`, `?check-sky-on`,
//! `?check-sky-openness`, `?check-fill-paths` and `?check-fill-openness`. Each
//! logs `PASS` or `FAIL` per check and a last line saying whether all passed.
//!
//! With the argument `budget`, it measures what voxel edits cost: once the
//! world has settled, it digs single voxels and then a ball of radius 8 at the
//! surface, and logs for each how many frames its light took to land and
//! settle, and the most time relights took in one frame. On WASM relights run
//! on the main thread, within the world's `wasm_relight_budget` each frame.
//!
//! A `sky_reach=N` argument sets the world's sky reach, 30 by default on
//! native and 16 on WASM, as a WASM application should use.
//!
//! On WASM it takes its arguments from the page address's query, split at
//! `&`, and draws into the page's canvas whose id is `bevy`. Build it with
//! `cargo test --no-run --release --features light_propagation --test
//! light_render --target wasm32-unknown-unknown`, run `wasm-bindgen --target
//! web` on the `light_render-*.wasm` file in
//! `target/wasm32-unknown-unknown/release/deps`, and serve a page that loads
//! it. Its output goes to the browser's console.

use bevy::{
    camera::RenderTarget,
    core_pipeline::tonemapping::{DebandDither, Tonemapping},
    light::CascadeShadowConfigBuilder,
    prelude::*,
    render::{
        render_resource::{TextureFormat, TextureUsages},
        view::screenshot::{save_to_disk, Screenshot, ScreenshotCaptured},
    },
    window::ExitCondition,
};
use bevy_voxel_world::prelude::*;
use noise::{NoiseFn, Perlin};

#[path = "../benches/common/fill_worlds.rs"]
mod fill_worlds;

use fill_worlds::FILL_SCENES;

/// Writes a line of the checks' output: to standard output on native, and to
/// the browser's console on WASM.
macro_rules! report {
    ($($line:tt)*) => {{
        #[cfg(not(target_arch = "wasm32"))]
        println!($($line)*);
        #[cfg(target_arch = "wasm32")]
        info!($($line)*);
    }};
}

const GRASS: u8 = 0;
const DIRT: u8 = 1;
const STONE: u8 = 2;
const LAMP: u8 = 3;
const DIM_LAMP: u8 = 4;

/// The level the dim lamp is declared at.
const DIM_LEVEL: f32 = 8.0;

// --- The scene ---------------------------------------------------------------

/// Noise caves stop here, and everything below is rock.
const CAVE_FLOOR: i32 = -480;

/// A box, minimum inclusive and maximum exclusive.
#[derive(Clone, Copy)]
struct Bounds {
    min: [i32; 3],
    max: [i32; 3],
}

impl Bounds {
    const fn new(min: [i32; 3], max: [i32; 3]) -> Self {
        Self { min, max }
    }

    fn contains(&self, x: i32, y: i32, z: i32) -> bool {
        (0..3).all(|axis| {
            let value = [x, y, z][axis];
            value >= self.min[axis] && value < self.max[axis]
        })
    }
}

struct Ellipsoid {
    centre: [f64; 3],
    radii: [f64; 3],
}

impl Ellipsoid {
    fn contains(&self, x: i32, y: i32, z: i32) -> bool {
        let d = |value: i32, axis: usize| {
            (value as f64 + 0.5 - self.centre[axis]) / self.radii[axis]
        };
        let (dx, dy, dz) = (d(x, 0), d(y, 1), d(z, 2));
        dx * dx + dy * dy + dz * dz < 1.0
    }
}

/// A disc of ground at one height, blended into the rolling hills.
struct Plateau {
    x: f64,
    z: f64,
    flat: f64,
    blend: f64,
    height: f64,
}

/// A cavern under a hill, lit from above through a shaft one voxel wide.
const SHAFT: (i32, i32) = (-320, 0);
const SHAFT_TOP: i32 = 120;
const CAVERN_ROOF: i32 = SHAFT_TOP - 100;
const CAVERN: Ellipsoid = Ellipsoid {
    centre: [-319.5, -10.0, 0.5],
    radii: [70.0, 30.0, 70.0],
};

/// A sealed pocket two voxels below the grass.
const POCKET_TOP: i32 = 48;
const POCKET: Bounds =
    Bounds::new([-326, POCKET_TOP - 8, 122], [-314, POCKET_TOP - 2, 134]);

/// A sealed room lit only by its lamps.
const LAMP_ROOM: Bounds = Bounds::new([300, -40, -340], [340, -20, -300]);
/// The lamps on its floor.
const BRIGHT_LAMP_AT: IVec3 = IVec3::new(320, -41, -320);
const DIM_LAMP_AT: IVec3 = IVec3::new(332, -41, -310);

/// An overhang: a slab sticking out of a mesa.
const MESA: Bounds = Bounds::new([-48, 32, 290], [48, 84, 340]);
const SLAB: Bounds = Bounds::new([-48, 72, 340], [48, 84, 404]);

const PLATEAUS: [Plateau; 4] = [
    Plateau {
        x: -319.5,
        z: 0.5,
        flat: 20.0,
        blend: 120.0,
        height: SHAFT_TOP as f64,
    },
    Plateau {
        x: 320.5,
        z: -320.5,
        flat: 64.0,
        blend: 150.0,
        height: 40.0,
    },
    Plateau {
        x: 0.0,
        z: 360.0,
        flat: 90.0,
        blend: 170.0,
        height: 32.0,
    },
    // Last, so that it wins inside the cavern hill's blend.
    Plateau {
        x: -320.0,
        z: 128.0,
        flat: 16.0,
        blend: 56.0,
        height: POCKET_TOP as f64,
    },
];

/// No noise caves here, so that the placed spaces stay as sealed as placed.
const QUIET: [Bounds; 4] = [
    Bounds::new([-406, -56, -86], [-234, 36, 86]),
    Bounds::new([-336, CAVERN_ROOF, -16], [-304, SHAFT_TOP + 8, 16]),
    Bounds::new([-342, 24, 112], [-298, 60, 144]),
    Bounds::new([284, -56, -356], [356, -4, -284]),
];

/// What voxel is where: a pure function of position.
struct Scene {
    hills: Perlin,
    region: Perlin,
    big: Perlin,
    worm_a: Perlin,
    worm_b: Perlin,
}

impl Scene {
    fn new() -> Self {
        Self {
            hills: Perlin::new(11),
            region: Perlin::new(19),
            big: Perlin::new(23),
            worm_a: Perlin::new(37),
            worm_b: Perlin::new(41),
        }
    }

    /// The height of the ground: voxels below it are underground.
    fn surface(&self, x: i32, z: i32) -> i32 {
        let (fx, fz) = (x as f64, z as f64);
        let mut height = 32.0
            + self.hills.get([fx / 160.0, fz / 160.0]) * 18.0
            + self.hills.get([fx / 47.0, fz / 47.0]) * 4.0;
        for plateau in &PLATEAUS {
            let distance = (fx - plateau.x).hypot(fz - plateau.z);
            let t = ((distance - plateau.flat) / (plateau.blend - plateau.flat))
                .clamp(0.0, 1.0);
            let t = t * t * (3.0 - 2.0 * t);
            height = plateau.height + (height - plateau.height) * t;
        }
        height.floor() as i32
    }

    fn noise_cave(&self, x: i32, y: i32, z: i32) -> bool {
        let (fx, fy, fz) = (x as f64, y as f64, z as f64);
        if self.region.get([fx / 220.0, fy / 160.0, fz / 220.0]) < -0.05 {
            return false;
        }
        if self.big.get([fx / 72.0, fy / 48.0, fz / 72.0]) > 0.42 {
            return true;
        }
        let at = [fx / 36.0, fy / 28.0, fz / 36.0];
        self.worm_a.get(at).abs() < 0.04 && self.worm_b.get(at).abs() < 0.04
    }

    fn open(&self, x: i32, y: i32, z: i32, surface: i32) -> bool {
        if y >= surface {
            return !(MESA.contains(x, y, z) || SLAB.contains(x, y, z));
        }
        if (x == SHAFT.0 && z == SHAFT.1 && y >= CAVERN_ROOF - 1)
            || CAVERN.contains(x, y, z)
            || POCKET.contains(x, y, z)
            || LAMP_ROOM.contains(x, y, z)
        {
            return true;
        }
        y >= CAVE_FLOOR
            && !QUIET.iter().any(|quiet| quiet.contains(x, y, z))
            && self.noise_cave(x, y, z)
    }

    fn voxel(&self, position: IVec3) -> WorldVoxel {
        let IVec3 { x, y, z } = position;
        let surface = self.surface(x, z);
        if self.open(x, y, z, surface) {
            return WorldVoxel::Air;
        }
        WorldVoxel::Solid(if position == BRIGHT_LAMP_AT {
            LAMP
        } else if position == DIM_LAMP_AT {
            DIM_LAMP
        } else if y >= surface {
            STONE
        } else if y + 1 >= surface {
            GRASS
        } else if y + 4 >= surface {
            DIRT
        } else {
            STONE
        })
    }
}

// --- The world ---------------------------------------------------------------

/// How far apart along x the sky fill worlds lie for their pixel checks.
const FILL_WORLD_SPACING: i32 = 4096;

/// Where the sky fill world `index` of `FILL_SCENES` puts its origin: in the
/// middle of its stretch of x, so the worlds beside it are out of reach of
/// its pixel checks' cameras.
fn fill_world_at(index: usize) -> IVec3 {
    IVec3::X * (index as i32 * FILL_WORLD_SPACING + FILL_WORLD_SPACING / 2)
}

/// The voxel at `position` in the sky fill worlds, laid out one after the
/// other along x from 0, grass on top of stone.
fn fill_world_voxel(position: IVec3) -> WorldVoxel {
    let stretch = position.x.div_euclid(FILL_WORLD_SPACING);
    let index = stretch.rem_euclid(FILL_SCENES.len() as i32) as usize;
    let local =
        position - IVec3::X * (stretch * FILL_WORLD_SPACING + FILL_WORLD_SPACING / 2);
    let scene = FILL_SCENES[index];
    if !scene.solid(local) {
        WorldVoxel::Air
    } else if scene.solid(local + IVec3::Y) {
        WorldVoxel::Solid(STONE)
    } else {
        WorldVoxel::Solid(GRASS)
    }
}

#[derive(Resource, Clone)]
struct CaveWorld {
    /// For the pixel checks: with sky light, but no sky fill, or with no sky
    /// light.
    check_sky: Option<bool>,
    /// For the pixel checks: with sky light and sky fill under sky openness.
    check_openness: bool,
    /// For the pixel checks of the sky fill worlds: with sky light and sky
    /// fill under this model, over the worlds laid out by [`fill_world_at`]
    /// in place of the caves.
    check_fill: Option<SkyFill>,
    /// How far sky light spreads past where it falls.
    sky_reach: u8,
    /// How light looks. The pixel checks change it while running.
    look: LightLook,
    /// How many levels light holds. The pixel checks change it while running.
    precision: LightPrecision,
}

impl Default for CaveWorld {
    fn default() -> Self {
        Self {
            check_sky: None,
            check_openness: false,
            check_fill: None,
            sky_reach: sky_reach(),
            look: LightLook::default(),
            precision: LightPrecision::default(),
        }
    }
}

/// The sky reach from a `sky_reach=N` argument, else 16 on WASM and the
/// default of 30 elsewhere.
fn sky_reach() -> u8 {
    arguments()
        .iter()
        .find_map(|argument| argument.strip_prefix("sky_reach=")?.parse().ok())
        .unwrap_or(if cfg!(target_arch = "wasm32") { 16 } else { 30 })
}

/// The arguments: those on the command line after the program's
/// name, or on WASM the page address's query split at `&`.
fn arguments() -> Vec<String> {
    #[cfg(not(target_arch = "wasm32"))]
    let arguments = std::env::args().skip(1).collect();
    #[cfg(target_arch = "wasm32")]
    let arguments = web_sys::window()
        .and_then(|window| window.location().search().ok())
        .unwrap_or_default()
        .trim_start_matches('?')
        .split('&')
        .filter(|argument| !argument.is_empty())
        .map(String::from)
        .collect();
    arguments
}

/// The plugins of an app with no window of its own on native. In the browser
/// the app draws into the page's canvas with the id `bevy`, since WebGL2
/// needs one to render at all.
fn plugins_without_window() -> bevy::app::PluginGroupBuilder {
    #[cfg(not(target_arch = "wasm32"))]
    let plugins = DefaultPlugins
        .set(WindowPlugin {
            primary_window: None,
            exit_condition: ExitCondition::DontExit,
            ..default()
        })
        .disable::<bevy::winit::WinitPlugin>()
        .add(bevy::app::ScheduleRunnerPlugin::run_loop(
            std::time::Duration::from_secs_f64(1.0 / 60.0),
        ));
    #[cfg(target_arch = "wasm32")]
    let plugins = DefaultPlugins.set(WindowPlugin {
        primary_window: Some(Window {
            canvas: Some("#bevy".into()),
            ..default()
        }),
        exit_condition: ExitCondition::DontExit,
        ..default()
    });
    plugins
}

impl VoxelWorldConfig for CaveWorld {
    type MaterialIndex = u8;
    type ChunkUserBundle = ();

    fn spawning_distance(&self) -> u32 {
        8
    }

    fn voxel_lookup_delegate(&self) -> VoxelLookupDelegate<Self::MaterialIndex> {
        if self.check_fill.is_some() {
            return Box::new(|_, _, _| {
                Box::new(|position, _| fill_world_voxel(position))
            });
        }
        Box::new(|_, _, _| {
            let scene = Scene::new();
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
        let light = LightPropagation {
            sky_reach: self.sky_reach,
            look: self.look.clone(),
            precision: self.precision,
            ..default()
        };
        Some(match self.check_sky {
            _ if self.check_fill.is_some() => LightPropagation {
                sky_fill: self.check_fill.clone(),
                ..light
            },
            _ if self.check_openness => LightPropagation {
                sky_fill: Some(SkyFill::Openness(OpenFill::default())),
                ..light
            },
            None => light,
            Some(true) => LightPropagation {
                sky_fill: None,
                ..light
            },
            Some(false) => LightPropagation {
                sky: Sky::Off,
                sky_fill: None,
                ..light
            },
        })
    }

    fn emissive_materials(&self) -> Vec<(Self::MaterialIndex, Emission)> {
        vec![
            (LAMP, Emission::level(15.0)),
            (DIM_LAMP, Emission::level(DIM_LEVEL)),
        ]
    }
}

// --- The cameras -------------------------------------------------------------

struct FixedCamera {
    position: Vec3,
    look_at: Vec3,
}

const fn camera(position: [f32; 3], look_at: [f32; 3]) -> FixedCamera {
    FixedCamera {
        position: Vec3::from_array(position),
        look_at: Vec3::from_array(look_at),
    }
}

const LAMP_ROOM_CAMERA: FixedCamera =
    camera([306.0, -30.0, -332.0], [320.5, -40.0, -319.5]);
const POCKET_CAMERA: FixedCamera = camera([-324.5, 43.0, 124.0], [-316.0, 42.0, 132.0]);

/// Looks under the slab sticking out of the mesa, at the mesa's shaded face.
const OVERHANG_CAMERA: FixedCamera = camera([0.0, 44.0, 396.0], [0.0, 42.0, 340.0]);
/// Looks at the mesa's sunlit face and the ground below it.
const SUNLIT_FACE_CAMERA: FixedCamera = camera([20.0, 60.0, 250.0], [0.0, 40.0, 290.0]);
/// Looks past the slab from the side the sun shines in from, at ground whose
/// sky light falls off beside the slab and the cave mouths beyond it.
const DRIP_LINE_CAMERA: FixedCamera = camera([90.0, 50.0, 372.0], [30.0, 34.0, 372.0]);

fn transform_of(camera: &FixedCamera) -> Transform {
    Transform::from_translation(camera.position).looking_at(camera.look_at, Vec3::Y)
}

fn add_lights(commands: &mut Commands) {
    commands.spawn((
        DirectionalLight {
            color: Color::srgb(0.98, 0.95, 0.82),
            shadow_maps_enabled: true,
            ..default()
        },
        Transform::from_xyz(0.0, 0.0, 0.0)
            .looking_at(Vec3::new(-0.15, -0.1, 0.15), Vec3::Y),
        CascadeShadowConfigBuilder::default().build(),
    ));
    commands.insert_resource(GlobalAmbientLight {
        color: Color::srgb(0.98, 0.95, 0.82),
        brightness: 600.0,
        affects_lightmapped_meshes: true,
    });
}

fn main() {
    match arguments().first().map(String::as_str) {
        None | Some("check") => check(),
        Some("check-sky-off") => check_part(false, false),
        Some("check-sky-on") => check_part(true, false),
        Some("check-sky-openness") => check_part(true, true),
        Some("check-fill-paths") => {
            check_fill_part("paths", SkyFill::Paths(PathFill::default()))
        }
        Some("check-fill-openness") => {
            check_fill_part("openness", SkyFill::Openness(OpenFill::default()))
        }
        Some("budget") => budget(),
        Some(other) => {
            eprintln!(
                "unknown argument {other:?}: expected check, check-sky-off, check-sky-on, \
                 check-sky-openness, check-fill-paths, check-fill-openness or budget"
            );
            std::process::exit(2);
        }
    }
}

// --- The pixel checks --------------------------------------------------------

const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;

/// One check: whether sky light is on, where the camera stands, whether the
/// sun has a shadow map, and what it asserts of the picture, given the
/// picture of an earlier check.
struct Check {
    name: &'static str,
    file: &'static str,
    sky: bool,
    camera: &'static FixedCamera,
    shadow_maps: bool,
    /// The file of the earlier check whose picture the judge gets.
    earlier: Option<&'static str>,
    judge: Judge,
    /// Whether it switches the slot lamps while judging a picture each frame.
    switches_lamps: bool,
    /// The look the world has for the picture.
    look: fn() -> LightLook,
    /// Whether it takes the picture on the frame after setting the look,
    /// without waiting for the world to settle. The camera and the sun must
    /// be as the check before left them.
    next_frame: bool,
    /// How many levels light holds for the picture.
    precision: LightPrecision,
}

type Judge =
    fn(&Image, &Camera, &GlobalTransform, Option<&Image>) -> Result<String, String>;

/// A check with the default look that waits for the world to settle.
const fn settled_check(
    name: &'static str,
    file: &'static str,
    sky: bool,
    camera: &'static FixedCamera,
    shadow_maps: bool,
    earlier: Option<&'static str>,
    judge: Judge,
) -> Check {
    Check {
        name,
        file,
        sky,
        camera,
        shadow_maps,
        earlier,
        judge,
        switches_lamps: false,
        look: LightLook::default,
        next_frame: false,
        precision: LightPrecision::Steps16,
    }
}

/// A check that sets `look`, with the camera and the sun as the check before
/// left them, and judges the picture of the next frame.
const fn look_check(
    name: &'static str,
    file: &'static str,
    sky: bool,
    camera: &'static FixedCamera,
    earlier: Option<&'static str>,
    judge: Judge,
    look: fn() -> LightLook,
) -> Check {
    Check {
        name,
        file,
        sky,
        camera,
        shadow_maps: true,
        earlier,
        judge,
        switches_lamps: false,
        look,
        next_frame: true,
        precision: LightPrecision::Steps16,
    }
}

/// The look with both falloffs and the sun following `curve`, with its entry
/// for level 0 set to 1, which the crate reads as 0, and minimum brightness 0.
fn curve_look(curve: LevelCurve) -> LightLook {
    let mut table = [0.0; 16];
    for (level, entry) in table.iter_mut().enumerate() {
        *entry = curve.at(level as f32);
    }
    table[0] = 1.0;
    let curve = LevelCurve::from_table(table);
    LightLook {
        sky_falloff: curve.clone(),
        voxel_falloff: curve.clone(),
        sun: SunGate::ByLevel(curve),
        minimum_brightness: 0.0,
        ..default()
    }
}

const CHECKS: [Check; 23] = [
    settled_check(
        "a lamp declared below the top level still glows, scaled by its level",
        "lamp_room.png",
        false,
        &LAMP_ROOM_CAMERA,
        true,
        None,
        dim_lamp_glows,
    ),
    look_check(
        "a change of voxel brightness shows on the next frame",
        "look_voxel_brightness.png",
        false,
        &LAMP_ROOM_CAMERA,
        Some("lamp_room.png"),
        differs_from_earlier,
        || LightLook {
            voxel_brightness: 250.0,
            ..default()
        },
    ),
    look_check(
        "a change of voxel falloff shows on the next frame",
        "look_voxel_falloff.png",
        false,
        &LAMP_ROOM_CAMERA,
        Some("lamp_room.png"),
        differs_from_earlier,
        || LightLook {
            voxel_falloff: LevelCurve::linear(),
            ..default()
        },
    ),
    look_check(
        "stepped light renders the lamp room, on the next frame",
        "look_stepped.png",
        false,
        &LAMP_ROOM_CAMERA,
        Some("lamp_room.png"),
        differs_from_earlier,
        || LightLook {
            smoothing: LightSmoothing::Stepped,
            ..default()
        },
    ),
    look_check(
        "a minimum brightness shows on the next frame",
        "look_minimum_brightness.png",
        false,
        &LAMP_ROOM_CAMERA,
        Some("lamp_room.png"),
        differs_from_earlier,
        || LightLook {
            minimum_brightness: 0.05,
            ..default()
        },
    ),
    look_check(
        "a change of glow shows on the next frame, still scaled by each lamp's level",
        "look_emissive_glow.png",
        false,
        &LAMP_ROOM_CAMERA,
        Some("lamp_room.png"),
        glow_changes_and_scales,
        || LightLook {
            emissive_glow: 2.0,
            ..default()
        },
    ),
    settled_check(
        "a sun with no shadow map leaves propagated shade as it was",
        "lamp_room_no_shadow_map.png",
        false,
        &LAMP_ROOM_CAMERA,
        false,
        Some("lamp_room.png"),
        matches_earlier,
    ),
    settled_check(
        "a sealed pocket with no light in reach is exactly black",
        "sealed_pocket.png",
        true,
        &POCKET_CAMERA,
        true,
        None,
        pocket_is_black,
    ),
    look_check(
        "a sealed pocket stays exactly black under a flat curve",
        "pocket_flat_curve.png",
        true,
        &POCKET_CAMERA,
        None,
        pocket_is_black,
        || curve_look(LevelCurve::from_fn(|_| 1.0)),
    ),
    look_check(
        "a sealed pocket stays exactly black under the linear curve",
        "pocket_linear_curve.png",
        true,
        &POCKET_CAMERA,
        None,
        pocket_is_black,
        || curve_look(LevelCurve::linear()),
    ),
    look_check(
        "a sealed pocket stays exactly black under the gamma curve",
        "pocket_gamma_curve.png",
        true,
        &POCKET_CAMERA,
        None,
        pocket_is_black,
        || curve_look(LevelCurve::gamma(2.2)),
    ),
    look_check(
        "a sealed pocket stays exactly black under a slow exponential curve, stepped",
        "pocket_exponential_curve.png",
        true,
        &POCKET_CAMERA,
        None,
        pocket_is_black,
        || LightLook {
            smoothing: LightSmoothing::Stepped,
            ao: LightAo::BaseColor,
            ..curve_look(LevelCurve::exponential(0.95))
        },
    ),
    // After the look checks, which judge the next frame and so can't follow a
    // change of precision: chunks show flat light until they relight.
    Check {
        precision: LightPrecision::Steps256,
        ..settled_check(
            "at 256 levels a sealed pocket is exactly black",
            "sealed_pocket_256.png",
            true,
            &POCKET_CAMERA,
            true,
            None,
            pocket_is_black,
        )
    },
    Check {
        name: "a slot given to another chunk the frame after it's freed never shows the other chunk's light",
        file: "slot_room.png",
        sky: true,
        camera: &SLOT_ROOM_CAMERA,
        shadow_maps: true,
        earlier: None,
        judge: pocket_is_black,
        switches_lamps: true,
        look: LightLook::default,
        next_frame: false,
        precision: LightPrecision::Steps16,
    },
    settled_check(
        "a sun with no shadow map leaves a sealed pocket exactly black",
        "sealed_pocket_no_shadow_map.png",
        true,
        &POCKET_CAMERA,
        false,
        None,
        pocket_is_black,
    ),
    settled_check(
        "smoothed light renders the overhang",
        "overhang.png",
        true,
        &OVERHANG_CAMERA,
        true,
        None,
        renders,
    ),
    look_check(
        "stepped light renders the overhang, on the next frame",
        "overhang_stepped.png",
        true,
        &OVERHANG_CAMERA,
        Some("overhang.png"),
        renders_and_differs,
        || LightLook {
            smoothing: LightSmoothing::Stepped,
            ..default()
        },
    ),
    look_check(
        "turning ambient occlusion off shows on the next frame",
        "overhang_ao_off.png",
        true,
        &OVERHANG_CAMERA,
        Some("overhang.png"),
        renders_and_differs,
        || LightLook {
            ao: LightAo::Off,
            ..default()
        },
    ),
    look_check(
        "a change of sky falloff shows on the next frame",
        "overhang_sky_falloff.png",
        true,
        &OVERHANG_CAMERA,
        Some("overhang.png"),
        renders_and_differs,
        || LightLook {
            sky_falloff: LevelCurve::gamma(2.2),
            ..default()
        },
    ),
    settled_check(
        "smoothed light renders the sunlit face of the mesa",
        "sunlit_face.png",
        true,
        &SUNLIT_FACE_CAMERA,
        true,
        None,
        renders,
    ),
    look_check(
        "ambient occlusion on the base colour shows on the next frame",
        "sunlit_face_ao_base_color.png",
        true,
        &SUNLIT_FACE_CAMERA,
        Some("sunlit_face.png"),
        renders_and_differs,
        || LightLook {
            ao: LightAo::BaseColor,
            ..default()
        },
    ),
    settled_check(
        "smoothed light renders the overhang's drip line",
        "drip_line.png",
        true,
        &DRIP_LINE_CAMERA,
        true,
        None,
        renders,
    ),
    look_check(
        "a sun following sky light by level shows on the next frame",
        "drip_line_sun_by_level.png",
        true,
        &DRIP_LINE_CAMERA,
        Some("drip_line.png"),
        renders_and_differs,
        || LightLook {
            sun: SunGate::ByLevel(LevelCurve::linear()),
            ..default()
        },
    ),
];

/// The checks run with sky fill on under sky openness.
const OPENNESS_CHECKS: [Check; 1] = [settled_check(
    "a sealed pocket under sky openness is exactly black",
    "sealed_pocket_openness.png",
    true,
    &POCKET_CAMERA,
    true,
    None,
    pocket_is_black,
)];

/// The pictures of the checks so far, by file.
#[derive(Resource, Default)]
struct Pictures(std::collections::HashMap<&'static str, Image>);

// --- The slot check's rooms --------------------------------------------------
//
// Below the cave floor everything is rock. The check carves two rooms there,
// in the same place in two chunks that aren't neighbours. The dark room has
// no lamp, and its chunk has a lamp of its own in a sealed cell one voxel big
// behind its wall, so the chunk needs a slot only while that lamp is on. The
// lit room's lamp lights its room, so its chunk needs a slot only while that
// lamp is on. The lamps switch in turn, so each chunk takes the slot the
// other just gave back. Had the dark room's chunk read the lit room's bytes
// for a frame, its walls would show the lamp's light.

/// The chunk holding the dark room.
const DARK_CHUNK: IVec3 = IVec3::new(0, -20, 0);
/// The chunk holding the lit room, three chunks along x.
const LIT_CHUNK: IVec3 = IVec3::new(3, -20, 0);
/// Each room, in its chunk's voxels.
const ROOM: Bounds = Bounds::new([4, 4, 4], [28, 28, 28]);
/// The lamp in the lit room, in its chunk's voxels.
const LIT_LAMP: IVec3 = IVec3::new(16, 4, 16);
/// The sealed cell behind the dark room's wall, and its lamp.
const SEALED_CELL: IVec3 = IVec3::new(1, 1, 1);
const SEALED_LAMP: IVec3 = IVec3::new(2, 1, 1);
/// Frames between switching the lamps, and frames judged in all.
const SWITCH_EVERY: u32 = 3;
const SWITCHED_FRAMES: u32 = 90;

const SLOT_ROOM_CAMERA: FixedCamera = camera([16.0, -624.0, 6.0], [16.0, -626.0, 27.0]);

fn carve_slot_rooms(voxel_world: &mut VoxelWorld<CaveWorld>) {
    for chunk in [DARK_CHUNK, LIT_CHUNK] {
        for x in ROOM.min[0]..ROOM.max[0] {
            for y in ROOM.min[1]..ROOM.max[1] {
                for z in ROOM.min[2]..ROOM.max[2] {
                    voxel_world
                        .set_voxel(chunk * 32 + IVec3::new(x, y, z), WorldVoxel::Air);
                }
            }
        }
    }
    voxel_world.set_voxel(DARK_CHUNK * 32 + SEALED_CELL, WorldVoxel::Air);
}

/// Switches on the lit room's lamp and off the sealed one, or the other way.
fn switch_lamps(voxel_world: &mut VoxelWorld<CaveWorld>, lit_room: bool) {
    let lamp = |on: bool| {
        if on {
            WorldVoxel::Solid(LAMP)
        } else {
            WorldVoxel::Solid(STONE)
        }
    };
    voxel_world.set_voxel(
        LIT_CHUNK * 32 + LIT_LAMP,
        if lit_room {
            lamp(true)
        } else {
            WorldVoxel::Air
        },
    );
    voxel_world.set_voxel(DARK_CHUNK * 32 + SEALED_LAMP, lamp(!lit_room));
}

#[derive(Resource)]
struct CheckPlan {
    target: Handle<Image>,
    /// The checks this run makes, in order.
    checks: Vec<&'static Check>,
    next: usize,
    moved_at_frame: u32,
    /// The last frame on which a chunk wasn't Ready or light hadn't settled.
    last_busy_frame: u32,
    frame: u32,
    waiting: bool,
    failed: bool,
    /// Frames judged so far while switching the slot lamps.
    switched: u32,
    /// Pictures of the switching frames not yet judged, and how many failed.
    shots_out: u32,
    shots_failed: u32,
    /// Where to save the pictures, if anywhere.
    directory: Option<std::path::PathBuf>,
    /// The look the current check wants, which `follow_plan_look` gives the
    /// world.
    look: LightLook,
    /// The precision the current check wants, which `follow_plan_look` gives
    /// the world.
    precision: LightPrecision,
}

/// Runs the checks with sky light off, then with it on, then with sky fill
/// under sky openness, then those of the sky fill worlds under each model,
/// each in a page load of its own in the browser.
#[cfg(target_arch = "wasm32")]
fn check() {
    error!(
        "in the browser, run ?check-sky-off, ?check-sky-on, ?check-sky-openness, \
         ?check-fill-paths and ?check-fill-openness, each in a page load"
    );
}

/// Runs the checks with sky light off, then with it on, then with sky fill
/// under sky openness, then those of the sky fill worlds under each model,
/// each in a process of its own, since a world's sky rule is read once.
#[cfg(not(target_arch = "wasm32"))]
fn check() {
    let mut failed = false;
    for sky in [
        "sky-off",
        "sky-on",
        "sky-openness",
        "fill-paths",
        "fill-openness",
    ] {
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .arg(format!("check-{sky}"))
            .args(std::env::args().skip(2))
            .status()
            .unwrap();
        failed |= !status.success();
    }
    if failed {
        std::process::exit(1);
    }
}

/// Runs the checks with sky light on or off, or those with sky fill under sky
/// openness.
fn check_part(sky: bool, openness: bool) {
    let checks: Vec<&'static Check> = if openness {
        OPENNESS_CHECKS.iter().collect()
    } else {
        CHECKS.iter().filter(|check| check.sky == sky).collect()
    };
    run_check_app(
        CaveWorld {
            check_sky: Some(sky),
            check_openness: openness,
            ..default()
        },
        checks,
    );
}

/// Runs the checks of the sky fill worlds with sky fill under `fill`, named
/// `model` in the checks and their files.
fn check_fill_part(model: &'static str, fill: SkyFill) {
    run_check_app(
        CaveWorld {
            check_sky: Some(true),
            check_fill: Some(fill),
            ..default()
        },
        fill_checks(model),
    );
}

fn run_check_app(world: CaveWorld, checks: Vec<&'static Check>) {
    App::new()
        .add_plugins(plugins_without_window())
        .add_plugins(VoxelWorldPlugin::with_config(world))
        .init_resource::<Pictures>()
        .add_systems(
            Startup,
            move |commands: Commands, images: ResMut<Assets<Image>>| {
                setup_check(commands, images, checks.clone())
            },
        )
        .add_systems(Update, (run_checks, follow_plan_look).chain())
        .run();
}

/// A check of each shot of each sky fill world under the model named
/// `model`: one asserting exact black where the shot sees only a space
/// sealed from the sky, and one taking the picture for a person to judge
/// elsewhere.
fn fill_checks(model: &'static str) -> Vec<&'static Check> {
    let mut checks = Vec::new();
    for (index, scene) in FILL_SCENES.into_iter().enumerate() {
        let origin = fill_world_at(index).as_vec3();
        for shot in scene.shots() {
            let camera: &'static FixedCamera = Box::leak(Box::new(FixedCamera {
                position: shot.at + origin,
                look_at: shot.target + origin,
            }));
            let check = settled_check(
                leak(format!(
                    "{} {} under {model}{}",
                    scene.name(),
                    shot.name,
                    if shot.sealed { " is exactly black" } else { "" }
                )),
                leak(format!("fill_{model}_{}_{}.png", scene.name(), shot.name)),
                true,
                camera,
                true,
                None,
                if shot.sealed {
                    pocket_is_black
                } else {
                    lit_share
                },
            );
            checks.push(&*Box::leak(Box::new(check)));
        }
    }
    checks
}

fn leak(text: String) -> &'static str {
    Box::leak(text.into_boxed_str())
}

/// Passes, noting how many of `image`'s pixels aren't black and the mean of
/// their brightest channels, for a picture a person judges.
fn lit_share(
    image: &Image,
    _: &Camera,
    _: &GlobalTransform,
    _: Option<&Image>,
) -> Result<String, String> {
    let data = image.data.as_ref().ok_or("no pixels")?;
    let pixels = data.as_chunks::<4>().0;
    let brightest: Vec<u8> = pixels
        .iter()
        .map(|pixel| pixel[0].max(pixel[1]).max(pixel[2]))
        .collect();
    let lit = brightest.iter().filter(|level| **level > 0).count();
    let mean =
        brightest.iter().map(|level| *level as f64).sum::<f64>() / pixels.len() as f64;
    Ok(format!(
        "{lit} of {} pixels lit, mean brightest channel {mean:.1}",
        pixels.len()
    ))
}

/// Gives the world the look the current check wants.
fn follow_plan_look(plan: Res<CheckPlan>, mut world: ResMut<CaveWorld>) {
    if world.look != plan.look {
        world.look = plan.look.clone();
    }
    if world.precision != plan.precision {
        world.precision = plan.precision;
    }
}

fn setup_check(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    checks: Vec<&'static Check>,
) {
    let mut image =
        Image::new_target_texture(WIDTH, HEIGHT, TextureFormat::Rgba8UnormSrgb, None);
    image.texture_descriptor.usage |= TextureUsages::COPY_SRC;
    let target = images.add(image);
    commands.spawn((
        Camera3d::default(),
        RenderTarget::Image(target.clone().into()),
        Tonemapping::None,
        DebandDither::Disabled,
        transform_of(checks[0].camera),
        VoxelWorldCamera::<CaveWorld>::default(),
    ));
    add_lights(&mut commands);
    commands.insert_resource(CheckPlan {
        target,
        checks,
        next: 0,
        moved_at_frame: 0,
        last_busy_frame: 0,
        frame: 0,
        waiting: false,
        failed: false,
        switched: 0,
        shots_out: 0,
        shots_failed: 0,
        directory: arguments()
            .get(1)
            .filter(|argument| !argument.contains('='))
            .map(Into::into),
        look: LightLook::default(),
        precision: LightPrecision::Steps16,
    });
}

/// Moves to each check's camera, waits for the world to settle there, and
/// judges a screenshot.
fn run_checks(
    mut commands: Commands,
    mut plan: ResMut<CheckPlan>,
    mut cameras: Query<
        (&mut Transform, &Camera, &GlobalTransform),
        With<VoxelWorldCamera<CaveWorld>>,
    >,
    mut suns: Query<&mut DirectionalLight>,
    chunks: Query<&Chunk<CaveWorld>>,
    mut voxel_world: VoxelWorld<CaveWorld>,
    mut exit: MessageWriter<AppExit>,
) {
    plan.frame += 1;
    let frame = plan.frame;
    if !voxel_world.light_is_settled()
        || chunks
            .iter()
            .any(|chunk| voxel_world.chunk_state(chunk.key()) != Some(ChunkStatus::Ready))
    {
        plan.last_busy_frame = frame;
    }
    if plan.waiting {
        return;
    }
    let Some(check) = plan.checks.get(plan.next).copied() else {
        if plan.next == plan.checks.len() {
            plan.next += 1;
            if plan.failed {
                report!("Some checks FAILED");
            } else {
                report!("All checks passed");
            }
        }
        exit.write(if plan.failed {
            AppExit::error()
        } else {
            AppExit::Success
        });
        return;
    };
    let (mut transform, camera, global) = cameras.single_mut().unwrap();
    let wanted = transform_of(check.camera);
    let shadow_maps_wrong = suns
        .iter()
        .any(|sun| sun.shadow_maps_enabled != check.shadow_maps);
    let look = (check.look)();
    if *transform != wanted
        || shadow_maps_wrong
        || plan.look != look
        || plan.precision != check.precision
    {
        *transform = wanted;
        plan.look = look;
        plan.precision = check.precision;
        for mut sun in &mut suns {
            sun.shadow_maps_enabled = check.shadow_maps;
        }
        plan.moved_at_frame = frame;
        if check.switches_lamps {
            carve_slot_rooms(&mut voxel_world);
            switch_lamps(&mut voxel_world, true);
        }
        return;
    }
    if check.switches_lamps && plan.switched > 0 {
        switch_and_judge(&mut commands, &mut plan, &mut voxel_world, check.name);
        return;
    }
    let waited = frame - plan.moved_at_frame;
    let quiet = frame - plan.last_busy_frame.max(plan.moved_at_frame);
    // A look change shows on the next frame, with no waiting.
    if !check.next_frame && !(quiet >= 60 && waited >= 120) {
        if waited >= 3600 {
            report!("FAIL {}: the world never settled", check.name);
            plan.failed = true;
            plan.next += 1;
        }
        return;
    }

    if check.switches_lamps {
        switch_and_judge(&mut commands, &mut plan, &mut voxel_world, check.name);
        return;
    }

    plan.waiting = true;
    let (camera, global) = (camera.clone(), *global);
    let (name, file, earlier, judge) =
        (check.name, check.file, check.earlier, check.judge);
    let mut screenshot = commands.spawn(Screenshot::image(plan.target.clone()));
    if let Some(directory) = &plan.directory {
        std::fs::create_dir_all(directory).unwrap();
        screenshot.observe(save_to_disk(directory.join(check.file)));
    }
    screenshot.observe(
        move |captured: On<ScreenshotCaptured>,
              mut plan: ResMut<CheckPlan>,
              mut pictures: ResMut<Pictures>| {
            let earlier = earlier.and_then(|file| pictures.0.get(file));
            match judge(&captured.image, &camera, &global, earlier) {
                Ok(note) => report!("PASS {name}: {note}"),
                Err(note) => {
                    report!("FAIL {name}: {note}");
                    plan.failed = true;
                }
            }
            pictures.0.insert(file, captured.image.clone());
            plan.waiting = false;
            plan.next += 1;
        },
    );
}

/// Whether `image` matches the earlier check's picture, within one step of
/// rounding in each channel.
fn matches_earlier(
    image: &Image,
    _: &Camera,
    _: &GlobalTransform,
    earlier: Option<&Image>,
) -> Result<String, String> {
    let data = image.data.as_ref().ok_or("no pixels")?;
    let earlier = earlier
        .and_then(|earlier| earlier.data.as_ref())
        .ok_or("no earlier picture")?;
    let differing = data
        .iter()
        .zip(earlier)
        .filter(|(now, before)| now.abs_diff(**before) > 1)
        .count();
    if differing == 0 {
        Ok(format!("{} pixels as before", data.len() / 4))
    } else {
        Err(format!("{differing} channels differ by more than 1"))
    }
}

/// Switches the slot lamps every few frames and judges a picture of every
/// frame black, then moves on once every picture is judged.
fn switch_and_judge(
    commands: &mut Commands,
    plan: &mut CheckPlan,
    voxel_world: &mut VoxelWorld<CaveWorld>,
    name: &'static str,
) {
    if plan.switched < SWITCHED_FRAMES {
        if plan.switched.is_multiple_of(SWITCH_EVERY) {
            let lit_room = (plan.switched / SWITCH_EVERY) % 2 == 1;
            switch_lamps(voxel_world, lit_room);
        }
        plan.switched += 1;
        plan.shots_out += 1;
        let frame = plan.switched;
        commands
            .spawn(Screenshot::image(plan.target.clone()))
            .observe(
                move |captured: On<ScreenshotCaptured>, mut plan: ResMut<CheckPlan>| {
                    plan.shots_out -= 1;
                    if let Err(note) = pocket_is_black(
                        &captured.image,
                        &Camera::default(),
                        &GlobalTransform::default(),
                        None,
                    ) {
                        report!("  frame {frame}: {note}");
                        plan.shots_failed += 1;
                    }
                },
            );
        return;
    }
    if plan.shots_out > 0 {
        return;
    }
    if plan.shots_failed == 0 {
        report!("PASS {name}: {SWITCHED_FRAMES} frames black");
    } else {
        report!(
            "FAIL {name}: {} of {SWITCHED_FRAMES} frames weren't black",
            plan.shots_failed
        );
        plan.failed = true;
    }
    plan.next += 1;
}

/// Whether `image` differs from the earlier check's picture in at least 0.1% of
/// its channels by more than one step of rounding.
fn differs_from_earlier(
    image: &Image,
    _: &Camera,
    _: &GlobalTransform,
    earlier: Option<&Image>,
) -> Result<String, String> {
    let data = image.data.as_ref().ok_or("no pixels")?;
    let earlier = earlier
        .and_then(|earlier| earlier.data.as_ref())
        .ok_or("no earlier picture")?;
    let differing = data
        .iter()
        .zip(earlier)
        .filter(|(now, before)| now.abs_diff(**before) > 1)
        .count();
    let note = format!("{differing} channels differ by more than 1");
    if differing * 1000 >= data.len() {
        Ok(note)
    } else {
        Err(note)
    }
}

/// Whether at least a tenth of `image`'s pixels aren't black.
fn renders(
    image: &Image,
    _: &Camera,
    _: &GlobalTransform,
    _: Option<&Image>,
) -> Result<String, String> {
    let data = image.data.as_ref().ok_or("no pixels")?;
    let pixels = data.as_chunks::<4>().0;
    let lit = pixels
        .iter()
        .filter(|pixel| pixel[0] != 0 || pixel[1] != 0 || pixel[2] != 0)
        .count();
    let note = format!("{lit} of {} pixels lit", pixels.len());
    if lit * 10 >= pixels.len() {
        Ok(note)
    } else {
        Err(note)
    }
}

fn renders_and_differs(
    image: &Image,
    camera: &Camera,
    global: &GlobalTransform,
    earlier: Option<&Image>,
) -> Result<String, String> {
    let rendered = renders(image, camera, global, earlier)?;
    let differs = differs_from_earlier(image, camera, global, earlier)?;
    Ok(format!("{rendered}, {differs}"))
}

/// Whether the dim lamp's glow changed from the earlier picture, and it still
/// glows less than the bright lamp.
fn glow_changes_and_scales(
    image: &Image,
    camera: &Camera,
    global: &GlobalTransform,
    earlier: Option<&Image>,
) -> Result<String, String> {
    let earlier = earlier.ok_or("no earlier picture")?;
    let top = Vec3::new(0.5, 1.0, 0.5);
    let bright = brightness_at(image, camera, global, BRIGHT_LAMP_AT.as_vec3() + top)?;
    let dim = brightness_at(image, camera, global, DIM_LAMP_AT.as_vec3() + top)?;
    let dim_before = brightness_at(earlier, camera, global, DIM_LAMP_AT.as_vec3() + top)?;
    let note = format!(
        "bright lamp {bright}, dim lamp {dim}, dim lamp before the change {dim_before}"
    );
    if dim > 0 && dim < bright && dim != dim_before {
        Ok(note)
    } else {
        Err(note)
    }
}

fn pocket_is_black(
    image: &Image,
    _: &Camera,
    _: &GlobalTransform,
    _: Option<&Image>,
) -> Result<String, String> {
    let data = image.data.as_ref().ok_or("no pixels")?;
    let lit = data
        .as_chunks::<4>()
        .0
        .iter()
        .filter(|pixel| pixel[0] != 0 || pixel[1] != 0 || pixel[2] != 0)
        .count();
    if lit == 0 {
        Ok(format!("{} pixels at 0", data.len() / 4))
    } else {
        Err(format!("{lit} pixels aren't black"))
    }
}

/// The brightest channel of the pixels around where `point` shows.
fn brightness_at(
    image: &Image,
    camera: &Camera,
    global: &GlobalTransform,
    point: Vec3,
) -> Result<u8, String> {
    let data = image.data.as_ref().ok_or("no pixels")?;
    let at = camera
        .world_to_viewport(global, point)
        .map_err(|error| format!("{point} isn't in view: {error:?}"))?;
    let (x, y) = (at.x as i32, at.y as i32);
    let mut brightest = 0;
    for dy in -1..=1 {
        for dx in -1..=1 {
            let (px, py) = (
                (x + dx).clamp(0, WIDTH as i32 - 1),
                (y + dy).clamp(0, HEIGHT as i32 - 1),
            );
            let index = (py as usize * WIDTH as usize + px as usize) * 4;
            brightest = brightest
                .max(data[index])
                .max(data[index + 1])
                .max(data[index + 2]);
        }
    }
    Ok(brightest)
}

fn dim_lamp_glows(
    image: &Image,
    camera: &Camera,
    global: &GlobalTransform,
    _: Option<&Image>,
) -> Result<String, String> {
    // The middle of each lamp's top face.
    let top = Vec3::new(0.5, 1.0, 0.5);
    let bright = brightness_at(image, camera, global, BRIGHT_LAMP_AT.as_vec3() + top)?;
    let dim = brightness_at(image, camera, global, DIM_LAMP_AT.as_vec3() + top)?;
    let note = format!("bright lamp {bright}, dim lamp at level {DIM_LEVEL} {dim}");
    if dim > 16 && dim < bright {
        Ok(note)
    } else {
        Err(note)
    }
}

// --- The relight budget ------------------------------------------------------

/// Where the budget mode digs: a single voxel at the top of each column, then
/// a ball around the top of the last.
const BUDGET_SITES: [[i32; 2]; 6] = [
    [-48, -100],
    [-16, -100],
    [16, -100],
    [48, -100],
    [-32, -70],
    [32, -70],
];
/// Looks down at the budget mode's sites.
const BUDGET_CAMERA: FixedCamera = camera([0.0, 110.0, -170.0], [0.0, 40.0, -90.0]);
const BOMB_RADIUS: i32 = 8;

/// The budget mode's progress, and what it measured since the edit it waits
/// on was set.
#[derive(Resource, Default)]
struct Budget {
    frame: u32,
    /// The last frame on which a chunk wasn't Ready or light hadn't settled.
    last_busy_frame: u32,
    /// The edit being measured: the voxel whose sky light shows the edited
    /// chunk's light landed, the frame the edit was set on, and the frame its
    /// light landed on.
    edit: Option<(IVec3, u32, Option<u32>)>,
    edits_done: usize,
    /// The world's light stats as the last frame ended.
    stats: bevy_voxel_world::light_bench::LightStats,
    /// Relights started, and the most relight time and the most main-thread
    /// light time in one frame, in milliseconds.
    relights: u64,
    worst_relight_frame_ms: f64,
    worst_light_frame_ms: f64,
}

fn budget() {
    App::new()
        .add_plugins(plugins_without_window())
        .add_plugins(VoxelWorldPlugin::with_config(CaveWorld::default()))
        .init_resource::<Budget>()
        .add_systems(Startup, setup_budget)
        .add_systems(Update, dig_and_measure)
        .add_systems(Last, take_light_stats)
        .run();
}

fn setup_budget(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
    let image =
        Image::new_target_texture(WIDTH, HEIGHT, TextureFormat::Rgba8UnormSrgb, None);
    commands.spawn((
        Camera3d::default(),
        RenderTarget::Image(images.add(image).into()),
        transform_of(&BUDGET_CAMERA),
        VoxelWorldCamera::<CaveWorld>::default(),
    ));
    add_lights(&mut commands);
    report!("budget: sky reach {}", sky_reach());
}

/// The highest solid voxel of the column at `x`, `z`, below y = 200.
fn top_solid(voxel_world: &VoxelWorld<CaveWorld>, x: i32, z: i32) -> Option<IVec3> {
    (-64..200)
        .rev()
        .map(|y| IVec3::new(x, y, z))
        .find(|position| voxel_world.get_voxel(*position).is_solid())
}

/// Waits for the world to settle, then digs at each site in turn, waiting for
/// each edit's light to land and settle before the next.
fn dig_and_measure(
    mut budget: ResMut<Budget>,
    chunks: Query<&Chunk<CaveWorld>>,
    mut voxel_world: VoxelWorld<CaveWorld>,
) {
    budget.frame += 1;
    let frame = budget.frame;
    let busy = !voxel_world.light_is_settled()
        || chunks.iter().any(|chunk| {
            voxel_world.chunk_state(chunk.key()) != Some(ChunkStatus::Ready)
        });
    if busy {
        budget.last_busy_frame = frame;
    }
    if let Some((watched, set_at, landed)) = budget.edit {
        let lit = voxel_world
            .light_at(watched)
            .is_some_and(|light| light.sky > 0.0);
        let landed = landed.or(lit.then_some(frame));
        budget.edit = Some((watched, set_at, landed));
        if busy {
            return;
        }
        report!(
            "budget: edit {} landed at frame {}, settled at frame {}, {} relights, \
             at most {:.2} ms of relights and {:.2} ms of light on the main thread in \
             a frame",
            budget.edits_done,
            landed.map_or("never".into(), |landed| (landed - set_at).to_string()),
            (budget.last_busy_frame + 1).saturating_sub(set_at),
            budget.relights,
            budget.worst_relight_frame_ms,
            budget.worst_light_frame_ms,
        );
        budget.edit = None;
        budget.edits_done += 1;
        if budget.edits_done == BUDGET_SITES.len() + 1 {
            report!(
                "budget: done; the longest relight on the main thread took {:.2} ms",
                budget.stats.max_relight_ms
            );
        }
        return;
    }
    // Each edit waits for 60 quiet frames.
    if frame - budget.last_busy_frame < 60 || budget.edits_done > BUDGET_SITES.len() {
        return;
    }
    let bomb = budget.edits_done == BUDGET_SITES.len();
    let [x, z] = BUDGET_SITES[budget.edits_done.min(BUDGET_SITES.len() - 1)];
    let Some(top) = top_solid(&voxel_world, x, z) else {
        report!("budget: no solid voxel at {x}, {z}");
        budget.edits_done += 1;
        return;
    };
    // The lowest voxel dug in the column, open to the sky once dug, reads sky
    // light once the edited chunk's light lands.
    let radius = if bomb { BOMB_RADIUS } else { 0 };
    for dz in -radius..=radius {
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                let offset = IVec3::new(dx, dy, dz);
                if offset.length_squared() <= radius * radius {
                    voxel_world.set_voxel(top + offset, WorldVoxel::Air);
                }
            }
        }
    }
    budget.edit = Some((top - IVec3::Y * radius, frame, None));
    budget.relights = 0;
    budget.worst_relight_frame_ms = 0.0;
    budget.worst_light_frame_ms = 0.0;
}

/// Adds what light cost this frame, from the world's light stats.
fn take_light_stats(world: &mut World) {
    let Some(stats) = bevy_voxel_world::light_bench::light_stats::<CaveWorld>(world)
    else {
        return;
    };
    let mut budget = world.resource_mut::<Budget>();
    let before = budget.stats;
    budget.relights += stats.relights - before.relights;
    let relight_ms = stats.relight_ms - before.relight_ms;
    let light_ms = stats.main_thread_ms - before.main_thread_ms;
    budget.worst_relight_frame_ms = budget.worst_relight_frame_ms.max(relight_ms);
    budget.worst_light_frame_ms = budget.worst_light_frame_ms.max(light_ms);
    budget.stats = stats;
}
