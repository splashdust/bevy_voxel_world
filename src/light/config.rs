//! The settings of light propagation, which a world returns from
//! [`VoxelWorldConfig::light_propagation`](crate::prelude::VoxelWorldConfig::light_propagation).
//!
//! Each field's documentation says when a change to it takes effect:
//!
//! - **Read once at plugin build:** changing it later does nothing.
//! - **Relights on change:** every chunk with a solid near its open voxels
//!   works its light out again. Old light shows until new light lands.
//! - **Replaces the texture on change:** as a relight, and the world's light
//!   texture is made again in the new format.
//! - **Shows on the next frame:** only the shader reads it.
//! - **A limit:** it caps what the world may use.
//!
//! Out-of-range values are clamped, with a warning logged once, where the crate
//! reads them. The value the world returns is never rewritten.

use std::time::Duration;

use bevy::log::warn_once;
use bevy::reflect::Reflect;

/// Light propagation for one world.
///
/// A world opts in by returning `Some` from
/// [`VoxelWorldConfig::light_propagation`](crate::prelude::VoxelWorldConfig::light_propagation),
/// and returning `None`, the default, costs nothing: no memory, no chunk task
/// time, no mesh attribute, no texture binding and no shader change.
///
/// Light spreads from its sources through the world's open voxels, `Air` and
/// `Unset`, and stops at `Solid` ones. A voxel with no open path to a source is
/// pitch black, however close it is to the surface. There are two kinds of
/// light, each a level from 0 to 15 at every voxel:
///
/// - **Sky light** enters from the open sky and falls straight down, with no
///   loss, to the first solid voxel. Past that it spreads sideways and down,
///   fading with distance.
/// - **Voxel light** comes from emissive voxels, declared by material in
///   [`emissive_materials`](crate::prelude::VoxelWorldConfig::emissive_materials),
///   and spreads the same way.
///
/// Each world has its own light: one world's solids never shadow another
/// world. Up is +Y in every world.
///
/// Every field says when a change to it takes effect, as one of: read once at
/// plugin build, relights on change, replaces the texture on change, shows on
/// the next frame, or a limit.
///
/// Light needs the crate's `light_propagation` cargo feature.
///
/// # Showing light
///
/// With [`shown`](Self::shown) on, the crate draws the world's chunks with its
/// material wrapped in
/// [`VoxelWorldBindings`](crate::rendering::VoxelWorldBindings), and:
///
/// - The crate owns the `MeshTag` of every chunk in the world, and overwrites
///   any tag the world's own systems set.
/// - Lit chunk materials are forward only: voxel light is added in the forward
///   pass, so a chunk drawn in a deferred pass shows none.
/// - Exact black is zero light before tonemapping. Where neither kind of light
///   reaches, a face is exactly black, which Bevy's default tonemapper maps to
///   1/255, with dither noise of up to 1/255.
///
/// ```
/// use bevy::prelude::*;
/// use bevy_voxel_world::prelude::*;
///
/// #[derive(Resource, Clone, Default)]
/// struct MyWorld;
///
/// impl VoxelWorldConfig for MyWorld {
///     type MaterialIndex = u8;
///     type ChunkUserBundle = ();
///
///     fn light_propagation(&self) -> Option<LightPropagation> {
///         Some(LightPropagation {
///             voxel_reach: 10,
///             ..default()
///         })
///     }
///
///     fn emissive_materials(&self) -> Vec<(u8, Emission)> {
///         vec![(3, Emission::level(15.0))]
///     }
/// }
/// ```
#[derive(Reflect, Clone, PartialEq, Debug)]
pub struct LightPropagation {
    /// Whether chunks are drawn with propagated light. With `false`, light is
    /// worked out and can be read, but chunks keep Bevy's lighting.
    ///
    /// Read once at plugin build. Defaults to `true`.
    pub shown: bool,

    /// Where sky light enters the world.
    ///
    /// Relights on change. Defaults to [`Sky::Clearance`] of 16 chunks.
    pub sky: Sky,

    /// How many voxels sky light spreads past where it falls straight down,
    /// from 1 to 32.
    ///
    /// Relights on change. Defaults to 30.
    pub sky_reach: u8,

    /// How many voxels voxel light spreads from an emissive voxel at level 15,
    /// from 1 to 32.
    ///
    /// Relights on change. Defaults to 14, which loses one level per voxel.
    pub voxel_reach: u8,

    /// The shape light spreads in, for both kinds of light.
    ///
    /// Relights on change. Defaults to [`LightShape::Round`].
    pub shape: LightShape,

    /// How sky light reaches past where it falls and spreads, such as under a
    /// high overhang. `None` turns it off.
    ///
    /// Relights on change. Defaults to [`SkyFill::Paths`] with its defaults.
    pub sky_fill: Option<SkyFill>,

    /// How many light levels each voxel holds.
    ///
    /// Replaces the texture on change. Defaults to [`LightPrecision::Steps16`].
    pub precision: LightPrecision,

    /// How many chunks the world's light texture holds at once, at least 1.
    ///
    /// A chunk takes a slot in the texture when its faces see more than one
    /// level of light. Each slot costs 39,304 bytes of GPU memory at 16
    /// levels, so the default of 1,024 is about 38 MiB. The texture is made
    /// when the first chunk needs a slot, so a world where every chunk's
    /// faces see one level costs nothing.
    ///
    /// The count is clamped to what the device's largest 3D texture holds,
    /// with a warning: WebGL2 guarantees 256 texels a side, which is 343
    /// slots.
    ///
    /// When every slot is taken, the chunks farthest from the camera give
    /// theirs to nearer ones, and the world logs a warning once. A chunk
    /// with no slot shows one level of each kind of light on all its faces:
    /// it looks flat, losing its shadows, and its lamps' faces lose their
    /// glow. Raise the count if that warning appears.
    ///
    /// Replaces the texture on change. Defaults to 1,024.
    pub slot_capacity: u32,

    /// How light looks on chunk faces.
    ///
    /// Shows on the next frame. See [`LightLook`] for its defaults.
    pub look: LightLook,

    /// The most memory, in bytes, the world keeps about which columns are open
    /// to the sky.
    ///
    /// A limit. Defaults to 16 MiB.
    pub sky_record_cap: usize,

    /// The most main-thread time per frame spent relighting on WASM, where
    /// relights don't run on other threads.
    ///
    /// A limit. Defaults to 4 ms.
    pub wasm_relight_budget: Duration,
}

impl Default for LightPropagation {
    fn default() -> Self {
        Self {
            shown: true,
            sky: Sky::default(),
            sky_reach: 30,
            voxel_reach: 14,
            shape: LightShape::default(),
            sky_fill: Some(SkyFill::default()),
            precision: LightPrecision::default(),
            slot_capacity: 1024,
            look: LightLook::default(),
            sky_record_cap: 16 * 1024 * 1024,
            wasm_relight_budget: Duration::from_millis(4),
        }
    }
}

impl LightPropagation {
    /// This config with every out-of-range value clamped, logging a warning
    /// once for each kind of value clamped.
    pub(crate) fn clamped(&self) -> Self {
        let mut config = self.clone();
        config.sky_reach = clamp_reach(config.sky_reach, "sky_reach");
        config.voxel_reach = clamp_reach(config.voxel_reach, "voxel_reach");
        if config.slot_capacity == 0 {
            warn_once!("light propagation: slot_capacity 0 is below 1, using 1");
            config.slot_capacity = 1;
        }
        if let Sky::Clearance { chunks: 0 } = config.sky {
            warn_once!(
                "light propagation: a sky clearance of 0 chunks is below 1, using 1"
            );
            config.sky = Sky::Clearance { chunks: 1 };
        }
        config.sky_fill = config.sky_fill.map(|fill| match fill {
            SkyFill::Paths(paths) => SkyFill::Paths(paths.clamped()),
            SkyFill::Openness(open) => SkyFill::Openness(open.clamped()),
        });
        config.look = config.look.clamped();
        config
    }
}

fn clamp_reach(reach: u8, name: &str) -> u8 {
    if (1..=32).contains(&reach) {
        return reach;
    }
    let clamped = reach.clamp(1, 32);
    warn_once!("light propagation: {name} {reach} is outside 1 to 32, using {clamped}");
    clamped
}

/// `value` kept within `min..=max`, logging a warning once if it wasn't. NaN
/// becomes `min`.
fn clamp_f32(value: f32, min: f32, max: f32, name: &str) -> f32 {
    if value >= min && value <= max {
        return value;
    }
    let clamped = if value > max { max } else { min };
    warn_once!(
        "light propagation: {name} {value} is outside {min} to {max}, using {clamped}"
    );
    clamped
}

fn at_least_one(value: u32, name: &str) -> u32 {
    if value >= 1 {
        return value;
    }
    warn_once!("light propagation: {name} 0 is below 1, using 1");
    1
}

/// Where sky light enters the world.
///
/// With sky light on, the world's voxel lookup must return the same voxel for
/// a position every time it's asked, leaving voxel edits aside, since a column
/// is judged open or blocked from what the lookups return.
#[derive(Reflect, Clone, PartialEq, Debug)]
pub enum Sky {
    /// Sky light enters the top of a chunk in a column when that column holds
    /// no solid voxel in the `chunks` whole chunks straight above it, at least 1.
    ///
    /// A shadow ends about this far below what casts it, and a sealed space
    /// taller than it lights up on its floor.
    Clearance { chunks: u32 },
    /// Sky light enters a column that holds no solid voxel up to `y`, rounded
    /// up to the top of its chunk. Voxel edits above it cast no shadow.
    Ceiling { y: i32 },
    /// No sky light. Voxel light still works.
    Off,
}

impl Default for Sky {
    fn default() -> Self {
        Sky::Clearance { chunks: 16 }
    }
}

/// How sky light is carried past where it falls and spreads, worked out per
/// 8-voxel cell.
#[derive(Reflect, Clone, PartialEq, Debug)]
pub enum SkyFill {
    /// Light passes from open space to open space through the openings between
    /// them, losing more through narrower openings.
    Paths(PathFill),
    /// Each cell is lit by how much of the sky it sees.
    Openness(OpenFill),
}

impl Default for SkyFill {
    fn default() -> Self {
        SkyFill::Paths(PathFill::default())
    }
}

/// The settings of [`SkyFill::Paths`].
#[derive(Reflect, Clone, PartialEq, Debug)]
pub struct PathFill {
    /// The light level of open space beside the open sky, from 0 to 15.
    /// Defaults to 12.
    pub beside_open_sky: f32,
    /// The levels lost passing through a whole open face of 8-voxel cells, at
    /// least 0. Defaults to 2.
    pub whole_face_loss: f32,
    /// How much more a narrower opening loses, at least 0. Passing through an
    /// opening loses `whole_face_loss × (16 / f) ^ opening_power` levels, where
    /// `f`, from 1 to 16, is how many cells of the receiving space's widest
    /// cross-section the opening fills. Defaults to 2.
    pub opening_power: f32,
}

impl Default for PathFill {
    fn default() -> Self {
        Self {
            beside_open_sky: 12.0,
            whole_face_loss: 2.0,
            opening_power: 2.0,
        }
    }
}

impl PathFill {
    fn clamped(self) -> Self {
        Self {
            beside_open_sky: clamp_f32(
                self.beside_open_sky,
                0.0,
                15.0,
                "beside_open_sky",
            ),
            whole_face_loss: clamp_f32(
                self.whole_face_loss,
                0.0,
                f32::MAX,
                "whole_face_loss",
            ),
            opening_power: clamp_f32(self.opening_power, 0.0, f32::MAX, "opening_power"),
        }
    }
}

/// The settings of [`SkyFill::Openness`].
#[derive(Reflect, Clone, PartialEq, Debug)]
pub struct OpenFill {
    /// The share of the sky, from 0 to 1, a cell must see to count as open to
    /// the sky and get the sun. Defaults to 0.8.
    pub sun_at: f32,
    /// How bright a cell is for the share of the sky it sees, at least 0.
    /// Defaults to 1.
    pub gain: f32,
    /// How many rays each cell casts toward the sky, at least 1. Defaults to 32.
    pub rays: u32,
    /// How far, in voxels, a ray goes looking for the sky, at least 1. Defaults
    /// to 128.
    pub reach: u32,
}

impl Default for OpenFill {
    fn default() -> Self {
        Self {
            sun_at: 0.8,
            gain: 1.0,
            rays: 32,
            reach: 128,
        }
    }
}

impl OpenFill {
    fn clamped(self) -> Self {
        Self {
            sun_at: clamp_f32(self.sun_at, 0.0, 1.0, "sun_at"),
            gain: clamp_f32(self.gain, 0.0, f32::MAX, "gain"),
            rays: at_least_one(self.rays, "rays"),
            reach: at_least_one(self.reach, "reach"),
        }
    }
}

/// The shape light spreads in.
#[derive(Reflect, Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum LightShape {
    /// Pools of light come out nearly circular.
    #[default]
    Round,
    /// Light loses a level per face step, so pools come out diamond shaped.
    Diamond,
}

/// How many light levels each voxel holds. Levels read from 0 to 15 either way.
#[derive(Reflect, Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum LightPrecision {
    /// 16 levels per kind of light, 1 byte per voxel.
    #[default]
    Steps16,
    /// 256 levels per kind of light, 2 bytes per voxel, with smoother sky fill.
    Steps256,
}

/// How light looks on chunk faces. Every field shows on the next frame.
#[derive(Reflect, Clone, PartialEq, Debug)]
pub struct LightLook {
    /// Whether light blends between neighbouring voxels on a face.
    /// Defaults to [`LightSmoothing::Smoothed`].
    pub smoothing: LightSmoothing,
    /// The brightness of each sky light level. Defaults to `0.7^(15 - level)`,
    /// with level 0 exactly black.
    pub sky_falloff: LevelCurve,
    /// The brightness of each voxel light level. Defaults to
    /// `0.7^(15 - level)`, with level 0 exactly black.
    pub voxel_falloff: LevelCurve,
    /// Where Bevy's directional lights shine. Defaults to
    /// [`SunGate::DirectSkyOnly`].
    pub sun: SunGate,
    /// The brightness of voxel light at level 15, multiplied by the camera's
    /// exposure, at least 0. Defaults to 1,000.
    pub voxel_brightness: f32,
    /// How brightly an emissive voxel's own faces glow, as a multiple of voxel
    /// light at level 15, at least 0. Defaults to 6.
    pub emissive_glow: f32,
    /// What ambient occlusion darkens. Defaults to [`LightAo::IndirectLight`].
    pub ao: LightAo,
    /// The least brightness of any face, at least 0, so caves needn't be pitch
    /// black. Defaults to 0.
    pub minimum_brightness: f32,
}

impl Default for LightLook {
    fn default() -> Self {
        Self {
            smoothing: LightSmoothing::default(),
            sky_falloff: LevelCurve::default(),
            voxel_falloff: LevelCurve::default(),
            sun: SunGate::default(),
            voxel_brightness: 1000.0,
            emissive_glow: 6.0,
            ao: LightAo::default(),
            minimum_brightness: 0.0,
        }
    }
}

impl LightLook {
    fn clamped(self) -> Self {
        Self {
            voxel_brightness: clamp_f32(
                self.voxel_brightness,
                0.0,
                f32::MAX,
                "voxel_brightness",
            ),
            emissive_glow: clamp_f32(self.emissive_glow, 0.0, f32::MAX, "emissive_glow"),
            minimum_brightness: clamp_f32(
                self.minimum_brightness,
                0.0,
                f32::MAX,
                "minimum_brightness",
            ),
            ..self
        }
    }
}

/// Whether light blends between neighbouring voxels on a face.
#[derive(Reflect, Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum LightSmoothing {
    /// Light blends smoothly across a face.
    #[default]
    Smoothed,
    /// Each voxel's light shows as a flat step.
    Stepped,
}

/// The brightness of each light level from 0 to 15.
#[derive(Reflect, Clone, PartialEq, Debug)]
pub struct LevelCurve {
    pub(crate) brightness: [f32; 16],
}

impl Default for LevelCurve {
    /// `0.7^(15 - level)`, so each level is 70% as bright as the one above it.
    fn default() -> Self {
        Self {
            brightness: std::array::from_fn(|level| 0.7_f32.powi(15 - level as i32)),
        }
    }
}

/// Where Bevy's directional lights shine.
#[derive(Reflect, Clone, PartialEq, Debug, Default)]
pub enum SunGate {
    /// Only where a voxel has direct sky, so the sun never shines through rock
    /// that hasn't spawned.
    #[default]
    DirectSkyOnly,
    /// Scaled by the sky light level, through this curve.
    ByLevel(LevelCurve),
}

/// What ambient occlusion darkens on a lit chunk.
#[derive(Reflect, Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum LightAo {
    /// Bevy's ambient light and voxel light, so rock keeps its shape in dim
    /// places.
    #[default]
    IndirectLight,
    /// The base colour, as without light propagation.
    BaseColor,
    /// No ambient occlusion.
    Off,
}

/// The light an emissive material gives off.
///
/// Built with [`Emission::level`].
#[derive(Reflect, Clone, Copy, PartialEq, Debug)]
pub struct Emission {
    level: f32,
}

impl Emission {
    /// Light at `level`, from 0 to 15. A voxel at level 8 lights about half as
    /// far as one at 15.
    pub fn level(level: f32) -> Self {
        Self { level }
    }

    /// The level as a whole level from 0 to 15.
    pub(crate) fn whole_level(&self) -> u8 {
        self.level.round().clamp(0.0, 15.0) as u8
    }

    /// This emission with its level clamped to 0 to 15, logging a warning once
    /// if it wasn't.
    pub(crate) fn clamped(self) -> Self {
        Self {
            level: clamp_f32(self.level, 0.0, 15.0, "emission level"),
        }
    }
}
