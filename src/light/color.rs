//! Light colour
//!
//! This module says what light colour each emissive material gives, how light
//! of several colours mixes where it meets, how a light colour packs into 16
//! bits, and how a chunk holds the light colour of its voxel light. The
//! [`EmissionTable`] of a world names the level and light colour of each
//! emissive material. The inputs of a chunk note which emissive materials the
//! chunk holds, so a solve finds the light colours around its chunk from
//! those notes alone. The solver runs the kernel once per light colour and
//! mixes the runs with a [`RunMixer`], and a chunk keeps the result as a
//! [`ChunkColor`].
//!
//! A light colour is linear RGB scaled so that its brightest component is 1.
//! Only voxel light has one. Where every emissive material around a chunk
//! gives white, the chunk solves and stores its light as it would with no
//! colour at all.
//!
//! # Why the level alone sets brightness
//!
//! Everything that reads light works in levels, from the rows of bits of the
//! kernel to the `voxel` field of a `light_at` reading. A level already says
//! how far light reaches and how bright it is. A light colour then says only
//! the hue and saturation of that light, so it is scaled to a brightest
//! component of 1 and never changes a level. A dim red and a bright red give
//! the same light, and a coloured lamp reaches exactly as far as a white one.
//! At the same level, blue light looks darker than red or green light, as
//! pure blue looks darker than pure red or green.
//!
//! # Why a run per light colour
//!
//! Where a light box holds emissive voxels of two or more light colours, the
//! solver runs the kernel once per colour, each run from the emissive voxels
//! of that colour alone. It then mixes the runs voxel by voxel with
//! [`mix_weighted`]. One run can't carry the colours, because its front stops
//! at voxels it has already lit. A dimmer colour would then never reach the
//! voxels that a brighter one reached first, and pools would meet in a hard
//! seam. Separate runs each see the whole pool of their colour, so a dim
//! colour still tints a brighter pool where it reaches.
//!
//! The level at a voxel is the highest level among the runs. One run of all
//! the emissive voxels gives the same level, so everything that reads levels
//! sees what it would see in a white world. The cost of a solve grows with the
//! light colours in its light box.
//!
//! # Why a fixed mixing curve
//!
//! Each component of the light colour at a voxel is the largest, over the
//! colours reaching the voxel, of `colour × mixing_weight(level)`, where
//! `level` is the level that colour reaches the voxel at. The result is then
//! scaled so that its brightest component is 1. The weight follows
//! `0.7^(15 - level)`, so each level weighs 70% of the level above it,
//! shifted and scaled so that level 15 weighs 1 and level 1 weighs 0. For
//! example, red at level 14 and blue at level 10 mix to `(1, 0, 0.23)`.
//!
//! The mix uses this curve whatever voxel falloff the look of the world uses.
//! The default voxel falloff is the same curve, and under it the mix matches
//! the look exactly: each component of the brightness at a voxel is the
//! largest, over the colours, of colour times brightness. A light colour is
//! part of the light, not of the look. So changing the falloff never relights,
//! and `light_at` reads the same colour under any look. Under another falloff,
//! overlaps still mix in the proportions of the default curve, which looks
//! plausible.
//!
//! The weight falls to 0 at level 1, so a pool fades out of the pool of
//! another colour instead of ending in a step of hue. Where every colour at a
//! voxel is at level 1, the weights are all 0, and the colours mix evenly.

use std::sync::Arc;

use bevy::log::warn_once;
use bevy::prelude::*;

use super::config::Emission;
use crate::chunk::CHUNK_SIZE_U;

const CHUNK: usize = CHUNK_SIZE_U as usize;

/// The packed light colour of white, which a voxel with no voxel light also
/// holds.
pub(crate) const WHITE: u16 = 0;

/// How many emissive materials the [`EmissionTable`] tells apart. An
/// [`EmissiveVoxel`](super::inputs::EmissiveVoxel) holds the index of its
/// material in 13 bits.
pub(crate) const MAX_EMISSIVE_MATERIALS: usize = 1 << 13;

/// How many steps each packed component other than the brightest has.
const STEPS: f32 = 127.0;

/// Returns `color` as a light colour, and whether any of its components was
/// negative.
///
/// The light colour is linear RGB with alpha 1, scaled so that its brightest
/// component is 1. A negative or NaN component reads as 0. Black gives black,
/// and so does a colour with an infinite component.
pub(crate) fn scaled(color: LinearRgba) -> (LinearRgba, bool) {
    let components = [color.red, color.green, color.blue];
    let negative = components.iter().any(|component| *component < 0.0);
    // `f32::max` returns 0 for a NaN component too.
    let [red, green, blue] = components.map(|component| component.max(0.0));
    let brightest = red.max(green).max(blue);
    if brightest <= 0.0 || !brightest.is_finite() {
        return (LinearRgba::BLACK, negative);
    }
    (
        LinearRgba::rgb(red / brightest, green / brightest, blue / brightest),
        negative,
    )
}

/// Returns the light colour `color` packed into 16 bits.
///
/// Bits 14 and 15 name the brightest component, from 0 for red to 2 for blue,
/// and the lowest of them on a tie. Bits 7 to 13 hold how far the first of the
/// other two components falls short of 1, in steps of 1/127, and bits 0 to 6
/// hold the same for the second. For example, `(1, 0.5, 0)` packs to 0 in the
/// top bits, 64 in bits 7 to 13 and 127 in bits 0 to 6.
///
/// White packs to [`WHITE`], and so does black. White and the primaries are
/// exact, and 3 in the top bits never names a colour.
pub(crate) fn pack(color: LinearRgba) -> u16 {
    let components = [color.red, color.green, color.blue].map(|c| c.max(0.0));
    let brightest = components[0].max(components[1]).max(components[2]);
    if brightest <= 0.0 {
        return WHITE;
    }
    let short = components
        .map(|c| ((1.0 - c / brightest).clamp(0.0, 1.0) * STEPS).round() as u16);
    // The lowest component that rounds to the brightest names it, so each
    // colour has only one packing.
    let top = short.iter().position(|short| *short == 0).unwrap_or(0);
    let [first, second] = others(top).map(|component| short[component]);
    (top as u16) << 14 | first << 7 | second
}

/// Returns the packed light colour `packed` cut to the 14 bits that a tag
/// with no slot holds.
///
/// The top 2 bits name the brightest component, and each of the other two
/// components takes 6 bits, in steps of 1/63. White cuts to 0, so the tag of
/// a white chunk is the tag it would have in a world with no light colour.
pub(crate) fn tag_cut(packed: u16) -> u32 {
    let cut = |short: u16| short.div_ceil(2).min(63) as u32;
    let top = (packed >> 14) as u32;
    top << 12 | cut(packed >> 7 & 0x7f) << 6 | cut(packed & 0x7f)
}

/// The top 2 bits of an entry that names a colour slot. They hold 3, the one
/// value that no packed light colour has there.
pub(crate) const SPARE: u16 = 3 << 14;

/// How many colour slots an entry can name, in the 14 bits below [`SPARE`].
pub(crate) const MAX_COLOR_SLOTS: u32 = 1 << 14;

/// Returns the entry of a chunk that holds the colour slot `index`: [`SPARE`]
/// in the top 2 bits, and the index in the other 14.
pub(crate) fn color_slot_entry(index: u32) -> u16 {
    debug_assert!(index < MAX_COLOR_SLOTS);
    SPARE | index as u16
}

/// Returns the light colour that `packed` holds. The top 2 bits of `packed`
/// must name a component, so they can't be 3.
pub(crate) fn unpack(packed: u16) -> LinearRgba {
    let top = (packed >> 14) as usize;
    debug_assert!(
        top < 3,
        "a packed light colour names its brightest component"
    );
    let mut components = [1.0; 3];
    let [first, second] = others(top.min(2));
    components[first] = 1.0 - (packed >> 7 & 0x7f) as f32 / STEPS;
    components[second] = 1.0 - (packed & 0x7f) as f32 / STEPS;
    LinearRgba::rgb(components[0], components[1], components[2])
}

/// The base of the fixed mixing curve, which is also the base of the default
/// voxel falloff.
const MIXING_BASE: f32 = 0.7;

/// Returns the weight that a light colour at `level`, on the 0 to 15 scale,
/// has where light of several colours meets.
///
/// The weight follows the fixed mixing curve,
/// `(0.7^(15 - level) - 0.7^14) / (1 - 0.7^14)`. It is 1 at level 15 and 0 at
/// level 1 and below. For example, level 14 weighs about 0.70, and level 10
/// about 0.16.
#[inline]
pub(crate) fn mixing_weight(level: f32) -> f32 {
    let floor = MIXING_BASE.powi(14);
    ((MIXING_BASE.powf(15.0 - level) - floor) / (1.0 - floor)).max(0.0)
}

/// Mixes the runs of a solve, one for each light colour. At each voxel the
/// mix gives the level of the brightest run, and the light colours of the
/// runs that light the voxel, mixed by [`mix_weighted`].
///
/// A voxel that only one run lights takes the light colour of that run as it
/// is. Level-0 and coarse chunks both mix this way.
pub(crate) struct RunMixer<'a> {
    colors: &'a [u16],
    unpacked: Vec<LinearRgba>,
    /// The [`mixing_weight`] of each level, indexed in steps of the precision.
    weights: Vec<f32>,
}

impl<'a> RunMixer<'a> {
    /// Returns a mixer for runs of the packed light colours `colors`, whose
    /// levels are in steps of `precision`.
    pub fn new(colors: &'a [u16], precision: super::config::LightPrecision) -> Self {
        Self {
            colors,
            unpacked: colors.iter().map(|color| unpack(*color)).collect(),
            weights: (0..=precision.top())
                .map(|steps| mixing_weight(precision.level_of(steps)))
                .collect(),
        }
    }

    /// Returns the level and packed light colour at a voxel where run `i`
    /// reaches level `level(i)`, or `None` where no run lights the voxel.
    #[inline]
    pub fn mix(&self, level: impl Fn(usize) -> u8) -> Option<(u8, u16)> {
        let (mut top, mut lit, mut only) = (0, 0, 0);
        for run in 0..self.colors.len() {
            let level = level(run);
            if level > 0 {
                top = top.max(level);
                lit += 1;
                only = run;
            }
        }
        match lit {
            0 => None,
            1 => Some((top, self.colors[only])),
            _ => Some((
                top,
                pack(mix_weighted((0..self.colors.len()).filter_map(|run| {
                    let level = level(run);
                    (level > 0)
                        .then(|| (self.unpacked[run], self.weights[level as usize]))
                }))),
            )),
        }
    }
}

/// Returns the light colour that `lights` mix to under [`mix_weighted`].
/// Each light is a light colour and its level, on the 0 to 15 scale. A light
/// at level 0 doesn't count.
#[cfg(any(test, feature = "sub_grids"))]
pub(crate) fn mix(lights: impl IntoIterator<Item = (LinearRgba, f32)>) -> LinearRgba {
    mix_weighted(
        lights
            .into_iter()
            .filter(|(_, level)| *level > 0.0)
            .map(|(color, level)| (color, mixing_weight(level))),
    )
}

/// Returns the light colour that `lights` mix to where they meet at a voxel.
/// Each light is a light colour that reaches the voxel, with its
/// [`mixing_weight`] at the level it reaches the voxel at.
///
/// Each component is the largest, over the lights, of colour times weight,
/// and the result is scaled so that its brightest component is 1. Where
/// every light weighs 0, at level 1, each weighs 1 instead, so they mix
/// evenly. No light at all mixes to white.
///
/// The `mix_color` and `mixed_color` functions of the light shader mirror
/// this rule and [`mixing_weight`], so a change here must change them too.
pub(crate) fn mix_weighted(
    lights: impl IntoIterator<Item = (LinearRgba, f32)>,
) -> LinearRgba {
    let mut weighted = Vec3::ZERO;
    let mut even = Vec3::ZERO;
    let mut any = false;
    for (color, weight) in lights {
        let color = Vec3::from_array(color.to_f32_array_no_alpha());
        weighted = weighted.max(color * weight);
        even = even.max(color);
        any = true;
    }
    if !any {
        return LinearRgba::WHITE;
    }
    let mixed = if weighted.max_element() > 0.0 {
        weighted
    } else {
        even
    };
    let brightest = mixed.max_element();
    if brightest <= 0.0 {
        return LinearRgba::WHITE;
    }
    let [red, green, blue] = (mixed / brightest).to_array();
    LinearRgba::rgb(red, green, blue)
}

/// Returns the two components other than `top`, in order.
fn others(top: usize) -> [usize; 2] {
    match top {
        0 => [1, 2],
        1 => [0, 2],
        _ => [0, 1],
    }
}

/// How a chunk or light entry holds the light colour of its voxel light.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub(crate) enum ChunkColor {
    /// The voxel light of every voxel is white, so the chunk holds nothing
    /// beside its levels.
    #[default]
    White,
    /// The voxel light of every lit voxel has this one packed light colour,
    /// which is never [`WHITE`].
    One(u16),
    /// The packed light colour of each voxel, at `x + y × 32 + z × 32²`, for
    /// a chunk that light of two or more light colours reaches. It takes
    /// 64 KiB. A voxel with no voxel light holds [`WHITE`].
    Mixed(Arc<[u16]>),
}

impl ChunkColor {
    /// Returns how a chunk holds the packed light colour `packed` for all of
    /// its voxels.
    pub fn of(packed: u16) -> Self {
        if packed == WHITE {
            ChunkColor::White
        } else {
            ChunkColor::One(packed)
        }
    }

    /// Returns the packed light colour at the open voxel `x`, `y`, `z` of the
    /// chunk. The voxel must have voxel light, since under
    /// [`ChunkColor::One`] an unlit voxel reads the colour too.
    pub fn packed_at(&self, x: usize, y: usize, z: usize) -> u16 {
        match self {
            ChunkColor::White => WHITE,
            ChunkColor::One(packed) => *packed,
            ChunkColor::Mixed(packed) => packed[voxel_index(x, y, z)],
        }
    }

    /// Returns the light colour at the open voxel `x`, `y`, `z` of the chunk.
    /// The voxel must have voxel light, as for [`packed_at`](Self::packed_at).
    pub fn at(&self, x: usize, y: usize, z: usize) -> LinearRgba {
        match self {
            ChunkColor::White => LinearRgba::WHITE,
            ChunkColor::One(packed) => unpack(*packed),
            ChunkColor::Mixed(packed) => unpack(packed[voxel_index(x, y, z)]),
        }
    }
}

/// Returns where the voxel `x`, `y`, `z` of a chunk is in
/// [`ChunkColor::Mixed`].
#[inline]
pub(crate) fn voxel_index(x: usize, y: usize, z: usize) -> usize {
    x + y * CHUNK + z * CHUNK * CHUNK
}

/// What an emissive material gives off, as its emissive voxels hold it: a
/// whole level from 0 to 15, and the index of the material in the
/// [`EmissionTable`] of the world, which names its light colour.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Emitter {
    pub level: u8,
    pub material: u16,
}

/// The emissive materials of a world, each with its level and light colour,
/// built from the emissive list of the world.
///
/// The index of a material is its place in the list. Past
/// [`MAX_EMISSIVE_MATERIALS`], the world warns once, and each further
/// material takes the index of the earlier one whose light colour is nearest
/// its own. A material whose light is level 0 or black holds white, so it
/// never makes a world coloured.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct EmissionTable<I> {
    materials: Vec<(I, Emitter)>,
    /// The packed light colour at each index.
    colors: Vec<u16>,
    /// Whether any index holds a light colour other than white.
    colored: bool,
}

impl<I> Default for EmissionTable<I> {
    fn default() -> Self {
        Self {
            materials: Vec::new(),
            colors: Vec::new(),
            colored: false,
        }
    }
}

impl<I: Copy + PartialEq> EmissionTable<I> {
    /// Returns the table of the emissive list `emission`.
    pub fn new(emission: &[(I, Emission)]) -> Self {
        let mut table = Self::default();
        let mut scaled = Vec::new();
        for (material, emission) in emission {
            let level = emission.whole_level();
            let color = if level == 0 {
                LinearRgba::WHITE
            } else {
                emission.color()
            };
            let index = if table.colors.len() < MAX_EMISSIVE_MATERIALS {
                table.colors.push(pack(color));
                scaled.push(color);
                table.colors.len() - 1
            } else {
                warn_once!(
                    "light propagation: more than {MAX_EMISSIVE_MATERIALS} emissive \
                     materials; each one past that gives the light colour of the \
                     earlier one nearest its own"
                );
                nearest(&scaled, color)
            };
            table.materials.push((
                *material,
                Emitter {
                    level,
                    material: index as u16,
                },
            ));
        }
        table.colored = table.colors.iter().any(|color| *color != WHITE);
        table
    }

    /// Returns a table of white materials at the whole levels `levels`.
    pub fn white(levels: &[(I, u8)]) -> Self {
        let emission: Vec<(I, Emission)> = levels
            .iter()
            .map(|(material, level)| (*material, Emission::level(*level as f32)))
            .collect();
        Self::new(&emission)
    }

    /// Returns what `material` gives off, or `None` if it isn't emissive.
    #[inline]
    pub fn get(&self, material: I) -> Option<Emitter> {
        self.materials
            .iter()
            .find(|(emissive, _)| *emissive == material)
            .map(|(_, emitter)| *emitter)
    }

    pub fn is_empty(&self) -> bool {
        self.materials.is_empty()
    }

    /// Returns the material indices whose packed light colour differs in
    /// `new`, where only light colours changed between the two tables.
    ///
    /// Returns `None` if a material came or went, or if the level or index of
    /// any material changed.
    pub fn recolored(&self, new: &Self) -> Option<Vec<u16>> {
        if self.materials != new.materials {
            return None;
        }
        let changed = self
            .colors
            .iter()
            .zip(&new.colors)
            .enumerate()
            .filter(|(_, (old, new))| old != new)
            .map(|(index, _)| index as u16)
            .collect();
        Some(changed)
    }
}

impl<I> EmissionTable<I> {
    /// Returns whether any material gives a light colour other than white. A
    /// world whose table gives only white solves and stores its light with no
    /// colour.
    #[inline]
    pub fn is_colored(&self) -> bool {
        self.colored
    }

    /// Returns the packed light colour of each material index, in order.
    pub fn colors(&self) -> &[u16] {
        &self.colors
    }

    /// Returns the packed light colour of the material index `material`, or
    /// [`WHITE`] for an index past the end of the table.
    #[inline]
    pub fn color(&self, material: u16) -> u16 {
        self.colors.get(material as usize).copied().unwrap_or(WHITE)
    }
}

/// Returns the index in `colors` of the light colour nearest `color` in
/// linear RGB, or 0 if `colors` is empty.
fn nearest(colors: &[LinearRgba], color: LinearRgba) -> usize {
    let distance = |other: &LinearRgba| {
        Vec3::from_array(other.to_f32_array_no_alpha())
            .distance_squared(Vec3::from_array(color.to_f32_array_no_alpha()))
    };
    colors
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| distance(a).total_cmp(&distance(b)))
        .map_or(0, |(index, _)| index)
}

#[cfg(test)]
mod tests {
    use super::*;

    const STEP: f32 = 1.0 / 127.0;

    fn close(a: LinearRgba, b: LinearRgba, within: f32) -> bool {
        a.to_f32_array_no_alpha()
            .iter()
            .zip(b.to_f32_array_no_alpha())
            .all(|(a, b)| (a - b).abs() <= within)
    }

    #[test]
    fn white_packs_to_0_and_the_primaries_are_exact() {
        assert_eq!(pack(LinearRgba::WHITE), WHITE);
        for color in [
            LinearRgba::WHITE,
            LinearRgba::RED,
            LinearRgba::GREEN,
            LinearRgba::BLUE,
        ] {
            assert_eq!(unpack(pack(color)), color);
        }
    }

    #[test]
    fn the_14_bit_tag_cut_keeps_white_at_0_and_the_primaries_exact() {
        assert_eq!(tag_cut(pack(LinearRgba::WHITE)), 0);
        assert_eq!(tag_cut(pack(LinearRgba::RED)), 63 << 6 | 63);
        assert_eq!(tag_cut(pack(LinearRgba::GREEN)), 1 << 12 | 63 << 6 | 63);
        assert_eq!(tag_cut(pack(LinearRgba::BLUE)), 2 << 12 | 63 << 6 | 63);
        // Half falls 64 of 127 steps short of 1, which cuts to 32 of 63.
        assert_eq!(tag_cut(pack(LinearRgba::rgb(1.0, 0.5, 0.5))), 32 << 6 | 32);
        for packed in [pack(LinearRgba::rgb(0.2, 0.7, 1.0)), u16::MAX >> 2] {
            assert!(tag_cut(packed) < 1 << 14);
        }
    }

    #[test]
    fn a_round_trip_lands_within_a_step() {
        for color in [
            LinearRgba::rgb(1.0, 0.5, 0.25),
            LinearRgba::rgb(0.3, 1.0, 0.01),
            LinearRgba::rgb(0.123, 0.987, 1.0),
            LinearRgba::rgb(1.0, 0.0, 0.999),
        ] {
            let back = unpack(pack(color));
            assert!(close(back, color, STEP), "{color:?} came back {back:?}");
        }
    }

    #[test]
    fn a_tie_names_the_lowest_component() {
        assert_eq!(pack(LinearRgba::rgb(1.0, 1.0, 0.0)) >> 14, 0);
        assert_eq!(pack(LinearRgba::rgb(0.0, 1.0, 1.0)) >> 14, 1);
        assert_eq!(pack(LinearRgba::rgb(1.0, 0.2, 1.0)) >> 14, 0);
        // A component within half a step of the brightest ties with it.
        assert_eq!(
            pack(LinearRgba::rgb(0.999, 1.0, 0.0)),
            pack(LinearRgba::rgb(1.0, 1.0, 0.0))
        );
    }

    #[test]
    fn no_colour_packs_to_the_spare_value() {
        let steps = (0..=20).map(|step| step as f32 / 20.0);
        for red in steps.clone() {
            for green in steps.clone() {
                for blue in steps.clone() {
                    let (color, _) = scaled(LinearRgba::rgb(red, green, blue));
                    assert_ne!(pack(color) >> 14, 3, "{color:?}");
                }
            }
        }
    }

    #[test]
    fn scaling_makes_the_brightest_component_1() {
        let (color, negative) = scaled(LinearRgba::rgb(0.5, 0.25, 0.0));
        assert_eq!(color, LinearRgba::rgb(1.0, 0.5, 0.0));
        assert!(!negative);
        let (color, _) = scaled(LinearRgba::new(0.0, 0.0, 4.0, 0.2));
        assert_eq!(color, LinearRgba::BLUE, "alpha is ignored");
    }

    #[test]
    fn scaling_keeps_black_black_and_makes_grey_white() {
        assert_eq!(scaled(LinearRgba::BLACK), (LinearRgba::BLACK, false));
        assert_eq!(
            scaled(LinearRgba::rgb(0.2, 0.2, 0.2)),
            (LinearRgba::WHITE, false)
        );
    }

    #[test]
    fn scaling_reads_a_negative_component_as_0() {
        let (color, negative) = scaled(LinearRgba::rgb(-1.0, 0.5, 0.25));
        assert_eq!(color, LinearRgba::rgb(0.0, 1.0, 0.5));
        assert!(negative);
        let (color, negative) = scaled(LinearRgba::rgb(-1.0, -0.5, 0.0));
        assert_eq!(color, LinearRgba::BLACK);
        assert!(negative);
    }

    /// Returns the weight of the fixed mixing curve at `level`, worked out from
    /// its formula.
    fn weight(level: f32) -> f32 {
        (0.7f32.powf(15.0 - level) - 0.7f32.powi(14)) / (1.0 - 0.7f32.powi(14))
    }

    #[test]
    fn two_colours_at_one_level_mix_to_their_brightest_components() {
        let red_and_green = mix([(LinearRgba::RED, 9.0), (LinearRgba::GREEN, 9.0)]);
        assert_eq!(red_and_green, LinearRgba::rgb(1.0, 1.0, 0.0));
        let orange = LinearRgba::rgb(1.0, 0.5, 0.0);
        let mixed = mix([(orange, 12.0), (LinearRgba::BLUE, 12.0)]);
        assert!(
            close(mixed, LinearRgba::rgb(1.0, 0.5, 1.0), 1e-6),
            "{mixed:?}"
        );
    }

    #[test]
    fn a_dimmer_colour_inside_a_brighter_pool_tints_it_by_its_weight() {
        let mixed = mix([(LinearRgba::RED, 14.0), (LinearRgba::BLUE, 10.0)]);
        let blue = weight(10.0) / weight(14.0);
        assert!(
            close(mixed, LinearRgba::rgb(1.0, 0.0, blue), 1e-6),
            "{mixed:?}"
        );
        // (0.7^5 - 0.7^14) / (0.7 - 0.7^14), worked out by hand.
        assert!((blue - 0.2326).abs() < 1e-3, "{blue}");
    }

    #[test]
    fn every_colour_at_level_1_mixes_evenly() {
        let mixed = mix([(LinearRgba::RED, 1.0), (LinearRgba::GREEN, 1.0)]);
        assert_eq!(mixed, LinearRgba::rgb(1.0, 1.0, 0.0));
        // A colour at level 2 outweighs any at level 1.
        let mixed = mix([
            (LinearRgba::RED, 1.0),
            (LinearRgba::GREEN, 2.0),
            (LinearRgba::BLUE, 1.0),
        ]);
        assert_eq!(mixed, LinearRgba::GREEN);
        // So does a colour one step of 1/17 above level 1, as at 256 levels.
        let mixed = mix([
            (LinearRgba::RED, 1.0),
            (LinearRgba::GREEN, 1.0 + 1.0 / 17.0),
        ]);
        assert_eq!(mixed, LinearRgba::GREEN);
    }

    #[test]
    fn no_light_mixes_to_white_and_one_colour_mixes_to_itself() {
        assert_eq!(mix([]), LinearRgba::WHITE);
        assert_eq!(mix([(LinearRgba::BLUE, 0.0)]), LinearRgba::WHITE);
        let orange = LinearRgba::rgb(1.0, 0.5, 0.0);
        assert!(close(mix([(orange, 3.0)]), orange, 1e-6));
        assert!(close(mix([(orange, 1.0)]), orange, 1e-6));
    }

    #[test]
    fn orange_hidden_at_a_face_by_red_and_green_shows_further_in() {
        let orange = LinearRgba::rgb(1.0, 0.5, 0.0);
        // At the face, orange and red are at 12, and green is at 11. The red
        // light beats orange in red, and the green light beats it in green, so
        // orange adds nothing.
        let at_face = mix([
            (orange, 12.0),
            (LinearRgba::RED, 12.0),
            (LinearRgba::GREEN, 11.0),
        ]);
        let without = mix([(LinearRgba::RED, 12.0), (LinearRgba::GREEN, 11.0)]);
        assert_eq!(at_face, without);
        assert!(close(
            at_face,
            LinearRgba::rgb(1.0, weight(11.0) / weight(12.0), 0.0),
            1e-6
        ));
        // Nine voxels in, orange and red are at 3, and green is at 2. Orange
        // now beats the green light in green.
        let inside = mix([
            (orange, 3.0),
            (LinearRgba::RED, 3.0),
            (LinearRgba::GREEN, 2.0),
        ]);
        assert!(0.5 * weight(3.0) > weight(2.0));
        assert!(
            close(inside, LinearRgba::rgb(1.0, 0.5, 0.0), 1e-6),
            "{inside:?}"
        );
    }

    #[test]
    fn colours_mix_by_the_fixed_curve_whatever_the_voxel_falloff() {
        use crate::prelude::{LevelCurve, LightLook};
        let lights = [
            (LinearRgba::rgb(1.0, 0.2, 0.0), 13.0),
            (LinearRgba::rgb(0.0, 0.4, 1.0), 9.0),
            (LinearRgba::GREEN, 6.0),
        ];
        // The max per component of colour times `brightness` at each level,
        // scaled so its brightest component is 1.
        let max_by = |brightness: &dyn Fn(f32) -> f32| {
            let max = |component: usize| {
                lights
                    .iter()
                    .map(|(color, level)| {
                        color.to_f32_array()[component] * brightness(*level)
                    })
                    .fold(0.0, f32::max)
            };
            let [red, green, blue] = [max(0), max(1), max(2)];
            let brightest = red.max(green).max(blue);
            LinearRgba::rgb(red / brightest, green / brightest, blue / brightest)
        };
        let expected = max_by(&weight);
        let default_falloff = LightLook::default().voxel_falloff;
        for falloff in [
            default_falloff.clone(),
            LevelCurve::default(),
            LevelCurve::linear(),
            LevelCurve::exponential(0.5),
        ] {
            let mixed = mix(lights);
            assert!(close(mixed, expected, 1e-6), "under {falloff:?}: {mixed:?}");
            let look = max_by(&|level| falloff.at(level));
            if falloff != default_falloff {
                assert!(!close(look, expected, 0.01), "{falloff:?} mixes alike");
            }
        }
    }

    fn colored(level: f32, color: LinearRgba) -> Emission {
        Emission::level(level).with_color(color)
    }

    #[test]
    fn the_emission_table_names_each_materials_level_and_light_colour() {
        let table = EmissionTable::new(&[
            (1u16, Emission::level(15.0)),
            (2, colored(8.0, LinearRgba::RED)),
            (3, colored(15.0, LinearRgba::rgb(0.5, 0.0, 0.0))),
            (4, colored(15.0, LinearRgba::BLACK)),
        ]);
        assert!(table.is_colored());
        let emitter = |material| table.get(material).unwrap();
        assert_eq!(emitter(1).level, 15);
        assert_eq!(table.color(emitter(1).material), WHITE);
        assert_eq!(emitter(2).level, 8);
        assert_eq!(unpack(table.color(emitter(2).material)), LinearRgba::RED);
        // Red and half red are equal once scaled.
        assert_eq!(
            table.color(emitter(3).material),
            table.color(emitter(2).material)
        );
        // Black gives no light, and no colour.
        assert_eq!(emitter(4).level, 0);
        assert_eq!(table.color(emitter(4).material), WHITE);
        assert_eq!(table.get(5), None);

        let grey = EmissionTable::new(&[(1u16, colored(15.0, LinearRgba::gray(0.3)))]);
        assert!(!grey.is_colored());
    }

    #[test]
    fn past_8192_materials_each_takes_the_index_of_the_earlier_one_nearest_its_colour() {
        let mut emission: Vec<(u32, Emission)> = (0..MAX_EMISSIVE_MATERIALS as u32)
            .map(|material| (material, colored(15.0, LinearRgba::rgb(1.0, 1.0, 0.0))))
            .collect();
        emission[7].1 = colored(15.0, LinearRgba::RED);
        emission[9].1 = colored(15.0, LinearRgba::BLUE);
        let past = MAX_EMISSIVE_MATERIALS as u32;
        emission.push((past, colored(10.0, LinearRgba::rgb(1.0, 0.1, 0.0))));
        emission.push((past + 1, colored(15.0, LinearRgba::rgb(0.0, 0.1, 1.0))));
        let table = EmissionTable::new(&emission);
        assert_eq!(
            table.get(past).unwrap(),
            Emitter {
                level: 10,
                material: 7
            }
        );
        assert_eq!(table.get(past + 1).unwrap().material, 9);
    }
}
