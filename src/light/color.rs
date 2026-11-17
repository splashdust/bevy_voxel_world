//! Light colour: what an emissive material's light colour is, how light of
//! several colours mixes where it meets, how a light colour is packed into 16
//! bits, and how a chunk holds the light colour of its voxel light.
//!
//! A light colour is linear RGB scaled so its brightest component is 1, so
//! the level alone says how bright light is. Only voxel light has one.
//!
//! The world's [`EmissionTable`] names each emissive material's level and
//! light colour. A chunk's inputs note which emissive materials it holds, so a
//! solve finds the light colours around its chunk from those notes alone.
//! Where every emissive material around a chunk gives white, the chunk solves
//! and stores its light as it would with no colour at all.
//!
//! # Why the level alone sets brightness
//!
//! Everything that reads light, from the kernel's rows of bits to
//! `light_at`'s `voxel`, works in levels, and a level already says how far
//! light reaches and how bright it is. A light colour then says only which
//! hue and saturation that light has, so it's scaled to a brightest
//! component of 1 and never changes a level. A dim red and a bright red give
//! the same light, a coloured lamp reaches exactly as far as a white one, and
//! blue reads darker than red or green at the same level, as those colours
//! are.
//!
//! # Why a run per light colour
//!
//! Where a light box holds emissive voxels of two or more light colours, the
//! solver runs the kernel once per colour, each run from that colour's
//! emissive voxels alone, and mixes the runs voxel by voxel with
//! [`mix_weighted`]. One run can't carry the colours: its front stops at
//! voxels it has already lit,
//! so a dimmer colour never reaches the voxels where a brighter one got first,
//! and pools would meet in a hard seam. Separate runs each see the whole of
//! their colour's pool, so a dim colour still tints a brighter pool where it
//! reaches. The cost of a solve grows with the light colours in its light box.
//! A voxel's level is the highest of the runs', which is what one run of all
//! the emissive voxels gives, so everything that reads levels sees what it
//! would in a white world.
//!
//! # Why a fixed mixing curve
//!
//! Each component of a voxel's light colour is the max over the colours of
//! colour times [`mixing_weight`] at that colour's level, whatever voxel
//! falloff the world's look uses. The weight is `0.7^(15 - level)`, each level
//! 70% as bright as the one above, shifted and scaled so level 1 weighs 0.
//! Under a voxel falloff of that same curve, the mix is exactly the look:
//! each component of a voxel's brightness is the max over the colours of
//! colour times brightness. A light colour is part of the light, not of the look, so
//! changing the falloff never relights and `light_at` reads the same colour
//! under any look. Under another falloff, overlaps mix in the default
//! curve's proportions, which nothing reads as a bug. The curve's weight falls
//! to 0 at level 1, so a pool fades out of another colour's pool instead of
//! ending in a hue step. Where every colour at a voxel is at level 1, the
//! weights are all 0 and the colours mix evenly.

use std::sync::Arc;

use bevy::log::warn_once;
use bevy::prelude::*;

use super::config::Emission;
use crate::chunk::CHUNK_SIZE_U;

const CHUNK: usize = CHUNK_SIZE_U as usize;

/// The packed light colour white. A voxel with no voxel light reads it.
pub(crate) const WHITE: u16 = 0;

/// How many emissive materials the [`EmissionTable`] tells apart: an
/// [`EmissiveVoxel`](super::inputs::EmissiveVoxel) holds its material's
/// index in 13 bits.
pub(crate) const MAX_EMISSIVE_MATERIALS: usize = 1 << 13;

/// Steps in each packed component below the brightest.
const STEPS: f32 = 127.0;

/// `color` as a light colour: linear RGB with alpha 1, each negative component
/// read as 0, scaled so its brightest component is 1. Black stays black.
/// Returns it, and whether any component was negative.
pub(crate) fn scaled(color: LinearRgba) -> (LinearRgba, bool) {
    let components = [color.red, color.green, color.blue];
    let negative = components.iter().any(|component| *component < 0.0);
    // `max` reads NaN as 0 too.
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

/// `color`, a light colour, packed into 16 bits.
///
/// Bits 14 and 15 name the brightest component, 0 for red to 2 for blue, the
/// lowest one on a tie. Bits 7 to 13 and 0 to 6 hold how far each of the other
/// two components, in order, falls short of 1, in steps of 1/127. So white
/// packs to [`WHITE`], white and the primaries are exact, and 3 in the top
/// bits never names a colour.
pub(crate) fn pack(color: LinearRgba) -> u16 {
    let components = [color.red, color.green, color.blue].map(|c| c.max(0.0));
    let brightest = components[0].max(components[1]).max(components[2]);
    if brightest <= 0.0 {
        return WHITE;
    }
    let short = components
        .map(|c| ((1.0 - c / brightest).clamp(0.0, 1.0) * STEPS).round() as u16);
    // The lowest component that rounds to the brightest names it, so a colour
    // packs one way only.
    let top = short.iter().position(|short| *short == 0).unwrap_or(0);
    let [first, second] = others(top).map(|component| short[component]);
    (top as u16) << 14 | first << 7 | second
}

/// The packed light colour `packed` cut to the 14 bits a tag with no slot
/// holds: the 2 bits naming the brightest component, then each of the other
/// two components in 6 bits, in steps of 1/63. White cuts to 0, so a white
/// chunk's tag is the tag it would have in a world with no light colour.
pub(crate) fn tag_cut(packed: u16) -> u32 {
    let cut = |short: u16| short.div_ceil(2).min(63) as u32;
    let top = (packed >> 14) as u32;
    top << 12 | cut(packed >> 7 & 0x7f) << 6 | cut(packed & 0x7f)
}

/// The top 2 bits of an entry naming a colour slot: 3, the spare value,
/// which no packed light colour holds there.
pub(crate) const SPARE: u16 = 3 << 14;

/// How many colour slots an entry can name: the 14 bits below [`SPARE`].
pub(crate) const MAX_COLOR_SLOTS: u32 = 1 << 14;

/// The entry of a chunk holding the colour slot `index`: [`SPARE`] in the
/// top 2 bits and the index in the other 14.
pub(crate) fn color_slot_entry(index: u32) -> u16 {
    debug_assert!(index < MAX_COLOR_SLOTS);
    SPARE | index as u16
}

/// The light colour `packed` holds, which must name one.
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

/// The base of the fixed mixing curve: the default voxel falloff's.
const MIXING_BASE: f32 = 0.7;

/// The weight a light colour at `level`, on the 0 to 15 scale, has where
/// light of several colours meets: the fixed mixing curve,
/// `(0.7^(15 - level) - 0.7^14) / (1 - 0.7^14)`. It's 1 at level 15 and 0 at
/// level 1 and below.
#[inline]
pub(crate) fn mixing_weight(level: f32) -> f32 {
    let floor = MIXING_BASE.powi(14);
    ((MIXING_BASE.powf(15.0 - level) - floor) / (1.0 - floor)).max(0.0)
}

/// [`mix_weighted`] of `lights`, each a light colour and its level on the 0
/// to 15 scale. A light at level 0 doesn't count.
#[cfg(test)]
fn mix(lights: impl IntoIterator<Item = (LinearRgba, f32)>) -> LinearRgba {
    mix_weighted(
        lights
            .into_iter()
            .filter(|(_, level)| *level > 0.0)
            .map(|(color, level)| (color, mixing_weight(level))),
    )
}

/// The light colour that `lights` mix to where they meet at a voxel, each a
/// light colour whose light reaches the voxel and its [`mixing_weight`] at
/// the level it reaches it at.
///
/// Each component is the max over the lights of colour times weight, scaled
/// so the brightest component is 1. Where every light weighs 0, at level 1,
/// each weighs 1 instead, so they mix evenly. No light mixes to white.
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

/// The two components other than `top`, in order.
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
    /// Every voxel's voxel light is white: nothing beside its levels.
    #[default]
    White,
    /// Every lit voxel's voxel light has the one packed light colour, never
    /// [`WHITE`].
    One(u16),
    /// The packed light colour of each voxel, at `x + y × 32 + z × 32²`, for
    /// a chunk that light of two or more light colours reaches: 64 KiB. A
    /// voxel with no voxel light holds [`WHITE`].
    Mixed(Arc<[u16]>),
}

impl ChunkColor {
    /// The packed light colour `packed` held for a whole chunk.
    pub fn of(packed: u16) -> Self {
        if packed == WHITE {
            ChunkColor::White
        } else {
            ChunkColor::One(packed)
        }
    }

    /// The packed light colour of the chunk's open voxel `x`, `y`, `z`, which
    /// has voxel light.
    pub fn packed_at(&self, x: usize, y: usize, z: usize) -> u16 {
        match self {
            ChunkColor::White => WHITE,
            ChunkColor::One(packed) => *packed,
            ChunkColor::Mixed(packed) => packed[voxel_index(x, y, z)],
        }
    }

    /// The light colour of the chunk's open voxel `x`, `y`, `z`, which has
    /// voxel light.
    pub fn at(&self, x: usize, y: usize, z: usize) -> LinearRgba {
        match self {
            ChunkColor::White => LinearRgba::WHITE,
            ChunkColor::One(packed) => unpack(*packed),
            ChunkColor::Mixed(packed) => unpack(packed[voxel_index(x, y, z)]),
        }
    }
}

/// Where the chunk's voxel `x`, `y`, `z` is in [`ChunkColor::Mixed`].
#[inline]
pub(crate) fn voxel_index(x: usize, y: usize, z: usize) -> usize {
    x + y * CHUNK + z * CHUNK * CHUNK
}

/// What an emissive material gives off, as its emissive voxels hold it: its
/// level, as a whole level from 0 to 15, and its index in the world's
/// [`EmissionTable`], which names its light colour.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Emitter {
    pub level: u8,
    pub material: u16,
}

/// Each emissive material of a world, with its level and light colour, built
/// from its emissive list.
///
/// A material's index is its place in the list. Past
/// [`MAX_EMISSIVE_MATERIALS`] the world warns once, and each further material
/// takes the index of the earlier one whose light colour is nearest its own.
/// A material whose light is level 0, or black, holds white, so it never
/// makes a world coloured.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct EmissionTable<I> {
    materials: Vec<(I, Emitter)>,
    /// The packed light colour of each index.
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
    /// The table of the emissive list `emission`.
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

    /// A table of white materials at the whole levels `levels`.
    pub fn white(levels: &[(I, u8)]) -> Self {
        let emission: Vec<(I, Emission)> = levels
            .iter()
            .map(|(material, level)| (*material, Emission::level(*level as f32)))
            .collect();
        Self::new(&emission)
    }

    /// What `material` gives off, if it's emissive.
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

    /// The material indices whose packed light colour differs in `new`, if
    /// `new` gives each material the same level and index as this table, so
    /// that only light colours changed. `None` if any material's level or
    /// index changed, or a material came or went.
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
    /// Whether any material gives a light colour other than white. A world
    /// whose table isn't solves and stores its light with no colour.
    #[inline]
    pub fn is_colored(&self) -> bool {
        self.colored
    }

    /// The packed light colour of each material index, in order.
    pub fn colors(&self) -> &[u16] {
        &self.colors
    }

    /// The packed light colour of the material index `material`.
    #[inline]
    pub fn color(&self, material: u16) -> u16 {
        self.colors.get(material as usize).copied().unwrap_or(WHITE)
    }
}

/// The index in `colors` of the light colour nearest `color`.
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
        // Half is 64 of 127 steps short of 1, so 32 of 63.
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

    /// The fixed mixing curve's weight at `level`, worked out from its formula.
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
        // So does one a step of 256 levels above level 1.
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
        // At the face: orange and red at 12, green at 11. Red's red and
        // green's green both beat orange's, so orange adds nothing.
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
        // Nine voxels in: orange and red at 3, green at 2. Orange's green now
        // beats green's.
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
        // Equal once scaled.
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
