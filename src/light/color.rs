//! Light colour: what an emissive material's light colour is, how one is
//! packed into 16 bits, and how a chunk holds the light colour of its voxel
//! light.
//!
//! A light colour is linear RGB scaled so its brightest component is 1, so
//! the level alone says how bright light is. Only voxel light has one.
//!
//! The world's [`EmissionTable`] names each emissive material's level and
//! light colour. A chunk's inputs note which emissive materials it holds, so a
//! solve finds the light colours around its chunk from those notes alone.
//! Where every emissive material around a chunk gives white, the chunk solves
//! and stores its light as it would with no colour at all.

use bevy::log::warn_once;
use bevy::prelude::*;

use super::config::Emission;

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

/// The two components other than `top`, in order.
fn others(top: usize) -> [usize; 2] {
    match top {
        0 => [1, 2],
        1 => [0, 2],
        _ => [0, 1],
    }
}

/// How a chunk or light entry holds the light colour of its voxel light.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub(crate) enum ChunkColor {
    /// Every voxel's voxel light is white: nothing beside its levels.
    #[default]
    White,
    /// Every lit voxel's voxel light has the one packed light colour, never
    /// [`WHITE`].
    One(u16),
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

    /// The packed light colour of an open voxel with voxel light.
    pub fn packed(&self) -> u16 {
        match self {
            ChunkColor::White => WHITE,
            ChunkColor::One(packed) => *packed,
        }
    }

    /// The light colour of an open voxel with voxel light.
    pub fn color(&self) -> LinearRgba {
        match self {
            ChunkColor::White => LinearRgba::WHITE,
            ChunkColor::One(packed) => unpack(*packed),
        }
    }
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
