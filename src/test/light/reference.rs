//! The reference flood
//!
//! This module works out what light should be by brute force, over a flat
//! array of a whole box of the world with every voxel in it known. It is
//! written from the rules of light propagation alone and shares no code with
//! the solver, so a test can hold the light of the solver against it. It is
//! slow on purpose: every round of spreading visits every voxel.
//!
//! Sky light and voxel light are each worked out from their sources. A source
//! is a voxel and the distance its light starts at, so another kind of
//! source, such as direct sky, adds sources without changing the flood.
//! Voxel light floods once for each light colour.
//!
//! Levels are held in steps, from 0 to `top`. Direct sky holds `top`, which is
//! 15 at 16 steps and 255 at 256 steps.

use bevy::prelude::*;

use crate::prelude::{LightShape, Sky, WorldVoxel};

mod fill;

pub(super) use fill::{fill_reference_everywhere, fill_reference_in_steps};

/// Not reached by any source.
const UNREACHED: u32 = u32::MAX;

/// The voxels of a box of the world and the light worked out in it.
pub(super) struct Reference {
    min: IVec3,
    size: IVec3,
    open: Vec<bool>,
    /// The declared level of each emissive voxel, rounded to a whole level.
    emissive: Vec<Option<u8>>,
    reach: u32,
    shape: LightShape,
    /// The top step: 15 at 16 levels, 255 at 256.
    pub top: u32,
    /// The voxel light level of every voxel.
    voxel: Vec<u8>,
    /// The sky light level of every voxel.
    sky: Vec<u8>,
    /// The light colour of the voxel light of every voxel, in linear RGB.
    color: Vec<[f32; 3]>,
}

/// Returns `color` scaled so its brightest component is 1, with each negative
/// component read as 0. Returns `None` for black.
fn brightest_at_1(color: [f32; 3]) -> Option<[f32; 3]> {
    let color = color.map(|component| component.max(0.0));
    let brightest = color[0].max(color[1]).max(color[2]);
    (brightest > 0.0).then(|| color.map(|component| component / brightest))
}

/// Returns the light colour that the light colours in `reaching` mix to, each
/// at its level on the 0 to 15 scale.
///
/// [`Reference::colored_voxel_light_in_steps`] describes the mix.
fn mixed(reaching: &[([f32; 3], f32)]) -> [f32; 3] {
    let weight = |level: f32| {
        (0.7f64.powf(15.0 - level as f64) - 0.7f64.powi(14)) / (1.0 - 0.7f64.powi(14))
    };
    let weights: Vec<f64> = reaching.iter().map(|(_, level)| weight(*level)).collect();
    let even = weights.iter().all(|weight| *weight <= 0.0);
    let mut mix = [0.0f64; 3];
    for ((color, _), weight) in reaching.iter().zip(weights) {
        let weight = if even { 1.0 } else { weight.max(0.0) };
        for component in 0..3 {
            mix[component] = mix[component].max(color[component] as f64 * weight);
        }
    }
    let brightest = mix[0].max(mix[1]).max(mix[2]);
    mix.map(|component| (component / brightest) as f32)
}

impl Reference {
    /// Works out voxel light in the box from `min` to `max`, exclusive, of the
    /// world whose voxels `voxel_at` returns. `emission` gives the declared
    /// level of an emissive material.
    ///
    /// Light from outside the box doesn't enter it, so only voxels at least
    /// `reach` inside its faces are sure to be right.
    pub fn voxel_light<I: Copy + PartialEq>(
        min: IVec3,
        max: IVec3,
        reach: u32,
        shape: LightShape,
        voxel_at: impl Fn(IVec3) -> WorldVoxel<I>,
        emission: impl Fn(I) -> Option<f32>,
    ) -> Self {
        Self::voxel_light_in_steps(15, min, max, reach, shape, voxel_at, emission)
    }

    /// [`voxel_light`](Self::voxel_light) with levels in steps up to `top`.
    pub fn voxel_light_in_steps<I: Copy + PartialEq>(
        top: u32,
        min: IVec3,
        max: IVec3,
        reach: u32,
        shape: LightShape,
        voxel_at: impl Fn(IVec3) -> WorldVoxel<I>,
        emission: impl Fn(I) -> Option<f32>,
    ) -> Self {
        Self::colored_voxel_light_in_steps(
            top,
            min,
            max,
            reach,
            shape,
            voxel_at,
            |material| emission(material).map(|level| (level, [1.0; 3])),
        )
    }

    /// Works as [`voxel_light_in_steps`](Self::voxel_light_in_steps), for
    /// emissive voxels that each give off a light colour. `emission` gives the
    /// declared level and the light colour, in linear RGB, of an emissive
    /// material.
    ///
    /// A light colour counts only by its hue and saturation. It is scaled so
    /// its brightest component is 1, and a negative component reads as 0.
    /// Black gives no light.
    ///
    /// Light of each light colour floods on its own. The voxel light level of
    /// a voxel is the highest level among the floods that reach it. The light
    /// colour of the voxel mixes those floods. Each component is the largest,
    /// over the floods, of the flood colour times a weight, and the mix is
    /// then scaled back to a brightest component of 1. A flood at `level`
    /// weighs `(0.7^(15 - level) - 0.7^14) / (1 - 0.7^14)`, which is 1 at
    /// level 15 and 0 at level 1. For example, red light at level 15 and green
    /// light at level 8 weigh 1 and about 0.076, so they mix to red with a
    /// green component of about 0.076.
    ///
    /// Where no weight is above 0, as where every flood reaches at level 1,
    /// each flood weighs 1. An emissive voxel has its own light colour. A
    /// voxel that no light reaches is white.
    pub fn colored_voxel_light_in_steps<I: Copy + PartialEq>(
        top: u32,
        min: IVec3,
        max: IVec3,
        reach: u32,
        shape: LightShape,
        voxel_at: impl Fn(IVec3) -> WorldVoxel<I>,
        emission: impl Fn(I) -> Option<(f32, [f32; 3])>,
    ) -> Self {
        let size = max - min;
        let count = (size.x * size.y * size.z) as usize;
        let mut open = Vec::with_capacity(count);
        let mut emissive = Vec::with_capacity(count);
        let mut own_color = Vec::with_capacity(count);
        for z in 0..size.z {
            for y in 0..size.y {
                for x in 0..size.x {
                    let voxel = voxel_at(min + IVec3::new(x, y, z));
                    open.push(!voxel.is_solid());
                    let declared = match voxel {
                        WorldVoxel::Solid(material) => emission(material),
                        _ => None,
                    };
                    let (level, color) = match declared {
                        Some((level, color)) => match brightest_at_1(color) {
                            Some(color) => (Some(level.round() as u8), color),
                            None => (Some(0), [1.0; 3]),
                        },
                        None => (None, [1.0; 3]),
                    };
                    emissive.push(level);
                    own_color.push(color);
                }
            }
        }
        let mut reference = Self {
            min,
            size,
            open,
            emissive,
            reach,
            shape,
            top,
            voxel: vec![0; count],
            sky: vec![0; count],
            color: vec![[1.0; 3]; count],
        };

        let mut colors: Vec<[f32; 3]> = Vec::new();
        for (level, color) in reference.emissive.iter().zip(&own_color) {
            if level.is_some_and(|level| level > 0) && !colors.contains(color) {
                colors.push(*color);
            }
        }
        // The level of every voxel in the flood of each light colour.
        let floods: Vec<([f32; 3], Vec<u8>)> = colors
            .iter()
            .map(|color| {
                let sources: Vec<(usize, u32)> = (0..count)
                    .filter(|&i| own_color[i] == *color)
                    .filter_map(|i| {
                        let level = reference.emissive[i]?;
                        reference
                            .start_distance(level as u32 * top / 15)
                            .map(|start| (i, start))
                    })
                    .collect();
                let distances = reference.flood(&sources);
                let levels = distances
                    .iter()
                    .enumerate()
                    .map(|(i, distance)| {
                        if reference.open[i] {
                            reference.level(*distance)
                        } else {
                            0
                        }
                    })
                    .collect();
                (*color, levels)
            })
            .collect();
        for i in 0..count {
            if let Some(level) = reference.emissive[i] {
                reference.voxel[i] = (level as u32 * top / 15) as u8;
                reference.color[i] = if level > 0 { own_color[i] } else { [1.0; 3] };
                continue;
            }
            let reaching: Vec<([f32; 3], f32)> = floods
                .iter()
                .filter(|(_, levels)| levels[i] > 0)
                .map(|(color, levels)| (*color, reference.read(levels[i])))
                .collect();
            let Some(level) = floods.iter().map(|(_, levels)| levels[i]).max() else {
                continue;
            };
            reference.voxel[i] = level;
            if level > 0 {
                reference.color[i] = mixed(&reaching);
            }
        }
        reference
    }

    /// Works out sky light in the box from `min` to `max`, exclusive, of the
    /// world whose voxels `voxel_at` returns, entering under the sky rule
    /// `sky`.
    ///
    /// An open voxel has direct sky, at level 15, when its column holds no
    /// solid from it up to the top of the chunk positions the rule says must
    /// be clear: with a sky clearance of N chunks, up to the top of the Nth
    /// chunk above the voxel's chunk; with a sky ceiling, up to the top of the
    /// chunk holding the ceiling, or of the voxel's own chunk if that is
    /// higher. Light from direct sky then spreads like any other, and voxel
    /// light is 0 everywhere. `voxel_at` is asked about voxels above the box
    /// too.
    ///
    /// Light from outside the box doesn't enter it, so only voxels at least
    /// `reach` inside its faces are sure to be right.
    pub fn sky_light<I: Copy + PartialEq>(
        min: IVec3,
        max: IVec3,
        reach: u32,
        shape: LightShape,
        sky: &Sky,
        voxel_at: impl Fn(IVec3) -> WorldVoxel<I>,
    ) -> Self {
        Self::sky_light_in_steps(15, min, max, reach, shape, sky, voxel_at)
    }

    /// [`sky_light`](Self::sky_light) with levels in steps up to `steps`.
    pub fn sky_light_in_steps<I: Copy + PartialEq>(
        steps: u32,
        min: IVec3,
        max: IVec3,
        reach: u32,
        shape: LightShape,
        sky: &Sky,
        voxel_at: impl Fn(IVec3) -> WorldVoxel<I>,
    ) -> Self {
        let mut reference =
            Self::voxel_light_in_steps(steps, min, max, reach, shape, &voxel_at, |_| {
                None
            });
        let size = reference.size;
        // The highest voxel each chunk's column must be clear up to.
        let top = |y: i32| -> Option<i32> {
            let chunk = y.div_euclid(32);
            match sky {
                Sky::Clearance { chunks } => Some((chunk + 1 + *chunks as i32) * 32 - 1),
                Sky::Ceiling { y } => Some((chunk.max(y.div_euclid(32)) + 1) * 32 - 1),
                Sky::Off => None,
            }
        };
        let mut direct = vec![false; reference.open.len()];
        if let Some(highest) = top(max.y - 1) {
            for z in 0..size.z {
                for x in 0..size.x {
                    // The lowest solid at or above the voxel, walking down.
                    let mut solid_from = i32::MAX;
                    for y in (min.y..=highest).rev() {
                        let p = IVec3::new(min.x + x, y, min.z + z);
                        let open = match reference.index(p) {
                            Some(i) => reference.open[i],
                            None => !voxel_at(p).is_solid(),
                        };
                        if !open {
                            solid_from = y;
                            continue;
                        }
                        if let Some(i) = reference.index(p) {
                            direct[i] = solid_from > top(y).unwrap();
                        }
                    }
                }
            }
        }
        let sources: Vec<(usize, u32)> = (0..direct.len())
            .filter(|&i| direct[i])
            .map(|i| (i, 0))
            .collect();
        let distances = reference.flood(&sources);
        reference.sky = distances
            .iter()
            .enumerate()
            .map(|(i, distance)| {
                if direct[i] {
                    steps as u8
                } else if !reference.open[i] {
                    0
                } else {
                    reference.level(*distance)
                }
            })
            .collect();
        reference
    }

    /// Works out sky light in the box from `min` to `max`, exclusive, spreading
    /// from the open voxels `direct` says have direct sky, at level 15. Voxel
    /// light is 0 everywhere.
    ///
    /// Light from outside the box doesn't enter it, so only voxels at least
    /// `reach` inside its faces are sure to be right.
    #[cfg(feature = "sub_grids")]
    pub fn sky_from<I: Copy + PartialEq>(
        min: IVec3,
        max: IVec3,
        reach: u32,
        shape: LightShape,
        voxel_at: impl Fn(IVec3) -> WorldVoxel<I>,
        direct: impl Fn(IVec3) -> bool,
    ) -> Self {
        let mut reference =
            Self::voxel_light(min, max, reach, shape, &voxel_at, |_| None);
        let size = reference.size;
        let position = |i: usize| {
            let i = i as i32;
            min + IVec3::new(i % size.x, (i / size.x) % size.y, i / (size.x * size.y))
        };
        let direct: Vec<bool> = (0..reference.open.len())
            .map(|i| reference.open[i] && direct(position(i)))
            .collect();
        let sources: Vec<(usize, u32)> = (0..direct.len())
            .filter(|&i| direct[i])
            .map(|i| (i, 0))
            .collect();
        let distances = reference.flood(&sources);
        reference.sky = distances
            .iter()
            .enumerate()
            .map(|(i, distance)| {
                if direct[i] {
                    reference.top as u8
                } else if !reference.open[i] {
                    0
                } else {
                    reference.level(*distance)
                }
            })
            .collect();
        reference
    }

    /// The voxel light level at `position`, in steps.
    pub fn voxel_at(&self, position: IVec3) -> u8 {
        self.voxel[self.index(position).expect("a position in the box")]
    }

    /// Returns whether `position` lies in the box.
    pub fn holds(&self, position: IVec3) -> bool {
        self.index(position).is_some()
    }

    /// Returns the light colour of the voxel light at `position`, in linear
    /// RGB.
    pub fn color_at(&self, position: IVec3) -> [f32; 3] {
        self.color[self.index(position).expect("a position in the box")]
    }

    /// The sky light level at `position`, in steps.
    pub fn sky_at(&self, position: IVec3) -> u8 {
        self.sky[self.index(position).expect("a position in the box")]
    }

    /// `steps` read on the 0 to 15 scale `light_at` reads.
    pub fn read(&self, steps: u8) -> f32 {
        steps as f32 * 15.0 / self.top as f32
    }

    fn index(&self, position: IVec3) -> Option<usize> {
        let local = position - self.min;
        if local.cmplt(IVec3::ZERO).any() || local.cmpge(self.size).any() {
            return None;
        }
        Some(
            (local.x + local.y * self.size.x + local.z * self.size.x * self.size.y)
                as usize,
        )
    }

    /// The level light holds `distance` steps from where it starts at the top
    /// step.
    fn level(&self, distance: u32) -> u8 {
        if distance == 0 || distance > self.reach {
            return 0;
        }
        let top = self.top as f32;
        let t = 1.0 - distance as f32 / (self.reach + 1) as f32;
        ((top * t).round()).clamp(1.0, top - 1.0) as u8
    }

    /// The distance at which an emissive voxel at `level`, in steps, starts:
    /// the first one at which light holds no more than its level. `None` if
    /// light within reach is always brighter, or the level is 0.
    fn start_distance(&self, level: u32) -> Option<u32> {
        if level >= self.top {
            return Some(0);
        }
        (1..=self.reach)
            .find(|&distance| self.level(distance) as u32 <= level && level > 0)
    }

    /// The distance at which light first reaches each voxel from `sources`.
    ///
    /// Light starts at a source's voxel at its distance, then spreads one round
    /// at a time into open voxels. A diamond round is one face step. A round
    /// shape makes every second round three passes instead, along x, then y,
    /// then z, each spreading one step from everything reached so far.
    fn flood(&self, sources: &[(usize, u32)]) -> Vec<u32> {
        let count = self.open.len();
        let mut distance = vec![UNREACHED; count];
        if sources.is_empty() {
            return distance;
        }
        // When each voxel was reached, as a count of passes, so a pass spreads
        // only from voxels reached before it began.
        let mut reached_at = vec![UNREACHED; count];
        let mut pass = 0;
        for round in 1..=self.reach {
            for &(i, start) in sources {
                if start + 1 == round {
                    reached_at[i] = pass;
                }
            }
            pass += 1;
            let axes: &[usize] = if self.shape == LightShape::Round && round % 2 == 0 {
                &[0, 1, 2]
            } else {
                &[3]
            };
            for &axis in axes {
                for i in 0..count {
                    if !self.open[i] || reached_at[i] != UNREACHED {
                        continue;
                    }
                    let from_before =
                        self.neighbours(i, axis).any(|n| reached_at[n] < pass);
                    if from_before {
                        reached_at[i] = pass;
                        distance[i] = round;
                    }
                }
                pass += 1;
            }
        }
        distance
    }

    /// The voxels a step from voxel `i` along `axis`, or along every axis for
    /// axis 3.
    fn neighbours(&self, i: usize, axis: usize) -> impl Iterator<Item = usize> + '_ {
        let size = self.size;
        let local = IVec3::new(
            i as i32 % size.x,
            (i as i32 / size.x) % size.y,
            i as i32 / (size.x * size.y),
        );
        let axes = if axis == 3 { 0..3 } else { axis..axis + 1 };
        axes.flat_map(move |axis| {
            [-1, 1].into_iter().map(move |way| {
                let mut step = IVec3::ZERO;
                step[axis] = way;
                local + step
            })
        })
        .filter_map(move |neighbour| self.index(neighbour + self.min))
    }
}
