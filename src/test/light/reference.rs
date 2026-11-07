//! The reference flood: what light should be, worked out by brute force over a
//! flat array of a whole region, with every voxel in it known.
//!
//! It is written from the rules of light propagation alone and shares no code
//! with the crate's solver, so a test can hold the solver's light against it.
//! It is slow on purpose: every round of spreading visits every voxel.
//!
//! A channel is worked out from its sources, each a voxel and the distance it
//! starts at, so another kind of source, such as direct sky, adds sources
//! without changing the flood.

use bevy::prelude::*;

use crate::prelude::{LightShape, Sky, WorldVoxel};

mod fill;

pub(super) use fill::fill_reference;

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
    /// The voxel light level of every voxel.
    voxel: Vec<u8>,
    /// The sky light level of every voxel.
    sky: Vec<u8>,
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
        let size = max - min;
        let count = (size.x * size.y * size.z) as usize;
        let mut open = Vec::with_capacity(count);
        let mut emissive = Vec::with_capacity(count);
        for z in 0..size.z {
            for y in 0..size.y {
                for x in 0..size.x {
                    let voxel = voxel_at(min + IVec3::new(x, y, z));
                    open.push(!voxel.is_solid());
                    emissive.push(match voxel {
                        WorldVoxel::Solid(material) => {
                            emission(material).map(|level| level.round() as u8)
                        }
                        _ => None,
                    });
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
            voxel: vec![0; count],
            sky: vec![0; count],
        };

        let sources: Vec<(usize, u32)> = (0..count)
            .filter_map(|i| {
                let level = reference.emissive[i]?;
                reference.start_distance(level).map(|start| (i, start))
            })
            .collect();
        let distances = reference.flood(&sources);
        reference.voxel = distances
            .iter()
            .enumerate()
            .map(|(i, distance)| match reference.emissive[i] {
                Some(level) => level,
                None if !reference.open[i] => 0,
                None => reference.level(*distance),
            })
            .collect();
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
        let mut reference =
            Self::voxel_light(min, max, reach, shape, &voxel_at, |_| None);
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
                    15
                } else if !reference.open[i] {
                    0
                } else {
                    reference.level(*distance)
                }
            })
            .collect();
        reference
    }

    /// The voxel light level at `position`.
    pub fn voxel_at(&self, position: IVec3) -> u8 {
        self.voxel[self.index(position).expect("a position in the box")]
    }

    /// The sky light level at `position`.
    pub fn sky_at(&self, position: IVec3) -> u8 {
        self.sky[self.index(position).expect("a position in the box")]
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

    /// The level light holds `distance` steps from where it starts at level 15.
    fn level(&self, distance: u32) -> u8 {
        if distance == 0 || distance > self.reach {
            return 0;
        }
        let t = 1.0 - distance as f32 / (self.reach + 1) as f32;
        ((15.0 * t).round() as u8).clamp(1, 14)
    }

    /// The distance at which an emissive voxel at `level` starts: the first one
    /// at which light holds no more than its level. `None` if light within
    /// reach is always brighter, or the level is 0.
    fn start_distance(&self, level: u8) -> Option<u32> {
        if level >= 15 {
            return Some(0);
        }
        (1..=self.reach).find(|&distance| self.level(distance) <= level && level > 0)
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
