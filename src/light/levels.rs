//! Light levels: the scale gameplay code reads, and how distance from a source
//! turns into a level.

/// The light at a voxel, as read by
/// [`VoxelWorld::light_at`](crate::prelude::VoxelWorld::light_at).
///
/// Each kind of light is a level from 0, pitch black, to 15. Sky light at 15
/// means the voxel has direct sky.
#[derive(Clone, Copy, PartialEq, Debug, Default)]
#[non_exhaustive]
pub struct LightLevels {
    /// Light from the open sky.
    pub sky: f32,
    /// Light from emissive voxels.
    pub voxel: f32,
}

impl LightLevels {
    pub(crate) fn new(sky: u8, voxel: u8) -> Self {
        Self {
            sky: sky as f32,
            voxel: voxel as f32,
        }
    }

    /// The levels a slot's texel holds: sky light in its low nibble and voxel
    /// light in its high one, or for a lamp's own cell, 15 in the high nibble
    /// and the lamp's level in the low one.
    pub(crate) fn of_texel(texel: u8) -> Self {
        let (sky, voxel) = (texel & 15, texel >> 4);
        if voxel == 15 {
            Self::new(0, sky)
        } else {
            Self::new(sky, voxel)
        }
    }

    /// Whether sky light falls straight onto the voxel with no solid in the
    /// way, which is where Bevy's sun shines by default.
    pub fn direct_sky(&self) -> bool {
        self.sky >= 15.0
    }
}

/// The level light holds at each distance from a source, for one reach.
///
/// At `d` steps from a level-15 source, `t = 1 - d / (reach + 1)` and an open
/// voxel holds `round(15t)`, kept within 1 to 14, so only a source holds 15
/// and light within reach never fades to 0. At reach 14 this loses exactly one
/// level per step.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct LevelScale {
    reach: u32,
    /// The level at each distance, up to `reach + 1`.
    levels: Vec<u8>,
    /// By level, what [`LevelScale::first_distance`] returns.
    first: [u32; 16],
    /// By level, what [`LevelScale::last_distance`] returns.
    last: [u32; 16],
}

impl LevelScale {
    pub fn new(reach: u32) -> Self {
        let levels = (0..=reach + 1)
            .map(|distance| {
                if distance == 0 {
                    15
                } else if distance > reach {
                    0
                } else {
                    let t = 1.0 - distance as f32 / (reach + 1) as f32;
                    ((15.0 * t).round() as u8).clamp(1, 14)
                }
            })
            .collect();
        Self::with_levels(reach, levels)
    }

    /// The levels of a chunk above LOD level 0, whose cells are `width`
    /// voxels wide, for light of `reach` counted in voxels, so pools keep
    /// their size in voxels whatever the level.
    ///
    /// At `d` cells from a source, `t = 1 - (d - 0.5) × width / (reach + 1)`:
    /// a cell holds what its middle would, so the first cell away isn't
    /// charged its whole width. A cell holds `round(15t)`, kept within 1 to
    /// 14, or 0 once `t ≤ 0`. The scale's reach is one cell past the last
    /// lit one, at most [`MAX_REACH`](super::kernel::MAX_REACH), so a box of
    /// that reach also spreads light into a chunk's padding.
    pub fn coarse(reach: u32, width: u32) -> Self {
        let level = |distance: u32| -> u8 {
            if distance == 0 {
                return 15;
            }
            let t = 1.0 - (distance as f32 - 0.5) * width as f32 / (reach + 1) as f32;
            if t <= 0.0 {
                0
            } else {
                ((15.0 * t).round() as u8).clamp(1, 14)
            }
        };
        let lit = (1..).take_while(|&distance| level(distance) > 0).count() as u32;
        let box_reach = (lit + 1).min(super::kernel::MAX_REACH);
        let levels = (0..=box_reach + 1)
            .map(|distance| {
                if distance >= box_reach {
                    0
                } else {
                    level(distance)
                }
            })
            .collect();
        Self::with_levels(box_reach, levels)
    }

    fn with_levels(reach: u32, levels: Vec<u8>) -> Self {
        let mut scale = Self {
            reach,
            levels,
            first: [0; 16],
            last: [0; 16],
        };
        for level in 0..16u8 {
            scale.first[level as usize] = scale.find_first_distance(level);
            scale.last[level as usize] = scale.find_last_distance(level);
        }
        scale
    }

    pub fn reach(&self) -> u32 {
        self.reach
    }

    /// The level `distance` steps from a level-15 source.
    pub fn level(&self, distance: u32) -> u8 {
        self.levels
            .get(distance as usize)
            .copied()
            .unwrap_or_default()
    }

    /// The first distance from a level-15 source at which an open voxel holds
    /// `level`, or `reach + 1` for level 0. A level no distance holds counts
    /// as the first distance holding a dimmer one.
    #[inline]
    pub fn first_distance(&self, level: u8) -> u32 {
        self.first[level as usize & 15]
    }

    /// The last distance from a level-15 source at which an open voxel holds
    /// `level`, or `reach + 1` for level 0. A level no distance holds counts
    /// as the last distance holding a brighter one.
    #[inline]
    pub fn last_distance(&self, level: u8) -> u32 {
        self.last[level as usize & 15]
    }

    fn find_first_distance(&self, level: u8) -> u32 {
        (1..=self.reach)
            .find(|&distance| self.level(distance) <= level)
            .unwrap_or(self.reach + 1)
    }

    fn find_last_distance(&self, level: u8) -> u32 {
        if level == 0 {
            return self.reach + 1;
        }
        (1..=self.reach)
            .rev()
            .find(|&distance| self.level(distance) >= level)
            .unwrap_or(0)
    }

    /// The distance at which light from a source at `level` starts: the first
    /// distance whose level is no brighter than it, so the source's own voxel
    /// reads its level and the voxels beside it the level after. `None` when
    /// such a source lights nothing within reach.
    pub fn start(&self, level: u8) -> Option<u8> {
        if level == 0 {
            return None;
        }
        (0..=self.reach)
            .find(|&distance| self.level(distance) <= level)
            .filter(|&distance| distance < self.reach)
            .map(|distance| distance as u8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coarse_levels_count_reach_in_voxels_from_each_cells_middle() {
        // Cells 2 voxels wide at reach 14: t = 1 - (d - 0.5) × 2 / 15.
        let scale = LevelScale::coarse(14, 2);
        let levels: Vec<u8> = (0..=9).map(|distance| scale.level(distance)).collect();
        assert_eq!(levels, vec![15, 14, 12, 10, 8, 6, 4, 2, 0, 0]);
        assert_eq!(scale.reach(), 8, "one cell past the last lit one");

        // Cells about as wide as the reach light only the cells beside a
        // source.
        let wide = LevelScale::coarse(14, 16);
        assert_eq!((wide.level(1), wide.level(2), wide.reach()), (7, 0, 2));
    }
}
