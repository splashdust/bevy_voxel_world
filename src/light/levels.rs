//! Light levels: the scale gameplay code reads, and how distance from a source
//! turns into a level.
//!
//! Inside the crate a level is a whole number of steps of the world's
//! [`LightPrecision`]: 0 to 15 at 16 levels, 0 to 255 at 256. Gameplay code
//! reads every level on the 0 to 15 scale, where one step at 256 levels is
//! 1/17 of a level.
//!
//! # References
//!
//! - [Minecraft Wiki, "Light"](https://minecraft.wiki/w/Light):
//!   a 0 to 15 scale for sky light and for light from blocks, the scale
//!   gameplay code reads here.

use super::config::LightPrecision;

impl LightPrecision {
    /// The top step: direct sky, and the marker of a lamp's own cell in a
    /// slot.
    #[inline]
    pub(crate) fn top(self) -> u8 {
        match self {
            LightPrecision::Steps16 => 15,
            LightPrecision::Steps256 => 255,
        }
    }

    /// How many bits a level has, which is how many bit planes a channel
    /// stores for a chunk.
    #[inline]
    pub(crate) fn bits(self) -> usize {
        match self {
            LightPrecision::Steps16 => 4,
            LightPrecision::Steps256 => 8,
        }
    }

    /// Steps per whole level of the 0 to 15 scale: 1, or 17 at 256 levels.
    #[inline]
    fn per_level(self) -> u32 {
        self.top() as u32 / 15
    }

    /// The steps of the whole level `level`, from 0 to 15.
    #[inline]
    pub(crate) fn of_whole(self, level: u8) -> u8 {
        (level.min(15) as u32 * self.per_level()) as u8
    }

    /// `steps` read on the 0 to 15 scale.
    #[inline]
    pub(crate) fn level_of(self, steps: u8) -> f32 {
        steps as f32 / self.per_level() as f32
    }

    /// `steps` of the precision `from` as steps of this one, rounded to the
    /// nearest. The top step stays the top step, and no other level becomes
    /// it or rounds to 0, so direct sky and darkness keep their meaning.
    #[inline]
    pub(crate) fn convert(self, steps: u8, from: LightPrecision) -> u8 {
        if steps == from.top() {
            return self.top();
        }
        if steps == 0 {
            return 0;
        }
        let (to, from) = (self.per_level(), from.per_level());
        let rounded = (steps as u32 * to + from / 2) / from;
        (rounded as u8).clamp(1, self.top() - 1)
    }
}

/// The light at a voxel, as read by
/// [`VoxelWorld::light_at`](crate::prelude::VoxelWorld::light_at).
///
/// Each kind of light is a level from 0, pitch black, to 15. Sky light at 15
/// means the voxel has direct sky. At 16 levels every level is a whole
/// number, and at 256 levels a step of 1/17.
#[derive(Clone, Copy, PartialEq, Debug, Default)]
#[non_exhaustive]
pub struct LightLevels {
    /// Light from the open sky.
    pub sky: f32,
    /// Light from emissive voxels.
    pub voxel: f32,
}

impl LightLevels {
    /// Whole levels `sky` and `voxel`, from 0 to 15.
    pub(crate) fn new(sky: u8, voxel: u8) -> Self {
        Self {
            sky: sky as f32,
            voxel: voxel as f32,
        }
    }

    /// Levels held as steps of `precision`.
    pub(crate) fn of_steps(sky: u8, voxel: u8, precision: LightPrecision) -> Self {
        Self {
            sky: precision.level_of(sky),
            voxel: precision.level_of(voxel),
        }
    }

    /// Whether sky light falls straight onto the voxel with no solid in the
    /// way, which is where Bevy's sun shines by default.
    pub fn direct_sky(&self) -> bool {
        self.sky >= 15.0
    }
}

/// The level light holds at each distance from a source, for one reach and
/// precision, in steps of the precision.
///
/// At `d` steps from a source at the top step, `t = 1 - d / (reach + 1)` and
/// an open voxel holds `round(top × t)`, kept within 1 to `top - 1`, so only
/// a source holds the top step and light within reach never fades to 0. At 16
/// levels and reach 14 this loses exactly one level per step.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct LevelScale {
    reach: u32,
    precision: LightPrecision,
    /// The level at each distance, up to `reach + 1`.
    levels: Vec<u8>,
    /// By level, what [`LevelScale::first_distance`] returns.
    first: Vec<u32>,
    /// By level, what [`LevelScale::last_distance`] returns.
    last: Vec<u32>,
}

impl LevelScale {
    pub fn new(reach: u32, precision: LightPrecision) -> Self {
        let top = precision.top();
        let levels = (0..=reach + 1)
            .map(|distance| {
                if distance == 0 {
                    top
                } else if distance > reach {
                    0
                } else {
                    let t = 1.0 - distance as f32 / (reach + 1) as f32;
                    ((top as f32 * t).round() as u8).clamp(1, top - 1)
                }
            })
            .collect();
        Self::with_levels(reach, precision, levels)
    }

    /// The levels of a chunk above LOD level 0, whose cells are `width`
    /// voxels wide, for light of `reach` counted in voxels, so pools keep
    /// their size in voxels whatever the level.
    ///
    /// At `d` cells from a source, `t = 1 - (d - 0.5) × width / (reach + 1)`:
    /// a cell holds what its middle would, so the first cell away isn't
    /// charged its whole width. A cell holds `round(top × t)`, kept within 1
    /// to `top - 1`, or 0 once `t ≤ 0`. The scale's reach is one cell past the
    /// last lit one, at most [`MAX_REACH`](super::kernel::MAX_REACH), so a box
    /// of that reach also spreads light into a chunk's padding.
    pub fn coarse(reach: u32, width: u32, precision: LightPrecision) -> Self {
        let top = precision.top();
        let level = |distance: u32| -> u8 {
            if distance == 0 {
                return top;
            }
            let t = 1.0 - (distance as f32 - 0.5) * width as f32 / (reach + 1) as f32;
            if t <= 0.0 {
                0
            } else {
                ((top as f32 * t).round() as u8).clamp(1, top - 1)
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
        Self::with_levels(box_reach, precision, levels)
    }

    fn with_levels(reach: u32, precision: LightPrecision, levels: Vec<u8>) -> Self {
        let mut scale = Self {
            reach,
            precision,
            levels,
            first: Vec::new(),
            last: Vec::new(),
        };
        let steps = 0..=precision.top();
        scale.first = steps
            .clone()
            .map(|level| scale.find_first_distance(level))
            .collect();
        scale.last = steps.map(|level| scale.find_last_distance(level)).collect();
        scale
    }

    pub fn reach(&self) -> u32 {
        self.reach
    }

    pub fn precision(&self) -> LightPrecision {
        self.precision
    }

    /// The level `distance` steps from a source at the top step.
    pub fn level(&self, distance: u32) -> u8 {
        self.levels
            .get(distance as usize)
            .copied()
            .unwrap_or_default()
    }

    /// The first distance from a source at the top step at which an open
    /// voxel holds `level`, or `reach + 1` for level 0. A level no distance
    /// holds counts as the first distance holding a dimmer one.
    #[inline]
    pub fn first_distance(&self, level: u8) -> u32 {
        self.first[(level as usize).min(self.first.len() - 1)]
    }

    /// The last distance from a source at the top step at which an open
    /// voxel holds `level`, or `reach + 1` for level 0. A level no distance
    /// holds counts as the last distance holding a brighter one.
    #[inline]
    pub fn last_distance(&self, level: u8) -> u32 {
        self.last[(level as usize).min(self.last.len() - 1)]
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
        let scale = LevelScale::coarse(14, 2, LightPrecision::Steps16);
        let levels: Vec<u8> = (0..=9).map(|distance| scale.level(distance)).collect();
        assert_eq!(levels, vec![15, 14, 12, 10, 8, 6, 4, 2, 0, 0]);
        assert_eq!(scale.reach(), 8, "one cell past the last lit one");

        // Cells about as wide as the reach light only the cells beside a
        // source.
        let wide = LevelScale::coarse(14, 16, LightPrecision::Steps16);
        assert_eq!((wide.level(1), wide.level(2), wide.reach()), (7, 0, 2));
    }

    #[test]
    fn at_256_levels_a_voxel_holds_round_255t_kept_within_1_to_254() {
        let scale = LevelScale::new(30, LightPrecision::Steps256);
        // t = 1 - d / 31.
        assert_eq!(scale.level(0), 255);
        assert_eq!(scale.level(1), 247);
        assert_eq!(scale.level(15), 132);
        assert_eq!(scale.level(30), 8);
        assert_eq!(scale.level(31), 0);
        // Reach 254 would hold 1 at its far end, but reach stops at 32: the
        // clamp keeps the voxel beside a source below direct sky.
        let short = LevelScale::new(1, LightPrecision::Steps256);
        assert_eq!(short.level(1), 128);
    }

    #[test]
    fn levels_convert_between_precisions_on_the_0_to_15_scale() {
        use LightPrecision::*;
        assert_eq!(Steps256.convert(14, Steps16), 238);
        assert_eq!(Steps16.convert(238, Steps256), 14);
        // Only direct sky is the top level, and only darkness is 0.
        assert_eq!(Steps16.convert(247, Steps256), 14);
        assert_eq!(Steps16.convert(255, Steps256), 15);
        assert_eq!(Steps16.convert(8, Steps256), 1);
        assert_eq!(Steps256.level_of(247), 247.0 / 17.0);
        assert_eq!(Steps256.of_whole(15), 255);
    }
}
