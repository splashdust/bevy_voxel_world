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
