//! Sky openness, a model of sky fill: each cell is lit by how much of the sky
//! it sees.
//!
//! Each open cell that isn't direct sky casts `rays` rays from its middle,
//! spread over the sky above it as a cosine-weighted hemisphere, so rays near
//! straight up count for more, as they do for light falling on a floor. A ray
//! steps cell by cell through the masks of the positions the field knows,
//! and reaches the sky when it enters a direct-sky cell before a cell holding
//! a solid, within `reach` voxels. A position the field doesn't know stops a
//! ray, so only columns proved clear count as sky. A cell's openness is the
//! share of its rays that reach the sky.
//!
//! Under the default sky falloff, `0.7 ^ (15 - level)`, a cell's brightness
//! is its openness × `gain`: its level is the one whose brightness that is,
//! up to 14. A cell seeing at least `sun_at` of the sky counts as direct sky
//! instead: it reads 15 and gets the sun. So the ground under a high floating
//! island loses the straight-down shade that spreading light can't fill, and
//! keeps only the sun's own shadow. A cave is black past what its opening
//! shows of the sky.
//!
//! Rays only rise, so a change to a position's cells casts again the cells of
//! the known positions below and beside it within `reach`, and its own.

use std::time::Duration;

use bevy::platform::collections::{HashMap, HashSet};
use bevy::platform::time::Instant;
use bevy::prelude::*;

use super::{CellFill, PositionCells, CELL};
use crate::light::config::OpenFill;

/// The highest level a cell that isn't direct sky holds, in sixteenths: 14.
const MAX_LEVEL: f32 = 14.0 * 16.0;

/// The level of a direct-sky cell, in sixteenths: 15.
const SKY_LEVEL: u8 = 15 * 16;

/// One direction of the rays every cell casts, ready for stepping cell by
/// cell from a cell's middle.
#[derive(Clone, Copy)]
struct Ray {
    /// By axis, the cell step: -1, 0 or 1.
    step: [i32; 3],
    /// By axis, how far along the ray, in cells, one cell's width is.
    delta: [f32; 3],
}

impl Ray {
    fn new(direction: Vec3) -> Self {
        let mut step = [0; 3];
        let mut delta = [f32::INFINITY; 3];
        for axis in 0..3 {
            let d = direction[axis];
            if d.abs() > 1e-6 {
                step[axis] = d.signum() as i32;
                delta[axis] = 1.0 / d.abs();
            }
        }
        Self { step, delta }
    }
}

/// `rays` directions over the hemisphere above a cell, cosine weighted, spread
/// around the vertical by the golden angle.
fn directions(rays: u32) -> Vec<Ray> {
    (0..rays)
        .map(|i| {
            let u = (i as f32 + 0.5) / rays as f32;
            let r = u.sqrt();
            let phi = i as f32 * 2.399_963;
            Ray::new(Vec3::new(r * phi.cos(), (1.0 - u).sqrt(), r * phi.sin()))
        })
        .collect()
}

struct Entry {
    cells: PositionCells,
    fill: CellFill,
    /// Whether `fill` has been worked out since the position was learned.
    worked_out: bool,
    /// The unknown positions that stopped a ray of a cell that isn't direct
    /// sky.
    asks: Vec<IVec3>,
}

/// What a cell's ray reads of a position.
#[derive(Clone, Copy)]
enum Look {
    NotRead,
    Unknown,
    Known(PositionCells),
}

/// The field of [`SkyFill::Openness`](crate::prelude::SkyFill::Openness).
pub(crate) struct OpenField {
    sun_at: f32,
    gain: f32,
    /// How far a ray goes, in cells.
    reach: f32,
    /// How many positions up and to each side the rays of a position's cells
    /// can reach.
    span: i32,
    rays: Vec<Ray>,
    entries: HashMap<IVec3, Entry>,
    /// What was learned (`Some`) or forgotten (`None`) since the last update.
    pending: HashMap<IVec3, Option<PositionCells>>,
    /// Known positions whose cells cast again.
    recast: HashSet<IVec3>,
    /// Forgotten positions not yet returned by an update.
    gone: Vec<IVec3>,
    /// Scratch for a position's rays: what each position they can reach
    /// reads as.
    looks: Vec<Look>,
}

impl OpenField {
    pub fn new(open: &OpenFill) -> Self {
        let reach = open.reach as f32 / CELL as f32;
        let span = (reach.ceil() as i32 + 3) / 4;
        Self {
            sun_at: open.sun_at,
            gain: open.gain,
            reach,
            span,
            rays: directions(open.rays),
            entries: HashMap::new(),
            pending: HashMap::new(),
            recast: HashSet::new(),
            gone: Vec::new(),
            looks: Vec::new(),
        }
    }

    pub fn learn(&mut self, position: IVec3, cells: PositionCells) {
        self.pending.insert(position, Some(cells));
    }

    pub fn forget(&mut self, position: IVec3) {
        self.pending.insert(position, None);
    }

    pub fn has_pending(&self) -> bool {
        !self.pending.is_empty() || !self.recast.is_empty()
    }

    pub fn cells(&self, position: IVec3) -> Option<&CellFill> {
        self.entries
            .get(&position)
            .filter(|entry| entry.worked_out)
            .map(|entry| &entry.fill)
    }

    /// Unknown positions that stopped a ray of a cell that could still get
    /// brighter, each with the position of that cell.
    pub fn wanted_probes(&self) -> impl Iterator<Item = (IVec3, IVec3)> + '_ {
        self.entries.iter().flat_map(move |(position, entry)| {
            entry
                .asks
                .iter()
                .filter(|asked| !self.entries.contains_key(*asked))
                .map(move |asked| (*asked, *position))
        })
    }

    pub fn bytes(&self) -> usize {
        self.entries.len() * (std::mem::size_of::<Entry>() + 16)
            + self
                .entries
                .values()
                .map(|entry| entry.asks.len() * std::mem::size_of::<IVec3>())
                .sum::<usize>()
    }

    pub fn update(&mut self, budget: Duration) -> Vec<IVec3> {
        let started = Instant::now();
        for (position, cells) in std::mem::take(&mut self.pending) {
            self.set(position, cells);
        }
        let mut changed = std::mem::take(&mut self.gone);
        let mut work: Vec<IVec3> = self.recast.drain().collect();
        while let Some(position) = work.pop() {
            if self.cast(position) {
                changed.push(position);
            }
            if started.elapsed() >= budget {
                break;
            }
        }
        // What the budget left waits for the next update.
        self.recast.extend(work);
        changed
    }

    /// Replaces what is known of `position` with `cells`, or forgets it, and
    /// marks the cells whose rays could pass through it to cast again.
    fn set(&mut self, position: IVec3, cells: Option<PositionCells>) {
        let old = self.entries.get(&position).map(|entry| entry.cells);
        if old == cells {
            return;
        }
        match cells {
            Some(cells) => {
                let entry = self.entries.entry(position).or_insert_with(|| Entry {
                    cells,
                    fill: CellFill {
                        open: 0,
                        direct: 0,
                        sunlit: 0,
                        levels: [0; 64],
                    },
                    worked_out: false,
                    asks: Vec::new(),
                });
                entry.cells = cells;
                self.recast.insert(position);
            }
            None => {
                let entry = self.entries.remove(&position);
                self.recast.remove(&position);
                if entry.is_some_and(|entry| entry.worked_out) {
                    self.gone.push(position);
                }
            }
        }
        let span = self.span;
        for y in -span..=0 {
            for z in -span..=span {
                for x in -span..=span {
                    let below = position + IVec3::new(x, y, z);
                    let casts = self
                        .entries
                        .get(&below)
                        .is_some_and(|entry| casting(entry.cells) != 0);
                    if casts {
                        self.recast.insert(below);
                    }
                }
            }
        }
    }

    /// Casts the rays of the cells of `position` and works out its cells
    /// again. Returns whether they changed, or were worked out for the first
    /// time.
    fn cast(&mut self, position: IVec3) -> bool {
        let Some(cells) = self.entries.get(&position).map(|entry| entry.cells) else {
            return false;
        };
        let span = self.span;
        let side = (2 * span + 1) as usize;
        self.looks.clear();
        self.looks
            .resize(side * side * (span as usize + 1), Look::NotRead);
        let mut levels = [0u8; 64];
        let mut sun = 0u64;
        let mut asks = Vec::new();
        let mut left = cells.open;
        while left != 0 {
            let bit = left.trailing_zeros() as usize;
            left &= left - 1;
            if cells.direct >> bit & 1 == 1 {
                levels[bit] = SKY_LEVEL;
                continue;
            }
            let cell =
                IVec3::new((bit & 3) as i32, (bit >> 2 & 3) as i32, (bit >> 4) as i32);
            let mut reached = 0;
            let mut asked = Vec::new();
            for ray in &self.rays {
                let mut at = cell;
                let mut next = ray.delta.map(|delta| delta * 0.5);
                loop {
                    let axis = if next[0] < next[1] {
                        if next[0] < next[2] {
                            0
                        } else {
                            2
                        }
                    } else if next[1] < next[2] {
                        1
                    } else {
                        2
                    };
                    if next[axis] > self.reach {
                        break;
                    }
                    at[axis] += ray.step[axis];
                    next[axis] += ray.delta[axis];
                    let offset = IVec3::new(at.x >> 2, at.y >> 2, at.z >> 2);
                    let index = (offset.x + span) as usize
                        + offset.y as usize * side
                        + (offset.z + span) as usize * side * (span as usize + 1);
                    let look = match self.looks[index] {
                        Look::NotRead => {
                            let look = match self.entries.get(&(position + offset)) {
                                Some(entry) => Look::Known(entry.cells),
                                None => Look::Unknown,
                            };
                            self.looks[index] = look;
                            look
                        }
                        look => look,
                    };
                    let Look::Known(seen) = look else {
                        asked.push(position + offset);
                        break;
                    };
                    let bit = (at.x & 3) + (at.y & 3) * 4 + (at.z & 3) * 16;
                    if seen.open >> bit & 1 == 0 {
                        break;
                    }
                    if seen.direct >> bit & 1 == 1 {
                        reached += 1;
                        break;
                    }
                }
            }
            let openness = reached as f32 / self.rays.len() as f32;
            if openness > 0.0 && openness >= self.sun_at {
                sun |= 1 << bit;
                levels[bit] = SKY_LEVEL;
                continue;
            }
            levels[bit] = level_of(openness * self.gain);
            asks.append(&mut asked);
        }
        asks.sort_unstable_by_key(|at| (at.x, at.y, at.z));
        asks.dedup();
        let fill = CellFill {
            open: cells.open,
            direct: cells.direct | sun,
            sunlit: sun & !cells.direct,
            levels,
        };
        let entry = self.entries.get_mut(&position).unwrap();
        entry.asks = asks;
        if entry.worked_out && entry.fill == fill {
            return false;
        }
        entry.fill = fill;
        entry.worked_out = true;
        true
    }
}

/// The cells of a position with `cells` that cast rays: open, and not direct
/// sky.
fn casting(cells: PositionCells) -> u64 {
    cells.open & !cells.direct
}

/// The level, in sixteenths, whose brightness under the default sky falloff,
/// `0.7 ^ (15 - level)`, is `brightness`, up to 14 levels.
fn level_of(brightness: f32) -> u8 {
    if brightness <= 0.0 {
        return 0;
    }
    let level = 15.0 + brightness.ln() / (1.0f32 / 0.7).ln();
    (level * 16.0).round().clamp(0.0, MAX_LEVEL) as u8
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::light::fill::{cell_bit, direct_cells};

    fn field() -> OpenField {
        OpenField::new(&OpenFill::default())
    }

    #[test]
    fn brightness_maps_to_the_level_of_the_default_falloff() {
        assert_eq!(level_of(0.0), 0);
        assert_eq!(level_of(0.7), 14 * 16);
        assert_eq!(level_of(0.49), 13 * 16);
        assert_eq!(level_of(2.0), 14 * 16);
    }

    #[test]
    fn a_cell_far_under_a_small_roof_counts_as_direct_sky_and_one_right_under_it_doesnt()
    {
        let mut field = field();
        // Air at y = 0 and up whose top cells all see the sky, except under a
        // roof of one cell in the position at the origin.
        let roof = cell_bit(1, 3, 1);
        for z in -5..=5 {
            for x in -5..=5 {
                for y in 0..=5 {
                    let position = IVec3::new(x, y, z);
                    let open = if position == IVec3::ZERO { !roof } else { !0 };
                    let top = if position.x == 0 && position.z == 0 && y == 0 {
                        !(1 << (1 + 4))
                    } else {
                        !0
                    };
                    field.learn(
                        position,
                        PositionCells {
                            open,
                            direct: direct_cells(open, top),
                        },
                    );
                }
            }
        }
        field.update(Duration::MAX);
        let cells = field.cells(IVec3::ZERO).unwrap();
        // Two cells below the roof, a cell sees most of the sky past its
        // edges.
        let far = cell_bit(1, 0, 1).trailing_zeros() as usize;
        assert_eq!(cells.direct >> far & 1, 1);
        assert_eq!(cells.sunlit >> far & 1, 1);
        assert_eq!(cells.levels[far], SKY_LEVEL);
        // Right under it, the roof hides about half the sky.
        let under = cell_bit(1, 2, 1).trailing_zeros() as usize;
        assert_eq!(cells.direct >> under & 1, 0);
        assert!((1..SKY_LEVEL).contains(&cells.levels[under]));
    }
}
