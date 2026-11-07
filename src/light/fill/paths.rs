//! Paths, the default model of sky fill: levels on the open regions of
//! positions, passed between regions through the openings between them.
//!
//! A position's open cells split into regions, the groups of open cells
//! joined by faces, so air above and below a floor stays apart. A region
//! passes light to a region of a neighbouring position through the cells
//! they share on that face, losing `whole_face_loss × (16 / f) ^
//! opening_power` levels, where `f`, from 1 to 16, is how much of the
//! receiving region's widest layer of cells the opening fills. The opening's
//! cost is judged against the region behind it: a small door into a big room
//! loses a lot, while a pocket open across its whole width loses as little as
//! a whole open face. A 1×1 shaft holds no open cell, so it carries nothing,
//! and a sealed space has no path in.
//!
//! The sky is one more neighbour, reached through a region's top cells whose
//! 64 columns are all proved clear: a region is worth one whole face's loss
//! more than `beside_open_sky` when its whole top sees the sky, so the region
//! beside it holds `beside_open_sky`, and less through fewer top cells.
//!
//! Levels are kept in sixteenths of a level, and kept current as positions
//! are learned and forgotten with two queues, as a light engine keeps light
//! current: a removal queue takes out every level that came through what
//! changed, then an increase queue floods again from what is left standing.
//! Each step through a whole face costs 2 levels by default, so a change
//! reaches about 7 positions.
//!
//! Two limits follow from the rule. A cave behind an opening narrower than
//! the cave is black past the pool of light the opening's cells give it. And
//! a sky full of small floating things gets no fill: a top cell rarely has
//! all 64 columns clear, so no region sees the sky.

use std::collections::BinaryHeap;
use std::time::Duration;

use bevy::platform::collections::{HashMap, HashSet};
use bevy::platform::time::Instant;
use bevy::prelude::*;

use super::{CellFill, PositionCells};
use crate::light::config::PathFill;

/// The six steps to a position's face neighbours, in the order of a region's
/// faces: -X, +X, -Y, +Y, -Z, +Z. A face's opposite is at `face ^ 1`.
const STEPS: [IVec3; 6] = [
    IVec3::NEG_X,
    IVec3::X,
    IVec3::NEG_Y,
    IVec3::Y,
    IVec3::NEG_Z,
    IVec3::Z,
];

const CELL_X0: u64 = 0x1111_1111_1111_1111;
const CELL_X3: u64 = 0x8888_8888_8888_8888;
const CELL_Y0: u64 = 0x000F_000F_000F_000F;
const CELL_Y3: u64 = 0xF000_F000_F000_F000;
/// The cells of a position's top layer.
const TOP: u64 = 0xF000_F000_F000_F000;

/// The highest level a region can hold, in sixteenths: 15.
const MAX_LEVEL: i32 = 15 * 16;

/// `cells` grown by one face step, kept within `open`.
#[inline]
fn grow(cells: u64, open: u64) -> u64 {
    (cells
        | (cells & !CELL_X3) << 1
        | (cells & !CELL_X0) >> 1
        | (cells & !CELL_Y3) << 4
        | (cells & !CELL_Y0) >> 4
        | cells << 16
        | cells >> 16)
        & open
}

/// The 16 cells on each face of `cells`, as bits `a + b × 4`, by face in
/// [`STEPS`] order, laid out so a face and its opposite on the neighbour
/// line up.
fn faces(cells: u64) -> [u16; 6] {
    if cells == !0 {
        return [!0; 6];
    }
    let mut faces = [0u16; 6];
    let at = |x: usize, y: usize, z: usize| cells >> (x + y * 4 + z * 16) & 1 == 1;
    for b in 0..4 {
        for a in 0..4 {
            let bit = 1u16 << (a + b * 4);
            let on = [
                at(0, a, b),
                at(3, a, b),
                at(a, 0, b),
                at(a, 3, b),
                at(a, b, 0),
                at(a, b, 3),
            ];
            for (face, on) in faces.iter_mut().zip(on) {
                if on {
                    *face |= bit;
                }
            }
        }
    }
    faces
}

/// The most cells in any one layer of `cells`, along any axis.
fn widest(cells: u64) -> u8 {
    if cells == !0 {
        return 16;
    }
    let mut most = 1;
    for layer in 0..4 {
        let (mut x, mut y, mut z) = (0u8, 0u8, 0u8);
        for b in 0..4 {
            for a in 0..4 {
                x += (cells >> (layer + a * 4 + b * 16) & 1) as u8;
                y += (cells >> (a + layer * 4 + b * 16) & 1) as u8;
                z += (cells >> (a + b * 4 + layer * 16) & 1) as u8;
            }
        }
        most = most.max(x).max(y).max(z);
    }
    most
}

/// What the rule costs, in sixteenths of a level.
#[derive(Clone, Copy)]
struct Costs {
    /// The level beside open sky.
    beside_open_sky: i32,
    /// By how many cells of a whole face an opening holds, judged on its
    /// own: what entering a region from the sky costs.
    absolute: [i32; 17],
    /// By the cells of the opening and then the receiving region's widest
    /// layer: what passing through the opening costs.
    through: [[i32; 17]; 17],
}

impl Costs {
    fn new(paths: &PathFill) -> Self {
        let loss = |f: f32| -> i32 {
            let levels = paths.whole_face_loss * (16.0 / f).powf(paths.opening_power);
            (levels * 16.0).min(MAX_LEVEL as f32 * 8.0) as i32
        };
        let mut absolute = [i32::MAX; 17];
        let mut through = [[i32::MAX; 17]; 17];
        for aperture in 1..=16 {
            absolute[aperture] = loss(aperture as f32);
            for (widest, through) in through[aperture].iter_mut().enumerate().skip(1) {
                *through = loss(aperture as f32 * 16.0 / widest as f32);
            }
        }
        Self {
            beside_open_sky: (paths.beside_open_sky * 16.0).round() as i32,
            absolute,
            through,
        }
    }

    /// What a region is worth from the sky through `cells` top cells that
    /// see it: one whole face's loss above `beside_open_sky` through a whole
    /// top, so its neighbour through a whole face holds `beside_open_sky`.
    fn sky_gives(&self, cells: u32) -> i32 {
        if cells == 0 {
            return 0;
        }
        (self.beside_open_sky + 2 * self.absolute[16] - self.absolute[cells as usize])
            .clamp(0, MAX_LEVEL)
    }

    /// What light at `level` holds past an opening of `aperture` cells into
    /// a region whose widest layer holds `widest` cells.
    #[inline]
    fn through(&self, level: i32, aperture: u32, widest: u8) -> i32 {
        level - self.through[aperture as usize][widest as usize]
    }
}

#[derive(Clone, Copy)]
struct Region {
    cells: u64,
    faces: [u16; 6],
    widest: u8,
    /// In sixteenths of a level.
    level: i32,
    /// What the sky gives it, through its own top cells.
    from_sky: i32,
}

struct Entry {
    cells: PositionCells,
    regions: Vec<Region>,
    fill: CellFill,
}

/// A region at a position, by its index in the position's regions.
type At = (IVec3, usize);

fn key(at: At) -> (i32, i32, i32, usize) {
    (at.0.x, at.0.y, at.0.z, at.1)
}

fn unkey(key: (i32, i32, i32, usize)) -> At {
    (IVec3::new(key.0, key.1, key.2), key.3)
}

/// The field of [`SkyFill::Paths`](crate::prelude::SkyFill::Paths).
pub(crate) struct PathField {
    costs: Costs,
    entries: HashMap<IVec3, Entry>,
    /// What was learned (`Some`) or forgotten (`None`) since the last update.
    pending: HashMap<IVec3, Option<PositionCells>>,
    /// Positions whose levels may have changed during the current update.
    touched: HashSet<IVec3>,
    /// Positions learned during the current update that the field didn't
    /// know before.
    created: HashSet<IVec3>,
    /// Unknown positions beside a region lit well enough to light them.
    wanted: HashSet<IVec3>,
    regions: usize,
}

impl PathField {
    pub fn new(paths: &PathFill) -> Self {
        Self {
            costs: Costs::new(paths),
            entries: HashMap::new(),
            pending: HashMap::new(),
            touched: HashSet::new(),
            created: HashSet::new(),
            wanted: HashSet::new(),
            regions: 0,
        }
    }

    pub fn learn(&mut self, position: IVec3, cells: PositionCells) {
        self.pending.insert(position, Some(cells));
    }

    pub fn forget(&mut self, position: IVec3) {
        self.pending.insert(position, None);
    }

    pub fn has_pending(&self) -> bool {
        !self.pending.is_empty()
    }

    pub fn cells(&self, position: IVec3) -> Option<&CellFill> {
        self.entries.get(&position).map(|entry| &entry.fill)
    }

    pub fn wanted_probes(&self) -> impl Iterator<Item = IVec3> + '_ {
        self.wanted.iter().copied()
    }

    pub fn bytes(&self) -> usize {
        self.entries.len() * (std::mem::size_of::<Entry>() + 16)
            + self.regions * std::mem::size_of::<Region>()
    }

    pub fn update(&mut self, budget: Duration) -> Vec<IVec3> {
        let started = Instant::now();
        self.touched.clear();
        let mut pending: Vec<(IVec3, Option<PositionCells>)> =
            self.pending.drain().collect();
        while let Some((position, cells)) = pending.pop() {
            self.set(position, cells);
            if started.elapsed() >= budget {
                break;
            }
        }
        // What the budget left waits for the next update.
        self.pending.extend(pending);
        let touched: Vec<IVec3> = self.touched.drain().collect();
        let mut changed = Vec::new();
        for position in &touched {
            if self.refresh(*position) || self.created.contains(position) {
                changed.push(*position);
            }
        }
        self.created.clear();
        for position in touched {
            self.refresh_wanted(position);
            for step in STEPS {
                self.refresh_wanted(position + step);
            }
        }
        changed
    }

    /// Works out the cells of `position` again from its regions. Returns
    /// whether they changed, or the position came or went.
    fn refresh(&mut self, position: IVec3) -> bool {
        let Some(entry) = self.entries.get_mut(&position) else {
            return true;
        };
        let mut levels = [0u8; 64];
        let top = self.costs.beside_open_sky;
        for region in &entry.regions {
            let level = region.level.clamp(0, top) as u8;
            let mut cells = region.cells;
            while cells != 0 {
                let bit = cells.trailing_zeros() as usize;
                cells &= cells - 1;
                levels[bit] = if entry.cells.direct >> bit & 1 == 1 {
                    top.min(MAX_LEVEL) as u8
                } else {
                    level
                };
            }
        }
        let fill = CellFill {
            open: entry.cells.open,
            direct: entry.cells.direct,
            levels,
        };
        if entry.fill == fill {
            return false;
        }
        entry.fill = fill;
        true
    }

    /// Whether the unknown `position` could be lit by a known neighbour, so
    /// probing it could raise a level.
    fn refresh_wanted(&mut self, position: IVec3) {
        if self.entries.contains_key(&position) {
            self.wanted.remove(&position);
            return;
        }
        let whole_face = self.costs.through[16][16];
        let lit = STEPS.iter().enumerate().any(|(face, step)| {
            let Some(entry) = self.entries.get(&(position - *step)) else {
                return false;
            };
            entry
                .regions
                .iter()
                .any(|region| region.faces[face] != 0 && region.level > whole_face)
        });
        if lit {
            self.wanted.insert(position);
        } else {
            self.wanted.remove(&position);
        }
    }

    /// The regions of a position with `cells`.
    fn regions(&self, cells: PositionCells) -> Vec<Region> {
        let mut regions = Vec::new();
        let mut left = cells.open;
        while left != 0 {
            let mut region = left.isolate_lowest_one();
            if cells.open == !0 {
                region = !0;
            }
            loop {
                let next = grow(region, cells.open);
                if next == region {
                    break;
                }
                region = next;
            }
            let top = (region & TOP & cells.direct).count_ones();
            regions.push(Region {
                cells: region,
                faces: faces(region),
                widest: widest(region),
                level: 0,
                from_sky: self.costs.sky_gives(top),
            });
            left &= !region;
        }
        regions
    }

    /// Replaces what is known of `position` with `cells`, or forgets it, and
    /// brings every level it changes up to date.
    fn set(&mut self, position: IVec3, cells: Option<PositionCells>) {
        let old = self.entries.get(&position);
        if old.map(|entry| entry.cells) == cells {
            return;
        }
        if old.is_none() && cells.is_none() {
            return;
        }
        let costs = self.costs;
        self.touched.insert(position);
        let mut removal: Vec<(IVec3, Region)> = Vec::new();
        if let Some(old) = old {
            for region in &old.regions {
                if region.level > 0 {
                    removal.push((position, *region));
                }
            }
            self.regions -= old.regions.len();
        }
        match cells {
            Some(cells) => {
                if old.is_none() {
                    self.created.insert(position);
                }
                let regions = self.regions(cells);
                self.regions += regions.len();
                let fill = self
                    .entries
                    .remove(&position)
                    .map(|entry| entry.fill)
                    .unwrap_or(CellFill {
                        open: 0,
                        direct: 0,
                        levels: [0; 64],
                    });
                self.entries.insert(
                    position,
                    Entry {
                        cells,
                        regions,
                        fill,
                    },
                );
            }
            None => {
                self.entries.remove(&position);
            }
        }

        // Takes out every level that came through what was there.
        let mut heap: BinaryHeap<(i32, (i32, i32, i32, usize))> = BinaryHeap::new();
        let mut cleared: Vec<At> = Vec::new();
        while let Some((at, region)) = removal.pop() {
            for (face, step) in STEPS.iter().enumerate() {
                if region.faces[face] == 0 {
                    continue;
                }
                let next = at + *step;
                let Some(entry) = self.entries.get_mut(&next) else {
                    continue;
                };
                for (index, other) in entry.regions.iter_mut().enumerate() {
                    let aperture =
                        (region.faces[face] & other.faces[face ^ 1]).count_ones();
                    if aperture == 0 || other.level == 0 {
                        continue;
                    }
                    let through = costs.through(region.level, aperture, other.widest);
                    if other.level <= through {
                        removal.push((next, *other));
                        other.level = 0;
                        cleared.push((next, index));
                        self.touched.insert(next);
                    } else {
                        heap.push((other.level, key((next, index))));
                    }
                }
            }
        }

        // Floods again from the sky, into what is new and what was cleared,
        // and from the neighbours of what is new.
        if let Some(entry) = self.entries.get(&position) {
            for (index, region) in entry.regions.iter().enumerate() {
                if region.from_sky > 0 {
                    heap.push((region.from_sky, key((position, index))));
                }
                for (face, step) in STEPS.iter().enumerate() {
                    if region.faces[face] == 0 {
                        continue;
                    }
                    let Some(neighbour) = self.entries.get(&(position + *step)) else {
                        continue;
                    };
                    for other in &neighbour.regions {
                        let aperture =
                            (region.faces[face] & other.faces[face ^ 1]).count_ones();
                        if aperture == 0 || other.level == 0 {
                            continue;
                        }
                        let through = costs.through(other.level, aperture, region.widest);
                        if through > 0 {
                            heap.push((through, key((position, index))));
                        }
                    }
                }
            }
        }
        for (at, index) in cleared {
            let region = self.entries[&at].regions[index];
            if region.from_sky > 0 {
                heap.push((region.from_sky, key((at, index))));
            }
        }
        while let Some((level, at)) = heap.pop() {
            let (at, index) = unkey(at);
            let Some(region) = self
                .entries
                .get_mut(&at)
                .and_then(|entry| entry.regions.get_mut(index))
            else {
                continue;
            };
            if region.level > level {
                continue;
            }
            if region.level < level {
                region.level = level;
                self.touched.insert(at);
            }
            let faces = region.faces;
            for (face, step) in STEPS.iter().enumerate() {
                if faces[face] == 0 {
                    continue;
                }
                let next = at + *step;
                let Some(entry) = self.entries.get(&next) else {
                    continue;
                };
                for (other_index, other) in entry.regions.iter().enumerate() {
                    let aperture = (faces[face] & other.faces[face ^ 1]).count_ones();
                    if aperture == 0 {
                        continue;
                    }
                    let through = costs.through(level, aperture, other.widest);
                    if through > other.level {
                        heap.push((through, key((next, other_index))));
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::light::fill::{cell_bit, direct_cells};

    fn field() -> PathField {
        PathField::new(&PathFill::default())
    }

    fn air(direct: u64) -> PositionCells {
        PositionCells { open: !0, direct }
    }

    #[test]
    fn a_region_beside_one_whose_whole_top_sees_the_sky_holds_beside_open_sky() {
        let mut field = field();
        field.learn(IVec3::ZERO, air(direct_cells(!0, !0)));
        field.learn(IVec3::X, air(0));
        field.update(Duration::MAX);
        let beside = field.cells(IVec3::X).unwrap();
        assert!(beside.levels.iter().all(|level| *level == 12 * 16));
        // Through a second whole face, 2 levels less.
        field.learn(IVec3::new(2, 0, 0), air(0));
        field.update(Duration::MAX);
        let next = field.cells(IVec3::new(2, 0, 0)).unwrap();
        assert_eq!(next.levels[0], 10 * 16);
    }

    #[test]
    fn forgetting_a_position_takes_out_the_light_that_came_through_it() {
        let mut field = field();
        field.learn(IVec3::ZERO, air(direct_cells(!0, !0)));
        field.learn(IVec3::X, air(0));
        field.learn(IVec3::new(2, 0, 0), air(0));
        field.update(Duration::MAX);
        field.forget(IVec3::X);
        let changed = field.update(Duration::MAX);
        assert!(changed.contains(&IVec3::new(2, 0, 0)));
        assert_eq!(field.cells(IVec3::new(2, 0, 0)).unwrap().levels[0], 0);
        assert!(field.wanted_probes().any(|position| position == IVec3::X));
    }

    #[test]
    fn a_small_opening_into_a_big_region_loses_more_than_a_whole_face() {
        let mut field = field();
        field.learn(IVec3::ZERO, air(direct_cells(!0, !0)));
        // A wall at x = 0 of the next position with one open cell in it.
        let wall = (0..4)
            .flat_map(|z| (0..4).map(move |y| cell_bit(0, y, z)))
            .fold(0, |wall, bit| wall | bit);
        let door = cell_bit(0, 1, 1);
        field.learn(
            IVec3::X,
            PositionCells {
                open: !wall | door,
                direct: 0,
            },
        );
        field.update(Duration::MAX);
        let room = field.cells(IVec3::X).unwrap().levels
            [cell_bit(2, 1, 1).trailing_zeros() as usize];
        // 14 levels less 2 × 16², far past black.
        assert_eq!(room, 0);
    }
}
