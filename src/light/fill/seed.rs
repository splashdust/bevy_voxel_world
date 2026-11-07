//! Seeding a chunk's solve with sky fill, the part every model of sky fill
//! shares.
//!
//! A solve reads the cells of the 27 positions around its chunk onto a grid
//! of 12×12×12 cells. A blur over open cells softens the steps between
//! positions: twice along each axis in turn, each cell becomes a quarter of
//! each neighbour along that axis and half itself, a closed neighbour or one
//! past the grid counting as the cell itself. The cells of a position the
//! field doesn't know count as open and unlit, so knowing less can only
//! darken what the blur gives, never brighten it. The blurred
//! level of each lit cell, rounded to a whole level, seeds every voxel of the
//! cell inside the kernel's box. A direct-sky cell isn't a seed, since its
//! voxels have direct sky already, but its level takes part in the blur.
//!
//! Each level's seeds then run 7 voxels through open voxels with no loss, so
//! a surface inside the next cell doesn't lose a level per voxel, and enter
//! the kernel at the distance from a level-15 source that holds their level.

use std::sync::Arc;

use super::{CellFill, CELL};
use crate::light::kernel::Row;
use crate::light::levels::LevelScale;

/// The cells of the grid along each axis: 4 per position, 3 positions.
const GRID: usize = 12;

/// How many voxels each seed runs through open voxels with no loss.
const RUN: usize = 7;

/// How many times the blur passes along each axis.
const BLUR: usize = 2;

/// The cells of the 27 positions around a chunk, at the index
/// [`neighbour_index`](crate::light::solver::neighbour_index) gives.
pub(crate) type CellsAround = [Option<Arc<CellFill>>; 27];

/// Which cells of a chunk's grid seed which level.
pub(crate) struct FillSeeds {
    /// By level from 1 to 14, the grid cells, at `x + y × 12 + z × 144`,
    /// that seed it.
    levels: [Vec<u16>; 15],
}

impl FillSeeds {
    /// The seeds of the chunk at the middle of `around`, for a kernel box
    /// reaching `reach` voxels past each of its faces. `None` if nothing in
    /// the box holds fill.
    pub fn of(around: &CellsAround, reach: usize) -> Option<Self> {
        let lit_anywhere = around.iter().flatten().any(|cells| {
            cells
                .levels
                .iter()
                .enumerate()
                .any(|(cell, level)| *level > 0 && cells.direct >> cell & 1 == 0)
        });
        if !lit_anywhere {
            return None;
        }
        let mut levels = [0i32; GRID * GRID * GRID];
        let mut open = [false; GRID * GRID * GRID];
        let mut lit = [false; GRID * GRID * GRID];
        for (index, cells) in around.iter().enumerate() {
            let (px, py, pz) = (index % 3, index / 3 % 3, index / 9);
            let Some(cells) = cells else {
                for z in pz * 4..pz * 4 + 4 {
                    for y in py * 4..py * 4 + 4 {
                        for x in px * 4..px * 4 + 4 {
                            open[x + y * GRID + z * GRID * GRID] = true;
                        }
                    }
                }
                continue;
            };
            let mut left = cells.open;
            while left != 0 {
                let bit = left.trailing_zeros() as usize;
                left &= left - 1;
                let (cx, cy, cz) = (bit & 3, bit >> 2 & 3, bit >> 4);
                let at =
                    (px * 4 + cx) + (py * 4 + cy) * GRID + (pz * 4 + cz) * GRID * GRID;
                open[at] = true;
                levels[at] = cells.levels[bit] as i32;
                lit[at] = cells.levels[bit] > 0 && cells.direct >> bit & 1 == 0;
            }
        }
        let mut before = levels;
        for _ in 0..BLUR {
            for stride in [1, GRID, GRID * GRID] {
                before.copy_from_slice(&levels);
                for at in 0..levels.len() {
                    if !open[at] {
                        continue;
                    }
                    let along = at / stride % GRID;
                    let here = before[at];
                    let low = if along > 0 && open[at - stride] {
                        before[at - stride]
                    } else {
                        here
                    };
                    let high = if along + 1 < GRID && open[at + stride] {
                        before[at + stride]
                    } else {
                        here
                    };
                    levels[at] = (2 * here + low + high) / 4;
                }
            }
        }
        let mut seeds = Self {
            levels: Default::default(),
        };
        let mut any = false;
        for at in 0..levels.len() {
            if !lit[at] {
                continue;
            }
            let (x, y, z) = (at % GRID, at / GRID % GRID, at / (GRID * GRID));
            if [x, y, z].iter().any(|axis| !overlaps(*axis, reach)) {
                continue;
            }
            let level = ((levels[at] + 8) / 16).clamp(0, 14) as usize;
            if level > 0 {
                seeds.levels[level].push(at as u16);
                any = true;
            }
        }
        any.then_some(seeds)
    }

    /// Each level's seeds as rows of a kernel box of `reach`, run through the
    /// box's `open` rows, with the round of the kernel each joins at, which
    /// `scale` gives. Levels that light nothing within reach are left out.
    pub fn rows<R: Row>(
        &self,
        open: &[R],
        reach: usize,
        scale: &LevelScale,
    ) -> Vec<(usize, Vec<R>)> {
        let side = crate::chunk::CHUNK_SIZE_U as usize + 2 * reach;
        let mut out = Vec::new();
        for (level, cells) in self.levels.iter().enumerate() {
            if cells.is_empty() {
                continue;
            }
            let Some(start) = scale.start(level as u8) else {
                continue;
            };
            let mut rows = vec![R::ZERO; side * side];
            for &at in cells {
                let at = at as usize;
                let (x, y, z) = (at % GRID, at / GRID % GRID, at / (GRID * GRID));
                let span = |cell: usize| -> (usize, usize) {
                    let low = (cell * CELL + reach) as i32 - 32;
                    let high = low + CELL as i32;
                    (low.max(0) as usize, (high.min(side as i32)) as usize)
                };
                let ((x0, x1), (y0, y1), (z0, z1)) = (span(x), span(y), span(z));
                let bits = ones::<R>(z1 - z0) << z0;
                for by in y0..y1 {
                    for bx in x0..x1 {
                        rows[bx + by * side] |= bits;
                    }
                }
            }
            run(&mut rows, open, side);
            out.push((start as usize, rows));
        }
        out.sort_by_key(|(start, _)| *start);
        out
    }
}

/// Whether the grid cell at `cell` along an axis overlaps a kernel box
/// reaching `reach` voxels past the chunk.
fn overlaps(cell: usize, reach: usize) -> bool {
    let low = (cell * CELL) as i32 - 32;
    let high = low + CELL as i32;
    high > -(reach as i32) && low < 32 + reach as i32
}

/// Runs `rows` [`RUN`] face steps through the `open` rows of a box with
/// `side` rows along x and y, keeping only open voxels.
fn run<R: Row>(rows: &mut [R], open: &[R], side: usize) {
    for (row, open) in rows.iter_mut().zip(open) {
        *row &= *open;
    }
    let mut before = rows.to_vec();
    for _ in 0..RUN {
        before.copy_from_slice(rows);
        for y in 1..side - 1 {
            for x in 1..side - 1 {
                let i = x + y * side;
                let c = before[i];
                rows[i] = (c
                    | c << 1
                    | c >> 1
                    | before[i - 1]
                    | before[i + 1]
                    | before[i - side]
                    | before[i + side])
                    & open[i];
            }
        }
    }
}

/// `count` set bits from bit 0 up.
fn ones<R: Row>(count: usize) -> R {
    if count >= R::BITS {
        !R::ZERO
    } else {
        !(!R::ZERO << count)
    }
}
