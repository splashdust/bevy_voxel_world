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
//! level of each lit cell, rounded to a step of the world's precision, seeds
//! every voxel of the cell inside the kernel's box, so at 256 levels fill's
//! fractions reach the voxels it lights. A direct-sky cell isn't a seed, since its
//! voxels have direct sky already, but its level takes part in the blur.
//!
//! Seeds enter the kernel at the distance from a source at the top level that
//! holds their level. The seeds entering at each distance run 7 voxels
//! through open voxels with no loss, so a surface inside the next cell
//! doesn't lose a level per voxel. Seeds are grouped by that distance, not by
//! level, since levels the kernel joins at the same round light the same
//! voxels: at 256 levels this keeps the runs to at most one per round.
//!
//! A solve finds its seeds before anything else and keeps a hash of them, so
//! a relight started only because sky fill's cells changed can stop before
//! the run when the chunk's seeds are the ones its light was solved from: a
//! change to a cell's level often leaves every seed's round as it was.
//!
//! The run is most of seeding's cost, so it works on columns of grid cells
//! rather than voxels where it can: see [`FillSeeds::rows`].

use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::time::Duration;

use bevy::platform::time::Instant;

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

/// Which cells of a chunk's grid seed light at which distance.
#[derive(Hash)]
pub(crate) struct FillSeeds {
    /// By the distance from a source at the top level their light starts at,
    /// the grid cells, at `x + y × 12 + z × 144`, that seed it.
    starts: Vec<Vec<u16>>,
}

/// A solve's sky fill seeds, found before it solves.
pub(crate) struct Seeding {
    /// `None` if nothing in the box holds fill.
    pub seeds: Option<FillSeeds>,
    /// A hash of `seeds`, the same for two solves seeding the same voxels at
    /// the same levels.
    pub hash: u64,
    /// The time finding and hashing them took.
    pub took: Duration,
}

impl Seeding {
    /// The seeds of the chunk at the middle of `around`, for a kernel box
    /// whose levels `scale` gives.
    pub fn of(around: &CellsAround, scale: &LevelScale) -> Self {
        let started = Instant::now();
        let seeds = FillSeeds::of(around, scale);
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        seeds.hash(&mut hasher);
        Self {
            seeds,
            hash: hasher.finish(),
            took: started.elapsed(),
        }
    }
}

impl FillSeeds {
    /// The seeds of the chunk at the middle of `around`, for a kernel box
    /// whose levels `scale` gives. `None` if nothing in the box holds fill.
    pub fn of(around: &CellsAround, scale: &LevelScale) -> Option<Self> {
        let reach = scale.reach() as usize;
        let top = scale.precision().top() as i32;
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
            // Each axis's stride, and the strides of the other two, which
            // pick a line of cells along it.
            for (stride, across) in [
                (1, [GRID, GRID * GRID]),
                (GRID, [1, GRID * GRID]),
                (GRID * GRID, [1, GRID]),
            ] {
                before.copy_from_slice(&levels);
                for a in 0..GRID {
                    for b in 0..GRID {
                        let line = a * across[0] + b * across[1];
                        for along in 0..GRID {
                            let at = line + along * stride;
                            if !open[at] {
                                continue;
                            }
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
            }
        }
        let mut seeds = Self {
            starts: vec![Vec::new(); reach + 1],
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
            // Sixteenths of a whole level to steps, rounded to the nearest.
            let level = ((levels[at] * top / 15 + 8) / 16).clamp(0, top - 1) as u8;
            if let Some(start) = scale.start(level) {
                seeds.starts[start as usize].push(at as u16);
                any = true;
            }
        }
        any.then_some(seeds)
    }

    /// The seeds joining at each round as rows of a kernel box of `reach`,
    /// run through the box's `open` rows, with that round, sorted by round.
    /// A round stands for the levels joining at it below, the brightest
    /// first.
    ///
    /// A level's rows don't hold every voxel its run reaches, only enough
    /// that the kernel lights each voxel from the brightest level reaching
    /// it, as it would with every voxel:
    ///
    /// - The kernel reads a level's rows only where its round still works:
    ///   from `round` to `side - round` along each axis of a box `side`
    ///   voxels wide. Rows past that may differ.
    /// - A voxel some brighter level's run reaches has joined at an earlier
    ///   round, so joining it again changes nothing. A level's run doesn't
    ///   pass through a brighter level's seeds: past one, everything is
    ///   within that brighter level's run already.
    pub fn rows<R: Row>(&self, open: &[R], reach: usize) -> Vec<(usize, Vec<R>)> {
        let side = crate::chunk::CHUNK_SIZE_U as usize + 2 * reach;
        let mut tiles = Tiles::new(open, reach, side);
        let mut out = Vec::new();
        let mut spare = vec![R::ZERO; side * side];
        let mut kept = vec![R::ZERO; side * side];
        for (start, cells) in self.starts.iter().enumerate() {
            if cells.is_empty() {
                continue;
            }
            // A run moves a seed at most RUN rows, so seeds farther than that
            // outside the rows the kernel reads can't reach them.
            let near = Span(start.saturating_sub(RUN), (side + RUN - start).min(side));
            if let Some(rows) = tiles.run(cells, open, near, &mut spare, &mut kept) {
                out.push((start, rows));
            }
            tiles.take(cells);
        }
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

/// Rows from the first index up to the second along an axis of a box.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Span(usize, usize);

impl Span {
    fn is_empty(self) -> bool {
        self.0 >= self.1
    }

    fn within(self, other: Span) -> Span {
        Span(self.0.max(other.0), self.1.min(other.1))
    }

    fn contains(self, other: Span) -> bool {
        other.0 >= self.0 && other.1 <= self.1
    }
}

/// How a run treats a tile at one level.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    /// No seed in it or the tiles around it, so its rows stay clear. So do
    /// the rows of a tile outside the box, which has none.
    Quiet,
    /// Every voxel of it closed, or taken by a brighter level's seeds, so its
    /// rows stay clear.
    Closed,
    /// Its rows all hold the same seeds and open voxels, and the tiles
    /// around it hold no other seeds or open voxels, so nothing reaches it
    /// that its own rows don't reach along z, and one row stands for all.
    Same,
    /// Stepped row by row.
    Rows,
}

/// The tiles of a kernel box: the 8×8 rows along x and y of each column of
/// grid cells along z. A face step reaches only the tiles beside a tile's
/// faces, and a run of [`RUN`] steps stays within the tiles around it.
///
/// Seeds are whole cells, so a tile's rows hold the same seeds unless the
/// tile is cut by the rows a level reaches. Where they also have the same
/// open voxels, as in open air, a run steps one row for the tile instead of
/// 64.
struct Tiles<R> {
    side: usize,
    reach: usize,
    /// By grid cell, the cell's rows along an axis: empty for cells outside
    /// the box.
    spans: [Span; GRID],
    /// By tile at `x + y × 12`, the open voxels all its rows share, if they
    /// do.
    open: [Option<R>; GRID * GRID],
    /// By tile, the voxels open in any of its rows.
    any_open: [R; GRID * GRID],
    /// By tile, the voxels brighter levels' seeds take.
    taken: [R; GRID * GRID],
}

impl<R: Row> Tiles<R> {
    fn new(open: &[R], reach: usize, side: usize) -> Self {
        let spans: [Span; GRID] = std::array::from_fn(|cell| span(cell, reach, side));
        let any_open = std::array::from_fn(|tile| {
            let (x, y): (Span, Span) = (spans[tile % GRID], spans[tile / GRID]);
            let mut any = R::ZERO;
            for by in y.0..y.1 {
                for row in &open[x.0 + by * side..x.1.max(x.0) + by * side] {
                    any |= *row;
                }
            }
            any
        });
        let open = std::array::from_fn(|tile| {
            let (x, y): (Span, Span) = (spans[tile % GRID], spans[tile / GRID]);
            if x.is_empty() || y.is_empty() {
                return None;
            }
            let first = open[x.0 + y.0 * side];
            (y.0..y.1)
                .all(|by| {
                    open[x.0 + by * side..x.1 + by * side]
                        .iter()
                        .all(|row| *row == first)
                })
                .then_some(first)
        });
        Self {
            side,
            reach,
            spans,
            open,
            any_open,
            taken: [R::ZERO; GRID * GRID],
        }
    }

    /// Takes the voxels of `cells` from the levels run after.
    fn take(&mut self, cells: &[u16]) {
        for &at in cells {
            let (tile, bits) = self.cell(at);
            self.taken[tile] |= bits;
        }
    }

    /// The tile of the grid cell at `at`, and the bits of its voxels along z
    /// within the box.
    fn cell(&self, at: u16) -> (usize, R) {
        let at = at as usize;
        let (x, y, z) = (at % GRID, at / GRID % GRID, at / (GRID * GRID));
        let z = span(z, self.reach, self.side);
        (x + y * GRID, ones::<R>(z.1 - z.0) << z.0)
    }

    /// The rows of `tile` along x and y.
    fn rows_of(&self, tile: usize) -> (Span, Span) {
        (self.spans[tile % GRID], self.spans[tile / GRID])
    }

    /// The open voxels all rows of `tile` share, less those brighter levels
    /// take, if its rows share them.
    fn open_of(&self, tile: usize) -> Option<R> {
        self.open[tile].map(|open| open & !self.taken[tile])
    }

    /// Seeds `cells` within `near` along x and y and runs them [`RUN`] face
    /// steps through the box's `open` rows, keeping only open voxels not
    /// taken by brighter levels. Returns the rows, or `None` if no seed lies
    /// within `near`. `spare` is a box of clear rows the run works in, left
    /// clear for the next, and `kept` a box it keeps the voxels a step may
    /// reach in.
    ///
    /// Rows past `near` and the box's outer rows keep their seeds, as the
    /// box's outer rows do when every row is stepped. Past `near` that
    /// changes nothing the kernel reads: rows there hold no seed, and what
    /// differs moves one row a step, never reaching back inside `near`.
    fn run(
        &self,
        cells: &[u16],
        open: &[R],
        near: Span,
        spare: &mut Vec<R>,
        kept: &mut [R],
    ) -> Option<Vec<R>> {
        let side = self.side;
        let stepped = Span(near.0.max(1), near.1.min(side - 1));
        let mut seeds = [R::ZERO; GRID * GRID];
        for &at in cells {
            let (tile, bits) = self.cell(at);
            seeds[tile] |= bits;
        }
        let mut rows = vec![R::ZERO; side * side];
        let mut any = false;
        // Whether every row of a tile holds its seeds, none cut by `near`.
        let mut whole = [true; GRID * GRID];
        for tile in 0..GRID * GRID {
            if seeds[tile] == R::ZERO {
                continue;
            }
            let (x, y) = self.rows_of(tile);
            let (cut_x, cut_y) = (x.within(near), y.within(near));
            whole[tile] = cut_x == x && cut_y == y;
            if cut_x.is_empty() || cut_y.is_empty() {
                continue;
            }
            any = true;
            for by in cut_y.0..cut_y.1 {
                let at = cut_x.0 + by * side..cut_x.1 + by * side;
                for (row, open) in rows[at.clone()].iter_mut().zip(&open[at]) {
                    *row = seeds[tile] & *open;
                }
            }
        }
        if !any {
            return None;
        }
        let kinds: [Kind; GRID * GRID] =
            std::array::from_fn(|tile| self.kind(tile, &seeds, &whole, stepped));
        // The row standing for each tile of kind `Same`.
        let mut same: [R; GRID * GRID] = std::array::from_fn(|tile| match kinds[tile] {
            Kind::Same => seeds[tile] & self.open_of(tile).unwrap_or(R::ZERO),
            _ => R::ZERO,
        });
        for (tile, kind) in kinds.iter().enumerate() {
            if *kind != Kind::Quiet {
                let (x, y) = self.rows_of(tile);
                for by in y.0..y.1 {
                    let at = x.0 + by * side..x.1 + by * side;
                    spare[at.clone()].copy_from_slice(&rows[at]);
                }
            }
        }
        for (tile, kind) in kinds.iter().enumerate() {
            if *kind == Kind::Rows {
                let (x, y) = self.rows_of(tile);
                for by in y.0..y.1 {
                    let at = x.0 + by * side..x.1 + by * side;
                    for (kept, open) in kept[at.clone()].iter_mut().zip(&open[at]) {
                        *kept = *open & !self.taken[tile];
                    }
                }
            }
        }
        // Tiles that changed in the last step. A tile steps only if it or a
        // tile beside it changed: otherwise it can't change, and both boxes
        // already hold its rows.
        let mut changed = [true; GRID * GRID];
        for _ in 0..RUN {
            let mut now = [false; GRID * GRID];
            for tile in 0..GRID * GRID {
                if kinds[tile] == Kind::Same {
                    let c = same[tile];
                    let row =
                        (c | c << 1 | c >> 1) & self.open_of(tile).unwrap_or(R::ZERO);
                    now[tile] = row != c;
                    same[tile] = row;
                    let (x, y) = self.rows_of(tile);
                    for by in y.0..y.1 {
                        spare[x.0 + by * side..x.1 + by * side].fill(row);
                    }
                }
            }
            let stepping = |tile: usize| {
                let (tx, ty) = (tile % GRID, tile / GRID);
                kinds[tile] == Kind::Rows
                    && (changed[tile]
                        || (tx > 0 && changed[tile - 1])
                        || (tx + 1 < GRID && changed[tile + 1])
                        || (ty > 0 && changed[tile - GRID])
                        || (ty + 1 < GRID && changed[tile + GRID]))
            };
            // Tiles side by side along x step as one span.
            for ty in 0..GRID {
                let mut tx = 0;
                while tx < GRID {
                    if !stepping(tx + ty * GRID) {
                        tx += 1;
                        continue;
                    }
                    let first = tx;
                    while tx < GRID && stepping(tx + ty * GRID) {
                        tx += 1;
                    }
                    let x =
                        Span(self.spans[first].0, self.spans[tx - 1].1).within(stepped);
                    let y = self.spans[ty].within(stepped);
                    if !x.is_empty()
                        && !y.is_empty()
                        && step(&rows, spare, kept, side, x, y)
                    {
                        now[first + ty * GRID..tx + ty * GRID].fill(true);
                    }
                }
            }
            std::mem::swap(&mut rows, spare);
            changed = now;
            if !changed.contains(&true) {
                break;
            }
        }
        for (tile, kind) in kinds.iter().enumerate() {
            if *kind != Kind::Quiet {
                let (x, y) = self.rows_of(tile);
                for by in y.0..y.1 {
                    spare[x.0 + by * side..x.1 + by * side].fill(R::ZERO);
                }
            }
        }
        Some(rows)
    }

    /// How a run with `seeds` treats `tile`. `whole` says whether every row
    /// of a tile holds its seeds, and `stepped` which rows take steps.
    fn kind(
        &self,
        tile: usize,
        seeds: &[R; GRID * GRID],
        whole: &[bool; GRID * GRID],
        stepped: Span,
    ) -> Kind {
        let (x, y) = self.rows_of(tile);
        let (tx, ty) = ((tile % GRID) as i32, (tile / GRID) as i32);
        let around = (-1..=1)
            .flat_map(move |dy| (-1..=1).map(move |dx| (tx + dx, ty + dy)))
            .filter(|(x, y)| (0..GRID as i32).contains(x) && (0..GRID as i32).contains(y))
            .map(|(x, y)| x as usize + y as usize * GRID);
        if x.is_empty()
            || y.is_empty()
            || around.clone().all(|tile| seeds[tile] == R::ZERO)
        {
            return Kind::Quiet;
        }
        let Some(own) = self.open_of(tile) else {
            return Kind::Rows;
        };
        if own == R::ZERO {
            return Kind::Closed;
        }
        let same = whole[tile]
            && stepped.contains(x)
            && stepped.contains(y)
            && around.into_iter().all(|other| {
                let open = self.any_open[other] & !self.taken[other];
                open & !own == R::ZERO && seeds[other] & !seeds[tile] == R::ZERO
            });
        if same {
            Kind::Same
        } else {
            Kind::Rows
        }
    }
}

/// One face step from `rows` into `next`, over the rows `x` and `y` of a box
/// `side` rows wide, keeping only the voxels of `open`. Returns whether it
/// reached a voxel `rows` didn't hold.
fn step<R: Row>(
    rows: &[R],
    next: &mut [R],
    open: &[R],
    side: usize,
    x: Span,
    y: Span,
) -> bool {
    let width = x.1 - x.0;
    let mut changed = R::ZERO;
    for by in y.0..y.1 {
        let i = x.0 + by * side;
        // Zipped, since indexing rows of a run-time width doesn't vectorise.
        let before = rows[i..i + width]
            .iter()
            .zip(&rows[i - 1..i - 1 + width])
            .zip(&rows[i + 1..i + 1 + width])
            .zip(&rows[i - side..i - side + width])
            .zip(&rows[i + side..i + side + width])
            .zip(&open[i..i + width]);
        for (out, (((((&c, &xm), &xp), &ym), &yp), &open)) in
            next[i..i + width].iter_mut().zip(before)
        {
            let row = (c | c << 1 | c >> 1 | xm | xp | ym | yp) & open;
            *out = row;
            changed |= row & !c;
        }
    }
    changed != R::ZERO
}

/// The voxels of the grid cell `cell` along an axis of a box of `reach`,
/// `side` voxels wide, cut to the box.
fn span(cell: usize, reach: usize, side: usize) -> Span {
    let low = (cell * CELL + reach) as i32 - 32;
    let high = low + CELL as i32;
    Span(
        low.clamp(0, side as i32) as usize,
        high.clamp(0, side as i32) as usize,
    )
}

/// `count` set bits from bit 0 up.
fn ones<R: Row>(count: usize) -> R {
    if count >= R::BITS {
        !R::ZERO
    } else {
        !(!R::ZERO << count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::light::config::LightPrecision;

    /// A small xorshift generator, so the test needs no seedable crate.
    struct Random(u64);

    impl Random {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
    }

    /// Each round's seeds run through every row of the box, the plain way.
    fn plain_rows<R: Row>(
        seeds: &FillSeeds,
        open: &[R],
        reach: usize,
    ) -> Vec<(usize, Vec<R>)> {
        let side = crate::chunk::CHUNK_SIZE_U as usize + 2 * reach;
        let mut out = Vec::new();
        for (start, cells) in seeds.starts.iter().enumerate() {
            if cells.is_empty() {
                continue;
            }
            let mut rows = vec![R::ZERO; side * side];
            for &at in cells {
                let at = at as usize;
                let (x, y, z) = (at % GRID, at / GRID % GRID, at / (GRID * GRID));
                let (x, y, z) = (
                    span(x, reach, side),
                    span(y, reach, side),
                    span(z, reach, side),
                );
                let (z0, z1) = (z.0, z.1);
                for by in y.0..y.1 {
                    for bx in x.0..x.1 {
                        rows[bx + by * side] |= ones::<R>(z1 - z0) << z0;
                    }
                }
            }
            for (row, open) in rows.iter_mut().zip(open) {
                *row &= *open;
            }
            for _ in 0..RUN {
                let before = rows.clone();
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
            out.push((start, rows));
        }
        out
    }

    /// Random cells around a chunk, mostly open, with levels from 0 to 15,
    /// and random open voxels in a box of `reach`.
    fn random_case<R: Row>(random: &mut Random, reach: usize) -> (CellsAround, Vec<R>) {
        let around = std::array::from_fn(|_| {
            if random.next().is_multiple_of(8) {
                return None;
            }
            let open = random.next() | random.next();
            let direct = random.next() & random.next() & random.next() & open;
            let levels = std::array::from_fn(|_| (random.next() % 241) as u8);
            Some(Arc::new(CellFill {
                open,
                direct,
                levels,
            }))
        });
        let side = crate::chunk::CHUNK_SIZE_U as usize + 2 * reach;
        let open = (0..side * side)
            .map(|_| {
                let solid = random.next() & random.next() & random.next();
                let row = R::from_u64(!solid)
                    | R::from_u64(!(solid >> 7)) << 64.min(R::BITS - 1);
                row & ones::<R>(side)
            })
            .collect();
        (around, open)
    }

    /// Open air over a floor, with a few solid blocks in it and a level per
    /// layer of cells that falls with height, so whole tiles share their
    /// seeds and open voxels.
    fn layered_case<R: Row>(random: &mut Random, reach: usize) -> (CellsAround, Vec<R>) {
        let fall = 1 + random.next() % 3;
        let around = std::array::from_fn(|index| {
            let py = (index / 3 % 3) as u64;
            let levels = std::array::from_fn(|cell| {
                let y = py * 4 + (cell / 4 % 4) as u64;
                let wobble = random.next() % 4 * (random.next() % 3 / 2);
                (240u64.saturating_sub(y * fall * 16) + wobble * 16).min(240) as u8
            });
            Some(Arc::new(CellFill {
                open: if py == 0 { !0xffff } else { !0 },
                direct: if py == 2 { 0xf000_0000_0000_0000 } else { 0 },
                levels,
            }))
        });
        let side = crate::chunk::CHUNK_SIZE_U as usize + 2 * reach;
        let blocks: Vec<[usize; 6]> = (0..1 + random.next() % 4)
            .map(|_| {
                let mut corner = || (random.next() % side as u64) as usize;
                let (x, y, z) = (corner(), corner(), corner());
                let mut size = || 1 + (random.next() % 12) as usize;
                [x, y, z, x + size(), y + size(), z + size()]
            })
            .collect();
        let floor = reach.saturating_sub(32) + 8;
        let open = (0..side * side)
            .map(|i| {
                let (x, y) = (i % side, i / side);
                if y < floor {
                    return R::ZERO;
                }
                let mut row = ones::<R>(side);
                for b in &blocks {
                    if (b[0]..b[3]).contains(&x) && (b[1]..b[4]).contains(&y) {
                        let z1 = b[5].min(side);
                        row &= !(ones::<R>(z1 - b[2].min(z1)) << b[2].min(z1));
                    }
                }
                row
            })
            .collect();
        (around, open)
    }

    /// By voxel of a box `side` wide, the first round at which some entry of
    /// `rows` joins it, where the kernel reads that entry.
    fn first_rounds<R: Row>(rows: &[(usize, Vec<R>)], side: usize) -> Vec<usize> {
        let mut first = vec![usize::MAX; side * side * side];
        for (start, rows) in rows {
            for y in *start..side - start {
                for x in *start..side - start {
                    let row = rows[x + y * side];
                    for z in *start..side - start {
                        if row >> z & R::ONE == R::ONE {
                            let at = &mut first[x + y * side + z * side * side];
                            *at = (*at).min(*start);
                        }
                    }
                }
            }
        }
        first
    }

    fn check<R: Row>(reach: usize, cases: usize, precision: LightPrecision) {
        let side = crate::chunk::CHUNK_SIZE_U as usize + 2 * reach;
        let scale = LevelScale::new(reach as u32, precision);
        let mut random = Random(0x2545_f491_4f6c_dd1d ^ reach as u64);
        for case in 0..cases {
            let (around, open) = if case % 2 == 0 {
                random_case::<R>(&mut random, reach)
            } else {
                layered_case::<R>(&mut random, reach)
            };
            let Some(seeds) = FillSeeds::of(&around, &scale) else {
                continue;
            };
            let fast = seeds.rows(&open, reach);
            assert!(fast.windows(2).all(|pair| pair[0].0 <= pair[1].0));
            let plain = plain_rows(&seeds, &open, reach);
            assert!(
                first_rounds(&fast, side) == first_rounds(&plain, side),
                "reach {reach}, case {case}"
            );
        }
    }

    #[test]
    fn rows_join_each_voxel_when_running_every_row_would() {
        for precision in [LightPrecision::Steps16, LightPrecision::Steps256] {
            check::<u64>(5, 40, precision);
            check::<u64>(14, 40, precision);
            check::<u64>(16, 40, precision);
            check::<crate::light::kernel::TwoWords>(17, 40, precision);
            check::<crate::light::kernel::TwoWords>(30, 40, precision);
            check::<crate::light::kernel::TwoWords>(32, 40, precision);
        }
    }
}
