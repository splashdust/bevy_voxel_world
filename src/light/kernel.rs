//! The kernel: spreads one channel of light from its sources through the open
//! voxels of a box around a chunk, by bit dilation.
//!
//! The box is `32 + 2 × reach` voxels on a side, centred on the chunk, so it
//! holds every voxel light can come from. Each row of the box runs along z and
//! is one `u64`, bit `z` for the voxel at `z`, which fits a box up to reach 16.
//! A round of spreading ORs each reached row with its neighbours, shifted
//! along z and taken from the rows beside it along x and y, and keeps only open
//! voxels.
//!
//! Round `d` only matters within `reach - d` voxels of the chunk, since light
//! has `reach - d` rounds left to get there. A round moves light at most one
//! voxel along each axis, so each round works on a box one voxel smaller on
//! every side than the last.
//!
//! The kernel knows nothing about which channel it spreads. More channels, such
//! as coloured voxel light, are more runs of it.

use super::config::LightShape;
use super::levels::LevelScale;
use crate::chunk::CHUNK_SIZE_U;

const CHUNK: usize = CHUNK_SIZE_U as usize;

/// How many bits a stored light level has.
pub(crate) const LEVEL_BITS: usize = 4;

/// One bit of a light level for each voxel of a chunk: a `u32` column along z
/// at `x + y × 32`.
pub(crate) type Plane = [u32; CHUNK * CHUNK];

/// The highest reach a box of one-`u64` rows holds.
pub(crate) const MAX_ONE_WORD_REACH: u32 = 16;

/// A source of light in a box: the voxel at `x`, `y` and `z` of the box, and
/// how many steps from a level-15 source its light starts at.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Seed {
    pub x: u8,
    pub y: u8,
    pub z: u8,
    pub start: u8,
}

/// The rows of a box, and the scratch rows the kernel works in. It is kept
/// between runs so a run allocates nothing.
pub(crate) struct LightBox {
    reach: usize,
    side: usize,
    /// Which voxels light passes through.
    open: Vec<u64>,
    reached: Vec<u64>,
    next: Vec<u64>,
    pass: Vec<u64>,
}

impl LightBox {
    /// A box for light of `reach`, at most [`MAX_ONE_WORD_REACH`], with every
    /// voxel closed.
    pub fn new(reach: u32) -> Self {
        debug_assert!((1..=MAX_ONE_WORD_REACH).contains(&reach));
        let reach = reach as usize;
        let side = CHUNK + 2 * reach;
        Self {
            reach,
            side,
            open: vec![0; side * side],
            reached: vec![0; side * side],
            next: vec![0; side * side],
            pass: vec![0; side * side],
        }
    }

    /// How many voxels the box spans on a side.
    pub fn side(&self) -> usize {
        self.side
    }

    pub fn reach(&self) -> usize {
        self.reach
    }

    /// The rows of open voxels, at `x + y × side`, to fill before a run. Bits at
    /// or above `side` must be clear.
    pub fn open_rows_mut(&mut self) -> &mut [u64] {
        &mut self.open
    }

    /// Spreads light from `seeds` through the box's open voxels in `shape`, and
    /// writes the level each of the chunk's voxels reaches into `planes`, one
    /// plane per bit of the level. Voxels light doesn't reach, and closed ones,
    /// keep level 0.
    ///
    /// `seeds` must be sorted by their start. A seed's own voxel is never
    /// written: it holds the level of its source, not of light reaching it.
    pub fn spread(
        &mut self,
        seeds: &[Seed],
        shape: LightShape,
        scale: &LevelScale,
        planes: &mut [Plane; LEVEL_BITS],
    ) {
        debug_assert!(seeds.windows(2).all(|pair| pair[0].start <= pair[1].start));
        debug_assert_eq!(scale.reach() as usize, self.reach);
        for plane in planes.iter_mut() {
            plane.fill(0);
        }
        let (reach, side) = (self.reach, self.side);
        self.reached.fill(0);
        let mut next_seed = 0;

        for round in 1..=reach {
            // Seeds starting at the step before this round join the light, and
            // light leaves through their own voxels for this round only.
            let first_seed = next_seed;
            while next_seed < seeds.len() && seeds[next_seed].start as usize + 1 == round
            {
                let seed = seeds[next_seed];
                let row = seed.x as usize + seed.y as usize * side;
                self.reached[row] |= 1 << seed.z;
                self.open[row] |= 1 << seed.z;
                next_seed += 1;
            }
            let round_seeds = &seeds[first_seed..next_seed];

            let low = round;
            let high = side - round;
            let z_mask = ones(low, high - low);
            let changed = if shape == LightShape::Round && round % 2 == 0 {
                self.three_passes(low, high, z_mask)
            } else {
                self.face_step(low, high, z_mask)
            };

            for seed in round_seeds {
                let row = seed.x as usize + seed.y as usize * side;
                self.open[row] &= !(1 << seed.z);
            }

            let level = scale.level(round as u32);
            for y in 0..CHUNK {
                let row = reach + (y + reach) * side;
                let new = &self.next[row..row + CHUNK];
                let old = &self.reached[row..row + CHUNK];
                for (bit, plane) in planes.iter_mut().enumerate() {
                    if level >> bit & 1 == 0 {
                        continue;
                    }
                    let columns = &mut plane[y * CHUNK..(y + 1) * CHUNK];
                    for ((column, new), old) in columns.iter_mut().zip(new).zip(old) {
                        *column |= ((new & !old) >> reach) as u32;
                    }
                }
            }
            std::mem::swap(&mut self.reached, &mut self.next);
            if changed == 0 && next_seed == seeds.len() {
                break;
            }
        }
    }

    /// One face step from every reached voxel, over rows `low..high` along x
    /// and y. Returns the bits newly reached.
    fn face_step(&mut self, low: usize, high: usize, z_mask: u64) -> u64 {
        let side = self.side;
        let width = high - low;
        let mut changed = 0;
        for y in low..high {
            let i = low + y * side;
            // Zipped, since indexing rows of a run-time width doesn't vectorise.
            let rows = self.reached[i..i + width]
                .iter()
                .zip(&self.reached[i - 1..i - 1 + width])
                .zip(&self.reached[i + 1..i + 1 + width])
                .zip(&self.reached[i - side..i - side + width])
                .zip(&self.reached[i + side..i + side + width])
                .zip(&self.open[i..i + width]);
            for (out, (((((c, xm), xp), ym), yp), open)) in
                self.next[i..i + width].iter_mut().zip(rows)
            {
                let row = (c | c << 1 | c >> 1 | xm | xp | ym | yp) & open & z_mask;
                *out = row;
                changed |= row & !c;
            }
        }
        changed
    }

    /// A step along x, then along y, then along z, each from everything the
    /// step before reached, over rows `low..high` along x and y. Returns the
    /// bits newly reached.
    fn three_passes(&mut self, low: usize, high: usize, z_mask: u64) -> u64 {
        let side = self.side;
        let width = high - low;
        // Along x into `pass`, on one more row each side along y, which the
        // step along y reads.
        for y in low - 1..high + 1 {
            let i = low + y * side;
            let rows = self.reached[i..i + width]
                .iter()
                .zip(&self.reached[i - 1..i - 1 + width])
                .zip(&self.reached[i + 1..i + 1 + width])
                .zip(&self.open[i..i + width]);
            for (out, (((c, xm), xp), open)) in
                self.pass[i..i + width].iter_mut().zip(rows)
            {
                *out = (c | xm | xp) & open;
            }
        }
        // Along y into `next`.
        for y in low..high {
            let i = low + y * side;
            let rows = self.pass[i..i + width]
                .iter()
                .zip(&self.pass[i - side..i - side + width])
                .zip(&self.pass[i + side..i + side + width])
                .zip(&self.open[i..i + width]);
            for (out, (((c, ym), yp), open)) in
                self.next[i..i + width].iter_mut().zip(rows)
            {
                *out = (c | ym | yp) & open;
            }
        }
        // Along z, in place.
        let mut changed = 0;
        for y in low..high {
            let i = low + y * side;
            for (out, (open, old)) in self.next[i..i + width].iter_mut().zip(
                self.open[i..i + width]
                    .iter()
                    .zip(&self.reached[i..i + width]),
            ) {
                let c = *out;
                let row = (c | c << 1 | c >> 1) & open & z_mask;
                *out = row;
                changed |= row & !old;
            }
        }
        changed
    }
}

/// `count` set bits from bit `low` up.
fn ones(low: usize, count: usize) -> u64 {
    let bits = if count >= 64 { !0 } else { (1u64 << count) - 1 };
    bits << low
}
