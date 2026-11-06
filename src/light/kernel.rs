//! The kernel: spreads one channel of light from its sources through the open
//! voxels of a box around a chunk, by bit dilation.
//!
//! The box is `32 + 2 × reach` voxels on a side, centred on the chunk, so it
//! holds every voxel light can come from. Each row of the box runs along z, bit
//! `z` for the voxel at `z`. A row is one `u64` up to reach 16, and two, as a
//! `u128`, up to reach 32, where the box still fits in the 3×3×3 chunks around
//! the chunk. A round of spreading ORs each reached row with its neighbours,
//! shifted along z and taken from the rows beside it along x and y, and keeps
//! only open voxels.
//!
//! Light starts from two kinds of source. Seeds are single voxels, such as
//! emissive voxels, that join the light in the round their level says. Lit
//! voxels, such as voxels with direct sky, hold full light before the first
//! round.
//!
//! Round `d` only matters within `reach - d` voxels of the chunk, since light
//! has `reach - d` rounds left to get there. A round moves light at most one
//! voxel along each axis, so each round works on a box one voxel smaller on
//! every side than the last.
//!
//! The kernel knows nothing about which channel it spreads. More channels, such
//! as coloured voxel light, are more runs of it.
//!
//! A run also hashes, for each of the chunk's six faces, when light reached
//! each voxel on that face: see [`FaceHashes`](super::faces::FaceHashes).

use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, Not, Shl, Shr};

use super::config::LightShape;
use super::faces::{mix, Face};
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

/// The highest reach the kernel takes, which a box of two-`u64` rows holds.
pub(crate) const MAX_REACH: u32 = 32;

/// A row of a box along z: a `u64` up to reach 16, a `u128` up to reach 32.
pub(crate) trait Row:
    Copy
    + Eq
    + BitOr<Output = Self>
    + BitOrAssign
    + BitAnd<Output = Self>
    + BitAndAssign
    + Not<Output = Self>
    + Shl<usize, Output = Self>
    + Shr<usize, Output = Self>
    + Send
    + 'static
{
    const BITS: usize;
    const ZERO: Self;
    const ONE: Self;

    /// The row's low 32 bits.
    fn low_u32(self) -> u32;

    /// `column` as the row's low 32 bits.
    fn from_u32(column: u32) -> Self;
}

impl Row for u64 {
    const BITS: usize = 64;
    const ZERO: Self = 0;
    const ONE: Self = 1;

    #[inline]
    fn low_u32(self) -> u32 {
        self as u32
    }

    #[inline]
    fn from_u32(column: u32) -> Self {
        column as u64
    }
}

impl Row for u128 {
    const BITS: usize = 128;
    const ZERO: Self = 0;
    const ONE: Self = 1;

    #[inline]
    fn low_u32(self) -> u32 {
        self as u32
    }

    #[inline]
    fn from_u32(column: u32) -> Self {
        column as u128
    }
}

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
pub(crate) struct LightBox<R> {
    reach: usize,
    side: usize,
    /// Which voxels light passes through.
    open: Vec<R>,
    /// The voxels light has reached. Before a run that starts lit, the lit
    /// voxels.
    reached: Vec<R>,
    next: Vec<R>,
    pass: Vec<R>,
    /// Scratch for the chunk's newly reached voxels, as a [`Plane`].
    fresh: Vec<u32>,
    /// The voxels of `fresh` on the faces at z = 0 and z = 31, each a word
    /// along x, by y.
    z_faces: [[u32; 2]; CHUNK],
}

impl<R: Row> LightBox<R> {
    /// A box for light of `reach`, with every voxel closed and none lit. A row
    /// must hold `32 + 2 × reach` bits.
    pub fn new(reach: u32) -> Self {
        debug_assert!((1..=MAX_REACH).contains(&reach));
        let reach = reach as usize;
        let side = CHUNK + 2 * reach;
        debug_assert!(side <= R::BITS);
        Self {
            reach,
            side,
            open: vec![R::ZERO; side * side],
            reached: vec![R::ZERO; side * side],
            next: vec![R::ZERO; side * side],
            pass: vec![R::ZERO; side * side],
            fresh: vec![0; CHUNK * CHUNK],
            z_faces: [[0; 2]; CHUNK],
        }
    }

    pub fn reach(&self) -> usize {
        self.reach
    }

    /// The rows of open voxels, at `x + y × side`, to fill before a run. Bits at
    /// or above `side` must be clear.
    pub fn open_rows_mut(&mut self) -> &mut [R] {
        &mut self.open
    }

    /// The rows of lit voxels, at `x + y × side`, to fill before a run that
    /// starts lit. A lit voxel must be open.
    pub fn lit_rows_mut(&mut self) -> &mut [R] {
        &mut self.reached
    }

    /// Spreads light from `seeds` through the box's open voxels in `shape`, and
    /// writes the level each of the chunk's voxels reaches into `planes`, one
    /// plane per bit of the level. Voxels light doesn't reach, and closed ones,
    /// keep level 0.
    ///
    /// `seeds` must be sorted by their start. A seed's own voxel is never
    /// written: it holds the level of its source, not of light reaching it.
    ///
    /// With `lit`, light also spreads from the voxels filled in through
    /// [`lit_rows_mut`](Self::lit_rows_mut), each a source at distance 0 whose
    /// own voxel isn't written either. Without it, the lit rows are ignored.
    ///
    /// It also writes into `faces` a hash per face, indexed by [`Face`], of
    /// the round and pass in which light reached each of the chunk's voxels on
    /// that face, lit voxels counting as round 0. A face no light reaches
    /// hashes to 0.
    pub fn spread(
        &mut self,
        seeds: &[Seed],
        lit: bool,
        shape: LightShape,
        scale: &LevelScale,
        planes: &mut [Plane; LEVEL_BITS],
        faces: &mut [u64; 6],
    ) {
        debug_assert!(seeds.windows(2).all(|pair| pair[0].start <= pair[1].start));
        debug_assert_eq!(scale.reach() as usize, self.reach);
        for plane in planes.iter_mut() {
            plane.fill(0);
        }
        *faces = [0; 6];
        let (reach, side) = (self.reach, self.side);
        if lit {
            self.hash_lit_faces(faces);
        } else {
            self.reached.fill(R::ZERO);
        }
        let mut next_seed = 0;

        for round in 1..=reach {
            // Seeds starting at the step before this round join the light, and
            // light leaves through their own voxels for this round only.
            let first_seed = next_seed;
            while next_seed < seeds.len() && seeds[next_seed].start as usize + 1 == round
            {
                let seed = seeds[next_seed];
                let row = seed.x as usize + seed.y as usize * side;
                self.reached[row] |= R::ONE << seed.z as usize;
                self.open[row] |= R::ONE << seed.z as usize;
                next_seed += 1;
            }
            let round_seeds = &seeds[first_seed..next_seed];

            let low = round;
            let high = side - round;
            let z_mask = ones::<R>(low, high - low);
            let changed = if shape == LightShape::Round && round % 2 == 0 {
                self.three_passes(low, high, z_mask, round, faces)
            } else {
                self.face_step(low, high, z_mask)
            };

            for seed in round_seeds {
                let row = seed.x as usize + seed.y as usize * side;
                self.open[row] &= !(R::ONE << seed.z as usize);
            }

            let level = scale.level(round as u32);
            let fresh = self.fresh_rows();
            if fresh != 0 {
                for (bit, plane) in planes.iter_mut().enumerate() {
                    if level >> bit & 1 == 1 {
                        for (column, fresh) in plane.iter_mut().zip(&self.fresh) {
                            *column |= fresh;
                        }
                    }
                }
                self.hash_faces(round, faces);
            }
            std::mem::swap(&mut self.reached, &mut self.next);
            if changed == R::ZERO && next_seed == seeds.len() {
                break;
            }
        }
    }

    /// One face step from every reached voxel, over rows `low..high` along x
    /// and y. Returns the bits newly reached.
    fn face_step(&mut self, low: usize, high: usize, z_mask: R) -> R {
        let side = self.side;
        let width = high - low;
        let mut changed = R::ZERO;
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
            for (out, (((((&c, &xm), &xp), &ym), &yp), &open)) in
                self.next[i..i + width].iter_mut().zip(rows)
            {
                let row = (c | c << 1 | c >> 1 | xm | xp | ym | yp) & open & z_mask;
                *out = row;
                changed |= row & !c;
            }
        }
        changed
    }

    /// Writes into `fresh` the chunk's voxels that `next` holds and that
    /// weren't reached before the round, one `u32` column along z at
    /// `x + y × 32`. Returns all of them ORed together.
    fn fresh_rows(&mut self) -> u32 {
        let (reach, side) = (self.reach, self.side);
        let rows = &self.next;
        let mut any = 0;
        for y in 0..CHUNK {
            let row = reach + (y + reach) * side;
            let fresh = &mut self.fresh[y * CHUNK..(y + 1) * CHUNK];
            // The Z faces' voxels, gathered along x while the row is at hand.
            let (mut near, mut far) = (0u32, 0u32);
            for (x, ((fresh, &new), &old)) in fresh
                .iter_mut()
                .zip(&rows[row..row + CHUNK])
                .zip(&self.reached[row..row + CHUNK])
                .enumerate()
            {
                let column = ((new & !old) >> reach).low_u32();
                *fresh = column;
                any |= column;
                near |= (column & 1) << x;
                far |= (column >> (CHUNK - 1)) << x;
            }
            self.z_faces[y] = [near, far];
        }
        any
    }

    /// Mixes into `faces` the chunk's lit voxels on each face, as reached in
    /// round 0.
    fn hash_lit_faces(&mut self, faces: &mut [u64; 6]) {
        let (reach, side) = (self.reach, self.side);
        for y in 0..CHUNK {
            let row = reach + (y + reach) * side;
            let (mut near, mut far) = (0u32, 0u32);
            for x in 0..CHUNK {
                let column = (self.reached[row + x] >> reach).low_u32();
                self.fresh[x + y * CHUNK] = column;
                near |= (column & 1) << x;
                far |= (column >> (CHUNK - 1)) << x;
            }
            self.z_faces[y] = [near, far];
        }
        self.hash_faces(0, faces);
    }

    /// Mixes into the Y faces' hashes the voxels on them that the pass along x
    /// of `round` reached, read straight from `pass`.
    fn hash_y_faces_after_x(&self, round: usize, faces: &mut [u64; 6]) {
        let (reach, side) = (self.reach, self.side);
        let tag = Stage::AfterX(round).tag();
        for (face, y) in [(Face::NegY, 0), (Face::PosY, CHUNK - 1)] {
            let row = reach + (y + reach) * side;
            for (x, (&new, &old)) in self.pass[row..row + CHUNK]
                .iter()
                .zip(&self.reached[row..row + CHUNK])
                .enumerate()
            {
                mix_word(
                    &mut faces[face as usize],
                    tag,
                    x,
                    ((new & !old) >> reach).low_u32(),
                );
            }
        }
    }

    /// Mixes into the Z faces' hashes the voxels on them that the passes along
    /// x and y of `round` reached, read straight from `next`.
    fn hash_z_faces_after_y(&self, round: usize, faces: &mut [u64; 6]) {
        let (reach, side) = (self.reach, self.side);
        let tag = Stage::AfterY(round).tag();
        let far_bit = reach + CHUNK - 1;
        for y in 0..CHUNK {
            let row = reach + (y + reach) * side;
            let (mut near, mut far) = (0u32, 0u32);
            for (x, (&new, &old)) in self.next[row..row + CHUNK]
                .iter()
                .zip(&self.reached[row..row + CHUNK])
                .enumerate()
            {
                let fresh = new & !old;
                near |= ((fresh >> reach).low_u32() & 1) << x;
                far |= ((fresh >> far_bit).low_u32() & 1) << x;
            }
            mix_word(&mut faces[Face::NegZ as usize], tag, y, near);
            mix_word(&mut faces[Face::PosZ as usize], tag, y, far);
        }
    }

    /// Mixes into `faces` the chunk's face voxels in `fresh`, which `round`
    /// reached.
    ///
    /// Every face is hashed after a whole round. With round spread, the Y
    /// faces are also hashed after the pass along x, and the Z faces after the
    /// pass along y: a voxel reached by then sends light across those faces in
    /// the same round, one reached later only in the next.
    fn hash_faces(&self, round: usize, faces: &mut [u64; 6]) {
        let fresh = &self.fresh;
        let tag = Stage::Round(round).tag();
        let last = (CHUNK - 1) * CHUNK;
        for i in 0..CHUNK {
            let row = &fresh[i * CHUNK..(i + 1) * CHUNK];
            mix_word(&mut faces[Face::NegX as usize], tag, i, row[0]);
            mix_word(&mut faces[Face::PosX as usize], tag, i, row[CHUNK - 1]);
            mix_word(&mut faces[Face::NegY as usize], tag, i, fresh[i]);
            mix_word(&mut faces[Face::PosY as usize], tag, i, fresh[last + i]);
        }
        for (y, [near, far]) in self.z_faces.iter().enumerate() {
            mix_word(&mut faces[Face::NegZ as usize], tag, y, *near);
            mix_word(&mut faces[Face::PosZ as usize], tag, y, *far);
        }
    }

    /// A step along x, then along y, then along z, each from everything the
    /// step before reached, over rows `low..high` along x and y. Returns the
    /// bits newly reached.
    fn three_passes(
        &mut self,
        low: usize,
        high: usize,
        z_mask: R,
        round: usize,
        faces: &mut [u64; 6],
    ) -> R {
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
            for (out, (((&c, &xm), &xp), &open)) in
                self.pass[i..i + width].iter_mut().zip(rows)
            {
                *out = (c | xm | xp) & open;
            }
        }
        self.hash_y_faces_after_x(round, faces);
        // Along y into `next`.
        for y in low..high {
            let i = low + y * side;
            let rows = self.pass[i..i + width]
                .iter()
                .zip(&self.pass[i - side..i - side + width])
                .zip(&self.pass[i + side..i + side + width])
                .zip(&self.open[i..i + width]);
            for (out, (((&c, &ym), &yp), &open)) in
                self.next[i..i + width].iter_mut().zip(rows)
            {
                *out = (c | ym | yp) & open;
            }
        }
        self.hash_z_faces_after_y(round, faces);
        // Along z, in place.
        let mut changed = R::ZERO;
        for y in low..high {
            let i = low + y * side;
            for (out, (&open, &old)) in self.next[i..i + width].iter_mut().zip(
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

/// When in a run face voxels were reached.
#[derive(Clone, Copy)]
enum Stage {
    /// By the end of the round.
    Round(usize),
    /// By the end of the round's pass along x.
    AfterX(usize),
    /// By the end of the round's pass along y.
    AfterY(usize),
}

impl Stage {
    fn tag(self) -> u64 {
        match self {
            Stage::Round(round) => (round as u64) << 2,
            Stage::AfterX(round) => (round as u64) << 2 | 1,
            Stage::AfterY(round) => (round as u64) << 2 | 2,
        }
    }
}

/// Mixes one row of a face's newly reached voxels into its hash. Empty rows
/// are left out, so a face no light reaches keeps its hash of 0.
#[inline]
fn mix_word(hash: &mut u64, tag: u64, row: usize, word: u32) {
    if word != 0 {
        *hash = mix(*hash, tag << 40 | (row as u64) << 32 | word as u64);
    }
}

/// `count` set bits from bit `low` up.
fn ones<R: Row>(low: usize, count: usize) -> R {
    let bits = if count >= R::BITS {
        !R::ZERO
    } else {
        !(!R::ZERO << count)
    };
    bits << low
}
