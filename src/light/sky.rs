//! Direct sky: which columns of a chunk sky light falls into from above.
//!
//! A world picks its sky rule with [`Sky`](super::config::Sky). With a sky
//! clearance of N chunks, sky light enters the top of a chunk in a column when
//! that column holds no solid voxel in the N whole chunks straight above. So a
//! shadow ends about N chunks below what casts it, in a flat line on a chunk
//! boundary: the first chunk whose N chunks above no longer reach the solid
//! gets direct sky across its whole top. Counting whole chunks keeps a
//! column's answer the same for every voxel of a chunk, and lets answers for
//! chunk positions combine with a plain AND. With a sky ceiling at y, rounded
//! up to the top of its chunk, sky light enters a column holding no solid up to
//! there, so nothing above the ceiling casts a shadow.
//!
//! Direct sky falls at level 15 from the top of the chunk down to the first
//! solid voxel in the column, at any distance, and the solver spreads it from
//! there sideways and down.
//!
//! A chunk works out its own column answer in its chunk task, settling columns
//! in this order and stopping once every column is settled:
//!
//! 1. Its padding row above it: a solid there blocks the column.
//! 2. Fill hints of the chunk positions above it: a `Uniform` air hint counts
//!    as clear and a `Uniform` solid hint as blocked, with no lookups. A hint
//!    is ignored for a chunk position holding voxel edits.
//! 3. The [sky record](super::record): the masks of the positions above it the
//!    world has already seen, read when the task starts.
//! 4. A walk upward a row at a time through the world's voxel lookups, over
//!    the positions the hints and the record leave unknown, if the walk budget
//!    can carry every open column through at least one chunk's height.
//!    Otherwise the task skips it.
//!
//! When the chunk lands, its answer is checked against the record again, which
//! may have learned more meanwhile. Columns still unsettled then wait for
//! [sky probes](super::probe).
//!
//! A column with a solid proved above it gets no direct sky. Every other
//! column is provisional and gets direct sky until something proves otherwise,
//! so a column nothing has settled yet shows direct sky. A chunk's own answer
//! decides its direct sky, whatever the chunk above it shows.
//!
//! Sky light reads columns from the world's lookups as well as from loaded
//! chunks, so it needs a lookup to return the same voxel for a position every
//! time it is asked.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use bevy::platform::collections::HashMap;
use bevy::prelude::*;

use super::config::Sky;
use crate::chunk::{ChunkData, FillType, CHUNK_SIZE_I, CHUNK_SIZE_U, PADDED_CHUNK_SIZE};
use crate::configuration::{FillHint, VoxelLookupDelegate};
use crate::voxel::WorldVoxel;

const CHUNK: usize = CHUNK_SIZE_U as usize;
const PADDED: usize = PADDED_CHUNK_SIZE as usize;

/// How many voxel lookups a chunk task's walk may make.
///
/// The walk runs only if this carries every column still open through at
/// least one chunk's height, so at most 343 open columns. Chunks under open
/// sky leave most columns open and skip the walk, showing direct sky
/// provisionally, which is right for them. Chunks under solid ground have
/// only their padding row's gaps open, and a walk through the chunks above
/// settles those.
///
/// Chosen at 11,000: streaming the sine terrain of the `chunk_lifecycle`
/// benchmark in, 69 of 4,169 chunk tasks walked, at about 9,400 lookups and
/// 84 µs each, about 1.4 µs per chunk task overall.
pub(crate) const WALK_BUDGET: u32 = 11_000;

/// One bit per column of a chunk, bit `z` of word `x`: 128 B.
pub(crate) type Columns = [u32; CHUNK];

/// A chunk's column answer: which of its columns sky light enters at its top.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct ColumnAnswer {
    /// The columns sky light enters, proved clear or provisional.
    pub sky: Columns,
    /// The columns among `sky` that nothing has proved clear yet.
    pub provisional: Columns,
}

impl ColumnAnswer {
    /// No column has direct sky, as with sky light off.
    pub fn none() -> Self {
        Self::default()
    }

    fn all_proved() -> Self {
        Self {
            sky: [!0; CHUNK],
            provisional: [0; CHUNK],
        }
    }

    pub fn has_sky(&self) -> bool {
        self.sky.iter().any(|column| *column != 0)
    }
}

/// The sky rule as chunk tasks read it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SkyRule {
    /// Sky light enters a chunk's column holding no solid in this many chunks
    /// straight above.
    Clearance(u32),
    /// Sky light enters a chunk's column holding no solid up to the top of the
    /// chunk at this height, in chunks.
    Ceiling(i32),
}

impl SkyRule {
    /// The rule for `sky`, or `None` with sky light off.
    pub fn of(sky: &Sky) -> Option<Self> {
        match sky {
            Sky::Clearance { chunks } => Some(SkyRule::Clearance(*chunks)),
            Sky::Ceiling { y } => Some(SkyRule::Ceiling(y.div_euclid(CHUNK_SIZE_I))),
            Sky::Off => None,
        }
    }

    /// How many chunk positions straight above the level-0 chunk at height
    /// `chunk_y`, in chunks, a column must be clear in for sky light to enter.
    pub fn window(self, chunk_y: i32) -> u32 {
        match self {
            SkyRule::Clearance(chunks) => chunks,
            SkyRule::Ceiling(top) => {
                (top as i64 - chunk_y as i64).clamp(0, u32::MAX as i64) as u32
            }
        }
    }
}

/// What a fill hint says about a chunk position above a chunk.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Hinted {
    Unknown,
    Air,
    Solid,
}

impl Hinted {
    /// What `hint` says, or `Unknown` for a chunk position voxel edits reach.
    pub fn of<I: Copy + PartialEq>(hint: FillHint<I>, edited: bool) -> Self {
        match hint {
            _ if edited => Hinted::Unknown,
            FillHint::Unknown => Hinted::Unknown,
            FillHint::Uniform(voxel) if voxel.is_solid() => Hinted::Solid,
            FillHint::Uniform(_) => Hinted::Air,
        }
    }
}

/// A chunk position above a chunk, as its walk reads it.
pub(crate) struct Above<I> {
    hinted: Hinted,
    /// The position's columns holding a solid, from the sky record.
    recorded: Option<Columns>,
    /// The voxel edits in the position, read before the world's lookups.
    edits: Option<Arc<HashMap<IVec3, WorldVoxel<I>>>>,
}

impl<I> Above<I> {
    /// The position with fill hint `hint`, its mask `recorded` in the sky
    /// record, and voxel edits `edits`. A hint is ignored where there are
    /// edits.
    pub fn new(
        hint: FillHint<I>,
        recorded: Option<Columns>,
        edits: Option<Arc<HashMap<IVec3, WorldVoxel<I>>>>,
    ) -> Self
    where
        I: Copy + PartialEq,
    {
        Self {
            hinted: Hinted::of(hint, edits.is_some()),
            recorded,
            edits,
        }
    }
}

/// What a chunk task needs to work out its column answer: the positions above
/// it within the sky rule, and the world's lookups to walk them.
pub(crate) struct SkyWalk<I> {
    /// The level-0 chunk the answer is for.
    pub position: IVec3,
    /// How many chunk positions above the chunk a column must be clear in.
    pub window: u32,
    /// The lowest positions of the window, those a walk can reach within its
    /// budget, lowest first.
    pub above: Vec<Above<I>>,
    pub delegate: VoxelLookupDelegate<I>,
    /// What the main thread needs to check the answer when the chunk lands.
    pub found: SkyFound,
}

/// What a chunk task found out about its columns besides its column answer,
/// so that the main thread can check the answer against the sky record when
/// the chunk lands.
#[derive(Clone, Debug, Default)]
pub(crate) struct SkyFound {
    /// The columns open in the chunk's padding row above it.
    pub open: Columns,
    /// By position above the chunk, lowest first, whether the answer takes
    /// in its voxels: from a fill hint, the sky record or a walk.
    pub known: Vec<bool>,
    /// The edit count of each position above the chunk when the task
    /// started.
    pub edits: Vec<u64>,
    /// The edit count of the chunk's own position when the task started.
    pub own_edits: u64,
    /// The sky record's generation when the task started.
    pub generation: u64,
}

/// How many positions above a chunk a walk can reach within its budget.
pub(crate) fn walkable(window: u32) -> u32 {
    window.min(WALK_BUDGET / (CHUNK_SIZE_U - 1) + 1)
}

/// Counts of sky walk work, for tests and benchmarks.
#[derive(Default)]
pub(crate) struct WalkCounters {
    /// Chunk tasks that walked.
    pub walks: AtomicU64,
    /// Voxel lookups walks made.
    pub lookups: AtomicU64,
}

impl<I: Copy + PartialEq> SkyWalk<I> {
    /// The column answer of `chunk`, the level-0 chunk the walk is for, and
    /// what was found on the way.
    pub fn answer(
        mut self,
        chunk: &ChunkData<I>,
        counters: &WalkCounters,
    ) -> (ColumnAnswer, SkyFound) {
        let mut found = std::mem::take(&mut self.found);
        found.known = vec![false; self.above.len()];
        if self.window == 0 {
            found.open = [!0; CHUNK];
            return (ColumnAnswer::all_proved(), found);
        }
        found.open = padding_row(chunk);
        let mut open = found.open;
        if self.above.iter().any(|above| above.hinted == Hinted::Solid) {
            return (ColumnAnswer::none(), found);
        }
        for (known, above) in found.known.iter_mut().zip(&self.above) {
            if let Some(recorded) = &above.recorded {
                for (open, solid) in open.iter_mut().zip(recorded) {
                    *open &= !solid;
                }
            }
            *known = above.hinted == Hinted::Air || above.recorded.is_some();
        }
        if open.iter().all(|column| *column == 0) {
            return (ColumnAnswer::none(), found);
        }
        let complete = self.above.len() as u32 == self.window;
        if complete && found.known.iter().all(|known| *known) {
            return (proved(open), found);
        }

        let open_count: u32 = open.iter().map(|column| column.count_ones()).sum();
        if open_count * CHUNK_SIZE_U > WALK_BUDGET {
            return (provisional(open), found);
        }
        counters.walks.fetch_add(1, Ordering::Relaxed);
        let (blocked, lookups) = self.walk(&mut open, &mut found.known);
        counters
            .lookups
            .fetch_add(lookups as u64, Ordering::Relaxed);
        let answer = if blocked || complete && found.known.iter().all(|known| *known) {
            proved(open)
        } else {
            provisional(open)
        };
        (answer, found)
    }

    /// Walks the columns still set in `open` upward a row at a time through
    /// the positions not yet `known`, clearing each column a solid blocks,
    /// until every column is blocked, the window ends or the budget runs out.
    /// Marks each position it walks through whole as known.
    ///
    /// Returns whether every column was blocked, and how many lookups it
    /// made.
    fn walk(&self, open: &mut Columns, known: &mut [bool]) -> (bool, u32) {
        let mut budget = WALK_BUDGET;
        let mut lookups = 0;
        for (index, above) in self.above.iter().enumerate() {
            if known[index] {
                continue;
            }
            let position = self.position + IVec3::new(0, index as i32 + 1, 0);
            let mut lookup = (self.delegate)(position, 0, None);
            let min = position * CHUNK_SIZE_I;
            // The chunk's padding row is the first row of the position above.
            let first_row = if index == 0 { 1 } else { 0 };
            for row in first_row..CHUNK {
                for (x, open) in open.iter_mut().enumerate() {
                    let mut column = *open;
                    while column != 0 {
                        let z = column.trailing_zeros() as usize;
                        column &= column - 1;
                        if budget == 0 {
                            return (false, lookups);
                        }
                        budget -= 1;
                        lookups += 1;
                        let at = min + IVec3::new(x as i32, row as i32, z as i32);
                        let voxel = above
                            .edits
                            .as_ref()
                            .and_then(|edits| edits.get(&at).copied())
                            .unwrap_or_else(|| lookup(at, None));
                        if voxel.is_solid() {
                            *open &= !(1 << z);
                        }
                    }
                }
                if open.iter().all(|column| *column == 0) {
                    return (true, lookups);
                }
            }
            known[index] = true;
        }
        (false, lookups)
    }
}

/// The answer for the columns `sky`, each proved clear.
pub(crate) fn proved(sky: Columns) -> ColumnAnswer {
    ColumnAnswer {
        sky,
        provisional: [0; CHUNK],
    }
}

/// The answer for the columns `sky`, none of them proved clear yet.
pub(crate) fn provisional(sky: Columns) -> ColumnAnswer {
    ColumnAnswer {
        sky,
        provisional: sky,
    }
}

/// The columns whose voxel in the chunk's padding row above it is open.
pub(crate) fn padding_row<I: Copy + PartialEq>(chunk: &ChunkData<I>) -> Columns {
    match (&chunk.fill_type, &chunk.voxels) {
        (FillType::Empty, _) => [!0; CHUNK],
        (FillType::Uniform(voxel), _) => {
            if voxel.is_solid() {
                [0; CHUNK]
            } else {
                [!0; CHUNK]
            }
        }
        (FillType::Mixed, Some(voxels)) => {
            let mut open = [0; CHUNK];
            let top = (PADDED - 1) * PADDED;
            for z in 0..CHUNK {
                let row = 1 + top + (z + 1) * PADDED * PADDED;
                for (x, voxel) in voxels[row..row + CHUNK].iter().enumerate() {
                    open[x] |= (!voxel.is_solid() as u32) << z;
                }
            }
            open
        }
        (FillType::Mixed, None) => unreachable!("a mixed chunk stores its voxels"),
    }
}
