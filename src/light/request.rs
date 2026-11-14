//! Sky light worked out on request, for `light_at` where nothing renders: in a
//! coarse chunk with no mesh, and in hinted air holding no light entry.
//!
//! No chunk holds light there, so a reading walks up the column above the
//! asked voxel on the caller's thread, along the rows a coarse chunk's walk
//! takes (see [`CoarseWalk`](super::coarse::CoarseWalk)): starting one cell
//! above it, a cell's width apart, doubling the step every 8 rows, up to
//! where the world's sky rule stops looking. Each row reads the voxel edit
//! there, then the fill hint of the level-0 chunk holding it, then the
//! world's lookup for that chunk. The column is open, and the voxel reads sky
//! light 15, if no row is solid; otherwise it reads 0. Rows can step past a
//! roof thinner than the step, so the answer is plausible rather than exact.
//!
//! Answers go in a [`WalkCache`], so a game reading the same voxel every
//! frame walks it once.

use std::sync::atomic::{AtomicU64, Ordering};

use bevy::platform::collections::HashMap;
use bevy::prelude::*;

use super::sky::SkyRule;
use crate::chunk::{chunk_containing, CHUNK_SIZE_I};
use crate::configuration::VoxelWorldConfig;
use crate::level_0_reader::{ChunkLookups, Hinted};
use crate::voxel_edits::VoxelEditReader;

/// How many walked voxels the cache keeps answers for.
///
/// A game reading the light of its entities each frame needs one answer per
/// entity standing still; one that moves walks again for each voxel it
/// enters. At this size the cache takes at most about 170 KiB. Once it is
/// full it empties and starts again, which costs one walk per voxel still
/// read.
pub(crate) const WALK_CACHE_CAP: usize = 4096;

/// The answers of walks on request: whether the column above a voxel is open,
/// by the voxel and the width of the cells its rows step by.
///
/// Any voxel edit, a `reset_light` and a change of light settings empty it.
#[derive(Default)]
pub(crate) struct WalkCache {
    answers: HashMap<(IVec3, u32), bool>,
}

impl WalkCache {
    pub fn get(&self, position: IVec3, width: u32) -> Option<bool> {
        self.answers.get(&(position, width)).copied()
    }

    pub fn insert(&mut self, position: IVec3, width: u32, open: bool) {
        if self.answers.len() >= WALK_CACHE_CAP {
            self.answers.clear();
        }
        self.answers.insert((position, width), open);
    }

    pub fn clear(&mut self) {
        self.answers.clear();
    }

    /// The bytes the cache holds now, counting the table's spare room.
    pub fn bytes(&self) -> usize {
        // Each bucket holds an entry and a control byte.
        self.answers.capacity() * (std::mem::size_of::<((IVec3, u32), bool)>() + 1)
    }
}

/// How many walks on request ran and how many voxel lookups they made.
#[derive(Default)]
pub(crate) struct RequestCounters {
    pub walks: AtomicU64,
    pub lookups: AtomicU64,
}

/// Whether sky light falls straight down the column above the voxel at
/// `position` under `rule`, walking rows `width` voxels apart from the voxel
/// up. `top` is the height just above the chunk holding the voxel, which the
/// sky rule's clearance counts from.
pub(crate) fn column_is_open<C: VoxelWorldConfig>(
    position: IVec3,
    width: u32,
    top: i32,
    rule: SkyRule,
    world: &C,
    edits: &VoxelEditReader<C::MaterialIndex>,
    counters: &RequestCounters,
) -> bool {
    let limit = match rule {
        SkyRule::Clearance(chunks) => top as i64 + chunks as i64 * CHUNK_SIZE_I as i64,
        SkyRule::Ceiling(chunk) => (chunk as i64 + 1) * CHUNK_SIZE_I as i64,
    }
    .min(i32::MAX as i64);
    let mut lookups = ChunkLookups::new(world.voxel_lookup_delegate());
    let mut made = 0;
    let (mut y, mut step, mut rows) = (position.y as i64, width as i64, 0);
    let open = loop {
        y += step;
        if y >= limit {
            break true;
        }
        rows += 1;
        if rows % 8 == 0 {
            step *= 2;
        }
        let at = IVec3::new(position.x, y as i32, position.z);
        let voxel = match edits.get(at) {
            Some(voxel) => voxel,
            // The edit at the voxel was read first, so the hint speaks for
            // the voxel whatever else is edited there.
            None => match lookups.hinted(world, chunk_containing(at)) {
                Hinted::Air => continue,
                Hinted::Solid => break false,
                Hinted::Unknown => {
                    made += 1;
                    lookups.lookup(at)
                }
            },
        };
        if voxel.is_solid() {
            break false;
        }
    };
    counters.walks.fetch_add(1, Ordering::Relaxed);
    counters.lookups.fetch_add(made, Ordering::Relaxed);
    open
}
