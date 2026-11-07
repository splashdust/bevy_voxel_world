//! Sky fill: sky light beyond direct sky, worked out per 8-voxel cell, so the
//! ground under a high floating island and the mouth of a cave behind a wide
//! opening get light that spreading from direct sky can't carry that far.
//!
//! Every chunk position is cut into 4×4×4 cells of 8×8×8 voxels. A cell is
//! open when it holds no solid voxel, and a direct-sky cell when its voxels
//! all have direct sky, proved clear above. The world's sky record keeps both
//! masks for each position it has seen (see [`record`](super::record)).
//!
//! A field, one per world on the main thread, works out a level for every
//! open cell of the positions it knows from those masks. How it does that is
//! a model, picked by [`SkyFill`]: [`SkyFillField`] holds one, and the rest
//! of the crate never knows which. A model also says which cells count as
//! direct sky, which positions to work out again after a change, and which
//! positions are worth probing. The default model, [`paths`], puts a level on
//! each open region of a position and passes it between regions through the
//! openings between them, losing more through an opening that fills less of
//! the region behind it. So a whole open face between two big spaces loses
//! little, and a small door into a big room carries little.
//!
//! Everything per voxel is shared by every model: a solve reads the cells of
//! the 27 positions around its chunk onto a grid of 12×12×12 cells, blurs
//! their levels over open cells to soften the steps between positions, and
//! seeds the kernel with them, after running each seed 7 voxels through open
//! voxels with no loss, so a surface inside the next cell doesn't lose a
//! level per voxel (see [`seed`]).
//!
//! A position the field doesn't know reads as blocked: fill is exact once
//! every position that could raise a level is known, and darker than the
//! truth before, never brighter. Fill probes, which run after everything
//! else, read the positions a model asks for.
//!
//! A cell's level changing relights the chunks around it that take fill, at
//! most every 500 ms: see [`driver`].

pub(crate) mod driver;
pub(crate) mod paths;
pub(crate) mod seed;

use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};
use std::time::Duration;

use bevy::log::warn_once;
use bevy::platform::collections::HashMap;
use bevy::prelude::*;

use super::config::SkyFill;
use super::inputs::Occupancy;
use super::sky::Columns;
use crate::chunk::CHUNK_SIZE_U;
use paths::PathField;

const CHUNK: usize = CHUNK_SIZE_U as usize;

/// How many voxels a cell is along each axis.
pub(crate) const CELL: usize = 8;

/// What the world knows of a position's cells, one bit per cell at
/// `x + y × 4 + z × 16` for the cell at `x`, `y`, `z`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct PositionCells {
    /// The cells holding no solid voxel.
    pub open: u64,
    /// The open cells whose voxels all have direct sky, proved clear.
    pub direct: u64,
}

/// What a solve reads of a position's cells: which are open and which are
/// direct sky, and each one's level in sixteenths of a light level.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct CellFill {
    pub open: u64,
    pub direct: u64,
    /// By cell, the level the blur starts from, in sixteenths of a level.
    pub levels: [u8; 64],
}

/// The bit of the cell at `x`, `y`, `z` of a position.
#[inline]
pub(crate) fn cell_bit(x: usize, y: usize, z: usize) -> u64 {
    1 << (x + y * 4 + z * 16)
}

/// The cells of a chunk with `occupancy` that hold no solid voxel.
pub(crate) fn open_cells(occupancy: &Occupancy) -> u64 {
    let columns = match occupancy {
        Occupancy::Open => return !0,
        Occupancy::Solid => return 0,
        Occupancy::Mixed(columns) => columns,
    };
    let mut open = 0;
    for cy in 0..4 {
        for cx in 0..4 {
            let mut solid = 0u32;
            for y in cy * CELL..(cy + 1) * CELL {
                for x in cx * CELL..(cx + 1) * CELL {
                    solid |= columns[x + y * CHUNK];
                }
            }
            for cz in 0..4 {
                if solid >> (cz * CELL) & 0xff == 0 {
                    open |= cell_bit(cx, cy, cz);
                }
            }
        }
    }
    open
}

/// The top cells, as bits `x + z × 4`, whose 64 columns are all in `sky`.
pub(crate) fn sky_top_cells(sky: &Columns) -> u16 {
    let mut cells = 0;
    for cz in 0..4 {
        for cx in 0..4 {
            let all = (cx * CELL..(cx + 1) * CELL)
                .all(|x| sky[x] >> (cz * CELL) & 0xff == 0xff);
            if all {
                cells |= 1 << (cx + cz * 4);
            }
        }
    }
    cells
}

/// The cells of `open` that have direct sky when the top cells `top` see the
/// sky: each open cell below such a top cell, down to the first cell that
/// isn't open.
pub(crate) fn direct_cells(open: u64, top: u16) -> u64 {
    let mut direct = 0;
    for cz in 0..4 {
        for cx in 0..4 {
            if top >> (cx + cz * 4) & 1 == 0 {
                continue;
            }
            for cy in (0..4).rev() {
                let bit = cell_bit(cx, cy, cz);
                if open & bit == 0 {
                    break;
                }
                direct |= bit;
            }
        }
    }
    direct
}

/// The cells of the positions a world's field knows, shared with the tasks
/// that solve light, which read them the way they read occupancy. Only the
/// main thread writes it.
#[derive(Clone, Default)]
pub(crate) struct FillCells(Arc<RwLock<HashMap<IVec3, Arc<CellFill>>>>);

impl FillCells {
    pub fn read(&self) -> RwLockReadGuard<'_, HashMap<IVec3, Arc<CellFill>>> {
        self.0.read().unwrap()
    }

    pub fn write(&self) -> RwLockWriteGuard<'_, HashMap<IVec3, Arc<CellFill>>> {
        self.0.write().unwrap()
    }
}

/// Sky light beyond direct sky, per cell, as one of the models of
/// [`SkyFill`]. The solver reads only its cells, and never knows how they
/// were worked out.
///
/// Each call matches the model once, never per voxel.
pub(crate) enum SkyFillField {
    Paths(PathField),
}

impl SkyFillField {
    /// The field of `fill`, or `None` for a model not built yet.
    pub fn new(fill: &SkyFill) -> Option<Self> {
        match fill {
            SkyFill::Paths(paths) => Some(SkyFillField::Paths(PathField::new(paths))),
            SkyFill::Openness(_) => {
                warn_once!(
                    "light propagation: SkyFill::Openness isn't built yet, so sky fill \
                     is off"
                );
                None
            }
        }
    }

    /// Takes in what is known of the cells of `position`, from its voxels and
    /// its sky answer. Takes effect at the next [`update`](Self::update).
    pub fn learn(&mut self, position: IVec3, cells: PositionCells) {
        match self {
            SkyFillField::Paths(field) => field.learn(position, cells),
        }
    }

    /// Forgets `position`, which then reads as blocked. Takes effect at the
    /// next [`update`](Self::update).
    pub fn forget(&mut self, position: IVec3) {
        match self {
            SkyFillField::Paths(field) => field.forget(position),
        }
    }

    /// Works through what was learned and forgotten, for at most about
    /// `budget`, leaving the rest for the next call. Returns the positions
    /// whose cells changed.
    pub fn update(&mut self, budget: Duration) -> Vec<IVec3> {
        match self {
            SkyFillField::Paths(field) => field.update(budget),
        }
    }

    /// Whether something learned or forgotten waits for an update.
    pub fn has_pending(&self) -> bool {
        match self {
            SkyFillField::Paths(field) => field.has_pending(),
        }
    }

    /// Positions the field doesn't know whose voxels could still raise a
    /// level.
    pub fn wanted_probes(&self) -> impl Iterator<Item = IVec3> + '_ {
        match self {
            SkyFillField::Paths(field) => field.wanted_probes(),
        }
    }

    /// The cells of `position`, if the field knows it.
    pub fn cells(&self, position: IVec3) -> Option<&CellFill> {
        match self {
            SkyFillField::Paths(field) => field.cells(position),
        }
    }

    /// Roughly how many bytes the field holds.
    pub fn bytes(&self) -> usize {
        match self {
            SkyFillField::Paths(field) => field.bytes(),
        }
    }
}
