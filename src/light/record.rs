//! The sky record: which columns of each level-0 chunk position hold a solid
//! voxel, for the chunk positions a world has seen.
//!
//! A chunk's column answer is the AND of the masks of the positions in its
//! sky rule's window: a column is clear only where no position above holds a
//! solid in it. So a chunk respawning under positions the record still holds
//! gets its exact direct sky on its first publish, however long ago the
//! chunks above it retired.
//!
//! Every generated level-0 chunk and every sky probe adds its position's mask.
//! The record is a cache capped in bytes, forgetting the least recently used
//! positions first. Forgetting costs only time: a forgotten position is
//! probed again when a chunk below it needs it, and gives the same mask.
//!
//! With sky fill on, each entry also holds the position's cells for sky fill:
//! which of its 8-voxel cells are open and which have direct sky. A
//! world's sky fill field is counted against the record's cap too.
//!
//! Each mask notes the edit count of its position when its voxels were read.
//! A mask whose position has been edited since reads as missing, so a voxel
//! edit never leaves a stale mask behind.

use std::collections::VecDeque;

use bevy::platform::collections::HashMap;
use bevy::prelude::*;

use super::fill::PositionCells;
use super::sky::Columns;

const CHUNK: usize = crate::chunk::CHUNK_SIZE_U as usize;

/// What an entry costs besides its mask, counted against the cap: its key,
/// its tag, its counts and its place in the eviction order.
const ENTRY_BYTES: usize = 64;

/// The bytes of a mask of a mixed position.
const MASK_BYTES: usize = std::mem::size_of::<Columns>();

/// The bytes of an entry's cells for sky fill.
const CELL_BYTES: usize = std::mem::size_of::<PositionCells>();

/// The smallest cap a world can set, so the record can hold a sky clearance's
/// worth of mixed positions.
pub(crate) const MIN_CAP: usize = 16 * (ENTRY_BYTES + MASK_BYTES);

/// Which columns of a chunk position hold a solid voxel.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Mask {
    /// No column holds a solid.
    Clear,
    /// Every column holds a solid.
    Blocked,
    Mixed(Box<Columns>),
}

impl Mask {
    fn of(columns: &Columns) -> Self {
        if columns.iter().all(|column| *column == 0) {
            Mask::Clear
        } else if columns.iter().all(|column| *column == !0) {
            Mask::Blocked
        } else {
            Mask::Mixed(Box::new(*columns))
        }
    }

    fn columns(&self) -> Columns {
        match self {
            Mask::Clear => [0; CHUNK],
            Mask::Blocked => [!0; CHUNK],
            Mask::Mixed(columns) => **columns,
        }
    }

    fn bytes(&self) -> usize {
        match self {
            Mask::Mixed(_) => ENTRY_BYTES + MASK_BYTES,
            _ => ENTRY_BYTES,
        }
    }
}

struct Entry {
    mask: Mask,
    /// With sky fill on, the position's cells.
    cells: Option<PositionCells>,
    /// The position's edit count when its voxels were read.
    edits: u64,
    /// When it was last read or written, on the record's clock.
    used: u64,
}

/// The sky record of one world.
pub(crate) struct SkyRecord {
    entries: HashMap<IVec3, Entry>,
    /// Positions in the order they were used, oldest first. A position used
    /// again is pushed again, and its older place is skipped when evicting.
    order: VecDeque<(IVec3, u64)>,
    clock: u64,
    bytes: usize,
    cap: usize,
    /// Changes whenever the record is emptied, so work that read the record
    /// before can tell.
    generation: u64,
    /// Entries forgotten to stay within the cap.
    pub evictions: u64,
    /// Whether entries keep their cells for sky fill.
    keeps_cells: bool,
    /// The bytes of the world's sky fill field, counted against the cap.
    field_bytes: usize,
    /// With sky fill on, the positions whose cells were recorded since
    /// [`take_noted`](Self::take_noted), and those forgotten since
    /// [`take_forgotten`](Self::take_forgotten).
    noted: Vec<IVec3>,
    forgotten: Vec<IVec3>,
}

impl SkyRecord {
    /// An empty record holding at most `cap` bytes.
    pub fn new(cap: usize) -> Self {
        Self {
            entries: HashMap::new(),
            order: VecDeque::new(),
            clock: 0,
            bytes: 0,
            cap,
            generation: 0,
            evictions: 0,
            keeps_cells: false,
            field_bytes: 0,
            noted: Vec::new(),
            forgotten: Vec::new(),
        }
    }

    /// Whether entries keep their cells for sky fill.
    pub fn keeps_cells(&self) -> bool {
        self.keeps_cells
    }

    /// Makes entries keep their cells for sky fill from now on, or drops
    /// every entry's cells.
    pub fn set_keeps_cells(&mut self, keeps: bool) {
        self.keeps_cells = keeps;
        if !keeps {
            for entry in self.entries.values_mut() {
                if entry.cells.take().is_some() {
                    self.bytes -= CELL_BYTES;
                }
            }
            self.noted.clear();
            self.forgotten.clear();
            self.field_bytes = 0;
        }
    }

    /// Counts `bytes` of the world's sky fill field against the cap,
    /// forgetting the least recently used positions past it.
    pub fn set_field_bytes(&mut self, bytes: usize) {
        self.field_bytes = bytes;
        self.evict_past_cap();
    }

    /// The cells of `position`, if the record holds them, made at edit count
    /// `edits`.
    pub fn cells(&mut self, position: IVec3, edits: u64) -> Option<PositionCells> {
        let entry = self.entries.get(&position)?;
        if entry.edits != edits {
            self.remove(position);
            return None;
        }
        entry.cells
    }

    /// Records which cells of `position` have direct sky, if the record holds
    /// its cells.
    pub fn set_direct(&mut self, position: IVec3, direct: u64) {
        if let Some(cells) = self
            .entries
            .get_mut(&position)
            .and_then(|entry| entry.cells.as_mut())
        {
            cells.direct = direct;
        }
    }

    /// The positions whose cells the record holds, with those cells.
    pub fn all_cells(&self) -> Vec<(IVec3, PositionCells)> {
        self.entries
            .iter()
            .filter_map(|(position, entry)| Some((*position, entry.cells?)))
            .collect()
    }

    /// The positions whose cells were recorded since this was last called.
    pub fn take_noted(&mut self) -> Vec<IVec3> {
        std::mem::take(&mut self.noted)
    }

    /// The positions whose entries were forgotten since this was last called.
    pub fn take_forgotten(&mut self) -> Vec<IVec3> {
        std::mem::take(&mut self.forgotten)
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// The columns holding a solid at `position`, or `None` if the record
    /// doesn't hold the position, or holds a mask made before the edit count
    /// moved from `edits`.
    pub fn get(&mut self, position: IVec3, edits: u64) -> Option<Columns> {
        let entry = self.entries.get_mut(&position)?;
        if entry.edits != edits {
            self.remove(position);
            return None;
        }
        self.clock += 1;
        entry.used = self.clock;
        self.order.push_back((position, self.clock));
        let columns = entry.mask.columns();
        self.compact();
        Some(columns)
    }

    /// Records the columns holding a solid at `position`, read at edit count
    /// `edits`, forgetting the least recently used positions if the record
    /// grows past its cap.
    ///
    /// With sky fill on, `open` holds the position's open cells, if known.
    /// Their direct sky is recorded later, with
    /// [`set_direct`](Self::set_direct).
    pub fn insert(
        &mut self,
        position: IVec3,
        columns: &Columns,
        open: Option<u64>,
        edits: u64,
    ) {
        self.remove(position);
        let mask = Mask::of(columns);
        self.clock += 1;
        self.bytes += mask.bytes();
        let cells = open
            .filter(|_| self.keeps_cells)
            .map(|open| PositionCells { open, direct: 0 });
        if cells.is_some() {
            self.bytes += CELL_BYTES;
            self.noted.push(position);
        }
        self.entries.insert(
            position,
            Entry {
                mask,
                cells,
                edits,
                used: self.clock,
            },
        );
        self.order.push_back((position, self.clock));
        self.evict_past_cap();
    }

    /// Holds at most `cap` bytes from now on, forgetting the least recently
    /// used positions past it.
    pub fn set_cap(&mut self, cap: usize) {
        self.cap = cap;
        self.evict_past_cap();
    }

    fn evict_past_cap(&mut self) {
        while self.bytes + self.field_bytes > self.cap {
            let Some((oldest, used)) = self.order.pop_front() else {
                break;
            };
            if self
                .entries
                .get(&oldest)
                .is_some_and(|entry| entry.used == used)
            {
                self.remove(oldest);
                self.evictions += 1;
            }
        }
        self.compact();
    }

    /// Forgets every position, as evicting all of them would.
    #[cfg(test)]
    pub fn forget_all(&mut self) {
        if self.keeps_cells {
            self.forgotten.extend(self.entries.keys().copied());
        }
        self.entries.clear();
        self.order.clear();
        self.bytes = 0;
    }

    /// Forgets every position, and makes the masks of work that read the
    /// record before stale, as after the world's lookups change.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.order.clear();
        self.bytes = 0;
        self.generation += 1;
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// The bytes of the world's sky fill field it counts against its cap.
    pub fn field_bytes(&self) -> usize {
        self.field_bytes
    }

    fn remove(&mut self, position: IVec3) {
        if let Some(entry) = self.entries.remove(&position) {
            self.bytes -= entry.mask.bytes();
            if entry.cells.is_some() {
                self.bytes -= CELL_BYTES;
                self.forgotten.push(position);
            }
        }
    }

    /// Drops the eviction order's skipped places once they outnumber the
    /// entries.
    fn compact(&mut self) {
        if self.order.len() <= 2 * self.entries.len() + 64 {
            return;
        }
        let entries = &self.entries;
        self.order.retain(|(position, used)| {
            entries.get(position).is_some_and(|e| e.used == *used)
        });
    }
}
