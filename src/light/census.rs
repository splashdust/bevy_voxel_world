//! Counting how often a world's shown light changes after chunks first show
//! it, and why, for the `chunk_lifecycle` bench. A world counts only once
//! [`light_bench::count_changes`](super::bench::count_changes) turns it on,
//! since comparing light voxel by voxel costs more than landing it.
//!
//! A chunk's light is visible where an open voxel touches a solid voxel by a
//! face, since a mesh's faces show the light in front of them. A change is
//! put down to one cause:
//!
//! - a neighbour arriving: a relight that a chunk task's light landing
//!   started, or that rippled on from one;
//! - a column found out: a relight started because a chunk's direct sky
//!   changed, when sky probes proved or disproved a column it had guessed;
//! - a shell filling in: light landing in a chunk's border voxels in front of
//!   a loaded neighbour's solid voxels, which that neighbour's faces show;
//! - anything else, such as sky fill or a voxel edit.
//!
//! Changes are counted in batches: at most one of each kind per chunk per
//! frame, after the frame its own light first landed.

use bevy::platform::collections::HashMap;
use bevy::prelude::*;

use super::faces::Face;
use super::inputs::Occupancy;
use super::solver::{ChannelLight, ChunkLight};
use super::view::ViewMaps;
use crate::chunk::CHUNK_SIZE_U;

const CHUNK: usize = CHUNK_SIZE_U as usize;

/// Why a chunk's light was solved again.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum Cause {
    /// A neighbour's chunk task landed light that could reach it.
    Arrival,
    /// Its direct sky changed as sky probes found out its columns.
    Column,
    #[default]
    Other,
}

/// What changed a chunk's shown light, as counted in [`ChangeCensus`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Change {
    Own(Cause),
    Shell,
}

/// Counts of changes to shown light, by cause.
#[derive(Clone, Copy, Debug, Default)]
pub struct ChangeCounts {
    /// Level-0 chunks whose chunk task's light landed for the first time.
    pub arrivals: u64,
    /// Of those, the chunks with a voxel whose light is visible.
    pub shown: u64,
    /// Relights that landed light other than the chunk's.
    pub changed_relights: u64,
    /// Frames after a chunk's arrival in which its light changed, summed
    /// over chunks.
    pub changed_batches: u64,
    /// Frames after a chunk's arrival in which a relight changed its
    /// visible light, summed over chunks, by cause.
    pub visible_from_arrival: u64,
    pub visible_from_column: u64,
    pub visible_from_other: u64,
    /// Frames after a chunk's arrival in which its visible light or the light
    /// in front of its solid voxels on its faces changed, summed over chunks.
    pub visible_with_shells: u64,
    /// Of those, the frames in which only the light in front of its faces
    /// changed.
    pub visible_from_shell: u64,
}

/// The kinds of change counted at most once per chunk per frame.
#[derive(Clone, Copy)]
enum Batch {
    Changed,
    OwnVisible,
    Visible,
}

/// Each chunk's frames of note, and the counts so far.
#[derive(Default)]
pub(crate) struct ChangeCensus {
    frame: u64,
    arrived: HashMap<IVec3, u64>,
    changed: HashMap<IVec3, u64>,
    own_visible: HashMap<IVec3, u64>,
    visible: HashMap<IVec3, u64>,
    pub counts: ChangeCounts,
}

impl ChangeCensus {
    /// Starts a frame of landing.
    pub fn next_frame(&mut self) {
        self.frame += 1;
    }

    /// Notes that the chunk at `position` retired.
    pub fn forget(&mut self, position: IVec3) {
        self.arrived.remove(&position);
        self.changed.remove(&position);
        self.own_visible.remove(&position);
        self.visible.remove(&position);
    }

    /// Counts a level-0 chunk's light landing at `position`: `old` is the
    /// light it replaced, `None` on the chunk's arrival, and `cause` why it
    /// was solved again. `view` holds the chunks around it.
    pub fn landed(
        &mut self,
        position: IVec3,
        old: Option<&ChunkLight>,
        new: &ChunkLight,
        cause: Cause,
        view: &ViewMaps,
    ) {
        match old {
            None => {
                self.arrived.insert(position, self.frame);
                self.counts.arrivals += 1;
                if visible_voxels(&new.inputs.occupancy)
                    .iter()
                    .any(|c| *c != 0)
                {
                    self.counts.shown += 1;
                }
            }
            Some(old) => {
                if old.same_levels(new) {
                    return;
                }
                self.counts.changed_relights += 1;
                if self.is_new_batch(position, Batch::Changed) {
                    self.counts.changed_batches += 1;
                }
                if own_change_shows(old, new) {
                    self.count_visible(position, Change::Own(cause));
                }
            }
        }
        for face in Face::ALL {
            let beside = position + face.offset();
            let Some(neighbour) = view.get(&beside).filter(|chunk| !chunk.is_entry())
            else {
                continue;
            };
            let solid = &neighbour.light.inputs.occupancy;
            if shell_change_shows(old, new, face, solid) {
                self.count_visible(beside, Change::Shell);
            }
        }
    }

    /// Whether this frame comes after the arrival of the chunk at `position`
    /// and is the first with a change of `batch`'s kind.
    fn is_new_batch(&mut self, position: IVec3, batch: Batch) -> bool {
        let Some(arrived) = self.arrived.get(&position) else {
            return false;
        };
        if *arrived == self.frame {
            return false;
        }
        let last = match batch {
            Batch::Changed => &mut self.changed,
            Batch::OwnVisible => &mut self.own_visible,
            Batch::Visible => &mut self.visible,
        };
        last.insert(position, self.frame) != Some(self.frame)
    }

    fn count_visible(&mut self, position: IVec3, change: Change) {
        if let Change::Own(cause) = change {
            if self.is_new_batch(position, Batch::OwnVisible) {
                let counts = &mut self.counts;
                *match cause {
                    Cause::Arrival => &mut counts.visible_from_arrival,
                    Cause::Column => &mut counts.visible_from_column,
                    Cause::Other => &mut counts.visible_from_other,
                } += 1;
            }
        }
        if self.is_new_batch(position, Batch::Visible) {
            self.counts.visible_with_shells += 1;
            if change == Change::Shell {
                self.counts.visible_from_shell += 1;
            }
        }
    }
}

/// The chunk's open voxels touching one of its solid voxels by a face, one
/// `u32` column along z at `x + y × 32`.
fn visible_voxels(occupancy: &Occupancy) -> Vec<u32> {
    let Occupancy::Mixed(solid) = occupancy else {
        return vec![0; CHUNK * CHUNK];
    };
    let mut visible = vec![0; CHUNK * CHUNK];
    for y in 0..CHUNK {
        for x in 0..CHUNK {
            let column = solid[x + y * CHUNK];
            let mut touching = column << 1 | column >> 1;
            if x > 0 {
                touching |= solid[x - 1 + y * CHUNK];
            }
            if x + 1 < CHUNK {
                touching |= solid[x + 1 + y * CHUNK];
            }
            if y > 0 {
                touching |= solid[x + (y - 1) * CHUNK];
            }
            if y + 1 < CHUNK {
                touching |= solid[x + (y + 1) * CHUNK];
            }
            visible[x + y * CHUNK] = touching & !column;
        }
    }
    visible
}

/// Whether light changed between `old` and `new` at a voxel the chunk's own
/// faces show.
fn own_change_shows(old: &ChunkLight, new: &ChunkLight) -> bool {
    let visible = visible_voxels(&new.inputs.occupancy);
    let channels = [(&old.sky, &new.sky), (&old.voxel, &new.voxel)];
    channels
        .into_iter()
        .filter(|(old, new)| old != new)
        .any(|(old, new)| {
            visible.iter().enumerate().any(|(column, bits)| {
                let (x, y) = (column % CHUNK, column / CHUNK);
                let mut bits = *bits;
                while bits != 0 {
                    let z = bits.trailing_zeros() as usize;
                    bits &= bits - 1;
                    if old.at(x, y, z) != new.at(x, y, z) {
                        return true;
                    }
                }
                false
            })
        })
}

/// Whether light changed between `old`, `None` for none, and `new` at an
/// open voxel on `face` in front of a solid voxel of the neighbour across
/// it, whose occupancy is `solid`.
fn shell_change_shows(
    old: Option<&ChunkLight>,
    new: &ChunkLight,
    face: Face,
    solid: &Occupancy,
) -> bool {
    let level =
        |light: &ChunkLight, channel: fn(&ChunkLight) -> &ChannelLight, x, y, z| {
            channel(light).at(x, y, z)
        };
    let channels: [fn(&ChunkLight) -> &ChannelLight; 2] =
        [|light| &light.sky, |light| &light.voxel];
    (0..CHUNK).any(|v| {
        (0..CHUNK).any(|u| {
            let (x, y, z) = face.voxel(u, v);
            let (ax, ay, az) = face.opposite().voxel(u, v);
            if new.inputs.occupancy.is_solid(x, y, z) || !solid.is_solid(ax, ay, az) {
                return false;
            }
            channels.iter().any(|channel| {
                let before = old.map_or(0, |old| level(old, *channel, x, y, z));
                before != level(new, *channel, x, y, z)
            })
        })
    })
}
