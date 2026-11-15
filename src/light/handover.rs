//! What light does for a handover: it notes the relights the handover's
//! commit starts, tells the handover once they have solved, and holds the
//! light of the handover's chunks until it lands. Compiled only with the
//! `sub_grids` feature.
//!
//! The commit's relights are those its voxel edits trigger as they are taken
//! into light: the world's chunks within reach of the voxels it changed and
//! below the columns it opened or closed, the sub-grids' chunks within reach,
//! and a new sub-grid chunk's first solve. The relights their light starts as
//! it lands, the relight ripple, aren't waited for. A relight ripple from a
//! chunk whose light is held starts once the handover lands, rather than as
//! the light is held: a neighbour's solve reads the inputs around it, which
//! already hold the commit's voxels, so its light would otherwise show the
//! change before the handover does.
//!
//! A noted chunk counts as solved once a relight started after the commit has
//! finished, whatever then happens to its light, so a chunk edited every
//! frame, whose relights each read voxels that changed since, can't hold its
//! handover. Light that lands for a chunk the handover holds is held, as the
//! light of a chunk being re-meshed is held for its new mesh, and lands in
//! the handover's frame with the meshes. Waiting for solved rather than
//! landed light keeps the two holds from waiting on each other.

use bevy::platform::collections::HashMap;

use super::census::Cause;
use crate::grid::GridKey;

/// A relight a held chunk's light calls for: the chunk to relight, whether
/// its relight is urgent, and why.
pub(crate) type Ripple = (GridKey, bool, Cause);

/// The chunks of a world whose light handovers hold, and the relights they
/// wait for.
#[derive(Default)]
pub(crate) struct HandoverWaits {
    /// While a handover's commit is taken into light, the world's edit frame
    /// its relights start at or after.
    capturing: Option<u64>,
    /// Each chunk whose light a handover holds, with the edit frame a
    /// relight must have started at or after for the handover to stop
    /// waiting, or `None` once one has solved.
    waits: HashMap<GridKey, Option<u64>>,
    /// The chunks noted since [`take_captured`](Self::take_captured).
    captured: Vec<GridKey>,
    /// The relights the light held for each chunk calls for, started once
    /// its handover lands.
    ripples: HashMap<GridKey, Vec<Ripple>>,
}

impl HandoverWaits {
    /// Notes the relights triggered from now on as the commit's of a
    /// handover, at the world's edit frame `since`, or stops noting them.
    pub fn capture(&mut self, since: Option<u64>) {
        self.capturing = since;
    }

    /// Notes the relight of the chunk at `key`, which was just triggered, if
    /// a handover's commit is being taken in.
    pub fn triggered(&mut self, key: GridKey) {
        if let Some(since) = self.capturing {
            self.waits.insert(key, Some(since));
            self.captured.push(key);
        }
    }

    /// The chunks noted since this was last called.
    pub fn take_captured(&mut self) -> Vec<GridKey> {
        std::mem::take(&mut self.captured)
    }

    /// Holds the light of the chunk at `key` until
    /// [`release`](Self::release), without waiting for a relight of it.
    pub fn hold(&mut self, key: GridKey) {
        self.waits.entry(key).or_insert(None);
    }

    /// Stops waiting for and holding the light of the chunk at `key`, and
    /// returns the relights its held light calls for.
    pub fn release(&mut self, key: GridKey) -> Vec<Ripple> {
        self.waits.remove(&key);
        self.ripples.remove(&key).unwrap_or_default()
    }

    /// Keeps the relights `ripples` that the light held for the chunk at
    /// `key` calls for, until [`release`](Self::release).
    pub fn defer(&mut self, key: GridKey, ripples: impl IntoIterator<Item = Ripple>) {
        self.ripples.entry(key).or_default().extend(ripples);
    }

    /// Whether a handover holds the light of the chunk at `key`.
    pub fn holds(&self, key: GridKey) -> bool {
        self.waits.contains_key(&key)
    }

    /// Notes that a relight of the chunk at `key`, started at the world's
    /// edit frame `edit_frame`, finished.
    pub fn solved(&mut self, key: GridKey, edit_frame: u64) {
        if let Some(wait) = self.waits.get_mut(&key) {
            if wait.is_some_and(|since| edit_frame >= since) {
                *wait = None;
            }
        }
    }

    /// Whether the chunk at `key` waits for a relight, unless `relights`
    /// says it no longer relights.
    pub fn waits_for(&self, key: GridKey, relights: impl Fn(GridKey) -> bool) -> bool {
        self.waits.get(&key).is_some_and(Option::is_some) && relights(key)
    }

    /// Whether no handover holds any chunk's light.
    pub fn is_empty(&self) -> bool {
        self.waits.is_empty()
    }
}
