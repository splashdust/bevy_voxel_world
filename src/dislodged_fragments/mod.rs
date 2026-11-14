//! Dislodged fragments: the voxels an edit cuts off from the rest of a grid.
//!
//! An edit that clears a voxel, or sets it to a material that doesn't carry,
//! can leave the solids around it with nothing holding them to the rest of
//! their grid. The fragment check floods out from those voxels and reports
//! each piece that ran out of voxels before the others, while the largest
//! piece keeps its place. [`Connectivity`] says which neighbours hold a voxel
//! in place.
//!
//! A world opts in through
//! [`detect_dislodged_fragments`](crate::prelude::VoxelWorldConfig::detect_dislodged_fragments),
//! read once at plugin build. A world that doesn't opt in gets no detection
//! systems. In one that does, right after each commit, one check starts on
//! the task pool for the world's voxel edits that commit's local sets and
//! clears changed, ahead of the pool budget. It reads the voxels as the
//! screen shows them up close, from the voxel edits, the published level-0
//! chunks and the world's voxel lookup, and reads its
//! [`FragmentDetection`] settings as it starts. Applied edit deltas, forgets
//! and regenerating chunks start no check.
//!
//! A check lands in a later frame, never in the frame it started. If a
//! commit since then changed the voxel edits of a chunk the check read from,
//! its result may be out of date, so it runs again with the newest commit's
//! check, their seeds together. That way a support placed before the check
//! lands keeps the piece up. A region edited in every frame holds its check
//! back until the edits stop. A forget drops the checks still running.
//! Otherwise each dislodged fragment arrives as a [`FragmentDislodged`]
//! message.
//!
//! A check's cost grows with the [`limit`](FragmentDetection::limit), which
//! bounds how far it floods into anything still attached, and with how many
//! separate pieces a commit's digs touch.
//!
//! The crate only reports. A game without sub-grids turns a fragment into
//! debris by setting air wherever [`FragmentDislodged::contains`] holds, for
//! instance with
//! [`set_voxels_in`](crate::prelude::VoxelWorld::set_voxels_in) over the
//! fragment's box. In a multiplayer game, opt in only the authority's world,
//! whose edits then reach the clients as edit deltas, so that clients never
//! disagree about what fell.

use std::marker::PhantomData;
use std::sync::Arc;

use bevy::prelude::*;

use crate::configuration::VoxelWorldConfig;

mod check;
mod detection;

pub(crate) use detection::build;
#[cfg(test)]
pub(crate) use detection::FragmentChecks;
pub use detection::{fragment_stats, FragmentStats};

/// Which of a voxel's neighbours hold it to the rest of its grid.
///
/// Two solids that touch this way belong to the same piece, so a piece held
/// up by a neighbour of either belongs with it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Connectivity {
    /// The 6 voxels sharing a face.
    #[default]
    Faces,
    /// The 18 voxels sharing a face or an edge.
    Edges,
    /// All 26 voxels around it, sharing a face, an edge or a corner.
    Corners,
}

/// How a world finds dislodged fragments, returned by
/// [`detect_dislodged_fragments`](crate::prelude::VoxelWorldConfig::detect_dislodged_fragments).
/// `I` is the world's material index.
///
/// Each check reads these afresh, so they can change while the game runs.
/// Build it with `..default()`, since later versions may add settings.
///
/// A check's cost grows with `limit`, and with how many separate pieces a
/// commit's digs touch. A region edited in every frame holds its check back
/// until the edits stop. In a multiplayer game, only the authority's world
/// should detect fragments.
#[derive(Clone, Debug, PartialEq)]
pub struct FragmentDetection<I> {
    /// How many voxels a flood visits before its piece counts as attached:
    /// 32,768 by default, one chunk's worth.
    ///
    /// A dig into a large mass stops after about this many voxels, and a
    /// piece larger than it is never reported. A larger limit finds larger
    /// fragments and costs more on each dig into anything attached.
    pub limit: u32,
    /// Which neighbours hold a voxel in place: faces only by default.
    pub connectivity: Connectivity,
    /// Materials that read as air for the check, so they never hold a piece
    /// up, such as water. Empty by default.
    pub non_carrying: Vec<I>,
}

impl<I> Default for FragmentDetection<I> {
    fn default() -> Self {
        Self {
            limit: 32_768,
            connectivity: Connectivity::Faces,
            non_carrying: Vec::new(),
        }
    }
}

/// One dislodged fragment of the voxel world `C`: voxels an edit cut off
/// from the rest of the world, found by a check that started right after
/// the edit's commit and landed a few frames later.
///
/// It holds only solids, none of a
/// [`non_carrying`](FragmentDetection::non_carrying) material, and
/// [`contains`](Self::contains) tells which. A game without
/// sub-grids turns it into debris by setting air wherever `contains` holds.
///
/// It is written in `PreUpdate`, in [`VoxelWorldSet`], so systems running
/// after the set read it in the frame its check lands.
///
/// [`VoxelWorldSet`]: crate::prelude::VoxelWorldSet
pub struct FragmentDislodged<C: VoxelWorldConfig> {
    /// The lowest corner of the box around the fragment's voxels, inclusive.
    pub min: IVec3,
    /// The highest corner of the box around the fragment's voxels, exclusive.
    pub max: IVec3,
    /// How many voxels the fragment holds.
    pub voxel_count: u32,
    /// The fragment's voxel positions, sorted by `check::position_order`.
    voxels: Arc<[IVec3]>,
    _world: PhantomData<C>,
}

impl<C: VoxelWorldConfig> FragmentDislodged<C> {
    /// Whether the voxel at `position` belongs to the fragment.
    pub fn contains(&self, position: IVec3) -> bool {
        position.cmpge(self.min).all()
            && position.cmplt(self.max).all()
            && self
                .voxels
                .binary_search_by_key(&check::position_order(position), |voxel| {
                    check::position_order(*voxel)
                })
                .is_ok()
    }

    fn of(fragment: check::Fragment) -> Self {
        Self {
            min: fragment.min,
            max: fragment.max,
            voxel_count: fragment.voxel_count,
            voxels: fragment.voxels.into(),
            _world: PhantomData,
        }
    }
}

impl<C: VoxelWorldConfig> Clone for FragmentDislodged<C> {
    fn clone(&self) -> Self {
        Self {
            min: self.min,
            max: self.max,
            voxel_count: self.voxel_count,
            voxels: self.voxels.clone(),
            _world: PhantomData,
        }
    }
}

impl<C: VoxelWorldConfig> std::fmt::Debug for FragmentDislodged<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FragmentDislodged")
            .field("min", &self.min)
            .field("max", &self.max)
            .field("voxel_count", &self.voxel_count)
            .finish_non_exhaustive()
    }
}

impl<C: VoxelWorldConfig> Message for FragmentDislodged<C> {}
