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
//! the task pool for each grid whose voxel edits that commit's local sets
//! and clears changed, ahead of the pool budget. It reads the voxels as the
//! screen shows them up close, from the voxel edits, the published level-0
//! chunks and the world's voxel lookup, and reads its
//! [`FragmentDetection`] settings as it starts. Applied edit deltas, forgets
//! and regenerating chunks start no check.
//!
//! With `sub_grids`, a check runs in each grid a commit cut, the world and
//! each sub-grid apart. Only connections within one grid count, so a
//! fragment resting on a sub-grid, or a sub-grid's piece resting on the
//! world, is still dislodged. A sub-grid is checked with no limit, so the
//! largest piece of a cut always keeps the sub-grid and a check's cost grows
//! with the sub-grid's size. A lift or a stamp starts a check like any
//! commit, from the air it leaves. A check of a sub-grid despawned, or
//! stamped away, before the check lands reports nothing.
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
//! The crate only reports, unless the world asks it to lift. A game without
//! sub-grids turns a fragment into debris by setting air wherever
//! [`FragmentDislodged::contains`] holds, for instance with
//! [`set_voxels_in`](crate::prelude::VoxelWorld::set_voxels_in) over the
//! fragment's box. With `sub_grids`, a game hands a fragment `f` to lift as
//! `voxel_world.lift(f.grid, f.min, f.max, |p, _| f.contains(p))`, and
//! `FragmentDetection::auto_lift` has the crate make that call for each
//! fragment as its check lands. In a multiplayer game, opt in only the
//! authority's world, whose edits then reach the clients as edit deltas, so
//! that clients never disagree about what fell.

use std::marker::PhantomData;
use std::sync::Arc;

use bevy::prelude::*;

use crate::configuration::VoxelWorldConfig;
#[cfg(feature = "sub_grids")]
use crate::sub_grid::Grid;

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
    ///
    /// With `sub_grids`, it bounds the world's checks only. A sub-grid is
    /// checked with no limit, so its largest piece keeps the sub-grid.
    pub limit: u32,
    /// Which neighbours hold a voxel in place: faces only by default.
    pub connectivity: Connectivity,
    /// Materials that read as air for the check, so they never hold a piece
    /// up, such as water. Empty by default.
    pub non_carrying: Vec<I>,
    /// Whether the crate lifts each fragment into a new sub-grid as its
    /// check lands: `false` by default.
    ///
    /// The lift is the handoff a game would make itself,
    /// `voxel_world.lift(f.grid, f.min, f.max, |p, _| f.contains(p))`, made
    /// in the frame the check lands, before its
    /// [`FragmentDislodged`] is written. The message names the new sub-grid
    /// in [`lifted`](FragmentDislodged::lifted), and its
    /// [`Lifted`](crate::prelude::Lifted) follows when the lift's handover
    /// lands, so a game only adds physics. Lifts from one check commit in
    /// the next `PreUpdate`, together. A lift leaves air where its fragment
    /// was, with nothing that carries beside it, so its commit dislodges
    /// nothing further.
    ///
    /// Sub-grids are checked with no limit, so a cut through a sub-grid
    /// lifts the smaller piece whatever its size, and the check's cost grows
    /// with the sub-grid's size. Only connections within one grid count, so
    /// a fragment resting on a sub-grid is still dislodged and lifted.
    #[cfg(feature = "sub_grids")]
    pub auto_lift: bool,
}

impl<I> Default for FragmentDetection<I> {
    fn default() -> Self {
        Self {
            limit: 32_768,
            connectivity: Connectivity::Faces,
            non_carrying: Vec::new(),
            #[cfg(feature = "sub_grids")]
            auto_lift: false,
        }
    }
}

/// One dislodged fragment of the voxel world `C`: voxels an edit cut off
/// from the rest of their grid, found by a check that started right after
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
    /// The grid the fragment was cut off in, whose voxel positions `min`,
    /// `max` and [`contains`](Self::contains) use.
    ///
    /// Only connections within this grid count, so a fragment resting on
    /// another grid, such as a sub-grid, is still dislodged. A sub-grid is
    /// checked with no limit: the smaller piece of a cut through it is
    /// reported whatever its size, and the check's cost grows with the
    /// sub-grid's size. A game lifts the fragment with
    /// `voxel_world.lift(f.grid, f.min, f.max, |p, _| f.contains(p))`.
    #[cfg(feature = "sub_grids")]
    pub grid: Grid,
    /// The lowest corner of the box around the fragment's voxels, inclusive.
    pub min: IVec3,
    /// The highest corner of the box around the fragment's voxels, exclusive.
    pub max: IVec3,
    /// How many voxels the fragment holds.
    pub voxel_count: u32,
    /// The sub-grid the fragment was lifted into, with
    /// [`auto_lift`](FragmentDetection::auto_lift) on: the result of
    /// `voxel_world.lift(f.grid, f.min, f.max, |p, _| f.contains(p))`, whose
    /// [`Lifted`](crate::prelude::Lifted) follows when its handover lands.
    ///
    /// `None` with `auto_lift` off, and where the lift found no solid voxel
    /// left to lift.
    #[cfg(feature = "sub_grids")]
    pub lifted: Option<Entity>,
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

    fn of(fragment: check::Fragment, #[cfg(feature = "sub_grids")] grid: Grid) -> Self {
        Self {
            #[cfg(feature = "sub_grids")]
            grid,
            #[cfg(feature = "sub_grids")]
            lifted: None,
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
            #[cfg(feature = "sub_grids")]
            grid: self.grid,
            #[cfg(feature = "sub_grids")]
            lifted: self.lifted,
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
        let mut f = f.debug_struct("FragmentDislodged");
        #[cfg(feature = "sub_grids")]
        f.field("grid", &self.grid).field("lifted", &self.lifted);
        f.field("min", &self.min)
            .field("max", &self.max)
            .field("voxel_count", &self.voxel_count)
            .finish_non_exhaustive()
    }
}

impl<C: VoxelWorldConfig> Message for FragmentDislodged<C> {}
