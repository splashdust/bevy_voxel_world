//! Dislodged fragments: the voxels an edit cuts off from the rest of a grid.
//!
//! An edit that clears a voxel, or sets it to a material that doesn't carry,
//! can leave the solids around it with nothing holding them to the rest of
//! their grid. The fragment check floods out from those voxels and reports
//! each piece that ran out of voxels before the others, while the largest
//! piece keeps its place. [`Connectivity`] says which neighbours hold a voxel
//! in place.

mod check;

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
