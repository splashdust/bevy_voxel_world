//! Just enough of sub-grids' and fragment detection's internals for the
//! crate's benchmarks: `chunk_lifecycle` reports what fragment checks and
//! the tag map cost. Not public API.

#[cfg(feature = "dislodged_fragments")]
pub use crate::dislodged_fragments::{fragment_stats, FragmentStats};

#[cfg(all(feature = "sub_grids", feature = "light_propagation"))]
pub use crate::light::tag_map::{tag_map_stats, TagMapStats};
