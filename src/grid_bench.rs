//! Just enough of sub-grids' and fragment detection's internals for the
//! crate's benchmarks: `chunk_lifecycle` reports what fragment checks cost.
//! Not public API.

#[cfg(feature = "dislodged_fragments")]
pub use crate::dislodged_fragments::{fragment_stats, FragmentStats};
