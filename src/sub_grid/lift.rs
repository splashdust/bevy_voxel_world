//! Lift: moving the voxels a game picks out of a grid into a new sub-grid.

use std::marker::PhantomData;
use std::sync::Arc;

use bevy::prelude::*;

use super::{world, Grid, SubGrids};
use crate::chunk_map::ChunkMap;
use crate::configuration::VoxelWorldConfig;
use crate::level_0_reader::Level0Reader;
use crate::sub_grid::handover::Handovers;
use crate::voxel::WorldVoxel;
use crate::voxel_edits::VoxelEdits;

/// What [`lift`] reads and edits: the parts of a `VoxelWorld` it needs.
pub(crate) struct LiftParts<'a, 'w, 's, C: VoxelWorldConfig> {
    pub sub_grids: &'a SubGrids<C>,
    pub edits: &'a VoxelEdits<C, C::MaterialIndex>,
    pub chunk_map: &'a ChunkMap<C, C::MaterialIndex>,
    pub configuration: &'a Arc<C>,
    pub handovers: &'a Handovers<C>,
    pub commands: &'a mut Commands<'w, 's>,
}

/// Lifts the solid voxels `pick` chooses in the box from `min` inclusive to
/// `max` exclusive out of `source` into a new sub-grid, and returns the
/// sub-grid: see [`VoxelWorld::lift`](crate::prelude::VoxelWorld::lift).
///
/// It reads the whole box first, keeping each voxel `pick` chose, then sets
/// those voxels in the new sub-grid and air over them in `source`. Its
/// memory grows with the voxels lifted, not with the box.
pub(crate) fn lift<C: VoxelWorldConfig>(
    parts: LiftParts<C>,
    source: Grid,
    min: IVec3,
    max: IVec3,
    mut pick: impl FnMut(IVec3, WorldVoxel<C::MaterialIndex>) -> bool,
) -> Result<Entity, LiftError> {
    let (parent, grid, source_pose) =
        world::place(parts.sub_grids, source).ok_or(LiftError::NoSuchGrid)?;
    let mut picked: Vec<(IVec3, WorldVoxel<C::MaterialIndex>)> = Vec::new();
    let (mut lowest, mut highest) = (IVec3::MAX, IVec3::MIN);
    {
        let chunk_map = parts.chunk_map.get_map();
        let configuration = parts.configuration.clone();
        let mut reader = match source {
            Grid::World => {
                Level0Reader::new(configuration, parts.edits.reader(), chunk_map)
            }
            Grid::SubGrid(_) => Level0Reader::for_sub_grid(
                configuration,
                parts.edits.reader_in(grid),
                chunk_map,
            ),
        };
        for z in min.z..max.z {
            for y in min.y..max.y {
                for x in min.x..max.x {
                    let position = IVec3::new(x, y, z);
                    let voxel = reader.read(position);
                    if pick(position, voxel) && voxel.is_solid() {
                        lowest = lowest.min(position);
                        highest = highest.max(position);
                        picked.push((position, voxel));
                    }
                }
            }
        }
    }
    if lowest.cmpgt(highest).any() {
        return Err(LiftError::NoSolid);
    }
    let origin = lift_origin(lowest);

    let offset = Transform::from_translation(origin.as_vec3());
    let (entity, lifted_grid) = world::spawn_under(
        parts.sub_grids,
        parts.commands,
        parent,
        offset,
        source_pose.mul_transform(offset),
    );
    let mut air: Vec<(IVec3, WorldVoxel<C::MaterialIndex>)> = picked
        .iter()
        .map(|(position, _)| (*position, WorldVoxel::Air))
        .collect();
    parts.edits.set_voxels_in(grid, &mut air);
    drop(air);
    for (position, _) in &mut picked {
        *position -= origin;
    }
    parts.edits.set_voxels_in(lifted_grid, &mut picked);
    parts.handovers.lifted(Lifted::new(source, entity));
    Ok(entity)
}

/// Why [`VoxelWorld::lift`](crate::prelude::VoxelWorld::lift) lifted
/// nothing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LiftError {
    /// The pick chose no solid voxel, so no sub-grid was spawned and no edit
    /// was made.
    NoSolid,
    /// The source names an entity that isn't a live sub-grid of this world.
    NoSuchGrid,
}

impl std::fmt::Display for LiftError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoSolid => write!(f, "the lift chose no solid voxel"),
            Self::NoSuchGrid => write!(f, "the lift's source is not a live grid"),
        }
    }
}

impl std::error::Error for LiftError {}

/// Written once per [`VoxelWorld::lift`](crate::prelude::VoxelWorld::lift)
/// call, in `PreUpdate` of the frame its handover lands: from then on the
/// lifted voxels show in `sub_grid` and no longer in `source`.
///
/// A game typically re-parents the sub-grid here, keeping its world pose,
/// and starts its physics. A sub-grid despawned before its handover lands
/// gets none. Only lifts made by this world's own calls write it: a world
/// applying another world's edit deltas sees a lift land through
/// [`ChunkPublished`](crate::prelude::ChunkPublished).
pub struct Lifted<C: VoxelWorldConfig> {
    /// The grid the voxels were lifted from.
    pub source: Grid,
    /// The sub-grid the lift spawned.
    pub sub_grid: Entity,
    _world: PhantomData<fn() -> C>,
}

impl<C: VoxelWorldConfig> Lifted<C> {
    pub(crate) fn new(source: Grid, sub_grid: Entity) -> Self {
        Self {
            source,
            sub_grid,
            _world: PhantomData,
        }
    }
}

impl<C: VoxelWorldConfig> Clone for Lifted<C> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<C: VoxelWorldConfig> Copy for Lifted<C> {}

impl<C: VoxelWorldConfig> std::fmt::Debug for Lifted<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Lifted")
            .field("source", &self.source)
            .field("sub_grid", &self.sub_grid)
            .finish()
    }
}

impl<C: VoxelWorldConfig> PartialEq for Lifted<C> {
    fn eq(&self, other: &Self) -> bool {
        (self.source, self.sub_grid) == (other.source, other.sub_grid)
    }
}

impl<C: VoxelWorldConfig> Eq for Lifted<C> {}

impl<C: VoxelWorldConfig> Message for Lifted<C> {}

/// The grid origin, in the source grid, of a sub-grid lifted with the
/// minimum corner of its solid voxels' box at `lowest`: one voxel below it on
/// each axis, so that corner sits at (1, 1, 1) in the sub-grid. A sub-grid's
/// chunks start at multiples of the chunk size, so a piece starting at
/// (1, 1, 1) spans the fewest chunks its padded voxels allow.
pub(crate) fn lift_origin(lowest: IVec3) -> IVec3 {
    lowest - 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_origin_sits_one_voxel_below_the_minimum_solid_on_each_axis() {
        assert_eq!(lift_origin(IVec3::new(2, -3, -1)), IVec3::new(1, -4, -2));
        assert_eq!(lift_origin(IVec3::ONE), IVec3::ZERO);
        assert_eq!(lift_origin(IVec3::new(-32, 0, 31)), IVec3::new(-33, -1, 30));
    }
}
