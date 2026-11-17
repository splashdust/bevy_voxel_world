//! Stamp: writing a sub-grid's voxels into another grid at the nearest pose
//! that lines up with that grid, and despawning the sub-grid.

use std::marker::PhantomData;
use std::sync::Arc;

use bevy::{math::I64Vec3, prelude::*};

use super::handover::Handovers;
use super::{solid_at, world, Grid, SubGrid, SubGrids};
use crate::chunk_lifecycle::ChunkRequests;
use crate::chunk_map::ChunkMap;
use crate::configuration::VoxelWorldConfig;
use crate::grid::GridId;
use crate::level_0_reader::Level0Reader;
use crate::sub_grid::snap::{snap, GridRotation, Snap};
use crate::voxel::WorldVoxel;
use crate::voxel_edits::VoxelEdits;

/// What [`stamp`] and [`preview`] read and edit: the parts of a `VoxelWorld`
/// they need.
pub(crate) struct StampParts<'a, C: VoxelWorldConfig> {
    pub sub_grids: &'a SubGrids<C>,
    pub edits: &'a VoxelEdits<C, C::MaterialIndex>,
    pub chunk_map: &'a ChunkMap<C, C::MaterialIndex>,
    pub configuration: &'a Arc<C>,
    pub handovers: &'a Handovers<C>,
    pub requests: &'a ChunkRequests<C>,
}

/// A stamp worked out from the pose snapshot, before anything is written.
struct Stamping<I> {
    sub_grid: GridId,
    target: GridId,
    target_pose: GlobalTransform,
    snap: Snap,
    /// The box holding the sub-grid's solid voxels, from `min` inclusive to
    /// `max` exclusive, or `None` for an empty sub-grid.
    bounds: Option<(IVec3, IVec3)>,
    /// Each of the sub-grid's solid voxels, where it lands and what the
    /// target holds there.
    voxels: Vec<StampVoxel<I>>,
}

/// Snaps the sub-grid `sub_grid` against `target` and reads each of its
/// solid voxels with the target's voxel where it lands. Its cost grows with
/// the box holding the sub-grid's solids, and with the target's chunk
/// positions it lands on that have no published level-0 chunk and no
/// uniform fill hint.
fn work_out<C: VoxelWorldConfig>(
    parts: &StampParts<C>,
    sub_grid: Entity,
    target: Grid,
) -> Result<Stamping<C::MaterialIndex>, StampError> {
    let (grid, pose) = parts
        .sub_grids
        .live(sub_grid)
        .ok_or(StampError::NoSuchSubGrid)?;
    let (_, target_grid, target_pose) =
        world::place(parts.sub_grids, target).ok_or(StampError::NoSuchTarget)?;
    if let Grid::SubGrid(target) = target {
        if parts.sub_grids.is_within(target, sub_grid) {
            return Err(StampError::IntoItself);
        }
    }
    // A stamped sub-grid's voxels are already on their way into another
    // grid, and it despawns once they land, with anything stamped into it.
    let stamped = |grid: Grid| match grid {
        Grid::World => false,
        Grid::SubGrid(entity) => parts.handovers.is_stamped_away(entity),
    };
    if stamped(Grid::SubGrid(sub_grid)) || stamped(target) {
        return Err(StampError::AlreadyStamped);
    }
    let bounds = parts
        .edits
        .solid_bounds_in(grid)
        .map(|(min, max)| (min, max + 1));
    // An empty sub-grid snaps about its grid origin.
    let (min, max) = bounds.unwrap_or_default();
    let snap = snap(pose.reparented_to(&target_pose), min, max);

    let incoming = parts.edits.reader_in(grid);
    let configuration = parts.configuration.clone();
    let chunk_map = parts.chunk_map.get_map();
    let mut existing = match target {
        Grid::World => Level0Reader::new(configuration, parts.edits.reader(), chunk_map),
        Grid::SubGrid(_) => Level0Reader::for_sub_grid(
            configuration,
            parts.edits.reader_in(target_grid),
            chunk_map,
        ),
    };
    let mut voxels = Vec::new();
    for z in min.z..max.z {
        for y in min.y..max.y {
            for x in min.x..max.x {
                let incoming = solid_at(&incoming, IVec3::new(x, y, z));
                if !incoming.is_solid() {
                    continue;
                }
                let position = snap.target_position(IVec3::new(x, y, z));
                voxels.push(StampVoxel {
                    position,
                    incoming,
                    existing: existing.read(position),
                    rotation: snap.rotation,
                });
            }
        }
    }
    Ok(Stamping {
        sub_grid: grid,
        target: target_grid,
        target_pose,
        snap,
        bounds,
        voxels,
    })
}

/// What stamping `sub_grid` into `target` would do: see
/// [`VoxelWorld::stamp_preview`](crate::prelude::VoxelWorld::stamp_preview).
pub(crate) fn preview<C: VoxelWorldConfig>(
    parts: StampParts<C>,
    sub_grid: Entity,
    target: Grid,
) -> Option<StampPreview> {
    let stamping = work_out(&parts, sub_grid, target).ok()?;
    let snap = &stamping.snap;
    Some(StampPreview {
        pose: stamping
            .target_pose
            .mul_transform(snap.pose)
            .compute_transform(),
        rotation: snap.rotation,
        angle: snap.angle,
        offset: snap.offset,
        scale: snap.scale,
        overlaps: stamping
            .voxels
            .iter()
            .filter(|voxel| voxel.existing.is_solid())
            .map(|voxel| voxel.position)
            .collect(),
    })
}

/// Stamps `sub_grid` into `target`, writing each voxel `rule` returns: see
/// [`VoxelWorld::stamp`](crate::prelude::VoxelWorld::stamp).
///
/// It works out every voxel first, then writes the target's voxels and
/// removes the sub-grid's, so both commit together.
pub(crate) fn stamp<C: VoxelWorldConfig>(
    parts: StampParts<C>,
    sub_grid: Entity,
    target: Grid,
    mut rule: impl FnMut(
        &StampVoxel<C::MaterialIndex>,
    ) -> Option<WorldVoxel<C::MaterialIndex>>,
) -> Result<(), StampError> {
    let stamping = work_out(&parts, sub_grid, target)?;
    if let Some((min, max)) = stamping.bounds {
        let (a, b) = (
            stamping.snap.target_position(min),
            stamping.snap.target_position(max - 1),
        );
        let (low, past_high) = (a.min(b), a.max(b) + 1);
        let size = (past_high - low).as_i64vec3();
        let index = |position: IVec3| {
            let i: I64Vec3 = (position - low).as_i64vec3();
            ((i.z * size.y + i.y) * size.x + i.x) as usize
        };
        let mut written = vec![None; (size.x * size.y * size.z) as usize];
        for voxel in &stamping.voxels {
            written[index(voxel.position)] = rule(voxel);
        }
        parts
            .edits
            .set_region_in(stamping.target, low, past_high, |position| {
                written[index(position)]
            });
        parts.edits.remove_region_in(stamping.sub_grid, min, max);
    }
    parts.requests.mark_handover();
    parts.handovers.stamped(Stamped::new(sub_grid, target));
    Ok(())
}

/// What stamping a sub-grid would do, from
/// [`VoxelWorld::stamp_preview`](crate::prelude::VoxelWorld::stamp_preview).
#[derive(Clone, Debug, PartialEq)]
pub struct StampPreview {
    /// The snapped pose against the world's root, where the stamp puts the
    /// sub-grid's voxels. A sub-grid that is a child of the world's root can
    /// take it as its `Transform` before stamping, so the snap doesn't show.
    pub pose: Transform,
    /// The rotation the snap turns the sub-grid's voxels by, against the
    /// target grid.
    pub rotation: GridRotation,
    /// The angle, in radians, between the sub-grid's rotation against the
    /// target grid and the snapped `rotation`.
    pub angle: f32,
    /// How far the snap moves the centre of the box holding the sub-grid's
    /// solid voxels, in the target grid's voxels. Each component is at most
    /// half a voxel.
    pub offset: Vec3,
    /// The sub-grid's scale against the target grid, which the snap sets
    /// to 1.
    pub scale: Vec3,
    /// Each target grid position where a solid voxel of the sub-grid lands
    /// on a solid voxel of the target, in no promised order.
    pub overlaps: Vec<IVec3>,
}

/// One voxel of a stamp, as the stamp's rule sees it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StampVoxel<I> {
    /// The target grid position the voxel lands on.
    pub position: IVec3,
    /// The sub-grid's solid voxel.
    pub incoming: WorldVoxel<I>,
    /// The target's voxel at `position` before the stamp, as the screen shows
    /// it up close: the voxel edit, else the published level-0 chunk's voxel,
    /// else its fill hint, else the world's voxel lookup. A sub-grid target
    /// reads air where it holds no solid.
    pub existing: WorldVoxel<I>,
    /// The rotation the stamp turns the sub-grid's voxels by, which a rule
    /// can turn a facing kept in the material index with.
    pub rotation: GridRotation,
}

/// The stamp rule that writes each incoming voxel over whatever the target
/// holds.
pub fn incoming_replaces<I: Copy>(voxel: &StampVoxel<I>) -> Option<WorldVoxel<I>> {
    Some(voxel.incoming)
}

/// Why [`VoxelWorld::stamp`](crate::prelude::VoxelWorld::stamp) stamped
/// nothing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StampError {
    /// The sub-grid isn't a live sub-grid of this world.
    NoSuchSubGrid,
    /// The target names an entity that isn't a live sub-grid of this world.
    NoSuchTarget,
    /// The target is the sub-grid itself or one of its descendants.
    IntoItself,
    /// The sub-grid, or the target sub-grid, was stamped already, and that
    /// stamp hasn't landed yet.
    AlreadyStamped,
}

impl std::fmt::Display for StampError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoSuchSubGrid => write!(f, "the stamped sub-grid is not live"),
            Self::NoSuchTarget => write!(f, "the stamp's target is not a live grid"),
            Self::IntoItself => {
                write!(f, "the stamp's target is the sub-grid or inside it")
            }
            Self::AlreadyStamped => {
                write!(f, "the sub-grid or the stamp's target was stamped already")
            }
        }
    }
}

impl std::error::Error for StampError {}

/// Written once per [`VoxelWorld::stamp`](crate::prelude::VoxelWorld::stamp)
/// call, in `PreUpdate` of the frame its handover lands: from then on the
/// stamped voxels show in `target`, and `sub_grid` shows none.
///
/// The sub-grid's entity still exists while systems in `Update` read this,
/// and despawns in that frame's `PostUpdate`. Its child sub-grids have
/// moved onto the target grid's entity by then, keeping their world pose, so
/// a game re-parents anything else it attached to the sub-grid here. A
/// sub-grid despawned before its handover lands gets none. Only stamps made
/// by this world's own calls write it: a world applying another world's edit
/// deltas sees a stamp land through
/// [`ChunkPublished`](crate::prelude::ChunkPublished) and
/// [`ChunkWillDespawn`](crate::prelude::ChunkWillDespawn).
pub struct Stamped<C: VoxelWorldConfig> {
    /// The sub-grid that was stamped.
    pub sub_grid: Entity,
    /// The grid it was stamped into.
    pub target: Grid,
    _world: PhantomData<fn() -> C>,
}

impl<C: VoxelWorldConfig> Stamped<C> {
    pub(crate) fn new(sub_grid: Entity, target: Grid) -> Self {
        Self {
            sub_grid,
            target,
            _world: PhantomData,
        }
    }
}

impl<C: VoxelWorldConfig> Clone for Stamped<C> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<C: VoxelWorldConfig> Copy for Stamped<C> {}

impl<C: VoxelWorldConfig> std::fmt::Debug for Stamped<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Stamped")
            .field("sub_grid", &self.sub_grid)
            .field("target", &self.target)
            .finish()
    }
}

impl<C: VoxelWorldConfig> PartialEq for Stamped<C> {
    fn eq(&self, other: &Self) -> bool {
        (self.sub_grid, self.target) == (other.sub_grid, other.target)
    }
}

impl<C: VoxelWorldConfig> Eq for Stamped<C> {}

impl<C: VoxelWorldConfig> Message for Stamped<C> {}

/// Moves each child sub-grid of a sub-grid whose stamp landed this frame
/// onto the target grid's entity, keeping its `GlobalTransform`. Runs after
/// [`Stamped`] is written. A target that is gone by then takes none, and they
/// despawn with the stamped sub-grid.
pub(crate) fn move_stamped_children<C: VoxelWorldConfig>(
    mut stamped: MessageReader<Stamped<C>>,
    sub_grids: Res<SubGrids<C>>,
    children: Query<&Children>,
    globals: Query<&GlobalTransform>,
    child_sub_grids: Query<(), With<SubGrid<C>>>,
    mut commands: Commands,
) {
    for message in stamped.read() {
        let Some((parent, _, _)) = world::place(&sub_grids, message.target) else {
            continue;
        };
        let Ok(kids) = children.get(message.sub_grid) else {
            continue;
        };
        let parent_global = globals.get(parent).copied().unwrap_or_default();
        for kid in kids.iter() {
            if !child_sub_grids.contains(kid) {
                continue;
            }
            let Ok(global) = globals.get(kid) else {
                continue;
            };
            commands
                .entity(kid)
                .try_insert((ChildOf(parent), global.reparented_to(&parent_global)));
        }
    }
}
