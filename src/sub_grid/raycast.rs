//! Raycasts across every grid of a world, and the query for the sub-grids
//! holding a point.
//!
//! A sub-grid is traced in its own coordinates: the ray is carried into its
//! grid by the inverse of its pose, meets the box around its solids there,
//! and is traced through its voxels as the world's grid is. The distance and
//! entry face come from the hit voxel's own box, not from the trace, so they
//! hold under a scaled pose and wherever the trace started.

use std::{
    borrow::Borrow,
    sync::{Arc, RwLock},
};

use bevy::{math::Affine3A, prelude::*};

use crate::voxel_world::{raycast::trace, voxel_at, VoxelWorld};
use crate::{
    chunk_map::{ChunkMap, ChunkMapData},
    configuration::VoxelWorldConfig,
    grid::GridId,
    sub_grid::{solid_at, Grid},
    voxel::WorldVoxel,
    voxel_edits::{VoxelEditReader, VoxelEdits},
};

/// A ray's hit on a voxel of one of a world's grids, from
/// [`VoxelWorld::raycast_grids`].
///
/// `position` and `normal` are on the hit voxel's grid, so
/// [`grid_mut(hit.grid)`](VoxelWorld::grid_mut) edits the voxel at
/// `position`, and the voxel beside the face is at `position + normal`.
/// `point`, `distance` and `world_normal` are in the frame of the world's
/// root, the frame a sub-grid's [`pose`](crate::prelude::GridVoxels::pose)
/// is against.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GridRaycastHit<I = u8> {
    /// The grid holding the voxel.
    pub grid: Grid,
    /// The voxel's position on its grid.
    pub position: IVec3,
    /// The normal of the face the ray entered the voxel through, on the
    /// voxel's grid, or `None` when the ray starts inside the voxel.
    pub normal: Option<IVec3>,
    /// The same face's normal in the root's frame, of length 1: `normal`
    /// turned by the grid's pose.
    pub world_normal: Option<Vec3>,
    /// Where the ray met the voxel, in the root's frame: the ray's origin
    /// when it starts inside the voxel.
    pub point: Vec3,
    /// The distance from the ray's origin to `point`.
    pub distance: f32,
    /// The voxel the ray hit.
    pub voxel: WorldVoxel<I>,
}

/// The closure [`VoxelWorld::raycast_grids_fn`] returns. It takes what
/// [`VoxelWorld::raycast_grids`] takes.
pub type GridRaycastFn<I> = dyn Fn(
        Ray3d,
        f32,
        &dyn Fn(Grid) -> bool,
        &dyn Fn(Grid, IVec3, WorldVoxel<I>) -> bool,
    ) -> Option<GridRaycastHit<I>>
    + Send
    + Sync;

/// The voxel filter of a grid raycast.
type VoxelFilter<'a, I> = &'a dyn Fn(Grid, IVec3, WorldVoxel<I>) -> bool;

impl<C: VoxelWorldConfig> VoxelWorld<'_, '_, C> {
    /// Casts `ray` across the world and every sub-grid, and returns the
    /// nearest hit within `max_distance` of the ray's origin, or `None`.
    ///
    /// The ray is in the frame of the world's root, which is world space
    /// while the root sits at the identity. Each sub-grid is hit where its
    /// [`pose`](crate::prelude::GridVoxels::pose) puts it, which is the pose
    /// shown last frame: a `Transform` changed this frame shows here from
    /// the next frame, and a sub-grid spawned this frame is hit at the
    /// transform it was given. The world's voxels are read as
    /// [`raycast`](Self::raycast) reads them, and a sub-grid's as
    /// [`grid(g).get_voxel`](crate::prelude::GridVoxels::get_voxel) does,
    /// voxels set this frame included.
    ///
    /// `grid_filter` is called once for each grid, the world included,
    /// before that grid is traced, and a grid it rejects is skipped.
    /// `voxel_filter` is called with the grid, the position on that grid
    /// and the voxel for each solid voxel the ray meets, and the ray passes
    /// through a voxel it rejects. When the world and a sub-grid are hit at
    /// exactly the same distance, the hit names the sub-grid; between two
    /// sub-grids no winner is promised.
    ///
    /// Rays read voxel edits before published chunk data, while the screen
    /// shows only published chunks. So during a stamp's handover, a ray hits
    /// the stamped voxels at their snapped pose in the few frames before
    /// they show.
    ///
    /// The cost grows with the live sub-grids, and with the voxels the ray
    /// passes in each grid it reaches before the nearest hit.
    pub fn raycast_grids(
        &self,
        ray: Ray3d,
        max_distance: f32,
        grid_filter: &impl Fn(Grid) -> bool,
        voxel_filter: &impl Fn(Grid, IVec3, WorldVoxel<C::MaterialIndex>) -> bool,
    ) -> Option<GridRaycastHit<C::MaterialIndex>> {
        let sub_grids =
            self.sub_grids
                .live_grids()
                .into_iter()
                .map(|(entity, grid, pose)| {
                    (entity, move || {
                        SubGridRay::new(grid, pose, &self.voxel_edits)
                    })
                });
        nearest(
            ray,
            max_distance,
            grid_filter,
            voxel_filter,
            &self.world_ray(),
            sub_grids,
        )
    }

    /// A sendable closure that casts rays as
    /// [`raycast_grids`](Self::raycast_grids) does, for tasks off the main
    /// thread.
    ///
    /// It keeps the sub-grids live at the call, with their poses and the
    /// boxes around their solids. It sees the voxel edits set before it was
    /// made, and committed edits and published chunks as they are when it
    /// casts. Making it costs as much as finding the bounds of every live
    /// sub-grid.
    pub fn raycast_grids_fn(&self) -> Arc<GridRaycastFn<C::MaterialIndex>> {
        let world = self.world_ray();
        let sub_grids: Vec<_> = self
            .sub_grids
            .live_grids()
            .into_iter()
            .map(|(entity, grid, pose)| {
                (entity, SubGridRay::new(grid, pose, &self.voxel_edits))
            })
            .collect();
        Arc::new(move |ray, max_distance, grid_filter, voxel_filter| {
            let sub_grids = sub_grids
                .iter()
                .map(|(entity, sub_grid)| (*entity, move || sub_grid));
            nearest(
                ray,
                max_distance,
                grid_filter,
                voxel_filter,
                &world,
                sub_grids,
            )
        })
    }

    /// The live sub-grids whose [`bounds`](crate::prelude::GridVoxels::bounds)
    /// box encloses `point`, a point in the frame of the world's root, in no
    /// promised order.
    ///
    /// Each box is placed by the sub-grid's
    /// [`pose`](crate::prelude::GridVoxels::pose), the pose shown last frame.
    /// A point in air inside the box counts, as in a cabin's room, and so
    /// does one on its faces. A sub-grid with no solid voxel has no box and
    /// is never listed. The cost grows with the live sub-grids and the work
    /// of finding each one's bounds.
    pub fn sub_grids_at_point(&self, point: Vec3) -> impl Iterator<Item = Entity> {
        let found: Vec<Entity> = self
            .sub_grids
            .live_grids()
            .into_iter()
            .filter(|(_, grid, pose)| {
                let Some((min, max)) = self.voxel_edits.solid_bounds_in(*grid) else {
                    return false;
                };
                let local = pose.affine().inverse().transform_point3(point);
                local.cmpge(min.as_vec3()).all() && local.cmple((max + 1).as_vec3()).all()
            })
            .map(|(entity, _, _)| entity)
            .collect();
        found.into_iter()
    }

    /// What a grid raycast reads of the world's grid, taken now.
    fn world_ray(&self) -> WorldRay<C> {
        WorldRay {
            edits: self.voxel_edits.reader(),
            chunk_map: self.chunk_map.get_map(),
            configuration: self.configuration.0.clone(),
        }
    }
}

/// What a grid raycast reads of the world's grid.
struct WorldRay<C: VoxelWorldConfig> {
    edits: VoxelEditReader<C::MaterialIndex>,
    chunk_map: Arc<RwLock<ChunkMapData<C::MaterialIndex>>>,
    configuration: Arc<C>,
}

impl<C: VoxelWorldConfig> WorldRay<C> {
    /// The world's nearest hit within `max_distance`, traced through the
    /// published chunks' bounds as [`VoxelWorld::raycast`] traces.
    fn cast(
        &self,
        ray: Ray3d,
        max_distance: f32,
        voxel_filter: VoxelFilter<C::MaterialIndex>,
    ) -> Option<GridRaycastHit<C::MaterialIndex>> {
        let bounds = ChunkMap::<C, C::MaterialIndex>::get_world_bounds(
            &self.chunk_map.read().unwrap(),
        );
        cast_grid(
            Grid::World,
            &Affine3A::IDENTITY,
            &bounds,
            ray,
            max_distance,
            |position| {
                voxel_at(
                    &self.edits,
                    &self.chunk_map,
                    self.configuration.as_ref(),
                    position,
                )
            },
            voxel_filter,
        )
    }
}

/// What a grid raycast reads of one sub-grid.
struct SubGridRay<I> {
    /// The sub-grid's pose against the world's root.
    pose: Affine3A,
    /// The box around its solids, one voxel wider on every side, or `None`
    /// when it has none.
    bounds: Option<Aabb3d>,
    edits: VoxelEditReader<I>,
}

impl<I: Copy + PartialEq> SubGridRay<I> {
    fn new<C>(grid: GridId, pose: GlobalTransform, edits: &VoxelEdits<C, I>) -> Self {
        // The wider box puts the trace's first voxel in air, so a trace that
        // starts where the ray meets the box never starts on a solid.
        let bounds = edits.solid_bounds_in(grid).map(|(min, max)| Aabb3d {
            min: (min - 1).as_vec3a(),
            max: (max + 2).as_vec3a(),
        });
        Self {
            pose: pose.affine(),
            bounds,
            edits: edits.reader_in(grid),
        }
    }

    fn cast(
        &self,
        entity: Entity,
        ray: Ray3d,
        max_distance: f32,
        voxel_filter: VoxelFilter<I>,
    ) -> Option<GridRaycastHit<I>> {
        cast_grid(
            Grid::SubGrid(entity),
            &self.pose,
            self.bounds.as_ref()?,
            ray,
            max_distance,
            |position| solid_at(&self.edits, position),
            voxel_filter,
        )
    }
}

/// The nearest hit within `max_distance` across the world and `sub_grids`,
/// whose rays are made only for the sub-grids `grid_filter` takes.
///
/// The world is traced first, and each sub-grid only as far as the nearest
/// hit so far, which a sub-grid hit at the same distance replaces.
fn nearest<C, F, S>(
    ray: Ray3d,
    max_distance: f32,
    grid_filter: &dyn Fn(Grid) -> bool,
    voxel_filter: VoxelFilter<C::MaterialIndex>,
    world: &WorldRay<C>,
    sub_grids: impl Iterator<Item = (Entity, F)>,
) -> Option<GridRaycastHit<C::MaterialIndex>>
where
    C: VoxelWorldConfig,
    F: FnOnce() -> S,
    S: Borrow<SubGridRay<C::MaterialIndex>>,
{
    let mut nearest = None;
    if grid_filter(Grid::World) {
        nearest = world.cast(ray, max_distance, voxel_filter);
    }
    for (entity, sub_grid) in sub_grids {
        if !grid_filter(Grid::SubGrid(entity)) {
            continue;
        }
        let reach = nearest
            .as_ref()
            .map_or(max_distance, |hit: &GridRaycastHit<_>| hit.distance);
        if let Some(hit) = sub_grid().borrow().cast(entity, ray, reach, voxel_filter) {
            nearest = Some(hit);
        }
    }
    nearest
}

/// Casts `ray`, in the root's frame, through the voxels `voxel_at` reads on
/// the grid at `pose`, inside `bounds` on that grid. Returns the first solid
/// voxel `voxel_filter` takes, if the ray meets it within `max_distance`.
fn cast_grid<I: Copy + PartialEq>(
    grid: Grid,
    pose: &Affine3A,
    bounds: &Aabb3d,
    ray: Ray3d,
    max_distance: f32,
    voxel_at: impl Fn(IVec3) -> WorldVoxel<I>,
    voxel_filter: VoxelFilter<I>,
) -> Option<GridRaycastHit<I>> {
    let from_root = pose.inverse();
    let origin = from_root.transform_point3(ray.origin);
    // How far the ray moves on the grid for each unit it moves in the root's
    // frame, so a distance along it is a distance in the root's frame.
    let step = from_root.transform_vector3(*ray.direction);
    let on_grid = Ray3d::new(origin, Dir3::new(step).ok()?);
    // The trace runs a voxel past the limit, since rounding in the change of
    // frame could otherwise cut off a voxel met at the limit; the voxel's
    // own distance is checked against the limit below.
    let hit = trace(
        on_grid,
        bounds,
        max_distance * step.length() + 1.0,
        voxel_at,
        |position, voxel| voxel.is_solid() && voxel_filter(grid, position, voxel),
    )?;
    let (distance, normal) = entry(origin, step, hit.position);
    if distance > max_distance {
        return None;
    }
    let turn = pose.matrix3.inverse().transpose();
    Some(GridRaycastHit {
        grid,
        position: hit.position,
        normal,
        world_normal: normal.map(|normal| (turn * normal.as_vec3()).normalize()),
        point: ray.get_point(distance),
        distance,
        voxel: hit.voxel,
    })
}

/// How far a ray from `origin`, moving by `step` per unit of distance, goes
/// before it enters the voxel at `position`, and the normal of the face it
/// enters through. A ray starting inside the voxel goes 0 and enters through
/// no face.
///
/// The ray enters the voxel's box where it has crossed the near plane of
/// every axis, so at the last of those crossings.
fn entry(origin: Vec3, step: Vec3, position: IVec3) -> (f32, Option<IVec3>) {
    let low = position.as_vec3();
    let mut last: Option<(f32, IVec3)> = None;
    for (axis, normal) in IVec3::AXES.into_iter().enumerate() {
        let (plane, normal) = if step[axis] > 0.0 {
            (low[axis], -normal)
        } else if step[axis] < 0.0 {
            (low[axis] + 1.0, normal)
        } else {
            continue;
        };
        let distance = (plane - origin[axis]) / step[axis];
        if last.is_none_or(|(furthest, _)| distance > furthest) {
            last = Some((distance, normal));
        }
    }
    match last {
        Some((distance, normal)) if distance >= 0.0 => (distance, Some(normal)),
        _ => (0.0, None),
    }
}
