//! Tracing a ray through one grid's voxels.

use bevy::prelude::{Aabb3d, IVec3, Ray3d, RayCast3d, Vec3};

use crate::{
    traversal_alg::voxel_line_traversal,
    voxel::{VoxelFace, WorldVoxel},
};

/// The voxel a [`trace`] stopped at.
pub(crate) struct TraceHit<I> {
    pub position: IVec3,
    /// The face the ray entered the voxel through, or `VoxelFace::None` when
    /// it is the first voxel the trace visits.
    pub face: VoxelFace,
    pub voxel: WorldVoxel<I>,
}

/// Traces `ray` through the voxels inside `bounds`, and returns the first
/// solid voxel that `accept` takes, or `None` if the ray leaves the bounds or
/// passes `max_distance` first.
///
/// A ray starting outside `bounds` is traced from where it meets them. Each
/// voxel along the ray is read with `voxel_at`; `accept` is asked about each
/// one that isn't `WorldVoxel::Unset`, so the ray passes through a voxel it
/// rejects. The cost grows with the voxels the ray passes before it stops.
pub(crate) fn trace<I: Copy + PartialEq>(
    ray: Ray3d,
    bounds: &Aabb3d,
    max_distance: f32,
    voxel_at: impl Fn(IVec3) -> WorldVoxel<I>,
    mut accept: impl FnMut(IVec3, WorldVoxel<I>) -> bool,
) -> Option<TraceHit<I>> {
    let p = ray.origin;
    let d = *ray.direction;

    let (trace_start, start_distance) = if p.cmplt(bounds.min.into()).any()
        || p.cmpgt(bounds.max.into()).any()
    {
        let t = RayCast3d::from_ray(ray, max_distance).aabb_intersection_at(bounds)?;
        (ray.get_point(t), t)
    } else {
        (p, 0.0)
    };

    let mut trace_end = bounds_exit(trace_start, d, bounds);
    let remaining = max_distance - start_distance;
    if trace_start.distance(trace_end) > remaining {
        trace_end = trace_start + d * remaining;
    }

    let mut hit = None;
    voxel_line_traversal(trace_start, trace_end, |position, _time, face| {
        let voxel = voxel_at(position);
        if voxel.is_unset() || !accept(position, voxel) || !voxel.is_solid() {
            return true;
        }
        hit = Some(TraceHit {
            position,
            face,
            voxel,
        });
        false
    });
    hit
}

/// Where a ray from `start`, a point within `bounds`, leaves them. `direction`
/// must not be zero.
///
/// The ray stops at the nearest of the faces it heads towards. Measuring from
/// `start` keeps the result precise however large the bounds are.
fn bounds_exit(start: Vec3, direction: Vec3, bounds: &Aabb3d) -> Vec3 {
    let (min, max) = (Vec3::from(bounds.min), Vec3::from(bounds.max));
    let mut distance = f32::INFINITY;
    for axis in 0..3 {
        let face = if direction[axis] > 0.0 {
            max[axis]
        } else if direction[axis] < 0.0 {
            min[axis]
        } else {
            continue;
        };
        distance = distance.min((face - start[axis]) / direction[axis]);
    }
    // Rounding can leave `start` just outside the bounds.
    start + direction * distance.max(0.0)
}
