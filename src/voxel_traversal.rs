use crate::voxel::{VoxelFace, VOXEL_SIZE};
use bevy::math::{IVec3, Vec3};
use bevy::prelude::{FromReflect, Struct};

/// Traverses the voxel grid along a fixed, grid-aligned direction, applying `visit_voxel` to
/// every voxel along the way (from `start` included to `end` **excluded**).
pub fn voxel_cartesian_traversal<F: FnMut(IVec3) -> bool + Sized>(
    start: IVec3,
    end: IVec3,
    mut visit_voxel: F,
) {
    let delta = end - start;

    // Make sure one and only one component of the direction we follow is non-zero
    debug_assert!(
        delta
            .signum()
            .abs()
            .iter_fields()
            .map(|f| { i32::from_reflect(f).unwrap() })
            .sum::<i32>()
            == 1
    );

    let distance = delta.abs().max_element();
    let direction = delta / distance;

    for d in 0..distance {
        let voxel_coords = start + direction * d;
        if !visit_voxel(voxel_coords) {
            break;
        }
    }
}

/// Fast voxel traversal between arbitrary world locations. Given two world positions, visits all
/// voxels along the ray from `start` to `end` (included). If you want to traverse the grid along
/// a cardinal dimension, use voxel_cartesian_traversal instead.
///
/// Specify a function or closure for `visit_voxel`, which will get executed for every voxel
/// traversed by the ray. `visit_voxel` will be called with:
/// - The current voxel coordinates on the grid
/// - The normalized time `t` along the ray at the moment the ray intersects with the current
///   voxel (such that `IntersectionPoint = t * (end - start)`)
/// - The face through which the voxel was entered by the ray
///
/// # Example
/// ```
/// use bevy::color::palettes::css;
/// use bevy::prelude::*;
/// use bevy_voxel_world::prelude::*;
/// use bevy_voxel_world::traversal_alg::*;
///
/// fn draw_trace(trace_start: Vec3, trace_end: Vec3, mut gizmos: Gizmos) {
///     gizmos.line(trace_start, trace_end, css::RED);
///
///     voxel_line_traversal(trace_start, trace_end, |voxel_coord, time, face| {
///         let voxel_center = voxel_coord.as_vec3() + Vec3::splat(VOXEL_SIZE / 2.);
///
///         // Draw a debug cube for the currently visited voxel
///         gizmos.cuboid(
///             Transform::from_translation(voxel_center).with_scale(Vec3::splat(VOXEL_SIZE)),
///             css::PINK,
///         );
///
///         // Draw a debug dot where the ray entered this voxel
///         gizmos.sphere(
///             trace_start + (trace_end - trace_start) * time,
///             Quat::IDENTITY,
///             0.1,
///             Color::BLACK);
///
///         // If this is not the very first voxel visited (ie, the one including `start`), draw
///         // a debug circle on the face through which the trace entered the current voxel
///         if let Ok(entered_face_normal) = face.try_into() {
///             gizmos.circle(
///                 voxel_center + (entered_face_normal * VOXEL_SIZE / 2.),
///                 Dir3::new(entered_face_normal).unwrap(),
///                 0.8 * VOXEL_SIZE / 2.,
///                 css::RED.with_alpha(0.5));
///         }
///
///         // Keep drawing until trace has finished visiting all voxels along the way
///         const NEVER_STOP: bool = true;
///         NEVER_STOP
///     });
/// }
/// ```
///
/// # Note
/// This algorithm visits all voxels along a ray, using the uniform partitioned grid to determine
/// what fixed increments of time over the ray are necessary to iterate each voxel exactly once.
/// This is an implementation of J. Amanatides, A. Woo, "A Fast Voxel Traversal Algorithm for Ray
/// Tracing", accessible online at http://www.cse.yorku.ca/~amana/research/grid.pdf
pub fn voxel_line_traversal<F: FnMut(IVec3, f32, VoxelFace) -> bool + Sized>(
    start: Vec3,
    end: Vec3,
    mut visit_voxel: F,
) {
    let ray = end - start;
    let end_t = ray.length();
    let ray_dir = ray / end_t;
    let r_ray_dir = ray_dir.recip();
    let delta_t = (VOXEL_SIZE * r_ray_dir).abs();

    let step = ray_dir.signum().as_ivec3();

    let start_voxel = start.floor().as_ivec3();
    let end_voxel = end.floor().as_ivec3();

    let mut voxel = start_voxel;
    let mut max_t = Vec3::ZERO;

    max_t.x = if step.x == 0 {
        end_t
    } else {
        let o = if step.x > 0 { 1 } else { 0 };
        let plane = (start_voxel.x + o) as f32 * VOXEL_SIZE;
        (plane - start.x) * r_ray_dir.x
    };

    max_t.y = if step.y == 0 {
        end_t
    } else {
        let o = if step.y > 0 { 1 } else { 0 };
        let plane = (start_voxel.y + o) as f32 * VOXEL_SIZE;
        (plane - start.y) * r_ray_dir.y
    };

    max_t.z = if step.z == 0 {
        end_t
    } else {
        let o = if step.z > 0 { 1 } else { 0 };
        let plane = (start_voxel.z + o) as f32 * VOXEL_SIZE;
        (plane - start.z) * r_ray_dir.z
    };

    let r_end_t = 1. / end_t;
    let mut time = max_t.min_element() * r_end_t;
    let mut face = VoxelFace::None;

    let out_of_bounds = end_voxel + step;
    let mut reached_end = voxel == end_voxel;
    let mut keep_going = visit_voxel(voxel, time, face);

    let x_face = if step.x > 0 {
        VoxelFace::Left
    } else {
        VoxelFace::Right
    };
    let y_face = if step.y > 0 {
        VoxelFace::Bottom
    } else {
        VoxelFace::Top
    };
    let z_face = if step.z > 0 {
        VoxelFace::Back
    } else {
        VoxelFace::Forward
    };

    while keep_going && !reached_end {
        if max_t.x < max_t.y && max_t.x < max_t.z {
            time = max_t.x * r_end_t;
            face = x_face;

            voxel.x += step.x;
            max_t.x += delta_t.x;

            reached_end = voxel.x == out_of_bounds.x;
        } else if max_t.y < max_t.z {
            time = max_t.y * r_end_t;
            face = y_face;

            voxel.y += step.y;
            max_t.y += delta_t.y;

            reached_end = voxel.y == out_of_bounds.y;
        } else {
            time = max_t.z * r_end_t;
            face = z_face;

            voxel.z += step.z;
            max_t.z += delta_t.z;

            reached_end = voxel.z == out_of_bounds.z;
        }

        if !reached_end {
            keep_going = visit_voxel(voxel, time, face);
        }
    }
}
