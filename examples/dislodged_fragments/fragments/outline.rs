//! Outlines of the fragments reported while auto-lift is off.
//!
//! Without auto-lift the crate only reports each dislodged fragment, and the
//! fragment stays where it is. Each one is outlined from its box and
//! `contains`, with lines drawn through its grid's `pose()`, so an outline
//! on a falling piece moves with it.

use std::collections::HashSet;

use bevy::prelude::*;
use bevy_voxel_world::prelude::*;

/// How many outlines are kept. A new one replaces the oldest.
const MAX_OUTLINES: usize = 24;

const OUTLINE_COLOUR: Color = Color::srgb(1.0, 0.85, 0.2);

/// The edges of the flat faces of the voxels in the box from `min` inclusive
/// to `max` exclusive that `contains` holds, as segments one voxel long on
/// the grid, where voxel `v` fills `v` to `v + 1`. Each segment appears once.
///
/// A face shows where a voxel meets one it doesn't hold. Edges between two
/// faces that lie side by side in one plane are left out, so a flat side
/// shows as one outline rather than as a grid of squares.
pub fn outline_segments(
    min: IVec3,
    max: IVec3,
    contains: impl Fn(IVec3) -> bool,
) -> Vec<(Vec3, Vec3)> {
    let exposed =
        |voxel: IVec3, normal: IVec3| contains(voxel) && !contains(voxel + normal);
    let mut segments = HashSet::new();
    for z in min.z..max.z {
        for y in min.y..max.y {
            for x in min.x..max.x {
                let voxel = IVec3::new(x, y, z);
                if !contains(voxel) {
                    continue;
                }
                for axis in 0..3 {
                    for sign in [-1, 1] {
                        let normal = IVec3::AXES[axis] * sign;
                        if !exposed(voxel, normal) {
                            continue;
                        }
                        let mut corner = voxel;
                        corner[axis] += i32::from(sign > 0);
                        for (along, across) in [
                            ((axis + 1) % 3, (axis + 2) % 3),
                            ((axis + 2) % 3, (axis + 1) % 3),
                        ] {
                            for side in [0, 1] {
                                let beside = voxel + IVec3::AXES[across] * (2 * side - 1);
                                if exposed(beside, normal) {
                                    continue;
                                }
                                let mut start = corner;
                                start[across] += side;
                                let end = start + IVec3::AXES[along];
                                segments.insert((start, end));
                            }
                        }
                    }
                }
            }
        }
    }
    segments
        .into_iter()
        .map(|(start, end)| (start.as_vec3(), end.as_vec3()))
        .collect()
}

/// A reported fragment's outline, in its grid's voxel positions.
struct Outline {
    grid: Grid,
    /// The fragment's box and size, which tell a repeated report apart.
    key: (IVec3, IVec3, u32),
    segments: Vec<(Vec3, Vec3)>,
}

#[derive(Resource, Default)]
pub struct Outlines(Vec<Outline>);

impl Outlines {
    pub fn len(&self) -> usize {
        self.0.len()
    }
}

/// Outlines each fragment reported without a lift. Turning auto-lift on
/// clears them.
pub fn keep_outlines<C: super::FragmentsConfig>(
    mut reports: MessageReader<FragmentDislodged<C>>,
    config: Res<C>,
    mut outlines: ResMut<Outlines>,
) {
    if config.is_changed() && config.detection().auto_lift {
        outlines.0.clear();
    }
    for fragment in reports.read() {
        if fragment.lifted.is_some() {
            continue;
        }
        let key = (fragment.min, fragment.max, fragment.voxel_count);
        if outlines
            .0
            .iter()
            .any(|outline| outline.grid == fragment.grid && outline.key == key)
        {
            continue;
        }
        if outlines.0.len() == MAX_OUTLINES {
            outlines.0.remove(0);
        }
        outlines.0.push(Outline {
            grid: fragment.grid,
            key,
            segments: outline_segments(fragment.min, fragment.max, |voxel| {
                fragment.contains(voxel)
            }),
        });
    }
}

/// Draws each outline through its grid's pose, and drops those whose grid
/// is gone.
pub fn draw_outlines<C: VoxelWorldConfig>(
    voxel_world: VoxelWorld<C>,
    mut outlines: ResMut<Outlines>,
    mut gizmos: Gizmos,
) {
    outlines.0.retain(|outline| {
        let Some(pose) = voxel_world.grid(outline.grid).map(|grid| grid.pose()) else {
            return false;
        };
        for &(start, end) in &outline.segments {
            gizmos.line(
                pose.transform_point(start),
                pose.transform_point(end),
                OUTLINE_COLOUR,
            );
        }
        true
    });
}
