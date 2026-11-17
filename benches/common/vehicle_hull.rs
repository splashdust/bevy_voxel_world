//! The hull `chunk_lifecycle`'s vehicle scenarios fly: an airship of voxels
//! built in a sub-grid by voxel edits.
//!
//! Its envelope is a hollow ellipsoid, long along z, with walls
//! [`WALL`] voxels thick. A gondola hangs under it: a box with walls as
//! thick, joined to the envelope through its roof, whose cabin is closed
//! but for a door-sized opening in one side wall. Every voxel comes from
//! [`hull_voxel`], so the hull is the same in every run.

use std::collections::HashSet;

use bevy::prelude::*;
use bevy_voxel_world::custom_meshing::CHUNK_SIZE_I;
use bevy_voxel_world::prelude::*;

/// The material of the envelope's voxels.
pub const ENVELOPE: u8 = 0;

/// The material of the gondola's voxels.
pub const GONDOLA: u8 = 1;

/// How thick the envelope's and the gondola's walls are, in voxels.
pub const WALL: i32 = 3;

/// The lowest corner of the box holding every voxel of the hull, inclusive.
pub const MIN: IVec3 = IVec3::ZERO;

/// The highest corner of the box holding every voxel of the hull,
/// exclusive. The box is 3 by 4 by 8 chunks, and no voxel of the hull
/// touches its faces, so the hull's chunks all lie inside it.
pub const MAX: IVec3 = IVec3::new(96, 128, 256);

/// The centre of the envelope on the hull's grid, which the scenarios turn
/// the hull about.
pub const CENTRE: Vec3 = Vec3::new(48.0, 80.0, 128.0);

/// The envelope's outer half-widths along x, y and z.
const RADII: Vec3 = Vec3::new(46.0, 46.0, 126.0);

/// The gondola's lowest corner, inclusive. Its roof reaches into the
/// envelope's lowest wall, which joins the two.
const GONDOLA_MIN: IVec3 = IVec3::new(36, 12, 104);

/// The gondola's highest corner, exclusive.
const GONDOLA_MAX: IVec3 = IVec3::new(60, 40, 152);

/// The y the cabin's inside reaches up to, exclusive. The gondola is solid
/// from here up to `GONDOLA_MAX`.
const CABIN_TOP: i32 = 34;

/// The door through the gondola's wall at its lowest x, from its lowest
/// corner, inclusive, to its highest, exclusive: 3 voxels wide and 4 high,
/// standing on the cabin's floor.
const DOOR_MIN: IVec3 = IVec3::new(36, 15, 126);
const DOOR_MAX: IVec3 = IVec3::new(39, 19, 129);

/// Where lamp voxels go when [`build`] is given lamps: under the cabin's
/// roof, towards each end of the cabin.
pub const LAMPS: [IVec3; 2] = [IVec3::new(48, 33, 116), IVec3::new(48, 33, 140)];

/// What [`build`] put in the sub-grid.
pub struct Hull {
    /// How many solid voxels the hull holds, lamps included.
    pub solid_voxels: u32,
    /// The positions of the level-0 chunks the sub-grid should have once the
    /// hull commits: those whose voxels, with one voxel of padding around
    /// them, hold one of the hull's solid voxels.
    pub chunks: HashSet<IVec3>,
}

/// The material of the hull's voxel at `position`, or `None` where the hull
/// holds air. With `lamp`, each voxel at [`LAMPS`] holds the material `lamp`
/// gives for its position.
pub fn hull_voxel(position: IVec3, lamp: Option<fn(IVec3) -> u8>) -> Option<u8> {
    if let Some(lamp) = lamp.filter(|_| LAMPS.contains(&position)) {
        return Some(lamp(position));
    }
    let centre = position.as_vec3() + 0.5 - CENTRE;
    let inside = |radii: Vec3| (centre / radii).length_squared() <= 1.0;
    if inside(RADII) && !inside(RADII - WALL as f32) {
        return Some(ENVELOPE);
    }
    let within =
        |min: IVec3, max: IVec3| position.cmpge(min).all() && position.cmplt(max).all();
    let cabin_min = GONDOLA_MIN + WALL;
    let cabin_max = IVec3::new(GONDOLA_MAX.x - WALL, CABIN_TOP, GONDOLA_MAX.z - WALL);
    (within(GONDOLA_MIN, GONDOLA_MAX)
        && !within(cabin_min, cabin_max)
        && !within(DOOR_MIN, DOOR_MAX))
    .then_some(GONDOLA)
}

/// Fills `grid` with the hull, placing lamp voxels in the cabin of the
/// materials `lamp` gives when it is given, and returns what it put there.
///
/// It sets voxels and removes none, so `grid` is best an empty sub-grid.
/// The voxels commit in the next `PreUpdate`.
pub fn build<C: VoxelWorldConfig<MaterialIndex = u8>>(
    grid: &mut GridVoxelsMut<C>,
    lamp: Option<fn(IVec3) -> u8>,
) -> Hull {
    grid.set_voxels_in(MIN, MAX, |position| {
        hull_voxel(position, lamp).map(WorldVoxel::Solid)
    });
    let mut hull = Hull {
        solid_voxels: 0,
        chunks: HashSet::new(),
    };
    for x in MIN.x..MAX.x {
        for y in MIN.y..MAX.y {
            for z in MIN.z..MAX.z {
                let position = IVec3::new(x, y, z);
                if hull_voxel(position, lamp).is_some() {
                    hull.solid_voxels += 1;
                    hull.chunks.extend(chunks_holding(position));
                }
            }
        }
    }
    hull
}

/// The positions of the level-0 chunks whose voxels, with one voxel of
/// padding around them, include `position`.
pub fn chunks_holding(position: IVec3) -> HashSet<IVec3> {
    if !on_chunk_face(position) {
        return HashSet::from([chunk_of(position)]);
    }
    let around = (-1..=1).flat_map(|x| {
        (-1..=1).flat_map(move |y| (-1..=1).map(move |z| IVec3::new(x, y, z)))
    });
    around.map(|offset| chunk_of(position + offset)).collect()
}

/// The position of the level-0 chunk whose own voxels include `position`.
pub fn chunk_of(position: IVec3) -> IVec3 {
    position.div_euclid(IVec3::splat(CHUNK_SIZE_I))
}

/// Whether `position` is in the outermost layer of its chunk's voxels,
/// which a neighbouring chunk holds as padding.
fn on_chunk_face(position: IVec3) -> bool {
    let offset = position.rem_euclid(IVec3::splat(CHUNK_SIZE_I));
    offset.cmpeq(IVec3::ZERO).any() || offset.cmpeq(IVec3::splat(CHUNK_SIZE_I - 1)).any()
}

/// One voxel on the hull's surface in each of its chunks that has one,
/// held by that chunk alone, so an edit at one of them changes one chunk.
///
/// Each is the first solid voxel with an air neighbour found in its chunk,
/// scanning from the top of the chunk down, outside the outermost layer of
/// the chunk's voxels. There is an odd number of them, so a scenario that
/// visits them in turn, setting on even frames and digging on odd ones,
/// alternates between setting and digging each one.
pub fn edit_sites() -> Vec<IVec3> {
    let faces = [
        IVec3::X,
        IVec3::NEG_X,
        IVec3::Y,
        IVec3::NEG_Y,
        IVec3::Z,
        IVec3::NEG_Z,
    ];
    let on_surface = |position: IVec3| {
        hull_voxel(position, None).is_some()
            && faces
                .iter()
                .any(|face| hull_voxel(position + *face, None).is_none())
    };
    let mut sites = Vec::new();
    for chunk_z in chunk_of(MIN).z..chunk_of(MAX).z {
        for chunk_y in chunk_of(MIN).y..chunk_of(MAX).y {
            for chunk_x in chunk_of(MIN).x..chunk_of(MAX).x {
                let min = IVec3::new(chunk_x, chunk_y, chunk_z) * CHUNK_SIZE_I + 1;
                let max = min + CHUNK_SIZE_I - 2;
                let site = (min.y..max.y).rev().find_map(|y| {
                    (min.x..max.x).find_map(|x| {
                        (min.z..max.z)
                            .map(|z| IVec3::new(x, y, z))
                            .find(|position| on_surface(*position))
                    })
                });
                sites.extend(site);
            }
        }
    }
    if sites.len().is_multiple_of(2) {
        sites.pop();
    }
    sites
}
