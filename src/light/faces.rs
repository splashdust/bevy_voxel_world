//! What a chunk's light shows its six face neighbours: a hash per face of when
//! light reached it, and the comparison an arriving chunk makes with each
//! neighbour's light.

use bevy::prelude::*;

use super::levels::LevelScale;
use super::solver::{ChannelLight, ChunkLight};
use crate::chunk::CHUNK_SIZE_U;

const CHUNK: usize = CHUNK_SIZE_U as usize;

/// One of a chunk's six faces, named by the direction it faces.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Face {
    NegX,
    PosX,
    NegY,
    PosY,
    NegZ,
    PosZ,
}

impl Face {
    pub const ALL: [Face; 6] = [
        Face::NegX,
        Face::PosX,
        Face::NegY,
        Face::PosY,
        Face::NegZ,
        Face::PosZ,
    ];

    /// The step from a chunk to the neighbour across this face.
    pub fn offset(self) -> IVec3 {
        match self {
            Face::NegX => IVec3::NEG_X,
            Face::PosX => IVec3::X,
            Face::NegY => IVec3::NEG_Y,
            Face::PosY => IVec3::Y,
            Face::NegZ => IVec3::NEG_Z,
            Face::PosZ => IVec3::Z,
        }
    }

    /// The neighbour's face that touches this one.
    pub fn opposite(self) -> Face {
        Face::ALL[self as usize ^ 1]
    }

    /// The chunk's voxel at `u` and `v` on this face, `u` and `v` running
    /// along the other two axes in x, y, z order. A face and its opposite
    /// give touching voxels the same `u` and `v`.
    pub fn voxel(self, u: usize, v: usize) -> (usize, usize, usize) {
        let far = self as usize & 1 == 1;
        let side = if far { CHUNK - 1 } else { 0 };
        match self {
            Face::NegX | Face::PosX => (side, u, v),
            Face::NegY | Face::PosY => (u, side, v),
            Face::NegZ | Face::PosZ => (u, v, side),
        }
    }
}

/// Mixes `value` into `hash`.
#[inline]
pub(crate) fn mix(hash: u64, value: u64) -> u64 {
    (hash.rotate_left(5) ^ value).wrapping_mul(0x517c_c1b7_2722_0a95)
}

/// Per channel and per face, a hash of when light reached each of a chunk's
/// voxels on that face: 96 B a chunk.
///
/// A neighbour's light can only change with a chunk's light where light
/// crosses their shared face at another time, so a relight that leaves a
/// face's hash alone needn't relight the neighbour across it. Round spread
/// hashes passes too, since in a round of three passes, along x, then y, then
/// z, a voxel reached in the pass along x still sends light across a Y face in
/// the same round, and one reached by the pass along y across a Z face.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct FaceHashes {
    pub sky: [u64; 6],
    pub voxel: [u64; 6],
}

impl FaceHashes {
    /// Whether light crosses `face` at another time in `other`.
    pub fn differs_at(&self, other: &FaceHashes, face: Face) -> bool {
        let face = face as usize;
        self.sky[face] != other.sky[face] || self.voxel[face] != other.voxel[face]
    }

    /// The hashes of a channel holding `level` in every open voxel, which no
    /// kernel run spread.
    pub fn uniform(level: u8) -> [u64; 6] {
        [if level == 0 { 0 } else { mix(0, level as u64) }; 6]
    }
}

/// Whether light in `from` could reach its neighbour across `face`, lit by
/// `to`, sooner than `to`'s levels say light got there, so the neighbour must
/// relight.
///
/// It holds when, in either channel, at some pair of touching open voxels, the
/// first distance from a source that holds `from`'s level is less than the
/// last distance that holds `to`'s, with level 0 counting as `reach + 1`.
/// Direct sky, at the top level, is a source itself, so a neighbour beside it
/// is one step from a source. `sky_scale` is `None` with sky light off.
pub(crate) fn could_reach_sooner(
    from: &ChunkLight,
    to: &ChunkLight,
    face: Face,
    voxel_scale: &LevelScale,
    sky_scale: Option<&LevelScale>,
) -> bool {
    // Light of another precision is from before a change of precision, which
    // relights every chunk anyway.
    if from.precision != to.precision {
        return true;
    }
    let channel_reaches = |scale: &LevelScale, of: fn(&ChunkLight) -> &ChannelLight| {
        if let ChannelLight::Uniform(0) = of(from) {
            return false;
        }
        let back = face.opposite();
        for u in 0..CHUNK {
            for v in 0..CHUNK {
                let (fx, fy, fz) = face.voxel(u, v);
                let (tx, ty, tz) = back.voxel(u, v);
                if from.inputs.occupancy.is_solid(fx, fy, fz)
                    || to.inputs.occupancy.is_solid(tx, ty, tz)
                {
                    continue;
                }
                let first = scale.first_distance(of(from).at(fx, fy, fz));
                let last = scale.last_distance(of(to).at(tx, ty, tz));
                if first < last {
                    return true;
                }
            }
        }
        false
    };
    channel_reaches(voxel_scale, |light| &light.voxel)
        || sky_scale.is_some_and(|scale| channel_reaches(scale, |light| &light.sky))
}
