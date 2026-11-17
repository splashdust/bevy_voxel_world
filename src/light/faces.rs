//! What a chunk's light shows its six face neighbours: a hash per face of when
//! light reached it, and the comparison an arriving chunk makes with each
//! neighbour's light.
//!
//! The comparison looks at each pair of touching open voxels on the shared
//! face, and at each emissive voxel on the face. An emissive voxel is solid,
//! so it holds no light to compare, yet its light crosses the face into the
//! open voxel beside it. Where no open voxel of its own chunk on the face
//! carries that light across, as with an emissive voxel set in a solid wall,
//! comparing open voxels alone would never relight the neighbour, which would
//! keep light dimmer than solving it again gives. Only voxel light needs this:
//! direct sky, the other source on a face, is held by open voxels.
//!
//! # Light colours
//!
//! Where emissive voxels of several light colours light a chunk, its solve
//! runs once per light colour (see [`solver`](super::solver)), and each run is
//! hashed and compared on its own.
//!
//! A face's hash mixes the hash of each run whose light crosses it, keyed by
//! the run's light colour. A change of light colour never changes a level, so
//! without the key, an edit swapping an emissive voxel for one of another
//! light colour would leave every hash as it was, and ripple nowhere. A run
//! that doesn't cross a face adds nothing to its hash, and white light adds
//! its hash unkeyed, so a face only white light crosses hashes as it would in
//! a white world.
//!
//! An arrival compares each run with the run of the same light colour across
//! the face. Comparing levels alone misses colour: a neighbour lit green on
//! its face at a level above what an arriving red gives it still has to
//! relight, since red would tint it. A chunk stores only the highest level of
//! its runs at each voxel, so it also keeps [`BorderLevels`]: each run's
//! levels on each face where two or more runs light its open voxels. On any
//! other face one run at most lights the face's voxels, and its levels and
//! light colour are what the chunk stores.
//!
//! An emissive voxel on the face is compared with the run of its own light
//! colour across the face, for the same reason: the neighbour may be lit
//! brighter there by another colour, which the emissive voxel's light would
//! still tint.
//!
//! Border levels take a bit plane of the face per bit of a level, for each
//! run, on each face two or more runs light, so they grow with the light
//! colours that meet on a chunk's faces.

use std::sync::Arc;

use bevy::prelude::*;

use super::color::{voxel_index, EmissionTable, WHITE};
use super::config::LightPrecision;
use super::inputs::Occupancy;
use super::levels::LevelScale;
use super::solver::{ChunkLevels, ChunkLight};
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
        let side = self.side();
        match self {
            Face::NegX | Face::PosX => (side, u, v),
            Face::NegY | Face::PosY => (u, side, v),
            Face::NegZ | Face::PosZ => (u, v, side),
        }
    }

    /// The `u` and `v` of the chunk's voxel at `x`, `y`, `z` on this face, or
    /// `None` when the voxel isn't on it.
    pub fn uv(self, x: usize, y: usize, z: usize) -> Option<(usize, usize)> {
        let (across, u, v) = match self {
            Face::NegX | Face::PosX => (x, y, z),
            Face::NegY | Face::PosY => (y, x, z),
            Face::NegZ | Face::PosZ => (z, x, y),
        };
        (across == self.side()).then_some((u, v))
    }

    /// The coordinate of the chunk's voxels on this face along its axis: 0, or
    /// `CHUNK - 1` for a face toward positive coordinates.
    fn side(self) -> usize {
        let far = self as usize & 1 == 1;
        if far {
            CHUNK - 1
        } else {
            0
        }
    }
}

/// Mixes `value` into `hash`.
#[inline]
pub(crate) fn mix(hash: u64, value: u64) -> u64 {
    (hash.rotate_left(5) ^ value).wrapping_mul(0x517c_c1b7_2722_0a95)
}

/// The face hashes of voxel light solved in runs, one per light colour, from
/// each run's packed light colour and face hashes, in order of light colour.
///
/// Each face mixes the hash of every run whose light crosses it, keyed by its
/// light colour. White's hash goes in unkeyed, so a face only white light
/// crosses keeps the hash a white world gives it.
pub(crate) fn run_hashes(runs: impl IntoIterator<Item = (u16, [u64; 6])>) -> [u64; 6] {
    let mut faces = [0; 6];
    for (color, hashes) in runs {
        for (face, hash) in faces.iter_mut().zip(hashes) {
            if hash == 0 {
                continue;
            }
            let keyed = if color == WHITE {
                hash
            } else {
                mix(mix(0, color as u64), hash)
            };
            *face = if *face == 0 { keyed } else { mix(*face, keyed) };
        }
    }
    faces
}

/// For sky light and voxel light and per face, a hash of when light reached
/// each of a chunk's voxels on that face: 96 B a chunk.
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

    /// The hashes of sky light or voxel light holding `level` in every open
    /// voxel, which no kernel run spread.
    pub fn uniform(level: u8) -> [u64; 6] {
        [if level == 0 { 0 } else { mix(0, level as u64) }; 6]
    }
}

/// Each run's voxel light on each face of a chunk where two or more runs, of
/// different light colours, light the chunk's open voxels on that face.
///
/// A chunk stores only the highest level of its runs at each voxel, which
/// can't say how far each light colour got. An arrival compares runs of the
/// same light colour across a face, so it reads them here: see the module
/// docs. Nothing else reads them, and two chunks' light that differs only in
/// them is the same light.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct BorderLevels(Option<Arc<[Box<[FaceRun]>; 6]>>);

/// One run's levels on one face: its packed light colour, and one bit plane
/// of the face per bit of a level, a `u32` per `u` with a bit per `v`.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct FaceRun {
    color: u16,
    planes: Box<[[u32; CHUNK]]>,
}

impl FaceRun {
    fn level(&self, u: usize, v: usize) -> u8 {
        self.planes
            .iter()
            .enumerate()
            .map(|(bit, plane)| ((plane[u] >> v & 1) as u8) << bit)
            .sum()
    }
}

impl BorderLevels {
    /// The border levels of a chunk with `occupancy`, lit by runs of the
    /// packed light colours in `runs`, each with its level in steps of
    /// `precision` at every voxel, at [`voxel_index`].
    pub fn of<'a>(
        runs: impl IntoIterator<Item = (u16, &'a [u8])> + Clone,
        occupancy: &Occupancy,
        precision: LightPrecision,
    ) -> Self {
        let mut faces: [Box<[FaceRun]>; 6] = Default::default();
        let mut any = false;
        for face in Face::ALL {
            let mut lit = Vec::new();
            for (color, levels) in runs.clone() {
                let mut planes = vec![[0; CHUNK]; precision.bits()];
                let mut lights = false;
                for u in 0..CHUNK {
                    for v in 0..CHUNK {
                        let (x, y, z) = face.voxel(u, v);
                        let level = levels[voxel_index(x, y, z)];
                        if level == 0 || occupancy.is_solid(x, y, z) {
                            continue;
                        }
                        lights = true;
                        for (bit, plane) in planes.iter_mut().enumerate() {
                            plane[u] |= ((level >> bit & 1) as u32) << v;
                        }
                    }
                }
                if lights {
                    lit.push(FaceRun {
                        color,
                        planes: planes.into_boxed_slice(),
                    });
                }
            }
            if lit.len() >= 2 {
                faces[face as usize] = lit.into_boxed_slice();
                any = true;
            }
        }
        BorderLevels(any.then(|| Arc::new(faces)))
    }

    /// Whether the chunk keeps no border levels on any face.
    pub fn is_empty(&self) -> bool {
        self.0.is_none()
    }

    /// The runs whose levels the chunk keeps on `face`, if it keeps any there.
    fn on(&self, face: Face) -> Option<&[FaceRun]> {
        let runs = &self.0.as_ref()?[face as usize];
        (!runs.is_empty()).then_some(&runs[..])
    }
}

/// The level the run of the packed light colour `color` gives `light`'s open
/// voxel at `u` and `v` on `face`.
fn run_level(light: &ChunkLight, face: Face, u: usize, v: usize, color: u16) -> u8 {
    if let Some(runs) = light.borders.on(face) {
        return runs
            .iter()
            .find(|run| run.color == color)
            .map_or(0, |run| run.level(u, v));
    }
    let (x, y, z) = face.voxel(u, v);
    if light.color.packed_at(x, y, z) == color {
        light.voxel.at(x, y, z)
    } else {
        0
    }
}

/// Whether light in `from` could reach its neighbour across `face`, lit by
/// `to`, sooner than `to`'s levels say light got there, so the neighbour must
/// relight.
///
/// It holds when, in sky light or voxel light, at some pair of touching open
/// voxels, the first distance from a source that holds `from`'s level is less
/// than the last distance that holds `to`'s, with level 0 counting as
/// `reach + 1`. Direct sky, at the top level, is a source itself, so a
/// neighbour beside it is one step from a source. `sky_scale` is `None` with sky light off.
///
/// It also holds when an emissive voxel of `from` on the face could light the
/// open voxel of `to` across it sooner than `to`'s voxel light there says:
/// see [`emissive_reaches_sooner`].
///
/// `colors` is the world's emissive list where it holds a light colour other
/// than white, and `None` in a white world. With it, voxel light is compared
/// run by run, each run with the run of the same light colour across the
/// face.
pub(crate) fn could_reach_sooner<I>(
    from: &ChunkLight,
    to: &ChunkLight,
    face: Face,
    voxel_scale: &LevelScale,
    sky_scale: Option<&LevelScale>,
    colors: Option<&EmissionTable<I>>,
) -> bool {
    // Light of another precision is from before a change of precision, which
    // relights every chunk anyway.
    if from.precision != to.precision {
        return true;
    }
    let reaches = |scale: &LevelScale, of: fn(&ChunkLight) -> &ChunkLevels| {
        if let ChunkLevels::Uniform(0) = of(from) {
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
    // Between two chunks lit only by white, the one run of each is its
    // stored levels.
    let voxel = match colors {
        Some(_) if !(from.is_white() && to.is_white()) => {
            runs_reach_sooner(from, to, face, voxel_scale)
        }
        _ => reaches(voxel_scale, |light| &light.voxel),
    };
    voxel
        || emissive_reaches_sooner(from, to, face, voxel_scale, colors)
        || sky_scale.is_some_and(|scale| reaches(scale, |light| &light.sky))
}

/// Whether, at some pair of touching open voxels on `face`, a run of `from`
/// could reach `to` sooner than the run of the same light colour in `to` says
/// light got there, comparing levels as [`could_reach_sooner`] does.
fn runs_reach_sooner(
    from: &ChunkLight,
    to: &ChunkLight,
    face: Face,
    scale: &LevelScale,
) -> bool {
    if let ChunkLevels::Uniform(0) = from.voxel {
        return false;
    }
    let back = face.opposite();
    let sooner = |u: usize, v: usize, color: u16, level: u8| {
        level > 0
            && scale.first_distance(level)
                < scale.last_distance(run_level(to, back, u, v, color))
    };
    for u in 0..CHUNK {
        for v in 0..CHUNK {
            let (fx, fy, fz) = face.voxel(u, v);
            let (tx, ty, tz) = back.voxel(u, v);
            if from.inputs.occupancy.is_solid(fx, fy, fz)
                || to.inputs.occupancy.is_solid(tx, ty, tz)
            {
                continue;
            }
            let reaches = match from.borders.on(face) {
                Some(runs) => runs
                    .iter()
                    .any(|run| sooner(u, v, run.color, run.level(u, v))),
                None => sooner(
                    u,
                    v,
                    from.color.packed_at(fx, fy, fz),
                    from.voxel.at(fx, fy, fz),
                ),
            };
            if reaches {
                return true;
            }
        }
    }
    false
}

/// Whether an emissive voxel of `from` on `face` could light the open voxel of
/// `to` across it sooner than `to`'s voxel light there says light got there.
///
/// Light from an emissive voxel starts at the distance [`LevelScale::start`]
/// gives for its level, and the voxels beside it are lit at the next distance.
/// So it lights the voxel across the face one step past its start, unlike an
/// open voxel, whose light can cross a face at the distance that reached it.
/// With `colors`, `to`'s voxel light there is that of the run of the emissive
/// voxel's own light colour.
///
/// It looks at every emissive voxel of `from`, so its cost grows with how many
/// the chunk has.
fn emissive_reaches_sooner<I>(
    from: &ChunkLight,
    to: &ChunkLight,
    face: Face,
    scale: &LevelScale,
    colors: Option<&EmissionTable<I>>,
) -> bool {
    let back = face.opposite();
    from.inputs.emissive.iter().any(|voxel| {
        let Some((u, v)) =
            face.uv(voxel.x() as usize, voxel.y() as usize, voxel.z() as usize)
        else {
            return false;
        };
        let (tx, ty, tz) = back.voxel(u, v);
        if to.inputs.occupancy.is_solid(tx, ty, tz) {
            return false;
        }
        let Some(start) = scale.start(scale.precision().of_whole(voxel.level())) else {
            return false;
        };
        let level = match colors {
            Some(colors) => run_level(to, back, u, v, colors.color(voxel.material())),
            None => to.voxel.at(tx, ty, tz),
        };
        start as u32 + 1 < scale.last_distance(level)
    })
}
