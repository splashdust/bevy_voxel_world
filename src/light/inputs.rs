//! What a chunk keeps for the solves of the chunks around it: its occupancy,
//! its emissive voxels and its column answer.

use bevy::prelude::*;

use crate::chunk::{ChunkData, FillType, CHUNK_SIZE_I, CHUNK_SIZE_U, PADDED_CHUNK_SIZE};
use crate::voxel::WorldVoxel;
use crate::voxel_edits::EditBucket;

use super::kernel::Plane;
use super::sky::{ColumnAnswer, Columns};

const CHUNK: usize = CHUNK_SIZE_U as usize;
const PADDED: usize = PADDED_CHUNK_SIZE as usize;

/// Which of a chunk's voxels are solid. Only a chunk with both stores them,
/// 4 KiB as one `u32` column along z at `x + y × 32`, a set bit for a solid.
#[derive(Clone, Debug)]
pub(crate) enum Occupancy {
    Open,
    Solid,
    Mixed(Box<Plane>),
}

impl Occupancy {
    /// The column of solid voxels at `x` and `y`, bit `z` for the voxel at `z`.
    #[inline]
    pub fn solid_column(&self, x: usize, y: usize) -> u32 {
        match self {
            Occupancy::Open => 0,
            Occupancy::Solid => !0,
            Occupancy::Mixed(columns) => columns[x + y * CHUNK],
        }
    }

    #[inline]
    pub fn is_solid(&self, x: usize, y: usize, z: usize) -> bool {
        self.solid_column(x, y) >> z & 1 == 1
    }

    pub fn has_open_voxel(&self) -> bool {
        !matches!(self, Occupancy::Solid)
    }

    /// Makes the voxel at `x`, `y`, `z` solid or open. Call
    /// [`normalize`](Self::normalize) once done setting voxels.
    pub fn set(&mut self, x: usize, y: usize, z: usize, solid: bool) {
        if self.is_solid(x, y, z) == solid {
            return;
        }
        let columns = match self {
            Occupancy::Mixed(columns) => columns,
            _ => {
                let fill = if matches!(self, Occupancy::Solid) {
                    !0
                } else {
                    0
                };
                *self = Occupancy::Mixed(Box::new([fill; CHUNK * CHUNK]));
                let Occupancy::Mixed(columns) = self else {
                    unreachable!()
                };
                columns
            }
        };
        columns[x + y * CHUNK] ^= 1 << z;
    }

    /// Stores a chunk whose voxels are all solid or all open as such, after
    /// [`set`](Self::set).
    pub fn normalize(&mut self) {
        let Occupancy::Mixed(columns) = self else {
            return;
        };
        if columns.iter().all(|column| *column == 0) {
            *self = Occupancy::Open;
        } else if columns.iter().all(|column| *column == !0) {
            *self = Occupancy::Solid;
        }
    }

    /// Which of the chunk's columns along y hold a solid voxel, as the sky
    /// record keeps them.
    pub fn solid_columns(&self) -> Columns {
        match self {
            Occupancy::Open => [0; CHUNK],
            Occupancy::Solid => [!0; CHUNK],
            Occupancy::Mixed(columns) => {
                let mut solid = [0; CHUNK];
                for row in columns.as_chunks::<CHUNK>().0 {
                    for (x, column) in row.iter().enumerate() {
                        solid[x] |= column;
                    }
                }
                solid
            }
        }
    }
}

/// An emissive voxel of a chunk: where it is in the chunk, and its level.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct EmissiveVoxel {
    pub x: u8,
    pub y: u8,
    pub z: u8,
    pub level: u8,
}

/// What a chunk's neighbours read of it when they solve.
#[derive(Debug)]
pub(crate) struct LightInputs {
    pub occupancy: Occupancy,
    pub emissive: Vec<EmissiveVoxel>,
    /// Which columns sky light enters at the chunk's top.
    ///
    /// Its `sky` is kept current. Its `provisional` is as of the last time
    /// `sky` changed: columns proved clear since are tracked by the world's
    /// [`SkyProbes`](super::probe::SkyProbes), which don't replace a chunk's
    /// inputs when only that changes.
    pub columns: ColumnAnswer,
    /// Whether a voxel in the chunk's padding, the layer of voxels just
    /// outside its faces, is solid, as its voxels were generated. A voxel
    /// edit can turn it on but never off.
    pub ring: bool,
}

impl LightInputs {
    /// What a chunk's voxels hold for light, with its column answer
    /// `columns`. With an empty `emission` list nothing looks for emissive
    /// voxels. Returns whether it looked.
    ///
    /// `chunk` must be a level-0 chunk with the default data shape.
    pub fn of<I: Copy + PartialEq>(
        chunk: &ChunkData<I>,
        emission: &[(I, u8)],
        columns: ColumnAnswer,
    ) -> (Self, bool) {
        let scanned = !emission.is_empty();
        let voxels = match (&chunk.fill_type, &chunk.voxels) {
            (FillType::Empty, _) => {
                return (
                    Self {
                        occupancy: Occupancy::Open,
                        emissive: Vec::new(),
                        columns,
                        ring: false,
                    },
                    false,
                )
            }
            (FillType::Uniform(voxel), _) => {
                let occupancy = if voxel.is_solid() {
                    Occupancy::Solid
                } else {
                    Occupancy::Open
                };
                let emissive = match uniform_emission(*voxel, emission) {
                    Some(level) => every_voxel(level),
                    None => Vec::new(),
                };
                return (
                    Self {
                        occupancy,
                        emissive,
                        columns,
                        ring: voxel.is_solid(),
                    },
                    scanned,
                );
            }
            (FillType::Mixed, Some(voxels)) => voxels,
            (FillType::Mixed, None) => unreachable!("a mixed chunk stores its voxels"),
        };

        let mut solid: Box<Plane> = Box::new([0; CHUNK * CHUNK]);
        let mut emissive = Vec::new();
        let mut solids = 0;
        for z in 0..CHUNK {
            for y in 0..CHUNK {
                let start = 1 + (y + 1) * PADDED + (z + 1) * PADDED * PADDED;
                let row = &voxels[start..start + CHUNK];
                let solid = &mut solid[y * CHUNK..(y + 1) * CHUNK];
                for (column, voxel) in solid.iter_mut().zip(row) {
                    let solid = voxel.is_solid() as u32;
                    *column |= solid << z;
                    solids += solid;
                }
                if scanned {
                    find_emissive(row, emission, y, z, &mut emissive);
                }
            }
        }
        let occupancy = match solids as usize {
            0 => Occupancy::Open,
            n if n == CHUNK * CHUNK * CHUNK => Occupancy::Solid,
            _ => Occupancy::Mixed(solid),
        };
        (
            Self {
                occupancy,
                emissive,
                columns,
                ring: true,
            },
            scanned,
        )
    }
}

/// What a relight reads to find a chunk's emissive voxels again after the
/// world's emissive list changed: the chunk's published data, and the voxel
/// edits of its position, which may be newer than that data.
pub(crate) struct Rescan<I> {
    pub data: ChunkData<I>,
    pub edits: Option<EditBucket<I>>,
}

impl LightInputs {
    /// These inputs with the emissive voxels of `rescan` under the emissive
    /// list `emission`, keeping the occupancy and column answer, which no
    /// emissive list changes.
    ///
    /// The chunk at `position` must be a level-0 chunk with the default data
    /// shape.
    pub fn rescanned<I: Copy + PartialEq>(
        &self,
        position: IVec3,
        rescan: &Rescan<I>,
        emission: &[(I, u8)],
    ) -> Self {
        let mut emissive = scan_emissive(&rescan.data, emission);
        if let (Some(edits), false) = (&rescan.edits, emission.is_empty()) {
            let min = position * CHUNK_SIZE_I;
            for (at, voxel) in edits.iter() {
                let local = at - min;
                let (x, y, z) = (local.x as u8, local.y as u8, local.z as u8);
                emissive.retain(|emitter| (emitter.x, emitter.y, emitter.z) != (x, y, z));
                if let Some(level) = uniform_emission(voxel, emission) {
                    emissive.push(EmissiveVoxel { x, y, z, level });
                }
            }
        }
        Self {
            occupancy: self.occupancy.clone(),
            emissive,
            columns: self.columns.clone(),
            ring: self.ring,
        }
    }

    /// These inputs with the column answer `columns`.
    pub fn with_columns(&self, columns: ColumnAnswer) -> Self {
        Self {
            occupancy: self.occupancy.clone(),
            emissive: self.emissive.clone(),
            columns,
            ring: self.ring,
        }
    }

    /// Whether the chunk takes sky fill: it has an open voxel touching a
    /// solid by a face, an edge or a corner, counting the solids in its
    /// padding. The solver also leaves out a chunk whose open voxels all have
    /// direct sky.
    ///
    /// A chunk with both open and solid voxels always has one, and a chunk of
    /// air has one exactly when its padding holds a solid.
    pub fn touches_solid(&self) -> bool {
        match self.occupancy {
            Occupancy::Mixed(_) => true,
            Occupancy::Open => self.ring,
            Occupancy::Solid => false,
        }
    }

    /// Whether every open voxel of the chunk has direct sky. False for a
    /// chunk with no open voxel.
    pub fn all_open_direct(&self) -> bool {
        let Some(direct) = self.direct_sky() else {
            return false;
        };
        (0..CHUNK * CHUNK).all(|column| {
            direct[column] == !self.occupancy.solid_column(column % CHUNK, column / CHUNK)
        })
    }

    /// The chunk's voxels with direct sky, as a [`Plane`]: in each column its
    /// answer lets sky light into, the open voxels from the chunk's top down
    /// to the first solid voxel. `None` if no voxel has direct sky.
    pub fn direct_sky(&self) -> Option<Box<Plane>> {
        if !self.columns.has_sky() || !self.occupancy.has_open_voxel() {
            return None;
        }
        let mut plane: Box<Plane> = Box::new([0; CHUNK * CHUNK]);
        for x in 0..CHUNK {
            let mut falling = self.columns.sky[x];
            for y in (0..CHUNK).rev() {
                if falling == 0 {
                    break;
                }
                falling &= !self.occupancy.solid_column(x, y);
                plane[x + y * CHUNK] = falling;
            }
        }
        Some(plane)
    }
}

/// The emissive voxels of the level-0 chunk `chunk`, of the default data
/// shape, under the emissive list `emission`.
fn scan_emissive<I: Copy + PartialEq>(
    chunk: &ChunkData<I>,
    emission: &[(I, u8)],
) -> Vec<EmissiveVoxel> {
    if emission.is_empty() {
        return Vec::new();
    }
    let voxels = match (&chunk.fill_type, &chunk.voxels) {
        (FillType::Empty, _) => return Vec::new(),
        (FillType::Uniform(voxel), _) => {
            return match uniform_emission(*voxel, emission) {
                Some(level) => every_voxel(level),
                None => Vec::new(),
            };
        }
        (FillType::Mixed, Some(voxels)) => voxels,
        (FillType::Mixed, None) => unreachable!("a mixed chunk stores its voxels"),
    };
    let mut emissive = Vec::new();
    for z in 0..CHUNK {
        for y in 0..CHUNK {
            let start = 1 + (y + 1) * PADDED + (z + 1) * PADDED * PADDED;
            find_emissive(&voxels[start..start + CHUNK], emission, y, z, &mut emissive);
        }
    }
    emissive
}

/// The emissive voxels of one row of voxels along x, at `y` and `z`.
#[inline]
fn find_emissive<I: Copy + PartialEq>(
    row: &[WorldVoxel<I>],
    emission: &[(I, u8)],
    y: usize,
    z: usize,
    found: &mut Vec<EmissiveVoxel>,
) {
    // Materials come in runs, so the last one's answer is kept.
    let mut last: Option<(I, Option<u8>)> = None;
    for (x, voxel) in row.iter().enumerate() {
        let WorldVoxel::Solid(material) = *voxel else {
            continue;
        };
        let level = match last {
            Some((seen, level)) if seen == material => level,
            _ => {
                let level = emission
                    .iter()
                    .find(|(emissive, _)| *emissive == material)
                    .map(|(_, level)| *level);
                last = Some((material, level));
                level
            }
        };
        if let Some(level) = level {
            found.push(EmissiveVoxel {
                x: x as u8,
                y: y as u8,
                z: z as u8,
                level,
            });
        }
    }
}

fn uniform_emission<I: Copy + PartialEq>(
    voxel: WorldVoxel<I>,
    emission: &[(I, u8)],
) -> Option<u8> {
    let WorldVoxel::Solid(material) = voxel else {
        return None;
    };
    emission
        .iter()
        .find(|(emissive, _)| *emissive == material)
        .map(|(_, level)| *level)
}

fn every_voxel(level: u8) -> Vec<EmissiveVoxel> {
    let mut voxels = Vec::with_capacity(CHUNK * CHUNK * CHUNK);
    for z in 0..CHUNK as u8 {
        for y in 0..CHUNK as u8 {
            for x in 0..CHUNK as u8 {
                voxels.push(EmissiveVoxel { x, y, z, level });
            }
        }
    }
    voxels
}
