//! What a chunk keeps for the solves of the chunks around it: its occupancy
//! and its emissive voxels.

use crate::chunk::{ChunkData, FillType, CHUNK_SIZE_U, PADDED_CHUNK_SIZE};
use crate::voxel::WorldVoxel;

use super::kernel::Plane;

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
}

impl LightInputs {
    /// What a chunk's voxels hold for light. With an empty `emission` list
    /// nothing looks for emissive voxels. Returns whether it looked.
    ///
    /// `chunk` must be a level-0 chunk with the default data shape.
    pub fn of<I: Copy + PartialEq>(
        chunk: &ChunkData<I>,
        emission: &[(I, u8)],
    ) -> (Self, bool) {
        let scanned = !emission.is_empty();
        let voxels = match (&chunk.fill_type, &chunk.voxels) {
            (FillType::Empty, _) => {
                return (
                    Self {
                        occupancy: Occupancy::Open,
                        emissive: Vec::new(),
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
                    },
                    scanned,
                );
            }
            (FillType::Mixed, Some(voxels)) => voxels,
            (FillType::Mixed, None) => unreachable!("a mixed chunk stores its voxels"),
        };

        let mut columns: Box<Plane> = Box::new([0; CHUNK * CHUNK]);
        let mut emissive = Vec::new();
        let mut solids = 0;
        for z in 0..CHUNK {
            for y in 0..CHUNK {
                let start = 1 + (y + 1) * PADDED + (z + 1) * PADDED * PADDED;
                let row = &voxels[start..start + CHUNK];
                let columns = &mut columns[y * CHUNK..(y + 1) * CHUNK];
                for (column, voxel) in columns.iter_mut().zip(row) {
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
            _ => Occupancy::Mixed(columns),
        };
        (
            Self {
                occupancy,
                emissive,
            },
            scanned,
        )
    }
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
