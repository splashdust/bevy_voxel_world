//! What a chunk keeps for the solves of the chunks around it: its occupancy,
//! its emissive voxels and its column answer.

use bevy::prelude::*;

use crate::chunk::{ChunkData, FillType, CHUNK_SIZE_I, CHUNK_SIZE_U, PADDED_CHUNK_SIZE};
use crate::voxel::WorldVoxel;
use crate::voxel_edits::EditBucket;

use super::color::{EmissionTable, Emitter, MAX_EMISSIVE_MATERIALS};
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

/// An emissive voxel of a chunk: where it is in the chunk, its level and its
/// material's index in the world's [`EmissionTable`], packed into 32 bits.
///
/// Its position takes 5 bits per axis, from bit 0 for x, its level 4 bits
/// from bit 15, and the index 13 bits from bit 19. An emissive voxel is
/// always one of its chunk's own voxels, never one of its padding.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct EmissiveVoxel(u32);

impl EmissiveVoxel {
    /// The voxel at `x`, `y`, `z` of its chunk, each below 32, giving off
    /// what `emitter` says.
    #[inline]
    pub fn new(x: u8, y: u8, z: u8, emitter: Emitter) -> Self {
        debug_assert!(x < 32 && y < 32 && z < 32 && emitter.level < 16);
        debug_assert!((emitter.material as usize) < MAX_EMISSIVE_MATERIALS);
        Self(
            x as u32
                | (y as u32) << 5
                | (z as u32) << 10
                | (emitter.level as u32) << 15
                | (emitter.material as u32) << 19,
        )
    }

    #[inline]
    pub fn x(self) -> u8 {
        (self.0 & 31) as u8
    }

    #[inline]
    pub fn y(self) -> u8 {
        (self.0 >> 5 & 31) as u8
    }

    #[inline]
    pub fn z(self) -> u8 {
        (self.0 >> 10 & 31) as u8
    }

    /// Where it is in its chunk.
    #[inline]
    pub fn at(self) -> (u8, u8, u8) {
        (self.x(), self.y(), self.z())
    }

    /// Its level, as a whole level from 0 to 15.
    #[inline]
    pub fn level(self) -> u8 {
        (self.0 >> 15 & 15) as u8
    }

    /// Its material's index in the world's [`EmissionTable`].
    #[inline]
    pub fn material(self) -> u16 {
        (self.0 >> 19) as u16
    }
}

impl std::fmt::Debug for EmissiveVoxel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EmissiveVoxel")
            .field("at", &self.at())
            .field("level", &self.level())
            .field("material", &self.material())
            .finish()
    }
}

/// A chunk's emissive voxels, and a note of the emissive materials among them
/// by their index in the world's [`EmissionTable`].
///
/// A solve reads the notes of the chunks around its chunk to find the light
/// colours that could reach it, without reading their emissive voxels.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub(crate) struct Emissive {
    voxels: Vec<EmissiveVoxel>,
    /// The distinct material indices of `voxels`, in order.
    materials: Box<[u16]>,
}

impl Emissive {
    /// The material indices of the emissive voxels, each once, in order.
    #[inline]
    pub fn materials(&self) -> &[u16] {
        &self.materials
    }
}

impl From<Vec<EmissiveVoxel>> for Emissive {
    fn from(voxels: Vec<EmissiveVoxel>) -> Self {
        let mut materials: Vec<u16> = Vec::new();
        let mut last = None;
        for voxel in &voxels {
            let material = voxel.material();
            if last != Some(material) {
                last = Some(material);
                if !materials.contains(&material) {
                    materials.push(material);
                }
            }
        }
        materials.sort_unstable();
        Self {
            voxels,
            materials: materials.into_boxed_slice(),
        }
    }
}

impl<'a> IntoIterator for &'a Emissive {
    type Item = &'a EmissiveVoxel;
    type IntoIter = std::slice::Iter<'a, EmissiveVoxel>;

    fn into_iter(self) -> Self::IntoIter {
        self.voxels.iter()
    }
}

impl std::ops::Deref for Emissive {
    type Target = [EmissiveVoxel];

    #[inline]
    fn deref(&self) -> &[EmissiveVoxel] {
        &self.voxels
    }
}

/// What a chunk's neighbours read of it when they solve.
#[derive(Debug)]
pub(crate) struct LightInputs {
    pub occupancy: Occupancy,
    pub emissive: Emissive,
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
        emission: &EmissionTable<I>,
        columns: ColumnAnswer,
    ) -> (Self, bool) {
        let scanned = !emission.is_empty();
        let voxels = match (&chunk.fill_type, &chunk.voxels) {
            (FillType::Empty, _) => {
                return (
                    Self {
                        occupancy: Occupancy::Open,
                        emissive: Emissive::default(),
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
                    Some(emitter) => every_voxel(emitter),
                    None => Vec::new(),
                };
                return (
                    Self {
                        occupancy,
                        emissive: emissive.into(),
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
                emissive: emissive.into(),
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
        emission: &EmissionTable<I>,
    ) -> Self {
        let mut emissive = scan_emissive(&rescan.data, emission);
        if let (Some(edits), false) = (&rescan.edits, emission.is_empty()) {
            let min = position * CHUNK_SIZE_I;
            for (at, voxel) in edits.iter() {
                let local = at - min;
                let (x, y, z) = (local.x as u8, local.y as u8, local.z as u8);
                emissive.retain(|emitter| emitter.at() != (x, y, z));
                if let Some(emitter) = uniform_emission(voxel, emission) {
                    emissive.push(EmissiveVoxel::new(x, y, z, emitter));
                }
            }
        }
        Self {
            occupancy: self.occupancy.clone(),
            emissive: emissive.into(),
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

    /// Whether the chunk is air with no solid in its padding. Its solve takes
    /// no sky fill, so its stored light can miss what the sky fill field's
    /// cells know, and the light shown at each of its voxels is the larger
    /// of the two.
    pub fn reads_fill_cells(&self) -> bool {
        matches!(self.occupancy, Occupancy::Open) && !self.ring
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
    emission: &EmissionTable<I>,
) -> Vec<EmissiveVoxel> {
    if emission.is_empty() {
        return Vec::new();
    }
    let voxels = match (&chunk.fill_type, &chunk.voxels) {
        (FillType::Empty, _) => return Vec::new(),
        (FillType::Uniform(voxel), _) => {
            return match uniform_emission(*voxel, emission) {
                Some(emitter) => every_voxel(emitter),
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
    emission: &EmissionTable<I>,
    y: usize,
    z: usize,
    found: &mut Vec<EmissiveVoxel>,
) {
    // Materials come in runs, so the last one's answer is kept.
    let mut last: Option<(I, Option<Emitter>)> = None;
    for (x, voxel) in row.iter().enumerate() {
        let WorldVoxel::Solid(material) = *voxel else {
            continue;
        };
        let emitter = match last {
            Some((seen, emitter)) if seen == material => emitter,
            _ => {
                let emitter = emission.get(material);
                last = Some((material, emitter));
                emitter
            }
        };
        if let Some(emitter) = emitter {
            found.push(EmissiveVoxel::new(x as u8, y as u8, z as u8, emitter));
        }
    }
}

/// What `voxel` gives off under the emissive list `emission`, if it's
/// emissive.
pub(crate) fn uniform_emission<I: Copy + PartialEq>(
    voxel: WorldVoxel<I>,
    emission: &EmissionTable<I>,
) -> Option<Emitter> {
    let WorldVoxel::Solid(material) = voxel else {
        return None;
    };
    emission.get(material)
}

fn every_voxel(emitter: Emitter) -> Vec<EmissiveVoxel> {
    let mut voxels = Vec::with_capacity(CHUNK * CHUNK * CHUNK);
    for z in 0..CHUNK as u8 {
        for y in 0..CHUNK as u8 {
            for x in 0..CHUNK as u8 {
                voxels.push(EmissiveVoxel::new(x, y, z, emitter));
            }
        }
    }
    voxels
}
