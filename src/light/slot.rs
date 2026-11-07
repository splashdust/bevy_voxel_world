//! What goes into a chunk's slot, and whether a chunk needs one.
//!
//! A chunk's faces read light from the open voxels in front of them: its own
//! voxels, and through the slot's shell its six face neighbours' border voxels.
//! A chunk needs a slot unless every open voxel touching one of its solid
//! voxels by a face, an edge or a corner holds one level per channel. Edges
//! and corners count because a face off the voxel grid, from a mesher other
//! than cubes, reads the voxels around it and not only the one in front.
//!
//! An emissive voxel touching an open voxel always needs a slot, since only a
//! slot holds the lamp's own cell, which makes its faces glow.
//!
//! A slot's format follows the world's precision, and is chosen in one place,
//! [`texel_bytes`] and [`texture_format`]: at 16 levels a texel is one
//! `R8Uint` byte, sky light in its low nibble and voxel light in its high
//! one; at 256 levels two bytes of `Rg8Uint`, sky light then voxel light.
//!
//! A solve works out what the chunk's own voxels show its faces, as
//! [`own_faces_see`]. Landing works out what each loaded face neighbour's
//! border shows them, as [`across`], whenever either chunk's light changes. A
//! neighbour that isn't loaded shows them nothing.

use bevy::prelude::*;
use bevy::render::render_resource::TextureFormat;

use super::config::LightPrecision;
use super::faces::Face;
use super::inputs::{EmissiveVoxel, LightInputs, Occupancy};
use super::kernel::Plane;
use super::levels::LightLevels;
use super::solver::{ChannelLight, ChunkLight};
use crate::chunk::CHUNK_SIZE_U;

const CHUNK: usize = CHUNK_SIZE_U as usize;
/// Cells along each side of a slot: a chunk and a shell one voxel deep.
const SIDE: usize = CHUNK + 2;
/// The texels of one slot.
pub(crate) const SLOT_TEXELS: usize = SIDE * SIDE * SIDE;
/// The texels of one shell plane, at `u + v × 32` for the cell in front of
/// the chunk's voxel [`Face::voxel`]`(u, v)`.
pub(crate) const PLANE_TEXELS: usize = CHUNK * CHUNK;

/// The bytes of a shell plane, in the format of a precision.
pub(crate) type ShellPlane = [u8];

/// How many bytes a texel has at `precision`.
#[inline]
pub(crate) fn texel_bytes(precision: LightPrecision) -> usize {
    match precision {
        LightPrecision::Steps16 => 1,
        LightPrecision::Steps256 => 2,
    }
}

/// The format of a light texture at `precision`.
pub(crate) fn texture_format(precision: LightPrecision) -> TextureFormat {
    match precision {
        LightPrecision::Steps16 => TextureFormat::R8Uint,
        LightPrecision::Steps256 => TextureFormat::Rg8Uint,
    }
}

/// Writes sky light `sky` and voxel light `voxel`, steps of `precision`, into
/// the texel at `texel` of `bytes`.
#[inline]
pub(crate) fn put_texel(
    bytes: &mut [u8],
    texel: usize,
    sky: u8,
    voxel: u8,
    precision: LightPrecision,
) {
    match precision {
        LightPrecision::Steps16 => bytes[texel] = sky | voxel << 4,
        LightPrecision::Steps256 => {
            bytes[2 * texel] = sky;
            bytes[2 * texel + 1] = voxel;
        }
    }
}

/// The sky half and voxel half of the texel at `texel` of `bytes`.
#[inline]
pub(crate) fn texel_at(
    bytes: &[u8],
    texel: usize,
    precision: LightPrecision,
) -> (u8, u8) {
    match precision {
        LightPrecision::Steps16 => (bytes[texel] & 15, bytes[texel] >> 4),
        LightPrecision::Steps256 => (bytes[2 * texel], bytes[2 * texel + 1]),
    }
}

/// The light of a texel's sky half `sky` and voxel half `voxel`: a lamp's own
/// cell, whose voxel half holds the top level, holds no sky light and the
/// lamp's level, from its sky half, as voxel light.
pub(crate) fn texel_levels(sky: u8, voxel: u8, precision: LightPrecision) -> LightLevels {
    if voxel == precision.top() {
        LightLevels::of_steps(0, sky, precision)
    } else {
        LightLevels::of_steps(sky, voxel, precision)
    }
}

/// Writes the marker of a lamp at the whole level `level` into the texel at
/// `texel`: the top level, which no open voxel holds, in its voxel half, and
/// its level in its sky half, which a solid never uses.
pub(crate) fn put_lamp(
    bytes: &mut [u8],
    texel: usize,
    level: u8,
    precision: LightPrecision,
) {
    put_texel(
        bytes,
        texel,
        precision.of_whole(level),
        precision.top(),
        precision,
    );
}

/// `bytes`, a slot of `from`'s format, in the format of `to`.
pub(crate) fn convert_texels(
    bytes: &[u8],
    from: LightPrecision,
    to: LightPrecision,
) -> Box<[u8]> {
    if from == to {
        return bytes.into();
    }
    let texels = bytes.len() / texel_bytes(from);
    let mut out = vec![0; texels * texel_bytes(to)].into_boxed_slice();
    for texel in 0..texels {
        let (sky, voxel) = texel_at(bytes, texel, from);
        put_texel(
            &mut out,
            texel,
            to.convert(sky, from),
            to.convert(voxel, from),
            to,
        );
    }
    out
}

/// For each byte, a `u64` with bit `i` of the byte moved to bit 0 of byte `i`.
const SPREAD: [u64; 256] = {
    let mut table = [0u64; 256];
    let mut byte = 0;
    while byte < 256 {
        let mut bit = 0;
        while bit < 8 {
            if byte >> bit & 1 == 1 {
                table[byte] |= 1 << (bit * 8);
            }
            bit += 1;
        }
        byte += 1;
    }
    table
};

/// What some open voxels hold of one channel: no voxel at all, one level, or
/// more than one.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum Seen {
    #[default]
    Nothing,
    One(u8),
    Mixed,
}

impl Seen {
    pub fn join(self, other: Seen) -> Seen {
        match (self, other) {
            (Seen::Nothing, seen) | (seen, Seen::Nothing) => seen,
            (Seen::One(a), Seen::One(b)) if a == b => Seen::One(a),
            _ => Seen::Mixed,
        }
    }
}

/// What a chunk's faces read from some of the open voxels in front of them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct FacesSee {
    pub sky: Seen,
    pub voxel: Seen,
    /// Whether an emissive voxel of the chunk touches one of these voxels.
    pub lamp: bool,
}

impl FacesSee {
    pub fn join(self, other: FacesSee) -> FacesSee {
        FacesSee {
            sky: self.sky.join(other.sky),
            voxel: self.voxel.join(other.voxel),
            lamp: self.lamp || other.lamp,
        }
    }

    pub fn needs_slot(&self) -> bool {
        self.lamp || self.sky == Seen::Mixed || self.voxel == Seen::Mixed
    }
}

/// What a chunk's faces read from its own open voxels, given its `light`.
pub(crate) fn own_faces_see(light: &ChunkLight) -> FacesSee {
    let Occupancy::Mixed(solid) = &light.inputs.occupancy else {
        return FacesSee::default();
    };
    let touching = touching_open(solid);
    FacesSee {
        sky: seen_over(&light.sky, &touching),
        voxel: seen_over(&light.voxel, &touching),
        lamp: light
            .inputs
            .emissive
            .iter()
            .any(|lamp| touches_open(solid, lamp)),
    }
}

/// The open voxels of a chunk touching one of its `solid` voxels by a face,
/// an edge or a corner, as columns along z like `solid`'s.
fn touching_open(solid: &Plane) -> Box<Plane> {
    let mut along_z: Box<Plane> = Box::new([0; CHUNK * CHUNK]);
    for (spread, column) in along_z.iter_mut().zip(solid.iter()) {
        *spread = column | column << 1 | column >> 1;
    }
    let mut touching: Box<Plane> = Box::new([0; CHUNK * CHUNK]);
    for y in 0..CHUNK {
        for x in 0..CHUNK {
            let mut near = 0;
            for ny in y.saturating_sub(1)..(y + 2).min(CHUNK) {
                for nx in x.saturating_sub(1)..(x + 2).min(CHUNK) {
                    near |= along_z[nx + ny * CHUNK];
                }
            }
            touching[x + y * CHUNK] = near & !solid[x + y * CHUNK];
        }
    }
    touching
}

/// Whether the emissive voxel `lamp` touches an open voxel of its own chunk.
fn touches_open(solid: &Plane, lamp: &EmissiveVoxel) -> bool {
    let (x, y, z) = (lamp.x as usize, lamp.y as usize, lamp.z as usize);
    let around = ((0b111u64 << z) >> 1) as u32;
    (y.saturating_sub(1)..(y + 2).min(CHUNK)).any(|ny| {
        (x.saturating_sub(1)..(x + 2).min(CHUNK))
            .any(|nx| !solid[nx + ny * CHUNK] & around != 0)
    })
}

/// What one channel holds in the voxels set in `voxels`.
fn seen_over(channel: &ChannelLight, voxels: &Plane) -> Seen {
    let mut seen = Seen::Nothing;
    match channel {
        ChannelLight::Uniform(level) => {
            if voxels.iter().any(|column| *column != 0) {
                seen = Seen::One(*level);
            }
        }
        ChannelLight::Planes(planes) => {
            for (column, &these) in voxels.iter().enumerate() {
                if these == 0 {
                    continue;
                }
                let z = these.trailing_zeros();
                let level: u8 = planes
                    .iter()
                    .enumerate()
                    .map(|(bit, plane)| ((plane[column] >> z & 1) as u8) << bit)
                    .sum();
                let same = planes.iter().enumerate().all(|(bit, plane)| {
                    let want = if level >> bit & 1 == 1 { these } else { 0 };
                    plane[column] & these == want
                });
                seen = if same {
                    seen.join(Seen::One(level))
                } else {
                    Seen::Mixed
                };
                if seen == Seen::Mixed {
                    break;
                }
            }
        }
    }
    seen
}

/// The light a chunk's slot holds for its own voxels, with its shell at 0, in
/// the format of the light's precision.
///
/// Open voxels hold their sky light and voxel light. Solid voxels hold 0,
/// apart from each emissive voxel, which holds a lamp's marker: see
/// [`put_lamp`].
pub(crate) fn slot_bytes(light: &ChunkLight) -> Box<[u8]> {
    let precision = light.precision;
    let wide = texel_bytes(precision);
    let mut bytes = vec![0u8; SLOT_TEXELS * wide].into_boxed_slice();
    for y in 0..CHUNK {
        for x in 0..CHUNK {
            let column = x + y * CHUNK;
            let open = !light.inputs.occupancy.solid_column(x, y);
            let row = 1 + (x + 1) * SIDE + (y + 1) * SIDE * SIDE;
            for shift in (0..CHUNK).step_by(8) {
                let open_byte = (open >> shift) as usize & 0xFF;
                let sky = levels(&light.sky, column, shift, open_byte);
                let voxel = levels(&light.voxel, column, shift, open_byte);
                let at = (row + shift) * wide;
                match precision {
                    LightPrecision::Steps16 => bytes[at..at + 8]
                        .copy_from_slice(&(sky | voxel << 4).to_le_bytes()),
                    LightPrecision::Steps256 => bytes[at..at + 16].copy_from_slice(
                        &(every_other_byte(sky) | every_other_byte(voxel) << 8)
                            .to_le_bytes(),
                    ),
                }
            }
        }
    }
    for lamp in &light.inputs.emissive {
        let (x, y, z) = (lamp.x as usize, lamp.y as usize, lamp.z as usize);
        let texel = (z + 1) + (x + 1) * SIDE + (y + 1) * SIDE * SIDE;
        put_lamp(&mut bytes, texel, lamp.level, precision);
    }
    bytes
}

/// Byte `i` of `bytes` moved to byte `2i`.
#[inline]
fn every_other_byte(bytes: u64) -> u128 {
    let mut wide = bytes as u128;
    wide = (wide | wide << 32) & 0x0000_0000_FFFF_FFFF_0000_0000_FFFF_FFFF;
    wide = (wide | wide << 16) & 0x0000_FFFF_0000_FFFF_0000_FFFF_0000_FFFF;
    (wide | wide << 8) & 0x00FF_00FF_00FF_00FF_00FF_00FF_00FF_00FF
}

/// The levels of one channel at 8 voxels of a column from `shift` up, one in
/// each byte. Only the voxels set in `open` get a level.
#[inline]
fn levels(channel: &ChannelLight, column: usize, shift: usize, open: usize) -> u64 {
    match channel {
        ChannelLight::Uniform(level) => SPREAD[open] * *level as u64,
        ChannelLight::Planes(planes) => {
            let mut texels = 0;
            for (bit, plane) in planes.iter().enumerate() {
                let levels = (plane[column] >> shift) as usize & open;
                texels |= SPREAD[levels] << bit;
            }
            texels
        }
    }
}

/// The cell of a slot's shell in front of the chunk's voxel `face.voxel(u,
/// v)`, as x, y, z in the slot's padded coordinates.
pub(crate) fn shell_cell(face: Face, u: usize, v: usize) -> UVec3 {
    let (x, y, z) = face.voxel(u, v);
    let mut cell = UVec3::new(x as u32 + 1, y as u32 + 1, z as u32 + 1);
    let axis = face as usize / 2;
    cell[axis] = if face as usize & 1 == 1 {
        SIDE as u32 - 1
    } else {
        0
    };
    cell
}

/// The index of the texel of the cell at `cell` in a slot, in padded
/// coordinates.
pub(crate) fn cell_index(cell: UVec3) -> usize {
    cell.z as usize + cell.x as usize * SIDE + cell.y as usize * SIDE * SIDE
}

/// Writes `plane` into the shell of `bytes` on `face`, or zeros with no plane,
/// both in the format of `precision`.
pub(crate) fn write_shell(
    bytes: &mut [u8],
    face: Face,
    plane: Option<&ShellPlane>,
    precision: LightPrecision,
) {
    let wide = texel_bytes(precision);
    for v in 0..CHUNK {
        for u in 0..CHUNK {
            let to = cell_index(shell_cell(face, u, v)) * wide;
            let from = (u + v * CHUNK) * wide;
            match plane {
                Some(plane) => {
                    bytes[to..to + wide].copy_from_slice(&plane[from..from + wide])
                }
                None => bytes[to..to + wide].fill(0),
            }
        }
    }
}

/// The light a chunk's voxels on `face` hold, as a slot of `precision` holds
/// them: the plane that goes into the shell of the neighbour across `face`.
pub(crate) fn border_plane(
    light: &ChunkLight,
    face: Face,
    precision: LightPrecision,
) -> Box<ShellPlane> {
    let mut plane = vec![0; PLANE_TEXELS * texel_bytes(precision)].into_boxed_slice();
    let occupancy = &light.inputs.occupancy;
    let from = light.precision;
    for v in 0..CHUNK {
        for u in 0..CHUNK {
            let (x, y, z) = face.voxel(u, v);
            if !occupancy.is_solid(x, y, z) {
                let sky = precision.convert(light.sky.at(x, y, z), from);
                let voxel = precision.convert(light.voxel.at(x, y, z), from);
                put_texel(&mut plane, u + v * CHUNK, sky, voxel, precision);
            }
        }
    }
    for_lamps_on(&light.inputs.emissive, face, |u, v, lamp| {
        put_lamp(&mut plane, u + v * CHUNK, lamp.level, precision);
    });
    plane
}

/// What the faces of a chunk whose voxels are `own` read across `face` from
/// the neighbour there, lit by `neighbour`: the neighbour's open border voxels
/// touching one of the chunk's solid voxels, with levels in steps of
/// `precision`.
pub(crate) fn across(
    own: &LightInputs,
    face: Face,
    neighbour: &ChunkLight,
    precision: LightPrecision,
) -> FacesSee {
    let back = face.opposite();
    let solid = rows(|u, v| {
        let (x, y, z) = face.voxel(u, v);
        own.occupancy.is_solid(x, y, z)
    });
    if solid.iter().all(|row| *row == 0) {
        return FacesSee::default();
    }
    let open = rows(|u, v| {
        let (x, y, z) = back.voxel(u, v);
        !neighbour.inputs.occupancy.is_solid(x, y, z)
    });
    let near = spread(solid);
    let mut lamps = [0u32; CHUNK];
    for_lamps_on(&own.emissive, face, |u, v, _| lamps[v] |= 1 << u);
    let near_lamps = spread(lamps);

    let mut seen = FacesSee::default();
    for v in 0..CHUNK {
        let mut cells = open[v] & near[v];
        seen.lamp |= open[v] & near_lamps[v] != 0;
        while cells != 0 {
            let u = cells.trailing_zeros() as usize;
            cells &= cells - 1;
            let (x, y, z) = back.voxel(u, v);
            let level = |channel: &ChannelLight| {
                Seen::One(precision.convert(channel.at(x, y, z), neighbour.precision))
            };
            seen.sky = seen.sky.join(level(&neighbour.sky));
            seen.voxel = seen.voxel.join(level(&neighbour.voxel));
            // Once the chunk needs a slot, the levels its faces see no longer
            // matter.
            if seen.needs_slot() {
                return seen;
            }
        }
    }
    seen
}

/// The most common level per channel among the open voxels touching a solid
/// voxel of the chunk lit by `light`, as sky and voxel light in steps of the
/// light's precision.
pub(crate) fn most_common(light: &ChunkLight) -> (u8, u8) {
    let touching = match &light.inputs.occupancy {
        Occupancy::Mixed(solid) => Some(touching_open(solid)),
        _ => None,
    };
    let most_common_level = |channel: &ChannelLight| match (channel, &touching) {
        (ChannelLight::Uniform(level), _) => *level,
        (ChannelLight::Planes(_), Some(touching)) => {
            let mut counts = [0u32; 256];
            for (column, &these) in touching.iter().enumerate() {
                let mut voxels = these;
                while voxels != 0 {
                    let z = voxels.trailing_zeros() as usize;
                    voxels &= voxels - 1;
                    let level = channel.at(column % CHUNK, column / CHUNK, z);
                    counts[level as usize] += 1;
                }
            }
            (0..=light.precision.top() as usize)
                .max_by_key(|level| counts[*level])
                .unwrap_or(0) as u8
        }
        (ChannelLight::Planes(_), None) => 0,
    };
    (
        most_common_level(&light.sky),
        most_common_level(&light.voxel),
    )
}

/// A layer of a chunk's voxels as 32 rows of bits: bit `u` of row `v`.
fn rows(is_set: impl Fn(usize, usize) -> bool) -> [u32; CHUNK] {
    std::array::from_fn(|v| (0..CHUNK).fold(0, |row, u| row | (is_set(u, v) as u32) << u))
}

/// Each set cell of `rows` and the cells touching it by an edge or a corner.
fn spread(rows: [u32; CHUNK]) -> [u32; CHUNK] {
    let wide = rows.map(|row| row | row << 1 | row >> 1);
    std::array::from_fn(|v| {
        wide[v]
            | if v > 0 { wide[v - 1] } else { 0 }
            | wide.get(v + 1).copied().unwrap_or(0)
    })
}

/// Calls `found` with `u`, `v` and the voxel for each of `emissive` lying on
/// the chunk's `face`.
fn for_lamps_on(
    emissive: &[EmissiveVoxel],
    face: Face,
    mut found: impl FnMut(usize, usize, &EmissiveVoxel),
) {
    let side = if face as usize & 1 == 1 { CHUNK - 1 } else { 0 };
    for lamp in emissive {
        let (x, y, z) = (lamp.x as usize, lamp.y as usize, lamp.z as usize);
        let (on, u, v) = match face {
            Face::NegX | Face::PosX => (x, y, z),
            Face::NegY | Face::PosY => (y, x, z),
            Face::NegZ | Face::PosZ => (z, x, y),
        };
        if on == side {
            found(u, v, lamp);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_shell_cell_lies_in_front_of_its_border_voxel() {
        for face in Face::ALL {
            for (u, v) in [(0, 0), (5, 17), (31, 31)] {
                let (x, y, z) = face.voxel(u, v);
                let border = UVec3::new(x as u32, y as u32, z as u32) + UVec3::ONE;
                let step = face.offset();
                let front = (border.as_ivec3() + step).as_uvec3();
                assert_eq!(shell_cell(face, u, v), front, "{face:?} {u} {v}");
            }
        }
    }

    #[test]
    fn seen_joins_to_mixed_only_on_two_levels() {
        assert_eq!(Seen::Nothing.join(Seen::One(3)), Seen::One(3));
        assert_eq!(Seen::One(3).join(Seen::One(3)), Seen::One(3));
        assert_eq!(Seen::One(3).join(Seen::One(4)), Seen::Mixed);
        assert_eq!(Seen::Mixed.join(Seen::Nothing), Seen::Mixed);
    }
}
