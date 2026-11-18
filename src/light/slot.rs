//! What goes into a chunk's slot, and whether a chunk needs one.
//!
//! A chunk's faces read light from the open voxels in front of them: its own
//! voxels, and through the slot's shell its six face neighbours' border voxels.
//! A chunk needs a slot unless every open voxel touching one of its solid
//! voxels by a face, an edge or a corner holds one sky light level and one
//! voxel light level. Edges and corners count because a face off the voxel
//! grid, from a mesher other than cubes, reads the voxels around it and not
//! only the one in front.
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

use super::color::ChunkColor;
use super::config::LightPrecision;
use super::faces::Face;
use super::inputs::{EmissiveVoxel, LightInputs, Occupancy};
use super::kernel::Plane;
use super::levels::LightLevels;
use super::solver::{ChunkLevels, ChunkLight};
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

/// What some voxels hold of one kind of light: no voxel at all, one value,
/// or more than one. The value is a level, or a packed light colour.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum Seen<T = u8> {
    #[default]
    Nothing,
    One(T),
    Mixed,
}

impl<T: Copy + PartialEq> Seen<T> {
    pub fn join(self, other: Seen<T>) -> Seen<T> {
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
    /// The packed light colours of these voxels' voxel light, and of the
    /// emissive voxels touching them. Only a coloured world's chunks read
    /// colours; elsewhere it's [`Seen::Nothing`].
    pub color: Seen<u16>,
}

impl FacesSee {
    pub fn join(self, other: FacesSee) -> FacesSee {
        FacesSee {
            sky: self.sky.join(other.sky),
            voxel: self.voxel.join(other.voxel),
            lamp: self.lamp || other.lamp,
            color: self.color.join(other.color),
        }
    }

    /// Whether the chunk needs a slot: its faces see more than one level, an
    /// emissive voxel of its own touches one of these voxels, or it needs a
    /// colour slot, which takes a slot too.
    pub fn needs_slot(&self) -> bool {
        self.lamp
            || self.sky == Seen::Mixed
            || self.voxel == Seen::Mixed
            || self.needs_color_slot()
    }

    /// Whether the chunk needs a colour slot: its faces see more than one
    /// light colour.
    pub fn needs_color_slot(&self) -> bool {
        self.color == Seen::Mixed
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
        color: Seen::Nothing,
    }
}

/// What a solve works out for its chunk's slot off the main thread, for
/// landing to place.
#[derive(Default)]
pub(crate) struct SlotParts {
    /// The bytes of the slot, with the shell at 0, for a chunk whose own
    /// voxels call for a slot: see [`slot_bytes`].
    pub bytes: Option<Box<[u8]>>,
    /// In a coloured world, the light colours of the chunk's own voxels.
    pub own_colors: Option<OwnColors>,
}

/// What a solve in a coloured world works out about the light colours of its
/// chunk's own voxels, so that landing its light doesn't.
pub(crate) struct OwnColors {
    /// The light colours its faces read from its own voxels: see
    /// [`own_colors_seen`].
    pub seen: Seen<u16>,
    /// Its colour slot's bytes, with the shell at 0, when its own voxels see
    /// more than one light colour: see [`color_slot_bytes`].
    pub bytes: Option<Box<[u8]>>,
}

impl OwnColors {
    /// What the chunk lit by `light` reads of its own light colours, where
    /// `colors` gives the packed light colour of each emissive material
    /// index.
    pub fn of(light: &ChunkLight, colors: &[u16]) -> Self {
        let lamp_color = |material: u16| {
            colors
                .get(material as usize)
                .copied()
                .unwrap_or(super::color::WHITE)
        };
        let seen = own_colors_seen(light, lamp_color);
        let bytes = (seen == Seen::Mixed).then(|| color_slot_bytes(light, lamp_color));
        Self { seen, bytes }
    }
}

/// The packed light colours a chunk's faces read from its own voxels, given
/// its `light`: the voxel light of its open voxels touching one of its solid
/// voxels, and each of its emissive voxels touching an open voxel, whose
/// material index `lamp_color` gives the packed colour of.
///
/// Its cost grows with the chunk's columns, and with its voxels where light of several colours reaches it.
pub(crate) fn own_colors_seen(
    light: &ChunkLight,
    lamp_color: impl Fn(u16) -> u16,
) -> Seen<u16> {
    let Occupancy::Mixed(solid) = &light.inputs.occupancy else {
        return Seen::Nothing;
    };
    let seen = colors_seen(light, &touching_open(solid));
    lamps_seen(light, solid, seen, lamp_color)
}

/// The packed light colours of every open voxel's voxel light in the chunk
/// lit by `light`, and of each of its emissive voxels touching an open voxel,
/// as [`own_colors_seen`] gives them.
///
/// Unlike [`own_colors_seen`], it reads open voxels far from any solid too.
pub(crate) fn open_colors_seen(
    light: &ChunkLight,
    lamp_color: impl Fn(u16) -> u16,
) -> Seen<u16> {
    let Some(open) = open_voxels(&light.inputs.occupancy) else {
        return Seen::Nothing;
    };
    let seen = colors_seen(light, &open);
    match &light.inputs.occupancy {
        Occupancy::Mixed(solid) => lamps_seen(light, solid, seen, lamp_color),
        _ => seen,
    }
}

/// `seen` joined with the packed light colour of each emissive voxel of
/// `light` touching an open voxel, a chunk whose solid voxels are `solid`.
fn lamps_seen(
    light: &ChunkLight,
    solid: &Plane,
    mut seen: Seen<u16>,
    lamp_color: impl Fn(u16) -> u16,
) -> Seen<u16> {
    for lamp in light.inputs.emissive.iter() {
        if seen == Seen::Mixed {
            break;
        }
        if touches_open(solid, lamp) {
            seen = seen.join(Seen::One(lamp_color(lamp.material())));
        }
    }
    seen
}

/// The packed light colours of the voxel light at the voxels set in
/// `voxels`, of the chunk lit by `light`. A voxel with no voxel light has no
/// colour, so it adds none.
pub(crate) fn colors_seen(light: &ChunkLight, voxels: &Plane) -> Seen<u16> {
    let lit = lit_voxels(&light.voxel, voxels, None);
    let mut seen = Seen::Nothing;
    for_each_voxel(&lit, |x, y, z| {
        seen = seen.join(Seen::One(light.color.packed_at(x, y, z)));
        // A chunk held in one colour has it at every lit voxel.
        !matches!(light.color, ChunkColor::Mixed(_)) || seen == Seen::Mixed
    });
    seen
}

/// Calls `visit` with each voxel set in `voxels`, as x, y and z, until it
/// returns true.
fn for_each_voxel(voxels: &Plane, mut visit: impl FnMut(usize, usize, usize) -> bool) {
    for (column, &these) in voxels.iter().enumerate() {
        let mut these = these;
        while these != 0 {
            let z = these.trailing_zeros() as usize;
            these &= these - 1;
            if visit(column % CHUNK, column / CHUNK, z) {
                return;
            }
        }
    }
}

/// Adds to `counts` how many of the voxels set in `voxels`, of the chunk lit
/// by `light`, have voxel light of each packed light colour, counting only
/// those at the voxel light level `level` if it's given, a step of the
/// light's precision.
pub(crate) fn count_colors(
    light: &ChunkLight,
    voxels: &Plane,
    level: Option<u8>,
    counts: &mut Vec<(u16, u32)>,
) {
    let lit = lit_voxels(&light.voxel, voxels, level);
    let mut add = |packed: u16, count: u32| match counts
        .iter_mut()
        .find(|(color, _)| *color == packed)
    {
        Some((_, counted)) => *counted += count,
        None => counts.push((packed, count)),
    };
    match &light.color {
        ChunkColor::Mixed(_) => for_each_voxel(&lit, |x, y, z| {
            add(light.color.packed_at(x, y, z), 1);
            false
        }),
        held => {
            let count: u32 = lit.iter().map(|column| column.count_ones()).sum();
            if count > 0 {
                add(held.packed_at(0, 0, 0), count);
            }
        }
    }
}

/// The voxels set in `voxels` that hold voxel light in `levels`: any level
/// above 0, or only `level` if it's given, as columns along z like a
/// [`Plane`].
fn lit_voxels(levels: &ChunkLevels, voxels: &Plane, level: Option<u8>) -> Box<Plane> {
    let mut lit: Box<Plane> = Box::new([0; CHUNK * CHUNK]);
    if level == Some(0) {
        return lit;
    }
    match levels {
        ChunkLevels::Uniform(held) => {
            if *held != 0 && level.is_none_or(|level| level == *held) {
                lit.copy_from_slice(voxels);
            }
        }
        ChunkLevels::Planes(planes) => {
            for (column, lit) in lit.iter_mut().enumerate() {
                if voxels[column] == 0 {
                    continue;
                }
                let at = match level {
                    None => planes.iter().fold(0, |lit, plane| lit | plane[column]),
                    Some(level) => {
                        planes.iter().enumerate().fold(!0, |at, (bit, plane)| {
                            if level >> bit & 1 == 1 {
                                at & plane[column]
                            } else {
                                at & !plane[column]
                            }
                        })
                    }
                };
                *lit = at & voxels[column];
            }
        }
    }
    lit
}

/// The open voxels of a chunk touching one of its `solid` voxels by a face,
/// an edge or a corner, as columns along z like `solid`'s.
pub(crate) fn touching_open(solid: &Plane) -> Box<Plane> {
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
    let (x, y, z) = (lamp.x() as usize, lamp.y() as usize, lamp.z() as usize);
    let around = ((0b111u64 << z) >> 1) as u32;
    (y.saturating_sub(1)..(y + 2).min(CHUNK)).any(|ny| {
        (x.saturating_sub(1)..(x + 2).min(CHUNK))
            .any(|nx| !solid[nx + ny * CHUNK] & around != 0)
    })
}

/// What `levels` hold in the voxels set in `voxels`.
fn seen_over(levels: &ChunkLevels, voxels: &Plane) -> Seen {
    let mut seen = Seen::Nothing;
    match levels {
        ChunkLevels::Uniform(level) => {
            if voxels.iter().any(|column| *column != 0) {
                seen = Seen::One(*level);
            }
        }
        ChunkLevels::Planes(planes) => {
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
        let (x, y, z) = (lamp.x() as usize, lamp.y() as usize, lamp.z() as usize);
        let texel = (z + 1) + (x + 1) * SIDE + (y + 1) * SIDE * SIDE;
        put_lamp(&mut bytes, texel, lamp.level(), precision);
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

/// The levels in `light` at 8 voxels of a column from `shift` up, one in
/// each byte. Only the voxels set in `open` get a level.
#[inline]
fn levels(light: &ChunkLevels, column: usize, shift: usize, open: usize) -> u64 {
    match light {
        ChunkLevels::Uniform(level) => SPREAD[open] * *level as u64,
        ChunkLevels::Planes(planes) => {
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
/// both `wide` bytes a texel: a slot's [`texel_bytes`], or a colour slot's
/// [`COLOR_TEXEL_BYTES`].
pub(crate) fn write_shell(
    bytes: &mut [u8],
    face: Face,
    plane: Option<&ShellPlane>,
    wide: usize,
) {
    match wide {
        1 => write_shell_texels::<1>(bytes, face, plane),
        2 => write_shell_texels::<2>(bytes, face, plane),
        _ => unreachable!("a texel is 1 or 2 bytes"),
    }
}

/// [`write_shell`] for texels of `W` bytes.
///
/// It steps through the shell's cells from the one in front of the face's
/// voxel at `u` and `v` 0, rather than working out each cell with
/// [`shell_cell`].
fn write_shell_texels<const W: usize>(
    bytes: &mut [u8],
    face: Face,
    plane: Option<&ShellPlane>,
) {
    let edge = if face as usize & 1 == 1 { SIDE - 1 } else { 0 };
    // The first cell, and the steps to the next cell along u and along v.
    let (first, along_u, along_v) = match face {
        Face::NegX | Face::PosX => (1 + edge * SIDE + SIDE * SIDE, SIDE * SIDE, 1),
        Face::NegY | Face::PosY => (1 + SIDE + edge * SIDE * SIDE, SIDE, 1),
        Face::NegZ | Face::PosZ => (edge + SIDE + SIDE * SIDE, SIDE, SIDE * SIDE),
    };
    for v in 0..CHUNK {
        for u in 0..CHUNK {
            let to = (first + u * along_u + v * along_v) * W;
            let texel: [u8; W] = match plane {
                Some(plane) => {
                    let from = (u + v * CHUNK) * W;
                    plane[from..from + W].try_into().unwrap()
                }
                None => [0; W],
            };
            bytes[to..to + W].copy_from_slice(&texel);
        }
    }
}

/// The light a chunk's voxels on `face` hold, as a slot of `precision` holds
/// them: the plane that goes into the shell of the neighbour across `face`.
///
/// It reads the levels a column along z at a time, 8 voxels to a `u64`, as
/// [`slot_bytes`] does. Light of another precision is converted voxel by
/// voxel.
pub(crate) fn border_plane(
    light: &ChunkLight,
    face: Face,
    precision: LightPrecision,
) -> Box<ShellPlane> {
    if light.precision != precision {
        return converted_border_plane(light, face, precision);
    }
    let mut plane = vec![0; PLANE_TEXELS * texel_bytes(precision)].into_boxed_slice();
    let occupancy = &light.inputs.occupancy;
    let side = if face as usize & 1 == 1 { CHUNK - 1 } else { 0 };
    // Puts the 8 texels of `sky` and `voxel`, a level in each byte, at
    // `texel` and every `step` texels after it.
    let mut put = |texel: usize, step: usize, sky: u64, voxel: u64| {
        for i in 0..8 {
            let (sky, voxel) = ((sky >> (8 * i)) as u8, (voxel >> (8 * i)) as u8);
            put_texel(&mut plane, texel + i * step, sky, voxel, precision);
        }
    };
    match face {
        Face::NegX | Face::PosX | Face::NegY | Face::PosY => {
            // A column along z on the face is a row of the plane along v.
            for u in 0..CHUNK {
                let (x, y) = match face {
                    Face::NegX | Face::PosX => (side, u),
                    _ => (u, side),
                };
                let column = x + y * CHUNK;
                let open = !occupancy.solid_column(x, y);
                for shift in (0..CHUNK).step_by(8) {
                    let open = (open >> shift) as usize & 0xFF;
                    let sky = levels(&light.sky, column, shift, open);
                    let voxel = levels(&light.voxel, column, shift, open);
                    put(u + shift * CHUNK, CHUNK, sky, voxel);
                }
            }
        }
        Face::NegZ | Face::PosZ => {
            // Each column holds one voxel of the face. A row of the plane is
            // gathered across 32 columns into the bits of a `u32`, as a
            // column is held, and read as one.
            for y in 0..CHUNK {
                let row = y * CHUNK;
                let open = (0..CHUNK).fold(0u32, |open, x| {
                    open | ((!occupancy.solid_column(x, y) >> side) & 1) << x
                });
                let sky = gathered(&light.sky, row, side);
                let voxel = gathered(&light.voxel, row, side);
                for shift in (0..CHUNK).step_by(8) {
                    let open = (open >> shift) as usize & 0xFF;
                    put(
                        row + shift,
                        1,
                        row_levels(&sky, shift, open),
                        row_levels(&voxel, shift, open),
                    );
                }
            }
        }
    }
    for_lamps_on(&light.inputs.emissive, face, |u, v, lamp| {
        put_lamp(&mut plane, u + v * CHUNK, lamp.level(), precision);
    });
    plane
}

/// [`border_plane`] for light of another precision than `precision`.
fn converted_border_plane(
    light: &ChunkLight,
    face: Face,
    precision: LightPrecision,
) -> Box<ShellPlane> {
    let mut plane = vec![0; PLANE_TEXELS * texel_bytes(precision)].into_boxed_slice();
    let from = light.precision;
    for_border_voxels(light, face, false, |texel, x, y, z| {
        let sky = precision.convert(light.sky.at(x, y, z), from);
        let voxel = precision.convert(light.voxel.at(x, y, z), from);
        put_texel(&mut plane, texel, sky, voxel, precision);
    });
    for_lamps_on(&light.inputs.emissive, face, |u, v, lamp| {
        put_lamp(&mut plane, u + v * CHUNK, lamp.level(), precision);
    });
    plane
}

/// The levels of one row of a chunk's voxels across 32 columns, as
/// [`gathered`] gives them.
enum Row {
    /// One level at every voxel.
    Uniform(u8),
    /// Bit planes of the row: bit `x` of plane `b` is bit `b` of the level
    /// in the row's column `x`.
    Planes([u32; 8]),
}

/// The levels of `levels` at bit `z` of the 32 columns from `row` on.
fn gathered(levels: &ChunkLevels, row: usize, z: usize) -> Row {
    match levels {
        ChunkLevels::Uniform(level) => Row::Uniform(*level),
        ChunkLevels::Planes(planes) => {
            let mut bits = [0u32; 8];
            for (bits, plane) in bits.iter_mut().zip(planes.iter()) {
                *bits = (0..CHUNK)
                    .fold(0, |row_bits, x| row_bits | ((plane[row + x] >> z) & 1) << x);
            }
            Row::Planes(bits)
        }
    }
}

/// The levels at 8 voxels of `row` from `shift` up, one in each byte. Only
/// the voxels set in `open` get a level.
#[inline]
fn row_levels(row: &Row, shift: usize, open: usize) -> u64 {
    match row {
        Row::Uniform(level) => SPREAD[open] * *level as u64,
        Row::Planes(bits) => bits.iter().enumerate().fold(0, |texels, (bit, bits)| {
            texels | SPREAD[(*bits >> shift) as usize & open] << bit
        }),
    }
}

/// Calls `visit` with each open voxel of the chunk lit by `light` on `face`,
/// as the texel of its shell plane and its x, y and z, or only with those
/// holding voxel light if `lit`.
///
/// It reads the chunk's voxels a column along z at a time: one column per
/// border voxel on the faces across z, and 32 border voxels per column on
/// the others.
fn for_border_voxels(
    light: &ChunkLight,
    face: Face,
    lit: bool,
    mut visit: impl FnMut(usize, usize, usize, usize),
) {
    let side = if face as usize & 1 == 1 { CHUNK - 1 } else { 0 };
    let lit_in = |column: usize| match &light.voxel {
        _ if !lit => !0,
        ChunkLevels::Uniform(0) => 0,
        ChunkLevels::Uniform(_) => !0,
        ChunkLevels::Planes(planes) => {
            planes.iter().fold(0, |lit, plane| lit | plane[column])
        }
    };
    let occupancy = &light.inputs.occupancy;
    let mut column = |x: usize, y: usize, on_face: u32| {
        let mut voxels = on_face & !occupancy.solid_column(x, y);
        if voxels != 0 {
            voxels &= lit_in(x + y * CHUNK);
        }
        while voxels != 0 {
            let z = voxels.trailing_zeros() as usize;
            voxels &= voxels - 1;
            let (u, v) = match face {
                Face::NegX | Face::PosX => (y, z),
                Face::NegY | Face::PosY => (x, z),
                Face::NegZ | Face::PosZ => (x, y),
            };
            visit(u + v * CHUNK, x, y, z);
        }
    };
    match face {
        Face::NegX | Face::PosX => (0..CHUNK).for_each(|y| column(side, y, !0)),
        Face::NegY | Face::PosY => (0..CHUNK).for_each(|x| column(x, side, !0)),
        Face::NegZ | Face::PosZ => {
            for y in 0..CHUNK {
                for x in 0..CHUNK {
                    column(x, y, 1 << side);
                }
            }
        }
    }
}

/// How many bytes a texel of a colour slot has: one packed light colour, in
/// little-endian order.
pub(crate) const COLOR_TEXEL_BYTES: usize = 2;

/// Writes the packed light colour `packed` into the texel at `texel` of
/// `bytes`, a colour slot or a colour shell plane.
#[inline]
pub(crate) fn put_color(bytes: &mut [u8], texel: usize, packed: u16) {
    let at = texel * COLOR_TEXEL_BYTES;
    bytes[at..at + COLOR_TEXEL_BYTES].copy_from_slice(&packed.to_le_bytes());
}

/// The packed light colour in the texel at `texel` of `bytes`, a colour
/// slot.
#[inline]
pub(crate) fn color_texel_at(bytes: &[u8], texel: usize) -> u16 {
    let at = texel * COLOR_TEXEL_BYTES;
    u16::from_le_bytes([bytes[at], bytes[at + 1]])
}

/// The packed light colours a chunk's colour slot holds for its own voxels,
/// with its shell at 0, given its `light`, in the layout of a slot.
///
/// An open voxel holds the colour of its voxel light, or 0 where it has
/// none. A solid voxel holds 0, apart from each emissive voxel, which holds
/// its own colour: `lamp_color` gives the packed colour of a material index.
///
/// Its cost grows with the chunk's lit voxels.
pub(crate) fn color_slot_bytes(
    light: &ChunkLight,
    lamp_color: impl Fn(u16) -> u16,
) -> Box<[u8]> {
    let mut bytes = vec![0u8; SLOT_TEXELS * COLOR_TEXEL_BYTES].into_boxed_slice();
    if let Some(open) = open_voxels(&light.inputs.occupancy) {
        let lit = lit_voxels(&light.voxel, &open, None);
        for_each_voxel(&lit, |x, y, z| {
            let cell = UVec3::new(x as u32 + 1, y as u32 + 1, z as u32 + 1);
            put_color(&mut bytes, cell_index(cell), light.color.packed_at(x, y, z));
            false
        });
    }
    for lamp in &light.inputs.emissive {
        let cell = UVec3::new(lamp.x() as u32, lamp.y() as u32, lamp.z() as u32) + 1;
        put_color(&mut bytes, cell_index(cell), lamp_color(lamp.material()));
    }
    bytes
}

/// The packed light colours a chunk's voxels on `face` hold, as a colour
/// slot holds them: the colour shell plane that goes into the colour slot of
/// the chunk across `face`. `lamp_color` gives the packed colour of an
/// emissive material index.
pub(crate) fn border_colors(
    light: &ChunkLight,
    face: Face,
    lamp_color: impl Fn(u16) -> u16,
) -> Box<ShellPlane> {
    let mut plane = vec![0; PLANE_TEXELS * COLOR_TEXEL_BYTES].into_boxed_slice();
    // White packs to 0, which the plane already holds.
    let unlit = matches!(light.voxel, ChunkLevels::Uniform(0));
    if !unlit && light.color != ChunkColor::White {
        for_border_voxels(light, face, true, |texel, x, y, z| {
            put_color(&mut plane, texel, light.color.packed_at(x, y, z));
        });
    }
    for_lamps_on(&light.inputs.emissive, face, |u, v, lamp| {
        put_color(&mut plane, u + v * CHUNK, lamp_color(lamp.material()));
    });
    plane
}

/// What the faces of a chunk whose voxels are `own` read across `face` from
/// the neighbour there, lit by `neighbour`: the neighbour's open border voxels
/// touching one of the chunk's solid voxels, with levels in steps of
/// `precision`. With `colored`, it reads their light colours too.
pub(crate) fn across(
    own: &LightInputs,
    face: Face,
    neighbour: &ChunkLight,
    precision: LightPrecision,
    colored: bool,
) -> FacesSee {
    let mut seen = across_levels(own, face, neighbour, precision);
    if colored {
        seen.color = colors_seen(neighbour, &across_voxels(own, face, &neighbour.inputs));
    }
    seen
}

/// The voxels of the neighbour across `face` from a chunk whose voxels are
/// `own` that the chunk's faces read: the neighbour's open border voxels
/// touching one of the chunk's solid voxels, as columns along z like a
/// [`Plane`], in the neighbour's coordinates.
pub(crate) fn across_voxels(
    own: &LightInputs,
    face: Face,
    neighbour: &LightInputs,
) -> Box<Plane> {
    let back = face.opposite();
    let solid = rows(|u, v| {
        let (x, y, z) = face.voxel(u, v);
        own.occupancy.is_solid(x, y, z)
    });
    let mut voxels: Box<Plane> = Box::new([0; CHUNK * CHUNK]);
    if solid.iter().all(|row| *row == 0) {
        return voxels;
    }
    let near = spread(solid);
    for (v, near) in near.iter().enumerate() {
        let mut cells = *near;
        while cells != 0 {
            let u = cells.trailing_zeros() as usize;
            cells &= cells - 1;
            let (x, y, z) = back.voxel(u, v);
            if !neighbour.occupancy.is_solid(x, y, z) {
                voxels[x + y * CHUNK] |= 1 << z;
            }
        }
    }
    voxels
}

/// The open voxels on `face` of a chunk whose voxels are `inputs`, as columns
/// along z like a [`Plane`].
pub(crate) fn border_voxels(inputs: &LightInputs, face: Face) -> Box<Plane> {
    let mut voxels: Box<Plane> = Box::new([0; CHUNK * CHUNK]);
    if let Occupancy::Solid = inputs.occupancy {
        return voxels;
    }
    for v in 0..CHUNK {
        for u in 0..CHUNK {
            let (x, y, z) = face.voxel(u, v);
            if !inputs.occupancy.is_solid(x, y, z) {
                voxels[x + y * CHUNK] |= 1 << z;
            }
        }
    }
    voxels
}

/// What the faces of a chunk whose voxels are `own` read across `face` of
/// the levels of `neighbour`, as [`across`] gives them.
fn across_levels(
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
            let level = |levels: &ChunkLevels| {
                Seen::One(precision.convert(levels.at(x, y, z), neighbour.precision))
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

/// The most common sky light level and voxel light level among the open
/// voxels touching a solid voxel of the chunk lit by `light`, as sky and voxel
/// light in steps of the light's precision.
pub(crate) fn most_common(light: &ChunkLight) -> (u8, u8) {
    let touching = match &light.inputs.occupancy {
        Occupancy::Mixed(solid) => Some(touching_open(solid)),
        _ => None,
    };
    let level = |levels| most_common_over(levels, touching.as_deref(), light.precision);
    (level(&light.sky), level(&light.voxel))
}

/// The open voxels of a chunk of `occupancy`, as columns along z like a
/// [`Plane`], or `None` for a chunk with none.
pub(crate) fn open_voxels(occupancy: &Occupancy) -> Option<Box<Plane>> {
    match occupancy {
        Occupancy::Open => Some(Box::new([!0; CHUNK * CHUNK])),
        Occupancy::Solid => None,
        Occupancy::Mixed(solid) => Some(Box::new(solid.map(|column| !column))),
    }
}

/// What every open voxel of the chunk lit by `light` holds, as sky light and
/// voxel light. A chunk with no open voxel holds [`Seen::Nothing`].
///
/// Unlike [`own_faces_see`], it reads open voxels far from any solid too.
pub(crate) fn open_voxels_see(light: &ChunkLight) -> (Seen, Seen) {
    let Some(open) = open_voxels(&light.inputs.occupancy) else {
        return (Seen::Nothing, Seen::Nothing);
    };
    (seen_over(&light.sky, &open), seen_over(&light.voxel, &open))
}

/// What the open voxels on `face` of the chunk lit by `light` hold, as sky
/// light and voxel light in steps of `precision`: what the shell plane of
/// the chunk across `face` holds in front of them. A face with no open voxel
/// holds [`Seen::Nothing`].
pub(crate) fn border_sees(
    light: &ChunkLight,
    face: Face,
    precision: LightPrecision,
) -> (Seen, Seen) {
    let occupancy = &light.inputs.occupancy;
    let (mut sky, mut voxel) = (Seen::Nothing, Seen::Nothing);
    if let Occupancy::Solid = occupancy {
        return (sky, voxel);
    }
    let level = |levels: &ChunkLevels, (x, y, z)| {
        Seen::One(precision.convert(levels.at(x, y, z), light.precision))
    };
    for v in 0..CHUNK {
        for u in 0..CHUNK {
            let voxel_at = face.voxel(u, v);
            let (x, y, z) = voxel_at;
            if occupancy.is_solid(x, y, z) {
                continue;
            }
            sky = sky.join(level(&light.sky, voxel_at));
            voxel = voxel.join(level(&light.voxel, voxel_at));
            if sky == Seen::Mixed && voxel == Seen::Mixed {
                return (sky, voxel);
            }
        }
    }
    (sky, voxel)
}

/// The most common sky light level and voxel light level among every open
/// voxel of the chunk lit by `light`, as sky and voxel light in steps of the
/// light's precision.
///
/// Unlike [`most_common`], it counts open voxels far from any solid too, so
/// an open chunk lit voxel by voxel gets its own most common level and not 0.
/// Its cost grows with the chunk's open voxels.
pub(crate) fn most_common_open(light: &ChunkLight) -> (u8, u8) {
    let open = open_voxels(&light.inputs.occupancy);
    let level = |levels| most_common_over(levels, open.as_deref(), light.precision);
    (level(&light.sky), level(&light.voxel))
}

/// The most common packed light colour among the open voxels of the chunk
/// lit by `light` whose voxel light is at `level`, a step of the light's
/// precision. White where none is.
///
/// Its cost grows with the chunk's open voxels.
#[cfg(feature = "sub_grids")]
pub(crate) fn most_common_open_color(light: &ChunkLight, level: u8) -> u16 {
    use super::color::WHITE;
    let Some(open) = open_voxels(&light.inputs.occupancy) else {
        return WHITE;
    };
    let mut counts = Vec::new();
    count_colors(light, &open, Some(level), &mut counts);
    counts
        .into_iter()
        .max_by_key(|(color, count)| (*count, std::cmp::Reverse(*color)))
        .map_or(WHITE, |(color, _)| color)
}

/// The most common level of `levels`, steps of `precision`, among the voxels
/// set in `voxels`. With no voxels, levels held as planes give 0.
fn most_common_over(
    levels: &ChunkLevels,
    voxels: Option<&Plane>,
    precision: LightPrecision,
) -> u8 {
    match (levels, voxels) {
        (ChunkLevels::Uniform(level), _) => *level,
        (ChunkLevels::Planes(_), Some(voxels)) => {
            let mut counts = [0u32; 256];
            for (column, &these) in voxels.iter().enumerate() {
                let mut voxels = these;
                while voxels != 0 {
                    let z = voxels.trailing_zeros() as usize;
                    voxels &= voxels - 1;
                    let level = levels.at(column % CHUNK, column / CHUNK, z);
                    counts[level as usize] += 1;
                }
            }
            (0..=precision.top() as usize)
                .max_by_key(|level| counts[*level])
                .unwrap_or(0) as u8
        }
        (ChunkLevels::Planes(_), None) => 0,
    }
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
        let (x, y, z) = (lamp.x() as usize, lamp.y() as usize, lamp.z() as usize);
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

    use std::sync::Arc;

    use crate::light::sky::ColumnAnswer;

    /// The light of a chunk of `occupancy` at 16 levels whose sky light is
    /// `sky` and whose voxel light is `voxel` everywhere.
    fn light(occupancy: Occupancy, sky: ChunkLevels, voxel: u8) -> ChunkLight {
        ChunkLight {
            inputs: Arc::new(LightInputs {
                occupancy,
                emissive: Default::default(),
                columns: ColumnAnswer::none(),
                ring: false,
            }),
            sky,
            voxel: ChunkLevels::Uniform(voxel),
            color: Default::default(),
            borders: Default::default(),
            precision: LightPrecision::Steps16,
        }
    }

    /// Levels at 16 steps holding `level` at every voxel, as planes.
    fn planes_of(level: u8) -> Box<[Plane]> {
        (0..4)
            .map(|bit| [if level >> bit & 1 == 1 { !0 } else { 0 }; CHUNK * CHUNK])
            .collect()
    }

    /// Sets the voxel at `x`, `y`, `z` of `planes` to `level`.
    fn set_level(planes: &mut [Plane], (x, y, z): (usize, usize, usize), level: u8) {
        for (bit, plane) in planes.iter_mut().enumerate() {
            let column = &mut plane[x + y * CHUNK];
            *column = *column & !(1 << z) | ((level >> bit & 1) as u32) << z;
        }
    }

    /// A number for the voxel at `x`, `y`, `z` that looks random.
    fn scatter(x: usize, y: usize, z: usize) -> u32 {
        let h = (x as u32).wrapping_mul(0x8da6_b343)
            ^ (y as u32).wrapping_mul(0xd816_3841)
            ^ (z as u32).wrapping_mul(0xcb1a_b31f);
        h ^ h >> 13
    }

    /// A chunk of scattered solids, with lamps on two of its faces, whose
    /// open voxels hold scattered levels of sky light and voxel light, and
    /// scattered light colours.
    fn scattered() -> ChunkLight {
        let lamps = [(0, 3, 4), (5, 6, 31)];
        let mut occupancy = Occupancy::Open;
        let (mut sky, mut voxel) = (planes_of(0), planes_of(0));
        let mut colors = vec![0; CHUNK * CHUNK * CHUNK];
        for (x, y, z) in (0..CHUNK).flat_map(|z| {
            (0..CHUNK).flat_map(move |y| (0..CHUNK).map(move |x| (x, y, z)))
        }) {
            let h = scatter(x, y, z);
            if h.is_multiple_of(3) || lamps.contains(&(x, y, z)) {
                occupancy.set(x, y, z, true);
                continue;
            }
            set_level(&mut sky, (x, y, z), (h >> 4) as u8 & 15);
            let level = (h >> 8) as u8 & 15;
            set_level(&mut voxel, (x, y, z), level);
            if level > 0 {
                colors[crate::light::color::voxel_index(x, y, z)] = (h >> 12) as u16 | 1;
            }
        }
        occupancy.normalize();
        let emissive: Vec<EmissiveVoxel> = lamps
            .iter()
            .enumerate()
            .map(|(material, &(x, y, z))| {
                let emitter = crate::light::color::Emitter {
                    level: 12,
                    material: material as u16,
                };
                EmissiveVoxel::new(x as u8, y as u8, z as u8, emitter)
            })
            .collect();
        ChunkLight {
            inputs: Arc::new(LightInputs {
                occupancy,
                emissive: emissive.into(),
                columns: ColumnAnswer::none(),
                ring: false,
            }),
            sky: ChunkLevels::Planes(sky),
            voxel: ChunkLevels::Planes(voxel),
            color: ChunkColor::Mixed(colors.into()),
            borders: Default::default(),
            precision: LightPrecision::Steps16,
        }
    }

    #[test]
    fn a_shell_plane_lands_in_the_cells_in_front_of_its_face() {
        for face in Face::ALL {
            for wide in [1, 2] {
                let plane: Vec<u8> = (0..PLANE_TEXELS * wide)
                    .map(|i| (i % 251) as u8 + 1)
                    .collect();
                let mut bytes = vec![0u8; SLOT_TEXELS * wide];
                write_shell(&mut bytes, face, Some(&plane), wide);
                for v in 0..CHUNK {
                    for u in 0..CHUNK {
                        let to = cell_index(shell_cell(face, u, v)) * wide;
                        let from = (u + v * CHUNK) * wide;
                        assert_eq!(
                            bytes[to..to + wide],
                            plane[from..from + wide],
                            "{face:?}"
                        );
                    }
                }
                let written = bytes.iter().filter(|byte| **byte != 0).count();
                assert_eq!(
                    written,
                    PLANE_TEXELS * wide,
                    "{face:?} wrote only its shell"
                );
                write_shell(&mut bytes, face, None, wide);
                assert!(bytes.iter().all(|byte| *byte == 0), "{face:?}");
            }
        }
    }

    #[test]
    fn a_white_or_one_colour_chunks_border_colours_hold_its_colour_at_lit_voxels_and_its_lamps_own(
    ) {
        let scattered = scattered();
        let lamp_color = |material: u16| 0x7000 + material;
        let orange = 0x1234;
        for (color, lit) in [(ChunkColor::White, 0), (ChunkColor::One(orange), orange)] {
            let light = ChunkLight {
                inputs: scattered.inputs.clone(),
                sky: scattered.sky.clone(),
                voxel: scattered.voxel.clone(),
                color,
                borders: Default::default(),
                precision: scattered.precision,
            };
            for face in Face::ALL {
                let colors = border_colors(&light, face, lamp_color);
                for v in 0..CHUNK {
                    for u in 0..CHUNK {
                        let (x, y, z) = face.voxel(u, v);
                        let lamp = light.inputs.emissive.iter().find(|lamp| {
                            (lamp.x() as usize, lamp.y() as usize, lamp.z() as usize)
                                == (x, y, z)
                        });
                        let expected = if let Some(lamp) = lamp {
                            lamp_color(lamp.material())
                        } else if light.inputs.occupancy.is_solid(x, y, z)
                            || (scatter(x, y, z) >> 8) & 15 == 0
                        {
                            0
                        } else {
                            lit
                        };
                        let texel = color_texel_at(&colors, u + v * CHUNK);
                        assert_eq!(texel, expected, "{face:?} at {x}, {y}, {z}");
                    }
                }
            }
        }
    }

    #[test]
    fn border_planes_hold_the_light_and_colours_of_each_border_voxel() {
        let light = scattered();
        let lamp_color = |material: u16| 0x7000 + material;
        for face in Face::ALL {
            let plane = border_plane(&light, face, LightPrecision::Steps16);
            let colors = border_colors(&light, face, lamp_color);
            for v in 0..CHUNK {
                for u in 0..CHUNK {
                    let (x, y, z) = face.voxel(u, v);
                    let texel = u + v * CHUNK;
                    let lamp = light.inputs.emissive.iter().find(|lamp| {
                        (lamp.x() as usize, lamp.y() as usize, lamp.z() as usize)
                            == (x, y, z)
                    });
                    let h = scatter(x, y, z);
                    let (levels, color) = if let Some(lamp) = lamp {
                        (12 | 15 << 4, lamp_color(lamp.material()))
                    } else if light.inputs.occupancy.is_solid(x, y, z) {
                        (0, 0)
                    } else {
                        let voxel = (h >> 8) & 15;
                        let color = if voxel > 0 { (h >> 12) as u16 | 1 } else { 0 };
                        ((h >> 4 & 15 | voxel << 4) as u8, color)
                    };
                    let at = format!("{face:?} at {x}, {y}, {z}");
                    assert_eq!(plane[texel], levels, "{at}");
                    assert_eq!(color_texel_at(&colors, texel), color, "{at}");
                }
            }
        }
    }

    /// One solid voxel at the chunk's middle.
    fn one_solid() -> Occupancy {
        let mut occupancy = Occupancy::Open;
        occupancy.set(16, 16, 16, true);
        occupancy.normalize();
        occupancy
    }

    #[test]
    fn a_chunk_whose_open_voxels_hold_one_level_reports_it() {
        let held_whole = light(Occupancy::Open, ChunkLevels::Uniform(9), 4);
        assert_eq!(open_voxels_see(&held_whole), (Seen::One(9), Seen::One(4)));
        assert_eq!(most_common_open(&held_whole), (9, 4));

        for occupancy in [Occupancy::Open, one_solid()] {
            let planes = light(occupancy, ChunkLevels::Planes(planes_of(9)), 4);
            assert_eq!(open_voxels_see(&planes), (Seen::One(9), Seen::One(4)));
            assert_eq!(most_common_open(&planes), (9, 4));
        }
    }

    #[test]
    fn one_open_voxel_at_another_level_makes_the_chunk_mixed() {
        let mut planes = planes_of(9);
        set_level(&mut planes, (3, 30, 7), 8);
        let light = light(one_solid(), ChunkLevels::Planes(planes), 4);

        assert_eq!(open_voxels_see(&light), (Seen::Mixed, Seen::One(4)));
        assert_eq!(most_common_open(&light).0, 9);
    }

    #[test]
    fn the_most_common_open_level_counts_voxels_far_from_solids() {
        // The 26 open voxels around the solid one hold 5, and every other
        // open voxel holds 9.
        let mut planes = planes_of(9);
        for z in 15..18 {
            for y in 15..18 {
                for x in 15..18 {
                    set_level(&mut planes, (x, y, z), 5);
                }
            }
        }
        let light = light(one_solid(), ChunkLevels::Planes(planes), 0);

        assert_eq!(most_common(&light).0, 5);
        assert_eq!(most_common_open(&light).0, 9);
    }

    #[test]
    fn a_solid_chunk_has_no_open_voxel_to_report() {
        let light = light(Occupancy::Solid, ChunkLevels::Planes(planes_of(9)), 0);
        assert_eq!(open_voxels_see(&light), (Seen::Nothing, Seen::Nothing));
    }

    #[test]
    fn seen_joins_to_mixed_only_on_two_levels() {
        assert_eq!(Seen::Nothing.join(Seen::One(3)), Seen::One(3));
        assert_eq!(Seen::One(3).join(Seen::One(3)), Seen::One(3));
        assert_eq!(Seen::One(3).join(Seen::One(4)), Seen::Mixed);
        assert_eq!(Seen::<u8>::Mixed.join(Seen::Nothing), Seen::Mixed);
    }
}
