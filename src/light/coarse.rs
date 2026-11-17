//! Light in coarse chunks: chunks above LOD level 0, and level-0 chunks whose
//! `chunk_data_shape` isn't the default, so that their data voxels aren't
//! world voxels.
//!
//! A coarse chunk works out its light once, in its own chunk task, from its
//! own 32³ cells and their padding, each cell `1 << L` voxels wide at level
//! `L`. It runs the same kernel as a level-0 chunk on those cells, reads no
//! neighbour and gives no neighbour anything to read. So **a coarse chunk's
//! light never relights**: it changes only when the chunk task runs again,
//! and a voxel edit outside the chunk and its padding doesn't reach it until
//! then. It takes no part in the world view, the sky record or sky fill.
//!
//! Two changes of settings solve coarse chunks again from their published
//! voxels on the task pool, and
//! [`light_is_settled`](crate::prelude::VoxelWorld::light_is_settled) waits
//! for those solves. A change of precision solves every coarse chunk. A change
//! to the emissive list solves the coarse chunks holding a material whose
//! entry was added, dropped, or given a new level or a new scaled light
//! colour, in a white world too. For that, each solve notes
//! the solid materials of its cells, emissive or not, which costs a solve a
//! little more for each material its cells hold. Old light shows until new
//! light lands. No other change of settings reaches a coarse chunk.
//!
//! A cell is the data voxel at its middle. Where data voxels are finer than
//! cells, a cell is open if any data voxel in it is open.
//!
//! Sky light falls into a column of cells from the chunk's top when the
//! chunk's own cells above it are open and a walk above the chunk finds the
//! column clear (see [`CoarseWalk`]). Light spreads from there and from the
//! emissive cells as at level 0, with its reach counted in voxels, so a pool
//! of light keeps its size in voxels as the level rises (see
//! [`LevelScale::coarse`]).
//!
//! The light is plausible, not exact. A lamp, or a shaft one voxel wide,
//! shows only if a cell's sample lands on it, and the underside of a floating
//! solid goes black in the middle, where only sky fill would light it.
//!
//! A coarse chunk's slot holds its own cells, and its shell holds its own
//! padding cells' light rather than its neighbours' border planes. The tag of
//! a chunk with a slot carries its LOD level in bits 24–27, which the shader
//! divides the chunk's local positions by.
//!
//! # Light colour
//!
//! A coarse chunk with an emissive cell of a light colour other than white
//! spreads voxel light once per light colour among its own emissive cells,
//! white included, and mixes the runs cell by cell as level 0 does (see
//! [`color`](super::color)). Materials of one colour share a run, so the
//! cost grows with the light colours among its emissive cells. Since all of
//! them are its own, every colour reaches it and none is skipped. A chunk
//! whose emissive cells all give white solves as it would in a white world.
//!
//! It takes a colour slot when the cells its faces read for levels, and its
//! emissive cells touching an open cell, see more than one light colour.
//! The colour slot's shell holds its own padding cells' colours, with its
//! edges and corners left at 0, and taking one takes a slot too. With a slot
//! and one light colour seen, its entry holds that colour.
//!
//! # Why the fallback takes the brightest tenth's colour
//!
//! A coarse chunk with no slot shows the brightest voxel light level a tenth
//! of the cells its faces read see, and the most common light colour among
//! those cells at that level or brighter. A far chunk then shows the colour
//! of the light that sets its level. Mixing the colours it holds by
//! brightness would turn a red pool and a blue pool magenta, a colour no lamp
//! gives. Counting every lit cell would let a large dim pool outvote the
//! bright one that sets its level. Counting only the cells at exactly that
//! level can pick out a thin ring at another pool's edge. At a level of 0 it
//! shows white. A chunk that keeps its slot but finds no colour slot holds
//! the same colour in its entry.
//!
//! # References
//!
//! - [Cubyz's lighting](https://github.com/PixelGuys/Cubyz/blob/master/src/renderer/lighting.zig):
//!   each LOD level spreads its own light, losing more per cell as cells widen.
//! - [Terasology's batch propagator](https://github.com/MovingBlocks/Terasology/blob/develop/engine/src/main/java/org/terasology/engine/world/propagation/StandardBatchPropagator.java):
//!   spreads light with a scale, which its LOD chunks set so light fades faster
//!   per cell.

use std::cell::RefCell;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::thread::LocalKey;

use bevy::platform::collections::HashMap;
use bevy::platform::time::Instant;
use bevy::prelude::*;

use super::color::{mix_weighted, mixing_weight, pack, unpack, EmissionTable, WHITE};
use super::config::{LightPrecision, LightShape};
use super::faces::Face;
use super::kernel::{LightBox, Plane, Row, Seed, TwoWords, MAX_ONE_WORD_REACH};
use super::levels::LevelScale;
use super::sky::{Columns, SkyRule, WALK_BUDGET};
use super::slot::{
    cell_index, put_color, put_lamp, put_texel, shell_cell, texel_at, texel_bytes, Seen,
    COLOR_TEXEL_BYTES, SLOT_TEXELS,
};
use super::solver::{empty_planes, LightCounters, LightTaskSettings};

use crate::chunk::{
    ChunkData, ChunkKey, FillType, CHUNK_SIZE_I, CHUNK_SIZE_U, PADDED_CHUNK_SIZE,
};
use crate::configuration::{VoxelLookupDelegate, VoxelWorldConfig};
use crate::level_0_reader::{ChunkLookups, Hinted};
use crate::voxel::WorldVoxel;
use crate::voxel_edits::{VoxelEditReader, VoxelEdits};

const CHUNK: usize = CHUNK_SIZE_U as usize;
/// Cells along each side of a coarse chunk with its padding, as in its slot.
const SIDE: usize = CHUNK + 2;
/// A column of cells along z with its padding, bit `z` for the cell at `z`.
const ALL_CELLS: u64 = (1 << SIDE) - 1;

/// Whether the chunk with `key` and the data shape `data_shape` takes the
/// coarse path for its light.
pub(crate) fn takes_coarse_path(key: ChunkKey, data_shape: UVec3) -> bool {
    key.lod_level > 0 || data_shape != UVec3::splat(PADDED_CHUNK_SIZE)
}

/// A coarse chunk's light, as its task worked it out.
pub(crate) struct CoarseSolved<I> {
    /// The sky light and voxel light its tag holds without a slot: the one
    /// level of each that its faces see, or else the brightest level a tenth
    /// of them see.
    pub levels: (u8, u8),
    /// The precision of its levels and of its slot's format.
    pub precision: LightPrecision,
    /// The bytes of its slot, in a world showing its light, if its faces see
    /// more than one level or a lamp.
    pub bytes: Option<Box<[u8]>>,
    /// The light colour it shows.
    pub color: CoarseColor,
    /// The solid materials of the cells that can hold a lamp, emissive or
    /// not, so a change to the emissive list finds the chunk.
    pub materials: Box<[I]>,
    /// The settings it solved under.
    pub settings: Arc<LightTaskSettings<I>>,
}

/// The light colour a coarse chunk shows, as packed light colours. All of it
/// is white in a chunk whose emissive cells all give white.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct CoarseColor {
    /// The colour its tag holds without a slot: the most common light colour
    /// among the cells its faces read at the voxel light level its tag holds
    /// or brighter. White at a voxel light level of 0.
    pub tag: u16,
    /// The colour its entry holds with a slot and no colour slot: the one
    /// light colour its faces see, or else `tag`'s.
    pub entry: u16,
    /// The bytes of its colour slot, in a world showing its light, if its
    /// faces see more than one light colour.
    pub bytes: Option<Box<[u8]>>,
}

/// Whether light solved under `old` differs from light solved under `new` in
/// a coarse chunk holding `materials`: whether the entry of one of them in
/// the emissive list was added, dropped, or given a new level or a new
/// scaled light colour.
pub(crate) fn emission_changed<I: Copy + PartialEq>(
    materials: &[I],
    old: &LightTaskSettings<I>,
    new: &LightTaskSettings<I>,
) -> bool {
    let same_list = old.generation == new.generation
        && old.emission_generation == new.emission_generation
        && old.color_generation == new.color_generation;
    if same_list {
        return false;
    }
    let entry = |settings: &LightTaskSettings<I>, material: I| {
        let emission = &settings.emission;
        emission
            .get(material)
            .map(|emitter| (emitter.level, emission.color(emitter.material)))
    };
    materials
        .iter()
        .any(|material| entry(old, *material) != entry(new, *material))
}

/// A coarse chunk task's part in solving light.
pub(crate) struct CoarseTask<I> {
    pub settings: Arc<LightTaskSettings<I>>,
    /// The walk above the chunk, with sky light on.
    pub walk: Option<CoarseWalk<I>>,
}

impl<I: Copy + PartialEq> CoarseTask<I> {
    /// Works out the light of the coarse `chunk` from its own cells.
    pub fn solve(self, chunk: &ChunkData<I>) -> CoarseSolved<I> {
        let started = Instant::now();
        let settings = &self.settings;
        let width = 1u32 << chunk.key.lod_level;
        let precision = settings.precision;
        let cells = Cells::of(chunk, &settings.emission);

        let lit = match (self.walk, &settings.sky_scale) {
            (Some(walk), Some(_)) => {
                let answer = walk.answer(&cells.top_open(), settings);
                Some(cells.direct_sky(&answer))
            }
            _ => None,
        };
        let sky = match (&lit, &settings.sky_scale) {
            (Some(lit), Some(scale)) if lit.iter().any(|column| *column != 0) => {
                if lit == &cells.open {
                    lit_levels(lit, precision.top())
                } else {
                    let scale = LevelScale::coarse(scale.reach(), width, precision);
                    spread(&cells.open, Some(lit), &[], &scale, settings.shape)
                }
            }
            _ => vec![0; SLOT_TEXELS],
        };
        let counters = &settings.counters;
        let colors = cells.light_colors();
        let (voxel, cell_colors) = if cells.lamps.is_empty() {
            (vec![0; SLOT_TEXELS], None)
        } else {
            let scale =
                LevelScale::coarse(settings.voxel_scale.reach(), width, precision);
            if colors.is_empty() {
                let voxel =
                    spread(&cells.open, None, &cells.lamps, &scale, settings.shape);
                (voxel, None)
            } else {
                let (voxel, cell_colors) =
                    spread_colors(&cells, &colors, &scale, settings.shape, counters);
                (voxel, Some(cell_colors))
            }
        };

        let bytes = cells.slot_bytes(&sky, &voxel, precision);
        let seen = cells.faces_see(&bytes, precision, cell_colors.as_deref());
        let color = match &cell_colors {
            None => CoarseColor::default(),
            Some(cell_colors) => CoarseColor {
                tag: seen.fallback,
                entry: match seen.color {
                    Seen::One(color) => color,
                    Seen::Mixed => seen.fallback,
                    Seen::Nothing => WHITE,
                },
                bytes: (settings.routed && seen.color == Seen::Mixed)
                    .then(|| cells.color_slot_bytes(cell_colors)),
            },
        };
        counters.coarse_solves.fetch_add(1, Ordering::Relaxed);
        counters
            .coarse_nanos
            .fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
        CoarseSolved {
            levels: seen.levels,
            precision,
            bytes: (settings.routed && seen.needs_slot).then_some(bytes),
            color,
            materials: cells.materials.into_boxed_slice(),
            settings: self.settings,
        }
    }
}

/// An emissive cell of a coarse chunk.
#[derive(Clone, Copy)]
struct Lamp {
    /// Where it is, in padded cell coordinates.
    cell: UVec3,
    /// Its whole level.
    level: u8,
    /// Its packed light colour.
    color: u16,
}

/// What a coarse chunk's faces read from its cells.
struct CoarseFacesSee {
    /// The sky light and voxel light its tag holds without a slot.
    levels: (u8, u8),
    needs_slot: bool,
    /// The packed light colours of the cells they read with voxel light, and
    /// of its emissive cells touching an open cell, or [`Seen::Nothing`] in a
    /// chunk whose emissive cells all give white.
    color: Seen<u16>,
    /// The light colour its tag holds without a slot: see
    /// [`CoarseColor::tag`].
    fallback: u16,
}

/// A coarse chunk's cells: which are open, its emissive cells, and the
/// materials of the cells that could be emissive.
struct Cells<I> {
    /// A column along z per cell `x`, `y`, at `x + y × 34`, padding included.
    open: Vec<u64>,
    lamps: Vec<Lamp>,
    /// Each material of a solid cell, once. A chunk solid through its
    /// padding holds none, since it lights no cell whatever its material.
    materials: Vec<I>,
}

impl<I: Copy + PartialEq> Cells<I> {
    fn of(chunk: &ChunkData<I>, emission: &EmissionTable<I>) -> Self {
        let uniform = |open: bool| Self {
            open: vec![if open { ALL_CELLS } else { 0 }; SIDE * SIDE],
            lamps: Vec::new(),
            materials: Vec::new(),
        };
        let voxels = match (&chunk.fill_type, &chunk.voxels) {
            (FillType::Empty, _) => return uniform(true),
            // A chunk whose padding is solid too lights no cell.
            (FillType::Uniform(voxel), _) => return uniform(!voxel.is_solid()),
            (FillType::Mixed, Some(voxels)) => voxels,
            (FillType::Mixed, None) => unreachable!("a mixed chunk stores its voxels"),
        };
        let shape = chunk.data_shape;
        let lod_level = chunk.key.lod_level;
        let ranges: [Vec<CellRange>; 3] = std::array::from_fn(|axis| {
            (0..SIDE)
                .map(|cell| CellRange::of(cell, shape[axis], lod_level))
                .collect()
        });
        let (row, slab) = (shape.x as usize, (shape.x * shape.y) as usize);
        let at = |x: u32, y: u32, z: u32| {
            voxels[x as usize + y as usize * row + z as usize * slab]
        };
        let mut open = vec![0; SIDE * SIDE];
        let mut lamps = Vec::new();
        let mut materials = Vec::new();
        // Neighbouring cells mostly hold the same material, so the last one
        // and what it gives off are kept to skip looking them up again.
        let mut last: Option<(I, Option<(u8, u16)>)> = None;
        for z in 0..SIDE {
            let rz = &ranges[2][z];
            for y in 0..SIDE {
                let ry = &ranges[1][y];
                for x in 0..SIDE {
                    let rx = &ranges[0][x];
                    let middle = at(rx.middle, ry.middle, rz.middle);
                    let is_open = if rx.single() && ry.single() && rz.single() {
                        !middle.is_solid()
                    } else {
                        (rz.low..=rz.high).any(|dz| {
                            (ry.low..=ry.high).any(|dy| {
                                (rx.low..=rx.high).any(|dx| !at(dx, dy, dz).is_solid())
                            })
                        })
                    };
                    if is_open {
                        open[x + y * SIDE] |= 1 << z;
                    } else if let WorldVoxel::Solid(material) = middle {
                        let emits = match last {
                            Some((held, emits)) if held == material => emits,
                            _ => {
                                if !materials.contains(&material) {
                                    materials.push(material);
                                }
                                let emits = emission.get(material).map(|emitter| {
                                    (emitter.level, emission.color(emitter.material))
                                });
                                last = Some((material, emits));
                                emits
                            }
                        };
                        if let Some((level, color)) = emits {
                            lamps.push(Lamp {
                                cell: UVec3::new(x as u32, y as u32, z as u32),
                                level,
                                color,
                            });
                        }
                    }
                }
            }
        }
        Self {
            open,
            lamps,
            materials,
        }
    }

    fn is_open(&self, x: usize, y: usize, z: usize) -> bool {
        self.open[x + y * SIDE] >> z & 1 == 1
    }

    /// The packed light colours of the emissive cells, each once, in order,
    /// or none if they all give white.
    fn light_colors(&self) -> Vec<u16> {
        if self.lamps.iter().all(|lamp| lamp.color == WHITE) {
            return Vec::new();
        }
        let mut colors: Vec<u16> = Vec::new();
        for lamp in &self.lamps {
            if !colors.contains(&lamp.color) {
                colors.push(lamp.color);
            }
        }
        colors.sort_unstable();
        colors
    }

    /// The bytes of the chunk's colour slot, given `colors`, the packed light
    /// colour of each cell's voxel light in slot order: those colours, each
    /// emissive cell's own colour, and the shell's edges and corners left at
    /// 0.
    fn color_slot_bytes(&self, colors: &[u16]) -> Box<[u8]> {
        let mut bytes = vec![0; SLOT_TEXELS * COLOR_TEXEL_BYTES].into_boxed_slice();
        for (texel, color) in colors.iter().enumerate() {
            if *color != WHITE {
                put_color(&mut bytes, texel, *color);
            }
        }
        for lamp in &self.lamps {
            put_color(&mut bytes, cell_index(lamp.cell), lamp.color);
        }
        clear_shell_edges(&mut bytes, COLOR_TEXEL_BYTES);
        bytes
    }

    /// The interior columns whose cell in the padding row above the chunk is
    /// open.
    fn top_open(&self) -> Columns {
        let mut top = [0; CHUNK];
        for (x, column) in top.iter_mut().enumerate() {
            let row = self.open[x + 1 + (SIDE - 1) * SIDE];
            *column = (row >> 1) as u32;
        }
        top
    }

    /// The cells with direct sky, padding included, given which interior
    /// columns the walk found clear. A padding column takes the answer of the
    /// interior column beside it.
    fn direct_sky(&self, clear: &Columns) -> Vec<u64> {
        let mut lit = vec![0; SIDE * SIDE];
        for z in 0..SIDE {
            let iz = z.clamp(1, CHUNK) - 1;
            for x in 0..SIDE {
                let ix = x.clamp(1, CHUNK) - 1;
                if clear[ix] >> iz & 1 == 0 {
                    continue;
                }
                for y in (0..SIDE).rev() {
                    if !self.is_open(x, y, z) {
                        break;
                    }
                    lit[x + y * SIDE] |= 1 << z;
                }
            }
        }
        lit
    }

    /// The bytes of the chunk's slot, in the format of `precision`: each
    /// cell's sky light and voxel light, a lamp marker in each emissive cell,
    /// and the shell's edges and corners left at 0.
    fn slot_bytes(
        &self,
        sky: &[u8],
        voxel: &[u8],
        precision: LightPrecision,
    ) -> Box<[u8]> {
        let wide = texel_bytes(precision);
        let mut bytes = vec![0; SLOT_TEXELS * wide].into_boxed_slice();
        for (texel, (sky, voxel)) in sky.iter().zip(voxel).enumerate() {
            put_texel(&mut bytes, texel, *sky, *voxel, precision);
        }
        for lamp in &self.lamps {
            put_lamp(&mut bytes, cell_index(lamp.cell), lamp.level, precision);
        }
        clear_shell_edges(&mut bytes, wide);
        bytes
    }

    /// What the chunk's faces read from `bytes`, a slot of `precision`'s
    /// format, and from `colors`, the packed light colour of each cell's
    /// voxel light in slot order, given in a chunk whose emissive cells don't
    /// all give white.
    ///
    /// Faces read the open cells touching one of the chunk's own solid cells
    /// by a face, an edge or a corner, as at level 0, padding included. A
    /// chunk with no such cell reads all its open cells. A chunk whose faces
    /// see more than one light colour needs a slot.
    fn faces_see(
        &self,
        bytes: &[u8],
        precision: LightPrecision,
        colors: Option<&[u16]>,
    ) -> CoarseFacesSee {
        let interior = ALL_CELLS & !1 & !(1 << (SIDE - 1));
        let solid = |x: usize, y: usize| -> u64 {
            if (1..=CHUNK).contains(&x) && (1..=CHUNK).contains(&y) {
                let column = !self.open[x + y * SIDE] & interior;
                column | column << 1 | column >> 1
            } else {
                0
            }
        };
        let mut sky = [0u32; 256];
        let mut voxel = [0u32; 256];
        let mut color = Seen::Nothing;
        // How many cells of each light colour are at each voxel light level.
        let mut by_color: Vec<(u16, Box<[u32; 256]>)> = Vec::new();
        let mut count = |cells: &dyn Fn(usize, usize) -> u64| {
            let mut any = false;
            for y in 0..SIDE {
                for x in 0..SIDE {
                    let mut bits = cells(x, y);
                    while bits != 0 {
                        let z = bits.trailing_zeros();
                        bits &= bits - 1;
                        let cell = cell_index(UVec3::new(x as u32, y as u32, z));
                        let (sky_level, voxel_level) = texel_at(bytes, cell, precision);
                        sky[sky_level as usize] += 1;
                        voxel[voxel_level as usize] += 1;
                        any = true;
                        let Some(colors) = colors.filter(|_| voxel_level > 0) else {
                            continue;
                        };
                        let packed = colors[cell];
                        color = color.join(Seen::One(packed));
                        let at = by_color.iter().position(|(held, _)| *held == packed);
                        let at = at.unwrap_or_else(|| {
                            by_color.push((packed, Box::new([0; 256])));
                            by_color.len() - 1
                        });
                        by_color[at].1[voxel_level as usize] += 1;
                    }
                }
            }
            any
        };
        let touching = |x: usize, y: usize| -> u64 {
            let edge_x = x == 0 || x == SIDE - 1;
            let edge_y = y == 0 || y == SIDE - 1;
            if edge_x && edge_y {
                return 0;
            }
            let mut near = 0;
            for ny in y.saturating_sub(1)..(y + 2).min(SIDE) {
                for nx in x.saturating_sub(1)..(x + 2).min(SIDE) {
                    near |= solid(nx, ny);
                }
            }
            let mut cells = near & self.open[x + y * SIDE] & ALL_CELLS;
            if edge_x || edge_y {
                cells &= interior;
            }
            cells
        };
        if !count(&touching) {
            count(&|x, y| {
                if (1..=CHUNK).contains(&x) && (1..=CHUNK).contains(&y) {
                    self.open[x + y * SIDE] & interior
                } else {
                    0
                }
            });
        }
        let touches_open = |lamp: &&Lamp| {
            let cell = lamp.cell;
            let (x, y, z) = (cell.x as usize, cell.y as usize, cell.z as usize);
            let inside = |v: usize| (1..=CHUNK).contains(&v);
            if !(inside(x) && inside(y) && inside(z)) {
                return false;
            }
            let around = (0b111u64 << z) >> 1;
            (y - 1..=y + 1).any(|ny| {
                (x - 1..=x + 1).any(|nx| self.open[nx + ny * SIDE] & around != 0)
            })
        };
        let mut lamp = false;
        for touching in self.lamps.iter().filter(touches_open) {
            lamp = true;
            if colors.is_none() {
                break;
            }
            color = color.join(Seen::One(touching.color));
        }
        let levels =
            |counts: &[u32; 256]| counts.iter().filter(|count| **count > 0).count();
        let needs_slot =
            lamp || levels(&sky) > 1 || levels(&voxel) > 1 || color == Seen::Mixed;
        let tenth = brightest_tenth(&voxel);
        // The most common colour at the tag's voxel light level or brighter,
        // the lowest packed colour on a tie. At level 0 no cell counts.
        let fallback = by_color
            .iter()
            .map(|(packed, counts)| {
                let count: u32 = counts[tenth as usize..].iter().sum();
                (count, std::cmp::Reverse(*packed))
            })
            .filter(|(count, _)| *count > 0 && tenth > 0)
            .max()
            .map_or(WHITE, |(_, packed)| packed.0);
        CoarseFacesSee {
            levels: (brightest_tenth(&sky), tenth),
            needs_slot,
            color,
            fallback,
        }
    }
}

/// Sets the shell's edges and corners in `bytes`, a slot or a colour slot of
/// `wide` bytes a texel, to 0.
fn clear_shell_edges(bytes: &mut [u8], wide: usize) {
    let edge = |v: usize| v == 0 || v == SIDE - 1;
    for y in 0..SIDE {
        for x in 0..SIDE {
            let zs: &[usize] = match (edge(x), edge(y)) {
                (true, true) => &[],
                (true, false) | (false, true) => &[0, SIDE - 1],
                (false, false) => continue,
            };
            let cell = |z: usize| cell_index(UVec3::new(x as u32, y as u32, z as u32));
            let mut clear =
                |z: usize| bytes[cell(z) * wide..(cell(z) + 1) * wide].fill(0);
            if zs.is_empty() {
                (0..SIDE).for_each(&mut clear);
            } else {
                zs.iter().copied().for_each(&mut clear);
            }
        }
    }
}

/// Spreads voxel light through the chunk's open cells once per light colour
/// in `colors`, each run from that colour's emissive cells alone, as at
/// level 0. Returns each cell's level, the highest of the runs', and the
/// packed light colour of its voxel light, both in slot order, padding
/// included.
///
/// A cell one run lights takes that run's colour, and one that more light
/// takes their colours at their levels, mixed by [`mix_weighted`]. A cell no
/// run lights holds white. The cost grows with the light colours in
/// `colors`.
fn spread_colors<I>(
    cells: &Cells<I>,
    colors: &[u16],
    scale: &LevelScale,
    shape: LightShape,
    counters: &LightCounters,
) -> (Vec<u8>, Vec<u16>) {
    let runs: Vec<Vec<u8>> = colors
        .iter()
        .map(|color| {
            let lamps: Vec<Lamp> = cells
                .lamps
                .iter()
                .filter(|lamp| lamp.color == *color)
                .copied()
                .collect();
            spread(&cells.open, None, &lamps, scale, shape)
        })
        .collect();
    if runs.len() > 1 {
        counters
            .color_runs
            .fetch_add(runs.len() as u64, Ordering::Relaxed);
    }
    let precision = scale.precision();
    let weights: Vec<f32> = (0..=precision.top())
        .map(|steps| mixing_weight(precision.level_of(steps)))
        .collect();
    let unpacked: Vec<LinearRgba> = colors.iter().map(|color| unpack(*color)).collect();
    let mut levels = vec![0; SLOT_TEXELS];
    let mut packed = vec![WHITE; SLOT_TEXELS];
    for texel in 0..SLOT_TEXELS {
        let (mut top, mut lit, mut only) = (0, 0, 0);
        for (run, run_levels) in runs.iter().enumerate() {
            if run_levels[texel] > 0 {
                top = top.max(run_levels[texel]);
                lit += 1;
                only = run;
            }
        }
        if lit == 0 {
            continue;
        }
        levels[texel] = top;
        packed[texel] = if lit == 1 {
            colors[only]
        } else {
            pack(mix_weighted(
                runs.iter()
                    .zip(&unpacked)
                    .filter(|(run_levels, _)| run_levels[texel] > 0)
                    .map(|(run_levels, color)| {
                        (*color, weights[run_levels[texel] as usize])
                    }),
            ))
        };
    }
    (levels, packed)
}

/// The brightest level that, with every brighter one, makes up at least a
/// tenth of `counts`, or 0 with nothing counted.
fn brightest_tenth(counts: &[u32; 256]) -> u8 {
    let total: u32 = counts.iter().sum();
    let mut covered = 0;
    for level in (0..256).rev() {
        covered += counts[level];
        if covered > 0 && covered * 10 >= total {
            return level as u8;
        }
    }
    0
}

/// Which data voxels along one axis make up a cell.
struct CellRange {
    /// The data voxel at the cell's middle.
    middle: u32,
    /// The first and last data voxels inside the cell, which are the middle
    /// one alone unless data voxels are finer than cells.
    low: u32,
    high: u32,
}

impl CellRange {
    /// The data voxels of padded cell `cell`, in a chunk at `lod_level` with
    /// `extent` data voxels along the axis.
    fn of(cell: usize, extent: u32, lod_level: u8) -> Self {
        if extent == PADDED_CHUNK_SIZE {
            let cell = cell as u32;
            return Self {
                middle: cell,
                low: cell,
                high: cell,
            };
        }
        let width = 1i32 << lod_level;
        let spacing =
            (CHUNK_SIZE_I << lod_level) as f32 / extent.saturating_sub(2).max(1) as f32;
        // Data voxel `c` covers the voxels from `floor((c - 1) × spacing)`.
        let covering = |offset: i32| {
            let coordinate = ((offset as f32 + 1.0) / spacing).ceil() as i64;
            coordinate.clamp(0, extent as i64 - 1) as u32
        };
        let start = (cell as i32 - 1) * width;
        let middle = covering(start + width / 2);
        if spacing >= width as f32 {
            Self {
                middle,
                low: middle,
                high: middle,
            }
        } else {
            Self {
                middle,
                low: covering(start),
                high: covering(start + width - 1),
            }
        }
    }

    fn single(&self) -> bool {
        self.low == self.high
    }
}

/// Whether the padded cell `cell` of the coarse `chunk` is open, as its
/// chunk task saw it: a cell is open if any data voxel in it is.
pub(crate) fn cell_is_open<I: Copy + PartialEq>(
    chunk: &ChunkData<I>,
    cell: UVec3,
) -> bool {
    let voxels = match (&chunk.fill_type, &chunk.voxels) {
        (FillType::Empty, _) => return true,
        (FillType::Uniform(voxel), _) => return !voxel.is_solid(),
        (FillType::Mixed, Some(voxels)) => voxels,
        (FillType::Mixed, None) => unreachable!("a mixed chunk stores its voxels"),
    };
    let shape = chunk.data_shape;
    let [rx, ry, rz] = std::array::from_fn(|axis| {
        CellRange::of(cell[axis] as usize, shape[axis], chunk.key.lod_level)
    });
    let (row, slab) = (shape.x as usize, (shape.x * shape.y) as usize);
    (rz.low..=rz.high).any(|z| {
        (ry.low..=ry.high).any(|y| {
            (rx.low..=rx.high).any(|x| {
                !voxels[x as usize + y as usize * row + z as usize * slab].is_solid()
            })
        })
    })
}

/// The level `top` in the `lit` cells and 0 elsewhere, in slot order.
fn lit_levels(lit: &[u64], top: u8) -> Vec<u8> {
    let mut levels = vec![0; SLOT_TEXELS];
    write_bits(&mut levels, lit, top);
    levels
}

/// Writes `level` into `levels`, in slot order, at each cell set in `cells`.
fn write_bits(levels: &mut [u8], cells: &[u64], level: u8) {
    for y in 0..SIDE {
        for x in 0..SIDE {
            let mut bits = cells[x + y * SIDE];
            while bits != 0 {
                let z = bits.trailing_zeros();
                bits &= bits - 1;
                levels[cell_index(UVec3::new(x as u32, y as u32, z))] = level;
            }
        }
    }
}

thread_local! {
    /// Each thread's boxes for coarse chunks, one per row width and reach,
    /// kept apart from level 0's so neither replaces the other's.
    static ONE_WORD_BOXES: RefCell<Vec<LightBox<u64>>> = const { RefCell::new(Vec::new()) };
    static TWO_WORD_BOXES: RefCell<Vec<LightBox<TwoWords>>> = const { RefCell::new(Vec::new()) };
}

/// Spreads sky light or voxel light through the `open` cells, from the `lit`
/// cells and the emissive cells `lamps`. Returns each cell's level in slot
/// order, padding included.
fn spread(
    open: &[u64],
    lit: Option<&[u64]>,
    lamps: &[Lamp],
    scale: &LevelScale,
    shape: LightShape,
) -> Vec<u8> {
    let (planes, shell) = if scale.reach() <= MAX_ONE_WORD_REACH {
        run(&ONE_WORD_BOXES, open, lit, lamps, scale, shape)
    } else {
        run(&TWO_WORD_BOXES, open, lit, lamps, scale, shape)
    };
    let mut levels = vec![0; SLOT_TEXELS];
    for y in 0..CHUNK {
        for x in 0..CHUNK {
            let column = x + y * CHUNK;
            let mut reached = planes.iter().fold(0, |any, plane| any | plane[column]);
            while reached != 0 {
                let z = reached.trailing_zeros() as usize;
                reached &= reached - 1;
                let level: u8 = planes
                    .iter()
                    .enumerate()
                    .map(|(bit, plane)| ((plane[column] >> z & 1) as u8) << bit)
                    .sum();
                let cell = UVec3::new(x as u32 + 1, y as u32 + 1, z as u32 + 1);
                levels[cell_index(cell)] = level;
            }
        }
    }
    for face in Face::ALL {
        for v in 0..CHUNK {
            for u in 0..CHUNK {
                let level = shell[face as usize][u + v * CHUNK];
                if level > 0 {
                    levels[cell_index(shell_cell(face, u, v))] = level;
                }
            }
        }
    }
    if let Some(lit) = lit {
        write_bits(&mut levels, lit, scale.precision().top());
    }
    levels
}

fn run<R: Row>(
    cached: &'static LocalKey<RefCell<Vec<LightBox<R>>>>,
    open: &[u64],
    lit: Option<&[u64]>,
    lamps: &[Lamp],
    scale: &LevelScale,
    shape: LightShape,
) -> (Box<[Plane]>, Box<super::kernel::ShellLevels>) {
    cached.with_borrow_mut(|boxes| {
        let reach = scale.reach() as usize;
        let index = match boxes
            .iter()
            .position(|light_box| light_box.reach() == reach)
        {
            Some(index) => index,
            None => {
                boxes.push(LightBox::new(scale.reach()));
                boxes.len() - 1
            }
        };
        let light_box = &mut boxes[index];
        light_box.capture_shell();
        // The box holds the chunk's cells with its padding one cell inside
        // the box's reach.
        let offset = reach - 1;
        let side = CHUNK + 2 * reach;
        let fill = |rows: &mut [R], cells: &[u64]| {
            rows.fill(R::ZERO);
            for y in 0..SIDE {
                for x in 0..SIDE {
                    rows[x + offset + (y + offset) * side] =
                        R::from_u64(cells[x + y * SIDE]) << offset;
                }
            }
        };
        fill(light_box.open_rows_mut(), open);
        if let Some(lit) = lit {
            fill(light_box.lit_rows_mut(), lit);
        }
        let mut seeds: Vec<Seed> = lamps
            .iter()
            .filter_map(|lamp| {
                Some(Seed {
                    x: (lamp.cell.x as usize + offset) as u8,
                    y: (lamp.cell.y as usize + offset) as u8,
                    z: (lamp.cell.z as usize + offset) as u8,
                    start: scale.start(scale.precision().of_whole(lamp.level))?,
                })
            })
            .collect();
        seeds.sort_by_key(|seed| seed.start);
        let mut planes = empty_planes(scale.precision());
        let mut faces = [0; 6];
        light_box.spread(
            &seeds,
            lit.is_some(),
            &[],
            shape,
            scale,
            &mut planes,
            &mut faces,
        );
        let shell = Box::new(*light_box.shell_levels().expect("the shell is captured"));
        (planes, shell)
    })
}

/// The walk above a coarse chunk that finds which of its columns sky light
/// falls into.
///
/// It reads the world's lookups for the level-0 chunks it passes through,
/// with voxel edits read first, since the chunk's own data can't answer for
/// voxels outside it. Its rows start one cell above the chunk's padding row
/// and their step doubles every 8 rows, so a thin roof just above is seen and
/// a thick one far above still is, up to where the world's sky rule stops
/// looking. Rows in a chunk position a fill hint says is air cost no lookups,
/// and a position a hint says is solid blocks every column.
///
/// It walks as many columns as [`WALK_BUDGET`] lookups carry: each column
/// alone if it can, and otherwise one column in the middle of each square
/// block of columns, blocks up to 32 wide, answering for the whole block.
pub(crate) struct CoarseWalk<I> {
    /// The voxel at the chunk's minimum corner.
    min: IVec3,
    width: i32,
    /// The height of each row, and what a fill hint says of the chunk
    /// position holding it.
    rows: Vec<(i32, Hinted)>,
    /// The world's voxel edits, if any lie in the positions the walk passes.
    edits: Option<VoxelEditReader<I>>,
    delegate: VoxelLookupDelegate<I>,
}

impl<I: Copy + PartialEq> CoarseWalk<I> {
    /// The walk above the coarse chunk with `key` under `rule`, reading the
    /// fill hints of world `world` now, on the main thread.
    pub fn new<C: VoxelWorldConfig<MaterialIndex = I>>(
        key: ChunkKey,
        rule: SkyRule,
        world: &C,
        edits: &VoxelEdits<C, I>,
        delegate: VoxelLookupDelegate<I>,
    ) -> Self {
        let width = 1i64 << key.lod_level;
        let span = CHUNK_SIZE_I as i64 * width;
        let min = key.min_voxel();
        let top = min.y as i64 + span;
        let limit = match rule {
            SkyRule::Clearance(chunks) => top + chunks as i64 * CHUNK_SIZE_I as i64,
            SkyRule::Ceiling(chunk) => (chunk as i64 + 1) * CHUNK_SIZE_I as i64,
        }
        .min(i32::MAX as i64);
        let reader = edits.reader();
        let committed = edits.committed();
        let mut edited = false;
        let mut hints = HashMap::<i64, Hinted>::new();
        let mut rows = Vec::new();
        let (mut y, mut step) = (top + width, width);
        while y < limit {
            let region = (y - top) / span;
            let hinted = *hints.entry(region).or_insert_with(|| {
                let above = ChunkKey::new(
                    key.position + IVec3::Y * ((region as i32 + 1) * key.extent()),
                    key.lod_level,
                );
                let reached = committed.reach(above, IVec3::ZERO);
                edited |= reached;
                Hinted::of(world.chunk_fill_hint(above), reached)
            });
            rows.push((y as i32, hinted));
            if rows.len() % 8 == 0 {
                step *= 2;
            }
            y += step;
        }
        Self {
            min,
            width: width as i32,
            rows,
            edits: edited.then_some(reader),
            delegate,
        }
    }

    /// Which of the chunk's interior columns set in `open` sky light falls
    /// into, bit `z` of word `x` as in `open`.
    fn answer(self, open: &Columns, settings: &LightTaskSettings<I>) -> Columns {
        if self.rows.iter().any(|(_, hinted)| *hinted == Hinted::Solid) {
            return [0; CHUNK];
        }
        let rows: Vec<i32> = self
            .rows
            .iter()
            .filter(|(_, hinted)| *hinted == Hinted::Unknown)
            .map(|(y, _)| *y)
            .collect();
        if rows.is_empty() {
            return *open;
        }
        let mut block = 1;
        while block < CHUNK
            && blocks_with_open(open, block) * rows.len() > WALK_BUDGET as usize
        {
            block *= 2;
        }

        let mut lookups = ChunkLookups::new(self.delegate);
        let mut made = 0u64;
        let mut answer = [0; CHUNK];
        let blocks = CHUNK / block;
        let block_mask = if block == CHUNK {
            !0u32
        } else {
            (1u32 << block) - 1
        };
        for bz in 0..blocks {
            for bx in 0..blocks {
                let columns = block_mask << (bz * block);
                let xs = bx * block..(bx + 1) * block;
                if xs.clone().all(|x| open[x] & columns == 0) {
                    continue;
                }
                let middle = IVec3::new(
                    self.min.x + (bx * block + block / 2) as i32 * self.width,
                    0,
                    self.min.z + (bz * block + block / 2) as i32 * self.width,
                );
                let clear = rows.iter().all(|&y| {
                    let at = IVec3::new(middle.x, y, middle.z);
                    made += 1;
                    let edit = self.edits.as_ref().and_then(|edits| edits.get(at));
                    let voxel = edit.unwrap_or_else(|| lookups.lookup(at));
                    !voxel.is_solid()
                });
                if clear {
                    for x in xs {
                        answer[x] |= open[x] & columns;
                    }
                }
            }
        }
        settings
            .counters
            .coarse_lookups
            .fetch_add(made, Ordering::Relaxed);
        answer
    }
}

/// How many square blocks `block` columns wide hold a column set in `open`.
fn blocks_with_open(open: &Columns, block: usize) -> usize {
    let blocks = CHUNK / block;
    let mask = if block == CHUNK {
        !0u32
    } else {
        (1u32 << block) - 1
    };
    (0..blocks)
        .map(|bz| {
            (0..blocks)
                .filter(|bx| {
                    (bx * block..(bx + 1) * block)
                        .any(|x| open[x] >> (bz * block) & mask != 0)
                })
                .count()
        })
        .sum()
}
