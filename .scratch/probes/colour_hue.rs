//! Throwaway probe for the coloured-light map, after its ticket "What each
//! colour candidate costs": `colour_label.rs` with one more candidate, the
//! **blended hue**, and how far each one-run candidate's hue is from the
//! exact max mix. Not crate code. `colour_hue_build.sh` copies it into the
//! probe worktree's `examples/`, with the kernel patched to carry a colour
//! index and a hue with the light (`colour_hue.patch`), for one build with the
//! bench profile.
//!
//! - **blended hue**: one run of every lamp, as white, each newly lit voxel
//!   taking the mean hue (8-bit red, green and blue) of the voxels that lit
//!   it in that step. Those voxels all have the same level, one step brighter,
//!   so weighting them by level changes nothing; dimmer neighbours have no hue
//!   yet when a voxel is lit. Checked: where one colour alone is the
//!   brightest, the hue is that colour's.
//!
//! Hue error, over lit chunk voxels above level 1 under the eight and three
//! colour palettes: each candidate's colour normalised to its brightest
//! component, against the exact max mix (each component the max over the
//! groups' runs of colour × the default falloff at the run's level) normalised
//! the same way; the error is the largest difference over the components, 0
//! to 1. Per-component levels are scored the same way, as a reference.
//!
//! - **carried index**: one run of every lamp, as white, each newly lit voxel
//!   taking the colour index of a voxel that lit it, held as a bit plane per
//!   bit of the index. Each voxel ends with the colour of the brightest light
//!   reaching it, so pools meet at a hard seam. Checked against the groups'
//!   runs: a voxel's index must name a colour whose own run reaches it at the
//!   white run's level.
//!
//! It cuts a kernel box around every chunk of three scenes and solves the
//! chunk's voxel light with the crate's own kernel (`light_bench::KernelBox`),
//! for each candidate:
//!
//! - **white**: one run of every lamp, as today.
//! - **groups**: a run per light colour with a lamp in the box. Where two or
//!   more runs reach the chunk they're mixed, each component the max over runs
//!   of colour × brightness under the default voxel falloff, black at level 1,
//!   into three 8-bit components (`mix`). Where one run reaches the chunk, it
//!   keeps today's planes and a colour index. `keep` is the other way to store
//!   two or more runs: each run's planes as they are, mixed in the shader.
//! - **per-component**: a run per component, each lamp seeding a component at
//!   the whole level whose brightness is its colour times its own, as the look
//!   probe seeded it. Components with the same seeds run once, so white is one
//!   run and today's planes.
//!
//! Palettes: every lamp white; every lamp orange; three colours; eight
//! colours. A lamp's colour is a hash of its position.
//!
//! Timings are the best of `REPEATS`, per chunk; a line reports their median
//! and maximum over the chunks with a lamp in their box. Packing a slot is
//! timed with packers written like the crate's `slot_bytes`, and checked
//! against it at today's format; shell planes likewise.
//!
//! Arguments: `out=<file.csv>`, `scenes=caves,lamp_world,many_lamps`,
//! `reaches=14,32`.

use std::collections::BTreeMap;
use std::hint::black_box;
use std::io::Write;
use std::time::Instant;

use bevy::prelude::*;
use bevy_voxel_world::light_bench::{KernelBox, SlotBox};
use bevy_voxel_world::prelude::LightShape;

#[path = "../benches/common/light_scene.rs"]
mod light_scene;

use light_scene::SceneTable;

const REPEATS: usize = 5;
type Plane = [u32; 1024];
const SIDE: usize = 34;
const SHAPE: LightShape = LightShape::Round;

// --- Scenes ----------------------------------------------------------------------

/// A block of chunks sampled into a bit per voxel, with 32 voxels around it,
/// and its lamps.
struct Region {
    name: &'static str,
    chunks: Vec<IVec3>,
    min: IVec3,
    size: IVec3,
    bits: Vec<u64>,
    lamps: Vec<IVec3>,
}

impl Region {
    fn sample(
        name: &'static str,
        first: IVec3,
        last: IVec3,
        open: impl Fn(IVec3) -> bool + Sync,
        lamp: impl Fn(IVec3, &dyn Fn(IVec3) -> bool) -> bool,
    ) -> Self {
        let started = Instant::now();
        let min = first * 32 - IVec3::splat(32);
        let size = (last - first + IVec3::ONE) * 32 + IVec3::splat(64);
        let (w, h, d) = (size.x as usize, size.y as usize, size.z as usize);
        let words = (w * h).div_ceil(64);
        let mut bits = vec![0u64; words * d];
        std::thread::scope(|scope| {
            for (z0, slabs) in bits.chunks_mut(words * d.div_ceil(12)).enumerate() {
                let open = &open;
                scope.spawn(move || {
                    for (dz, slab) in slabs.chunks_mut(words).enumerate() {
                        let z = z0 * d.div_ceil(12) + dz;
                        for y in 0..h {
                            for x in 0..w {
                                if open(min + IVec3::new(x as i32, y as i32, z as i32)) {
                                    let bit = x + y * w;
                                    slab[bit / 64] |= 1 << (bit % 64);
                                }
                            }
                        }
                    }
                });
            }
        });
        let mut region = Self {
            name,
            chunks: Vec::new(),
            min,
            size,
            bits,
            lamps: Vec::new(),
        };
        for z in first.z..=last.z {
            for y in first.y..=last.y {
                for x in first.x..=last.x {
                    region.chunks.push(IVec3::new(x, y, z));
                }
            }
        }
        let is_open = |p: IVec3| region.open(p);
        let mut lamps = Vec::new();
        for z in 0..d as i32 {
            for y in 0..h as i32 - 1 {
                for x in 0..w as i32 {
                    let p = min + IVec3::new(x, y, z);
                    if !is_open(p) && lamp(p, &is_open) {
                        lamps.push(p);
                    }
                }
            }
        }
        region.lamps = lamps;
        eprintln!(
            "{name}: {} chunks, {} lamps, sampled in {:.1?}",
            region.chunks.len(),
            region.lamps.len(),
            started.elapsed()
        );
        region
    }

    #[inline]
    fn open(&self, p: IVec3) -> bool {
        let q = p - self.min;
        if q.cmplt(IVec3::ZERO).any() || q.cmpge(self.size).any() {
            return true;
        }
        let (w, h) = (self.size.x as usize, self.size.y as usize);
        let words = (w * h).div_ceil(64);
        let bit = q.x as usize + q.y as usize * w;
        self.bits[q.z as usize * words + bit / 64] >> (bit % 64) & 1 == 1
    }
}

fn hash3(p: IVec3) -> u32 {
    let mut h = (p.x as u32).wrapping_mul(0x9e37_79b1)
        ^ (p.y as u32).wrapping_mul(0x85eb_ca77)
        ^ (p.z as u32).wrapping_mul(0xc2b2_ae3d);
    h ^= h >> 15;
    h = h.wrapping_mul(0x2c1b_3c6d);
    h ^ (h >> 12)
}

/// The caves bench's scene around its camera, in the deep cavern at the foot
/// of the shaft: lamps on one in 32 exposed floor voxels and the torch
/// columns, as the reference scene has them. With `dense`, also on one in 4
/// exposed floor voxels: the scene of many lamps.
fn caves(dense: bool) -> Region {
    let first = IVec3::new(-3, -8, -3);
    let last = IVec3::new(4, -5, 4);
    let min = first * 32 - IVec3::splat(32);
    let max = (last + IVec3::ONE) * 32 + IVec3::splat(32);
    let table = SceneTable::new(min, max.with_y(max.y + 1), 12);
    let name = if dense { "many_lamps" } else { "caves" };
    Region::sample(
        name,
        first,
        last,
        |p| table.open(p),
        |p, open| {
            table.emissive(p) || (dense && open(p + IVec3::Y) && hash3(p) % 4 == 0)
        },
    )
}

/// `chunk_lifecycle`'s `LampWorld`: the sine terrain with a lamp on one in 64
/// surface voxels.
fn lamp_world() -> Region {
    let height =
        |x: i32, z: i32| (8.0 * (x as f32 / 23.0).sin() + 8.0 * (z as f32 / 17.0).cos()) as i32;
    Region::sample(
        "lamp_world",
        IVec3::new(0, -1, 0),
        IVec3::new(7, 0, 7),
        |p| p.y >= height(p.x, p.z),
        |p, _| {
            let hash = (p.x.wrapping_mul(73_856_093) ^ p.z.wrapping_mul(19_349_663)) as u32;
            p.y == height(p.x, p.z) - 1 && hash.is_multiple_of(64)
        },
    )
}

// --- Colour ----------------------------------------------------------------------

const WHITE: [f32; 3] = [1.0, 1.0, 1.0];
const ORANGE: [f32; 3] = [1.0, 0.5, 0.15];
const EIGHT: [[f32; 3]; 8] = [
    ORANGE,
    [1.0, 0.15, 0.1],
    [0.2, 1.0, 0.25],
    [0.25, 0.35, 1.0],
    [0.1, 0.85, 0.8],
    [0.7, 0.2, 1.0],
    [1.0, 0.9, 0.3],
    [1.0, 0.45, 0.75],
];

struct Palette {
    name: &'static str,
    colours: Vec<[f32; 3]>,
}

fn palettes() -> Vec<Palette> {
    vec![
        Palette { name: "white", colours: vec![WHITE] },
        Palette { name: "one", colours: vec![ORANGE] },
        Palette { name: "three", colours: EIGHT[..3].to_vec() },
        Palette { name: "eight", colours: EIGHT.to_vec() },
    ]
}

/// The default voxel falloff from "How coloured light looks": black at level 1.
fn falloff(level: f32) -> f32 {
    let whole = |l: usize| -> f32 {
        if l == 0 {
            return 0.0;
        }
        let exp = |l: usize| 0.7f32.powi(15 - l as i32);
        (exp(l) - exp(1)) / (1.0 - exp(1))
    };
    if level <= 0.0 {
        return 0.0;
    }
    let level = level.min(15.0);
    let low = level.floor() as usize;
    let high = (low + 1).min(15);
    let f = level - low as f32;
    whole(low) * (1.0 - f) + whole(high) * f
}

fn inverse(b: f32) -> f32 {
    if b <= 0.0 {
        return 0.0;
    }
    let (mut low, mut high) = (0.0f32, 15.0f32);
    for _ in 0..40 {
        let mid = 0.5 * (low + high);
        if falloff(mid) < b {
            low = mid;
        } else {
            high = mid;
        }
    }
    high
}

/// The whole level a lamp of level 15 and `colour` seeds component `c` at,
/// under per-component levels.
fn component_seed(colour: [f32; 3], c: usize) -> u8 {
    inverse(colour[c] * falloff(15.0)).round() as u8
}

/// Per run colour and component, the 8-bit component (1/17 of a level) each
/// level of the run mixes to: the level whose brightness is colour × the run's.
fn luts(colours: &[[f32; 3]], p256: bool) -> Vec<[Vec<u8>; 3]> {
    let steps = if p256 { 256 } else { 16 };
    let per_level = if p256 { 17.0 } else { 1.0 };
    colours
        .iter()
        .map(|colour| {
            std::array::from_fn(|c| {
                (0..steps)
                    .map(|s| {
                        let level = s as f32 / per_level;
                        (inverse(colour[c] * falloff(level)) * 17.0).round().min(255.0) as u8
                    })
                    .collect()
            })
        })
        .collect()
}

// --- Packing, as the crate's slot_bytes does it -------------------------------------

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

#[inline]
fn group_levels(planes: &[Plane], column: usize, shift: usize, open: usize) -> u64 {
    let mut texels = 0;
    for (bit, plane) in planes.iter().enumerate() {
        texels |= SPREAD[(plane[column] >> shift) as usize & open] << bit;
    }
    texels
}

#[inline]
fn every_other_byte(bytes: u64) -> u128 {
    let mut wide = bytes as u128;
    wide = (wide | wide << 32) & 0x0000_0000_FFFF_FFFF_0000_0000_FFFF_FFFF;
    wide = (wide | wide << 16) & 0x0000_FFFF_0000_FFFF_0000_FFFF_0000_FFFF;
    (wide | wide << 8) & 0x00FF_00FF_00FF_00FF_00FF_00FF_00FF_00FF
}

#[inline]
fn every_fourth_byte(bytes: u32) -> u128 {
    let mut wide = bytes as u128;
    wide = (wide | wide << 48) & 0x0000_0000_0000_FFFF_0000_0000_0000_FFFF;
    (wide | wide << 24) & 0x0000_00FF_0000_00FF_0000_00FF_0000_00FF
}

/// The open voxels of the chunk, a `u32` column along z per `x + y × 32`.
fn open_columns(region: &Region, chunk: IVec3) -> Vec<u32> {
    let origin = chunk * 32;
    let mut columns = vec![0u32; 1024];
    for y in 0..32 {
        for x in 0..32 {
            for z in 0..32 {
                if region.open(origin + IVec3::new(x, y, z)) {
                    columns[(x + y * 32) as usize] |= 1 << z;
                }
            }
        }
    }
    columns
}

/// Where a slot holds a channel's levels.
enum Texels<'a> {
    /// Today's: one voxel channel, 1 byte a texel at 16 levels and 2 at 256.
    Today(&'a [Plane]),
    /// Three components of 4 bits, `R16Uint`, 2 bytes a texel; at 16 levels.
    Nibbles([&'a [Plane]; 3]),
    /// Three components of 8 bits from bit planes, `Rgba8Uint`, 4 bytes.
    Bytes([&'a [Plane]; 3]),
    /// Three components of 8 bits already as a byte per voxel, `Rgba8Uint`.
    Mixed(&'a [Vec<u8>; 3]),
}

fn texel_bytes(texels: &Texels, p256: bool) -> usize {
    match texels {
        Texels::Today(_) => 1 + p256 as usize,
        Texels::Nibbles(_) => 2,
        Texels::Bytes(_) | Texels::Mixed(_) => 4,
    }
}

/// A slot's bytes, sky light at the top step in every open voxel, as the
/// crate's `SlotBox` holds it.
fn pack(open: &[u32], texels: &Texels, p256: bool) -> Box<[u8]> {
    let wide = texel_bytes(texels, p256);
    let top: u64 = if p256 { 255 } else { 15 };
    let mut bytes = vec![0u8; SIDE * SIDE * SIDE * wide].into_boxed_slice();
    for y in 0..32 {
        for x in 0..32 {
            let column = x + y * 32;
            let row = 1 + (x + 1) * SIDE + (y + 1) * SIDE * SIDE;
            for shift in (0..32).step_by(8) {
                let open_byte = (open[column] >> shift) as usize & 0xFF;
                let sky = SPREAD[open_byte] * top;
                let at = (row + shift) * wide;
                match texels {
                    Texels::Today(planes) => {
                        let voxel = group_levels(planes, column, shift, open_byte);
                        if p256 {
                            bytes[at..at + 16].copy_from_slice(
                                &(every_other_byte(sky) | every_other_byte(voxel) << 8)
                                    .to_le_bytes(),
                            );
                        } else {
                            bytes[at..at + 8].copy_from_slice(&(sky | voxel << 4).to_le_bytes());
                        }
                    }
                    Texels::Nibbles([r, g, b]) => {
                        let low = sky | group_levels(r, column, shift, open_byte) << 4;
                        let high = group_levels(g, column, shift, open_byte)
                            | group_levels(b, column, shift, open_byte) << 4;
                        bytes[at..at + 16].copy_from_slice(
                            &(every_other_byte(low) | every_other_byte(high) << 8).to_le_bytes(),
                        );
                    }
                    Texels::Bytes([r, g, b]) => {
                        let lanes = [
                            sky,
                            group_levels(r, column, shift, open_byte),
                            group_levels(g, column, shift, open_byte),
                            group_levels(b, column, shift, open_byte),
                        ];
                        put_four(&mut bytes[at..at + 32], lanes);
                    }
                    Texels::Mixed(out) => {
                        let base = column * 32 + shift;
                        let lane = |c: usize| {
                            u64::from_le_bytes(out[c][base..base + 8].try_into().unwrap())
                                & (SPREAD[open_byte] * 0xFF)
                        };
                        put_four(&mut bytes[at..at + 32], [sky, lane(0), lane(1), lane(2)]);
                    }
                }
            }
        }
    }
    bytes
}

#[inline]
fn put_four(bytes: &mut [u8], lanes: [u64; 4]) {
    for half in 0..2 {
        let mut texels = 0u128;
        for (i, lane) in lanes.iter().enumerate() {
            texels |= every_fourth_byte((lane >> (32 * half)) as u32) << (8 * i);
        }
        bytes[16 * half..16 * half + 16].copy_from_slice(&texels.to_le_bytes());
    }
}

/// The level of `planes` at `x`, `y`, `z`, as `ChannelLight::at` reads it.
#[inline]
fn level_at(planes: &[Plane], x: usize, y: usize, z: usize) -> u8 {
    planes
        .iter()
        .enumerate()
        .map(|(bit, plane)| ((plane[x + y * 32] >> z & 1) as u8) << bit)
        .sum()
}

/// Shell planes for every face, as landing writes them: each neighbour's
/// border plane read voxel by voxel, then copied into the slot's shell. Every
/// neighbour holds the chunk's own light, with its own open voxels, as in the
/// crate's `SlotBox`.
fn shell_planes(bytes: &mut [u8], neighbours: &[Vec<u32>; 6], texels: &Texels, p256: bool) {
    let wide = texel_bytes(texels, p256);
    let mut plane = vec![0u8; 1024 * wide];
    for face in 0..6 {
        let (axis, positive) = (face / 2, face % 2 == 1);
        let open = &neighbours[face];
        plane.fill(0);
        for v in 0..32 {
            for u in 0..32 {
                let mut p = [0usize; 3];
                p[axis] = if positive { 0 } else { 31 };
                p[(axis + 1) % 3] = u;
                p[(axis + 2) % 3] = v;
                let (x, y, z) = (p[0], p[1], p[2]);
                if open[x + y * 32] >> z & 1 == 0 {
                    continue;
                }
                let sky = if p256 { 255u8 } else { 15 };
                let at = (u + v * 32) * wide;
                match texels {
                    Texels::Today(planes) => {
                        let voxel = level_at(planes, x, y, z);
                        if p256 {
                            plane[at] = sky;
                            plane[at + 1] = voxel;
                        } else {
                            plane[at] = sky | voxel << 4;
                        }
                    }
                    Texels::Nibbles([r, g, b]) => {
                        let texel = sky as u16
                            | (level_at(r, x, y, z) as u16) << 4
                            | (level_at(g, x, y, z) as u16) << 8
                            | (level_at(b, x, y, z) as u16) << 12;
                        plane[at..at + 2].copy_from_slice(&texel.to_le_bytes());
                    }
                    Texels::Bytes([r, g, b]) => {
                        plane[at..at + 4].copy_from_slice(&[
                            sky,
                            level_at(r, x, y, z),
                            level_at(g, x, y, z),
                            level_at(b, x, y, z),
                        ]);
                    }
                    Texels::Mixed(out) => {
                        let i = (x + y * 32) * 32 + z;
                        plane[at..at + 4].copy_from_slice(&[sky, out[0][i], out[1][i], out[2][i]]);
                    }
                }
            }
        }
        for v in 0..32 {
            for u in 0..32 {
                let mut cell = [0usize; 3];
                cell[axis] = if positive { SIDE - 1 } else { 0 };
                cell[(axis + 1) % 3] = u + 1;
                cell[(axis + 2) % 3] = v + 1;
                let to = (cell[2] + cell[0] * SIDE + cell[1] * SIDE * SIDE) * wide;
                let from = (u + v * 32) * wide;
                bytes[to..to + wide].copy_from_slice(&plane[from..from + wide]);
            }
        }
    }
}

// --- Mixing -----------------------------------------------------------------------

/// Mixes runs reaching the chunk into three 8-bit components, each the max
/// over runs of the run colour's table at the run's level.
fn mix(runs: &[(usize, Box<[Plane]>)], luts: &[[Vec<u8>; 3]], out: &mut [Vec<u8>; 3]) {
    for c in out.iter_mut() {
        c.fill(0);
    }
    for (colour, planes) in runs {
        let lut = &luts[*colour];
        for column in 0..1024 {
            let any = planes.iter().fold(0, |any, plane| any | plane[column]);
            if any == 0 {
                continue;
            }
            for shift in (0..32).step_by(8) {
                let byte = (any >> shift) as usize & 0xFF;
                if byte == 0 {
                    continue;
                }
                let levels = group_levels(planes, column, shift, 0xFF).to_le_bytes();
                let base = column * 32 + shift;
                for (i, &level) in levels.iter().enumerate() {
                    if level == 0 {
                        continue;
                    }
                    for c in 0..3 {
                        let value = lut[c][level as usize];
                        let slot = &mut out[c][base + i];
                        if value > *slot {
                            *slot = value;
                        }
                    }
                }
            }
        }
    }
}

fn lit_in_chunk(planes: &[Plane]) -> bool {
    planes.iter().any(|plane| plane.iter().any(|&column| column != 0))
}

// --- Measuring --------------------------------------------------------------------

fn best_us(mut work: impl FnMut()) -> f64 {
    (0..REPEATS)
        .map(|_| {
            let start = Instant::now();
            work();
            start.elapsed().as_secs_f64() * 1e6
        })
        .fold(f64::MAX, f64::min)
}

/// One chunk's costs under one palette.
#[derive(Default, Clone)]
struct Costs {
    white_us: f64,
    group_runs: usize,
    group_colours_in_chunk: usize,
    group_run_us: f64,
    group_mix_us: f64,
    group_pack_us: f64,
    group_shell_us: f64,
    component_runs: usize,
    component_distinct: bool,
    component_run_us: f64,
    component_pack_us: f64,
    component_shell_us: f64,
    today_pack_us: f64,
    today_shell_us: f64,
    crate_pack_us: f64,
    crate_shell_us: f64,
    grouping_us: f64,
    /// Bytes per voxel of the chunk's voxel light in the world view.
    today_bytes: f64,
    group_mix_bytes: f64,
    group_keep_bytes: f64,
    component_bytes: f64,
    label_us: f64,
    /// Lit chunk voxels, and those whose carried index names a colour whose
    /// own run doesn't reach them at the white run's level.
    label_lit: usize,
    label_wrong: usize,
    /// Lit chunk voxels where two colours tie for the brightest.
    label_ties: usize,
    hue_us: f64,
    /// Lit chunk voxels with one brightest colour whose blended hue isn't it.
    hue_wrong: usize,
    /// Lit chunk voxels above level 1, scored for hue error.
    scored: usize,
    /// Per one-run candidate (index, hue) and per-component: summed hue
    /// error, and voxels whose error is over 0.1.
    error_sum: [f64; 3],
    error_over: [usize; 3],
}

struct Config {
    reach: u32,
    p256: bool,
}

/// Bits of a colour index for `colours` colours.
fn index_bits(colours: usize) -> usize {
    (usize::BITS - (colours - 1).leading_zeros()) as usize
}

fn measure_chunk(
    region: &Region,
    chunk: IVec3,
    config: &Config,
    palettes: &[Palette],
    luts: &[Vec<[Vec<u8>; 3]>],
) -> Option<Vec<Costs>> {
    let reach = config.reach as i32;
    let box_min = chunk * 32 - IVec3::splat(reach);
    let side = 32 + 2 * reach;
    // Lamps whose light can reach the chunk: within `reach` of it, as the
    // solver's seeds are for a lamp at the top level.
    let lamps: Vec<IVec3> = region
        .lamps
        .iter()
        .map(|p| *p - box_min)
        .filter(|p| p.cmpge(IVec3::ZERO).all() && p.cmplt(IVec3::splat(side)).all())
        .filter(|p| {
            let outside = (IVec3::splat(reach) - *p)
                .max(*p - IVec3::splat(reach + 31))
                .max_element()
                .max(1);
            outside <= reach
        })
        .collect();
    if lamps.is_empty() {
        return None;
    }
    let mut kernel = KernelBox::new(config.reach, |p| region.open(box_min + p), &[]);
    if config.p256 {
        kernel = kernel.at_256_levels();
    }
    let open = open_columns(region, chunk);
    let neighbours: [Vec<u32>; 6] = std::array::from_fn(|face| {
        let mut offset = IVec3::ZERO;
        offset[face / 2] = if face % 2 == 1 { 1 } else { -1 };
        open_columns(region, chunk + offset)
    });
    let bits = if config.p256 { 8.0 } else { 4.0 };
    let mut all = Vec::new();
    for (palette, luts) in palettes.iter().zip(luts) {
        let mut costs = Costs::default();
        let colour_of = |p: IVec3| (hash3(p + box_min) as usize) % palette.colours.len();

        // White: every lamp in one run, as today.
        let white: Vec<(IVec3, u8)> = lamps.iter().map(|p| (*p, 15)).collect();
        let (white_us, white_planes) = run(&mut kernel, &white);
        costs.white_us = white_us;

        // Today's slot and shells, the crate's own and the probe's.
        let lit = lit_in_chunk(&white_planes);
        costs.today_bytes = if lit { bits / 8.0 } else { 0.0 };
        if lit {
            let today = Texels::Today(&white_planes);
            costs.today_pack_us = best_us(|| {
                black_box(pack(&open, &today, config.p256));
            });
            let mut bytes = pack(&open, &today, config.p256);
            costs.today_shell_us = best_us(|| {
                shell_planes(&mut bytes, &neighbours, &today, config.p256);
                black_box(&bytes);
            });
            kernel.set_sources(&white);
            kernel.spread(SHAPE);
            let slot = SlotBox::new(&kernel, |p| region.open(chunk * 32 + p));
            costs.crate_pack_us = best_us(|| {
                black_box(slot.slot_bytes());
            });
            let mut bytes = slot.slot_bytes();
            costs.crate_shell_us = best_us(|| {
                slot.shell_planes(&mut bytes);
                black_box(&bytes);
            });
        }

        // Groups: a run per light colour in the box.
        let grouped = || {
            let mut groups: BTreeMap<usize, Vec<(IVec3, u8)>> = BTreeMap::new();
            for p in &lamps {
                groups.entry(colour_of(*p)).or_default().push((*p, 15));
            }
            groups
        };
        costs.grouping_us = best_us(|| {
            black_box(grouped());
        });
        let groups = grouped();
        costs.group_runs = groups.len();
        let mut reaching = Vec::new();
        for (colour, sources) in &groups {
            let (us, planes) = run(&mut kernel, sources);
            costs.group_run_us += us;
            if lit_in_chunk(&planes) {
                reaching.push((*colour, planes));
            }
        }
        costs.group_colours_in_chunk = reaching.len();

        // Carried index: one run of every lamp, carrying each lamp's colour.
        let index_bits = index_bits(palette.colours.len());
        let mut kernel_labels: Vec<Plane> = Vec::new();
        if index_bits == 0 {
            costs.label_us = costs.white_us;
        } else {
            let coloured: Vec<(IVec3, u8, u8)> =
                lamps.iter().map(|p| (*p, 15, colour_of(*p) as u8)).collect();
            kernel.set_coloured_sources(&coloured, index_bits);
            costs.label_us = best_us(|| {
                kernel.spread(SHAPE);
                black_box(&kernel);
            });
            assert!(*kernel.planes() == *white_planes, "a carried index changed the levels");
            let labels = kernel.label_planes().to_vec();
            kernel_labels = labels.clone();
            for z in 0..32 {
                for y in 0..32 {
                    for x in 0..32 {
                        let level = level_at(&white_planes, x, y, z);
                        if level == 0 {
                            continue;
                        }
                        costs.label_lit += 1;
                        let index = level_at(&labels, x, y, z) as usize;
                        let brightest: Vec<usize> = reaching
                            .iter()
                            .filter(|(_, planes)| level_at(planes, x, y, z) == level)
                            .map(|(colour, _)| *colour)
                            .collect();
                        if brightest.len() > 1 {
                            costs.label_ties += 1;
                        }
                        if !brightest.contains(&index) {
                            costs.label_wrong += 1;
                        }
                    }
                }
            }
            kernel.set_sources(&white);
        }
        let labels = if index_bits > 0 { kernel_labels } else { Vec::new() };

        // Blended hue: one run of every lamp, carrying each lamp's hue.
        let rgb = |colour: [f32; 3]| -> u32 {
            let byte = |c: f32| (c * 255.0).round() as u32;
            byte(colour[0]) | byte(colour[1]) << 8 | byte(colour[2]) << 16
        };
        let mut hues = Vec::new();
        if palette.colours.len() == 1 {
            costs.hue_us = costs.white_us;
        } else {
            let hued: Vec<(IVec3, u8, u32)> = lamps
                .iter()
                .map(|p| (*p, 15, rgb(palette.colours[colour_of(*p)])))
                .collect();
            kernel.set_hued_sources(&hued);
            costs.hue_us = best_us(|| {
                kernel.spread(SHAPE);
                black_box(&kernel);
            });
            assert!(*kernel.planes() == *white_planes, "a blended hue changed the levels");
            hues = kernel.hues().to_vec();
            for z in 0..32 {
                for y in 0..32 {
                    for x in 0..32 {
                        let level = level_at(&white_planes, x, y, z);
                        let hue = hues[(x + y * 32) * 32 + z];
                        if level == 0 {
                            continue;
                        }
                        assert!(hue != 0, "a lit voxel has no hue");
                        let brightest: Vec<usize> = reaching
                            .iter()
                            .filter(|(_, planes)| level_at(planes, x, y, z) == level)
                            .map(|(colour, _)| *colour)
                            .collect();
                        if brightest.len() == 1 && hue != rgb(palette.colours[brightest[0]]) {
                            costs.hue_wrong += 1;
                        }
                    }
                }
            }
            kernel.set_sources(&white);
        }
        let k = reaching.len();
        costs.group_mix_bytes = match k {
            0 => 0.0,
            1 => bits / 8.0,
            _ => 3.0,
        };
        costs.group_keep_bytes = k as f64 * bits / 8.0;
        if k >= 2 {
            let mut out: [Vec<u8>; 3] = std::array::from_fn(|_| vec![0u8; 32768]);
            costs.group_mix_us = best_us(|| {
                mix(&reaching, luts, &mut out);
                black_box(&out);
            });
            let mixed = Texels::Mixed(&out);
            costs.group_pack_us = best_us(|| {
                black_box(pack(&open, &mixed, config.p256));
            });
            let mut bytes = pack(&open, &mixed, config.p256);
            costs.group_shell_us = best_us(|| {
                shell_planes(&mut bytes, &neighbours, &mixed, config.p256);
                black_box(&bytes);
            });
        } else {
            costs.group_pack_us = costs.today_pack_us;
            costs.group_shell_us = costs.today_shell_us;
        }

        // Per-component levels: a run per component, components with the
        // same seeds running once.
        let mut seeds: Vec<Vec<(IVec3, u8)>> = (0..3)
            .map(|c| {
                lamps
                    .iter()
                    .map(|p| (*p, component_seed(palette.colours[colour_of(*p)], c)))
                    .filter(|(_, level)| *level > 0)
                    .collect()
            })
            .collect();
        let distinct = !(seeds[0] == seeds[1] && seeds[1] == seeds[2]);
        costs.component_distinct = distinct;
        if !distinct {
            seeds.truncate(1);
        }
        let mut components: Vec<Box<[Plane]>> = Vec::new();
        for sources in &seeds {
            if sources.is_empty() {
                components.push(vec![[0u32; 1024]; bits as usize].into());
                continue;
            }
            let (us, planes) = run(&mut kernel, sources);
            costs.component_runs += 1;
            costs.component_run_us += us;
            components.push(planes);
        }
        let any = components.iter().any(|planes| lit_in_chunk(planes));
        if !distinct {
            costs.component_bytes = if any { bits / 8.0 } else { 0.0 };
            costs.component_pack_us = costs.today_pack_us;
            costs.component_shell_us = costs.today_shell_us;
        } else if any {
            costs.component_bytes = 3.0 * bits / 8.0;
            let refs = [&*components[0], &*components[1], &*components[2]];
            let texels = if config.p256 { Texels::Bytes(refs) } else { Texels::Nibbles(refs) };
            costs.component_pack_us = best_us(|| {
                black_box(pack(&open, &texels, config.p256));
            });
            let mut bytes = pack(&open, &texels, config.p256);
            costs.component_shell_us = best_us(|| {
                shell_planes(&mut bytes, &neighbours, &texels, config.p256);
                black_box(&bytes);
            });
        }

        // Hue error against the exact max mix, for palettes of several colours.
        if palette.colours.len() > 1 && distinct {
            let normalised = |colour: [f32; 3]| -> Option<[f32; 3]> {
                let top = colour[0].max(colour[1]).max(colour[2]);
                (top > 0.0).then(|| colour.map(|c| c / top))
            };
            let error = |a: [f32; 3], b: [f32; 3]| -> f64 {
                (0..3).map(|c| (a[c] - b[c]).abs()).fold(0.0f32, f32::max) as f64
            };
            for z in 0..32 {
                for y in 0..32 {
                    for x in 0..32 {
                        if level_at(&white_planes, x, y, z) <= 1 {
                            continue;
                        }
                        let mut exact = [0.0f32; 3];
                        for (colour, planes) in &reaching {
                            let brightness = falloff(level_at(planes, x, y, z) as f32);
                            for c in 0..3 {
                                exact[c] = exact[c].max(palette.colours[*colour][c] * brightness);
                            }
                        }
                        let Some(exact) = normalised(exact) else {
                            continue;
                        };
                        let index = palette.colours[level_at(&labels, x, y, z) as usize];
                        let hue = hues[(x + y * 32) * 32 + z];
                        let hue = [0, 8, 16].map(|s| (hue >> s & 0xff) as f32 / 255.0);
                        let component: [f32; 3] =
                            std::array::from_fn(|c| falloff(level_at(&components[c], x, y, z) as f32));
                        costs.scored += 1;
                        for (k, estimate) in [index, hue, component].into_iter().enumerate() {
                            let e = normalised(estimate).map_or(1.0, |estimate| error(exact, estimate));
                            costs.error_sum[k] += e;
                            if e > 0.1 {
                                costs.error_over[k] += 1;
                            }
                        }
                    }
                }
            }
        }
        all.push(costs);
    }
    Some(all)
}

/// Spreads `sources` through the box, best of `REPEATS`. Returns the time and
/// the planes it left.
fn run(kernel: &mut KernelBox, sources: &[(IVec3, u8)]) -> (f64, Box<[Plane]>) {
    kernel.set_sources(sources);
    let us = best_us(|| {
        kernel.spread(SHAPE);
        black_box(&*kernel);
    });
    (us, kernel.planes().into())
}

fn percentile(values: &[f64], p: f64) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[((sorted.len() - 1) as f64 * p).round() as usize]
}

fn main() {
    let mut out_path = None;
    let mut scenes: Vec<String> = vec!["caves".into(), "lamp_world".into(), "many_lamps".into()];
    let mut reaches = vec![14u32, 32];
    let mut precisions = vec![false, true];
    for arg in std::env::args().skip(1) {
        if let Some(path) = arg.strip_prefix("out=") {
            out_path = Some(path.to_string());
        } else if let Some(list) = arg.strip_prefix("scenes=") {
            scenes = list.split(',').map(String::from).collect();
        } else if let Some(list) = arg.strip_prefix("reaches=") {
            reaches = list.split(',').map(|r| r.parse().unwrap()).collect();
        } else if let Some(list) = arg.strip_prefix("levels=") {
            precisions = list.split(',').map(|l| l == "256").collect();
        }
    }
    let palettes = palettes();
    let mut csv = String::from(
        "scene,reach,levels,palette,chunks_lit,stat,white_us,group_runs,group_colours_in_chunk,\
         group_run_us,group_mix_us,group_solve_us,group_pack_us,group_shell_us,component_runs,\
         component_solve_us,component_pack_us,component_shell_us,today_pack_us,today_shell_us,\
         crate_pack_us,crate_shell_us,grouping_us,today_bytes,group_mix_bytes,group_keep_bytes,\
         component_bytes,share_group_mixed,share_component_distinct,share_group_over_3,\
         label_us,share_label_wrong,share_label_ties,hue_us,share_hue_wrong,\
         error_mean_index,error_mean_hue,error_mean_component,\
         share_error_index,share_error_hue,share_error_component\n",
    );
    for scene in &scenes {
        let region = match scene.as_str() {
            "caves" => caves(false),
            "many_lamps" => caves(true),
            "lamp_world" => lamp_world(),
            other => panic!("no scene {other}"),
        };
        for &reach in &reaches {
            for &p256 in &precisions {
                let config = Config { reach, p256 };
                let luts: Vec<_> = palettes.iter().map(|p| luts(&p.colours, p256)).collect();
                let started = Instant::now();
                let mut per_palette: Vec<Vec<Costs>> = vec![Vec::new(); palettes.len()];
                for &chunk in &region.chunks {
                    if let Some(costs) = measure_chunk(&region, chunk, &config, &palettes, &luts) {
                        for (all, costs) in per_palette.iter_mut().zip(costs) {
                            all.push(costs);
                        }
                    }
                }
                let levels = if p256 { 256 } else { 16 };
                for (palette, costs) in palettes.iter().zip(&per_palette) {
                    let n = costs.len();
                    let col = |f: &dyn Fn(&Costs) -> f64| -> Vec<f64> { costs.iter().map(f).collect() };
                    let mean = |f: &dyn Fn(&Costs) -> f64| col(f).iter().sum::<f64>() / n.max(1) as f64;
                    let mixed: Vec<&Costs> =
                        costs.iter().filter(|c| c.group_colours_in_chunk >= 2).collect();
                    let distinct: Vec<&Costs> = costs
                        .iter()
                        .filter(|c| c.component_distinct && c.component_bytes > 0.0)
                        .collect();
                    let over3 = costs.iter().filter(|c| c.group_colours_in_chunk > 3).count();
                    let share_mixed = mixed.len() as f64 / n.max(1) as f64;
                    let share_distinct = distinct.len() as f64 / n.max(1) as f64;
                    let share_over3 = over3 as f64 / n.max(1) as f64;
                    let scored = costs.iter().map(|c| c.scored).sum::<usize>().max(1) as f64;
                    let error_mean =
                        |k: usize| costs.iter().map(|c| c.error_sum[k]).sum::<f64>() / scored;
                    let error_share = |k: usize| {
                        costs.iter().map(|c| c.error_over[k]).sum::<usize>() as f64 / scored
                    };
                    for (stat, p) in [("p50", 0.5), ("p90", 0.9), ("max", 1.0)] {
                        let q = |f: &dyn Fn(&Costs) -> f64| percentile(&col(f), p);
                        let q_of = |set: &[&Costs], f: &dyn Fn(&Costs) -> f64| {
                            percentile(&set.iter().map(|c| f(c)).collect::<Vec<_>>(), p)
                        };
                        let row = [
                            q(&|c| c.white_us),
                            q(&|c| c.group_runs as f64),
                            q(&|c| c.group_colours_in_chunk as f64),
                            q(&|c| c.group_run_us),
                            q_of(&mixed, &|c| c.group_mix_us),
                            q(&|c| c.group_run_us + c.group_mix_us),
                            q_of(&mixed, &|c| c.group_pack_us),
                            q_of(&mixed, &|c| c.group_shell_us),
                            q(&|c| c.component_runs as f64),
                            q(&|c| c.component_run_us),
                            q_of(&distinct, &|c| c.component_pack_us),
                            q_of(&distinct, &|c| c.component_shell_us),
                            q(&|c| c.today_pack_us),
                            q(&|c| c.today_shell_us),
                            q(&|c| c.crate_pack_us),
                            q(&|c| c.crate_shell_us),
                            q(&|c| c.grouping_us),
                            mean(&|c| c.today_bytes),
                            mean(&|c| c.group_mix_bytes),
                            mean(&|c| c.group_keep_bytes),
                            mean(&|c| c.component_bytes),
                            share_mixed,
                            share_distinct,
                            share_over3,
                            q(&|c| c.label_us),
                            costs.iter().map(|c| c.label_wrong).sum::<usize>() as f64
                                / costs.iter().map(|c| c.label_lit).sum::<usize>().max(1) as f64,
                            costs.iter().map(|c| c.label_ties).sum::<usize>() as f64
                                / costs.iter().map(|c| c.label_lit).sum::<usize>().max(1) as f64,
                            q(&|c| c.hue_us),
                            costs.iter().map(|c| c.hue_wrong).sum::<usize>() as f64
                                / costs.iter().map(|c| c.label_lit).sum::<usize>().max(1) as f64,
                            error_mean(0),
                            error_mean(1),
                            error_mean(2),
                            error_share(0),
                            error_share(1),
                            error_share(2),
                        ];
                        csv.push_str(&format!(
                            "{},{reach},{levels},{},{n},{stat},{}\n",
                            region.name,
                            palette.name,
                            row.iter().map(|v| format!("{v:.2}")).collect::<Vec<_>>().join(",")
                        ));
                        if stat == "p50" {
                            println!(
                                "{:<10} r{reach:<2} {levels:>3} {:<5} n={n:<3} white {:6.1} | carried {:6.1} (wrong {:.4}%) | blended {:6.1} (wrong {:.4}%) | groups {:4.1} runs {:7.1} (+mix {:6.1}) mixed {:3.0}% | comp {:3.1} runs {:7.1} | error mean index {:.3} hue {:.3} comp {:.3}, over 0.1 index {:.1}% hue {:.1}% comp {:.1}%",
                                region.name,
                                palette.name,
                                row[0],
                                row[24],
                                row[25] * 100.0,
                                row[27],
                                row[28] * 100.0,
                                mean(&|c| c.group_runs as f64),
                                row[5],
                                row[4],
                                share_mixed * 100.0,
                                mean(&|c| c.component_runs as f64),
                                row[9],
                                row[29],
                                row[30],
                                row[31],
                                row[32] * 100.0,
                                row[33] * 100.0,
                                row[34] * 100.0,
                            );
                        }
                    }
                }
                eprintln!(
                    "{} reach {reach} at {levels} levels in {:.1?}",
                    region.name,
                    started.elapsed()
                );
            }
        }
    }
    if let Some(path) = out_path {
        std::fs::File::create(&path).unwrap().write_all(csv.as_bytes()).unwrap();
        eprintln!("wrote {path}");
    }
}
