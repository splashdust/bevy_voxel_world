//! Just enough of light propagation's internals for the crate's benchmarks:
//! `light_kernels` times the kernel and the emissive scan, and
//! `chunk_lifecycle` reports what solving light costs in chunk tasks. Not
//! public API.

use std::hint::black_box;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use bevy::prelude::*;

use super::inputs::LightInputs;
use super::kernel::{LightBox, Plane, Row, Seed, LEVEL_BITS, MAX_ONE_WORD_REACH};
use super::levels::LevelScale;
use super::sky::ColumnAnswer;
use super::solver::{neighbour_index, solve_sky, solve_voxel};
use super::view::Snapshot;
use super::LightSettings;
use crate::chunk::{ChunkData, FillType, CHUNK_SIZE_U, PADDED_CHUNK_SIZE};
use crate::configuration::VoxelWorldConfig;
use crate::prelude::{LightShape, WorldVoxel};

const CHUNK: usize = CHUNK_SIZE_U as usize;

/// A box of one row width.
enum Rows {
    OneWord(LightBox<u64>),
    TwoWord(LightBox<u128>),
}

/// A kernel box filled once, which can be spread again and again.
pub struct KernelBox {
    rows: Rows,
    seeds: Vec<Seed>,
    /// The lit rows, copied into the box before each spread, if it starts lit.
    lit: Option<Vec<u128>>,
    scale: LevelScale,
    planes: Box<[Plane; LEVEL_BITS]>,
    faces: [u64; 6],
}

impl KernelBox {
    /// A box for voxel light of `reach`, from 1 to 32. `open` says whether
    /// light passes the voxel at a position in the box, from 0 to
    /// `32 + 2 × reach` on each axis. Each source is a position in the box and
    /// its level.
    pub fn new(
        reach: u32,
        open: impl Fn(IVec3) -> bool,
        sources: &[(IVec3, u8)],
    ) -> Self {
        let scale = LevelScale::new(reach);
        let mut seeds: Vec<Seed> = sources
            .iter()
            .filter_map(|&(position, level)| {
                Some(Seed {
                    x: position.x as u8,
                    y: position.y as u8,
                    z: position.z as u8,
                    start: scale.start(level)?,
                })
            })
            .collect();
        seeds.sort_by_key(|seed| seed.start);
        Self::filled(reach, open, seeds, None)
    }

    /// A box for sky light of `reach`, from 1 to 32, spreading from the open
    /// voxels where `direct` says a position in the box has direct sky.
    pub fn with_direct_sky(
        reach: u32,
        open: impl Fn(IVec3) -> bool,
        direct: impl Fn(IVec3) -> bool,
    ) -> Self {
        let side = 32 + 2 * reach as usize;
        let lit = rows(side, |p| open(p) && direct(p));
        Self::filled(reach, open, Vec::new(), Some(lit))
    }

    fn filled(
        reach: u32,
        open: impl Fn(IVec3) -> bool,
        seeds: Vec<Seed>,
        lit: Option<Vec<u128>>,
    ) -> Self {
        let side = 32 + 2 * reach as usize;
        let open = rows(side, open);
        let rows = if reach <= MAX_ONE_WORD_REACH {
            let mut light_box = LightBox::<u64>::new(reach);
            copy_rows(&open, light_box.open_rows_mut());
            Rows::OneWord(light_box)
        } else {
            let mut light_box = LightBox::<u128>::new(reach);
            copy_rows(&open, light_box.open_rows_mut());
            Rows::TwoWord(light_box)
        };
        Self {
            rows,
            seeds,
            lit,
            scale: LevelScale::new(reach),
            planes: Box::new([[0; CHUNK * CHUNK]; LEVEL_BITS]),
            faces: [0; 6],
        }
    }

    /// Spreads light through the box in `shape`, hashing the chunk's faces
    /// as every solve does.
    pub fn spread(&mut self, shape: LightShape) {
        match &mut self.rows {
            Rows::OneWord(light_box) => spread_rows(
                light_box,
                &self.seeds,
                self.lit.as_deref(),
                shape,
                &self.scale,
                &mut self.planes,
                &mut self.faces,
            ),
            Rows::TwoWord(light_box) => spread_rows(
                light_box,
                &self.seeds,
                self.lit.as_deref(),
                shape,
                &self.scale,
                &mut self.planes,
                &mut self.faces,
            ),
        }
    }

    /// The face hashes the last spread left, one per face of the chunk.
    pub fn face_hashes(&self) -> [u64; 6] {
        self.faces
    }

    /// The level the last spread left at the chunk's voxel `local`. A voxel
    /// with direct sky reads 0, as the kernel leaves it.
    pub fn level(&self, local: UVec3) -> u8 {
        let column = local.x as usize + local.y as usize * CHUNK;
        (0..LEVEL_BITS)
            .map(|bit| ((self.planes[bit][column] >> local.z & 1) as u8) << bit)
            .sum()
    }
}

fn spread_rows<R: Row>(
    light_box: &mut LightBox<R>,
    seeds: &[Seed],
    lit: Option<&[u128]>,
    shape: LightShape,
    scale: &LevelScale,
    planes: &mut [Plane; LEVEL_BITS],
    faces: &mut [u64; 6],
) {
    if let Some(lit) = lit {
        copy_rows(lit, light_box.lit_rows_mut());
    }
    light_box.spread(seeds, lit.is_some(), shape, scale, planes, faces);
}

/// The rows of a box `side` voxels on a side, at `x + y × side`, with bit `z`
/// set where `set` holds.
fn rows(side: usize, set: impl Fn(IVec3) -> bool) -> Vec<u128> {
    let mut rows = vec![0; side * side];
    for y in 0..side {
        for x in 0..side {
            for z in 0..side {
                if set(IVec3::new(x as i32, y as i32, z as i32)) {
                    rows[x + y * side] |= 1 << z;
                }
            }
        }
    }
    rows
}

fn copy_rows<R: Row>(from: &[u128], to: &mut [R]) {
    for (to, from) in to.iter_mut().zip(from) {
        *to = R::from_u32(*from as u32)
            | R::from_u32((*from >> 32) as u32) << 32
            | if R::BITS > 64 {
                R::from_u32((*from >> 64) as u32) << 64
                    | R::from_u32((*from >> 96) as u32) << 96
            } else {
                R::ZERO
            };
    }
}

/// Scans a chunk's voxels, 34 on a side with their padding, for its occupancy
/// and, with a non-empty `emission` list of materials and levels, its emissive
/// voxels. Returns how many emissive voxels it found.
pub fn emissive_scan<I: Copy + PartialEq + std::hash::Hash>(
    voxels: &Arc<[WorldVoxel<I>]>,
    emission: &[(I, u8)],
) -> usize {
    debug_assert_eq!(voxels.len(), (PADDED_CHUNK_SIZE as usize).pow(3));
    let mut chunk = ChunkData::new();
    chunk.fill_type = FillType::Mixed;
    chunk.voxels = Some(voxels.clone());
    LightInputs::of(&chunk, emission, ColumnAnswer::none())
        .0
        .emissive
        .len()
}

/// Counts of a lit world's light work so far.
#[derive(Clone, Copy, Debug, Default)]
pub struct LightStats {
    /// Chunks whose voxels were scanned for emissive voxels.
    pub emissive_scans: u64,
    pub solves: u64,
    /// Time spent solving in chunk tasks, summed over them, in milliseconds.
    pub solve_ms: f64,
    /// Time spent reading the world view before solving, summed over chunk
    /// tasks and relight tasks, in milliseconds.
    pub snapshot_ms: f64,
    /// Relight tasks started.
    pub relights: u64,
    /// Main-thread time spent landing light and starting relights, in
    /// milliseconds.
    pub main_thread_ms: f64,
    /// The most main-thread time spent landing light and starting relights
    /// in one frame, in milliseconds.
    pub max_frame_ms: f64,
    /// Time chunk tasks spent working out their column answers, summed over
    /// them, in milliseconds.
    pub column_ms: f64,
    /// Chunk tasks that walked the columns above them.
    pub walks: u64,
    /// Voxel lookups those walks made.
    pub walk_lookups: u64,
}

/// The light work world `C` has done, or `None` if it has no light.
pub fn light_stats<C: VoxelWorldConfig>(world: &World) -> Option<LightStats> {
    let counters = world.get_resource::<LightSettings<C>>()?.counters();
    let ms = |nanos: &AtomicU64| nanos.load(Ordering::Relaxed) as f64 / 1e6;
    Some(LightStats {
        emissive_scans: counters.emissive_scans.load(Ordering::Relaxed),
        solves: counters.solves.load(Ordering::Relaxed),
        solve_ms: ms(&counters.solve_nanos),
        snapshot_ms: ms(&counters.snapshot_nanos),
        relights: counters.relights.load(Ordering::Relaxed),
        main_thread_ms: ms(&counters.main_thread_nanos),
        max_frame_ms: ms(&counters.max_frame_nanos),
        column_ms: ms(&counters.column_nanos),
        walks: counters.walks.walks.load(Ordering::Relaxed),
        walk_lookups: counters.walks.lookups.load(Ordering::Relaxed),
    })
}

/// Solves sky light and voxel light again for every level-0 chunk of world `C` with light,
/// one after another on this thread, from the world view as it is now.
/// Returns how many it solved and the time it took in milliseconds.
pub fn solve_published_serially<C: VoxelWorldConfig>(world: &World) -> (usize, f64) {
    let Some(settings) = world.get_resource::<LightSettings<C>>() else {
        return (0, 0.0);
    };
    let view = settings.view.read();
    let mut boxes = Vec::new();
    for (position, chunk) in view.iter() {
        let mut snapshot = Snapshot::read(&view, *position);
        snapshot.neighbours[neighbour_index(IVec3::ZERO)] =
            Some(chunk.light.inputs.clone());
        boxes.push(snapshot.neighbours);
    }
    let task = &settings.task;
    let start = Instant::now();
    for neighbours in &boxes {
        if let Some(sky_scale) = &task.sky_scale {
            black_box(solve_sky(neighbours, sky_scale, task.shape));
        }
        black_box(solve_voxel(neighbours, &task.voxel_scale, task.shape));
    }
    (boxes.len(), start.elapsed().as_secs_f64() * 1000.0)
}
