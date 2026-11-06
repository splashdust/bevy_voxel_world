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
use super::kernel::{LightBox, Plane, Seed, LEVEL_BITS};
use super::levels::LevelScale;
use super::solver::{neighbour_index, solve_channel};
use super::LightSettings;
use crate::chunk::{ChunkData, FillType, CHUNK_SIZE_U, PADDED_CHUNK_SIZE};
use crate::chunk_map::ChunkMap;
use crate::configuration::VoxelWorldConfig;
use crate::prelude::{LightShape, WorldVoxel};

const CHUNK: usize = CHUNK_SIZE_U as usize;

/// A kernel box filled once, which can be spread again and again.
pub struct KernelBox {
    light_box: LightBox,
    seeds: Vec<Seed>,
    scale: LevelScale,
    planes: Box<[Plane; LEVEL_BITS]>,
}

impl KernelBox {
    /// A box for light of `reach`, from 1 to 16. `open` says whether light
    /// passes the voxel at a position in the box, from 0 to `32 + 2 × reach`
    /// on each axis. Each source is a position in the box and its level.
    pub fn new(
        reach: u32,
        open: impl Fn(IVec3) -> bool,
        sources: &[(IVec3, u8)],
    ) -> Self {
        let scale = LevelScale::new(reach);
        let mut light_box = LightBox::new(reach);
        let side = light_box.side();
        let rows = light_box.open_rows_mut();
        for y in 0..side {
            for x in 0..side {
                let mut row = 0u64;
                for z in 0..side {
                    if open(IVec3::new(x as i32, y as i32, z as i32)) {
                        row |= 1 << z;
                    }
                }
                rows[x + y * side] = row;
            }
        }
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
        Self {
            light_box,
            seeds,
            scale,
            planes: Box::new([[0; CHUNK * CHUNK]; LEVEL_BITS]),
        }
    }

    /// Spreads light through the box in `shape`.
    pub fn spread(&mut self, shape: LightShape) {
        self.light_box
            .spread(&self.seeds, shape, &self.scale, &mut self.planes);
    }

    /// The level the last spread left at the chunk's voxel `local`.
    pub fn level(&self, local: UVec3) -> u8 {
        let column = local.x as usize + local.y as usize * CHUNK;
        (0..LEVEL_BITS)
            .map(|bit| ((self.planes[bit][column] >> local.z & 1) as u8) << bit)
            .sum()
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
    LightInputs::of(&chunk, emission).0.emissive.len()
}

/// Counts of a lit world's light work so far.
#[derive(Clone, Copy, Debug, Default)]
pub struct LightStats {
    /// Chunks whose voxels were scanned for emissive voxels.
    pub emissive_scans: u64,
    pub solves: u64,
    /// Time spent solving in chunk tasks, summed over them, in milliseconds.
    pub solve_ms: f64,
    /// Main-thread time spent reading the inputs of chunk tasks' neighbours,
    /// in milliseconds.
    pub snapshot_ms: f64,
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
    })
}

/// Solves voxel light again for every published level-0 chunk of world `C`,
/// one after another on this thread, from the inputs published now. Returns
/// how many it solved and the time it took in milliseconds.
pub fn solve_published_serially<C: VoxelWorldConfig>(world: &World) -> (usize, f64) {
    let Some(settings) = world.get_resource::<LightSettings<C>>() else {
        return (0, 0.0);
    };
    let chunk_map = world.resource::<ChunkMap<C, C::MaterialIndex>>();
    let map = chunk_map.get_read_lock();
    let mut tasks = Vec::new();
    for (key, chunk) in map.iter() {
        let Some(light) = &chunk.light else {
            continue;
        };
        let Some(mut task) = settings.task_for(*key, chunk.data_shape, &map) else {
            continue;
        };
        task.neighbours[neighbour_index(IVec3::ZERO)] = Some(light.inputs.clone());
        tasks.push(task);
    }
    let start = Instant::now();
    for task in &tasks {
        black_box(solve_channel(
            &task.neighbours,
            &task.settings.voxel_scale,
            task.settings.shape,
        ));
    }
    (tasks.len(), start.elapsed().as_secs_f64() * 1000.0)
}
