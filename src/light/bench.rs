//! Just enough of light propagation's internals for the crate's benchmarks:
//! `light_kernels` times the kernel and the emissive scan, and
//! `chunk_lifecycle` reports what solving light costs in chunk tasks. Not
//! public API.

use std::hint::black_box;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use bevy::prelude::*;

use super::config::LightPrecision;
use super::faces::Face;
use super::inputs::LightInputs;
use super::inputs::{EmissiveVoxel, Occupancy};
use super::kernel::{LightBox, Plane, Row, Seed, MAX_ONE_WORD_REACH};
use super::levels::LevelScale;
use super::route::LightRoute;
use super::sky::ColumnAnswer;
use super::slot::{border_plane, slot_bytes, write_shell};
use super::solver::{empty_planes, neighbour_index, solve_sky, solve_voxel};
use super::solver::{ChannelLight, ChunkLight};
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
    planes: Box<[Plane]>,
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
        let scale = LevelScale::new(reach, LightPrecision::Steps16);
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
            scale: LevelScale::new(reach, LightPrecision::Steps16),
            planes: empty_planes(LightPrecision::Steps16),
            faces: [0; 6],
        }
    }

    /// This box spreading light at 256 levels, where it spreads at 16 by
    /// default. Seeds keep the distance they start at.
    pub fn at_256_levels(mut self) -> Self {
        let precision = LightPrecision::Steps256;
        self.scale = LevelScale::new(self.scale.reach(), precision);
        self.planes = empty_planes(precision);
        self
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
        self.planes
            .iter()
            .enumerate()
            .map(|(bit, plane)| ((plane[column] >> local.z & 1) as u8) << bit)
            .sum()
    }
}

fn spread_rows<R: Row>(
    light_box: &mut LightBox<R>,
    seeds: &[Seed],
    lit: Option<&[u128]>,
    shape: LightShape,
    scale: &LevelScale,
    planes: &mut [Plane],
    faces: &mut [u64; 6],
) {
    if let Some(lit) = lit {
        copy_rows(lit, light_box.lit_rows_mut());
    }
    light_box.spread(seeds, lit.is_some(), &[], shape, scale, planes, faces);
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

/// A chunk's light and its six face neighbours', for timing what landing
/// does to fill a slot.
pub struct SlotBox {
    light: ChunkLight,
    neighbours: [ChunkLight; 6],
}

impl SlotBox {
    /// The light the last spread of `kernel_box` left, at its precision, in a
    /// chunk whose open voxels `open` says, at positions from -32 to 63
    /// relative to the chunk. Each face neighbour gets the same light, with
    /// its own open voxels.
    pub fn new(kernel_box: &KernelBox, open: impl Fn(IVec3) -> bool) -> Self {
        let precision = kernel_box.scale.precision();
        let chunk = |offset: IVec3| {
            let mut solid: Box<Plane> = Box::new([0; CHUNK * CHUNK]);
            for y in 0..CHUNK {
                for x in 0..CHUNK {
                    for z in 0..CHUNK {
                        let local = IVec3::new(x as i32, y as i32, z as i32);
                        if !open(local + offset * CHUNK as i32) {
                            solid[x + y * CHUNK] |= 1 << z;
                        }
                    }
                }
            }
            ChunkLight {
                inputs: Arc::new(LightInputs {
                    occupancy: Occupancy::Mixed(solid),
                    emissive: Vec::<EmissiveVoxel>::new(),
                    columns: ColumnAnswer::none(),
                    ring: true,
                }),
                sky: ChannelLight::Uniform(precision.top()),
                voxel: ChannelLight::Planes(kernel_box.planes.clone()),
                precision,
            }
        };
        Self {
            light: chunk(IVec3::ZERO),
            neighbours: Face::ALL.map(|face| chunk(face.offset())),
        }
    }

    /// Turns the chunk's bit planes into a slot's bytes with the table, in
    /// the format of the light's precision.
    pub fn slot_bytes(&self) -> Box<[u8]> {
        slot_bytes(&self.light)
    }

    /// Copies the six face neighbours' border planes into the shell of
    /// `bytes`.
    pub fn shell_planes(&self, bytes: &mut [u8]) {
        for face in Face::ALL {
            let precision = self.light.precision;
            let neighbour = &self.neighbours[face as usize];
            let plane = border_plane(neighbour, face.opposite(), precision);
            write_shell(bytes, face, Some(&plane), precision);
        }
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
    /// In a world showing its light, slots given to chunks, counting late ones
    /// and evictions.
    pub slot_allocations: u64,
    /// Slots given to chunks whose faces saw one level, when a neighbour's
    /// border plane stopped matching it.
    pub late_allocations: u64,
    /// Slots given back by chunks whose faces came to see one level.
    pub slots_given_back: u64,
    /// Slots taken from a chunk farther from the camera for a nearer one.
    pub evictions: u64,
    /// Shell planes written alone, after a neighbour's light changed.
    pub plane_writes: u64,
    /// Slots written whole.
    pub full_writes: u64,
    /// Sky probes started.
    pub probes: u64,
    /// Voxel lookups sky probes made.
    pub probe_lookups: u64,
    /// Sky probes whose mask was thrown away.
    pub probes_discarded: u64,
    /// Chunk positions in the sky record.
    pub record_entries: u64,
    /// Bytes the sky record counts against its cap.
    pub record_bytes: u64,
    /// Positions the sky record forgot to stay within its cap.
    pub record_evictions: u64,
    /// Coarse chunks solved in their chunk tasks.
    pub coarse_solves: u64,
    /// Time coarse chunk tasks spent solving light, walks included, summed
    /// over them, in milliseconds.
    pub coarse_solve_ms: f64,
    /// Voxel lookups coarse chunks' walks made.
    pub coarse_lookups: u64,
    /// The bytes coarse chunks keep now as copies of their slots.
    pub coarse_copy_bytes: u64,
    /// Solves that seeded sky fill.
    pub fill_seeded: u64,
    /// Time solves spent seeding sky fill, in milliseconds.
    pub fill_seed_ms: f64,
    /// Relights started because a sky fill cell changed.
    pub fill_relights: u64,
    /// Main-thread time the sky fill field spent updating, in milliseconds.
    pub fill_update_ms: f64,
    /// Main-thread time spent on sky fill in all, in milliseconds.
    pub fill_main_thread_ms: f64,
    /// Fill probes started.
    pub fill_probes: u64,
    /// Voxel lookups fill probes made.
    pub fill_probe_lookups: u64,
}

/// The light work world `C` has done, or `None` if it has no light.
pub fn light_stats<C: VoxelWorldConfig>(world: &World) -> Option<LightStats> {
    let settings = world.get_resource::<LightSettings<C>>()?;
    let counters = settings.counters();
    let (record_entries, record_bytes, record_evictions) = {
        let record = settings.record.lock().unwrap();
        (record.len() as u64, record.bytes() as u64, record.evictions)
    };
    let ms = |nanos: &AtomicU64| nanos.load(Ordering::Relaxed) as f64 / 1e6;
    let route = world.get_resource::<LightRoute<C>>();
    let slots = route.map(|route| route.counters).unwrap_or_default();
    Some(LightStats {
        slot_allocations: slots.allocations,
        late_allocations: slots.late_allocations,
        slots_given_back: slots.given_back,
        evictions: slots.evictions,
        plane_writes: slots.plane_writes,
        full_writes: slots.full_writes,
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
        probes: counters.probes.load(Ordering::Relaxed),
        probe_lookups: counters.probe_lookups.load(Ordering::Relaxed),
        probes_discarded: counters.probes_discarded.load(Ordering::Relaxed),
        record_entries,
        record_bytes,
        record_evictions,
        coarse_solves: counters.coarse_solves.load(Ordering::Relaxed),
        coarse_solve_ms: ms(&counters.coarse_nanos),
        coarse_lookups: counters.coarse_lookups.load(Ordering::Relaxed),
        coarse_copy_bytes: route.map_or(0, |route| route.coarse_copy_bytes() as u64),
        fill_seeded: counters.fill_seeded.load(Ordering::Relaxed),
        fill_seed_ms: ms(&counters.fill_seed_nanos),
        fill_relights: counters.fill_relights.load(Ordering::Relaxed),
        fill_update_ms: ms(&counters.fill_update_nanos),
        fill_main_thread_ms: ms(&counters.fill_main_thread_nanos),
        fill_probes: counters.fill_probes.load(Ordering::Relaxed),
        fill_probe_lookups: counters.fill_probe_lookups.load(Ordering::Relaxed),
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
            black_box(solve_sky(
                neighbours,
                sky_scale,
                task.shape,
                None,
                &task.counters,
            ));
        }
        black_box(solve_voxel(neighbours, &task.voxel_scale, task.shape));
    }
    (boxes.len(), start.elapsed().as_secs_f64() * 1000.0)
}
