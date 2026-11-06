//! The solver: works out a chunk's light from what is within light reach of
//! it.
//!
//! A chunk's light is a pure function of the occupancy and emissive voxels of
//! the 3×3×3 chunks around it, read into a box that reaches `reach` voxels past
//! each of its faces. It never reads a neighbour's light, so the order chunks
//! arrive in can't change a chunk's light, and no light outlives what made it.
//!
//! A neighbour that isn't loaded reads as solid: no light passes through it
//! and it holds no emissive voxels. Nothing is kept for chunks that have
//! retired, so light beside a chunk that isn't loaded, whether it hasn't
//! arrived yet or has retired, can be darker than the truth, never brighter.
//! It is exact for a chunk whose whole box is loaded.
//!
//! A chunk's first solve runs in its chunk task, reading its neighbours from
//! the [world view](super::view) as the task finds it. Neighbours arriving
//! later relight it in a relight task, which runs the same solve: see
//! [`relight`](super::relight).

use std::cell::RefCell;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use bevy::prelude::*;

use super::config::LightShape;
use super::faces::{could_reach_sooner, Face, FaceHashes};
use super::inputs::{LightInputs, Occupancy};
use super::kernel::{LightBox, Plane, Seed, LEVEL_BITS};
use super::levels::{LevelScale, LightLevels};
use super::view::{LightView, Snapshot};
use crate::chunk::{ChunkData, CHUNK_SIZE_I, CHUNK_SIZE_U};
use crate::voxel::WorldVoxel;

const CHUNK: usize = CHUNK_SIZE_U as usize;

/// One channel of a chunk's light: one level for every open voxel, or a level
/// per voxel as bit planes, 16 KiB at 16 levels.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum ChannelLight {
    Uniform(u8),
    Planes(Box<[Plane; LEVEL_BITS]>),
}

impl ChannelLight {
    /// The level at the chunk's voxel `x`, `y`, `z`, which must be open.
    pub fn at(&self, x: usize, y: usize, z: usize) -> u8 {
        match self {
            ChannelLight::Uniform(level) => *level,
            ChannelLight::Planes(planes) => planes
                .iter()
                .enumerate()
                .map(|(bit, plane)| ((plane[x + y * CHUNK] >> z & 1) as u8) << bit)
                .sum(),
        }
    }
}

/// A level-0 chunk's light, and what it keeps for its neighbours' solves.
#[derive(Debug)]
pub(crate) struct ChunkLight {
    pub inputs: Arc<LightInputs>,
    /// One level for every open voxel until sky light is worked out.
    pub sky: u8,
    pub voxel: ChannelLight,
}

impl ChunkLight {
    /// Whether `other` holds the same levels as this light in every voxel.
    pub fn same_levels(&self, other: &ChunkLight) -> bool {
        self.sky == other.sky && self.voxel == other.voxel
    }

    /// The light at the chunk's voxel `local`, which holds `voxel`. A solid
    /// voxel holds no light, except an emissive one, which holds its level.
    pub fn levels_at<I: Copy + PartialEq>(
        &self,
        local: UVec3,
        voxel: WorldVoxel<I>,
        emission: &[(I, u8)],
    ) -> LightLevels {
        match voxel {
            WorldVoxel::Solid(material) => {
                let level = emission
                    .iter()
                    .find(|(emissive, _)| *emissive == material)
                    .map_or(0, |(_, level)| *level);
                LightLevels::new(0, level)
            }
            _ => {
                let (x, y, z) = (local.x as usize, local.y as usize, local.z as usize);
                LightLevels::new(self.sky, self.voxel.at(x, y, z))
            }
        }
    }
}

/// What a world's chunk tasks need to solve light, shared by all of them.
pub(crate) struct LightTaskSettings<I> {
    pub voxel_scale: LevelScale,
    pub shape: LightShape,
    /// The level of sky light in every open voxel, until sky light is solved.
    pub sky: u8,
    /// Whether the world's chunks show their light, so a solve turns a
    /// chunk's light into the bytes of its slot.
    pub routed: bool,
    /// Each emissive material and its level.
    pub emission: Vec<(I, u8)>,
    pub counters: Arc<LightCounters>,
}

/// Counts of light work in a world, for tests and benchmarks.
#[derive(Default)]
pub(crate) struct LightCounters {
    /// Chunks whose voxels were scanned for emissive voxels.
    pub emissive_scans: AtomicU64,
    pub solves: AtomicU64,
    /// Time spent solving, in nanoseconds, summed over every chunk task.
    pub solve_nanos: AtomicU64,
    /// Time spent reading the world view before solving, in nanoseconds,
    /// summed over every chunk task and relight task.
    pub snapshot_nanos: AtomicU64,
    /// Relight tasks started.
    pub relights: AtomicU64,
    /// Main-thread time spent landing light and starting relights, in
    /// nanoseconds.
    pub main_thread_nanos: AtomicU64,
    /// The most main-thread time spent landing light and starting relights
    /// in one frame, in nanoseconds.
    pub max_frame_nanos: AtomicU64,
}

/// The 3×3×3 chunk positions around a chunk, at
/// `(x + 1) + (y + 1) × 3 + (z + 1) × 9` for offsets from -1 to 1.
pub(crate) type Neighbourhood = [Option<Arc<LightInputs>>; 27];

pub(crate) fn neighbour_index(offset: IVec3) -> usize {
    ((offset.x + 1) + (offset.y + 1) * 3 + (offset.z + 1) * 9) as usize
}

/// A finished solve: the chunk's light and what landing it needs.
pub(crate) struct Solved {
    pub light: Arc<ChunkLight>,
    pub faces: FaceHashes,
    /// The version of each face neighbour's light the solve compared against,
    /// by [`Face`], or `None` where it had no neighbour.
    pub seen: [Option<u64>; 6],
    /// By [`Face`], whether the chunk's light could reach the neighbour sooner
    /// than the neighbour's light says, judged against the version in `seen`.
    pub reaches: [bool; 6],
    /// The bytes of the chunk's slot, in a world showing its light, for a
    /// chunk with both solid and open voxels, so with faces, whose light
    /// varies.
    pub bytes: Option<Box<[u8]>>,
}

/// A chunk task's part in solving light: the world's settings and its world
/// view.
pub(crate) struct LightTask<I> {
    pub settings: Arc<LightTaskSettings<I>>,
    pub view: LightView,
}

impl<I: Copy + PartialEq> LightTask<I> {
    /// Scans the chunk's voxels and solves its light, reading the chunks around
    /// it from the world view. `chunk` must be a level-0 chunk with the default
    /// data shape.
    pub fn solve(self, chunk: &ChunkData<I>) -> Solved {
        let settings = &self.settings;
        let (inputs, scanned) = LightInputs::of(chunk, &settings.emission);
        if scanned {
            settings
                .counters
                .emissive_scans
                .fetch_add(1, Ordering::Relaxed);
        }
        let snapshot = read_snapshot(&self.view, chunk.key.position, settings);
        solve(snapshot, Arc::new(inputs), settings)
    }
}

/// Solves the light of the chunk at `position` again from the world view, as a
/// relight does. Returns `None` if the chunk is no longer in the view.
pub(crate) fn relight<I>(
    settings: &LightTaskSettings<I>,
    view: &LightView,
    position: IVec3,
) -> Option<Solved> {
    let inputs = view.read().get(&position)?.light.inputs.clone();
    let snapshot = read_snapshot(view, position, settings);
    Some(solve(snapshot, inputs, settings))
}

fn read_snapshot<I>(
    view: &LightView,
    position: IVec3,
    settings: &LightTaskSettings<I>,
) -> Snapshot {
    let started = Instant::now();
    let snapshot = Snapshot::read(&view.read(), position);
    settings
        .counters
        .snapshot_nanos
        .fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
    snapshot
}

/// Solves the light of the chunk whose own inputs are `inputs`, among the
/// chunks around it in `snapshot`, and compares it with its face neighbours'
/// light.
fn solve<I>(
    mut snapshot: Snapshot,
    inputs: Arc<LightInputs>,
    settings: &LightTaskSettings<I>,
) -> Solved {
    snapshot.neighbours[neighbour_index(IVec3::ZERO)] = Some(inputs.clone());
    let started = Instant::now();
    let (voxel, voxel_faces) =
        solve_channel(&snapshot.neighbours, &settings.voxel_scale, settings.shape);
    let counters = &settings.counters;
    counters.solves.fetch_add(1, Ordering::Relaxed);
    counters
        .solve_nanos
        .fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);

    let light = Arc::new(ChunkLight {
        inputs,
        sky: settings.sky,
        voxel,
    });
    let faces = FaceHashes {
        sky: FaceHashes::uniform(settings.sky),
        voxel: voxel_faces,
    };
    let mut seen = [None; 6];
    let mut reaches = [false; 6];
    for face in Face::ALL {
        if let Some((neighbour, version)) = &snapshot.faces[face as usize] {
            seen[face as usize] = Some(*version);
            reaches[face as usize] =
                could_reach_sooner(&light, neighbour, face, &settings.voxel_scale);
        }
    }
    let has_faces = matches!(light.inputs.occupancy, Occupancy::Mixed(_));
    let bytes = (settings.routed && has_faces)
        .then(|| super::route::slot_bytes(&light))
        .flatten();
    Solved {
        light,
        faces,
        seen,
        reaches,
        bytes,
    }
}

thread_local! {
    /// Each thread's box, kept between solves so a solve allocates nothing.
    static LIGHT_BOX: RefCell<Option<LightBox>> = const { RefCell::new(None) };
}

/// Solves voxel light for the chunk at the centre of `neighbours`. Returns it
/// with its face hashes, indexed by [`Face`].
pub(crate) fn solve_channel(
    neighbours: &Neighbourhood,
    scale: &LevelScale,
    shape: LightShape,
) -> (ChannelLight, [u64; 6]) {
    let centre = neighbours[neighbour_index(IVec3::ZERO)]
        .as_ref()
        .expect("the chunk's own inputs");
    let unlit = (ChannelLight::Uniform(0), FaceHashes::uniform(0));
    if !centre.occupancy.has_open_voxel() {
        return unlit;
    }
    let seeds = seeds(neighbours, scale);
    if seeds.is_empty() {
        return unlit;
    }

    LIGHT_BOX.with_borrow_mut(|light_box| {
        let light_box = match light_box {
            Some(light_box) if light_box.reach() == scale.reach() as usize => light_box,
            _ => light_box.insert(LightBox::new(scale.reach())),
        };
        fill_open_rows(light_box, neighbours);
        let mut planes: Box<[Plane; LEVEL_BITS]> =
            Box::new([[0; CHUNK * CHUNK]; LEVEL_BITS]);
        let mut faces = [0; 6];
        light_box.spread(&seeds, shape, scale, &mut planes, &mut faces);
        let light = one_level(&planes, &centre.occupancy)
            .map_or(ChannelLight::Planes(planes), ChannelLight::Uniform);
        (light, faces)
    })
}

/// The emissive voxels whose light could reach the chunk, as seeds in its box,
/// sorted by where their light starts.
fn seeds(neighbours: &Neighbourhood, scale: &LevelScale) -> Vec<Seed> {
    let reach = scale.reach() as i32;
    let side = CHUNK_SIZE_I + 2 * reach;
    let mut seeds = Vec::new();
    for (index, inputs) in neighbours.iter().enumerate() {
        let Some(inputs) = inputs else {
            continue;
        };
        let index = index as i32;
        let offset = IVec3::new(index % 3, index / 3 % 3, index / 9) - IVec3::ONE;
        for voxel in &inputs.emissive {
            let Some(start) = scale.start(voxel.level) else {
                continue;
            };
            let position = offset * CHUNK_SIZE_I
                + IVec3::new(voxel.x as i32, voxel.y as i32, voxel.z as i32)
                + IVec3::splat(reach);
            if position.cmplt(IVec3::ZERO).any()
                || position.cmpge(IVec3::splat(side)).any()
            {
                continue;
            }
            // How far outside the chunk the voxel is, at least one step.
            let outside = (IVec3::splat(reach) - position)
                .max(position - IVec3::splat(reach + CHUNK_SIZE_I - 1))
                .max_element()
                .max(1);
            if outside + start as i32 > reach {
                continue;
            }
            seeds.push(Seed {
                x: position.x as u8,
                y: position.y as u8,
                z: position.z as u8,
                start,
            });
        }
    }
    seeds.sort_by_key(|seed| seed.start);
    seeds
}

/// Fills the box's open rows from the occupancy of the chunks around it. A
/// chunk that isn't loaded is solid.
fn fill_open_rows(light_box: &mut LightBox, neighbours: &Neighbourhood) {
    let reach = light_box.reach() as i32;
    let side = light_box.side();
    let side_mask = if side >= 64 { !0 } else { (1u64 << side) - 1 };
    let rows = light_box.open_rows_mut();
    for by in 0..side {
        let wy = by as i32 - reach;
        let (cy, ly) = (
            wy.div_euclid(CHUNK_SIZE_I),
            wy.rem_euclid(CHUNK_SIZE_I) as usize,
        );
        for bx in 0..side {
            let wx = bx as i32 - reach;
            let (cx, lx) = (
                wx.div_euclid(CHUNK_SIZE_I),
                wx.rem_euclid(CHUNK_SIZE_I) as usize,
            );
            let mut row = 0u64;
            for cz in -1..=1 {
                let index = neighbour_index(IVec3::new(cx, cy, cz));
                let Some(inputs) = &neighbours[index] else {
                    continue;
                };
                let open = !inputs.occupancy.solid_column(lx, ly) as u64;
                let shift = cz * CHUNK_SIZE_I + reach;
                row |= if shift >= 0 {
                    open << shift
                } else {
                    open >> -shift
                };
            }
            rows[bx + by * side] = row & side_mask;
        }
    }
}

/// The level every open voxel of the chunk holds, if they all hold the same.
fn one_level(planes: &[Plane; LEVEL_BITS], occupancy: &Occupancy) -> Option<u8> {
    let mut level = None;
    for y in 0..CHUNK {
        for x in 0..CHUNK {
            let open = !occupancy.solid_column(x, y);
            if open == 0 {
                continue;
            }
            let column = x + y * CHUNK;
            let level = *level.get_or_insert_with(|| {
                let z = open.trailing_zeros();
                (0..LEVEL_BITS)
                    .map(|bit| ((planes[bit][column] >> z & 1) as u8) << bit)
                    .sum::<u8>()
            });
            for (bit, plane) in planes.iter().enumerate() {
                let want = if level >> bit & 1 == 1 { open } else { 0 };
                if plane[column] & open != want {
                    return None;
                }
            }
        }
    }
    Some(level.unwrap_or(0))
}
