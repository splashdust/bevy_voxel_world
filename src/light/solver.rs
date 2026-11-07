//! The solver: works out a chunk's light from what is within light reach of
//! it.
//!
//! A chunk's light is a pure function of the occupancy, direct sky and emissive
//! voxels of the 3×3×3 chunks around it, read into a box per channel that
//! reaches that channel's reach past each of the chunk's faces. It never reads
//! a neighbour's light, so the order chunks arrive in can't change a chunk's
//! light, and no light outlives what made it.
//!
//! Sky light spreads from voxels with direct sky, each from its own chunk's
//! column answer (see [`sky`](super::sky)). Voxel light spreads from emissive
//! voxels. A channel with no source in its box does no work.
//!
//! A neighbour that isn't loaded reads as solid: no light passes through it
//! and it holds no sources. Nothing is kept for chunks that have retired, so
//! light beside a chunk that isn't loaded, whether it hasn't arrived yet or has
//! retired, can be darker than the truth, never brighter. It is exact for a
//! chunk whose whole box is loaded.
//!
//! A chunk's first solve runs in its chunk task, reading its neighbours from
//! the [world view](super::view) as the task finds it. Neighbours arriving
//! later relight it in a relight task, which runs the same solve: see
//! [`relight`](super::relight).

use std::cell::RefCell;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::thread::LocalKey;
use std::time::Instant;

use bevy::prelude::*;

use super::config::LightShape;
use super::faces::{could_reach_sooner, Face, FaceHashes};
use super::fill::seed::{CellsAround, FillSeeds};
use super::fill::FillCells;
use super::inputs::{LightInputs, Occupancy, Rescan};
use super::kernel::{LightBox, Plane, Row, Seed, LEVEL_BITS, MAX_ONE_WORD_REACH};
use super::levels::{LevelScale, LightLevels};
use super::sky::{ColumnAnswer, SkyFound, SkyWalk, WalkCounters};
use super::slot::{own_faces_see, slot_bytes, FacesSee};
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
    pub sky: ChannelLight,
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
                LightLevels::new(self.sky.at(x, y, z), self.voxel.at(x, y, z))
            }
        }
    }
}

/// What a world's chunk tasks need to solve light, shared by all of them.
pub(crate) struct LightTaskSettings<I> {
    pub voxel_scale: LevelScale,
    /// The levels of sky light, or `None` with sky light off.
    pub sky_scale: Option<LevelScale>,
    pub shape: LightShape,
    /// Whether the world's chunks show their light, so a solve turns a
    /// chunk's light into the bytes of its slot.
    pub routed: bool,
    /// Each emissive material and its level.
    pub emission: Vec<(I, u8)>,
    pub counters: Arc<LightCounters>,
    /// Counts the world's changes to these settings, so light solved under
    /// older settings can be told apart as it lands.
    pub generation: u64,
    /// The `generation` of the last change to the sky rule.
    pub sky_generation: u64,
    /// The `generation` of the last change to the emissive list. A chunk's
    /// emissive voxels found before it are stale.
    pub emission_generation: u64,
    /// The cells of the world's sky fill field, with sky light and sky fill
    /// on.
    pub fill: Option<FillCells>,
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
    /// Time chunk tasks spent working out their column answers, in
    /// nanoseconds.
    pub column_nanos: AtomicU64,
    pub walks: WalkCounters,
    /// Sky probes started.
    pub probes: AtomicU64,
    /// Voxel lookups sky probes made.
    pub probe_lookups: AtomicU64,
    /// Sky probes whose mask was thrown away, because their position was
    /// edited while they ran or the sky record was emptied.
    pub probes_discarded: AtomicU64,
    /// Coarse chunks solved, in their chunk tasks.
    pub coarse_solves: AtomicU64,
    /// Time coarse chunk tasks spent solving light, walks included, in
    /// nanoseconds.
    pub coarse_nanos: AtomicU64,
    /// Voxel lookups coarse chunks' walks made.
    pub coarse_lookups: AtomicU64,
    /// Solves that seeded sky fill.
    pub fill_seeded: AtomicU64,
    /// Time solves spent seeding sky fill, from reading the cells to the
    /// end of the 7-voxel run, in nanoseconds.
    pub fill_seed_nanos: AtomicU64,
    /// Relights started because a sky fill cell changed.
    pub fill_relights: AtomicU64,
    /// Main-thread time spent updating the sky fill field, in nanoseconds.
    pub fill_update_nanos: AtomicU64,
    /// Main-thread time spent on sky fill in all, from collecting fill
    /// probes to marking chunks to relight, in nanoseconds.
    pub fill_main_thread_nanos: AtomicU64,
    /// Fill probes started.
    pub fill_probes: AtomicU64,
    /// Voxel lookups fill probes made.
    pub fill_probe_lookups: AtomicU64,
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
    /// In a world showing its light, what the chunk's faces read from its
    /// own voxels.
    pub own: FacesSee,
    /// The bytes of the chunk's slot, in a world showing its light, for a
    /// chunk whose own voxels already call for a slot.
    pub bytes: Option<Box<[u8]>>,
    /// What a chunk task found out about the chunk's columns, with sky light
    /// on. A relight finds nothing.
    pub sky: Option<SkyFound>,
    /// The edit count of the chunk's position when its chunk task started,
    /// which its voxels reflect. 0 for a relight, which keeps the inputs it
    /// read.
    pub own_edits: u64,
    /// The world's edit frame when its chunk task started, or 0 for a
    /// relight: see [`LightSettings::edit_frame`](super::LightSettings).
    pub edit_frame: u64,
    /// The [generation](LightTaskSettings::generation) of the settings it was
    /// solved under.
    pub generation: u64,
    /// The [emission generation](LightTaskSettings::emission_generation) its
    /// inputs' emissive voxels were found under.
    pub emission: u64,
    /// For a relight that found the chunk's emissive voxels again, the inputs
    /// it read from the world view, which its light's inputs replace.
    pub rescanned: Option<Arc<LightInputs>>,
}

impl Solved {
    /// The chunk's inputs as the world view held them when the solve read
    /// them.
    pub fn read_inputs(&self) -> &Arc<LightInputs> {
        self.rescanned.as_ref().unwrap_or(&self.light.inputs)
    }
}

/// A chunk task's part in solving light: the world's settings, its world view
/// and, with sky light on, what the chunk needs to work out its direct sky.
pub(crate) struct LightTask<I> {
    pub settings: Arc<LightTaskSettings<I>>,
    pub view: LightView,
    pub walk: Option<SkyWalk<I>>,
    /// The edit count of the chunk's position when the task started.
    pub own_edits: u64,
    /// The world's edit frame when the task started.
    pub edit_frame: u64,
}

impl<I: Copy + PartialEq> LightTask<I> {
    /// Works out the chunk's column answer, scans its voxels and solves its
    /// light, reading the chunks around it from the world view. `chunk` must be
    /// a level-0 chunk with the default data shape.
    pub fn solve(self, chunk: &ChunkData<I>) -> Solved {
        let settings = &self.settings;
        let (columns, found) = match self.walk {
            Some(walk) => {
                let started = Instant::now();
                let (columns, found) = walk.answer(chunk, &settings.counters.walks);
                settings
                    .counters
                    .column_nanos
                    .fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
                (columns, Some(found))
            }
            None => (ColumnAnswer::none(), None),
        };
        let (inputs, scanned) = LightInputs::of(chunk, &settings.emission, columns);
        if scanned {
            settings
                .counters
                .emissive_scans
                .fetch_add(1, Ordering::Relaxed);
        }
        let snapshot = read_snapshot(&self.view, chunk.key.position, settings);
        let mut solved = solve(snapshot, Arc::new(inputs), settings);
        solved.sky = found;
        solved.own_edits = self.own_edits;
        solved.edit_frame = self.edit_frame;
        solved
    }
}

/// A chunk task's part when the world view already holds light for the
/// chunk's voxels: a chunk re-meshed after a voxel edit, whose light the edit
/// relights on its own. The task skips its solve and works out only what the
/// chunk's new mesh needs from that light.
pub(crate) struct KeptTask<I> {
    pub settings: Arc<LightTaskSettings<I>>,
    pub light: Arc<ChunkLight>,
    /// The world's edit frame when the task started.
    pub edit_frame: u64,
}

/// The light a chunk task kept, as [`KeptTask`] reads it.
pub(crate) struct KeptLight {
    pub light: Arc<ChunkLight>,
    /// In a world showing its light, what the chunk's faces read from its
    /// own voxels.
    pub own: FacesSee,
    /// The bytes of the chunk's slot, for a chunk whose own voxels call for
    /// one.
    pub bytes: Option<Box<[u8]>>,
    pub edit_frame: u64,
}

impl<I> KeptTask<I> {
    pub fn solve(self) -> KeptLight {
        let own = if self.settings.routed {
            own_faces_see(&self.light)
        } else {
            FacesSee::default()
        };
        let bytes = own.needs_slot().then(|| slot_bytes(&self.light));
        KeptLight {
            light: self.light,
            own,
            bytes,
            edit_frame: self.edit_frame,
        }
    }
}

/// Solves the light of the chunk at `position` again from the world view, as a
/// relight does. Returns `None` if the chunk is no longer in the view.
///
/// With `rescan`, it first finds the chunk's emissive voxels again under the
/// world's emissive list, for a chunk whose emissive voxels were found under
/// an older one.
pub(crate) fn relight<I: Copy + PartialEq>(
    settings: &LightTaskSettings<I>,
    view: &LightView,
    position: IVec3,
    rescan: Option<Rescan<I>>,
) -> Option<Solved> {
    let read = view.read().get(&position)?.light.inputs.clone();
    let (inputs, rescanned) = match rescan {
        Some(rescan) => {
            if !settings.emission.is_empty() {
                settings
                    .counters
                    .emissive_scans
                    .fetch_add(1, Ordering::Relaxed);
            }
            let inputs = read.rescanned(position, &rescan, &settings.emission);
            (Arc::new(inputs), Some(read))
        }
        None => (read, None),
    };
    let snapshot = read_snapshot(view, position, settings);
    let mut solved = solve(snapshot, inputs, settings);
    solved.rescanned = rescanned;
    Some(solved)
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
    let fill = settings
        .fill
        .as_ref()
        .filter(|_| inputs.touches_solid())
        .map(|cells| read_cells(cells, snapshot.position));
    let (sky, sky_faces) = match &settings.sky_scale {
        Some(scale) => solve_sky(
            &snapshot.neighbours,
            scale,
            settings.shape,
            fill.as_ref(),
            &settings.counters,
        ),
        None => (ChannelLight::Uniform(0), FaceHashes::uniform(0)),
    };
    let (voxel, voxel_faces) =
        solve_voxel(&snapshot.neighbours, &settings.voxel_scale, settings.shape);
    let counters = &settings.counters;
    counters.solves.fetch_add(1, Ordering::Relaxed);
    counters
        .solve_nanos
        .fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);

    let light = Arc::new(ChunkLight { inputs, sky, voxel });
    let faces = FaceHashes {
        sky: sky_faces,
        voxel: voxel_faces,
    };
    let mut seen = [None; 6];
    let mut reaches = [false; 6];
    for face in Face::ALL {
        if let Some((neighbour, version)) = &snapshot.faces[face as usize] {
            seen[face as usize] = Some(*version);
            reaches[face as usize] = could_reach_sooner(
                &light,
                neighbour,
                face,
                &settings.voxel_scale,
                settings.sky_scale.as_ref(),
            );
        }
    }
    let own = if settings.routed {
        own_faces_see(&light)
    } else {
        FacesSee::default()
    };
    let bytes = own.needs_slot().then(|| slot_bytes(&light));
    Solved {
        light,
        faces,
        seen,
        reaches,
        own,
        bytes,
        sky: None,
        own_edits: 0,
        edit_frame: 0,
        generation: settings.generation,
        emission: settings.emission_generation,
        rescanned: None,
    }
}

thread_local! {
    /// Each thread's boxes, one per row width, kept between solves so a solve
    /// allocates no box.
    static ONE_WORD_BOX: RefCell<Option<LightBox<u64>>> = const { RefCell::new(None) };
    static TWO_WORD_BOX: RefCell<Option<LightBox<u128>>> = const { RefCell::new(None) };
}

/// Solves voxel light for the chunk at the centre of `neighbours`. Returns it
/// with its face hashes, indexed by [`Face`].
pub(crate) fn solve_voxel(
    neighbours: &Neighbourhood,
    scale: &LevelScale,
    shape: LightShape,
) -> (ChannelLight, [u64; 6]) {
    let centre = centre(neighbours);
    let unlit = (ChannelLight::Uniform(0), FaceHashes::uniform(0));
    if !centre.occupancy.has_open_voxel() {
        return unlit;
    }
    let seeds = seeds(neighbours, scale);
    if seeds.is_empty() {
        return unlit;
    }
    let (planes, faces) = spread(neighbours, scale, shape, &seeds, None, None);
    (channel_light(planes, &centre.occupancy), faces)
}

/// The sky fill cells of the 27 positions around `position`.
fn read_cells(cells: &FillCells, position: IVec3) -> CellsAround {
    let cells = cells.read();
    std::array::from_fn(|index| {
        let index = index as i32;
        let offset = IVec3::new(index % 3, index / 3 % 3, index / 9) - IVec3::ONE;
        cells.get(&(position + offset)).cloned()
    })
}

/// Solves sky light for the chunk at the centre of `neighbours`, from the
/// voxels with direct sky around it and, for a chunk that takes sky fill,
/// the sky fill cells `fill` around it. Returns it with its face hashes,
/// indexed by [`Face`].
///
/// Direct sky holds level 15 and doesn't fade. A chunk whose open voxels all
/// have direct sky needs no spreading, and takes no sky fill.
pub(crate) fn solve_sky(
    neighbours: &Neighbourhood,
    scale: &LevelScale,
    shape: LightShape,
    fill: Option<&CellsAround>,
    counters: &LightCounters,
) -> (ChannelLight, [u64; 6]) {
    let centre = centre(neighbours);
    let unlit = (ChannelLight::Uniform(0), FaceHashes::uniform(0));
    if !centre.occupancy.has_open_voxel() {
        return unlit;
    }
    let direct: [Option<Box<Plane>>; 27] = std::array::from_fn(|index| {
        neighbours[index]
            .as_ref()
            .and_then(|inputs| inputs.direct_sky())
    });
    let own = direct[neighbour_index(IVec3::ZERO)].as_deref();
    if let Some(own) = own {
        let all_direct = (0..CHUNK * CHUNK).all(|column| {
            own[column]
                == !centre
                    .occupancy
                    .solid_column(column % CHUNK, column / CHUNK)
        });
        if all_direct {
            return (ChannelLight::Uniform(15), FaceHashes::uniform(15));
        }
    }
    let seeding = Instant::now();
    let seeds = fill.and_then(|around| FillSeeds::of(around, scale.reach() as usize));
    if direct.iter().all(Option::is_none) && seeds.is_none() {
        return unlit;
    }
    let (mut planes, faces) = spread(
        neighbours,
        scale,
        shape,
        &[],
        Some(&direct),
        seeds.as_ref().map(|seeds| (seeds, seeding, counters)),
    );
    if let Some(own) = own {
        for plane in planes.iter_mut() {
            for (column, direct) in plane.iter_mut().zip(own.iter()) {
                *column |= direct;
            }
        }
    }
    (channel_light(planes, &centre.occupancy), faces)
}

fn centre(neighbours: &Neighbourhood) -> &LightInputs {
    neighbours[neighbour_index(IVec3::ZERO)]
        .as_ref()
        .expect("the chunk's own inputs")
}

fn channel_light(
    planes: Box<[Plane; LEVEL_BITS]>,
    occupancy: &Occupancy,
) -> ChannelLight {
    one_level(&planes, occupancy)
        .map_or(ChannelLight::Planes(planes), ChannelLight::Uniform)
}

/// Sky fill's seeds for a solve, when it started seeding, and where to count
/// the time seeding took.
type Filling<'a> = (&'a FillSeeds, Instant, &'a LightCounters);

/// Spreads light through the box around the chunk at the centre of
/// `neighbours`, from `seeds`, from the voxels set in `lit`, a plane per
/// chunk of the neighbourhood, and from sky fill's seeds in `fill`. Returns
/// the chunk's levels and face hashes.
fn spread(
    neighbours: &Neighbourhood,
    scale: &LevelScale,
    shape: LightShape,
    seeds: &[Seed],
    lit: Option<&[Option<Box<Plane>>; 27]>,
    fill: Option<Filling<'_>>,
) -> (Box<[Plane; LEVEL_BITS]>, [u64; 6]) {
    if scale.reach() <= MAX_ONE_WORD_REACH {
        spread_in(&ONE_WORD_BOX, neighbours, scale, shape, seeds, lit, fill)
    } else {
        spread_in(&TWO_WORD_BOX, neighbours, scale, shape, seeds, lit, fill)
    }
}

fn spread_in<R: Row>(
    cached: &'static LocalKey<RefCell<Option<LightBox<R>>>>,
    neighbours: &Neighbourhood,
    scale: &LevelScale,
    shape: LightShape,
    seeds: &[Seed],
    lit: Option<&[Option<Box<Plane>>; 27]>,
    fill: Option<Filling<'_>>,
) -> (Box<[Plane; LEVEL_BITS]>, [u64; 6]) {
    cached.with_borrow_mut(|light_box| {
        let light_box = match light_box {
            Some(light_box) if light_box.reach() == scale.reach() as usize => light_box,
            _ => light_box.insert(LightBox::new(scale.reach())),
        };
        let reach = light_box.reach();
        fill_rows(light_box.open_rows_mut(), reach, |index, x, y| {
            let inputs = neighbours[index].as_ref()?;
            Some(!inputs.occupancy.solid_column(x, y))
        });
        if let Some(lit) = lit {
            fill_rows(light_box.lit_rows_mut(), reach, |index, x, y| {
                Some(lit[index].as_ref()?[x + y * CHUNK])
            });
        }
        let fill_rows = match fill {
            Some((seeds, started, counters)) => {
                let rows = seeds.rows(light_box.open_rows_mut(), reach, scale);
                counters.fill_seeded.fetch_add(1, Ordering::Relaxed);
                counters
                    .fill_seed_nanos
                    .fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
                rows
            }
            None => Vec::new(),
        };
        let mut planes: Box<[Plane; LEVEL_BITS]> =
            Box::new([[0; CHUNK * CHUNK]; LEVEL_BITS]);
        let mut faces = [0; 6];
        light_box.spread(
            seeds,
            lit.is_some(),
            &fill_rows,
            shape,
            scale,
            &mut planes,
            &mut faces,
        );
        (planes, faces)
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

/// Fills a box's `rows`, at `x + y × side`, from a `u32` column along z of
/// each chunk around the box's chunk. `column` gives the column of the chunk
/// at neighbour index `index` at its voxel `x` and `y`, or `None` for a chunk
/// that isn't loaded, whose bits stay clear.
fn fill_rows<R: Row>(
    rows: &mut [R],
    reach: usize,
    column: impl Fn(usize, usize, usize) -> Option<u32>,
) {
    let side = CHUNK + 2 * reach;
    let side_mask = if side >= R::BITS {
        !R::ZERO
    } else {
        !(!R::ZERO << side)
    };
    let reach = reach as i32;
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
            let mut row = R::ZERO;
            for cz in -1..=1 {
                let index = neighbour_index(IVec3::new(cx, cy, cz));
                let Some(bits) = column(index, lx, ly) else {
                    continue;
                };
                let bits = R::from_u32(bits);
                let shift = cz * CHUNK_SIZE_I + reach;
                row |= if shift >= 0 {
                    bits << shift as usize
                } else {
                    bits >> (-shift) as usize
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
