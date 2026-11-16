//! The solver: works out a chunk's light from what is within light reach of
//! it.
//!
//! A chunk's light is a pure function of the occupancy, direct sky and emissive
//! voxels of the 3×3×3 chunks around it, read into a box for sky light and one
//! for voxel light, each reaching that light's reach past each of the chunk's
//! faces. It never reads a neighbour's light, so the order chunks arrive in
//! can't change a chunk's light, and no light outlives what made it.
//!
//! Sky light spreads from voxels with direct sky, each from its own chunk's
//! column answer (see [`sky`](super::sky)). Voxel light spreads from emissive
//! voxels. A kind of light with no source in its box does no work. A solve can
//! instead be given its box's voxels with direct sky as [`DirectSky::Cells`],
//! so sky can enter a voxel from any side, and can be told to take no sky fill.
//!
//! A neighbour that isn't loaded reads as solid: no light passes through it
//! and it holds no sources. A solve can be told to read it as air instead,
//! with [`Missing`]. Air a fill hint keeps from spawning reads as what its
//! hint says: see [`hinted`](super::hinted). Every world solve reads missing
//! neighbours as solid, takes its direct sky from column answers and takes sky
//! fill where the world has it. Nothing is kept for chunks that have retired, so
//! light beside a chunk that isn't loaded, whether it hasn't arrived yet or has
//! retired, can be darker than the truth, never brighter. It is exact for a
//! chunk whose whole box is loaded.
//!
//! A chunk's first solve runs in its chunk task, reading its neighbours from
//! the [world view](super::view) as the task finds it. Neighbours arriving
//! later relight it in a relight task, which runs the same solve: see
//! [`relight`](super::relight).
//!
//! # References
//!
//! - [Veloren's terrain mesher](https://gitlab.com/veloren/veloren/-/blob/master/voxygen/src/mesh/terrain.rs):
//!   works a chunk's light out again for each mesh, from the voxels in a halo
//!   around the chunk.
//! - [rustcraft's lighting notes](https://github.com/ArturTrumpfheller/rustcraft/blob/main/docs/LIGHTING.md):
//!   works a region's light out again from a halo as wide as light's reach,
//!   reading voxels it doesn't know as solid.
//! - [tesserae's light propagation](https://github.com/mkappworks-dev/tesserae/pull/3):
//!   found that solving a chunk again from its neighbours' light can't take
//!   light away, since two chunks can hold each other's light up. So a solve
//!   here reads only voxels.

use std::cell::RefCell;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::thread::LocalKey;
use std::time::Duration;

use bevy::platform::time::Instant;
use bevy::prelude::*;

use super::config::LightPrecision;
use super::config::LightShape;
use super::faces::{could_reach_sooner, Face, FaceHashes};
use super::fill::seed::{CellsAround, FillSeeds, Seeding};
use super::fill::{FillCells, CELL};
use super::inputs::{LightInputs, Occupancy, Rescan};
use super::kernel::{LightBox, Plane, Row, Seed, TwoWords, MAX_ONE_WORD_REACH};
use super::levels::{LevelScale, LightLevels};
use super::sky::{ColumnAnswer, SkyFound, SkyWalk, WalkCounters};
use super::slot::{own_faces_see, slot_bytes, FacesSee};
use super::view::{world_position, LightView, Missing, Snapshot};
use crate::chunk::{ChunkData, CHUNK_SIZE_I, CHUNK_SIZE_U};
use crate::grid::{GridId, GridKey};
use crate::voxel::WorldVoxel;

const CHUNK: usize = CHUNK_SIZE_U as usize;

/// One kind of a chunk's light, sky or voxel, in steps of its precision: one
/// level for every open voxel, or a level per voxel as one bit plane per bit of
/// the level, 16 KiB at 16 levels and 32 KiB at 256.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum ChunkLevels {
    Uniform(u8),
    Planes(Box<[Plane]>),
}

impl ChunkLevels {
    /// The level at the chunk's voxel `x`, `y`, `z`, which must be open.
    pub fn at(&self, x: usize, y: usize, z: usize) -> u8 {
        match self {
            ChunkLevels::Uniform(level) => *level,
            ChunkLevels::Planes(planes) => planes
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
    pub sky: ChunkLevels,
    pub voxel: ChunkLevels,
    /// The precision its levels are steps of: the world's when it was
    /// solved, which a change of precision leaves behind until it relights.
    pub precision: LightPrecision,
}

impl ChunkLight {
    /// Whether `other` holds the same levels as this light in every voxel.
    pub fn same_levels(&self, other: &ChunkLight) -> bool {
        self.precision == other.precision
            && self.sky == other.sky
            && self.voxel == other.voxel
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
                LightLevels::of_steps(
                    self.sky.at(x, y, z),
                    self.voxel.at(x, y, z),
                    self.precision,
                )
            }
        }
    }
}

/// What a world's chunk tasks need to solve light, shared by all of them.
pub(crate) struct LightTaskSettings<I> {
    pub precision: LightPrecision,
    pub voxel_scale: LevelScale,
    /// The levels of sky light, or `None` with sky light off.
    pub sky_scale: Option<LevelScale>,
    pub shape: LightShape,
    /// Whether the world's chunks show their light, so a solve turns a
    /// chunk's light into the bytes of its slot.
    pub routed: bool,
    /// Each emissive material and its level, as a whole level from 0 to 15.
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
    /// Of those, the relight tasks started for sub-grid chunks.
    #[cfg(feature = "sub_grids")]
    pub sub_grid_relights: AtomicU64,
    /// Sub-grid chunks solved in their chunk tasks. A sub-grid's commit
    /// leaves its chunks' solves to relights, so this counts only chunks
    /// that got no inputs from a commit.
    #[cfg(feature = "sub_grids")]
    pub sub_grid_chunk_solves: AtomicU64,
    /// Time relight tasks spent, from reading the world view to the end of
    /// their solve, in nanoseconds.
    pub relight_nanos: AtomicU64,
    /// Relight tasks started for light entries.
    pub entry_relights: AtomicU64,
    /// Time relight tasks for light entries spent, in nanoseconds.
    pub entry_relight_nanos: AtomicU64,
    /// Main-thread time spent landing light and starting relights, in
    /// nanoseconds.
    pub main_thread_nanos: AtomicU64,
    /// The most main-thread time spent landing light and starting relights
    /// in one frame, in nanoseconds.
    pub max_frame_nanos: AtomicU64,
    /// The most time relights ran on the main thread in one frame, in
    /// nanoseconds. Relights run there only on WASM.
    pub max_relight_frame_nanos: AtomicU64,
    /// The longest relight run on the main thread, in nanoseconds.
    pub max_relight_nanos: AtomicU64,
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
    /// Time solves spent seeding sky fill, from the cells around the chunk
    /// to the end of the 7-voxel run, in nanoseconds. Filling the box's open
    /// and lit rows, which every solve of sky light does, isn't counted.
    pub fill_seed_nanos: AtomicU64,
    /// Relights started because a sky fill cell changed.
    pub fill_relights: AtomicU64,
    /// Of those, the relights that stopped before solving because the
    /// chunk's seeds were the ones its light was solved from.
    pub fill_relights_skipped: AtomicU64,
    /// Main-thread time spent updating the sky fill field, in nanoseconds.
    pub fill_update_nanos: AtomicU64,
    /// Main-thread time spent on sky fill in all, from collecting fill
    /// probes to marking chunks to relight, in nanoseconds.
    pub fill_main_thread_nanos: AtomicU64,
    /// Fill probes started.
    pub fill_probes: AtomicU64,
    /// Voxel lookups fill probes made.
    pub fill_probe_lookups: AtomicU64,
    /// Walks `light_at` ran on request, where nothing renders.
    pub requests: super::request::RequestCounters,
}

/// The 3×3×3 chunk positions around a chunk, at
/// `(x + 1) + (y + 1) × 3 + (z + 1) × 9` for offsets from -1 to 1.
pub(crate) type Neighbourhood = [Option<Arc<LightInputs>>; 27];

/// A plane of voxels for each chunk position of a [`Neighbourhood`], at the
/// same index, or `None` where none is set.
pub(crate) type DirectCells = [Option<Box<Plane>>; 27];

/// Where a solve finds the voxels of its box that have direct sky.
pub(crate) enum DirectSky {
    /// In each chunk's column answer: in each column the answer lets sky
    /// light into, the open voxels from the chunk's top down to the first
    /// solid voxel. Sky light only enters going down the grid's -Y.
    Columns,
    /// In these cells, worked out before the solve, which can give a voxel
    /// direct sky from any side. Every cell set must be open. A column answer
    /// adds nothing to them.
    Cells(Box<DirectCells>),
}

/// How a solve of the chunk at `key` reads its box: how it reads positions
/// holding nothing, and whether it takes sky fill.
///
/// A world chunk reads missing chunks as solid and takes sky fill. A
/// sub-grid's chunk reads its grid's missing chunks as air and takes no sky
/// fill.
fn reading(key: GridKey) -> (Missing, bool) {
    if key.is_world() {
        (Missing::Solid, true)
    } else {
        (Missing::Air, false)
    }
}

/// Where the solve of the chunk at `key`, whose own inputs are `inputs`,
/// finds direct sky in the box `snapshot` read, with sky light on.
///
/// A world chunk takes it from column answers. A sub-grid's chunk takes it
/// from its enclosure where the snapshot holds its grid's line record: see
/// the `enclosure` module. Otherwise every open voxel of its own
/// has direct sky, so its own sky is the top level everywhere.
fn direct_sky(key: GridKey, snapshot: &Snapshot, inputs: &LightInputs) -> DirectSky {
    if key.is_world() {
        return DirectSky::Columns;
    }
    #[cfg(feature = "sub_grids")]
    if let Some(lines) = &snapshot.lines {
        return DirectSky::Cells(super::sub_grid::enclosure::direct_cells(
            &snapshot.neighbours,
            inputs,
            key.position(),
            lines,
        ));
    }
    #[cfg(not(feature = "sub_grids"))]
    let _ = snapshot;
    let mut own: Box<Plane> = Box::new([0; CHUNK * CHUNK]);
    for (index, column) in own.iter_mut().enumerate() {
        *column = !inputs.occupancy.solid_column(index % CHUNK, index / CHUNK);
    }
    let mut cells: Box<DirectCells> = Box::new(std::array::from_fn(|_| None));
    cells[neighbour_index(IVec3::ZERO)] = Some(own);
    DirectSky::Cells(cells)
}

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
    /// The edit stamp of the chunk's position when its chunk task started,
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
    /// The hash of the sky fill seeds it was solved from, for a chunk that
    /// takes sky fill: see [`Seeding`].
    pub fill_seeds: Option<u64>,
    /// The inputs of the chunks and hinted air around the chunk as the solve
    /// read them, its own included.
    pub read: Box<Neighbourhood>,
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
    /// The grid of the chunk the task solves.
    pub grid: GridId,
    pub view: LightView,
    pub walk: Option<SkyWalk<I>>,
    /// The edit stamp of the chunk's position when the task started.
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
        let key = GridKey::new(self.grid, chunk.key);
        let (missing, take_fill) = reading(key);
        let seeding = seeding(settings, &inputs, key, take_fill);
        let snapshot = read_snapshot(&self.view, key, missing, settings);
        let direct = direct_sky_if_lit(settings, key, &snapshot, &inputs);
        let mut solved = solve(snapshot, Arc::new(inputs), settings, seeding, direct);
        #[cfg(feature = "sub_grids")]
        if self.grid != GridId::WORLD {
            settings
                .counters
                .sub_grid_chunk_solves
                .fetch_add(1, Ordering::Relaxed);
        }
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

/// Solves the light of the chunk at `key` again from the world view, as a
/// relight does, reading the chunks around it on its grid. Returns `None` if
/// the chunk is no longer in the view.
///
/// With `rescan`, it first finds the chunk's emissive voxels again under the
/// world's emissive list, for a chunk whose emissive voxels were found under
/// an older one.
pub(crate) fn relight<I: Copy + PartialEq>(
    settings: &LightTaskSettings<I>,
    view: &LightView,
    key: GridKey,
    rescan: Option<Rescan<I>>,
    unless_seeds: Option<u64>,
) -> Option<Solved> {
    let read = view.read().chunks.get(&key)?.light.inputs.clone();
    let (inputs, rescanned) = match rescan {
        Some(rescan) => {
            if !settings.emission.is_empty() {
                settings
                    .counters
                    .emissive_scans
                    .fetch_add(1, Ordering::Relaxed);
            }
            let inputs = read.rescanned(key.position(), &rescan, &settings.emission);
            (Arc::new(inputs), Some(read))
        }
        None => (read, None),
    };
    let (missing, take_fill) = reading(key);
    let seeding = seeding(settings, &inputs, key, take_fill);
    if let (Some(seeding), Some(solved_from)) = (&seeding, unless_seeds) {
        if seeding.hash == solved_from && rescanned.is_none() {
            settings
                .counters
                .fill_relights_skipped
                .fetch_add(1, Ordering::Relaxed);
            return None;
        }
    }
    let snapshot = read_snapshot(view, key, missing, settings);
    let direct = direct_sky_if_lit(settings, key, &snapshot, &inputs);
    let mut solved = solve(snapshot, inputs, settings, seeding, direct);
    solved.rescanned = rescanned;
    Some(solved)
}

/// The [`direct_sky`] of the chunk at `key`, worked out only with sky light
/// on. With it off no solve reads direct sky, so it gives the column answers,
/// which cost nothing to name.
fn direct_sky_if_lit<I>(
    settings: &LightTaskSettings<I>,
    key: GridKey,
    snapshot: &Snapshot,
    inputs: &LightInputs,
) -> DirectSky {
    if settings.sky_scale.is_none() {
        return DirectSky::Columns;
    }
    direct_sky(key, snapshot, inputs)
}

/// The sky fill seeds of the chunk at `key` with `inputs`, for a chunk
/// that takes sky fill in a world with sky light and sky fill on. A solve
/// that doesn't `take_fill` seeds none. Only the world has sky fill, so
/// another grid's chunk takes none either way.
fn seeding<I>(
    settings: &LightTaskSettings<I>,
    inputs: &LightInputs,
    key: GridKey,
    take_fill: bool,
) -> Option<Seeding> {
    if !take_fill {
        return None;
    }
    let (cells, scale) = (settings.fill.as_ref()?, settings.sky_scale.as_ref()?);
    let position = world_position(&key)?;
    if !inputs.touches_solid() {
        return None;
    }
    Some(Seeding::of(&read_cells(cells, position), scale))
}

fn read_snapshot<I>(
    view: &LightView,
    key: GridKey,
    missing: Missing,
    settings: &LightTaskSettings<I>,
) -> Snapshot {
    let started = Instant::now();
    let snapshot = Snapshot::read(&view.read(), key, missing);
    settings
        .counters
        .snapshot_nanos
        .fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
    snapshot
}

/// Solves the light of the chunk whose own inputs are `inputs`, among the
/// chunks around it in `snapshot`, with its sky fill seeds `seeding` and its
/// box's direct sky from `direct`, and compares it with its face neighbours'
/// light.
fn solve<I>(
    mut snapshot: Snapshot,
    inputs: Arc<LightInputs>,
    settings: &LightTaskSettings<I>,
    seeding: Option<Seeding>,
    direct: DirectSky,
) -> Solved {
    snapshot.neighbours[neighbour_index(IVec3::ZERO)] = Some(inputs.clone());
    let started = Instant::now();
    let fill_seeds = seeding.as_ref().map(|seeding| seeding.hash);
    let (sky, sky_faces) = match &settings.sky_scale {
        Some(scale) => solve_sky(
            &snapshot.neighbours,
            snapshot.missing,
            direct,
            scale,
            settings.shape,
            seeding.as_ref(),
            &settings.counters,
        ),
        None => (ChunkLevels::Uniform(0), FaceHashes::uniform(0)),
    };
    let (voxel, voxel_faces) = solve_voxel(
        &snapshot.neighbours,
        snapshot.missing,
        &settings.voxel_scale,
        settings.shape,
    );
    let counters = &settings.counters;
    counters.solves.fetch_add(1, Ordering::Relaxed);
    counters
        .solve_nanos
        .fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);

    let light = Arc::new(ChunkLight {
        inputs,
        sky,
        voxel,
        precision: settings.precision,
    });
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
        fill_seeds,
        read: Box::new(snapshot.neighbours),
    }
}

thread_local! {
    /// Each thread's boxes, one per row width, kept between solves so a solve
    /// allocates no box.
    static ONE_WORD_BOX: RefCell<Option<LightBox<u64>>> = const { RefCell::new(None) };
    static TWO_WORD_BOX: RefCell<Option<LightBox<TwoWords>>> = const { RefCell::new(None) };
}

/// Solves voxel light for the chunk at the centre of `neighbours`, reading
/// the positions that hold nothing as `missing` says. Returns it with its
/// face hashes, indexed by [`Face`].
pub(crate) fn solve_voxel(
    neighbours: &Neighbourhood,
    missing: Missing,
    scale: &LevelScale,
    shape: LightShape,
) -> (ChunkLevels, [u64; 6]) {
    let centre = centre(neighbours);
    let unlit = (ChunkLevels::Uniform(0), FaceHashes::uniform(0));
    if !centre.occupancy.has_open_voxel() {
        return unlit;
    }
    let seeds = seeds(neighbours, scale);
    if seeds.is_empty() {
        return unlit;
    }
    let (planes, faces) = spread(neighbours, missing, scale, shape, &seeds, None, None);
    (chunk_levels(planes, &centre.occupancy), faces)
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

/// Solves sky light for the chunk at the centre of `neighbours`, reading the
/// positions that hold nothing as `missing` says. It spreads from the voxels
/// `direct` gives direct sky and, for a chunk that takes sky fill, the sky
/// fill seeds `fill` found from the cells around it, and the voxels of the
/// cells sky fill counts as direct sky. Returns it with its face hashes,
/// indexed by [`Face`].
///
/// Direct sky holds the top level and doesn't fade. A chunk whose open voxels
/// all have direct sky needs no spreading, and takes no sky fill.
pub(crate) fn solve_sky(
    neighbours: &Neighbourhood,
    missing: Missing,
    direct: DirectSky,
    scale: &LevelScale,
    shape: LightShape,
    fill: Option<&Seeding>,
    counters: &LightCounters,
) -> (ChunkLevels, [u64; 6]) {
    let centre = centre(neighbours);
    let unlit = (ChunkLevels::Uniform(0), FaceHashes::uniform(0));
    if !centre.occupancy.has_open_voxel() {
        return unlit;
    }
    let mut direct: DirectCells = match direct {
        DirectSky::Columns => std::array::from_fn(|index| {
            neighbours[index]
                .as_ref()
                .and_then(|inputs| inputs.direct_sky())
        }),
        DirectSky::Cells(cells) => *cells,
    };
    if let Some(seeding) = fill {
        add_sunlit(&mut direct, neighbours, &seeding.sunlit);
    }
    let own = direct[neighbour_index(IVec3::ZERO)].as_deref();
    if let Some(own) = own {
        let all_direct = (0..CHUNK * CHUNK).all(|column| {
            own[column]
                == !centre
                    .occupancy
                    .solid_column(column % CHUNK, column / CHUNK)
        });
        if all_direct {
            let top = scale.precision().top();
            return (ChunkLevels::Uniform(top), FaceHashes::uniform(top));
        }
    }
    let seeds = fill.and_then(|seeding| Some((seeding.seeds.as_ref()?, seeding.took)));
    if direct.iter().all(Option::is_none) && seeds.is_none() {
        return unlit;
    }
    let (mut planes, faces) = spread(
        neighbours,
        missing,
        scale,
        shape,
        &[],
        Some(&direct),
        seeds.map(|(seeds, took)| (seeds, took, counters)),
    );
    if let Some(own) = own {
        for plane in planes.iter_mut() {
            for (column, direct) in plane.iter_mut().zip(own.iter()) {
                *column |= direct;
            }
        }
    }
    (chunk_levels(planes, &centre.occupancy), faces)
}

/// Adds to `direct` the voxels of the cells `sunlit` holds for each position
/// of `neighbours` that has inputs, which sky fill counts as direct sky.
fn add_sunlit(direct: &mut DirectCells, neighbours: &Neighbourhood, sunlit: &[u64; 27]) {
    for (index, (plane, cells)) in direct.iter_mut().zip(sunlit).enumerate() {
        if *cells == 0 || neighbours[index].is_none() {
            continue;
        }
        let plane = plane.get_or_insert_with(|| Box::new([0; CHUNK * CHUNK]));
        let mut left = *cells;
        while left != 0 {
            let bit = left.trailing_zeros() as usize;
            left &= left - 1;
            let (cx, cy, cz) = (bit & 3, bit >> 2 & 3, bit >> 4);
            let z_bits = 0xffu32 << (cz * CELL);
            for y in cy * CELL..(cy + 1) * CELL {
                for x in cx * CELL..(cx + 1) * CELL {
                    plane[x + y * CHUNK] |= z_bits;
                }
            }
        }
    }
}

fn centre(neighbours: &Neighbourhood) -> &LightInputs {
    neighbours[neighbour_index(IVec3::ZERO)]
        .as_ref()
        .expect("the chunk's own inputs")
}

fn chunk_levels(planes: Box<[Plane]>, occupancy: &Occupancy) -> ChunkLevels {
    one_level(&planes, occupancy)
        .map_or(ChunkLevels::Planes(planes), ChunkLevels::Uniform)
}

/// Sky fill's seeds for a solve, the time finding them took, and where to
/// count the time seeding took.
type Filling<'a> = (&'a FillSeeds, Duration, &'a LightCounters);

/// Spreads light through the box around the chunk at the centre of
/// `neighbours`, from `seeds`, from the voxels set in `lit`, a plane per
/// chunk of the neighbourhood, and from sky fill's seeds in `fill`. The
/// positions of `neighbours` that hold nothing read as `missing` says.
/// Returns the chunk's levels and face hashes.
fn spread(
    neighbours: &Neighbourhood,
    missing: Missing,
    scale: &LevelScale,
    shape: LightShape,
    seeds: &[Seed],
    lit: Option<&DirectCells>,
    fill: Option<Filling<'_>>,
) -> (Box<[Plane]>, [u64; 6]) {
    let around = (neighbours, missing);
    if scale.reach() <= MAX_ONE_WORD_REACH {
        spread_in(&ONE_WORD_BOX, around, scale, shape, seeds, lit, fill)
    } else {
        spread_in(&TWO_WORD_BOX, around, scale, shape, seeds, lit, fill)
    }
}

fn spread_in<R: Row>(
    cached: &'static LocalKey<RefCell<Option<LightBox<R>>>>,
    (neighbours, missing): (&Neighbourhood, Missing),
    scale: &LevelScale,
    shape: LightShape,
    seeds: &[Seed],
    lit: Option<&DirectCells>,
    fill: Option<Filling<'_>>,
) -> (Box<[Plane]>, [u64; 6]) {
    cached.with_borrow_mut(|light_box| {
        let light_box = match light_box {
            Some(light_box) if light_box.reach() == scale.reach() as usize => light_box,
            _ => light_box.insert(LightBox::new(scale.reach())),
        };
        let reach = light_box.reach();
        fill_rows(light_box.open_rows_mut(), reach, |index, x, y| {
            match (&neighbours[index], missing) {
                (Some(inputs), _) => Some(!inputs.occupancy.solid_column(x, y)),
                (None, Missing::Solid) => None,
                (None, Missing::Air) => Some(!0),
            }
        });
        if let Some(lit) = lit {
            fill_rows(light_box.lit_rows_mut(), reach, |index, x, y| {
                Some(lit[index].as_ref()?[x + y * CHUNK])
            });
        }
        let fill_rows = match fill {
            Some((seeds, finding, counters)) => {
                let started = Instant::now();
                let rows = seeds.rows(light_box.open_rows_mut(), reach);
                let took = finding + started.elapsed();
                counters.fill_seeded.fetch_add(1, Ordering::Relaxed);
                counters
                    .fill_seed_nanos
                    .fetch_add(took.as_nanos() as u64, Ordering::Relaxed);
                rows
            }
            None => Vec::new(),
        };
        let mut planes = empty_planes(scale.precision());
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
            let level = scale.precision().of_whole(voxel.level);
            let Some(start) = scale.start(level) else {
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

/// Bit planes of level 0 for every voxel of a chunk, one per bit of a level
/// at `precision`.
pub(crate) fn empty_planes(precision: LightPrecision) -> Box<[Plane]> {
    vec![[0; CHUNK * CHUNK]; precision.bits()].into_boxed_slice()
}

/// The level every open voxel of the chunk holds, if they all hold the same.
fn one_level(planes: &[Plane], occupancy: &Occupancy) -> Option<u8> {
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
                planes
                    .iter()
                    .enumerate()
                    .map(|(bit, plane)| ((plane[column] >> z & 1) as u8) << bit)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::light::fill::CellFill;
    use crate::light::inputs::EmissiveVoxel;
    use crate::light::view::{in_world, Missing};

    /// A chunk whose voxels are all solid except `open`.
    fn solid_except(open: impl IntoIterator<Item = (usize, usize, usize)>) -> Occupancy {
        let mut occupancy = Occupancy::Solid;
        for (x, y, z) in open {
            occupancy.set(x, y, z, false);
        }
        occupancy.normalize();
        occupancy
    }

    fn inputs(
        occupancy: Occupancy,
        emissive: Vec<EmissiveVoxel>,
        columns: ColumnAnswer,
    ) -> Arc<LightInputs> {
        Arc::new(LightInputs {
            occupancy,
            emissive,
            columns,
            ring: true,
        })
    }

    fn around(chunks: &[(IVec3, Arc<LightInputs>)]) -> Neighbourhood {
        let mut neighbours: Neighbourhood = Default::default();
        for (offset, inputs) in chunks {
            neighbours[neighbour_index(*offset)] = Some(inputs.clone());
        }
        neighbours
    }

    /// At 16 levels and reach 14, light loses one level per step.
    fn scale() -> LevelScale {
        LevelScale::new(14, LightPrecision::Steps16)
    }

    fn sky_columns() -> ColumnAnswer {
        ColumnAnswer {
            sky: [!0; CHUNK],
            provisional: [0; CHUNK],
        }
    }

    #[test]
    fn lamp_light_passes_through_a_missing_neighbour_read_as_air() {
        // The chunk is solid but for one open voxel on its +X face, and a lamp
        // ten voxels along that face. The only way between them runs outside
        // the chunk, through the +X neighbour, which isn't loaded.
        let lamp = EmissiveVoxel {
            x: 31,
            y: 10,
            z: 20,
            level: 15,
        };
        let centre = inputs(
            solid_except([(31, 10, 10)]),
            vec![lamp],
            ColumnAnswer::none(),
        );
        let alone = around(&[(IVec3::ZERO, centre.clone())]);
        let shape = LightShape::Diamond;

        let (air, _) = solve_voxel(&alone, Missing::Air, &scale(), shape);
        // One step out of the lamp, ten along z, one step back in.
        assert_eq!(air.at(31, 10, 10), 15 - 12);
        let open_beside = around(&[
            (IVec3::ZERO, centre.clone()),
            (
                IVec3::X,
                inputs(Occupancy::Open, Vec::new(), ColumnAnswer::none()),
            ),
        ]);
        let (open, _) = solve_voxel(&open_beside, Missing::Solid, &scale(), shape);
        assert_eq!(air, open);

        let (solid, _) = solve_voxel(&alone, Missing::Solid, &scale(), shape);
        assert_eq!(solid, ChunkLevels::Uniform(0));
    }

    #[test]
    fn direct_cells_a_column_answer_marks_light_as_the_column_answer_does() {
        // A floor below y = 4 and a roof at y = 20 over x < 16, under sky in
        // every column. Sky light spreads in under the roof from the side.
        let mut occupancy = Occupancy::Open;
        for x in 0..CHUNK {
            for z in 0..CHUNK {
                for y in 0..4 {
                    occupancy.set(x, y, z, true);
                }
                if x < 16 {
                    occupancy.set(x, 20, z, true);
                }
            }
        }
        let mut centre_cells: Box<Plane> = Box::new([0; CHUNK * CHUNK]);
        for y in 0..CHUNK {
            for x in 0..CHUNK {
                if (x < 16 && y > 20) || (x >= 16 && y >= 4) {
                    centre_cells[x + y * CHUNK] = !0;
                }
            }
        }
        let mut cells: DirectCells = Default::default();
        cells[neighbour_index(IVec3::ZERO)] = Some(centre_cells);
        cells[neighbour_index(IVec3::Y)] = Some(Box::new([!0; CHUNK * CHUNK]));

        let answered = around(&[
            (
                IVec3::ZERO,
                inputs(occupancy.clone(), Vec::new(), sky_columns()),
            ),
            (IVec3::Y, inputs(Occupancy::Open, Vec::new(), sky_columns())),
        ]);
        let unanswered = around(&[
            (
                IVec3::ZERO,
                inputs(occupancy, Vec::new(), ColumnAnswer::none()),
            ),
            (
                IVec3::Y,
                inputs(Occupancy::Open, Vec::new(), ColumnAnswer::none()),
            ),
        ]);
        let counters = LightCounters::default();
        for shape in [LightShape::Round, LightShape::Diamond] {
            let from_columns = solve_sky(
                &answered,
                Missing::Solid,
                DirectSky::Columns,
                &scale(),
                shape,
                None,
                &counters,
            );
            let from_cells = solve_sky(
                &unanswered,
                Missing::Solid,
                DirectSky::Cells(Box::new(cells.clone())),
                &scale(),
                shape,
                None,
                &counters,
            );
            assert_eq!(from_cells, from_columns);
            // Under the roof, two steps in from its edge.
            assert_eq!(from_cells.0.at(14, 10, 5), 15 - 2);
        }
    }

    #[test]
    fn sky_spreads_from_a_direct_cell_open_toward_a_side() {
        // A tunnel along x at y = 10, z = 10, roofed all along, opening into
        // open air across +X. Its mouth is the one direct cell.
        let tunnel = solid_except((0..CHUNK).map(|x| (x, 10, 10)));
        let neighbours = around(&[
            (
                IVec3::ZERO,
                inputs(tunnel, Vec::new(), ColumnAnswer::none()),
            ),
            (
                IVec3::X,
                inputs(Occupancy::Open, Vec::new(), ColumnAnswer::none()),
            ),
        ]);
        let mut mouth: Box<Plane> = Box::new([0; CHUNK * CHUNK]);
        mouth[31 + 10 * CHUNK] = 1 << 10;
        let mut cells: DirectCells = Default::default();
        cells[neighbour_index(IVec3::ZERO)] = Some(mouth);
        let counters = LightCounters::default();

        let (sky, _) = solve_sky(
            &neighbours,
            Missing::Solid,
            DirectSky::Cells(Box::new(cells)),
            &scale(),
            LightShape::Diamond,
            None,
            &counters,
        );

        for steps in 0..CHUNK {
            let expected = if steps > 14 { 0 } else { 15 - steps as u8 };
            assert_eq!(sky.at(31 - steps, 10, 10), expected, "{steps} steps in");
        }
        let (columns, _) = solve_sky(
            &neighbours,
            Missing::Solid,
            DirectSky::Columns,
            &scale(),
            LightShape::Diamond,
            None,
            &counters,
        );
        assert_eq!(columns, ChunkLevels::Uniform(0));
    }

    fn settings_with_fill(fill: FillCells) -> LightTaskSettings<u8> {
        LightTaskSettings {
            precision: LightPrecision::Steps16,
            voxel_scale: scale(),
            sky_scale: Some(scale()),
            shape: LightShape::Round,
            routed: false,
            emission: Vec::new(),
            counters: Arc::new(LightCounters::default()),
            generation: 0,
            sky_generation: 0,
            emission_generation: 0,
            fill: Some(fill),
        }
    }

    #[test]
    fn a_solve_that_takes_no_fill_seeds_none_and_a_world_solve_seeds_its_cells() {
        // Each position around the chunk has one sunlit cell of its own.
        let position = IVec3::new(3, -1, 2);
        let fill = FillCells::default();
        let sunlit = |index: usize| 1u64 << index;
        for index in 0..27 {
            let offset = IVec3::new(index % 3, index / 3 % 3, index / 9) - IVec3::ONE;
            fill.write().insert(
                position + offset,
                Arc::new(CellFill {
                    open: !0,
                    direct: sunlit(index as usize),
                    sunlit: sunlit(index as usize),
                    levels: [128; 64],
                }),
            );
        }
        let settings = settings_with_fill(fill);
        let mixed = inputs(solid_except([(0, 0, 0)]), Vec::new(), ColumnAnswer::none());

        let world = seeding(&settings, &mixed, in_world(position), true)
            .expect("a world solve takes its cells' fill");
        let expected: [u64; 27] = std::array::from_fn(sunlit);
        assert_eq!(world.sunlit, expected);
        assert!(world.seeds.is_some());

        assert!(seeding(&settings, &mixed, in_world(position), false).is_none());
    }
}
