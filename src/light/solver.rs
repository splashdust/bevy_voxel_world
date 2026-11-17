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
//! An emissive voxel joins a solve only if its light could reach the chunk in
//! the rounds it has left, which the solve tells from how far outside the
//! chunk it is along each axis. With [`LightShape::Diamond`] every round is
//! one face step, so the sum of those distances bounds it. With
//! [`LightShape::Round`] every even round moves light a voxel along every
//! axis at once, so neither that sum nor a path distance bounds how far light
//! gets. The solve counts the rounds instead: each even round left takes one
//! off every distance, and the face steps of the other rounds must cover what
//! is left. Where no emissive voxel passes, voxel light isn't spread at all.
//! The check costs a few additions per emissive voxel in the box.
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

use super::color::{
    mix_weighted, mixing_weight, pack, unpack, ChunkColor, EmissionTable, Emitter, WHITE,
};
use super::config::LightPrecision;
use super::config::LightShape;
use super::faces::{could_reach_sooner, mix as mix_hash, Face, FaceHashes};
use super::fill::seed::{CellsAround, FillSeeds, Seeding};
use super::fill::{FillCells, CELL};
use super::inputs::{EmissiveVoxel, LightInputs, Occupancy, Rescan};
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
    /// The light colour of its voxel light.
    pub color: ChunkColor,
    /// The precision its levels are steps of: the world's when it was
    /// solved, which a change of precision leaves behind until it relights.
    pub precision: LightPrecision,
}

impl ChunkLight {
    /// Whether `other` holds the same levels and light colours as this light
    /// in every voxel.
    pub fn same_light(&self, other: &ChunkLight) -> bool {
        self.precision == other.precision
            && self.sky == other.sky
            && self.voxel == other.voxel
            && self.color == other.color
    }

    /// The light at the chunk's voxel `local`, which holds `voxel`. A solid
    /// voxel holds no light, except an emissive one, which holds its level
    /// and light colour.
    pub fn levels_at<I: Copy + PartialEq>(
        &self,
        local: UVec3,
        voxel: WorldVoxel<I>,
        emission: &EmissionTable<I>,
    ) -> LightLevels {
        match voxel {
            WorldVoxel::Solid(material) => emitted(emission, emission.get(material)),
            _ => {
                self.open_levels_at(local.x as usize, local.y as usize, local.z as usize)
            }
        }
    }

    /// The light at the chunk's open voxel `x`, `y`, `z`.
    pub fn open_levels_at(&self, x: usize, y: usize, z: usize) -> LightLevels {
        LightLevels::of_steps(
            self.sky.at(x, y, z),
            self.voxel.at(x, y, z),
            self.precision,
        )
        .with_voxel_color(self.color.at(x, y, z))
    }
}

/// The light at an emissive voxel giving off what `emitter` says, under the
/// emissive list `emission`, or at a solid voxel that isn't emissive.
pub(crate) fn emitted<I: Copy + PartialEq>(
    emission: &EmissionTable<I>,
    emitter: Option<Emitter>,
) -> LightLevels {
    let Some(emitter) = emitter else {
        return LightLevels::new(0, 0);
    };
    LightLevels::new(0, emitter.level)
        .with_voxel_color(unpack(emission.color(emitter.material)))
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
    /// Each emissive material, its level and its light colour.
    pub emission: EmissionTable<I>,
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
    /// Runs of voxel light, each of one light colour, on light boxes holding
    /// two or more light colours. A light box of one colour runs as in a white
    /// world, uncounted.
    pub color_runs: AtomicU64,
    /// Solves of voxel light that spread none, because no emissive voxel
    /// around the chunk could light it, though some chunk around it held
    /// emissive voxels.
    pub voxel_boxes_culled: AtomicU64,
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
        let mut solved =
            solve(key, snapshot, Arc::new(inputs), settings, seeding, direct);
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
    let mut solved = solve(key, snapshot, inputs, settings, seeding, direct);
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
/// light. `key` is the chunk's.
fn solve<I>(
    key: GridKey,
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
    // A sub-grid's own voxel light stays white.
    let (voxel, voxel_faces, color) = if settings.emission.is_colored() && key.is_world()
    {
        solve_voxel_colors(
            &snapshot.neighbours,
            snapshot.missing,
            &settings.voxel_scale,
            settings.shape,
            &settings.emission,
            &settings.counters,
        )
    } else {
        let (voxel, faces) = solve_voxel(
            &snapshot.neighbours,
            snapshot.missing,
            &settings.voxel_scale,
            settings.shape,
            &settings.counters,
        );
        (voxel, faces, ChunkColor::White)
    };
    let counters = &settings.counters;
    counters.solves.fetch_add(1, Ordering::Relaxed);
    counters
        .solve_nanos
        .fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);

    let light = Arc::new(ChunkLight {
        inputs,
        sky,
        voxel,
        color,
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
    counters: &LightCounters,
) -> (ChunkLevels, [u64; 6]) {
    let centre = centre(neighbours);
    let unlit = (ChunkLevels::Uniform(0), FaceHashes::uniform(0));
    if !centre.occupancy.has_open_voxel() {
        return unlit;
    }
    let seeds = seeds(neighbours, scale, shape, |_| true);
    if seeds.is_empty() {
        let held = neighbours
            .iter()
            .flatten()
            .any(|inputs| !inputs.emissive.is_empty());
        if held {
            counters.voxel_boxes_culled.fetch_add(1, Ordering::Relaxed);
        }
        return unlit;
    }
    let (planes, faces) = spread(neighbours, missing, scale, shape, &seeds, None, None);
    (chunk_levels(planes, &centre.occupancy), faces)
}

/// Solves voxel light for the chunk at the centre of `neighbours`, as
/// [`solve_voxel`] does, in a world whose emissive list `emission` holds a
/// light colour other than white. Returns its levels, face hashes and light
/// colour.
///
/// The light colours come from the notes of the emissive materials each
/// chunk around it holds. With one, light is solved in one run, as in a
/// white world. With more, it runs once per light colour, from that colour's
/// emissive voxels alone, and runs that light no voxel of the chunk are
/// dropped. A voxel's level is then the highest of the runs', which is the
/// level one run of all of them gives, and its light colour the runs' colours
/// at their levels, mixed by [`mix_weighted`].
pub(crate) fn solve_voxel_colors<I>(
    neighbours: &Neighbourhood,
    missing: Missing,
    scale: &LevelScale,
    shape: LightShape,
    emission: &EmissionTable<I>,
    counters: &LightCounters,
) -> (ChunkLevels, [u64; 6], ChunkColor) {
    let colors = light_colors(neighbours, emission);
    if colors.len() <= 1 {
        let (voxel, faces) = solve_voxel(neighbours, missing, scale, shape, counters);
        let color = match colors.first() {
            Some(color) if voxel != ChunkLevels::Uniform(0) => ChunkColor::of(*color),
            _ => ChunkColor::White,
        };
        return (voxel, faces, color);
    }
    let unlit = (
        ChunkLevels::Uniform(0),
        FaceHashes::uniform(0),
        ChunkColor::White,
    );
    let centre = centre(neighbours);
    if !centre.occupancy.has_open_voxel() {
        return unlit;
    }
    let mut runs = Vec::new();
    let mut spread_any = false;
    for color in colors {
        let seeds = seeds(neighbours, scale, shape, |voxel| {
            emission.color(voxel.material()) == color
        });
        if seeds.is_empty() {
            continue;
        }
        spread_any = true;
        let (planes, faces) =
            spread(neighbours, missing, scale, shape, &seeds, None, None);
        counters.color_runs.fetch_add(1, Ordering::Relaxed);
        if lights_any(&planes, &centre.occupancy) {
            runs.push(Run {
                color,
                planes,
                faces,
            });
        }
    }
    if !spread_any {
        counters.voxel_boxes_culled.fetch_add(1, Ordering::Relaxed);
    }
    match runs.len() {
        0 => unlit,
        1 => {
            let run = runs.pop().expect("one run");
            (
                chunk_levels(run.planes, &centre.occupancy),
                run.faces,
                ChunkColor::of(run.color),
            )
        }
        _ => mixed(&runs, &centre.occupancy, scale.precision()),
    }
}

/// One light colour's run over a light box.
struct Run {
    /// The packed light colour of the emissive voxels it spread from.
    color: u16,
    planes: Box<[Plane]>,
    faces: [u64; 6],
}

/// The packed light colours of the emissive materials the chunks of
/// `neighbours` note, each once, in order.
fn light_colors<I>(neighbours: &Neighbourhood, emission: &EmissionTable<I>) -> Vec<u16> {
    let mut colors = Vec::new();
    let materials = neighbours
        .iter()
        .flatten()
        .flat_map(|inputs| inputs.emissive.materials());
    for material in materials {
        let color = emission.color(*material);
        if !colors.contains(&color) {
            colors.push(color);
        }
    }
    colors.sort_unstable();
    colors
}

/// Whether `planes` light any open voxel of the chunk.
fn lights_any(planes: &[Plane], occupancy: &Occupancy) -> bool {
    planes.iter().any(|plane| {
        plane.iter().enumerate().any(|(column, bits)| {
            bits & !occupancy.solid_column(column % CHUNK, column / CHUNK) != 0
        })
    })
}

/// The levels, face hashes and light colour of a chunk two or more of whose
/// `runs` light its voxels, with levels in steps of `precision`.
///
/// A voxel only one run lights takes that run's light colour as it is. The
/// face hashes mix each run's with its light colour, so they change wherever
/// a run's light crosses a face at another time.
fn mixed(
    runs: &[Run],
    occupancy: &Occupancy,
    precision: LightPrecision,
) -> (ChunkLevels, [u64; 6], ChunkColor) {
    let weights: Vec<f32> = (0..=precision.top())
        .map(|steps| mixing_weight(precision.level_of(steps)))
        .collect();
    let levels: Vec<Vec<u8>> = runs.iter().map(|run| voxel_levels(&run.planes)).collect();
    let colors: Vec<LinearRgba> = runs.iter().map(|run| unpack(run.color)).collect();
    let mut packed = vec![WHITE; CHUNK * CHUNK * CHUNK];
    let mut planes = empty_planes(precision);
    let (mut one, mut uniform) = (None, true);
    for column in 0..CHUNK * CHUNK {
        let mut open = !occupancy.solid_column(column % CHUNK, column / CHUNK);
        while open != 0 {
            let z = open.trailing_zeros() as usize;
            open &= open - 1;
            let index = column + z * CHUNK * CHUNK;
            let (mut top, mut lit, mut only) = (0, 0, 0);
            for (run, levels) in levels.iter().enumerate() {
                if levels[index] > 0 {
                    top = top.max(levels[index]);
                    lit += 1;
                    only = run;
                }
            }
            if lit == 0 {
                continue;
            }
            packed[index] = if lit == 1 {
                runs[only].color
            } else {
                pack(mix_weighted(
                    levels
                        .iter()
                        .zip(&colors)
                        .filter(|(levels, _)| levels[index] > 0)
                        .map(|(levels, color)| (*color, weights[levels[index] as usize])),
                ))
            };
            // Whether every lit voxel has the one light colour.
            uniform &= *one.get_or_insert(packed[index]) == packed[index];
            for (bit, plane) in planes.iter_mut().enumerate() {
                plane[column] |= ((top >> bit & 1) as u32) << z;
            }
        }
    }
    let mut faces = [0; 6];
    for run in runs {
        for (face, hash) in faces.iter_mut().zip(run.faces) {
            if hash != 0 {
                *face = mix_hash(mix_hash(*face, run.color as u64), hash);
            }
        }
    }
    let color = match one {
        Some(one) if uniform => ChunkColor::of(one),
        _ => ChunkColor::Mixed(packed.into()),
    };
    (chunk_levels(planes, occupancy), faces, color)
}

/// The level in `planes` of each voxel of the chunk, at
/// [`voxel_index`](super::color::voxel_index).
fn voxel_levels(planes: &[Plane]) -> Vec<u8> {
    let mut levels = vec![0; CHUNK * CHUNK * CHUNK];
    for (bit, plane) in planes.iter().enumerate() {
        for (column, bits) in plane.iter().enumerate() {
            let mut bits = *bits;
            while bits != 0 {
                let z = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                levels[column + z * CHUNK * CHUNK] |= 1 << bit;
            }
        }
    }
    levels
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

/// The emissive voxels whose light could reach the chunk, spreading in
/// `shape`, as seeds in its box, sorted by where their light starts. Only the
/// emissive voxels `keep` holds for count.
fn seeds(
    neighbours: &Neighbourhood,
    scale: &LevelScale,
    shape: LightShape,
    keep: impl Fn(&EmissiveVoxel) -> bool,
) -> Vec<Seed> {
    let reach = scale.reach() as i32;
    let last = scale.last_distance(1) as i32;
    let side = CHUNK_SIZE_I + 2 * reach;
    let mut seeds = Vec::new();
    for (index, inputs) in neighbours.iter().enumerate() {
        let Some(inputs) = inputs else {
            continue;
        };
        let index = index as i32;
        let offset = IVec3::new(index % 3, index / 3 % 3, index / 9) - IVec3::ONE;
        for voxel in &inputs.emissive {
            if !keep(voxel) {
                continue;
            }
            let level = scale.precision().of_whole(voxel.level());
            let Some(start) = scale.start(level) else {
                continue;
            };
            let position = offset * CHUNK_SIZE_I
                + IVec3::new(voxel.x() as i32, voxel.y() as i32, voxel.z() as i32)
                + IVec3::splat(reach);
            if position.cmplt(IVec3::ZERO).any()
                || position.cmpge(IVec3::splat(side)).any()
            {
                continue;
            }
            let outside = (IVec3::splat(reach) - position)
                .max(position - IVec3::splat(reach + CHUNK_SIZE_I - 1))
                .max(IVec3::ZERO);
            if !could_reach(outside, start as i32, last, shape) {
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

/// Whether light from a source `outside` voxels outside the chunk along each
/// axis, starting at step `start`, could reach a voxel of the chunk by the
/// last lit round `last`, spreading in `shape`.
///
/// Light joins in the round after `start`. Of the rounds left, a round
/// spreading by three passes moves light a voxel along every axis, and any
/// other round a face step along one. So light gets there only if the face
/// steps left cover what the three-pass rounds leave of each distance.
fn could_reach(outside: IVec3, start: i32, last: i32, shape: LightShape) -> bool {
    let rounds = last - start;
    let three_pass = match shape {
        LightShape::Round => last / 2 - start / 2,
        LightShape::Diamond => 0,
    };
    let left = (outside - IVec3::splat(three_pass)).max(IVec3::ZERO);
    rounds > 0 && left.element_sum() <= rounds - three_pass
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
            emissive: emissive.into(),
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
        let lamp = EmissiveVoxel::new(
            31,
            10,
            20,
            Emitter {
                level: 15,
                material: 0,
            },
        );
        let centre = inputs(
            solid_except([(31, 10, 10)]),
            vec![lamp],
            ColumnAnswer::none(),
        );
        let alone = around(&[(IVec3::ZERO, centre.clone())]);
        let shape = LightShape::Diamond;

        let (air, _) = solve_voxel(
            &alone,
            Missing::Air,
            &scale(),
            shape,
            &LightCounters::default(),
        );
        // One step out of the lamp, ten along z, one step back in.
        assert_eq!(air.at(31, 10, 10), 15 - 12);
        let open_beside = around(&[
            (IVec3::ZERO, centre.clone()),
            (
                IVec3::X,
                inputs(Occupancy::Open, Vec::new(), ColumnAnswer::none()),
            ),
        ]);
        let (open, _) = solve_voxel(
            &open_beside,
            Missing::Solid,
            &scale(),
            shape,
            &LightCounters::default(),
        );
        assert_eq!(air, open);

        let (solid, _) = solve_voxel(
            &alone,
            Missing::Solid,
            &scale(),
            shape,
            &LightCounters::default(),
        );
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
            emission: EmissionTable::default(),
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

    /// A neighbourhood holding only one lamp at `level`, at `position` in the
    /// box of light of `reach` around the chunk, in the chunk position that
    /// holds it.
    fn lamp_at(position: IVec3, reach: i32, level: u8) -> Neighbourhood {
        let voxel = position - IVec3::splat(reach);
        let chunk = voxel.div_euclid(IVec3::splat(CHUNK_SIZE_I));
        let local = voxel.rem_euclid(IVec3::splat(CHUNK_SIZE_I)).as_uvec3();
        let lamp = EmissiveVoxel::new(
            local.x as u8,
            local.y as u8,
            local.z as u8,
            Emitter { level, material: 0 },
        );
        let held = inputs(Occupancy::Open, vec![lamp], ColumnAnswer::none());
        around(&[(chunk, held)])
    }

    /// Whether light from lamps at `positions` in the box, each starting at
    /// `start`, reaches a voxel of the chunk, as the kernel spreads it
    /// through a box of air.
    fn kernel_reaches(
        positions: &[IVec3],
        start: u8,
        scale: &LevelScale,
        shape: LightShape,
    ) -> bool {
        let seeds: Vec<Seed> = positions
            .iter()
            .map(|position| Seed {
                x: position.x as u8,
                y: position.y as u8,
                z: position.z as u8,
                start,
            })
            .collect();
        let mut air = around(&[]);
        air[neighbour_index(IVec3::ZERO)] =
            Some(inputs(Occupancy::Open, Vec::new(), ColumnAnswer::none()));
        let (planes, _) = spread(&air, Missing::Air, scale, shape, &seeds, None, None);
        planes.iter().flatten().any(|column| *column != 0)
    }

    #[test]
    fn the_cull_keeps_every_lamp_whose_light_reaches_the_chunk() {
        for reach in [14, 17] {
            for precision in [LightPrecision::Steps16, LightPrecision::Steps256] {
                let scale = LevelScale::new(reach, precision);
                for shape in [LightShape::Round, LightShape::Diamond] {
                    let side = CHUNK_SIZE_I + 2 * reach as i32;
                    // The box and the kernel are the same mirrored along
                    // each axis, so one corner of the box stands for all.
                    let half = side / 2;
                    for level in 1..=15 {
                        let Some(start) = scale.start(precision.of_whole(level)) else {
                            continue;
                        };
                        let mut culled = Vec::new();
                        for z in 0..half {
                            for y in 0..half {
                                for x in 0..half {
                                    let position = IVec3::new(x, y, z);
                                    let neighbours =
                                        lamp_at(position, reach as i32, level);
                                    if seeds(&neighbours, &scale, shape, |_| true)
                                        .is_empty()
                                    {
                                        culled.push(position);
                                    }
                                }
                            }
                        }
                        assert!(!culled.is_empty());
                        // Light only adds up, so if all of them together
                        // light nothing, none of them does alone.
                        if !kernel_reaches(&culled, start, &scale, shape) {
                            continue;
                        }
                        let reaching = culled
                            .iter()
                            .find(|position| {
                                kernel_reaches(&[**position], start, &scale, shape)
                            })
                            .expect("one lamp reaches the chunk alone");
                        panic!(
                            "the cull drops a lamp at level {level} at {reaching} in \
                             the box, whose light reaches the chunk at reach {reach}, \
                             {precision:?}, {shape:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn the_cull_drops_a_lamp_past_a_corner_whose_light_cant_reach_the_chunk() {
        // Five voxels outside the chunk along every axis: 15 face steps away,
        // and the light of a lamp at level 15 takes 14.
        let position = IVec3::splat(14 - 5);
        let neighbours = lamp_at(position, 14, 15);

        assert!(seeds(&neighbours, &scale(), LightShape::Diamond, |_| true).is_empty());
        assert!(!kernel_reaches(
            &[position],
            0,
            &scale(),
            LightShape::Diamond
        ));
        // Round light moves along every axis at once in its even rounds, so
        // it gets there.
        assert_eq!(
            seeds(&neighbours, &scale(), LightShape::Round, |_| true).len(),
            1
        );
        assert!(kernel_reaches(&[position], 0, &scale(), LightShape::Round));
    }
}
