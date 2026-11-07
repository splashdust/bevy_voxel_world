//! Sky probes: settling the columns a chunk's own task left provisional.
//!
//! A chunk whose task couldn't settle every column it lets sky light into
//! waits here, with the positions above it whose voxels its answer takes in.
//! A sky probe reads one of the other positions through the world's lookups
//! and keeps only its mask of columns holding a solid: it creates no entity,
//! mesh or voxel data, and doesn't touch the streaming policy. The mask goes
//! into the [sky record](super::record) and into every waiting chunk below.
//! A chunk whose direct sky changes relights, and its relight ripples on to
//! the chunks its light reaches.
//!
//! Probes start after chunk tasks, with the room left in the world's
//! [`PoolBudget`]: one keeps running while any chunk waits, more start while
//! the task pool has idle threads, and none once nothing waits. A probe of a
//! position that is mostly air reads most of its voxels, much more than a
//! chunk task's walk, so more than one at a time while chunks stream in would
//! slow the stream. Chunks with the
//! fewest provisional columns go first, since they are the most likely to show
//! wrong light and the cheapest to settle, and the nearest among equals.
//!
//! A probe notes its position's edit count when it starts. If the count moved
//! by the time it finishes, its mask is thrown away and the position is probed
//! again.
//!
//! A waiting chunk that retires ripples relights from its face neighbours,
//! which may have read its provisional direct sky. A chunk retiring with every
//! column settled relights nothing.

use std::marker::PhantomData;
use std::sync::atomic::Ordering;
use std::sync::Arc;

use bevy::platform::collections::{HashMap, HashSet};
use bevy::prelude::*;
use bevy::tasks::{AsyncComputeTaskPool, Task};
use futures_lite::future;

use super::faces::Face;
use super::fill::driver::{field_of, SkyFillWork};
use super::fill::open_cells;
use super::record::SkyRecord;
use super::relight::LightWork;
use super::sky::{
    padding_row, proved, provisional, ColumnAnswer, Columns, Hinted, SkyRule,
};
use super::solver::{ChunkLight, Solved};
use super::view::{LightView, ENTRY};
use super::LightSettings;
use crate::chunk::{ChunkKey, CHUNK_SIZE_F, CHUNK_SIZE_I, CHUNK_SIZE_U};
use crate::chunk_map::ChunkMap;
use crate::configuration::VoxelWorldConfig;
use crate::pool_budget::{first_in_start_order, JobKind, PoolBudget};
use crate::voxel_edits::VoxelEdits;
use crate::voxel_world::VoxelWorldCamera;

const CHUNK: usize = CHUNK_SIZE_U as usize;

/// A chunk with columns it lets sky light into that nothing has proved clear.
struct Waiting {
    entity: Entity,
    /// The columns it lets sky light into.
    sky: Columns,
    /// By position above the chunk within its sky rule's window that a probe
    /// can reach, lowest first, whether its voxels are taken into `sky`.
    known: Vec<bool>,
    /// Whether `known` covers the whole window, so knowing every position
    /// proves the columns left in `sky` clear.
    complete: bool,
}

impl Waiting {
    fn is_proved(&self) -> bool {
        self.complete && self.known.iter().all(|known| *known)
    }

    /// The lowest position above the chunk at `position` whose voxels it
    /// doesn't know yet, and that `skip` doesn't hold.
    fn next(&self, position: IVec3, skip: &HashSet<IVec3>) -> Option<IVec3> {
        self.known
            .iter()
            .enumerate()
            .filter(|(_, known)| !**known)
            .map(|(index, _)| position + IVec3::Y * (index as i32 + 1))
            .find(|above| !skip.contains(above))
    }

    fn answer(&self) -> ColumnAnswer {
        if self.is_proved() {
            proved(self.sky)
        } else {
            provisional(self.sky)
        }
    }
}

struct Probe {
    position: IVec3,
    /// The position's edit count when the probe started.
    edits: u64,
    /// The sky record's generation when the probe started.
    generation: u64,
    task: Task<ProbeRead>,
}

/// What a probe read of a position.
pub(crate) struct ProbeRead {
    /// The columns holding a solid.
    pub solid: Columns,
    /// The cells holding no solid, when asked for.
    pub open: Option<u64>,
    pub lookups: u64,
}

/// Reads the voxels of `position` through `lookup`, voxel edits first, for
/// which of its columns hold a solid and, with `cells`, which of its cells
/// hold none.
///
/// Each column is read upward to its first solid. With `cells`, a column goes
/// on past it, skipping only voxels whose cell already holds a solid.
pub(crate) fn probe_position<I: Copy + PartialEq>(
    position: IVec3,
    mut lookup: impl FnMut(IVec3) -> crate::voxel::WorldVoxel<I>,
    edits: Option<&HashMap<IVec3, crate::voxel::WorldVoxel<I>>>,
    cells: bool,
) -> ProbeRead {
    let min = position * CHUNK_SIZE_I;
    let mut solid = [0; CHUNK];
    let mut closed = 0u64;
    let mut lookups = 0;
    for z in 0..CHUNK {
        for (x, solid) in solid.iter_mut().enumerate() {
            for y in 0..CHUNK {
                let cell = crate::light::fill::cell_bit(x / 8, y / 8, z / 8);
                let column_blocked = *solid >> z & 1 == 1;
                if column_blocked && (!cells || closed & cell != 0) {
                    if !cells {
                        break;
                    }
                    continue;
                }
                let at = min + IVec3::new(x as i32, y as i32, z as i32);
                lookups += 1;
                let voxel = edits
                    .and_then(|edits| edits.get(&at).copied())
                    .unwrap_or_else(|| lookup(at));
                if voxel.is_solid() {
                    *solid |= 1 << z;
                    closed |= cell;
                }
            }
        }
    }
    ProbeRead {
        solid,
        open: cells.then_some(!closed),
        lookups,
    }
}

/// The sky probes of world `C`: the chunks waiting for their columns to
/// settle, and the probes running for them. Only a world with sky light on has
/// one.
#[derive(Resource)]
pub(crate) struct SkyProbes<C> {
    rule: SkyRule,
    waiting: HashMap<IVec3, Waiting>,
    /// The heights of the waiting chunks, by their x and z.
    columns: HashMap<IVec2, Vec<i32>>,
    running: Vec<Probe>,
    /// The positions running probes read.
    probing: HashSet<IVec3>,
    /// Chunks whose direct sky changed, to relight.
    relight: Vec<IVec3>,
    /// Chunks that started or stopped waiting, so whether their columns are
    /// proved changed, for sky fill.
    touched: Vec<IVec3>,
    /// Whether probes also read which cells of a position hold no solid, for
    /// sky fill.
    pub reads_cells: bool,
    _world: PhantomData<C>,
}

impl<C: VoxelWorldConfig> SkyProbes<C> {
    pub fn new(rule: SkyRule) -> Self {
        Self {
            rule,
            waiting: HashMap::new(),
            columns: HashMap::new(),
            running: Vec::new(),
            probing: HashSet::new(),
            relight: Vec::new(),
            touched: Vec::new(),
            reads_cells: false,
            _world: PhantomData,
        }
    }

    /// Whether no probe is running, and no chunk waits for one.
    pub fn is_settled(&self) -> bool {
        let none = HashSet::new();
        self.running.is_empty()
            && self
                .waiting
                .iter()
                .all(|(position, chunk)| chunk.next(*position, &none).is_none())
    }

    /// Checks the column answer of a chunk task's `solved` light for the chunk
    /// `entity` at `position` against the sky record before it lands, and
    /// records the chunk's own mask.
    ///
    /// The record may have learned positions above the chunk since its task
    /// started. A position edited meanwhile makes what the task found there
    /// stale, so the answer is worked out again from its padding row, the fill
    /// hints and the record. If the direct sky changed, `solved` gets the new
    /// answer and the chunk is queued to relight.
    #[allow(clippy::too_many_arguments)]
    pub fn chunk_landed(
        &mut self,
        position: IVec3,
        entity: Entity,
        solved: &mut Solved,
        record: &mut SkyRecord,
        edits: &VoxelEdits<C, C::MaterialIndex>,
        world: &C,
        view: &LightView,
    ) {
        let Some(found) = solved.sky.take() else {
            return;
        };
        let inputs = solved.light.inputs.clone();
        let counts = edit_counts(edits, position, found.known.len());
        let fresh = found.generation == record.generation();
        let stale = !fresh
            || found
                .known
                .iter()
                .zip(found.edits.iter().zip(&counts))
                .any(|(known, (noted, now))| *known && noted != now);
        let (sky, known) = if stale {
            from_hints(world, position, found.open, &counts)
        } else {
            (inputs.columns.sky, found.known)
        };
        let answer = self.settle(position, entity, sky, known, &counts, record);
        if answer != inputs.columns {
            if answer.sky != inputs.columns.sky {
                self.relight.push(position);
            }
            let light = &solved.light;
            solved.light = Arc::new(ChunkLight {
                inputs: Arc::new(inputs.with_columns(answer)),
                sky: light.sky.clone(),
                voxel: light.voxel.clone(),
                precision: light.precision,
            });
        }

        if fresh && edits.edit_count(position) == found.own_edits {
            let mask = inputs.occupancy.solid_columns();
            let open = record.keeps_cells().then(|| open_cells(&inputs.occupancy));
            record.insert(position, &mask, open, found.own_edits);
            self.learned(position, &mask, view);
        }
    }

    /// Takes the column at `x`, `z` out of the columns of the chunk at
    /// `position`, which a voxel edit above it has blocked. Returns the
    /// chunk's new column answer if it was waiting for probes.
    pub fn column_blocked(
        &mut self,
        position: IVec3,
        x: usize,
        z: usize,
    ) -> Option<ColumnAnswer> {
        let chunk = self.waiting.get_mut(&position)?;
        chunk.sky[x] &= !(1 << z);
        let answer = chunk.answer();
        if answer.provisional.iter().all(|column| *column == 0) {
            self.remove(position);
        }
        Some(answer)
    }

    /// Lets sky light into the column at `x`, `z` of the chunk `entity` at
    /// `position`, which a voxel edit above it has opened, and returns the
    /// chunk's new column answer. `current` is its answer so far.
    ///
    /// `known` says, by position above the chunk that a probe can reach,
    /// lowest first, whether its voxels are known, and `complete` whether
    /// those positions make up the sky rule's whole window. With every
    /// position known, the column is proved clear; otherwise the chunk waits
    /// for probes of the unknown ones, showing direct sky until then.
    pub fn column_opened(
        &mut self,
        position: IVec3,
        entity: Entity,
        (x, z): (usize, usize),
        current: &ColumnAnswer,
        known: Vec<bool>,
        complete: bool,
    ) -> ColumnAnswer {
        let bit = 1 << z;
        if let Some(chunk) = self.waiting.get_mut(&position) {
            chunk.sky[x] |= bit;
            for (was, now) in chunk.known.iter_mut().zip(&known) {
                *was &= *now;
            }
            let answer = chunk.answer();
            if answer.provisional.iter().all(|column| *column == 0) {
                self.remove(position);
            }
            return answer;
        }
        let mut sky = current.sky;
        sky[x] |= bit;
        if complete && known.iter().all(|known| *known) {
            let mut answer = current.clone();
            answer.sky = sky;
            return answer;
        }
        let chunk = Waiting {
            entity,
            sky,
            known,
            complete,
        };
        let answer = chunk.answer();
        self.waiting.insert(position, chunk);
        if self.reads_cells {
            self.touched.push(position);
        }
        self.columns
            .entry(position.xz())
            .or_default()
            .push(position.y);
        answer
    }

    /// Forgets the waiting chunk `entity` at `position` as it retires. Returns
    /// whether it was waiting, so its neighbours may have read provisional
    /// direct sky from it.
    pub fn forget(&mut self, position: IVec3, entity: Entity) -> bool {
        if self
            .waiting
            .get(&position)
            .is_none_or(|chunk| chunk.entity != entity)
        {
            return false;
        }
        self.remove(position);
        true
    }

    /// Whether the chunk at `position` waits for its columns to be proved.
    pub fn is_waiting(&self, position: IVec3) -> bool {
        self.waiting.contains_key(&position)
    }

    /// The chunks that started or stopped waiting since this was last called.
    pub fn take_touched(&mut self) -> Vec<IVec3> {
        std::mem::take(&mut self.touched)
    }

    /// The chunks whose direct sky changed since this was last called.
    pub fn take_relights(&mut self) -> Vec<IVec3> {
        std::mem::take(&mut self.relight)
    }

    /// Works out the column answer of the chunk `entity` at `position` from
    /// the columns `sky` and the positions `known` above it, taking in the
    /// masks the record holds for the others, and keeps the chunk waiting if
    /// it still has provisional columns. `counts` are the edit counts of the
    /// positions above it.
    fn settle(
        &mut self,
        position: IVec3,
        entity: Entity,
        mut sky: Columns,
        mut known: Vec<bool>,
        counts: &[u64],
        record: &mut SkyRecord,
    ) -> ColumnAnswer {
        self.remove(position);
        for (index, known) in known.iter_mut().enumerate() {
            if *known {
                continue;
            }
            let above = position + IVec3::Y * (index as i32 + 1);
            if let Some(mask) = record.get(above, counts[index]) {
                for (sky, solid) in sky.iter_mut().zip(&mask) {
                    *sky &= !solid;
                }
                *known = true;
            }
        }
        let chunk = Waiting {
            entity,
            sky,
            complete: known.len() as u32 == self.rule.window(position.y),
            known,
        };
        let answer = chunk.answer();
        if answer.provisional.iter().any(|column| *column != 0) {
            self.waiting.insert(position, chunk);
            if self.reads_cells {
                self.touched.push(position);
            }
            self.columns
                .entry(position.xz())
                .or_default()
                .push(position.y);
        }
        answer
    }

    /// Takes the mask of `position`, the columns holding a solid there, into
    /// every waiting chunk below it. A chunk whose direct sky changes gets its
    /// new answer in the world view and is queued to relight.
    pub fn learned(&mut self, position: IVec3, mask: &Columns, view: &LightView) {
        let Some(heights) = self.columns.get(&position.xz()) else {
            return;
        };
        let below: Vec<IVec3> = heights
            .iter()
            .filter(|y| **y < position.y)
            .map(|y| IVec3::new(position.x, *y, position.z))
            .collect();
        for chunk_position in below {
            let index = (position.y - chunk_position.y - 1) as usize;
            let chunk = self.waiting.get_mut(&chunk_position).unwrap();
            if chunk.known.get(index) != Some(&false) {
                continue;
            }
            chunk.known[index] = true;
            let old = chunk.sky;
            for (sky, solid) in chunk.sky.iter_mut().zip(mask) {
                *sky &= !solid;
            }
            let answer = chunk.answer();
            let entity = chunk.entity;
            if answer.sky != old {
                replace_columns(view, chunk_position, entity, answer.clone());
                self.relight.push(chunk_position);
            }
            if answer.provisional.iter().all(|column| *column == 0) {
                self.remove(chunk_position);
            }
        }
    }

    fn remove(&mut self, position: IVec3) {
        if self.waiting.remove(&position).is_none() {
            return;
        }
        if self.reads_cells {
            self.touched.push(position);
        }
        if let Some(heights) = self.columns.get_mut(&position.xz()) {
            heights.retain(|y| *y != position.y);
            if heights.is_empty() {
                self.columns.remove(&position.xz());
            }
        }
    }

    /// Works out the column answer of the loaded chunk `entity` at `position`
    /// again from `padding`, the columns open in its padding row above it, the
    /// fill hints and the sky record, as if its task had found nothing more.
    /// It waits for probes if columns stay provisional.
    pub fn answer_again(
        &mut self,
        position: IVec3,
        entity: Entity,
        padding: Columns,
        record: &mut SkyRecord,
        edits: &VoxelEdits<C, C::MaterialIndex>,
        world: &C,
    ) -> ColumnAnswer {
        let window = self.rule.window(position.y);
        let open = if window == 0 { [!0; CHUNK] } else { padding };
        let counts = edit_counts(edits, position, super::sky::walkable(window) as usize);
        let (sky, known) = from_hints(world, position, open, &counts);
        self.settle(position, entity, sky, known, &counts, record)
    }

    /// Works out the column answer of hinted air at `position` from the fill
    /// hints and the sky record, and waits for probes if columns stay
    /// provisional, as [`answer_again`](Self::answer_again) does for a chunk.
    /// The padding row above it is the bottom of the position above, which
    /// the answer takes in with the rest of the window.
    pub fn answer_air(
        &mut self,
        position: IVec3,
        record: &mut SkyRecord,
        edits: &VoxelEdits<C, C::MaterialIndex>,
        world: &C,
    ) -> ColumnAnswer {
        self.answer_again(position, ENTRY, [!0; CHUNK], record, edits, world)
    }

    /// Gives the waiting chunk at `position` the entity `entity`: a chunk
    /// retiring in hinted air leaves its columns waiting for the light entry
    /// or hinted air that takes its place.
    pub fn rekey(&mut self, position: IVec3, entity: Entity) {
        if let Some(chunk) = self.waiting.get_mut(&position) {
            chunk.entity = entity;
        }
    }

    /// Follows the sky rule `rule` from now on. Waiting chunks must then have
    /// their answers worked out again: see [`answer_loaded_again`].
    pub fn set_rule(&mut self, rule: SkyRule) {
        self.rule = rule;
    }

    /// Forgets every waiting chunk and drops the running probes.
    fn clear(&mut self) {
        if self.reads_cells {
            self.touched.extend(self.waiting.keys().copied());
        }
        self.waiting.clear();
        self.columns.clear();
        self.running.clear();
        self.probing.clear();
        self.relight.clear();
    }

    /// Collects finished probes: records each mask whose position wasn't
    /// edited while it ran, takes it into the waiting chunks below, and
    /// relights the chunks whose direct sky changed.
    pub fn collect(
        mut probes: ResMut<Self>,
        mut work: ResMut<LightWork<C>>,
        mut budget: ResMut<PoolBudget<C>>,
        settings: Res<LightSettings<C>>,
        edits: Res<VoxelEdits<C, C::MaterialIndex>>,
    ) {
        let mut finished = 0;
        if !probes.running.is_empty() {
            #[cfg(feature = "trace")]
            let _span = bevy::log::info_span!("collect_sky_probes").entered();
            let mut record = settings.record.lock().unwrap();
            let counters = settings.counters();
            let mut index = 0;
            while index < probes.running.len() {
                let probe = &mut probes.running[index];
                let Some(read) = future::block_on(future::poll_once(&mut probe.task))
                else {
                    index += 1;
                    continue;
                };
                finished += 1;
                let probe = probes.running.swap_remove(index);
                probes.probing.remove(&probe.position);
                counters
                    .probe_lookups
                    .fetch_add(read.lookups, Ordering::Relaxed);
                if probe.generation != record.generation()
                    || edits.edit_count(probe.position) != probe.edits
                {
                    counters.probes_discarded.fetch_add(1, Ordering::Relaxed);
                    continue;
                }
                record.insert(probe.position, &read.solid, read.open, probe.edits);
                probes.learned(probe.position, &read.solid, &settings.view);
            }
        }
        for position in probes.take_relights() {
            work.trigger(position);
        }
        budget.set_finished_last_frame(JobKind::SkyProbe, finished);
        budget.set_running(JobKind::SkyProbe, probes.running.len());
    }

    /// Starts probes for the waiting chunks, fewest provisional columns first
    /// and nearest a camera among equals: one if none is running, and as many
    /// more as the task pool has idle threads.
    pub fn start(
        mut probes: ResMut<Self>,
        mut budget: ResMut<PoolBudget<C>>,
        settings: Res<LightSettings<C>>,
        edits: Res<VoxelEdits<C, C::MaterialIndex>>,
        cameras: Query<&GlobalTransform, With<VoxelWorldCamera<C>>>,
        configuration: Res<C>,
    ) {
        if probes.waiting.is_empty() {
            return;
        }
        #[cfg(feature = "trace")]
        let _span = bevy::log::info_span!("start_sky_probes").entered();
        // One probe always runs while any chunk waits, so streaming can't
        // starve them, but no more than one takes a thread chunk tasks want.
        let floor = probes.running.is_empty() as usize;
        let count = budget.idle().max(floor);
        if count == 0 {
            return;
        }
        let cameras: Vec<Vec3> =
            cameras.iter().map(|camera| camera.translation()).collect();
        let candidates: Vec<((u32, u32), IVec3)> = probes
            .waiting
            .iter()
            .filter_map(|(position, chunk)| {
                let next = chunk.next(*position, &probes.probing)?;
                let unproved = chunk.sky.iter().map(|column| column.count_ones()).sum();
                Some(((unproved, distance_to_nearest(*position, &cameras)), next))
            })
            .collect();
        if candidates.is_empty() {
            return;
        }
        // Chunks above each other wait on the same positions, so more
        // candidates than probes are taken to find enough different ones.
        let mut targets = Vec::with_capacity(count);
        for target in first_in_start_order(candidates, count * 4) {
            if targets.len() == count {
                break;
            }
            if !targets.contains(&target) {
                targets.push(target);
            }
        }

        let thread_pool = AsyncComputeTaskPool::get();
        let generation = settings.record.lock().unwrap().generation();
        let delegate = configuration.voxel_lookup_delegate();
        for position in targets {
            let mut lookup = delegate(position, 0, None);
            let bucket = edits.buckets(std::iter::once(position)).pop().flatten();
            let cells = probes.reads_cells;
            let task = thread_pool.spawn(async move {
                #[cfg(feature = "trace")]
                let _span = bevy::log::info_span!("sky_probe").entered();
                probe_position(position, |at| lookup(at, None), bucket.as_deref(), cells)
            });
            probes.probing.insert(position);
            probes.running.push(Probe {
                position,
                edits: edits.edit_count(position),
                generation,
                task,
            });
            budget.started(JobKind::SkyProbe);
            settings.counters().probes.fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// Picks up a `reset_light` call: forgets the sky record and the sky fill
/// field, works every loaded
/// chunk's column answer out again from its padding row, the fill hints and
/// the loaded chunks' own voxels, and relights every chunk.
pub(crate) fn reset_light<C: VoxelWorldConfig>(
    settings: Res<LightSettings<C>>,
    mut work: ResMut<LightWork<C>>,
    probes: Option<ResMut<SkyProbes<C>>>,
    fill: Option<ResMut<SkyFillWork<C>>>,
    chunk_map: Res<ChunkMap<C, C::MaterialIndex>>,
    edits: Res<VoxelEdits<C, C::MaterialIndex>>,
    configuration: Res<C>,
) {
    if !settings.reset.swap(false, Ordering::AcqRel) {
        return;
    }
    if let (Some(mut fill), Some(field)) = (fill, field_of(&settings.config)) {
        fill.reset(field);
    }
    if let Some(mut probes) = probes {
        let mut record = settings.record.lock().unwrap();
        record.clear();
        answer_loaded_again(
            &mut probes,
            &settings.view,
            &mut record,
            &chunk_map,
            &edits,
            &configuration,
        );
    }
    work.trigger_all();
}

/// Forgets every waiting chunk, records every loaded chunk's own mask, and
/// works each one's column answer out again from its padding row, the fill
/// hints and `record`, under the sky rule of `probes`. Chunks whose answer
/// changed get it in the world view `view`.
///
/// Returns the positions of the chunks whose direct sky changed.
pub(crate) fn answer_loaded_again<C: VoxelWorldConfig>(
    probes: &mut SkyProbes<C>,
    view: &LightView,
    record: &mut SkyRecord,
    chunk_map: &ChunkMap<C, C::MaterialIndex>,
    edits: &VoxelEdits<C, C::MaterialIndex>,
    world: &C,
) -> Vec<IVec3> {
    probes.clear();
    let loaded: Vec<(IVec3, Entity, ColumnAnswer)> = {
        let view = view.read();
        for (position, chunk) in view.iter() {
            let occupancy = &chunk.light.inputs.occupancy;
            let open = record.keeps_cells().then(|| open_cells(occupancy));
            record.insert(
                *position,
                &occupancy.solid_columns(),
                open,
                edits.edit_count(*position),
            );
        }
        view.iter()
            .map(|(position, chunk)| {
                (*position, chunk.entity, chunk.light.inputs.columns.clone())
            })
            .collect()
    };
    let air: Vec<(IVec3, Entity, ColumnAnswer)> = view
        .read()
        .air
        .iter()
        .map(|(position, air)| (*position, ENTRY, air.inputs.columns.clone()))
        .collect();
    let chunk_map = chunk_map.get_read_lock();
    let mut changed = Vec::new();
    for (position, entity, columns) in loaded.into_iter().chain(air) {
        let answer = if entity == ENTRY {
            probes.answer_air(position, record, edits, world)
        } else {
            let Some(data) = chunk_map.get(&ChunkKey::new(position, 0)) else {
                continue;
            };
            probes.answer_again(position, entity, padding_row(data), record, edits, world)
        };
        if answer.sky != columns.sky {
            changed.push(position);
        }
        if answer != columns {
            replace_columns(view, position, entity, answer);
        }
    }
    changed
}

/// The edit counts of the `count` positions above the chunk at `position`,
/// lowest first.
fn edit_counts<C, I: Copy + PartialEq>(
    edits: &VoxelEdits<C, I>,
    position: IVec3,
    count: usize,
) -> Vec<u64> {
    (1..=count as i32)
        .map(|y| edits.edit_count(position + IVec3::Y * y))
        .collect()
}

/// The columns open in a chunk's padding row, `open`, as the fill hints of the
/// positions above the chunk at `position` leave them, and which of those
/// positions the hints settle. A hint is ignored for a position whose edit
/// count in `counts` isn't 0.
fn from_hints<C: VoxelWorldConfig>(
    world: &C,
    position: IVec3,
    open: Columns,
    counts: &[u64],
) -> (Columns, Vec<bool>) {
    let mut sky = open;
    let mut known = vec![false; counts.len()];
    for (index, count) in counts.iter().enumerate() {
        let above = position + IVec3::Y * (index as i32 + 1);
        match Hinted::of(world.chunk_fill_hint(ChunkKey::new(above, 0)), *count > 0) {
            Hinted::Air => known[index] = true,
            Hinted::Solid => sky = [0; CHUNK],
            Hinted::Unknown => {}
        }
    }
    (sky, known)
}

/// The column answer of hinted air at `position` under `rule`, from the fill
/// hints and `record` alone, without waiting for probes: the answer
/// [`SkyProbes::answer_air`] gives it while nothing more is known.
pub(crate) fn air_answer<C: VoxelWorldConfig>(
    rule: SkyRule,
    position: IVec3,
    record: &mut SkyRecord,
    edits: &VoxelEdits<C, C::MaterialIndex>,
    world: &C,
) -> ColumnAnswer {
    let window = rule.window(position.y);
    let counts = edit_counts(edits, position, super::sky::walkable(window) as usize);
    let (mut sky, mut known) = from_hints(world, position, [!0; CHUNK], &counts);
    for (index, known) in known.iter_mut().enumerate() {
        if *known {
            continue;
        }
        let above = position + IVec3::Y * (index as i32 + 1);
        if let Some(mask) = record.get(above, counts[index]) {
            for (sky, solid) in sky.iter_mut().zip(&mask) {
                *sky &= !solid;
            }
            *known = true;
        }
    }
    if known.len() as u32 == window && known.iter().all(|known| *known) {
        proved(sky)
    } else {
        provisional(sky)
    }
}

/// Gives the chunk `entity` at `position` the column answer `answer` in the
/// world view, keeping its light. Relights then read the new answer. With
/// [`ENTRY`] it gives it to a light entry or to hinted air.
pub(crate) fn replace_columns(
    view: &LightView,
    position: IVec3,
    entity: Entity,
    answer: ColumnAnswer,
) {
    let mut view = view.write();
    if entity == ENTRY && view.is_air(position) {
        let air = view.air.get_mut(&position).unwrap();
        air.inputs = Arc::new(air.inputs.with_columns(answer));
        return;
    }
    let Some(chunk) = view
        .get_mut(&position)
        .filter(|chunk| chunk.entity == entity)
    else {
        return;
    };
    let light = &chunk.light;
    chunk.light = Arc::new(ChunkLight {
        inputs: Arc::new(light.inputs.with_columns(answer)),
        sky: light.sky.clone(),
        voxel: light.voxel.clone(),
        precision: light.precision,
    });
}

/// The face neighbours of the chunk at `position`, which a retiring chunk
/// with provisional columns ripples relights from.
pub(crate) fn face_neighbours(position: IVec3) -> impl Iterator<Item = IVec3> {
    Face::ALL
        .into_iter()
        .map(move |face| position + face.offset())
}

/// How far the middle of the level-0 chunk at `position` is from the nearest
/// of `cameras`, in whole voxels.
fn distance_to_nearest(position: IVec3, cameras: &[Vec3]) -> u32 {
    let middle = (position.as_vec3() + 0.5) * CHUNK_SIZE_F;
    cameras
        .iter()
        .map(|camera| camera.distance(middle))
        .fold(f32::INFINITY, f32::min)
        .min(u32::MAX as f32) as u32
}
