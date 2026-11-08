//! Keeping a world's sky fill field current on the main thread, relighting
//! the chunks whose fill changed, and running fill probes.
//!
//! The field learns a position's cells whenever the sky record takes in its
//! voxels or forgets it, a chunk lands or retires there, or whether its
//! columns are proved clear changes. A loaded chunk's open cells come from its
//! occupancy and its direct sky from its own column answer, once no column
//! of it waits for sky probes. Any other position's come from the record: its
//! cells, and its columns clear through the sky rule's window, read from the
//! masks the record holds for the positions above it. A position above that
//! the record doesn't hold makes every column blocked, so only columns proved
//! clear count as sky.
//!
//! When a position's cells change, the chunks in the 3×3×3 around it that
//! take sky fill relight, at most every 500 ms, after chunk tasks. A chunk
//! takes sky fill when it has an open voxel touching a solid by a face, an
//! edge or a corner, counting the solids in its padding, unless its open
//! voxels all have direct sky: those are the chunks whose faces can read the
//! fill around them.
//!
//! Fill probes read the positions the field asks for that could raise a
//! level of a position whose 3×3×3 holds a chunk taking sky fill, with
//! whatever room the task pool has left after everything else, and nothing
//! at rest. A position the record forgot to stay within its cap isn't read
//! again until a chunk lands or retires, a voxel edit changes a column, or
//! sky probes start or stop waiting: a record too small for what the field
//! asks for would otherwise forget one position for each it reads, for ever.

use std::marker::PhantomData;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;

use bevy::platform::collections::HashSet;
use bevy::platform::time::Instant;
use bevy::prelude::*;
use bevy::tasks::{AsyncComputeTaskPool, Task};
use futures_lite::future;

use super::{
    direct_cells, open_cells, sky_top_cells, FillCells, PositionCells, SkyFillField,
};
use crate::chunk::{ChunkKey, CHUNK_SIZE_F};
use crate::configuration::VoxelWorldConfig;
use crate::light::config::LightPropagation;
use crate::light::inputs::LightInputs;
use crate::light::probe::{probe_position, ProbeRead, SkyProbes};
use crate::light::record::SkyRecord;
use crate::light::relight::LightWork;
use crate::light::sky::{walkable, Hinted, SkyRule};
use crate::light::view::LightView;
use crate::light::LightSettings;
use crate::pool_budget::{first_in_start_order, JobKind, PoolBudget};
use crate::voxel_edits::VoxelEdits;
use crate::voxel_world::VoxelWorldCamera;

/// The least time between two relights of the chunks whose fill changed.
const RELIGHT_DELAY: Duration = Duration::from_millis(500);

/// How long the field may take updating in a frame. What is left waits for
/// the next frame.
const UPDATE_BUDGET: Duration = Duration::from_millis(1);

/// How many chunk positions below a changed position a sky ceiling's window
/// is followed down.
const CEILING_DEPTH: i32 = 64;

struct FillProbe {
    position: IVec3,
    edits: u64,
    generation: u64,
    task: Task<ProbeRead>,
}

/// The sky fill of world `C`: its field, the work keeping it current and its
/// fill probes. Only a world with sky light and sky fill on has one.
#[derive(Resource)]
pub(crate) struct SkyFillWork<C> {
    field: SkyFillField,
    /// The field's cells, as the tasks that solve light read them.
    cells: FillCells,
    /// The positions the field has learned and not forgotten.
    known: HashSet<IVec3>,
    /// The positions whose cells to work out again.
    dirty: HashSet<IVec3>,
    /// The positions whose cells changed since the last relight.
    changed: HashSet<IVec3>,
    last_relight: Option<Instant>,
    running: Vec<FillProbe>,
    probing: HashSet<IVec3>,
    /// How many positions fill probes have yet to read.
    wanted: usize,
    /// Positions the record forgot to stay within its cap since anything
    /// else changed, which fill probes don't read.
    evicted: HashSet<IVec3>,
    /// Whether anything changed since fill probes last looked for positions
    /// to read. With nothing changed, they find none they didn't find then.
    recheck: bool,
    _world: PhantomData<C>,
}

impl<C: VoxelWorldConfig> SkyFillWork<C> {
    /// The work of a world whose field is `field`, which writes its cells
    /// into `cells`.
    pub fn new(field: SkyFillField, cells: FillCells) -> Self {
        cells.write().clear();
        Self {
            field,
            cells,
            known: HashSet::new(),
            dirty: HashSet::new(),
            changed: HashSet::new(),
            last_relight: None,
            running: Vec::new(),
            probing: HashSet::new(),
            wanted: 0,
            evicted: HashSet::new(),
            recheck: true,
            _world: PhantomData,
        }
    }

    /// Whether the field is up to date with everything it was told, no fill
    /// relight waits for its delay, and no fill probe runs or waits.
    pub fn is_settled(&self) -> bool {
        self.dirty.is_empty()
            && !self.field.has_pending()
            && self.changed.is_empty()
            && self.running.is_empty()
            && self.wanted == 0
    }

    /// Whether every running fill probe's task has finished, for its read to
    /// be collected on the next frame.
    #[cfg(test)]
    pub(crate) fn tasks_finished(&self) -> bool {
        self.running.iter().all(|probe| probe.task.is_finished())
    }

    /// Starts again from an empty field of `field`, as after `reset_light`.
    pub fn reset(&mut self, field: SkyFillField) {
        *self = Self::new(field, self.cells.clone());
    }

    /// Learns every loaded chunk and every position the record holds cells
    /// for, and brings the field up to date at once, for a change of
    /// settings. Loaded chunks' cells go into the record.
    #[allow(clippy::too_many_arguments)]
    pub fn learn_everything(
        &mut self,
        view: &LightView,
        record: &mut SkyRecord,
        probes: Option<&SkyProbes<C>>,
        rule: SkyRule,
        edits: &VoxelEdits<C, C::MaterialIndex>,
        world: &C,
    ) {
        for (position, chunk) in view.read().iter() {
            let occupancy = &chunk.light.inputs.occupancy;
            let count = edits.edit_count(*position);
            if chunk.edits == count {
                let open = Some(open_cells(occupancy));
                record.insert(*position, &occupancy.solid_columns(), open, count);
            }
            self.dirty.insert(*position);
        }
        self.dirty.extend(view.read().air.keys().copied());
        for (position, _) in record.all_cells() {
            self.dirty.insert(position);
        }
        record.take_noted();
        record.take_forgotten();
        self.work_out_dirty(view, record, probes, rule, edits, world);
        let changed = self.field.update(Duration::MAX);
        self.publish(changed);
        // The settings relight that follows covers every chunk taking fill.
        self.changed.clear();
    }

    /// Works out the cells of every dirty position again and tells the field.
    fn work_out_dirty(
        &mut self,
        view: &LightView,
        record: &mut SkyRecord,
        probes: Option<&SkyProbes<C>>,
        rule: SkyRule,
        edits: &VoxelEdits<C, C::MaterialIndex>,
        world: &C,
    ) {
        if self.dirty.is_empty() {
            return;
        }
        let view = view.read();
        for position in std::mem::take(&mut self.dirty) {
            let count = edits.edit_count(position);
            // Hinted air is learned from its hint, with no lookups.
            let inputs = view.inputs(position);
            let open = match inputs {
                Some(inputs) => Some(open_cells(&inputs.occupancy)),
                None => record.cells(position, count).map(|cells| cells.open),
            };
            let Some(open) = open else {
                if self.known.remove(&position) {
                    self.field.forget(position);
                }
                continue;
            };
            let top = match inputs {
                Some(_) if probes.is_some_and(|probes| probes.is_waiting(position)) => 0,
                Some(inputs) => sky_top_cells(&inputs.columns.sky),
                None => top_from_record(position, rule, &view, record, edits, world),
            };
            let direct = direct_cells(open, top);
            record.set_direct(position, direct);
            self.field.learn(position, PositionCells { open, direct });
            self.known.insert(position);
        }
    }

    /// Writes the cells of the positions `changed` for the tasks to read,
    /// and notes them for relighting.
    fn publish(&mut self, changed: Vec<IVec3>) {
        if changed.is_empty() {
            return;
        }
        let mut cells = self.cells.write();
        for position in changed {
            match self.field.cells(position) {
                Some(fill) => {
                    cells.insert(position, Arc::new(fill.clone()));
                }
                None => {
                    cells.remove(&position);
                }
            }
            self.changed.insert(position);
        }
    }

    /// Positions to work out again when `position` was recorded or
    /// forgotten: it, and the known positions below it whose sky rule's
    /// window holds it.
    fn note_with_below(&mut self, position: IVec3, rule: SkyRule) {
        self.dirty.insert(position);
        let depth = match rule {
            SkyRule::Clearance(chunks) => chunks as i32,
            SkyRule::Ceiling(top) if position.y <= top => CEILING_DEPTH,
            SkyRule::Ceiling(_) => 0,
        };
        for below in 1..=depth {
            let below = position - IVec3::Y * below;
            if self.known.contains(&below) {
                self.dirty.insert(below);
            }
        }
    }

    /// Collects fill probes, takes in what changed for sky fill this frame,
    /// updates the field, and relights the chunks whose fill changed once
    /// the last such relight is 500 ms old.
    #[allow(clippy::too_many_arguments)]
    pub fn follow(
        mut fill: ResMut<Self>,
        mut work: ResMut<LightWork<C>>,
        mut probes: Option<ResMut<SkyProbes<C>>>,
        settings: Res<LightSettings<C>>,
        edits: Res<VoxelEdits<C, C::MaterialIndex>>,
        world: Res<C>,
        mut budget: ResMut<PoolBudget<C>>,
    ) {
        let Some(rule) = settings.sky_rule else {
            return;
        };
        #[cfg(feature = "trace")]
        let _span = bevy::log::info_span!("follow_sky_fill").entered();
        let started = Instant::now();
        let fill = fill.as_mut();
        let mut record = settings.record.lock().unwrap();
        let counters = settings.counters();

        let mut finished = 0;
        let mut index = 0;
        while index < fill.running.len() {
            let Some(read) =
                future::block_on(future::poll_once(&mut fill.running[index].task))
            else {
                index += 1;
                continue;
            };
            finished += 1;
            let probe = fill.running.swap_remove(index);
            fill.probing.remove(&probe.position);
            counters
                .fill_probe_lookups
                .fetch_add(read.lookups, Ordering::Relaxed);
            if probe.generation != record.generation()
                || edits.edit_count(probe.position) != probe.edits
            {
                continue;
            }
            record.insert(probe.position, &read.solid, read.open, probe.edits);
            if let Some(probes) = probes.as_mut() {
                probes.learned(probe.position, &read.solid, &settings.view);
            }
        }
        budget.set_finished_last_frame(JobKind::FillProbe, finished);
        budget.set_running(JobKind::FillProbe, fill.running.len());

        let touched: Vec<IVec3> = probes
            .as_mut()
            .map(|probes| probes.take_touched())
            .unwrap_or_default()
            .into_iter()
            .chain(work.take_fill_touched())
            .collect();
        if !touched.is_empty() {
            fill.evicted.clear();
        }
        fill.dirty.extend(touched);
        for position in record.take_forgotten() {
            // A position replaced in the record is held again.
            if record.cells(position, edits.edit_count(position)).is_none() {
                fill.evicted.insert(position);
            }
            fill.note_with_below(position, rule);
        }
        for position in record.take_noted() {
            fill.note_with_below(position, rule);
        }
        fill.recheck |=
            finished > 0 || !fill.dirty.is_empty() || fill.field.has_pending();
        fill.work_out_dirty(
            &settings.view,
            &mut record,
            probes.as_deref(),
            rule,
            &edits,
            &world,
        );
        let updating = Instant::now();
        let changed = fill.field.update(UPDATE_BUDGET);
        counters
            .fill_update_nanos
            .fetch_add(updating.elapsed().as_nanos() as u64, Ordering::Relaxed);
        fill.publish(changed);
        record.set_field_bytes(fill.field.bytes());
        drop(record);

        let due = fill
            .last_relight
            .is_none_or(|last| last.elapsed() >= RELIGHT_DELAY);
        if !fill.changed.is_empty() && due {
            fill.last_relight = Some(Instant::now());
            let view = settings.view.read();
            let mut marked = HashSet::new();
            for position in fill.changed.drain() {
                for around in around(position) {
                    if marked.insert(around)
                        && view
                            .get(&around)
                            .is_some_and(|chunk| takes_fill(&chunk.light.inputs))
                    {
                        work.trigger_fill(around);
                    }
                }
            }
        }
        counters
            .fill_main_thread_nanos
            .fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }

    /// Starts fill probes of the positions the field asks for that could
    /// raise a level of a position whose 3×3×3 holds a chunk taking sky fill,
    /// nearest a camera first, as many as the task pool has idle threads.
    #[allow(clippy::too_many_arguments)]
    pub fn start_probes(
        mut fill: ResMut<Self>,
        mut budget: ResMut<PoolBudget<C>>,
        settings: Res<LightSettings<C>>,
        edits: Res<VoxelEdits<C, C::MaterialIndex>>,
        cameras: Query<&GlobalTransform, With<VoxelWorldCamera<C>>>,
        world: Res<C>,
    ) {
        let fill = fill.as_mut();
        // Looking costs a scan of the chunks around every position the field
        // wants, so at rest it isn't repeated every frame.
        if !fill.recheck && fill.wanted == 0 {
            return;
        }
        fill.recheck = false;
        if fill.field.wanted_probes().next().is_none() {
            fill.wanted = 0;
            return;
        }
        let candidates: Vec<IVec3> = {
            let view = settings.view.read();
            let mut candidates = HashSet::new();
            for (position, raises) in fill.field.wanted_probes() {
                if candidates.contains(&position)
                    || fill.probing.contains(&position)
                    || fill.evicted.contains(&position)
                {
                    continue;
                }
                let matters = around(raises).any(|around| {
                    view.get(&around)
                        .is_some_and(|chunk| takes_fill(&chunk.light.inputs))
                });
                if matters {
                    candidates.insert(position);
                }
            }
            candidates.into_iter().collect()
        };
        fill.wanted = candidates.len();
        let count = budget.idle();
        if candidates.is_empty() || count == 0 {
            return;
        }
        let cameras: Vec<Vec3> =
            cameras.iter().map(|camera| camera.translation()).collect();
        let candidates = candidates
            .into_iter()
            .map(|position| (distance_to_nearest(position, &cameras), position))
            .collect();
        let generation = settings.record.lock().unwrap().generation();
        let delegate = world.voxel_lookup_delegate();
        let thread_pool = AsyncComputeTaskPool::get();
        for position in first_in_start_order(candidates, count) {
            let mut lookup = delegate(position, 0, None);
            let bucket = edits.buckets(std::iter::once(position)).pop().flatten();
            let task = thread_pool.spawn(async move {
                #[cfg(feature = "trace")]
                let _span = bevy::log::info_span!("fill_probe").entered();
                probe_position(position, |at| lookup(at, None), bucket.as_deref(), true)
            });
            fill.probing.insert(position);
            fill.running.push(FillProbe {
                position,
                edits: edits.edit_count(position),
                generation,
                task,
            });
            fill.wanted -= 1;
            budget.started(JobKind::FillProbe);
            settings
                .counters()
                .fill_probes
                .fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// Whether the chunk with `inputs` takes sky fill: it has an open voxel
/// touching a solid, counting its padding, and not all its open voxels have
/// direct sky.
pub(crate) fn takes_fill(inputs: &LightInputs) -> bool {
    inputs.touches_solid() && !inputs.all_open_direct()
}

/// The top cells of the position `position`, which no chunk holds, whose
/// columns are all clear through the sky rule's window, as the record and
/// fill hints tell. A position above that neither tells about blocks every
/// column.
fn top_from_record<C: VoxelWorldConfig>(
    position: IVec3,
    rule: SkyRule,
    view: &bevy::platform::collections::HashMap<IVec3, crate::light::view::ViewChunk>,
    record: &mut SkyRecord,
    edits: &VoxelEdits<C, C::MaterialIndex>,
    world: &C,
) -> u16 {
    let window = rule.window(position.y);
    if window > walkable(window) {
        return 0;
    }
    let mut sky = [!0u32; crate::chunk::CHUNK_SIZE_U as usize];
    for above in 1..=window as i32 {
        let above = position + IVec3::Y * above;
        let count = edits.edit_count(above);
        let solid = match view.get(&above) {
            Some(chunk) if chunk.edits == count => {
                Some(chunk.light.inputs.occupancy.solid_columns())
            }
            _ => record.get(above, count),
        };
        let solid = match solid {
            Some(solid) => solid,
            None => match Hinted::of(
                world.chunk_fill_hint(ChunkKey::new(above, 0)),
                count > 0,
            ) {
                Hinted::Air => continue,
                Hinted::Solid | Hinted::Unknown => return 0,
            },
        };
        for (sky, solid) in sky.iter_mut().zip(&solid) {
            *sky &= !solid;
        }
    }
    sky_top_cells(&sky)
}

/// The 27 chunk positions of the 3×3×3 around `position`, itself included.
fn around(position: IVec3) -> impl Iterator<Item = IVec3> {
    (-1..=1).flat_map(move |z| {
        (-1..=1).flat_map(move |y| (-1..=1).map(move |x| position + IVec3::new(x, y, z)))
    })
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

/// The field of `config`, if it has sky light and sky fill on.
pub(crate) fn field_of(config: &LightPropagation) -> Option<SkyFillField> {
    SkyRule::of(&config.sky)?;
    Some(SkyFillField::new(config.sky_fill.as_ref()?))
}
