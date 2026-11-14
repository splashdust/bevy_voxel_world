//! The systems that run fragment checks for a world that detects dislodged
//! fragments: one check on the task pool per commit with local cuts, landed
//! in a later frame, run again when what it read moved on.

use std::marker::PhantomData;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use bevy::platform::collections::{HashMap, HashSet};
use bevy::platform::time::Instant;
use bevy::prelude::*;
use bevy::tasks::{AsyncComputeTaskPool, Task};
use futures_lite::future;

use super::check::{check_fragments, FragmentCheck, FragmentRules};
use super::{FragmentDetection, FragmentDislodged};
use crate::chunk_lifecycle::{ChunkLifecycle, ChunkRequests};
use crate::chunk_map::ChunkMap;
use crate::configuration::VoxelWorldConfig;
use crate::grid::GridId;
use crate::level_0_reader::Level0Reader;
use crate::pool_budget::{JobKind, PoolBudget};
use crate::voxel::WorldVoxel;
use crate::voxel_edits::VoxelEdits;
use crate::voxel_world::{ConfigCopy, VoxelWorldSet};
use crate::voxel_world_internal::Internals;

/// Sets up fragment detection for world `C`, if it opts in.
///
/// The message is added either way, so a system reading it runs in any
/// world. Only a world whose `detect_dislodged_fragments` returns settings
/// gets the detection system, and commits that record where each change
/// came from.
pub(crate) fn build<C: VoxelWorldConfig>(app: &mut App, world: &C) {
    app.add_message::<FragmentDislodged<C>>();
    if world.detect_dislodged_fragments().is_none() {
        return;
    }
    // The plugin's setup inits the voxel edits only where none were inserted.
    app.insert_resource(VoxelEdits::<C, C::MaterialIndex>::tracking_origins())
        .init_resource::<FragmentChecks<C>>()
        .add_systems(
            PreUpdate,
            FragmentChecks::<C>::follow_commit
                .after(Internals::<C>::commit_voxel_edits)
                .before(ChunkLifecycle::<C>::collect_chunk_tasks)
                .in_set(VoxelWorldSet::<C>::default()),
        );
}

/// The fragment checks of world `C` running on the task pool.
#[derive(Resource)]
pub(crate) struct FragmentChecks<C: VoxelWorldConfig> {
    running: Vec<Running>,
    counters: Arc<Counters>,
    _world: PhantomData<C>,
}

impl<C: VoxelWorldConfig> Default for FragmentChecks<C> {
    fn default() -> Self {
        Self {
            running: Vec::new(),
            counters: Arc::default(),
            _world: PhantomData,
        }
    }
}

/// One check on the task pool, with what it started from.
struct Running {
    task: Task<Landed>,
    inputs: Inputs,
}

/// What a check starts from.
#[derive(Default)]
struct Inputs {
    /// Positions whose voxels local edits changed, any of which may no
    /// longer carry.
    seeds: Vec<IVec3>,
    /// Positions whose edits were removed, which read the world's voxel
    /// lookup since their published chunks may still show the edit.
    cleared: HashSet<IVec3>,
    /// The forget epoch when the check started.
    epoch: u64,
}

impl Inputs {
    /// Takes in `other`'s seeds and cleared positions, so one check covers
    /// both.
    fn join(&mut self, other: Inputs) {
        self.seeds.extend(other.seeds);
        self.cleared.extend(other.cleared);
    }
}

/// What a check found, with the edit stamp of each chunk position it read
/// from, noted before its first read there.
struct Landed {
    check: FragmentCheck,
    stamps: HashMap<IVec3, u64>,
}

/// What a world's fragment checks have done, for the crate's benchmarks.
#[derive(Default)]
struct Counters {
    checks: AtomicU64,
    visited: AtomicU64,
    reruns: AtomicU64,
    task_nanos: AtomicU64,
}

impl<C: VoxelWorldConfig> FragmentChecks<C> {
    /// Lands the checks that finished, then starts this frame's check, run
    /// right after the commit while `ChunkRequests::edited` holds its change
    /// sets.
    ///
    /// Checks started in earlier frames land here, so a check never lands in
    /// the frame it started, and an edit set in that frame commits first. A
    /// check whose noted edit stamps moved runs again with this frame's
    /// check, and one that started before a forget is dropped. This frame's
    /// check starts from the voxels the commit's local edits left air,
    /// `Unset` or a non-carrying material, or removed, and only if the
    /// world's settings still ask for one.
    #[allow(clippy::too_many_arguments)]
    pub fn follow_commit(
        mut checks: ResMut<Self>,
        requests: Res<ChunkRequests<C>>,
        edits: Res<VoxelEdits<C, C::MaterialIndex>>,
        chunk_map: Res<ChunkMap<C, C::MaterialIndex>>,
        configuration: Res<ConfigCopy<C>>,
        mut budget: ResMut<PoolBudget<C>>,
        mut dislodged: MessageWriter<FragmentDislodged<C>>,
    ) {
        let checks = &mut *checks;
        let epoch = edits.forget_epoch();
        let mut inputs = Inputs::default();
        let mut finished = 0;
        let mut i = 0;
        while i < checks.running.len() {
            let Some(landed) =
                future::block_on(future::poll_once(&mut checks.running[i].task))
            else {
                i += 1;
                continue;
            };
            finished += 1;
            let Running { inputs: ran, .. } = checks.running.swap_remove(i);
            // The edits the check followed were forgotten.
            if ran.epoch != epoch {
                continue;
            }
            let moved = landed
                .stamps
                .iter()
                .any(|(chunk, stamp)| edits.edit_stamp(*chunk) != *stamp);
            if moved {
                checks.counters.reruns.fetch_add(1, Ordering::Relaxed);
                inputs.join(ran);
                continue;
            }
            dislodged.write_batch(
                landed
                    .check
                    .fragments
                    .into_iter()
                    .map(FragmentDislodged::of),
            );
        }
        budget.set_finished_last_frame(JobKind::FragmentCheck, finished);

        let detection = configuration.0.detect_dislodged_fragments();
        if let Some(detection) = &detection {
            gather(&requests, detection, &mut inputs);
        }
        if let (Some(detection), false) = (detection, inputs.seeds.is_empty()) {
            inputs.epoch = epoch;
            checks.start(inputs, detection, &edits, &chunk_map, &configuration);
        }
        budget.set_running(JobKind::FragmentCheck, checks.running.len());
    }

    /// Starts a check on the task pool from `inputs`, by `detection`.
    fn start(
        &mut self,
        inputs: Inputs,
        detection: FragmentDetection<C::MaterialIndex>,
        edits: &VoxelEdits<C, C::MaterialIndex>,
        chunk_map: &ChunkMap<C, C::MaterialIndex>,
        configuration: &ConfigCopy<C>,
    ) {
        let mut reader = Level0Reader::for_fragment_check(
            configuration.0.clone(),
            edits.reader(),
            chunk_map.get_map(),
            edits.stamps(),
            inputs.cleared.clone(),
        );
        let seeds = inputs.seeds.clone();
        let counters = self.counters.clone();
        counters.checks.fetch_add(1, Ordering::Relaxed);
        let task = AsyncComputeTaskPool::get().spawn(async move {
            let started = Instant::now();
            let rules = FragmentRules {
                limit: Some(detection.limit),
                connectivity: detection.connectivity,
                non_carrying: &detection.non_carrying,
            };
            // A seed set back to a carrying voxel since, as by a support
            // placed where a voxel was dug, no longer seeds.
            let open: Vec<IVec3> = seeds
                .into_iter()
                .filter(|&seed| !rules.carries(&reader.read(seed)))
                .collect();
            let check = check_fragments(open, |p| reader.read(p), &rules);
            counters
                .visited
                .fetch_add(check.visited as u64, Ordering::Relaxed);
            counters
                .task_nanos
                .fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
            Landed {
                check,
                stamps: reader.into_noted_stamps(),
            }
        });
        self.running.push(Running { task, inputs });
    }

    /// Whether no check is running.
    #[cfg(test)]
    pub fn is_idle(&self) -> bool {
        self.running.is_empty()
    }
}

/// Adds to `inputs` the seeds and cleared positions of this frame's change
/// sets of the world's grid.
///
/// Every removed edit counts as cleared, since its published chunk may still
/// show it. Only a local change seeds, and only one that may leave the voxel
/// not carrying: a removed edit, whose voxel the world generates, or a voxel
/// set to air, `Unset` or a material `detection` lists as not carrying.
fn gather<C: VoxelWorldConfig>(
    requests: &ChunkRequests<C>,
    detection: &FragmentDetection<C::MaterialIndex>,
    inputs: &mut Inputs,
) {
    for set in requests
        .edited
        .iter()
        .filter(|set| set.grid == GridId::WORLD)
    {
        for (i, (position, voxel)) in set.voxels.iter().enumerate() {
            if voxel.is_none() {
                inputs.cleared.insert(*position);
            }
            let seeds = match voxel {
                Some(WorldVoxel::Solid(material)) => {
                    detection.non_carrying.contains(material)
                }
                _ => true,
            };
            if seeds && set.origin.is_local(i) {
                inputs.seeds.push(*position);
            }
        }
    }
}

/// What the fragment checks of a world have done since it started.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FragmentStats {
    /// Checks started, reruns among them.
    pub checks: u64,
    /// Voxels the checks' floods visited.
    pub voxels_visited: u64,
    /// Checks that landed after what they read moved on, and ran again.
    pub reruns: u64,
    /// Time the checks took on the task pool, in microseconds.
    pub task_micros: u64,
}

/// What the fragment checks of world `C` have done, or `None` if it doesn't
/// detect dislodged fragments.
pub fn fragment_stats<C: VoxelWorldConfig>(world: &World) -> Option<FragmentStats> {
    let counters = &world.get_resource::<FragmentChecks<C>>()?.counters;
    let read = |counter: &AtomicU64| counter.load(Ordering::Relaxed);
    Some(FragmentStats {
        checks: read(&counters.checks),
        voxels_visited: read(&counters.visited),
        reruns: read(&counters.reruns),
        task_micros: read(&counters.task_nanos) / 1000,
    })
}
