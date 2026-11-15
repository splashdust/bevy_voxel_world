//! The systems that run fragment checks for a world that detects dislodged
//! fragments: one check on the task pool per grid a commit's local edits
//! cut, landed in a later frame, run again when what it read moved on.

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
#[cfg(feature = "sub_grids")]
use crate::chunk_lifecycle::handover::Handovers;
use crate::chunk_lifecycle::{ChunkLifecycle, ChunkRequests};
use crate::chunk_map::ChunkMap;
use crate::configuration::VoxelWorldConfig;
use crate::grid::{GridId, GridKey};
use crate::level_0_reader::Level0Reader;
use crate::pool_budget::{JobKind, PoolBudget};
#[cfg(feature = "sub_grids")]
use crate::sub_grid::{Grid, LiftParts, SubGrids};
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
    // After the handovers take this frame's lifts and stamps, so an
    // automatic lift lands with the commit that takes its edits, the next
    // frame's.
    #[cfg(feature = "sub_grids")]
    app.add_systems(
        PreUpdate,
        FragmentChecks::<C>::lift_landed
            .run_if(|checks: Res<FragmentChecks<C>>| !checks.lifting.is_empty())
            .after(ChunkLifecycle::<C>::apply_requests)
            .in_set(VoxelWorldSet::<C>::default()),
    );
}

/// The fragment checks of world `C` running on the task pool.
#[derive(Resource)]
pub(crate) struct FragmentChecks<C: VoxelWorldConfig> {
    running: Vec<Running>,
    /// Fragments of checks landed this frame to lift before they are
    /// written.
    #[cfg(feature = "sub_grids")]
    lifting: Vec<FragmentDislodged<C>>,
    counters: Arc<Counters>,
    _world: PhantomData<C>,
}

impl<C: VoxelWorldConfig> Default for FragmentChecks<C> {
    fn default() -> Self {
        Self {
            running: Vec::new(),
            #[cfg(feature = "sub_grids")]
            lifting: Vec::new(),
            counters: Arc::default(),
            _world: PhantomData,
        }
    }
}

/// One check on the task pool, with what it started from.
struct Running {
    task: Task<Landed>,
    inputs: Inputs,
    /// Whether to lift each fragment it finds, as the settings said when it
    /// started.
    #[cfg(feature = "sub_grids")]
    auto_lift: bool,
}

/// What a check of one grid starts from.
#[derive(Default)]
struct Inputs {
    /// The grid checked.
    grid: GridId,
    /// The entity of the sub-grid checked, or `None` for the world.
    #[cfg(feature = "sub_grids")]
    sub_grid: Option<Entity>,
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
    /// What a check of `grid` starts from, with nothing to check yet.
    fn of(grid: GridId) -> Self {
        Self { grid, ..default() }
    }

    /// The grid checked, as the messages name it.
    #[cfg(feature = "sub_grids")]
    fn grid(&self) -> Grid {
        self.sub_grid.map_or(Grid::World, Grid::SubGrid)
    }

    /// Takes in `other`'s seeds and cleared positions, a check of the same
    /// grid's, so one check covers both.
    fn join(&mut self, other: Inputs) {
        #[cfg(feature = "sub_grids")]
        {
            self.sub_grid = other.sub_grid;
        }
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
    /// Lands the checks that finished, then starts this frame's checks, run
    /// right after the commit while `ChunkRequests::edited` holds its change
    /// sets.
    ///
    /// Checks started in earlier frames land here, so a check never lands in
    /// the frame it started, and an edit set in that frame commits first. A
    /// check whose noted edit stamps moved runs again with this frame's
    /// check of its grid. A check of the world that started before a forget
    /// is dropped, and so is a check of a sub-grid no longer live, or
    /// stamped. This frame's checks start, one per grid, from the voxels the
    /// commit's local edits left air, `Unset` or a non-carrying material, or
    /// removed, and only if the world's settings still ask for them.
    #[allow(clippy::too_many_arguments)]
    pub fn follow_commit(
        mut checks: ResMut<Self>,
        requests: Res<ChunkRequests<C>>,
        edits: Res<VoxelEdits<C, C::MaterialIndex>>,
        chunk_map: Res<ChunkMap<C, C::MaterialIndex>>,
        configuration: Res<ConfigCopy<C>>,
        mut budget: ResMut<PoolBudget<C>>,
        mut dislodged: MessageWriter<FragmentDislodged<C>>,
        #[cfg(feature = "sub_grids")] sub_grids: Res<SubGrids<C>>,
        #[cfg(feature = "sub_grids")] handovers: Res<Handovers<C>>,
    ) {
        let checks = &mut *checks;
        let epoch = edits.forget_epoch();
        let mut inputs: Vec<Inputs> = Vec::new();
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
            let running = checks.running.swap_remove(i);
            let ran = running.inputs;
            #[cfg(feature = "sub_grids")]
            let gone = match ran.sub_grid {
                // A sub-grid's voxels go when it despawns, and a stamp moves
                // them into another grid.
                Some(entity) => {
                    sub_grids.live(entity).map(|(grid, _)| grid) != Some(ran.grid)
                        || handovers.is_stamped_away(entity)
                }
                // The edits the check followed were forgotten.
                None => ran.epoch != epoch,
            };
            #[cfg(not(feature = "sub_grids"))]
            let gone = ran.epoch != epoch;
            if gone {
                continue;
            }
            let moved = landed.stamps.iter().any(|(chunk, stamp)| {
                edits.edit_stamp_in(GridKey::level_0(ran.grid, *chunk)) != *stamp
            });
            if moved {
                checks.counters.reruns.fetch_add(1, Ordering::Relaxed);
                inputs_of(&mut inputs, ran.grid).join(ran);
                continue;
            }
            let fragments = landed.check.fragments.into_iter().map(|fragment| {
                FragmentDislodged::of(
                    fragment,
                    #[cfg(feature = "sub_grids")]
                    ran.grid(),
                )
            });
            #[cfg(feature = "sub_grids")]
            if running.auto_lift {
                checks.lifting.extend(fragments);
                continue;
            }
            dislodged.write_batch(fragments);
        }
        budget.set_finished_last_frame(JobKind::FragmentCheck, finished);

        let detection = configuration.0.detect_dislodged_fragments();
        if let Some(detection) = &detection {
            gather(
                &requests,
                detection,
                #[cfg(feature = "sub_grids")]
                |grid| {
                    sub_grids
                        .entity_of(grid)
                        .filter(|&entity| !handovers.is_stamped_away(entity))
                },
                &mut inputs,
            );
        }
        if let Some(detection) = detection {
            for mut inputs in inputs {
                if inputs.seeds.is_empty() {
                    continue;
                }
                inputs.epoch = epoch;
                checks.start(inputs, &detection, &edits, &chunk_map, &configuration);
            }
        }
        budget.set_running(JobKind::FragmentCheck, checks.running.len());
    }

    /// Lifts each fragment of the checks landed this frame that asked for
    /// it into a new sub-grid, then writes its `FragmentDislodged`, naming
    /// the sub-grid.
    ///
    /// It runs after the handovers take this frame's lifts and stamps, so
    /// each lift lands with the next frame's commit, which takes its edits.
    /// Lifts commit in the order their fragments were found.
    #[cfg(feature = "sub_grids")]
    #[allow(clippy::too_many_arguments)]
    pub fn lift_landed(
        mut checks: ResMut<Self>,
        sub_grids: Res<SubGrids<C>>,
        edits: Res<VoxelEdits<C, C::MaterialIndex>>,
        chunk_map: Res<ChunkMap<C, C::MaterialIndex>>,
        configuration: Res<ConfigCopy<C>>,
        handovers: Res<Handovers<C>>,
        mut commands: Commands,
        mut dislodged: MessageWriter<FragmentDislodged<C>>,
    ) {
        if checks.lifting.is_empty() {
            return;
        }
        for mut fragment in checks.lifting.drain(..) {
            let parts = LiftParts {
                sub_grids: &sub_grids,
                edits: &edits,
                chunk_map: &chunk_map,
                configuration: &configuration.0,
                handovers: &handovers,
                commands: &mut commands,
            };
            let (grid, min, max) = (fragment.grid, fragment.min, fragment.max);
            fragment.lifted =
                crate::sub_grid::lift(parts, grid, min, max, |p, _| fragment.contains(p))
                    .ok();
            dislodged.write(fragment);
        }
    }

    /// Starts a check on the task pool from `inputs`, by `detection`.
    fn start(
        &mut self,
        inputs: Inputs,
        detection: &FragmentDetection<C::MaterialIndex>,
        edits: &VoxelEdits<C, C::MaterialIndex>,
        chunk_map: &ChunkMap<C, C::MaterialIndex>,
        configuration: &ConfigCopy<C>,
    ) {
        let world = configuration.0.clone();
        let stamps = edits.stamps_in(inputs.grid);
        // A sub-grid is checked with no limit, so its largest piece keeps it.
        let (mut reader, limit) = match inputs.grid {
            #[cfg(feature = "sub_grids")]
            grid if grid != GridId::WORLD => (
                Level0Reader::for_sub_grid_fragment_check(
                    world,
                    edits.reader_in(grid),
                    chunk_map.get_map(),
                    stamps,
                ),
                None,
            ),
            _ => (
                Level0Reader::for_fragment_check(
                    world,
                    edits.reader(),
                    chunk_map.get_map(),
                    stamps,
                    inputs.cleared.clone(),
                ),
                Some(detection.limit),
            ),
        };
        let connectivity = detection.connectivity;
        let non_carrying = detection.non_carrying.clone();
        let seeds = inputs.seeds.clone();
        let counters = self.counters.clone();
        counters.checks.fetch_add(1, Ordering::Relaxed);
        let task = AsyncComputeTaskPool::get().spawn(async move {
            let started = Instant::now();
            let rules = FragmentRules {
                limit,
                connectivity,
                non_carrying: &non_carrying,
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
        self.running.push(Running {
            task,
            inputs,
            #[cfg(feature = "sub_grids")]
            auto_lift: detection.auto_lift,
        });
    }

    /// Whether no check is running.
    #[cfg(test)]
    pub fn is_idle(&self) -> bool {
        self.running.is_empty()
    }
}

/// The inputs of `grid`'s check in `inputs`, added if it has none yet.
fn inputs_of(inputs: &mut Vec<Inputs>, grid: GridId) -> &mut Inputs {
    match inputs.iter().position(|inputs| inputs.grid == grid) {
        Some(i) => &mut inputs[i],
        None => {
            inputs.push(Inputs::of(grid));
            inputs.last_mut().unwrap()
        }
    }
}

/// Adds to `inputs` the seeds and cleared positions of this frame's change
/// sets, by grid.
///
/// Every removed edit counts as cleared, since its published chunk may still
/// show it. Only a local change seeds, and only one that may leave the voxel
/// not carrying: a removed edit, whose voxel the world generates, or a voxel
/// set to air, `Unset` or a material `detection` lists as not carrying.
///
/// With `sub_grids`, a sub-grid's change sets count only where `checked`
/// gives its entity: one live and not stamped.
fn gather<C: VoxelWorldConfig>(
    requests: &ChunkRequests<C>,
    detection: &FragmentDetection<C::MaterialIndex>,
    #[cfg(feature = "sub_grids")] checked: impl Fn(GridId) -> Option<Entity>,
    inputs: &mut Vec<Inputs>,
) {
    for set in &requests.edited {
        #[cfg(feature = "sub_grids")]
        let sub_grid = match set.grid {
            GridId::WORLD => None,
            grid => match checked(grid) {
                Some(entity) => Some(entity),
                None => continue,
            },
        };
        let into = inputs_of(inputs, set.grid);
        #[cfg(feature = "sub_grids")]
        {
            into.sub_grid = sub_grid;
        }
        for (i, (position, voxel)) in set.voxels.iter().enumerate() {
            if voxel.is_none() {
                into.cleared.insert(*position);
            }
            let seeds = match voxel {
                Some(WorldVoxel::Solid(material)) => {
                    detection.non_carrying.contains(material)
                }
                _ => true,
            };
            if seeds && set.origin.is_local(i) {
                into.seeds.push(*position);
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
