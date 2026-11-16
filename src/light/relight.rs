//! Relights: solving a chunk's light again when chunks arriving around it
//! could light it, and landing every solve's light on the main thread.
//!
//! A chunk's light can change after it appears, because its neighbours arrive
//! after its first solve, which reads a neighbour that isn't loaded as solid.
//! When a chunk's light lands, it is compared with each loaded face
//! neighbour's: a neighbour relights if this chunk's light could get to a
//! voxel on their shared face sooner than the neighbour's level there says
//! light did. The chunk task makes that comparison against the neighbours it
//! read, and landing makes it again only for neighbours whose light has
//! changed since. Landing also checks the other way for those, so a chunk
//! whose task missed a neighbour arriving relights itself.
//!
//! A relight goes on as a relight ripple: when its light lands, it relights a
//! neighbour only across faces where light reached the chunk at another time
//! than before. Each solve keeps a hash per face of the round in which light
//! reached each voxel on that face, and with round spread the pass too,
//! because in a round of three passes a voxel reached early in the round
//! still sends light across some faces in that same round. Light crossing
//! every face at the same times as before can't light the neighbours any
//! differently, so the ripple stops there.
//!
//! Relights are tasks of their own. They never change a chunk's
//! `ChunkState` and never re-mesh it, so a relit chunk stays Ready and
//! visible. Each chunk has a light state beside its chunk state: clean, dirty
//! or relighting. A chunk runs at most one relight: a newer trigger marks it
//! dirty, the running relight still lands, and one more follows.
//!
//! Relights a voxel edit triggers are urgent (see [`edit`](super::edit)):
//! they start at once, ahead of the pool budget, and so does their ripple. A
//! chunk being re-meshed for the edit holds its relit light until a mesh of
//! the voxels that relight read lands, so its old mesh never shows light
//! worked out for its new voxels, such as a placed block's old floor face
//! reading the new solid's level of 0.
//!
//! A handover holds the light of its chunks the same way, being re-meshed
//! or not, until it lands, and waits for the relights its commit started to
//! solve: see the `handover` module, with `sub_grids`.
//!
//! A change of light settings marks chunks dirty in a queue of its own,
//! whose relights start after chunk tasks, nearest a camera first. A relight
//! solved under older settings is thrown away as it lands and started again,
//! and a chunk task's light solved under them lands and is relit. When the
//! emissive list changed, a relight first finds the chunk's emissive voxels
//! again from its published voxels and voxel edits, and the chunks around it
//! relight if they changed.
//!
//! On WASM, where tasks get no thread of their own and would all run between
//! two frames, relights run on the main thread as they start, in the same
//! order from the same queues, until this frame's relights have taken the
//! world's `wasm_relight_budget`. Urgent relights go first there too, those
//! of chunks holding edited voxels before the rest, so an edit's own light
//! still lands within 2 frames. The rest wait for the next frame.
//!
//! Light entries, the light of hinted air beside loaded solids, relight like
//! chunks. A relight that would go on into hinted air holding no light entry
//! passes on through it to the chunks beside it: see
//! [`hinted`](super::hinted).
//!
//! A sub-grid's chunks relight with the world's, but never from arrivals or
//! ripples: the commit that changes a sub-grid relights every chunk of it
//! whose light can change. See the `sub_grid` module.
//!
//! A chunk that retires relights nothing, unless it had provisional columns.
//! Its neighbours keep the light it gave them, which was the truth while it
//! was loaded, and their next solve reads it as solid. A chunk with provisional
//! columns may have given them direct sky it doesn't have, so its face
//! neighbours relight, reading it as solid.
//!
//! # References
//!
//! - [Luanti's light algorithms](https://github.com/luanti-org/luanti/blob/master/src/voxelalgorithms.cpp):
//!   a block marks the borders its light couldn't cross into a neighbour that
//!   wasn't loaded, and `update_block_border_lighting` lights them again once
//!   the neighbour arrives.
//! - [Starlight's technical details](https://github.com/PaperMC/Starlight/blob/fabric/TECHNICAL_DETAILS.md):
//!   when a chunk is lit, an edge check compares its border with what its
//!   neighbours say it should be.
//! - [tesserae's light propagation](https://github.com/mkappworks-dev/tesserae/pull/3):
//!   compares the light on a shared face to decide whether a neighbour solves
//!   again, which costs far less than the solve.

use std::collections::VecDeque;
use std::marker::PhantomData;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;

use bevy::mesh::MeshTag;
use bevy::platform::collections::{HashMap, HashSet};
use bevy::platform::time::Instant;
use bevy::prelude::*;
use bevy::render::renderer::RenderDevice;
use bevy::tasks::{AsyncComputeTaskPool, Task};
use futures_lite::future;

use super::census::{Cause, ChangeCensus};
use super::coarse::CoarseSolved;
use super::faces::{could_reach_sooner, Face};
use super::fill::open_cells;
use super::hinted::{self, changed_across_air, passes_into_air};
use super::inputs::Rescan;
use super::levels::LevelScale;
use super::levels::LightLevels;
use super::probe::{face_neighbours, SkyProbes};
use super::route::LightRoute;
use super::sky::{padding_row, ColumnAnswer};
use super::slot::FacesSee;
use super::solver::{relight, ChunkLight, KeptLight, Solved};
use super::view::{
    chunk_centre, in_world, world_position, AirInputs, LightView, ViewChunk, ENTRY,
};
use super::{LightSettings, SolvedLight, TaskLight};
use crate::chunk::{ChunkKey, CHUNK_SIZE_U};
use crate::chunk_lifecycle::{ChunkState, ChunkStateOf};
use crate::chunk_map::ChunkMap;
use crate::configuration::VoxelWorldConfig;
use crate::grid::GridKey;
use crate::pool_budget::{first_in_start_order, JobKind, PoolBudget};
use crate::voxel_edits::VoxelEdits;
use crate::voxel_world::VoxelWorldCamera;

/// How long landing light may take on the main thread each frame. Results
/// left over land on the next frame.
const LANDING_BUDGET: Duration = Duration::from_millis(1);

/// A chunk's light state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LightState {
    Clean,
    /// Waiting for a relight to start.
    Dirty,
    /// A relight is running, or its result waits to land. `dirty` turns true
    /// when something triggers another one meanwhile.
    Relighting {
        dirty: bool,
    },
}

struct LightChunk {
    entity: Entity,
    state: LightState,
    /// Whether its next relight starts at once, ahead of the pool budget, as
    /// relights rippling from a voxel edit do.
    urgent: bool,
    /// Why its next relight, or the one running, was started.
    cause: Cause,
}

/// Where a result to land came from.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Source {
    /// A chunk task: the chunk's first solve, or the solve of its new voxels.
    ChunkTask,
    Relight,
}

struct Landing {
    key: GridKey,
    entity: Entity,
    source: Source,
    solved: Solved,
    /// Whether the relights it triggers are urgent.
    urgent: bool,
    /// For a relight, the world's edit frame when it started: see
    /// [`LightSettings::edit_frame`].
    edit_frame: u64,
    /// For a relight, the version of the chunk's light in the world view when
    /// it started.
    version: u64,
    /// For a relight, why it was started.
    cause: Cause,
}

/// A relight that started.
struct Started {
    key: GridKey,
    entity: Entity,
    urgent: bool,
    edit_frame: u64,
    version: u64,
    cause: Cause,
}

/// A relight running on the task pool.
struct Running {
    task: Task<Option<Solved>>,
    relight: Started,
}

/// What a chunk retiring leaves for light where it was.
enum Left {
    Nothing,
    /// A light entry, keeping the chunk's light.
    Entry(Arc<ChunkLight>),
    /// Hinted air holding no light entry.
    Air,
}

/// A chunk's light that changed as it landed, for the shader.
struct LandedLight {
    key: GridKey,
    entity: Entity,
    light: Arc<ChunkLight>,
    own: FacesSee,
    bytes: Option<Box<[u8]>>,
    /// Whether it came from a chunk task, whose voxels may be new.
    new_voxels: bool,
}

/// The light work of world `C` on the main thread: each chunk's light state,
/// the queue of relights to start, the running relight tasks, and the results
/// waiting to land.
///
/// It finds a level-0 chunk or light entry by its grid key, and a coarse
/// chunk by its entity. Sky fill and hinted air are the world's alone, so
/// what it notes for them holds the world's level-0 chunk positions.
#[derive(Resource)]
pub(crate) struct LightWork<C> {
    chunks: HashMap<GridKey, LightChunk>,
    /// The grid key of each level-0 chunk with light, by entity.
    positions: HashMap<Entity, GridKey>,
    /// The dirty chunks, waiting for a relight to start.
    queue: HashSet<GridKey>,
    /// The dirty chunks a settings change marked, waiting for a relight to
    /// start after chunk tasks have started. A chunk is in one queue at most.
    settings_queue: HashSet<GridKey>,
    /// The dirty chunks a sky fill cell's change marked, waiting for a
    /// relight to start after chunk tasks and before the settings queue. A
    /// chunk is in one queue at most.
    fill_queue: HashSet<GridKey>,
    /// Chunks that landed or retired since sky fill last looked, whose cells
    /// sky fill works out again.
    fill_touched: Vec<IVec3>,
    /// Whether the world has sky fill, so `fill_touched` is kept.
    pub tracks_fill: bool,
    running: Vec<Running>,
    landing: VecDeque<Landing>,
    /// Relit light of chunks being re-meshed, held until their new mesh
    /// lands, by grid key.
    held: HashMap<GridKey, Landing>,
    /// The light kept by the chunk tasks of chunks re-meshed after a voxel
    /// edit, waiting to reach the shader with their new mesh.
    kept_landing: Vec<(GridKey, Entity, KeptLight)>,
    /// For each chunk whose chunk task published, the world's edit frame when
    /// that task started, so its mesh shows every edit before then.
    meshed: HashMap<Entity, u64>,
    /// The light coarse chunk tasks solved, waiting to land.
    coarse_landing: Vec<(ChunkKey, Entity, CoarseSolved)>,
    /// The key of each coarse chunk whose light has landed, and the levels
    /// its tag holds without a slot, as in a texel.
    coarse: HashMap<Entity, (ChunkKey, LightLevels)>,
    /// Coarse chunks whose light landed at another precision than the
    /// world's, waiting to be solved again: see
    /// [`solve_coarse_again`](Self::solve_coarse_again).
    coarse_queue: Vec<(ChunkKey, Entity)>,
    /// Coarse chunks being solved again on the task pool.
    coarse_running: Vec<(ChunkKey, Entity, Task<CoarseSolved>)>,
    /// The version the next light to land gets.
    clock: u64,
    /// Main-thread time spent this frame, in nanoseconds.
    frame_nanos: u64,
    /// Main-thread time this frame's voxel edits spent, in nanoseconds,
    /// which landing takes off its budget.
    edit_nanos: u64,
    /// The world view, where relights find the hinted air they pass on
    /// through.
    pub view: LightView,
    /// Chunks that landed, retired or had voxels edited since hinted air was
    /// last looked at, around which light entries may be made or dropped:
    /// see [`hinted::follow`](super::hinted::follow).
    hinted_touched: Vec<IVec3>,
    /// Light entries whose first solve hasn't landed. It changes no inputs,
    /// so it ripples to no neighbour.
    fresh: HashSet<GridKey>,
    /// Counts of changes to shown light, once a bench turns them on.
    pub census: Option<ChangeCensus>,
    /// Chunks holding voxels edited since their last relight started. When
    /// relights run on the main thread, theirs start first.
    edited: HashSet<GridKey>,
    /// Time relights ran on the main thread this frame.
    relight_time: Duration,
    /// The relights handovers wait for, and the chunks whose light they
    /// hold.
    #[cfg(feature = "sub_grids")]
    handovers: super::sub_grid::handover::HandoverWaits,
    _world: PhantomData<C>,
}

impl<C> Default for LightWork<C> {
    fn default() -> Self {
        Self {
            chunks: HashMap::new(),
            positions: HashMap::new(),
            queue: HashSet::new(),
            settings_queue: HashSet::new(),
            fill_queue: HashSet::new(),
            fill_touched: Vec::new(),
            tracks_fill: false,
            running: Vec::new(),
            landing: VecDeque::new(),
            held: HashMap::new(),
            kept_landing: Vec::new(),
            meshed: HashMap::new(),
            coarse_landing: Vec::new(),
            coarse: HashMap::new(),
            coarse_queue: Vec::new(),
            coarse_running: Vec::new(),
            clock: 0,
            frame_nanos: 0,
            edit_nanos: 0,
            view: LightView::default(),
            hinted_touched: Vec::new(),
            fresh: HashSet::new(),
            census: None,
            edited: HashSet::new(),
            relight_time: Duration::ZERO,
            #[cfg(feature = "sub_grids")]
            handovers: Default::default(),
            _world: PhantomData,
        }
    }
}

impl<C: VoxelWorldConfig> LightWork<C> {
    /// Whether no relight is queued or running, and no light waits to land,
    /// for a chunk's mesh or for a handover.
    pub fn is_settled(&self) -> bool {
        #[cfg(feature = "sub_grids")]
        if !self.handovers.is_empty() {
            return false;
        }
        self.queue.is_empty()
            && self.settings_queue.is_empty()
            && self.fill_queue.is_empty()
            && self.running.is_empty()
            && self.landing.is_empty()
            && self.held.is_empty()
            && self.coarse_queue.is_empty()
            && self.coarse_running.is_empty()
    }

    /// Whether every running relight's task has finished, for its light to
    /// be collected on the next frame.
    #[cfg(test)]
    pub(crate) fn tasks_finished(&self) -> bool {
        self.running
            .iter()
            .all(|running| running.task.is_finished())
    }

    /// Queues the light a chunk task solved for the chunk `entity` with
    /// `key`, to land with the chunk's voxels. It replaces any light a chunk
    /// task solved for the chunk that hasn't landed yet.
    pub fn arrive(&mut self, key: GridKey, entity: Entity, solved: SolvedLight) {
        match solved {
            SolvedLight::Level0(solved) => {
                self.meshed.insert(entity, solved.edit_frame);
                self.arrive_level_0(key, entity, solved)
            }
            SolvedLight::Kept(kept) => {
                self.meshed.insert(entity, kept.edit_frame);
                self.kept_landing.retain(|(_, held, _)| *held != entity);
                self.kept_landing.push((key, entity, kept));
            }
            SolvedLight::Coarse(solved) => {
                self.drop_coarse_work(entity);
                self.coarse_landing.push((key.key(), entity, solved));
            }
        }
    }

    /// The levels the coarse chunk `entity`'s tag holds without a slot, as in
    /// a texel, once its light has landed.
    pub fn coarse_levels(&self, entity: Entity) -> Option<LightLevels> {
        self.coarse.get(&entity).map(|(_, levels)| *levels)
    }

    /// Queues every coarse chunk whose light has landed to be solved again
    /// from its published voxels, after a change of precision.
    ///
    /// A coarse chunk otherwise solves only in its own chunk task. Its light
    /// is converted to the new precision at once, but light worked out at 16
    /// levels shows 16 levels until it is solved again. One whose light is
    /// still to land is queued as it lands.
    pub fn solve_coarse_again(&mut self) {
        self.coarse_running.clear();
        self.coarse_queue = self
            .coarse
            .iter()
            .map(|(entity, (key, _))| (*key, *entity))
            .collect();
    }

    /// Drops the light waiting to land for the coarse chunk `entity`, and
    /// its solve queued or running.
    fn drop_coarse_work(&mut self, entity: Entity) {
        self.coarse_landing.retain(|(_, held, _)| *held != entity);
        self.coarse_queue.retain(|(_, queued)| *queued != entity);
        self.coarse_running
            .retain(|(_, running, _)| *running != entity);
    }

    /// Starts solving the queued coarse chunks again, as many as `budget`
    /// leaves room for under the cap `cap`.
    fn start_coarse(
        &mut self,
        settings: &LightSettings<C>,
        budget: &mut PoolBudget<C>,
        cap: usize,
        chunk_map: &ChunkMap<C, C::MaterialIndex>,
        edits: &VoxelEdits<C, C::MaterialIndex>,
        world: &C,
    ) {
        let count = budget
            .slots(JobKind::Relight, cap)
            .min(self.coarse_queue.len());
        let pool = AsyncComputeTaskPool::get();
        for (key, entity) in self.coarse_queue.drain(..count).collect::<Vec<_>>() {
            let Some(data) = chunk_map.get_read_lock().get(&key).cloned() else {
                continue;
            };
            let delegate = world.voxel_lookup_delegate();
            let Some(TaskLight::Coarse(task)) = settings.task_for(
                GridKey::world(key),
                entity,
                data.data_shape,
                world,
                edits,
                Some(delegate),
                false,
            ) else {
                continue;
            };
            let solve = pool.spawn(async move { task.solve(&data) });
            self.coarse_running.push((key, entity, solve));
            budget.started(JobKind::Relight);
        }
    }

    fn arrive_level_0(&mut self, key: GridKey, entity: Entity, solved: Solved) {
        let held = self.landing.iter_mut().find(|landing| {
            landing.entity == entity && landing.source == Source::ChunkTask
        });
        match held {
            Some(held) => held.solved = solved,
            None => self.landing.push_back(Landing {
                key,
                entity,
                source: Source::ChunkTask,
                solved,
                urgent: false,
                edit_frame: 0,
                version: 0,
                cause: Cause::Other,
            }),
        }
    }

    /// Whether the chunk `entity` shows a mesh of its voxels as of the
    /// world's edit frame `edit_frame`: it is `ready`, or the task of the mesh
    /// it shows started no earlier.
    fn shows_mesh_of(&self, entity: Entity, edit_frame: u64, ready: bool) -> bool {
        ready
            || self
                .meshed
                .get(&entity)
                .is_some_and(|meshed| *meshed >= edit_frame)
    }

    /// Whether a handover holds the light of the chunk at `key` until it
    /// lands.
    fn held_for_handover(&self, key: GridKey) -> bool {
        #[cfg(feature = "sub_grids")]
        return self.handovers.holds(key);
        #[cfg(not(feature = "sub_grids"))]
        {
            let _ = key;
            false
        }
    }

    /// Notes for the handovers that a relight of the chunk at `key`, started
    /// at the world's edit frame `edit_frame`, has finished solving.
    fn note_solved(&mut self, key: GridKey, edit_frame: u64) {
        #[cfg(feature = "sub_grids")]
        self.handovers.solved(key, edit_frame);
        #[cfg(not(feature = "sub_grids"))]
        let _ = (key, edit_frame);
    }

    /// Marks every chunk with light for a relight.
    pub fn trigger_all(&mut self) {
        let keys: Vec<GridKey> = self.chunks.keys().copied().collect();
        for key in keys {
            self.trigger(key);
        }
    }

    /// Marks the chunk or light entry at `key` for a relight, if it has
    /// light. Hinted air with no light entry passes the relight on to the
    /// chunks around it.
    pub fn trigger(&mut self, key: GridKey) {
        self.trigger_as(key, false, Cause::Other);
    }

    /// Marks the chunk at `key` for a relight, as
    /// [`trigger`](Self::trigger) does, because its direct sky changed.
    pub fn trigger_column(&mut self, key: GridKey) {
        self.trigger_as(key, false, Cause::Column);
    }

    /// Marks the chunk at `key` for a relight that starts at once, ahead
    /// of the pool budget and whatever its chunk state, if it has light. The
    /// relights its ripple triggers are urgent too.
    pub fn trigger_urgent(&mut self, key: GridKey) {
        self.trigger_as(key, true, Cause::Other);
    }

    /// Marks the chunk at `key`, which holds edited voxels, for an
    /// urgent relight, as [`trigger_urgent`](Self::trigger_urgent) does. When
    /// relights run on the main thread, its relight starts before other
    /// urgent ones, so its light lands within 2 frames however many chunks
    /// the edit relights.
    pub fn trigger_edited(&mut self, key: GridKey) {
        // Only a chunk with light leaves `edited` when its relight starts.
        if self.chunks.contains_key(&key) {
            self.edited.insert(key);
        }
        self.trigger_urgent(key);
    }

    fn trigger_as(&mut self, key: GridKey, urgent: bool, cause: Cause) {
        self.trigger_from(key, key, urgent, cause);
    }

    /// Marks the chunk or light entry at `key` for a relight, as
    /// [`trigger_as`](Self::trigger_as) does. Where `key` is hinted air
    /// holding no light entry, the relight passes on through it instead,
    /// within the 3×3×3 around `origin`: see [`pass_on`](Self::pass_on).
    fn trigger_from(
        &mut self,
        key: GridKey,
        origin: GridKey,
        urgent: bool,
        cause: Cause,
    ) {
        let Some(chunk) = self.chunks.get_mut(&key) else {
            if self.view.read().is_air(key) {
                self.pass_on(key, origin, urgent, cause);
            }
            return;
        };
        #[cfg(feature = "sub_grids")]
        if chunk.entity != ENTRY {
            self.handovers.triggered(key);
        }
        chunk.urgent |= urgent;
        // The first trigger since the last relight started names its cause.
        if matches!(
            chunk.state,
            LightState::Clean | LightState::Relighting { dirty: false }
        ) {
            chunk.cause = cause;
        }
        chunk.state = match chunk.state {
            LightState::Clean | LightState::Dirty => {
                self.settings_queue.remove(&key);
                self.fill_queue.remove(&key);
                self.queue.insert(key);
                LightState::Dirty
            }
            LightState::Relighting { .. } => LightState::Relighting { dirty: true },
        };
    }

    /// Passes a relight on through the hinted air at `air`, which holds no
    /// light entry: the chunks and light entries beside it relight instead,
    /// and beside any such air it touches, within the 3×3×3 around `origin`,
    /// where the change that started the relight is.
    ///
    /// Light crosses hinted air, but such air holds no light to compare, so a
    /// relight can't stop there. A change at `origin` can't reach a chunk
    /// past the 3×3×3 around it, whose boxes reach no further. The relights
    /// it starts are for `cause`.
    fn pass_on(&mut self, air: GridKey, origin: GridKey, urgent: bool, cause: Cause) {
        let within =
            |key: GridKey| (key.position() - origin.position()).abs().max_element() <= 1;
        let mut relight = Vec::new();
        {
            let view = self.view.read();
            let mut seen = HashSet::from([air]);
            let mut next = vec![air];
            while let Some(key) = next.pop() {
                for face in Face::ALL {
                    let beside = key + face.offset();
                    if !within(beside) || !seen.insert(beside) {
                        continue;
                    }
                    if view.is_air(beside) {
                        next.push(beside);
                    } else if self.chunks.contains_key(&beside) {
                        relight.push(beside);
                    }
                }
            }
        }
        for key in relight {
            self.trigger_as(key, urgent, cause);
        }
    }

    /// Adds a light entry at `key`, which the world view now holds, and
    /// queues its first solve.
    pub fn add_entry(&mut self, key: GridKey) {
        self.chunks.insert(
            key,
            LightChunk {
                entity: ENTRY,
                state: LightState::Clean,
                urgent: false,
                cause: Cause::Other,
            },
        );
        self.fresh.insert(key);
        self.fill_touch(key);
        self.trigger(key);
    }

    /// Forgets the light entry at `key`, which the world view no longer
    /// holds.
    pub fn forget_entry(&mut self, key: GridKey) {
        if self
            .chunks
            .get(&key)
            .is_some_and(|chunk| chunk.entity == ENTRY)
        {
            self.chunks.remove(&key);
            self.queue.remove(&key);
            self.settings_queue.remove(&key);
            self.fill_queue.remove(&key);
            self.held.remove(&key);
            self.fresh.remove(&key);
            self.fill_touch(key);
        }
    }

    /// Adds the sub-grid chunk `entity` at `key`, whose inputs the world view
    /// holds, and queues its first solve as a relight that starts at once,
    /// ahead of the pool budget.
    #[cfg(feature = "sub_grids")]
    pub fn add_sub_grid_chunk(&mut self, key: GridKey, entity: Entity) {
        if let Some(old) = self.chunks.get(&key).map(|chunk| chunk.entity) {
            self.positions.remove(&old);
        }
        self.chunks.insert(
            key,
            LightChunk {
                entity,
                state: LightState::Clean,
                urgent: false,
                cause: Cause::Other,
            },
        );
        self.positions.insert(entity, key);
        self.trigger_edited(key);
    }

    /// Notes the relights triggered from now on as those of a commit landing
    /// as a handover, taken in at the world's edit frame `since`, or stops
    /// noting them with `None`. See the `handover` module.
    #[cfg(feature = "sub_grids")]
    pub fn capture_handover(&mut self, since: Option<u64>) {
        self.handovers.capture(since);
    }

    /// Notes the first solve of the new sub-grid chunk at `key` as a relight
    /// of the handover's commit, if one is being taken in.
    #[cfg(feature = "sub_grids")]
    pub fn capture_new_chunk(&mut self, key: GridKey) {
        self.handovers.triggered(key);
    }

    /// The chunks whose relights the commits taken in since this was last
    /// called noted for their handovers.
    #[cfg(feature = "sub_grids")]
    pub fn take_handover_relights(&mut self) -> Vec<GridKey> {
        self.handovers.take_captured()
    }

    /// Holds the light landing for the chunk at `key` until
    /// [`release_for_handover`](Self::release_for_handover), for a chunk a
    /// handover holds whose relight it doesn't wait for.
    #[cfg(feature = "sub_grids")]
    pub fn hold_for_handover(&mut self, key: GridKey) {
        self.handovers.hold(key);
    }

    /// Lets the light of the chunk at `key` land, once its handover has
    /// landed or no longer holds it. Light held for it lands in this frame's
    /// [`Relights::land`], with the chunk's new mesh if it has one.
    ///
    /// The relight ripple from the light held for it starts now.
    #[cfg(feature = "sub_grids")]
    pub fn release_for_handover(&mut self, key: GridKey) {
        for (target, urgent, cause) in self.handovers.release(key) {
            self.trigger_from(target, key, urgent, cause);
        }
    }

    /// Whether every relight a handover waits for among the chunks at `keys`
    /// has solved. A chunk with no light of its own any more, or only a
    /// light entry's, waits for nothing.
    #[cfg(feature = "sub_grids")]
    pub fn handover_solved(&self, keys: &[GridKey]) -> bool {
        let relights = |key: GridKey| {
            self.chunks
                .get(&key)
                .is_some_and(|chunk| chunk.entity != ENTRY)
        };
        !keys
            .iter()
            .any(|key| self.handovers.waits_for(*key, relights))
    }

    /// Whether the light entry at `key` is still there, so light solved
    /// for it lands.
    fn has_entry(&self, key: GridKey) -> bool {
        self.chunks
            .get(&key)
            .is_some_and(|chunk| chunk.entity == ENTRY)
    }

    /// The version the next light to land in the world view gets.
    pub fn next_version(&mut self) -> u64 {
        self.clock += 1;
        self.clock - 1
    }

    /// The positions of the world's chunks that landed, retired or had voxels
    /// edited since this was last called, around which hinted air changes.
    pub fn take_hinted_touched(&mut self) -> Vec<IVec3> {
        std::mem::take(&mut self.hinted_touched)
    }

    /// Notes that the voxels of the chunk at `key` changed, so light
    /// entries around it may be made or dropped. Only the world has hinted
    /// air, so another grid's chunk notes nothing.
    pub fn hinted_touch(&mut self, key: GridKey) {
        if let Some(position) = world_position(&key) {
            self.hinted_touched.push(position);
        }
    }

    /// Marks the chunk at `key` for a relight after a settings change,
    /// if it has light. The relight starts after chunk tasks, unless
    /// something else triggers it meanwhile. A chunk already dirty or
    /// relighting needs nothing more: a relight running now solves under the
    /// old settings, so it is thrown away and started again as it lands.
    pub fn trigger_settings(&mut self, key: GridKey) {
        let Some(chunk) = self.chunks.get_mut(&key) else {
            return;
        };
        if chunk.state == LightState::Clean {
            chunk.state = LightState::Dirty;
            self.settings_queue.insert(key);
        }
    }

    /// Marks the chunk at `key` for a relight with the settings relights, as
    /// [`trigger_settings`](Self::trigger_settings) does, after a change that
    /// leaves the settings' generation alone, such as switching
    /// `sub_grid_sky`. A relight running now read what the change replaced,
    /// so the chunk relights again once it lands.
    #[cfg(feature = "sub_grids")]
    pub fn trigger_again(&mut self, key: GridKey) {
        let Some(chunk) = self.chunks.get_mut(&key) else {
            return;
        };
        match chunk.state {
            LightState::Clean => {
                chunk.state = LightState::Dirty;
                self.settings_queue.insert(key);
            }
            LightState::Dirty => {}
            LightState::Relighting { .. } => {
                chunk.state = LightState::Relighting { dirty: true };
            }
        }
    }

    /// Marks the chunk at `key` for a relight because a sky fill cell it
    /// reads changed, if it has light. The relight starts after chunk tasks,
    /// unless something else triggers it meanwhile. A chunk already dirty
    /// needs nothing more, and a chunk relighting is marked to relight again.
    pub fn trigger_fill(&mut self, key: GridKey) {
        let Some(chunk) = self.chunks.get_mut(&key) else {
            return;
        };
        match chunk.state {
            LightState::Clean => {
                chunk.state = LightState::Dirty;
                self.fill_queue.insert(key);
            }
            LightState::Dirty => {}
            LightState::Relighting { .. } => {
                chunk.state = LightState::Relighting { dirty: true };
            }
        }
    }

    /// Notes that the chunk at `key` landed, retired or had its column
    /// answer changed, for sky fill to work out its cells again. Only the
    /// world has sky fill, so another grid's chunk notes nothing.
    pub fn fill_touch(&mut self, key: GridKey) {
        if let (true, Some(position)) = (self.tracks_fill, world_position(&key)) {
            self.fill_touched.push(position);
        }
    }

    /// The positions of the world's chunks noted with
    /// [`fill_touch`](Self::fill_touch) since this was last called.
    pub fn take_fill_touched(&mut self) -> Vec<IVec3> {
        std::mem::take(&mut self.fill_touched)
    }

    /// Ends the chunk's relight, starting another if one was triggered
    /// meanwhile.
    fn relight_done(&mut self, key: GridKey, entity: Entity) {
        let Some(chunk) = self.chunks.get_mut(&key) else {
            return;
        };
        if chunk.entity != entity {
            return;
        }
        if let LightState::Relighting { dirty } = chunk.state {
            chunk.state = LightState::Clean;
            if dirty {
                let cause = chunk.cause;
                self.trigger_as(key, false, cause);
            }
        }
    }

    /// Lands one result: writes its light into the world view and triggers
    /// the relights it calls for. `scales` are the levels of voxel light and,
    /// with sky light on, of sky light.
    ///
    /// Returns the chunk's light if it changed, with the bytes of its slot,
    /// for the shader.
    ///
    /// A relight of a chunk being re-meshed, which isn't `ready` and shows a
    /// mesh older than the edits the relight read, triggers its ripple at
    /// once but holds its light until a new enough mesh lands: see
    /// [`hold`](Self::hold).
    ///
    /// A chunk task's light solved under settings older than `generation`
    /// lands, and the chunk is marked for the settings relight.
    ///
    /// A sub-grid's chunk relights none of its neighbours as its light
    /// lands, except where its emissive voxels were found again: see
    /// [`passes_light_on`].
    fn land(
        &mut self,
        landing: Landing,
        view: &LightView,
        scales: (&LevelScale, Option<&LevelScale>),
        ready: bool,
        generation: u64,
    ) -> Option<LandedLight> {
        let mut landing = landing;
        if landing.source == Source::Relight {
            rebase(&mut landing, view);
        }
        if landing.source == Source::Relight
            && (!self.shows_mesh_of(landing.entity, landing.edit_frame, ready)
                || self.held_for_handover(landing.key))
        {
            self.hold(landing, view);
            return None;
        }
        let Landing {
            key,
            entity,
            source,
            mut solved,
            urgent,
            cause,
            ..
        } = landing;
        let old_settings = solved.generation != generation;
        let landed = LandedLight {
            key,
            entity,
            light: solved.light.clone(),
            own: solved.own,
            bytes: solved.bytes.take(),
            new_voxels: source == Source::ChunkTask,
        };
        let mut triggers = Vec::new();
        // Hinted air beside an arriving chunk that its light passes on into.
        let mut passes = Vec::new();
        // For the census, the light this landing replaces, and whether the
        // chunk arrived with it.
        let mut replaced = None;
        let mut arrived = false;
        {
            let mut view = view.write();
            match source {
                Source::Relight => {
                    let chunk = view.chunks.get_mut(&key)?;
                    // A chunk task landed new voxels since the relight
                    // started, and solved them from a newer view.
                    if chunk.entity != entity
                        || !Arc::ptr_eq(&chunk.light.inputs, solved.read_inputs())
                    {
                        return None;
                    }
                    // A light entry's first solve changes no inputs a
                    // neighbour reads, so it ripples nowhere.
                    let fresh = self.fresh.remove(&key);
                    if solved.rescanned.is_none() && chunk.light.same_light(&solved.light)
                    {
                        // The light already there is the light these seeds
                        // give.
                        if chunk.solved_from_seeds().is_some() {
                            chunk.fill_seeds = solved
                                .fill_seeds
                                .map(|hash| (Arc::downgrade(&chunk.light), hash));
                        }
                        chunk.faces = solved.faces;
                        self.held.remove(&key);
                        return None;
                    }
                    for face in Face::ALL {
                        if !fresh
                            && passes_light_on(key)
                            && chunk.faces.differs_at(&solved.faces, face)
                        {
                            triggers.push(key + face.offset());
                        }
                    }
                    if let Some(read) = &solved.rescanned {
                        chunk.emission = solved.emission;
                        if read.emissive != solved.light.inputs.emissive {
                            triggers.extend(around(key));
                        }
                    }
                    chunk.fill_seeds = solved
                        .fill_seeds
                        .map(|hash| (Arc::downgrade(&solved.light), hash));
                    let old = std::mem::replace(&mut chunk.light, solved.light);
                    if self.census.is_some() && chunk.entity != ENTRY {
                        replaced = Some(old);
                    }
                    chunk.faces = solved.faces;
                    chunk.version = self.clock;
                    self.clock += 1;
                }
                Source::ChunkTask => {
                    let new = ViewChunk {
                        entity,
                        light: solved.light.clone(),
                        faces: solved.faces,
                        version: self.clock,
                        edits: solved.own_edits,
                        emission: solved.emission,
                        fill_seeds: solved
                            .fill_seeds
                            .map(|hash| (Arc::downgrade(&solved.light), hash)),
                    };
                    self.clock += 1;
                    let old = view
                        .chunks
                        .insert(key, new)
                        .filter(|old| old.entity == entity);
                    if self.census.is_some() {
                        match &old {
                            Some(old) => replaced = Some(old.light.clone()),
                            None => arrived = true,
                        }
                    }
                    view.air.remove(&key);
                    if changed_across_air(&view, key, &solved.read) {
                        triggers.push(key);
                    }
                    for face in Face::ALL.into_iter().filter(|_| passes_light_on(key)) {
                        let beside = key + face.offset();
                        if view.is_air(beside) {
                            if passes_into_air(
                                &solved.light,
                                &view.air[&beside].inputs,
                                face,
                            ) {
                                passes.push(beside);
                            }
                            continue;
                        }
                        let Some(neighbour) = view.chunks.get(&beside) else {
                            continue;
                        };
                        let index = face as usize;
                        let reaches = if solved.seen[index] == Some(neighbour.version) {
                            solved.reaches[index]
                        } else {
                            // The neighbour's light changed since the task
                            // read it, and its own landing couldn't compare
                            // with this chunk's light, which hadn't landed.
                            if could_reach_sooner(
                                &neighbour.light,
                                &solved.light,
                                face.opposite(),
                                scales.0,
                                scales.1,
                            ) {
                                triggers.push(key);
                            }
                            could_reach_sooner(
                                &solved.light,
                                &neighbour.light,
                                face,
                                scales.0,
                                scales.1,
                            )
                        };
                        let rippled = old
                            .as_ref()
                            .is_some_and(|old| old.faces.differs_at(&solved.faces, face));
                        if reaches || rippled {
                            triggers.push(key + face.offset());
                        }
                    }
                }
            }
        }

        if let Some(census) = &mut self.census {
            if replaced.is_some() || arrived {
                census.landed(
                    key,
                    replaced.as_deref(),
                    &landed.light,
                    cause,
                    &view.read(),
                );
            }
        }

        if source == Source::ChunkTask {
            // Light held for the chunk's old inputs is stale.
            self.held.remove(&key);
            if let Some(chunk) = self.chunks.get_mut(&key) {
                if chunk.entity != entity {
                    self.positions.remove(&chunk.entity);
                    self.fresh.remove(&key);
                    self.queue.remove(&key);
                    self.settings_queue.remove(&key);
                    self.fill_queue.remove(&key);
                    self.chunks.remove(&key);
                } else if let LightState::Relighting { .. } = chunk.state {
                    // The running relight read the old inputs, so it is
                    // thrown away as it lands, and another must follow.
                    chunk.state = LightState::Relighting { dirty: true };
                }
            }
            self.chunks.entry(key).or_insert(LightChunk {
                entity,
                state: LightState::Clean,
                urgent: false,
                cause: Cause::Other,
            });
            self.positions.insert(entity, key);
            self.fill_touch(key);
            self.hinted_touch(key);
        }
        // An arrival's light starts relights for its neighbours; a relight's
        // ripple goes on for the reason the relight started.
        let cause = match source {
            Source::ChunkTask => Cause::Arrival,
            Source::Relight => cause,
        };
        for target in triggers {
            self.trigger_from(target, key, urgent, cause);
        }
        for air in passes {
            self.pass_on(air, key, urgent, cause);
        }
        if source == Source::ChunkTask && old_settings {
            self.trigger_settings(key);
        }
        Some(landed)
    }

    /// Whether `landing` is a chunk task's light older than the inputs the
    /// world view holds for its chunk: the task started before a voxel edit
    /// the view has taken in since.
    fn is_stale_task(&self, landing: &Landing, view: &LightView) -> bool {
        landing.source == Source::ChunkTask
            && view.read().chunks.get(&landing.key).is_some_and(|chunk| {
                chunk.entity == landing.entity && chunk.edits > landing.solved.own_edits
            })
    }

    /// Holds a relight's light for a chunk being re-meshed, so its old mesh
    /// never shows light worked out for its new voxels. Its ripple goes on at
    /// once: the neighbours across faces whose hashes changed relight, and
    /// the world view keeps the new hashes. A newer held light replaces an
    /// older one.
    fn hold(&mut self, landing: Landing, view: &LightView) {
        let key = landing.key;
        let mut triggers = Vec::new();
        let same = {
            let mut view = view.write();
            let Some(chunk) = view.chunks.get_mut(&key) else {
                return;
            };
            if chunk.entity != landing.entity
                || !Arc::ptr_eq(&chunk.light.inputs, landing.solved.read_inputs())
            {
                return;
            }
            for face in Face::ALL {
                if passes_light_on(key)
                    && chunk.faces.differs_at(&landing.solved.faces, face)
                {
                    triggers.push(key + face.offset());
                }
            }
            chunk.faces = landing.solved.faces;
            landing.solved.rescanned.is_none()
                && chunk.light.same_light(&landing.solved.light)
        };
        let (urgent, cause) = (landing.urgent, landing.cause);
        if same {
            self.held.remove(&key);
        } else {
            self.held.insert(key, landing);
        }
        // Light a handover holds starts its ripple as the handover lands.
        #[cfg(feature = "sub_grids")]
        if self.handovers.holds(key) {
            let ripples = triggers.into_iter().map(|target| (target, urgent, cause));
            self.handovers.defer(key, ripples);
            return;
        }
        for key in triggers {
            self.trigger_as(key, urgent, cause);
        }
    }

    /// Lands the light held for the chunk at `key` once its new mesh has
    /// landed. Returns it for the shader, unless the chunk's inputs changed
    /// meanwhile.
    fn release(&mut self, key: GridKey, view: &LightView) -> Option<LandedLight> {
        let mut landing = self.held.remove(&key)?;
        rebase(&mut landing, view);
        let Landing {
            entity, mut solved, ..
        } = landing;
        let emissive_changed = {
            let mut view = view.write();
            let chunk = view.chunks.get_mut(&key)?;
            if chunk.entity != entity
                || !Arc::ptr_eq(&chunk.light.inputs, solved.read_inputs())
            {
                return None;
            }
            let emissive_changed = solved.rescanned.as_ref().is_some_and(|read| {
                chunk.emission = solved.emission;
                read.emissive != solved.light.inputs.emissive
            });
            chunk.light = solved.light.clone();
            chunk.version = self.clock;
            self.clock += 1;
            emissive_changed
        };
        if emissive_changed {
            for neighbour in around(key) {
                self.trigger(neighbour);
            }
        }
        Some(LandedLight {
            key,
            entity,
            light: solved.light,
            own: solved.own,
            bytes: solved.bytes.take(),
            new_voxels: true,
        })
    }

    /// Forgets the despawned chunk `entity`: its light state, its queued
    /// relight and its entry in the world view. Relights nothing. Returns
    /// where the chunk was, if it had light, and what it left there.
    ///
    /// A world chunk in hinted air, as `hinted_air` tells of its position,
    /// leaves its voxels for the solves around it, which are the hint's: a
    /// light entry with its light where a loaded chunk's solid touches it,
    /// otherwise its inputs as hinted air.
    fn forget(
        &mut self,
        entity: Entity,
        view: &LightView,
        hinted_air: impl Fn(IVec3) -> bool,
    ) -> Option<(GridKey, Left)> {
        self.landing.retain(|landing| landing.entity != entity);
        self.held.retain(|_, landing| landing.entity != entity);
        self.kept_landing.retain(|(_, held, _)| *held != entity);
        self.meshed.remove(&entity);
        self.drop_coarse_work(entity);
        self.coarse.remove(&entity);
        let key = self.positions.remove(&entity)?;
        if let Some(census) = &mut self.census {
            census.forget(key);
        }
        // A chunk that despawned drops out of the handover holding it, and
        // relights nothing.
        #[cfg(feature = "sub_grids")]
        self.handovers.release(key);
        let mut clean = true;
        if self
            .chunks
            .get(&key)
            .is_some_and(|chunk| chunk.entity == entity)
        {
            clean = self.chunks.remove(&key).unwrap().state == LightState::Clean;
            self.queue.remove(&key);
            self.settings_queue.remove(&key);
            self.fill_queue.remove(&key);
            self.edited.remove(&key);
        }
        self.fill_touch(key);
        self.hinted_touch(key);
        let mut view = view.write();
        if !view
            .chunks
            .get(&key)
            .is_some_and(|chunk| chunk.entity == entity)
        {
            return Some((key, Left::Nothing));
        }
        // Only the world has hinted air.
        let Some(position) =
            world_position(&key).filter(|position| hinted_air(*position))
        else {
            view.chunks.remove(&key);
            return Some((key, Left::Nothing));
        };
        if hinted::touched(&view, position) {
            let chunk = view.chunks.get_mut(&key).unwrap();
            chunk.entity = ENTRY;
            let light = chunk.light.clone();
            self.chunks.insert(
                key,
                LightChunk {
                    entity: ENTRY,
                    state: LightState::Clean,
                    urgent: false,
                    cause: Cause::Other,
                },
            );
            // Its relight was cut short with the chunk.
            if !clean {
                self.trigger(key);
            }
            return Some((key, Left::Entry(light)));
        }
        let chunk = view.chunks.remove(&key).unwrap();
        view.air.insert(
            key,
            AirInputs {
                inputs: chunk.light.inputs.clone(),
                edits: chunk.edits,
            },
        );
        Some((key, Left::Air))
    }

    /// Starts relights for the Ready chunks of `queue`, nearest a camera
    /// first, as many as `budget` leaves room for under the cap `cap`. Each
    /// chunk's middle is placed by its grid's pose in `poses`.
    #[allow(clippy::too_many_arguments)]
    fn start_nearest(
        &mut self,
        queue: Queue,
        settings: &LightSettings<C>,
        budget: &mut PoolBudget<C>,
        cap: usize,
        rescans: &Rescans<'_, C>,
        states: &Query<&ChunkStateOf<C>>,
        cameras: &Query<&GlobalTransform, With<VoxelWorldCamera<C>>>,
        poses: &Poses<'_, C>,
    ) {
        let keys = match queue {
            Queue::Arrivals => &self.queue,
            Queue::Fill => &self.fill_queue,
            Queue::Settings => &self.settings_queue,
        };
        if keys.is_empty() {
            return;
        }
        let slots = if settings.main_thread {
            keys.len()
        } else {
            budget.slots(JobKind::Relight, cap)
        };
        if slots == 0 || !self.has_relight_time(settings) {
            return;
        }
        let cameras: Vec<Vec3> =
            cameras.iter().map(|camera| camera.translation()).collect();
        let candidates: Vec<(u32, GridKey)> = keys
            .iter()
            .filter(|key| {
                self.chunks.get(*key).is_some_and(|chunk| {
                    chunk.entity == ENTRY
                        || matches!(states.get(chunk.entity), Ok(ChunkState::Ready))
                })
            })
            .map(|key| {
                let distance = distance_to_nearest(*key, &poses.pose_of(*key), &cameras);
                (distance, *key)
            })
            .collect();
        for key in first_in_start_order(candidates, slots) {
            if !self.has_relight_time(settings) {
                break;
            }
            let fill_only = matches!(queue, Queue::Fill);
            self.start_relight(key, settings, budget, rescans, fill_only);
        }
    }

    /// Whether another relight may start this frame: always when relights
    /// run on the task pool, and while this frame's relights have taken less
    /// than the world's `wasm_relight_budget` when they run on the main
    /// thread. The last one to start may take the frame over the budget.
    fn has_relight_time(&self, settings: &LightSettings<C>) -> bool {
        !settings.main_thread || self.relight_time < settings.config.wasm_relight_budget
    }

    /// Queues the light of a relight that finished to land, urgent light
    /// ahead of the rest, or ends the relight if it stopped before solving.
    fn finished(&mut self, relight: Started, solved: Option<Solved>) {
        let Started {
            key,
            entity,
            urgent,
            edit_frame,
            version,
            cause,
        } = relight;
        let Some(solved) = solved else {
            self.note_solved(key, edit_frame);
            self.relight_done(key, entity);
            return;
        };
        let landing = Landing {
            key,
            entity,
            source: Source::Relight,
            solved,
            urgent,
            edit_frame,
            version,
            cause,
        };
        if urgent {
            self.landing.push_front(landing);
        } else {
            self.landing.push_back(landing);
        }
    }

    fn add_frame_time(&mut self, started: Instant) {
        self.frame_nanos += started.elapsed().as_nanos() as u64;
    }

    /// Adds the main-thread time this frame's voxel edits took since
    /// `started`, which landing takes off its budget.
    pub fn add_edit_time(&mut self, started: Instant) {
        let nanos = started.elapsed().as_nanos() as u64;
        self.edit_nanos += nanos;
        self.frame_nanos += nanos;
    }

    /// Starts a relight task for the dirty chunk at `key`.
    ///
    /// A chunk whose emissive voxels were found under an older emissive list
    /// has them found again in the task, from its published data in
    /// `rescans` and its voxel edits.
    ///
    /// A relight that is `fill_only`, started only because sky fill cells
    /// changed, stops before solving when the chunk's seeds are the ones its
    /// light was solved from.
    fn start_relight(
        &mut self,
        key: GridKey,
        settings: &LightSettings<C>,
        budget: &mut PoolBudget<C>,
        rescans: &Rescans<'_, C>,
        fill_only: bool,
    ) {
        self.queue.remove(&key);
        self.settings_queue.remove(&key);
        self.fill_queue.remove(&key);
        self.edited.remove(&key);
        let chunk = self.chunks.get_mut(&key).unwrap();
        chunk.state = LightState::Relighting { dirty: false };
        let urgent = std::mem::take(&mut chunk.urgent);
        let (entity, cause) = (chunk.entity, chunk.cause);
        let edit_frame = settings.edit_frame.load(Ordering::Relaxed);
        let (version, emission, solved_from) = settings
            .view
            .read()
            .chunks
            .get(&key)
            .map_or((0, 0, None), |chunk| {
                (chunk.version, chunk.emission, chunk.solved_from_seeds())
            });
        let unless_seeds = solved_from.filter(|_| fill_only);
        let rescan = (emission != settings.task.emission_generation)
            .then(|| rescans.of(key))
            .flatten();
        let task_settings = settings.task.clone();
        let view = settings.view.clone();
        let solve = move || {
            #[cfg(feature = "trace")]
            let _span = bevy::log::info_span!("relight_task").entered();
            let started = Instant::now();
            let solved = relight(&task_settings, &view, key, rescan, unless_seeds);
            let nanos = started.elapsed().as_nanos() as u64;
            let counters = &task_settings.counters;
            counters.relight_nanos.fetch_add(nanos, Ordering::Relaxed);
            if entity == ENTRY {
                counters
                    .entry_relight_nanos
                    .fetch_add(nanos, Ordering::Relaxed);
            }
            solved
        };
        let relight = Started {
            key,
            entity,
            urgent,
            edit_frame,
            version,
            cause,
        };
        settings.counters().relights.fetch_add(1, Ordering::Relaxed);
        #[cfg(feature = "sub_grids")]
        if key.grid() != crate::grid::GridId::WORLD {
            settings
                .counters()
                .sub_grid_relights
                .fetch_add(1, Ordering::Relaxed);
        }
        if settings.main_thread {
            let started = Instant::now();
            let solved = solve();
            let took = started.elapsed();
            self.relight_time += took;
            self.frame_nanos += took.as_nanos() as u64;
            let counters = settings.counters();
            counters
                .max_relight_nanos
                .fetch_max(took.as_nanos() as u64, Ordering::Relaxed);
            counters
                .max_relight_frame_nanos
                .fetch_max(self.relight_time.as_nanos() as u64, Ordering::Relaxed);
            self.finished(relight, solved);
        } else {
            let task = AsyncComputeTaskPool::get().spawn(async move { solve() });
            self.running.push(Running { task, relight });
            budget.started(JobKind::Relight);
        }
        if entity == ENTRY {
            settings
                .counters()
                .entry_relights
                .fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// The systems that relight the chunks of world `C` and land their light.
pub(crate) struct Relights<C>(PhantomData<C>);

impl<C: VoxelWorldConfig> Relights<C> {
    /// Forgets the light of despawned chunks. A chunk that retires with
    /// provisional columns relights its face neighbours, which may have read
    /// direct sky from it that it doesn't have.
    pub fn forget_despawned(
        mut removed: RemovedComponents<ChunkStateOf<C>>,
        mut work: ResMut<LightWork<C>>,
        settings: Res<LightSettings<C>>,
        world: Res<C>,
        mut route: Option<ResMut<LightRoute<C>>>,
        mut probes: Option<ResMut<SkyProbes<C>>>,
    ) {
        for entity in removed.read() {
            let left = work.forget(entity, &settings.view, |position| {
                hinted::is_hinted_air(world.as_ref(), position)
            });
            if let Some(route) = route.as_mut() {
                route.forget(entity);
                if let Some((key, Left::Entry(light))) = &left {
                    route.entry_landed(*key, light.clone());
                }
            }
            // Only the world's chunks have sky probes.
            let Some((key, left)) = left else {
                continue;
            };
            let (Some(probes), Some(position)) = (probes.as_mut(), world_position(&key))
            else {
                continue;
            };
            if !matches!(left, Left::Nothing) {
                // Hinted air keeps the columns its chunk was waiting on.
                probes.rekey(position, ENTRY);
            } else if probes.forget(position, entity) {
                for neighbour in face_neighbours(position) {
                    work.trigger_from(in_world(neighbour), key, false, Cause::Column);
                }
            }
        }
    }

    /// Collects finished relight tasks, queueing their light to land.
    pub fn collect(mut work: ResMut<LightWork<C>>, mut budget: ResMut<PoolBudget<C>>) {
        #[cfg(feature = "trace")]
        let _span = bevy::log::info_span!("collect_relights").entered();
        work.frame_nanos = 0;
        work.edit_nanos = 0;
        work.relight_time = Duration::ZERO;
        let mut finished = 0;
        let mut index = 0;
        while index < work.running.len() {
            let running = &mut work.running[index];
            let Some(solved) = future::block_on(future::poll_once(&mut running.task))
            else {
                index += 1;
                continue;
            };
            finished += 1;
            let Running { relight, .. } = work.running.swap_remove(index);
            work.finished(relight, solved);
        }
        let mut index = 0;
        while index < work.coarse_running.len() {
            let task = &mut work.coarse_running[index].2;
            let Some(solved) = future::block_on(future::poll_once(task)) else {
                index += 1;
                continue;
            };
            finished += 1;
            let (key, entity, _) = work.coarse_running.swap_remove(index);
            work.coarse_landing.push((key, entity, solved));
        }
        budget.set_finished_last_frame(JobKind::Relight, finished);
        budget.set_running(
            JobKind::Relight,
            work.running.len() + work.coarse_running.len(),
        );
    }

    /// Lands queued light, oldest first, for as long as the frame's landing
    /// budget allows, and carries the rest over to the next frame.
    ///
    /// In a world showing its light, a chunk whose light changed gets its new
    /// tag and slot as its light lands, and its neighbours' shells follow.
    /// Slots freed meanwhile go to chunks waiting for one.
    ///
    /// With sky light on, a chunk task's column answer is checked against the
    /// sky record as it lands, and the chunk's own mask goes into the record:
    /// see [`SkyProbes::chunk_landed`].
    #[allow(clippy::too_many_arguments)]
    pub fn land(
        mut commands: Commands,
        mut work: ResMut<LightWork<C>>,
        settings: Res<LightSettings<C>>,
        chunks: Query<&ChunkStateOf<C>>,
        mut route: Option<ResMut<LightRoute<C>>>,
        cameras: Query<&GlobalTransform, With<VoxelWorldCamera<C>>>,
        device: Option<Res<RenderDevice>>,
        mut probes: Option<ResMut<SkyProbes<C>>>,
        edits: Res<VoxelEdits<C, C::MaterialIndex>>,
        configuration: Res<C>,
        chunk_map: Res<ChunkMap<C, C::MaterialIndex>>,
        #[cfg(feature = "sub_grids")] sub_grids: Option<
            Res<crate::sub_grid::SubGrids<C>>,
        >,
    ) {
        let refill = route.as_ref().is_some_and(|route| route.wants_refill());
        if work.landing.is_empty()
            && work.coarse_landing.is_empty()
            && work.held.is_empty()
            && work.kept_landing.is_empty()
            && !refill
        {
            return;
        }
        // A light entry has no chunk to wait for.
        let ready = |entity: Entity| {
            entity == ENTRY || matches!(chunks.get(entity), Ok(ChunkState::Ready))
        };
        #[cfg(feature = "trace")]
        let _span = bevy::log::info_span!("land_light").entered();
        let started = Instant::now();
        if let Some(census) = &mut work.census {
            census.next_frame();
        }
        if let Some(route) = route.as_mut() {
            route.set_cameras(cameras.iter().map(|camera| camera.translation()));
            route.set_device(device.as_deref());
            #[cfg(feature = "sub_grids")]
            if let Some(sub_grids) = &sub_grids {
                route.set_poses(sub_grids.live_grids());
            }
        }
        // Coarse chunks' light lands whole every frame: it relights nothing
        // and costs a few map writes.
        for (key, entity, solved) in std::mem::take(&mut work.coarse_landing) {
            if chunks.get(entity).is_err() {
                continue;
            }
            let (sky, voxel) = solved.levels;
            let levels = LightLevels::of_steps(sky, voxel, solved.precision);
            work.coarse.insert(entity, (key, levels));
            if solved.precision != settings.task.precision
                && !work
                    .coarse_queue
                    .iter()
                    .any(|(_, queued)| *queued == entity)
            {
                // Solved before a change of precision.
                work.coarse_queue.push((key, entity));
            }
            if let Some(route) = route.as_mut() {
                route.coarse_landed(
                    entity,
                    key,
                    solved.levels,
                    solved.precision,
                    solved.bytes,
                );
            }
        }
        // Chunks whose new mesh landed this frame get the light held for them,
        // or the light their chunk task kept, whatever the budget, so mesh and
        // light show together.
        let released: Vec<GridKey> = work
            .held
            .iter()
            .filter(|(key, landing)| {
                work.shows_mesh_of(
                    landing.entity,
                    landing.edit_frame,
                    ready(landing.entity),
                ) && !work.held_for_handover(**key)
            })
            .map(|(key, _)| *key)
            .collect();
        for key in released {
            let landed = work.release(key, &settings.view);
            if let (Some(route), Some(landed)) = (route.as_mut(), landed) {
                route.light_landed(
                    landed.key,
                    landed.entity,
                    landed.light,
                    landed.own,
                    landed.bytes,
                    landed.new_voxels,
                );
            }
        }
        for (key, entity, kept) in std::mem::take(&mut work.kept_landing) {
            // Light that changed since the task kept it is landing on its own.
            let current = settings.view.read().chunks.get(&key).is_some_and(|chunk| {
                chunk.entity == entity && Arc::ptr_eq(&chunk.light, &kept.light)
            });
            if let (true, Some(route)) = (current, route.as_mut()) {
                route.light_landed(key, entity, kept.light, kept.own, kept.bytes, true);
            }
        }

        let scales = (&settings.task.voxel_scale, settings.task.sky_scale.as_ref());
        let generation = settings.task.generation;
        let budget = LANDING_BUDGET.saturating_sub(Duration::from_nanos(work.edit_nanos));
        let mut record = probes.is_some().then(|| settings.record.lock().unwrap());
        while let Some(mut landing) = work.landing.pop_front() {
            let (key, entity, source) = (landing.key, landing.entity, landing.source);
            if work.is_stale_task(&landing, &settings.view) {
                continue;
            }
            if source == Source::Relight && landing.solved.generation != generation {
                // Solved under settings changed since: it starts again.
                work.relight_done(key, entity);
                work.trigger(key);
                continue;
            }
            if source == Source::Relight {
                work.note_solved(key, landing.edit_frame);
            }
            let alive = if entity == ENTRY {
                work.has_entry(key)
            } else {
                chunks.get(entity).is_ok()
            };
            if alive {
                // Only the world's chunks have sky probes and a sky record.
                let world = world_position(&key);
                if source == Source::ChunkTask
                    && landing.solved.generation < settings.task.sky_generation
                {
                    // The task worked out its direct sky under the old sky
                    // rule, so it is worked out again under the new one.
                    let padding = chunk_map
                        .get_read_lock()
                        .get_in(&key)
                        .map(padding_row)
                        .or(landing.solved.sky.take().map(|found| found.open))
                        .unwrap_or([!0; CHUNK_SIZE_U as usize]);
                    landing.solved.sky = None;
                    let answer = match (probes.as_mut(), record.as_mut(), world) {
                        (Some(probes), Some(record), Some(position)) => {
                            let own = landing.solved.own_edits;
                            if edits.edit_stamp(position) == own {
                                let occupancy = &landing.solved.light.inputs.occupancy;
                                let open =
                                    record.keeps_cells().then(|| open_cells(occupancy));
                                record.insert(
                                    position,
                                    &occupancy.solid_columns(),
                                    open,
                                    own,
                                );
                            }
                            probes.answer_again(
                                position,
                                entity,
                                padding,
                                record,
                                &edits,
                                &configuration,
                            )
                        }
                        _ => ColumnAnswer::none(),
                    };
                    let light = &landing.solved.light;
                    if answer != light.inputs.columns {
                        // Light on other inputs than its seeds were found for.
                        landing.solved.fill_seeds = None;
                        landing.solved.light = Arc::new(ChunkLight {
                            inputs: Arc::new(light.inputs.with_columns(answer)),
                            sky: light.sky.clone(),
                            voxel: light.voxel.clone(),
                            color: light.color,
                            precision: light.precision,
                        });
                    }
                }
                if let (Some(probes), Some(record), Some(position)) =
                    (probes.as_mut(), record.as_mut(), world)
                {
                    probes.chunk_landed(
                        position,
                        entity,
                        &mut landing.solved,
                        record,
                        &edits,
                        &configuration,
                        &settings.view,
                    );
                }
                let landed =
                    work.land(landing, &settings.view, scales, ready(entity), generation);
                if let Some(probes) = probes.as_mut() {
                    for position in probes.take_relights() {
                        work.trigger_column(in_world(position));
                    }
                }
                if let (Some(route), Some(landed)) = (route.as_mut(), landed) {
                    if landed.entity == ENTRY {
                        route.entry_landed(landed.key, landed.light);
                    } else {
                        route.light_landed(
                            landed.key,
                            landed.entity,
                            landed.light,
                            landed.own,
                            landed.bytes,
                            landed.new_voxels,
                        );
                    }
                }
            }
            if source == Source::Relight {
                work.relight_done(key, entity);
            }
            if started.elapsed() >= budget {
                break;
            }
        }
        if let Some(route) = route.as_mut() {
            route.refill();
            for (entity, tag) in route.take_changed_tags() {
                commands.entity(entity).try_insert(MeshTag::new(tag));
            }
        }
        work.add_frame_time(started);
    }

    /// Starts relights for dirty chunks, nearest a camera first, as many as
    /// the world's [`PoolBudget`] leaves room for. They start before chunk
    /// tasks, which get the room left over.
    ///
    /// A chunk that isn't Ready waits: its chunk task, once it lands, may
    /// call for no relight at all.
    #[allow(clippy::too_many_arguments)]
    pub fn start(
        mut work: ResMut<LightWork<C>>,
        mut budget: ResMut<PoolBudget<C>>,
        settings: Res<LightSettings<C>>,
        states: Query<&ChunkStateOf<C>>,
        cameras: Query<&GlobalTransform, With<VoxelWorldCamera<C>>>,
        configuration: Res<C>,
        chunk_map: Res<ChunkMap<C, C::MaterialIndex>>,
        edits: Res<VoxelEdits<C, C::MaterialIndex>>,
        #[cfg(feature = "sub_grids")] sub_grids: Option<
            Res<crate::sub_grid::SubGrids<C>>,
        >,
    ) {
        if work.queue.is_empty() {
            return;
        }
        #[cfg(feature = "trace")]
        let _span = bevy::log::info_span!("start_relights").entered();
        let started = Instant::now();
        let rescans = Rescans {
            chunk_map: &chunk_map,
            edits: &edits,
        };
        #[cfg(feature = "sub_grids")]
        let poses = Poses::of(sub_grids.as_deref());
        #[cfg(not(feature = "sub_grids"))]
        let poses = Poses::world();
        // Urgent relights go straight to the pool. On the main thread they
        // start within the budget, edited chunks first, then nearest a camera
        // first.
        let camera_positions: Vec<Vec3> =
            cameras.iter().map(|camera| camera.translation()).collect();
        let mut urgent: Vec<(bool, u32, GridKey)> = work
            .queue
            .iter()
            .filter(|key| work.chunks[*key].urgent)
            .map(|key| {
                let later = !work.edited.contains(key);
                let distance =
                    distance_to_nearest(*key, &poses.pose_of(*key), &camera_positions);
                (later, distance, *key)
            })
            .collect();
        urgent.sort_unstable_by_key(|(later, distance, key)| {
            (*later, *distance, key.position().to_array())
        });
        for (_, _, key) in urgent {
            if !work.has_relight_time(&settings) {
                break;
            }
            work.start_relight(key, &settings, &mut budget, &rescans, false);
        }
        let cap = configuration.max_active_chunk_threads();
        work.start_nearest(
            Queue::Arrivals,
            &settings,
            &mut budget,
            cap,
            &rescans,
            &states,
            &cameras,
            &poses,
        );
        work.add_frame_time(started);
    }

    /// Starts relights for the chunks whose sky fill cells changed, nearest
    /// a camera first, with the room in the world's [`PoolBudget`] that chunk
    /// tasks leave.
    #[allow(clippy::too_many_arguments)]
    pub fn start_fill(
        mut work: ResMut<LightWork<C>>,
        mut budget: ResMut<PoolBudget<C>>,
        settings: Res<LightSettings<C>>,
        states: Query<&ChunkStateOf<C>>,
        cameras: Query<&GlobalTransform, With<VoxelWorldCamera<C>>>,
        configuration: Res<C>,
        chunk_map: Res<ChunkMap<C, C::MaterialIndex>>,
        edits: Res<VoxelEdits<C, C::MaterialIndex>>,
    ) {
        if work.fill_queue.is_empty() {
            return;
        }
        #[cfg(feature = "trace")]
        let _span = bevy::log::info_span!("start_fill_relights").entered();
        let started = Instant::now();
        let rescans = Rescans {
            chunk_map: &chunk_map,
            edits: &edits,
        };
        let cap = configuration.max_active_chunk_threads();
        let before = settings.counters().relights.load(Ordering::Relaxed);
        // Only the world has sky fill.
        work.start_nearest(
            Queue::Fill,
            &settings,
            &mut budget,
            cap,
            &rescans,
            &states,
            &cameras,
            &Poses::world(),
        );
        let started_now = settings.counters().relights.load(Ordering::Relaxed) - before;
        settings
            .counters()
            .fill_relights
            .fetch_add(started_now, Ordering::Relaxed);
        work.add_frame_time(started);
    }

    /// Starts relights for the chunks a settings change marked, nearest a
    /// camera first, then solves of the coarse chunks it queued, with the
    /// room in the world's [`PoolBudget`] that chunk tasks leave. Then counts
    /// this frame's main-thread time.
    #[allow(clippy::too_many_arguments)]
    pub fn start_settings(
        mut work: ResMut<LightWork<C>>,
        mut budget: ResMut<PoolBudget<C>>,
        settings: Res<LightSettings<C>>,
        states: Query<&ChunkStateOf<C>>,
        cameras: Query<&GlobalTransform, With<VoxelWorldCamera<C>>>,
        configuration: Res<C>,
        chunk_map: Res<ChunkMap<C, C::MaterialIndex>>,
        edits: Res<VoxelEdits<C, C::MaterialIndex>>,
        #[cfg(feature = "sub_grids")] sub_grids: Option<
            Res<crate::sub_grid::SubGrids<C>>,
        >,
    ) {
        if !work.settings_queue.is_empty() {
            #[cfg(feature = "trace")]
            let _span = bevy::log::info_span!("start_settings_relights").entered();
            let started = Instant::now();
            let rescans = Rescans {
                chunk_map: &chunk_map,
                edits: &edits,
            };
            #[cfg(feature = "sub_grids")]
            let poses = Poses::of(sub_grids.as_deref());
            #[cfg(not(feature = "sub_grids"))]
            let poses = Poses::world();
            let cap = configuration.max_active_chunk_threads();
            work.start_nearest(
                Queue::Settings,
                &settings,
                &mut budget,
                cap,
                &rescans,
                &states,
                &cameras,
                &poses,
            );
            work.add_frame_time(started);
        }
        if !work.coarse_queue.is_empty() {
            let started = Instant::now();
            let cap = configuration.max_active_chunk_threads();
            work.start_coarse(
                &settings,
                &mut budget,
                cap,
                &chunk_map,
                &edits,
                &configuration,
            );
            work.add_frame_time(started);
        }
        if work.frame_nanos > 0 {
            let counters = settings.counters();
            counters
                .main_thread_nanos
                .fetch_add(work.frame_nanos, Ordering::Relaxed);
            counters
                .max_frame_nanos
                .fetch_max(work.frame_nanos, Ordering::Relaxed);
        }
    }
}

/// Which of a world's queues of dirty chunks to start relights from.
#[derive(Clone, Copy)]
enum Queue {
    /// Chunks triggered by light arriving, by voxel edits and by sky probes.
    Arrivals,
    /// Chunks whose sky fill cells changed.
    Fill,
    /// Chunks a settings change marked.
    Settings,
}

/// Where relights find a chunk's voxels to look for its emissive voxels
/// again: its published data and its voxel edits.
struct Rescans<'a, C: VoxelWorldConfig> {
    chunk_map: &'a ChunkMap<C, C::MaterialIndex>,
    edits: &'a VoxelEdits<C, C::MaterialIndex>,
}

impl<C: VoxelWorldConfig> Rescans<'_, C> {
    /// What a relight of the level-0 chunk at `key` reads to find its
    /// emissive voxels again, or `None` if its data isn't published.
    fn of(&self, key: GridKey) -> Option<Rescan<C::MaterialIndex>> {
        let data = self.chunk_map.get_read_lock().get_in(&key).cloned()?;
        let edits = self
            .edits
            .buckets_in(key.grid(), std::iter::once(key.position()))
            .pop()
            .flatten();
        Some(Rescan { data, edits })
    }
}

/// The 26 chunks around the chunk at `key`, on its grid, whose solves read
/// its emissive voxels.
fn around(key: GridKey) -> impl Iterator<Item = GridKey> {
    (-1..=1).flat_map(move |z| {
        (-1..=1).flat_map(move |y| {
            (-1..=1)
                .map(move |x| IVec3::new(x, y, z))
                .filter(|offset| *offset != IVec3::ZERO)
                .map(move |offset| key + offset)
        })
    })
}

/// Whether light landing for the chunk at `key` relights the chunks beside
/// it where it could light them differently: its arrival, and its relight
/// ripple.
///
/// Only a world chunk does. A sub-grid's chunk solves with every chunk of
/// its grid present, and a voxel edit relights every chunk of the grid whose
/// solve box holds an edited voxel (see the `sub_grid` module), so its light
/// has nothing to pass on.
fn passes_light_on(key: GridKey) -> bool {
    key.is_world()
}

/// Moves a relight's light onto the inputs the world view holds for its chunk
/// now, if voxel edits gave the chunk new inputs while the relight ran and no
/// other light has landed for it since. Its light is then a step behind the
/// edits, and the relight they triggered follows; landing it keeps light
/// moving while a chunk is edited every frame.
fn rebase(landing: &mut Landing, view: &LightView) {
    let view = view.read();
    let Some(chunk) = view.chunks.get(&landing.key) else {
        return;
    };
    // Light with emissive voxels found again belongs to the inputs it read,
    // and lands only on those.
    if landing.solved.rescanned.is_some() {
        return;
    }
    let light = &landing.solved.light;
    if chunk.entity != landing.entity
        || chunk.version != landing.version
        || Arc::ptr_eq(&chunk.light.inputs, &light.inputs)
    {
        return;
    }
    // Light on other inputs than its seeds were found for.
    landing.solved.fill_seeds = None;
    landing.solved.light = Arc::new(ChunkLight {
        inputs: chunk.light.inputs.clone(),
        sky: light.sky.clone(),
        voxel: light.voxel.clone(),
        color: light.color,
        precision: light.precision,
    });
}

/// The pose of each grid, which places its chunks for the order relights
/// start in: the identity for the world, and a sub-grid's last pose against
/// the world's root.
///
/// The sub-grids' poses are read once, the first time a sub-grid's chunk
/// asks, so a frame starting only world relights reads none.
struct Poses<'a, C> {
    #[cfg(feature = "sub_grids")]
    sub_grids: Option<&'a crate::sub_grid::SubGrids<C>>,
    #[cfg(feature = "sub_grids")]
    read: std::cell::OnceCell<HashMap<crate::grid::GridId, GlobalTransform>>,
    _world: PhantomData<&'a C>,
}

impl<'a, C: VoxelWorldConfig> Poses<'a, C> {
    /// Poses that place every chunk as a world chunk, for relights of world
    /// chunks alone.
    fn world() -> Self {
        Self {
            #[cfg(feature = "sub_grids")]
            sub_grids: None,
            #[cfg(feature = "sub_grids")]
            read: std::cell::OnceCell::new(),
            _world: PhantomData,
        }
    }

    /// The poses of the world and its live `sub_grids`.
    #[cfg(feature = "sub_grids")]
    fn of(sub_grids: Option<&'a crate::sub_grid::SubGrids<C>>) -> Self {
        Self {
            sub_grids,
            read: std::cell::OnceCell::new(),
            _world: PhantomData,
        }
    }

    /// The pose of the grid of the chunk at `key`. A sub-grid that has ended
    /// reads the identity.
    fn pose_of(&self, key: GridKey) -> GlobalTransform {
        #[cfg(feature = "sub_grids")]
        if !key.is_world() {
            let poses = self.read.get_or_init(|| {
                self.sub_grids.map_or_else(HashMap::new, |sub_grids| {
                    sub_grids
                        .live_grids()
                        .into_iter()
                        .map(|(_, grid, pose)| (grid, pose))
                        .collect()
                })
            });
            return poses
                .get(&key.grid())
                .copied()
                .unwrap_or(GlobalTransform::IDENTITY);
        }
        let _ = key;
        GlobalTransform::IDENTITY
    }
}

/// How far the middle of the level-0 chunk at `key`, placed by its grid's
/// `pose`, is from the nearest of `cameras`, in whole voxels. See
/// [`chunk_centre`].
fn distance_to_nearest(key: GridKey, pose: &GlobalTransform, cameras: &[Vec3]) -> u32 {
    let middle = chunk_centre(key, pose);
    cameras
        .iter()
        .map(|camera| camera.distance(middle))
        .fold(f32::INFINITY, f32::min)
        .min(u32::MAX as f32) as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::configuration::DefaultWorld;

    #[test]
    fn a_chunks_middle_follows_its_grids_pose() {
        let key = in_world(IVec3::new(1, -2, 0));
        let identity = GlobalTransform::IDENTITY;
        assert_eq!(chunk_centre(key, &identity), Vec3::new(48.0, -48.0, 16.0));
        let camera = [Vec3::new(48.0, -48.0, 26.0)];
        assert_eq!(distance_to_nearest(key, &identity, &camera), 10);

        // A quarter turn about y takes x to -z and z to x.
        let pose = GlobalTransform::from(
            Transform::from_xyz(100.0, 0.0, 0.0)
                .with_rotation(Quat::from_rotation_y(std::f32::consts::FRAC_PI_2)),
        );
        let middle = chunk_centre(key, &pose);
        assert!(
            middle.abs_diff_eq(Vec3::new(116.0, -48.0, -48.0), 1e-3),
            "{middle}"
        );
        let camera = [Vec3::new(116.0, -48.0, -37.5)];
        assert_eq!(distance_to_nearest(key, &pose, &camera), 10);
    }

    #[test]
    fn an_edit_where_no_chunk_has_light_marks_nothing_edited() {
        let mut work = LightWork::<DefaultWorld>::default();
        work.trigger_edited(in_world(IVec3::new(3, -1, 2)));

        assert!(work.edited.is_empty());
        assert!(work.is_settled());
    }

    #[cfg(feature = "sub_grids")]
    #[test]
    fn the_chunks_around_a_sub_grid_chunk_are_on_its_grid() {
        use crate::grid::GridId;

        let key = GridKey::level_0(GridId::new(2), IVec3::new(-1, 0, 5));
        let around: Vec<GridKey> = around(key).collect();

        assert_eq!(around.len(), 26);
        for neighbour in around {
            assert_eq!(neighbour.grid(), GridId::new(2));
            let offset = neighbour.position() - key.position();
            assert_eq!(offset.abs().max_element(), 1);
        }
    }
}
