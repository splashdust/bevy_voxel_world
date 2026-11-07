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
//! A chunk that retires relights nothing, unless it had provisional columns.
//! Its neighbours keep the light it gave them, which was the truth while it
//! was loaded, and their next solve reads it as solid. A chunk with provisional
//! columns may have given them direct sky it doesn't have, so its face
//! neighbours relight, reading it as solid.

use std::collections::VecDeque;
use std::marker::PhantomData;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{Duration, Instant};

use bevy::mesh::MeshTag;
use bevy::platform::collections::{HashMap, HashSet};
use bevy::prelude::*;
use bevy::render::renderer::RenderDevice;
use bevy::tasks::{AsyncComputeTaskPool, Task};
use futures_lite::future;

use super::coarse::CoarseSolved;
use super::faces::{could_reach_sooner, Face};
use super::levels::LevelScale;
use super::probe::{face_neighbours, SkyProbes};
use super::route::LightRoute;
use super::slot::FacesSee;
use super::solver::{relight, ChunkLight, Solved};
use super::view::{LightView, ViewChunk};
use super::{LightSettings, SolvedLight};
use crate::chunk::{ChunkKey, CHUNK_SIZE_F};
use crate::chunk_lifecycle::{ChunkState, ChunkStateOf};
use crate::configuration::VoxelWorldConfig;
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
}

/// Where a result to land came from.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Source {
    /// A chunk task: the chunk's first solve, or the solve of its new voxels.
    ChunkTask,
    Relight,
}

struct Landing {
    position: IVec3,
    entity: Entity,
    source: Source,
    solved: Solved,
}

struct Running {
    position: IVec3,
    entity: Entity,
    task: Task<Option<Solved>>,
}

/// A chunk's light that changed as it landed, for the shader.
struct LandedLight {
    position: IVec3,
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
#[derive(Resource)]
pub(crate) struct LightWork<C> {
    chunks: HashMap<IVec3, LightChunk>,
    positions: HashMap<Entity, IVec3>,
    /// The dirty chunks, waiting for a relight to start.
    queue: HashSet<IVec3>,
    running: Vec<Running>,
    landing: VecDeque<Landing>,
    /// The light coarse chunk tasks solved, waiting to land.
    coarse_landing: Vec<(ChunkKey, Entity, CoarseSolved)>,
    /// The levels each coarse chunk's tag holds without a slot, as in a
    /// texel.
    coarse: HashMap<Entity, u8>,
    /// The version the next light to land gets.
    clock: u64,
    /// Main-thread time spent this frame, in nanoseconds.
    frame_nanos: u64,
    _world: PhantomData<C>,
}

impl<C> Default for LightWork<C> {
    fn default() -> Self {
        Self {
            chunks: HashMap::new(),
            positions: HashMap::new(),
            queue: HashSet::new(),
            running: Vec::new(),
            landing: VecDeque::new(),
            coarse_landing: Vec::new(),
            coarse: HashMap::new(),
            clock: 0,
            frame_nanos: 0,
            _world: PhantomData,
        }
    }
}

impl<C: VoxelWorldConfig> LightWork<C> {
    /// Whether no relight is queued or running, and no light waits to land.
    pub fn is_settled(&self) -> bool {
        self.queue.is_empty() && self.running.is_empty() && self.landing.is_empty()
    }

    /// Queues the light a chunk task solved for the chunk `entity` with
    /// `key`, to land with the chunk's voxels. It replaces any light a chunk
    /// task solved for the chunk that hasn't landed yet.
    pub fn arrive(&mut self, key: ChunkKey, entity: Entity, solved: SolvedLight) {
        match solved {
            SolvedLight::Level0(solved) => {
                self.arrive_level_0(key.position, entity, solved)
            }
            SolvedLight::Coarse(solved) => {
                self.coarse_landing.retain(|(_, held, _)| *held != entity);
                self.coarse_landing.push((key, entity, solved));
            }
        }
    }

    /// The levels the coarse chunk `entity`'s tag holds without a slot, as in
    /// a texel, once its light has landed.
    pub fn coarse_levels(&self, entity: Entity) -> Option<u8> {
        self.coarse.get(&entity).copied()
    }

    fn arrive_level_0(&mut self, position: IVec3, entity: Entity, solved: Solved) {
        let held = self.landing.iter_mut().find(|landing| {
            landing.entity == entity && landing.source == Source::ChunkTask
        });
        match held {
            Some(held) => held.solved = solved,
            None => self.landing.push_back(Landing {
                position,
                entity,
                source: Source::ChunkTask,
                solved,
            }),
        }
    }

    /// Marks every chunk with light for a relight.
    pub fn trigger_all(&mut self) {
        let positions: Vec<IVec3> = self.chunks.keys().copied().collect();
        for position in positions {
            self.trigger(position);
        }
    }

    /// Marks the chunk at `position` for a relight, if it has light.
    pub fn trigger(&mut self, position: IVec3) {
        let Some(chunk) = self.chunks.get_mut(&position) else {
            return;
        };
        chunk.state = match chunk.state {
            LightState::Clean | LightState::Dirty => {
                self.queue.insert(position);
                LightState::Dirty
            }
            LightState::Relighting { .. } => LightState::Relighting { dirty: true },
        };
    }

    /// Ends the chunk's relight, starting another if one was triggered
    /// meanwhile.
    fn relight_done(&mut self, position: IVec3, entity: Entity) {
        let Some(chunk) = self.chunks.get_mut(&position) else {
            return;
        };
        if chunk.entity != entity {
            return;
        }
        if let LightState::Relighting { dirty } = chunk.state {
            chunk.state = LightState::Clean;
            if dirty {
                self.trigger(position);
            }
        }
    }

    /// Lands one result: writes its light into the world view and triggers
    /// the relights it calls for. `scales` are the levels of voxel light and,
    /// with sky light on, of sky light.
    ///
    /// Returns the chunk's light if it changed, with the bytes of its slot,
    /// for the shader.
    fn land(
        &mut self,
        landing: Landing,
        view: &LightView,
        scales: (&LevelScale, Option<&LevelScale>),
    ) -> Option<LandedLight> {
        let Landing {
            position,
            entity,
            source,
            mut solved,
        } = landing;
        let landed = LandedLight {
            position,
            entity,
            light: solved.light.clone(),
            own: solved.own,
            bytes: solved.bytes.take(),
            new_voxels: source == Source::ChunkTask,
        };
        let mut triggers = Vec::new();
        {
            let mut view = view.write();
            match source {
                Source::Relight => {
                    let chunk = view.get_mut(&position)?;
                    // A chunk task landed new voxels since the relight
                    // started, and solved them from a newer view.
                    if chunk.entity != entity
                        || !Arc::ptr_eq(&chunk.light.inputs, &solved.light.inputs)
                    {
                        return None;
                    }
                    if chunk.light.same_levels(&solved.light) {
                        chunk.faces = solved.faces;
                        return None;
                    }
                    for face in Face::ALL {
                        if chunk.faces.differs_at(&solved.faces, face) {
                            triggers.push(position + face.offset());
                        }
                    }
                    chunk.light = solved.light;
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
                    };
                    self.clock += 1;
                    let old = view
                        .insert(position, new)
                        .filter(|old| old.entity == entity);
                    for face in Face::ALL {
                        let Some(neighbour) = view.get(&(position + face.offset()))
                        else {
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
                                triggers.push(position);
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
                            triggers.push(position + face.offset());
                        }
                    }
                }
            }
        }

        if source == Source::ChunkTask {
            if let Some(chunk) = self.chunks.get(&position) {
                if chunk.entity != entity {
                    self.positions.remove(&chunk.entity);
                    self.queue.remove(&position);
                    self.chunks.remove(&position);
                }
            }
            self.chunks.entry(position).or_insert(LightChunk {
                entity,
                state: LightState::Clean,
            });
            self.positions.insert(entity, position);
        }
        for position in triggers {
            self.trigger(position);
        }
        Some(landed)
    }

    /// Forgets the despawned chunk `entity`: its light state, its queued
    /// relight and its entry in the world view. Relights nothing. Returns
    /// where the chunk was, if it had light.
    fn forget(&mut self, entity: Entity, view: &LightView) -> Option<IVec3> {
        self.landing.retain(|landing| landing.entity != entity);
        self.coarse_landing.retain(|(_, held, _)| *held != entity);
        self.coarse.remove(&entity);
        let position = self.positions.remove(&entity)?;
        if self
            .chunks
            .get(&position)
            .is_some_and(|chunk| chunk.entity == entity)
        {
            self.chunks.remove(&position);
            self.queue.remove(&position);
        }
        let mut view = view.write();
        if view
            .get(&position)
            .is_some_and(|chunk| chunk.entity == entity)
        {
            view.remove(&position);
        }
        Some(position)
    }

    fn add_frame_time(&mut self, started: Instant) {
        self.frame_nanos += started.elapsed().as_nanos() as u64;
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
        mut route: Option<ResMut<LightRoute<C>>>,
        mut probes: Option<ResMut<SkyProbes<C>>>,
    ) {
        for entity in removed.read() {
            let position = work.forget(entity, &settings.view);
            if let Some(route) = route.as_mut() {
                route.forget(entity);
            }
            if let (Some(probes), Some(position)) = (probes.as_mut(), position) {
                if probes.forget(position, entity) {
                    for neighbour in face_neighbours(position) {
                        work.trigger(neighbour);
                    }
                }
            }
        }
    }

    /// Collects finished relight tasks, queueing their light to land.
    pub fn collect(mut work: ResMut<LightWork<C>>, mut budget: ResMut<PoolBudget<C>>) {
        #[cfg(feature = "trace")]
        let _span = bevy::log::info_span!("collect_relights").entered();
        work.frame_nanos = 0;
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
            let Running {
                position, entity, ..
            } = work.running.swap_remove(index);
            match solved {
                Some(solved) => work.landing.push_back(Landing {
                    position,
                    entity,
                    source: Source::Relight,
                    solved,
                }),
                None => work.relight_done(position, entity),
            }
        }
        budget.set_finished_last_frame(JobKind::Relight, finished);
        budget.set_running(JobKind::Relight, work.running.len());
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
        chunks: Query<(), With<ChunkStateOf<C>>>,
        mut route: Option<ResMut<LightRoute<C>>>,
        cameras: Query<&GlobalTransform, With<VoxelWorldCamera<C>>>,
        device: Option<Res<RenderDevice>>,
        mut probes: Option<ResMut<SkyProbes<C>>>,
        edits: Res<VoxelEdits<C, C::MaterialIndex>>,
        configuration: Res<C>,
    ) {
        let refill = route.as_ref().is_some_and(|route| route.wants_refill());
        if work.landing.is_empty() && work.coarse_landing.is_empty() && !refill {
            return;
        }
        #[cfg(feature = "trace")]
        let _span = bevy::log::info_span!("land_light").entered();
        let started = Instant::now();
        if let Some(route) = route.as_mut() {
            route.set_cameras(cameras.iter().map(|camera| camera.translation()));
            route.set_device(device.as_deref());
        }
        // Coarse chunks' light lands whole every frame: it relights nothing
        // and costs a few map writes.
        for (key, entity, solved) in std::mem::take(&mut work.coarse_landing) {
            if !chunks.contains(entity) {
                continue;
            }
            work.coarse.insert(entity, solved.levels);
            if let Some(route) = route.as_mut() {
                route.coarse_landed(entity, key, solved.levels, solved.bytes);
            }
        }
        let scales = (&settings.task.voxel_scale, settings.task.sky_scale.as_ref());
        let mut record = probes.is_some().then(|| settings.record.lock().unwrap());
        while let Some(mut landing) = work.landing.pop_front() {
            let (position, entity, source) =
                (landing.position, landing.entity, landing.source);
            if chunks.contains(entity) {
                if let (Some(probes), Some(record)) = (probes.as_mut(), record.as_mut()) {
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
                let landed = work.land(landing, &settings.view, scales);
                if let Some(probes) = probes.as_mut() {
                    for position in probes.take_relights() {
                        work.trigger(position);
                    }
                }
                if let (Some(route), Some(landed)) = (route.as_mut(), landed) {
                    route.light_landed(
                        landed.position,
                        landed.entity,
                        landed.light,
                        landed.own,
                        landed.bytes,
                        landed.new_voxels,
                    );
                }
            }
            if source == Source::Relight {
                work.relight_done(position, entity);
            }
            if started.elapsed() >= LANDING_BUDGET {
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
    pub fn start(
        mut work: ResMut<LightWork<C>>,
        mut budget: ResMut<PoolBudget<C>>,
        settings: Res<LightSettings<C>>,
        states: Query<&ChunkStateOf<C>>,
        cameras: Query<&GlobalTransform, With<VoxelWorldCamera<C>>>,
        configuration: Res<C>,
    ) {
        if !work.queue.is_empty() {
            #[cfg(feature = "trace")]
            let _span = bevy::log::info_span!("start_relights").entered();
            let started = Instant::now();
            let slots =
                budget.slots(JobKind::Relight, configuration.max_active_chunk_threads());
            if slots > 0 {
                let cameras: Vec<Vec3> =
                    cameras.iter().map(|camera| camera.translation()).collect();
                let candidates: Vec<(u32, IVec3)> = work
                    .queue
                    .iter()
                    .filter(|position| {
                        work.chunks.get(*position).is_some_and(|chunk| {
                            matches!(states.get(chunk.entity), Ok(ChunkState::Ready))
                        })
                    })
                    .map(|position| (distance_to_nearest(*position, &cameras), *position))
                    .collect();
                let thread_pool = AsyncComputeTaskPool::get();
                for position in first_in_start_order(candidates, slots) {
                    work.queue.remove(&position);
                    let chunk = work.chunks.get_mut(&position).unwrap();
                    chunk.state = LightState::Relighting { dirty: false };
                    let entity = chunk.entity;
                    let task_settings = settings.task.clone();
                    let view = settings.view.clone();
                    let task = thread_pool.spawn(async move {
                        #[cfg(feature = "trace")]
                        let _span = bevy::log::info_span!("relight_task").entered();
                        relight(&task_settings, &view, position)
                    });
                    work.running.push(Running {
                        position,
                        entity,
                        task,
                    });
                    budget.started(JobKind::Relight);
                    settings.counters().relights.fetch_add(1, Ordering::Relaxed);
                }
            }
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
