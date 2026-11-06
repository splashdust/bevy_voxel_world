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
//! A chunk that retires relights nothing. Its neighbours keep the light it
//! gave them, which was the truth while it was loaded, and their next solve
//! reads it as solid.

use std::collections::VecDeque;
use std::marker::PhantomData;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{Duration, Instant};

use bevy::mesh::MeshTag;
use bevy::platform::collections::{HashMap, HashSet};
use bevy::prelude::*;
use bevy::tasks::{AsyncComputeTaskPool, Task};
use futures_lite::future;

use super::faces::{could_reach_sooner, Face};
use super::levels::LevelScale;
use super::route::LightRoute;
use super::solver::{relight, ChunkLight, Solved};
use super::view::{LightView, ViewChunk};
use super::LightSettings;
use crate::chunk::CHUNK_SIZE_F;
use crate::chunk_lifecycle::{ChunkState, ChunkStateOf};
use crate::configuration::VoxelWorldConfig;
use crate::pool_budget::{first_in_start_order, JobKind, PoolBudget};
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
    entity: Entity,
    light: Arc<ChunkLight>,
    bytes: Option<Box<[u8]>>,
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

    /// Queues the light a chunk task solved for the chunk `entity` at
    /// `position`, to land with the chunk's voxels. It replaces any light a
    /// chunk task solved for the chunk that hasn't landed yet.
    pub fn arrive(&mut self, position: IVec3, entity: Entity, solved: Solved) {
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

    /// Marks the chunk at `position` for a relight, if it has light.
    fn trigger(&mut self, position: IVec3) {
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
    /// the relights it calls for.
    ///
    /// Returns the chunk's light if it changed, with the bytes of its slot,
    /// for the shader.
    fn land(
        &mut self,
        landing: Landing,
        view: &LightView,
        scale: &LevelScale,
    ) -> Option<LandedLight> {
        let Landing {
            position,
            entity,
            source,
            mut solved,
        } = landing;
        let landed = LandedLight {
            entity,
            light: solved.light.clone(),
            bytes: solved.bytes.take(),
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
                                scale,
                            ) {
                                triggers.push(position);
                            }
                            could_reach_sooner(
                                &solved.light,
                                &neighbour.light,
                                face,
                                scale,
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
    /// relight and its entry in the world view. Relights nothing.
    fn forget(&mut self, entity: Entity, view: &LightView) {
        self.landing.retain(|landing| landing.entity != entity);
        let Some(position) = self.positions.remove(&entity) else {
            return;
        };
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
    }

    fn add_frame_time(&mut self, started: Instant) {
        self.frame_nanos += started.elapsed().as_nanos() as u64;
    }
}

/// The systems that relight the chunks of world `C` and land their light.
pub(crate) struct Relights<C>(PhantomData<C>);

impl<C: VoxelWorldConfig> Relights<C> {
    /// Forgets the light of despawned chunks.
    pub fn forget_despawned(
        mut removed: RemovedComponents<ChunkStateOf<C>>,
        mut work: ResMut<LightWork<C>>,
        settings: Res<LightSettings<C>>,
        mut route: Option<ResMut<LightRoute<C>>>,
    ) {
        for entity in removed.read() {
            work.forget(entity, &settings.view);
            if let Some(route) = route.as_mut() {
                route.forget(entity);
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
    /// tag and slot as its light lands.
    pub fn land(
        mut commands: Commands,
        mut work: ResMut<LightWork<C>>,
        settings: Res<LightSettings<C>>,
        chunks: Query<(), With<ChunkStateOf<C>>>,
        mut route: Option<ResMut<LightRoute<C>>>,
    ) {
        if work.landing.is_empty() {
            return;
        }
        #[cfg(feature = "trace")]
        let _span = bevy::log::info_span!("land_light").entered();
        let started = Instant::now();
        let scale = &settings.task.voxel_scale;
        while let Some(landing) = work.landing.pop_front() {
            let (position, entity, source) =
                (landing.position, landing.entity, landing.source);
            if chunks.contains(entity) {
                let landed = work.land(landing, &settings.view, scale);
                if let (Some(route), Some(landed)) = (route.as_mut(), landed) {
                    let LandedLight {
                        entity,
                        light,
                        bytes,
                    } = landed;
                    if let Some(tag) = route.light_landed(entity, &light, bytes) {
                        commands.entity(entity).try_insert(MeshTag::new(tag));
                    }
                }
            }
            if source == Source::Relight {
                work.relight_done(position, entity);
            }
            if started.elapsed() >= LANDING_BUDGET {
                break;
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
