///
/// Chunk lifecycle
/// The state of every chunk, the chunk tasks that generate and mesh chunks, and
/// publishing their results. Spawn, retire and LOD decisions and voxel edits are
/// made elsewhere and reach the lifecycle as `ChunkRequests`.
///
use std::{
    collections::VecDeque,
    marker::PhantomData,
    sync::{Arc, RwLockReadGuard},
};

use bevy::{
    platform::collections::{HashMap, HashSet},
    prelude::*,
    tasks::{AsyncComputeTaskPool, Task},
};
use futures_lite::future;

use crate::{
    chunk::{Chunk, ChunkTask, CHUNK_SIZE_F},
    chunk_map::{ChunkMap, ChunkMapData, ChunkMapRemoveBuffer, ChunkMapUpdateBuffer},
    configuration::{LodLevel, VoxelWorldConfig},
    mesh_cache::{MeshCache, MeshCacheInsertBuffer, MeshRef},
    prelude::default_chunk_meshing_delegate,
    streaming_policy::chunk_centre,
    voxel_edits::VoxelEdits,
    voxel_world::{
        ChunkPublished, ChunkWillChangeLod, ChunkWillDespawn, ChunkWillRemesh,
        ChunkWillSpawn, ChunkWillUpdate, VoxelWorldCamera,
    },
    voxel_world_internal::{NeedsMaterial, SpawnMeshes, WorldRoot},
};

/// The fewest chunk tasks kept running per task pool thread. More run once the
/// pool finishes more than half as many in a frame.
const MIN_TASKS_PER_THREAD: usize = 4;

/// Where a chunk is in the chunk lifecycle, as seen from outside the crate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ChunkStatus {
    /// The chunk exists and is waiting for a chunk task to start.
    Pending,
    /// The chunk's task is running.
    Generating,
    /// The chunk's latest task has been published. It may have no mesh.
    Ready,
    /// The chunk will be despawned on the next frame.
    Retiring,
}

/// The chunk state of a chunk. `T` is the handle of its running chunk task.
///
/// Public only because `VoxelWorld` queries it; this module isn't exported.
#[derive(Component)]
pub enum ChunkState<T: Send + Sync + 'static> {
    /// `priority` chunks get task slots before the others: follow-ups of
    /// superseded tasks and remeshes after an edit.
    Pending {
        priority: bool,
    },
    /// `superseded` turns true when an edit or LOD change makes the running
    /// task's inputs stale; a follow-up task starts once it is published.
    Generating {
        task: T,
        superseded: bool,
    },
    Ready,
    /// `kept` turns true once the chunk has lived through a frame while retiring.
    Retiring {
        kept: bool,
    },
}

/// Something that happens to a chunk.
pub(crate) enum ChunkEvent<T> {
    /// A chunk task started for the chunk.
    TaskStarted(T),
    /// The chunk's task finished with a result to publish.
    TaskFinished,
    /// The chunk's task relied on a cached mesh that expired before it finished.
    CachedMeshExpired,
    /// A voxel edit made the chunk's voxel data stale.
    Edited,
    /// A LOD change gave the chunk a new data or mesh shape.
    Reshaped,
    /// The chunk was chosen for despawning.
    Retire,
    /// The despawn step of a frame ran.
    DespawnStep,
}

/// What the lifecycle systems do after a transition, beyond storing the new state.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Effect {
    None,
    /// Publish the finished task's result.
    Publish,
    /// The chunk started retiring; announce it.
    AnnounceRetiring,
    /// Despawn the chunk.
    Despawn,
}

impl<T: Send + Sync + 'static> ChunkState<T> {
    /// Moves the chunk to its next state and returns what remains to be done.
    pub(crate) fn apply(&mut self, event: ChunkEvent<T>) -> Effect {
        let state = std::mem::replace(self, ChunkState::Ready);
        let (state, effect) = transition(state, event);
        *self = state;
        effect
    }

    pub fn status(&self) -> ChunkStatus {
        match self {
            ChunkState::Pending { .. } => ChunkStatus::Pending,
            ChunkState::Generating { .. } => ChunkStatus::Generating,
            ChunkState::Ready => ChunkStatus::Ready,
            ChunkState::Retiring { .. } => ChunkStatus::Retiring,
        }
    }

    fn has_priority(&self) -> bool {
        matches!(self, ChunkState::Pending { priority: true })
    }
}

/// The rules of the chunk lifecycle: the state a chunk moves to when `event`
/// happens to it in `state`. Retiring is the only way a running task is dropped.
pub(crate) fn transition<T: Send + Sync + 'static>(
    state: ChunkState<T>,
    event: ChunkEvent<T>,
) -> (ChunkState<T>, Effect) {
    use ChunkEvent as E;
    use ChunkState as S;

    match (state, event) {
        (S::Retiring { kept: false }, E::DespawnStep) => {
            (S::Retiring { kept: true }, Effect::None)
        }
        (S::Retiring { kept: true }, E::DespawnStep) => {
            (S::Retiring { kept: true }, Effect::Despawn)
        }
        (state @ S::Retiring { .. }, _) => (state, Effect::None),
        (_, E::Retire) => (S::Retiring { kept: false }, Effect::AnnounceRetiring),
        (S::Pending { .. }, E::TaskStarted(task)) => (
            S::Generating {
                task,
                superseded: false,
            },
            Effect::None,
        ),
        (S::Generating { superseded, .. }, E::TaskFinished) => {
            let state = if superseded {
                S::Pending { priority: true }
            } else {
                S::Ready
            };
            (state, Effect::Publish)
        }
        // The chunk already had its turn, so it shouldn't wait for another.
        (S::Generating { .. }, E::CachedMeshExpired) => {
            (S::Pending { priority: true }, Effect::None)
        }
        // The running task finishes and is published, then a follow-up starts.
        (S::Generating { task, .. }, E::Edited | E::Reshaped) => (
            S::Generating {
                task,
                superseded: true,
            },
            Effect::None,
        ),
        (S::Pending { .. } | S::Ready, E::Edited) => {
            (S::Pending { priority: true }, Effect::None)
        }
        (S::Ready, E::Reshaped) => (S::Pending { priority: false }, Effect::None),
        (state, _) => (state, Effect::None),
    }
}

/// The handle of a running chunk task of the voxel world `C`.
pub(crate) type ChunkTaskHandle<C> =
    Task<ChunkTask<C, <C as VoxelWorldConfig>::MaterialIndex>>;

/// The chunk state component of the voxel world `C`.
pub(crate) type ChunkStateOf<C> = ChunkState<ChunkTaskHandle<C>>;

/// The live chunks of the voxel world `C` by chunk position, from spawning until
/// despawning.
///
/// Public only because `VoxelWorld` reads it; this module isn't exported.
#[derive(Resource)]
pub struct ChunkIndex<C> {
    entities: HashMap<IVec3, Entity>,
    // Chunks by chunk state, so each lifecycle system only visits the chunks it
    // acts on. Ready chunks aren't tracked.
    /// Pending chunks with priority, oldest first. May hold chunks that have
    /// since moved on.
    priority: VecDeque<Entity>,
    /// The other Pending chunks, with their chunk positions.
    pending: HashMap<Entity, IVec3>,
    generating: HashSet<Entity>,
    retiring: HashSet<Entity>,
    /// How many chunk tasks finished in the last frame.
    finished_last_frame: usize,
    _marker: PhantomData<C>,
}

impl<C> Default for ChunkIndex<C> {
    fn default() -> Self {
        Self {
            entities: HashMap::new(),
            priority: VecDeque::new(),
            pending: HashMap::new(),
            generating: HashSet::new(),
            retiring: HashSet::new(),
            finished_last_frame: 0,
            _marker: PhantomData,
        }
    }
}

impl<C> ChunkIndex<C> {
    pub fn get(&self, chunk_pos: IVec3) -> Option<Entity> {
        self.entities.get(&chunk_pos).copied()
    }

    pub fn contains(&self, chunk_pos: IVec3) -> bool {
        self.entities.contains_key(&chunk_pos)
    }

    fn spawned(&mut self, chunk_pos: IVec3, entity: Entity) {
        self.entities.insert(chunk_pos, entity);
        self.pending.insert(entity, chunk_pos);
    }

    fn despawned(&mut self, chunk_pos: IVec3, entity: Entity) {
        self.entities.remove(&chunk_pos);
        self.retiring.remove(&entity);
    }

    /// How many chunk tasks to keep running, within `cap`: enough to keep the
    /// task pool busy until the next frame, judged by how many finished in the
    /// last one.
    fn running_tasks_target(&self, cap: usize) -> usize {
        let threads = AsyncComputeTaskPool::get().thread_num().max(1);
        (2 * self.finished_last_frame)
            .max(MIN_TASKS_PER_THREAD * threads)
            .min(cap)
    }

    /// How many chunks could start a task right away, beyond those waiting for
    /// one, with `cap` running tasks at most.
    pub(crate) fn spawn_budget(&self, cap: usize) -> usize {
        self.running_tasks_target(cap).saturating_sub(
            self.generating.len() + self.pending.len() + self.priority.len(),
        )
    }

    /// Applies `event` to the state of the chunk at `chunk_pos`, and files the
    /// chunk under its new state.
    fn apply<T: Send + Sync + 'static>(
        &mut self,
        entity: Entity,
        chunk_pos: IVec3,
        state: &mut ChunkState<T>,
        event: ChunkEvent<T>,
    ) -> Effect {
        let before = (state.status(), state.has_priority());
        let effect = state.apply(event);
        let after = (state.status(), state.has_priority());
        if before != after {
            match before {
                (ChunkStatus::Pending, false) => {
                    self.pending.remove(&entity);
                }
                // Left in the queue; starting tasks skips chunks no longer Pending.
                (ChunkStatus::Pending, true) => {}
                (ChunkStatus::Generating, _) => {
                    self.generating.remove(&entity);
                }
                (ChunkStatus::Retiring, _) => {
                    self.retiring.remove(&entity);
                }
                (ChunkStatus::Ready, _) => {}
            }
            match after {
                (ChunkStatus::Pending, true) => self.priority.push_back(entity),
                (ChunkStatus::Pending, false) => {
                    self.pending.insert(entity, chunk_pos);
                }
                (ChunkStatus::Generating, _) => {
                    self.generating.insert(entity);
                }
                (ChunkStatus::Retiring, _) => {
                    self.retiring.insert(entity);
                }
                (ChunkStatus::Ready, _) => {}
            }
        }
        effect
    }
}

/// The squared distance from the chunk's centre to the nearest camera.
fn distance_squared_to_nearest_camera(chunk_pos: IVec3, cameras: &[Vec3]) -> f32 {
    let centre = chunk_centre(chunk_pos);
    cameras
        .iter()
        .map(|camera| camera.distance_squared(centre))
        .fold(f32::INFINITY, f32::min)
}

/// What the rest of the crate asks of the chunk lifecycle this frame, by chunk
/// position. Requests for chunks that don't exist, or that are retiring, are
/// ignored, except spawning.
#[derive(Resource)]
pub(crate) struct ChunkRequests<C> {
    /// Chunks whose voxels were edited.
    pub edited: Vec<IVec3>,
    /// Chunks to spawn, with their LOD level.
    pub spawn: Vec<(IVec3, LodLevel)>,
    /// Chunks whose LOD level should change.
    pub change_lod: Vec<(IVec3, LodLevel)>,
    /// Chunks to despawn.
    pub retire: Vec<IVec3>,
    _marker: PhantomData<C>,
}

impl<C> Default for ChunkRequests<C> {
    fn default() -> Self {
        Self {
            edited: Vec::new(),
            spawn: Vec::new(),
            change_lod: Vec::new(),
            retire: Vec::new(),
            _marker: PhantomData,
        }
    }
}

/// The systems of the chunk lifecycle of the voxel world `C`.
pub(crate) struct ChunkLifecycle<C>(PhantomData<C>);

impl<C: VoxelWorldConfig> ChunkLifecycle<C> {
    /// Applies this frame's requests: edits, then spawns, LOD changes and
    /// retirements.
    #[allow(clippy::type_complexity)]
    pub fn apply_requests(
        mut commands: Commands,
        mut requests: ResMut<ChunkRequests<C>>,
        mut index: ResMut<ChunkIndex<C>>,
        mut chunks: Query<(&mut Chunk<C>, &mut ChunkStateOf<C>)>,
        world_root: Query<Entity, With<WorldRoot<C>>>,
        configuration: Res<C>,
        mut messages: (
            MessageWriter<ChunkWillUpdate<C>>,
            MessageWriter<ChunkWillChangeLod<C>>,
            MessageWriter<ChunkWillDespawn<C>>,
        ),
    ) {
        let (ev_update, ev_change_lod, ev_despawn) = &mut messages;
        let requests = &mut *requests;

        let edited: HashSet<IVec3> = requests.edited.drain(..).collect();
        for chunk_pos in edited {
            let Some(entity) = index.get(chunk_pos) else {
                continue;
            };
            if let Ok((_, mut state)) = chunks.get_mut(entity) {
                index.apply(entity, chunk_pos, &mut state, ChunkEvent::Edited);
            }
            ev_update.write(ChunkWillUpdate::<C>::new(chunk_pos, entity));
        }

        let world_root = world_root.single().unwrap();
        let attach_to_root = configuration.attach_chunks_to_root();
        for (chunk_pos, lod_level) in requests.spawn.drain(..) {
            if index.contains(chunk_pos) {
                continue;
            }
            let mut chunk_entity = commands.spawn((
                ChunkStateOf::<C>::Pending { priority: false },
                Transform::from_translation(chunk_pos.as_vec3() * CHUNK_SIZE_F - 1.0),
            ));
            let entity = chunk_entity.id();
            chunk_entity.insert(Chunk::<C>::new(
                chunk_pos,
                lod_level,
                entity,
                configuration.chunk_data_shape(lod_level),
                configuration.chunk_meshing_shape(lod_level),
            ));
            if attach_to_root {
                commands.entity(world_root).add_child(entity);
            }
            index.spawned(chunk_pos, entity);
        }

        for (chunk_pos, lod_level) in requests.change_lod.drain(..) {
            let Some(entity) = index.get(chunk_pos) else {
                continue;
            };
            let Ok((mut chunk, mut state)) = chunks.get_mut(entity) else {
                continue;
            };
            if chunk.lod_level == lod_level || state.status() == ChunkStatus::Retiring {
                continue;
            }

            ev_change_lod.write(ChunkWillChangeLod::<C>::new(chunk_pos, entity));

            let data_shape = configuration.chunk_data_shape(lod_level);
            let mesh_shape = configuration.chunk_meshing_shape(lod_level);
            let reshaped =
                chunk.data_shape != data_shape || chunk.mesh_shape != mesh_shape;
            chunk.lod_level = lod_level;
            chunk.data_shape = data_shape;
            chunk.mesh_shape = mesh_shape;
            if reshaped {
                index.apply(entity, chunk_pos, &mut state, ChunkEvent::Reshaped);
            }
        }

        for chunk_pos in requests.retire.drain(..) {
            let Some(entity) = index.get(chunk_pos) else {
                continue;
            };
            let Ok((_, mut state)) = chunks.get_mut(entity) else {
                continue;
            };
            if index.apply(entity, chunk_pos, &mut state, ChunkEvent::Retire)
                == Effect::AnnounceRetiring
            {
                ev_despawn.write(ChunkWillDespawn::<C>::new(chunk_pos, entity));
            }
        }
    }

    /// Starts chunk tasks for Pending chunks: priority chunks first, oldest
    /// first, then the chunks nearest a camera. Priority chunks are few and
    /// waited for, so they start whenever the cap allows. Of the others, only as
    /// many run at once as keep the task pool busy until the next frame, so the
    /// chunks still waiting can be started in whatever order the cameras are in
    /// by then.
    #[allow(clippy::too_many_arguments)]
    pub fn start_chunk_tasks(
        mut index: ResMut<ChunkIndex<C>>,
        mut chunks: Query<(&Chunk<C>, &mut ChunkStateOf<C>)>,
        cameras: Query<&GlobalTransform, With<VoxelWorldCamera<C>>>,
        mut ev_chunk_will_remesh: MessageWriter<ChunkWillRemesh<C>>,
        mesh_cache: Res<MeshCache<C>>,
        voxel_edits: Res<VoxelEdits<C, C::MaterialIndex>>,
        chunk_map: Res<ChunkMap<C, C::MaterialIndex>>,
        configuration: Res<C>,
    ) {
        let cap = configuration.max_active_chunk_threads();
        if index.generating.len() >= cap {
            return;
        }

        let chunk_map_read_lock = chunk_map.get_read_lock();
        let starter = TaskStarter::<C> {
            thread_pool: AsyncComputeTaskPool::get(),
            configuration: configuration.as_ref(),
            mesh_cache: mesh_cache.as_ref(),
            voxel_edits: voxel_edits.as_ref(),
            chunk_map: &chunk_map_read_lock,
        };

        while index.generating.len() < cap {
            let Some(entity) = index.priority.pop_front() else {
                break;
            };
            starter.start(entity, &mut index, &mut chunks, &mut ev_chunk_will_remesh);
        }

        let target = index.running_tasks_target(cap);
        let slots = target.saturating_sub(index.generating.len());
        if slots == 0 || index.pending.is_empty() {
            return;
        }

        let camera_positions: Vec<Vec3> = cameras
            .iter()
            .map(|transform| transform.translation())
            .collect();
        let mut nearest: Vec<(f32, Entity)> = index
            .pending
            .iter()
            .map(|(&entity, &chunk_pos)| {
                (
                    distance_squared_to_nearest_camera(chunk_pos, &camera_positions),
                    entity,
                )
            })
            .collect();
        if nearest.len() > slots {
            nearest.select_nth_unstable_by(slots - 1, |a, b| a.0.total_cmp(&b.0));
            nearest.truncate(slots);
        }
        // The task pool runs tasks in the order they start.
        nearest.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
        for (_, entity) in nearest {
            starter.start(entity, &mut index, &mut chunks, &mut ev_chunk_will_remesh);
        }
    }

    /// Collects finished chunk tasks, queues their data for publishing and, when
    /// the world spawns meshes, gives each chunk its mesh.
    #[allow(clippy::type_complexity)]
    pub fn collect_chunk_tasks(
        mut commands: Commands,
        mut index: ResMut<ChunkIndex<C>>,
        mut chunks: Query<(&Chunk<C>, &mut ChunkStateOf<C>, &Transform)>,
        mesh_assets: Option<ResMut<Assets<Mesh>>>,
        spawn_meshes: Res<SpawnMeshes<C>>,
        buffers: (
            ResMut<ChunkMapUpdateBuffer<C, C::MaterialIndex>>,
            ResMut<MeshCacheInsertBuffer<C>>,
        ),
        mesh_cache: Res<MeshCache<C>>,
    ) {
        let (mut chunk_map_update_buffer, mut mesh_cache_insert_buffer) = buffers;
        let mut mesh_assets = mesh_assets.filter(|_| spawn_meshes.0);

        let mut finished = Vec::new();
        for &entity in &index.generating {
            let Ok((_, mut state, _)) = chunks.get_mut(entity) else {
                continue;
            };
            let ChunkState::Generating { task, .. } = &mut *state else {
                continue;
            };
            if let Some(chunk_task) = future::block_on(future::poll_once(task)) {
                finished.push((entity, chunk_task));
            }
        }
        index.finished_last_frame = finished.len();

        for (entity, mut chunk_task) in finished {
            let Ok((chunk, mut state, transform)) = chunks.get_mut(entity) else {
                continue;
            };

            if let Some(mesh_assets) = mesh_assets.as_mut() {
                let hash = chunk_task.voxels_hash();
                let mesh_handle = if chunk_task.is_empty() || chunk_task.is_full() {
                    None
                } else if let Some(mesh_handle) = mesh_cache.get_mesh_handle(&hash) {
                    if let Some(user_bundle) = mesh_cache.get_user_bundle(&hash) {
                        commands.entity(entity).insert(user_bundle);
                    }
                    Some(mesh_handle)
                } else if let Some(mesh) = chunk_task.mesh.take() {
                    // Bevy 0.19 mesh slab allocator can emit use-after-free
                    // errors if zero-vertex meshes are uploaded.
                    if mesh.count_vertices() == 0 {
                        None
                    } else {
                        let mesh_ref = Arc::new(mesh_assets.add(mesh));
                        let user_bundle = chunk_task.user_bundle.take();
                        mesh_cache_insert_buffer.push((
                            hash,
                            mesh_ref.clone(),
                            user_bundle.clone(),
                        ));
                        if let Some(bundle) = user_bundle {
                            commands.entity(entity).insert(bundle);
                        }
                        Some(mesh_ref)
                    }
                } else {
                    index.apply(
                        entity,
                        chunk.position,
                        &mut state,
                        ChunkEvent::CachedMeshExpired,
                    );
                    continue;
                };

                if let Some(mesh_handle) = mesh_handle {
                    commands.entity(entity).try_insert((
                        *transform,
                        MeshRef(mesh_handle),
                        NeedsMaterial::<C>::new(),
                    ));
                } else {
                    commands
                        .entity(entity)
                        .remove::<Mesh3d>()
                        .remove::<MeshRef>()
                        .remove::<NeedsMaterial<C>>()
                        .remove::<C::ChunkUserBundle>();
                }
            }

            if index.apply(entity, chunk.position, &mut state, ChunkEvent::TaskFinished)
                == Effect::Publish
            {
                chunk_map_update_buffer.push((
                    chunk.position,
                    chunk_task.chunk_data,
                    entity,
                ));
            }
        }
    }

    /// Publishes collected chunk data and meshes, and removes despawned chunks'
    /// data.
    pub fn flush_chunk_map_buffers(
        mut chunk_map_update_buffer: ResMut<ChunkMapUpdateBuffer<C, C::MaterialIndex>>,
        mut chunk_map_remove_buffer: ResMut<ChunkMapRemoveBuffer<C>>,
        mut ev_chunk_published: MessageWriter<ChunkPublished<C>>,
        mut ev_chunk_will_spawn: MessageWriter<ChunkWillSpawn<C>>,
        chunk_map: Res<ChunkMap<C, C::MaterialIndex>>,
    ) {
        chunk_map.apply_buffers(
            &mut chunk_map_update_buffer,
            &mut chunk_map_remove_buffer,
            &mut ev_chunk_published,
            &mut ev_chunk_will_spawn,
        );
    }

    pub fn flush_mesh_cache_buffers(
        mut mesh_cache_insert_buffer: ResMut<MeshCacheInsertBuffer<C>>,
        mesh_cache: Res<MeshCache<C>>,
    ) {
        mesh_cache.apply_buffers(&mut mesh_cache_insert_buffer);
    }

    /// Despawns chunks that have been retiring for a frame.
    pub fn despawn_retired_chunks(
        mut commands: Commands,
        mut chunks: Query<(&Chunk<C>, &mut ChunkStateOf<C>)>,
        mut index: ResMut<ChunkIndex<C>>,
        mut chunk_map_remove_buffer: ResMut<ChunkMapRemoveBuffer<C>>,
    ) {
        let retiring: Vec<Entity> = index.retiring.iter().copied().collect();
        for entity in retiring {
            let Ok((chunk, mut state)) = chunks.get_mut(entity) else {
                continue;
            };
            if index.apply(entity, chunk.position, &mut state, ChunkEvent::DespawnStep)
                == Effect::Despawn
            {
                commands.entity(entity).despawn();
                index.despawned(chunk.position, entity);
                chunk_map_remove_buffer.push(chunk.position);
            }
        }
    }
}

/// What starting a chunk task needs from the world, taken once per frame.
struct TaskStarter<'a, C: VoxelWorldConfig> {
    thread_pool: &'a AsyncComputeTaskPool,
    configuration: &'a C,
    mesh_cache: &'a MeshCache<C>,
    voxel_edits: &'a VoxelEdits<C, C::MaterialIndex>,
    chunk_map: &'a RwLockReadGuard<'a, ChunkMapData<C::MaterialIndex>>,
}

impl<C: VoxelWorldConfig> TaskStarter<'_, C> {
    /// Starts the chunk's task, unless the chunk is no longer Pending, and says
    /// whether it did.
    #[allow(clippy::type_complexity)]
    fn start(
        &self,
        entity: Entity,
        index: &mut ChunkIndex<C>,
        chunks: &mut Query<(&Chunk<C>, &mut ChunkStateOf<C>)>,
        ev_chunk_will_remesh: &mut MessageWriter<ChunkWillRemesh<C>>,
    ) -> bool {
        let Ok((chunk, mut state)) = chunks.get_mut(entity) else {
            // Despawned behind the lifecycle's back.
            index.pending.remove(&entity);
            return false;
        };
        if state.status() != ChunkStatus::Pending {
            return false;
        }

        let previous_chunk_data =
            ChunkMap::<C, C::MaterialIndex>::get(&chunk.position, self.chunk_map);
        let lod_level = chunk.lod_level;
        let regenerate_strategy = self.configuration.chunk_regenerate_strategy();
        let voxel_data_fn = (self.configuration.voxel_lookup_delegate())(
            chunk.position,
            lod_level,
            previous_chunk_data.clone(),
        );
        let data_shape = chunk.data_shape;
        let mesh_shape = chunk.mesh_shape;
        let chunk_meshing_fn = (self
            .configuration
            .chunk_meshing_delegate()
            .unwrap_or(Box::new(default_chunk_meshing_delegate)))(
            chunk.position,
            lod_level,
            data_shape,
            mesh_shape,
            previous_chunk_data.clone(),
        );
        let texture_index_mapper = self.configuration.texture_index_mapper().clone();

        let mut chunk_task = ChunkTask::<C, C::MaterialIndex>::new(
            chunk.entity,
            chunk.position,
            lod_level,
            data_shape,
            mesh_shape,
            self.voxel_edits.chunk_edits(chunk.position),
        );

        let mesh_map = self.mesh_cache.get_mesh_map();

        let task = self.thread_pool.spawn(async move {
            #[cfg(feature = "trace")]
            let _span = bevy::log::info_span!("chunk_task").entered();
            chunk_task.generate(
                voxel_data_fn,
                previous_chunk_data.clone(),
                regenerate_strategy,
            );

            // No need to mesh if the chunk is empty or full
            if chunk_task.is_empty() || chunk_task.is_full() {
                return chunk_task;
            }

            // Also no need to mesh if a matching mesh is already cached
            let mesh_cache_hit = {
                #[cfg(feature = "trace")]
                let _span = bevy::log::info_span!("wait_mesh_cache_read").entered();
                mesh_map.read().unwrap()
            }
            .contains_key(&chunk_task.voxels_hash());
            if !mesh_cache_hit {
                chunk_task.mesh(chunk_meshing_fn, texture_index_mapper);
            }

            chunk_task
        });

        index.apply(
            entity,
            chunk.position,
            &mut state,
            ChunkEvent::TaskStarted(task),
        );
        ev_chunk_will_remesh
            .write(ChunkWillRemesh::<C>::new(chunk.position, chunk.entity));
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type State = ChunkState<&'static str>;
    type Event = ChunkEvent<&'static str>;

    fn status_and_effect(state: State, event: Event) -> (ChunkStatus, Effect) {
        let (state, effect) = transition(state, event);
        (state.status(), effect)
    }

    fn generating(superseded: bool) -> State {
        ChunkState::Generating {
            task: "task",
            superseded,
        }
    }

    fn live_states() -> [State; 5] {
        [
            ChunkState::Pending { priority: false },
            ChunkState::Pending { priority: true },
            generating(false),
            generating(true),
            ChunkState::Ready,
        ]
    }

    #[test]
    fn a_started_task_makes_a_pending_chunk_generating() {
        for priority in [false, true] {
            let (state, effect) = transition(
                ChunkState::Pending { priority },
                ChunkEvent::TaskStarted("task"),
            );
            assert!(matches!(
                state,
                ChunkState::Generating {
                    task: "task",
                    superseded: false
                }
            ));
            assert_eq!(effect, Effect::None);
        }
    }

    #[test]
    fn a_finished_task_is_published_and_makes_the_chunk_ready() {
        assert_eq!(
            status_and_effect(generating(false), ChunkEvent::TaskFinished),
            (ChunkStatus::Ready, Effect::Publish)
        );
    }

    #[test]
    fn an_expired_cached_mesh_sends_the_chunk_back_to_the_front_unpublished() {
        for superseded in [false, true] {
            let (state, effect) =
                transition(generating(superseded), ChunkEvent::CachedMeshExpired);
            assert!(matches!(state, ChunkState::Pending { priority: true }));
            assert_eq!(effect, Effect::None);
        }
    }

    #[test]
    fn an_edit_or_lod_change_supersedes_a_running_task_without_dropping_it() {
        for edited in [true, false] {
            for superseded in [false, true] {
                let event = if edited {
                    ChunkEvent::Edited
                } else {
                    ChunkEvent::Reshaped
                };
                let (state, effect) = transition(generating(superseded), event);
                assert!(matches!(
                    state,
                    ChunkState::Generating {
                        task: "task",
                        superseded: true
                    }
                ));
                assert_eq!(effect, Effect::None);
            }
        }
    }

    #[test]
    fn a_superseded_task_is_published_then_followed_up_first() {
        let (state, effect) = transition(generating(true), ChunkEvent::TaskFinished);
        assert!(matches!(state, ChunkState::Pending { priority: true }));
        assert_eq!(effect, Effect::Publish);
    }

    #[test]
    fn an_edited_chunk_waits_for_a_task_at_the_front() {
        let states: [State; 3] = [
            ChunkState::Pending { priority: false },
            ChunkState::Pending { priority: true },
            ChunkState::Ready,
        ];
        for state in states {
            let (state, effect) = transition(state, ChunkEvent::Edited);
            assert!(matches!(state, ChunkState::Pending { priority: true }));
            assert_eq!(effect, Effect::None);
        }
    }

    #[test]
    fn a_reshaped_chunk_waits_for_a_task_in_line() {
        for priority in [false, true] {
            let (state, _) =
                transition(State::Pending { priority }, ChunkEvent::Reshaped);
            assert!(
                matches!(state, ChunkState::Pending { priority: p } if p == priority)
            );
        }
        let (state, effect) = transition(State::Ready, ChunkEvent::Reshaped);
        assert!(matches!(state, ChunkState::Pending { priority: false }));
        assert_eq!(effect, Effect::None);
    }

    #[test]
    fn retiring_a_live_chunk_announces_it_once() {
        for state in live_states() {
            let (state, effect) = transition(state, ChunkEvent::Retire);
            assert_eq!(
                (state.status(), effect),
                (ChunkStatus::Retiring, Effect::AnnounceRetiring)
            );
            assert_eq!(
                status_and_effect(state, ChunkEvent::Retire),
                (ChunkStatus::Retiring, Effect::None)
            );
        }
    }

    #[test]
    fn a_retiring_chunk_lives_through_one_despawn_step() {
        let (state, effect) = transition(ChunkState::Ready, ChunkEvent::Retire);
        assert_eq!(effect, Effect::AnnounceRetiring);
        let (state, effect) = transition(state, ChunkEvent::DespawnStep);
        assert_eq!(
            (state.status(), effect),
            (ChunkStatus::Retiring, Effect::None)
        );
        assert_eq!(
            status_and_effect(state, ChunkEvent::DespawnStep),
            (ChunkStatus::Retiring, Effect::Despawn)
        );
    }

    #[test]
    fn a_retiring_chunk_ignores_everything_but_despawning() {
        for event in [
            ChunkEvent::TaskStarted("task"),
            ChunkEvent::TaskFinished,
            ChunkEvent::CachedMeshExpired,
            ChunkEvent::Edited,
            ChunkEvent::Reshaped,
        ] {
            assert_eq!(
                status_and_effect(ChunkState::Retiring { kept: false }, event),
                (ChunkStatus::Retiring, Effect::None)
            );
        }
    }

    #[test]
    fn the_despawn_step_leaves_live_chunks_alone() {
        for state in live_states() {
            let status = state.status();
            assert_eq!(
                status_and_effect(state, ChunkEvent::DespawnStep),
                (status, Effect::None)
            );
        }
    }
}
