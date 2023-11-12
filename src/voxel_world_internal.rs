///
/// Voxel World internals
/// This module contains the internal systems and resources used to implement bevy_voxel_world.
///
use bevy::{
    ecs::system::SystemParam,
    prelude::*,
    tasks::AsyncComputeTaskPool,
    utils::{HashMap, HashSet},
};
use futures_lite::future;
use ndshape::ConstShape;
use std::{
    collections::VecDeque,
    hash::{Hash, Hasher},
    sync::{Arc, RwLock, RwLockReadGuard, Weak},
};
use weak_table::WeakValueHashMap;

use crate::{
    chunk::{self, NeedsRemesh, CHUNK_SIZE_F, CHUNK_SIZE_I},
    configuration::{ChunkDespawnStrategy, ChunkSpawnStrategy, VoxelWorldConfiguration},
    voxel::WorldVoxel,
    voxel_material::{LoadingTexture, StandardVoxelMaterialHandle},
    voxel_world::{ChunkWillDespawn, ChunkWillRemesh, ChunkWillSpawn, VoxelWorldCamera},
};

#[derive(SystemParam, Deref)]
pub struct CameraInfo<'w, 's>(
    Query<'w, 's, (&'static Camera, &'static GlobalTransform), With<VoxelWorldCamera>>,
);

/// Holds a map of modified voxels that will persist between chunk spawn/despawn
#[derive(Resource, Deref, DerefMut, Clone)]
pub struct ModifiedVoxels(Arc<RwLock<HashMap<IVec3, WorldVoxel>>>);

impl Default for ModifiedVoxels {
    fn default() -> Self {
        Self(Arc::new(RwLock::new(HashMap::new())))
    }
}

impl ModifiedVoxels {
    pub fn get_voxel(&self, position: &IVec3) -> Option<WorldVoxel> {
        let modified_voxels = self.read().unwrap();
        modified_voxels.get(position).cloned()
    }
}

/// Holds a map of all chunks that are currently spawned spawned
/// The chunks also exist as entities that can be queried in the ECS,
/// but having this map in addition allows for faster spatial lookups
#[derive(Resource)]
pub struct ChunkMap {
    map: Arc<RwLock<HashMap<IVec3, chunk::ChunkData>>>,
}

impl ChunkMap {
    pub fn get(
        position: &IVec3,
        read_lock: &RwLockReadGuard<HashMap<IVec3, chunk::ChunkData>>,
    ) -> Option<chunk::ChunkData> {
        read_lock.get(position).cloned()
    }

    pub fn contains_chunk(
        position: &IVec3,
        read_lock: &RwLockReadGuard<HashMap<IVec3, chunk::ChunkData>>,
    ) -> bool {
        read_lock.contains_key(position)
    }

    pub fn get_read_lock(&self) -> RwLockReadGuard<HashMap<IVec3, chunk::ChunkData>> {
        self.map.read().unwrap()
    }

    pub fn get_map(&self) -> Arc<RwLock<HashMap<IVec3, chunk::ChunkData>>> {
        self.map.clone()
    }

    pub(crate) fn apply_buffers(
        &self,
        insert_buffer: &mut ChunkMapInsertBuffer,
        update_buffer: &mut ChunkMapUpdateBuffer,
        remove_buffer: &mut ChunkMapRemoveBuffer,
    ) {
        if let Ok(mut write_lock) = self.map.try_write() {
            for (position, chunk_data) in insert_buffer.iter() {
                write_lock.insert(*position, chunk_data.clone());
            }
            insert_buffer.clear();

            for (position, chunk_data) in update_buffer.iter() {
                write_lock.insert(*position, chunk_data.clone());
            }
            update_buffer.clear();

            for position in remove_buffer.iter() {
                write_lock.remove(position);
            }
            remove_buffer.clear();
        }
    }
}

impl Default for ChunkMap {
    fn default() -> Self {
        Self {
            map: Arc::new(RwLock::new(HashMap::with_capacity(1000))),
        }
    }
}

#[derive(Resource, Deref, DerefMut, Default)]
pub(crate) struct ChunkMapInsertBuffer(Vec<(IVec3, chunk::ChunkData)>);

#[derive(Resource, Deref, DerefMut, Default)]
pub(crate) struct ChunkMapUpdateBuffer(Vec<(IVec3, chunk::ChunkData)>);

#[derive(Resource, Deref, DerefMut, Default)]
pub(crate) struct ChunkMapRemoveBuffer(Vec<IVec3>);

/// This is used to keep a reference to a mesh handle in each chunk entity. This ensures that the WeakMap
/// we use to look up mesh handles can drop handles that no chunks are using anymore.
#[derive(Component)]
struct MeshRef(Arc<Handle<Mesh>>);

type WeakMeshMap = WeakValueHashMap<u64, Weak<Handle<Mesh>>>;

/// This map keeps track of mesh handles generated for a certain configuration of voxels.
/// Using this map, we can avoid generating the same mesh multiple times, and reusing mesh handles
/// should allow Bevy to automatically batch draw identical chunks (large flat areas for example)
#[derive(Resource, Clone)]
pub(crate) struct MeshCache {
    map: Arc<RwLock<WeakMeshMap>>,
}

impl MeshCache {
    pub fn apply_buffers(&self, insert_buffer: &mut MeshCacheInsertBuffer) {
        if insert_buffer.len() == 0 {
            return;
        }

        if let Ok(mut map) = self.map.try_write() {
            for (voxels, mesh) in insert_buffer.drain(..) {
                map.insert(voxels, mesh);
            }
            map.remove_expired();
        }
    }

    pub fn get(&self, voxels_hash: &u64) -> Option<Arc<Handle<Mesh>>> {
        self.map.read().unwrap().get(voxels_hash)
    }

    pub fn get_map(&self) -> Arc<RwLock<WeakMeshMap>> {
        self.map.clone()
    }
}

impl Default for MeshCache {
    fn default() -> Self {
        Self {
            map: Arc::new(RwLock::new(WeakMeshMap::with_capacity(2000))),
        }
    }
}

#[derive(Resource, Deref, DerefMut, Default)]
pub(crate) struct MeshCacheInsertBuffer(Vec<(u64, Arc<Handle<Mesh>>)>);

/// A temporary buffer for voxel modifications that will get flushed to the `ModifiedVoxels` resource
/// at the end of the frame.
#[derive(Resource, Deref, DerefMut, Default)]
pub struct VoxelWriteBuffer(Vec<(IVec3, WorldVoxel)>);

/// Init the resources used internally by bevy_voxel_world
pub(crate) fn setup_internals(mut commands: Commands) {
    commands.init_resource::<ChunkMap>();
    commands.init_resource::<ChunkMapInsertBuffer>();
    commands.init_resource::<ChunkMapUpdateBuffer>();
    commands.init_resource::<ChunkMapRemoveBuffer>();
    commands.init_resource::<MeshCache>();
    commands.init_resource::<MeshCacheInsertBuffer>();
    commands.init_resource::<ModifiedVoxels>();
    commands.init_resource::<VoxelWriteBuffer>();
}

/// Find and spawn chunks in need of spawning
pub(crate) fn spawn_chunks(
    mut commands: Commands,
    mut chunk_map_write_buffer: ResMut<ChunkMapInsertBuffer>,
    chunk_map: Res<ChunkMap>,
    configuration: Res<VoxelWorldConfiguration>,
    camera_info: CameraInfo,
) {
    let (camera, cam_gtf) = camera_info.get_single().unwrap();
    let cam_pos = cam_gtf.translation().as_ivec3();

    let spawning_distance = configuration.spawning_distance as i32;
    let spawning_distance_squared = spawning_distance.pow(2);

    let viewport_size = camera.physical_viewport_size().unwrap_or_default();

    let mut visited = HashSet::new();
    let mut chunks_deque = VecDeque::with_capacity((spawning_distance.pow(2) * 3) as usize);

    let chunk_map_read_lock = chunk_map.get_read_lock();

    // Shoots a ray from the given point, and queue all (non-spawned) chunks intersecting the ray
    let queue_chunks_intersecting_ray_from_point = |point: Vec2, queue: &mut VecDeque<IVec3>| {
        let ray = camera.viewport_to_world(cam_gtf, point).unwrap_or_default();
        let mut current = ray.origin;
        let mut t = 0.0;
        for _ in 0..spawning_distance {
            let chunk_pos = current.as_ivec3() / chunk::CHUNK_SIZE_I;
            if let Some(chunk) = ChunkMap::get(&chunk_pos, &chunk_map_read_lock) {
                if chunk.is_full {
                    // If we hit a full chunk, we can stop the ray early
                    break;
                }
            } else {
                queue.push_back(chunk_pos);
            }
            t += chunk::CHUNK_SIZE_F;
            current = ray.origin + ray.direction * t;
        }
    };

    // Each frame we pick some random points on the screen
    let m = configuration.spawning_ray_margin;
    for _ in 0..configuration.spawning_rays {
        let random_point_in_viewport = {
            let x = rand::random::<f32>() * (viewport_size.x + m * 2) as f32 - m as f32;
            let y = rand::random::<f32>() * (viewport_size.y + m * 2) as f32 - m as f32;
            Vec2::new(x, y)
        };

        // Then, for each point, we cast a ray, picking up any unspawned chunks along the ray
        queue_chunks_intersecting_ray_from_point(random_point_in_viewport, &mut chunks_deque);
    }

    // We also queue the chunks closest to the camera to make sure they will always spawn early
    let chunk_at_camera = cam_pos / chunk::CHUNK_SIZE_I;
    for x in -1..=1 {
        for y in -1..=1 {
            for z in -1..=1 {
                let queue_pos = chunk_at_camera + IVec3::new(x, y, z);
                chunks_deque.push_back(queue_pos);
            }
        }
    }

    // Then, when we have a queue of chunks, we can set them up for spawning
    while let Some(chunk_position) = chunks_deque.pop_front() {
        if visited.contains(&chunk_position)
            || chunks_deque.len() > configuration.max_spawn_per_frame
        {
            continue;
        }
        visited.insert(chunk_position);

        if chunk_position.distance_squared(chunk_at_camera) > spawning_distance_squared {
            continue;
        }

        let has_chunk = ChunkMap::contains_chunk(&chunk_position, &chunk_map_read_lock);

        if !has_chunk {
            let chunk = chunk::Chunk {
                position: chunk_position,
                entity: commands.spawn(chunk::NeedsRemesh).id(),
            };

            chunk_map_write_buffer.push((
                chunk_position,
                chunk::ChunkData {
                    voxels: Arc::new([WorldVoxel::Unset; chunk::PaddedChunkShape::SIZE as usize]),
                    voxels_hash: 0,
                    entity: chunk.entity,
                    is_full: false,
                },
            ));

            commands
                .entity(chunk.entity)
                .insert(chunk)
                .insert(Transform::from_translation(
                    chunk_position.as_vec3() * CHUNK_SIZE_F - 1.0,
                ));
        } else {
            continue;
        }

        if configuration.chunk_spawn_strategy != ChunkSpawnStrategy::Close {
            continue;
        }

        // If we get here, we queue the neighbors
        for x in -1..=1 {
            for y in -1..=1 {
                for z in -1..=1 {
                    let queue_pos = chunk_position + IVec3::new(x, y, z);
                    if queue_pos == chunk_position {
                        continue;
                    }
                    chunks_deque.push_back(queue_pos);
                }
            }
        }
    }
}

/// Tags chunks that are eligible for despawning
pub fn retire_chunks(
    mut commands: Commands,
    all_chunks: Query<(&chunk::Chunk, Option<&ViewVisibility>)>,
    configuration: Res<VoxelWorldConfiguration>,
    camera_info: CameraInfo,
    mut ev_chunk_will_despawn: EventWriter<ChunkWillDespawn>,
) {
    let spawning_distance = configuration.spawning_distance as i32;
    let spawning_distance_squared = spawning_distance.pow(2);

    let (_, cam_gtf) = camera_info.get_single().unwrap();
    let cam_pos = cam_gtf.translation().as_ivec3();

    let chunk_at_camera = cam_pos / CHUNK_SIZE_I;

    let chunks_to_remove = {
        let mut remove = Vec::with_capacity(1000);
        for (chunk, view_visibility) in all_chunks.iter() {
            let should_be_culled = {
                match configuration.chunk_despawn_strategy {
                    ChunkDespawnStrategy::FarAway => false,
                    ChunkDespawnStrategy::FarAwayOrOutOfView => {
                        if let Some(visibility) = view_visibility {
                            !visibility.get()
                        } else {
                            false
                        }
                    }
                }
            };
            let dist_squared = chunk.position.distance_squared(chunk_at_camera);
            if should_be_culled || dist_squared > spawning_distance_squared + 1 {
                remove.push(chunk);
            }
        }
        remove
    };

    for chunk in chunks_to_remove {
        commands.entity(chunk.entity).insert(chunk::NeedsDespawn);

        ev_chunk_will_despawn.send(ChunkWillDespawn {
            chunk_key: chunk.position,
            entity: chunk.entity,
        });
    }
}

/// Despawns chunks that have been tagged for despawning
pub(crate) fn despawn_retired_chunks(
    mut commands: Commands,
    mut chunk_map_remove_buffer: ResMut<ChunkMapRemoveBuffer>,
    chunk_map: Res<ChunkMap>,
    retired_chunks: Query<&chunk::Chunk, With<chunk::NeedsDespawn>>,
) {
    let read_lock = chunk_map.get_read_lock();
    for chunk in retired_chunks.iter() {
        if let Some(chunk_data) = ChunkMap::get(&chunk.position, &read_lock) {
            commands.entity(chunk_data.entity).despawn_recursive();
            chunk_map_remove_buffer.push(chunk.position);
        }
    }
}

/// Spawn a thread for each chunk that has been marked by NeedsRemesh
pub(crate) fn remesh_dirty_chunks(
    mut commands: Commands,
    mut ev_chunk_will_remesh: EventWriter<ChunkWillRemesh>,
    dirty_chunks: Query<&chunk::Chunk, With<chunk::NeedsRemesh>>,
    mesh_cache: Res<MeshCache>,
    modified_voxels: Res<ModifiedVoxels>,
    configuration: Res<VoxelWorldConfiguration>,
) {
    let thread_pool = AsyncComputeTaskPool::get();

    for chunk in dirty_chunks.iter() {
        let voxel_data_fn = (configuration.voxel_lookup_delegate)(chunk.position);
        let texture_index_mapper = configuration.texture_index_mapper.clone();

        let mut chunk_task = chunk::ChunkTask {
            position: chunk.position,
            voxels: Arc::new([WorldVoxel::Unset; chunk::PaddedChunkShape::SIZE as usize]),
            voxels_hash: 0,
            modified_voxels: modified_voxels.clone(),
            mesh: None,
            is_empty: true,
            is_full: false,
        };

        let mesh_map = Arc::new(mesh_cache.get_map());
        let thread = thread_pool.spawn(async move {
            chunk_task.generate(voxel_data_fn);

            // No need to mesh if the chunk is empty or full
            if chunk_task.is_empty || chunk_task.is_full {
                return chunk_task;
            }

            // Pre-compute a hash for the voxels array
            chunk_task.voxels_hash = {
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                chunk_task.voxels.hash(&mut hasher);
                hasher.finish()
            };

            // Also no need to mesh if a matching mesh is already cached
            let mesh_cache_hit = mesh_map
                .read()
                .unwrap()
                .contains_key(&chunk_task.voxels_hash);
            if !mesh_cache_hit {
                chunk_task.mesh(texture_index_mapper);
            }

            chunk_task
        });

        commands
            .entity(chunk.entity)
            .insert(chunk::ChunkThread::new(thread))
            .remove::<chunk::NeedsRemesh>();

        ev_chunk_will_remesh.send(ChunkWillRemesh {
            chunk_key: chunk.position,
            entity: chunk.entity,
        });
    }
}

/// Inserts new meshes for chunks that have just finished remeshing
pub(crate) fn spawn_meshes(
    mut commands: Commands,
    mut chunking_threads: Query<
        (&mut chunk::ChunkThread, &mut chunk::Chunk, &Transform),
        Without<chunk::NeedsRemesh>,
    >,
    mut mesh_assets: ResMut<Assets<Mesh>>,
    mut ev_chunk_will_spawn: EventWriter<ChunkWillSpawn>,
    buffers: (ResMut<ChunkMapUpdateBuffer>, ResMut<MeshCacheInsertBuffer>),
    res: (
        Res<ChunkMap>,
        Res<MeshCache>,
        Res<StandardVoxelMaterialHandle>,
        Res<LoadingTexture>,
    ),
) {
    let (chunk_map, mesh_cache, material_handle, loading_texture) = res;

    if !loading_texture.is_loaded {
        return;
    }

    let (mut chunk_map_update_buffer, mut mesh_cache_insert_buffer) = buffers;

    for (mut thread, chunk, transform) in &mut chunking_threads {
        let thread_result = future::block_on(future::poll_once(&mut thread.0));

        if thread_result.is_none() {
            continue;
        }

        let chunk_task = thread_result.unwrap();

        if !chunk_task.is_empty {
            if !chunk_task.is_full {
                let mesh_handle = {
                    if let Some(mesh_handle) = mesh_cache.get(&chunk_task.voxels_hash) {
                        mesh_handle
                    } else {
                        if chunk_task.mesh.is_none() {
                            commands
                                .entity(chunk.entity)
                                .insert(NeedsRemesh)
                                .remove::<chunk::ChunkThread>();
                            continue;
                        }
                        let mesh_ref = Arc::new(mesh_assets.add(chunk_task.mesh.unwrap()));
                        mesh_cache_insert_buffer.push((chunk_task.voxels_hash, mesh_ref.clone()));
                        mesh_ref
                    }
                };

                commands
                    .entity(chunk.entity)
                    .insert((
                        MaterialMeshBundle {
                            mesh: (*mesh_handle).clone(),
                            material: material_handle.0.clone(),
                            transform: *transform,
                            ..default()
                        },
                        MeshRef(mesh_handle),
                    ))
                    .remove::<bevy::render::primitives::Aabb>();

                ev_chunk_will_spawn.send(ChunkWillSpawn {
                    chunk_key: chunk_task.position,
                    entity: chunk.entity,
                });
            }

            let chunk_map_read_lock = chunk_map.get_read_lock();
            if let Some(chunk_data) = ChunkMap::get(&chunk.position, &chunk_map_read_lock) {
                chunk_map_update_buffer.push((
                    chunk.position,
                    chunk::ChunkData {
                        voxels: chunk_task.voxels,
                        voxels_hash: chunk_task.voxels_hash,
                        is_full: chunk_task.is_full,
                        entity: chunk_data.entity,
                    },
                ));
            }
        }

        commands.entity(chunk.entity).remove::<chunk::ChunkThread>();
    }
}

pub(crate) fn flush_voxel_write_buffer(
    mut commands: Commands,
    mut buffer: ResMut<VoxelWriteBuffer>,
    mut chunk_map_insert_buffer: ResMut<ChunkMapInsertBuffer>,
    chunk_map: Res<ChunkMap>,
    modified_voxels: ResMut<ModifiedVoxels>,
) {
    let chunk_map_read_lock = chunk_map.get_read_lock();
    let mut modified_voxels = modified_voxels.write().unwrap();

    for (position, voxel) in buffer.iter() {
        let (chunk_pos, _vox_pos) = get_chunk_voxel_position(*position);
        modified_voxels.insert(*position, *voxel);

        if let Some(chunk_data) = ChunkMap::get(&chunk_pos, &chunk_map_read_lock) {
            // Mark the chunk as needing remeshing
            commands
                .entity(chunk_data.entity)
                .insert(chunk::NeedsRemesh);
        } else {
            let chunk = chunk::Chunk {
                position: chunk_pos,
                entity: commands.spawn(chunk::NeedsRemesh).id(),
            };
            chunk_map_insert_buffer.push((
                chunk_pos,
                chunk::ChunkData {
                    voxels: Arc::new([WorldVoxel::Unset; chunk::PaddedChunkShape::SIZE as usize]),
                    voxels_hash: 0,
                    entity: chunk.entity,
                    is_full: false,
                },
            ));
            commands.entity(chunk.entity).insert(chunk);
        }
    }
    buffer.clear();
}

pub(crate) fn flush_mesh_cache_buffers(
    mut mesh_cache_insert_buffer: ResMut<MeshCacheInsertBuffer>,
    mesh_cache: Res<MeshCache>,
) {
    mesh_cache.apply_buffers(&mut mesh_cache_insert_buffer);
}

pub(crate) fn flush_chunk_map_buffers(
    mut chunk_map_insert_buffer: ResMut<ChunkMapInsertBuffer>,
    mut chunk_map_update_buffer: ResMut<ChunkMapUpdateBuffer>,
    mut chunk_map_remove_buffer: ResMut<ChunkMapRemoveBuffer>,
    chunk_map: Res<ChunkMap>,
) {
    chunk_map.apply_buffers(
        &mut chunk_map_insert_buffer,
        &mut chunk_map_update_buffer,
        &mut chunk_map_remove_buffer,
    );
}

/// Check if the given world point is within the camera's view
#[inline]
#[allow(dead_code)]
fn is_in_view(world_point: Vec3, camera: &Camera, cam_global_transform: &GlobalTransform) -> bool {
    if let Some(chunk_vp) = camera.world_to_ndc(cam_global_transform, world_point) {
        // When the position is within the viewport the values returned will be between
        // -1.0 and 1.0 on the X and Y axes, and between 0.0 and 1.0 on the Z axis.
        chunk_vp.x >= -1.0
            && chunk_vp.x <= 1.0
            && chunk_vp.y >= -1.0
            && chunk_vp.y <= 1.0
            && chunk_vp.z >= 0.0
            && chunk_vp.z <= 1.0
    } else {
        false
    }
}

/// Returns a tuple of the chunk position and the voxel position within the chunk.
#[inline]
pub(crate) fn get_chunk_voxel_position(position: IVec3) -> (IVec3, UVec3) {
    let chunk_position = IVec3 {
        x: (position.x as f32 / CHUNK_SIZE_F).floor() as i32,
        y: (position.y as f32 / CHUNK_SIZE_F).floor() as i32,
        z: (position.z as f32 / CHUNK_SIZE_F).floor() as i32,
    };

    let voxel_position = (position - chunk_position * CHUNK_SIZE_I).as_uvec3() + 1;

    (chunk_position, voxel_position)
}
