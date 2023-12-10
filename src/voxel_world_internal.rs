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
use std::{
    collections::VecDeque,
    sync::{Arc, RwLock},
};

use crate::{
    chunk::*,
    chunk_map::*,
    configuration::{ChunkDespawnStrategy, ChunkSpawnStrategy, VoxelWorldConfiguration},
    mesh_cache::*,
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
    let mut chunks_deque =
        VecDeque::with_capacity(configuration.spawning_rays * spawning_distance as usize);

    let chunk_map_read_lock = chunk_map.get_read_lock();

    // Shoots a ray from the given point, and queue all (non-spawned) chunks intersecting the ray
    let queue_chunks_intersecting_ray_from_point = |point: Vec2, queue: &mut VecDeque<IVec3>| {
        let ray = camera.viewport_to_world(cam_gtf, point).unwrap_or_default();
        let mut current = ray.origin;
        let mut t = 0.0;
        while t < (spawning_distance * CHUNK_SIZE_I) as f32 {
            let chunk_pos = current.as_ivec3() / CHUNK_SIZE_I;
            if let Some(chunk) = ChunkMap::get(&chunk_pos, &chunk_map_read_lock) {
                if chunk.is_full {
                    // If we hit a full chunk, we can stop the ray early
                    break;
                }
            } else {
                queue.push_back(chunk_pos);
            }
            t += CHUNK_SIZE_F;
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
    let chunk_at_camera = cam_pos / CHUNK_SIZE_I;
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
            let chunk = Chunk {
                position: chunk_position,
                entity: commands.spawn(NeedsRemesh).id(),
            };

            chunk_map_write_buffer.push((chunk_position, ChunkData::with_entity(chunk.entity)));

            commands.entity(chunk.entity).insert((
                chunk,
                Transform::from_translation(chunk_position.as_vec3() * CHUNK_SIZE_F - 1.0),
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
    all_chunks: Query<(&Chunk, Option<&ViewVisibility>)>,
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
        commands.entity(chunk.entity).insert(NeedsDespawn);

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
    retired_chunks: Query<(Entity, &Chunk), With<NeedsDespawn>>,
) {
    let read_lock = chunk_map.get_read_lock();
    for (entity, chunk) in retired_chunks.iter() {
        if ChunkMap::contains_chunk(&chunk.position, &read_lock) {
            commands.entity(entity).despawn_recursive();
            chunk_map_remove_buffer.push(chunk.position);
        }
    }
}

/// Spawn a thread for each chunk that has been marked by NeedsRemesh
#[allow(clippy::too_many_arguments)]
pub(crate) fn remesh_dirty_chunks(
    mut commands: Commands,
    mut ev_chunk_will_remesh: EventWriter<ChunkWillRemesh>,
    dirty_chunks: Query<&Chunk, With<NeedsRemesh>>,
    mesh_cache: Res<MeshCache>,
    modified_voxels: Res<ModifiedVoxels>,
    configuration: Res<VoxelWorldConfiguration>,
) {
    let thread_pool = AsyncComputeTaskPool::get();

    for chunk in dirty_chunks.iter() {
        let voxel_data_fn = (configuration.voxel_lookup_delegate)(chunk.position);
        let texture_index_mapper = configuration.texture_index_mapper.clone();

        let mut chunk_task = ChunkTask::new(chunk.entity, chunk.position, modified_voxels.clone());

        let mesh_map = Arc::new(mesh_cache.get_map());
        let thread = thread_pool.spawn(async move {
            chunk_task.generate(voxel_data_fn);

            // No need to mesh if the chunk is empty or full
            if chunk_task.is_empty() || chunk_task.is_full() {
                return chunk_task;
            }

            // Also no need to mesh if a matching mesh is already cached
            let mesh_cache_hit = mesh_map
                .read()
                .unwrap()
                .contains_key(&chunk_task.voxels_hash());
            if !mesh_cache_hit {
                chunk_task.mesh(texture_index_mapper);
            }

            chunk_task
        });

        commands
            .entity(chunk.entity)
            .insert(ChunkThread::new(thread, chunk.position))
            .remove::<NeedsRemesh>();

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
        (Entity, &mut ChunkThread, &mut Chunk, &Transform),
        Without<NeedsRemesh>,
    >,
    mut mesh_assets: ResMut<Assets<Mesh>>,
    buffers: (ResMut<ChunkMapUpdateBuffer>, ResMut<MeshCacheInsertBuffer>),
    res: (
        Res<MeshCache>,
        Res<StandardVoxelMaterialHandle>,
        Res<LoadingTexture>,
    ),
) {
    let (mesh_cache, material_handle, loading_texture) = res;

    if !loading_texture.is_loaded {
        return;
    }

    let (mut chunk_map_update_buffer, mut mesh_cache_insert_buffer) = buffers;

    for (entity, mut thread, chunk, transform) in &mut chunking_threads {
        let thread_result = future::block_on(future::poll_once(&mut thread.0));

        if thread_result.is_none() {
            continue;
        }

        let chunk_task = thread_result.unwrap();

        if !chunk_task.is_empty() {
            if !chunk_task.is_full() {
                let mesh_handle = {
                    if let Some(mesh_handle) = mesh_cache.get(&chunk_task.voxels_hash()) {
                        mesh_handle
                    } else {
                        if chunk_task.mesh.is_none() {
                            commands
                                .entity(chunk.entity)
                                .insert(NeedsRemesh)
                                .remove::<ChunkThread>();
                            continue;
                        }
                        let hash = chunk_task.voxels_hash();
                        let mesh_ref = Arc::new(mesh_assets.add(chunk_task.mesh.unwrap()));
                        mesh_cache_insert_buffer.push((hash, mesh_ref.clone()));
                        mesh_ref
                    }
                };

                commands
                    .entity(entity)
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
            }

            chunk_map_update_buffer.push((
                chunk.position,
                chunk_task.chunk_data,
                ChunkWillSpawn {
                    chunk_key: chunk_task.position,
                    entity,
                },
            ));
        }

        commands.entity(chunk.entity).remove::<ChunkThread>();
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
            commands.entity(chunk_data.entity).insert(NeedsRemesh);
        } else {
            let chunk = Chunk {
                position: chunk_pos,
                entity: commands.spawn(NeedsRemesh).id(),
            };
            chunk_map_insert_buffer.push((chunk_pos, ChunkData::with_entity(chunk.entity)));
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
    mut ev_chunk_will_spawn: EventWriter<ChunkWillSpawn>,
    chunk_map: Res<ChunkMap>,
) {
    chunk_map.apply_buffers(
        &mut chunk_map_insert_buffer,
        &mut chunk_map_update_buffer,
        &mut chunk_map_remove_buffer,
        &mut ev_chunk_will_spawn,
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
