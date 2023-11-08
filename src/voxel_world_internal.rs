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
    sync::{Arc, RwLock},
};

use crate::{
    chunk::{self, CHUNK_SIZE_F, CHUNK_SIZE_I},
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
        let modified_voxels = (*self).read().unwrap();
        modified_voxels.get(position).cloned()
    }
}

/// Holds a map of all chunks that are currently spawned spawned
/// The chunks also exist as entities that can be queried in the ECS,
/// but having this map in addition allows for faster spatial lookups
#[derive(Resource, Deref, DerefMut)]
pub struct ChunkMap(Arc<RwLock<HashMap<IVec3, chunk::ChunkData>>>);

impl Default for ChunkMap {
    fn default() -> Self {
        Self(Arc::new(RwLock::new(HashMap::new())))
    }
}

/// A temporary buffer for voxel modifications that will get flushed to the `ModifiedVoxels` resource
/// at the end of the frame.
#[derive(Resource, Deref, DerefMut, Default)]
pub struct VoxelWriteBuffer(Vec<(IVec3, WorldVoxel)>);

/// Init the resources used internally by bevy_voxel_world
pub(crate) fn setup_internals(mut commands: Commands) {
    commands.init_resource::<ChunkMap>();
    commands.init_resource::<ModifiedVoxels>();
    commands.init_resource::<VoxelWriteBuffer>();
}

/// Find and spawn chunks in need of spawning
pub(crate) fn spawn_chunks(
    mut commands: Commands,
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

    let queue_chunks_intersecting_ray_from_point = |point: Vec2, queue: &mut VecDeque<IVec3>| {
        let ray = camera.viewport_to_world(cam_gtf, point).unwrap_or_default();
        let mut current = ray.origin;
        let mut t = 0.0;
        for _ in 0..spawning_distance {
            let chunk_pos = current.as_ivec3() / chunk::CHUNK_SIZE_I;
            let has_chunk = {
                let chunk_map = (*chunk_map).read().unwrap();
                chunk_map.contains_key(&chunk_pos)
            };
            if !has_chunk {
                queue.push_back(chunk_pos);
            }
            t += chunk::CHUNK_SIZE_F;
            current = ray.origin + ray.direction * t;
        }
    };

    // Each frame we pick a random point on the screen
    let random_point_in_viewport = {
        let x = rand::random::<f32>() * viewport_size.x as f32;
        let y = rand::random::<f32>() * viewport_size.y as f32;
        Vec2::new(x, y)
    };

    // We then cast a ray from this point, picking up any unspawned chunks along the ray
    queue_chunks_intersecting_ray_from_point(random_point_in_viewport, &mut chunks_deque);

    // We also queue the chunk closest to the camera to make sure it will always spawn early
    let chunk_at_camera = cam_pos / chunk::CHUNK_SIZE_I;
    chunks_deque.push_back(chunk_at_camera);

    let spawn_stratey = &configuration.chunk_spawn_strategy;
    let despawn_strategy = &configuration.chunk_despawn_strategy;
    let max_spawn_per_frame = configuration.max_spawn_per_frame;

    // Then, when we have an initial queue of chunks, we do kind of a flood fill to spawn
    // any new chunks we come across within the spawning distance.
    while let Some(chunk_position) = chunks_deque.pop_front() {
        if visited.contains(&chunk_position) || chunks_deque.len() > max_spawn_per_frame {
            continue;
        }
        visited.insert(chunk_position);

        if chunk_position.distance_squared(chunk_at_camera) > spawning_distance_squared {
            continue;
        }

        let has_chunk = {
            let chunk_map = (*chunk_map).read().unwrap();
            chunk_map.contains_key(&chunk_position)
        };

        if !has_chunk {
            let chunk = chunk::Chunk {
                position: chunk_position,
                entity: commands.spawn(chunk::NeedsRemesh).id(),
            };

            {
                let mut chunk_map_write = (*chunk_map).write().unwrap();
                chunk_map_write.insert(
                    chunk_position,
                    chunk::ChunkData {
                        voxels: Arc::new(
                            [WorldVoxel::Unset; chunk::PaddedChunkShape::SIZE as usize],
                        ),
                        entity: chunk.entity,
                    },
                );
            }

            commands
                .entity(chunk.entity)
                .insert(chunk)
                .insert(Transform::from_translation(
                    chunk_position.as_vec3() * CHUNK_SIZE_F - 1.0,
                ));

            if spawn_stratey != &ChunkSpawnStrategy::Close
                && despawn_strategy != &ChunkDespawnStrategy::FarAway
            {
                // If this chunk is not in view, it should be just outside of view, and we can
                // skip queing any neighbors, effectively culling the neighboring chunks
                if !is_in_view(chunk_position.as_vec3() * CHUNK_SIZE_F, camera, cam_gtf) {
                    continue;
                }
            }
        } else {
            // If the chunk was already spawned, we can move on without queueing any neighbors
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
pub fn despawn_retired_chunks(
    mut commands: Commands,
    chunk_map: Res<ChunkMap>,
    retired_chunks: Query<&chunk::Chunk, With<chunk::NeedsDespawn>>,
) {
    for chunk in retired_chunks.iter() {
        let mut chunk_map_write = (*chunk_map).write().unwrap();
        if let Some(chunk_data) = chunk_map_write.remove(&chunk.position) {
            commands.entity(chunk_data.entity).despawn_recursive();
        }
    }
}

/// Spawn a thread for each chunk that has been marked by NeedsRemesh
pub fn remesh_dirty_chunks(
    mut commands: Commands,
    mut ev_chunk_will_remesh: EventWriter<ChunkWillRemesh>,
    dirty_chunks: Query<&chunk::Chunk, With<chunk::NeedsRemesh>>,
    chunk_map: Res<ChunkMap>,
    modified_voxels: Res<ModifiedVoxels>,
    configuration: Res<VoxelWorldConfiguration>,
) {
    let thread_pool = AsyncComputeTaskPool::get();

    for chunk in dirty_chunks.iter() {
        let voxel_data_fn = (configuration.voxel_lookup_delegate)(chunk.position);
        let texture_index_mapper = configuration.texture_index_mapper.clone();

        let chunk_opt = {
            let chunk_map = (*chunk_map).read().unwrap();
            chunk_map.get(&chunk.position).cloned()
        };

        if let Some(chunk_data) = chunk_opt {
            let mut chunk_task = chunk::ChunkTask {
                position: chunk.position,
                voxels: chunk_data.voxels.clone(),
                modified_voxels: modified_voxels.clone(),
                mesh: None,
                is_empty: true,
                is_full: false,
            };

            let thread = thread_pool.spawn(async move {
                chunk_task.generate(voxel_data_fn);
                chunk_task.mesh(texture_index_mapper);
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
}

pub fn flush_voxel_write_buffer(
    mut commands: Commands,
    mut buffer: ResMut<VoxelWriteBuffer>,
    modified_voxels: ResMut<ModifiedVoxels>,
    chunk_map: Res<ChunkMap>,
) {
    let mut chunk_map = (*chunk_map).write().unwrap();
    let mut modified_voxels = (*modified_voxels).write().unwrap();

    for (position, voxel) in buffer.iter() {
        let (chunk_pos, _vox_pos) = get_chunk_voxel_position(*position);
        modified_voxels.insert(*position, *voxel);

        let chunk_opt = { chunk_map.get(&chunk_pos).cloned() };

        if let Some(chunk_data) = chunk_opt {
            // Mark the chunk as needing remeshing
            commands
                .entity(chunk_data.entity)
                .insert(chunk::NeedsRemesh);
        } else {
            let chunk = chunk::Chunk {
                position: chunk_pos,
                entity: commands.spawn(chunk::NeedsRemesh).id(),
            };
            chunk_map.insert(
                chunk_pos,
                chunk::ChunkData {
                    voxels: Arc::new([WorldVoxel::Unset; chunk::PaddedChunkShape::SIZE as usize]),
                    entity: chunk.entity,
                },
            );
            commands.entity(chunk.entity).insert(chunk);
        }
    }
    buffer.clear();
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
    chunk_map: Res<ChunkMap>,
    loading_texture: Res<LoadingTexture>,
    material_handle: Res<StandardVoxelMaterialHandle>,
) {
    if !loading_texture.is_loaded {
        return;
    }

    for (mut thread, chunk, transform) in &mut chunking_threads {
        let thread_result = future::block_on(future::poll_once(&mut thread.0));

        if thread_result.is_none() {
            continue;
        }

        if let Some(chunk_task) = thread_result {
            if !chunk_task.is_empty {
                if !chunk_task.is_full {
                    commands
                        .entity(chunk.entity)
                        .insert(MaterialMeshBundle {
                            mesh: mesh_assets.add(chunk_task.mesh.unwrap()),
                            material: material_handle.0.clone(),
                            transform: *transform,
                            ..default()
                        })
                        .remove::<bevy::render::primitives::Aabb>();

                    ev_chunk_will_spawn.send(ChunkWillSpawn {
                        chunk_key: chunk_task.position,
                        entity: chunk.entity,
                    });
                }
                let mut chunk_map_write = (*chunk_map).write().unwrap();
                let chunk_data_mut = chunk_map_write.get_mut(&chunk.position).unwrap();
                chunk_data_mut.voxels = chunk_task.voxels;
            }
        }

        commands.entity(chunk.entity).remove::<chunk::ChunkThread>();
    }
}

/// Check if the given world point is within the camera's view
#[inline]
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
