///
/// Voxel World internals
/// This module contains the internal systems and resources used to implement bevy_voxel_world.
///
use bevy::{
    ecs::system::SystemParam,
    platform::collections::{HashMap, HashSet},
    prelude::*,
    tasks::AsyncComputeTaskPool,
};
use futures_lite::future;
use std::{
    collections::VecDeque,
    marker::PhantomData,
    sync::{Arc, RwLock},
};

use crate::{
    chunk::*,
    chunk_map::*,
    configuration::{ChunkDespawnStrategy, ChunkSpawnStrategy, VoxelWorldConfig},
    mesh_cache::*,
    plugin::VoxelWorldMaterialHandle,
    prelude::default_chunk_meshing_delegate,
    voxel::WorldVoxel,
    voxel_material::LoadingTexture,
    voxel_world::{
        get_chunk_voxel_position, ChunkWillDespawn, ChunkWillRemesh, ChunkWillSpawn,
        ChunkWillUpdate, VoxelWorldCamera,
    },
};

#[derive(SystemParam, Deref)]
pub struct CameraInfo<'w, 's, C: VoxelWorldConfig>(
    Query<'w, 's, (&'static Camera, &'static GlobalTransform), With<VoxelWorldCamera<C>>>,
);

/// Holds a map of modified voxels that will persist between chunk spawn/despawn
#[derive(Resource, Deref, DerefMut, Clone)]
pub struct ModifiedVoxels<C, I>(
    #[deref] Arc<RwLock<HashMap<IVec3, WorldVoxel<I>>>>,
    PhantomData<C>,
);

impl<C: VoxelWorldConfig> Default for ModifiedVoxels<C, C::MaterialIndex> {
    fn default() -> Self {
        Self(Arc::new(RwLock::new(HashMap::new())), PhantomData)
    }
}

impl<C: VoxelWorldConfig> ModifiedVoxels<C, C::MaterialIndex> {
    pub fn get_voxel(&self, position: &IVec3) -> Option<WorldVoxel<C::MaterialIndex>> {
        let modified_voxels = self.0.read().unwrap();
        modified_voxels.get(position).cloned()
    }
}

/// A temporary buffer for voxel modifications that will get flushed to the `ModifiedVoxels` resource
/// at the end of the frame.
#[derive(Resource, Deref, DerefMut, Default)]
pub struct VoxelWriteBuffer<C, I>(#[deref] Vec<(IVec3, WorldVoxel<I>)>, PhantomData<C>);

#[derive(Component)]
pub(crate) struct NeedsMaterial<C>(PhantomData<C>);

pub(crate) struct Internals<C>(PhantomData<C>);

#[derive(Component)]
pub struct WorldRoot<C>(PhantomData<C>);

impl<C> Internals<C>
where
    C: VoxelWorldConfig,
{
    /// Init the resources used internally by bevy_voxel_world
    pub fn setup(mut commands: Commands, configuration: Res<C>) {
        commands.init_resource::<ChunkMap<C, C::MaterialIndex>>();
        commands.init_resource::<ChunkMapInsertBuffer<C, C::MaterialIndex>>();
        commands.init_resource::<ChunkMapUpdateBuffer<C, C::MaterialIndex>>();
        commands.init_resource::<ChunkMapRemoveBuffer<C>>();
        commands.init_resource::<MeshCache<C>>();
        commands.init_resource::<MeshCacheInsertBuffer<C>>();
        commands.init_resource::<ModifiedVoxels<C, C::MaterialIndex>>();
        commands.init_resource::<VoxelWriteBuffer<C, C::MaterialIndex>>();

        // Create the root node and allow to modify it by the configuration.
        let world_root = commands
            .spawn((
                WorldRoot::<C>(PhantomData),
                Visibility::default(),
                Transform::default(),
            ))
            .id();
        configuration.init_root(commands, world_root)
    }

    /// Find and spawn chunks in need of spawning
    pub fn spawn_chunks(
        mut commands: Commands,
        mut chunk_map_insert_buffer: ResMut<ChunkMapInsertBuffer<C, C::MaterialIndex>>,
        world_root: Query<Entity, With<WorldRoot<C>>>,
        chunk_map: Res<ChunkMap<C, C::MaterialIndex>>,
        configuration: Res<C>,
        camera_info: CameraInfo<C>,
    ) {
        // Panic if no root exists as it is already inserted in the setup.
        let world_root = world_root.single().unwrap();

        let Ok((camera, cam_gtf)) = camera_info.single() else {
            return;
        };
        let cam_pos = cam_gtf.translation().as_ivec3();

        let spawning_distance = configuration.max_spawning_distance() as i32;
        let spawning_distance_squared = spawning_distance.pow(2);

        let viewport_size = camera.physical_viewport_size().unwrap_or_default();

        let mut visited = HashSet::new();
        let mut chunks_deque = VecDeque::with_capacity(
            configuration.spawning_rays() * spawning_distance as usize,
        );

        let chunk_map_read_lock = chunk_map.get_read_lock();

        // Shoots a ray from the given point, and queue all (non-spawned) chunks intersecting the ray
        let queue_chunks_intersecting_ray_from_point =
            |point: Vec2, queue: &mut VecDeque<IVec3>| {
                let Ok(ray) = camera.viewport_to_world(cam_gtf, point) else {
                    return;
                };
                let mut current = ray.origin;
                let mut t = 0.0;
                while t < (spawning_distance * CHUNK_SIZE_I) as f32 {
                    let chunk_pos = current.as_ivec3() / CHUNK_SIZE_I;
                    if let Some(chunk) = ChunkMap::<C, C::MaterialIndex>::get(
                        &chunk_pos,
                        &chunk_map_read_lock,
                    ) {
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
        let m = configuration.spawning_ray_margin();
        for _ in 0..configuration.spawning_rays() {
            let random_point_in_viewport = {
                let x =
                    rand::random::<f32>() * (viewport_size.x + m * 2) as f32 - m as f32;
                let y =
                    rand::random::<f32>() * (viewport_size.y + m * 2) as f32 - m as f32;
                Vec2::new(x, y)
            };

            // Then, for each point, we cast a ray, picking up any unspawned chunks along the ray
            queue_chunks_intersecting_ray_from_point(
                random_point_in_viewport,
                &mut chunks_deque,
            );
        }

        // We also queue the chunks closest to the camera to make sure they will always spawn early
        let chunk_at_camera = cam_pos / CHUNK_SIZE_I;
        let distance = configuration.min_spawning_distance() as i32;
        for x in -distance..=distance {
            for y in -distance..=distance {
                for z in -distance..=distance {
                    let queue_pos = chunk_at_camera + IVec3::new(x, y, z);
                    chunks_deque.push_back(queue_pos);
                }
            }
        }

        // Then, when we have a queue of chunks, we can set them up for spawning
        while let Some(chunk_position) = chunks_deque.pop_front() {
            if visited.contains(&chunk_position)
                || chunks_deque.len() > configuration.max_spawn_per_frame()
            {
                continue;
            }
            visited.insert(chunk_position);

            if chunk_position.distance_squared(chunk_at_camera)
                > spawning_distance_squared
            {
                continue;
            }

            let has_chunk = ChunkMap::<C, C::MaterialIndex>::contains_chunk(
                &chunk_position,
                &chunk_map_read_lock,
            );

            if !has_chunk {
                let chunk_entity = commands.spawn(NeedsRemesh).id();
                commands.entity(world_root).add_child(chunk_entity);
                let chunk = Chunk::<C>::new(chunk_position, chunk_entity);

                chunk_map_insert_buffer
                    .push((chunk_position, ChunkData::with_entity(chunk.entity)));

                commands.entity(chunk.entity).try_insert((
                    chunk,
                    Transform::from_translation(
                        chunk_position.as_vec3() * CHUNK_SIZE_F - 1.0,
                    ),
                ));
            } else {
                continue;
            }

            if configuration.chunk_spawn_strategy() != ChunkSpawnStrategy::Close {
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
        all_chunks: Query<(&Chunk<C>, Option<&ViewVisibility>)>,
        configuration: Res<C>,
        camera_info: CameraInfo<C>,
        mut ev_chunk_will_despawn: EventWriter<ChunkWillDespawn<C>>,
    ) {
        let spawning_distance = configuration.max_spawning_distance() as i32;
        let spawning_distance_squared = spawning_distance.pow(2);

        let (_, cam_gtf) = camera_info.single().unwrap();
        let cam_pos = cam_gtf.translation().as_ivec3();

        let chunk_at_camera = cam_pos / CHUNK_SIZE_I;

        let chunks_to_remove = {
            let mut remove = Vec::with_capacity(1000);
            for (chunk, view_visibility) in all_chunks.iter() {
                let should_be_culled = {
                    match configuration.chunk_despawn_strategy() {
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
                let near_camera = dist_squared
                    <= (CHUNK_SIZE_I * configuration.min_spawning_distance() as i32)
                        .pow(2);
                if (should_be_culled && !near_camera)
                    || dist_squared > spawning_distance_squared + 1
                {
                    remove.push(chunk);
                }
            }
            remove
        };

        for chunk in chunks_to_remove {
            commands.entity(chunk.entity).try_insert(NeedsDespawn);
            ev_chunk_will_despawn
                .write(ChunkWillDespawn::<C>::new(chunk.position, chunk.entity));
        }
    }

    /// Despawns chunks that have been tagged for despawning
    pub fn despawn_retired_chunks(
        mut commands: Commands,
        mut chunk_map_remove_buffer: ResMut<ChunkMapRemoveBuffer<C>>,
        chunk_map: Res<ChunkMap<C, C::MaterialIndex>>,
        retired_chunks: Query<(Entity, &Chunk<C>), With<NeedsDespawn>>,
    ) {
        let read_lock = chunk_map.get_read_lock();
        for (entity, chunk) in retired_chunks.iter() {
            if ChunkMap::<C, C::MaterialIndex>::contains_chunk(
                &chunk.position,
                &read_lock,
            ) {
                commands.entity(entity).despawn();
                chunk_map_remove_buffer.push(chunk.position);
            }
        }
    }

    /// Spawn a thread for each chunk that has been marked by NeedsRemesh
    #[allow(clippy::too_many_arguments)]
    pub fn remesh_dirty_chunks(
        mut commands: Commands,
        mut ev_chunk_will_remesh: EventWriter<ChunkWillRemesh<C>>,
        dirty_chunks: Query<&Chunk<C>, With<NeedsRemesh>>,
        mesh_cache: Res<MeshCache<C>>,
        modified_voxels: Res<ModifiedVoxels<C, C::MaterialIndex>>,
        configuration: Res<C>,
    ) {
        let thread_pool = AsyncComputeTaskPool::get();

        for chunk in dirty_chunks.iter() {
            let voxel_data_fn = (configuration.voxel_lookup_delegate())(chunk.position);
            let chunk_meshing_fn = (configuration
                .chunk_meshing_delegate()
                .unwrap_or(Box::new(default_chunk_meshing_delegate)))(
                chunk.position
            );
            let texture_index_mapper = configuration.texture_index_mapper().clone();

            let mut chunk_task = ChunkTask::<C, C::MaterialIndex>::new(
                chunk.entity,
                chunk.position,
                modified_voxels.clone(),
            );

            let mesh_map = mesh_cache.get_mesh_map();

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
                    chunk_task.mesh(chunk_meshing_fn, texture_index_mapper);
                }

                chunk_task
            });

            commands
                .entity(chunk.entity)
                .try_insert(ChunkThread::<C, C::MaterialIndex>::new(
                    thread,
                    chunk.position,
                ))
                .remove::<NeedsRemesh>();

            ev_chunk_will_remesh
                .write(ChunkWillRemesh::<C>::new(chunk.position, chunk.entity));
        }
    }

    /// Inserts new meshes for chunks that have just finished remeshing
    #[allow(clippy::type_complexity)]
    pub fn spawn_meshes(
        mut commands: Commands,
        mut chunking_threads: Query<
            (
                Entity,
                &mut ChunkThread<C, C::MaterialIndex>,
                &mut Chunk<C>,
                &Transform,
            ),
            Without<NeedsRemesh>,
        >,
        mut mesh_assets: ResMut<Assets<Mesh>>,
        buffers: (
            ResMut<ChunkMapUpdateBuffer<C, C::MaterialIndex>>,
            ResMut<MeshCacheInsertBuffer<C>>,
        ),
        res: (Res<MeshCache<C>>, Res<LoadingTexture>),
    ) {
        let (mesh_cache, loading_texture) = res;

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
                        if let Some(mesh_handle) =
                            mesh_cache.get_mesh_handle(&chunk_task.voxels_hash())
                        {
                            if let Some(user_bundle) =
                                mesh_cache.get_user_bundle(&chunk_task.voxels_hash())
                            {
                                commands.entity(entity).insert(user_bundle);
                            }

                            mesh_handle
                        } else {
                            if chunk_task.mesh.is_none() {
                                commands
                                    .entity(chunk.entity)
                                    .try_insert(NeedsRemesh)
                                    .remove::<ChunkThread<C, C::MaterialIndex>>();
                                continue;
                            }
                            let hash = chunk_task.voxels_hash();
                            let mesh_ref =
                                Arc::new(mesh_assets.add(chunk_task.mesh.unwrap()));
                            let user_bundle = chunk_task.user_bundle;

                            mesh_cache_insert_buffer.push((
                                hash,
                                mesh_ref.clone(),
                                user_bundle.clone(),
                            ));
                            if let Some(bundle) = user_bundle {
                                commands.entity(entity).insert(bundle);
                            }
                            mesh_ref
                        }
                    };

                    commands
                        .entity(entity)
                        .try_insert((
                            *transform,
                            MeshRef(mesh_handle),
                            NeedsMaterial::<C>(PhantomData),
                        ))
                        .remove::<bevy::render::primitives::Aabb>();
                }
            } else {
                commands
                    .entity(entity)
                    .remove::<Mesh3d>()
                    .remove::<MeshRef>();
            }

            chunk_map_update_buffer.push((
                chunk.position,
                chunk_task.chunk_data,
                ChunkWillSpawn::<C>::new(chunk_task.position, entity),
            ));

            commands
                .entity(chunk.entity)
                .remove::<ChunkThread<C, C::MaterialIndex>>();
        }
    }

    pub fn flush_voxel_write_buffer(
        mut commands: Commands,
        mut buffer: ResMut<VoxelWriteBuffer<C, C::MaterialIndex>>,
        mut ev_chunk_will_update: EventWriter<ChunkWillUpdate<C>>,
        chunk_map: Res<ChunkMap<C, C::MaterialIndex>>,
        modified_voxels: ResMut<ModifiedVoxels<C, C::MaterialIndex>>,
    ) {
        let chunk_map_read_lock = chunk_map.get_read_lock();
        let mut modified_voxels = modified_voxels.write().unwrap();

        let mut updated_chunks = HashSet::<(Entity, IVec3)>::new();

        for (position, voxel) in buffer.iter() {
            let (chunk_pos, _vox_pos) = get_chunk_voxel_position(*position);
            modified_voxels.insert(*position, *voxel);

            // Mark the chunk as needing remeshing or spawn a new chunk if it doesn't exist
            if let Some(chunk_data) =
                ChunkMap::<C, C::MaterialIndex>::get(&chunk_pos, &chunk_map_read_lock)
            {
                if let Ok(mut ent) = commands.get_entity(chunk_data.entity) {
                    ent.try_insert(NeedsRemesh);
                    updated_chunks.insert((chunk_data.entity, chunk_pos));
                }
            }
        }

        for (entity, chunk_pos) in updated_chunks {
            ev_chunk_will_update.write(ChunkWillUpdate::<C>::new(chunk_pos, entity));
        }

        buffer.clear();
    }

    pub fn flush_mesh_cache_buffers(
        mut mesh_cache_insert_buffer: ResMut<MeshCacheInsertBuffer<C>>,
        mesh_cache: Res<MeshCache<C>>,
    ) {
        mesh_cache.apply_buffers(&mut mesh_cache_insert_buffer);
    }

    pub fn flush_chunk_map_buffers(
        mut chunk_map_insert_buffer: ResMut<ChunkMapInsertBuffer<C, C::MaterialIndex>>,
        mut chunk_map_update_buffer: ResMut<ChunkMapUpdateBuffer<C, C::MaterialIndex>>,
        mut chunk_map_remove_buffer: ResMut<ChunkMapRemoveBuffer<C>>,
        mut ev_chunk_will_spawn: EventWriter<ChunkWillSpawn<C>>,
        chunk_map: Res<ChunkMap<C, C::MaterialIndex>>,
    ) {
        chunk_map.apply_buffers(
            &mut chunk_map_insert_buffer,
            &mut chunk_map_update_buffer,
            &mut chunk_map_remove_buffer,
            &mut ev_chunk_will_spawn,
        );
    }

    pub(crate) fn assign_material<M: Material>(
        mut commands: Commands,
        mut needs_material: Query<(Entity, &MeshRef, &Transform), With<NeedsMaterial<C>>>,
        material_handle: Option<Res<VoxelWorldMaterialHandle<M>>>,
    ) {
        let Some(material_handle) = material_handle else {
            return;
        };

        for (entity, mesh_ref, transform) in needs_material.iter_mut() {
            commands
                .entity(entity)
                .insert(Mesh3d((*mesh_ref.0).clone()))
                .insert(MeshMaterial3d(material_handle.handle.clone()))
                .insert(*transform)
                .remove::<NeedsMaterial<C>>();
        }
    }
}

/// Check if the given world point is within the camera's view
#[inline]
#[allow(dead_code)]
fn is_in_view(
    world_point: Vec3,
    camera: &Camera,
    cam_global_transform: &GlobalTransform,
) -> bool {
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
