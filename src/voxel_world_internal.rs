///
/// Voxel World internals
/// This module contains the internal systems and resources used to implement bevy_voxel_world.
///
use bevy::{
    ecs::system::SystemParam,
    platform::collections::{HashMap, HashSet},
    prelude::*,
};
use std::{
    collections::VecDeque,
    marker::PhantomData,
    sync::{Arc, RwLock},
};

use crate::{
    chunk::*,
    chunk_lifecycle::{ChunkIndex, ChunkRequests},
    chunk_map::*,
    configuration::{ChunkDespawnStrategy, ChunkSpawnStrategy, VoxelWorldConfig},
    mesh_cache::*,
    plugin::VoxelWorldMaterialHandle,
    voxel::WorldVoxel,
    voxel_material::LoadingTexture,
    voxel_world::{get_affected_chunk_positions, VoxelWorldCamera},
};

#[derive(SystemParam, Deref)]
pub struct CameraInfo<'w, 's, C: VoxelWorldConfig>(
    Query<'w, 's, (&'static Camera, &'static GlobalTransform), With<VoxelWorldCamera<C>>>,
);

/// The edited voxels of one chunk, shared with the chunk tasks that read them.
type EditBucket<I> = Arc<HashMap<IVec3, WorldVoxel<I>>>;

fn chunk_of(position: IVec3) -> IVec3 {
    position.div_euclid(IVec3::splat(CHUNK_SIZE_I))
}

/// Holds every voxel edit, kept between chunk spawn and despawn. Edits are
/// grouped by chunk, so a chunk task takes the edits it reads when it starts and
/// holds no lock while it runs.
#[derive(Resource, Clone)]
pub struct ModifiedVoxels<C, I>(
    Arc<RwLock<HashMap<IVec3, EditBucket<I>>>>,
    PhantomData<C>,
);

impl<C, I> Default for ModifiedVoxels<C, I> {
    fn default() -> Self {
        Self(Arc::new(RwLock::new(HashMap::new())), PhantomData)
    }
}

impl<C, I: Copy + PartialEq> ModifiedVoxels<C, I> {
    pub fn get_voxel(&self, position: &IVec3) -> Option<WorldVoxel<I>> {
        let buckets = self.0.read().unwrap();
        buckets.get(&chunk_of(*position))?.get(position).copied()
    }

    /// The edits a chunk task at `chunk_pos` reads.
    pub(crate) fn chunk_edits(&self, chunk_pos: IVec3) -> ChunkEdits<I> {
        let buckets = self.0.read().unwrap();
        let mut edits = ChunkEdits {
            chunk_pos,
            buckets: std::array::from_fn(|_| None),
        };
        if buckets.is_empty() {
            return edits;
        }
        for (i, bucket) in edits.buckets.iter_mut().enumerate() {
            let i = i as i32;
            let offset = IVec3::new(i % 3, i / 3 % 3, i / 9) - 1;
            *bucket = buckets.get(&(chunk_pos + offset)).cloned();
        }
        edits
    }

    /// Writes `edits`, calling `changed` with each position whose voxel changed.
    /// A chunk's edits are copied only while a running chunk task holds them.
    pub(crate) fn write(
        &self,
        edits: &[(IVec3, WorldVoxel<I>)],
        mut changed: impl FnMut(IVec3),
    ) {
        let mut buckets = {
            #[cfg(feature = "trace")]
            let _span = bevy::log::info_span!("wait_modified_voxels_write").entered();
            self.0.write().unwrap()
        };
        for &(position, voxel) in edits {
            let bucket = buckets.entry(chunk_of(position)).or_default();
            // Skip writes that don't change the voxel, so setting the same value
            // every frame doesn't remesh the chunk every frame.
            if bucket.get(&position) == Some(&voxel) {
                continue;
            }
            Arc::make_mut(bucket).insert(position, voxel);
            changed(position);
        }
    }
}

/// The voxel edits a chunk task reads, taken when it starts: those of its chunk
/// and of the neighbouring chunks its padding reaches into.
pub(crate) struct ChunkEdits<I> {
    chunk_pos: IVec3,
    /// By offset from `chunk_pos`, x varying fastest.
    buckets: [Option<EditBucket<I>>; 27],
}

impl<I> Default for ChunkEdits<I> {
    fn default() -> Self {
        Self {
            chunk_pos: IVec3::ZERO,
            buckets: std::array::from_fn(|_| None),
        }
    }
}

impl<I: Copy> ChunkEdits<I> {
    pub fn is_empty(&self) -> bool {
        self.buckets.iter().all(Option::is_none)
    }

    pub fn get(&self, position: IVec3) -> Option<WorldVoxel<I>> {
        let offset = chunk_of(position) - self.chunk_pos + 1;
        if offset.cmplt(IVec3::ZERO).any() || offset.cmpgt(IVec3::splat(2)).any() {
            return None;
        }
        let i = (offset.x + offset.y * 3 + offset.z * 9) as usize;
        self.buckets[i].as_ref()?.get(&position).copied()
    }
}

/// A temporary buffer for voxel modifications that will get flushed to the `ModifiedVoxels` resource
/// at the end of the frame.
#[derive(Resource, Deref, DerefMut, Default)]
pub struct VoxelWriteBuffer<C, I>(#[deref] Vec<(IVec3, WorldVoxel<I>)>, PhantomData<C>);

#[derive(Component)]
pub(crate) struct NeedsMaterial<C>(PhantomData<C>);

impl<C> NeedsMaterial<C> {
    pub fn new() -> Self {
        Self(PhantomData)
    }
}

/// Whether chunks get meshes. Off in `minimal()` worlds, which only keep voxel data.
#[derive(Resource)]
pub(crate) struct SpawnMeshes<C>(pub bool, PhantomData<C>);

impl<C> SpawnMeshes<C> {
    pub fn new(spawn_meshes: bool) -> Self {
        Self(spawn_meshes, PhantomData)
    }
}

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
        commands.init_resource::<ChunkMapUpdateBuffer<C, C::MaterialIndex>>();
        commands.init_resource::<ChunkMapRemoveBuffer<C>>();
        commands.init_resource::<MeshCache<C>>();
        commands.init_resource::<MeshCacheInsertBuffer<C>>();
        commands.init_resource::<ModifiedVoxels<C, C::MaterialIndex>>();
        commands.init_resource::<VoxelWriteBuffer<C, C::MaterialIndex>>();
        commands.init_resource::<ChunkIndex<C>>();
        commands.init_resource::<ChunkRequests<C>>();

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

    /// Find chunks in need of spawning and request them
    pub fn spawn_chunks(
        mut requests: ResMut<ChunkRequests<C>>,
        index: Res<ChunkIndex<C>>,
        chunk_map: Res<ChunkMap<C, C::MaterialIndex>>,
        configuration: Res<C>,
        camera_info: CameraInfo<C>,
    ) {
        let cameras: Vec<_> = camera_info
            .iter()
            .map(|(camera, transform)| TrackedCamera::new(camera, transform))
            .collect();
        if cameras.is_empty() {
            return;
        }

        let spawning_distance = configuration.spawning_distance() as i32;
        let spawning_distance_squared = spawning_distance.pow(2);
        let spawn_strategy = configuration.chunk_spawn_strategy();

        let visibility_margin = 0.0f32;
        let protected_chunk_radius_sq =
            (configuration.min_despawn_distance() as i32).pow(2);

        let mut visited = HashSet::new();
        let mut chunks_deque = VecDeque::with_capacity(
            configuration.spawning_rays() * spawning_distance as usize,
        );

        let chunk_map_read_lock = chunk_map.get_read_lock();

        // Shoots a ray from the given point, and queue all (non-spawned) chunks intersecting the ray
        let queue_chunks_intersecting_ray_from_point =
            |camera: &TrackedCamera, point: Vec2, queue: &mut VecDeque<IVec3>| {
                let Ok(ray) = camera.camera.viewport_to_world(camera.transform, point)
                else {
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
            let camera = &cameras[rand::random_range(0..cameras.len())];
            let viewport_size =
                camera.camera.physical_viewport_size().unwrap_or_default();
            let random_point_in_viewport = {
                let x =
                    rand::random::<f32>() * (viewport_size.x + m * 2) as f32 - m as f32;
                let y =
                    rand::random::<f32>() * (viewport_size.y + m * 2) as f32 - m as f32;
                Vec2::new(x, y)
            };

            // Then, for each point, we cast a ray, picking up any unspawned chunks along the ray
            queue_chunks_intersecting_ray_from_point(
                camera,
                random_point_in_viewport,
                &mut chunks_deque,
            );
        }

        // We also queue the chunks closest to every camera to make sure they will always spawn early.
        let protected_distance = configuration.min_despawn_distance() as i32;
        for camera in &cameras {
            for x in -protected_distance..=protected_distance {
                for y in -protected_distance..=protected_distance {
                    for z in -protected_distance..=protected_distance {
                        let queue_pos = camera.chunk_position + IVec3::new(x, y, z);
                        chunks_deque.push_back(queue_pos);
                    }
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

            if !chunk_is_close_to_any_camera(
                &cameras,
                chunk_position,
                spawning_distance_squared,
            ) {
                continue;
            }

            if spawn_strategy == ChunkSpawnStrategy::CloseAndInView
                && !chunk_visible_to_any_camera(
                    &cameras,
                    chunk_position,
                    visibility_margin,
                )
                && !chunk_is_close_to_any_camera(
                    &cameras,
                    chunk_position,
                    protected_chunk_radius_sq,
                )
            {
                continue;
            }

            if index.contains(chunk_position) {
                continue;
            }
            let camera_position =
                closest_camera_position_to_chunk(&cameras, chunk_position);
            let lod_level =
                configuration.chunk_lod(chunk_position, None, camera_position);
            requests.spawn.push((chunk_position, lod_level));

            if spawn_strategy != ChunkSpawnStrategy::Close {
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

    /// Request LOD changes for chunks whose LOD level should change.
    pub fn update_chunk_lods(
        mut requests: ResMut<ChunkRequests<C>>,
        chunks: Query<&Chunk<C>>,
        configuration: Res<C>,
        camera_info: CameraInfo<C>,
    ) {
        let cameras: Vec<_> = camera_info
            .iter()
            .map(|(camera, transform)| TrackedCamera::new(camera, transform))
            .collect();
        if cameras.is_empty() {
            return;
        }

        for chunk in chunks.iter() {
            let camera_position =
                closest_camera_position_to_chunk(&cameras, chunk.position);
            let target_lod = configuration.chunk_lod(
                chunk.position,
                Some(chunk.lod_level),
                camera_position,
            );
            if target_lod != chunk.lod_level {
                requests.change_lod.push((chunk.position, target_lod));
            }
        }
    }

    /// Request despawning of chunks that are eligible for it
    pub fn retire_chunks(
        mut requests: ResMut<ChunkRequests<C>>,
        all_chunks: Query<(&Chunk<C>, Option<&ViewVisibility>)>,
        configuration: Res<C>,
        camera_info: CameraInfo<C>,
    ) {
        let spawning_distance = configuration.spawning_distance() as i32;
        let spawning_distance_squared = spawning_distance.pow(2);

        let cameras: Vec<_> = camera_info
            .iter()
            .map(|(camera, transform)| TrackedCamera::new(camera, transform))
            .collect();
        if cameras.is_empty() {
            return;
        }

        let chunks_to_remove = {
            let mut remove = Vec::with_capacity(1000);
            for (chunk, view_visibility) in all_chunks.iter() {
                let should_be_culled = {
                    match configuration.chunk_despawn_strategy() {
                        ChunkDespawnStrategy::FarAway => false,
                        ChunkDespawnStrategy::FarAwayOrOutOfView => {
                            let frustum_culled = !chunk_visible_to_any_camera(
                                &cameras,
                                chunk.position,
                                0.0,
                            );
                            if let Some(visibility) = view_visibility {
                                !visibility.get() || frustum_culled
                            } else {
                                frustum_culled
                            }
                        }
                    }
                };
                let near_camera = chunk_is_close_to_any_camera(
                    &cameras,
                    chunk.position,
                    (CHUNK_SIZE_I * configuration.min_despawn_distance() as i32).pow(2),
                );
                if (should_be_culled && !near_camera)
                    || !chunk_is_close_to_any_camera(
                        &cameras,
                        chunk.position,
                        spawning_distance_squared + 1,
                    )
                {
                    remove.push(chunk);
                }
            }
            remove
        };

        requests
            .retire
            .extend(chunks_to_remove.into_iter().map(|chunk| chunk.position));
    }

    pub fn flush_voxel_write_buffer(
        mut buffer: ResMut<VoxelWriteBuffer<C, C::MaterialIndex>>,
        mut requests: ResMut<ChunkRequests<C>>,
        modified_voxels: ResMut<ModifiedVoxels<C, C::MaterialIndex>>,
    ) {
        if buffer.is_empty() {
            return;
        }
        modified_voxels.write(&buffer, |position| {
            requests
                .edited
                .extend(get_affected_chunk_positions(position));
        });
        buffer.clear();
    }

    /// Shows the meshes of chunks once the voxel texture has loaded.
    pub(crate) fn assign_material<M: Material>(
        mut commands: Commands,
        mut needs_material: Query<(Entity, &MeshRef, &Transform), With<NeedsMaterial<C>>>,
        material_handle: Option<Res<VoxelWorldMaterialHandle<M>>>,
        loading_texture: Res<LoadingTexture>,
    ) {
        let Some(material_handle) = material_handle else {
            return;
        };
        if !loading_texture.is_loaded {
            return;
        }

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

struct TrackedCamera<'a> {
    camera: &'a Camera,
    transform: &'a GlobalTransform,
    position: Vec3,
    chunk_position: IVec3,
}

impl<'a> TrackedCamera<'a> {
    fn new(camera: &'a Camera, transform: &'a GlobalTransform) -> Self {
        let position = transform.translation();
        Self {
            camera,
            transform,
            position,
            chunk_position: position.as_ivec3() / CHUNK_SIZE_I,
        }
    }
}

fn chunk_is_close_to_any_camera(
    cameras: &[TrackedCamera],
    chunk_position: IVec3,
    distance_squared: i32,
) -> bool {
    cameras.iter().any(|camera| {
        chunk_position.distance_squared(camera.chunk_position) <= distance_squared
    })
}

fn closest_camera_position_to_chunk(
    cameras: &[TrackedCamera],
    chunk_position: IVec3,
) -> Vec3 {
    let chunk_world_position = chunk_position.as_vec3() * CHUNK_SIZE_F;
    cameras
        .iter()
        .min_by(|a, b| {
            a.position
                .distance_squared(chunk_world_position)
                .total_cmp(&b.position.distance_squared(chunk_world_position))
        })
        .map(|camera| camera.position)
        .unwrap_or(Vec3::ZERO)
}

fn chunk_visible_to_any_camera(
    cameras: &[TrackedCamera],
    chunk_position: IVec3,
    ndc_margin: f32,
) -> bool {
    cameras.iter().any(|camera| {
        chunk_visible_to_camera(
            camera.camera,
            camera.transform,
            chunk_position,
            ndc_margin,
        )
    })
}

fn chunk_visible_to_camera(
    camera: &Camera,
    cam_gtf: &GlobalTransform,
    chunk_position: IVec3,
    ndc_margin: f32,
) -> bool {
    let chunk_min = chunk_position.as_vec3() * CHUNK_SIZE_F;
    let chunk_max = chunk_min + Vec3::splat(CHUNK_SIZE_F);

    let cam_pos = cam_gtf.translation();
    if cam_pos.x >= chunk_min.x
        && cam_pos.x <= chunk_max.x
        && cam_pos.y >= chunk_min.y
        && cam_pos.y <= chunk_max.y
        && cam_pos.z >= chunk_min.z
        && cam_pos.z <= chunk_max.z
    {
        return true;
    }

    let limit = 1.0 + ndc_margin;
    let point_in_ndc = |point: Vec3| -> bool {
        if let Some(ndc) = camera.world_to_ndc(cam_gtf, point) {
            ndc.x >= -limit
                && ndc.x <= limit
                && ndc.y >= -limit
                && ndc.y <= limit
                && ndc.z >= -ndc_margin
                && ndc.z <= 1.0 + ndc_margin
        } else {
            false
        }
    };

    if point_in_ndc((chunk_min + chunk_max) * 0.5) {
        return true;
    }

    for &x in &[chunk_min.x, chunk_max.x] {
        for &y in &[chunk_min.y, chunk_max.y] {
            for &z in &[chunk_min.z, chunk_max.z] {
                if point_in_ndc(Vec3::new(x, y, z)) {
                    return true;
                }
            }
        }
    }

    false
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
