///
/// Voxel World internals
/// This module contains the internal systems and resources used to implement bevy_voxel_world.
///
use bevy::{ecs::system::SystemParam, platform::collections::HashMap, prelude::*};
use std::{
    marker::PhantomData,
    sync::{Arc, RwLock},
};

use crate::{
    chunk::*,
    chunk_lifecycle::{ChunkIndex, ChunkRequests},
    chunk_map::*,
    configuration::VoxelWorldConfig,
    mesh_cache::*,
    plugin::VoxelWorldMaterialHandle,
    streaming_policy::{self, CameraSnapshot, ExistingChunk, Occupancy},
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
        buckets
            .get(&chunk_containing(*position))?
            .get(position)
            .copied()
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
            let bucket = buckets.entry(chunk_containing(position)).or_default();
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
        let offset = chunk_containing(position) - self.chunk_pos + 1;
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

    /// Plan this frame's spawns, LOD changes and retirements, and request them.
    pub fn stream_chunks(
        mut requests: ResMut<ChunkRequests<C>>,
        index: Res<ChunkIndex<C>>,
        chunk_map: Res<ChunkMap<C, C::MaterialIndex>>,
        chunks: Query<&Chunk<C>>,
        configuration: Res<C>,
        camera_info: CameraInfo<C>,
    ) {
        let cameras: Vec<_> = camera_info
            .iter()
            .map(|(camera, transform)| {
                CameraSnapshot::new(
                    transform.affine(),
                    camera.clip_from_view(),
                    camera.physical_viewport_size().unwrap_or_default(),
                )
            })
            .collect();
        if cameras.is_empty() {
            return;
        }

        let chunk_map_read_lock = chunk_map.get_read_lock();
        let occupancy = |position: IVec3| {
            if !index.contains(position) {
                return Occupancy::Missing;
            }
            match chunk_map_read_lock.get(&position) {
                Some(chunk_data) if chunk_data.is_full => Occupancy::Full,
                Some(_) => Occupancy::Published,
                None => Occupancy::Spawned,
            }
        };
        let existing = chunks.iter().map(|chunk| ExistingChunk {
            position: chunk.position,
            lod_level: chunk.lod_level,
        });

        let plan = streaming_policy::plan(
            configuration.as_ref(),
            &cameras,
            existing,
            occupancy,
            &mut rand::rng(),
        );
        requests.spawn.extend(plan.spawn);
        requests.change_lod.extend(plan.change_lod);
        requests.retire.extend(plan.retire);
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
