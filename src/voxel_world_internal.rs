///
/// Voxel World internals
/// This module contains the internal systems and resources used to implement bevy_voxel_world.
///
use bevy::{ecs::system::SystemParam, prelude::*};
use std::marker::PhantomData;

use crate::{
    chunk::*,
    chunk_lifecycle::{ChunkIndex, ChunkRequests},
    chunk_map::*,
    configuration::VoxelWorldConfig,
    mesh_cache::*,
    plugin::VoxelWorldMaterialHandle,
    streaming_policy::{self, CameraSnapshot, ExistingChunk, Occupancy},
    voxel::WorldVoxel,
    voxel_edits::VoxelEdits,
    voxel_material::LoadingTexture,
    voxel_world::VoxelWorldCamera,
};

#[derive(SystemParam, Deref)]
pub struct CameraInfo<'w, 's, C: VoxelWorldConfig>(
    Query<'w, 's, (&'static Camera, &'static GlobalTransform), With<VoxelWorldCamera<C>>>,
);

/// Voxel edits set since the last frame, committed to `VoxelEdits` at the start
/// of the next frame's `VoxelWorldSet`.
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
        commands.init_resource::<VoxelEdits<C, C::MaterialIndex>>();
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
                _ => Occupancy::Spawned,
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

    /// Commits the voxel edits set since the last frame, and requests that the
    /// chunks they make stale be updated.
    pub fn commit_voxel_edits(
        mut buffer: ResMut<VoxelWriteBuffer<C, C::MaterialIndex>>,
        mut requests: ResMut<ChunkRequests<C>>,
        voxel_edits: Res<VoxelEdits<C, C::MaterialIndex>>,
    ) {
        if buffer.is_empty() {
            return;
        }
        requests.edited.extend(voxel_edits.commit(&buffer));
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
