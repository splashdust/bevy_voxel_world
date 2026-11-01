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
    configuration::{LodLevel, VoxelWorldConfig},
    mesh_cache::*,
    plugin::VoxelWorldMaterialHandle,
    streaming_policy::{self, CameraSnapshot, ChunkLookup, Occupancy},
    voxel_edits::{CommittedEdits, VoxelEdits},
    voxel_material::LoadingTexture,
    voxel_world::VoxelWorldCamera,
};

/// What the streaming policy sees of the chunks.
///
/// The chunk index tells which chunks exist, and the chunk map which are
/// published. The committed voxel edits tell which chunks' fill hints can't be
/// trusted.
struct Chunks<'a, C: VoxelWorldConfig> {
    index: &'a ChunkIndex<C>,
    chunk_map: &'a ChunkMapData<C::MaterialIndex>,
    edits: &'a CommittedEdits<'a, C::MaterialIndex>,
}

impl<C: VoxelWorldConfig> Chunks<'_, C> {
    /// The occupancy of a chunk that exists.
    fn published(&self, key: ChunkKey) -> Occupancy {
        self.chunk_map
            .get(&key)
            .map_or(Occupancy::Unpublished, Occupancy::of_data)
    }
}

impl<C: VoxelWorldConfig> ChunkLookup for Chunks<'_, C> {
    fn covering(&self, chunk_position: IVec3) -> Option<(ChunkKey, Occupancy)> {
        let key = self.index.covering_chunk(chunk_position)?;
        Some((key, self.published(key)))
    }

    fn occupancy(&self, key: ChunkKey) -> Occupancy {
        if !self.index.contains(key) {
            return Occupancy::Missing;
        }
        self.published(key)
    }

    fn coarsest_level(&self) -> LodLevel {
        self.index.levels_in_use().last().unwrap_or(0)
    }

    fn edits_reach(&self, key: ChunkKey) -> bool {
        self.edits.reach(key)
    }
}

#[derive(SystemParam, Deref)]
pub struct CameraInfo<'w, 's, C: VoxelWorldConfig>(
    Query<'w, 's, (&'static Camera, &'static GlobalTransform), With<VoxelWorldCamera<C>>>,
);

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

    /// Plans this frame's spawns and retirements, and requests them.
    pub fn stream_chunks(
        mut requests: ResMut<ChunkRequests<C>>,
        index: Res<ChunkIndex<C>>,
        chunk_map: Res<ChunkMap<C, C::MaterialIndex>>,
        voxel_edits: Res<VoxelEdits<C, C::MaterialIndex>>,
        chunks: Query<&Chunk<C>>,
        configuration: Res<C>,
        camera_info: CameraInfo<C>,
    ) {
        #[cfg(feature = "trace")]
        let _span = bevy::log::info_span!("stream_chunks").entered();
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
        let committed_edits = voxel_edits.committed();
        let lookup = Chunks::<C> {
            index: &index,
            chunk_map: &chunk_map_read_lock,
            edits: &committed_edits,
        };

        #[cfg(feature = "trace")]
        let _plan_span = bevy::log::info_span!("streaming_policy_plan").entered();
        let plan = streaming_policy::plan(
            configuration.as_ref(),
            &cameras,
            chunks.iter().map(|chunk| chunk.key),
            &lookup,
            index.spawn_budget(configuration.max_active_chunk_threads()),
            &mut rand::rng(),
        );
        requests.spawn.extend(plan.spawn);
        requests.retire.extend(plan.retire);
    }

    /// Commits the voxel edits set since the last frame, and requests that the
    /// chunks they make stale be updated.
    pub fn commit_voxel_edits(
        mut requests: ResMut<ChunkRequests<C>>,
        voxel_edits: Res<VoxelEdits<C, C::MaterialIndex>>,
    ) {
        #[cfg(feature = "trace")]
        let _span = bevy::log::info_span!("commit_voxel_edits").entered();
        requests.edited.extend(voxel_edits.commit());
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
