use std::{
    marker::PhantomData,
    sync::{Arc, RwLock, Weak},
};

use bevy::{prelude::*, platform::collections::HashMap};
use weak_table::WeakValueHashMap;

use crate::prelude::VoxelWorldConfig;

/// This is used to keep a reference to a mesh handle in each chunk entity. This ensures that the WeakMap
/// we use to look up mesh handles can drop handles that no chunks are using anymore.
#[derive(Component)]
pub(crate) struct MeshRef(pub Arc<Handle<Mesh>>);

type WeakMeshMap = WeakValueHashMap<u64, Weak<Handle<Mesh>>>;

// TODO: Refactor this out of MeshCache. This is a bit of an afterthough, since user bundles are added via
// the meshing delegate function, which only runs if there's no chached mesh for the chunk. Hence the need
// to cache the user bundle here as well.
type UserBundleMap<UB> = HashMap<u64, UB>;

/// MeshCache uses a weak map to keep track of mesh handles generated for a certain configuration of voxels.
/// Using this map, we can avoid generating the same mesh multiple times, and reusing mesh handles
/// should allow Bevy to automatically batch draw identical chunks (large flat areas for example)
#[derive(Resource, Clone)]
pub(crate) struct MeshCache<C: VoxelWorldConfig> {
    mesh_handles: Arc<RwLock<WeakMeshMap>>,
    user_bundes: Arc<RwLock<UserBundleMap<C::ChunkUserBundle>>>,
    _marker: std::marker::PhantomData<C>,
}

impl<C: VoxelWorldConfig> MeshCache<C> {
    pub fn apply_buffers(&self, insert_buffer: &mut MeshCacheInsertBuffer<C>) {
        if insert_buffer.len() == 0 {
            return;
        }

        if let (Ok(mut mesh_handles), Ok(mut user_bundles)) =
            (self.mesh_handles.try_write(), self.user_bundes.try_write())
        {
            for (voxels, mesh, user_bundle) in insert_buffer.drain(..) {
                mesh_handles.insert(voxels, mesh);
                if let Some(user_bundle) = user_bundle {
                    user_bundles.insert(voxels, user_bundle);
                }
            }
            mesh_handles.remove_expired();
            //user_bundles.remove_expired();
        }
    }

    pub fn get_mesh_handle(&self, voxels_hash: &u64) -> Option<Arc<Handle<Mesh>>> {
        self.mesh_handles.read().unwrap().get(voxels_hash)
    }

    pub fn get_mesh_map(&self) -> Arc<RwLock<WeakMeshMap>> {
        self.mesh_handles.clone()
    }

    pub fn get_user_bundle(&self, voxels_hash: &u64) -> Option<C::ChunkUserBundle> {
        self.user_bundes.read().unwrap().get(voxels_hash).cloned()
    }
}

impl<C: VoxelWorldConfig> Default for MeshCache<C> {
    fn default() -> Self {
        Self {
            mesh_handles: Arc::new(RwLock::new(WeakMeshMap::with_capacity(2000))),
            user_bundes: Arc::new(RwLock::new(UserBundleMap::with_capacity(2000))),
            _marker: std::marker::PhantomData,
        }
    }
}

type MeshHandleRef = Arc<Handle<Mesh>>;

#[derive(Resource, Deref, DerefMut)]
pub(crate) struct MeshCacheInsertBuffer<C: VoxelWorldConfig>(
    #[deref] Vec<(u64, MeshHandleRef, Option<C::ChunkUserBundle>)>,
    PhantomData<C>,
);

impl<C: VoxelWorldConfig> Default for MeshCacheInsertBuffer<C> {
    fn default() -> Self {
        Self(Vec::with_capacity(1000), PhantomData)
    }
}
