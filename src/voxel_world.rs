use std::sync::{Arc, RwLock};

use bevy::{
    ecs::system::SystemParam,
    prelude::*,
    tasks::{AsyncComputeTaskPool, Task},
    utils::HashMap,
};
use block_mesh::ndshape::{ConstShape, ConstShape3u32};
use futures_lite::future;

use crate::{
    configuration::VoxelWorldConfiguration, meshing, voxel::WorldVoxel,
    voxel_material::VoxelTextureMaterialHandle,
};

pub const CHUNK_SIZE_U: u32 = 32;
pub const CHUNK_SIZE_I: i32 = CHUNK_SIZE_U as i32;
pub const CHUNK_SIZE_F: f32 = CHUNK_SIZE_U as f32;

#[derive(Component)]
pub struct VoxelWorldCamera;

/// Grants access to the VoxelWorld in systems
#[derive(SystemParam)]
pub struct VoxelWorld<'w, 's> {
    chunk_map: Res<'w, ChunkMap>,
    modified_voxels: Res<'w, ModifiedVoxels>,
    chunks: Query<'w, 's, &'static Chunk>,

    commands: Commands<'w, 's>,
}

impl<'w, 's> VoxelWorld<'w, 's> {
    /// Get the voxel at the given position, or None if there is no voxel at that position
    pub fn get_voxel(&self, position: IVec3) -> WorldVoxel {
        self.get_voxel_fn()(position)
    }

    /// Get a sendable closure that can be used to get the voxel at the given position
    /// This is useful for spawning tasks that need to access the voxel world
    pub fn get_voxel_fn(&self) -> Arc<dyn Fn(IVec3) -> WorldVoxel + Send + Sync> {
        let modified_voxels = self.modified_voxels.0.clone();
        let chunks = self
            .chunks
            .iter()
            .map(|chunk| {
                (
                    chunk.position,
                    (*self.chunks.get(chunk.entity).unwrap()).clone(),
                )
            })
            .collect::<HashMap<IVec3, _>>();

        Arc::new(move |position| {
            let (chunk_pos, vox_pos) = get_chunk_voxel_position(position);

            {
                let modified_voxels = modified_voxels.read().unwrap();
                if let Some(voxel) = modified_voxels.get(&position) {
                    return *voxel;
                }
            }

            *chunks
                .get(&chunk_pos)
                .and_then(|chunk| chunk.get_voxel(vox_pos))
                .unwrap_or(&WorldVoxel::Unset)
        })
    }

    /// Get the closes surface voxel to the given position
    /// Returns None if there is no surface voxel at or below the given position
    pub fn get_closest_surface_voxel(&self, position: IVec3) -> Option<(IVec3, WorldVoxel)> {
        let get_voxel = self.get_voxel_fn();
        let mut current_pos = position;
        let current_voxel = get_voxel(current_pos);

        let is_surface = |pos: IVec3| {
            let above = pos + IVec3::Y;
            (get_voxel(pos) != WorldVoxel::Unset && get_voxel(pos) != WorldVoxel::Air)
                && (get_voxel(above) == WorldVoxel::Unset || get_voxel(above) == WorldVoxel::Air)
        };

        if current_voxel == WorldVoxel::Unset || current_voxel == WorldVoxel::Air {
            while !is_surface(current_pos) {
                current_pos -= IVec3::Y;
                if current_pos.y < -256 {
                    return None;
                }
            }

            return Some((current_pos, get_voxel(current_pos)));
        }

        None
    }

    /// Get a randowm surface voxel within the given radius of the given position
    /// Returns None if no surface voxel was found within the given radius
    pub fn get_random_surface_voxel(
        &self,
        position: IVec3,
        radius: u32,
    ) -> Option<(IVec3, WorldVoxel)> {
        let mut tries = 0;

        while tries < 100 {
            tries += 1;

            let r = radius as f32;
            let x = rand::random::<f32>() * r * 2.0 - r;
            let y = rand::random::<f32>() * r * 2.0 - r;
            let z = rand::random::<f32>() * r * 2.0 - r;

            if y < 0.0 {
                continue;
            }

            let d = x * x + y * y + z * z;
            if d > r * r {
                continue;
            }

            let pos = position + IVec3::new(x as i32, y as i32, z as i32);
            if let Some(result) = self.get_closest_surface_voxel(pos) {
                return Some(result);
            }
        }

        None
    }

    /// Get first surface voxel at the given Vec2 position
    pub fn get_surface_voxel_at_2d_pos(&self, pos_2d: Vec2) -> Option<(IVec3, WorldVoxel)> {
        self.get_closest_surface_voxel(IVec3 {
            x: pos_2d.x.floor() as i32,
            y: 256,
            z: pos_2d.y.floor() as i32,
        })
    }

    /// Set the voxel at the given position. This will create a new chunk if one does not exist at
    /// the given position.
    pub fn set_voxel(&mut self, position: IVec3, voxel: WorldVoxel) {
        let (chunk_pos, _vox_pos) = get_chunk_voxel_position(position);

        // Set the voxel in the modified_voxels map. This map persists when chunks are despawned
        {
            let mut modified_voxels_write = (*self.modified_voxels).write().unwrap();
            modified_voxels_write.insert(position, voxel);
        };

        let chunk_entity_opt = {
            let chunk_map = (*self.chunk_map).read().unwrap();
            chunk_map.get(&chunk_pos).cloned()
        };

        if let Some(chunk_entity) = chunk_entity_opt {
            if let Ok(chunk) = self.chunks.get(chunk_entity) {
                // Mark the chunk as needing remeshing
                self.commands.entity(chunk.entity).insert(NeedsRemesh);
            }
        } else {
            let chunk = Chunk {
                position: chunk_pos,
                voxels: Arc::new([WorldVoxel::Unset; PaddedChunkShape::SIZE as usize]),
                entity: self.commands.spawn(NeedsRemesh).id(),
            };
            let mut chunk_map_write = (*self.chunk_map).write().unwrap();
            chunk_map_write.insert(chunk_pos, chunk.entity);
            self.commands.entity(chunk.entity).insert(chunk);
        }
    }
}

/// This is used internally by the plugin to manage the world
#[derive(SystemParam)]
pub(crate) struct VoxelWorldInternal<'w, 's> {
    commands: Commands<'w, 's>,

    chunk_map: Res<'w, ChunkMap>,
    modified_voxels: Res<'w, ModifiedVoxels>,
    configuration: Res<'w, VoxelWorldConfiguration>,

    dirty_chunks: Query<'w, 's, &'static Chunk, With<NeedsRemesh>>,
    retired_chunks: Query<'w, 's, &'static Chunk, With<NeedsDespawn>>,
    camera: Query<'w, 's, &'static Transform, With<VoxelWorldCamera>>,

    ev_chunk_will_spawn: EventWriter<'w, ChunkWillSpawn>,
    ev_chunk_will_despawn: EventWriter<'w, ChunkWillDespawn>,
    ev_chunk_will_remesh: EventWriter<'w, ChunkWillRemesh>,
}

impl<'w, 's> VoxelWorldInternal<'w, 's> {
    /// Spawn chunks within the given distance of the camera
    pub fn spawn_chunks_in_view(&mut self) {
        let camera_position = self
            .camera
            .get_single()
            .unwrap_or(&Transform::default())
            .translation;
        let distance = self.configuration.spawning_distance as i32;

        let min = camera_position.as_ivec3() - IVec3::splat(distance * CHUNK_SIZE_I);
        let max = camera_position.as_ivec3() + IVec3::splat(distance * CHUNK_SIZE_I);

        // TODO: This naive implementation needs to be optimized to only spawn chunks
        // that are within the view frustum and that actually have voxels in them.
        // Also, it might make more sense to start closest to the camera and work outwards.
        for x in (min.x..=max.x).step_by(CHUNK_SIZE_U as usize) {
            for y in (0..2 * CHUNK_SIZE_I).step_by(CHUNK_SIZE_U as usize) {
                for z in (min.z..=max.z).step_by(CHUNK_SIZE_U as usize) {
                    let chunk_position = IVec3::new(x, y, z) / CHUNK_SIZE_I;

                    let has_chunk = {
                        let chunk_map = (*self.chunk_map).read().unwrap();
                        chunk_map.contains_key(&chunk_position)
                    };

                    if !has_chunk {
                        let chunk = Chunk {
                            position: chunk_position,
                            voxels: Arc::new([WorldVoxel::Unset; PaddedChunkShape::SIZE as usize]),
                            entity: self.commands.spawn(NeedsRemesh).id(),
                        };

                        self.ev_chunk_will_spawn.send(ChunkWillSpawn {
                            chunk_key: chunk_position,
                            entity: chunk.entity,
                        });

                        let mut chunk_map_write = (*self.chunk_map).write().unwrap();

                        chunk_map_write.insert(chunk_position, chunk.entity);

                        self.commands.entity(chunk.entity).insert(chunk).insert(
                            Transform::from_translation(
                                chunk_position.as_vec3() * CHUNK_SIZE_F - 1.0,
                            ),
                        );
                    }
                }
            }
        }
    }

    /// Remove chunks that are outside the given distance of the camera
    pub fn remove_chunks_out_of_view(&mut self) {
        let camera_position = self
            .camera
            .get_single()
            .unwrap_or(&Transform::default())
            .translation;
        let distance = self.configuration.spawning_distance as i32;

        let min =
            (camera_position.as_ivec3() - IVec3::splat(distance * CHUNK_SIZE_I)) / CHUNK_SIZE_I;
        let max =
            (camera_position.as_ivec3() + IVec3::splat(distance * CHUNK_SIZE_I)) / CHUNK_SIZE_I;

        let chunks_to_remove = {
            let mut remove = Vec::new();
            let chunk_map_read = (*self.chunk_map).read().unwrap();
            for (chunk_position, _) in chunk_map_read.iter() {
                if chunk_position.x < min.x
                    || chunk_position.x > max.x
                    || chunk_position.z < min.z
                    || chunk_position.z > max.z
                {
                    remove.push(*chunk_position);
                }
            }
            remove
        };

        for chunk_position in chunks_to_remove {
            let mut chunk_map_write = (*self.chunk_map).write().unwrap();
            if let Some(entity) = chunk_map_write.remove(&chunk_position) {
                self.commands.entity(entity).insert(NeedsDespawn);
                self.ev_chunk_will_despawn.send(ChunkWillDespawn {
                    chunk_key: chunk_position,
                    entity,
                });
            }
        }
    }

    pub fn despawn_retired_chunks(&mut self) {
        for chunk in self.retired_chunks.iter() {
            self.commands.entity(chunk.entity).despawn_recursive();
        }
    }

    /// Remesh dirty chunks
    /// This function will spawn a new thread for each chunk that needs remeshing
    pub fn remesh_dirty_chunks(&mut self) {
        let thread_pool = AsyncComputeTaskPool::get();

        for chunk in self.dirty_chunks.iter() {
            let voxel_data_fn = (self.configuration.voxel_lookup_delegate)(chunk.position);
            let texture_index_mapper = self.configuration.texture_index_mapper.clone();

            let mut chunk_task = ChunkTask {
                position: chunk.position,
                voxels: chunk.voxels.clone(),
                modified_voxels: self.modified_voxels.clone(),
                mesh: None,
                is_empty: true,
            };

            let thread = thread_pool.spawn(async move {
                chunk_task.generate(voxel_data_fn);
                chunk_task.mesh(texture_index_mapper);
                chunk_task
            });

            self.commands
                .entity(chunk.entity)
                .insert(ChunkThread(thread))
                .remove::<NeedsRemesh>();

            self.ev_chunk_will_remesh.send(ChunkWillRemesh {
                chunk_key: chunk.position,
                entity: chunk.entity,
            });
        }
    }
}

#[derive(Resource)]
pub(crate) struct LoadingTexture {
    pub is_loaded: bool,
    pub handle: Handle<Image>,
}

#[derive(SystemParam)]
pub(crate) struct VoxelWorldMeshSpawner<'w, 's> {
    commands: Commands<'w, 's>,
    chunking_threads: Query<
        'w,
        's,
        (
            &'static mut ChunkThread,
            &'static mut Chunk,
            &'static Transform,
        ),
        Without<NeedsRemesh>,
    >,
    mesh_assets: ResMut<'w, Assets<Mesh>>,
    material_handle: Res<'w, VoxelTextureMaterialHandle>,
    loading_texture: ResMut<'w, LoadingTexture>,
}

impl<'w, 's> VoxelWorldMeshSpawner<'w, 's> {
    /// Spawn meshes for chunks that have finished remeshing
    pub fn spawn_meshes(&mut self) {
        if !self.loading_texture.is_loaded {
            return;
        }

        for (mut thread, mut chunk, transform) in &mut self.chunking_threads {
            let thread_result = future::block_on(future::poll_once(&mut thread.0));

            if thread_result.is_none() {
                continue;
            }

            if let Some(chunk_data) = thread_result {
                if !chunk_data.is_empty {
                    self.commands
                        .entity(chunk.entity)
                        .insert(MaterialMeshBundle {
                            mesh: self.mesh_assets.add(chunk_data.mesh.unwrap()),
                            material: self.material_handle.0.clone(),
                            transform: *transform,
                            ..default()
                        })
                        .remove::<bevy::render::primitives::Aabb>();

                    chunk.voxels = chunk_data.voxels;
                }
            }

            self.commands.entity(chunk.entity).remove::<ChunkThread>();
        }
    }
}

// A chunk with 1-voxel boundary padding.
pub(crate) const PADDED_CHUNK_SIZE: u32 = CHUNK_SIZE_U + 2;
pub(crate) type PaddedChunkShape =
    ConstShape3u32<PADDED_CHUNK_SIZE, PADDED_CHUNK_SIZE, PADDED_CHUNK_SIZE>;

#[derive(Resource, Deref, DerefMut)]
pub struct ChunkMap(Arc<RwLock<HashMap<IVec3, Entity>>>);

impl Default for ChunkMap {
    fn default() -> Self {
        Self(Arc::new(RwLock::new(HashMap::new())))
    }
}

#[derive(Component, Clone)]
pub struct Chunk {
    position: IVec3,
    voxels: Arc<[WorldVoxel; PaddedChunkShape::SIZE as usize]>,
    entity: Entity,
}

impl Chunk {
    fn get_voxel(&self, position: UVec3) -> Option<&WorldVoxel> {
        let i = PaddedChunkShape::linearize(position.to_array()) as usize;
        self.voxels.get(i)
    }
}

#[derive(Component)]
pub(crate) struct ChunkTask {
    position: IVec3,
    voxels: Arc<[WorldVoxel; PaddedChunkShape::SIZE as usize]>,
    modified_voxels: ModifiedVoxels,
    is_empty: bool,
    mesh: Option<Mesh>,
}

impl ChunkTask {
    pub fn generate<F>(&mut self, mut voxel_data_fn: F)
    where
        F: FnMut(IVec3) -> WorldVoxel + Send + 'static,
    {
        let mut filled_count = 0;
        let modified_voxels = (*self.modified_voxels).read().unwrap();
        let mut voxels = [WorldVoxel::Unset; PaddedChunkShape::SIZE as usize];

        for i in 0..PaddedChunkShape::SIZE {
            let chunk_block = PaddedChunkShape::delinearize(i);

            let block_pos = IVec3 {
                x: chunk_block[0] as i32 + (self.position.x * CHUNK_SIZE_I) - 1,
                y: chunk_block[1] as i32 + (self.position.y * CHUNK_SIZE_I) - 1,
                z: chunk_block[2] as i32 + (self.position.z * CHUNK_SIZE_I) - 1,
            };

            if let Some(voxel) = modified_voxels.get(&block_pos) {
                voxels[i as usize] = *voxel;
                if !voxel.is_unset() {
                    filled_count += 1;
                }
                continue;
            }

            let voxel = voxel_data_fn(block_pos);

            voxels[i as usize] = voxel;

            if let WorldVoxel::Solid(_) = voxel {
                filled_count += 1;
            }
        }

        self.voxels = Arc::new(voxels);

        // If the chunk is empty or full, we don't need to mesh it.
        self.is_empty = filled_count == PaddedChunkShape::SIZE - 1 || filled_count == 0;
    }

    pub fn mesh(&mut self, texture_index_mapper: Arc<dyn Fn(u8) -> [u32; 3] + Send + Sync>) {
        if self.mesh.is_none() {
            self.mesh = Some(meshing::generate_chunk_mesh(
                self.voxels.clone(),
                self.position,
                texture_index_mapper,
            ));
        }
    }
}

/// Returns a tuple of the chunk position and the voxel position within the chunk.
#[inline]
fn get_chunk_voxel_position(position: IVec3) -> (IVec3, UVec3) {
    let chunk_position = IVec3 {
        x: (position.x as f32 / CHUNK_SIZE_F).floor() as i32,
        y: (position.y as f32 / CHUNK_SIZE_F).floor() as i32,
        z: (position.z as f32 / CHUNK_SIZE_F).floor() as i32,
    };

    let voxel_position = (position - chunk_position * CHUNK_SIZE_I).as_uvec3() + 1;

    (chunk_position, voxel_position)
}

#[derive(Component)]
#[component(storage = "SparseSet")]
pub(crate) struct ChunkThread(Task<ChunkTask>);

#[derive(Component)]
#[component(storage = "SparseSet")]
pub(crate) struct NeedsRemesh;

#[derive(Component)]
pub struct NeedsDespawn;

#[derive(Event)]
pub struct ChunkWillDespawn {
    pub chunk_key: IVec3,
    pub entity: Entity,
}

#[derive(Event)]
pub struct ChunkWillSpawn {
    pub chunk_key: IVec3,
    pub entity: Entity,
}

#[derive(Event)]
pub struct ChunkWillRemesh {
    pub chunk_key: IVec3,
    pub entity: Entity,
}

#[derive(Resource, Deref, DerefMut, Clone)]
pub struct ModifiedVoxels(Arc<RwLock<HashMap<IVec3, WorldVoxel>>>);

impl Default for ModifiedVoxels {
    fn default() -> Self {
        Self(Arc::new(RwLock::new(HashMap::new())))
    }
}
