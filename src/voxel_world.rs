///
/// VoxelWorld
/// This module implements most of the public API for bevy_voxel_world.
///
use std::marker::PhantomData;
use std::sync::Arc;

use bevy::{ecs::system::SystemParam, math::bounding::RayCast3d, prelude::*};

use crate::{
    chunk::ChunkData,
    chunk_map::ChunkMap,
    configuration::VoxelWorldConfig,
    traversal_alg::voxel_line_traversal,
    voxel::WorldVoxel,
    voxel_world_internal::{get_chunk_voxel_position, ModifiedVoxels, VoxelWriteBuffer},
};

/// This component is used to mark the Camera that bevy_voxel_world should use to determine
/// which chunks to spawn and despawn.
#[derive(Component)]
pub struct VoxelWorldCamera<C> {
    _marker: PhantomData<C>,
}

impl<C> Default for VoxelWorldCamera<C> {
    fn default() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

#[derive(Event)]
pub struct ChunkEvent<C> {
    pub chunk_key: IVec3,
    pub entity: Entity,
    _marker: PhantomData<C>,
}

impl<C> ChunkEvent<C> {
    pub fn new(chunk_key: IVec3, entity: Entity) -> Self {
        Self {
            chunk_key,
            entity,
            _marker: PhantomData,
        }
    }

    pub fn clone(&self) -> Self {
        Self {
            chunk_key: self.chunk_key,
            entity: self.entity,
            _marker: PhantomData,
        }
    }
}

/// Fired when a chunk is about to be despawned.
pub type ChunkWillDespawn<C> = ChunkEvent<C>;

/// Fired when a chunk is about to be spawned.
pub type ChunkWillSpawn<C> = ChunkEvent<C>;

/// Fired when a chunk is about to be remeshed.
pub type ChunkWillRemesh<C> = ChunkEvent<C>;

pub trait FilterFn<I> {
    fn call(&self, input: (Vec3, WorldVoxel<I>)) -> bool;
}

impl<F: Fn((Vec3, WorldVoxel<I>)) -> bool, I> FilterFn<I> for F {
    fn call(&self, input: (Vec3, WorldVoxel<I>)) -> bool {
        self(input)
    }
}

pub type RaycastFn<I> =
    dyn Fn(Ray3d, &dyn FilterFn<I>) -> Option<VoxelRaycastResult<I>> + Send + Sync;

#[derive(Default, Debug, PartialEq, Clone)]
pub struct VoxelRaycastResult<I = u8> {
    pub position: Vec3,
    pub normal: Option<Vec3>,
    pub voxel: WorldVoxel<I>,
}

impl<I> VoxelRaycastResult<I> {
    /// Get the voxel position of the raycast result
    pub fn voxel_pos(&self) -> IVec3 {
        self.position.floor().as_ivec3()
    }

    /// Get the face normal of the ray hit
    pub fn voxel_normal(&self) -> Option<IVec3> {
        self.normal.map(|n| n.floor().as_ivec3())
    }
}

/// Grants access to the VoxelWorld in systems
#[derive(SystemParam)]
pub struct VoxelWorld<'w, C: VoxelWorldConfig> {
    chunk_map: Res<'w, ChunkMap<C, <C as VoxelWorldConfig>::MaterialIndex>>,
    modified_voxels: Res<'w, ModifiedVoxels<C, <C as VoxelWorldConfig>::MaterialIndex>>,
    voxel_write_buffer: ResMut<'w, VoxelWriteBuffer<C, <C as VoxelWorldConfig>::MaterialIndex>>,
    #[allow(unused)]
    configuration: Res<'w, C>,
}

impl<C: VoxelWorldConfig> VoxelWorld<'_, C> {
    /// Get the voxel at the given position. The voxel will be WorldVoxel::Unset if there is no voxel at that position
    pub fn get_voxel(&self, position: IVec3) -> WorldVoxel<C::MaterialIndex> {
        self.get_voxel_fn()(position)
    }

    /// Set the voxel at the given position. This will create a new chunk if one does not exist at
    /// the given position.
    pub fn set_voxel(&mut self, position: IVec3, voxel: WorldVoxel<C::MaterialIndex>) {
        self.voxel_write_buffer.push((position, voxel));
    }

    /// Get a sendable closure that can be used to get the voxel at the given position
    /// This is useful for spawning tasks that need to access the voxel world
    pub fn get_voxel_fn(&self) -> Arc<dyn Fn(IVec3) -> WorldVoxel<C::MaterialIndex> + Send + Sync> {
        let chunk_map = self.chunk_map.get_map();
        let write_buffer = self.voxel_write_buffer.clone();
        let modified_voxels = self.modified_voxels.clone();

        Arc::new(move |position| {
            let (chunk_pos, vox_pos) = get_chunk_voxel_position(position);

            if let Some(voxel) = write_buffer
                .iter()
                .find(|(pos, _)| *pos == position)
                .map(|(_, voxel)| *voxel)
            {
                return voxel;
            }

            {
                if let Some(voxel) = modified_voxels.get_voxel(&position) {
                    return voxel;
                }
            }

            let chunk_opt = {
                let chun_map_read = chunk_map.read().unwrap();
                chun_map_read.get(&chunk_pos).cloned()
            };

            if let Some(chunk_data) = chunk_opt {
                chunk_data.get_voxel(vox_pos)
            } else {
                WorldVoxel::Unset
            }
        })
    }

    /// Get the ChunkData for the given chunk position
    ///
    /// The position should be the chunk position, measured in CHUNK_SIZE units (32 by default)
    ///
    /// You can `floor(voxel_position / CHUNK_SIZE)` to get the chunk position from a voxel position
    pub fn get_chunk_data(&self, chunk_pos: IVec3) -> Option<ChunkData<C::MaterialIndex>> {
        self.chunk_map
            .get_map()
            .read()
            .unwrap()
            .get(&chunk_pos)
            .cloned()
    }

    pub fn get_chunk_data_fn(
        &self,
    ) -> Arc<dyn Fn(IVec3) -> Option<ChunkData<C::MaterialIndex>> + Send + Sync> {
        let chunk_map = self.chunk_map.get_map();
        Arc::new(move |chunk_pos| chunk_map.read().unwrap().get(&chunk_pos).cloned())
    }

    /// Get the closes surface voxel to the given position
    /// Returns None if there is no surface voxel at or below the given position
    #[deprecated(since = "0.11.0", note = "Use raycast to find a surface instead")]
    pub fn get_closest_surface_voxel(
        &self,
        position: IVec3,
    ) -> Option<(IVec3, WorldVoxel<C::MaterialIndex>)> {
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
    #[deprecated(since = "0.11.0", note = "Use raycast to find a surface instead")]
    pub fn get_random_surface_voxel(
        &self,
        position: IVec3,
        radius: u32,
    ) -> Option<(IVec3, WorldVoxel<C::MaterialIndex>)> {
        let mut tries = 0;

        while tries < 100 {
            tries += 1;

            let r = radius as f32;
            let x = rand::random::<f32>() * r * 2.0 - r;
            let z = rand::random::<f32>() * r * 2.0 - r;

            let pos = position + IVec3::new(x as i32, position.y, z as i32);
            #[allow(deprecated)]
            if let Some(result) = self.get_closest_surface_voxel(pos) {
                return Some(result);
            }
        }

        None
    }

    /// Get first surface voxel at the given Vec2 position
    #[deprecated(since = "0.11.0", note = "Use raycast to find a surface instead")]
    pub fn get_surface_voxel_at_2d_pos(
        &self,
        pos_2d: Vec2,
    ) -> Option<(IVec3, WorldVoxel<C::MaterialIndex>)> {
        #[allow(deprecated)]
        self.get_closest_surface_voxel(IVec3 {
            x: pos_2d.x.floor() as i32,
            y: 256,
            z: pos_2d.y.floor() as i32,
        })
    }

    /// Get the first solid voxel intersecting with the given ray.
    /// The `filter` function can be used to filter out voxels that should not be considered for the raycast.
    ///
    /// Returns a `VoxelRaycastResult` with position, normal and voxel info. The position is given in world space.
    /// Returns `None` if no voxel was intersected
    ///
    /// # Example
    /// ```
    /// use bevy::prelude::*;
    /// use bevy_voxel_world::prelude::*;
    ///
    /// fn do_raycast(
    ///     voxel_world: VoxelWorld<DefaultWorld>,
    ///     camera_info: Query<(&Camera, &GlobalTransform), With<VoxelWorldCamera<DefaultWorld>>>,
    ///     mut cursor_evr: EventReader<CursorMoved>,
    /// ) {
    ///     for ev in cursor_evr.read() {
    ///         // Get a ray from the cursor position into the world
    ///         let (camera, cam_gtf) = camera_info.single();
    ///         let Ok(ray) = camera.viewport_to_world(cam_gtf, ev.position) else {
    ///            return;
    ///         };
    ///
    ///         if let Some(result) = voxel_world.raycast(ray, &|(_pos, _vox)| true) {
    ///             println!("vox_pos: {:?}, normal: {:?}, vox: {:?}", result.position, result.normal, result.voxel);
    ///         }
    ///     }
    /// }
    /// ```
    pub fn raycast(
        &self,
        ray: Ray3d,
        filter: &impl Fn((Vec3, WorldVoxel<C::MaterialIndex>)) -> bool,
    ) -> Option<VoxelRaycastResult<C::MaterialIndex>> {
        let raycast_fn = self.raycast_fn();
        raycast_fn(ray, filter)
    }

    /// Get a sendable closure that can be used to raycast into the voxel world
    pub fn raycast_fn(&self) -> Arc<RaycastFn<C::MaterialIndex>> {
        let chunk_map = self.chunk_map.get_map();
        let get_voxel = self.get_voxel_fn();

        Arc::new(move |ray, filter| {
            let p = ray.origin;
            let d = ray.direction;

            let loaded_aabb =
                ChunkMap::<C, C::MaterialIndex>::get_world_bounds(&chunk_map.read().unwrap());
            let trace_start =
                if p.cmplt(loaded_aabb.min.into()).any() || p.cmpgt(loaded_aabb.max.into()).any() {
                    if let Some(trace_start_t) =
                        RayCast3d::from_ray(ray, f32::MAX).aabb_intersection_at(&loaded_aabb)
                    {
                        ray.get_point(trace_start_t)
                    } else {
                        return None;
                    }
                } else {
                    p
                };

            // To find where we get out of the loaded cuboid, we can intersect from a point
            // guaranteed to be on the other side of the cube and in the opposite direction
            // of the ray.
            let trace_end_orig =
                trace_start + d * loaded_aabb.min.distance_squared(loaded_aabb.max);
            let trace_end_t = RayCast3d::new(trace_end_orig, -ray.direction, f32::MAX)
                .aabb_intersection_at(&loaded_aabb)
                .unwrap();
            let trace_end = Ray3d::new(trace_end_orig, -d).get_point(trace_end_t);

            let mut raycast_result = None;
            voxel_line_traversal(trace_start, trace_end, |voxel_coords, _time, face| {
                let voxel = get_voxel(voxel_coords);

                if !voxel.is_unset() && filter.call((voxel_coords.as_vec3(), voxel)) {
                    if voxel.is_solid() {
                        raycast_result = Some(VoxelRaycastResult {
                            position: voxel_coords.as_vec3(),
                            normal: face.try_into().ok(),
                            voxel,
                        });

                        // Found solid voxel - stop traversing
                        false
                    } else {
                        // Voxel is not solid - continue traversing
                        true
                    }
                } else {
                    // Ignoring this voxel bc of filter - continue traversing
                    true
                }
            });

            raycast_result
        })
    }
}
