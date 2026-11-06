///
/// VoxelWorld
/// This module implements most of the public API for bevy_voxel_world.
///
use std::marker::PhantomData;
use std::sync::{Arc, RwLock};

use bevy::{ecs::system::SystemParam, prelude::*};

use crate::{
    chunk::{chunk_containing, ChunkData, ChunkKey, CHUNK_SIZE_I},
    chunk_lifecycle::{ChunkIndex, ChunkStateOf, ChunkStatus},
    chunk_map::{ChunkMap, ChunkMapData},
    configuration::{FillHint, VoxelWorldConfig},
    traversal_alg::voxel_line_traversal,
    voxel::WorldVoxel,
    voxel_edits::{VoxelEditReader, VoxelEdits},
};

/// This component is used to mark Cameras that bevy_voxel_world should use to determine
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

/// The systems that run the chunk lifecycle of the voxel world `C`, in `PreUpdate`.
///
/// Within a frame, they apply voxel edits, spawn and retire chunks, start chunk
/// tasks, collect finished ones, publish their data and despawn retired chunks.
/// A system ordered after this set sees every chunk published so far, and the
/// `ChunkPublished` and `ChunkWillSpawn` messages of this frame.
#[derive(SystemSet)]
pub struct VoxelWorldSet<C>(PhantomData<fn() -> C>);

impl<C> Default for VoxelWorldSet<C> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<C> Clone for VoxelWorldSet<C> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<C> Copy for VoxelWorldSet<C> {}

impl<C> PartialEq for VoxelWorldSet<C> {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

impl<C> Eq for VoxelWorldSet<C> {}

impl<C> std::hash::Hash for VoxelWorldSet<C> {
    fn hash<H: std::hash::Hasher>(&self, _: &mut H) {}
}

impl<C> std::fmt::Debug for VoxelWorldSet<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "VoxelWorldSet<{}>", std::any::type_name::<C>())
    }
}

pub trait ChunkEventType: Send + Sync + 'static {}

/// A message about a chunk, which it names by its chunk key and its entity.
#[derive(Event)]
pub struct ChunkEvent<C, E>
where
    C: VoxelWorldConfig,
    E: ChunkEventType,
{
    pub chunk_key: ChunkKey,
    pub entity: Entity,
    _marker: (PhantomData<C>, PhantomData<E>),
}

impl<C, E> ChunkEvent<C, E>
where
    C: VoxelWorldConfig,
    E: ChunkEventType,
{
    pub fn new(chunk_key: ChunkKey, entity: Entity) -> Self {
        Self {
            chunk_key,
            entity,
            _marker: (PhantomData, PhantomData),
        }
    }

    pub fn clone(&self) -> Self {
        Self {
            chunk_key: self.chunk_key,
            entity: self.entity,
            _marker: (PhantomData, PhantomData),
        }
    }
}

impl<C, E> Message for ChunkEvent<C, E>
where
    C: VoxelWorldConfig,
    E: ChunkEventType,
{
}

/// Fired when a chunk is about to be despawned.
pub type ChunkWillDespawn<C> = ChunkEvent<C, WillDespawn>;
pub struct WillDespawn;
impl ChunkEventType for WillDespawn {}

/// Fired once per chunk, when its voxel data is first published and readable
/// through `VoxelWorld`. A chunk spawned again after despawning fires it again.
pub type ChunkWillSpawn<C> = ChunkEvent<C, WillSpawn>;
pub struct WillSpawn;
impl ChunkEventType for WillSpawn {}

/// Fired every time a chunk's voxel data is published, after which it is
/// readable through `VoxelWorld`: on its first publish and after every remesh.
pub type ChunkPublished<C> = ChunkEvent<C, Published>;
pub struct Published;
impl ChunkEventType for Published {}

/// Fired when a chunk is about to be remeshed.
pub type ChunkWillRemesh<C> = ChunkEvent<C, WillRemesh>;
pub struct WillRemesh;
impl ChunkEventType for WillRemesh {}

/// Fired when a chunk is about to be updated, typically when `set_voxel` was called on a voxel
/// within the chunk.
pub type ChunkWillUpdate<C> = ChunkEvent<C, WillUpdate>;
pub struct WillUpdate;
impl ChunkEventType for WillUpdate {}

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
pub struct VoxelWorld<'w, 's, C: VoxelWorldConfig> {
    chunk_index: Res<'w, ChunkIndex<C>>,
    chunk_states: Query<'w, 's, &'static ChunkStateOf<C>>,
    chunk_map: Res<'w, ChunkMap<C, <C as VoxelWorldConfig>::MaterialIndex>>,
    voxel_edits: Res<'w, VoxelEdits<C, <C as VoxelWorldConfig>::MaterialIndex>>,
    configuration: Res<'w, C>,
    #[cfg(feature = "light_propagation")]
    light: Option<Res<'w, crate::light::LightSettings<C>>>,
}

impl<C: VoxelWorldConfig> VoxelWorld<'_, '_, C> {
    /// Where the chunk with the given key is in the chunk lifecycle, or `None`
    /// if no chunk with that key is spawned. [`VoxelWorld::chunk_at`] finds the
    /// key of the chunk covering a voxel.
    pub fn chunk_state(&self, key: ChunkKey) -> Option<ChunkStatus> {
        let entity = self.chunk_index.get(key)?;
        // A chunk spawned this frame has no state component until commands apply.
        Some(
            self.chunk_states
                .get(entity)
                .map_or(ChunkStatus::Pending, |state| state.status()),
        )
    }

    /// The key of the finest spawned chunk that covers the voxel at
    /// `voxel_position`, or `None` if no chunk does.
    pub fn chunk_at(&self, voxel_position: IVec3) -> Option<ChunkKey> {
        self.chunk_index.chunk_at(voxel_position)
    }

    /// Get the voxel at the given position: its voxel edit if it has one, else
    /// its voxel in the finest published chunk covering it, else what the
    /// [fill hint](VoxelWorldConfig::chunk_fill_hint) of its level-0 chunk says.
    /// The voxel will be `WorldVoxel::Unset` if none of these has a voxel there.
    pub fn get_voxel(&self, position: IVec3) -> WorldVoxel<C::MaterialIndex> {
        voxel_at(
            &self.voxel_edits.reader(),
            &self.chunk_map.get_map(),
            self.configuration.as_ref(),
            position,
        )
    }

    /// The light at the voxel at `position`, as levels from 0 to 15.
    ///
    /// It reads the light published with the level-0 chunk holding the voxel:
    /// a solid voxel reads 0, an emissive voxel its own level, and an open
    /// voxel the light that reached it. Light beside a chunk that isn't loaded,
    /// or that was published after the chunk holding the voxel started its
    /// task, can read darker than the truth, never brighter.
    ///
    /// Returns `None` when the world has no light propagation, or no published
    /// level-0 chunk holds the voxel.
    #[cfg(feature = "light_propagation")]
    pub fn light_at(&self, position: IVec3) -> Option<crate::light::LightLevels> {
        let light = self.light.as_ref()?;
        light.light_at(&self.chunk_map.get_read_lock(), position)
    }

    /// Set the voxel at the given position. It replaces the generated voxel there for
    /// as long as the world exists, and `get_voxel` returns it right away. Setting a
    /// voxel doesn't spawn a chunk: a spawned chunk that contains it remeshes to show
    /// it, and a chunk spawned later shows it from the start. When the same position
    /// is set more than once, the last voxel set wins.
    pub fn set_voxel(&mut self, position: IVec3, voxel: WorldVoxel<C::MaterialIndex>) {
        self.voxel_edits.set(position, voxel);
    }

    /// Get a sendable closure that can be used to get the voxel at the given position
    /// This is useful for spawning tasks that need to access the voxel world
    pub fn get_voxel_fn(
        &self,
    ) -> Arc<dyn Fn(IVec3) -> WorldVoxel<C::MaterialIndex> + Send + Sync> {
        let voxel_edits = self.voxel_edits.reader();
        let chunk_map = self.chunk_map.get_map();
        let configuration = self.configuration.as_ref().clone();

        Arc::new(move |position| {
            voxel_at(&voxel_edits, &chunk_map, &configuration, position)
        })
    }

    /// The published data of the chunk with the given key, or `None` if no chunk
    /// with that key is spawned or published.
    ///
    /// The level-0 chunk at a chunk position has the key
    /// `ChunkKey::new(chunk_pos, 0)`, where `chunk_pos` is
    /// `floor(voxel_position / CHUNK_SIZE)`; [`VoxelWorld::chunk_at`] finds the
    /// key of the chunk covering a voxel at any level.
    pub fn get_chunk_data(&self, key: ChunkKey) -> Option<ChunkData<C::MaterialIndex>> {
        self.chunk_map.get_map().read().unwrap().get(&key).cloned()
    }

    pub fn get_chunk_data_fn(
        &self,
    ) -> Arc<dyn Fn(ChunkKey) -> Option<ChunkData<C::MaterialIndex>> + Send + Sync> {
        let chunk_map = self.chunk_map.get_map();
        Arc::new(move |key| chunk_map.read().unwrap().get(&key).cloned())
    }

    /// Get the closes surface voxel to the given position
    /// Returns None if there is no surface voxel at or below the given position
    #[deprecated(since = "0.10.2", note = "Use raycast to find a surface instead")]
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
                && (get_voxel(above) == WorldVoxel::Unset
                    || get_voxel(above) == WorldVoxel::Air)
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
    #[deprecated(since = "0.10.2", note = "Use raycast to find a surface instead")]
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
    #[deprecated(since = "0.10.2", note = "Use raycast to find a surface instead")]
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
    /// The ray is traced through the bounds of the published chunks, reading
    /// voxels as [`get_voxel`](Self::get_voxel) does, so within those bounds a
    /// voxel no chunk covers is what its chunk's fill hint says.
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
    ///     mut cursor_evr: MessageReader<CursorMoved>,
    /// ) {
    ///     for ev in cursor_evr.read() {
    ///         // Get a ray from the cursor position into the world
    ///         let (camera, cam_gtf) = camera_info.single().unwrap();
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

            let loaded_aabb = ChunkMap::<C, C::MaterialIndex>::get_world_bounds(
                &chunk_map.read().unwrap(),
            );
            let trace_start = if p.cmplt(loaded_aabb.min.into()).any()
                || p.cmpgt(loaded_aabb.max.into()).any()
            {
                {
                    let trace_start_t = RayCast3d::from_ray(ray, f32::MAX)
                        .aabb_intersection_at(&loaded_aabb)?;
                    ray.get_point(trace_start_t)
                }
            } else {
                p
            };

            let trace_end = bounds_exit(trace_start, *d, &loaded_aabb);

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

/// Returns a tuple of the chunk position and the voxel position within the chunk.
#[inline]
pub fn get_chunk_voxel_position(position: IVec3) -> (IVec3, UVec3) {
    let chunk_position = chunk_containing(position);
    let voxel_position = (position - chunk_position * CHUNK_SIZE_I).as_uvec3() + 1;

    (chunk_position, voxel_position)
}

/// Where a ray from `start`, a point within `bounds`, leaves them. `direction`
/// must not be zero.
///
/// The ray stops at the nearest of the faces it heads towards. Measuring from
/// `start` keeps the result precise however large the bounds are.
fn bounds_exit(start: Vec3, direction: Vec3, bounds: &Aabb3d) -> Vec3 {
    let (min, max) = (Vec3::from(bounds.min), Vec3::from(bounds.max));
    let mut distance = f32::INFINITY;
    for axis in 0..3 {
        let face = if direction[axis] > 0.0 {
            max[axis]
        } else if direction[axis] < 0.0 {
            min[axis]
        } else {
            continue;
        };
        distance = distance.min((face - start[axis]) / direction[axis]);
    }
    // Rounding can leave `start` just outside the bounds.
    start + direction * distance.max(0.0)
}

/// The voxel at `position`: its voxel edit if it has one, else its voxel in the
/// published data of the finest chunk covering it, else the voxel the fill hint
/// of its level-0 chunk says is everywhere in it.
fn voxel_at<C: VoxelWorldConfig>(
    voxel_edits: &VoxelEditReader<C::MaterialIndex>,
    chunk_map: &RwLock<ChunkMapData<C::MaterialIndex>>,
    configuration: &C,
    position: IVec3,
) -> WorldVoxel<C::MaterialIndex> {
    if let Some(voxel) = voxel_edits.get(position) {
        return voxel;
    }
    let published = chunk_map
        .read()
        .unwrap()
        .finest_covering(position)
        .and_then(|chunk_data| chunk_data.get_voxel_at_world_position(position));
    match published {
        Some(voxel) => voxel,
        None => match configuration.chunk_fill_hint(ChunkKey::containing(position, 0)) {
            FillHint::Uniform(voxel) => voxel,
            FillHint::Unknown => WorldVoxel::Unset,
        },
    }
}
