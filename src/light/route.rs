//! The route light takes from a chunk task to the shader: a texture of slots
//! per world, and a `MeshTag` on each chunk that the crate owns.
//!
//! Chunk meshes are shared through the mesh cache, keyed on the chunk's
//! voxels, and their positions are local to the chunk. Two chunks with the
//! same voxels can hold different light, so light can't go into a mesh's
//! vertices. It reaches the shader through two things the mesh doesn't hold:
//!
//! - **A slot** in the world's light texture, for a chunk whose faces see
//!   more than one level (see [`slot`](super::slot) for which chunks need
//!   one). A slot is 34³ texels: the chunk's 32³ voxels and a shell one voxel
//!   deep around them, so a slot's cell is a voxel of the chunk's padded local
//!   coordinates. Each texel is a byte, sky light in the low nibble and voxel
//!   light in the high one. Texel rows run along the voxels' z, so one column
//!   of the solver's bit planes is one row.
//! - **The tag**, a `MeshTag` on the chunk entity. Bit 31 says whether the
//!   chunk has a slot. With one, bits 0–23 are the slot's index. Without one,
//!   bits 0–7 hold the one sky level and voxel level every face of the chunk
//!   sees, as in a texel. Bits 28–30 are always clear. The crate sets the tag
//!   on every chunk of a lit world showing its light, replacing any tag the
//!   world's own systems set.
//!
//! A slot's shell holds copies of the six face neighbours' border planes, so
//! a face on a chunk boundary reads the light in front of it. The shell is
//! only for display: it never feeds back into light. Its edges and corners
//! stay 0, and so does a plane whose neighbour isn't loaded. When a
//! neighbour's light lands, only the plane in front of it is written again,
//! and only if it changed; the chunk doesn't relight. A neighbour that
//! retires leaves its last plane in place.
//!
//! A solve, in a chunk task or a relight task, turns the solver's bit planes
//! into a slot's bytes for a chunk whose own voxels already call for a slot.
//! When the light lands on the main thread, a chunk showing its mesh gets its
//! slot, its shell and its tag, and the bytes are queued. A chunk whose mesh
//! isn't shown yet keeps them until it is. A chunk that needed no slot takes
//! one later if a neighbour's plane stops matching its levels, and one whose
//! faces come to see one level gives its slot back.
//!
//! The texture is made when the first chunk takes a slot, so a world whose
//! chunks all see one level holds none. Its slot count is clamped to the
//! largest 3D texture the device makes. When every slot is taken, the slot
//! holder farthest from the cameras gives way to a nearer chunk, and the
//! world warns once. A chunk with no slot shows the most common level of the
//! voxels touching its solids, so it loses its shadows and its lamps' glow;
//! it takes a slot again when one comes free.
//!
//! The render world writes queued bytes into the slot's part of the texture
//! during the frame they were extracted with, so the mesh, the tag and the
//! light show in the same frame. A slot freed and given to another chunk is
//! written in the frame that extracts the new owner's tag, after every frame
//! drawn with the old one has been submitted, so it never shows the other
//! chunk's light. The texture's `Image` asset is never changed after it's
//! made, so nothing is written at rest.

use std::marker::PhantomData;
use std::sync::Arc;

use bevy::asset::RenderAssetUsages;
use bevy::ecs::entity::{EntityHashMap, EntityHashSet};
use bevy::ecs::system::EntityCommands;
use bevy::log::warn_once;
use bevy::mesh::MeshTag;
use bevy::platform::collections::HashMap;
use bevy::platform::hash::FixedHasher;
use bevy::prelude::*;
use bevy::render::{
    render_asset::RenderAssets,
    render_resource::{
        Extent3d, Origin3d, TexelCopyBufferLayout, TexelCopyTextureInfo, TextureAspect,
        TextureDimension, TextureFormat, TextureUsages,
    },
    renderer::{RenderDevice, RenderQueue},
    texture::GpuImage,
    ExtractSchedule, MainWorld, Render, RenderApp, RenderSystems,
};
use std::hash::BuildHasher;

use super::config::LightPropagation;
use super::faces::Face;
use super::slot::{
    across, border_plane, most_common, shell_cell, slot_bytes, write_shell, FacesSee,
    Seen, ShellPlane, PLANE_BYTES,
};
use super::solver::{ChannelLight, ChunkLight};
use crate::chunk::Chunk;
use crate::chunk::{CHUNK_SIZE_F, CHUNK_SIZE_U};
use crate::configuration::VoxelWorldConfig;

/// Texels along each side of a slot: a chunk and a shell one voxel deep.
pub(crate) const SLOT_SIDE: u32 = CHUNK_SIZE_U + 2;

/// The tag bit of a chunk with a slot.
pub(crate) const HAS_SLOT: u32 = 1 << 31;

/// The levels of a chunk with no slot, as the low byte of its tag.
fn tag_levels(sky: u8, voxel: u8) -> u8 {
    sky | voxel << 4
}

/// How many slots the texture holds along each of its axes, for `capacity`
/// slots on a device whose 3D textures are at most `max_side` texels on a
/// side: the smallest axis doubles, x first, until they hold them all.
///
/// Returns the slot count the texture holds, which is less than `capacity`
/// only if the device can't make a texture that large. Then each axis holds
/// as many slots as fit.
fn texture_shape(capacity: u32, max_side: Option<u32>) -> (u32, UVec3) {
    let limit = max_side.map_or(u32::MAX, |side| (side / SLOT_SIDE).max(1));
    let mut axes = UVec3::ONE;
    while axes.element_product() < capacity {
        let Some(axis) = (0..3)
            .filter(|&axis| axes[axis] * 2 <= limit)
            .min_by_key(|&axis| axes[axis])
        else {
            let axes = UVec3::splat(limit);
            return (axes.element_product().min(capacity), axes);
        };
        axes[axis] *= 2;
    }
    (capacity, axes)
}

/// Counts of what the route has done with slots, for benchmarks.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct SlotCounters {
    /// Slots given to chunks, counting late ones and evictions.
    pub allocations: u64,
    /// Slots given to chunks that had none because their faces saw one
    /// level, when a neighbour's plane stopped matching it.
    pub late_allocations: u64,
    /// Slots given back by chunks whose faces came to see one level.
    pub given_back: u64,
    /// Slots taken from a farther chunk for a nearer one.
    pub evictions: u64,
    /// Shell planes written alone, after a neighbour's light changed.
    pub plane_writes: u64,
    /// Slots written whole.
    pub full_writes: u64,
}

/// What the render world writes into a slot.
enum Upload {
    /// The whole slot.
    Full(Box<[u8]>),
    /// Some of its shell planes, by [`Face`].
    Planes([Option<Box<ShellPlane>>; 6]),
}

/// A level-0 chunk the route knows of: one whose light has landed, or which
/// shows its mesh.
struct RouteChunk {
    position: IVec3,
    shown: bool,
    light: Option<Arc<ChunkLight>>,
    /// What the chunk's faces read from its own voxels.
    own: FacesSee,
    /// The bytes its last solve made of its own voxels, until they're placed.
    bytes: Option<Box<[u8]>>,
    /// What its faces read across each face, by [`Face`].
    across: [FacesSee; 6],
    slot: Option<u32>,
    /// A hash of each shell plane in the slot, by [`Face`], or `None` for a
    /// plane of zeros.
    shell: [Option<u64>; 6],
    tag: u32,
}

/// World `C`'s light texture and the slots in it. Only a lit world that shows
/// its light has one.
#[derive(Resource)]
pub(crate) struct LightRoute<C> {
    /// A texture of one texel, which the world's material binds until a chunk
    /// needs a slot.
    pub placeholder: Handle<Image>,
    texture: Option<Handle<Image>>,
    /// Whether the texture waits to be made.
    texture_wanted: bool,
    /// The slot count, clamped to the device once the texture's shape is set.
    capacity: u32,
    slots_per_axis: Option<UVec3>,
    /// The device's largest 3D texture side, if there's a device.
    max_side: Option<u32>,
    next_slot: u32,
    /// Slots given back, the last one first.
    free: Vec<u32>,
    holders: HashMap<u32, Entity>,
    chunks: EntityHashMap<RouteChunk>,
    /// The chunk at each level-0 position whose light has landed.
    at: HashMap<IVec3, Entity>,
    /// Chunks needing a slot that have none.
    starved: EntityHashSet,
    /// The writes for each slot, taken by the render world each frame.
    uploads: HashMap<u32, Upload>,
    /// Tags changed for chunks, waiting to be set on them.
    changed_tags: Vec<(Entity, u32)>,
    cameras: Vec<Vec3>,
    /// The tag's levels for a chunk with no light of its own, such as a chunk
    /// above LOD level 0, or one whose light hasn't landed yet.
    unlit_levels: u8,
    warned_full: bool,
    pub counters: SlotCounters,
    _world: PhantomData<C>,
}

impl<C: VoxelWorldConfig> LightRoute<C> {
    /// A route for `config`, which must be clamped, with its placeholder
    /// texture added to `images`.
    pub fn new(config: &LightPropagation, sky: u8, images: &mut Assets<Image>) -> Self {
        let mut placeholder = Image::new_fill(
            Extent3d::default(),
            TextureDimension::D3,
            &[0],
            TextureFormat::R8Uint,
            RenderAssetUsages::RENDER_WORLD,
        );
        placeholder.texture_descriptor.usage = TextureUsages::TEXTURE_BINDING;
        Self {
            placeholder: images.add(placeholder),
            texture: None,
            texture_wanted: false,
            capacity: config.slot_capacity,
            slots_per_axis: None,
            max_side: None,
            next_slot: 0,
            free: Vec::new(),
            holders: HashMap::default(),
            chunks: EntityHashMap::default(),
            at: HashMap::default(),
            starved: EntityHashSet::default(),
            uploads: HashMap::default(),
            changed_tags: Vec::new(),
            cameras: Vec::new(),
            unlit_levels: tag_levels(sky, 0),
            warned_full: false,
            counters: SlotCounters::default(),
            _world: PhantomData,
        }
    }

    /// Notes the largest 3D texture side of the device, if there is one,
    /// which the texture's shape is clamped to when it's made.
    pub fn set_device(&mut self, device: Option<&RenderDevice>) {
        if self.slots_per_axis.is_none() {
            self.max_side = device.map(|device| device.limits().max_texture_dimension_3d);
        }
    }

    pub fn set_cameras(&mut self, cameras: impl Iterator<Item = Vec3>) {
        self.cameras.clear();
        self.cameras.extend(cameras);
    }

    /// Whether the texture waits to be made.
    pub fn texture_wanted(&self) -> bool {
        self.texture_wanted
    }

    /// The texture of the shape the slots need, made once.
    pub fn texture_image(&self) -> Image {
        let size = self.slots_per_axis.unwrap_or(UVec3::ONE) * SLOT_SIDE;
        let mut image = Image::new_uninit(
            Extent3d {
                width: size.x,
                height: size.y,
                depth_or_array_layers: size.z,
            },
            TextureDimension::D3,
            TextureFormat::R8Uint,
            RenderAssetUsages::default(),
        );
        image.texture_descriptor.usage =
            TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST;
        image
    }

    /// The texture a material made now binds: the light texture once it's
    /// made, or else the placeholder.
    pub fn texture(&self) -> Handle<Image> {
        self.texture
            .clone()
            .unwrap_or_else(|| self.placeholder.clone())
    }

    pub fn texture_made(&mut self, texture: Handle<Image>) {
        self.texture = Some(texture);
        self.texture_wanted = false;
    }

    /// Whether the world has warned that its slots ran out.
    #[cfg_attr(not(test), expect(dead_code, reason = "read by tests"))]
    pub fn warned_full(&self) -> bool {
        self.warned_full
    }

    /// Takes the light that landed for the chunk `entity` at the level-0
    /// position `position`: what its faces read from its own voxels, and the
    /// bytes its solve made of it, if any. `new_voxels` says whether its
    /// voxels may have changed since its last landing.
    ///
    /// A chunk showing its mesh gets its new tag and slot, and its face
    /// neighbours' shells and tags follow. Changed tags wait in
    /// [`take_changed_tags`](Self::take_changed_tags).
    pub fn light_landed(
        &mut self,
        position: IVec3,
        entity: Entity,
        light: Arc<ChunkLight>,
        own: FacesSee,
        bytes: Option<Box<[u8]>>,
        new_voxels: bool,
    ) {
        let unlit = self.unlit_levels as u32;
        let first = !self
            .chunks
            .get(&entity)
            .is_some_and(|chunk| chunk.light.is_some());
        let chunk = self
            .chunks
            .entry(entity)
            .or_insert_with(|| RouteChunk::new(position, unlit));
        chunk.light = Some(light.clone());
        chunk.own = own;
        chunk.bytes = bytes;
        self.at.insert(position, entity);
        if first || new_voxels {
            let across = Face::ALL.map(|face| {
                self.neighbour_light(position, face)
                    .map_or(FacesSee::default(), |neighbour| {
                        across(&light.inputs, face, &neighbour)
                    })
            });
            self.chunks.get_mut(&entity).unwrap().across = across;
        }
        self.settle(entity, None);

        for face in Face::ALL {
            let Some(&neighbour) = self.at.get(&(position + face.offset())) else {
                continue;
            };
            let back = face.opposite();
            let Some(chunk) = self.chunks.get_mut(&neighbour) else {
                continue;
            };
            let Some(neighbour_light) = &chunk.light else {
                continue;
            };
            chunk.across[back as usize] = across(&neighbour_light.inputs, back, &light);
            self.settle(neighbour, Some(back));
        }
    }

    /// The tag of the chunk `entity` as its mesh is shown, placing light that
    /// landed for it meanwhile. `position` is its level-0 position, or `None`
    /// for a chunk above level 0, which gets the levels of a chunk with no
    /// light, as does one whose light hasn't landed.
    fn show(&mut self, entity: Entity, position: Option<IVec3>) -> u32 {
        let Some(position) = position else {
            return self.unlit_levels as u32;
        };
        let unlit = self.unlit_levels as u32;
        let chunk = self
            .chunks
            .entry(entity)
            .or_insert_with(|| RouteChunk::new(position, unlit));
        if !chunk.shown {
            chunk.shown = true;
            self.settle(entity, None);
        }
        self.chunks[&entity].tag
    }

    /// Forgets the despawned chunk `entity`, freeing its slot. Its
    /// neighbours' shells keep its last planes.
    pub fn forget(&mut self, entity: Entity) {
        let Some(chunk) = self.chunks.remove(&entity) else {
            return;
        };
        if self.at.get(&chunk.position) == Some(&entity) {
            self.at.remove(&chunk.position);
        }
        self.starved.remove(&entity);
        if let Some(slot) = chunk.slot {
            self.free_slot(slot);
        }
    }

    /// Whether a chunk needing a slot waits for one that is free.
    pub fn wants_refill(&self) -> bool {
        !self.starved.is_empty() && self.has_free_slot()
    }

    /// Gives free slots to the chunks waiting for one, nearest the cameras
    /// first.
    pub fn refill(&mut self) {
        while self.wants_refill() {
            let nearest = self
                .starved
                .iter()
                .map(|entity| (self.distance(self.chunks[entity].position), *entity))
                .min_by(|a, b| a.0.total_cmp(&b.0))
                .map(|(_, entity)| entity);
            let Some(entity) = nearest else {
                return;
            };
            self.starved.remove(&entity);
            self.settle(entity, None);
        }
    }

    /// The tags changed since this was last called, to set on their chunks.
    pub fn take_changed_tags(&mut self) -> std::vec::Drain<'_, (Entity, u32)> {
        self.changed_tags.drain(..)
    }

    /// Brings the shown chunk `entity`'s slot and tag in line with its light.
    /// With `face`, only the plane in front of that face changed, and the
    /// slot is written again only there; otherwise it's written whole.
    fn settle(&mut self, entity: Entity, face: Option<Face>) {
        let Some(chunk) = self.chunks.get(&entity) else {
            return;
        };
        let (true, Some(light)) = (chunk.shown, chunk.light.clone()) else {
            return;
        };
        let sees = chunk
            .across
            .iter()
            .fold(chunk.own, |sees, across| sees.join(*across));
        let slot = chunk.slot;
        self.starved.remove(&entity);
        let tag = if sees.needs_slot() {
            match (slot, face) {
                (Some(slot), Some(face)) => {
                    self.write_plane(entity, slot, face);
                    HAS_SLOT | slot
                }
                (Some(slot), None) => {
                    self.write_full(entity, slot, &light);
                    HAS_SLOT | slot
                }
                (None, _) => match self.take_slot(entity) {
                    Some(slot) => {
                        if face.is_some() {
                            self.counters.late_allocations += 1;
                        }
                        self.write_full(entity, slot, &light);
                        HAS_SLOT | slot
                    }
                    None => {
                        self.starved.insert(entity);
                        let (sky, voxel) = most_common(&light);
                        tag_levels(sky, voxel) as u32
                    }
                },
            }
        } else {
            if let Some(slot) = slot {
                self.chunks.get_mut(&entity).unwrap().slot = None;
                self.free_slot(slot);
                self.counters.given_back += 1;
            }
            flat_tag(&sees, &light)
        };
        let chunk = self.chunks.get_mut(&entity).unwrap();
        if chunk.slot.is_none() {
            chunk.bytes = None;
        }
        if chunk.tag != tag {
            chunk.tag = tag;
            self.changed_tags.push((entity, tag));
        }
    }

    /// Queues the whole slot of the chunk `entity`: its own voxels' light and
    /// its neighbours' border planes.
    fn write_full(&mut self, entity: Entity, slot: u32, light: &ChunkLight) {
        let position = self.chunks[&entity].position;
        let mut bytes = self
            .chunks
            .get_mut(&entity)
            .unwrap()
            .bytes
            .take()
            .unwrap_or_else(|| slot_bytes(light));
        let mut shell = [None; 6];
        for face in Face::ALL {
            if let Some(neighbour) = self.neighbour_light(position, face) {
                let plane = border_plane(&neighbour, face.opposite());
                write_shell(&mut bytes, face, Some(&plane));
                shell[face as usize] = Some(FixedHasher.hash_one(&plane[..]));
            }
        }
        self.chunks.get_mut(&entity).unwrap().shell = shell;
        self.uploads.insert(slot, Upload::Full(bytes));
        self.counters.full_writes += 1;
    }

    /// Queues the shell plane of the chunk `entity` in front of `face`, if
    /// the neighbour's border plane there changed.
    fn write_plane(&mut self, entity: Entity, slot: u32, face: Face) {
        let position = self.chunks[&entity].position;
        let Some(neighbour) = self.neighbour_light(position, face) else {
            return;
        };
        let plane = border_plane(&neighbour, face.opposite());
        let hash = Some(FixedHasher.hash_one(&plane[..]));
        let shell = &mut self.chunks.get_mut(&entity).unwrap().shell[face as usize];
        if *shell == hash {
            return;
        }
        *shell = hash;
        self.counters.plane_writes += 1;
        match self.uploads.get_mut(&slot) {
            Some(Upload::Full(bytes)) => write_shell(bytes, face, Some(&plane)),
            Some(Upload::Planes(planes)) => planes[face as usize] = Some(plane),
            None => {
                let mut planes: [Option<Box<ShellPlane>>; 6] = Default::default();
                planes[face as usize] = Some(plane);
                self.uploads.insert(slot, Upload::Planes(planes));
            }
        }
    }

    /// A slot for the chunk `entity`: a free one, or one taken from the
    /// holder farthest from the cameras if it's farther than this chunk.
    fn take_slot(&mut self, entity: Entity) -> Option<u32> {
        if self.slots_per_axis.is_none() {
            let (capacity, axes) = texture_shape(self.capacity, self.max_side);
            if capacity < self.capacity {
                warn_once!(
                    "light propagation: slot_capacity {} is more than this device's 3D \
                     textures hold, using {capacity}",
                    self.capacity
                );
            }
            self.capacity = capacity;
            self.slots_per_axis = Some(axes);
            self.texture_wanted = true;
        }
        let slot = match self.free.pop() {
            Some(slot) => slot,
            None if self.next_slot < self.capacity => {
                self.next_slot += 1;
                self.next_slot - 1
            }
            None => self.evict_for(entity)?,
        };
        self.holders.insert(slot, entity);
        self.chunks.get_mut(&entity).unwrap().slot = Some(slot);
        self.counters.allocations += 1;
        Some(slot)
    }

    /// Takes the slot of the holder farthest from the cameras, if it's
    /// farther than the chunk `entity`, falling it back to flat light.
    fn evict_for(&mut self, entity: Entity) -> Option<u32> {
        if !self.warned_full {
            self.warned_full = true;
            warn!(
                "light propagation: every one of the {} light slots is taken, so the \
                 chunks farthest from the camera show flat light; raise slot_capacity \
                 to give them slots",
                self.capacity
            );
        }
        let near = self.distance(self.chunks[&entity].position);
        let (slot, holder, far) = self
            .holders
            .iter()
            .map(|(slot, holder)| {
                (*slot, *holder, self.distance(self.chunks[holder].position))
            })
            .max_by(|a, b| a.2.total_cmp(&b.2))?;
        if far <= near {
            return None;
        }
        self.holders.remove(&slot);
        let chunk = self.chunks.get_mut(&holder).unwrap();
        chunk.slot = None;
        let light = chunk.light.clone();
        let tag = light.map_or(self.unlit_levels as u32, |light| {
            let (sky, voxel) = most_common(&light);
            tag_levels(sky, voxel) as u32
        });
        chunk.tag = tag;
        self.changed_tags.push((holder, tag));
        self.starved.insert(holder);
        self.counters.evictions += 1;
        Some(slot)
    }

    fn free_slot(&mut self, slot: u32) {
        self.holders.remove(&slot);
        self.free.push(slot);
    }

    fn has_free_slot(&self) -> bool {
        !self.free.is_empty() || self.next_slot < self.capacity
    }

    /// The light of the chunk across `face` from the level-0 position
    /// `position`, if its light has landed.
    fn neighbour_light(&self, position: IVec3, face: Face) -> Option<Arc<ChunkLight>> {
        let entity = self.at.get(&(position + face.offset()))?;
        self.chunks.get(entity)?.light.clone()
    }

    /// How far the middle of the level-0 chunk at `position` is from the
    /// nearest camera.
    fn distance(&self, position: IVec3) -> f32 {
        let middle = (position.as_vec3() + 0.5) * CHUNK_SIZE_F;
        self.cameras
            .iter()
            .map(|camera| camera.distance(middle))
            .fold(f32::INFINITY, f32::min)
    }

    /// The bytes queued whole for `slot` and not yet taken by the render
    /// world, with any planes written since.
    #[cfg(test)]
    pub fn queued_upload(&self, slot: u32) -> Option<&[u8]> {
        match self.uploads.get(&slot)? {
            Upload::Full(bytes) => Some(bytes),
            Upload::Planes(_) => None,
        }
    }
}

impl RouteChunk {
    fn new(position: IVec3, tag: u32) -> Self {
        Self {
            position,
            shown: false,
            light: None,
            own: FacesSee::default(),
            bytes: None,
            across: [FacesSee::default(); 6],
            slot: None,
            shell: [None; 6],
            tag,
        }
    }
}

/// The tag of a chunk with no slot whose faces see `sees`. A channel no face
/// reads takes the chunk's own level, if it has one.
fn flat_tag(sees: &FacesSee, light: &ChunkLight) -> u32 {
    let level = |seen: Seen, channel: &ChannelLight| match (seen, channel) {
        (Seen::One(level), _) => level,
        (_, ChannelLight::Uniform(level)) => *level,
        _ => 0,
    };
    tag_levels(level(sees.sky, &light.sky), level(sees.voxel, &light.voxel)) as u32
}

/// Sets the tag of a chunk of world `C` as its mesh is shown, and queues its
/// slot's bytes. Runs as the chunk gets its `Mesh3d`.
pub(crate) fn tag_chunk<C: VoxelWorldConfig>(chunk: &mut EntityCommands) {
    chunk.queue(|mut chunk: EntityWorldMut| {
        let entity = chunk.id();
        let position = chunk
            .get::<Chunk<C>>()
            .map(|chunk| chunk.key())
            .filter(|key| key.lod_level == 0)
            .map(|key| key.position);
        chunk.world_scope(|world| {
            world.resource_scope::<LightRoute<C>, _>(|world, mut route| {
                route.set_device(world.get_resource::<RenderDevice>());
                let tag = route.show(entity, position);
                for (changed, tag) in route.take_changed_tags() {
                    if let Ok(mut changed) = world.get_entity_mut(changed) {
                        changed.insert(MeshTag::new(tag));
                    }
                }
                world.entity_mut(entity).insert(MeshTag::new(tag));
            });
        });
    });
}

/// Sets up writing world `C`'s queued slots into its texture, in the render
/// world. An app with no renderer gets nothing.
pub(crate) fn add_render_systems<C: VoxelWorldConfig>(app: &mut App) {
    let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
        return;
    };
    render_app
        .insert_resource(RenderUploads::<C> {
            texture: None,
            slots_per_axis: UVec3::ONE,
            pending: Vec::new(),
            _world: PhantomData,
        })
        .add_systems(ExtractSchedule, extract_uploads::<C>)
        .add_systems(
            Render,
            write_uploads::<C>.in_set(RenderSystems::PrepareResources),
        );
}

/// World `C`'s slot writes waiting to be made, in the render world.
#[derive(Resource)]
struct RenderUploads<C> {
    texture: Option<AssetId<Image>>,
    slots_per_axis: UVec3,
    /// Each slot and what to write into it, oldest first.
    pending: Vec<(u32, Upload)>,
    _world: PhantomData<C>,
}

/// Takes the slot writes queued in the main world this frame.
fn extract_uploads<C: VoxelWorldConfig>(
    mut main_world: ResMut<MainWorld>,
    mut uploads: ResMut<RenderUploads<C>>,
) {
    let Some(mut route) = main_world.get_resource_mut::<LightRoute<C>>() else {
        return;
    };
    if route.uploads.is_empty() {
        return;
    }
    let (Some(texture), Some(axes)) = (&route.texture, route.slots_per_axis) else {
        return;
    };
    uploads.texture = Some(texture.id());
    uploads.slots_per_axis = axes;
    uploads.pending.extend(route.uploads.drain());
}

/// Writes each pending slot write into its part of the texture, once the
/// texture exists on the GPU.
fn write_uploads<C: VoxelWorldConfig>(
    mut uploads: ResMut<RenderUploads<C>>,
    images: Res<RenderAssets<GpuImage>>,
    queue: Res<RenderQueue>,
) {
    if uploads.pending.is_empty() {
        return;
    }
    let Some(image) = uploads.texture.and_then(|texture| images.get(texture)) else {
        return;
    };
    let axes = uploads.slots_per_axis;
    let write = |slot: u32, origin: UVec3, extent: UVec3, bytes: &[u8]| {
        let slot_origin = UVec3::new(
            slot % axes.x,
            slot / axes.x % axes.y,
            slot / (axes.x * axes.y),
        ) * SLOT_SIDE;
        let origin = slot_origin + origin;
        queue.write_texture(
            TexelCopyTextureInfo {
                texture: &image.texture,
                mip_level: 0,
                origin: Origin3d {
                    x: origin.x,
                    y: origin.y,
                    z: origin.z,
                },
                aspect: TextureAspect::All,
            },
            bytes,
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(extent.x),
                rows_per_image: Some(extent.y),
            },
            Extent3d {
                width: extent.x,
                height: extent.y,
                depth_or_array_layers: extent.z,
            },
        );
    };
    for (slot, upload) in std::mem::take(&mut uploads.pending) {
        match upload {
            Upload::Full(bytes) => {
                write(slot, UVec3::ZERO, UVec3::splat(SLOT_SIDE), &bytes)
            }
            Upload::Planes(planes) => {
                for face in Face::ALL {
                    if let Some(plane) = &planes[face as usize] {
                        let (origin, extent, bytes) = plane_texels(face, plane);
                        write(slot, origin, extent, &bytes);
                    }
                }
            }
        }
    }
}

/// Where a shell plane on `face` lies in its slot's texels, as the region's
/// origin and extent, and its bytes in the region's order.
///
/// A slot's texel x, y, z is its padded cell z, x, y.
fn plane_texels(face: Face, plane: &ShellPlane) -> (UVec3, UVec3, Vec<u8>) {
    let texel = |u: usize, v: usize| {
        let cell = shell_cell(face, u, v);
        UVec3::new(cell.z, cell.x, cell.y)
    };
    let origin = texel(0, 0);
    let extent =
        texel(CHUNK_SIZE_U as usize - 1, CHUNK_SIZE_U as usize - 1) - origin + UVec3::ONE;
    let mut bytes = vec![0; PLANE_BYTES];
    for v in 0..CHUNK_SIZE_U as usize {
        for u in 0..CHUNK_SIZE_U as usize {
            let at = texel(u, v) - origin;
            bytes[(at.x + at.y * extent.x + at.z * extent.x * extent.y) as usize] =
                plane[u + v * CHUNK_SIZE_U as usize];
        }
    }
    (origin, extent, bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_texture_shape_follows_the_slot_count() {
        assert_eq!(texture_shape(1024, None), (1024, UVec3::new(16, 8, 8)));
        assert_eq!(
            texture_shape(1024, None).1 * SLOT_SIDE,
            UVec3::new(544, 272, 272)
        );
        assert_eq!(texture_shape(1, None), (1, UVec3::ONE));
        assert!(texture_shape(1000, None).1.element_product() >= 1000);
    }

    #[test]
    fn the_slot_count_is_clamped_to_the_devices_3d_textures() {
        // WebGL2 guarantees 256 texels a side, which holds 7 slots.
        assert_eq!(texture_shape(1024, Some(256)), (343, UVec3::splat(7)));
        assert_eq!(
            texture_shape(1024, Some(2048)),
            (1024, UVec3::new(16, 8, 8))
        );
        assert_eq!(texture_shape(40, Some(256)), (40, UVec3::new(4, 4, 4)));
    }

    #[test]
    fn a_shell_plane_covers_one_layer_of_its_slot() {
        let mut plane = [0u8; PLANE_BYTES];
        plane[3 + 5 * 32] = 7;
        for face in Face::ALL {
            let (origin, extent, bytes) = plane_texels(face, &plane);
            assert_eq!(extent.element_product(), 32 * 32);
            let cell = shell_cell(face, 3, 5);
            let at = UVec3::new(cell.z, cell.x, cell.y) - origin;
            let index = (at.x + at.y * extent.x + at.z * extent.x * extent.y) as usize;
            assert_eq!(bytes[index], 7, "{face:?}");
        }
    }
}
