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
//!   coordinates. At 16 levels each texel is an `R8Uint` byte, sky light in
//!   the low nibble and voxel light in the high one, and at 256 levels two
//!   bytes of `Rg8Uint`, sky light then voxel light (see
//!   [`slot`](super::slot)). Texel rows run along the voxels' z, so one
//!   column of the solver's bit planes is one row.
//! - **The tag**, a `MeshTag` on the chunk entity. Bit 31 says whether the
//!   chunk has a slot. With one, bits 0–23 are the slot's index. Without one,
//!   bits 0–15 hold the one sky level and voxel level every face of the chunk
//!   sees: 4 bits each at 16 levels, sky in bits 0–3 and voxel in 4–7, and 8
//!   bits each at 256 levels, sky in bits 0–7 and voxel in 8–15. Bits 28–30
//!   are always clear. The crate sets the tag on every chunk of a lit world
//!   showing its light, replacing any tag the world's own systems set.
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
//! A coarse chunk (see [`coarse`](super::coarse)) takes a slot on the same
//! rule from its own cells, and its shell holds its own padding cells. Its
//! tag with a slot also carries its LOD level in bits 24–27. It keeps a copy
//! of its slot's bytes while it holds one, so `light_at` can read its cells.
//! When it gives way to a nearer chunk it drops the copy and shows the
//! brightest level a tenth of its faces see, and it takes no slot again until
//! its chunk task runs again, since a coarse chunk that a camera nears is
//! split into finer chunks anyway.
//!
//! The render world writes queued bytes into the slot's part of the texture
//! during the frame they were extracted with, so the mesh, the tag and the
//! light show in the same frame. A slot freed and given to another chunk is
//! written in the frame that extracts the new owner's tag, after every frame
//! drawn with the old one has been submitted, so it never shows the other
//! chunk's light. The texture's `Image` asset is never changed after it's
//! made, so nothing is written at rest.
//!
//! A change of precision or slot capacity replaces the texture: see
//! [`LightRoute::replace_texture`]. Every chunk with a slot drops to its one
//! level per channel, and takes a slot in the new texture as its light lands
//! again.

use std::marker::PhantomData;
use std::sync::Arc;
use std::time::{Duration, Instant};

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

use super::coarse::takes_coarse_path;
use super::config::{LightPrecision, LightPropagation};
use super::faces::Face;
use super::levels::LightLevels;
use super::slot::{
    across, border_plane, cell_index, convert_texels, most_common, shell_cell,
    slot_bytes, texel_at, texel_bytes, texel_levels, texture_format, write_shell,
    FacesSee, Seen, ShellPlane, PLANE_TEXELS,
};
use super::solver::{ChannelLight, ChunkLight};
use crate::chunk::{Chunk, ChunkKey};
use crate::chunk::{CHUNK_SIZE_F, CHUNK_SIZE_U};
use crate::configuration::VoxelWorldConfig;

/// Texels along each side of a slot: a chunk and a shell one voxel deep.
pub(crate) const SLOT_SIDE: u32 = CHUNK_SIZE_U + 2;

/// The tag bit of a chunk with a slot.
pub(crate) const HAS_SLOT: u32 = 1 << 31;

/// The lowest of the tag bits holding a coarse chunk's LOD level when it has
/// a slot.
const LOD_SHIFT: u32 = 24;

/// The tag of a chunk with no slot showing sky light `sky` and voxel light
/// `voxel`, steps of `precision`.
pub(crate) fn tag_levels(sky: u8, voxel: u8, precision: LightPrecision) -> u32 {
    match precision {
        LightPrecision::Steps16 => sky as u32 | (voxel as u32) << 4,
        LightPrecision::Steps256 => sky as u32 | (voxel as u32) << 8,
    }
}

/// How long refilling slots may take in a frame.
const REFILL_BUDGET: Duration = Duration::from_millis(1);

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

/// A coarse chunk the route knows of.
struct CoarseChunk {
    key: ChunkKey,
    shown: bool,
    /// The sky light and voxel light its tag holds without a slot, in steps
    /// of the route's precision.
    levels: (u8, u8),
    /// Its slot's bytes: kept until it's shown, and as a copy while it holds
    /// a slot.
    bytes: Option<Box<[u8]>>,
    slot: Option<u32>,
    tag: u32,
}

impl CoarseChunk {
    fn new(key: ChunkKey, levels: (u8, u8), precision: LightPrecision) -> Self {
        Self {
            key,
            shown: false,
            levels,
            bytes: None,
            slot: None,
            tag: tag_levels(levels.0, levels.1, precision),
        }
    }
}

/// World `C`'s light texture and the slots in it. Only a lit world that shows
/// its light has one.
#[derive(Resource)]
pub(crate) struct LightRoute<C> {
    /// A texture of one texel, which the world's material binds until a chunk
    /// needs a slot.
    pub placeholder: Handle<Image>,
    texture: Option<Handle<Image>>,
    /// Textures replaced by a change of precision or slot capacity, until
    /// the world's materials stop binding them.
    released: Vec<Handle<Image>>,
    /// Whether the texture waits to be made.
    texture_wanted: bool,
    /// The precision of the texture's format, and of the levels in tags.
    precision: LightPrecision,
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
    coarse: EntityHashMap<CoarseChunk>,
    /// The chunk at each level-0 position whose light has landed.
    at: HashMap<IVec3, Entity>,
    /// Chunks needing a slot that have none.
    starved: EntityHashSet,
    /// The writes for each slot, taken by the render world each frame.
    uploads: HashMap<u32, Upload>,
    /// Tags changed for chunks, waiting to be set on them.
    changed_tags: Vec<(Entity, u32)>,
    cameras: Vec<Vec3>,
    /// The sky light, a whole level, of a chunk with no light of its own,
    /// such as a chunk above LOD level 0, or one whose light hasn't landed
    /// yet.
    unlit_sky: u8,
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
            released: Vec::new(),
            texture_wanted: false,
            precision: config.precision,
            capacity: config.slot_capacity,
            slots_per_axis: None,
            max_side: None,
            next_slot: 0,
            free: Vec::new(),
            holders: HashMap::default(),
            chunks: EntityHashMap::default(),
            coarse: EntityHashMap::default(),
            at: HashMap::default(),
            starved: EntityHashSet::default(),
            uploads: HashMap::default(),
            changed_tags: Vec::new(),
            cameras: Vec::new(),
            unlit_sky: sky,
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

    /// Makes chunks with no light of their own show sky light at the whole
    /// level `sky` from now on, after the world's sky rule changed.
    pub fn set_unlit_sky(&mut self, sky: u8) {
        self.unlit_sky = sky;
    }

    /// The tag of a chunk with no light of its own.
    fn unlit_tag(&self) -> u32 {
        let sky = self.precision.of_whole(self.unlit_sky);
        tag_levels(sky, 0, self.precision)
    }

    /// The tag of a chunk showing `levels`, steps of `from`, with no slot.
    fn flat_levels(&self, (sky, voxel): (u8, u8), from: LightPrecision) -> u32 {
        let to = self.precision;
        tag_levels(to.convert(sky, from), to.convert(voxel, from), to)
    }

    /// Replaces the texture with one of `precision`'s format holding
    /// `capacity` slots, after either changed.
    ///
    /// Every chunk with a slot drops it and shows its most common level per
    /// channel, and changed tags wait in
    /// [`take_changed_tags`](Self::take_changed_tags). The old texture waits
    /// in [`take_released`](Self::take_released) for the world's materials to
    /// stop binding it, and the new one is made only when a chunk next takes
    /// a slot. A level-0 chunk takes one again when its light lands at the
    /// new precision, or, with the precision unchanged, as slots are
    /// refilled. A coarse chunk's slot is converted to the new format and
    /// placed again.
    pub fn replace_texture(&mut self, precision: LightPrecision, capacity: u32) {
        let old = self.precision;
        self.precision = precision;
        self.capacity = capacity;
        self.released.extend(self.texture.take());
        self.texture_wanted = false;
        self.slots_per_axis = None;
        self.next_slot = 0;
        self.free.clear();
        self.holders.clear();
        self.uploads.clear();
        self.starved.clear();
        self.warned_full = false;
        let unlit = self.unlit_tag();
        let entities: Vec<Entity> = self.chunks.keys().copied().collect();
        for entity in entities {
            let chunk = &self.chunks[&entity];
            let had_slot = chunk.slot.is_some();
            let tag = match &chunk.light {
                None => unlit,
                Some(light) => {
                    let sees = chunk.sees();
                    if had_slot || sees.needs_slot() {
                        self.flat_levels(most_common(light), light.precision)
                    } else {
                        self.flat_levels(flat_tag(&sees, light), light.precision)
                    }
                }
            };
            let current = chunk
                .light
                .as_ref()
                .is_some_and(|light| light.precision == precision);
            let starved = current && (had_slot || chunk.sees().needs_slot());
            let chunk = self.chunks.get_mut(&entity).unwrap();
            let convert = |seen: Seen| match seen {
                Seen::One(level) => Seen::One(precision.convert(level, old)),
                seen => seen,
            };
            for across in &mut chunk.across {
                across.sky = convert(across.sky);
                across.voxel = convert(across.voxel);
            }
            chunk.slot = None;
            chunk.shell = [None; 6];
            chunk.bytes = None;
            if starved && chunk.shown {
                self.starved.insert(entity);
            }
            if chunk.tag != tag {
                chunk.tag = tag;
                if chunk.shown {
                    self.changed_tags.push((entity, tag));
                }
            }
        }
        let coarse: Vec<Entity> = self.coarse.keys().copied().collect();
        for entity in coarse {
            let chunk = self.coarse.get_mut(&entity).unwrap();
            chunk.levels = (
                precision.convert(chunk.levels.0, old),
                precision.convert(chunk.levels.1, old),
            );
            chunk.slot = None;
            chunk.bytes = chunk
                .bytes
                .take()
                .map(|bytes| convert_texels(&bytes, old, precision));
            let tag = tag_levels(chunk.levels.0, chunk.levels.1, precision);
            if chunk.tag != tag {
                chunk.tag = tag;
                if chunk.shown {
                    self.changed_tags.push((entity, tag));
                }
            }
            self.settle_coarse(entity);
        }
    }

    /// The textures changes of precision or slot capacity replaced, once.
    pub fn take_released(&mut self) -> Vec<Handle<Image>> {
        std::mem::take(&mut self.released)
    }

    /// Whether a replaced texture waits for the world's materials to stop
    /// binding it.
    pub fn has_released(&self) -> bool {
        !self.released.is_empty()
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
            texture_format(self.precision),
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
        let unlit = self.unlit_tag();
        // A voxel edit gives a chunk new inputs without a chunk task.
        let first = !self.chunks.get(&entity).is_some_and(|chunk| {
            chunk
                .light
                .as_ref()
                .is_some_and(|old| Arc::ptr_eq(&old.inputs, &light.inputs))
        });
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
                        across(&light.inputs, face, &neighbour, self.precision)
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
            chunk.across[back as usize] =
                across(&neighbour_light.inputs, back, &light, self.precision);
            self.settle(neighbour, Some(back));
        }
    }

    /// The tag of the chunk `entity` with `key` as its mesh is shown, placing
    /// light that landed for it meanwhile. A chunk whose light hasn't landed
    /// gets the levels of a chunk with no light. `coarse` says whether it's a
    /// coarse chunk.
    fn show(&mut self, entity: Entity, key: ChunkKey, coarse: bool) -> u32 {
        if coarse {
            let unlit = (self.precision.of_whole(self.unlit_sky), 0);
            let precision = self.precision;
            let chunk = self
                .coarse
                .entry(entity)
                .or_insert_with(|| CoarseChunk::new(key, unlit, precision));
            if !chunk.shown {
                chunk.shown = true;
                self.settle_coarse(entity);
            }
            return self.coarse[&entity].tag;
        }
        let position = key.position;
        let unlit = self.unlit_tag();
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
        if let Some(chunk) = self.coarse.remove(&entity) {
            if let Some(slot) = chunk.slot {
                self.free_slot(slot);
            }
            return;
        }
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

    /// Takes the light that landed for the coarse chunk `entity` with `key`:
    /// the sky and voxel light its tag holds without a slot, and its slot's
    /// bytes if it needs one, both of `precision`.
    pub fn coarse_landed(
        &mut self,
        entity: Entity,
        key: ChunkKey,
        levels: (u8, u8),
        precision: LightPrecision,
        bytes: Option<Box<[u8]>>,
    ) {
        let to = self.precision;
        let levels = (
            to.convert(levels.0, precision),
            to.convert(levels.1, precision),
        );
        let bytes = bytes.map(|bytes| convert_texels(&bytes, precision, to));
        let chunk = self
            .coarse
            .entry(entity)
            .or_insert_with(|| CoarseChunk::new(key, levels, to));
        chunk.levels = levels;
        chunk.bytes = bytes;
        self.settle_coarse(entity);
    }

    /// The light of the slot of the coarse chunk `entity` at `cell`, in
    /// padded cell coordinates, while it holds a slot.
    pub fn coarse_cell(&self, entity: Entity, cell: UVec3) -> Option<LightLevels> {
        let chunk = self.coarse.get(&entity)?;
        chunk.slot?;
        let (sky, voxel) =
            texel_at(chunk.bytes.as_ref()?, cell_index(cell), self.precision);
        Some(texel_levels(sky, voxel, self.precision))
    }

    /// The bytes coarse chunks keep as copies of their slots.
    pub fn coarse_copy_bytes(&self) -> usize {
        self.coarse
            .values()
            .filter(|chunk| chunk.slot.is_some())
            .filter_map(|chunk| chunk.bytes.as_ref())
            .map(|bytes| bytes.len())
            .sum()
    }

    /// Brings the shown coarse chunk `entity`'s slot and tag in line with its
    /// light. One with bytes takes a slot if it can, and otherwise drops them
    /// and shows its levels.
    fn settle_coarse(&mut self, entity: Entity) {
        let Some(chunk) = self.coarse.get(&entity) else {
            return;
        };
        if !chunk.shown {
            return;
        }
        let (key, wants_slot, held) = (chunk.key, chunk.bytes.is_some(), chunk.slot);
        let slot = match (wants_slot, held) {
            (true, Some(slot)) => Some(slot),
            (true, None) => self.take_slot(entity, self.distance_to(key.centre())),
            (false, held) => {
                if let Some(slot) = held {
                    self.free_slot(slot);
                    self.counters.given_back += 1;
                }
                None
            }
        };
        let chunk = self.coarse.get_mut(&entity).unwrap();
        chunk.slot = slot;
        let tag = match slot {
            Some(slot) => {
                let bytes = chunk
                    .bytes
                    .clone()
                    .expect("a chunk taking a slot has bytes");
                self.uploads.insert(slot, Upload::Full(bytes));
                self.counters.full_writes += 1;
                HAS_SLOT | (key.lod_level as u32) << LOD_SHIFT | slot
            }
            None => {
                chunk.bytes = None;
                tag_levels(chunk.levels.0, chunk.levels.1, self.precision)
            }
        };
        let chunk = self.coarse.get_mut(&entity).unwrap();
        if chunk.tag != tag {
            chunk.tag = tag;
            self.changed_tags.push((entity, tag));
        }
    }

    /// Whether a chunk needing a slot waits for one that is free.
    pub fn wants_refill(&self) -> bool {
        !self.starved.is_empty() && self.has_free_slot()
    }

    /// Gives free slots to the chunks waiting for one, nearest the cameras
    /// first, for as long as [`REFILL_BUDGET`] allows. Chunks left waiting
    /// get theirs in later frames.
    pub fn refill(&mut self) {
        let started = Instant::now();
        while self.wants_refill() && started.elapsed() < REFILL_BUDGET {
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
        let sees = chunk.sees();
        let (slot, position) = (chunk.slot, chunk.position);
        self.starved.remove(&entity);
        let tag = if light.precision != self.precision {
            // Light solved before a change of precision shows flat until it
            // lands again at the new one.
            if let Some(slot) = slot {
                self.chunks.get_mut(&entity).unwrap().slot = None;
                self.free_slot(slot);
            }
            self.flat_levels(most_common(&light), light.precision)
        } else if sees.needs_slot() {
            match (slot, face) {
                (Some(slot), Some(face)) => {
                    self.write_plane(entity, slot, face);
                    HAS_SLOT | slot
                }
                (Some(slot), None) => {
                    self.write_full(entity, slot, &light);
                    HAS_SLOT | slot
                }
                (None, _) => match self.take_slot(entity, self.distance(position)) {
                    Some(slot) => {
                        self.chunks.get_mut(&entity).unwrap().slot = Some(slot);
                        if face.is_some() {
                            self.counters.late_allocations += 1;
                        }
                        self.write_full(entity, slot, &light);
                        HAS_SLOT | slot
                    }
                    None => {
                        self.starved.insert(entity);
                        self.flat_levels(most_common(&light), light.precision)
                    }
                },
            }
        } else {
            if let Some(slot) = slot {
                self.chunks.get_mut(&entity).unwrap().slot = None;
                self.free_slot(slot);
                self.counters.given_back += 1;
            }
            self.flat_levels(flat_tag(&sees, &light), light.precision)
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
                let plane = border_plane(&neighbour, face.opposite(), self.precision);
                write_shell(&mut bytes, face, Some(&plane), self.precision);
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
        let plane = border_plane(&neighbour, face.opposite(), self.precision);
        let hash = Some(FixedHasher.hash_one(&plane[..]));
        let shell = &mut self.chunks.get_mut(&entity).unwrap().shell[face as usize];
        if *shell == hash {
            return;
        }
        *shell = hash;
        self.counters.plane_writes += 1;
        match self.uploads.get_mut(&slot) {
            Some(Upload::Full(bytes)) => {
                write_shell(bytes, face, Some(&plane), self.precision)
            }
            Some(Upload::Planes(planes)) => planes[face as usize] = Some(plane),
            None => {
                let mut planes: [Option<Box<ShellPlane>>; 6] = Default::default();
                planes[face as usize] = Some(plane);
                self.uploads.insert(slot, Upload::Planes(planes));
            }
        }
    }

    /// A slot for the chunk `entity`, `near` from the cameras: a free one, or
    /// one taken from the holder farthest from the cameras if it's farther
    /// than this chunk.
    fn take_slot(&mut self, entity: Entity, near: f32) -> Option<u32> {
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
            None => self.evict_for(near)?,
        };
        self.holders.insert(slot, entity);
        self.counters.allocations += 1;
        Some(slot)
    }

    /// Takes the slot of the holder farthest from the cameras, if it's
    /// farther than `near`, falling it back to flat light.
    fn evict_for(&mut self, near: f32) -> Option<u32> {
        if !self.warned_full {
            self.warned_full = true;
            warn!(
                "light propagation: every one of the {} light slots is taken, so the \
                 chunks farthest from the camera show flat light; raise slot_capacity \
                 to give them slots",
                self.capacity
            );
        }
        let (slot, holder, far) = self
            .holders
            .iter()
            .map(|(slot, holder)| (*slot, *holder, self.holder_distance(*holder)))
            .max_by(|a, b| a.2.total_cmp(&b.2))?;
        if far <= near {
            return None;
        }
        self.holders.remove(&slot);
        self.counters.evictions += 1;
        if let Some(chunk) = self.coarse.get_mut(&holder) {
            chunk.slot = None;
            chunk.bytes = None;
            chunk.tag = tag_levels(chunk.levels.0, chunk.levels.1, self.precision);
            self.changed_tags.push((holder, chunk.tag));
            return Some(slot);
        }
        let light = self.chunks[&holder].light.clone();
        let tag = light.map_or(self.unlit_tag(), |light| {
            self.flat_levels(most_common(&light), light.precision)
        });
        let chunk = self.chunks.get_mut(&holder).unwrap();
        chunk.slot = None;
        chunk.tag = tag;
        self.changed_tags.push((holder, tag));
        self.starved.insert(holder);
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
        self.distance_to((position.as_vec3() + 0.5) * CHUNK_SIZE_F)
    }

    /// How far the slot holder `holder` is from the nearest camera.
    fn holder_distance(&self, holder: Entity) -> f32 {
        match self.coarse.get(&holder) {
            Some(chunk) => self.distance_to(chunk.key.centre()),
            None => self.distance(self.chunks[&holder].position),
        }
    }

    /// How far `point` is from the nearest camera.
    fn distance_to(&self, point: Vec3) -> f32 {
        self.cameras
            .iter()
            .map(|camera| camera.distance(point))
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

    /// What its faces read, from its own voxels and across each face.
    fn sees(&self) -> FacesSee {
        self.across
            .iter()
            .fold(self.own, |sees, across| sees.join(*across))
    }
}

/// The levels of a chunk with no slot whose faces see `sees`, as sky and
/// voxel light. A channel no face reads takes the chunk's own level, if it
/// has one.
fn flat_tag(sees: &FacesSee, light: &ChunkLight) -> (u8, u8) {
    let level = |seen: Seen, channel: &ChannelLight| match (seen, channel) {
        (Seen::One(level), _) => level,
        (_, ChannelLight::Uniform(level)) => *level,
        _ => 0,
    };
    (level(sees.sky, &light.sky), level(sees.voxel, &light.voxel))
}

/// Sets the tag of a chunk of world `C` as its mesh is shown, and queues its
/// slot's bytes. Runs as the chunk gets its `Mesh3d`.
pub(crate) fn tag_chunk<C: VoxelWorldConfig>(chunk: &mut EntityCommands) {
    chunk.queue(|mut chunk: EntityWorldMut| {
        let entity = chunk.id();
        let Some((key, coarse)) = chunk.get::<Chunk<C>>().map(|chunk| {
            let key = chunk.key();
            (key, takes_coarse_path(key, chunk.data_shape()))
        }) else {
            return;
        };
        chunk.world_scope(|world| {
            world.resource_scope::<LightRoute<C>, _>(|world, mut route| {
                route.set_device(world.get_resource::<RenderDevice>());
                let tag = route.show(entity, key, coarse);
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
            texel_bytes: 1,
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
    texel_bytes: usize,
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
    // Writes for a texture since replaced would not fit the new one.
    if uploads.texture != Some(texture.id()) {
        uploads.pending.clear();
    }
    uploads.texture = Some(texture.id());
    uploads.slots_per_axis = axes;
    uploads.texel_bytes = texel_bytes(route.precision);
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
    let (axes, wide) = (uploads.slots_per_axis, uploads.texel_bytes);
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
                bytes_per_row: Some(extent.x * wide as u32),
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
                        let (origin, extent, bytes) = plane_texels(face, plane, wide);
                        write(slot, origin, extent, &bytes);
                    }
                }
            }
        }
    }
}

/// Where a shell plane on `face` lies in its slot's texels, as the region's
/// origin and extent, and its bytes in the region's order, `wide` bytes a
/// texel.
///
/// A slot's texel x, y, z is its padded cell z, x, y.
fn plane_texels(face: Face, plane: &ShellPlane, wide: usize) -> (UVec3, UVec3, Vec<u8>) {
    let texel = |u: usize, v: usize| {
        let cell = shell_cell(face, u, v);
        UVec3::new(cell.z, cell.x, cell.y)
    };
    let origin = texel(0, 0);
    let extent =
        texel(CHUNK_SIZE_U as usize - 1, CHUNK_SIZE_U as usize - 1) - origin + UVec3::ONE;
    let mut bytes = vec![0; PLANE_TEXELS * wide];
    for v in 0..CHUNK_SIZE_U as usize {
        for u in 0..CHUNK_SIZE_U as usize {
            let at = texel(u, v) - origin;
            let to =
                (at.x + at.y * extent.x + at.z * extent.x * extent.y) as usize * wide;
            let from = (u + v * CHUNK_SIZE_U as usize) * wide;
            bytes[to..to + wide].copy_from_slice(&plane[from..from + wide]);
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
        let mut plane = [0u8; PLANE_TEXELS];
        plane[3 + 5 * 32] = 7;
        for face in Face::ALL {
            let (origin, extent, bytes) = plane_texels(face, &plane, 1);
            assert_eq!(extent.element_product(), 32 * 32);
            let cell = shell_cell(face, 3, 5);
            let at = UVec3::new(cell.z, cell.x, cell.y) - origin;
            let index = (at.x + at.y * extent.x + at.z * extent.x * extent.y) as usize;
            assert_eq!(bytes[index], 7, "{face:?}");
        }
    }
}
