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
//!   bits each at 256 levels, sky in bits 0–7 and voxel in 8–15. Bit 30 is
//!   set on every chunk of a sub-grid and never on a world chunk, and bits
//!   28 and 29 are always clear. Every reader of a field masks it to its
//!   width, so bit 30 never reads as part of the slot, the LOD level or the
//!   levels. The crate sets the tag on every chunk of a lit world showing
//!   its light, replacing any tag the world's own systems set.
//!
//! A slot's shell holds copies of the six face neighbours' border planes, so
//! a face on a chunk boundary reads the light in front of it. The shell is
//! only for display: it never feeds back into light. Its edges and corners
//! stay 0, and so does a plane whose neighbour isn't loaded. A sub-grid's
//! chunk copies only the neighbours on its own grid, never a world chunk or
//! another sub-grid's chunk at the same position, and a plane toward a
//! position where its grid has no chunk stays 0. When a
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
//! world warns once. A sub-grid's chunk is measured at its grid's last pose. A chunk with no slot shows the most common level of the
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
//!
//! With `sub_grids`, a world with a sub-grid also keeps a tag map beside the
//! light texture, from a world chunk position to the tag shown there. Every
//! change to a tag goes through one place, which notes the position for the
//! tag map's upkeep, and the tag map's writes go to the render world in the
//! same extract as the slots'. A chunk at a position the tag map holds is
//! kept, which places its light whether or not its mesh shows, under a
//! wider slot rule that reads all of its open voxels: see
//! [`tag_map`](super::sub_grid::tag_map).
//!
//! # References
//!
//! - [all-is-cubes' light texture](https://github.com/kpreid/all-is-cubes/blob/main/all-is-cubes-gpu/src/light_texture.rs):
//!   light kept in a 3D texture apart from meshes, written in parts as it
//!   changes, and read by position in the fragment shader.

use std::marker::PhantomData;
use std::sync::Arc;
use std::time::Duration;

use bevy::asset::RenderAssetUsages;
use bevy::ecs::entity::EntityHashMap;
use bevy::ecs::system::EntityCommands;
use bevy::log::warn_once;
use bevy::mesh::MeshTag;
use bevy::platform::collections::{HashMap, HashSet};
use bevy::platform::hash::FixedHasher;
use bevy::platform::time::Instant;
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
use super::fill::{cell_level, CellFill, FillCells, CELL};
use super::kernel::Plane;
use super::levels::LightLevels;
use super::slot::{
    across, border_plane, border_sees, cell_index, convert_texels, most_common,
    most_common_open, open_voxels_see, shell_cell, slot_bytes, texel_at, texel_bytes,
    texel_levels, texture_format, write_shell, FacesSee, Seen, ShellPlane, PLANE_TEXELS,
};
use super::solver::{ChannelLight, ChunkLight};
use super::view::{chunk_centre, world_position};
use crate::chunk::CHUNK_SIZE_U;
use crate::chunk::{Chunk, ChunkKey};
use crate::configuration::VoxelWorldConfig;
use crate::grid::GridKey;
#[cfg(feature = "sub_grids")]
use {
    super::sub_grid::tag_map::columns::{HintedColumns, Hinting},
    super::sub_grid::tag_map::{PendingWrites, Placed, TagMap, TagMapWrites},
    super::view::in_world,
    crate::chunk::MAX_LOD_LEVEL,
    crate::configuration::LodLevel,
    crate::grid::GridId,
};

/// Texels along each side of a slot: a chunk and a shell one voxel deep.
pub(crate) const SLOT_SIDE: u32 = CHUNK_SIZE_U + 2;

/// The tag bit of a chunk with a slot.
pub(crate) const HAS_SLOT: u32 = 1 << 31;

/// The tag bit of a sub-grid's chunk.
pub(crate) const SUB_GRID: u32 = 1 << 30;

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
    /// The most slots held at once.
    pub peak_held: u64,
}

/// What the render world writes into a slot.
enum Upload {
    /// The whole slot.
    Full(Box<[u8]>),
    /// Some of its shell planes, by [`Face`].
    Planes([Option<Box<ShellPlane>>; 6]),
}

/// How the route names a level-0 chunk: by its entity, or by its grid key
/// for a chunk with no entity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum RouteId {
    Entity(Entity),
    #[cfg_attr(not(test), expect(dead_code, reason = "made by tests"))]
    Key(GridKey),
}

/// A chunk whose tag the route sets: a level-0 chunk or a coarse one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Tagged {
    Chunk(RouteId),
    Coarse(Entity),
}

/// What a chunk's tag shows: its slot, or one sky level and voxel level in
/// steps of the route's precision.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Shows {
    Slot(u32),
    Levels(u8, u8),
}

/// A level-0 chunk the route knows of: one whose light has landed, which
/// shows its mesh, or which is kept.
struct RouteChunk {
    key: GridKey,
    shown: bool,
    /// Whether its light is placed, with a slot if it needs one, whether or
    /// not its mesh shows.
    kept: bool,
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
    /// A coarse chunk with `key` showing `levels`, whose tag
    /// [`retag`](LightRoute::retag) sets.
    fn new(key: ChunkKey, levels: (u8, u8)) -> Self {
        Self {
            key,
            shown: false,
            levels,
            bytes: None,
            slot: None,
            tag: 0,
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
    holders: HashMap<u32, Tagged>,
    chunks: HashMap<RouteId, RouteChunk>,
    coarse: EntityHashMap<CoarseChunk>,
    /// The level-0 chunk at each grid key whose light has landed.
    at: HashMap<GridKey, RouteId>,
    /// The light of each light entry, which the shells of the chunks beside
    /// it read: see [`hinted`](super::hinted).
    entries: HashMap<GridKey, Arc<ChunkLight>>,
    /// Level-0 chunks needing a slot that have none.
    starved: HashSet<RouteId>,
    /// The writes for each slot, taken by the render world each frame.
    uploads: HashMap<u32, Upload>,
    /// Tags changed for chunks, waiting to be set on them.
    changed_tags: Vec<(Entity, u32)>,
    cameras: Vec<Vec3>,
    /// Each live sub-grid's last pose against the world's root, which places
    /// its chunks for the farthest-from-cameras rule.
    #[cfg(feature = "sub_grids")]
    poses: HashMap<GridId, GlobalTransform>,
    /// The sky light, a whole level, of a chunk with no light of its own,
    /// such as a chunk above LOD level 0, or one whose light hasn't landed
    /// yet.
    unlit_sky: u8,
    warned_full: bool,
    pub counters: SlotCounters,
    /// The cells of the world's sky fill field, which a kept chunk of air
    /// with no solid in its padding shows where they're brighter than its
    /// own light.
    fill_cells: FillCells,
    /// The tag map, once the world has a sub-grid, with what the route
    /// keeps to answer it.
    #[cfg(feature = "sub_grids")]
    tag_map: Option<TagMapping>,
    _world: PhantomData<C>,
}

/// A tag map, and what the route keeps to give the tag at each of its
/// positions: see [`tag_map`](super::sub_grid::tag_map).
#[cfg(feature = "sub_grids")]
struct TagMapping {
    map: TagMap,
    shown: ShownChunks,
    /// The kept chunks of air whose light reads the sky fill field's cells,
    /// with the cells their tags and slots were last made from.
    filled: HashMap<RouteId, Option<Arc<CellFill>>>,
    /// The column answers of the hinted air the map holds.
    columns: HintedColumns,
}

/// The world chunks showing their tags, by position, for a tag map.
#[cfg(feature = "sub_grids")]
#[derive(Default)]
struct ShownChunks {
    /// The level-0 world chunk showing its mesh at each position, or else
    /// the one kept there while the tag map holds the position.
    level_0: HashMap<IVec3, RouteId>,
    /// The coarse world chunk showing its mesh at each key.
    coarse: HashMap<ChunkKey, Entity>,
    /// How many of `coarse` are at each LOD level.
    coarse_levels: [u32; MAX_LOD_LEVEL as usize + 1],
    /// Level-0 positions whose tag may have changed since the last upkeep.
    touched: HashSet<IVec3>,
    /// Coarse chunks whose tag may have changed since the last upkeep.
    touched_coarse: HashSet<ChunkKey>,
}

#[cfg(feature = "sub_grids")]
impl ShownChunks {
    /// Notes that the coarse chunk `entity` with `key` shows its mesh.
    fn coarse_shown(&mut self, key: ChunkKey, entity: Entity) {
        if self.coarse.insert(key, entity).is_none() {
            self.coarse_levels[key.lod_level as usize] += 1;
        }
        self.touched_coarse.insert(key);
    }

    /// Notes that the coarse chunk `entity` with `key` is gone.
    fn coarse_gone(&mut self, key: ChunkKey, entity: Entity) {
        if self.coarse.get(&key) == Some(&entity) {
            self.coarse.remove(&key);
            self.coarse_levels[key.lod_level as usize] -= 1;
            self.touched_coarse.insert(key);
        }
    }
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
            chunks: HashMap::default(),
            coarse: EntityHashMap::default(),
            at: HashMap::default(),
            entries: HashMap::default(),
            starved: HashSet::default(),
            uploads: HashMap::default(),
            changed_tags: Vec::new(),
            cameras: Vec::new(),
            #[cfg(feature = "sub_grids")]
            poses: HashMap::default(),
            unlit_sky: sky,
            warned_full: false,
            counters: SlotCounters::default(),
            fill_cells: FillCells::default(),
            #[cfg(feature = "sub_grids")]
            tag_map: None,
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

    /// Makes kept chunks read the cells of the world's sky fill field from
    /// `cells`.
    pub fn set_fill_cells(&mut self, cells: FillCells) {
        self.fill_cells = cells;
    }

    /// What a chunk with no light of its own shows.
    fn unlit(&self) -> Shows {
        Shows::Levels(self.precision.of_whole(self.unlit_sky), 0)
    }

    /// What a chunk with no slot showing `levels`, steps of `from`, shows.
    fn flat_levels(&self, (sky, voxel): (u8, u8), from: LightPrecision) -> Shows {
        let to = self.precision;
        Shows::Levels(to.convert(sky, from), to.convert(voxel, from))
    }

    /// Sets the tag of `chunk` to show `shows`. Every tag the route gives a
    /// chunk is set here. A changed tag of a chunk showing its mesh waits in
    /// [`take_changed_tags`](Self::take_changed_tags).
    fn retag(&mut self, chunk: Tagged, shows: Shows) {
        let precision = self.precision;
        let levels = |sky, voxel| tag_levels(sky, voxel, precision);
        let (tag, shown, entity) = match chunk {
            Tagged::Chunk(id) => {
                let Some(chunk) = self.chunks.get_mut(&id) else {
                    return;
                };
                let mut tag = match shows {
                    Shows::Slot(slot) => HAS_SLOT | slot,
                    Shows::Levels(sky, voxel) => levels(sky, voxel),
                };
                if !chunk.key.is_world() {
                    tag |= SUB_GRID;
                }
                if chunk.tag == tag {
                    return;
                }
                chunk.tag = tag;
                #[cfg(feature = "sub_grids")]
                if let (true, Some(mapping)) = (chunk.placed(), self.tag_map.as_mut()) {
                    mapping.shown.touched.extend(world_position(&chunk.key));
                }
                let entity = match id {
                    RouteId::Entity(entity) => Some(entity),
                    RouteId::Key(_) => None,
                };
                (tag, chunk.shown, entity)
            }
            Tagged::Coarse(entity) => {
                let Some(chunk) = self.coarse.get_mut(&entity) else {
                    return;
                };
                let tag = match shows {
                    Shows::Slot(slot) => {
                        HAS_SLOT | (chunk.key.lod_level as u32) << LOD_SHIFT | slot
                    }
                    Shows::Levels(sky, voxel) => levels(sky, voxel),
                };
                if chunk.tag == tag {
                    return;
                }
                chunk.tag = tag;
                #[cfg(feature = "sub_grids")]
                if let (true, Some(mapping)) = (chunk.shown, self.tag_map.as_mut()) {
                    mapping.shown.touched_coarse.insert(chunk.key);
                }
                (tag, chunk.shown, Some(entity))
            }
        };
        if let (true, Some(entity)) = (shown, entity) {
            self.changed_tags.push((entity, tag));
        }
    }

    /// Adds the level-0 chunk `id` at `key`, showing the levels of a chunk
    /// with no light, unless the route knows of it.
    fn add_chunk(&mut self, id: RouteId, key: GridKey) {
        if self.chunks.contains_key(&id) {
            return;
        }
        self.chunks.insert(id, RouteChunk::new(key));
        self.retag(Tagged::Chunk(id), self.unlit());
    }

    /// Adds the coarse chunk `entity` with `key`, showing `levels`, steps of
    /// the route's precision, unless the route knows of it.
    fn add_coarse(&mut self, entity: Entity, key: ChunkKey, levels: (u8, u8)) {
        if self.coarse.contains_key(&entity) {
            return;
        }
        self.coarse.insert(entity, CoarseChunk::new(key, levels));
        self.retag(Tagged::Coarse(entity), Shows::Levels(levels.0, levels.1));
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
        let unlit = self.unlit();
        let ids: Vec<RouteId> = self.chunks.keys().copied().collect();
        for id in ids {
            let chunk = &self.chunks[&id];
            let had_slot = chunk.slot.is_some();
            let current = chunk
                .light
                .as_ref()
                .is_some_and(|light| light.precision == precision);
            let (shows, needs_slot) = match self.placed_light(id) {
                None => (unlit, false),
                Some((_, sees)) if had_slot || sees.needs_slot() => {
                    (self.most_common_shows(id), sees.needs_slot())
                }
                Some((light, sees)) => (
                    self.flat_levels(flat_tag(&sees, &light), light.precision),
                    false,
                ),
            };
            let starved = current && (had_slot || needs_slot);
            let chunk = self.chunks.get_mut(&id).unwrap();
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
            if starved && chunk.placed() {
                self.starved.insert(id);
            }
            self.retag(Tagged::Chunk(id), shows);
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
            let (sky, voxel) = chunk.levels;
            self.retag(Tagged::Coarse(entity), Shows::Levels(sky, voxel));
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

    /// Places each live sub-grid's chunks at its pose in `grids`, given as
    /// entity, id and pose, for the farthest-from-cameras rule from now on.
    /// Its cost grows with the live sub-grids.
    #[cfg(feature = "sub_grids")]
    pub fn set_poses(&mut self, grids: Vec<(Entity, GridId, GlobalTransform)>) {
        self.poses.clear();
        self.poses
            .extend(grids.into_iter().map(|(_, grid, pose)| (grid, pose)));
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

    /// Takes the light that landed for the level-0 chunk `entity` at `key`:
    /// what its faces read from its own voxels, and the bytes its solve made
    /// of it, if any. `new_voxels` says whether its voxels may have changed
    /// since its last landing.
    ///
    /// A chunk showing its mesh gets its new tag and slot, and its face
    /// neighbours' shells and tags follow. Changed tags wait in
    /// [`take_changed_tags`](Self::take_changed_tags).
    pub fn light_landed(
        &mut self,
        key: GridKey,
        entity: Entity,
        light: Arc<ChunkLight>,
        own: FacesSee,
        bytes: Option<Box<[u8]>>,
        new_voxels: bool,
    ) {
        let id = RouteId::Entity(entity);
        self.landed(id, key, light, own, bytes, new_voxels);
    }

    /// Takes the light that landed for the level-0 chunk `id` at `key`, as
    /// [`light_landed`](Self::light_landed) does.
    fn landed(
        &mut self,
        id: RouteId,
        key: GridKey,
        light: Arc<ChunkLight>,
        own: FacesSee,
        bytes: Option<Box<[u8]>>,
        new_voxels: bool,
    ) {
        // A voxel edit gives a chunk new inputs without a chunk task.
        let first = !self.chunks.get(&id).is_some_and(|chunk| {
            chunk
                .light
                .as_ref()
                .is_some_and(|old| Arc::ptr_eq(&old.inputs, &light.inputs))
        });
        self.add_chunk(id, key);
        let chunk = self.chunks.get_mut(&id).unwrap();
        chunk.light = Some(light.clone());
        chunk.own = own;
        chunk.bytes = bytes;
        #[cfg(not(feature = "sub_grids"))]
        self.at.insert(key, id);
        #[cfg(feature = "sub_grids")]
        {
            let before = self.at.insert(key, id);
            self.tag_map_landed(id, key, before);
        }
        self.entries.remove(&key);
        if first || new_voxels {
            let across = Face::ALL.map(|face| {
                self.neighbour_light(key, face)
                    .map_or(FacesSee::default(), |neighbour| {
                        across(&light.inputs, face, &neighbour, self.precision)
                    })
            });
            self.chunks.get_mut(&id).unwrap().across = across;
        }
        self.settle(id, None);
        self.neighbours_follow(key, &light);
    }

    /// Makes the level-0 chunk `id` at `key` place its light while `kept`,
    /// whether or not its mesh shows: it takes a slot if its light needs
    /// one, and the slot's bytes are queued. A chunk with no entity that is
    /// no longer kept is forgotten, and one with an entity whose mesh isn't
    /// shown gives its slot back.
    ///
    /// A kept chunk takes a slot unless every one of its open voxels, and
    /// every open voxel its slot's shell copies from its face neighbours,
    /// holds one level per channel. Without a slot it shows the most common
    /// level among its open voxels. A chunk showing its mesh that is no
    /// longer kept goes back to the rule in [`slot`](super::slot).
    #[cfg_attr(
        not(any(test, feature = "sub_grids")),
        expect(dead_code, reason = "called by the tag map")
    )]
    fn keep(&mut self, id: RouteId, key: GridKey, kept: bool) {
        if kept {
            self.add_chunk(id, key);
        }
        let Some(chunk) = self.chunks.get_mut(&id) else {
            return;
        };
        if chunk.kept == kept {
            return;
        }
        chunk.kept = kept;
        let shown = chunk.shown;
        #[cfg(feature = "sub_grids")]
        if let (Some(mapping), Some(position)) =
            (self.tag_map.as_mut(), world_position(&key))
        {
            if kept && !shown {
                mapping.shown.level_0.entry(position).or_insert(id);
                mapping.shown.touched.insert(position);
            }
            if !kept {
                mapping.filled.remove(&id);
            }
        }
        if kept || shown {
            self.settle(id, None);
            return;
        }
        if let RouteId::Key(_) = id {
            self.forget_chunk(id);
            return;
        }
        let chunk = self.chunks.get_mut(&id).unwrap();
        let slot = chunk.slot.take();
        chunk.bytes = None;
        if let Some(slot) = slot {
            self.free_slot(slot);
        }
        self.starved.remove(&id);
        let shows = self.most_common_shows(id);
        self.retag(Tagged::Chunk(id), shows);
        #[cfg(feature = "sub_grids")]
        self.tag_map_forget(id, key);
    }

    /// Takes the light that landed for the light entry at `key`, which
    /// the chunks beside it read through their shells, as they read a
    /// chunk's.
    pub fn entry_landed(&mut self, key: GridKey, light: Arc<ChunkLight>) {
        if self.at.contains_key(&key) {
            return;
        }
        self.entries.insert(key, light.clone());
        #[cfg(feature = "sub_grids")]
        self.tag_map_touch(key);
        self.neighbours_follow(key, &light);
    }

    /// Forgets the light entry at `key`. The shells beside it keep its
    /// last planes, as they keep a despawned chunk's.
    pub fn entry_dropped(&mut self, key: GridKey) {
        if self.entries.remove(&key).is_some() {
            #[cfg(feature = "sub_grids")]
            self.tag_map_touch(key);
        }
    }

    /// Brings what the face neighbours on its grid of the chunk or light
    /// entry at `key` read across their faces in line with its new `light`,
    /// and their shells and tags with it.
    fn neighbours_follow(&mut self, key: GridKey, light: &Arc<ChunkLight>) {
        for face in Face::ALL {
            let Some(&neighbour) = self.at.get(&(key + face.offset())) else {
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
                across(&neighbour_light.inputs, back, light, self.precision);
            self.settle(neighbour, Some(back));
        }
    }

    /// The tag of the chunk `entity` with `key` as its mesh is shown, placing
    /// light that landed for it meanwhile. A chunk whose light hasn't landed
    /// gets the levels of a chunk with no light. `coarse` says whether it's a
    /// coarse chunk.
    fn show(&mut self, entity: Entity, key: GridKey, coarse: bool) -> u32 {
        if coarse {
            let unlit = (self.precision.of_whole(self.unlit_sky), 0);
            self.add_coarse(entity, key.key(), unlit);
            let chunk = self.coarse.get_mut(&entity).unwrap();
            if !chunk.shown {
                chunk.shown = true;
                #[cfg(feature = "sub_grids")]
                if let Some(mapping) = self.tag_map.as_mut() {
                    mapping.shown.coarse_shown(key.key(), entity);
                }
                self.settle_coarse(entity);
            }
            return self.coarse[&entity].tag;
        }
        let id = RouteId::Entity(entity);
        self.add_chunk(id, key);
        let chunk = self.chunks.get_mut(&id).unwrap();
        if !chunk.shown {
            chunk.shown = true;
            #[cfg(feature = "sub_grids")]
            if let (Some(mapping), Some(position)) =
                (self.tag_map.as_mut(), world_position(&key))
            {
                mapping.shown.level_0.insert(position, id);
                mapping.shown.touched.insert(position);
            }
            self.settle(id, None);
        }
        self.chunks[&id].tag
    }

    /// Forgets the despawned chunk `entity`, freeing its slot. Its
    /// neighbours' shells keep its last planes.
    pub fn forget(&mut self, entity: Entity) {
        if let Some(chunk) = self.coarse.remove(&entity) {
            if let Some(slot) = chunk.slot {
                self.free_slot(slot);
            }
            #[cfg(feature = "sub_grids")]
            if let (true, Some(mapping)) = (chunk.shown, self.tag_map.as_mut()) {
                mapping.shown.coarse_gone(chunk.key, entity);
            }
            return;
        }
        self.forget_chunk(RouteId::Entity(entity));
    }

    /// Forgets the level-0 chunk `id`, freeing its slot.
    fn forget_chunk(&mut self, id: RouteId) {
        let Some(chunk) = self.chunks.remove(&id) else {
            return;
        };
        if self.at.get(&chunk.key) == Some(&id) {
            self.at.remove(&chunk.key);
        }
        self.starved.remove(&id);
        if let Some(slot) = chunk.slot {
            self.free_slot(slot);
        }
        #[cfg(feature = "sub_grids")]
        self.tag_map_forget(id, chunk.key);
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
        self.add_coarse(entity, key, levels);
        let chunk = self.coarse.get_mut(&entity).unwrap();
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
            (true, None) => {
                self.take_slot(Tagged::Coarse(entity), self.distance_to(key.centre()))
            }
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
        let shows = match slot {
            Some(slot) => {
                let bytes = chunk
                    .bytes
                    .clone()
                    .expect("a chunk taking a slot has bytes");
                self.uploads.insert(slot, Upload::Full(bytes));
                self.counters.full_writes += 1;
                Shows::Slot(slot)
            }
            None => {
                chunk.bytes = None;
                Shows::Levels(chunk.levels.0, chunk.levels.1)
            }
        };
        self.retag(Tagged::Coarse(entity), shows);
    }

    /// How many chunks need a slot and have none.
    pub fn starved_count(&self) -> usize {
        self.starved.len()
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
                .map(|id| (self.distance(self.chunks[id].key), *id))
                .min_by(|a, b| a.0.total_cmp(&b.0))
                .map(|(_, id)| id);
            let Some(id) = nearest else {
                return;
            };
            self.starved.remove(&id);
            self.settle(id, None);
        }
    }

    /// The tags changed since this was last called, to set on their chunks.
    pub fn take_changed_tags(&mut self) -> std::vec::Drain<'_, (Entity, u32)> {
        self.changed_tags.drain(..)
    }

    /// Brings the level-0 chunk `id`'s slot and tag in line with its light,
    /// if it shows its mesh or is kept. With `face`, only the plane in front
    /// of that face changed, and the slot is written again only there;
    /// otherwise it's written whole.
    fn settle(&mut self, id: RouteId, face: Option<Face>) {
        let Some(chunk) = self.chunks.get(&id) else {
            return;
        };
        if !chunk.placed() {
            return;
        }
        let (slot, key) = (chunk.slot, chunk.key);
        let Some((light, sees)) = self.placed_light(id) else {
            return;
        };
        #[cfg(feature = "sub_grids")]
        self.note_fill_cells(id);
        self.starved.remove(&id);
        let shows = if light.precision != self.precision {
            // Light solved before a change of precision shows flat until it
            // lands again at the new one.
            if let Some(slot) = slot {
                self.chunks.get_mut(&id).unwrap().slot = None;
                self.free_slot(slot);
            }
            self.most_common_shows(id)
        } else if sees.needs_slot() {
            match (slot, face) {
                (Some(slot), Some(face)) => {
                    self.write_plane(id, slot, face);
                    Shows::Slot(slot)
                }
                (Some(slot), None) => {
                    self.write_full(id, slot, &light);
                    Shows::Slot(slot)
                }
                (None, _) => {
                    match self.take_slot(Tagged::Chunk(id), self.distance(key)) {
                        Some(slot) => {
                            self.chunks.get_mut(&id).unwrap().slot = Some(slot);
                            if face.is_some() {
                                self.counters.late_allocations += 1;
                            }
                            self.write_full(id, slot, &light);
                            Shows::Slot(slot)
                        }
                        None => {
                            self.starved.insert(id);
                            self.most_common_shows(id)
                        }
                    }
                }
            }
        } else {
            if let Some(slot) = slot {
                self.chunks.get_mut(&id).unwrap().slot = None;
                self.free_slot(slot);
                self.counters.given_back += 1;
            }
            self.flat_levels(flat_tag(&sees, &light), light.precision)
        };
        let chunk = self.chunks.get_mut(&id).unwrap();
        if chunk.slot.is_none() {
            chunk.bytes = None;
        }
        self.retag(Tagged::Chunk(id), shows);
    }

    /// The light the level-0 chunk `id` shows, and what the voxels its slot
    /// rule reads hold, once its light has landed. A kept chunk reads all of
    /// its open voxels and its shell's, and shows its light as
    /// [`kept_light`](Self::kept_light) gives it. Any other chunk reads the
    /// voxels [`slot`](super::slot) describes.
    fn placed_light(&self, id: RouteId) -> Option<(Arc<ChunkLight>, FacesSee)> {
        let chunk = self.chunks.get(&id)?;
        let light = chunk.light.as_ref()?;
        if !chunk.kept {
            return Some((light.clone(), chunk.sees()));
        }
        let light = self.kept_light(chunk.key, light);
        let (sky, voxel) = open_voxels_see(&light);
        let mut sees = FacesSee {
            sky,
            voxel,
            lamp: chunk.own.lamp,
        };
        for face in Face::ALL {
            if sees.needs_slot() {
                break;
            }
            if let Some(neighbour) = self.neighbour_light(chunk.key, face) {
                let (sky, voxel) =
                    border_sees(&neighbour, face.opposite(), self.precision);
                sees = sees.join(FacesSee {
                    sky,
                    voxel,
                    lamp: false,
                });
            }
        }
        Some((light, sees))
    }

    /// What the level-0 chunk `id` shows with no slot while its light needs
    /// one: the most common level per channel among the voxels its slot rule
    /// reads, or the levels of a chunk with no light if its light hasn't
    /// landed.
    fn most_common_shows(&self, id: RouteId) -> Shows {
        let chunk = &self.chunks[&id];
        let Some(light) = &chunk.light else {
            return self.unlit();
        };
        if chunk.kept {
            let light = self.kept_light(chunk.key, light);
            self.flat_levels(most_common_open(&light), light.precision)
        } else {
            self.flat_levels(most_common(light), light.precision)
        }
    }

    /// The light a kept chunk at `key` lit by `light` shows. A chunk of air
    /// with no solid in its padding shows, at each voxel, the larger of its
    /// own sky light and that of the sky fill field's cell holding the voxel,
    /// as `light_at` reads it. Any other chunk shows its own light.
    fn kept_light(&self, key: GridKey, light: &Arc<ChunkLight>) -> Arc<ChunkLight> {
        let position = world_position(&key).filter(|_| light.inputs.reads_fill_cells());
        let cells = self.fill_cells.read();
        match position.and_then(|position| cells.get(&position)) {
            Some(fill) => with_fill_cells(light, fill),
            None => light.clone(),
        }
    }

    /// Notes which sky fill field cells the kept chunk `id` was settled with,
    /// if its light reads them, so that upkeep settles it again when they
    /// change: see [`follow_fill_cells`](Self::follow_fill_cells).
    #[cfg(feature = "sub_grids")]
    fn note_fill_cells(&mut self, id: RouteId) {
        let chunk = &self.chunks[&id];
        let reads = chunk.kept
            && chunk
                .light
                .as_ref()
                .is_some_and(|light| light.inputs.reads_fill_cells());
        let position = world_position(&chunk.key);
        let Some(mapping) = self.tag_map.as_mut() else {
            return;
        };
        match (reads, position) {
            (true, Some(position)) => {
                let cells = self.fill_cells.read().get(&position).cloned();
                mapping.filled.insert(id, cells);
            }
            _ => {
                mapping.filled.remove(&id);
            }
        }
    }

    /// Settles again each kept chunk whose sky fill field cells changed since
    /// it was last settled. A chunk of air with no solid in its padding takes
    /// no sky fill in its solve, so no relight of it follows the cells.
    #[cfg(feature = "sub_grids")]
    fn follow_fill_cells(&mut self) {
        let Some(mapping) = self.tag_map.as_ref() else {
            return;
        };
        if mapping.filled.is_empty() {
            return;
        }
        let changed: Vec<RouteId> = {
            let cells = self.fill_cells.read();
            mapping
                .filled
                .iter()
                .filter(|(id, used)| {
                    let now = self
                        .chunks
                        .get(*id)
                        .and_then(|chunk| world_position(&chunk.key))
                        .and_then(|position| cells.get(&position));
                    match (now, used) {
                        (Some(now), Some(used)) => !Arc::ptr_eq(now, used),
                        (None, None) => false,
                        _ => true,
                    }
                })
                .map(|(id, _)| *id)
                .collect()
        };
        for id in changed {
            self.settle(id, None);
        }
    }

    /// Queues the whole slot of the level-0 chunk `id`: its own voxels' light
    /// and its neighbours' border planes.
    fn write_full(&mut self, id: RouteId, slot: u32, light: &ChunkLight) {
        let key = self.chunks[&id].key;
        let mut bytes = self
            .chunks
            .get_mut(&id)
            .unwrap()
            .bytes
            .take()
            .unwrap_or_else(|| slot_bytes(light));
        let mut shell = [None; 6];
        for face in Face::ALL {
            if let Some(neighbour) = self.neighbour_light(key, face) {
                let plane = border_plane(&neighbour, face.opposite(), self.precision);
                write_shell(&mut bytes, face, Some(&plane), self.precision);
                shell[face as usize] = Some(FixedHasher.hash_one(&plane[..]));
            }
        }
        self.chunks.get_mut(&id).unwrap().shell = shell;
        self.uploads.insert(slot, Upload::Full(bytes));
        self.counters.full_writes += 1;
    }

    /// Queues the shell plane of the level-0 chunk `id` in front of `face`,
    /// if the neighbour's border plane there changed.
    fn write_plane(&mut self, id: RouteId, slot: u32, face: Face) {
        let key = self.chunks[&id].key;
        let Some(neighbour) = self.neighbour_light(key, face) else {
            return;
        };
        let plane = border_plane(&neighbour, face.opposite(), self.precision);
        let hash = Some(FixedHasher.hash_one(&plane[..]));
        let shell = &mut self.chunks.get_mut(&id).unwrap().shell[face as usize];
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

    /// A slot for `chunk`, `near` from the cameras: a free one, or one taken
    /// from the holder farthest from the cameras if it's farther than this
    /// chunk.
    fn take_slot(&mut self, chunk: Tagged, near: f32) -> Option<u32> {
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
        self.holders.insert(slot, chunk);
        self.counters.allocations += 1;
        self.counters.peak_held = self.counters.peak_held.max(self.holders.len() as u64);
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
        match holder {
            Tagged::Coarse(entity) => {
                let chunk = self.coarse.get_mut(&entity).unwrap();
                chunk.slot = None;
                chunk.bytes = None;
                let (sky, voxel) = chunk.levels;
                self.retag(holder, Shows::Levels(sky, voxel));
            }
            Tagged::Chunk(id) => {
                let shows = self.most_common_shows(id);
                self.chunks.get_mut(&id).unwrap().slot = None;
                self.retag(holder, shows);
                self.starved.insert(id);
            }
        }
        Some(slot)
    }

    fn free_slot(&mut self, slot: u32) {
        self.holders.remove(&slot);
        self.free.push(slot);
    }

    fn has_free_slot(&self) -> bool {
        !self.free.is_empty() || self.next_slot < self.capacity
    }

    /// The light of the chunk across `face` from the level-0 chunk at `key`,
    /// on its grid, if its light has landed.
    fn neighbour_light(&self, key: GridKey, face: Face) -> Option<Arc<ChunkLight>> {
        let beside = key + face.offset();
        match self.at.get(&beside) {
            Some(entity) => self.chunks.get(entity)?.light.clone(),
            None => self.entries.get(&beside).cloned(),
        }
    }

    /// How far the middle of the level-0 chunk at `key` is from the
    /// nearest camera, with a sub-grid's chunk placed at its grid's last
    /// pose: see [`chunk_centre`]. The chunks of a sub-grid with no pose
    /// noted yet are placed as the world's.
    fn distance(&self, key: GridKey) -> f32 {
        #[cfg(feature = "sub_grids")]
        if let Some(pose) = self.poses.get(&key.grid()) {
            return self.distance_to(chunk_centre(key, pose));
        }
        self.distance_to(chunk_centre(key, &GlobalTransform::IDENTITY))
    }

    /// How far the slot holder `holder` is from the nearest camera.
    fn holder_distance(&self, holder: Tagged) -> f32 {
        match holder {
            Tagged::Coarse(entity) => self.distance_to(self.coarse[&entity].key.centre()),
            Tagged::Chunk(id) => self.distance(self.chunks[&id].key),
        }
    }

    /// How far `point` is from the nearest camera.
    fn distance_to(&self, point: Vec3) -> f32 {
        self.cameras
            .iter()
            .map(|camera| camera.distance(point))
            .fold(f32::INFINITY, f32::min)
    }

    /// Brings the tag map in line with `bodies`, each sub-grid placed, and
    /// with the chunks' tags as they are now. Positions of hinted air read
    /// as `hinting` says, and take the answers of walks that finished.
    /// The tag map is made, with its texture added to `images`, once there
    /// is a sub-grid. Returns how many walks finished.
    ///
    /// The level-0 world chunk at each position the map comes to hold is
    /// kept, under the slot rule [`keep`](Self::keep) describes, and one at
    /// a position it stops holding is no longer kept, so its slot is freed in
    /// the frame its entry leaves. Changed tags of chunks showing their meshes
    /// wait in [`take_changed_tags`](Self::take_changed_tags).
    ///
    /// Its cost grows with the positions whose tags changed since the last
    /// upkeep, with what [`TagMap::place`] and [`TagMap::write_entries`]
    /// cost, and with the chunks of air near sub-grids whose light varies,
    /// whose slots' bytes are made from the light the world view stores.
    #[cfg(feature = "sub_grids")]
    pub fn upkeep_tag_map(
        &mut self,
        bodies: &[Placed],
        hinting: &Hinting,
        images: &mut Assets<Image>,
    ) -> usize {
        if self.tag_map.is_none() && bodies.is_empty() {
            return 0;
        }
        let started = Instant::now();
        if self.tag_map.is_none() {
            self.tag_map = Some(self.new_tag_mapping(images));
        }
        let placement = self.tag_map.as_mut().unwrap().map.place(bodies);
        // Slots freed first go to the chunks kept next.
        for position in &placement.removed {
            self.tag_map_holds(*position, false);
        }
        for position in &placement.added {
            self.tag_map_holds(*position, true);
        }
        self.follow_fill_cells();
        let mut mapping = self.tag_map.take().unwrap();
        mapping.columns.follow(hinting.generation);
        mapping.columns.release(placement.removed.iter().copied());
        let mut touched: Vec<IVec3> = Vec::new();
        let map = &mapping.map;
        let finished = mapping
            .columns
            .collect(|position| map.holds(position), &mut touched);
        touched.extend(mapping.shown.touched.drain());
        for key in mapping.shown.touched_coarse.drain() {
            let (first, side) = (key.position, key.extent());
            if side.pow(3) as usize <= mapping.map.len() {
                for x in 0..side {
                    for y in 0..side {
                        for z in 0..side {
                            touched.push(first + IVec3::new(x, y, z));
                        }
                    }
                }
            } else {
                let last = first + side;
                touched.extend(mapping.map.positions().filter(|position| {
                    position.cmpge(first).all() && position.cmplt(last).all()
                }));
            }
        }
        let nothing_known = tag_levels(self.precision.of_whole(15), 0, self.precision);
        let TagMapping {
            mut map,
            shown,
            filled,
            mut columns,
        } = mapping;
        let answer = |position: IVec3| {
            self.tag_at(&shown, &mut columns, hinting, position, nothing_known)
        };
        map.write_entries(
            placement.added,
            touched.into_iter(),
            answer,
            nothing_known,
            images,
        );
        map.counters.upkeep_nanos += started.elapsed().as_nanos() as u64;
        self.tag_map = Some(TagMapping {
            map,
            shown,
            filled,
            columns,
        });
        finished
    }

    /// Whether a position of hinted air in the tag map waits for a walk.
    #[cfg(feature = "sub_grids")]
    pub fn wants_hinted_walks(&self) -> bool {
        self.tag_map
            .as_ref()
            .is_some_and(|mapping| mapping.columns.has_wanted())
    }

    /// How many walks of hinted air in the tag map are running.
    #[cfg(feature = "sub_grids")]
    pub fn hinted_walks_running(&self) -> usize {
        self.tag_map
            .as_ref()
            .map_or(0, |mapping| mapping.columns.running())
    }

    /// Starts walks for up to `count` positions of hinted air in the tag
    /// map that wait for one, each in the task `spawn` makes for it.
    #[cfg(feature = "sub_grids")]
    pub fn start_hinted_walks(
        &mut self,
        count: usize,
        spawn: impl FnMut(IVec3) -> bevy::tasks::Task<bool>,
    ) {
        if let Some(mapping) = self.tag_map.as_mut() {
            mapping.columns.start(count, spawn);
        }
    }

    /// Keeps the level-0 world chunk at `position` while the tag map `held`
    /// the position, if its light has landed.
    #[cfg(feature = "sub_grids")]
    fn tag_map_holds(&mut self, position: IVec3, held: bool) {
        let key = in_world(position);
        if let Some(&id) = self.at.get(&key) {
            self.keep(id, key, held);
        }
    }

    /// Keeps the level-0 chunk `id`, whose light has just landed at `key`,
    /// if the tag map holds its position, and stops keeping `before`, the
    /// chunk whose light landed there last, if that was another chunk. Its
    /// light is settled after this.
    #[cfg(feature = "sub_grids")]
    fn tag_map_landed(&mut self, id: RouteId, key: GridKey, before: Option<RouteId>) {
        let (Some(mapping), Some(position)) =
            (self.tag_map.as_mut(), world_position(&key))
        else {
            return;
        };
        if mapping.map.holds(position) {
            let chunk = self.chunks.get_mut(&id).unwrap();
            chunk.kept = true;
            if !chunk.shown {
                mapping.shown.level_0.entry(position).or_insert(id);
                mapping.shown.touched.insert(position);
            }
        }
        if let Some(before) = before.filter(|before| *before != id) {
            self.keep(before, key, false);
        }
    }

    /// A tag map for a world that has just got its first sub-grid, with the
    /// chunks showing their meshes now.
    #[cfg(feature = "sub_grids")]
    fn new_tag_mapping(&self, images: &mut Assets<Image>) -> TagMapping {
        let mut mapping = TagMapping {
            map: TagMap::new(images),
            shown: ShownChunks::default(),
            filled: HashMap::default(),
            columns: HintedColumns::default(),
        };
        for (id, chunk) in &self.chunks {
            if let (true, Some(position)) = (chunk.shown, world_position(&chunk.key)) {
                mapping.shown.level_0.insert(position, *id);
            }
        }
        for (entity, chunk) in &self.coarse {
            if chunk.shown {
                mapping.shown.coarse_shown(chunk.key, *entity);
            }
        }
        mapping
    }

    /// The tag the screen shows at the level-0 world chunk position
    /// `position`: the level-0 chunk's there, or else the finest coarse
    /// chunk's covering it. Hinted air that `hinting` holds shows the most
    /// common level among its light entry's open voxels, or else its column
    /// answer from `columns`, or `nothing_known` until that lands. Anywhere
    /// else it's `nothing_known`.
    #[cfg(feature = "sub_grids")]
    fn tag_at(
        &self,
        shown: &ShownChunks,
        columns: &mut HintedColumns,
        hinting: &Hinting,
        position: IVec3,
        nothing_known: u32,
    ) -> u32 {
        if let Some(chunk) = shown
            .level_0
            .get(&position)
            .and_then(|id| self.chunks.get(id))
        {
            return chunk.tag;
        }
        for (level, count) in shown.coarse_levels.iter().enumerate() {
            if *count == 0 {
                continue;
            }
            let key = ChunkKey::containing_chunk(position, level as LodLevel);
            if let Some(chunk) = shown
                .coarse
                .get(&key)
                .and_then(|entity| self.coarse.get(entity))
            {
                return chunk.tag;
            }
        }
        if !hinting.holds(position) {
            return nothing_known;
        }
        let to = self.precision;
        if let Some(light) = self.entries.get(&in_world(position)) {
            let (sky, voxel) = most_common_open(light);
            let from = light.precision;
            return tag_levels(to.convert(sky, from), to.convert(voxel, from), to);
        }
        let dark = tag_levels(0, 0, to);
        if !hinting.sky {
            return dark;
        }
        match columns.answer(position) {
            Some(false) => dark,
            Some(true) | None => nothing_known,
        }
    }

    /// Notes for the tag map that the tag at the world chunk at `key` may
    /// have changed.
    #[cfg(feature = "sub_grids")]
    fn tag_map_touch(&mut self, key: GridKey) {
        if let (Some(mapping), Some(position)) =
            (self.tag_map.as_mut(), world_position(&key))
        {
            mapping.shown.touched.insert(position);
        }
    }

    /// Notes for the tag map that the level-0 chunk `id` at `key` is gone,
    /// or no longer shows its tag there. Another chunk at the same key
    /// showing its mesh, or else one kept there, takes its place.
    #[cfg(feature = "sub_grids")]
    fn tag_map_forget(&mut self, id: RouteId, key: GridKey) {
        let (Some(mapping), Some(position)) =
            (self.tag_map.as_mut(), world_position(&key))
        else {
            return;
        };
        mapping.filled.remove(&id);
        if mapping.shown.level_0.get(&position) != Some(&id) {
            return;
        }
        mapping.shown.level_0.remove(&position);
        mapping.shown.touched.insert(position);
        let other = self
            .chunks
            .iter()
            .filter(|(other, chunk)| **other != id && chunk.placed() && chunk.key == key)
            .max_by_key(|(_, chunk)| chunk.shown)
            .map(|(other, _)| *other);
        if let Some(other) = other {
            mapping.shown.level_0.insert(position, other);
        }
    }

    /// The tag map, once the world has a sub-grid.
    #[cfg(feature = "sub_grids")]
    pub fn tag_map(&self) -> Option<&TagMap> {
        self.tag_map.as_ref().map(|mapping| &mapping.map)
    }

    /// The tag map's writes waiting since the render world last took them.
    #[cfg(feature = "sub_grids")]
    pub fn take_tag_map_writes(&mut self) -> Option<TagMapWrites> {
        self.tag_map.as_mut()?.map.take_writes()
    }

    /// How many slots are held by chunks kept for the tag map that don't
    /// show a mesh: world chunks of air near sub-grids.
    #[cfg(feature = "sub_grids")]
    pub fn kept_unshown_slots(&self) -> usize {
        self.chunks
            .values()
            .filter(|chunk| chunk.kept && !chunk.shown && chunk.slot.is_some())
            .count()
    }

    /// How many slots chunks hold.
    #[cfg(all(test, feature = "sub_grids"))]
    pub fn slots_held(&self) -> usize {
        self.holders.len()
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
    /// A chunk at `key` with no light, whose tag
    /// [`retag`](LightRoute::retag) sets.
    fn new(key: GridKey) -> Self {
        Self {
            key,
            shown: false,
            kept: false,
            light: None,
            own: FacesSee::default(),
            bytes: None,
            across: [FacesSee::default(); 6],
            slot: None,
            shell: [None; 6],
            tag: 0,
        }
    }

    /// Whether its light is placed: it shows its mesh, or it's kept.
    fn placed(&self) -> bool {
        self.shown || self.kept
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

/// `light` with the sky light at each voxel raised to that of the sky fill
/// field's cell holding it in `fill`, where the cell's is larger. The same
/// light, shared, when no cell's is.
///
/// Its cost grows with the chunk's voxels.
fn with_fill_cells(light: &Arc<ChunkLight>, fill: &CellFill) -> Arc<ChunkLight> {
    let precision = light.precision;
    let cells: [u8; 64] = std::array::from_fn(|bit| {
        let cell = UVec3::new(bit as u32 % 4, bit as u32 / 4 % 4, bit as u32 / 16);
        precision.of_whole(cell_level(fill, cell * CELL as u32))
    });
    if cells.iter().all(|level| *level == 0) {
        return light.clone();
    }
    let side = CHUNK_SIZE_U as usize;
    let mut planes: Vec<Plane> = vec![[0; 32 * 32]; precision.bits()];
    let mut raised = false;
    for y in 0..side {
        for x in 0..side {
            let column = x + y * side;
            for z in 0..side {
                let stored = light.sky.at(x, y, z);
                let cell = cells[x / CELL + y / CELL * 4 + z / CELL * 16];
                raised |= cell > stored;
                let level = stored.max(cell);
                for (bit, plane) in planes.iter_mut().enumerate() {
                    plane[column] |= ((level >> bit & 1) as u32) << z;
                }
            }
        }
    }
    if !raised {
        return light.clone();
    }
    Arc::new(ChunkLight {
        inputs: light.inputs.clone(),
        sky: ChannelLight::Planes(planes.into_boxed_slice()),
        voxel: light.voxel.clone(),
        precision,
    })
}

/// Sets the tag of a chunk of world `C` as its mesh is shown, and queues its
/// slot's bytes. Runs as the chunk gets its `Mesh3d`.
pub(crate) fn tag_chunk<C: VoxelWorldConfig>(chunk: &mut EntityCommands) {
    chunk.queue(|mut chunk: EntityWorldMut| {
        let entity = chunk.id();
        let Some((key, coarse)) = chunk.get::<Chunk<C>>().map(|chunk| {
            let key = chunk.grid_key();
            (key, takes_coarse_path(key.key(), chunk.data_shape()))
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
            #[cfg(feature = "sub_grids")]
            tag_map: PendingWrites::default(),
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
    /// The tag map's writes.
    #[cfg(feature = "sub_grids")]
    tag_map: PendingWrites,
    _world: PhantomData<C>,
}

/// Takes the slot writes and the tag map's writes queued in the main world
/// this frame, in the extract that takes the chunks' tags.
fn extract_uploads<C: VoxelWorldConfig>(
    mut main_world: ResMut<MainWorld>,
    mut uploads: ResMut<RenderUploads<C>>,
) {
    let Some(mut route) = main_world.get_resource_mut::<LightRoute<C>>() else {
        return;
    };
    #[cfg(feature = "sub_grids")]
    if let Some(writes) = route.take_tag_map_writes() {
        uploads.tag_map.extend(writes);
    }
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

/// Writes each pending slot write into its part of the texture, and each
/// pending tag map write into the tag map's texture, once the texture exists
/// on the GPU.
fn write_uploads<C: VoxelWorldConfig>(
    mut uploads: ResMut<RenderUploads<C>>,
    images: Res<RenderAssets<GpuImage>>,
    queue: Res<RenderQueue>,
) {
    #[cfg(feature = "sub_grids")]
    uploads.tag_map.write(&images, &queue);
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
    use crate::configuration::DefaultWorld;
    #[cfg(feature = "sub_grids")]
    use crate::grid::GridId;
    use crate::light::inputs::{LightInputs, Occupancy};
    use crate::light::kernel::Plane;
    use crate::light::sky::ColumnAnswer;
    use crate::light::slot::own_faces_see;
    use crate::light::view::in_world;

    fn route() -> LightRoute<DefaultWorld> {
        let mut images = Assets::<Image>::default();
        LightRoute::new(&LightPropagation::default(), 15, &mut images)
    }

    /// The light of a chunk with one solid voxel, at its middle, with full
    /// sky light above the middle and none below, so its faces see two sky
    /// levels and it needs a slot.
    fn light_needing_a_slot() -> (Arc<ChunkLight>, FacesSee) {
        let precision = LightPropagation::default().precision;
        let mut occupancy = Occupancy::Open;
        occupancy.set(16, 16, 16, true);
        occupancy.normalize();
        let above: Plane = [!0 << 16; 32 * 32];
        let light = ChunkLight {
            inputs: Arc::new(LightInputs {
                occupancy,
                emissive: Vec::new(),
                columns: ColumnAnswer::none(),
                ring: false,
            }),
            sky: ChannelLight::Planes(vec![above; precision.bits()].into_boxed_slice()),
            voxel: ChannelLight::Uniform(0),
            precision,
        };
        let own = own_faces_see(&light);
        assert!(own.needs_slot());
        (Arc::new(light), own)
    }

    #[test]
    fn a_world_chunks_distance_is_from_its_middle() {
        let mut route = route();
        route.set_cameras([Vec3::new(48.0, -48.0, 26.0)].into_iter());
        assert_eq!(route.distance(in_world(IVec3::new(1, -2, 0))), 10.0);
    }

    #[test]
    fn a_kept_chunk_takes_a_slot_with_no_mesh() {
        let mut route = route();
        let (light, own) = light_needing_a_slot();
        let world = in_world(IVec3::ZERO);
        let entity = Entity::from_raw_u32(1).unwrap();
        route.light_landed(world, entity, light.clone(), own, None, true);
        // A chunk with no entity, far from the world chunk.
        let far = in_world(IVec3::new(5, 0, 0));
        let id = RouteId::Key(far);
        route.landed(id, far, light.clone(), own, None, true);
        assert!(route.holders.is_empty());

        route.keep(id, far, true);

        let tag = route.chunks[&id].tag;
        assert_ne!(tag & HAS_SLOT, 0);
        let queued = route.queued_upload(tag & 0xFF_FFFF).expect("bytes queued");
        // It has no neighbours, so its shell is all 0.
        assert_eq!(queued, &slot_bytes(&light)[..]);
        assert_eq!(route.take_changed_tags().count(), 0);

        // The world chunk takes a slot only once its mesh shows.
        assert_eq!(route.holders.len(), 1);
        let shown = route.show(entity, world, false);
        assert_ne!(shown & HAS_SLOT, 0);
        assert!(route.queued_upload(shown & 0xFF_FFFF).is_some());

        route.keep(id, far, false);

        assert!(!route.chunks.contains_key(&id));
        assert_eq!(route.holders.len(), 1);
    }

    #[cfg(feature = "sub_grids")]
    #[test]
    fn the_tag_map_answers_a_coarse_chunks_tag_wherever_no_finer_chunk_shows() {
        let mut route = route();
        let mut images = Assets::<Image>::default();
        let coarse = Entity::from_raw_u32(1).unwrap();
        let key = ChunkKey::new(IVec3::ZERO, 2);
        let bytes = vec![0; SLOT_SIDE.pow(3) as usize].into_boxed_slice();
        route.coarse_landed(coarse, key, (15, 0), LightPrecision::Steps16, Some(bytes));
        let coarse_tag = route.show(coarse, GridKey::world(key), true);
        assert_eq!(coarse_tag & HAS_SLOT, HAS_SLOT);
        assert_eq!(coarse_tag >> LOD_SHIFT & 0xF, 2);
        let fine = Entity::from_raw_u32(2).unwrap();
        let fine_tag = route.show(fine, in_world(IVec3::X), false);
        assert_ne!(fine_tag, coarse_tag);

        // Padded, the box runs from -1 to 160 along x: chunks -1 to 4.
        let pose = GlobalTransform::from_xyz(0.0, 10.0, 10.0);
        let body = Placed {
            grid: GridId::new(1),
            pose,
            bounds: Some((IVec3::ZERO, IVec3::new(159, 1, 1))),
        };
        route.upkeep_tag_map(&[body], &Hinting::none(), &mut images);

        let map = route.tag_map().unwrap();
        assert_eq!(map.len(), 6);
        for x in -1..=4 {
            let expected = match x {
                1 => fine_tag,
                0..=3 => coarse_tag,
                _ => 15,
            };
            assert_eq!(map.answer(IVec3::new(x, 0, 0)), Some(expected), "{x}");
        }

        // The coarse chunk gives its slot back, and its tag changes.
        route.coarse_landed(coarse, key, (9, 0), LightPrecision::Steps16, None);
        route.upkeep_tag_map(&[body], &Hinting::none(), &mut images);
        let flat = tag_levels(9, 0, LightPrecision::Steps16);
        let map = route.tag_map().unwrap();
        assert_eq!(map.answer(IVec3::new(2, 0, 0)), Some(flat));
        assert_eq!(map.answer(IVec3::X), Some(fine_tag));

        // Once it's gone, nothing is known where it was.
        route.forget(coarse);
        route.upkeep_tag_map(&[body], &Hinting::none(), &mut images);
        assert_eq!(
            route.tag_map().unwrap().answer(IVec3::new(2, 0, 0)),
            Some(15)
        );
    }

    /// The light of a chunk of air at 16 levels whose sky is 9 at every
    /// voxel but one, which holds 8, and whose voxel light is 0.
    #[cfg(feature = "sub_grids")]
    fn air_lit_voxel_by_voxel() -> Arc<ChunkLight> {
        let mut sky: Vec<Plane> = (0..4)
            .map(|bit| [if 9 >> bit & 1 == 1 { !0 } else { 0 }; 32 * 32])
            .collect();
        // 8 at x = 3, y = 30, z = 7: bit 3 only.
        for (bit, plane) in sky.iter_mut().enumerate() {
            let column = &mut plane[3 + 30 * 32];
            *column = *column & !(1 << 7) | ((8 >> bit & 1) as u32) << 7;
        }
        Arc::new(ChunkLight {
            inputs: Arc::new(LightInputs {
                occupancy: Occupancy::Open,
                emissive: Vec::new(),
                columns: ColumnAnswer::none(),
                ring: false,
            }),
            sky: ChannelLight::Planes(sky.into_boxed_slice()),
            voxel: ChannelLight::Uniform(0),
            precision: LightPrecision::Steps16,
        })
    }

    /// A route whose texture holds `capacity` slots, with a camera at the
    /// origin.
    #[cfg(feature = "sub_grids")]
    fn route_of(capacity: u32) -> LightRoute<DefaultWorld> {
        let mut images = Assets::<Image>::default();
        let config = LightPropagation {
            slot_capacity: capacity,
            ..default()
        };
        let mut route = LightRoute::new(&config, 15, &mut images);
        route.set_cameras([Vec3::ZERO].into_iter());
        route
    }

    /// A sub-grid of two voxels in the middle of the world chunk at
    /// `position`.
    #[cfg(feature = "sub_grids")]
    fn body_in(position: IVec3) -> Placed {
        Placed {
            grid: GridId::new(1),
            pose: GlobalTransform::from_translation(position.as_vec3() * 32.0 + 15.0),
            bounds: Some((IVec3::ZERO, IVec3::splat(2))),
        }
    }

    #[cfg(feature = "sub_grids")]
    #[test]
    fn an_air_chunk_in_the_tag_map_shows_its_most_common_open_level_until_a_slot_frees() {
        let mut route = route_of(1);
        let mut images = Assets::<Image>::default();
        let (light, own) = light_needing_a_slot();
        let holder = Entity::from_raw_u32(1).unwrap();
        let near = in_world(IVec3::ZERO);
        route.light_landed(near, holder, light, own, None, true);
        assert_ne!(route.show(holder, near, false) & HAS_SLOT, 0);
        // An air chunk with no mesh, farther from the camera than the holder.
        let air = Entity::from_raw_u32(2).unwrap();
        let far = in_world(IVec3::new(3, 0, 0));
        let air_light = air_lit_voxel_by_voxel();
        route.light_landed(
            far,
            air,
            air_light.clone(),
            own_faces_see(&air_light),
            None,
            true,
        );
        let body = body_in(IVec3::new(3, 0, 0));

        route.upkeep_tag_map(&[body], &Hinting::none(), &mut images);

        let flat = tag_levels(9, 0, LightPrecision::Steps16);
        let answer = |route: &LightRoute<DefaultWorld>| {
            route.tag_map().unwrap().answer(IVec3::new(3, 0, 0))
        };
        assert_eq!(answer(&route), Some(flat));
        assert_ne!(route.chunks[&RouteId::Entity(holder)].tag & HAS_SLOT, 0);

        // The holder despawns, and its slot goes to the air chunk.
        route.forget(holder);
        route.refill();
        route.upkeep_tag_map(&[body], &Hinting::none(), &mut images);

        let tag = answer(&route).unwrap();
        assert_ne!(tag & HAS_SLOT, 0);
        let queued = route.queued_upload(tag & 0xFF_FFFF).expect("bytes queued");
        assert_eq!(queued, &slot_bytes(&air_light)[..]);
    }

    #[cfg(feature = "sub_grids")]
    #[test]
    fn an_air_chunk_in_the_tag_map_takes_the_slot_of_a_farther_holder() {
        let mut route = route_of(1);
        let mut images = Assets::<Image>::default();
        let (light, own) = light_needing_a_slot();
        let holder = Entity::from_raw_u32(1).unwrap();
        let far = in_world(IVec3::new(5, 0, 0));
        route.light_landed(far, holder, light, own, None, true);
        assert_ne!(route.show(holder, far, false) & HAS_SLOT, 0);
        let air = Entity::from_raw_u32(2).unwrap();
        let near = in_world(IVec3::X);
        let air_light = air_lit_voxel_by_voxel();
        route.light_landed(
            near,
            air,
            air_light.clone(),
            own_faces_see(&air_light),
            None,
            true,
        );

        route.upkeep_tag_map(&[body_in(IVec3::X)], &Hinting::none(), &mut images);

        let tag = route.tag_map().unwrap().answer(IVec3::X).unwrap();
        assert_ne!(tag & HAS_SLOT, 0);
        assert_eq!(route.chunks[&RouteId::Entity(holder)].tag & HAS_SLOT, 0);
        assert_eq!(route.counters.evictions, 1);
    }

    #[cfg(feature = "sub_grids")]
    #[test]
    fn a_shown_chunk_takes_the_wider_rule_only_while_the_tag_map_holds_it() {
        let mut route = route_of(16);
        let mut images = Assets::<Image>::default();
        // One solid voxel in a chunk of the air above: the open voxels
        // around the solid one hold 9, but one far from it holds 8.
        let air = air_lit_voxel_by_voxel();
        let mut occupancy = Occupancy::Open;
        occupancy.set(16, 16, 16, true);
        occupancy.normalize();
        let light = Arc::new(ChunkLight {
            inputs: Arc::new(LightInputs {
                occupancy,
                emissive: Vec::new(),
                columns: ColumnAnswer::none(),
                ring: false,
            }),
            sky: match &air.sky {
                ChannelLight::Planes(planes) => ChannelLight::Planes(planes.clone()),
                ChannelLight::Uniform(level) => ChannelLight::Uniform(*level),
            },
            voxel: ChannelLight::Uniform(0),
            precision: LightPrecision::Steps16,
        });
        let own = own_faces_see(&light);
        assert!(!own.needs_slot());
        let entity = Entity::from_raw_u32(1).unwrap();
        let key = in_world(IVec3::X);
        route.light_landed(key, entity, light, own, None, true);
        let flat = route.show(entity, key, false);
        assert_eq!(flat, tag_levels(9, 0, LightPrecision::Steps16));

        route.upkeep_tag_map(&[body_in(IVec3::X)], &Hinting::none(), &mut images);
        let tag = route.chunks[&RouteId::Entity(entity)].tag;
        assert_ne!(tag & HAS_SLOT, 0);
        assert_eq!(route.tag_map().unwrap().answer(IVec3::X), Some(tag));
        assert_eq!(route.take_changed_tags().next_back(), Some((entity, tag)));

        // The body moves away, and the chunk gives its slot back.
        route.upkeep_tag_map(
            &[body_in(IVec3::new(9, 0, 0))],
            &Hinting::none(),
            &mut images,
        );
        assert_eq!(route.chunks[&RouteId::Entity(entity)].tag, flat);
        assert!(route.holders.is_empty());
        assert_eq!(route.tag_map().unwrap().answer(IVec3::X), None);
    }

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
