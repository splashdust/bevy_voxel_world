//! The tag map
//!
//! This module keeps the tag map of a lit world: a hash table on the GPU from
//! a level-0 world chunk position to the tag the screen shows there. A face
//! of a sub-grid looks up its world position in it to find the light of the
//! world there. The module is compiled only with `sub_grids` and
//! `light_propagation`.
//!
//! # What it holds
//!
//! Each sub-grid adds every level-0 world chunk position that the box around
//! its solid voxels touches, placed by the pose drawn this frame and padded by
//! one voxel. A sub-grid with no solid voxel adds none. Positions are on the
//! grid of the world, so the map never holds the chunks of a sub-grid.
//!
//! Each position holds the tag of the chunk the screen shows there, in the
//! encoding [`route`](crate::light::route) describes:
//!
//! - A position with a level-0 chunk holds the tag of that chunk.
//! - A position covered by a coarse chunk holds the tag of the coarse chunk,
//!   whose LOD bits say how wide its cells are.
//! - A position of hinted air holds levels: see [Hinted air](#hinted-air).
//! - Any other position with no chunk showing a tag, such as one past the
//!   spawning distance, holds sky 15 and voxel 0 in white, as open sky with no
//!   emissive voxels does.
//!
//! A tag carries the light colour of its chunk, so the entries of the map
//! carry it too.
//!
//! # Slots near a sub-grid
//!
//! While the map holds a position, the route keeps the level-0 world chunk
//! there, including a chunk of air with no mesh. It gives that chunk a slot,
//! unless every one of its open voxels holds one sky light level and one voxel
//! light level, and so does every open voxel that the shell of its slot copies
//! from its face neighbours. The bytes of the slot are made from the light the
//! world view already stores, so no relight runs. A chunk of air with no solid
//! in its padding shows the larger of its own sky light and the sky light of
//! the sky fill field, as `light_at` reads it.
//!
//! Slots come from the budget of the world under the rule the route
//! describes, with no priority for sub-grids. While a chunk has no slot, its
//! entry holds the most common level among its open voxels. When the map stops
//! holding the position, the slot of the chunk is freed in the frame its entry
//! leaves. A chunk showing its mesh then goes back to the slot rule in
//! [`slot`](crate::light::slot).
//!
//! # When entries are written
//!
//! Upkeep runs in `PostUpdate`, after transform propagation and after the last
//! change to the tag of a chunk. It writes an entry only when the positions
//! held change, or the tag at one of them does. The writes go to the render
//! world in the same extract as the tags of the chunks and the slots of the
//! light texture, so an entry never names a slot its chunk gave up in an
//! earlier frame. Nothing is written while the sub-grids and the tags they
//! touch stay as they are.
//!
//! # Cost
//!
//! Upkeep grows with the world chunk positions the boxes of the sub-grids
//! touch. A tumbling sub-grid has a box that touches more of them than a still
//! one does. Slots go to the chunks of air near sub-grids whose light varies,
//! and making the bytes of a slot grows with the voxels of the chunk.
//!
//! Each entry takes 16 bytes of texture, and the table keeps room for at least
//! twice as many entries as it holds. When it runs out of room it doubles,
//! replacing its texture, and every entry is written again in that frame. It
//! never shrinks. A world makes its tag map in the frame its first sub-grid
//! exists, and a world without light propagation makes none.
//!
//! # The shader
//!
//! The materials of the world bind the texture of the map at binding 202. The
//! shader looks up in it the world chunk in front of the face of a sub-grid:
//! see [`material`](crate::light::material). Until the world makes its tag
//! map, the materials bind a placeholder of one empty entry, where every
//! lookup misses, which has no upkeep. Each time the table grows into a new
//! texture, the materials bind the new texture in the same frame.
//!
//! # Hinted air
//!
//! A position of hinted air within the spawning distance of a camera holds the
//! light of its light entry, if it has one. Its sky light level and its voxel
//! light level are each the most common among the open voxels of the entry.
//! Its light colour is the most common light colour among the open voxels at
//! that voxel light level.
//!
//! Otherwise the position holds what `light_at` reads at its middle voxel with
//! sky fill left out. That is sky 15 if the column above the voxel is open and
//! sky 0 if not, with voxel 0. With sky light off it is sky 0.
//!
//! The column is walked in a task under `JobKind::HintedColumn`, and the
//! position holds sky 15 until the answer lands. The answer is kept while the
//! position stays in the map. The column walks again after any voxel edit of
//! the world, a `reset_light` or a change of light settings, and the position
//! keeps its old answer until the new one lands: see [`columns`].
//!
//! This costs one walk for each position of hinted air the box of a sub-grid
//! comes to touch, and one for each such position held after each of those
//! changes. Hinted air never gets a light entry or a slot for a sub-grid.
//!
//! # The table
//!
//! The table is open addressing with linear probing. It is held in an
//! `Rgba32Uint` 2D texture with one texel per entry: the x, y and z of the
//! position as `u32`s, then the tag. Entry `i` is texel `(i % TABLE_WIDTH, i /
//! TABLE_WIDTH)`. A lookup starts at [`hash`] of the position masked by the
//! capacity, a power of two. It steps to the next entry, wrapping at the end,
//! until it finds the position or an entry whose tag is [`EMPTY`]. A removal
//! shifts the entries after it back instead of leaving a marker, so that a
//! lookup stops at the first empty entry. A shader reading the table must
//! follow these steps exactly, so it stops where the lookup on the CPU does.

use bevy::asset::RenderAssetUsages;
use bevy::mesh::MeshTag;
use bevy::platform::collections::{HashMap, HashSet};
use bevy::prelude::*;
use bevy::render::{
    render_asset::RenderAssets,
    render_resource::{
        Extent3d, Origin3d, TexelCopyBufferLayout, TexelCopyTextureInfo, TextureAspect,
        TextureDimension, TextureFormat, TextureUsages,
    },
    renderer::RenderQueue,
    texture::GpuImage,
};
use bevy::tasks::AsyncComputeTaskPool;

use crate::chunk::CHUNK_SIZE_F;
use crate::configuration::VoxelWorldConfig;
use crate::grid::GridId;
use crate::pool_budget::{JobKind, PoolBudget};
use crate::sub_grid::SubGrids;
use crate::voxel_edits::VoxelEdits;
use crate::voxel_world::{ConfigCopy, VoxelWorldCamera};

use crate::light::route::LightRoute;
use crate::light::LightSettings;
use columns::{walked_voxel, Hinting};

pub(crate) mod columns;

/// The tag of an empty entry. A world chunk's tag never has bit 30 set, so
/// no entry holding a tag can match it.
pub(crate) const EMPTY: u32 = u32::MAX;

/// Entries in each row of the texture.
pub(crate) const TABLE_WIDTH: u32 = 256;

/// The room a new table has, in entries.
const FIRST_CAPACITY: usize = TABLE_WIDTH as usize;

/// One entry: a position's x, y and z, then its tag.
type Texel = [u32; 4];

const EMPTY_TEXEL: Texel = [0, 0, 0, EMPTY];

/// The hash of the level-0 chunk position `position`, from which a lookup
/// starts. Every step is a wrapping `u32` multiply, xor or shift, so a
/// shader can work it out the same way.
pub(crate) fn hash(position: IVec3) -> u32 {
    let mut hash = (position.x as u32).wrapping_mul(0x9E37_79B1)
        ^ (position.y as u32).wrapping_mul(0x85EB_CA77)
        ^ (position.z as u32).wrapping_mul(0xC2B2_AE3D);
    // MurmurHash3's finalizer, so nearby positions spread across the table.
    hash ^= hash >> 16;
    hash = hash.wrapping_mul(0x85EB_CA6B);
    hash ^= hash >> 13;
    hash = hash.wrapping_mul(0xC2B2_AE35);
    hash ^= hash >> 16;
    hash
}

/// The level-0 world chunk positions touched by the box of a grid's voxels
/// from `min` inclusive to `max` exclusive, placed by `pose`, the grid's
/// pose against the world's root, with the box's world bounds padded by
/// one voxel.
pub(crate) fn chunks_around(
    pose: &GlobalTransform,
    min: IVec3,
    max: IVec3,
) -> impl Iterator<Item = IVec3> {
    let (low, high) = (min.as_vec3(), max.as_vec3());
    let (mut world_min, mut world_max) = (Vec3::INFINITY, Vec3::NEG_INFINITY);
    for corner in 0..8 {
        let local = Vec3::select(
            BVec3::new(corner & 1 != 0, corner & 2 != 0, corner & 4 != 0),
            high,
            low,
        );
        let world = pose.transform_point(local);
        world_min = world_min.min(world);
        world_max = world_max.max(world);
    }
    let first = ((world_min - 1.0) / CHUNK_SIZE_F).floor().as_ivec3();
    let last = ((world_max + 1.0) / CHUNK_SIZE_F).ceil().as_ivec3() - 1;
    (first.x..=last.x).flat_map(move |x| {
        (first.y..=last.y)
            .flat_map(move |y| (first.z..=last.z).map(move |z| IVec3::new(x, y, z)))
    })
}

/// The hash table of a tag map, from a level-0 chunk position to a tag, laid
/// out as its texture's texels: see the module docs.
pub(crate) struct TagTable {
    texels: Vec<Texel>,
    len: usize,
    /// The entries changed since [`take_changed`](Self::take_changed) last
    /// ran, maybe more than once each.
    changed: Vec<u32>,
}

impl TagTable {
    /// An empty table with room for `capacity` entries, a power of two.
    pub fn with_capacity(capacity: usize) -> Self {
        debug_assert!(capacity.is_power_of_two());
        Self {
            texels: vec![EMPTY_TEXEL; capacity],
            len: 0,
            changed: Vec::new(),
        }
    }

    pub fn capacity(&self) -> usize {
        self.texels.len()
    }

    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the table holds `len` entries with at least as much room
    /// left empty, which keeps probes short.
    pub fn has_room_for(&self, len: usize) -> bool {
        len <= self.capacity() / 2
    }

    /// The tag at `position`, if the table holds it.
    pub fn get(&self, position: IVec3) -> Option<u32> {
        self.find(position).ok().map(|index| self.texels[index][3])
    }

    /// The entry holding `position`, or else the empty entry where a lookup
    /// of it stops.
    fn find(&self, position: IVec3) -> Result<usize, usize> {
        let mask = self.capacity() - 1;
        let mut index = hash(position) as usize & mask;
        loop {
            let texel = &self.texels[index];
            if texel[3] == EMPTY {
                return Err(index);
            }
            if texel[..3] == key(position) {
                return Ok(index);
            }
            index = (index + 1) & mask;
        }
    }

    /// Sets the tag at `position` to `tag`, which must not be [`EMPTY`].
    /// The table must keep at least one empty entry.
    pub fn insert(&mut self, position: IVec3, tag: u32) {
        debug_assert_ne!(tag, EMPTY);
        let index = match self.find(position) {
            Ok(index) if self.texels[index][3] == tag => return,
            Ok(index) => index,
            Err(index) => {
                debug_assert!(self.len + 1 < self.capacity(), "the table is full");
                self.len += 1;
                index
            }
        };
        let [x, y, z] = key(position);
        self.texels[index] = [x, y, z, tag];
        self.changed.push(index as u32);
    }

    /// Removes `position` from the table, moving the entries whose probes
    /// passed it back so every lookup still finds them.
    pub fn remove(&mut self, position: IVec3) {
        let Ok(mut hole) = self.find(position) else {
            return;
        };
        self.len -= 1;
        let mask = self.capacity() - 1;
        let mut next = hole;
        loop {
            next = (next + 1) & mask;
            let texel = self.texels[next];
            if texel[3] == EMPTY {
                break;
            }
            let home = hash(position_of(&texel)) as usize & mask;
            // An entry whose home lies after the hole, up to the entry
            // itself, is still found from its home.
            let found_anyway = if hole <= next {
                hole < home && home <= next
            } else {
                hole < home || home <= next
            };
            if found_anyway {
                continue;
            }
            self.texels[hole] = texel;
            self.changed.push(hole as u32);
            hole = next;
        }
        self.texels[hole] = EMPTY_TEXEL;
        self.changed.push(hole as u32);
    }

    /// A table with room for `capacity` entries, a power of two, holding
    /// this one's entries.
    pub fn grown_to(&self, capacity: usize) -> Self {
        let mut grown = Self::with_capacity(capacity);
        for (position, tag) in self.entries() {
            grown.insert(position, tag);
        }
        grown.changed.clear();
        grown
    }

    /// The entries changed since this last ran, each once, lowest first.
    pub fn take_changed(&mut self) -> Vec<u32> {
        let mut changed = std::mem::take(&mut self.changed);
        changed.sort_unstable();
        changed.dedup();
        changed
    }

    /// Each position the table holds, with its tag.
    pub fn entries(&self) -> impl Iterator<Item = (IVec3, u32)> + '_ {
        self.texels
            .iter()
            .filter(|texel| texel[3] != EMPTY)
            .map(|texel| (position_of(texel), texel[3]))
    }
}

/// The first three words of the texel holding `position`.
fn key(position: IVec3) -> [u32; 3] {
    position.to_array().map(|coordinate| coordinate as u32)
}

/// The position the texel `texel` holds.
fn position_of(texel: &Texel) -> IVec3 {
    IVec3::new(texel[0] as i32, texel[1] as i32, texel[2] as i32)
}

/// A sub-grid as the tag map's upkeep reads it.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Placed {
    pub grid: GridId,
    /// Its pose against the world's root, as drawn this frame.
    pub pose: GlobalTransform,
    /// The box around its solid voxels, from its minimum corner inclusive
    /// to its maximum corner exclusive, or `None` with no solid voxel.
    pub bounds: Option<(IVec3, IVec3)>,
}

/// The positions one [`TagMap::place`] added to the map and removed from it.
pub(crate) struct Placement {
    pub added: Vec<IVec3>,
    pub removed: Vec<IVec3>,
}

/// What a sub-grid adds to the tag map, as of its last upkeep.
struct Body {
    pose: GlobalTransform,
    /// The box around its solid voxels, from its minimum corner inclusive to
    /// its maximum corner exclusive.
    bounds: Option<(IVec3, IVec3)>,
    positions: Vec<IVec3>,
}

/// Counts of what a tag map has done, for benchmarks.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct TagMapCounters {
    /// Entries written into the texture, counting each entry of a full
    /// rewrite.
    pub entries_written: u64,
    /// Times the whole texture was written: once as it's made, and once
    /// each time it grows.
    pub full_rewrites: u64,
    /// Time spent on upkeep, on the main thread.
    pub upkeep_nanos: u64,
}

/// A lit world's tag map: its table, its texture, and which positions each
/// sub-grid adds. See the module docs.
pub(crate) struct TagMap {
    table: TagTable,
    texture: Handle<Image>,
    bodies: HashMap<GridId, Body>,
    /// How many sub-grids add each position the map holds.
    held: HashMap<IVec3, u32>,
    /// The tag of a position with nothing known, as of the last upkeep.
    nothing_known: u32,
    /// Entries changed since the render world last took the writes.
    written: HashSet<u32>,
    /// Whether the whole texture waits to be written.
    full: bool,
    /// Whether no upkeep has run yet, so the first counts the texture's
    /// first full write.
    fresh: bool,
    pub counters: TagMapCounters,
}

/// Writes into a tag map's texture, taken by the render world.
pub(crate) struct TagMapWrites {
    texture: AssetId<Image>,
    /// Runs of entries next to each other in one row, by their first entry.
    runs: Vec<(u32, Vec<Texel>)>,
}

impl TagMap {
    /// A tag map holding nothing, whose texture is added to `images`.
    pub fn new(images: &mut Assets<Image>) -> Self {
        Self {
            table: TagTable::with_capacity(FIRST_CAPACITY),
            texture: images.add(texture_image(FIRST_CAPACITY)),
            bodies: HashMap::default(),
            held: HashMap::default(),
            nothing_known: EMPTY,
            written: HashSet::default(),
            full: true,
            fresh: true,
            counters: TagMapCounters {
                full_rewrites: 1,
                ..default()
            },
        }
    }

    /// How many positions the map holds.
    pub fn len(&self) -> usize {
        self.table.len()
    }

    /// Each position the map holds.
    pub fn positions(&self) -> impl Iterator<Item = IVec3> + '_ {
        self.held.keys().copied()
    }

    /// The tag at `position`, if the map holds it.
    #[cfg(test)]
    pub fn answer(&self, position: IVec3) -> Option<u32> {
        self.table.get(position)
    }

    /// The map's texture, which the world's materials bind.
    pub fn texture(&self) -> &Handle<Image> {
        &self.texture
    }

    /// Whether the map holds `position`.
    pub fn holds(&self, position: IVec3) -> bool {
        self.held.contains_key(&position)
    }

    /// Brings the positions the map holds in line with `bodies`, each
    /// sub-grid placed. Positions no sub-grid adds any more leave the table
    /// at once. Returns the positions added and removed, and the added ones
    /// get their entries from [`write_entries`](Self::write_entries).
    ///
    /// Its cost grows with the positions the sub-grids add, for each
    /// sub-grid whose pose or box changed.
    pub fn place(&mut self, bodies: &[Placed]) -> Placement {
        let mut added = Vec::new();
        let mut removed = Vec::new();
        let live: HashSet<GridId> = bodies.iter().map(|body| body.grid).collect();
        let gone: Vec<GridId> = self
            .bodies
            .keys()
            .filter(|grid| !live.contains(*grid))
            .copied()
            .collect();
        for grid in gone {
            let body = self.bodies.remove(&grid).unwrap();
            self.release(&body.positions, &mut removed);
        }
        for &Placed { grid, pose, bounds } in bodies {
            if let Some(body) = self.bodies.get(&grid) {
                if body.pose == pose && body.bounds == bounds {
                    continue;
                }
            }
            let positions: Vec<IVec3> = bounds
                .map(|(min, max)| chunks_around(&pose, min, max).collect())
                .unwrap_or_default();
            for position in &positions {
                let count = self.held.entry(*position).or_insert(0);
                *count += 1;
                if *count == 1 {
                    added.push(*position);
                }
            }
            if let Some(old) = self.bodies.remove(&grid) {
                self.release(&old.positions, &mut removed);
            }
            self.bodies.insert(
                grid,
                Body {
                    pose,
                    bounds,
                    positions,
                },
            );
        }
        // A position released and taken again in the same upkeep stays.
        removed.retain(|position| !self.held.contains_key(position));
        added.retain(|position| self.table.get(*position).is_none());
        for position in &removed {
            self.table.remove(*position);
        }
        Placement { added, removed }
    }

    /// Writes the tags `answer` gives into the table: for each position in
    /// `added`, from [`place`](Self::place), and for each position whose
    /// tag may have changed since the last upkeep, from `touched`. A
    /// position with no tag known holds `nothing_known`.
    ///
    /// The table grows into a new texture added to `images` when it runs
    /// out of room. Its cost grows with `added` and `touched`.
    pub fn write_entries(
        &mut self,
        added: Vec<IVec3>,
        touched: impl Iterator<Item = IVec3>,
        mut answer: impl FnMut(IVec3) -> u32,
        nothing_known: u32,
        images: &mut Assets<Image>,
    ) {
        let mut whole = std::mem::take(&mut self.fresh);
        let new = added;
        let needed = self.table.len() + new.len();
        if !self.table.has_room_for(needed) {
            self.grow_for(needed, images);
            whole = true;
        }
        for position in new {
            self.table.insert(position, answer(position));
        }
        if nothing_known != self.nothing_known {
            self.nothing_known = nothing_known;
            for position in self.held.keys() {
                self.table.insert(*position, answer(*position));
            }
        } else {
            for position in touched {
                if self.held.contains_key(&position) {
                    self.table.insert(position, answer(position));
                }
            }
        }

        let changed = self.table.take_changed();
        if whole {
            self.counters.entries_written += self.table.capacity() as u64;
        } else {
            self.counters.entries_written += changed.len() as u64;
        }
        if !self.full {
            self.written.extend(changed);
        }
    }

    /// Takes `positions` from what one sub-grid adds, pushing each position
    /// no sub-grid adds any more onto `removed`.
    fn release(&mut self, positions: &[IVec3], removed: &mut Vec<IVec3>) {
        for position in positions {
            let count = self.held.get_mut(position).unwrap();
            *count -= 1;
            if *count == 0 {
                self.held.remove(position);
                removed.push(*position);
            }
        }
    }

    /// Doubles the table until it has room for `needed` entries, replacing
    /// its texture, whose every entry then waits to be written.
    fn grow_for(&mut self, needed: usize, images: &mut Assets<Image>) {
        let mut capacity = self.table.capacity();
        while needed > capacity / 2 {
            capacity *= 2;
        }
        self.table = self.table.grown_to(capacity);
        self.texture = images.add(texture_image(capacity));
        self.full = true;
        self.written.clear();
        self.counters.full_rewrites += 1;
    }

    /// The writes waiting since the render world last took them, if any.
    pub fn take_writes(&mut self) -> Option<TagMapWrites> {
        let texture = self.texture.id();
        if std::mem::take(&mut self.full) {
            self.written.clear();
            let runs = self
                .table
                .texels
                .chunks(TABLE_WIDTH as usize)
                .enumerate()
                .map(|(row, texels)| (row as u32 * TABLE_WIDTH, texels.to_vec()))
                .collect();
            return Some(TagMapWrites { texture, runs });
        }
        if self.written.is_empty() {
            return None;
        }
        let mut written: Vec<u32> = self.written.drain().collect();
        written.sort_unstable();
        let mut runs: Vec<(u32, Vec<Texel>)> = Vec::new();
        for index in written {
            let texel = self.table.texels[index as usize];
            match runs.last_mut() {
                Some((first, texels))
                    if *first + texels.len() as u32 == index
                        && index % TABLE_WIDTH != 0 =>
                {
                    texels.push(texel)
                }
                _ => runs.push((index, vec![texel])),
            }
        }
        Some(TagMapWrites { texture, runs })
    }
}

impl TagMapWrites {
    /// Each entry written, with the position and tag it holds. An empty
    /// entry gives `None`.
    #[cfg(test)]
    pub fn entries(&self) -> impl Iterator<Item = Option<(IVec3, u32)>> + '_ {
        self.runs.iter().flat_map(|(_, texels)| {
            texels
                .iter()
                .map(|texel| (texel[3] != EMPTY).then(|| (position_of(texel), texel[3])))
        })
    }
}

/// The texture of a table with room for `capacity` entries, with nothing
/// written into it.
fn texture_image(capacity: usize) -> Image {
    let mut image = Image::new_uninit(
        Extent3d {
            width: TABLE_WIDTH,
            height: capacity as u32 / TABLE_WIDTH,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        TextureFormat::Rgba32Uint,
        RenderAssetUsages::default(),
    );
    image.texture_descriptor.usage =
        TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST;
    image
}

/// A texture laid out as a tag map's, of one empty entry, so that every
/// lookup in it misses. A lit world's materials bind it until the world
/// makes its tag map.
pub(crate) fn placeholder_image() -> Image {
    let mut texel = [0; 16];
    texel[12..].copy_from_slice(&EMPTY.to_le_bytes());
    let mut image = Image::new_fill(
        Extent3d::default(),
        TextureDimension::D2,
        &texel,
        TextureFormat::Rgba32Uint,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.texture_descriptor.usage = TextureUsages::TEXTURE_BINDING;
    image
}

/// A tag map's writes waiting to be made, in the render world.
#[derive(Default)]
pub(crate) struct PendingWrites {
    texture: Option<AssetId<Image>>,
    /// Oldest first.
    pending: Vec<TagMapWrites>,
}

impl PendingWrites {
    /// Adds the writes extracted this frame. Writes into a texture since
    /// replaced are dropped, and a replaced texture's first writes are its
    /// whole table, so nothing is lost.
    pub fn extend(&mut self, writes: TagMapWrites) {
        if self.texture != Some(writes.texture) {
            self.pending.clear();
            self.texture = Some(writes.texture);
        }
        self.pending.push(writes);
    }

    /// Writes each pending write into the texture, once the texture exists
    /// on the GPU.
    pub fn write(&mut self, images: &RenderAssets<GpuImage>, queue: &RenderQueue) {
        if self.pending.is_empty() {
            return;
        }
        let Some(image) = self.texture.and_then(|texture| images.get(texture)) else {
            return;
        };
        for writes in std::mem::take(&mut self.pending) {
            for (first, texels) in writes.runs {
                let bytes: Vec<u8> = texels
                    .iter()
                    .flatten()
                    .flat_map(|word| word.to_le_bytes())
                    .collect();
                queue.write_texture(
                    TexelCopyTextureInfo {
                        texture: &image.texture,
                        mip_level: 0,
                        origin: Origin3d {
                            x: first % TABLE_WIDTH,
                            y: first / TABLE_WIDTH,
                            z: 0,
                        },
                        aspect: TextureAspect::All,
                    },
                    &bytes,
                    TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(texels.len() as u32 * 16),
                        rows_per_image: Some(1),
                    },
                    Extent3d {
                        width: texels.len() as u32,
                        height: 1,
                        depth_or_array_layers: 1,
                    },
                );
            }
        }
    }
}

/// Whether [`upkeep`] has anything to do: world `C` has a live sub-grid, or
/// a tag map to keep or drop.
pub(crate) fn has_upkeep<C: VoxelWorldConfig>(
    route: Res<LightRoute<C>>,
    sub_grids: Option<Res<SubGrids<C>>>,
) -> bool {
    route.tag_map().is_some() || sub_grids.is_some_and(|sub_grids| sub_grids.has_live())
}

/// Sets the `MeshTag` of each chunk whose tag [`upkeep`] changed.
///
/// It is a system of its own, with no system ordered after it, so `upkeep`
/// has no `Commands` and the systems ordered after `upkeep` wait for no sync
/// point. The tags apply by the end of `PostUpdate`, before the frame is
/// drawn.
pub(crate) fn set_changed_tags<C: VoxelWorldConfig>(
    mut commands: Commands,
    mut route: ResMut<LightRoute<C>>,
) {
    for (entity, tag) in route.take_changed_tags() {
        commands.entity(entity).try_insert(MeshTag::new(tag));
    }
}

/// Keeps world `C`'s tag map in line with its sub-grids and its chunks'
/// tags, and starts the walks of hinted air it waits on. Runs in
/// `PostUpdate` after the sub-grids' poses are kept and after the last
/// system that sets a chunk's tag. The tags of chunks whose slots it
/// changed are set by [`set_changed_tags`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn upkeep<C: VoxelWorldConfig>(
    mut route: ResMut<LightRoute<C>>,
    sub_grids: Option<Res<SubGrids<C>>>,
    edits: Res<VoxelEdits<C, C::MaterialIndex>>,
    settings: Res<LightSettings<C>>,
    configuration: Res<ConfigCopy<C>>,
    mut budget: ResMut<PoolBudget<C>>,
    cameras: Query<&GlobalTransform, With<VoxelWorldCamera<C>>>,
    mut images: ResMut<Assets<Image>>,
) {
    let Some(sub_grids) = sub_grids else {
        return;
    };
    let bodies: Vec<Placed> = sub_grids
        .live_grids()
        .into_iter()
        .map(|(_, grid, pose)| {
            let bounds = edits.solid_bounds_in(grid).map(|(min, max)| (min, max + 1));
            Placed { grid, pose, bounds }
        })
        .collect();
    if bodies.is_empty() && route.tag_map().is_none() {
        return;
    }
    let world = configuration.0.clone();
    let is_hinted_air =
        |position| crate::light::hinted::is_hinted_air(world.as_ref(), position);
    let hinting = Hinting {
        is_hinted_air: &is_hinted_air,
        cameras: cameras
            .iter()
            .map(|camera| {
                crate::chunk::chunk_containing(camera.translation().floor().as_ivec3())
            })
            .collect(),
        spawning_distance: world.spawning_distance(),
        sky: settings.sky_rule.is_some(),
        generation: settings.walks.lock().unwrap().generation(),
    };
    let finished = route.upkeep_tag_map(&bodies, &hinting, &mut images);
    budget.set_finished_last_frame(JobKind::HintedColumn, finished);

    if let Some(rule) = settings.sky_rule.filter(|_| route.wants_hinted_walks()) {
        let mut count = budget.idle();
        if route.hinted_walks_running() == 0 {
            count = count.max(1);
        }
        // On WASM tasks run between frames on the main thread.
        if settings.main_thread {
            count = count.min(1);
        }
        let pool = AsyncComputeTaskPool::get();
        let counters = settings.task.counters.clone();
        route.start_hinted_walks(count, |position| {
            let (world, counters) = (world.clone(), counters.clone());
            let reader = edits.reader();
            let top = (position.y + 1) * crate::chunk::CHUNK_SIZE_I;
            pool.spawn(async move {
                crate::light::request::column_is_open(
                    walked_voxel(position),
                    1,
                    top,
                    rule,
                    world.as_ref(),
                    &reader,
                    &counters.requests,
                )
            })
        });
    }
    budget.set_running(JobKind::HintedColumn, route.hinted_walks_running());
}

/// What the tag map of world `C` has done, for benchmarks.
#[derive(Clone, Copy, Debug, Default)]
pub struct TagMapStats {
    /// The positions the map holds now.
    pub entries: u64,
    /// Entries written into its texture, counting each entry of a full
    /// rewrite.
    pub entries_written: u64,
    /// Times its whole texture was written: as it was made, and each time it
    /// grew.
    pub full_rewrites: u64,
    /// Time spent on upkeep on the main thread, apart from light's own
    /// main-thread time.
    pub upkeep_micros: u64,
    /// Slots held now by world chunks the map holds that show no mesh,
    /// which are world chunks of air near sub-grids.
    pub air_slots: u64,
}

/// What the tag map of world `C` has done, or `None` if the world has no
/// tag map, as a world without light propagation, or one with no sub-grid
/// yet, has none.
pub fn tag_map_stats<C: VoxelWorldConfig>(world: &World) -> Option<TagMapStats> {
    let route = world.get_resource::<LightRoute<C>>()?;
    let map = route.tag_map()?;
    Some(TagMapStats {
        entries: map.len() as u64,
        entries_written: map.counters.entries_written,
        full_rewrites: map.counters.full_rewrites,
        upkeep_micros: map.counters.upkeep_nanos / 1000,
        air_slots: route.kept_unshown_slots() as u64,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A sequence of pseudo-random numbers, the same in every run.
    struct Numbers(u64);

    impl Numbers {
        fn next(&mut self, below: u64) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (self.0 >> 33) % below
        }
    }

    #[test]
    fn every_lookup_agrees_with_a_plain_map_after_inserts_removes_and_grows() {
        let mut numbers = Numbers(7);
        let mut table = TagTable::with_capacity(16);
        let mut plain: std::collections::HashMap<IVec3, u32> = Default::default();
        // Positions in a small box, so inserts and removes hit the same ones
        // and probes run into each other.
        let position = |numbers: &mut Numbers| {
            IVec3::new(
                numbers.next(9) as i32 - 4,
                numbers.next(5) as i32 - 2,
                numbers.next(9) as i32 - 4,
            )
        };
        for step in 0..20_000 {
            let at = position(&mut numbers);
            if numbers.next(3) == 0 {
                table.remove(at);
                plain.remove(&at);
            } else {
                if !table.has_room_for(table.len() + 1) {
                    table = table.grown_to(table.capacity() * 2);
                }
                let tag = numbers.next(1 << 31) as u32;
                table.insert(at, tag);
                plain.insert(at, tag);
            }
            assert_eq!(table.len(), plain.len(), "step {step}");
            if step % 97 == 0 {
                for x in -5..=5 {
                    for y in -3..=3 {
                        for z in -5..=5 {
                            let at = IVec3::new(x, y, z);
                            assert_eq!(table.get(at), plain.get(&at).copied(), "{at}");
                        }
                    }
                }
            }
        }
        let held: std::collections::HashMap<IVec3, u32> = table.entries().collect();
        assert_eq!(held, plain);
    }

    #[test]
    fn removing_every_entry_leaves_an_empty_table() {
        let mut table = TagTable::with_capacity(8);
        let positions = [IVec3::ZERO, IVec3::X, IVec3::NEG_Y, IVec3::new(9, -9, 3)];
        for (tag, position) in positions.iter().enumerate() {
            table.insert(*position, tag as u32);
        }
        for position in positions.iter().rev() {
            table.remove(*position);
        }
        assert_eq!(table.len(), 0);
        assert_eq!(table.entries().count(), 0);
        for position in positions {
            assert_eq!(table.get(position), None);
        }
    }

    #[test]
    fn a_rotated_and_scaled_box_touches_the_chunks_of_its_padded_world_bounds() {
        // Scaled by 2, the 4-voxel box spans 8 voxels: y from 58 to 66,
        // reaching the chunk at y = 2 only once padded. A quarter turn about
        // y sends its x along -z: z from -3 to 5, reaching z = -1.
        let pose = GlobalTransform::from(
            Transform::from_xyz(30.0, 58.0, 5.0)
                .with_rotation(Quat::from_rotation_y(std::f32::consts::FRAC_PI_2))
                .with_scale(Vec3::splat(2.0)),
        );
        let touched: HashSet<IVec3> =
            chunks_around(&pose, IVec3::ZERO, IVec3::splat(4)).collect();
        let mut expected = HashSet::new();
        for x in 0..=1 {
            for y in 1..=2 {
                for z in -1..=0 {
                    expected.insert(IVec3::new(x, y, z));
                }
            }
        }
        assert_eq!(touched, expected);
    }

    #[test]
    fn a_box_one_voxel_from_a_chunk_boundary_reaches_across_it() {
        // Voxels 2 to 30 along each axis: padded, 1 to 31, inside one chunk.
        let inside: HashSet<IVec3> = chunks_around(
            &GlobalTransform::IDENTITY,
            IVec3::splat(2),
            IVec3::splat(31),
        )
        .collect();
        assert_eq!(inside, HashSet::from_iter([IVec3::ZERO]));
        // Voxel 31 ends on the boundary at 32, and its padding crosses it.
        let across: HashSet<IVec3> = chunks_around(
            &GlobalTransform::IDENTITY,
            IVec3::new(31, 2, 2),
            IVec3::new(32, 3, 3),
        )
        .collect();
        assert_eq!(across, HashSet::from_iter([IVec3::ZERO, IVec3::X]));
    }
}
