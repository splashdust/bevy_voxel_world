//! The route light takes from a chunk task to the shader: a texture of slots
//! per world, and a `MeshTag` on each chunk that the crate owns.
//!
//! Chunk meshes are shared through the mesh cache, keyed on the chunk's
//! voxels, and their positions are local to the chunk. Two chunks with the
//! same voxels can hold different light, so light can't go into a mesh's
//! vertices. It reaches the shader through two things the mesh doesn't hold:
//!
//! - **A slot** in the world's light texture, for a chunk whose light varies.
//!   A slot is 34³ texels: the chunk's 32³ voxels and a shell one voxel deep
//!   around them, so a slot's cell is a voxel of the chunk's padded local
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
//! A lamp's own cell holds 15 in its voxel half, which no open voxel can hold,
//! and its declared level in its sky half, which a solid voxel never uses. The
//! shader reads such a cell as sky light 0 and voxel light at the lamp's level,
//! and makes a face glow when the cell behind it holds one.
//!
//! A solve, in a chunk task or a relight task, turns the solver's bit planes
//! into a slot's bytes, only for a chunk with faces whose light varies. When
//! the light lands on the main thread, a chunk showing its mesh gets its slot
//! and its tag, and the bytes are queued. A chunk whose mesh isn't shown yet
//! keeps them until it is. The render world writes them into the slot's part
//! of the texture during the frame they were extracted with, so the mesh, the
//! tag and the light show in the same frame. The texture's `Image` asset is
//! never changed, so nothing is written at rest.
//!
//! Shell cells stay 0, so faces on a chunk's border read their neighbours'
//! voxels as dark. Slots are given out until the texture is full, and a chunk
//! keeps its slot until it despawns. A chunk that finds the texture full is
//! drawn with no light from lamps.

use std::marker::PhantomData;

use bevy::asset::RenderAssetUsages;
use bevy::ecs::entity::EntityHashMap;
use bevy::ecs::system::EntityCommands;
use bevy::mesh::MeshTag;
use bevy::platform::collections::HashMap;
use bevy::prelude::*;
use bevy::render::{
    render_asset::RenderAssets,
    render_resource::{
        Extent3d, Origin3d, TexelCopyBufferLayout, TexelCopyTextureInfo, TextureAspect,
        TextureDimension, TextureFormat, TextureUsages,
    },
    renderer::RenderQueue,
    texture::GpuImage,
    ExtractSchedule, MainWorld, Render, RenderApp, RenderSystems,
};

use super::config::LightPropagation;
use super::solver::{ChannelLight, ChunkLight};
use crate::chunk::CHUNK_SIZE_U;
use crate::configuration::VoxelWorldConfig;

/// Texels along each side of a slot: a chunk and a shell one voxel deep.
pub(crate) const SLOT_SIDE: u32 = CHUNK_SIZE_U + 2;
const SIDE: usize = SLOT_SIDE as usize;
/// The bytes of one slot.
pub(crate) const SLOT_BYTES: usize = SIDE * SIDE * SIDE;

/// The tag bit of a chunk with a slot.
pub(crate) const HAS_SLOT: u32 = 1 << 31;
/// A lamp's own cell holds this in its voxel half.
const LAMP_MARKER: u8 = 15 << 4;

const CHUNK: usize = CHUNK_SIZE_U as usize;

/// For each byte, a `u64` with bit `i` of the byte moved to bit 0 of byte `i`.
const SPREAD: [u64; 256] = {
    let mut table = [0u64; 256];
    let mut byte = 0;
    while byte < 256 {
        let mut bit = 0;
        while bit < 8 {
            if byte >> bit & 1 == 1 {
                table[byte] |= 1 << (bit * 8);
            }
            bit += 1;
        }
        byte += 1;
    }
    table
};

/// The bytes of the slot of a chunk with `light`, or `None` if its light is
/// one level per channel and its tag can carry it.
///
/// Solid voxels hold 0, apart from each emissive voxel, which holds the lamp
/// marker. The shell is left at 0.
pub(crate) fn slot_bytes(light: &ChunkLight) -> Option<Box<[u8]>> {
    let ChannelLight::Planes(planes) = &light.voxel else {
        return None;
    };
    let mut bytes = vec![0u8; SLOT_BYTES].into_boxed_slice();
    let sky = light.sky as u64;
    for y in 0..CHUNK {
        for x in 0..CHUNK {
            let column = x + y * CHUNK;
            let open = !light.inputs.occupancy.solid_column(x, y);
            let row = 1 + (x + 1) * SIDE + (y + 1) * SIDE * SIDE;
            for shift in (0..CHUNK).step_by(8) {
                let open_byte = (open >> shift) as usize & 0xFF;
                let mut texels = SPREAD[open_byte] * sky;
                for (bit, plane) in planes.iter().enumerate() {
                    let levels = (plane[column] >> shift) as usize & open_byte;
                    texels |= SPREAD[levels] << (4 + bit);
                }
                bytes[row + shift..row + shift + 8]
                    .copy_from_slice(&texels.to_le_bytes());
            }
        }
    }
    for lamp in &light.inputs.emissive {
        let (x, y, z) = (lamp.x as usize, lamp.y as usize, lamp.z as usize);
        bytes[(z + 1) + (x + 1) * SIDE + (y + 1) * SIDE * SIDE] =
            LAMP_MARKER | lamp.level;
    }
    Some(bytes)
}

/// The levels of a chunk with no slot, as the low byte of its tag.
fn tag_levels(sky: u8, voxel: u8) -> u8 {
    sky | voxel << 4
}

/// How many slots the texture holds along each of its axes, for `capacity`
/// slots: the smallest axis doubles, x first, until they hold them all.
fn slots_per_axis(capacity: u32) -> UVec3 {
    let mut axes = UVec3::ONE;
    while axes.element_product() < capacity {
        let axis = (0..3).min_by_key(|&axis| axes[axis]).unwrap();
        axes[axis] *= 2;
    }
    axes
}

/// A chunk's landed light, as the shader will read it.
struct Landing {
    /// Its slot's bytes, if its light varies.
    bytes: Option<Box<[u8]>>,
    /// Its tag's levels if it gets no slot.
    levels: u8,
}

/// World `C`'s light texture and the slots in it. Only a lit world that shows
/// its light has one.
#[derive(Resource)]
pub(crate) struct LightRoute<C> {
    pub texture: Handle<Image>,
    slots_per_axis: UVec3,
    capacity: u32,
    next_slot: u32,
    slots: EntityHashMap<u32>,
    /// Light that landed for chunks not showing their mesh yet.
    landing: EntityHashMap<Landing>,
    /// The tag of each chunk showing its mesh.
    tags: EntityHashMap<u32>,
    /// The bytes to write into each slot, taken by the render world each frame.
    uploads: HashMap<u32, Box<[u8]>>,
    /// The tag's levels for a chunk with no light of its own, such as a chunk
    /// above LOD level 0, or one whose light hasn't landed yet.
    unlit_levels: u8,
    _world: PhantomData<C>,
}

impl<C: VoxelWorldConfig> LightRoute<C> {
    /// A route for `config`, which must be clamped, with its texture added to
    /// `images`. The texture starts at 0 in every texel.
    pub fn new(config: &LightPropagation, sky: u8, images: &mut Assets<Image>) -> Self {
        let slots_per_axis = slots_per_axis(config.slot_capacity);
        let size = slots_per_axis * SLOT_SIDE;
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
        Self {
            texture: images.add(image),
            slots_per_axis,
            capacity: config.slot_capacity,
            next_slot: 0,
            slots: EntityHashMap::default(),
            landing: EntityHashMap::default(),
            tags: EntityHashMap::default(),
            uploads: HashMap::default(),
            unlit_levels: tag_levels(sky, 0),
            _world: PhantomData,
        }
    }

    /// Takes the light that landed for the chunk `entity`, with the bytes its
    /// solve made of it. Returns the chunk's new tag if it shows its mesh, and
    /// otherwise keeps the light until it does.
    pub fn light_landed(
        &mut self,
        entity: Entity,
        light: &ChunkLight,
        bytes: Option<Box<[u8]>>,
    ) -> Option<u32> {
        let levels = match light.voxel {
            ChannelLight::Uniform(voxel) => tag_levels(light.sky, voxel),
            ChannelLight::Planes(_) => tag_levels(light.sky, 0),
        };
        let landing = Landing { bytes, levels };
        if !self.tags.contains_key(&entity) {
            self.landing.insert(entity, landing);
            return None;
        }
        Some(self.place(entity, landing))
    }

    /// The tag of the chunk `entity` as its mesh is shown, placing light that
    /// landed for it meanwhile. A chunk whose light hasn't landed gets the
    /// levels of a chunk with no light.
    fn show(&mut self, entity: Entity) -> u32 {
        match self.landing.remove(&entity) {
            Some(landing) => self.place(entity, landing),
            None => *self.tags.entry(entity).or_insert(self.unlit_levels as u32),
        }
    }

    /// Gives the chunk `entity` the tag for `landing`, and a slot with its
    /// bytes queued if its light varies. A chunk keeps the slot it had.
    fn place(&mut self, entity: Entity, landing: Landing) -> u32 {
        let slot = landing.bytes.and_then(|bytes| {
            let slot = match self.slots.get(&entity) {
                Some(slot) => *slot,
                None if self.next_slot < self.capacity => {
                    self.next_slot += 1;
                    self.slots.insert(entity, self.next_slot - 1);
                    self.next_slot - 1
                }
                None => return None,
            };
            self.uploads.insert(slot, bytes);
            Some(slot)
        });
        let tag = slot.map_or(landing.levels as u32, |slot| HAS_SLOT | slot);
        self.tags.insert(entity, tag);
        tag
    }

    /// Forgets the despawned chunk `entity`, apart from its slot.
    pub fn forget(&mut self, entity: Entity) {
        self.landing.remove(&entity);
        self.tags.remove(&entity);
    }

    /// The bytes queued for `slot` and not yet taken by the render world.
    #[cfg(test)]
    pub fn queued_upload(&self, slot: u32) -> Option<&[u8]> {
        self.uploads.get(&slot).map(|bytes| &bytes[..])
    }
}

/// Sets the tag of a chunk of world `C` as its mesh is shown, and queues its
/// slot's bytes. Runs as the chunk gets its `Mesh3d`.
pub(crate) fn tag_chunk<C: VoxelWorldConfig>(chunk: &mut EntityCommands) {
    chunk.queue(|mut chunk: EntityWorldMut| {
        let entity = chunk.id();
        let tag = chunk.resource_mut::<LightRoute<C>>().show(entity);
        chunk.insert(MeshTag::new(tag));
    });
}

/// Sets up writing world `C`'s queued slots into its texture, in the render
/// world. An app with no renderer gets nothing.
pub(crate) fn add_render_systems<C: VoxelWorldConfig>(app: &mut App) {
    let route = app.world().resource::<LightRoute<C>>();
    let uploads = RenderUploads::<C> {
        texture: route.texture.id(),
        slots_per_axis: route.slots_per_axis,
        pending: Vec::new(),
        _world: PhantomData,
    };
    let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
        return;
    };
    render_app
        .insert_resource(uploads)
        .add_systems(ExtractSchedule, extract_uploads::<C>)
        .add_systems(
            Render,
            write_uploads::<C>.in_set(RenderSystems::PrepareResources),
        );
}

/// World `C`'s slot bytes waiting to be written, in the render world.
#[derive(Resource)]
struct RenderUploads<C> {
    texture: AssetId<Image>,
    slots_per_axis: UVec3,
    /// Each slot and its bytes, oldest first.
    pending: Vec<(u32, Box<[u8]>)>,
    _world: PhantomData<C>,
}

/// Takes the slot bytes queued in the main world this frame.
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
    uploads.pending.extend(route.uploads.drain());
}

/// Writes each pending slot's bytes into its part of the texture, once the
/// texture exists on the GPU.
fn write_uploads<C: VoxelWorldConfig>(
    mut uploads: ResMut<RenderUploads<C>>,
    images: Res<RenderAssets<GpuImage>>,
    queue: Res<RenderQueue>,
) {
    if uploads.pending.is_empty() {
        return;
    }
    let Some(image) = images.get(uploads.texture) else {
        return;
    };
    let axes = uploads.slots_per_axis;
    for (slot, bytes) in uploads.pending.drain(..) {
        let origin = UVec3::new(
            slot % axes.x,
            slot / axes.x % axes.y,
            slot / (axes.x * axes.y),
        ) * SLOT_SIDE;
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
            &bytes,
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(SLOT_SIDE),
                rows_per_image: Some(SLOT_SIDE),
            },
            Extent3d {
                width: SLOT_SIDE,
                height: SLOT_SIDE,
                depth_or_array_layers: SLOT_SIDE,
            },
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_texture_shape_follows_the_slot_count() {
        assert_eq!(slots_per_axis(1024) * SLOT_SIDE, UVec3::new(544, 272, 272));
        assert_eq!(slots_per_axis(1), UVec3::ONE);
        assert!(slots_per_axis(1000).element_product() >= 1000);
    }
}
