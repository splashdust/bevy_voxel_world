//! The level-0 reader, which reads the world's voxels as the screen shows them
//! up close, and the steps and caches it shares with the crate's other voxel
//! reads.
//!
//! [`VoxelWorld::get_voxel`](crate::prelude::VoxelWorld::get_voxel) reads the
//! voxel edit, then the finest published chunk covering a position, which can
//! be a coarse chunk sampling it, then the fill hint. The [`Level0Reader`]
//! reads a position as the screen shows it up close instead: the voxel edit,
//! then the published level-0 chunk holding it, then that chunk's fill hint,
//! then the world's voxel lookup run on the caller's thread. Both share the
//! edit step, [`edit_step`], and the hint step, [`hint_step`].
//!
//! [`ChunkLookups`] keeps, for one read session, the fill hint and the voxel
//! lookup of each level-0 chunk position read, so each runs at most once per
//! chunk position. With light propagation, `column_is_open`, `CoarseWalk` and
//! `relight_edits` make their lookups on the caller's thread through it too.

use std::sync::{Arc, RwLock};

use bevy::platform::collections::HashMap;
use bevy::prelude::*;

use crate::chunk::{chunk_containing, ChunkData, ChunkKey};
use crate::chunk_map::ChunkMapData;
use crate::configuration::{
    FillHint, VoxelLookupDelegate, VoxelLookupFn, VoxelWorldConfig,
};
use crate::voxel::WorldVoxel;
use crate::voxel_edits::VoxelEditReader;

/// The voxel edit at `position`, if it has one: the first step of every
/// voxel read.
#[inline]
pub(crate) fn edit_step<I: Copy + PartialEq>(
    edits: &VoxelEditReader<I>,
    position: IVec3,
) -> Option<WorldVoxel<I>> {
    edits.get(position)
}

/// The voxel `hint` says fills its chunk, if it names one: the step of a
/// voxel read after the published chunks.
///
/// It speaks for a voxel only once the voxel's edit was read, since a fill
/// hint says nothing of edits.
#[inline]
pub(crate) fn hint_step<I>(hint: FillHint<I>) -> Option<WorldVoxel<I>> {
    match hint {
        FillHint::Uniform(voxel) => Some(voxel),
        FillHint::Unknown => None,
    }
}

/// What a fill hint says about a chunk position, without its material.
#[cfg_attr(not(feature = "light_propagation"), allow(dead_code))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Hinted {
    Unknown,
    Air,
    Solid,
}

#[cfg_attr(not(feature = "light_propagation"), allow(dead_code))]
impl Hinted {
    /// What `hint` says, or `Unknown` for a chunk position voxel edits reach.
    pub fn of<I: Copy + PartialEq>(hint: FillHint<I>, edited: bool) -> Self {
        match hint {
            _ if edited => Hinted::Unknown,
            FillHint::Unknown => Hinted::Unknown,
            FillHint::Uniform(voxel) if voxel.is_solid() => Hinted::Solid,
            FillHint::Uniform(_) => Hinted::Air,
        }
    }
}

/// The fill hint and the voxel lookup of each level-0 chunk position read in
/// one read session, each asked of the world at most once per position.
///
/// A lookup runs on the caller's thread, so its cost grows with the chunk
/// positions looked up and with what the world's generator does for each. A
/// session is one walk, one commit or one [`Level0Reader`]. It holds a lookup
/// closure per chunk position looked up, so it isn't kept across frames.
pub(crate) struct ChunkLookups<I> {
    delegate: VoxelLookupDelegate<I>,
    hints: HashMap<IVec3, FillHint<I>>,
    lookups: HashMap<IVec3, VoxelLookupFn<I>>,
}

impl<I: Copy + PartialEq> ChunkLookups<I> {
    /// A session that makes its lookups with `delegate`, a world's
    /// [`voxel_lookup_delegate`](VoxelWorldConfig::voxel_lookup_delegate).
    pub fn new(delegate: VoxelLookupDelegate<I>) -> Self {
        Self {
            delegate,
            hints: HashMap::new(),
            lookups: HashMap::new(),
        }
    }

    /// The fill hint of the level-0 chunk at `chunk` in `world`.
    pub fn hint<C: VoxelWorldConfig<MaterialIndex = I>>(
        &mut self,
        world: &C,
        chunk: IVec3,
    ) -> FillHint<I> {
        *self
            .hints
            .entry(chunk)
            .or_insert_with(|| world.chunk_fill_hint(ChunkKey::new(chunk, 0)))
    }

    /// What the fill hint of the level-0 chunk at `chunk` in `world` says,
    /// for a voxel whose edit was read first.
    #[cfg_attr(not(feature = "light_propagation"), allow(dead_code))]
    pub fn hinted<C: VoxelWorldConfig<MaterialIndex = I>>(
        &mut self,
        world: &C,
        chunk: IVec3,
    ) -> Hinted {
        Hinted::of(self.hint(world, chunk), false)
    }

    /// The voxel the world generates at `position`, from the lookup of the
    /// level-0 chunk holding it.
    pub fn lookup(&mut self, position: IVec3) -> WorldVoxel<I> {
        let chunk = chunk_containing(position);
        let delegate = &self.delegate;
        let lookup = self
            .lookups
            .entry(chunk)
            .or_insert_with(|| delegate(chunk, 0, None));
        lookup(position, None)
    }
}

/// Reads the world's voxels as the screen shows them up close, never from a
/// coarse chunk: the voxel edit, else the voxel in the published level-0
/// chunk holding the position, else that chunk's fill hint, else the world's
/// voxel lookup for that chunk, run on the caller's thread.
///
/// It never refuses and never waits, so a position far from every camera
/// reads what the world generates there. Its cost grows with the chunk
/// positions it reads that hold no published level-0 chunk and no uniform
/// fill hint, each running the world's generator on the caller's thread. A
/// position a coarse chunk covers counts among them: the reader looks past
/// the coarse sample to the voxel itself.
///
/// One reader is one read session, such as one lift or one stamp. It keeps
/// each chunk position's published chunk, fill hint and lookup from the first
/// read there, so a session's reads agree with each other, and a chunk
/// published after that read doesn't show.
#[allow(dead_code)]
pub(crate) struct Level0Reader<C: VoxelWorldConfig> {
    world: Arc<C>,
    edits: VoxelEditReader<C::MaterialIndex>,
    chunk_map: Arc<RwLock<ChunkMapData<C::MaterialIndex>>>,
    /// The published level-0 chunk at each chunk position read, or `None`
    /// where none was.
    published: HashMap<IVec3, Option<ChunkData<C::MaterialIndex>>>,
    lookups: ChunkLookups<C::MaterialIndex>,
}

#[allow(dead_code)]
impl<C: VoxelWorldConfig> Level0Reader<C> {
    /// A read session over world `world`'s voxel edits `edits` and published
    /// chunks `chunk_map`.
    pub fn new(
        world: Arc<C>,
        edits: VoxelEditReader<C::MaterialIndex>,
        chunk_map: Arc<RwLock<ChunkMapData<C::MaterialIndex>>>,
    ) -> Self {
        let lookups = ChunkLookups::new(world.voxel_lookup_delegate());
        Self {
            world,
            edits,
            chunk_map,
            published: HashMap::new(),
            lookups,
        }
    }

    /// The voxel at `position`.
    pub fn read(&mut self, position: IVec3) -> WorldVoxel<C::MaterialIndex> {
        if let Some(voxel) = edit_step(&self.edits, position) {
            return voxel;
        }
        let chunk = chunk_containing(position);
        let chunk_map = &self.chunk_map;
        let published = self.published.entry(chunk).or_insert_with(|| {
            chunk_map
                .read()
                .unwrap()
                .get(&ChunkKey::new(chunk, 0))
                .cloned()
        });
        if let Some(voxel) = published
            .as_ref()
            .and_then(|data| data.get_voxel_at_world_position(position))
        {
            return voxel;
        }
        if let Some(voxel) = hint_step(self.lookups.hint(self.world.as_ref(), chunk)) {
            return voxel;
        }
        self.lookups.lookup(position)
    }
}
