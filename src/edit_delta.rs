//! Edit deltas: the voxel edits of one level-0 chunk that changed in one
//! commit, as a value a game can send to another world or write to disk.
//!
//! Every commit writes an [`EditsCommitted`] message for each level-0 chunk
//! whose voxel edits changed, carrying the chunk's [`EditDelta`]. A world with
//! no networking can read it as an edit event that names the changed voxels.
//! In a networked game, the server forwards each delta to the clients whose
//! interest covers its chunk. A listen server's own world is the server, so it
//! forwards too. A pure client ignores the message, and a client never forwards
//! it: its deltas only repeat what it was sent and what it set itself.
//!
//! A delta states the chunk's edit sequence after it. A delta carrying `n`
//! voxels at sequence `seq` continues from a chunk whose sequence is
//! `seq - n`, so a receiver can tell when it missed one.
//!
//! A voxel's place in a delta is its index within the chunk,
//! `x + 32 * y + 1024 * z`, from the voxel's position relative to the chunk's
//! minimum voxel, each axis 0 to 31. This is not the padded position that
//! [`get_chunk_voxel_position`] returns, which counts from 1.
//!
//! A delta carrying few voxels lists them with their indices
//! ([`Voxels::Sparse`]). One carrying many takes the dense form
//! ([`Voxels::Dense`]): a [`Mask`] of the chunk's voxels, plus the voxels the
//! mask marks in index order.
//!
//! [`get_chunk_voxel_position`]: crate::prelude::get_chunk_voxel_position

use bevy::prelude::*;

use crate::{
    chunk::CHUNK_SIZE_I, configuration::VoxelWorldConfig, voxel::WorldVoxel,
    voxel_edits::ChangeSet,
};

/// How many voxels a level-0 chunk has.
pub(crate) const CHUNK_VOXELS: usize =
    (CHUNK_SIZE_I * CHUNK_SIZE_I * CHUNK_SIZE_I) as usize;

/// How many `u64` words a [`Mask`] holds: one bit per voxel of a chunk.
const MASK_WORDS: usize = CHUNK_VOXELS / 64;

/// A delta carrying this many voxels or more takes the dense form.
///
/// The sparse form stores an index beside every voxel, so its size grows with
/// the voxels. The dense form stores a mask the same size however few voxels
/// it marks, plus the voxels. This is about where the two take the same size
/// with a one-byte material index, so a delta with fewer voxels stays sparse
/// and one with more is as small as the dense form.
pub(crate) const DENSE_DELTA_AT: usize = 2048;

/// The index of the voxel at `local` within its level-0 chunk. `local` is its
/// position relative to the chunk's minimum voxel, 0 to 31 on each axis.
#[inline]
pub(crate) fn voxel_index(local: IVec3) -> usize {
    (local.x + local.y * CHUNK_SIZE_I + local.z * CHUNK_SIZE_I * CHUNK_SIZE_I) as usize
}

/// The position relative to its chunk's minimum voxel of the voxel at `index`:
/// the inverse of [`voxel_index`].
#[inline]
pub(crate) fn voxel_local(index: usize) -> IVec3 {
    let i = index as i32;
    IVec3::new(
        i % CHUNK_SIZE_I,
        i / CHUNK_SIZE_I % CHUNK_SIZE_I,
        i / (CHUNK_SIZE_I * CHUNK_SIZE_I),
    )
}

/// The voxel edits of one level-0 chunk that changed in one commit, with the
/// chunk's edit sequence after it.
///
/// The field order is part of the crate's public API, so that a serialised
/// delta keeps its layout across releases.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EditDelta<I> {
    /// The level-0 chunk position.
    pub chunk: IVec3,
    /// The chunk's edit sequence after this delta.
    pub seq: u64,
    /// Whether the delta carries every voxel edit the chunk has, replacing its
    /// edits, rather than the ones that changed.
    pub full: bool,
    /// The voxel edits, each with its new value. `WorldVoxel::Unset` is a
    /// value like any other.
    pub voxels: Voxels<I>,
}

/// The voxels of an [`EditDelta`], in one of two forms.
///
/// The crate writes [`Dense`](Self::Dense) for a delta of 2048 voxels or
/// more, and [`Sparse`](Self::Sparse) otherwise, but reads either.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Voxels<I> {
    /// Each voxel with its index within the chunk, in index order.
    Sparse(Vec<(u16, WorldVoxel<I>)>),
    /// The voxels the mask marks, in index order.
    Dense {
        mask: Mask,
        voxels: Vec<WorldVoxel<I>>,
    },
}

/// One bit per voxel of a level-0 chunk, saying which voxels a dense
/// [`EditDelta`] carries.
///
/// The bit for the voxel at index `i` is bit `i % 64` of word `i / 64`,
/// counting from the least significant bit.
#[derive(Clone, PartialEq, Eq)]
pub struct Mask(Box<[u64; MASK_WORDS]>);

impl<I: Copy> EditDelta<I> {
    /// How many voxels the delta carries.
    pub fn len(&self) -> usize {
        match &self.voxels {
            Voxels::Sparse(voxels) => voxels.len(),
            Voxels::Dense { voxels, .. } => voxels.len(),
        }
    }

    /// Whether the delta carries no voxels.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Every voxel the delta carries, by world position, in index order.
    pub fn iter(&self) -> Box<dyn Iterator<Item = (IVec3, WorldVoxel<I>)> + '_> {
        let min = self.chunk * CHUNK_SIZE_I;
        let at = move |index: usize| min + voxel_local(index);
        match &self.voxels {
            Voxels::Sparse(voxels) => {
                Box::new(voxels.iter().map(move |&(i, v)| (at(i as usize), v)))
            }
            Voxels::Dense { mask, voxels } => Box::new(
                mask.iter()
                    .zip(voxels.iter())
                    .map(move |(i, &v)| (at(i as usize), v)),
            ),
        }
    }

    /// The delta of the chunk at `chunk` carrying `voxels`, each by its index
    /// within the chunk, in the form its size calls for.
    pub(crate) fn new(
        chunk: IVec3,
        seq: u64,
        full: bool,
        mut voxels: Vec<(u16, WorldVoxel<I>)>,
    ) -> Self {
        voxels.sort_unstable_by_key(|&(index, _)| index);
        let voxels = if voxels.len() >= DENSE_DELTA_AT {
            let mut mask = Mask::default();
            for &(index, _) in &voxels {
                mask.insert(index);
            }
            Voxels::Dense {
                mask,
                voxels: voxels.into_iter().map(|(_, voxel)| voxel).collect(),
            }
        } else {
            Voxels::Sparse(voxels)
        };
        Self {
            chunk,
            seq,
            full,
            voxels,
        }
    }

    /// The delta of a commit's change set.
    pub(crate) fn from_change_set(set: &ChangeSet<I>) -> Self {
        // A replaced chunk's delta carries every edit it has left, which the
        // change set alone doesn't hold.
        debug_assert!(!set.replaced, "a replaced chunk's delta is its full delta");
        let min = set.chunk * CHUNK_SIZE_I;
        let voxels = set
            .voxels
            .iter()
            .filter_map(|&(position, voxel)| {
                Some((voxel_index(position - min) as u16, voxel?))
            })
            .collect();
        Self::new(set.chunk, set.seq, set.replaced, voxels)
    }
}

impl Mask {
    /// Whether the mask marks the voxel at `index`.
    pub fn contains(&self, index: u16) -> bool {
        let index = index as usize;
        self.0[index / 64] & (1 << (index % 64)) != 0
    }

    /// Marks the voxel at `index`.
    pub fn insert(&mut self, index: u16) {
        let index = index as usize;
        self.0[index / 64] |= 1 << (index % 64);
    }

    /// How many voxels the mask marks.
    pub fn len(&self) -> usize {
        self.0.iter().map(|word| word.count_ones() as usize).sum()
    }

    /// Whether the mask marks no voxels.
    pub fn is_empty(&self) -> bool {
        self.0.iter().all(|&word| word == 0)
    }

    /// The index of every voxel the mask marks, in order.
    pub fn iter(&self) -> impl Iterator<Item = u16> + '_ {
        self.0.iter().enumerate().flat_map(|(w, &word)| {
            let mut bits = word;
            std::iter::from_fn(move || {
                (bits != 0).then(|| {
                    let bit = bits.trailing_zeros();
                    bits &= bits - 1;
                    (w * 64) as u16 + bit as u16
                })
            })
        })
    }

    /// The mask's words, as laid out in the type's docs.
    pub fn words(&self) -> &[u64; MASK_WORDS] {
        &self.0
    }
}

impl Default for Mask {
    /// A mask marking no voxels.
    fn default() -> Self {
        Self(Box::new([0; MASK_WORDS]))
    }
}

impl std::fmt::Debug for Mask {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_set().entries(self.iter()).finish()
    }
}

/// Written by the commit of the voxel world `C`, in `PreUpdate` within
/// [`VoxelWorldSet`], once for each level-0 chunk whose voxel edits changed.
///
/// Systems running after the set read this frame's deltas. It carries no
/// chunk entity: a world whose chunks are coarser than level 0 has no entity at
/// the delta's chunk position.
///
/// [`VoxelWorldSet`]: crate::prelude::VoxelWorldSet
pub struct EditsCommitted<C: VoxelWorldConfig> {
    pub delta: EditDelta<C::MaterialIndex>,
}

impl<C: VoxelWorldConfig> Message for EditsCommitted<C> {}
