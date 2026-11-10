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
//! # Applying deltas
//!
//! A receiving world takes each delta through
//! [`VoxelWorld::apply_delta`], which checks the delta against the chunk's
//! edit sequence `s`, counting deltas applied to the chunk earlier in the
//! frame:
//!
//! - An incremental delta at `seq <= s` was already applied, and is skipped.
//! - One at `seq - n == s` applies, and the chunk's sequence becomes `seq`.
//! - Any other is a gap: the receiver missed a delta for the chunk. It is
//!   dropped and reported as an [`EditGap`].
//! - A full delta carries every edit the chunk has, so it needs nothing
//!   before it. One older than `s` is skipped, and any other replaces the
//!   chunk's edits.
//!
//! Applied voxels commit next `PreUpdate`, with the voxels set this frame, so
//! the chunk lifecycle and light handle them like any other voxel edit. A
//! receiver's commit writes [`EditsCommitted`] too, at the sequences it
//! received.
//!
//! # Channels and recovery
//!
//! Deltas belong on a reliable, ordered channel. The sequence check catches
//! a delta that arrives out of order or after one went missing, but not a
//! missing last delta, since nothing after it shows the gap. These flows are
//! game messages, each a chunk position one way and a full delta back, and
//! each heals what the channel let through:
//!
//! - **Gap request**: on an [`EditGap`], the client asks the server for the
//!   chunk's full delta, from [`VoxelWorld::full_delta`], and applies it.
//! - **Join**: a joining client receives the full delta of every chunk in
//!   [`VoxelWorld::edited_chunks`] that its interest covers. A client
//!   re-joining a server forgets its edits first, so that the server's
//!   sequences aren't skipped as older than its own.
//! - **Heartbeat**: on a channel that can drop deltas, the server now and
//!   then sends its edited chunks with their sequences. A client whose own
//!   sequence for a chunk is lower requests that chunk's full delta.
//!
//! # Schedule
//!
//! A delta applied in `Update` commits in the next frame's `PreUpdate`, the
//! same one frame later that [`VoxelWorld::set_voxel`] commits. A game that
//! wants a delta to commit in the frame it arrives applies it from a system
//! in `PreUpdate` ordered before [`VoxelWorldSet`].
//!
//! # Chunk layout
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
//! # Serialising
//!
//! With the crate's `serde` feature, [`EditDelta`], [`Voxels`], [`Mask`] and
//! [`WorldVoxel`] implement `Serialize` and `Deserialize`, in whatever format
//! the game picks. A delta's material index needs to implement them too; the
//! feature puts no bound on [`VoxelWorldConfig::MaterialIndex`].
//!
//! - The chunk position is three `i32`, x then y then z.
//! - The mask is a byte sequence of exactly 4096 bytes: its words in order,
//!   each little-endian. Any other length is a deserialise error.
//! - Everything else is serde's derived layout, so the order of the types'
//!   fields and variants is the layout. A change to that order is a breaking
//!   change to the crate's API.
//!
//! The crate adds no version to the layout. A game that keeps deltas across
//! releases, in a save file or between a client and a server of different
//! builds, wraps them in its own envelope that starts with a version. A good
//! start is `postcard` behind one version byte, with the bytes compressed by
//! `lz4_flex` when written to a file.
//!
//! [`get_chunk_voxel_position`]: crate::prelude::get_chunk_voxel_position
//! [`VoxelWorld::apply_delta`]: crate::prelude::VoxelWorld::apply_delta
//! [`VoxelWorld::full_delta`]: crate::prelude::VoxelWorld::full_delta
//! [`VoxelWorld::edited_chunks`]: crate::prelude::VoxelWorld::edited_chunks
//! [`VoxelWorld::set_voxel`]: crate::prelude::VoxelWorld::set_voxel
//! [`VoxelWorldSet`]: crate::prelude::VoxelWorldSet

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
/// A world applies a delta through
/// [`VoxelWorld::apply_delta`](crate::prelude::VoxelWorld::apply_delta),
/// which checks it against the chunk's edit sequence `s` there:
///
/// | delta | skipped | applied | reported as an [`EditGap`] |
/// |-------|---------|---------|----------------------------|
/// | incremental, `n` voxels | `seq <= s` | `seq - n == s` | otherwise |
/// | full | `seq < s` | `seq >= s` | never |
///
/// With the crate's `serde` feature, a delta serialises in any format, its
/// chunk position as three `i32` and a dense delta's [`Mask`] as 4096 bytes.
/// The order of its fields, and of the fields and variants of the types it
/// holds, is part of the crate's public API, so that a serialised delta keeps
/// its layout across releases. The layout carries no version: a game that
/// keeps deltas across its own releases puts one in the envelope it wraps them
/// in, such as a version byte before `postcard` bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EditDelta<I> {
    /// The level-0 chunk position.
    #[cfg_attr(feature = "serde", serde(with = "chunk_position"))]
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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
///
/// With the crate's `serde` feature, a mask serialises as a byte sequence of
/// exactly 4096 bytes, its words in order, each little-endian. Deserialising
/// any other length is an error.
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

    /// Whether a world whose chunk is at edit sequence `local` applies this
    /// delta, by the rule in the module docs. Returns `Ok(false)` for a delta
    /// already applied, and the gap for one that doesn't continue from
    /// `local` or doesn't fit a chunk.
    pub(crate) fn applies_at(&self, local: u64) -> Result<bool, EditGap> {
        let gap = EditGap {
            chunk: self.chunk,
            local,
            received: self.seq,
        };
        if !self.fits_a_chunk() {
            return Err(gap);
        }
        if self.full {
            return Ok(self.seq >= local);
        }
        if self.seq <= local {
            return Ok(false);
        }
        match self.seq.checked_sub(self.len() as u64) {
            Some(before) if before == local => Ok(true),
            _ => Err(gap),
        }
    }

    /// Whether every voxel index is within a chunk, and a dense delta has a
    /// voxel for each voxel its mask marks.
    fn fits_a_chunk(&self) -> bool {
        match &self.voxels {
            Voxels::Sparse(voxels) => voxels
                .iter()
                .all(|&(index, _)| (index as usize) < CHUNK_VOXELS),
            Voxels::Dense { mask, voxels } => mask.len() == voxels.len(),
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

    /// The mask with the given words, as laid out in the type's docs.
    pub fn from_words(words: [u64; MASK_WORDS]) -> Self {
        Self(Box::new(words))
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

/// How many bytes a serialised [`Mask`] takes.
#[cfg(feature = "serde")]
const MASK_BYTES: usize = MASK_WORDS * 8;

#[cfg(feature = "serde")]
impl serde::Serialize for Mask {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut bytes = [0; MASK_BYTES];
        for (to, word) in bytes.as_chunks_mut::<8>().0.iter_mut().zip(self.0.iter()) {
            *to = word.to_le_bytes();
        }
        serializer.serialize_bytes(&bytes)
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for Mask {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Self, D::Error> {
        deserializer.deserialize_bytes(MaskVisitor)
    }
}

#[cfg(feature = "serde")]
struct MaskVisitor;

#[cfg(feature = "serde")]
impl<'de> serde::de::Visitor<'de> for MaskVisitor {
    type Value = Mask;

    fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{MASK_BYTES} bytes")
    }

    fn visit_bytes<E: serde::de::Error>(self, bytes: &[u8]) -> Result<Mask, E> {
        if bytes.len() != MASK_BYTES {
            return Err(E::invalid_length(bytes.len(), &self));
        }
        let mut words = [0; MASK_WORDS];
        for (word, from) in words.iter_mut().zip(bytes.as_chunks::<8>().0) {
            *word = u64::from_le_bytes(*from);
        }
        Ok(Mask::from_words(words))
    }

    // Formats with no byte type, such as JSON, write bytes as a sequence.
    fn visit_seq<A: serde::de::SeqAccess<'de>>(
        self,
        mut seq: A,
    ) -> Result<Mask, A::Error> {
        let mut bytes = Vec::with_capacity(MASK_BYTES);
        while let Some(byte) = seq.next_element()? {
            bytes.push(byte);
            if bytes.len() > MASK_BYTES {
                return Err(serde::de::Error::invalid_length(bytes.len(), &self));
            }
        }
        self.visit_bytes(&bytes)
    }
}

/// Serialises a chunk position as three `i32`, the layout glam's own `serde`
/// feature gives an `IVec3`, without the crate turning that feature on.
#[cfg(feature = "serde")]
mod chunk_position {
    use bevy::math::IVec3;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(
        chunk: &IVec3,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        chunk.to_array().serialize(serializer)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<IVec3, D::Error> {
        <[i32; 3]>::deserialize(deserializer).map(IVec3::from_array)
    }
}

/// An edit delta that a world dropped because it doesn't continue from the
/// chunk's edit sequence: the world missed a delta for the chunk in between.
///
/// The game asks the sender for the chunk's full delta, and applies it.
/// [`VoxelWorld::apply_delta`] also reports a delta whose voxels don't fit a
/// chunk this way, since the full delta repairs that too.
///
/// [`VoxelWorld::apply_delta`]: crate::prelude::VoxelWorld::apply_delta
#[must_use]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EditGap {
    /// The level-0 chunk position.
    pub chunk: IVec3,
    /// The chunk's edit sequence in the world that dropped the delta,
    /// counting the deltas applied to it this frame.
    pub local: u64,
    /// The edit sequence of the dropped delta.
    pub received: u64,
}

impl std::fmt::Display for EditGap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "edit delta at sequence {} doesn't continue from sequence {} of chunk {}",
            self.received, self.local, self.chunk
        )
    }
}

impl std::error::Error for EditGap {}

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
