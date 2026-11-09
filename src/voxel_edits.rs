//! Every voxel edit of a voxel world, kept for as long as the world exists,
//! and which chunks each voxel edit makes stale.
//!
//! Voxel edits can be set from systems running in parallel, and are committed
//! once a frame. The module reads no Bevy world state, so it can be tested
//! with plain values.
//!
//! The edits of each level-0 chunk live in a [`Bucket`] with the chunk's edit
//! sequence: how many of the chunk's voxel edits have ever changed in a
//! commit. A bucket starts as a map from position to voxel and turns into an
//! array of every voxel in the chunk once it holds many edits, so that setting
//! most of a chunk, as an animation does every frame, costs an array write per
//! voxel rather than a hash map insert.
//!
//! A commit reports a [`ChangeSet`] per level-0 chunk whose edits changed: the
//! voxels whose committed edit changed, with their new edits. The chunk
//! lifecycle marks the chunks they reach stale through
//! [`ChangeSet::stale_boxes`], and light reads the voxels themselves.

use std::{
    marker::PhantomData,
    sync::{Arc, Mutex, RwLock, RwLockReadGuard},
};

use bevy::{platform::collections::HashMap, prelude::*};

use crate::{
    chunk::{chunk_containing, ChunkKey, CHUNK_SIZE_I},
    configuration::LodLevel,
    voxel::WorldVoxel,
};

type Voxels<I> = HashMap<IVec3, WorldVoxel<I>>;

/// How many voxels a level-0 chunk has.
const BUCKET_VOXELS: usize = (CHUNK_SIZE_I * CHUNK_SIZE_I * CHUNK_SIZE_I) as usize;

/// A bucket holding this many edits turns from a map into an array.
///
/// The array takes the same memory however few edits it holds, and a commit
/// compares every one of its slots; the map takes memory per edit and hashes
/// each read and write. This is about where the two take the same memory, so
/// a chunk with fewer edits stays a map and one with more is as small as an
/// array.
const DENSE_AT: usize = 2048;

/// A chunk with more changed voxels than this in one commit is marked stale as
/// the one box around them, rather than voxel by voxel.
///
/// Marking voxel by voxel finds exactly the chunks whose padding reaches a
/// changed voxel, but its cost grows with the voxels. The box costs the same
/// however many voxels it holds, and can mark a neighbouring chunk that none
/// of the voxels reach, which then remeshes for nothing.
const BOXED_AT: usize = 64;

/// The voxel edits of one level-0 chunk, with its edit sequence.
#[derive(Clone)]
pub(crate) struct Bucket<I> {
    /// The chunk's minimum voxel.
    min: IVec3,
    store: Store<I>,
    /// The chunk's edit sequence. Only committed buckets keep it.
    seq: u64,
}

#[derive(Clone)]
enum Store<I> {
    Sparse(Voxels<I>),
    Dense {
        /// `BUCKET_VOXELS` slots, x varying fastest.
        voxels: Box<[Option<WorldVoxel<I>>]>,
        /// How many slots are set.
        count: usize,
    },
}

impl<I: Copy + PartialEq> Bucket<I> {
    fn new(chunk_pos: IVec3) -> Self {
        Self {
            min: chunk_pos * CHUNK_SIZE_I,
            store: Store::Sparse(Voxels::default()),
            seq: 0,
        }
    }

    #[inline]
    fn index(&self, position: IVec3) -> usize {
        let local = position - self.min;
        (local.x + local.y * CHUNK_SIZE_I + local.z * CHUNK_SIZE_I * CHUNK_SIZE_I)
            as usize
    }

    fn position_of(&self, index: usize) -> IVec3 {
        let i = index as i32;
        self.min
            + IVec3::new(
                i % CHUNK_SIZE_I,
                i / CHUNK_SIZE_I % CHUNK_SIZE_I,
                i / (CHUNK_SIZE_I * CHUNK_SIZE_I),
            )
    }

    #[inline]
    pub fn get(&self, position: IVec3) -> Option<WorldVoxel<I>> {
        match &self.store {
            Store::Sparse(voxels) => voxels.get(&position).copied(),
            Store::Dense { voxels, .. } => voxels[self.index(position)],
        }
    }

    /// Sets the voxel at `position`. Returns whether that changed it.
    #[inline]
    pub fn insert(&mut self, position: IVec3, voxel: WorldVoxel<I>) -> bool {
        if let Store::Sparse(voxels) = &self.store {
            if voxels.len() >= DENSE_AT {
                self.make_dense();
            }
        }
        let index = self.index(position);
        match &mut self.store {
            Store::Sparse(voxels) => voxels.insert(position, voxel) != Some(voxel),
            Store::Dense { voxels, count } => {
                let slot = &mut voxels[index];
                if *slot == Some(voxel) {
                    return false;
                }
                if slot.is_none() {
                    *count += 1;
                }
                *slot = Some(voxel);
                true
            }
        }
    }

    /// Turns a sparse bucket dense, for a region write that will fill it.
    pub fn make_dense(&mut self) {
        if let Store::Sparse(sparse) = &self.store {
            let mut voxels = vec![None; BUCKET_VOXELS].into_boxed_slice();
            let min = self.min;
            for (&position, &voxel) in sparse {
                let local = position - min;
                voxels[(local.x
                    + local.y * CHUNK_SIZE_I
                    + local.z * CHUNK_SIZE_I * CHUNK_SIZE_I)
                    as usize] = Some(voxel);
            }
            let count = sparse.len();
            self.store = Store::Dense { voxels, count };
        }
    }

    /// Every edit in the bucket.
    pub fn iter(&self) -> Box<dyn Iterator<Item = (IVec3, WorldVoxel<I>)> + '_> {
        match &self.store {
            Store::Sparse(voxels) => Box::new(voxels.iter().map(|(p, v)| (*p, *v))),
            Store::Dense { voxels, .. } => Box::new(
                voxels
                    .iter()
                    .enumerate()
                    .filter_map(|(i, v)| v.map(|v| (self.position_of(i), v))),
            ),
        }
    }

    /// The edits in `pending` that differ from this bucket's.
    fn changes_from(&self, pending: &Bucket<I>) -> Vec<(IVec3, Option<WorldVoxel<I>>)> {
        let mut changes = Vec::new();
        match (&pending.store, &self.store) {
            (
                Store::Dense {
                    voxels: pending, ..
                },
                Store::Dense {
                    voxels: committed, ..
                },
            ) => {
                for (i, (p, c)) in pending.iter().zip(committed.iter()).enumerate() {
                    if let Some(v) = p {
                        if *c != Some(*v) {
                            changes.push((self.position_of(i), Some(*v)));
                        }
                    }
                }
            }
            _ => {
                for (position, voxel) in pending.iter() {
                    if self.get(position) != Some(voxel) {
                        changes.push((position, Some(voxel)));
                    }
                }
            }
        }
        changes
    }
}

/// The edited voxels of one chunk, shared with the chunk tasks that read them.
pub(crate) type EditBucket<I> = Arc<Bucket<I>>;

type BucketMap<I> = HashMap<IVec3, EditBucket<I>>;

/// Committed edits, by chunk.
type Committed<I> = Arc<RwLock<BucketMap<I>>>;

/// The voxels of one level-0 chunk whose committed edits changed in a commit.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ChangeSet<I> {
    /// The level-0 chunk position.
    pub chunk: IVec3,
    /// Whether the commit replaced the chunk's edits rather than changing
    /// some of them, so the edit sequence before the commit can't be worked
    /// out from the one after it and the voxels' count.
    pub replaced: bool,
    /// Each voxel whose committed edit changed, with its new edit, or `None`
    /// where its edit was removed.
    pub voxels: Vec<(IVec3, Option<WorldVoxel<I>>)>,
}

impl<I> ChangeSet<I> {
    /// Boxes holding every changed voxel: one box per voxel, or the one box
    /// around them when there are more than [`BOXED_AT`].
    /// [`chunks_including_box`] gives the chunks each box makes stale.
    pub fn stale_boxes(&self) -> impl Iterator<Item = EditedBox> + '_ {
        let positions = self.voxels.iter().map(|(position, _)| *position);
        let around = (self.voxels.len() > BOXED_AT).then(|| {
            positions.clone().fold(
                EditedBox {
                    min: IVec3::MAX,
                    max: IVec3::MIN,
                },
                |around, position| EditedBox {
                    min: around.min.min(position),
                    max: around.max.max(position),
                },
            )
        });
        let one_by_one =
            around
                .is_none()
                .then_some(positions.map(|position| EditedBox {
                    min: position,
                    max: position,
                }));
        around.into_iter().chain(one_by_one.into_iter().flatten())
    }
}

/// A box of voxels, from `min` to `max`, both inclusive.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct EditedBox {
    pub min: IVec3,
    pub max: IVec3,
}

fn bucket_mut<I: Copy + PartialEq>(
    map: &mut BucketMap<I>,
    chunk: IVec3,
) -> &mut Bucket<I> {
    Arc::make_mut(
        map.entry(chunk)
            .or_insert_with(|| Arc::new(Bucket::new(chunk))),
    )
}

/// Every voxel edit of the voxel world `C`.
///
/// Edits set since the last commit are pending: they are readable right away,
/// and only the last voxel set at a position is committed. Committed edits are
/// grouped by chunk, so a chunk task takes the edits it reads when it starts and
/// holds no lock while it runs.
///
/// Public only because `VoxelWorld` reads it; this module isn't exported.
#[derive(Resource)]
pub struct VoxelEdits<C, I> {
    /// The edits set since the last commit, by chunk. The map is copied only
    /// when an edit is set while a reader holds it.
    pending: Mutex<Arc<BucketMap<I>>>,
    committed: Committed<I>,
    _marker: PhantomData<C>,
}

impl<C, I> Default for VoxelEdits<C, I> {
    fn default() -> Self {
        Self {
            pending: Mutex::default(),
            committed: Arc::default(),
            _marker: PhantomData,
        }
    }
}

impl<C, I: Copy + PartialEq> VoxelEdits<C, I> {
    /// Sets the voxel at `position`, replacing any voxel set there since the
    /// last commit.
    pub(crate) fn set(&self, position: IVec3, voxel: WorldVoxel<I>) {
        let chunk = chunk_containing(position);
        let mut pending = self.pending.lock().unwrap();
        bucket_mut(Arc::make_mut(&mut pending), chunk).insert(position, voxel);
    }

    /// Sets the voxels in the box from `min` inclusive to `max` exclusive for
    /// which `voxel` returns one. A chunk the box covers a good part of gets a
    /// dense bucket up front.
    pub(crate) fn set_region(
        &self,
        min: IVec3,
        max: IVec3,
        mut voxel: impl FnMut(IVec3) -> Option<WorldVoxel<I>>,
    ) {
        if (max - min).cmple(IVec3::ZERO).any() {
            return;
        }
        let mut pending = self.pending.lock().unwrap();
        let map = Arc::make_mut(&mut pending);
        let chunk_min = chunk_containing(min);
        let chunk_max = chunk_containing(max - 1);
        for cz in chunk_min.z..=chunk_max.z {
            for cy in chunk_min.y..=chunk_max.y {
                for cx in chunk_min.x..=chunk_max.x {
                    let chunk = IVec3::new(cx, cy, cz);
                    let lo = (chunk * CHUNK_SIZE_I).max(min);
                    let hi = ((chunk + 1) * CHUNK_SIZE_I).min(max);
                    let bucket = bucket_mut(map, chunk);
                    if (hi - lo).element_product() as usize >= DENSE_AT {
                        bucket.make_dense();
                    }
                    for z in lo.z..hi.z {
                        for y in lo.y..hi.y {
                            for x in lo.x..hi.x {
                                let position = IVec3::new(x, y, z);
                                if let Some(v) = voxel(position) {
                                    bucket.insert(position, v);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// Returns a reader that works from any thread.
    ///
    /// The reader sees the pending edits as they were when it was made, and the
    /// committed edits as they are at the time of each read.
    pub(crate) fn reader(&self) -> VoxelEditReader<I> {
        VoxelEditReader {
            pending: self.pending.lock().unwrap().clone(),
            committed: self.committed.clone(),
        }
    }

    /// Commits the pending edits. Returns a change set for each level-0 chunk
    /// whose committed edits changed, and moves its edit sequence on by the
    /// number of voxels in it.
    ///
    /// An edit that sets a voxel to the voxel already committed there changes
    /// nothing. So setting the same voxel every frame, or setting a voxel and
    /// then back, doesn't remesh its chunk.
    ///
    /// A chunk's committed edits are copied only when a running chunk task
    /// still holds them.
    pub(crate) fn commit(&self) -> Vec<ChangeSet<I>> {
        // The pending edits stay locked until they are committed, so a reader
        // made during the commit finds each edit either pending or committed.
        let mut pending = self.pending.lock().unwrap();
        if pending.is_empty() {
            return Vec::new();
        }
        let mut buckets = {
            #[cfg(feature = "trace")]
            let _span = bevy::log::info_span!("wait_voxel_edits_write").entered();
            self.committed.write().unwrap()
        };
        let mut sets = Vec::new();
        for (&chunk, pending_bucket) in pending.iter() {
            let changes = match buckets.get(&chunk) {
                Some(committed) => committed.changes_from(pending_bucket),
                None => Bucket::new(chunk).changes_from(pending_bucket),
            };
            if changes.is_empty() {
                continue;
            }
            let committed = Arc::make_mut(
                buckets
                    .entry(chunk)
                    .or_insert_with(|| Arc::new(Bucket::new(chunk))),
            );
            if changes.len() >= DENSE_AT {
                committed.make_dense();
            }
            for &(position, voxel) in &changes {
                if let Some(voxel) = voxel {
                    committed.insert(position, voxel);
                }
            }
            committed.seq += changes.len() as u64;
            sets.push(ChangeSet {
                chunk,
                replaced: false,
                voxels: changes,
            });
        }
        match Arc::get_mut(&mut pending) {
            Some(map) => map.clear(),
            None => *pending = Arc::default(),
        }
        sets
    }

    /// The edits the chunk task of the chunk at `key` reads: those within the
    /// chunk's extent and its padding. `padding` is the chunk's [`padding_depth`].
    ///
    /// [`padding_depth`]: crate::chunk::padding_depth
    pub(crate) fn chunk_edits(&self, key: ChunkKey, padding: IVec3) -> ChunkEdits<I> {
        let buckets = self.committed.read().unwrap();
        let chunk_pos = key.position;
        // The padding of a level-0 chunk is at most its extent deep, so it stays
        // within the 27 level-0 chunks around it, whatever the chunk's shape.
        if key.lod_level == 0 {
            let mut around: [Option<EditBucket<I>>; 27] = std::array::from_fn(|_| None);
            if !buckets.is_empty() {
                for (i, bucket) in around.iter_mut().enumerate() {
                    *bucket = buckets.get(&(chunk_pos + around_offset(i))).cloned();
                }
            }
            return ChunkEdits {
                chunk_pos,
                buckets: Buckets::Around(around),
            };
        }

        let (min, max) = buckets_read_by(key, padding);
        let within = buckets
            .iter()
            .filter(|(position, _)| {
                position.cmpge(min).all() && position.cmplt(max).all()
            })
            .map(|(position, bucket)| (*position, bucket.clone()))
            .collect();
        ChunkEdits {
            chunk_pos,
            buckets: Buckets::Within(within),
        }
    }

    /// The committed edits in each level-0 chunk at `positions`, in order, or
    /// `None` for a chunk with none.
    #[cfg(feature = "light_propagation")]
    pub(crate) fn buckets(
        &self,
        positions: impl Iterator<Item = IVec3>,
    ) -> Vec<Option<EditBucket<I>>> {
        let buckets = self.committed.read().unwrap();
        positions
            .map(|position| buckets.get(&position).cloned())
            .collect()
    }

    /// The edit sequence of the level-0 chunk at `position`: 0 for a chunk
    /// with no edits. Work that reads the chunk's voxels notes it when it
    /// starts, and its result is stale if the sequence has moved since.
    #[cfg_attr(not(any(test, feature = "light_propagation")), expect(dead_code))]
    pub(crate) fn edit_sequence(&self, position: IVec3) -> u64 {
        let buckets = self.committed.read().unwrap();
        if buckets.is_empty() {
            return 0;
        }
        buckets.get(&position).map_or(0, |bucket| bucket.seq)
    }

    /// Whether edits have been set since the last commit.
    #[cfg(feature = "light_propagation")]
    pub(crate) fn has_pending(&self) -> bool {
        !self.pending.lock().unwrap().is_empty()
    }

    /// The committed edits, read once for a frame.
    pub(crate) fn committed(&self) -> CommittedEdits<'_, I> {
        CommittedEdits(self.committed.read().unwrap())
    }
}

/// The committed edits, held for reading.
pub(crate) struct CommittedEdits<'a, I>(RwLockReadGuard<'a, BucketMap<I>>);

impl<I> CommittedEdits<'_, I> {
    /// Whether an edit lies in a level-0 chunk that the chunk task of the chunk
    /// at `key` reads, within its extent or reached by its padding.
    ///
    /// Such an edit may change the chunk's voxels, so the chunk's fill hint
    /// can't be trusted. `padding` is the chunk's [`padding_depth`].
    ///
    /// [`padding_depth`]: crate::chunk::padding_depth
    pub fn reach(&self, key: ChunkKey, padding: IVec3) -> bool {
        let buckets = &self.0;
        if buckets.is_empty() {
            return false;
        }
        if key.lod_level == 0 {
            return (0..27)
                .any(|i| buckets.contains_key(&(key.position + around_offset(i))));
        }
        let (min, max) = buckets_read_by(key, padding);
        buckets
            .keys()
            .any(|position| position.cmpge(min).all() && position.cmplt(max).all())
    }
}

/// The offset of the `i`th of the 27 level-0 chunks around a chunk, x varying
/// fastest, as [`Buckets::Around`] stores them.
fn around_offset(i: usize) -> IVec3 {
    let i = i as i32;
    IVec3::new(i % 3, i / 3 % 3, i / 9) - 1
}

/// The level-0 chunk positions whose edits the chunk task of the chunk at
/// `key`, above level 0, reads. Returns `min` inclusive to `max` exclusive on
/// every axis.
///
/// `padding` is how many voxels deep the chunk's padding is along each axis. It
/// reaches that many voxels' worth of level-0 chunks past the extent.
fn buckets_read_by(key: ChunkKey, padding: IVec3) -> (IVec3, IVec3) {
    let padding = (padding + CHUNK_SIZE_I - 1) / CHUNK_SIZE_I;
    (
        key.position - padding,
        key.position + key.extent() + padding,
    )
}

/// The chunks at `lod_level` whose padded voxel data includes the voxel at
/// `position`.
///
/// That is the chunk containing the voxel, plus each neighbouring chunk across
/// a face, edge or corner that the voxel lies within `padding` voxels of along
/// each axis. `padding` is the [`padding_depth`] of the chunks at `lod_level`.
///
/// [`padding_depth`]: crate::chunk::padding_depth
#[cfg(test)]
pub(crate) fn chunks_including(
    position: IVec3,
    lod_level: LodLevel,
    padding: IVec3,
) -> impl Iterator<Item = ChunkKey> {
    chunks_including_box(position, position, lod_level, padding)
}

/// The chunks at `lod_level` whose padded voxel data includes any voxel of the
/// box from `min` to `max`, both inclusive.
///
/// A chunk at chunk coordinate `c` along an axis holds the padded voxels from
/// `c * E - P` up to but not including `c * E + E + P`, with `E` its voxel
/// extent and `P` its padding depth.
pub(crate) fn chunks_including_box(
    min: IVec3,
    max: IVec3,
    lod_level: LodLevel,
    padding: IVec3,
) -> impl Iterator<Item = ChunkKey> {
    let voxel_extent = IVec3::splat(CHUNK_SIZE_I << lod_level);
    let low = (min - padding - voxel_extent).div_euclid(voxel_extent) + 1;
    let high = (max + padding).div_euclid(voxel_extent);
    let extent = 1 << lod_level;
    (low.z..=high.z).flat_map(move |z| {
        (low.y..=high.y).flat_map(move |y| {
            (low.x..=high.x).map(move |x| ChunkKey {
                position: IVec3::new(x, y, z) * extent,
                lod_level,
            })
        })
    })
}

/// Reads voxel edits from any thread.
pub(crate) struct VoxelEditReader<I> {
    pending: Arc<BucketMap<I>>,
    committed: Committed<I>,
}

impl<I: Copy + PartialEq> VoxelEditReader<I> {
    pub fn get(&self, position: IVec3) -> Option<WorldVoxel<I>> {
        let chunk = chunk_containing(position);
        if let Some(voxel) = self.pending.get(&chunk).and_then(|b| b.get(position)) {
            return Some(voxel);
        }
        let buckets = self.committed.read().unwrap();
        buckets.get(&chunk)?.get(position)
    }
}

/// The voxel edits a chunk task reads, taken when it starts: those of the
/// level-0 chunks in its extent and of the ones its padding reaches into.
pub(crate) struct ChunkEdits<I> {
    /// The chunk's minimum corner, in level-0 chunks.
    chunk_pos: IVec3,
    buckets: Buckets<I>,
}

/// The edit buckets a chunk task reads, by level-0 chunk.
enum Buckets<I> {
    /// For a level-0 chunk: by offset from the chunk, x varying fastest.
    Around([Option<EditBucket<I>>; 27]),
    /// For a chunk above level 0: by level-0 chunk position.
    Within(HashMap<IVec3, EditBucket<I>>),
}

impl<I: Copy + PartialEq> ChunkEdits<I> {
    pub fn is_empty(&self) -> bool {
        match &self.buckets {
            Buckets::Around(around) => around.iter().all(Option::is_none),
            Buckets::Within(within) => within.is_empty(),
        }
    }

    // Kept out of the chunk task's body: inlined there, it changes how the
    // compiler lays out the task's loop for chunks without edits, which most
    // chunks are, and slows their generation though they never call it.
    #[inline(never)]
    pub fn get(&self, position: IVec3) -> Option<WorldVoxel<I>> {
        let chunk_pos = chunk_containing(position);
        match &self.buckets {
            Buckets::Around(around) => {
                let offset = chunk_pos - self.chunk_pos + 1;
                if offset.cmplt(IVec3::ZERO).any() || offset.cmpgt(IVec3::splat(2)).any()
                {
                    return None;
                }
                let i = (offset.x + offset.y * 3 + offset.z * 9) as usize;
                around[i].as_ref()?.get(position)
            }
            Buckets::Within(within) => within.get(&chunk_pos)?.get(position),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk::{padding_depth, PADDED_CHUNK_SIZE};
    use bevy::platform::collections::HashSet;
    use std::sync::atomic::{AtomicU32, Ordering};

    type Edits = VoxelEdits<(), u8>;

    const LAST: i32 = CHUNK_SIZE_I - 1;

    /// The padding depth of a chunk at `lod_level` with the default data shape.
    fn default_padding(lod_level: LodLevel) -> IVec3 {
        padding_depth(UVec3::splat(PADDED_CHUNK_SIZE), lod_level)
    }

    /// Sets one voxel and commits it, returning the voxels that changed.
    fn commit_one(edits: &Edits, position: IVec3, voxel: WorldVoxel) -> Vec<IVec3> {
        edits.set(position, voxel);
        changed_voxels(&edits.commit())
    }

    /// The positions of every voxel in `sets`.
    fn changed_voxels(sets: &[ChangeSet<u8>]) -> Vec<IVec3> {
        sets.iter()
            .flat_map(|set| set.voxels.iter().map(|(position, _)| *position))
            .collect()
    }

    /// The voxels of the chunk at `chunk_pos`, x varying fastest.
    fn every_voxel_of(chunk_pos: IVec3) -> impl Iterator<Item = IVec3> {
        let min = chunk_pos * CHUNK_SIZE_I;
        (0..CHUNK_SIZE_I).flat_map(move |z| {
            (0..CHUNK_SIZE_I).flat_map(move |y| {
                (0..CHUNK_SIZE_I).map(move |x| min + IVec3::new(x, y, z))
            })
        })
    }

    #[test]
    fn a_commit_reports_the_changed_voxels_of_each_chunk_with_their_new_edits() {
        let edits = Edits::default();
        let in_origin = IVec3::new(1, 2, 3);
        let below_origin = IVec3::new(-1, 2, 3);
        edits.set(in_origin, WorldVoxel::Solid(1));
        edits.set(below_origin, WorldVoxel::Air);

        let mut sets = edits.commit();
        sets.sort_by_key(|set| set.chunk.x);

        assert_eq!(sets.len(), 2);
        assert_eq!(sets[0].chunk, IVec3::new(-1, 0, 0));
        assert_eq!(sets[0].voxels, vec![(below_origin, Some(WorldVoxel::Air))]);
        assert_eq!(sets[1].chunk, IVec3::ZERO);
        assert_eq!(
            sets[1].voxels,
            vec![(in_origin, Some(WorldVoxel::Solid(1)))]
        );
        assert!(sets.iter().all(|set| !set.replaced));
    }

    #[test]
    fn unset_is_an_edit_like_any_other() {
        let edits = Edits::default();
        let position = IVec3::new(4, 4, 4);

        assert_eq!(
            commit_one(&edits, position, WorldVoxel::Unset),
            vec![position]
        );
        assert_eq!(edits.reader().get(position), Some(WorldVoxel::Unset));
    }

    #[test]
    fn the_edit_sequence_counts_every_voxel_whose_edit_changed() {
        let edits = Edits::default();
        assert_eq!(edits.edit_sequence(IVec3::ZERO), 0);

        for x in 0..3 {
            edits.set(IVec3::new(x, 0, 0), WorldVoxel::Solid(1));
        }
        edits.commit();
        assert_eq!(edits.edit_sequence(IVec3::ZERO), 3);

        // The same voxels again change nothing; one new voxel moves it on.
        edits.set(IVec3::ZERO, WorldVoxel::Solid(1));
        edits.set(IVec3::X, WorldVoxel::Solid(2));
        edits.commit();
        assert_eq!(edits.edit_sequence(IVec3::ZERO), 4);
        assert_eq!(edits.edit_sequence(IVec3::X), 0);
    }

    #[test]
    fn a_region_whose_voxels_all_return_none_leaves_no_edits() {
        let edits = Edits::default();
        edits.set_region(IVec3::ZERO, IVec3::splat(CHUNK_SIZE_I), |_| None);

        assert!(edits.commit().is_empty());
        assert!(!edits
            .committed()
            .reach(ChunkKey::new(IVec3::ZERO, 0), default_padding(0)));
    }

    #[test]
    fn an_empty_or_inverted_region_sets_nothing() {
        let edits = Edits::default();
        let set = |min: IVec3, max: IVec3| {
            edits.set_region(min, max, |position| {
                panic!("asked for {position} in an empty box")
            });
        };
        set(IVec3::ZERO, IVec3::ZERO);
        set(IVec3::ZERO, IVec3::new(4, 0, 4));
        set(IVec3::splat(4), IVec3::ZERO);

        assert!(edits.commit().is_empty());
    }

    #[test]
    fn a_region_sets_the_voxels_it_returns_across_chunks() {
        let edits = Edits::default();
        let (min, max) = (IVec3::new(-2, 0, 0), IVec3::new(2, 2, 1));
        edits.set_region(min, max, |position| {
            (position.y == 0).then_some(WorldVoxel::Solid(position.x as u8))
        });

        let mut changed = changed_voxels(&edits.commit());
        changed.sort_by_key(|position| position.x);
        assert_eq!(
            changed,
            (-2..2).map(|x| IVec3::new(x, 0, 0)).collect::<Vec<_>>()
        );
        let reader = edits.reader();
        assert_eq!(
            reader.get(IVec3::new(-1, 0, 0)),
            Some(WorldVoxel::Solid(255))
        );
        assert_eq!(reader.get(IVec3::new(-1, 1, 0)), None);
    }

    #[test]
    fn a_chunk_with_many_edits_keeps_each_of_them() {
        let edits = Edits::default();
        let voxel = |position: IVec3| WorldVoxel::Solid((position.x + position.y) as u8);
        let set: Vec<IVec3> = every_voxel_of(IVec3::ZERO).take(DENSE_AT + 100).collect();
        for &position in &set {
            edits.set(position, voxel(position));
        }

        assert_eq!(edits.commit()[0].voxels.len(), set.len());
        assert_eq!(edits.edit_sequence(IVec3::ZERO), set.len() as u64);
        let reader = edits.reader();
        for &position in &set {
            assert_eq!(reader.get(position), Some(voxel(position)), "{position}");
        }
        let unset = every_voxel_of(IVec3::ZERO).nth(DENSE_AT + 100).unwrap();
        assert_eq!(reader.get(unset), None);
    }

    #[test]
    fn a_region_equal_to_a_few_committed_edits_changes_only_the_others() {
        let edits = Edits::default();
        let (same, other) = (IVec3::new(1, 1, 1), IVec3::new(2, 2, 2));
        for position in [same, other] {
            edits.set(position, WorldVoxel::Solid(1));
        }
        edits.commit();

        // A region over the whole chunk, setting only those two voxels.
        edits.set_region(IVec3::ZERO, IVec3::splat(CHUNK_SIZE_I), |position| {
            (position == same || position == other)
                .then(|| WorldVoxel::Solid(if position == same { 1 } else { 2 }))
        });

        assert_eq!(changed_voxels(&edits.commit()), vec![other]);
        assert_eq!(edits.edit_sequence(IVec3::ZERO), 3);
    }

    #[test]
    fn a_few_voxels_equal_to_a_full_chunk_of_edits_change_only_the_others() {
        let edits = Edits::default();
        edits.set_region(IVec3::ZERO, IVec3::splat(CHUNK_SIZE_I), |_| {
            Some(WorldVoxel::Solid(1))
        });
        edits.commit();

        let other = IVec3::new(2, 2, 2);
        edits.set(IVec3::ONE, WorldVoxel::Solid(1));
        edits.set(other, WorldVoxel::Solid(2));

        assert_eq!(changed_voxels(&edits.commit()), vec![other]);
    }

    #[test]
    fn a_full_chunk_set_again_with_one_voxel_different_changes_that_voxel() {
        let edits = Edits::default();
        let full = IVec3::splat(CHUNK_SIZE_I);
        edits.set_region(IVec3::ZERO, full, |_| Some(WorldVoxel::Solid(1)));
        edits.commit();

        let other = IVec3::new(2, 2, 2);
        edits.set_region(IVec3::ZERO, full, |position| {
            Some(WorldVoxel::Solid(if position == other { 2 } else { 1 }))
        });

        assert_eq!(changed_voxels(&edits.commit()), vec![other]);
    }

    /// The stale boxes of a commit of the first `count` voxels of the slice
    /// z = 7 of the chunk at the origin, x varying fastest.
    fn boxes_for_voxels_in_a_row(count: i32) -> Vec<EditedBox> {
        let edits = Edits::default();
        for i in 0..count {
            edits.set(
                IVec3::new(i % CHUNK_SIZE_I, i / CHUNK_SIZE_I, 7),
                WorldVoxel::Solid(1),
            );
        }
        let sets = edits.commit();
        assert_eq!(sets.len(), 1);
        sets[0].stale_boxes().collect()
    }

    #[test]
    fn a_few_changed_voxels_are_reported_one_by_one() {
        let boxes = boxes_for_voxels_in_a_row(BOXED_AT as i32);
        assert_eq!(boxes.len(), BOXED_AT);
        assert!(boxes.iter().all(|voxels| voxels.min == voxels.max));
    }

    #[test]
    fn many_changed_voxels_are_reported_as_the_box_around_them() {
        assert_eq!(
            boxes_for_voxels_in_a_row(BOXED_AT as i32 + 1),
            vec![EditedBox {
                min: IVec3::new(0, 0, 7),
                max: IVec3::new(LAST, 2, 7)
            }]
        );
    }

    /// The level-0 chunks that an edit at `position` makes stale.
    fn stale_after_editing(position: IVec3) -> HashSet<ChunkKey> {
        chunks_including(position, 0, default_padding(0)).collect()
    }

    fn chunks(positions: &[[i32; 3]]) -> HashSet<ChunkKey> {
        positions
            .iter()
            .copied()
            .map(|position| ChunkKey::new(IVec3::from_array(position), 0))
            .collect()
    }

    fn level_1_chunks(positions: &[[i32; 3]]) -> HashSet<ChunkKey> {
        positions
            .iter()
            .copied()
            .map(|position| ChunkKey::new(IVec3::from_array(position), 1))
            .collect()
    }

    #[test]
    fn edits_are_read_back_before_and_after_they_are_committed() {
        let edits = Edits::default();
        let position = IVec3::new(3, -40, 70);
        edits.set(position, WorldVoxel::Solid(2));

        assert_eq!(edits.reader().get(position), Some(WorldVoxel::Solid(2)));
        edits.commit();
        let reader = edits.reader();
        assert_eq!(reader.get(position), Some(WorldVoxel::Solid(2)));
        assert_eq!(reader.get(position + IVec3::X), None);
    }

    #[test]
    fn the_last_voxel_set_is_read_and_committed() {
        let edits = Edits::default();
        edits.set(IVec3::ZERO, WorldVoxel::Solid(1));
        edits.set(IVec3::ZERO, WorldVoxel::Solid(2));

        assert_eq!(edits.reader().get(IVec3::ZERO), Some(WorldVoxel::Solid(2)));
        edits.commit();
        assert_eq!(edits.reader().get(IVec3::ZERO), Some(WorldVoxel::Solid(2)));
    }

    #[test]
    fn a_reader_keeps_the_pending_edits_it_was_made_with() {
        let edits = Edits::default();
        edits.set(IVec3::ZERO, WorldVoxel::Solid(1));
        let reader = edits.reader();
        edits.set(IVec3::ZERO, WorldVoxel::Solid(2));

        assert_eq!(reader.get(IVec3::ZERO), Some(WorldVoxel::Solid(1)));
    }

    #[test]
    fn a_reader_sees_later_edits_once_they_are_committed() {
        let edits = Edits::default();
        let reader = edits.reader();
        edits.set(IVec3::ZERO, WorldVoxel::Solid(1));

        assert_eq!(reader.get(IVec3::ZERO), None);
        edits.commit();
        assert_eq!(reader.get(IVec3::ZERO), Some(WorldVoxel::Solid(1)));
    }

    #[test]
    fn a_reader_made_during_a_commit_sees_each_edit_pending_or_committed() {
        const ROUNDS: u32 = 100_000;
        let edits = VoxelEdits::<(), u32>::default();
        // The last voxel read. The writer commits each edit only once a reader
        // has seen it pending, so a reader made during the commit would notice
        // if the edit went missing.
        let read = AtomicU32::new(0);
        let went_back = std::thread::scope(|scope| {
            scope.spawn(|| {
                for n in 1..=ROUNDS {
                    edits.set(IVec3::ZERO, WorldVoxel::Solid(n));
                    while read.load(Ordering::Acquire) < n {
                        std::thread::yield_now();
                    }
                    edits.commit();
                }
            });

            let mut last = 0;
            let mut went_back = None;
            while last < ROUNDS {
                let n = match edits.reader().get(IVec3::ZERO) {
                    Some(WorldVoxel::Solid(n)) => n,
                    _ => 0,
                };
                if n < last {
                    went_back = Some((last, n));
                    break;
                }
                last = n;
                read.store(n, Ordering::Release);
            }
            // Lets the writer finish if the voxel went back.
            read.store(u32::MAX, Ordering::Release);
            went_back
        });
        assert_eq!(
            went_back, None,
            "the voxel was read as one value, then as an older one"
        );
    }

    #[test]
    fn committing_the_same_voxel_again_changes_nothing() {
        let edits = Edits::default();
        let position = IVec3::new(5, 5, 5);

        assert_eq!(
            commit_one(&edits, position, WorldVoxel::Solid(1)),
            vec![position]
        );
        assert!(commit_one(&edits, position, WorldVoxel::Solid(1)).is_empty());
    }

    #[test]
    fn committing_a_different_voxel_changes_it_again() {
        let edits = Edits::default();
        let position = IVec3::new(5, 5, 5);
        commit_one(&edits, position, WorldVoxel::Solid(1));

        assert_eq!(
            commit_one(&edits, position, WorldVoxel::Solid(2)),
            vec![position]
        );
        assert_eq!(edits.reader().get(position), Some(WorldVoxel::Solid(2)));
    }

    #[test]
    fn setting_a_committed_voxel_and_back_changes_nothing() {
        let edits = Edits::default();
        let position = IVec3::new(5, 5, 5);
        commit_one(&edits, position, WorldVoxel::Solid(1));

        edits.set(position, WorldVoxel::Solid(2));
        edits.set(position, WorldVoxel::Solid(1));
        assert!(edits.commit().is_empty());
    }

    #[test]
    fn an_edit_inside_a_chunk_makes_only_that_chunk_stale() {
        assert_eq!(
            stale_after_editing(IVec3::new(1, 1, LAST - 1)),
            chunks(&[[0, 0, 0]])
        );
    }

    #[test]
    fn an_edit_on_a_face_makes_both_chunks_sharing_it_stale() {
        assert_eq!(
            stale_after_editing(IVec3::new(0, 5, 5)),
            chunks(&[[0, 0, 0], [-1, 0, 0]])
        );
        assert_eq!(
            stale_after_editing(IVec3::new(5, LAST, 5)),
            chunks(&[[0, 0, 0], [0, 1, 0]])
        );
    }

    #[test]
    fn an_edit_on_an_edge_makes_all_four_chunks_sharing_it_stale() {
        assert_eq!(
            stale_after_editing(IVec3::new(5, 0, LAST)),
            chunks(&[[0, 0, 0], [0, -1, 0], [0, 0, 1], [0, -1, 1]])
        );
    }

    #[test]
    fn an_edit_on_a_corner_makes_all_eight_chunks_sharing_it_stale() {
        assert_eq!(
            stale_after_editing(IVec3::ZERO),
            chunks(&[
                [0, 0, 0],
                [-1, 0, 0],
                [0, -1, 0],
                [0, 0, -1],
                [-1, -1, 0],
                [-1, 0, -1],
                [0, -1, -1],
                [-1, -1, -1],
            ])
        );
        assert_eq!(
            stale_after_editing(IVec3::splat(LAST)),
            chunks(&[
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 1, 0],
                [1, 0, 1],
                [0, 1, 1],
                [1, 1, 1],
            ])
        );
    }

    #[test]
    fn an_edit_at_negative_coordinates_belongs_to_the_chunk_below_zero() {
        // Voxel -1 is the last voxel of chunk -1, next to chunk 0.
        assert_eq!(
            stale_after_editing(IVec3::new(-1, 5, 5)),
            chunks(&[[-1, 0, 0], [0, 0, 0]])
        );
        // Voxel -CHUNK_SIZE is the first voxel of chunk -1, next to chunk -2.
        assert_eq!(
            stale_after_editing(IVec3::new(-CHUNK_SIZE_I, 5, 5)),
            chunks(&[[-1, 0, 0], [-2, 0, 0]])
        );
    }

    #[test]
    fn an_edit_within_a_data_voxel_of_a_coarse_chunks_face_makes_its_neighbour_stale() {
        // Level-1 chunks are 64 voxels wide and their padding two voxels deep.
        let stale = |x: i32| -> HashSet<ChunkKey> {
            chunks_including(IVec3::new(x, 5, 5), 1, default_padding(1)).collect()
        };
        assert_eq!(stale(2), level_1_chunks(&[[0, 0, 0]]));
        assert_eq!(stale(1), level_1_chunks(&[[0, 0, 0], [-2, 0, 0]]));
        assert_eq!(stale(61), level_1_chunks(&[[0, 0, 0]]));
        assert_eq!(stale(62), level_1_chunks(&[[0, 0, 0], [2, 0, 0]]));
        assert_eq!(stale(65), level_1_chunks(&[[0, 0, 0], [2, 0, 0]]));
        assert_eq!(stale(66), level_1_chunks(&[[2, 0, 0]]));
    }

    #[test]
    fn a_chunk_task_reads_the_edits_its_padding_reaches() {
        let edits = Edits::default();
        let position = IVec3::new(-1, 5, 5);
        commit_one(&edits, position, WorldVoxel::Solid(1));

        for chunk_pos in [IVec3::new(-1, 0, 0), IVec3::ZERO] {
            let chunk_edits =
                edits.chunk_edits(ChunkKey::new(chunk_pos, 0), default_padding(0));
            assert!(!chunk_edits.is_empty());
            assert_eq!(chunk_edits.get(position), Some(WorldVoxel::Solid(1)));
        }
        assert!(edits
            .chunk_edits(ChunkKey::new(IVec3::new(1, 0, 0), 0), default_padding(0))
            .is_empty());
    }

    #[test]
    fn a_coarse_chunk_task_reads_the_edits_in_its_extent_and_padding() {
        let edits = Edits::default();
        let inside = IVec3::new(40, 5, 5);
        let in_padding = IVec3::new(-1, 5, 5);
        let beyond = IVec3::new(200, 5, 5);
        for position in [inside, in_padding, beyond] {
            commit_one(&edits, position, WorldVoxel::Solid(1));
        }

        // Level-0 chunks 0 and 1, with chunks -1 and 2 as padding.
        let chunk_edits =
            edits.chunk_edits(ChunkKey::new(IVec3::ZERO, 1), default_padding(1));
        assert!(!chunk_edits.is_empty());
        assert_eq!(chunk_edits.get(inside), Some(WorldVoxel::Solid(1)));
        assert_eq!(chunk_edits.get(in_padding), Some(WorldVoxel::Solid(1)));
        assert_eq!(chunk_edits.get(beyond), None);

        let farther =
            edits.chunk_edits(ChunkKey::new(IVec3::new(4, 0, 0), 1), default_padding(1));
        assert_eq!(farther.get(beyond), Some(WorldVoxel::Solid(1)));
        assert_eq!(farther.get(inside), None);
        assert!(edits
            .chunk_edits(ChunkKey::new(IVec3::new(8, 0, 0), 1), default_padding(1))
            .is_empty());
    }

    #[test]
    fn an_edit_reaches_the_chunks_whose_tasks_read_it() {
        let edits = Edits::default();
        assert!(!edits
            .committed()
            .reach(ChunkKey::new(IVec3::ZERO, 0), default_padding(0)));

        // In chunk -1, one voxel from chunk 0.
        commit_one(&edits, IVec3::new(-1, 5, 5), WorldVoxel::Solid(1));
        let committed = edits.committed();

        for chunk_pos in [IVec3::new(-1, 0, 0), IVec3::ZERO] {
            assert!(
                committed.reach(ChunkKey::new(chunk_pos, 0), default_padding(0)),
                "{chunk_pos}"
            );
        }
        assert!(
            !committed.reach(ChunkKey::new(IVec3::new(1, 0, 0), 0), default_padding(0))
        );
        // Level-0 chunks 0 and 1, with chunks -1 and 2 as padding.
        assert!(committed.reach(ChunkKey::new(IVec3::ZERO, 1), default_padding(1)));
        assert!(
            committed.reach(ChunkKey::new(IVec3::new(-2, 0, 0), 1), default_padding(1))
        );
        assert!(
            !committed.reach(ChunkKey::new(IVec3::new(2, 0, 0), 1), default_padding(1))
        );
    }

    #[test]
    fn a_chunk_task_keeps_the_edits_it_started_with() {
        let edits = Edits::default();
        let position = IVec3::new(5, 5, 5);
        commit_one(&edits, position, WorldVoxel::Solid(1));
        let started_with =
            edits.chunk_edits(ChunkKey::new(IVec3::ZERO, 0), default_padding(0));

        commit_one(&edits, position, WorldVoxel::Solid(2));

        assert_eq!(started_with.get(position), Some(WorldVoxel::Solid(1)));
        assert_eq!(
            edits
                .chunk_edits(ChunkKey::new(IVec3::ZERO, 0), default_padding(0))
                .get(position),
            Some(WorldVoxel::Solid(2))
        );
    }

    #[test]
    fn the_padding_is_as_deep_as_the_data_voxels_are_apart_along_each_axis() {
        assert_eq!(default_padding(0), IVec3::ONE);
        assert_eq!(default_padding(3), IVec3::splat(8));
        // Denser than one data voxel per voxel still reaches one voxel.
        assert_eq!(padding_depth(UVec3::splat(66), 0), IVec3::ONE);
        assert_eq!(
            padding_depth(UVec3::new(18, 34, 10), 0),
            IVec3::new(2, 1, 4)
        );
        // One data voxel per chunk reaches across the whole chunk next to it.
        assert_eq!(padding_depth(UVec3::splat(3), 1), IVec3::splat(64));
    }

    #[test]
    fn an_edit_within_a_deeper_padding_makes_the_neighbour_stale_along_that_axis_only() {
        let padding = padding_depth(UVec3::new(18, 34, 34), 0);
        let stale = |position: IVec3| -> HashSet<ChunkKey> {
            chunks_including(position, 0, padding).collect()
        };

        assert_eq!(
            stale(IVec3::new(-2, 5, 5)),
            chunks(&[[-1, 0, 0], [0, 0, 0]])
        );
        assert_eq!(stale(IVec3::new(-3, 5, 5)), chunks(&[[-1, 0, 0]]));
        assert_eq!(stale(IVec3::new(1, 5, 5)), chunks(&[[0, 0, 0], [-1, 0, 0]]));
        assert_eq!(stale(IVec3::new(2, 5, 5)), chunks(&[[0, 0, 0]]));
        // Along y the data voxels are still one voxel apart.
        assert_eq!(stale(IVec3::new(5, -2, 5)), chunks(&[[0, -1, 0]]));
    }

    #[test]
    fn a_coarse_chunk_with_sparse_data_reads_edits_past_the_next_level_0_chunk() {
        let edits = Edits::default();
        // Level 2 is 128 voxels wide. With two data voxels inside, they are 64
        // voxels apart, so the padding sits two level-0 chunks past the minimum
        // corner.
        let padding = padding_depth(UVec3::splat(4), 2);
        assert_eq!(padding, IVec3::splat(64));
        let in_padding = IVec3::new(-64, 0, 0);
        commit_one(&edits, in_padding, WorldVoxel::Solid(1));

        let key = ChunkKey::new(IVec3::ZERO, 2);
        assert_eq!(
            edits.chunk_edits(key, padding).get(in_padding),
            Some(WorldVoxel::Solid(1))
        );
        assert!(edits.committed().reach(key, padding));
        assert!(chunks_including(in_padding, 2, padding).any(|stale| stale == key));
        // With the default data shape the padding doesn't reach that far.
        assert!(!edits.committed().reach(key, default_padding(2)));
    }
}
