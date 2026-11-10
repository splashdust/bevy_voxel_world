//! Every voxel edit of a voxel world, kept until it is cleared or forgotten,
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
//! voxel rather than a hash map insert. A commit that leaves an array few
//! edits turns it back into a map.
//!
//! A commit reports a [`ChangeSet`] per level-0 chunk whose edits changed: the
//! voxels whose committed edit changed, with their new edits. The chunk
//! lifecycle marks the chunks they reach stale through
//! [`ChangeSet::stale_boxes`], light reads the voxels themselves, and each
//! set becomes the chunk's [`EditDelta`](crate::prelude::EditDelta) in the
//! commit's message.
//!
//! Edit deltas applied from another world join the pending edits in the
//! order they arrive among the voxels set, and assign the chunk's edit
//! sequence rather than count towards it. A full delta marks the chunk's
//! pending bucket replaced: the commit then drops the chunk's committed edits
//! that the bucket lacks, and reports them as removed.
//!
//! A clear takes the same path. It marks the pending bucket of each chunk it
//! reaches replaced, filled with the chunk's committed edits under those set
//! earlier in the frame, less the edits in its box. Forgetting is immediate
//! instead: it drops every bucket with its edit sequence, and hands the
//! removed edits to the next commit apart from its change sets, since no
//! other world hears of them.

use std::{
    marker::PhantomData,
    sync::{Arc, Mutex, RwLock, RwLockReadGuard},
};

#[cfg(feature = "light_propagation")]
use std::sync::atomic::{AtomicU64, Ordering};

use bevy::{platform::collections::HashMap, prelude::*};

use crate::{
    chunk::{chunk_containing, ChunkKey, CHUNK_SIZE_I},
    configuration::LodLevel,
    edits::{voxel_index, voxel_local, EditDelta, EditGap, CHUNK_VOXELS},
    voxel::WorldVoxel,
};

type Voxels<I> = HashMap<IVec3, WorldVoxel<I>>;

/// A bucket holding this many edits turns from a map into an array.
///
/// The array takes the same memory however few edits it holds, and a commit
/// compares every one of its slots; the map takes memory per edit and hashes
/// each read and write. This is about where the two take the same memory, so
/// a chunk with fewer edits stays a map and one with more is as small as an
/// array.
const DENSE_AT: usize = 2048;

/// An array bucket that a commit leaves holding fewer edits than this turns
/// back into a map.
///
/// Only a commit that replaces a chunk's edits, a clear or an applied full
/// delta, can leave fewer. Half of [`DENSE_AT`] keeps a chunk whose edits
/// hover around that count from turning back and forth every commit.
const SPARSE_BELOW: usize = DENSE_AT / 2;

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
    /// Whether the commit replaces the chunk's committed edits with this
    /// bucket's, rather than setting this bucket's over them. Only pending
    /// buckets keep it.
    replaced: bool,
    /// The edit sequence that the last edit delta applied since the last
    /// commit assigned. Only pending buckets keep it.
    assigned: Option<Assigned>,
}

/// The edit sequence an applied edit delta assigned to a pending bucket.
#[derive(Clone, Copy)]
struct Assigned {
    /// The delta's edit sequence.
    seq: u64,
    /// How many voxels local sets have changed in the bucket since.
    sets_after: u64,
}

#[derive(Clone)]
enum Store<I> {
    Sparse(Voxels<I>),
    Dense {
        /// `CHUNK_VOXELS` slots, by [`voxel_index`].
        voxels: Box<[Option<WorldVoxel<I>>]>,
        /// How many slots are set.
        count: usize,
    },
}

impl<I> Bucket<I> {
    /// Whether the bucket holds no edit, as one left by a clear does.
    #[inline]
    fn is_empty(&self) -> bool {
        match &self.store {
            Store::Sparse(voxels) => voxels.is_empty(),
            Store::Dense { count, .. } => *count == 0,
        }
    }
}

impl<I: Copy + PartialEq> Bucket<I> {
    fn new(chunk_pos: IVec3) -> Self {
        Self {
            min: chunk_pos * CHUNK_SIZE_I,
            store: Store::Sparse(Voxels::default()),
            seq: 0,
            replaced: false,
            assigned: None,
        }
    }

    #[inline]
    fn index(&self, position: IVec3) -> usize {
        voxel_index(position - self.min)
    }

    fn position_of(&self, index: usize) -> IVec3 {
        self.min + voxel_local(index)
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

    /// Sets the voxel at `position` from a local set rather than an applied
    /// delta.
    #[inline]
    fn set(&mut self, position: IVec3, voxel: WorldVoxel<I>) {
        let changed = self.insert(position, voxel);
        self.count_sets(changed as u64);
    }

    /// Counts `changed` voxels changed by local sets towards the edit
    /// sequence the bucket's commit reaches.
    #[inline]
    fn count_sets(&mut self, changed: u64) {
        if let Some(assigned) = &mut self.assigned {
            assigned.sets_after += changed;
        }
    }

    /// Removes every edit in the bucket.
    fn clear(&mut self) {
        match &mut self.store {
            Store::Sparse(voxels) => voxels.clear(),
            Store::Dense { voxels, count } => {
                voxels.fill(None);
                *count = 0;
            }
        }
    }

    /// Whether the bucket holds an edit in the box from `min` to `max`, both
    /// inclusive and within the bucket's chunk.
    fn any_within(&self, min: IVec3, max: IVec3) -> bool {
        match &self.store {
            Store::Sparse(voxels) => voxels
                .keys()
                .any(|position| position.cmpge(min).all() && position.cmple(max).all()),
            Store::Dense { count: 0, .. } => false,
            Store::Dense { .. } => {
                box_positions(min, max).any(|position| self.get(position).is_some())
            }
        }
    }

    /// Removes every edit in the box from `min` to `max`, both inclusive and
    /// within the bucket's chunk. Returns how many it removed.
    fn remove_within(&mut self, min: IVec3, max: IVec3) -> u64 {
        let chunk_max = self.min + CHUNK_SIZE_I - 1;
        if min == self.min && max == chunk_max {
            let count = match &self.store {
                Store::Sparse(voxels) => voxels.len(),
                Store::Dense { count, .. } => *count,
            };
            self.clear();
            return count as u64;
        }
        let bucket_min = self.min;
        match &mut self.store {
            Store::Sparse(voxels) => {
                let before = voxels.len();
                voxels.retain(|position, _| {
                    !(position.cmpge(min).all() && position.cmple(max).all())
                });
                (before - voxels.len()) as u64
            }
            Store::Dense { voxels, count } => {
                let mut removed = 0;
                for position in box_positions(min, max) {
                    if voxels[voxel_index(position - bucket_min)].take().is_some() {
                        removed += 1;
                    }
                }
                *count -= removed;
                removed as u64
            }
        }
    }

    /// Clears the edits in the box from `min` to `max`, both inclusive and
    /// within the bucket's chunk, from this pending bucket. `committed` is the
    /// chunk's committed bucket, if it has one. Called only where one of the
    /// two holds an edit in the box.
    ///
    /// The bucket becomes replaced, holding the committed edits under those
    /// set earlier this frame, so the commit drops the edits in the box from
    /// both. Voxels set after the clear go over it as usual.
    fn clear_within(&mut self, committed: Option<&Bucket<I>>, min: IVec3, max: IVec3) {
        if !self.replaced {
            let committed_store = committed.map_or_else(
                || Store::Sparse(Voxels::default()),
                |committed| committed.store.clone(),
            );
            let earlier = std::mem::replace(&mut self.store, committed_store);
            self.replaced = true;
            let earlier = Bucket {
                store: earlier,
                ..Bucket::new(chunk_containing(self.min))
            };
            for (position, voxel) in earlier.iter() {
                self.insert(position, voxel);
            }
        }
        let removed = self.remove_within(min, max);
        self.count_sets(removed);
    }

    /// Turns a sparse bucket dense, for a region write that will fill it.
    pub fn make_dense(&mut self) {
        if let Store::Sparse(sparse) = &self.store {
            let mut voxels = vec![None; CHUNK_VOXELS].into_boxed_slice();
            let min = self.min;
            for (&position, &voxel) in sparse {
                voxels[voxel_index(position - min)] = Some(voxel);
            }
            let count = sparse.len();
            self.store = Store::Dense { voxels, count };
        }
    }

    /// The bucket's edits, as the map they fit in once fewer than
    /// [`SPARSE_BELOW`], else as they are.
    fn compacted_store(&self) -> Store<I> {
        match &self.store {
            Store::Dense { count, .. } if *count < SPARSE_BELOW => {
                Store::Sparse(self.iter().collect())
            }
            store => store.clone(),
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

    /// The edits that change when `pending` replaces this bucket's edits:
    /// each edit of this bucket that `pending` lacks, as removed, then each
    /// edit of `pending` that differs from this bucket's.
    fn replaced_by(&self, pending: &Bucket<I>) -> Vec<(IVec3, Option<WorldVoxel<I>>)> {
        let mut changes: Vec<_> = self
            .iter()
            .filter(|&(position, _)| pending.get(position).is_none())
            .map(|(position, _)| (position, None))
            .collect();
        changes.extend(self.changes_from(pending));
        changes
    }
}

/// Every position in the box from `min` to `max`, both inclusive.
fn box_positions(min: IVec3, max: IVec3) -> impl Iterator<Item = IVec3> {
    (min.z..=max.z).flat_map(move |z| {
        (min.y..=max.y)
            .flat_map(move |y| (min.x..=max.x).map(move |x| IVec3::new(x, y, z)))
    })
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
    /// Whether the edit sequence before the commit can't be worked out from
    /// the one after it and the voxels' count. That is so where the commit
    /// replaced the chunk's edits rather than changing some of them, and
    /// where an applied edit delta's sequence doesn't match the voxels that
    /// changed: voxels set before the delta committed with it, or the delta
    /// set voxels to the edits already committed. The chunk's message then
    /// carries its full delta.
    pub replaced: bool,
    /// The chunk's edit sequence after the commit.
    pub seq: u64,
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

    /// The box around every voxel whose edit was removed, both corners
    /// inclusive, or `None` if no edit was.
    pub fn removed_box(&self) -> Option<(IVec3, IVec3)> {
        self.voxels
            .iter()
            .filter(|(_, voxel)| voxel.is_none())
            .map(|(position, _)| (*position, *position))
            .reduce(|(min, max), (p, _)| (min.min(p), max.max(p)))
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
    /// A change set for each chunk whose edits were forgotten since the last
    /// commit, every edit it held removed.
    forgotten: Mutex<Vec<ChangeSet<I>>>,
    /// How far every [`edit_stamp`](Self::edit_stamp) runs ahead of its
    /// chunk's edit sequence. Changed only under the committed edits' write
    /// lock, and read under their read lock.
    #[cfg(feature = "light_propagation")]
    forget_epoch: AtomicU64,
    _marker: PhantomData<C>,
}

impl<C, I> Default for VoxelEdits<C, I> {
    fn default() -> Self {
        Self {
            pending: Mutex::default(),
            committed: Arc::default(),
            forgotten: Mutex::default(),
            #[cfg(feature = "light_propagation")]
            forget_epoch: AtomicU64::default(),
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
        bucket_mut(Arc::make_mut(&mut pending), chunk).set(position, voxel);
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
        if max.cmple(min).any() {
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
                    let mut changed = 0;
                    for z in lo.z..hi.z {
                        for y in lo.y..hi.y {
                            for x in lo.x..hi.x {
                                let position = IVec3::new(x, y, z);
                                if let Some(v) = voxel(position) {
                                    changed += bucket.insert(position, v) as u64;
                                }
                            }
                        }
                    }
                    bucket.count_sets(changed);
                }
            }
        }
    }

    /// Clears the edits in the box from `min` inclusive to `max` exclusive:
    /// the next commit removes the committed ones and those set earlier this
    /// frame. A box that is empty or inverted along any axis clears nothing.
    pub(crate) fn clear_region(&self, min: IVec3, max: IVec3) {
        if max.cmple(min).any() {
            return;
        }
        let max = max - 1;
        let (chunk_min, chunk_max) = (chunk_containing(min), chunk_containing(max));
        self.clear_chunks(
            |chunk| chunk.cmpge(chunk_min).all() && chunk.cmple(chunk_max).all(),
            |chunk| {
                let chunk_min = chunk * CHUNK_SIZE_I;
                (chunk_min.max(min), (chunk_min + CHUNK_SIZE_I - 1).min(max))
            },
        );
    }

    /// Clears every edit: the next commit removes them all.
    pub(crate) fn clear_all(&self) {
        self.clear_chunks(
            |_| true,
            |chunk| {
                let chunk_min = chunk * CHUNK_SIZE_I;
                (chunk_min, chunk_min + CHUNK_SIZE_I - 1)
            },
        );
    }

    /// Clears the edits of each chunk holding pending or committed edits
    /// that `reached` accepts, within the box `within` gives for it, both
    /// corners inclusive.
    fn clear_chunks(
        &self,
        reached: impl Fn(IVec3) -> bool,
        within: impl Fn(IVec3) -> (IVec3, IVec3),
    ) {
        let mut pending = self.pending.lock().unwrap();
        let committed = self.committed.read().unwrap();
        let holds = |bucket: Option<&EditBucket<I>>, (min, max)| {
            bucket.is_some_and(|bucket| bucket.any_within(min, max))
        };
        let mut chunks: Vec<IVec3> = committed
            .keys()
            .chain(pending.keys())
            .copied()
            .filter(|chunk| reached(*chunk))
            .filter(|chunk| {
                holds(committed.get(chunk), within(*chunk))
                    || holds(pending.get(chunk), within(*chunk))
            })
            .collect();
        if chunks.is_empty() {
            return;
        }
        chunks.sort_unstable_by_key(|chunk| chunk.to_array());
        chunks.dedup();
        let map = Arc::make_mut(&mut pending);
        for chunk in chunks {
            let (min, max) = within(chunk);
            bucket_mut(map, chunk).clear_within(
                committed.get(&chunk).map(|b| &**b),
                min,
                max,
            );
        }
    }

    /// Drops every edit and every edit sequence at once, pending edits
    /// included. The next commit hands the removed edits to
    /// [`take_forgotten`](Self::take_forgotten) rather than reporting them.
    pub(crate) fn forget(&self) {
        let mut pending = self.pending.lock().unwrap();
        match Arc::get_mut(&mut pending) {
            Some(map) => map.clear(),
            None => *pending = Arc::default(),
        }
        let mut buckets = self.committed.write().unwrap();
        if buckets.is_empty() {
            return;
        }
        let mut forgotten = self.forgotten.lock().unwrap();
        // The new epoch is past every stamp handed out so far: the old epoch
        // plus the largest sequence any chunk reached since.
        #[cfg(feature = "light_propagation")]
        let highest = buckets.values().map(|bucket| bucket.seq).max().unwrap_or(0);
        #[cfg(feature = "light_propagation")]
        self.forget_epoch.fetch_add(highest + 1, Ordering::Relaxed);
        for (chunk, bucket) in buckets.drain() {
            forgotten.push(ChangeSet {
                chunk,
                replaced: true,
                seq: 0,
                voxels: bucket
                    .iter()
                    .map(|(position, _)| (position, None))
                    .collect(),
            });
        }
    }

    /// A change set for each chunk whose edits were forgotten since this was
    /// last called, with every edit it held removed.
    pub(crate) fn take_forgotten(&self) -> Vec<ChangeSet<I>> {
        std::mem::take(&mut self.forgotten.lock().unwrap())
    }

    /// Applies `delta` if it continues from the edit sequence its chunk
    /// reaches this frame: the sequence the last delta applied to the chunk
    /// since the last commit assigned, else the committed one. See
    /// [`VoxelWorld::apply_delta`](crate::prelude::VoxelWorld::apply_delta).
    ///
    /// The delta's voxels join the chunk's pending edits after any set before
    /// them. A full delta first drops the chunk's pending edits and has the
    /// commit replace its committed edits.
    pub(crate) fn apply(&self, delta: EditDelta<I>) -> Result<(), EditGap> {
        let mut pending = self.pending.lock().unwrap();
        let local = match pending.get(&delta.chunk).and_then(|b| b.assigned) {
            Some(assigned) => assigned.seq,
            None => self.edit_sequence(delta.chunk),
        };
        if !delta.applies_at(local)? {
            return Ok(());
        }
        let bucket = bucket_mut(Arc::make_mut(&mut pending), delta.chunk);
        if delta.full {
            bucket.clear();
            bucket.replaced = true;
        }
        if delta.len() >= DENSE_AT {
            bucket.make_dense();
        }
        for (position, voxel) in delta.iter() {
            bucket.insert(position, voxel);
        }
        bucket.assigned = Some(Assigned {
            seq: delta.seq,
            sets_after: 0,
        });
        Ok(())
    }

    /// The full delta of the level-0 chunk at `chunk`, from its committed
    /// edits: an empty one at sequence 0 for a chunk with no bucket.
    pub(crate) fn full_delta(&self, chunk: IVec3) -> EditDelta<I> {
        let buckets = self.committed.read().unwrap();
        let Some(bucket) = buckets.get(&chunk) else {
            return EditDelta::new(chunk, 0, true, Vec::new());
        };
        let voxels = bucket
            .iter()
            .map(|(position, voxel)| (bucket.index(position) as u16, voxel))
            .collect();
        EditDelta::new(chunk, bucket.seq, true, voxels)
    }

    /// Every level-0 chunk position holding committed edits, or an empty
    /// bucket left by removing them, with its edit sequence.
    pub(crate) fn edited_chunks(&self) -> Vec<(IVec3, u64)> {
        let buckets = self.committed.read().unwrap();
        buckets
            .iter()
            .map(|(&chunk, bucket)| (chunk, bucket.seq))
            .collect()
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
    /// whose committed edits or edit sequence changed.
    ///
    /// A chunk's edit sequence moves on by the number of voxels in its change
    /// set. Where an edit delta was applied, it takes the delta's sequence
    /// instead, plus the voxels local sets changed after it.
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
            let empty;
            let committed = match buckets.get(&chunk) {
                Some(committed) => committed,
                None => {
                    empty = Bucket::new(chunk);
                    &empty
                }
            };
            let changes = if pending_bucket.replaced {
                committed.replaced_by(pending_bucket)
            } else {
                committed.changes_from(pending_bucket)
            };
            let before = committed.seq;
            let counted = before + changes.len() as u64;
            let seq = pending_bucket
                .assigned
                .map_or(counted, |assigned| assigned.seq + assigned.sets_after);
            if changes.is_empty() && seq == before {
                continue;
            }
            let committed = Arc::make_mut(
                buckets
                    .entry(chunk)
                    .or_insert_with(|| Arc::new(Bucket::new(chunk))),
            );
            if pending_bucket.replaced {
                committed.store = pending_bucket.compacted_store();
            } else {
                if changes.len() >= DENSE_AT {
                    committed.make_dense();
                }
                for &(position, voxel) in &changes {
                    if let Some(voxel) = voxel {
                        committed.insert(position, voxel);
                    }
                }
            }
            committed.seq = seq;
            sets.push(ChangeSet {
                chunk,
                // Local sets before an applied delta count towards its
                // voxels but not its sequence, and a delta's voxels equal to
                // the committed edits count towards its sequence but not its
                // voxels. Either way the sequence before can't be worked out
                // from the voxels.
                replaced: pending_bucket.replaced || seq != counted,
                seq,
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
    /// A bucket left empty by a clear is left out, so a chunk whose reach
    /// holds no edit reads none.
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
                    *bucket = buckets
                        .get(&(chunk_pos + around_offset(i)))
                        .filter(|bucket| !bucket.is_empty())
                        .cloned();
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
            .filter(|(position, bucket)| {
                position.cmpge(min).all()
                    && position.cmplt(max).all()
                    && !bucket.is_empty()
            })
            .map(|(position, bucket)| (*position, bucket.clone()))
            .collect();
        ChunkEdits {
            chunk_pos,
            buckets: Buckets::Within(within),
        }
    }

    /// The committed edits in each level-0 chunk at `positions`, in order, or
    /// `None` for a chunk with none, a bucket left empty by a clear included.
    #[cfg(feature = "light_propagation")]
    pub(crate) fn buckets(
        &self,
        positions: impl Iterator<Item = IVec3>,
    ) -> Vec<Option<EditBucket<I>>> {
        let buckets = self.committed.read().unwrap();
        positions
            .map(|position| {
                buckets
                    .get(&position)
                    .filter(|bucket| !bucket.is_empty())
                    .cloned()
            })
            .collect()
    }

    /// The edit sequence of the level-0 chunk at `position`: 0 for a chunk
    /// with no edits.
    pub(crate) fn edit_sequence(&self, position: IVec3) -> u64 {
        let buckets = self.committed.read().unwrap();
        if buckets.is_empty() {
            return 0;
        }
        buckets.get(&position).map_or(0, |bucket| bucket.seq)
    }

    /// A stamp of the voxel edits of the level-0 chunk at `position`. Work
    /// that reads the chunk's voxels notes it when it starts, and its result
    /// is stale if the stamp has moved since. A later stamp is larger.
    ///
    /// It is the chunk's edit sequence plus the forget epoch, which every
    /// forget raises past each stamp handed out before it. Forgetting sets
    /// the sequence back to 0, and the epoch keeps the stamp from ever going
    /// back to a value it had, when the chunk's edits may have been
    /// different. A forget moves the stamp of every position, edited or not.
    #[cfg(feature = "light_propagation")]
    pub(crate) fn edit_stamp(&self, position: IVec3) -> u64 {
        let buckets = self.committed.read().unwrap();
        let epoch = self.forget_epoch.load(Ordering::Relaxed);
        if buckets.is_empty() {
            return epoch;
        }
        epoch + buckets.get(&position).map_or(0, |bucket| bucket.seq)
    }

    /// Whether `stamp`, an [`edit_stamp`](Self::edit_stamp) read since the
    /// last forget, is of a chunk whose edit sequence is above 0: one that
    /// holds edits, or did since the last forget.
    #[cfg(feature = "light_propagation")]
    pub(crate) fn stamp_has_edits(&self, stamp: u64) -> bool {
        stamp > self.forget_epoch.load(Ordering::Relaxed)
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
    /// can't be trusted. A bucket left empty by a clear holds no edit.
    /// `padding` is the chunk's [`padding_depth`].
    ///
    /// [`padding_depth`]: crate::chunk::padding_depth
    pub fn reach(&self, key: ChunkKey, padding: IVec3) -> bool {
        let buckets = &self.0;
        if buckets.is_empty() {
            return false;
        }
        if key.lod_level == 0 {
            return (0..27).any(|i| {
                buckets
                    .get(&(key.position + around_offset(i)))
                    .is_some_and(|bucket| !bucket.is_empty())
            });
        }
        let (min, max) = buckets_read_by(key, padding);
        buckets.iter().any(|(position, bucket)| {
            position.cmpge(min).all() && position.cmplt(max).all() && !bucket.is_empty()
        })
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
        if let Some(bucket) = self.pending.get(&chunk) {
            if let Some(voxel) = bucket.get(position) {
                return Some(voxel);
            }
            // The commit replaces the committed edits with this bucket's.
            if bucket.replaced {
                return None;
            }
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

    /// An edit delta of the chunk at the origin, setting the voxels at
    /// `positions` to `Solid(1)`.
    fn delta(seq: u64, full: bool, positions: &[IVec3]) -> EditDelta<u8> {
        let voxels = positions
            .iter()
            .map(|&position| (voxel_index(position) as u16, WorldVoxel::Solid(1)))
            .collect();
        EditDelta::new(IVec3::ZERO, seq, full, voxels)
    }

    #[test]
    fn a_full_delta_commits_as_replaced_with_the_edits_it_removed() {
        let edits = Edits::default();
        let (removed, kept, added) = (IVec3::X, IVec3::Y, IVec3::Z);
        edits.set(removed, WorldVoxel::Solid(2));
        edits.set(kept, WorldVoxel::Solid(1));
        edits.commit();

        assert_eq!(edits.apply(delta(10, true, &[kept, added])), Ok(()));
        let sets = edits.commit();

        assert_eq!(
            sets,
            vec![ChangeSet {
                chunk: IVec3::ZERO,
                replaced: true,
                seq: 10,
                voxels: vec![(removed, None), (added, Some(WorldVoxel::Solid(1)))],
            }]
        );
        assert_eq!(sets[0].removed_box(), Some((removed, removed)));
        assert_eq!(edits.reader().get(removed), None);
    }

    #[test]
    fn a_clear_commits_as_replaced_with_the_edits_it_removed() {
        let edits = Edits::default();
        let (inside, outside) = (IVec3::new(1, 1, 1), IVec3::new(9, 1, 1));
        edits.set(inside, WorldVoxel::Solid(1));
        edits.set(outside, WorldVoxel::Solid(1));
        edits.commit();

        edits.clear_region(IVec3::ZERO, IVec3::splat(4));
        // The cleared edit is gone from the reader before the commit too.
        assert_eq!(edits.reader().get(inside), None);
        let sets = edits.commit();

        assert_eq!(
            sets,
            vec![ChangeSet {
                chunk: IVec3::ZERO,
                replaced: true,
                seq: 3,
                voxels: vec![(inside, None)],
            }]
        );
        assert_eq!(edits.reader().get(outside), Some(WorldVoxel::Solid(1)));
    }

    #[test]
    fn a_clear_in_a_dense_bucket_removes_only_the_edits_in_its_box() {
        let edits = Edits::default();
        let full = IVec3::splat(CHUNK_SIZE_I);
        edits.set_region(IVec3::ZERO, full, |_| Some(WorldVoxel::Solid(1)));
        edits.commit();

        edits.clear_region(IVec3::new(0, 0, 0), IVec3::new(2, 1, 1));
        let sets = edits.commit();

        assert_eq!(changed_voxels(&sets), vec![IVec3::ZERO, IVec3::X]);
        assert_eq!(edits.edit_sequence(IVec3::ZERO), CHUNK_VOXELS as u64 + 2);
        assert_eq!(edits.full_delta(IVec3::ZERO).len(), CHUNK_VOXELS - 2);
    }

    /// Whether the committed bucket of the chunk at `chunk` is an array.
    fn is_dense(edits: &Edits, chunk: IVec3) -> bool {
        matches!(
            edits.committed.read().unwrap()[&chunk].store,
            Store::Dense { .. }
        )
    }

    #[test]
    fn a_dense_bucket_cleared_of_most_of_its_edits_turns_back_into_a_map() {
        let edits = Edits::default();
        let full = IVec3::splat(CHUNK_SIZE_I);
        edits.set_region(IVec3::ZERO, full, |_| Some(WorldVoxel::Solid(1)));
        edits.commit();
        assert!(is_dense(&edits, IVec3::ZERO));

        // Clearing a few edits keeps the array.
        edits.clear_region(IVec3::ZERO, IVec3::new(2, 1, 1));
        edits.commit();
        assert!(is_dense(&edits, IVec3::ZERO));

        // All but the bottom slab of 32 x 32 voxels.
        edits.clear_region(IVec3::new(0, 1, 0), full);
        edits.commit();
        assert!(!is_dense(&edits, IVec3::ZERO));
        assert_eq!(edits.full_delta(IVec3::ZERO).len(), 32 * 32 - 2);
        assert_eq!(
            edits.reader().get(IVec3::new(5, 0, 5)),
            Some(WorldVoxel::Solid(1))
        );

        edits.clear_all();
        edits.commit();
        assert!(!is_dense(&edits, IVec3::ZERO));
        assert_eq!(edits.full_delta(IVec3::ZERO).len(), 0);
    }

    #[test]
    fn forgetting_hands_on_every_edit_removed_and_reports_none() {
        let edits = Edits::default();
        edits.set(IVec3::ONE, WorldVoxel::Solid(1));
        edits.commit();
        edits.set(IVec3::splat(2), WorldVoxel::Solid(1));

        edits.forget();

        assert!(edits.commit().is_empty());
        assert_eq!(edits.edit_sequence(IVec3::ZERO), 0);
        assert_eq!(edits.reader().get(IVec3::splat(2)), None);
        assert_eq!(
            edits.take_forgotten(),
            vec![ChangeSet {
                chunk: IVec3::ZERO,
                replaced: true,
                seq: 0,
                voxels: vec![(IVec3::ONE, None)],
            }]
        );
        assert!(edits.take_forgotten().is_empty());
    }

    #[cfg(feature = "light_propagation")]
    #[test]
    fn a_forgotten_chunks_edit_stamp_never_returns_to_a_value_it_had() {
        let edits = Edits::default();
        edits.set(IVec3::ONE, WorldVoxel::Solid(1));
        edits.commit();
        let before = edits.edit_stamp(IVec3::ZERO);

        edits.forget();
        assert!(edits.edit_stamp(IVec3::ZERO) > before);
        edits.set(IVec3::ONE, WorldVoxel::Solid(2));
        edits.commit();
        assert_eq!(edits.edit_sequence(IVec3::ZERO), 1);
        assert!(edits.edit_stamp(IVec3::ZERO) > before + 1);
    }

    #[cfg(feature = "light_propagation")]
    #[test]
    fn after_a_forget_every_stamp_is_past_each_stamp_handed_out_before() {
        let edits = Edits::default();
        // Three edits in the chunk at the origin, one in the chunk above it
        // along z.
        let in_z = IVec3::new(0, 0, 32);
        for x in 0..3 {
            edits.set(IVec3::new(x, 0, 0), WorldVoxel::Solid(1));
        }
        edits.set(in_z, WorldVoxel::Solid(1));
        edits.commit();
        let highest = edits
            .edit_stamp(IVec3::ZERO)
            .max(edits.edit_stamp(IVec3::Z));
        assert!(edits.stamp_has_edits(edits.edit_stamp(IVec3::ZERO)));

        edits.forget();
        for chunk in [IVec3::ZERO, IVec3::Z, IVec3::X] {
            assert!(edits.edit_stamp(chunk) > highest, "{chunk}");
            assert!(!edits.stamp_has_edits(edits.edit_stamp(chunk)), "{chunk}");
        }

        // An edit in the chunk along z alone, then a second forget.
        edits.set(in_z, WorldVoxel::Solid(2));
        edits.commit();
        let highest = edits.edit_stamp(IVec3::Z);
        assert!(edits.stamp_has_edits(highest));
        edits.forget();
        assert!(edits.edit_stamp(IVec3::ZERO) > highest);
    }

    #[test]
    fn voxels_set_after_an_applied_delta_move_the_sequence_on_from_it() {
        let edits = Edits::default();
        assert_eq!(
            edits.apply(delta(5, false, &[IVec3::X])),
            Err(EditGap {
                chunk: IVec3::ZERO,
                local: 0,
                received: 5,
            })
        );
        assert_eq!(edits.apply(delta(1, false, &[IVec3::X])), Ok(()));
        edits.set(IVec3::Y, WorldVoxel::Solid(3));

        let sets = edits.commit();
        assert_eq!((sets[0].seq, sets[0].replaced), (2, false));
        assert_eq!(edits.edit_sequence(IVec3::ZERO), 2);
    }

    #[test]
    fn voxels_set_before_an_applied_delta_leave_its_sequence_untold() {
        let edits = Edits::default();
        edits.set(IVec3::Y, WorldVoxel::Solid(3));
        edits.set(IVec3::Z, WorldVoxel::Solid(3));
        assert_eq!(edits.apply(delta(1, false, &[IVec3::X])), Ok(()));

        let sets = edits.commit();
        // Three voxels changed, but the sequence is the delta's, so the
        // sequence before can't be worked out from them.
        assert_eq!(sets[0].voxels.len(), 3);
        assert_eq!((sets[0].seq, sets[0].replaced), (1, true));
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
    fn a_chunk_cleared_of_its_edits_reaches_no_chunk_task() {
        let edits = Edits::default();
        commit_one(&edits, IVec3::new(-1, 5, 5), WorldVoxel::Solid(1));
        edits.clear_all();
        edits.commit();
        // The chunk keeps its sequence, so the clear reaches later worlds.
        assert_eq!(edits.edited_chunks(), vec![(IVec3::new(-1, 0, 0), 2)]);

        for key in [ChunkKey::new(IVec3::ZERO, 0), ChunkKey::new(IVec3::ZERO, 1)] {
            let padding = default_padding(key.lod_level);
            assert!(!edits.committed().reach(key, padding), "{key:?}");
            assert!(edits.chunk_edits(key, padding).is_empty(), "{key:?}");
        }
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
