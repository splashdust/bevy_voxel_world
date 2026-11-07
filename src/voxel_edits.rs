///
/// Voxel edits
/// Every voxel edit of a voxel world, kept for as long as the world exists, and
/// which chunks each voxel edit makes stale.
///
/// Voxel edits can be set from systems running in parallel, and are committed
/// once a frame. The module reads no Bevy world state, so it can be tested with
/// plain values.
///
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

/// The edited voxels of one chunk, shared with the chunk tasks that read them.
pub(crate) type EditBucket<I> = Arc<Voxels<I>>;

/// Committed edits, by chunk.
type Committed<I> = Arc<RwLock<HashMap<IVec3, EditBucket<I>>>>;

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
    /// The edits set since the last commit. The map is copied only when an edit
    /// is set while a reader holds it.
    pending: Mutex<Arc<Voxels<I>>>,
    committed: Committed<I>,
    /// How many times each level-0 chunk position's committed edits have
    /// changed, for positions with edits.
    #[cfg(feature = "light_propagation")]
    counts: RwLock<HashMap<IVec3, u64>>,
    _marker: PhantomData<C>,
}

impl<C, I> Default for VoxelEdits<C, I> {
    fn default() -> Self {
        Self {
            pending: Mutex::default(),
            committed: Arc::default(),
            #[cfg(feature = "light_propagation")]
            counts: RwLock::default(),
            _marker: PhantomData,
        }
    }
}

impl<C, I: Copy + PartialEq> VoxelEdits<C, I> {
    /// Sets the voxel at `position`, replacing any voxel set there since the
    /// last commit.
    pub(crate) fn set(&self, position: IVec3, voxel: WorldVoxel<I>) {
        let mut pending = self.pending.lock().unwrap();
        Arc::make_mut(&mut pending).insert(position, voxel);
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

    /// Commits the pending edits. Returns the positions of the voxels that
    /// changed; [`chunks_including`] gives the chunks each one makes stale.
    ///
    /// An edit that sets a voxel to the voxel already committed there changes
    /// nothing. So setting the same voxel every frame, or setting a voxel and
    /// then back, doesn't remesh its chunk.
    ///
    /// A chunk's committed edits are copied only when a running chunk task
    /// still holds them.
    pub(crate) fn commit(&self) -> Vec<IVec3> {
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
        let mut changed = Vec::new();
        for (&position, &voxel) in pending.iter() {
            let bucket = buckets.entry(chunk_containing(position)).or_default();
            if bucket.get(&position) == Some(&voxel) {
                continue;
            }
            Arc::make_mut(bucket).insert(position, voxel);
            changed.push(position);
        }
        #[cfg(feature = "light_propagation")]
        if !changed.is_empty() {
            let mut counts = self.counts.write().unwrap();
            for position in &changed {
                *counts.entry(chunk_containing(*position)).or_default() += 1;
            }
        }
        match Arc::get_mut(&mut pending) {
            Some(voxels) => voxels.clear(),
            None => *pending = Arc::default(),
        }
        changed
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

    /// How many times the committed edits of the level-0 chunk at `position`
    /// have changed: 0 for a position with no edits. Work that reads the
    /// position's voxels notes it when it starts, and its result is stale if
    /// it has moved since.
    #[cfg(feature = "light_propagation")]
    pub(crate) fn edit_count(&self, position: IVec3) -> u64 {
        let counts = self.counts.read().unwrap();
        if counts.is_empty() {
            return 0;
        }
        counts.get(&position).copied().unwrap_or(0)
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
pub(crate) struct CommittedEdits<'a, I>(
    RwLockReadGuard<'a, HashMap<IVec3, EditBucket<I>>>,
);

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
pub(crate) fn chunks_including(
    position: IVec3,
    lod_level: LodLevel,
    padding: IVec3,
) -> impl Iterator<Item = ChunkKey> {
    let key = ChunkKey::containing(position, lod_level);
    let in_chunk = position - key.min_voxel();
    let last = IVec3::splat(key.voxel_extent()) - padding;
    let offsets = |axis: usize| {
        let low = if in_chunk[axis] < padding[axis] {
            -1
        } else {
            0
        };
        let high = if in_chunk[axis] >= last[axis] { 1 } else { 0 };
        low..=high
    };
    let [xs, ys, zs] = [offsets(0), offsets(1), offsets(2)];
    let extent = key.extent();
    xs.flat_map(move |x| {
        let zs = zs.clone();
        ys.clone().flat_map(move |y| {
            zs.clone().map(move |z| ChunkKey {
                position: key.position + IVec3::new(x, y, z) * extent,
                lod_level,
            })
        })
    })
}

/// Reads voxel edits from any thread.
pub(crate) struct VoxelEditReader<I> {
    pending: Arc<Voxels<I>>,
    committed: Committed<I>,
}

impl<I: Copy> VoxelEditReader<I> {
    /// The voxel edit at `position`, if there is one.
    pub fn get(&self, position: IVec3) -> Option<WorldVoxel<I>> {
        if let Some(voxel) = self.pending.get(&position) {
            return Some(*voxel);
        }
        let buckets = self.committed.read().unwrap();
        buckets
            .get(&chunk_containing(position))?
            .get(&position)
            .copied()
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

impl<I: Copy> ChunkEdits<I> {
    pub fn is_empty(&self) -> bool {
        match &self.buckets {
            Buckets::Around(around) => around.iter().all(Option::is_none),
            Buckets::Within(within) => within.is_empty(),
        }
    }

    #[inline]
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
                around[i].as_ref()?.get(&position).copied()
            }
            Buckets::Within(within) => within.get(&chunk_pos)?.get(&position).copied(),
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
        edits.commit()
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
