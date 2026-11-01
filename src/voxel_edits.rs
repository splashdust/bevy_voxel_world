///
/// Voxel edits
/// Every voxel edit of a voxel world, kept for as long as the world exists, and
/// which chunks each edit makes stale. Edits can be set from systems running in
/// parallel, and are committed once a frame. It reads no Bevy world state, so it
/// can be tested with plain values.
///
use std::{
    marker::PhantomData,
    sync::{Arc, Mutex, RwLock},
};

use bevy::{
    platform::collections::{HashMap, HashSet},
    prelude::*,
};

use crate::{
    chunk::{chunk_containing, CHUNK_SIZE_I},
    voxel::WorldVoxel,
};

type Voxels<I> = HashMap<IVec3, WorldVoxel<I>>;

/// The edited voxels of one chunk, shared with the chunk tasks that read them.
type EditBucket<I> = Arc<Voxels<I>>;

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
    /// Copied only when an edit is set while a reader holds it.
    pending: Mutex<Arc<Voxels<I>>>,
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
        let mut pending = self.pending.lock().unwrap();
        Arc::make_mut(&mut pending).insert(position, voxel);
    }

    /// Reads the edits from any thread: the pending edits as they were when it
    /// was made, and the committed edits as they are when read.
    pub(crate) fn reader(&self) -> VoxelEditReader<I> {
        VoxelEditReader {
            pending: self.pending.lock().unwrap().clone(),
            committed: self.committed.clone(),
        }
    }

    /// Commits the pending edits and returns the chunks they make stale: those
    /// whose padded voxel data includes a voxel that changed. An edit that sets
    /// a voxel to its committed edit changes nothing, so setting the same voxel
    /// every frame, or setting a voxel and back, doesn't remesh its chunk. A
    /// chunk's edits are copied only while a running chunk task holds them.
    pub(crate) fn commit(&self) -> HashSet<IVec3> {
        // The pending edits stay locked until they are committed, so a reader
        // made meanwhile finds each edit pending or committed.
        let mut pending = self.pending.lock().unwrap();
        if pending.is_empty() {
            return HashSet::new();
        }
        let mut buckets = {
            #[cfg(feature = "trace")]
            let _span = bevy::log::info_span!("wait_voxel_edits_write").entered();
            self.committed.write().unwrap()
        };
        let mut stale = HashSet::new();
        for (&position, &voxel) in pending.iter() {
            let bucket = buckets.entry(chunk_containing(position)).or_default();
            if bucket.get(&position) == Some(&voxel) {
                continue;
            }
            Arc::make_mut(bucket).insert(position, voxel);
            add_chunks_including(position, &mut stale);
        }
        match Arc::get_mut(&mut pending) {
            Some(voxels) => voxels.clear(),
            None => *pending = Arc::default(),
        }
        stale
    }

    /// The edits a chunk task at `chunk_pos` reads.
    pub(crate) fn chunk_edits(&self, chunk_pos: IVec3) -> ChunkEdits<I> {
        let buckets = self.committed.read().unwrap();
        let mut edits = ChunkEdits {
            chunk_pos,
            buckets: std::array::from_fn(|_| None),
        };
        if buckets.is_empty() {
            return edits;
        }
        for (i, bucket) in edits.buckets.iter_mut().enumerate() {
            let i = i as i32;
            let offset = IVec3::new(i % 3, i / 3 % 3, i / 9) - 1;
            *bucket = buckets.get(&(chunk_pos + offset)).cloned();
        }
        edits
    }
}

/// Adds to `stale` the chunks whose padded voxel data includes the voxel at
/// `position`: its own chunk, and the chunks across each face, edge and corner
/// of it that the voxel lies on.
fn add_chunks_including(position: IVec3, stale: &mut HashSet<IVec3>) {
    const LAST: i32 = CHUNK_SIZE_I - 1;
    let chunk_pos = chunk_containing(position);
    let in_chunk = position - chunk_pos * CHUNK_SIZE_I;
    let offsets = |coordinate| match coordinate {
        0 => -1..=0,
        LAST => 0..=1,
        _ => 0..=0,
    };
    for x in offsets(in_chunk.x) {
        for y in offsets(in_chunk.y) {
            for z in offsets(in_chunk.z) {
                stale.insert(chunk_pos + IVec3::new(x, y, z));
            }
        }
    }
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

/// The voxel edits a chunk task reads, taken when it starts: those of its chunk
/// and of the neighbouring chunks its padding reaches into.
pub(crate) struct ChunkEdits<I> {
    chunk_pos: IVec3,
    /// By offset from `chunk_pos`, x varying fastest.
    buckets: [Option<EditBucket<I>>; 27],
}

impl<I> Default for ChunkEdits<I> {
    fn default() -> Self {
        Self {
            chunk_pos: IVec3::ZERO,
            buckets: std::array::from_fn(|_| None),
        }
    }
}

impl<I: Copy> ChunkEdits<I> {
    pub fn is_empty(&self) -> bool {
        self.buckets.iter().all(Option::is_none)
    }

    #[inline]
    pub fn get(&self, position: IVec3) -> Option<WorldVoxel<I>> {
        let offset = chunk_containing(position) - self.chunk_pos + 1;
        if offset.cmplt(IVec3::ZERO).any() || offset.cmpgt(IVec3::splat(2)).any() {
            return None;
        }
        let i = (offset.x + offset.y * 3 + offset.z * 9) as usize;
        self.buckets[i].as_ref()?.get(&position).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    type Edits = VoxelEdits<(), u8>;

    const LAST: i32 = CHUNK_SIZE_I - 1;

    /// Sets one voxel and commits it, returning the chunks it makes stale.
    fn commit_one(edits: &Edits, position: IVec3, voxel: WorldVoxel) -> HashSet<IVec3> {
        edits.set(position, voxel);
        edits.commit()
    }

    /// The chunks that one edit at `position` makes stale.
    fn stale_after_editing(position: IVec3) -> HashSet<IVec3> {
        commit_one(&Edits::default(), position, WorldVoxel::Solid(1))
    }

    fn chunks(positions: &[[i32; 3]]) -> HashSet<IVec3> {
        positions.iter().copied().map(IVec3::from_array).collect()
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
        // The last voxel read. Each edit is committed only once it has been read
        // while pending, so that a reader made during the commit could tell if
        // the edit went missing.
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
    fn committing_the_same_voxel_again_makes_no_chunk_stale() {
        let edits = Edits::default();
        let position = IVec3::new(5, 5, 5);

        assert_eq!(
            commit_one(&edits, position, WorldVoxel::Solid(1)),
            chunks(&[[0, 0, 0]])
        );
        assert!(commit_one(&edits, position, WorldVoxel::Solid(1)).is_empty());
    }

    #[test]
    fn committing_a_different_voxel_makes_its_chunk_stale_again() {
        let edits = Edits::default();
        let position = IVec3::new(5, 5, 5);
        commit_one(&edits, position, WorldVoxel::Solid(1));

        assert_eq!(
            commit_one(&edits, position, WorldVoxel::Solid(2)),
            chunks(&[[0, 0, 0]])
        );
        assert_eq!(edits.reader().get(position), Some(WorldVoxel::Solid(2)));
    }

    #[test]
    fn setting_a_committed_voxel_and_back_makes_no_chunk_stale() {
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
    fn a_chunk_task_reads_the_edits_its_padding_reaches() {
        let edits = Edits::default();
        let position = IVec3::new(-1, 5, 5);
        commit_one(&edits, position, WorldVoxel::Solid(1));

        for chunk_pos in [IVec3::new(-1, 0, 0), IVec3::ZERO] {
            let chunk_edits = edits.chunk_edits(chunk_pos);
            assert!(!chunk_edits.is_empty());
            assert_eq!(chunk_edits.get(position), Some(WorldVoxel::Solid(1)));
        }
        assert!(edits.chunk_edits(IVec3::new(1, 0, 0)).is_empty());
    }

    #[test]
    fn a_chunk_task_keeps_the_edits_it_started_with() {
        let edits = Edits::default();
        let position = IVec3::new(5, 5, 5);
        commit_one(&edits, position, WorldVoxel::Solid(1));
        let started_with = edits.chunk_edits(IVec3::ZERO);

        commit_one(&edits, position, WorldVoxel::Solid(2));

        assert_eq!(started_with.get(position), Some(WorldVoxel::Solid(1)));
        assert_eq!(
            edits.chunk_edits(IVec3::ZERO).get(position),
            Some(WorldVoxel::Solid(2))
        );
    }
}
