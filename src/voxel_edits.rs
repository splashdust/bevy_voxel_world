///
/// Voxel edits
/// Every voxel edit of a voxel world, kept for as long as the world exists, and
/// which chunks each edit makes stale. It reads no Bevy world state, so it can
/// be tested with plain values.
///
use std::{
    marker::PhantomData,
    sync::{Arc, RwLock},
};

use bevy::{
    platform::collections::{HashMap, HashSet},
    prelude::*,
};

use crate::{
    chunk::{chunk_containing, CHUNK_SIZE_I},
    voxel::WorldVoxel,
};

/// The edited voxels of one chunk, shared with the chunk tasks that read them.
type EditBucket<I> = Arc<HashMap<IVec3, WorldVoxel<I>>>;

/// Every voxel edit of the voxel world `C`. Edits are grouped by chunk, so a
/// chunk task takes the edits it reads when it starts and holds no lock while
/// it runs.
///
/// Public only because `VoxelWorld` reads it; this module isn't exported.
#[derive(Resource, Clone)]
pub struct VoxelEdits<C, I>(Arc<RwLock<HashMap<IVec3, EditBucket<I>>>>, PhantomData<C>);

impl<C, I> Default for VoxelEdits<C, I> {
    fn default() -> Self {
        Self(Arc::new(RwLock::new(HashMap::new())), PhantomData)
    }
}

impl<C, I: Copy + PartialEq> VoxelEdits<C, I> {
    /// Reads the committed edits, from any thread, including those committed
    /// after it was made.
    pub(crate) fn reader(&self) -> VoxelEditReader<I> {
        VoxelEditReader(self.0.clone())
    }

    /// Commits `edits` in order and returns the chunks they make stale: those
    /// whose padded voxel data includes a voxel that changed. Committing a voxel
    /// that is already committed changes nothing, so setting the same voxel
    /// every frame doesn't remesh its chunk every frame. A chunk's edits are
    /// copied only while a running chunk task holds them.
    pub(crate) fn commit(&self, edits: &[(IVec3, WorldVoxel<I>)]) -> HashSet<IVec3> {
        let mut buckets = {
            #[cfg(feature = "trace")]
            let _span = bevy::log::info_span!("wait_voxel_edits_write").entered();
            self.0.write().unwrap()
        };
        let mut stale = HashSet::new();
        for &(position, voxel) in edits {
            let bucket = buckets.entry(chunk_containing(position)).or_default();
            if bucket.get(&position) == Some(&voxel) {
                continue;
            }
            Arc::make_mut(bucket).insert(position, voxel);
            add_chunks_including(position, &mut stale);
        }
        stale
    }

    /// The edits a chunk task at `chunk_pos` reads.
    pub(crate) fn chunk_edits(&self, chunk_pos: IVec3) -> ChunkEdits<I> {
        let buckets = self.0.read().unwrap();
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

/// Reads committed voxel edits from any thread.
pub(crate) struct VoxelEditReader<I>(Arc<RwLock<HashMap<IVec3, EditBucket<I>>>>);

impl<I: Copy> VoxelEditReader<I> {
    /// The voxel edit at `position`, if there is one.
    pub fn get(&self, position: IVec3) -> Option<WorldVoxel<I>> {
        let buckets = self.0.read().unwrap();
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

    type Edits = VoxelEdits<(), u8>;

    const LAST: i32 = CHUNK_SIZE_I - 1;

    /// The chunks that one edit at `position` makes stale.
    fn stale_after_editing(position: IVec3) -> HashSet<IVec3> {
        Edits::default().commit(&[(position, WorldVoxel::Solid(1))])
    }

    fn chunks(positions: &[[i32; 3]]) -> HashSet<IVec3> {
        positions.iter().copied().map(IVec3::from_array).collect()
    }

    #[test]
    fn committed_edits_are_read_back() {
        let edits = Edits::default();
        let position = IVec3::new(3, -40, 70);
        edits.commit(&[(position, WorldVoxel::Solid(2))]);

        let reader = edits.reader();
        assert_eq!(reader.get(position), Some(WorldVoxel::Solid(2)));
        assert_eq!(reader.get(position + IVec3::X), None);
    }

    #[test]
    fn a_reader_sees_edits_committed_after_it_was_made() {
        let edits = Edits::default();
        let reader = edits.reader();
        edits.commit(&[(IVec3::ZERO, WorldVoxel::Solid(1))]);

        assert_eq!(reader.get(IVec3::ZERO), Some(WorldVoxel::Solid(1)));
    }

    #[test]
    fn committing_the_same_voxel_again_makes_no_chunk_stale() {
        let edits = Edits::default();
        let edit = [(IVec3::new(5, 5, 5), WorldVoxel::Solid(1))];

        assert_eq!(edits.commit(&edit), chunks(&[[0, 0, 0]]));
        assert!(edits.commit(&edit).is_empty());
    }

    #[test]
    fn committing_a_different_voxel_makes_its_chunk_stale_again() {
        let edits = Edits::default();
        let position = IVec3::new(5, 5, 5);
        edits.commit(&[(position, WorldVoxel::Solid(1))]);

        assert_eq!(
            edits.commit(&[(position, WorldVoxel::Solid(2))]),
            chunks(&[[0, 0, 0]])
        );
        assert_eq!(edits.reader().get(position), Some(WorldVoxel::Solid(2)));
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
        edits.commit(&[(position, WorldVoxel::Solid(1))]);

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
        edits.commit(&[(position, WorldVoxel::Solid(1))]);
        let started_with = edits.chunk_edits(IVec3::ZERO);

        edits.commit(&[(position, WorldVoxel::Solid(2))]);

        assert_eq!(started_with.get(position), Some(WorldVoxel::Solid(1)));
        assert_eq!(
            edits.chunk_edits(IVec3::ZERO).get(position),
            Some(WorldVoxel::Solid(2))
        );
    }
}
