//! Grids in the crate's chunk tables.
//!
//! A world's streamed voxels are one grid, and each of its sub-grids is
//! another. The chunk map, the chunk index and the voxel edits find a chunk by
//! its [`GridKey`]: its grid and its chunk key. Their methods that take a plain
//! `ChunkKey` or voxel position read the world's grid, so a sub-grid's chunks
//! and voxel edits never show through them.
//!
//! [`GridTable`] decides how the entries of different grids are stored, and
//! [`PerGrid`] holds a value for each grid beside a table, such as the LOD
//! levels in use, so one grid's chunks never count towards another's.
//!
//! [`GridTable`] keeps every grid in one hash map keyed by grid key. A lookup
//! costs what it does with one grid, and a scan over one grid's entries visits
//! every grid's, so its cost grows with the chunks of every grid.

use std::{
    hash::{BuildHasher, Hash, Hasher},
    mem::size_of,
};

use bevy::{
    platform::{collections::HashMap, hash::FixedHasher},
    prelude::*,
};

use crate::{chunk::ChunkKey, configuration::LodLevel};

/// Names a grid of a voxel world. The world's streamed voxels are
/// [`GridId::WORLD`].
///
/// With `sub_grids` off the world is the only grid, and the id takes no space.
#[cfg(feature = "sub_grids")]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub(crate) struct GridId(u16);

/// Names a grid of a voxel world. The world's streamed voxels are
/// [`GridId::WORLD`].
///
/// With `sub_grids` off the world is the only grid, and the id takes no space.
#[cfg(not(feature = "sub_grids"))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub(crate) struct GridId;

impl GridId {
    /// The world's streamed voxels.
    #[cfg(feature = "sub_grids")]
    pub const WORLD: Self = Self(0);
    /// The world's streamed voxels.
    #[cfg(not(feature = "sub_grids"))]
    pub const WORLD: Self = Self;

    /// The grid numbered `id`. 0 is the world.
    #[cfg(all(test, feature = "sub_grids"))]
    pub const fn new(id: u16) -> Self {
        Self(id)
    }

    /// The grid's number. 0 is the world.
    #[cfg(feature = "sub_grids")]
    fn number(self) -> u16 {
        self.0
    }

    /// The grid's number. 0 is the world.
    #[cfg(not(feature = "sub_grids"))]
    fn number(self) -> u16 {
        0
    }
}

#[cfg(not(feature = "sub_grids"))]
const _: () = assert!(size_of::<GridId>() == 0);

/// Finds a chunk in the crate's tables: the chunk's grid and its chunk key.
///
/// It holds the chunk key's fields beside the grid rather than a `ChunkKey`,
/// so the grid takes space the chunk key leaves as padding and the key stays
/// as small as a `ChunkKey`. Its equality and hash include the grid.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct GridKey {
    position: IVec3,
    lod_level: LodLevel,
    grid: GridId,
}

const _: () = assert!(size_of::<GridKey>() == size_of::<ChunkKey>());
const _: () = assert!(size_of::<GridKey>() == 16);

impl Hash for GridKey {
    /// Hashes the key as two words rather than a word per field: the tables
    /// hash a key on every lookup, and the hashers cost about the same per
    /// write whatever its width.
    fn hash<H: Hasher>(&self, state: &mut H) {
        let IVec3 { x, y, z } = self.position;
        state.write_u64(u64::from(x as u32) | u64::from(y as u32) << 32);
        state.write_u64(
            u64::from(z as u32)
                | u64::from(self.lod_level) << 32
                | u64::from(self.grid.number()) << 40,
        );
    }
}

impl GridKey {
    /// The chunk at `key` on `grid`.
    pub fn new(grid: GridId, key: ChunkKey) -> Self {
        Self {
            position: key.position,
            lod_level: key.lod_level,
            grid,
        }
    }

    /// The world's chunk at `key`.
    pub fn world(key: ChunkKey) -> Self {
        Self::new(GridId::WORLD, key)
    }

    /// The level-0 chunk at `chunk_position` on `grid`.
    pub fn level_0(grid: GridId, chunk_position: IVec3) -> Self {
        Self {
            position: chunk_position,
            lod_level: 0,
            grid,
        }
    }

    pub fn grid(&self) -> GridId {
        self.grid
    }

    /// The chunk key on the grid.
    pub fn key(&self) -> ChunkKey {
        ChunkKey {
            position: self.position,
            lod_level: self.lod_level,
        }
    }

    /// The chunk's minimum corner, in level-0 chunks of its grid.
    pub fn position(&self) -> IVec3 {
        self.position
    }

    pub fn lod_level(&self) -> LodLevel {
        self.lod_level
    }

    /// The key of the chunk one level above this one, on the same grid, that
    /// contains it.
    pub fn parent(&self) -> Self {
        debug_assert!(
            self.lod_level < crate::chunk::MAX_LOD_LEVEL,
            "{} has no parent",
            self.key()
        );
        Self::new(
            self.grid,
            ChunkKey::containing_chunk(self.position, self.lod_level + 1),
        )
    }

    /// The keys of the eight chunks one level below this one, on the same
    /// grid, that it contains.
    pub fn children(&self) -> impl Iterator<Item = Self> {
        debug_assert!(self.lod_level > 0, "{} has no children", self.key());
        let lod_level = self.lod_level.saturating_sub(1);
        let half = 1 << lod_level;
        let Self { position, grid, .. } = *self;
        (0..8).map(move |corner: i32| Self {
            position: position
                + IVec3::new(corner & 1, (corner >> 1) & 1, corner >> 2) * half,
            lod_level,
            grid,
        })
    }
}

/// A table of values by [`GridKey`], holding the entries of every grid.
///
/// `S` builds the hasher of the table's keys.
#[derive(Clone)]
pub(crate) struct GridTable<V, S = FixedHasher> {
    entries: HashMap<GridKey, V, S>,
}

impl<V, S: Default> Default for GridTable<V, S> {
    fn default() -> Self {
        Self {
            entries: HashMap::with_hasher(S::default()),
        }
    }
}

impl<V, S: BuildHasher + Default> GridTable<V, S> {
    /// A table with room for `capacity` entries before it grows.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            entries: HashMap::with_capacity_and_hasher(capacity, S::default()),
        }
    }

    /// Moves `grid`'s entries out into a table of their own.
    pub fn take_grid(&mut self, grid: GridId) -> Self {
        if self.holds_only(grid) {
            return std::mem::take(self);
        }
        Self {
            entries: self.entries.extract_if(|key, _| key.grid == grid).collect(),
        }
    }
}

impl<V, S: BuildHasher> GridTable<V, S> {
    #[inline]
    pub fn get(&self, key: &GridKey) -> Option<&V> {
        self.entries.get(key)
    }

    #[inline]
    pub fn contains(&self, key: &GridKey) -> bool {
        self.entries.contains_key(key)
    }

    /// Sets the value at `key`. Returns the value it replaced, if any.
    pub fn insert(&mut self, key: GridKey, value: V) -> Option<V> {
        self.entries.insert(key, value)
    }

    /// Removes the value at `key`, and returns it if there was one.
    pub fn remove(&mut self, key: &GridKey) -> Option<V> {
        self.entries.remove(key)
    }

    /// The value at `key`, set by `make` first if there is none.
    pub fn get_or_insert_with(
        &mut self,
        key: GridKey,
        make: impl FnOnce() -> V,
    ) -> &mut V {
        self.entries.entry(key).or_insert_with(make)
    }

    /// Whether no grid has an entry.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Whether every entry is `grid`'s, as it is in an empty table.
    pub fn holds_only(&self, grid: GridId) -> bool {
        self.entries.keys().all(|key| key.grid == grid)
    }

    /// Every entry of `grid`, in no particular order.
    pub fn grid_entries(&self, grid: GridId) -> impl Iterator<Item = (&GridKey, &V)> {
        self.entries.iter().filter(move |(key, _)| key.grid == grid)
    }

    /// The key of every entry of `grid`, in no particular order.
    pub fn grid_keys(&self, grid: GridId) -> impl Iterator<Item = GridKey> + '_ {
        self.entries
            .keys()
            .copied()
            .filter(move |key| key.grid == grid)
    }

    /// Removes every entry of `grid`.
    pub fn clear_grid(&mut self, grid: GridId) {
        if self.holds_only(grid) {
            self.entries.clear();
        } else {
            self.entries.retain(|key, _| key.grid != grid);
        }
    }

    /// Removes every entry of `grid`, and returns them in no particular order.
    pub fn drain_grid(
        &mut self,
        grid: GridId,
    ) -> impl Iterator<Item = (GridKey, V)> + '_ {
        self.entries.extract_if(move |key, _| key.grid == grid)
    }
}

impl<V, S> IntoIterator for GridTable<V, S> {
    type Item = (GridKey, V);
    type IntoIter = <HashMap<GridKey, V, S> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.entries.into_iter()
    }
}

/// A value for each grid that has one, such as an aggregate kept beside a
/// [`GridTable`]. The world's is always there.
pub(crate) struct PerGrid<A> {
    world: A,
    #[cfg(feature = "sub_grids")]
    others: HashMap<GridId, A>,
}

impl<A> PerGrid<A> {
    /// Holds `world` for the world, and nothing for any other grid.
    pub fn new(world: A) -> Self {
        Self {
            world,
            #[cfg(feature = "sub_grids")]
            others: HashMap::new(),
        }
    }

    /// `grid`'s value, if it has one.
    #[inline]
    pub fn get(&self, grid: GridId) -> Option<&A> {
        #[cfg(feature = "sub_grids")]
        if grid != GridId::WORLD {
            return self.others.get(&grid);
        }
        let _ = grid;
        Some(&self.world)
    }

    /// `grid`'s value, set by `make` first if it has none.
    #[inline]
    pub fn get_or_insert_with(
        &mut self,
        grid: GridId,
        make: impl FnOnce() -> A,
    ) -> &mut A {
        #[cfg(feature = "sub_grids")]
        if grid != GridId::WORLD {
            return self.others.entry(grid).or_insert_with(make);
        }
        let _ = (grid, make);
        &mut self.world
    }
}

#[cfg(all(test, feature = "sub_grids"))]
mod tests {
    use super::*;

    #[test]
    fn grid_keys_of_one_chunk_key_on_two_grids_differ_in_equality_and_hash() {
        let key = ChunkKey::new(IVec3::new(2, -4, 6), 1);
        let (world, other) = (GridKey::world(key), GridKey::new(GridId::new(1), key));

        assert_ne!(world, other);
        let hash = |key: &GridKey| FixedHasher.hash_one(key);
        assert_ne!(hash(&world), hash(&other));
        assert_eq!(other.key(), key);
        assert_eq!(other.parent().grid(), GridId::new(1));
        assert!(other.children().all(|child| child.grid() == GridId::new(1)));
    }

    #[test]
    fn a_scan_of_one_grid_leaves_the_others_out() {
        let mut table = GridTable::<u32>::default();
        let key = ChunkKey::new(IVec3::ZERO, 0);
        table.insert(GridKey::world(key), 1);
        table.insert(GridKey::new(GridId::new(1), key), 2);

        let world: Vec<u32> =
            table.grid_entries(GridId::WORLD).map(|(_, v)| *v).collect();
        assert_eq!(world, vec![1]);
        let taken = table.take_grid(GridId::new(1));
        assert_eq!(
            taken.into_iter().map(|(_, v)| v).collect::<Vec<_>>(),
            vec![2]
        );
        assert!(table.holds_only(GridId::WORLD));
    }
}
