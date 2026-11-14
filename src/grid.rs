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
//! [`GridTable`] keeps the world's entries in a hash map of their own, keyed as
//! they would be with no sub-grids, and each sub-grid's in a small one beside
//! it. A lookup picks the grid's table first, and a scan over one grid's
//! entries visits that grid's alone, so the world's streaming never pays for
//! sub-grids' chunks.

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

/// Hands out the ids of a world's sub-grids, and takes them back to hand out
/// again.
///
/// An id is free again only once it is released, which the crate does once
/// the despawned sub-grid's entries are gone from every table, so a new
/// sub-grid never finds an old one's voxel edits.
#[cfg(feature = "sub_grids")]
#[derive(Default)]
pub(crate) struct GridIds {
    /// The highest id handed out so far.
    highest: u16,
    /// Released ids, handed out again before any higher one.
    free: Vec<GridId>,
}

#[cfg(feature = "sub_grids")]
impl GridIds {
    /// A free id, or `None` when every id but the world's is in use.
    pub fn allocate(&mut self) -> Option<GridId> {
        if let Some(id) = self.free.pop() {
            return Some(id);
        }
        self.highest = self.highest.checked_add(1)?;
        Some(GridId(self.highest))
    }

    /// Frees `id`, which a sub-grid held until its entries were gone.
    pub fn release(&mut self, id: GridId) {
        debug_assert!(id != GridId::WORLD && id.0 <= self.highest);
        self.free.push(id);
    }
}

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
    /// Hashes the key as two words rather than a word per field: the sets of
    /// grid keys the chunk lifecycle builds hash a key on every insert, and
    /// the hashers cost about the same per write whatever its width.
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

/// The key of the chunk `offset` chunks of the same LOD level from this one,
/// on the same grid: a chunk's neighbours are always on its own grid.
impl std::ops::Add<IVec3> for GridKey {
    type Output = Self;

    #[inline]
    fn add(self, offset: IVec3) -> Self {
        Self {
            position: self.position + offset * (1 << self.lod_level),
            ..self
        }
    }
}

/// The key of an entry within its grid's table in a [`GridTable`]: what is
/// left of a [`GridKey`] once its grid has picked the table.
pub(crate) trait KeyInGrid: Copy + Eq + Hash {
    /// The key of the entry at `key` within its grid's table.
    fn of(key: &GridKey) -> Self;

    /// The grid key of the entry at this key in `grid`'s table.
    fn on(self, grid: GridId) -> GridKey;
}

impl KeyInGrid for ChunkKey {
    #[inline]
    fn of(key: &GridKey) -> Self {
        key.key()
    }

    #[inline]
    fn on(self, grid: GridId) -> GridKey {
        GridKey::new(grid, self)
    }
}

/// A level-0 chunk position, for a table whose entries are all at level 0.
impl KeyInGrid for IVec3 {
    #[inline]
    fn of(key: &GridKey) -> Self {
        debug_assert_eq!(key.lod_level, 0, "{} isn't at level 0", key.key());
        key.position
    }

    #[inline]
    fn on(self, grid: GridId) -> GridKey {
        GridKey::level_0(grid, self)
    }
}

/// A table of values by [`GridKey`], holding the entries of every grid.
///
/// The world's entries sit in a hash map of their own keyed by `K`, as they
/// would with no sub-grids, and each sub-grid with entries has a small one of
/// its own. `K` is the key within a grid's table, and `S` builds its hasher.
///
/// A sub-grid's table goes as soon as it holds no entry, so a table that
/// holds only the world's entries keeps no other table.
///
/// A lookup or scan of the world's entries inlines to the plain hash map
/// operation it would be with no sub-grids, and a sub-grid's picks its table
/// out of line. The streaming policy looks up the world's chunks many times
/// each frame, and code for sub-grids inlined into those callers made them
/// too big to inline in turn.
#[derive(Clone)]
pub(crate) struct GridTable<K, V, S = FixedHasher> {
    world: HashMap<K, V, S>,
    #[cfg(feature = "sub_grids")]
    sub_grids: HashMap<GridId, HashMap<K, V, S>>,
    /// Always empty: the table a scan of a grid with no entries reads, so a
    /// scan of the world's entries is a plain scan of its hash map.
    #[cfg(feature = "sub_grids")]
    empty: HashMap<K, V, S>,
}

impl<K, V, S: Default> Default for GridTable<K, V, S> {
    fn default() -> Self {
        Self {
            world: HashMap::with_hasher(S::default()),
            #[cfg(feature = "sub_grids")]
            sub_grids: HashMap::new(),
            #[cfg(feature = "sub_grids")]
            empty: HashMap::with_hasher(S::default()),
        }
    }
}

impl<K: KeyInGrid, V, S: BuildHasher + Default> GridTable<K, V, S> {
    /// A table with room for `capacity` of the world's entries before it
    /// grows.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            world: HashMap::with_capacity_and_hasher(capacity, S::default()),
            #[cfg(feature = "sub_grids")]
            sub_grids: HashMap::new(),
            #[cfg(feature = "sub_grids")]
            empty: HashMap::with_hasher(S::default()),
        }
    }

    /// Sets the value at `key`. Returns the value it replaced, if any.
    pub fn insert(&mut self, key: GridKey, value: V) -> Option<V> {
        self.table_or_new(key.grid).insert(K::of(&key), value)
    }

    /// The value at `key`, set by `make` first if there is none.
    pub fn get_or_insert_with(
        &mut self,
        key: GridKey,
        make: impl FnOnce() -> V,
    ) -> &mut V {
        self.table_or_new(key.grid)
            .entry(K::of(&key))
            .or_insert_with(make)
    }

    /// Moves `grid`'s entries out into a table of their own.
    pub fn take_grid(&mut self, grid: GridId) -> Self {
        let mut taken = Self::default();
        #[cfg(feature = "sub_grids")]
        if grid != GridId::WORLD {
            if let Some(table) = self.sub_grids.remove(&grid) {
                taken.sub_grids.insert(grid, table);
            }
            return taken;
        }
        let _ = grid;
        taken.world = std::mem::take(&mut self.world);
        taken
    }

    /// `grid`'s table, made empty first if it has none.
    #[inline]
    fn table_or_new(&mut self, grid: GridId) -> &mut HashMap<K, V, S> {
        #[cfg(feature = "sub_grids")]
        if grid != GridId::WORLD {
            return self
                .sub_grids
                .entry(grid)
                .or_insert_with(|| HashMap::with_hasher(S::default()));
        }
        let _ = grid;
        &mut self.world
    }
}

impl<K: KeyInGrid, V, S: BuildHasher> GridTable<K, V, S> {
    #[inline(always)]
    pub fn get(&self, key: &GridKey) -> Option<&V> {
        #[cfg(feature = "sub_grids")]
        if key.grid != GridId::WORLD {
            return self.sub_grid_get(key);
        }
        self.world.get(&K::of(key))
    }

    #[inline(always)]
    pub fn contains(&self, key: &GridKey) -> bool {
        #[cfg(feature = "sub_grids")]
        if key.grid != GridId::WORLD {
            return self.sub_grid_get(key).is_some();
        }
        self.world.contains_key(&K::of(key))
    }

    /// The value at `key`, which is a sub-grid's.
    #[cfg(feature = "sub_grids")]
    #[cold]
    #[inline(never)]
    fn sub_grid_get(&self, key: &GridKey) -> Option<&V> {
        self.sub_grids.get(&key.grid)?.get(&K::of(key))
    }

    /// Removes the value at `key`, and returns it if there was one.
    pub fn remove(&mut self, key: &GridKey) -> Option<V> {
        #[cfg(feature = "sub_grids")]
        if key.grid != GridId::WORLD {
            let table = self.sub_grids.get_mut(&key.grid)?;
            let removed = table.remove(&K::of(key));
            if table.is_empty() {
                self.sub_grids.remove(&key.grid);
            }
            return removed;
        }
        self.world.remove(&K::of(key))
    }

    /// Whether no grid has an entry.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.world.is_empty() && !self.has_sub_grids()
    }

    /// Whether every entry is `grid`'s, as it is in an empty table.
    pub fn holds_only(&self, grid: GridId) -> bool {
        #[cfg(feature = "sub_grids")]
        if grid != GridId::WORLD {
            return self.world.is_empty() && self.sub_grids.keys().all(|g| *g == grid);
        }
        let _ = grid;
        !self.has_sub_grids()
    }

    /// Every entry of `grid`, in no particular order. Its cost grows with
    /// `grid`'s entries alone.
    pub fn grid_entries(&self, grid: GridId) -> impl Iterator<Item = (GridKey, &V)> {
        self.table(grid)
            .iter()
            .map(move |(key, v)| (key.on(grid), v))
    }

    /// The key within the world's table of every entry of the world, in no
    /// particular order: a plain scan of the world's hash map, for the
    /// streaming policy, which plans only the world's chunks.
    #[inline]
    pub fn world_keys(&self) -> impl Iterator<Item = K> + '_ {
        self.world.keys().copied()
    }

    /// The key of every entry of `grid`, in no particular order. Its cost
    /// grows with `grid`'s entries alone.
    pub fn grid_keys(&self, grid: GridId) -> impl Iterator<Item = GridKey> + '_ {
        self.table(grid).keys().map(move |key| key.on(grid))
    }

    /// Removes every entry of `grid`.
    pub fn clear_grid(&mut self, grid: GridId) {
        #[cfg(feature = "sub_grids")]
        if grid != GridId::WORLD {
            self.sub_grids.remove(&grid);
            return;
        }
        let _ = grid;
        self.world.clear();
    }

    /// Removes every entry of `grid`, and returns them in no particular order.
    pub fn drain_grid(
        &mut self,
        grid: GridId,
    ) -> impl Iterator<Item = (GridKey, V)> + '_ {
        #[cfg(feature = "sub_grids")]
        let (world, sub_grid) = if grid == GridId::WORLD {
            (Some(&mut self.world), None)
        } else {
            (None, self.sub_grids.remove(&grid))
        };
        #[cfg(not(feature = "sub_grids"))]
        let (world, sub_grid) = (Some(&mut self.world), None::<HashMap<K, V, S>>);
        world
            .into_iter()
            .flat_map(|table| table.drain())
            .chain(sub_grid.into_iter().flatten())
            .map(move |(key, v)| (key.on(grid), v))
    }

    /// Every entry of every grid, in no particular order.
    pub fn into_entries(self) -> impl Iterator<Item = (GridKey, V)> {
        let entries = self
            .world
            .into_iter()
            .map(|(key, v)| (key.on(GridId::WORLD), v));
        #[cfg(feature = "sub_grids")]
        let entries =
            entries.chain(self.sub_grids.into_iter().flat_map(|(grid, table)| {
                table.into_iter().map(move |(key, v)| (key.on(grid), v))
            }));
        entries
    }

    /// Whether any sub-grid has an entry.
    #[inline]
    fn has_sub_grids(&self) -> bool {
        #[cfg(feature = "sub_grids")]
        return !self.sub_grids.is_empty();
        #[cfg(not(feature = "sub_grids"))]
        false
    }

    /// `grid`'s table, or an empty one if it has none.
    #[inline(always)]
    fn table(&self, grid: GridId) -> &HashMap<K, V, S> {
        #[cfg(feature = "sub_grids")]
        if grid != GridId::WORLD {
            return self.sub_grid_table(grid);
        }
        let _ = grid;
        &self.world
    }

    /// The table of `grid`, which isn't the world, or an empty one if it has
    /// none.
    #[cfg(feature = "sub_grids")]
    #[cold]
    #[inline(never)]
    fn sub_grid_table(&self, grid: GridId) -> &HashMap<K, V, S> {
        self.sub_grids.get(&grid).unwrap_or(&self.empty)
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

    /// `grid`'s value, if it has one. The world's inlines to a field read, as
    /// a [`GridTable`] lookup in the world's table does.
    #[inline(always)]
    pub fn get(&self, grid: GridId) -> Option<&A> {
        #[cfg(feature = "sub_grids")]
        if grid != GridId::WORLD {
            return self.other(grid);
        }
        let _ = grid;
        Some(&self.world)
    }

    /// The value of `grid`, which isn't the world, if it has one.
    #[cfg(feature = "sub_grids")]
    #[cold]
    #[inline(never)]
    fn other(&self, grid: GridId) -> Option<&A> {
        self.others.get(&grid)
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
        let mut table = GridTable::<ChunkKey, u32>::default();
        let key = ChunkKey::new(IVec3::ZERO, 0);
        table.insert(GridKey::world(key), 1);
        table.insert(GridKey::new(GridId::new(1), key), 2);

        let world: Vec<u32> =
            table.grid_entries(GridId::WORLD).map(|(_, v)| *v).collect();
        assert_eq!(world, vec![1]);
        let taken = table.take_grid(GridId::new(1));
        assert_eq!(
            taken.into_entries().map(|(_, v)| v).collect::<Vec<_>>(),
            vec![2]
        );
        assert!(table.holds_only(GridId::WORLD));
    }

    #[test]
    fn ids_run_out_past_the_last_sub_grid_and_a_released_id_comes_back() {
        let mut ids = GridIds::default();
        let first = ids.allocate().unwrap();
        assert_ne!(first, GridId::WORLD);
        for _ in 1..u16::MAX {
            assert!(ids.allocate().is_some());
        }
        assert_eq!(ids.allocate(), None);

        ids.release(first);
        assert_eq!(ids.allocate(), Some(first));
        assert_eq!(ids.allocate(), None);
    }

    #[test]
    fn a_table_is_empty_again_once_a_sub_grid_loses_its_last_entry() {
        let mut table = GridTable::<IVec3, u32>::default();
        let key = GridKey::level_0(GridId::new(1), IVec3::new(1, 2, 3));
        table.insert(key, 1);
        assert!(!table.is_empty());
        assert!(!table.holds_only(GridId::WORLD));

        assert_eq!(table.remove(&key), Some(1));
        assert!(table.is_empty());
        assert!(table.holds_only(GridId::WORLD));
        assert_eq!(table.grid_keys(GridId::new(1)).count(), 0);
    }
}
