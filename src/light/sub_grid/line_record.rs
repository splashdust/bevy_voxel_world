//! The line record: which lines of each of a sub-grid's chunks hold a solid,
//! along each of the sub-grid's three axes.
//!
//! Enclosure starts from the voxels where a line along one of the sub-grid's
//! axes leaves the sub-grid without meeting one of its voxels. A voxel's
//! line test in one of the six directions reads its own chunk's voxels past
//! it, then the record for each chunk further along its row. A chunk the
//! record doesn't hold counts as empty, so a hollow sub-grid a whole chunk
//! across still reads as closed. A line that stopped at the chunk's border
//! instead would light a sub-grid's inside wherever two of its chunks meet.
//!
//! Each chunk holds three 32×32 bit masks, one per axis, so the record grows
//! with the sub-grid's chunks. A line test's cost grows with the sub-grid's
//! extent in chunks along the line.
//!
//! When an edit changes a sub-grid's voxels, [`LineRecord::edit`] updates the
//! masks and works out which voxels' line tests flipped, as spans on the rows
//! through the edited voxels. Usually few flip: a solid placed on a row that
//! already holds solids on both sides of it flips nothing.
//!
//! The record is for sub-grids only. World light never reads it.

use std::ops::Range;

use bevy::platform::collections::{HashMap, HashSet};
use bevy::prelude::*;

use crate::chunk::{chunk_containing, CHUNK_SIZE_I, CHUNK_SIZE_U};

const CHUNK: usize = CHUNK_SIZE_U as usize;
const CHUNK_SHIFT: i32 = CHUNK_SIZE_U.trailing_zeros() as i32;

/// An axis of a sub-grid.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Axis {
    X,
    Y,
    Z,
}

impl Axis {
    pub const ALL: [Axis; 3] = [Axis::X, Axis::Y, Axis::Z];

    fn index(self) -> usize {
        self as usize
    }

    /// Splits a position into its coordinate along this axis and its two
    /// coordinates across it: (y, z) across X, (x, z) across Y, (x, y)
    /// across Z.
    fn split(self, position: IVec3) -> (i32, IVec2) {
        match self {
            Axis::X => (position.x, IVec2::new(position.y, position.z)),
            Axis::Y => (position.y, IVec2::new(position.x, position.z)),
            Axis::Z => (position.z, IVec2::new(position.x, position.y)),
        }
    }

    /// The position with coordinate `along` on this axis and `across` on the
    /// other two, as [`split`](Self::split) orders them.
    fn join(self, along: i32, across: IVec2) -> IVec3 {
        match self {
            Axis::X => IVec3::new(along, across.x, across.y),
            Axis::Y => IVec3::new(across.x, along, across.y),
            Axis::Z => IVec3::new(across.x, across.y, along),
        }
    }
}

/// One of the six directions along a sub-grid's axes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Direction {
    pub axis: Axis,
    /// Whether it points toward larger coordinates on `axis`.
    pub positive: bool,
}

impl Direction {
    pub const ALL: [Direction; 6] = [
        Direction::new(Axis::X, false),
        Direction::new(Axis::X, true),
        Direction::new(Axis::Y, false),
        Direction::new(Axis::Y, true),
        Direction::new(Axis::Z, false),
        Direction::new(Axis::Z, true),
    ];

    pub const fn new(axis: Axis, positive: bool) -> Self {
        Self { axis, positive }
    }

    fn bit(self) -> u8 {
        1 << (self.axis as u8 * 2 + self.positive as u8)
    }
}

/// One bit per line of a chunk along one axis: bit `v` of word `u`, for the
/// line at (u, v) across that axis.
type Mask = [u32; CHUNK];

/// Which lines of one chunk hold a solid, along each of the three axes.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct ChunkLines {
    masks: [Mask; 3],
}

impl ChunkLines {
    /// The lines of a chunk whose voxels at chunk-local positions are solid
    /// where `solid` says so. Reads every voxel of the chunk.
    #[cfg(test)]
    pub fn of(solid: impl Fn(UVec3) -> bool) -> Self {
        let mut lines = Self::default();
        for x in 0..CHUNK_SIZE_I {
            for y in 0..CHUNK_SIZE_I {
                for z in 0..CHUNK_SIZE_I {
                    let voxel = IVec3::new(x, y, z);
                    if solid(voxel.as_uvec3()) {
                        for axis in Axis::ALL {
                            lines.set(axis, axis.split(voxel).1.as_uvec2(), true);
                        }
                    }
                }
            }
        }
        lines
    }

    /// The lines of a chunk whose solid voxels are at the chunk-local
    /// positions `solids`. Its cost grows with the solids.
    pub fn of_solids(solids: impl IntoIterator<Item = UVec3>) -> Self {
        let mut lines = Self::default();
        for voxel in solids {
            for axis in Axis::ALL {
                lines.set(axis, axis.split(voxel.as_ivec3()).1.as_uvec2(), true);
            }
        }
        lines
    }

    /// Whether the line along `axis` at chunk-local `line` across it holds a
    /// solid.
    pub fn holds(&self, axis: Axis, line: UVec2) -> bool {
        self.masks[axis.index()][line.x as usize] & (1 << line.y) != 0
    }

    pub fn is_empty(&self) -> bool {
        self.masks.iter().flatten().all(|word| *word == 0)
    }

    fn set(&mut self, axis: Axis, line: UVec2, holds: bool) {
        let word = &mut self.masks[axis.index()][line.x as usize];
        let bit = 1 << line.y;
        if holds {
            *word |= bit;
        } else {
            *word &= !bit;
        }
    }
}

/// Which of a voxel's six lines leave the sub-grid without meeting a solid.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct OpenLines(u8);

impl OpenLines {
    #[cfg(test)]
    pub fn is_open(self, direction: Direction) -> bool {
        self.0 & direction.bit() != 0
    }

    /// Whether any line is open, which makes the voxel direct sky.
    pub fn any(self) -> bool {
        self.0 != 0
    }
}

/// A span of voxels on one chunk-local line whose line test in `direction`
/// flipped.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Flip {
    /// The level-0 chunk position of the chunk the voxels are in.
    pub chunk: IVec3,
    pub direction: Direction,
    /// The chunk-local line, across `direction`'s axis.
    pub line: UVec2,
    /// The chunk-local coordinates of the voxels along the axis.
    pub span: Range<i32>,
}

impl Flip {
    /// The grid positions of the voxels whose test flipped.
    pub fn voxels(&self) -> impl Iterator<Item = IVec3> + '_ {
        let axis = self.direction.axis;
        let (chunk_along, chunk_across) = axis.split(self.chunk * CHUNK_SIZE_I);
        let across = chunk_across + self.line.as_ivec2();
        self.span
            .clone()
            .map(move |t| axis.join(chunk_along + t, across))
    }
}

/// The line record of one sub-grid.
#[derive(Clone, Debug, Default)]
pub(crate) struct LineRecord {
    /// The chunks holding a solid on some line, by level-0 chunk position.
    chunks: HashMap<IVec3, ChunkLines>,
    /// The smallest and largest positions in `chunks`, if any.
    bounds: Option<(IVec3, IVec3)>,
}

impl LineRecord {
    /// Sets the lines of the chunk at `chunk`, without working out which line
    /// tests flip. For filling the record of a sub-grid whose light hasn't
    /// been solved yet.
    pub fn insert(&mut self, chunk: IVec3, lines: ChunkLines) {
        if lines.is_empty() {
            if self.chunks.remove(&chunk).is_some() {
                self.rebuild_bounds();
            }
        } else {
            self.chunks.insert(chunk, lines);
            self.bounds = Some(match self.bounds {
                Some((min, max)) => (min.min(chunk), max.max(chunk)),
                None => (chunk, chunk),
            });
        }
    }

    /// The six line tests of the voxel at grid position `voxel`.
    ///
    /// `solid` says whether a voxel of the sub-grid is solid, and is asked only
    /// about the voxels of `voxel`'s own chunk. Past that chunk, each test
    /// reads the record along the row of chunks, where a missing chunk counts
    /// as empty. The cost grows with the sub-grid's extent in chunks.
    pub fn open_lines(&self, voxel: IVec3, solid: impl Fn(IVec3) -> bool) -> OpenLines {
        let mut open = OpenLines::default();
        for direction in Direction::ALL {
            if self.line_is_open(voxel, direction, &solid) {
                open.0 |= direction.bit();
            }
        }
        open
    }

    fn line_is_open(
        &self,
        voxel: IVec3,
        direction: Direction,
        solid: impl Fn(IVec3) -> bool,
    ) -> bool {
        let axis = direction.axis;
        let (along, across) = axis.split(voxel);
        let step = if direction.positive { 1 } else { -1 };
        let chunk_along = along >> CHUNK_SHIFT;
        let mut t = along + step;
        while t >> CHUNK_SHIFT == chunk_along {
            if solid(axis.join(t, across)) {
                return false;
            }
            t += step;
        }
        !self.row_holds_past(axis, across, chunk_along, direction.positive)
    }

    /// Which lines along `axis` hold a solid in some chunk of the row of
    /// chunks at chunk position `across` across the axis, past the chunk at
    /// `chunk_along` in the given direction. Returns them as one mask: bit
    /// `v` of word `u` for the chunk-local line at (u, v) across the axis.
    ///
    /// A chunk the record doesn't hold counts as empty. The cost grows with
    /// the sub-grid's extent in chunks along the axis.
    pub fn lines_past(
        &self,
        axis: Axis,
        across: IVec2,
        chunk_along: i32,
        positive: bool,
    ) -> [u32; CHUNK] {
        let mut mask = [0; CHUNK];
        let Some((min, max)) = self.bounds else {
            return mask;
        };
        let (along_min, along_max) = (axis.split(min).0, axis.split(max).0);
        let chunks = if positive {
            chunk_along + 1..=along_max
        } else {
            along_min..=chunk_along - 1
        };
        for chunk in chunks {
            if let Some(lines) = self.chunks.get(&axis.join(chunk, across)) {
                for (word, held) in mask.iter_mut().zip(&lines.masks[axis.index()]) {
                    *word |= held;
                }
            }
        }
        mask
    }

    /// Whether a chunk of the row along `axis` through `across`, past the one
    /// at `chunk_along` in the given direction, holds a solid on the row.
    fn row_holds_past(
        &self,
        axis: Axis,
        across: IVec2,
        chunk_along: i32,
        positive: bool,
    ) -> bool {
        let Some((min, max)) = self.bounds else {
            return false;
        };
        let (row, line) = Row { axis, across }.chunk_line();
        let (min, max) = (axis.split(min).0, axis.split(max).0);
        let holds = |c: i32| {
            self.chunks
                .get(&axis.join(c, row))
                .is_some_and(|lines| lines.holds(axis, line))
        };
        if positive {
            (chunk_along + 1..=max).any(holds)
        } else {
            (min..chunk_along).any(holds)
        }
    }

    /// Records an edit to the sub-grid's voxels, and returns the voxels whose
    /// line tests it flipped.
    ///
    /// `changes` gives each edited voxel's grid position and whether it was
    /// solid before the edit, and `solid` reads the sub-grid's voxels after
    /// it. The flips cover the rows through the edited voxels, the voxels of
    /// their own chunks included, within the chunks the record holds before
    /// or after the edit. A voxel outside those chunks has an open line on
    /// every axis it lies outside them on, so its flips change no direct sky
    /// and aren't reported. A span may hold edited voxels themselves.
    ///
    /// The cost grows with the edited voxels times the sub-grid's extent in
    /// chunks.
    pub fn edit(
        &mut self,
        changes: impl IntoIterator<Item = (IVec3, bool)>,
        solid: impl Fn(IVec3) -> bool,
    ) -> Vec<Flip> {
        let changed: HashMap<IVec3, bool> = changes
            .into_iter()
            .filter(|(voxel, was_solid)| *was_solid != solid(*voxel))
            .collect();
        let solid_before =
            |voxel: IVec3| changed.get(&voxel).copied().unwrap_or_else(|| solid(voxel));
        let rows: HashSet<Row> = changed
            .keys()
            .flat_map(|voxel| {
                Axis::ALL.map(|axis| Row {
                    axis,
                    across: axis.split(*voxel).1,
                })
            })
            .collect();
        let ends_before: Vec<_> = rows
            .iter()
            .map(|row| self.row_ends(*row, &solid_before))
            .collect();
        let bounds_before = self.bounds;

        let mut touched: HashMap<IVec3, ChunkLines> = HashMap::default();
        for voxel in changed.keys() {
            let chunk = chunk_containing(*voxel);
            let lines = touched
                .entry(chunk)
                .or_insert_with(|| self.chunks.get(&chunk).cloned().unwrap_or_default());
            for axis in Axis::ALL {
                let (along, across) = axis.split(*voxel);
                let start = along & !(CHUNK_SIZE_I - 1);
                let holds =
                    (start..start + CHUNK_SIZE_I).any(|t| solid(axis.join(t, across)));
                lines.set(axis, Row { axis, across }.chunk_line().1, holds);
            }
        }
        for (chunk, lines) in touched {
            self.insert(chunk, lines);
        }

        let Some((min, max)) = [bounds_before, self.bounds].into_iter().flatten().reduce(
            |(a_min, a_max), (b_min, b_max)| (a_min.min(b_min), a_max.max(b_max)),
        ) else {
            return Vec::new();
        };
        let mut flips = Vec::new();
        for (row, before) in rows.iter().zip(ends_before) {
            let after = self.row_ends(*row, &solid);
            let bounds = row.axis.split(min).0 * CHUNK_SIZE_I
                ..(row.axis.split(max).0 + 1) * CHUNK_SIZE_I;
            for positive in [false, true] {
                let span = flipped_span(before, after, positive, bounds.clone());
                row.push_flips(Direction::new(row.axis, positive), span, &mut flips);
            }
        }
        flips
    }

    /// The coordinates of the first and last solid on a row, reading the
    /// record for which chunks hold it and `solid` within those chunks.
    fn row_ends(&self, row: Row, solid: &impl Fn(IVec3) -> bool) -> Option<(i32, i32)> {
        let (min, max) = self.bounds?;
        let axis = row.axis;
        let (row_chunk, line) = row.chunk_line();
        let holds = |c: &i32| {
            self.chunks
                .get(&axis.join(*c, row_chunk))
                .is_some_and(|lines| lines.holds(axis, line))
        };
        let solid_at = |t: &i32| solid(axis.join(*t, row.across));
        let voxels = |c: i32| c * CHUNK_SIZE_I..(c + 1) * CHUNK_SIZE_I;
        let chunks = axis.split(min).0..=axis.split(max).0;
        let first = chunks
            .clone()
            .filter(holds)
            .find_map(|c| voxels(c).find(solid_at))?;
        let last = chunks
            .rev()
            .filter(holds)
            .find_map(|c| voxels(c).rev().find(solid_at))?;
        Some((first, last))
    }

    fn rebuild_bounds(&mut self) {
        self.bounds = self.chunks.keys().fold(None, |bounds, chunk| {
            Some(match bounds {
                Some((min, max)) => (chunk.min(min), chunk.max(max)),
                None => (*chunk, *chunk),
            })
        });
    }
}

/// A row of voxels along `axis`, at grid position `across` across it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Row {
    axis: Axis,
    across: IVec2,
}

impl Row {
    /// The chunk position across the axis of the chunks the row passes
    /// through, and the chunk-local line it runs along in each of them.
    fn chunk_line(self) -> (IVec2, UVec2) {
        (
            self.across >> CHUNK_SHIFT,
            (self.across & (CHUNK_SIZE_I - 1)).as_uvec2(),
        )
    }

    /// Pushes the voxels at coordinates `span` along the row as flips in
    /// `direction`, one per chunk they fall in.
    fn push_flips(self, direction: Direction, span: Range<i32>, flips: &mut Vec<Flip>) {
        let (row_chunk, line) = self.chunk_line();
        let mut start = span.start;
        while start < span.end {
            let chunk_along = start >> CHUNK_SHIFT;
            let origin = chunk_along * CHUNK_SIZE_I;
            let end = span.end.min(origin + CHUNK_SIZE_I);
            flips.push(Flip {
                chunk: self.axis.join(chunk_along, row_chunk),
                direction,
                line,
                span: start - origin..end - origin,
            });
            start = end;
        }
    }
}

/// The coordinates along a row whose line test in one direction flips when
/// the row's first and last solid move from `before` to `after`, within
/// `bounds`.
///
/// Toward larger coordinates a voxel's test is open past the row's last
/// solid, and toward smaller ones before its first. So the tests that flip
/// lie between the old and the new end, or from that end to the edge of
/// `bounds` when the row held no solid before or holds none after.
fn flipped_span(
    before: Option<(i32, i32)>,
    after: Option<(i32, i32)>,
    positive: bool,
    bounds: Range<i32>,
) -> Range<i32> {
    let end = |ends: Option<(i32, i32)>| {
        ends.map(|(first, last)| if positive { last } else { first })
    };
    let span = match (end(before), end(after)) {
        (Some(a), Some(b)) => a.min(b) + 1..a.max(b),
        (None, Some(only)) | (Some(only), None) if positive => bounds.start..only,
        (None, Some(only)) | (Some(only), None) => only + 1..bounds.end,
        (None, None) => return 0..0,
    };
    span.start.max(bounds.start)..span.end.min(bounds.end)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lines_held(lines: &ChunkLines, axis: Axis) -> HashSet<UVec2> {
        (0..32)
            .flat_map(|u| (0..32).map(move |v| UVec2::new(u, v)))
            .filter(|line| lines.holds(axis, *line))
            .collect()
    }

    /// The solid voxels of a sub-grid, by grid position.
    type Solids = HashSet<IVec3>;

    /// A record whose chunks hold the lines of `solids`.
    fn record_of(solids: &Solids) -> LineRecord {
        let chunks: HashSet<IVec3> = solids
            .iter()
            .map(|voxel| chunk_containing(*voxel))
            .collect();
        let mut record = LineRecord::default();
        for chunk in chunks {
            let origin = chunk * CHUNK_SIZE_I;
            let lines =
                ChunkLines::of(|voxel| solids.contains(&(origin + voxel.as_ivec3())));
            record.insert(chunk, lines);
        }
        record
    }

    fn open_lines(record: &LineRecord, solids: &Solids, voxel: IVec3) -> OpenLines {
        record.open_lines(voxel, |voxel| solids.contains(&voxel))
    }

    /// The walls, floor and roof of a box from `min` to `max`, inclusive.
    fn cabin(min: IVec3, max: IVec3) -> Solids {
        let mut solids = Solids::default();
        for x in min.x..=max.x {
            for y in min.y..=max.y {
                for z in min.z..=max.z {
                    let voxel = IVec3::new(x, y, z);
                    if voxel.cmpeq(min).any() || voxel.cmpeq(max).any() {
                        solids.insert(voxel);
                    }
                }
            }
        }
        solids
    }

    const CABIN_MIN: IVec3 = IVec3::splat(-40);
    const CABIN_MAX: IVec3 = IVec3::splat(70);

    fn open_directions(lines: OpenLines) -> Vec<Direction> {
        Direction::ALL
            .into_iter()
            .filter(|direction| lines.is_open(*direction))
            .collect()
    }

    #[test]
    fn inside_a_cabin_spanning_several_chunks_every_line_is_blocked() {
        let solids = cabin(CABIN_MIN, CABIN_MAX);
        let record = record_of(&solids);
        for voxel in [
            IVec3::new(-39, -39, -39),
            IVec3::ZERO,
            IVec3::new(31, 32, 33),
            IVec3::new(69, 5, -10),
        ] {
            assert_eq!(
                open_directions(open_lines(&record, &solids, voxel)),
                [],
                "{voxel}"
            );
        }
    }

    #[test]
    fn in_line_with_a_doorway_only_the_doorways_axis_is_open() {
        let mut solids = cabin(CABIN_MIN, CABIN_MAX);
        let doorway = [IVec3::new(70, 10, 20), IVec3::new(70, 11, 20)];
        for voxel in doorway {
            solids.remove(&voxel);
        }
        let record = record_of(&solids);

        assert_eq!(
            open_directions(open_lines(&record, &solids, IVec3::new(-20, 10, 20))),
            [Direction::new(Axis::X, true)]
        );
        assert_eq!(
            open_directions(open_lines(&record, &solids, IVec3::new(50, 11, 20))),
            [Direction::new(Axis::X, true)]
        );
        assert_eq!(
            open_directions(open_lines(&record, &solids, IVec3::new(50, 12, 20))),
            []
        );
    }

    #[test]
    fn a_solid_two_chunks_along_a_row_blocks_the_line_past_a_missing_chunk() {
        let solids = Solids::from([IVec3::new(70, 5, 5)]);
        let record = record_of(&solids);

        let lines = open_lines(&record, &solids, IVec3::new(5, 5, 5));
        assert!(!lines.is_open(Direction::new(Axis::X, true)));
        assert_eq!(open_directions(lines).len(), 5);
    }

    #[test]
    fn a_solid_two_chunks_along_a_row_blocks_the_line_past_an_empty_one() {
        // The chunk between holds a solid, but not on the row.
        let solids = Solids::from([IVec3::new(70, 5, 5), IVec3::new(40, 6, 5)]);
        let record = record_of(&solids);

        let lines = open_lines(&record, &solids, IVec3::new(5, 5, 5));
        assert!(!lines.is_open(Direction::new(Axis::X, true)));
        assert_eq!(open_directions(lines).len(), 5);
    }

    #[test]
    fn a_voxels_own_chunk_counts_only_the_solids_past_it() {
        let solids = Solids::from([IVec3::new(10, 5, 5)]);
        let record = record_of(&solids);

        let before = open_lines(&record, &solids, IVec3::new(9, 5, 5));
        let after = open_lines(&record, &solids, IVec3::new(11, 5, 5));
        assert!(!before.is_open(Direction::new(Axis::X, true)));
        assert!(before.is_open(Direction::new(Axis::X, false)));
        assert!(after.is_open(Direction::new(Axis::X, true)));
        assert!(!after.is_open(Direction::new(Axis::X, false)));
    }

    /// Makes the voxels at `voxels` solid or open, in `solids` and in the
    /// record, and returns each flipped voxel with the direction its test
    /// flipped in.
    fn edit(
        record: &mut LineRecord,
        solids: &mut Solids,
        voxels: &[IVec3],
        make_solid: bool,
    ) -> HashSet<(IVec3, Direction)> {
        let changes: Vec<_> = voxels
            .iter()
            .map(|voxel| (*voxel, solids.contains(voxel)))
            .collect();
        for voxel in voxels {
            if make_solid {
                solids.insert(*voxel);
            } else {
                solids.remove(voxel);
            }
        }
        flipped(&record.edit(changes, |voxel| solids.contains(&voxel)))
    }

    fn flipped(flips: &[Flip]) -> HashSet<(IVec3, Direction)> {
        flips
            .iter()
            .flat_map(|flip| flip.voxels().map(move |voxel| (voxel, flip.direction)))
            .collect()
    }

    /// The voxels from `from` to `to` inclusive on the row along X at y 5 and
    /// z 5, each with `direction`.
    fn on_row(from: i32, to: i32, direction: Direction) -> HashSet<(IVec3, Direction)> {
        (from..=to)
            .map(|x| (IVec3::new(x, 5, 5), direction))
            .collect()
    }

    /// The flips along X on the row at y 5 and z 5.
    fn along_row(flips: HashSet<(IVec3, Direction)>) -> HashSet<(IVec3, Direction)> {
        flips
            .into_iter()
            .filter(|(voxel, direction)| {
                direction.axis == Axis::X && voxel.y == 5 && voxel.z == 5
            })
            .collect()
    }

    const TOWARD_X: Direction = Direction::new(Axis::X, true);
    const AWAY_FROM_X: Direction = Direction::new(Axis::X, false);

    #[test]
    fn a_chunks_first_solid_on_a_line_flips_it_up_to_the_nearest_chunk_holding_the_line()
    {
        // A solid in chunk -2 holds the row, and one in chunk 1 another row.
        let mut solids = Solids::from([IVec3::new(-50, 5, 5), IVec3::new(40, 6, 5)]);
        let mut record = record_of(&solids);

        let flips = along_row(edit(
            &mut record,
            &mut solids,
            &[IVec3::new(10, 5, 5)],
            true,
        ));

        // From past chunk -2's solid up to the new one, the line toward +X was
        // open and is now blocked. Toward -X, chunk -2 blocked it already.
        assert_eq!(flips, on_row(-49, 9, TOWARD_X));
    }

    #[test]
    fn a_solid_on_an_empty_row_flips_it_on_both_sides() {
        let mut solids = Solids::from([IVec3::new(-50, 6, 5), IVec3::new(80, 6, 5)]);
        let mut record = record_of(&solids);

        let flips = along_row(edit(
            &mut record,
            &mut solids,
            &[IVec3::new(10, 5, 5)],
            true,
        ));

        // The record's chunks run from -2 to 2 along X, so from voxel -64 to 95.
        let mut expected = on_row(-64, 9, TOWARD_X);
        expected.extend(on_row(11, 95, AWAY_FROM_X));
        assert_eq!(flips, expected);
    }

    #[test]
    fn clearing_a_chunks_last_solid_on_a_line_flips_it_back() {
        let mut solids = Solids::from([IVec3::new(-50, 5, 5), IVec3::new(40, 6, 5)]);
        let mut record = record_of(&solids);
        let voxel = IVec3::new(10, 5, 5);
        let set = edit(&mut record, &mut solids, &[voxel], true);

        let cleared = edit(&mut record, &mut solids, &[voxel], false);

        assert_eq!(cleared, set);
        assert_eq!(along_row(cleared), on_row(-49, 9, TOWARD_X));
    }

    #[test]
    fn a_solid_behind_nearer_blockers_flips_nothing() {
        let mut solids = cabin(CABIN_MIN, CABIN_MAX);
        let mut record = record_of(&solids);
        // Chunk 0 is all air inside the cabin, so this is its first solid on
        // each of the three lines through it.
        let voxel = IVec3::new(10, 10, 10);

        assert_eq!(
            edit(&mut record, &mut solids, &[voxel], true),
            HashSet::new()
        );
        assert_eq!(
            edit(&mut record, &mut solids, &[voxel], false),
            HashSet::new()
        );
    }

    /// The smallest and largest coordinate of a solid on each row that holds
    /// one, by axis and position across it.
    fn row_ends(solids: &Solids) -> HashMap<(Axis, IVec2), (i32, i32)> {
        let mut ends: HashMap<(Axis, IVec2), (i32, i32)> = HashMap::default();
        for voxel in solids {
            for axis in Axis::ALL {
                let (along, across) = axis.split(*voxel);
                let end = ends.entry((axis, across)).or_insert((along, along));
                *end = (end.0.min(along), end.1.max(along));
            }
        }
        ends
    }

    /// A voxel's line test worked out from the ends of the rows: open when no
    /// solid lies past the voxel on its row.
    fn line_test(
        ends: &HashMap<(Axis, IVec2), (i32, i32)>,
        voxel: IVec3,
        direction: Direction,
    ) -> bool {
        let (along, across) = direction.axis.split(voxel);
        match ends.get(&(direction.axis, across)) {
            None => true,
            Some((_, last)) if direction.positive => along > *last,
            Some((first, _)) => along < *first,
        }
    }

    /// The smallest and largest chunk positions holding one of the solids.
    fn chunk_bounds<'a>(
        solids: impl Iterator<Item = &'a IVec3>,
    ) -> Option<(IVec3, IVec3)> {
        solids
            .map(|voxel| chunk_containing(*voxel))
            .fold(None, |bounds, chunk| {
                Some(match bounds {
                    Some((min, max)) => (chunk.min(min), chunk.max(max)),
                    None => (chunk, chunk),
                })
            })
    }

    #[test]
    fn flips_from_random_edits_agree_with_every_line_test_worked_out_again() {
        use rand::{rngs::StdRng, RngExt, SeedableRng};

        let mut rng = StdRng::seed_from_u64(21);
        let mut solids = Solids::default();
        let mut record = LineRecord::default();
        // Voxels from -3 to 2 on each axis, so in chunks -1 and 0 on each.
        let random_voxel = |rng: &mut StdRng| {
            IVec3::new(
                rng.random_range(-3..3),
                rng.random_range(-3..3),
                rng.random_range(-3..3),
            )
        };
        for _ in 0..300 {
            let edited: Vec<IVec3> = (0..rng.random_range(1..4))
                .map(|_| random_voxel(&mut rng))
                .collect();
            let before = solids.clone();
            let flips = edit(&mut record, &mut solids, &edited, rng.random_bool(0.6));
            let (ends_before, ends_after) = (row_ends(&before), row_ends(&solids));

            // A voxel outside the chunks holding a solid always has an open
            // line, so its flips aren't reported. Only the rows holding a
            // solid can flip, so every other test stays as it was.
            let Some((low, high)) = chunk_bounds(before.iter().chain(&solids)) else {
                assert!(flips.is_empty());
                continue;
            };
            let checked = |voxel: &IVec3| {
                let chunk = chunk_containing(*voxel);
                chunk.cmpge(low).all()
                    && chunk.cmple(high).all()
                    && !before.contains(voxel)
                    && !solids.contains(voxel)
            };
            let flipped = |voxel: IVec3, direction: Direction| {
                line_test(&ends_before, voxel, direction)
                    != line_test(&ends_after, voxel, direction)
            };
            for (axis, across) in ends_before.keys().chain(ends_after.keys()) {
                let (from, to) = (axis.split(low).0, axis.split(high).0 + 1);
                for along in from * CHUNK_SIZE_I..to * CHUNK_SIZE_I {
                    let voxel = axis.join(along, *across);
                    for positive in [false, true] {
                        let direction = Direction::new(*axis, positive);
                        if checked(&voxel) && flipped(voxel, direction) {
                            assert!(
                                flips.contains(&(voxel, direction)),
                                "{voxel} {direction:?} not reported after editing {edited:?}"
                            );
                        }
                    }
                }
            }
            for (voxel, direction) in &flips {
                assert!(
                    !checked(voxel) || flipped(*voxel, *direction),
                    "{voxel} {direction:?} reported after editing {edited:?}"
                );
            }

            for voxel in [
                random_voxel(&mut rng),
                IVec3::new(-30, 1, 2),
                IVec3::new(20, -1, 31),
            ] {
                if !solids.contains(&voxel) {
                    let lines = open_lines(&record, &solids, voxel);
                    for direction in Direction::ALL {
                        assert_eq!(
                            lines.is_open(direction),
                            line_test(&ends_after, voxel, direction),
                            "{voxel} {direction:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn a_chunks_masks_hold_exactly_the_lines_through_its_solids() {
        let solids = HashSet::from([
            UVec3::new(1, 2, 3),
            UVec3::new(1, 2, 7),
            UVec3::new(31, 0, 0),
        ]);
        let lines = ChunkLines::of(|voxel| solids.contains(&voxel));

        // A line along X is named by its (y, z), along Y by (x, z), along Z by (x, y).
        let uv = |u, v| UVec2::new(u, v);
        assert_eq!(
            lines_held(&lines, Axis::X),
            HashSet::from([uv(2, 3), uv(2, 7), uv(0, 0)])
        );
        assert_eq!(
            lines_held(&lines, Axis::Y),
            HashSet::from([uv(1, 3), uv(1, 7), uv(31, 0)])
        );
        assert_eq!(
            lines_held(&lines, Axis::Z),
            HashSet::from([uv(1, 2), uv(31, 0)])
        );
    }

    #[test]
    fn an_empty_chunk_holds_no_line() {
        let lines = ChunkLines::of(|_| false);
        for axis in Axis::ALL {
            assert!(lines_held(&lines, axis).is_empty());
        }
    }
}
