//! Enclosure: which voxels of a sub-grid have direct sky, under
//! [`SubGridSky::Enclosed`]. Compiled only with the `sub_grids` feature.
//!
//! An open voxel of a sub-grid has direct sky when any of the six lines
//! along the sub-grid's axes from it leaves the sub-grid without meeting one
//! of its solid voxels. A sub-grid chunk's solve gives those voxels of its
//! box the top sky level and spreads sky from them as a world solve spreads
//! direct sky, with no sky fill. Lines run along the sub-grid's own axes, so
//! which way up it is changes nothing.
//!
//! Lines reach past a solve's box, so a solve reads the voxels of its box,
//! where a chunk position the sub-grid has no chunk at is air, and then the
//! sub-grid's [line record](super::line_record) for the chunks further along
//! each row. Each sub-grid has one record in the
//! [world view](crate::light::view), which solves read a snapshot of. The commit
//! keeps it current from the sub-grid's voxel edits, and relights the chunks
//! whose solve box holds a voxel whose line test the edit flipped. Under
//! [`SubGridSky::World`] there are no records, and every open voxel of a
//! sub-grid has direct sky.
//!
//! A record lasts as long as its sub-grid: it goes when the sub-grid ends,
//! and a later sub-grid given the same id starts with none.

use std::sync::Arc;

use bevy::platform::collections::{HashMap, HashSet};
use bevy::prelude::*;

use super::line_record::{Axis, ChunkLines, Flip, LineRecord};
use crate::chunk::{chunk_containing, CHUNK_SIZE_I, CHUNK_SIZE_U};
use crate::configuration::VoxelWorldConfig;
use crate::grid::{GridId, GridKey};
use crate::light::config::SubGridSky;
use crate::light::inputs::LightInputs;
use crate::light::kernel::Plane;
use crate::light::solver::{DirectCells, Neighbourhood};
use crate::light::view::{LightView, ViewMaps};
use crate::light::LightSettings;
use crate::sub_grid::SubGrids;
use crate::voxel_edits::{ChangeSet, CommittedEdits, VoxelEdits};

const CHUNK: usize = CHUNK_SIZE_U as usize;

/// The voxels along one axis of the 3×3×3 chunks around a chunk.
const SIDE: usize = 3 * CHUNK;

/// A sub-grid's line record in the world view, with the sub-grid entity it
/// belongs to.
pub(crate) struct GridLines {
    pub owner: Entity,
    pub record: Arc<LineRecord>,
}

/// The neighbour index of the chunk at offset (`x`, `y`, `z`), each from 0 to
/// 2, in the 3×3×3 around a chunk.
fn index(x: usize, y: usize, z: usize) -> usize {
    x + y * 3 + z * 9
}

/// The direct sky of the open voxels of the 3×3×3 chunks around the sub-grid
/// chunk at `position`, a plane per chunk position of `neighbours`. `own` is
/// the chunk's inputs, and `record` its grid's line record.
///
/// A position of `neighbours` that holds nothing reads as air, as the
/// sub-grid's solve reads it. The voxels of the 3×3×3 come from `neighbours`,
/// and the chunks past them along each row from `record`. Its cost grows
/// with the voxels of the 3×3×3 and with the sub-grid's extent in chunks.
pub(crate) fn direct_cells(
    neighbours: &Neighbourhood,
    own: &LightInputs,
    position: IVec3,
    record: &LineRecord,
) -> Box<DirectCells> {
    let centre = index(1, 1, 1);
    let solid = |at: usize, x: usize, y: usize| -> u32 {
        if at == centre {
            return own.occupancy.solid_column(x, y);
        }
        neighbours[at]
            .as_ref()
            .map_or(0, |inputs| inputs.occupancy.solid_column(x, y))
    };
    let mut planes: [Box<Plane>; 27] =
        std::array::from_fn(|_| Box::new([0; CHUNK * CHUNK]));
    let chunk_at = |offset: usize, along: i32| along + offset as i32 - 1;

    // Along z, each column of a chunk is one word, bit z for its voxel z.
    for oy in 0..3 {
        for ox in 0..3 {
            let across = IVec2::new(chunk_at(ox, position.x), chunk_at(oy, position.y));
            let past_up = record.lines_past(Axis::Z, across, position.z + 1, true);
            let past_down = record.lines_past(Axis::Z, across, position.z - 1, false);
            for y in 0..CHUNK {
                for x in 0..CHUNK {
                    let words: [u32; 3] =
                        std::array::from_fn(|oz| solid(index(ox, oy, oz), x, y));
                    // The record's lines across z are at (x, y): bit y of
                    // word x.
                    let blocked_up = past_up[x] >> y & 1 != 0;
                    let blocked_down = past_down[x] >> y & 1 != 0;
                    for oz in 0..3 {
                        let mut open = 0;
                        if !blocked_up && words[oz + 1..].iter().all(|word| *word == 0) {
                            open |= above_last(words[oz]);
                        }
                        if !blocked_down && words[..oz].iter().all(|word| *word == 0) {
                            open |= below_first(words[oz]);
                        }
                        planes[index(ox, oy, oz)][x + y * CHUNK] |= open;
                    }
                }
            }
        }
    }

    // Along x and y, a row is swept a voxel at a time, carrying which of a
    // column's 32 voxels along z have met a solid. The record's lines across
    // x are at (y, z), and across y at (x, z): in both, word u holds bit z.
    for oz in 0..3 {
        for oy in 0..3 {
            let across = IVec2::new(chunk_at(oy, position.y), chunk_at(oz, position.z));
            let past_up = record.lines_past(Axis::X, across, position.x + 1, true);
            let past_down = record.lines_past(Axis::X, across, position.x - 1, false);
            for y in 0..CHUNK {
                let at = |along: usize| (index(along / CHUNK, oy, oz), along % CHUNK);
                sweep(&mut planes, past_up[y], (0..SIDE).rev(), |along| {
                    let (chunk, x) = at(along);
                    (chunk, x + y * CHUNK, solid(chunk, x, y))
                });
                sweep(&mut planes, past_down[y], 0..SIDE, |along| {
                    let (chunk, x) = at(along);
                    (chunk, x + y * CHUNK, solid(chunk, x, y))
                });
            }
        }
        for ox in 0..3 {
            let across = IVec2::new(chunk_at(ox, position.x), chunk_at(oz, position.z));
            let past_up = record.lines_past(Axis::Y, across, position.y + 1, true);
            let past_down = record.lines_past(Axis::Y, across, position.y - 1, false);
            for x in 0..CHUNK {
                let at = |along: usize| (index(ox, along / CHUNK, oz), along % CHUNK);
                sweep(&mut planes, past_up[x], (0..SIDE).rev(), |along| {
                    let (chunk, y) = at(along);
                    (chunk, x + y * CHUNK, solid(chunk, x, y))
                });
                sweep(&mut planes, past_down[x], 0..SIDE, |along| {
                    let (chunk, y) = at(along);
                    (chunk, x + y * CHUNK, solid(chunk, x, y))
                });
            }
        }
    }

    let mut cells: Box<DirectCells> = Box::new(std::array::from_fn(|_| None));
    for (at, mut plane) in planes.into_iter().enumerate() {
        for (column, direct) in plane.iter_mut().enumerate() {
            *direct &= !solid(at, column % CHUNK, column / CHUNK);
        }
        if plane.iter().any(|column| *column != 0) {
            cells[at] = Some(plane);
        }
    }
    cells
}

/// Sweeps one row of columns, in the order `along` gives, marking each
/// column's voxels whose line onward from them, back toward where the sweep
/// started, meets no solid. `blocked` holds the voxels whose line meets a
/// solid past the row's start. `column` gives, for a place along the row,
/// the chunk's neighbour index, the column's index in its plane and its
/// solid voxels.
fn sweep(
    planes: &mut [Box<Plane>; 27],
    blocked: u32,
    along: impl Iterator<Item = usize>,
    column: impl Fn(usize) -> (usize, usize, u32),
) {
    let mut blocked = blocked;
    for place in along {
        let (chunk, at, solid) = column(place);
        planes[chunk][at] |= !blocked;
        blocked |= solid;
    }
}

/// The bits of a word above its highest set bit, or every bit of an empty
/// word.
fn above_last(word: u32) -> u32 {
    match word {
        0 => !0,
        _ => (!0u32).checked_shl(32 - word.leading_zeros()).unwrap_or(0),
    }
}

/// The bits of a word below its lowest set bit, or every bit of an empty
/// word.
fn below_first(word: u32) -> u32 {
    match word {
        0 => !0,
        _ => (1u32 << word.trailing_zeros()) - 1,
    }
}

/// The sub-grids' entities by grid, for those alive.
fn owners<C: VoxelWorldConfig>(sub_grids: &SubGrids<C>) -> HashMap<GridId, Entity> {
    sub_grids
        .live_grids()
        .into_iter()
        .map(|(entity, grid, _)| (grid, entity))
        .collect()
}

/// Drops the line record of each sub-grid that has ended. Its cost grows with
/// the live sub-grids, and it does nothing while no record is kept.
pub(crate) fn forget_ended<C: VoxelWorldConfig>(
    view: &LightView,
    sub_grids: &SubGrids<C>,
) {
    if view.read().lines.as_ref().is_none_or(HashMap::is_empty) {
        return;
    }
    let owners = owners(sub_grids);
    let mut view = view.write();
    if let Some(lines) = view.lines.as_mut() {
        lines.retain(|grid, lines| owners.get(grid) == Some(&lines.owner));
    }
}

/// Takes the sub-grid change sets `sets` of a commit into the line records
/// of `view`, before the commit's inputs replace those in `view`. Returns,
/// by grid, the voxels whose line tests flipped. Does nothing under
/// [`SubGridSky::World`].
///
/// A sub-grid with no record yet gets one from its edits, with no flips: its
/// chunks are all new, and solve anyway. Its cost grows with the edited
/// voxels times the sub-grid's extent in chunks, and with the voxels of
/// each edited chunk.
pub(crate) fn follow_edits<'a, C: VoxelWorldConfig>(
    view: &mut ViewMaps,
    sets: impl Iterator<Item = &'a ChangeSet<C::MaterialIndex>>,
    edits: &VoxelEdits<C, C::MaterialIndex>,
    sub_grids: &SubGrids<C>,
) -> Vec<(GridId, Flip)> {
    if view.lines.is_none() {
        return Vec::new();
    }
    let mut by_grid: HashMap<GridId, Vec<&ChangeSet<C::MaterialIndex>>> = HashMap::new();
    for set in sets {
        by_grid.entry(set.grid).or_default().push(set);
    }
    if by_grid.is_empty() {
        return Vec::new();
    }
    let owners = owners(sub_grids);
    let committed = edits.committed();
    let mut flips = Vec::new();
    for (grid, sets) in by_grid {
        let Some(owner) = owners.get(&grid).copied() else {
            continue;
        };
        let chunks: HashSet<IVec3> = sets.iter().map(|set| set.chunk).collect();
        let kept = view
            .lines
            .as_ref()
            .and_then(|lines| lines.get(&grid))
            .is_some_and(|lines| lines.owner == owner);
        if kept {
            // A removed edit was a solid, since a sub-grid's edits are its
            // solids. A new one was solid if the inputs before the commit
            // say so.
            let changes: Vec<(IVec3, bool)> = sets
                .iter()
                .flat_map(|set| set.voxels.iter())
                .map(|(p, voxel)| {
                    let key = GridKey::level_0(grid, chunk_containing(*p));
                    let was_solid = match view.chunks.get(&key) {
                        Some(chunk) => {
                            let local = (*p - key.position() * CHUNK_SIZE_I).as_uvec3();
                            chunk.light.inputs.occupancy.is_solid(
                                local.x as usize,
                                local.y as usize,
                                local.z as usize,
                            )
                        }
                        None => voxel.is_none(),
                    };
                    (*p, was_solid)
                })
                .collect();
            let lines = view.lines.as_mut().unwrap().get_mut(&grid).unwrap();
            let record = Arc::make_mut(&mut lines.record);
            let solid = |p: IVec3| {
                committed
                    .get_in(grid, p)
                    .is_some_and(|voxel| voxel.is_solid())
            };
            flips.extend(
                record
                    .edit(changes, solid)
                    .into_iter()
                    .map(|flip| (grid, flip)),
            );
            // The masks of the edited chunks, read from the edits, whatever
            // the inputs before said.
            for chunk in chunks {
                record.insert(
                    chunk,
                    chunk_lines(&committed, GridKey::level_0(grid, chunk)),
                );
            }
        } else {
            let mut record = LineRecord::default();
            for chunk in chunks {
                record.insert(
                    chunk,
                    chunk_lines(&committed, GridKey::level_0(grid, chunk)),
                );
            }
            view.lines.as_mut().unwrap().insert(
                grid,
                GridLines {
                    owner,
                    record: Arc::new(record),
                },
            );
        }
    }
    flips
}

/// The lines of the sub-grid chunk at `key` that hold a solid, from its
/// committed edits.
fn chunk_lines<I: Copy + PartialEq>(
    committed: &CommittedEdits<'_, I>,
    key: GridKey,
) -> ChunkLines {
    let Some(bucket) = committed.bucket_in(key) else {
        return ChunkLines::default();
    };
    let min = key.position() * CHUNK_SIZE_I;
    ChunkLines::of_solids(
        bucket
            .iter()
            .filter(|(_, voxel)| voxel.is_solid())
            .map(|(p, _)| (p - min).as_uvec3()),
    )
}

/// Switches world `view` to `sky`: under [`SubGridSky::Enclosed`] it builds
/// every live sub-grid's line record from its voxel edits, and under
/// [`SubGridSky::World`] it drops them. Returns the sub-grid chunks to
/// relight: those with an open voxel, whose sky changes.
///
/// Building grows with the sub-grids' voxel edits.
pub(crate) fn switch<C: VoxelWorldConfig>(
    view: &LightView,
    sky: SubGridSky,
    edits: &VoxelEdits<C, C::MaterialIndex>,
    sub_grids: Option<&SubGrids<C>>,
) -> Vec<GridKey> {
    let lines = match sky {
        SubGridSky::World => None,
        SubGridSky::Enclosed => {
            let mut lines = HashMap::new();
            for (owner, grid, _) in
                sub_grids.map(SubGrids::live_grids).unwrap_or_default()
            {
                let chunks: Vec<IVec3> = edits
                    .edited_chunks_in(grid)
                    .into_iter()
                    .map(|(chunk, _)| chunk)
                    .collect();
                let committed = edits.committed();
                let mut record = LineRecord::default();
                for chunk in chunks {
                    record.insert(
                        chunk,
                        chunk_lines(&committed, GridKey::level_0(grid, chunk)),
                    );
                }
                lines.insert(
                    grid,
                    GridLines {
                        owner,
                        record: Arc::new(record),
                    },
                );
            }
            Some(lines)
        }
    };
    let mut view = view.write();
    view.lines = lines;
    view.chunks
        .iter()
        .filter(|(key, chunk)| {
            !key.is_world() && chunk.light.inputs.occupancy.has_open_voxel()
        })
        .map(|(key, _)| *key)
        .collect()
}

impl<C: VoxelWorldConfig> LightSettings<C> {
    /// Whether enclosure closes in the voxel at `position` of the sub-grid
    /// `grid`, in a chunk position where the sub-grid has no chunk: whether,
    /// under [`SubGridSky::Enclosed`], every line along the sub-grid's axes
    /// from it meets one of its voxels. Such a voxel holds no light of its
    /// own, and sky spreading from an opening within sky reach of it is
    /// left out, so it reads darker than the truth there, never brighter.
    pub fn closed_in_at(&self, grid: GridId, position: IVec3) -> bool {
        let view = self.view.read();
        let Some(lines) = view.lines.as_ref() else {
            return false;
        };
        // A sub-grid with no record has no voxels to close anything in.
        lines
            .get(&grid)
            .is_some_and(|lines| !lines.record.open_lines(position, |_| false).any())
    }
}

#[cfg(test)]
mod tests {
    use super::super::line_record::OpenLines;
    use super::*;
    use crate::light::inputs::Occupancy;
    use crate::light::sky::ColumnAnswer;
    use crate::light::solver::neighbour_index;

    /// The inputs of a chunk whose solid voxels, by chunk-local position, are
    /// where `solid` says.
    fn inputs(solid: impl Fn(IVec3) -> bool) -> Arc<LightInputs> {
        let mut occupancy = Occupancy::Open;
        for z in 0..CHUNK {
            for y in 0..CHUNK {
                for x in 0..CHUNK {
                    if solid(IVec3::new(x as i32, y as i32, z as i32)) {
                        occupancy.set(x, y, z, true);
                    }
                }
            }
        }
        occupancy.normalize();
        Arc::new(LightInputs {
            occupancy,
            emissive: Vec::new(),
            columns: ColumnAnswer::none(),
            ring: false,
        })
    }

    #[test]
    fn direct_cells_are_the_open_voxels_with_an_open_line_test() {
        // Hollow shells, slabs and posts scattered over chunks -3 to 3 along
        // x and -1 to 1 across, so lines leave the 3×3×3 around the origin
        // and meet solids in the record past it, and some chunk positions
        // hold nothing.
        let solid = |p: IVec3| {
            let shell = (p.x.rem_euclid(48) == 5 || p.x.rem_euclid(48) == 40)
                && (-20..30).contains(&p.y)
                && (-12..44).contains(&p.z);
            let slab = p.y == 12 && (p.x.rem_euclid(70) < 20) && (0..20).contains(&p.z);
            let post = p.x.rem_euclid(9) == 0 && p.z.rem_euclid(13) == 0 && p.y > -30;
            let in_bounds = (-96..128).contains(&p.x)
                && (-32..64).contains(&p.y)
                && (-32..64).contains(&p.z);
            let hole = (-40..-34).contains(&p.y);
            in_bounds && !hole && (shell || slab || post)
        };
        let chunk_solid = |chunk: IVec3| {
            let min = chunk * CHUNK_SIZE_I;
            move |local: IVec3| solid(min + local)
        };
        let missing = [IVec3::new(1, 1, -1), IVec3::new(-1, -1, 1)];
        let mut record = LineRecord::default();
        for x in -3..=3 {
            for y in -1..=1 {
                for z in -1..=1 {
                    let chunk = IVec3::new(x, y, z);
                    if missing.contains(&chunk) {
                        continue;
                    }
                    let solid = chunk_solid(chunk);
                    record.insert(chunk, ChunkLines::of(|local| solid(local.as_ivec3())));
                }
            }
        }
        let solid_in_record =
            |p: IVec3| !missing.contains(&chunk_containing(p)) && solid(p);
        let neighbours: Neighbourhood = std::array::from_fn(|at| {
            let at = at as i32;
            let offset = IVec3::new(at % 3, at / 3 % 3, at / 9) - 1;
            if offset == IVec3::ZERO || missing.contains(&offset) {
                return None;
            }
            Some(inputs(chunk_solid(offset)))
        });
        let own = inputs(chunk_solid(IVec3::ZERO));

        let cells = direct_cells(&neighbours, &own, IVec3::ZERO, &record);
        let mut direct = 0;
        for offset in (0..27).map(|at| IVec3::new(at % 3, at / 3 % 3, at / 9) - 1) {
            let plane = cells[neighbour_index(offset)].as_deref();
            for z in 0..CHUNK_SIZE_I {
                for y in 0..CHUNK_SIZE_I {
                    for x in 0..CHUNK_SIZE_I {
                        let p = offset * CHUNK_SIZE_I + IVec3::new(x, y, z);
                        let lines: OpenLines = record.open_lines(p, solid_in_record);
                        let expected = !solid_in_record(p) && lines.any();
                        let got = plane.is_some_and(|plane| {
                            plane[(x + y * CHUNK_SIZE_I) as usize] >> z & 1 == 1
                        });
                        assert_eq!(got, expected, "at {p}");
                        direct += expected as u32;
                    }
                }
            }
        }
        assert!(direct > 0);
    }
}
