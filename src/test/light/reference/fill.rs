//! The reference for sky fill under either model, worked out from the agreed
//! rules by brute force, sharing no code with the crate's fields or seeding.
//!
//! The positions given as known are cut into cells of 8×8×8 voxels, each
//! open when it holds no solid voxel and direct sky when its voxels all have
//! direct sky. Each model gives each open cell a level, in sixteenths of a
//! level:
//!
//! - Paths: a position's open cells joined by faces make its regions. Each
//!   region's level is the most of what the sky gives it through its direct
//!   top cells and what each region beside it passes through the cells they
//!   share, worked out by repeating until nothing changes.
//! - Openness: each open cell that isn't direct sky casts rays from its
//!   middle, cell by cell, and its openness is the share of them that enter
//!   a direct cell before a closed or unknown one, within reach. A cell
//!   seeing at least `sun_at` counts as direct sky, and its voxels are
//!   sources of direct sky in the chunks that take fill.
//!
//! Every chunk then reads its 27 positions onto a grid of cells, blurs it,
//! seeds the lit cells' voxels, runs the seeds 7 voxels through open voxels
//! and floods the box around the chunk from them and from direct sky.

use bevy::platform::collections::{HashMap, HashSet};
use bevy::prelude::*;

use super::Reference;
use crate::prelude::{LightShape, OpenFill, PathFill, Sky, SkyFill, WorldVoxel};

const STEPS: [IVec3; 6] = [
    IVec3::NEG_X,
    IVec3::X,
    IVec3::NEG_Y,
    IVec3::Y,
    IVec3::NEG_Z,
    IVec3::Z,
];

struct Position {
    open: [bool; 64],
    direct: [bool; 64],
    /// Each region's cells.
    regions: Vec<Vec<IVec3>>,
}

/// What the blur and the seeds read of an open cell.
#[derive(Clone, Copy)]
struct Cell {
    /// The level the blur starts from, in sixteenths.
    value: i32,
    /// Whether its voxels seed fill.
    seeds: bool,
    /// Whether its voxels are sources of direct sky, though they may not
    /// have direct sky of their own.
    sunlit: bool,
}

/// By known position, its open cells.
type Cells = HashMap<IVec3, [Option<Cell>; 64]>;

fn cell_index(cell: IVec3) -> usize {
    (cell.x + cell.y * 4 + cell.z * 16) as usize
}

fn cells() -> impl Iterator<Item = IVec3> {
    (0..4)
        .flat_map(|z| (0..4).flat_map(move |y| (0..4).map(move |x| IVec3::new(x, y, z))))
}

/// Whether the column at `x`, `z` holds no solid from above the chunk at
/// height `chunk_y` up to the top of what the sky rule `sky` says must be
/// clear.
fn column_clear<I: Copy + PartialEq>(
    sky: &Sky,
    chunk_y: i32,
    x: i32,
    z: i32,
    voxel_at: &impl Fn(IVec3) -> WorldVoxel<I>,
) -> bool {
    let top = match sky {
        Sky::Clearance { chunks } => (chunk_y + 1 + *chunks as i32) * 32,
        Sky::Ceiling { y } => (chunk_y.max(y.div_euclid(32)) + 1) * 32,
        Sky::Off => return false,
    };
    ((chunk_y + 1) * 32..top).all(|y| !voxel_at(IVec3::new(x, y, z)).is_solid())
}

fn position<I: Copy + PartialEq>(
    at: IVec3,
    sky: &Sky,
    voxel_at: &impl Fn(IVec3) -> WorldVoxel<I>,
) -> Position {
    let min = at * 32;
    let mut open = [false; 64];
    for cell in cells() {
        open[cell_index(cell)] = (0..8).all(|z| {
            (0..8).all(|y| {
                (0..8).all(|x| !voxel_at(min + cell * 8 + IVec3::new(x, y, z)).is_solid())
            })
        });
    }
    let mut direct = [false; 64];
    for cz in 0..4 {
        for cx in 0..4 {
            let sees = (0..8).all(|z| {
                (0..8).all(|x| {
                    column_clear(
                        sky,
                        at.y,
                        min.x + cx * 8 + x,
                        min.z + cz * 8 + z,
                        voxel_at,
                    )
                })
            });
            if !sees {
                continue;
            }
            for cy in (0..4).rev() {
                let index = cell_index(IVec3::new(cx, cy, cz));
                if !open[index] {
                    break;
                }
                direct[index] = true;
            }
        }
    }
    let mut regions = Vec::new();
    let mut seen = [false; 64];
    for start in cells() {
        if !open[cell_index(start)] || seen[cell_index(start)] {
            continue;
        }
        let mut region = vec![start];
        seen[cell_index(start)] = true;
        let mut next = 0;
        while next < region.len() {
            let cell = region[next];
            next += 1;
            for step in STEPS {
                let beside = cell + step;
                if beside.cmplt(IVec3::ZERO).any() || beside.cmpgt(IVec3::splat(3)).any()
                {
                    continue;
                }
                let index = cell_index(beside);
                if open[index] && !seen[index] {
                    seen[index] = true;
                    region.push(beside);
                }
            }
        }
        regions.push(region);
    }
    Position {
        open,
        direct,
        regions,
    }
}

/// The most cells in any one layer of `cells`, along any axis.
fn widest(cells: &[IVec3]) -> i32 {
    (0..3)
        .flat_map(|axis| (0..4).map(move |layer| (axis, layer)))
        .map(|(axis, layer)| {
            cells.iter().filter(|cell| cell[axis] == layer).count() as i32
        })
        .max()
        .unwrap_or(0)
}

/// The cells of the known `positions` under paths.
fn path_cells(positions: &HashMap<IVec3, Position>, fill: &PathFill) -> Cells {
    let loss = |f: f32| -> i32 {
        let levels = fill.whole_face_loss * (16.0 / f).powf(fill.opening_power);
        (levels * 16.0).min(240.0 * 8.0) as i32
    };
    let top = (fill.beside_open_sky * 16.0).round() as i32;

    // Each region's level, from the sky and then from its neighbours.
    let mut levels: HashMap<(IVec3, usize), i32> = HashMap::new();
    for (at, position) in positions {
        for (index, region) in position.regions.iter().enumerate() {
            let sky_cells = region
                .iter()
                .filter(|cell| cell.y == 3 && position.direct[cell_index(**cell)])
                .count() as i32;
            let level = if sky_cells == 0 {
                0
            } else {
                (top + 2 * loss(16.0) - loss(sky_cells as f32)).clamp(0, 240)
            };
            levels.insert((*at, index), level);
        }
    }
    loop {
        let mut changed = false;
        for (at, position) in positions {
            for (index, region) in position.regions.iter().enumerate() {
                let mut best = levels[&(*at, index)];
                for step in STEPS {
                    let Some(beside) = positions.get(&(*at + step)) else {
                        continue;
                    };
                    for (other_index, other) in beside.regions.iter().enumerate() {
                        let aperture = region
                            .iter()
                            .filter(|cell| {
                                let mirror = (**cell + step).rem_euclid(IVec3::splat(4));
                                let on_face = (**cell + step).cmplt(IVec3::ZERO).any()
                                    || (**cell + step).cmpgt(IVec3::splat(3)).any();
                                on_face && other.contains(&mirror)
                            })
                            .count() as i32;
                        if aperture == 0 {
                            continue;
                        }
                        let f = aperture as f32 * 16.0 / widest(region) as f32;
                        best = best.max(levels[&(*at + step, other_index)] - loss(f));
                    }
                }
                if best > levels[&(*at, index)] {
                    levels.insert((*at, index), best);
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }

    positions
        .iter()
        .map(|(at, position)| {
            let mut cells = [None; 64];
            for (index, region) in position.regions.iter().enumerate() {
                let level = levels[&(*at, index)].clamp(0, top);
                for cell in region {
                    let direct = position.direct[cell_index(*cell)];
                    cells[cell_index(*cell)] = Some(Cell {
                        value: if direct { top } else { level },
                        seeds: level > 0 && !direct,
                        sunlit: false,
                    });
                }
            }
            (*at, cells)
        })
        .collect()
}

/// The cells of the known `positions` under sky openness.
fn open_cells(positions: &HashMap<IVec3, Position>, fill: &OpenFill) -> Cells {
    // Whether the cell at `cell`, counted in cells from the world's origin,
    // is open and whether it is direct, or None in an unknown position.
    let state = |cell: IVec3| -> Option<(bool, bool)> {
        let position = positions.get(&cell.div_euclid(IVec3::splat(4)))?;
        let index = cell_index(cell.rem_euclid(IVec3::splat(4)));
        Some((position.open[index], position.direct[index]))
    };
    let directions: Vec<Vec3> = (0..fill.rays)
        .map(|i| {
            let u = (i as f32 + 0.5) / fill.rays as f32;
            let r = u.sqrt();
            let phi = i as f32 * 2.399_963;
            Vec3::new(r * phi.cos(), (1.0 - u).sqrt(), r * phi.sin())
        })
        .collect();
    let reach = fill.reach as f32 / 8.0;
    // Whether a ray from the middle of `start` along `direction` enters a
    // direct cell before a closed or unknown one, within reach. It enters
    // cells in the order it crosses their faces.
    let reaches_sky = |start: IVec3, direction: Vec3| -> bool {
        let mut crossings: Vec<(f32, usize)> = Vec::new();
        for axis in 0..3 {
            let d = direction[axis].abs();
            if d <= 1e-6 {
                continue;
            }
            for k in 0.. {
                let t = (0.5 + k as f32) / d;
                if t > reach {
                    break;
                }
                crossings.push((t, axis));
            }
        }
        // On a tie, the later axis first.
        crossings.sort_by(|a, b| a.0.total_cmp(&b.0).then(b.1.cmp(&a.1)));
        let mut cell = start;
        for (_, axis) in crossings {
            cell[axis] += direction[axis].signum() as i32;
            match state(cell) {
                None | Some((false, _)) => return false,
                Some((true, true)) => return true,
                Some((true, false)) => {}
            }
        }
        false
    };

    positions
        .iter()
        .map(|(at, position)| {
            let mut out = [None; 64];
            for local in cells() {
                let index = cell_index(local);
                if !position.open[index] {
                    continue;
                }
                if position.direct[index] {
                    out[index] = Some(Cell {
                        value: 240,
                        seeds: false,
                        sunlit: false,
                    });
                    continue;
                }
                let start = *at * 4 + local;
                let seen = directions
                    .iter()
                    .filter(|direction| reaches_sky(start, **direction))
                    .count() as f32
                    / directions.len() as f32;
                out[index] = Some(if seen > 0.0 && seen >= fill.sun_at {
                    Cell {
                        value: 240,
                        seeds: false,
                        sunlit: true,
                    }
                } else {
                    let brightness = seen * fill.gain;
                    let value = if brightness <= 0.0 {
                        0
                    } else {
                        // The level whose brightness is 0.7 ^ (15 - level).
                        let level = 15.0 - (1.0 / brightness).ln() / (1.0f32 / 0.7).ln();
                        (level * 16.0).round().clamp(0.0, 224.0) as i32
                    };
                    Cell {
                        value,
                        seeds: value > 0,
                        sunlit: false,
                    }
                });
            }
            (*at, out)
        })
        .collect()
}

/// The sky light, with sky fill under `fill`, of every voxel of `chunks`,
/// with the positions `known` known to sky fill and every voxel's direct sky
/// worked out from `voxel_at` under the sky rule `sky`, with levels in steps
/// up to `steps`: 15 at 16 levels, 255 at 256.
#[allow(clippy::too_many_arguments)]
pub(in crate::test::light) fn fill_reference_in_steps<I: Copy + PartialEq>(
    steps: u32,
    chunks: &[IVec3],
    known: &[IVec3],
    reach: u32,
    shape: LightShape,
    sky: &Sky,
    fill: &SkyFill,
    voxel_at: impl Fn(IVec3) -> WorldVoxel<I>,
) -> HashMap<IVec3, u8> {
    reference_of(
        steps, chunks, known, reach, shape, sky, fill, voxel_at, false,
    )
}

/// [`fill_reference_in_steps`] at 16 levels with every chunk taking fill, as
/// if a face read every open voxel: the light `light_at` approximates where
/// no face reads.
pub(in crate::test::light) fn fill_reference_everywhere<I: Copy + PartialEq>(
    chunks: &[IVec3],
    known: &[IVec3],
    reach: u32,
    shape: LightShape,
    sky: &Sky,
    fill: &SkyFill,
    voxel_at: impl Fn(IVec3) -> WorldVoxel<I>,
) -> HashMap<IVec3, u8> {
    reference_of(15, chunks, known, reach, shape, sky, fill, voxel_at, true)
}

#[allow(clippy::too_many_arguments)]
fn reference_of<I: Copy + PartialEq>(
    steps: u32,
    chunks: &[IVec3],
    known: &[IVec3],
    reach: u32,
    shape: LightShape,
    sky: &Sky,
    fill: &SkyFill,
    voxel_at: impl Fn(IVec3) -> WorldVoxel<I>,
    every_chunk_takes_fill: bool,
) -> HashMap<IVec3, u8> {
    let known: HashSet<IVec3> = known.iter().copied().collect();
    let positions: HashMap<IVec3, Position> = known
        .iter()
        .map(|at| (*at, position(*at, sky, &voxel_at)))
        .collect();
    let cells = match fill {
        SkyFill::Paths(paths) => path_cells(&positions, paths),
        SkyFill::Openness(open) => open_cells(&positions, open),
    };

    let mut out = HashMap::new();
    for chunk in chunks {
        let min = *chunk * 32 - IVec3::splat(reach as i32);
        let max = *chunk * 32 + IVec3::splat(32 + reach as i32);
        let mut box_light =
            Reference::sky_light_in_steps(steps, min, max, reach, shape, sky, &voxel_at);
        let mut direct: Vec<bool> = box_light
            .sky
            .iter()
            .map(|level| *level as u32 == steps)
            .collect();

        // Which chunks take fill: an open voxel touching a solid, counting
        // the padding, and not every open voxel with direct sky.
        let own = |p: IVec3| {
            (p - *chunk * 32).cmpge(IVec3::ZERO).all()
                && (p - *chunk * 32).cmplt(IVec3::splat(32)).all()
        };
        let padded = (-1..33).flat_map(|z| {
            (-1..33).flat_map(move |y| {
                (-1..33).map(move |x| *chunk * 32 + IVec3::new(x, y, z))
            })
        });
        let mut any_open = false;
        let mut any_solid = false;
        let mut all_direct = true;
        for p in padded {
            let i = box_light.index(p).unwrap();
            if !box_light.open[i] {
                any_solid = true;
            } else if own(p) {
                any_open = true;
                all_direct &= direct[i];
            }
        }
        let takes = any_open && (any_solid || every_chunk_takes_fill) && !all_direct;

        let mut seeds: Vec<Option<u32>> = vec![None; box_light.open.len()];
        if takes {
            seed(&mut seeds, &box_light, *chunk, &cells);
            add_sunlit(&mut direct, &box_light, *chunk, &cells);
        }
        let sources: Vec<(usize, u32)> = (0..direct.len())
            .filter_map(|i| {
                if direct[i] {
                    Some((i, 0))
                } else {
                    seeds[i].map(|start| (i, start))
                }
            })
            .collect();
        let distances = box_light.flood(&sources);
        for (i, distance) in distances.iter().enumerate() {
            let level = if direct[i] {
                steps as u8
            } else if !box_light.open[i] {
                0
            } else {
                box_light.level((*distance).min(seeds[i].unwrap_or(u32::MAX)))
            };
            box_light.sky[i] = level;
        }
        for z in 0..32 {
            for y in 0..32 {
                for x in 0..32 {
                    let p = *chunk * 32 + IVec3::new(x, y, z);
                    out.insert(p, box_light.sky_at(p));
                }
            }
        }
    }
    out
}

/// Marks in `direct`, by voxel of `box_light`, the voxels of the sunlit
/// cells of the 27 positions around `chunk`.
fn add_sunlit(direct: &mut [bool], box_light: &Reference, chunk: IVec3, cells: &Cells) {
    for z in -1..=1 {
        for y in -1..=1 {
            for x in -1..=1 {
                let at = chunk + IVec3::new(x, y, z);
                let Some(position) = cells.get(&at) else {
                    continue;
                };
                for local in self::cells() {
                    if !position[cell_index(local)].is_some_and(|cell| cell.sunlit) {
                        continue;
                    }
                    let min = at * 32 + local * 8;
                    for p in (0..8).flat_map(|z| {
                        (0..8).flat_map(move |y| {
                            (0..8).map(move |x| min + IVec3::new(x, y, z))
                        })
                    }) {
                        if let Some(i) = box_light.index(p) {
                            direct[i] = true;
                        }
                    }
                }
            }
        }
    }
}

/// Writes into `seeds`, by voxel of `box_light`, the distance from a source
/// at the top step at which sky fill's seeds start for the chunk at `chunk`.
fn seed(seeds: &mut [Option<u32>], box_light: &Reference, chunk: IVec3, cells: &Cells) {
    const GRID: i32 = 12;
    let grid_index =
        |cell: IVec3| (cell.x + cell.y * GRID + cell.z * GRID * GRID) as usize;
    let mut open = vec![false; 1728];
    let mut value = vec![0i32; 1728];
    let mut lit = vec![false; 1728];
    for z in -1..=1 {
        for y in -1..=1 {
            for x in -1..=1 {
                let offset = IVec3::new(x, y, z);
                let at = chunk + offset;
                let base = (offset + IVec3::ONE) * 4;
                let Some(position) = cells.get(&at) else {
                    for cell in self::cells() {
                        open[grid_index(base + cell)] = true;
                    }
                    continue;
                };
                for local in self::cells() {
                    let Some(cell) = position[cell_index(local)] else {
                        continue;
                    };
                    let g = grid_index(base + local);
                    open[g] = true;
                    value[g] = cell.value;
                    lit[g] = cell.seeds;
                }
            }
        }
    }
    for _ in 0..2 {
        for axis in 0..3 {
            let before = value.clone();
            for z in 0..GRID {
                for y in 0..GRID {
                    for x in 0..GRID {
                        let cell = IVec3::new(x, y, z);
                        let g = grid_index(cell);
                        if !open[g] {
                            continue;
                        }
                        let mut step = IVec3::ZERO;
                        step[axis] = 1;
                        let read = |beside: IVec3| {
                            let inside = beside.cmpge(IVec3::ZERO).all()
                                && beside.cmplt(IVec3::splat(GRID)).all();
                            if inside && open[grid_index(beside)] {
                                before[grid_index(beside)]
                            } else {
                                before[g]
                            }
                        };
                        value[g] =
                            (2 * before[g] + read(cell - step) + read(cell + step)) / 4;
                    }
                }
            }
        }
    }
    let reach = box_light.reach;
    let steps = box_light.top as i32;
    let mut by_level: Vec<Vec<usize>> = vec![Vec::new(); steps as usize];
    for z in 0..GRID {
        for y in 0..GRID {
            for x in 0..GRID {
                let cell = IVec3::new(x, y, z);
                let g = grid_index(cell);
                if !lit[g] {
                    continue;
                }
                let level =
                    ((value[g] * steps / 15 + 8) / 16).clamp(0, steps - 1) as usize;
                if level == 0 {
                    continue;
                }
                let min = chunk * 32 + (cell - IVec3::splat(4)) * 8;
                for p in (0..8).flat_map(|z| {
                    (0..8)
                        .flat_map(move |y| (0..8).map(move |x| min + IVec3::new(x, y, z)))
                }) {
                    if let Some(i) = box_light.index(p) {
                        if box_light.open[i] {
                            by_level[level].push(i);
                        }
                    }
                }
            }
        }
    }
    // Each level's seeds run 7 face steps through open voxels, outward from
    // the seeded voxels. Passing over the whole box once per level instead is
    // too slow at 256 levels. `reached[i]` is the last level whose run
    // reached voxel `i`.
    let mut reached = vec![usize::MAX; seeds.len()];
    for (level, voxels) in by_level.into_iter().enumerate() {
        if voxels.is_empty() {
            continue;
        }
        let Some(start) =
            (0..=reach).find(|&d| level_at(reach, d, steps as u32) <= level as u32)
        else {
            continue;
        };
        if start >= reach {
            continue;
        }
        let mut frontier = Vec::new();
        for i in voxels {
            if reached[i] != level {
                reached[i] = level;
                frontier.push(i);
            }
        }
        let mut run = frontier.clone();
        for _ in 0..7 {
            let mut next = Vec::new();
            for i in frontier {
                for n in box_light.neighbours(i, 3) {
                    if box_light.open[n] && reached[n] != level {
                        reached[n] = level;
                        next.push(n);
                    }
                }
            }
            run.extend_from_slice(&next);
            frontier = next;
        }
        for i in run {
            seeds[i] = Some(seeds[i].map_or(start, |s| s.min(start)));
        }
    }
}

/// The level `distance` steps from a source at the top step `top`, at
/// `reach`.
fn level_at(reach: u32, distance: u32, top: u32) -> u32 {
    if distance == 0 {
        return top;
    }
    if distance > reach {
        return 0;
    }
    let t = 1.0 - distance as f32 / (reach + 1) as f32;
    ((top as f32 * t).round() as u32).clamp(1, top - 1)
}
