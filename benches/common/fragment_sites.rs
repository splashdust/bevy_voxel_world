//! Where `chunk_lifecycle`'s `fragments` scenario digs and cuts in the light
//! reference scene, found from the scene's voxels around a point.
//!
//! A dig sets one exposed solid voxel to air where its solid neighbours stay
//! joined around it, so it cuts nothing off. A cut sets every solid voxel on
//! the faces of a box to air, which leaves the solid voxels inside the box
//! with nothing holding them to the rest of the scene.

use std::collections::VecDeque;

use bevy::prelude::*;

use crate::light_scene::{SceneTable, SCENE_TOP};

/// How many digs [`find`] picks.
pub const DIGS: usize = 200;

/// The sides of the boxes cuts are made around, and how many cuts [`find`]
/// picks of each. The voxels inside a box of side `s` number `(s - 2)³`.
pub const CUT_SIDES: [(i32, usize); 3] = [(10, 4), (14, 4), (18, 4)];

/// How many solid voxels the piece inside a cut's box may hold.
pub const PIECE_VOXELS: std::ops::RangeInclusive<u32> = 300..=4000;

/// The side of the box of the cut that drops nothing: the 34³ solid voxels
/// inside it are more than the 32,768 a check floods by default before it
/// counts a piece as attached.
pub const LIMIT_CUT_SIDE: i32 = 36;

/// How many chunks from the centre's chunk sites are searched for, on each
/// axis.
const REACH: i32 = 6;

/// How far around a cut's box the voxels outside it must stay joined.
const MARGIN: i32 = 16;

/// How far apart cut boxes stay, so that no dig or other cut touches what a
/// cut's check reads.
const SPACING: i32 = 32;

const FACES: [IVec3; 6] = [
    IVec3::X,
    IVec3::NEG_X,
    IVec3::Y,
    IVec3::NEG_Y,
    IVec3::Z,
    IVec3::NEG_Z,
];

/// A box whose faces a cut sets to air.
pub struct Cut {
    /// The box's lowest corner, inclusive.
    pub min: IVec3,
    /// The box's highest corner, exclusive.
    pub max: IVec3,
    /// The solid voxels on the box's faces, which the cut sets to air.
    pub dug: Vec<IVec3>,
    /// The solid voxels inside the box, which form one piece joined by
    /// faces.
    pub piece: u32,
}

/// The digs and cuts of the `fragments` scenario.
pub struct Sites {
    /// Exposed solid voxels in separate chunks, alternately with air above
    /// them, as on a slope, and with air beside them under solid, as in a
    /// cave wall.
    pub digs: Vec<IVec3>,
    /// Cuts around pieces of [`PIECE_VOXELS`] with air under some of their
    /// voxels, as in a cave ceiling, an overhang or an arch.
    pub cuts: Vec<Cut>,
    /// A cut around more solid voxels than a check floods by default.
    pub limit_cut: Cut,
}

/// Finds [`DIGS`] digs, the cuts of [`CUT_SIDES`] and the limit cut within
/// [`REACH`] chunks of `centre`, nearest first, reading the voxels of
/// `table`.
///
/// # Panics
///
/// If the scene has too few places that fit around `centre`.
pub fn find(table: &SceneTable, centre: Vec3) -> Sites {
    let solid = |p: IVec3| !table.open(p);
    let centre_chunk = (centre / 32.0).floor().as_ivec3();
    let region_min = (centre_chunk - IVec3::splat(REACH)) * 32;
    let region_max = (centre_chunk + IVec3::splat(REACH + 1)) * 32;
    let region_max = region_max.with_y(region_max.y.min(SCENE_TOP));
    let centre = centre.as_ivec3();

    let mut taken: Vec<(IVec3, IVec3)> = Vec::new();
    let apart = |taken: &[(IVec3, IVec3)], min: IVec3, max: IVec3| {
        taken.iter().all(|(other_min, other_max)| {
            (min - SPACING).cmpge(*other_max).any()
                || (max + SPACING).cmple(*other_min).any()
        })
    };

    let limit_cut = boxes(region_min, region_max, LIMIT_CUT_SIDE, centre)
        .find_map(|min| limit_cut_at(min, &solid))
        .expect("a box of solid voxels for the limit cut");
    taken.push((limit_cut.min, limit_cut.max));

    let mut cuts = Vec::new();
    for (side, count) in CUT_SIDES {
        let mut found = 0;
        for min in boxes(region_min, region_max, side, centre) {
            if found == count {
                break;
            }
            if !apart(&taken, min, min + side) {
                continue;
            }
            if let Some(cut) = cut_at(min, side, &solid) {
                taken.push((cut.min, cut.max));
                cuts.push(cut);
                found += 1;
            }
        }
        assert_eq!(found, count, "too few pieces in boxes of side {side}");
    }

    let mut chunks: Vec<IVec3> = Vec::new();
    for z in -REACH..=REACH {
        for y in -REACH..=REACH {
            for x in -REACH..=REACH {
                let chunk = centre_chunk + IVec3::new(x, y, z);
                if chunk.y * 32 < SCENE_TOP {
                    chunks.push(chunk);
                }
            }
        }
    }
    chunks.sort_by_key(|chunk| {
        ((*chunk - centre_chunk).length_squared(), chunk.to_array())
    });
    let mut digs = Vec::new();
    for chunk in chunks {
        if digs.len() == DIGS {
            break;
        }
        let (min, max) = (chunk * 32, chunk * 32 + 32);
        if !apart(&taken, min, max) {
            continue;
        }
        let wall = digs.len() % 2 == 1;
        if let Some(dig) =
            dig_in(chunk, wall, &solid).or_else(|| dig_in(chunk, !wall, &solid))
        {
            digs.push((chunk, dig));
        }
    }
    assert_eq!(digs.len(), DIGS, "too few chunks with a place to dig");
    // Chunks alike in whether each coordinate is odd are never neighbours,
    // so digs grouped that way seldom follow one in a chunk the last one's
    // check reads.
    digs.sort_by_key(|(chunk, _)| (*chunk & 1).to_array());
    let digs = digs.into_iter().map(|(_, dig)| dig).collect();

    Sites {
        digs,
        cuts,
        limit_cut,
    }
}

/// The lowest corners of the boxes of `side` that tile the region from `min`
/// to `max`, nearest to `centre` first.
fn boxes(
    min: IVec3,
    max: IVec3,
    side: i32,
    centre: IVec3,
) -> impl Iterator<Item = IVec3> {
    let mut corners = Vec::new();
    for z in (min.z..=max.z - side).step_by(side as usize) {
        for y in (min.y..=max.y - side).step_by(side as usize) {
            for x in (min.x..=max.x - side).step_by(side as usize) {
                corners.push(IVec3::new(x, y, z));
            }
        }
    }
    corners.sort_by_key(|corner| {
        let middle = *corner + side / 2;
        ((middle - centre).length_squared(), corner.to_array())
    });
    corners.into_iter()
}

/// Every position in the box from `min` to `max`, maximum exclusive.
fn positions(min: IVec3, max: IVec3) -> impl Iterator<Item = IVec3> {
    (min.z..max.z).flat_map(move |z| {
        (min.y..max.y).flat_map(move |y| (min.x..max.x).map(move |x| IVec3::new(x, y, z)))
    })
}

fn within(min: IVec3, max: IVec3, p: IVec3) -> bool {
    p.cmpge(min).all() && p.cmplt(max).all()
}

/// The cut around the box of `side` at `min`, if the solid voxels inside it
/// form one piece of [`PIECE_VOXELS`] with air under some of them, held to
/// the voxels outside the box only through its faces, and the solid voxels
/// touching the box from outside stay joined to each other without it.
fn cut_at(min: IVec3, side: i32, solid: &impl Fn(IVec3) -> bool) -> Option<Cut> {
    let max = min + side;
    let (inner_min, inner_max) = (min + 1, max - 1);
    let inside: Vec<IVec3> = positions(inner_min, inner_max)
        .filter(|p| solid(*p))
        .collect();
    let piece = inside.len() as u32;
    if !PIECE_VOXELS.contains(&piece) {
        return None;
    }
    let hangs = inside
        .iter()
        .any(|p| within(inner_min, inner_max, *p - IVec3::Y) && !solid(*p - IVec3::Y));
    if !hangs {
        return None;
    }
    if flood(inner_min, inner_max, inside[0], solid).count != piece {
        return None;
    }
    let cut = shell(min, max, solid);
    let holds = cut.dug.iter().any(|p| {
        FACES
            .iter()
            .any(|face| within(inner_min, inner_max, *p + *face) && solid(*p + *face))
    });
    (holds && outside_stays_joined(&cut, solid)).then_some(Cut { piece, ..cut })
}

/// The limit cut around the box of [`LIMIT_CUT_SIDE`] at `min`, if every
/// voxel inside it is solid and the solid voxels touching the box from
/// outside stay joined to each other without it.
fn limit_cut_at(min: IVec3, solid: &impl Fn(IVec3) -> bool) -> Option<Cut> {
    let max = min + LIMIT_CUT_SIDE;
    if !positions(min + 1, max - 1).all(solid) {
        return None;
    }
    let cut = Cut {
        piece: (LIMIT_CUT_SIDE - 2).pow(3) as u32,
        ..shell(min, max, solid)
    };
    outside_stays_joined(&cut, solid).then_some(cut)
}

/// The cut setting to air the solid voxels on the faces of the box from
/// `min` to `max`, with no piece counted yet.
fn shell(min: IVec3, max: IVec3, solid: &impl Fn(IVec3) -> bool) -> Cut {
    let dug = positions(min, max)
        .filter(|p| !within(min + 1, max - 1, *p) && solid(*p))
        .collect();
    Cut {
        min,
        max,
        dug,
        piece: 0,
    }
}

/// Whether the solid voxels outside `cut`'s box that touch what it digs are
/// all joined to each other within [`MARGIN`] of the box, without passing
/// through it, and reach that margin's edge, as voxels held to the rest of
/// the scene would.
fn outside_stays_joined(cut: &Cut, solid: &impl Fn(IVec3) -> bool) -> bool {
    let outside = |p: IVec3| !within(cut.min, cut.max, p) && solid(p);
    let touching: Vec<IVec3> = cut
        .dug
        .iter()
        .flat_map(|p| FACES.map(|face| *p + face))
        .filter(|p| outside(*p))
        .collect();
    let Some(first) = touching.first() else {
        return false;
    };
    let joined = flood(cut.min - MARGIN, cut.max + MARGIN, *first, &outside);
    joined.reached_edge && touching.iter().all(|p| joined.contains(*p))
}

/// A voxel to dig in `chunk`, away from the chunk's faces: with air beside
/// it and solid above and below it when `wall`, with air above it otherwise.
/// It touches at least three solid voxels, which stay joined to each other
/// around it within three voxels of it.
fn dig_in(chunk: IVec3, wall: bool, solid: &impl Fn(IVec3) -> bool) -> Option<IVec3> {
    let origin = chunk * 32;
    positions(origin + 8, origin + 24).find(|p| {
        let p = *p;
        if !solid(p) {
            return false;
        }
        let kind = if wall {
            solid(p + IVec3::Y)
                && solid(p - IVec3::Y)
                && [IVec3::X, IVec3::NEG_X, IVec3::Z, IVec3::NEG_Z]
                    .iter()
                    .any(|side| !solid(p + *side))
        } else {
            !solid(p + IVec3::Y)
        };
        if !kind {
            return false;
        }
        let neighbours: Vec<IVec3> = FACES
            .map(|face| p + face)
            .into_iter()
            .filter(|n| solid(*n))
            .collect();
        if neighbours.len() < 3 {
            return false;
        }
        let around = |q: IVec3| q != p && solid(q);
        let joined = flood(p - 3, p + 4, neighbours[0], &around);
        joined.reached_edge && neighbours.iter().all(|n| joined.contains(*n))
    })
}

/// The voxels a flood from `start` through faces reaches where `solid`
/// holds, within the box from `min` to `max`.
struct Flood {
    min: IVec3,
    size: IVec3,
    reached: Vec<bool>,
    count: u32,
    /// Whether it reached a voxel on the box's faces.
    reached_edge: bool,
}

impl Flood {
    fn index(&self, p: IVec3) -> usize {
        let local = p - self.min;
        (local.x + self.size.x * (local.y + self.size.y * local.z)) as usize
    }

    fn contains(&self, p: IVec3) -> bool {
        within(self.min, self.min + self.size, p) && self.reached[self.index(p)]
    }
}

fn flood(min: IVec3, max: IVec3, start: IVec3, solid: &impl Fn(IVec3) -> bool) -> Flood {
    let size = max - min;
    let mut flood = Flood {
        min,
        size,
        reached: vec![false; (size.x * size.y * size.z) as usize],
        count: 0,
        reached_edge: false,
    };
    let mut queue = VecDeque::from([start]);
    let first = flood.index(start);
    flood.reached[first] = true;
    while let Some(p) = queue.pop_front() {
        flood.count += 1;
        flood.reached_edge |= p.cmpeq(min).any() || p.cmpeq(max - 1).any();
        for face in FACES {
            let next = p + face;
            if !within(min, max, next) || !solid(next) {
                continue;
            }
            let i = flood.index(next);
            if !flood.reached[i] {
                flood.reached[i] = true;
                queue.push_back(next);
            }
        }
    }
    flood
}
