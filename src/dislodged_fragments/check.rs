//! The fragment check: which pieces of one grid an edit cut off from the rest.
//!
//! [`check_fragments`] is a pure function of its seeds, a voxel reader and
//! [`FragmentRules`]. It reads one grid, so only connections within that
//! grid count.
//!
//! Each seed is a position whose voxel no longer carries. A flood starts from
//! every carrying voxel a seed touches, and a voxel that several seeds touch
//! starts one flood. Each step advances whichever flood has visited the
//! fewest voxels by one voxel, and floods that meet join. The check stops
//! when at most one flood is still growing, or when every flood still growing
//! has passed the limit. Every piece whose flood ran out is dislodged. The
//! last one standing keeps its place, and a flood past the limit counts as
//! attached.
//!
//! Since the smallest flood always advances, pieces run out in order of
//! size: a small piece cut free is found after about as many steps as it has
//! voxels, and a dig into a solid mass stops as soon as its floods have met,
//! or soon after the limit.
//!
//! Positions have one fixed order, [`position_order`]: lowest `y` first,
//! then lowest `x`, then lowest `z`. Of two floods that have visited as many
//! voxels, the one whose lowest voxel comes later in that order advances
//! first. So of two equal pieces, that one runs out first, and the piece
//! holding the lowest voxel position keeps its place. Nothing depends on the
//! order the seeds come in.
//!
//! The cost grows with the voxels the floods visit and the air around them:
//! every voxel of each dislodged fragment, and of each piece that keeps its
//! place, about as many as the largest dislodged fragment holds, or the limit
//! if that comes first. With no limit, as in a sub-grid's check, the cost
//! grows with the size of every piece but the largest.

use std::{
    cmp::Reverse,
    collections::{BinaryHeap, VecDeque},
};

use bevy::{platform::collections::HashMap, prelude::*};

use super::Connectivity;
use crate::voxel::WorldVoxel;

/// The rules a fragment check reads its grid by.
pub(crate) struct FragmentRules<'a, I> {
    /// A flood that has visited more voxels than this counts as attached.
    /// With `None`, every piece but the largest is flooded in full, as a
    /// sub-grid's check needs.
    pub limit: Option<u32>,
    /// Which neighbours of a voxel belong to its piece.
    pub connectivity: Connectivity,
    /// Materials that read as air, so they never hold a piece up.
    pub non_carrying: &'a [I],
}

impl<I: PartialEq> FragmentRules<'_, I> {
    /// Whether `voxel` holds up the voxels it touches. Air and `Unset` never
    /// do, and neither does a material in `non_carrying`.
    pub fn carries(&self, voxel: &WorldVoxel<I>) -> bool {
        match voxel {
            WorldVoxel::Solid(material) => !self.non_carrying.contains(material),
            WorldVoxel::Air | WorldVoxel::Unset => false,
        }
    }
}

/// One dislodged fragment found by [`check_fragments`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Fragment {
    /// The fragment's voxel positions, in [`position_order`].
    pub voxels: Vec<IVec3>,
    /// The lowest corner of the box around `voxels`, inclusive.
    pub min: IVec3,
    /// The highest corner of the box around `voxels`, exclusive.
    pub max: IVec3,
    /// How many voxels the fragment holds.
    pub voxel_count: u32,
}

/// What one fragment check found.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct FragmentCheck {
    /// The dislodged fragments, in [`position_order`] of their lowest voxel.
    pub fragments: Vec<Fragment>,
    /// How many carrying voxels the floods visited, kept or dislodged.
    pub visited: u32,
}

/// The fixed order of voxel positions the fragment check breaks ties by:
/// lowest `y` first, then lowest `x`, then lowest `z`.
pub(crate) fn position_order(position: IVec3) -> (i32, i32, i32) {
    (position.y, position.x, position.z)
}

/// Finds the pieces of one grid that no longer hold to the rest of it, as
/// the module docs describe.
///
/// `seeds` are positions whose voxels no longer carry, and `read` gives the
/// grid's voxel at a position. `read` is called at most once per position.
///
/// Returns the dislodged fragments and how many voxels the floods visited.
pub(crate) fn check_fragments<I: PartialEq>(
    seeds: impl IntoIterator<Item = IVec3>,
    mut read: impl FnMut(IVec3) -> WorldVoxel<I>,
    rules: &FragmentRules<I>,
) -> FragmentCheck {
    let neighbours = neighbours(rules.connectivity);
    let mut floods = Floods::default();
    let mut carries = |p: IVec3, seen: &mut HashMap<IVec3, Seen>| {
        let carries = rules.carries(&read(p));
        if !carries {
            seen.insert(p, Seen::Open);
        }
        carries
    };

    // Sorted, so floods are numbered the same whatever order the seeds came in.
    let mut seeds: Vec<IVec3> = seeds.into_iter().collect();
    seeds.sort_unstable_by_key(|&p| position_order(p));
    seeds.dedup();
    for seed in seeds {
        for &offset in neighbours {
            let p = seed + offset;
            if !floods.seen.contains_key(&p) && carries(p, &mut floods.seen) {
                floods.start(p);
            }
        }
    }

    while floods.growing > 1 {
        let Some(Reverse(step)) = floods.queue.pop() else {
            break;
        };
        if !floods.is_current(&step) {
            continue;
        }
        // The smallest flood still growing is past the limit, so all are.
        if rules.limit.is_some_and(|limit| step.visited > limit) {
            break;
        }
        let mut flood = step.flood as usize;
        let Some(voxel) = floods.list[flood].frontier.pop_front() else {
            floods.list[flood].ran_out = true;
            floods.growing -= 1;
            continue;
        };
        for &offset in neighbours {
            let p = voxel + offset;
            match floods.seen.get(&p) {
                Some(Seen::Open) => {}
                Some(&Seen::Flood(other)) => {
                    let other = floods.root(other as usize);
                    if other != flood {
                        flood = floods.join(flood, other);
                    }
                }
                None => {
                    if carries(p, &mut floods.seen) {
                        floods.claim(flood, p);
                    }
                }
            }
        }
        floods.queue_step(flood);
    }

    let mut fragments: Vec<Fragment> = floods
        .list
        .into_iter()
        .filter(|flood| flood.ran_out)
        .map(Flood::into_fragment)
        .collect();
    fragments.sort_unstable_by_key(|fragment| position_order(fragment.voxels[0]));
    FragmentCheck {
        fragments,
        visited: floods.visited,
    }
}

/// The 26 offsets to a voxel's neighbours: the 6 sharing a face, then the 12
/// sharing only an edge, then the 8 sharing only a corner.
const NEIGHBOURS: [IVec3; 26] = {
    let mut out = [IVec3::ZERO; 26];
    let mut faces = 0;
    let mut edges = 6;
    let mut corners = 18;
    let mut i = 0;
    while i < 27 {
        let offset = IVec3::new(i % 3 - 1, i / 3 % 3 - 1, i / 9 - 1);
        match offset.x.abs() + offset.y.abs() + offset.z.abs() {
            1 => {
                out[faces] = offset;
                faces += 1;
            }
            2 => {
                out[edges] = offset;
                edges += 1;
            }
            3 => {
                out[corners] = offset;
                corners += 1;
            }
            _ => {}
        }
        i += 1;
    }
    out
};

/// The offsets to the neighbours `connectivity` joins a voxel to.
fn neighbours(connectivity: Connectivity) -> &'static [IVec3] {
    match connectivity {
        Connectivity::Faces => &NEIGHBOURS[..6],
        Connectivity::Edges => &NEIGHBOURS[..18],
        Connectivity::Corners => &NEIGHBOURS,
    }
}

/// What the check found at a position it read.
enum Seen {
    /// A voxel that doesn't carry.
    Open,
    /// A carrying voxel, visited by this flood or by one it has joined since.
    Flood(u32),
}

/// One flood of the check, growing over a piece from where it started.
struct Flood {
    /// Visited voxels whose neighbours haven't been looked at yet.
    frontier: VecDeque<IVec3>,
    /// Every voxel this flood has visited, including those of floods that
    /// joined it.
    voxels: Vec<IVec3>,
    /// The first of `voxels` in [`position_order`].
    lowest: IVec3,
    /// Whether the flood ran out of voxels, so its piece is dislodged.
    ran_out: bool,
}

impl Flood {
    fn new(lowest: IVec3) -> Self {
        Self {
            frontier: VecDeque::new(),
            voxels: Vec::new(),
            lowest,
            ran_out: false,
        }
    }

    fn into_fragment(mut self) -> Fragment {
        self.voxels.sort_unstable_by_key(|&p| position_order(p));
        let (min, max) = self
            .voxels
            .iter()
            .fold((IVec3::MAX, IVec3::MIN), |(min, max), &p| {
                (min.min(p), max.max(p))
            });
        Fragment {
            voxel_count: self.voxels.len() as u32,
            voxels: self.voxels,
            min,
            max: max + IVec3::ONE,
        }
    }
}

/// A flood's place in the queue of floods waiting to advance. The smallest
/// comes out first, and of two that have visited as many voxels, the one
/// whose lowest voxel comes later in [`position_order`].
#[derive(PartialEq, Eq, PartialOrd, Ord)]
struct Step {
    visited: u32,
    lowest: Reverse<(i32, i32, i32)>,
    flood: u32,
}

/// Every flood of one check.
///
/// A flood that joins another stays in `list`, with `parent` leading on to
/// the flood that holds its voxels now. The queue holds a [`Step`] for each
/// flood still growing, and the steps left behind when a flood grew or
/// joined another, which [`Floods::is_current`] skips.
#[derive(Default)]
struct Floods {
    list: Vec<Flood>,
    parent: Vec<u32>,
    seen: HashMap<IVec3, Seen>,
    queue: BinaryHeap<Reverse<Step>>,
    growing: usize,
    visited: u32,
}

impl Floods {
    /// Starts a flood at the carrying voxel `p`.
    fn start(&mut self, p: IVec3) {
        let flood = self.list.len();
        self.list.push(Flood::new(p));
        self.parent.push(flood as u32);
        self.growing += 1;
        self.claim(flood, p);
        self.queue_step(flood);
    }

    /// Adds the carrying voxel `p` to `flood`.
    fn claim(&mut self, flood: usize, p: IVec3) {
        self.seen.insert(p, Seen::Flood(flood as u32));
        self.visited += 1;
        let flood = &mut self.list[flood];
        flood.frontier.push_back(p);
        flood.voxels.push(p);
        if position_order(p) < position_order(flood.lowest) {
            flood.lowest = p;
        }
    }

    /// The flood that holds `flood`'s voxels now.
    fn root(&mut self, flood: usize) -> usize {
        let mut root = flood;
        while self.parent[root] as usize != root {
            root = self.parent[root] as usize;
        }
        let mut at = flood;
        while at != root {
            at = std::mem::replace(&mut self.parent[at], root as u32) as usize;
        }
        root
    }

    /// Joins two growing floods that met, moving the smaller one's voxels
    /// into the larger. Returns the flood that holds them both.
    fn join(&mut self, a: usize, b: usize) -> usize {
        let (into, from) = if self.list[a].voxels.len() >= self.list[b].voxels.len() {
            (a, b)
        } else {
            (b, a)
        };
        let mut moved = std::mem::replace(&mut self.list[from], Flood::new(IVec3::ZERO));
        let into_flood = &mut self.list[into];
        into_flood.frontier.append(&mut moved.frontier);
        into_flood.voxels.append(&mut moved.voxels);
        if position_order(moved.lowest) < position_order(into_flood.lowest) {
            into_flood.lowest = moved.lowest;
        }
        self.parent[from] = into as u32;
        self.growing -= 1;
        into
    }

    /// Queues `flood` to advance at its current size.
    fn queue_step(&mut self, flood: usize) {
        let f = &self.list[flood];
        self.queue.push(Reverse(Step {
            visited: f.voxels.len() as u32,
            lowest: Reverse(position_order(f.lowest)),
            flood: flood as u32,
        }));
    }

    /// Whether `step` still describes a growing flood as it is now.
    fn is_current(&self, step: &Step) -> bool {
        let flood = step.flood as usize;
        let f = &self.list[flood];
        self.parent[flood] as usize == flood
            && !f.ran_out
            && f.voxels.len() as u32 == step.visited
            && Reverse(position_order(f.lowest)) == step.lowest
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::platform::collections::HashSet;

    const ROCK: u8 = 1;

    /// Every position in `min..max`.
    fn cuboid(min: IVec3, max: IVec3) -> Vec<IVec3> {
        let mut out = Vec::new();
        for y in min.y..max.y {
            for x in min.x..max.x {
                for z in min.z..max.z {
                    out.push(IVec3::new(x, y, z));
                }
            }
        }
        out
    }

    /// A grid holding rock at `solids` and air everywhere else.
    fn rock_at(solids: &HashSet<IVec3>) -> impl FnMut(IVec3) -> WorldVoxel<u8> + '_ {
        |p| {
            if solids.contains(&p) {
                WorldVoxel::Solid(ROCK)
            } else {
                WorldVoxel::Air
            }
        }
    }

    /// `rock_at(solids)` above rock that fills every position with `y < 0`.
    fn on_ground(solids: &HashSet<IVec3>) -> impl FnMut(IVec3) -> WorldVoxel<u8> + '_ {
        |p| {
            if p.y < 0 || solids.contains(&p) {
                WorldVoxel::Solid(ROCK)
            } else {
                WorldVoxel::Air
            }
        }
    }

    fn rules(limit: Option<u32>) -> FragmentRules<'static, u8> {
        FragmentRules {
            limit,
            connectivity: Connectivity::Faces,
            non_carrying: &[],
        }
    }

    /// Removes `cut` from `solids` and returns it as the seeds.
    fn carve(solids: &mut HashSet<IVec3>, cut: &[IVec3]) -> Vec<IVec3> {
        for p in cut {
            solids.remove(p);
        }
        cut.to_vec()
    }

    fn sorted(mut voxels: Vec<IVec3>) -> Vec<IVec3> {
        voxels.sort_by_key(|p| (p.y, p.x, p.z));
        voxels
    }

    #[test]
    fn a_piece_cut_from_under_an_overhang_is_reported_with_its_voxels_and_bounds() {
        // A wall on the ground with a shelf sticking out from its top, cut
        // through next to the wall.
        let mut solids: HashSet<IVec3> = cuboid(IVec3::ZERO, IVec3::new(4, 10, 4))
            .into_iter()
            .chain(cuboid(IVec3::new(4, 7, 0), IVec3::new(9, 9, 4)))
            .collect();
        let seeds = carve(
            &mut solids,
            &cuboid(IVec3::new(4, 7, 0), IVec3::new(5, 9, 4)),
        );

        let check = check_fragments(seeds, on_ground(&solids), &rules(Some(1000)));

        let piece = cuboid(IVec3::new(5, 7, 0), IVec3::new(9, 9, 4));
        assert_eq!(
            check.fragments,
            vec![Fragment {
                voxels: sorted(piece),
                min: IVec3::new(5, 7, 0),
                max: IVec3::new(9, 9, 4),
                voxel_count: 32,
            }]
        );
    }

    #[test]
    fn a_floating_rock_cut_in_two_drops_the_smaller_half() {
        let mut solids: HashSet<IVec3> = cuboid(IVec3::ZERO, IVec3::new(10, 4, 4))
            .into_iter()
            .collect();
        let seeds = carve(
            &mut solids,
            &cuboid(IVec3::new(6, 0, 0), IVec3::new(7, 4, 4)),
        );

        let check = check_fragments(seeds, rock_at(&solids), &rules(Some(1000)));

        let smaller = cuboid(IVec3::new(7, 0, 0), IVec3::new(10, 4, 4));
        assert_eq!(check.fragments.len(), 1);
        assert_eq!(check.fragments[0].voxels, sorted(smaller));
    }

    #[test]
    fn a_dig_into_a_solid_mass_reports_nothing_and_stops_within_a_few_voxels() {
        // Rock everywhere but a shaft, so a flood that never stopped would
        // never end.
        let shaft = cuboid(IVec3::new(0, -5, 0), IVec3::new(1, 0, 1));
        let dug: HashSet<IVec3> = shaft.iter().copied().collect();
        let read = |p| {
            if dug.contains(&p) {
                WorldVoxel::Air
            } else {
                WorldVoxel::Solid(ROCK)
            }
        };

        let check = check_fragments(shaft.clone(), read, &rules(Some(32_768)));

        assert!(check.fragments.is_empty());
        assert!(check.visited < 100, "visited {}", check.visited);
    }

    #[test]
    fn a_piece_larger_than_the_limit_counts_as_attached() {
        // The overhang of the first test, 32 voxels.
        let mut solids: HashSet<IVec3> = cuboid(IVec3::ZERO, IVec3::new(4, 10, 4))
            .into_iter()
            .chain(cuboid(IVec3::new(4, 7, 0), IVec3::new(9, 9, 4)))
            .collect();
        let seeds = carve(
            &mut solids,
            &cuboid(IVec3::new(4, 7, 0), IVec3::new(5, 9, 4)),
        );

        let check = check_fragments(seeds.clone(), on_ground(&solids), &rules(Some(31)));
        assert!(check.fragments.is_empty());
        // Two floods, each stopping within one step of passing the limit,
        // though the ground's flood would never run out.
        assert!(check.visited <= 2 * (31 + 6), "visited {}", check.visited);

        let check = check_fragments(seeds, on_ground(&solids), &rules(Some(32)));
        assert_eq!(check.fragments.len(), 1);
        assert_eq!(check.fragments[0].voxel_count, 32);
    }

    #[test]
    fn with_no_limit_every_piece_but_the_largest_is_dislodged() {
        // A floating beam cut twice into pieces of 1,000, 1,400 and 400.
        let mut solids: HashSet<IVec3> = cuboid(IVec3::ZERO, IVec3::new(30, 10, 10))
            .into_iter()
            .collect();
        let mut seeds = carve(
            &mut solids,
            &cuboid(IVec3::new(10, 0, 0), IVec3::new(11, 10, 10)),
        );
        seeds.extend(carve(
            &mut solids,
            &cuboid(IVec3::new(25, 0, 0), IVec3::new(26, 10, 10)),
        ));

        let limited = check_fragments(seeds.clone(), rock_at(&solids), &rules(Some(500)));
        let counts: Vec<u32> = limited.fragments.iter().map(|f| f.voxel_count).collect();
        assert_eq!(counts, [400]);

        let unlimited = check_fragments(seeds, rock_at(&solids), &rules(None));
        let found: Vec<(IVec3, IVec3, u32)> = unlimited
            .fragments
            .iter()
            .map(|f| (f.min, f.max, f.voxel_count))
            .collect();
        assert_eq!(
            found,
            [
                (IVec3::ZERO, IVec3::new(10, 10, 10), 1000),
                (IVec3::new(26, 0, 0), IVec3::new(30, 10, 10), 400),
            ]
        );
    }

    /// Whether a voxel touching a 64-voxel floating rock at `touch` is cut
    /// free when the rock between them at `seed` goes, under `connectivity`.
    fn cut_free(touch: IVec3, seeds: &[IVec3], connectivity: Connectivity) -> bool {
        let solids: HashSet<IVec3> = cuboid(IVec3::ZERO, IVec3::splat(4))
            .into_iter()
            .chain([touch])
            .collect();
        let rules = FragmentRules {
            connectivity,
            ..rules(Some(1000))
        };
        let check = check_fragments(seeds.to_vec(), rock_at(&solids), &rules);
        match check.fragments.as_slice() {
            [] => false,
            [fragment] => {
                assert_eq!(fragment.voxels, [touch]);
                true
            }
            more => panic!("{} fragments", more.len()),
        }
    }

    #[test]
    fn voxels_touching_by_an_edge_join_under_edges_and_corners() {
        // Diagonal to the rock's voxel (3, 3, 3) across an edge.
        let touch = IVec3::new(4, 4, 3);
        let seeds = [IVec3::new(4, 3, 3)];

        assert!(cut_free(touch, &seeds, Connectivity::Faces));
        assert!(!cut_free(touch, &seeds, Connectivity::Edges));
        assert!(!cut_free(touch, &seeds, Connectivity::Corners));
    }

    #[test]
    fn voxels_touching_by_a_corner_join_only_under_corners() {
        // Diagonal to the rock's voxel (3, 3, 3) across a corner.
        let touch = IVec3::new(4, 4, 4);
        let seeds = [IVec3::new(4, 3, 3), IVec3::new(4, 4, 3)];

        assert!(cut_free(touch, &seeds, Connectivity::Faces));
        assert!(cut_free(touch, &seeds, Connectivity::Edges));
        assert!(!cut_free(touch, &seeds, Connectivity::Corners));
    }

    #[test]
    fn a_piece_held_up_only_by_a_non_carrying_material_is_dislodged() {
        const WATER: u8 = 2;
        // A tower on the ground whose third layer an edit turned to water.
        let water: HashSet<IVec3> = cuboid(IVec3::new(0, 2, 0), IVec3::new(3, 3, 3))
            .into_iter()
            .collect();
        let rock: HashSet<IVec3> = cuboid(IVec3::ZERO, IVec3::new(3, 6, 3))
            .into_iter()
            .filter(|p| !water.contains(p))
            .collect();
        let read = |p: IVec3| {
            if water.contains(&p) {
                WorldVoxel::Solid(WATER)
            } else if p.y < 0 || rock.contains(&p) {
                WorldVoxel::Solid(ROCK)
            } else {
                WorldVoxel::Air
            }
        };
        let seeds: Vec<IVec3> = water.iter().copied().collect();

        let check = check_fragments(seeds.clone(), read, &rules(Some(1000)));
        assert!(check.fragments.is_empty());

        let rules = FragmentRules {
            non_carrying: &[WATER],
            ..rules(Some(1000))
        };
        let check = check_fragments(seeds, read, &rules);
        assert_eq!(check.fragments.len(), 1);
        assert_eq!(
            check.fragments[0].voxels,
            sorted(cuboid(IVec3::new(0, 3, 0), IVec3::new(3, 6, 3)))
        );
    }

    #[test]
    fn seeds_beside_the_same_piece_give_one_flood() {
        // A floating rock with dug voxels against two of its faces.
        let solids: HashSet<IVec3> =
            cuboid(IVec3::ZERO, IVec3::splat(4)).into_iter().collect();
        let mut seeds = cuboid(IVec3::new(4, 0, 0), IVec3::new(5, 4, 4));
        seeds.extend(cuboid(IVec3::new(0, 4, 0), IVec3::new(4, 5, 4)));

        let check = check_fragments(seeds, rock_at(&solids), &rules(None));

        // The floods from every voxel the seeds touch join into one, the
        // last one standing, before the rock is flooded through.
        assert!(check.fragments.is_empty());
        assert!(check.visited < 64, "visited {}", check.visited);
    }

    #[test]
    fn a_seed_with_no_carrying_neighbour_starts_no_flood() {
        let solids: HashSet<IVec3> =
            cuboid(IVec3::ZERO, IVec3::splat(4)).into_iter().collect();

        let check =
            check_fragments([IVec3::new(10, 10, 10)], rock_at(&solids), &rules(None));

        assert_eq!(check, FragmentCheck::default());
    }

    #[test]
    fn of_two_equal_pieces_the_one_holding_the_lowest_position_stays() {
        // Two rocks of 8 voxels. `low` holds the lowest y, `left` the lowest
        // x, and the lowest position goes by y first.
        let low = cuboid(IVec3::new(10, 0, 0), IVec3::new(12, 2, 2));
        let left = cuboid(IVec3::new(0, 1, 0), IVec3::new(2, 3, 2));
        let solids: HashSet<IVec3> = low.iter().chain(&left).copied().collect();
        let beside_low = IVec3::new(12, 0, 0);
        let beside_left = IVec3::new(2, 1, 0);

        for seeds in [[beside_low, beside_left], [beside_left, beside_low]] {
            let check = check_fragments(seeds, rock_at(&solids), &rules(None));
            assert_eq!(check.fragments.len(), 1);
            assert_eq!(check.fragments[0].voxels, sorted(left.clone()));
        }
    }
}
