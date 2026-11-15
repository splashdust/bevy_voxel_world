//! The meadow, its trees and its lamps, as voxels.
//!
//! The meadow is gentle noise. The trees and lamp posts are built once, into
//! a map from position to voxel, because where each branch's leaves may go
//! depends on every other branch's wood.

use std::collections::{HashMap, HashSet, VecDeque};

use bevy::prelude::*;
use bevy_voxel_world::prelude::*;
use noise::{NoiseFn, Perlin};

// Materials.
pub const GRASS: u8 = 0;
pub const DIRT: u8 = 1;
pub const STONE: u8 = 2;
pub const BARK: u8 = 3;
pub const LEAVES: u8 = 4;
pub const POST: u8 = 5;
pub const LAMP: u8 = 6;

/// The world's voxels: the meadow, and the trees and lamp posts standing on
/// it.
pub struct Scene {
    ground: Perlin,
    detail: Perlin,
    parts: HashMap<IVec3, Part>,
    branches: Vec<Branch>,
    trees: Vec<Tree>,
    lamps: Vec<IVec3>,
}

impl Scene {
    /// Every voxel below this height is stone.
    pub const SOLID_BELOW: i32 = -16;
    /// Every voxel from this height up is air.
    pub const AIR_FROM: i32 = 64;

    /// Builds the trees and lamp posts.
    pub fn new() -> Self {
        let mut scene = Self {
            ground: Perlin::new(SEED),
            detail: Perlin::new(SEED + 1),
            parts: HashMap::new(),
            branches: Vec::new(),
            trees: Vec::new(),
            lamps: Vec::new(),
        };
        // All the wood comes first, so that each branch's leaves can keep
        // clear of every other branch's wood.
        let mut tips = Vec::new();
        for (column, shape) in [
            (IVec2::ZERO, &LARGE),
            (IVec2::new(-27, 17), &SMALL),
            (IVec2::new(22, 27), &SMALL),
        ] {
            scene.grow(column, shape, &mut tips);
        }
        let noise = Perlin::new(SEED + 2);
        for (branch, tip, radius) in tips {
            scene.grow_leaves(branch, tip, radius, &noise);
        }
        for (column, height) in LAMPS {
            scene.stand_lamp(column, height);
        }
        scene
    }

    /// The voxel at `position`: a tree's or a lamp post's, or the meadow's.
    pub fn voxel(&self, position: IVec3) -> WorldVoxel<u8> {
        if position.y < Self::SOLID_BELOW {
            return WorldVoxel::Solid(STONE);
        }
        if position.y >= Self::AIR_FROM {
            return WorldVoxel::Air;
        }
        if let Some(part) = self.parts.get(&position) {
            return WorldVoxel::Solid(part.material);
        }
        match self.surface(position.xz()) - 1 - position.y {
            ..0 => WorldVoxel::Air,
            0 => WorldVoxel::Solid(GRASS),
            1..=3 => WorldVoxel::Solid(DIRT),
            _ => WorldVoxel::Solid(STONE),
        }
    }

    /// The height of the meadow in a column: the voxels below it are ground.
    fn surface(&self, column: IVec2) -> i32 {
        let [x, z] = column.as_dvec2().to_array();
        let height = 2.5 * self.ground.get([x / 56.0, z / 56.0])
            + 0.8 * self.detail.get([x / 19.0, z / 19.0]);
        height.round() as i32
    }

    fn is_ground(&self, position: IVec3) -> bool {
        position.y < self.surface(position.xz())
    }

    /// Adds a tree's wood standing in `column`, and where each of its
    /// branches ends in leaves to `tips`.
    fn grow(
        &mut self,
        column: IVec2,
        shape: &Shape,
        tips: &mut Vec<(usize, IVec3, f32)>,
    ) {
        let base = IVec3::new(column.x, self.surface(column), column.y);
        let trunk = self.branches.len();
        // The middle of the trunk at `height` voxels above its foot, where
        // it leans.
        let middle = |height: i32| {
            let lean = (shape.lean * height.max(0) as f32).round().as_ivec2();
            base + IVec3::new(lean.x, height, lean.y)
        };
        // The trunk flares out a little at its foot, and its roots reach a
        // few voxels into the ground.
        for height in -3..=shape.trunk_height {
            let flare = (2 - height).max(0) as f32 * 0.6;
            let radius = shape.trunk_radius + flare;
            let reach = radius.ceil() as i32;
            for z in -reach..=reach {
                for x in -reach..=reach {
                    if ((x * x + z * z) as f32) <= radius * radius {
                        self.place(middle(height) + IVec3::new(x, 0, z), BARK, trunk);
                    }
                }
            }
        }
        self.branches.push(Branch {
            path: run(middle(-3), middle(shape.trunk_height)),
            thin: false,
        });
        for limb in shape.limbs {
            let start = middle(limb.height);
            let end = start + heading(limb.angle, limb.rise, limb.length);
            let index = self.wood(start, end, limb.width);
            if limb.forks.is_empty() {
                tips.push((index, end, shape.leaves));
            }
            for fork in limb.forks {
                let path = &self.branches[index].path;
                let from = path[((path.len() - 1) as f32 * fork.at) as usize];
                let to = from + heading(limb.angle + fork.turn, fork.rise, fork.length);
                let fork = self.wood(from, to, 2);
                tips.push((fork, to, shape.leaves));
            }
        }
        self.trees.push(Tree {
            base,
            branches: trunk..self.branches.len(),
        });
    }

    /// Adds a branch of wood `width` voxels thick from `from` to `to`, and
    /// returns its index.
    ///
    /// The branch is a cube of wood at each position of a run that steps
    /// along one axis at a time, so its voxels share faces however it slants.
    /// Where it grows out of other wood, that wood stays theirs.
    fn wood(&mut self, from: IVec3, to: IVec3, width: i32) -> usize {
        let index = self.branches.len();
        let path = run(from, to);
        let low = -(width - 1) / 2;
        for step in &path {
            for z in low..low + width {
                for y in low..low + width {
                    for x in low..low + width {
                        self.place(*step + IVec3::new(x, y, z), BARK, index);
                    }
                }
            }
        }
        self.branches.push(Branch {
            path,
            thin: width <= 2,
        });
        index
    }

    fn place(&mut self, position: IVec3, material: u8, branch: usize) {
        self.parts.entry(position).or_insert(Part {
            material,
            branch: Some(branch),
        });
    }

    /// Adds a cluster of leaves around `tip`, the end of `branch`.
    ///
    /// A leaf goes only where none of its faces touches the ground or
    /// anything of another branch, and the cluster keeps only the leaves
    /// joined to the branch's own wood through each other. So each cluster
    /// belongs to its own branch alone: the default `Connectivity::Faces`
    /// finds a branch cut off with all its leaves, and leaves pressed against
    /// another branch's don't join the two.
    ///
    /// Leaves grow only nearer the tip than the branch's middle, so the
    /// inner half of the branch is bare wood, and a cut there takes all of
    /// them.
    fn grow_leaves(&mut self, branch: usize, tip: IVec3, radius: f32, noise: &Perlin) {
        let reach = (radius * 1.2).ceil() as i32;
        let path = &self.branches[branch].path;
        let middle = path[path.len() / 2];
        let clear = |position: IVec3| {
            !self.is_ground(position)
                && self
                    .parts
                    .get(&position)
                    .is_none_or(|part| part.branch == Some(branch))
        };
        let mut room = HashSet::new();
        for z in -reach..=reach {
            for y in -reach..=reach {
                for x in -reach..=reach {
                    let offset = IVec3::new(x, y, z);
                    let position = tip + offset;
                    let wobble = noise.get((position.as_dvec3() / 3.0).to_array()) as f32;
                    // Clusters are flatter than they are wide, so a
                    // branch that lands on its leaves rolls less far.
                    let stretched = offset.as_vec3() * Vec3::new(1.0, 1.6, 1.0);
                    if stretched.length() <= radius * (1.0 + 0.2 * wobble)
                        && offset.length_squared() < (position - middle).length_squared()
                        && !self.parts.contains_key(&position)
                        && clear(position)
                        && FACES.iter().all(|face| clear(position + *face))
                    {
                        room.insert(position);
                    }
                }
            }
        }
        let own_wood = |position: IVec3| {
            self.parts
                .get(&position)
                .is_some_and(|part| part.branch == Some(branch))
        };
        let mut queue: VecDeque<IVec3> = room
            .iter()
            .copied()
            .filter(|position| FACES.iter().any(|face| own_wood(*position + *face)))
            .collect();
        for position in &queue {
            room.remove(position);
        }
        while let Some(leaf) = queue.pop_front() {
            self.parts.insert(
                leaf,
                Part {
                    material: LEAVES,
                    branch: Some(branch),
                },
            );
            for face in FACES {
                if room.remove(&(leaf + face)) {
                    queue.push_back(leaf + face);
                }
            }
        }
    }

    /// Adds a post `height` voxels tall in `column`, with a lamp on top.
    fn stand_lamp(&mut self, column: IVec2, height: i32) {
        let foot = IVec3::new(column.x, self.surface(column), column.y);
        for y in 0..height {
            let post = Part {
                material: POST,
                branch: None,
            };
            self.parts.insert(foot + IVec3::Y * y, post);
        }
        let lamp = foot + IVec3::Y * height;
        let part = Part {
            material: LAMP,
            branch: None,
        };
        self.parts.insert(lamp, part);
        self.lamps.push(lamp);
    }
}

const SEED: u32 = 3;

const FACES: [IVec3; 6] = [
    IVec3::X,
    IVec3::NEG_X,
    IVec3::Y,
    IVec3::NEG_Y,
    IVec3::Z,
    IVec3::NEG_Z,
];

/// The positions from `from` to `to`, each one voxel along one axis from the
/// one before, keeping close to the straight line between them.
fn run(from: IVec3, to: IVec3) -> Vec<IVec3> {
    let total = (to - from).abs();
    let sign = (to - from).signum();
    let mut done = IVec3::ZERO;
    let mut path = vec![from];
    while done != total {
        // The axis furthest behind its share of the way steps next.
        let share = |axis: usize| (done[axis] as f32 + 0.5) / total[axis] as f32;
        let axis = (0..3)
            .filter(|axis| done[*axis] < total[*axis])
            .min_by(|a, b| share(*a).total_cmp(&share(*b)))
            .unwrap();
        done[axis] += 1;
        path.push(from + done * sign);
    }
    path
}

/// The offset `length` voxels long that heads `angle` degrees round from +x
/// towards +z, climbing `rise` voxels for each voxel it goes sideways.
fn heading(angle: f32, rise: f32, length: f32) -> IVec3 {
    let (sin, cos) = angle.to_radians().sin_cos();
    (Vec3::new(cos, rise, sin).normalize() * length)
        .round()
        .as_ivec3()
}

/// A tree's sizes.
struct Shape {
    trunk_radius: f32,
    trunk_height: i32,
    /// How far the trunk leans along x and z for each voxel it rises.
    lean: Vec2,
    limbs: &'static [Limb],
    /// The radius of the cluster of leaves at each branch's tip.
    leaves: f32,
}

/// A run of wood out of the trunk.
struct Limb {
    /// The height on the trunk it grows from.
    height: i32,
    /// Where it heads, in degrees round from +x towards +z.
    angle: f32,
    /// How many voxels it climbs for each voxel it goes sideways.
    rise: f32,
    length: f32,
    width: i32,
    /// The thin branches it forks into. A limb with none ends in leaves.
    forks: &'static [Fork],
}

/// A thin branch out of a limb, which ends in leaves.
struct Fork {
    /// How far along the limb it grows from, from 0 to 1.
    at: f32,
    /// How far it turns from the limb's heading, in degrees.
    turn: f32,
    rise: f32,
    length: f32,
}

const FORKS: &[Fork] = &[
    Fork {
        at: 0.45,
        turn: 55.0,
        rise: 1.3,
        length: 10.0,
    },
    Fork {
        at: 1.0,
        turn: -35.0,
        rise: 0.8,
        length: 8.0,
    },
];

const fn limb(height: i32, angle: f32, length: f32) -> Limb {
    Limb {
        height,
        angle,
        rise: 0.6,
        length,
        width: 3,
        forks: FORKS,
    }
}

/// The large tree: a thick trunk, and four limbs and a leader at its top,
/// which fork into thin branches.
///
/// The trunk leans and the limb on that side reaches furthest, so a tree
/// cut through near its foot has its weight past the stump and tips over.
const LARGE: Shape = Shape {
    trunk_radius: 4.2,
    lean: Vec2::new(0.3, 0.1),
    trunk_height: 20,
    limbs: &[
        limb(12, 20.0, 16.0),
        limb(16, 110.0, 11.0),
        limb(13, 200.0, 9.0),
        limb(17, 290.0, 11.0),
        Limb {
            height: 20,
            angle: 200.0,
            rise: 5.0,
            length: 8.0,
            width: 3,
            forks: &[
                Fork {
                    at: 1.0,
                    turn: 0.0,
                    rise: 0.8,
                    length: 6.0,
                },
                Fork {
                    at: 1.0,
                    turn: 180.0,
                    rise: 0.8,
                    length: 6.0,
                },
            ],
        },
    ],
    leaves: 5.0,
};

const fn thin_limb(height: i32, angle: f32) -> Limb {
    Limb {
        height,
        angle,
        rise: 0.8,
        length: 8.0,
        width: 2,
        forks: &[],
    }
}

/// A small tree: a thin trunk whose limbs are thin branches.
const SMALL: Shape = Shape {
    trunk_radius: 2.0,
    lean: Vec2::ZERO,
    trunk_height: 9,
    limbs: &[
        thin_limb(4, 30.0),
        thin_limb(5, 150.0),
        thin_limb(6, 270.0),
        Limb {
            height: 9,
            angle: 0.0,
            rise: 8.0,
            length: 6.0,
            width: 2,
            forks: &[],
        },
    ],
    leaves: 4.0,
};

/// The lamp posts: their columns and how tall they stand. The first three
/// stand under the large tree's limbs.
const LAMPS: [(IVec2, i32); 8] = [
    (IVec2::new(9, -4), 5),
    (IVec2::new(-6, 9), 4),
    (IVec2::new(-7, -8), 6),
    (IVec2::new(15, 10), 4),
    (IVec2::new(-16, -12), 5),
    (IVec2::new(27, -5), 3),
    (IVec2::new(-21, 11), 3),
    (IVec2::new(27, 22), 4),
];

/// A solid voxel the scene places on the meadow.
#[derive(Clone, Copy)]
struct Part {
    material: u8,
    /// The branch it belongs to, for wood and leaves.
    branch: Option<usize>,
}

/// One run of wood: a trunk, a limb or a branch.
struct Branch {
    /// The positions the run steps through, from where it grows to its tip.
    path: Vec<IVec3>,
    /// Whether one click through its middle cuts it off.
    #[cfg_attr(not(test), expect(dead_code, reason = "the tests cut thin branches"))]
    thin: bool,
}

/// Where a tree stands, and the branches that belong to it, which the tests
/// cut as a player would.
#[cfg_attr(not(test), expect(dead_code, reason = "the tests cut each tree"))]
struct Tree {
    base: IVec3,
    branches: std::ops::Range<usize>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CARVE_RADIUS;

    /// The solid voxels in a box around one tree, down into the ground
    /// under it, leaving out other trees.
    fn around(scene: &Scene, tree: &Tree) -> HashSet<IVec3> {
        let (mut min, mut max) = (tree.base, tree.base);
        for branch in &scene.branches[tree.branches.clone()] {
            for step in &branch.path {
                min = min.min(*step);
                max = max.max(*step);
            }
        }
        let (min, max) = (min - IVec3::new(8, 4, 8), max + IVec3::splat(8));
        let mut solid = HashSet::new();
        for z in min.z..=max.z {
            for y in min.y..=max.y {
                for x in min.x..=max.x {
                    let position = IVec3::new(x, y, z);
                    let theirs = branch_of(scene, position)
                        .is_some_and(|branch| !tree.branches.contains(&branch));
                    if scene.voxel(position).is_solid() && !theirs {
                        solid.insert(position);
                    }
                }
            }
        }
        solid
    }

    /// Carves a sphere as a click does.
    fn carve(solid: &mut HashSet<IVec3>, centre: IVec3) {
        solid.retain(|voxel| {
            (*voxel - centre).length_squared() > CARVE_RADIUS * CARVE_RADIUS
        });
    }

    /// The voxels of `solid` that no run of face-joined voxels joins to the
    /// lowest of them, which is in the ground under the tree.
    fn cut_off(solid: &HashSet<IVec3>) -> HashSet<IVec3> {
        let start = *solid.iter().min_by_key(|voxel| voxel.y).unwrap();
        let mut held = HashSet::from([start]);
        let mut queue = VecDeque::from([start]);
        while let Some(voxel) = queue.pop_front() {
            for step in [IVec3::X, IVec3::Y, IVec3::Z] {
                for next in [voxel + step, voxel - step] {
                    if solid.contains(&next) && held.insert(next) {
                        queue.push_back(next);
                    }
                }
            }
        }
        solid.difference(&held).copied().collect()
    }

    fn branch_of(scene: &Scene, voxel: IVec3) -> Option<usize> {
        scene.parts.get(&voxel).and_then(|part| part.branch)
    }

    #[test]
    fn every_tree_stands_joined_to_the_ground() {
        let scene = Scene::new();
        assert!(scene.trees.len() >= 3);
        for tree in &scene.trees {
            let loose = cut_off(&around(&scene, tree));
            assert!(loose.is_empty(), "{} voxels hang loose", loose.len());
        }
    }

    #[test]
    fn a_click_through_a_thin_branch_cuts_it_off_with_its_own_leaves_only() {
        let scene = Scene::new();
        let mut thin = 0;
        for tree in &scene.trees {
            let solid = around(&scene, tree);
            for index in tree.branches.clone() {
                let branch = &scene.branches[index];
                if !branch.thin {
                    continue;
                }
                thin += 1;
                let mut cut = solid.clone();
                carve(&mut cut, branch.path[branch.path.len() / 2]);
                let loose = cut_off(&cut);
                let leaves: HashSet<IVec3> = scene
                    .parts
                    .iter()
                    .filter(|(_, part)| {
                        part.material == LEAVES && part.branch == Some(index)
                    })
                    .map(|(voxel, _)| *voxel)
                    .collect();
                assert!(!leaves.is_empty(), "branch {index} has no leaves");
                assert!(
                    loose.is_superset(&leaves),
                    "branch {index} leaves leaves behind"
                );
                assert!(
                    loose
                        .iter()
                        .all(|voxel| branch_of(&scene, *voxel) == Some(index)),
                    "cutting branch {index} takes another branch's voxels"
                );
            }
        }
        assert!(thin >= 12, "{thin} thin branches");
    }

    #[test]
    fn one_click_on_the_large_trunk_leaves_it_standing_and_a_ring_of_them_fells_it() {
        let scene = Scene::new();
        let tree = &scene.trees[0];
        let mut solid = around(&scene, tree);
        // The middle of the trunk, four voxels above its foot.
        let trunk: Vec<IVec3> = scene
            .parts
            .iter()
            .filter(|(voxel, part)| {
                voxel.y == tree.base.y + 4 && part.branch == Some(tree.branches.start)
            })
            .map(|(voxel, _)| *voxel)
            .collect();
        let middle =
            trunk.iter().map(|voxel| voxel.as_vec3()).sum::<Vec3>() / trunk.len() as f32;
        let height = middle.round().as_ivec3();
        // The first click lands on the trunk's side, facing +x.
        let side = (0..16)
            .map(|x| height + IVec3::X * x)
            .take_while(|voxel| solid.contains(voxel))
            .last()
            .unwrap();
        carve(&mut solid, side);
        assert!(cut_off(&solid).is_empty());
        // Then the clicks go round the trunk and through its middle.
        for turn in 0..8 {
            let angle = turn as f32 * std::f32::consts::TAU / 8.0;
            let radius = (side - height).x as f32;
            let offset = Vec3::new(angle.cos(), 0.0, angle.sin()) * radius;
            carve(&mut solid, height + offset.round().as_ivec3());
        }
        for x in [-2, 0, 2] {
            for z in [-2, 0, 2] {
                carve(&mut solid, height + IVec3::new(x, 0, z));
            }
        }
        let felled = cut_off(&solid);
        // Over the smallest limit the panel offers and under the default.
        assert!(
            (4_096..32_768).contains(&felled.len()),
            "{} voxels felled",
            felled.len()
        );
        // The tree leans, so its weight hangs past the stump and it tips
        // over rather than dropping onto the stump and standing there.
        let weight = felled.iter().map(|voxel| voxel.as_vec3()).sum::<Vec3>()
            / felled.len() as f32;
        let stump = trunk
            .iter()
            .map(|voxel| (voxel.as_vec3() - middle).xz().length())
            .fold(0.0, f32::max);
        let overhang = (weight - middle).xz().length();
        assert!(
            overhang > stump,
            "the weight hangs {overhang} out, the stump is {stump} wide"
        );
    }

    #[test]
    fn lamps_stand_apart_from_the_trees_and_some_stand_under_the_canopy() {
        let scene = Scene::new();
        let trunk = scene.trees[0].base;
        let mut under = 0;
        let mut beyond = 0;
        for lamp in &scene.lamps {
            assert_eq!(scene.voxel(*lamp), WorldVoxel::Solid(LAMP));
            // Nothing of a tree above the ground within two voxels of the
            // lamp or its post.
            for y in -8..=2 {
                for z in -2..=2 {
                    for x in -2..=2 {
                        let near = *lamp + IVec3::new(x, y, z);
                        assert!(
                            scene.is_ground(near) || branch_of(&scene, near).is_none(),
                            "lamp at {lamp} touches a tree"
                        );
                    }
                }
            }
            if (1..32).any(|y| branch_of(&scene, *lamp + IVec3::Y * y).is_some()) {
                under += 1;
            }
            if (lamp.xz() - trunk.xz()).as_vec2().length() > 20.0 {
                beyond += 1;
            }
        }
        assert!(under >= 2, "{under} lamps under a canopy");
        assert!(beyond >= 1, "{beyond} lamps beyond the large tree");
    }
}
