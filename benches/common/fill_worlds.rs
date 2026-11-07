//! Worlds for comparing the models of sky fill, shared by `chunk_lifecycle`
//! and the `light_caves` example's pixel checks: floating islands at several
//! heights, overhangs and arches, caves with narrow and wide mouths, sparse
//! floating voxels with no ground, and buildings with rooms inside.
//!
//! Each world is a pure function of position, repeating along x and z, and
//! holds the places where sky fill decides what a player sees: the ground
//! under islands, the ground under overhangs, the inside of caves and rooms.
//! Two of them hold a space sealed from the sky, which every model must
//! leave exactly black.

#![allow(dead_code)]

use bevy::prelude::*;

/// One of the worlds.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FillScene {
    /// Flat ground below y = 0, with a slab of stone 64 wide and 8 thick
    /// every 128 voxels along x and z, its underside 12, 32, 72 or 160 voxels
    /// over the ground.
    Islands,
    /// Flat ground with a mesa every 128 voxels along x and z, a slab
    /// sticking 48 voxels out of its top 40 voxels up, a shelf sticking 32
    /// voxels out of its side 8 voxels up, and an arch beside it.
    Arches,
    /// Flat ground with a hill 64 voxels tall every 256 voxels along x and
    /// z. A tunnel with a mouth 24 wide and 16 tall leads into a chamber
    /// wider than it, a mouth 3 by 3 into another chamber, and a third
    /// chamber has no opening.
    Caves,
    /// No ground: islands on a jittered lattice of 192 by 160 by 192 voxels,
    /// with lone solid voxels scattered through the air, about 8 per chunk.
    Pebbles,
    /// Flat ground with a building every 96 voxels along x and z, three
    /// storeys with windows, a doorway and hatches between the storeys, and
    /// beside it a room with no opening.
    Interiors,
}

pub const FILL_SCENES: [FillScene; 5] = [
    FillScene::Islands,
    FillScene::Arches,
    FillScene::Caves,
    FillScene::Pebbles,
    FillScene::Interiors,
];

impl FillScene {
    pub fn name(self) -> &'static str {
        match self {
            FillScene::Islands => "islands",
            FillScene::Arches => "arches",
            FillScene::Caves => "caves",
            FillScene::Pebbles => "pebbles",
            FillScene::Interiors => "interiors",
        }
    }

    /// Whether the voxel at `p` is solid.
    pub fn solid(self, p: IVec3) -> bool {
        match self {
            FillScene::Islands => islands(p),
            FillScene::Arches => arches(p),
            FillScene::Caves => caves(p),
            FillScene::Pebbles => pebbles(p),
            FillScene::Interiors => interiors(p),
        }
    }

    /// Where the pixel checks look, each a camera position and the point it
    /// looks at.
    pub fn shots(self) -> Vec<FillShot> {
        let shot = |name, at: [f32; 3], target: [f32; 3], sealed| FillShot {
            name,
            at: Vec3::from_array(at),
            target: Vec3::from_array(target),
            sealed,
        };
        match self {
            FillScene::Islands => vec![
                shot(
                    "under_low_island",
                    [48.0, 5.0, 100.0],
                    [48.0, 0.0, 48.0],
                    false,
                ),
                shot(
                    "under_mid_island",
                    [176.0, 8.0, 100.0],
                    [176.0, 0.0, 48.0],
                    false,
                ),
                shot(
                    "under_high_island",
                    [432.0, 12.0, 110.0],
                    [432.0, 0.0, 48.0],
                    false,
                ),
                shot(
                    "island_heights",
                    [-40.0, 60.0, 140.0],
                    [240.0, 20.0, 48.0],
                    false,
                ),
            ],
            FillScene::Arches => vec![
                shot("overhang_lip", [64.0, 8.0, 64.0], [60.0, 2.0, 10.0], false),
                shot("low_shelf", [20.0, 4.0, 96.0], [20.0, 2.0, 50.0], false),
                shot("under_arch", [92.0, 8.0, 112.0], [92.0, 2.0, 80.0], false),
            ],
            FillScene::Caves => vec![
                shot("wide_mouth", [8.0, 8.0, 60.0], [60.0, 6.0, 60.0], false),
                shot(
                    "behind_wide_mouth",
                    [116.0, 12.0, 60.0],
                    [170.0, 4.0, 100.0],
                    false,
                ),
                shot(
                    "narrow_mouth",
                    [100.0, 10.0, 160.0],
                    [40.0, 2.0, 141.0],
                    false,
                ),
                shot(
                    "sealed_cave",
                    [168.0, 22.0, 168.0],
                    [190.0, 14.0, 190.0],
                    true,
                ),
            ],
            FillScene::Pebbles => {
                let (centre, radius, height) = pebble_island(0, 0, 0);
                let below = centre - Vec3::Y * (height + 24.0);
                vec![shot(
                    "under_pebble_island",
                    (below + Vec3::Z * (radius + 16.0)).to_array(),
                    (centre - Vec3::Y * height).to_array(),
                    false,
                )]
            }
            FillScene::Interiors => vec![
                shot("ground_floor", [12.0, 4.0, 28.0], [44.0, 2.0, 28.0], false),
                shot(
                    "upper_storey",
                    [42.0, 12.0, 42.0],
                    [14.0, 10.0, 14.0],
                    false,
                ),
                shot("sealed_room", [64.0, 5.0, 64.0], [80.0, 3.0, 80.0], true),
            ],
        }
    }
}

/// Where a pixel check looks, and whether everything it sees is sealed
/// from the sky, so must be exactly black.
pub struct FillShot {
    pub name: &'static str,
    pub at: Vec3,
    pub target: Vec3,
    pub sealed: bool,
}

/// Whether `p` lies in `lo..hi` on each axis.
fn inside(p: IVec3, lo: [i32; 3], hi: [i32; 3]) -> bool {
    (0..3).all(|axis| p[axis] >= lo[axis] && p[axis] < hi[axis])
}

/// `p` within the repeating tile of `size` voxels along x and z.
fn tile(p: IVec3, size: i32) -> (IVec3, IVec2) {
    let cell = IVec2::new(p.x.div_euclid(size), p.z.div_euclid(size));
    (
        IVec3::new(p.x.rem_euclid(size), p.y, p.z.rem_euclid(size)),
        cell,
    )
}

/// The heights of the islands' undersides, by tile: tile (n, 0) gets entry
/// n % 4 along x.
pub const ISLAND_HEIGHTS: [i32; 4] = [12, 32, 72, 160];

fn islands(p: IVec3) -> bool {
    let (local, cell) = tile(p, 128);
    let bottom = ISLAND_HEIGHTS[(cell.x + 2 * cell.y).rem_euclid(4) as usize];
    p.y < 0 || inside(local, [16, bottom, 16], [80, bottom + 8, 80])
}

fn arches(p: IVec3) -> bool {
    let (local, _) = tile(p, 128);
    p.y < 0
        // The mesa, and the slab sticking out of its top along x.
        || inside(local, [0, 0, 0], [40, 48, 40])
        || inside(local, [40, 40, 0], [88, 48, 40])
        // The shelf sticking out of its side along z.
        || inside(local, [0, 8, 40], [40, 16, 72])
        // The arch: two pillars and the deck over them.
        || inside(local, [64, 0, 72], [72, 32, 88])
        || inside(local, [112, 0, 72], [120, 32, 88])
        || inside(local, [64, 24, 72], [120, 32, 88])
}

fn caves(p: IVec3) -> bool {
    let (local, _) = tile(p, 256);
    if p.y < 0 {
        return true;
    }
    if !inside(local, [32, 0, 32], [224, 64, 224]) {
        return false;
    }
    let open = [
        // The wide mouth and its tunnel, then a chamber wider than both.
        ([32, 0, 48], [112, 16, 72]),
        ([112, 0, 40], [176, 24, 112]),
        // The narrow mouth and its chamber.
        ([32, 0, 140], [40, 3, 143]),
        ([40, 0, 128], [112, 20, 192]),
        // The sealed chamber.
        ([140, 12, 140], [196, 32, 196]),
    ];
    !open.iter().any(|(lo, hi)| inside(local, *lo, *hi))
}

/// A hash of a lattice point, for the pebble world.
fn hash3(x: i32, y: i32, z: i32) -> u32 {
    let mut h = (x as u32).wrapping_mul(0x8da6_b343)
        ^ (y as u32).wrapping_mul(0xd816_3841)
        ^ (z as u32).wrapping_mul(0xcb1a_b31f);
    h ^= h >> 13;
    h = h.wrapping_mul(0x5bd1_e995);
    h ^ (h >> 15)
}

/// The pebble world's island in lattice cell `(cx, cy, cz)`: its centre,
/// its radius along x and z and its half height.
pub fn pebble_island(cx: i32, cy: i32, cz: i32) -> (Vec3, f32, f32) {
    let (px, py, pz) = (192, 160, 192);
    let hash = hash3(cx, cy, cz);
    let jitter = |shift: u32| ((hash >> shift) & 31) as i32 - 16;
    let centre = IVec3::new(
        cx * px + px / 2 + jitter(0),
        cy * py + py / 2 + jitter(5),
        cz * pz + pz / 2 + jitter(10),
    );
    let radius = 30.0 + ((hash >> 15) & 31) as f32 * 1.6;
    let height = 8.0 + ((hash >> 20) & 15) as f32 * 1.4;
    (centre.as_vec3(), radius, height)
}

fn pebbles(p: IVec3) -> bool {
    let cell = IVec3::new(
        p.x.div_euclid(192),
        p.y.div_euclid(160),
        p.z.div_euclid(192),
    );
    let (centre, radius, height) = pebble_island(cell.x, cell.y, cell.z);
    let d = (p.as_vec3() - centre) / Vec3::new(radius, height, radius);
    d.length_squared() < 1.0 || hash3(p.x, p.y, p.z).is_multiple_of(4096)
}

fn interiors(p: IVec3) -> bool {
    let (l, _) = tile(p, 96);
    if p.y < 0 {
        return true;
    }
    // The sealed room: walls, roof and the ground under it closed.
    if inside(l, [60, 0, 60], [84, 12, 84]) {
        return !inside(l, [62, 0, 62], [82, 10, 82]);
    }
    if !inside(l, [8, 0, 8], [48, 26, 48]) {
        return false;
    }
    let storey_y = l.y.rem_euclid(8);
    let in_walls = !inside(l, [10, 0, 10], [46, 26, 46]);
    if in_walls {
        let window = l.y < 24
            && (3..6).contains(&storey_y)
            && ((l.x >= 46 && [20..24, 32..36].iter().any(|r| r.contains(&l.z)))
                || (l.z < 10 && [20..24, 32..36].iter().any(|r| r.contains(&l.x))));
        let door = l.x < 10 && (26..30).contains(&l.z) && l.y < 6;
        return !(window || door);
    }
    // The floors between storeys, with a hatch in each, and the roof.
    let hatch = inside(l, [12, 0, 12], [16, 26, 16]) && l.y < 24;
    l.y >= 24 || ((l.y == 8 || l.y == 16) && !hatch)
}
