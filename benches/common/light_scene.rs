//! The light reference scene, shared by the benches: hills over caves,
//! shafts, a sealed room and a floating island, with an emissive voxel on
//! one in 32 exposed floor voxels and on the floors of five torch columns.
//!
//! [`SceneTable`] samples a box of it into a bit per voxel, so a voxel
//! world can read it with cheap lookups.

#![allow(dead_code)]

use std::collections::HashMap;
use std::sync::Mutex;

use bevy::prelude::*;
use noise::{NoiseFn, Perlin};

/// Above this height the scene holds no solid voxel.
pub const SCENE_TOP: i32 = 200;

const CAVE_FLOOR: i32 = -480;

/// A box, minimum inclusive and maximum exclusive.
#[derive(Clone, Copy)]
struct Bounds([i32; 3], [i32; 3]);

impl Bounds {
    fn contains(&self, p: IVec3) -> bool {
        (0..3).all(|axis| p[axis] >= self.0[axis] && p[axis] < self.1[axis])
    }
}

struct Ellipsoid([f64; 3], [f64; 3]);

impl Ellipsoid {
    fn contains(&self, p: IVec3) -> bool {
        (0..3)
            .map(|axis| ((p[axis] as f64 + 0.5 - self.0[axis]) / self.1[axis]).powi(2))
            .sum::<f64>()
            < 1.0
    }
}

/// A disc of ground forced to one height: x, z, flat radius, blend radius,
/// height.
type Plateau = (f64, f64, f64, f64, f64);

const PLATEAUS: [Plateau; 6] = [
    (0.5, 0.5, 24.0, 190.0, 160.0),
    (-319.5, 0.5, 20.0, 120.0, 120.0),
    (320.5, 0.5, 64.0, 150.0, 40.0),
    (0.0, 360.0, 90.0, 170.0, 32.0),
    (-340.0, 360.0, 90.0, 170.0, 30.0),
    (-320.0, 128.0, 16.0, 56.0, 48.0),
];
const DEEP_CAVERN: Ellipsoid = Ellipsoid([0.5, -176.0, 0.5], [110.0, 36.0, 110.0]);
const REF_CAVERN: Ellipsoid = Ellipsoid([-319.5, -10.0, 0.5], [70.0, 30.0, 70.0]);
const POCKET: Bounds = Bounds([-326, 40, 122], [-314, 46, 134]);
const TORCH_ROOM: Bounds = Bounds([300, -40, -340], [340, -20, -300]);
const MESA: Bounds = Bounds([-48, 32, 290], [48, 84, 340]);
const SLAB: Bounds = Bounds([-48, 72, 340], [48, 84, 404]);
const ISLAND: Bounds = Bounds([-390, 150, 310], [-290, 176, 410]);
const QUIET: [Bounds; 7] = [
    Bounds([-126, -228, -126], [126, -124, 126]),
    Bounds([-16, -140, -16], [16, 168, 16]),
    Bounds([-406, -56, -86], [-234, 36, 86]),
    Bounds([-336, 20, -16], [-304, 128, 16]),
    Bounds([-342, 24, 112], [-298, 60, 144]),
    Bounds([256, -304, -64], [384, 48, 64]),
    Bounds([284, -56, -356], [356, -4, -284]),
];
const TORCH_COLUMNS: [(i32, i32); 5] =
    [(70, 0), (-70, 0), (0, 70), (0, -70), (320, -320)];

/// The light reference scene, with an emissive voxel on one in 32 exposed
/// floor voxels.
pub struct Scene {
    hills: Perlin,
    region: Perlin,
    big: Perlin,
    worm_a: Perlin,
    worm_b: Perlin,
    surfaces: HashMap<(i32, i32), i32>,
}

impl Scene {
    pub fn new() -> Self {
        Self {
            hills: Perlin::new(11),
            region: Perlin::new(19),
            big: Perlin::new(23),
            worm_a: Perlin::new(37),
            worm_b: Perlin::new(41),
            surfaces: HashMap::new(),
        }
    }

    /// The height of the ground at `x`, `z`: the first voxel above it.
    pub fn surface(&mut self, x: i32, z: i32) -> i32 {
        if let Some(height) = self.surfaces.get(&(x, z)) {
            return *height;
        }
        let height = self.surface_at(x, z);
        self.surfaces.insert((x, z), height);
        height
    }

    /// [`surface`](Self::surface), worked out again on every call.
    fn surface_at(&self, x: i32, z: i32) -> i32 {
        let (fx, fz) = (x as f64, z as f64);
        let mut height = 32.0
            + self.hills.get([fx / 160.0, fz / 160.0]) * 18.0
            + self.hills.get([fx / 47.0, fz / 47.0]) * 4.0;
        for (px, pz, flat, blend, level) in PLATEAUS {
            let t = (((fx - px).hypot(fz - pz) - flat) / (blend - flat)).clamp(0.0, 1.0);
            let t = t * t * (3.0 - 2.0 * t);
            height = level + (height - level) * t;
        }
        height.floor() as i32
    }

    fn noise_cave(&self, p: IVec3) -> bool {
        let (x, y, z) = (p.x as f64, p.y as f64, p.z as f64);
        if self.region.get([x / 220.0, y / 160.0, z / 220.0]) < -0.05 {
            return false;
        }
        if self.big.get([x / 72.0, y / 48.0, z / 72.0]) > 0.42 {
            return true;
        }
        let at = [x / 36.0, y / 28.0, z / 36.0];
        self.worm_a.get(at).abs() < 0.04 && self.worm_b.get(at).abs() < 0.04
    }

    fn placed_hollow(&self, p: IVec3) -> bool {
        (p.x == 0 && p.z == 0 && p.y >= -141)
            || (p.x == -320 && p.z == 0 && p.y >= 19)
            || (p.x == 320 && p.z == 0 && p.y >= 32)
            || DEEP_CAVERN.contains(p)
            || REF_CAVERN.contains(p)
            || POCKET.contains(p)
            || TORCH_ROOM.contains(p)
            || (p.y >= -288 && p.y < 32 && (p.x - 320).pow(2) + p.z.pow(2) < 48 * 48)
    }

    fn floating_solid(&self, p: IVec3) -> bool {
        if MESA.contains(p) || SLAB.contains(p) {
            return true;
        }
        ISLAND.contains(p)
            && p.y as f64
                >= 160.0 + 8.0 * self.hills.get([p.x as f64 / 16.0, p.z as f64 / 16.0])
    }

    pub fn open(&mut self, p: IVec3) -> bool {
        let surface = self.surface(p.x, p.z);
        self.open_over(p, surface)
    }

    /// Whether the voxel at `p` is open, in a column whose ground ends at
    /// `surface`.
    fn open_over(&self, p: IVec3, surface: i32) -> bool {
        if p.y >= surface {
            return !self.floating_solid(p);
        }
        self.placed_hollow(p)
            || (p.y >= CAVE_FLOOR
                && !QUIET.iter().any(|quiet| quiet.contains(p))
                && self.noise_cave(p))
    }

    /// Whether the voxel at `p`, which must be solid, is emissive.
    pub fn emissive(&mut self, p: IVec3) -> bool {
        self.open(p + IVec3::Y) && lamp_at(p)
    }
}

/// Whether a solid voxel at `p` with an open voxel above it is emissive.
fn lamp_at(p: IVec3) -> bool {
    TORCH_COLUMNS.contains(&(p.x, p.z))
        || (hash3(p).is_multiple_of(32) && !QUIET.iter().any(|quiet| quiet.contains(p)))
}

/// A box of the scene sampled into one bit per voxel, set where it's open.
/// Lookups outside the box ask the scene itself, which is far slower.
pub struct SceneTable {
    scene: Scene,
    min: IVec3,
    size: IVec3,
    bits: Vec<u64>,
}

impl SceneTable {
    /// Samples the box from `min` to `max`, maximum exclusive, on `threads`
    /// threads.
    pub fn new(min: IVec3, max: IVec3, threads: usize) -> Self {
        let scene = Scene::new();
        let size = max - min;
        let (w, h, d) = (size.x as usize, size.y as usize, size.z as usize);
        let mut surfaces = vec![0; w * d];
        for z in 0..d {
            for x in 0..w {
                surfaces[x + z * w] =
                    scene.surface_at(min.x + x as i32, min.z + z as i32);
            }
        }
        // One slab of whole words per z, so threads never share a word.
        let words_per_slab = (w * h).div_ceil(64);
        let mut bits = vec![0u64; words_per_slab * d];
        {
            let slabs = Mutex::new(bits.chunks_mut(words_per_slab).enumerate());
            let (scene, surfaces) = (&scene, &surfaces);
            std::thread::scope(|scope| {
                for _ in 0..threads {
                    scope.spawn(|| loop {
                        let Some((z, slab)) = slabs.lock().unwrap().next() else {
                            return;
                        };
                        for y in 0..h {
                            for x in 0..w {
                                let p = min + IVec3::new(x as i32, y as i32, z as i32);
                                if scene.open_over(p, surfaces[x + z * w]) {
                                    let bit = x + y * w;
                                    slab[bit / 64] |= 1 << (bit % 64);
                                }
                            }
                        }
                    });
                }
            });
        }
        Self {
            scene,
            min,
            size,
            bits,
        }
    }

    /// Whether the voxel at `p` is open.
    #[inline]
    pub fn open(&self, p: IVec3) -> bool {
        let local = p - self.min;
        if local.cmplt(IVec3::ZERO).any() || local.cmpge(self.size).any() {
            if p.y >= SCENE_TOP {
                return true;
            }
            return self.scene.open_over(p, self.scene.surface_at(p.x, p.z));
        }
        let (w, h) = (self.size.x as usize, self.size.y as usize);
        let words_per_slab = (w * h).div_ceil(64);
        let bit = local.x as usize + local.y as usize * w;
        self.bits[local.z as usize * words_per_slab + bit / 64] >> (bit % 64) & 1 == 1
    }

    /// Whether the voxel at `p`, which must be solid, is emissive.
    #[inline]
    pub fn emissive(&self, p: IVec3) -> bool {
        lamp_at(p) && self.open(p + IVec3::Y)
    }
}

fn hash3(p: IVec3) -> u32 {
    let mut h = (p.x as u32).wrapping_mul(0x8da6_b343)
        ^ (p.y as u32).wrapping_mul(0xd816_3841)
        ^ (p.z as u32).wrapping_mul(0xcb1a_b31f);
    h ^= h >> 13;
    h = h.wrapping_mul(0x5bd1_e995);
    h ^ (h >> 15)
}
