//! A house on a mountain ledge at dusk, with a cave beside it, lit by light
//! propagation together with Bevy's atmosphere, fog and lights. Walk through it,
//! and place and remove voxels to see the light follow.
//!
//! Light propagation decides where light can reach. Sky light falls in through
//! the door, the windows and the cave mouth, and voxel light spreads from the
//! lanterns, the hearth and the crystals. Some rooms have lamps, some only
//! windows, and the cellar has neither and stays dark however bright the sky.
//!
//! Bevy does the rest. The atmosphere draws the sky and lights the world through
//! its environment map, which propagated light dims wherever sky light is weak.
//! The sun shines only on faces with direct sky. Voxel light has no colour, so
//! each lamp also has a warm point light, and distance fog and a volume of mist
//! add haze.
//!
//! The trees can be felled. Remove a voxel of a trunk, and the tree above it
//! is cut loose. The world asks the crate to lift such a piece into a
//! sub-grid, which falls under avian3d physics and stays where it lands. The
//! "Felling trees" section below shows how.
//!
//! Run with `cargo run --release --example light_mountain_house --features
//! light_propagation,sub_grids,dislodged_fragments`.
//! Walk with WASD, hold Shift to run, jump with Space and look with the mouse.
//! A left click places a voxel against the face under the crosshair, a right
//! click removes the voxel there or fells a tree, and 1 to 4 choose stone, timber, a lantern or
//! a crystal to place. `[` and `]` lower and raise the sun, L carries a lantern
//! into the dark, H hides the help, and Esc frees the mouse.
//!
//! The scene renders `RENDER_HEIGHT` pixels tall and is stretched over the
//! window, so a larger window, or a high-density display, costs no more to draw.
//!
//! In the browser the example renders through WebGL2, which can't run the
//! atmosphere or volumetric fog. The fog's colour and an ambient light stand in
//! for the atmosphere, and the valley goes without its mist. The world ends a
//! little way past the ledge, the house and the cave, in thicker fog, so that
//! all of it spawns while the page loads. The website's live demo runs this
//! example there.

use std::sync::{Arc, LazyLock};

use avian3d::prelude::{AngularVelocity, Collider, PhysicsPlugins, RigidBody};
use bevy::{
    camera::Exposure,
    camera::RenderTarget,
    core_pipeline::tonemapping::Tonemapping,
    image::ImageSampler,
    input::mouse::AccumulatedMouseMotion,
    light::{
        atmosphere::ScatteringMedium, light_consts::lux, Atmosphere,
        AtmosphereEnvironmentMapLight, CascadeShadowConfigBuilder, FogVolume,
        NotShadowCaster, VolumetricFog, VolumetricLight,
    },
    pbr::AtmosphereSettings,
    platform::collections::HashMap,
    post_process::bloom::Bloom,
    prelude::*,
    render::render_resource::{Extent3d, TextureFormat},
    window::{CursorGrabMode, CursorOptions, PrimaryWindow},
};
use bevy_voxel_world::{custom_meshing::CHUNK_SIZE_I, prelude::*};
use noise::{Fbm, MultiFractal, NoiseFn, Perlin, RidgedMulti};

// Materials.
const GRASS: u8 = 0;
const DIRT: u8 = 1;
const STONE: u8 = 2;
const SNOWY_STONE: u8 = 3;
const COBBLESTONE: u8 = 4;
const DARK_PLANKS: u8 = 5;
const TIMBER: u8 = 6;
const ROOF: u8 = 7;
const LEAVES: u8 = 8;
const BARK: u8 = 9;
const LANTERN: u8 = 10;
const CRYSTAL: u8 = 11;
const GRAVEL: u8 = 12;
const FLOOR: u8 = 13;
const HEARTH: u8 = 14;

// Tiles of `assets/mountain_house_texture.png`, from the top.
const TILE_GRASS_TOP: u32 = 0;
const TILE_GRASS_SIDE: u32 = 1;
const TILE_DIRT: u32 = 2;
const TILE_STONE: u32 = 3;
const TILE_SNOW: u32 = 4;
const TILE_SNOWY_STONE_SIDE: u32 = 5;
const TILE_COBBLESTONE: u32 = 6;
const TILE_DARK_PLANKS: u32 = 7;
const TILE_TIMBER_SIDE: u32 = 8;
const TILE_TIMBER_TOP: u32 = 9;
const TILE_ROOF: u32 = 10;
const TILE_SPRUCE_LEAVES: u32 = 11;
const TILE_SPRUCE_BARK: u32 = 12;
const TILE_LANTERN: u32 = 13;
const TILE_CRYSTAL: u32 = 14;
const TILE_GRAVEL: u32 = 15;
const TILE_FLOOR_PLANKS: u32 = 16;
const TILE_HEARTH: u32 = 17;
const TILE_COUNT: u32 = 18;

/// The tiles of a material's top, sides and bottom.
fn texture_indices(material: u8) -> [u32; 3] {
    match material {
        GRASS => [TILE_GRASS_TOP, TILE_GRASS_SIDE, TILE_DIRT],
        DIRT => [TILE_DIRT; 3],
        STONE => [TILE_STONE; 3],
        SNOWY_STONE => [TILE_SNOW, TILE_SNOWY_STONE_SIDE, TILE_STONE],
        COBBLESTONE => [TILE_COBBLESTONE; 3],
        DARK_PLANKS => [TILE_DARK_PLANKS; 3],
        TIMBER => [TILE_TIMBER_TOP, TILE_TIMBER_SIDE, TILE_TIMBER_TOP],
        ROOF => [TILE_ROOF; 3],
        LEAVES => [TILE_SPRUCE_LEAVES; 3],
        BARK => [TILE_SPRUCE_BARK; 3],
        LANTERN => [TILE_LANTERN; 3],
        CRYSTAL => [TILE_CRYSTAL; 3],
        GRAVEL => [TILE_GRAVEL; 3],
        FLOOR => [TILE_FLOOR_PLANKS; 3],
        HEARTH => [TILE_HEARTH; 3],
        _ => [TILE_STONE; 3],
    }
}

// ---------------------------------------------------------------------------
// The ground

/// The top of the solid ground on the ledge that the house and the cave mouth
/// stand on.
const LEDGE_TOP: i32 = 39;
/// Where the cliff behind the ledge starts, along z.
const CLIFF_Z: f32 = 16.0;

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// The ground's height and steepness at one column.
#[derive(Clone, Copy)]
struct Column {
    height: f32,
    slope: f32,
}

/// The ground: a forested hillside rising to a flat ledge, a mountain rising from
/// a cliff behind the ledge to snowy peaks, and a range of hills across the valley
/// towards +x, which the low sun sets behind.
///
/// A voxel at height `y` is solid when `y < height(x, z)`.
struct Terrain {
    hills: Fbm<Perlin>,
    detail: Fbm<Perlin>,
    peaks: RidgedMulti<Perlin>,
    gullies: RidgedMulti<Perlin>,
    patches: Perlin,
}

impl Terrain {
    fn new() -> Self {
        Self {
            hills: Fbm::<Perlin>::new(11).set_octaves(4),
            detail: Fbm::<Perlin>::new(12).set_octaves(3),
            peaks: RidgedMulti::<Perlin>::new(13).set_octaves(5),
            gullies: RidgedMulti::<Perlin>::new(15).set_octaves(3),
            patches: Perlin::new(14),
        }
    }

    /// How far along x the ledge flattens the ground, from 0 to 1.
    fn along_ledge(x: f32) -> f32 {
        1.0 - smoothstep(0.0, 16.0, (x - 6.0).abs() - 40.0)
    }

    /// How much the ledge flattens the ground at a column, from 0 to 1.
    fn ledge_weight(x: f32, z: f32) -> f32 {
        Self::along_ledge(x) * (1.0 - smoothstep(0.0, 14.0, -14.0 - z))
    }

    fn height(&self, x: f32, z: f32) -> f32 {
        let (xd, zd) = (x as f64, z as f64);
        let ledge_top = LEDGE_TOP as f32 + 0.5;
        let hills = self.hills.get([xd * 0.006, zd * 0.006]) as f32;
        let detail = self.detail.get([xd * 0.05, zd * 0.05]) as f32;
        // How far in front of the cliff a column is. The foot of the mountain
        // wanders, except behind the ledge, and turns away past x = 110 to open
        // the valley.
        let front =
            z + 26.0 * hills * (1.0 - Self::along_ledge(x)) - 0.9 * (x - 110.0).max(0.0);

        // The hillside climbs from the valley floor to the height of the ledge.
        let valley = 9.0 + 6.0 * hills;
        let hill = valley
            + (ledge_top - 1.5 - valley) * smoothstep(-120.0, -14.0, front)
            + 2.0 * detail;
        let hill = hill + (ledge_top - hill) * Self::ledge_weight(x, z);

        // The range across the valley.
        let range_x = 225.0 + 30.0 * hills;
        let range = (66.0 + 16.0 * hills + 3.0 * detail)
            * (1.0 - smoothstep(0.0, 60.0, (x - range_x).abs()));

        // The mountain rises steeply from the cliff and levels off towards its
        // peaks, with gullies running down its face.
        let rise = (front - CLIFF_Z).max(0.0);
        let ridge = self.peaks.get([xd * 0.007, zd * 0.007]) as f32;
        let summit = 140.0 + 55.0 * ridge;
        let gullies = self.gullies.get([xd * 0.03, zd * 0.008]) as f32;
        let mountain = ledge_top
            + (summit - ledge_top) * (1.0 - (-rise / 30.0).exp())
            + (rise / 25.0).min(1.0) * (14.0 * gullies + 7.0 * detail);

        hill.max(range).max(mountain).clamp(1.0, 210.0)
    }

    fn column(&self, x: i32, z: i32) -> Column {
        let (x, z) = (x as f32, z as f32);
        let dx = self.height(x + 1.0, z) - self.height(x - 1.0, z);
        let dz = self.height(x, z + 1.0) - self.height(x, z - 1.0);
        Column {
            height: self.height(x, z),
            slope: 0.5 * (dx * dx + dz * dz).sqrt(),
        }
    }

    /// The material of the ground at `position`, `depth` voxels below its
    /// surface.
    ///
    /// The top voxel of every column above the snow line is snowy, so a steep
    /// face shows snow on the top of each step.
    fn material(&self, position: IVec3, column: Column, depth: f32) -> u8 {
        let (x, y, z) = (position.x as f64, position.y as f32, position.z as f64);
        let snow_line = 74.0 + 16.0 * self.patches.get([x * 0.04, z * 0.04]) as f32;
        let top = depth <= 1.0;
        if top && y > snow_line {
            SNOWY_STONE
        } else if column.slope < 1.4 && y < 80.0 {
            let rocky = self.patches.get([x * 0.11, z * 0.11]) > 0.5;
            if top && rocky {
                STONE
            } else if top {
                GRASS
            } else if depth < 4.0 {
                DIRT
            } else {
                STONE
            }
        } else {
            STONE
        }
    }
}

// ---------------------------------------------------------------------------
// The cave

/// A point on the cave's centre line: where it is, how far the cave reaches from
/// it, and the height of the cave's flat floor there.
#[derive(Clone, Copy)]
struct CavePoint {
    centre: Vec3,
    radius: f32,
    floor: f32,
}

const fn cave_point(centre: [f32; 3], radius: f32, floor: f32) -> CavePoint {
    CavePoint {
        centre: Vec3::from_array(centre),
        radius,
        floor,
    }
}

/// The tunnel from the mouth in the cliff beside the house to the cavern at its
/// end. Its first point stands on the ledge, so the mouth opens where the
/// tunnel meets the cliff.
const TUNNEL: [CavePoint; 5] = [
    cave_point([-26.0, 45.0, 4.0], 7.0, 40.0),
    cave_point([-26.0, 44.0, 26.0], 5.5, 40.0),
    cave_point([-32.0, 41.0, 44.0], 3.8, 38.0),
    cave_point([-41.0, 38.5, 58.0], 3.6, 35.0),
    cave_point([-44.0, 36.5, 74.0], 4.0, 33.0),
];

/// The cavern at the far end of the tunnel, an ellipsoid with a flat floor.
const CAVERN: CavePoint = cave_point([-42.0, 38.0, 90.0], 12.0, 32.0);
const CAVERN_RADII: Vec3 = Vec3::new(12.0, 8.0, 14.0);

/// The point on the tunnel's centre line nearest `p`, with the tunnel's radius
/// and floor there.
fn nearest_on_tunnel(p: Vec3) -> CavePoint {
    let mut best = TUNNEL[0];
    let mut best_distance = f32::MAX;
    for pair in TUNNEL.windows(2) {
        let (a, b) = (pair[0], pair[1]);
        let ab = b.centre - a.centre;
        let t = ((p - a.centre).dot(ab) / ab.length_squared()).clamp(0.0, 1.0);
        let centre = a.centre + ab * t;
        let distance = (p - centre).length_squared();
        if distance < best_distance {
            best_distance = distance;
            best = CavePoint {
                centre,
                radius: a.radius + (b.radius - a.radius) * t,
                floor: a.floor + (b.floor - a.floor) * t,
            };
        }
    }
    best
}

/// A box around the whole cave, outside which `in_cave` need not look.
const CAVE_MIN: Vec3 = Vec3::new(-60.0, 30.0, 0.0);
const CAVE_MAX: Vec3 = Vec3::new(-10.0, 53.0, 106.0);

/// Whether `position` is open space in the cave.
fn in_cave(position: IVec3) -> bool {
    let p = position.as_vec3() + 0.5;
    if p.cmplt(CAVE_MIN).any() || p.cmpgt(CAVE_MAX).any() {
        return false;
    }
    let cavern = (p - CAVERN.centre) / CAVERN_RADII;
    if cavern.length_squared() < 1.0 && p.y >= CAVERN.floor {
        return true;
    }
    let nearest = nearest_on_tunnel(p);
    // A little unevenness, so the walls aren't smooth tubes.
    let wobble = 0.2 * ((p.x * 0.7).sin() + (p.y * 0.9).cos() + (p.z * 0.5).sin());
    p.y >= nearest.floor && (p - nearest.centre).length() < nearest.radius + wobble
}

// ---------------------------------------------------------------------------
// Placed voxels: the house, the cave's supports and lamps, boulders and trees

/// Voxels placed by hand, which replace the ground and the cave where they are,
/// and the Bevy lights that go with the lamps among them.
#[derive(Default)]
struct Blueprint {
    voxels: HashMap<IVec3, WorldVoxel>,
    lamps: Vec<Lamp>,
    /// Added to every position given to `set`, `fill` and `lamp`.
    origin: IVec3,
}

#[derive(Clone, Copy, PartialEq)]
enum LampKind {
    Lantern,
    Hearth,
    Crystal,
}

/// A Bevy light that goes with an emissive voxel, placed in the open space
/// beside it.
#[derive(Clone, Copy)]
struct Lamp {
    at: Vec3,
    kind: LampKind,
}

impl Blueprint {
    fn set(&mut self, at: [i32; 3], voxel: WorldVoxel) {
        self.voxels
            .insert(self.origin + IVec3::from_array(at), voxel);
    }

    /// Sets every voxel from `min` to `max`, both included.
    fn fill(&mut self, min: [i32; 3], max: [i32; 3], voxel: WorldVoxel) {
        for x in min[0]..=max[0] {
            for y in min[1]..=max[1] {
                for z in min[2]..=max[2] {
                    self.set([x, y, z], voxel);
                }
            }
        }
    }

    fn solid(&mut self, min: [i32; 3], max: [i32; 3], material: u8) {
        self.fill(min, max, WorldVoxel::Solid(material));
    }

    fn air(&mut self, min: [i32; 3], max: [i32; 3]) {
        self.fill(min, max, WorldVoxel::Air);
    }

    fn lamp(&mut self, at: [f32; 3], kind: LampKind) {
        let at = self.origin.as_vec3() + Vec3::from_array(at);
        self.lamps.push(Lamp { at, kind });
    }
}

/// Everything placed by hand, built once on first use.
static BLUEPRINT: LazyLock<Blueprint> = LazyLock::new(|| {
    let terrain = Terrain::new();
    let mut blueprint = Blueprint::default();
    build_house(&mut blueprint);
    build_cave(&mut blueprint);
    place_boulders(&mut blueprint);
    plant_trees(&mut blueprint, &terrain);
    blueprint
});

/// The corner of the house at its lowest x and z, at the height of its floor.
const HOUSE: IVec3 = IVec3::new(0, LEDGE_TOP, 0);
/// The highest x and z of the house's outer walls, from `HOUSE`.
const HOUSE_X: i32 = 18;
const HOUSE_Z: i32 = 12;

/// Builds the house: a stone ground floor and a plank upper floor under a
/// gabled roof, with a cellar below. Its front faces -z.
///
/// Its rooms show the ways a room gets light. The hearth room and the lamp room
/// above it have lamps, which glow through their windows. The window room and
/// the attic have only windows, so they take sky light that spreads in through
/// them. The cellar has neither, and only a little sky light reaches it down
/// its stairs.
fn build_house(b: &mut Blueprint) {
    use WorldVoxel::Solid;
    b.origin = HOUSE;
    let (w, d) = (HOUSE_X, HOUSE_Z);

    // Clear the ground where the house stands and set it on a stone foundation.
    b.air([-2, 1, -2], [w + 2, 24, d + 2]);
    b.solid([0, -7, 0], [w, 0, d], COBBLESTONE);

    // Ground floor: stone walls on a plank floor.
    b.solid([1, 0, 1], [w - 1, 0, d - 1], FLOOR);
    for y in 1..=5 {
        b.solid([0, y, 0], [w, y, 0], COBBLESTONE);
        b.solid([0, y, d], [w, y, d], COBBLESTONE);
        b.solid([0, y, 0], [0, y, d], COBBLESTONE);
        b.solid([w, y, 0], [w, y, d], COBBLESTONE);
    }
    // Between the floors, planks inside a rim of timber beams.
    b.solid([0, 6, 0], [w, 6, d], TIMBER);
    b.solid([1, 6, 1], [w - 1, 6, d - 1], FLOOR);
    // Upper floor: plank walls.
    for y in 7..=12 {
        b.solid([0, y, 0], [w, y, 0], DARK_PLANKS);
        b.solid([0, y, d], [w, y, d], DARK_PLANKS);
        b.solid([0, y, 0], [0, y, d], DARK_PLANKS);
        b.solid([w, y, 0], [w, y, d], DARK_PLANKS);
    }
    // The roof steps in one voxel from each eave per row, up to the ridge, and
    // overhangs the walls by one voxel. The gable walls fill in beneath it.
    for k in 0..=7 {
        let y = 12 + k;
        b.solid([-1, y, k - 1], [w + 1, y, k - 1], ROOF);
        b.solid([-1, y, d + 1 - k], [w + 1, y, d + 1 - k], ROOF);
        if k >= 1 && k <= d - k {
            b.solid([0, y, k], [0, y, d - k], DARK_PLANKS);
            b.solid([w, y, k], [w, y, d - k], DARK_PLANKS);
        }
    }

    // Timber posts at the corners and along the front and back.
    for (x, z) in [(0, 0), (w, 0), (0, d), (w, d)] {
        b.solid([x, 1, z], [x, 12, z], TIMBER);
    }
    for z in [0, d] {
        for x in [4, 7, 11, 14] {
            b.solid([x, 1, z], [x, 5, z], TIMBER);
        }
        for x in [6, 9, 12] {
            b.solid([x, 7, z], [x, 12, z], TIMBER);
        }
    }
    // Two tie beams across the upper floor, under the roof.
    b.solid([4, 12, 1], [4, 12, d - 1], TIMBER);
    b.solid([14, 12, 1], [14, 12, d - 1], TIMBER);

    // The front door, with a timber lintel and a lamp on each side. One light
    // between the lamps stands in for both.
    b.air([8, 1, 0], [10, 3, 0]);
    b.solid([8, 4, 0], [10, 4, 0], TIMBER);
    b.set([7, 3, -1], Solid(LANTERN));
    b.set([11, 3, -1], Solid(LANTERN));
    b.lamp([9.5, 3.3, -1.8], LampKind::Lantern);

    // Ground floor: the hearth room, the hall in the middle, the window room.
    b.solid([6, 1, 1], [6, 5, d - 1], DARK_PLANKS);
    b.solid([12, 1, 1], [12, 5, d - 1], DARK_PLANKS);
    b.air([6, 1, 2], [6, 3, 3]);
    b.air([12, 1, 2], [12, 3, 3]);

    // The hearth room: a hearth with a chimney above it, a lamp on the back wall
    // and one on a stand in the front window, and a window in the side wall.
    b.solid([1, 1, 4], [2, 4, 8], COBBLESTONE);
    b.air([1, 1, 5], [2, 2, 7]);
    b.solid([1, 1, 5], [1, 1, 7], HEARTH);
    b.lamp([2.4, 1.6, 6.5], LampKind::Hearth);
    b.solid([1, 5, 5], [2, 21, 7], COBBLESTONE);
    b.set([4, 3, 11], Solid(LANTERN));
    b.air([1, 2, 0], [3, 4, 0]);
    b.set([2, 1, 1], Solid(TIMBER));
    b.set([2, 2, 1], Solid(LANTERN));
    b.lamp([2.5, 3.4, 1.5], LampKind::Lantern);
    b.air([0, 2, 9], [0, 4, 10]);

    // The window room: no lamp, and big windows in the front and side walls.
    b.air([15, 2, 0], [17, 4, 0]);
    b.air([w, 2, 5], [w, 4, 7]);

    // The hall: stairs up beside the window room, rising towards the back.
    for i in 1..=5 {
        b.solid([10, 1, 3 + i], [11, i, 3 + i], FLOOR);
    }
    b.air([10, 6, 4], [11, 6, 8]);

    // The hall: stairs down to the cellar beside the hearth room, falling towards
    // the front. The cellar lies under the hall and the window room.
    b.air([7, -5, 1], [w - 1, -1, d - 1]);
    b.air([7, 0, 6], [8, 0, 10]);
    for i in 1..=5 {
        b.solid([7, -6, 11 - i], [8, -i, 11 - i], FLOOR);
    }
    // Some crates in the far corner of the cellar.
    b.solid([15, -5, 9], [16, -4, 10], TIMBER);
    b.set([16, -5, 7], Solid(TIMBER));

    // Upper floor: a wall from the floor to the roof splits it into two rooms,
    // with a doorway at the back.
    for z in 1..d {
        let top = 12 + z.min(d - z);
        b.solid([9, 7, z], [9, top, z], DARK_PLANKS);
    }
    b.air([9, 7, 9], [9, 9, 10]);

    // The lamp room: a lamp on the back wall and one on a stand in a front
    // window, two windows in the front and one in the gable.
    b.set([3, 9, 11], Solid(LANTERN));
    b.air([2, 8, 0], [4, 10, 0]);
    b.set([3, 7, 1], Solid(TIMBER));
    b.set([3, 8, 1], Solid(LANTERN));
    b.lamp([3.5, 9.4, 1.5], LampKind::Lantern);
    b.air([7, 8, 0], [8, 10, 0]);
    b.air([0, 9, 9], [0, 10, 10]);

    // The attic: windows only, and the stairs coming up from the hall.
    b.air([10, 8, 0], [11, 10, 0]);
    b.air([15, 8, 0], [16, 10, 0]);
    b.air([w, 9, 5], [w, 11, 7]);

    // A gravel path from the door along the ledge to the cave mouth.
    b.solid([8, 0, -4], [10, 0, -1], GRAVEL);
    b.solid([-28, 0, -4], [10, 0, -2], GRAVEL);
    b.solid([-28, 0, -4], [-24, 0, 15], GRAVEL);
}

/// Builds a lantern on a post at the cave mouth, timber supports along the
/// tunnel with lamps hanging from some of them, and glowing crystals around the
/// cavern.
///
/// Sky light reaches some way into the tunnel through its wide mouth. Past that,
/// only the lamps and crystals light it.
fn build_cave(b: &mut Blueprint) {
    use WorldVoxel::Solid;
    b.origin = IVec3::ZERO;

    b.solid([-19, LEDGE_TOP + 1, 10], [-19, LEDGE_TOP + 3, 10], TIMBER);
    b.set([-19, LEDGE_TOP + 4, 10], Solid(LANTERN));
    b.lamp([-18.2, LEDGE_TOP as f32 + 4.5, 9.2], LampKind::Lantern);

    // (tunnel segment, how far along it, whether a lamp hangs from the support)
    let supports = [
        (0, 0.75, false),
        (1, 0.35, false),
        (1, 0.85, true),
        (2, 0.5, false),
        (3, 0.2, false),
        (3, 0.7, true),
    ];
    for (segment, t, lamp) in supports {
        let (a, c) = (TUNNEL[segment], TUNNEL[segment + 1]);
        let centre = a.centre.lerp(c.centre, t);
        let floor = (a.floor + (c.floor - a.floor) * t - 0.5).ceil() as i32;
        let along = c.centre - a.centre;
        let across = Vec3::new(-along.z, 0.0, along.x).normalize();
        let beam = floor + 4;

        // Each post stands against a wall, found by walking out from the centre.
        let post = |side: f32| {
            let mut last = centre.round().as_ivec3();
            for step in 0..12 {
                let at = (centre + across * side * step as f32).round().as_ivec3();
                let at = IVec3::new(at.x, floor + 1, at.z);
                if !in_cave(at) {
                    break;
                }
                last = at;
            }
            last
        };
        let (left, right) = (post(-1.0), post(1.0));
        for foot in [left, right] {
            b.solid([foot.x, floor, foot.z], [foot.x, beam, foot.z], TIMBER);
        }
        let span = (right - left).as_vec3();
        let steps = span.length().ceil() as i32;
        for i in 0..=steps {
            let at = (left.as_vec3() + span * i as f32 / steps as f32)
                .round()
                .as_ivec3();
            b.set([at.x, beam, at.z], Solid(TIMBER));
        }
        if lamp {
            let middle = ((left + right).as_vec3() * 0.5).round().as_ivec3();
            b.set([middle.x, beam - 1, middle.z], Solid(LANTERN));
            let (x, z) = (middle.x as f32 + 0.5, middle.z as f32 + 0.5);
            b.lamp([x, beam as f32 - 1.4, z], LampKind::Lantern);
        }
    }

    // Crystals: clusters on the cavern walls, found by walking out from its
    // centre, and a few standing on its floor.
    let floor = (CAVERN.floor - 0.5).ceil() as i32;
    for (i, angle) in [0.3f32, 1.5, 2.6, 3.8, 5.0].into_iter().enumerate() {
        let direction = Vec3::new(angle.cos(), 0.0, angle.sin());
        let height = floor + 1 + (i as i32 % 3);
        let wall = (0..20).find_map(|step| {
            let at = (CAVERN.centre + direction * step as f32).round().as_ivec3();
            let at = IVec3::new(at.x, height, at.z);
            (!in_cave(at)).then_some(at)
        });
        let Some(wall) = wall else {
            continue;
        };
        for offset in [
            [0, 0, 0],
            [0, 1, 0],
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1],
            [0, 2, 0],
        ] {
            b.set(
                (wall + IVec3::from_array(offset)).to_array(),
                Solid(CRYSTAL),
            );
        }
        let open = wall.as_vec3() + 0.5 - direction * 2.0;
        b.lamp(open.to_array(), LampKind::Crystal);
    }
    for (x, z, height) in [(-46, 86, 3), (-39, 94, 2), (-37, 84, 1), (-48, 95, 2)] {
        b.solid([x, floor, z], [x, floor + height - 1, z], CRYSTAL);
    }
}

/// Places a few boulders on the ledge.
fn place_boulders(b: &mut Blueprint) {
    b.origin = IVec3::ZERO;
    let boulders: [(i32, i32, i32); 6] = [
        (-12, -9, 2),
        (-40, 4, 3),
        (26, 10, 2),
        (30, -6, 1),
        (-20, 12, 2),
        (46, -2, 2),
    ];
    for (x, z, size) in boulders {
        for dx in 0..=size {
            for dz in 0..=size {
                let height = size - (dx - size / 2).abs().max((dz - size / 2).abs());
                let at = [x + dx, LEDGE_TOP + 1, z + dz];
                b.solid(at, [at[0], at[1] + height, at[2]], STONE);
            }
        }
    }
}

fn hash(x: i32, z: i32) -> u32 {
    let mut h =
        (x as u32).wrapping_mul(0x8da6_b343) ^ (z as u32).wrapping_mul(0xd816_3841);
    h ^= h >> 13;
    h = h.wrapping_mul(0x5bd1_e995);
    h ^ (h >> 15)
}

/// Plants spruces on gentle slopes below the snow, away from the ledge, and a
/// few beside the house.
///
/// The trees stand far enough apart that no two share leaves, so a tree cut
/// through its trunk falls on its own. The forest has one tree in each
/// `CELL` by `CELL` square at most, moved up to 4 voxels from its corner, so
/// two trunks are at least 8 voxels apart along x or z. A spruce's leaves
/// reach 3 voxels out along x and z, which leaves a gap of at least one voxel.
fn plant_trees(b: &mut Blueprint, terrain: &Terrain) {
    b.origin = IVec3::ZERO;
    const CELL: i32 = 12;
    for cell_x in -21..=21 {
        for cell_z in -21..=1 {
            let h = hash(cell_x, cell_z);
            if h % 100 > 85 {
                continue;
            }
            let x = cell_x * CELL + (h >> 8) as i32 % 5;
            let z = cell_z * CELL + (h >> 12) as i32 % 5;
            let column = terrain.column(x, z);
            let near_ledge = Terrain::ledge_weight(x as f32, z as f32) > 0.02;
            if column.slope > 1.0 || column.height > 74.0 || near_ledge {
                continue;
            }
            let base = IVec3::new(x, column.height.ceil() as i32, z);
            spruce(b, terrain, base, 8 + (h >> 16) as i32 % 9);
        }
    }
    // The trees beside the house stand on the ledge, where the forest has
    // none, far enough from its edge, the boulders and each other that their
    // leaves touch nothing.
    for (x, z, height) in [(24, -12, 13), (-30, -6, 15), (40, -10, 12), (44, 10, 14)] {
        spruce(b, terrain, IVec3::new(x, LEDGE_TOP + 1, z), height);
    }
}

/// A spruce: a trunk under tiers of leaves that narrow to a point.
fn spruce(b: &mut Blueprint, terrain: &Terrain, base: IVec3, height: i32) {
    use WorldVoxel::Solid;
    b.solid(
        base.to_array(),
        (base + IVec3::Y * (height - 3)).to_array(),
        BARK,
    );
    let mut ground = HashMap::new();
    for y in (2 + height / 6)..=height {
        let from_top = height - y;
        // Every third layer pulls in, so the leaves read as stacked boughs.
        let mut radius = 0.5 + from_top as f32 * 0.3;
        if from_top % 3 == 0 && from_top > 0 {
            radius -= 1.0;
        }
        let radius = radius.clamp(0.0, 3.8);
        let reach = radius.ceil() as i32;
        for dx in -reach..=reach {
            for dz in -reach..=reach {
                let distance = (dx * dx + dz * dz) as f32;
                let trunk = dx == 0 && dz == 0 && y <= height - 3;
                if trunk || distance > radius * radius + 0.3 {
                    continue;
                }
                let (x, z) = (base.x + dx, base.z + dz);
                // Leave out some leaves on the rim, so the edges are ragged.
                let rim = distance > (radius - 1.0).max(0.0).powi(2);
                if rim && hash(x * 31 + y, z * 17 - y).is_multiple_of(4) {
                    continue;
                }
                // Leaves on a slope stop short of the ground. A leaf sharing
                // a face with the ground would hold the tree up after its
                // trunk is cut, so none touches it from above or beside.
                let leaf = (base.y + y) as f32;
                let mut ground_at = |x: i32, z: i32| {
                    *ground
                        .entry((x, z))
                        .or_insert_with(|| terrain.height(x as f32, z as f32))
                };
                let clear = leaf - 1.0 >= ground_at(x, z)
                    && [(1, 0), (-1, 0), (0, 1), (0, -1)]
                        .into_iter()
                        .all(|(dx, dz)| leaf >= ground_at(x + dx, z + dz));
                if clear {
                    b.set([x, base.y + y, z], Solid(LEAVES));
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// The world

/// Decides a voxel: placed voxels first, then the cave, then the ground.
///
/// `columns` keeps each column's height and slope for the rest of the chunk.
fn scene_voxel(
    terrain: &Terrain,
    columns: &mut HashMap<(i32, i32), Column>,
    position: IVec3,
) -> WorldVoxel {
    if let Some(voxel) = BLUEPRINT.voxels.get(&position) {
        return *voxel;
    }
    if in_cave(position) {
        return WorldVoxel::Air;
    }
    let column = *columns
        .entry((position.x, position.z))
        .or_insert_with(|| terrain.column(position.x, position.z));
    let depth = column.height - position.y as f32;
    if depth <= 0.0 {
        return WorldVoxel::Air;
    }
    if in_cave(position + IVec3::Y) {
        return WorldVoxel::Solid(GRAVEL);
    }
    WorldVoxel::Solid(terrain.material(position, column, depth))
}

#[derive(Resource, Clone, Default)]
struct MountainWorld;

impl VoxelWorldConfig for MountainWorld {
    type MaterialIndex = u8;
    type ChunkUserBundle = ();

    // Light asks for the voxels of a position more than once, so the scene is a
    // pure function of position.
    fn voxel_lookup_delegate(&self) -> VoxelLookupDelegate<Self::MaterialIndex> {
        Box::new(|_, _, _| {
            let terrain = Terrain::new();
            let mut columns = HashMap::new();
            Box::new(move |position, _| scene_voxel(&terrain, &mut columns, position))
        })
    }

    // Every chunk within the spawning distance stays spawned, in view or not,
    // so turning the camera spawns and retires nothing, and the chunks behind
    // it still cast the sun's shadows.
    fn chunk_spawn_strategy(&self) -> ChunkSpawnStrategy {
        ChunkSpawnStrategy::Close
    }

    fn chunk_despawn_strategy(&self) -> ChunkDespawnStrategy {
        ChunkDespawnStrategy::FarAway
    }

    // A browser runs chunk tasks on the main thread, each to its end before
    // the next frame. Running one at a time keeps frames even while chunks
    // stream in.
    fn max_active_chunk_threads(&self) -> usize {
        if IN_BROWSER {
            1
        } else {
            usize::MAX
        }
    }

    // Everything lies between y = 0 and y = 224, with rock below and air above,
    // so spawning rays need not spawn chunks outside it.
    //
    // In the browser, the world also ends outside `BROWSER_CHUNKS`.
    fn chunk_fill_hint(&self, key: ChunkKey) -> FillHint {
        let (min, max) = BROWSER_CHUNKS;
        let column = key.position.xz();
        let outside = column.cmpgt(max).any() || (column + key.extent()).cmple(min).any();
        if IN_BROWSER && outside {
            FillHint::Uniform(WorldVoxel::Air)
        } else if key.position.y + key.extent() <= 0 {
            FillHint::Uniform(WorldVoxel::Solid(STONE))
        } else if key.position.y >= 7 {
            FillHint::Uniform(WorldVoxel::Air)
        } else {
            FillHint::Unknown
        }
    }

    fn texture_index_mapper(
        &self,
    ) -> Arc<dyn Fn(Self::MaterialIndex) -> [u32; 3] + Send + Sync> {
        Arc::new(texture_indices)
    }

    fn voxel_texture(&self) -> Option<(String, u32)> {
        Some(("mountain_house_texture.png".into(), TILE_COUNT))
    }

    fn light_propagation(&self) -> Option<LightPropagation> {
        Some(LightPropagation {
            // Nothing stands above y = 224, so sky light enters every column
            // that is open up to there, without sky probes of the chunks above.
            sky: Sky::Ceiling { y: 223 },
            // A browser keeps sky light's reach short, as `LightPropagation`
            // explains.
            #[cfg(target_arch = "wasm32")]
            sky_reach: 16,
            look: LightLook {
                // Voxel light is white. Less of it than the default lets the
                // lamps' warm Bevy lights colour the rooms.
                voxel_brightness: 400.0,
                ..default()
            },
            ..default()
        })
    }

    fn emissive_materials(&self) -> Vec<(Self::MaterialIndex, Emission)> {
        vec![
            (LANTERN, Emission::level(15.0)),
            (HEARTH, Emission::level(14.0)),
            (CRYSTAL, Emission::level(9.0)),
        ]
    }

    // Lift each piece that an edit cuts loose into a sub-grid, such as a tree
    // whose trunk was cut. "Felling trees" below makes it fall.
    fn detect_dislodged_fragments(
        &self,
    ) -> Option<FragmentDetection<Self::MaterialIndex>> {
        Some(FragmentDetection {
            auto_lift: true,
            ..default()
        })
    }
}

// ---------------------------------------------------------------------------
// The app

fn main() {
    let window = Window {
        title: "light_mountain_house".into(),
        resolution: (1600, 900).into(),
        #[cfg(target_arch = "wasm32")]
        canvas: Some("#bevy".into()),
        ..default()
    };
    App::new()
        .add_plugins(DefaultPlugins.set(ImagePlugin::default_nearest()).set(
            WindowPlugin {
                primary_window: Some(window),
                primary_cursor_options: Some(mouse(true)),
                ..default()
            },
        ))
        .add_plugins(MountainHousePlugin)
        .run();
}

/// The scene and its controls, without Bevy's default plugins.
///
/// The website's live demo includes this file and adds this plugin to its own
/// app, which sets up the window for a canvas in the page.
pub struct MountainHousePlugin;

impl Plugin for MountainHousePlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((
            VoxelWorldPlugin::with_config(MountainWorld),
            PhysicsPlugins::default(),
        ))
        .insert_resource(Sun::DUSK)
        .init_resource::<Placing>()
        .init_resource::<GroundColliders>()
        .add_systems(Startup, (setup, spawn_lamps, spawn_smoke))
        // Right after the voxel world's own systems, in the frame a lift's
        // voxels first show in its sub-grid, so the piece and the ground
        // under it have colliders before the next physics step.
        .add_systems(
            PreUpdate,
            (start_falling, build_piece_colliders, build_ground_colliders)
                .chain()
                .after(VoxelWorldSet::<MountainWorld>::default()),
        )
        .add_systems(
            Update,
            (
                // `edit_voxels` goes before `grab_mouse`, so the click that
                // captures the mouse doesn't also place a voxel.
                (look_around, walk, edit_voxels, grab_mouse).chain(),
                choose_voxel,
                turn_sun,
                follow_sun.run_if(resource_changed::<Sun>),
                fit_scene_image,
                flicker,
                drift_smoke,
                carry_lantern,
                toggle_help,
                drop_lost_pieces,
            ),
        );
    }
}

/// Where the camera's feet start: on the ledge, off the house's front corner.
const START: Vec3 = Vec3::new(-6.0, LEDGE_TOP as f32 + 1.0, -12.0);

/// The camera, which looks where its yaw and pitch point.
#[derive(Component, Default)]
struct Viewer {
    yaw: f32,
    pitch: f32,
    /// The turn the mouse has asked for that the camera hasn't made yet, as
    /// yaw and pitch in radians. `look_around` makes it over `LOOK_SMOOTHING`.
    unapplied: Vec2,
}

#[derive(Component)]
struct HelpText;

/// The distance fog at dusk, which `follow_sun` dims as the sun sets.
const FOG: Color = Color::srgb(0.42, 0.45, 0.58);
const FOG_GLOW: Color = Color::srgba(1.0, 0.62, 0.35, 0.5);
const MIST_AMBIENT: f32 = 0.3;

/// The ambient light that stands in for the atmosphere in the browser, at dusk
/// and before.
const BROWSER_AMBIENT: f32 = 600.0;

/// The smallest and largest chunk columns, along x and z, of the world in the
/// browser: the ledge, the house, the cave and a little of the valley.
///
/// Every chunk drawn costs the browser time each frame, and chunk tasks run
/// one at a time on the page's one thread. Ending the world here lets every
/// chunk of it spawn while the page loads, within the default spawning
/// distance from anywhere in it, so none stream in while walking. The walker
/// stops at the edge, where the chunks beyond are never generated.
const BROWSER_CHUNKS: (IVec2, IVec2) = (IVec2::new(-3, -3), IVec2::new(2, 3));

/// How far the distance fog lets you see, in voxels. The browser's shorter
/// distance hides the edge of its smaller world, and its sky is the fog's
/// colour, so the world fades into the sky there.
const FOG_VISIBILITY: f32 = if IN_BROWSER { 150.0 } else { 520.0 };

/// How many metres one voxel stands for, for the atmosphere's haze.
const METRES_PER_VOXEL: f32 = 14.0;

/// The sun's illuminance before the atmosphere filters it. It is well below real
/// sunlight, so that one exposure suits both the sky and the lamps.
const SUN_ILLUMINANCE: f32 = lux::RAW_SUNLIGHT / 40.0;

/// Whether the example runs in a browser, where it renders through WebGL2 and
/// leaves out what WebGL2 can't do.
const IN_BROWSER: bool = cfg!(target_arch = "wasm32");

/// How many rows of pixels the scene renders at, whatever the window's size.
/// The window shows them stretched to fit, so a larger window, or one on a
/// high-density display, shows the same pixels larger rather than drawing more.
const RENDER_HEIGHT: u32 = 640;

/// The image the camera renders the scene into, which `fit_scene_image` keeps
/// the shape of the window.
#[derive(Resource)]
struct SceneImage(Handle<Image>);

/// The size of the scene's image for `window`: `RENDER_HEIGHT` pixels tall and
/// as wide as the window's shape needs.
fn scene_size(window: &Window) -> UVec2 {
    let aspect = window.width() / window.height().max(1.0);
    let width = (RENDER_HEIGHT as f32 * aspect).round().max(1.0);
    UVec2::new(width as u32, RENDER_HEIGHT)
}

fn setup(
    mut commands: Commands,
    mut media: ResMut<Assets<ScatteringMedium>>,
    mut images: ResMut<Assets<Image>>,
    window: Single<&Window, With<PrimaryWindow>>,
    sun: Res<Sun>,
) {
    // The camera renders the scene into an image, which a second camera shows
    // stretched over the window, under the help text. Smoothing it as it
    // stretches keeps the pixels even at any window size.
    // WebGL2 can't view a texture in a second format, so in the browser the
    // image is sRGB throughout.
    let size = scene_size(&window);
    let (format, view_format) = if IN_BROWSER {
        (TextureFormat::Rgba8UnormSrgb, None)
    } else {
        (
            TextureFormat::Rgba8Unorm,
            Some(TextureFormat::Rgba8UnormSrgb),
        )
    };
    let mut scene = Image::new_target_texture(size.x, size.y, format, view_format);
    scene.sampler = ImageSampler::linear();
    let scene = images.add(scene);
    commands.insert_resource(SceneImage(scene.clone()));
    commands.spawn((Camera2d, Msaa::Off));
    commands.spawn((
        ImageNode::new(scene.clone()),
        Node {
            position_type: PositionType::Absolute,
            width: percent(100),
            height: percent(100),
            ..default()
        },
        GlobalZIndex(-1),
    ));

    // The atmosphere is a planet whose ground is at y = 0. Scaling it down makes
    // one voxel stand for several metres, so the haze of distance shows within
    // the few hundred voxels that are spawned.
    if !IN_BROWSER {
        commands.spawn((
            Atmosphere::earth(media.add(ScatteringMedium::default())),
            Transform::from_xyz(0.0, -6_360_000.0 / METRES_PER_VOXEL, 0.0)
                .with_scale(Vec3::splat(1.0 / METRES_PER_VOXEL)),
        ));
    }

    // The atmosphere draws the sky over the colour the camera clears to, and
    // lights the world through its environment map. Light propagation scales
    // the environment map down where sky light is weak, so it doesn't reach
    // into the cellar or deep into the cave.
    //
    // In the browser, where there is no atmosphere, the fog's colour and an
    // ambient light stand in for it, and `follow_sun` dims both after sunset.
    let (clear_color, ambient) = if IN_BROWSER {
        let ambient = GlobalAmbientLight {
            brightness: BROWSER_AMBIENT,
            ..default()
        };
        (FOG, ambient)
    } else {
        (Color::BLACK, GlobalAmbientLight::NONE)
    };
    commands.insert_resource(ambient);

    let camera = commands
        .spawn((
            Camera3d::default(),
            Camera {
                clear_color: ClearColorConfig::Custom(clear_color),
                // Renders before the window's camera shows its image.
                order: -1,
                ..default()
            },
            RenderTarget::Image(scene.into()),
            Projection::Perspective(PerspectiveProjection {
                fov: 60f32.to_radians(),
                far: 2000.0,
                ..default()
            }),
            Transform::from_translation(START + Vec3::Y * EYE_HEIGHT),
            // Facing the house, a little upwards.
            Viewer {
                yaw: -2.39,
                pitch: 0.16,
                ..default()
            },
            Walker {
                feet: START,
                ..default()
            },
            Exposure { ev100: 9.0 },
            Tonemapping::TonyMcMapface,
            // Makes the lamps and the faces of emissive voxels glow.
            Bloom::NATURAL,
            // Hides the edge of the spawned world in haze.
            DistanceFog {
                color: FOG,
                directional_light_color: FOG_GLOW,
                directional_light_exponent: 12.0,
                falloff: FogFalloff::from_visibility_colors(
                    FOG_VISIBILITY,
                    Color::srgb(0.32, 0.36, 0.5),
                    Color::srgb(0.62, 0.62, 0.72),
                ),
            },
            VoxelWorldCamera::<MountainWorld>::default(),
            // WebGL2 can't sample a multisampled depth texture, which the
            // camera's effects ask for.
            if IN_BROWSER {
                Msaa::Off
            } else {
                Msaa::default()
            },
        ))
        .id();
    // WebGL2 can't run the atmosphere or volumetric fog, so the browser goes
    // without both.
    if !IN_BROWSER {
        commands.entity(camera).insert((
            AtmosphereSettings::default(),
            AtmosphereEnvironmentMapLight {
                intensity: 2.5,
                ..default()
            },
            // Volumetric fog shows only inside a `FogVolume`. Lights with
            // `VolumetricLight` shine through it, and the ambient colour lights
            // it in their shadows.
            VolumetricFog {
                ambient_color: Color::srgb(0.55, 0.62, 0.8),
                ambient_intensity: MIST_AMBIENT,
                step_count: 48,
                ..default()
            },
        ));
    }

    // Light propagation lets the sun shine on chunks only on faces with direct
    // sky, so it can't reach into the cave through rock that hasn't spawned. It
    // keeps a directional light without shadow maps off chunks altogether.
    commands.spawn((
        DirectionalLight {
            illuminance: SUN_ILLUMINANCE,
            shadow_maps_enabled: true,
            ..default()
        },
        sun.transform(),
        VolumetricLight,
        CascadeShadowConfigBuilder {
            first_cascade_far_bound: 40.0,
            maximum_distance: 500.0,
            ..default()
        }
        .build(),
    ));

    // Mist in the valley below the ledge.
    if !IN_BROWSER {
        commands.spawn((
            FogVolume {
                density_factor: 0.08,
                absorption: 0.05,
                scattering: 0.4,
                ..default()
            },
            Transform::from_xyz(-25.0, 17.0, -110.0)
                .with_scale(Vec3::new(500.0, 34.0, 180.0)),
        ));
    }

    commands.spawn((
        HelpText,
        Text::new(help(0)),
        TextFont {
            font_size: FontSize::Px(14.0),
            ..default()
        },
        Node {
            position_type: PositionType::Absolute,
            top: px(12),
            left: px(12),
            padding: UiRect::all(px(8)),
            ..default()
        },
        BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.5)),
    ));

    // A crosshair in the middle of the window, at the voxel a click edits.
    commands
        .spawn(Node {
            position_type: PositionType::Absolute,
            width: percent(100),
            height: percent(100),
            justify_content: JustifyContent::Center,
            align_items: AlignItems::Center,
            ..default()
        })
        .with_children(|centre| {
            for (width, height) in [(16, 2), (2, 16)] {
                centre.spawn((
                    Node {
                        position_type: PositionType::Absolute,
                        width: px(width),
                        height: px(height),
                        ..default()
                    },
                    BackgroundColor(Color::srgba(1.0, 1.0, 1.0, 0.8)),
                ));
            }
        });
}

/// The help text, naming the voxel at `PLACEABLE[placing]` as the one a left
/// click places.
fn help(placing: usize) -> String {
    format!(
        "WASD  walk, Shift runs\n\
         Space  jump\n\
         left click  place {}\n\
         right click  remove, or fell a tree\n\
         1-4  stone, timber, lantern, crystal\n\
         [ ]  lower or raise the sun\n\
         L  carry a lantern\n\
         Esc  free the mouse\n\
         H  hide this",
        PLACEABLE[placing].2
    )
}

/// Keeps the scene's image the shape of the window, so the scene doesn't look
/// squashed or stretched after the window is resized.
fn fit_scene_image(
    window: Single<&Window, (With<PrimaryWindow>, Changed<Window>)>,
    scene: Res<SceneImage>,
    mut images: ResMut<Assets<Image>>,
) {
    let size = scene_size(&window);
    if images
        .get(&scene.0)
        .is_some_and(|image| image.size() != size)
    {
        if let Some(mut image) = images.get_mut(&scene.0) {
            image.resize(Extent3d {
                width: size.x,
                height: size.y,
                depth_or_array_layers: 1,
            });
        }
    }
}

/// Where the sun stands, in radians: its height above the horizon, and its
/// bearing, the angle from +x towards +z.
#[derive(Resource, Clone, Copy)]
struct Sun {
    elevation: f32,
    bearing: f32,
}

impl Sun {
    /// Just above the range across the valley, which shades the ledge and leaves
    /// the peaks in sunlight.
    const DUSK: Sun = Sun {
        elevation: 0.06,
        bearing: -0.55,
    };

    /// The direction from the world towards the sun.
    fn direction(&self) -> Vec3 {
        Vec3::new(
            self.elevation.cos() * self.bearing.cos(),
            self.elevation.sin(),
            self.elevation.cos() * self.bearing.sin(),
        )
    }

    fn transform(&self) -> Transform {
        Transform::default().looking_to(-self.direction(), Vec3::Y)
    }

    /// How much of the sun shows above the horizon, from 1 down to 0 as it
    /// sinks below.
    fn above_horizon(&self) -> f32 {
        smoothstep(-0.04, 0.04, self.elevation)
    }

    /// How much daylight is left, from 1 at dusk and above down to 0.08 once the
    /// sun is well below the horizon.
    fn daylight(&self) -> f32 {
        0.08 + 0.92 * smoothstep(-0.12, 0.06, self.elevation)
    }
}

fn turn_sun(time: Res<Time>, keys: Res<ButtonInput<KeyCode>>, mut sun: ResMut<Sun>) {
    let mut turn = 0.0;
    if keys.pressed(KeyCode::BracketLeft) {
        turn -= 1.0;
    }
    if keys.pressed(KeyCode::BracketRight) {
        turn += 1.0;
    }
    if turn != 0.0 {
        sun.elevation =
            (sun.elevation + turn * 0.25 * time.delta_secs()).clamp(-0.3, 1.4);
    }
}

/// Moves the directional light with the sun, and dims the fog and mist after it
/// sets. The atmosphere's sky follows the light on its own, but the fog's
/// colours are fixed values that would otherwise glow in the dark.
///
/// In the browser it also dims the sky colour and ambient light that stand in
/// for the atmosphere, and puts the sun out below the horizon, where the
/// atmosphere would have filtered it out.
fn follow_sun(
    sun: Res<Sun>,
    mut ambient: ResMut<GlobalAmbientLight>,
    mut lights: Query<(&mut Transform, &mut DirectionalLight)>,
    mut cameras: Query<(&mut Camera, &mut DistanceFog, Option<&mut VolumetricFog>)>,
) {
    for (mut transform, mut light) in &mut lights {
        *transform = sun.transform();
        if IN_BROWSER {
            light.illuminance = SUN_ILLUMINANCE * sun.above_horizon();
        }
    }
    let daylight = sun.daylight();
    let dim = |color: Color| {
        let linear = color.to_linear();
        Color::linear_rgba(
            linear.red * daylight,
            linear.green * daylight,
            linear.blue * daylight,
            linear.alpha,
        )
    };
    for (mut camera, mut fog, mist) in &mut cameras {
        fog.color = dim(FOG);
        fog.directional_light_color = dim(FOG_GLOW);
        if let Some(mut mist) = mist {
            mist.ambient_intensity = MIST_AMBIENT * daylight;
        }
        if IN_BROWSER {
            camera.clear_color = ClearColorConfig::Custom(dim(FOG));
        }
    }
    if IN_BROWSER {
        ambient.brightness = BROWSER_AMBIENT * daylight;
    }
}

fn spawn_lamps(mut commands: Commands) {
    for lamp in &BLUEPRINT.lamps {
        spawn_lamp(&mut commands, lamp.at, lamp.kind);
    }
}

/// The Bevy light of a lamp.
///
/// Light propagation lights the voxels around a lamp, but voxel light has no
/// colour, so a warm Bevy light adds it. The Bevy light also lights entities
/// that aren't chunks.
///
/// The Bevy lights cast no shadows, so each also shines through the walls
/// within its range. With shadows, each would render the world around it six
/// times every frame, and the lantern voxel beside it would cast a shadow
/// across the room.
#[derive(Component)]
struct LampLight;

/// How far a lamp's Bevy light can be from the voxels that glow with it. The
/// light goes out once none is left this close.
const LAMP_SPREAD: f32 = 2.5;

fn spawn_lamp(commands: &mut Commands, at: Vec3, kind: LampKind) {
    let light = match kind {
        LampKind::Lantern => PointLight {
            color: Color::srgb(1.0, 0.62, 0.28),
            intensity: 150_000.0,
            range: 12.0,
            radius: 0.2,
            ..default()
        },
        LampKind::Hearth => PointLight {
            color: Color::srgb(1.0, 0.45, 0.15),
            intensity: 200_000.0,
            range: 10.0,
            radius: 0.6,
            ..default()
        },
        LampKind::Crystal => PointLight {
            color: Color::srgb(0.35, 0.65, 1.0),
            intensity: 25_000.0,
            range: 12.0,
            radius: 0.5,
            ..default()
        },
    };
    let mut entity = commands.spawn((LampLight, Transform::from_translation(at), light));
    if kind != LampKind::Crystal {
        entity.insert(Flicker {
            lumens: light.intensity,
        });
    }
}

/// Whether a lantern, hearth or crystal voxel lies within `LAMP_SPREAD` of `at`.
fn lamp_voxel_near(voxel_world: &VoxelWorld<MountainWorld>, at: Vec3) -> bool {
    let min = (at - LAMP_SPREAD).floor().as_ivec3();
    let max = (at + LAMP_SPREAD).ceil().as_ivec3();
    (min.x..max.x).any(|x| {
        (min.y..max.y).any(|y| {
            (min.z..max.z).any(|z| {
                let position = IVec3::new(x, y, z);
                (position.as_vec3() + 0.5).distance(at) <= LAMP_SPREAD
                    && matches!(
                        voxel_world.get_voxel(position),
                        WorldVoxel::Solid(LANTERN | HEARTH | CRYSTAL)
                    )
            })
        })
    })
}

/// Makes a flame-lit lamp waver a little around its `lumens`.
#[derive(Component)]
struct Flicker {
    lumens: f32,
}

fn flicker(time: Res<Time>, mut lights: Query<(&mut PointLight, &Transform, &Flicker)>) {
    let t = time.elapsed_secs();
    for (mut light, transform, flicker) in &mut lights {
        let seed = transform.translation.x * 1.3 + transform.translation.z * 0.7;
        let wave = (t * 7.0 + seed).sin() * 0.5 + (t * 13.0 + seed * 2.0).sin() * 0.3;
        light.intensity = flicker.lumens * (1.0 + 0.08 * wave);
    }
}

/// Where smoke leaves the top of the chimney.
const CHIMNEY_TOP: Vec3 = Vec3::new(2.0, LEDGE_TOP as f32 + 22.5, 6.5);
const SMOKE_PUFFS: usize = 14;
const SMOKE_SECONDS: f32 = 10.0;

/// A puff of smoke from the chimney, `age` seconds after it left.
///
/// Puffs are Bevy meshes, not voxels, so they take Bevy's lights only.
#[derive(Component)]
struct Smoke {
    age: f32,
}

fn spawn_smoke(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let cube = meshes.add(Cuboid::from_length(1.0));
    for i in 0..SMOKE_PUFFS {
        let material = materials.add(StandardMaterial {
            base_color: Color::srgba(0.62, 0.62, 0.65, 0.0),
            alpha_mode: AlphaMode::Blend,
            perceptual_roughness: 1.0,
            ..default()
        });
        commands.spawn((
            Mesh3d(cube.clone()),
            MeshMaterial3d(material),
            Transform::from_translation(CHIMNEY_TOP),
            NotShadowCaster,
            Smoke {
                age: i as f32 * SMOKE_SECONDS / SMOKE_PUFFS as f32,
            },
        ));
    }
}

/// Lifts each puff, drifts it with the wind, grows it and fades it out, then
/// starts it again at the chimney.
fn drift_smoke(
    time: Res<Time>,
    mut puffs: Query<(
        &mut Transform,
        &MeshMaterial3d<StandardMaterial>,
        &mut Smoke,
    )>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    for (mut transform, material, mut smoke) in &mut puffs {
        smoke.age = (smoke.age + time.delta_secs()) % SMOKE_SECONDS;
        let t = smoke.age / SMOKE_SECONDS;
        let sway = (smoke.age * 1.3).sin() * 0.8;
        transform.translation =
            CHIMNEY_TOP + Vec3::new(4.0 * t * t + sway, 16.0 * t, 3.0 * t);
        transform.scale = Vec3::splat(1.0 + 2.2 * t);
        if let Some(mut material) = materials.get_mut(&material.0) {
            let fade_in = (t * 12.0).min(1.0);
            material.base_color.set_alpha(0.8 * fade_in * (1.0 - t * t));
        }
    }
}

/// A lantern the camera carries, switched on and off with L.
#[derive(Component)]
struct CarriedLantern;

/// A Bevy light lights chunks whatever their propagated light, so a carried
/// lantern shows the places light propagation leaves dark.
fn carry_lantern(
    mut commands: Commands,
    keys: Res<ButtonInput<KeyCode>>,
    cameras: Query<Entity, With<Viewer>>,
    carried: Query<Entity, With<CarriedLantern>>,
) {
    if !keys.just_pressed(KeyCode::KeyL) {
        return;
    }
    if let Ok(lantern) = carried.single() {
        commands.entity(lantern).despawn();
        return;
    }
    for camera in &cameras {
        commands.entity(camera).with_child((
            CarriedLantern,
            PointLight {
                color: Color::srgb(1.0, 0.62, 0.28),
                intensity: 60_000.0,
                range: 16.0,
                radius: 0.1,
                ..default()
            },
            Transform::from_xyz(0.4, -0.3, -0.5),
        ));
    }
}

/// How long mouse look takes to catch up with the mouse, in seconds.
///
/// The mouse reports its movement at its own rate, which doesn't match the
/// frame rate, so some frames receive two reports and others none. Turning by
/// each frame's reports would turn the camera in uneven steps; easing towards
/// where the mouse points spreads them out, at the cost of this much delay.
const LOOK_SMOOTHING: f32 = 0.03;

/// Turns the camera with the mouse while the mouse is captured.
fn look_around(
    time: Res<Time>,
    motion: Res<AccumulatedMouseMotion>,
    cursor: Single<&CursorOptions, With<PrimaryWindow>>,
    camera: Single<(&mut Transform, &mut Viewer)>,
) {
    let (mut transform, mut viewer) = camera.into_inner();
    if cursor.grab_mode == CursorGrabMode::Locked {
        viewer.unapplied += motion.delta * 0.003;
    }
    let step = viewer.unapplied * (1.0 - (-time.delta_secs() / LOOK_SMOOTHING).exp());
    viewer.unapplied -= step;
    viewer.yaw -= step.x;
    viewer.pitch = (viewer.pitch - step.y).clamp(-1.5, 1.5);
    transform.rotation = Quat::from_euler(EulerRot::YXZ, viewer.yaw, viewer.pitch, 0.0);
}

/// The walking body's size, in voxels: a box this wide and tall, with the eye
/// this far above the feet. It fits through doors and under ceilings two voxels
/// high.
const BODY_RADIUS: f32 = 0.3;
const BODY_HEIGHT: f32 = 1.75;
const EYE_HEIGHT: f32 = 1.6;
/// The gap the body keeps from the voxels it touches, so that rounding never
/// puts it inside one.
const SKIN: f32 = 0.001;
const WALK_SPEED: f32 = 4.5;
const RUN_SPEED: f32 = 9.0;
const GRAVITY: f32 = 25.0;
const MAX_FALL_SPEED: f32 = 40.0;
/// Lifts the feet about 1.3 voxels, enough to jump onto a voxel from the air.
const JUMP_SPEED: f32 = 8.0;

/// Makes the camera walk: a body under it falls, stops against solid voxels,
/// and climbs or steps down one voxel at a time, so it can take stairs and
/// rough ground without jumping.
#[derive(Component, Clone, Copy, Default)]
struct Walker {
    /// The centre of the body's base.
    feet: Vec3,
    /// How fast the body moves up, negative while falling.
    vertical_speed: f32,
    grounded: bool,
    /// How far the eye is from `EYE_HEIGHT` above the feet. A step moves the feet
    /// a whole voxel at once; this lets the eye follow over a moment instead.
    eye_offset: f32,
}

impl Walker {
    /// Moves the body for `seconds` at `horizontal` speed while it falls or
    /// rises. It moves along x, z and y in turn, so it slides along whatever
    /// stops it.
    fn advance(
        &mut self,
        horizontal: Vec3,
        seconds: f32,
        solid: &impl Fn(IVec3) -> bool,
    ) {
        for axis in [0, 2] {
            let distance = horizontal[axis] * seconds;
            if distance == 0.0 || !self.slide(axis, distance, solid) || !self.grounded {
                continue;
            }
            // Stopped on the ground: climb onto what's in the way if it is one
            // voxel high and there is room above it.
            let raised = self.feet + Vec3::Y;
            let mut onto = raised;
            onto[axis] += distance;
            if !body_hits(raised, solid) && !body_hits(onto, solid) {
                self.feet = onto;
                self.eye_offset -= 1.0;
            }
        }

        // Walked off an edge one voxel high: step down rather than fall, so
        // going down stairs doesn't bounce.
        let below = |feet: Vec3| body_hits(feet - Vec3::Y * 2.0 * SKIN, solid);
        if self.grounded && self.vertical_speed <= 0.0 && !below(self.feet) {
            let lowered = self.feet - Vec3::Y;
            if !body_hits(lowered, solid) && below(lowered) {
                self.feet = lowered;
                self.eye_offset += 1.0;
            }
        }
        self.eye_offset = self.eye_offset.clamp(-1.0, 1.0);

        self.vertical_speed =
            (self.vertical_speed - GRAVITY * seconds).max(-MAX_FALL_SPEED);
        let stopped = self.slide(1, self.vertical_speed * seconds, solid);
        self.grounded = stopped && self.vertical_speed < 0.0;
        if stopped {
            self.vertical_speed = 0.0;
        }
    }

    /// Moves the feet `distance` along `axis`, or up to the face of the voxel in
    /// the way. Returns whether a voxel stopped it.
    ///
    /// `distance` must be less than a voxel, so that the voxel in the way lies in
    /// the layer the body moves into.
    fn slide(
        &mut self,
        axis: usize,
        distance: f32,
        solid: &impl Fn(IVec3) -> bool,
    ) -> bool {
        let mut moved = self.feet;
        moved[axis] += distance;
        if !body_hits(moved, solid) {
            self.feet = moved;
            return false;
        }
        let (low, high) = if axis == 1 {
            (0.0, BODY_HEIGHT)
        } else {
            (-BODY_RADIUS, BODY_RADIUS)
        };
        moved[axis] = if distance > 0.0 {
            (moved[axis] + high).floor() - high - SKIN
        } else {
            (moved[axis] + low).ceil() - low + SKIN
        };
        if !body_hits(moved, solid) {
            self.feet = moved;
        }
        true
    }
}

/// Whether a body with its feet at `feet` overlaps a solid voxel.
fn body_hits(feet: Vec3, solid: &impl Fn(IVec3) -> bool) -> bool {
    let min = (feet - Vec3::new(BODY_RADIUS, 0.0, BODY_RADIUS))
        .floor()
        .as_ivec3();
    let max = (feet + Vec3::new(BODY_RADIUS, BODY_HEIGHT, BODY_RADIUS))
        .ceil()
        .as_ivec3();
    (min.x..max.x).any(|x| {
        (min.y..max.y).any(|y| (min.z..max.z).any(|z| solid(IVec3::new(x, y, z))))
    })
}

/// Whether the voxel at a position stops a walking body, or `None` while the
/// voxel isn't known yet.
///
/// A voxel reads `WorldVoxel::Unset` in a chunk that generated empty, and also
/// where no chunk has been generated yet. Only whether a published chunk covers
/// it tells the empty space from the unknown.
fn ground(voxel_world: &VoxelWorld<MountainWorld>) -> impl Fn(IVec3) -> Option<bool> {
    let voxel = voxel_world.get_voxel_fn();
    let chunk_data = voxel_world.get_chunk_data_fn();
    move |position| match voxel(position) {
        WorldVoxel::Solid(_) => Some(true),
        WorldVoxel::Air => Some(false),
        WorldVoxel::Unset => (0..=MAX_LOD_LEVEL)
            .any(|level| chunk_data(ChunkKey::containing(position, level)).is_some())
            .then_some(false),
    }
}

/// Moves the camera with WASD, Shift to run and Space to jump.
///
/// It asks the voxel world which voxels are solid, so it walks on whatever the
/// world holds, edits included.
fn walk(
    time: Res<Time>,
    keys: Res<ButtonInput<KeyCode>>,
    voxel_world: VoxelWorld<MountainWorld>,
    camera: Single<(&mut Transform, &mut Walker, &Viewer)>,
) {
    let (mut transform, mut walker, viewer) = camera.into_inner();
    let ground = ground(&voxel_world);
    let reached_unknown = std::cell::Cell::new(false);
    let solid = |position: IVec3| {
        ground(position).unwrap_or_else(|| {
            reached_unknown.set(true);
            true
        })
    };

    let mut direction = Vec3::ZERO;
    for (key, towards) in [
        (KeyCode::KeyW, Vec3::NEG_Z),
        (KeyCode::KeyS, Vec3::Z),
        (KeyCode::KeyA, Vec3::NEG_X),
        (KeyCode::KeyD, Vec3::X),
    ] {
        if keys.pressed(key) {
            direction += towards;
        }
    }
    let speed = if keys.pressed(KeyCode::ShiftLeft) {
        RUN_SPEED
    } else {
        WALK_SPEED
    };
    let horizontal =
        Quat::from_rotation_y(viewer.yaw) * direction.normalize_or_zero() * speed;
    if walker.grounded && keys.just_pressed(KeyCode::Space) {
        walker.vertical_speed = JUMP_SPEED;
        walker.grounded = false;
    }

    // Split the frame into steps short enough for `Walker::slide`, and cap it
    // so a long frame doesn't send the body far in one go.
    let seconds = time.delta_secs().min(0.1);
    let vertical = (walker.vertical_speed.abs() + GRAVITY * seconds).min(MAX_FALL_SPEED);
    let fastest = horizontal.abs().max_element().max(vertical);
    let steps = (fastest * seconds / 0.4).ceil().max(1.0);
    let before = *walker;
    for _ in 0..steps as u32 {
        walker.advance(horizontal, seconds / steps, &solid);
    }
    // Where the body reached voxels that aren't known yet, it could fall
    // through ground that hasn't been generated, so it waits where it was.
    if reached_unknown.get() {
        *walker = before;
        walker.vertical_speed = 0.0;
    }
    walker.eye_offset *= (-12.0 * seconds).exp();
    transform.translation = walker.feet + Vec3::Y * (EYE_HEIGHT + walker.eye_offset);
}

/// The mouse captured for looking around, hidden and locked to the window, or
/// free.
fn mouse(captured: bool) -> CursorOptions {
    CursorOptions {
        visible: !captured,
        grab_mode: if captured {
            CursorGrabMode::Locked
        } else {
            CursorGrabMode::None
        },
        ..default()
    }
}

/// Esc frees the mouse, and a left click captures it again.
fn grab_mouse(
    keys: Res<ButtonInput<KeyCode>>,
    buttons: Res<ButtonInput<MouseButton>>,
    mut cursor: Single<&mut CursorOptions, With<PrimaryWindow>>,
) {
    if keys.just_pressed(KeyCode::Escape) {
        **cursor = mouse(false);
    } else if buttons.just_pressed(MouseButton::Left) {
        **cursor = mouse(true);
    }
}

/// The voxels a left click can place, with the key that chooses each and its
/// name.
const PLACEABLE: [(KeyCode, u8, &str); 4] = [
    (KeyCode::Digit1, STONE, "stone"),
    (KeyCode::Digit2, TIMBER, "timber"),
    (KeyCode::Digit3, LANTERN, "lantern"),
    (KeyCode::Digit4, CRYSTAL, "crystal"),
];

/// The index in `PLACEABLE` of the voxel a left click places.
#[derive(Resource, Default)]
struct Placing(usize);

fn choose_voxel(
    keys: Res<ButtonInput<KeyCode>>,
    mut placing: ResMut<Placing>,
    mut help_text: Single<&mut Text, With<HelpText>>,
) {
    for (index, (key, ..)) in PLACEABLE.iter().enumerate() {
        if keys.just_pressed(*key) {
            placing.0 = index;
            help_text.0 = help(index);
        }
    }
}

/// How far from the eye a click reaches, in voxels.
const REACH: f32 = 8.0;

/// While the mouse is captured, a left click places the chosen voxel against
/// the face under the crosshair, and a right click removes the voxel there,
/// in the world or in a felled tree.
///
/// Light propagation relights the voxels around an edit on its own. The Bevy
/// lights that colour the lamps are the example's, so it adds one in front of
/// a placed lantern or crystal, and puts out those left with no lamp voxel
/// beside them.
fn edit_voxels(
    mut commands: Commands,
    buttons: Res<ButtonInput<MouseButton>>,
    cursor: Single<&CursorOptions, With<PrimaryWindow>>,
    placing: Res<Placing>,
    mut voxel_world: VoxelWorld<MountainWorld>,
    camera: Single<(&Transform, &Walker)>,
    lamps: Query<(Entity, &Transform), With<LampLight>>,
) {
    let place = buttons.just_pressed(MouseButton::Left);
    let remove = buttons.just_pressed(MouseButton::Right);
    if cursor.grab_mode != CursorGrabMode::Locked || !(place || remove) {
        return;
    }
    let (transform, walker) = *camera;
    let eye = transform.translation;
    let ray = Ray3d::new(eye, transform.forward());
    // `raycast_grids` checks the world and every felled tree, which is a
    // sub-grid, and returns the nearest hit. The hit names its grid, and its
    // position and normal are on that grid, which `grid_mut` edits with the
    // same calls for either.
    let Some(hit) = voxel_world.raycast_grids(ray, REACH, &|_| true, &|_, _, _| true)
    else {
        return;
    };

    if remove {
        if let Some(mut grid) = voxel_world.grid_mut(hit.grid) {
            grid.set_voxel(hit.position, WorldVoxel::Air);
        }
        for (lamp, transform) in &lamps {
            if !lamp_voxel_near(&voxel_world, transform.translation) {
                commands.entity(lamp).despawn();
            }
        }
        return;
    }
    let Some(normal) = hit.normal else {
        return;
    };
    let at = hit.position + normal;
    let (_, material, _) = PLACEABLE[placing.0];
    if hit.grid != Grid::World {
        // A voxel placed on a felled tree gets no Bevy light, which would
        // stay behind as the tree moves. A lamp there still lights the tree
        // itself through light propagation.
        if let Some(mut grid) = voxel_world.grid_mut(hit.grid) {
            grid.set_voxel(at, WorldVoxel::Solid(material));
        }
        return;
    }
    if body_hits(walker.feet, &|position| position == at) {
        return;
    }
    voxel_world.set_voxel(at, WorldVoxel::Solid(material));
    let light_at = at.as_vec3() + 0.5 + normal.as_vec3() * 0.9;
    match material {
        LANTERN => spawn_lamp(&mut commands, light_at, LampKind::Lantern),
        CRYSTAL => spawn_lamp(&mut commands, light_at, LampKind::Crystal),
        _ => {}
    }
}

fn toggle_help(
    keys: Res<ButtonInput<KeyCode>>,
    mut help_text: Single<&mut Visibility, With<HelpText>>,
) {
    if keys.just_pressed(KeyCode::KeyH) {
        help_text.toggle_inherited_hidden();
    }
}

// ---------------------------------------------------------------------------
// Felling trees
//
// The world returns `FragmentDetection` with `auto_lift` on, so after each
// edit the crate looks for voxels the edit cut off from the rest of the world,
// and lifts each such piece into a new sub-grid: a set of voxels on a grid of
// its own, with a `Transform`. Cutting through a trunk lifts the tree above
// the cut. The systems below make a piece fall:
//
// 1. `start_falling` gives a new piece a dynamic body in avian3d, turning
//    away from the camera, so a tree standing on its stump tips over.
// 2. `build_piece_colliders` gives the piece's chunks colliders made of its
//    voxels.
// 3. `build_ground_colliders` gives the world's chunks around each piece
//    colliders, so it has something to land on. The world is large, so only
//    the chunks near a piece get them.
// 4. `drop_lost_pieces` despawns a piece that falls out of the world, or has
//    no voxels left.
//
// A piece stays a sub-grid where it lands. The walker passes through it, since
// it walks on the world's voxels only, but clicks reach it, so a felled tree
// can be cut again. A cut through a piece lifts the smaller part into a new
// piece of its own.

/// A piece cut loose from the world.
#[derive(Component)]
struct Piece;

/// How fast a new piece starts turning away from the camera, in radians per
/// second.
const FELLING_SPIN: f32 = 0.6;

/// A piece below this height has fallen out of the world, and despawns.
const LOST_BELOW: f32 = -64.0;

/// The smallest box holding the voxels of the sub-grid `piece`, in the frame
/// of the world's root, as its minimum and maximum corners.
fn piece_box(
    voxel_world: &VoxelWorld<MountainWorld>,
    piece: Entity,
) -> Option<(Vec3, Vec3)> {
    let grid = voxel_world.grid(Grid::SubGrid(piece))?;
    let (min, max) = grid.bounds()?;
    let pose = grid.pose();
    let (min, max) = (min.as_vec3(), max.as_vec3());
    let corners = [
        Vec3::new(min.x, min.y, min.z),
        Vec3::new(max.x, min.y, min.z),
        Vec3::new(min.x, max.y, min.z),
        Vec3::new(max.x, max.y, min.z),
        Vec3::new(min.x, min.y, max.z),
        Vec3::new(max.x, min.y, max.z),
        Vec3::new(min.x, max.y, max.z),
        Vec3::new(max.x, max.y, max.z),
    ]
    .map(|corner| pose.transform_point(corner));
    let low = corners.into_iter().reduce(Vec3::min)?;
    let high = corners.into_iter().reduce(Vec3::max)?;
    Some((low, high))
}

/// Gives each piece whose lift has landed a dynamic body, turning so that its
/// top moves away from the camera.
///
/// `Lifted` arrives in the frame the piece's voxels first show in its
/// sub-grid. A lift spawns the piece as a child of the grid it was cut from.
/// A piece cut from the world is a child of the world's root already. One
/// cut from another piece would move along with that piece, so it moves to
/// the world's root, which is that piece's parent, keeping its place.
fn start_falling(
    mut lifted: MessageReader<Lifted<MountainWorld>>,
    voxel_world: VoxelWorld<MountainWorld>,
    camera: Single<&Transform, With<Viewer>>,
    parents: Query<&ChildOf>,
    mut commands: Commands,
) {
    for message in lifted.read() {
        let piece = message.sub_grid;
        if let Grid::SubGrid(source) = message.source {
            if let Ok(&ChildOf(root)) = parents.get(source) {
                commands.entity(piece).set_parent_in_place(root);
            }
        }
        let Some((low, high)) = piece_box(&voxel_world, piece) else {
            continue;
        };
        // Turning about this axis moves the piece's top along `away`.
        let away = ((low + high) / 2.0 - camera.translation)
            .with_y(0.0)
            .normalize_or_zero();
        let axis = Vec3::Y.cross(away);
        commands.entity(piece).try_insert((
            RigidBody::Dynamic,
            AngularVelocity(axis * FELLING_SPIN),
            Piece,
        ));
    }
}

/// Gives each chunk of a piece that just showed new voxels a collider made of
/// its solid voxels.
///
/// A sub-grid's chunks are children of the sub-grid, so avian adds their
/// colliders to the piece's body.
fn build_piece_colliders(
    mut published: MessageReader<ChunkPublished<MountainWorld>>,
    voxel_world: VoxelWorld<MountainWorld>,
    mut commands: Commands,
) {
    for message in published.read() {
        if message.grid == Grid::World {
            continue;
        }
        let Some(chunk) = voxel_world
            .grid(message.grid)
            .and_then(|grid| grid.get_chunk_data(message.chunk_key))
        else {
            continue;
        };
        // The chunk's entity sits one voxel below its first voxel, so its
        // voxels are at 1 to `CHUNK_SIZE` on each axis.
        let voxels: Vec<IVec3> =
            solid_voxels(chunk.key().min_voxel(), false, |position| {
                chunk.get_voxel_at_world_position(position)
            })
            .into_iter()
            .map(|offset| offset + IVec3::ONE)
            .collect();
        if voxels.is_empty() {
            commands.entity(message.entity).try_remove::<Collider>();
        } else {
            commands
                .entity(message.entity)
                .try_insert(Collider::voxels(Vec3::ONE, &voxels));
        }
    }
}

/// The world's chunks that have colliders, with the entity holding each one's
/// collider.
#[derive(Resource, Default)]
struct GroundColliders(HashMap<ChunkKey, Entity>);

/// How far around a piece the world's chunks get colliders, in voxels.
const GROUND_MARGIN: f32 = 4.0;

/// Gives the world's chunks around each piece colliders, and rebuilds the
/// collider of each such chunk whose voxels changed.
///
/// Each collider holds only the chunk's surface voxels, those with a face
/// against a voxel that isn't solid, since the ground is mostly buried rock.
fn build_ground_colliders(
    mut published: MessageReader<ChunkPublished<MountainWorld>>,
    pieces: Query<Entity, With<Piece>>,
    voxel_world: VoxelWorld<MountainWorld>,
    mut ground: ResMut<GroundColliders>,
    mut commands: Commands,
) {
    let mut chunks: Vec<ChunkKey> = published
        .read()
        .filter(|message| message.grid == Grid::World)
        .map(|message| message.chunk_key)
        .filter(|key| ground.0.contains_key(key))
        .collect();
    for piece in &pieces {
        let Some((low, high)) = piece_box(&voxel_world, piece) else {
            continue;
        };
        let size = CHUNK_SIZE_I as f32;
        let first = ((low - GROUND_MARGIN) / size).floor().as_ivec3();
        let last = ((high + GROUND_MARGIN) / size).floor().as_ivec3();
        for z in first.z..=last.z {
            for y in first.y..=last.y {
                for x in first.x..=last.x {
                    let key = ChunkKey::containing_chunk(IVec3::new(x, y, z), 0);
                    if !ground.0.contains_key(&key) && !chunks.contains(&key) {
                        chunks.push(key);
                    }
                }
            }
        }
    }

    for key in chunks {
        let first = key.min_voxel();
        let entity = *ground.0.entry(key).or_insert_with(|| {
            commands
                .spawn((
                    RigidBody::Static,
                    Transform::from_translation(first.as_vec3()),
                ))
                .id()
        });
        // A chunk not generated yet gets its collider when it publishes,
        // since it is in `GroundColliders` now.
        let voxels = voxel_world
            .get_chunk_data(key)
            .filter(|chunk| !chunk.is_empty())
            .map(|chunk| {
                solid_voxels(first, true, |position| {
                    chunk.get_voxel_at_world_position(position)
                })
            })
            .unwrap_or_default();
        if voxels.is_empty() {
            commands.entity(entity).try_remove::<Collider>();
        } else {
            commands
                .entity(entity)
                .try_insert(Collider::voxels(Vec3::ONE, &voxels));
        }
    }
}

/// The solid voxels of the level-0 chunk whose first voxel is `first`, as
/// offsets from `first`. `voxel_at` reads the chunk's voxel at a position on
/// its grid. With `surface_only`, it keeps only the voxels with a face against
/// a voxel that isn't solid.
///
/// A chunk's data repeats its neighbours' voxels in a border one voxel wide,
/// so `voxel_at` can tell whether a voxel at the chunk's edge is on the
/// surface.
fn solid_voxels(
    first: IVec3,
    surface_only: bool,
    voxel_at: impl Fn(IVec3) -> Option<WorldVoxel>,
) -> Vec<IVec3> {
    let solid = |position| voxel_at(position).is_some_and(|voxel| voxel.is_solid());
    let buried = |position: IVec3| {
        [IVec3::X, IVec3::Y, IVec3::Z]
            .into_iter()
            .all(|axis| solid(position - axis) && solid(position + axis))
    };
    let mut voxels = Vec::new();
    for z in 0..CHUNK_SIZE_I {
        for y in 0..CHUNK_SIZE_I {
            for x in 0..CHUNK_SIZE_I {
                let offset = IVec3::new(x, y, z);
                let position = first + offset;
                if solid(position) && !(surface_only && buried(position)) {
                    voxels.push(offset);
                }
            }
        }
    }
    voxels
}

/// Despawns each piece that has fallen out of the world, and each piece
/// whose last voxel was removed. The crate keeps an empty sub-grid until the
/// game despawns it.
fn drop_lost_pieces(
    mut voxel_world: VoxelWorld<MountainWorld>,
    pieces: Query<(Entity, &GlobalTransform), With<Piece>>,
) {
    for (entity, transform) in &pieces {
        let empty = voxel_world
            .grid(Grid::SubGrid(entity))
            .is_some_and(|grid| grid.bounds().is_none());
        if empty || transform.translation().y < LOST_BELOW {
            voxel_world.despawn_sub_grid(entity);
        }
    }
}
