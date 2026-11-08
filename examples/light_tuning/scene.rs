//! The world light is tuned in: a meadow at the foot of a mountain, with a
//! place to see each kind of setting at work.
//!
//! Every voxel is a pure function of its position, as sky light needs.

use std::sync::LazyLock;

use bevy::{platform::collections::HashMap, prelude::*};
use bevy_voxel_world::prelude::*;
use noise::{MultiFractal, NoiseFn, Perlin, RidgedMulti};

// Materials.
pub const GRASS: u8 = 0;
pub const DIRT: u8 = 1;
pub const STONE: u8 = 2;
pub const SNOWY_STONE: u8 = 3;
pub const COBBLESTONE: u8 = 4;
pub const DARK_PLANKS: u8 = 5;
pub const TIMBER: u8 = 6;
pub const ROOF: u8 = 7;
pub const LEAVES: u8 = 8;
pub const BARK: u8 = 9;
pub const LANTERN: u8 = 10;
pub const CRYSTAL: u8 = 11;
pub const GRAVEL: u8 = 12;
pub const FLOOR: u8 = 13;

/// The texture the materials' tiles come from, which the
/// `light_mountain_house` example uses too, and how many tiles it holds.
pub const TEXTURE: &str = "mountain_house_texture.png";
pub const TILE_COUNT: u32 = 18;

// Tiles of `TEXTURE`, from the top.
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

/// The tiles of a material's top, sides and bottom.
pub fn texture_indices(material: u8) -> [u32; 3] {
    match material {
        GRASS => [TILE_GRASS_TOP, TILE_GRASS_SIDE, TILE_DIRT],
        DIRT => [TILE_DIRT; 3],
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
        _ => [TILE_STONE; 3],
    }
}

// --- Places -------------------------------------------------------------------

/// A place to look from, each showing some settings at work.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Place {
    Meadow,
    Cave,
    Chamber,
    Island,
    Arch,
    House,
    Well,
    Overview,
}

impl Place {
    pub const ALL: [Place; 8] = [
        Place::Meadow,
        Place::Cave,
        Place::Chamber,
        Place::Island,
        Place::Arch,
        Place::House,
        Place::Well,
        Place::Overview,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Place::Meadow => "Meadow",
            Place::Cave => "Cave",
            Place::Chamber => "Chamber",
            Place::Island => "Island",
            Place::Arch => "Arch",
            Place::House => "House",
            Place::Well => "Well",
            Place::Overview => "Overview",
        }
    }

    /// The number key that moves the camera here.
    pub fn key(self) -> usize {
        Place::ALL
            .iter()
            .position(|place| *place == self)
            .unwrap_or(0)
            + 1
    }

    /// Where the camera stands, and what it looks at.
    pub fn view(self) -> (Vec3, Vec3) {
        let (eye, target) = match self {
            // The mountain, the cave mouth, the house and the well.
            Place::Meadow => ([36.0, 58.0, 92.0], [0.0, 42.0, -40.0]),
            // Just inside the cave mouth, looking down the hall.
            Place::Cave => ([-10.0, 44.0, -48.0], [4.0, 36.0, -100.0]),
            // The deep chamber behind the narrow passage, lit by crystals.
            Place::Chamber => ([4.0, 39.0, -129.0], [9.0, 33.0, -150.0]),
            // The floating island from the height of its underside: the
            // shade straight below it, and its sun shadow beyond, by the house.
            Place::Island => ([150.0, 74.0, 134.0], [72.0, 66.0, 58.0]),
            // The ground under the arch, where the sun stops at its edge.
            Place::Arch => ([-58.0, 44.0, 64.0], [-88.0, 36.0, 20.0]),
            // Inside the house, lit through its door and windows.
            Place::House => ([48.5, 37.5, -10.5], [58.0, 33.5, -19.0]),
            // The far end of the room off the bottom of the well, looking back
            // at the shaft's floor, which sky light falls on without loss, and
            // the light fading into the room from there.
            Place::Well => ([-12.5, 8.5, -10.0], [-30.0, 8.0, -10.0]),
            // Everything from above.
            Place::Overview => ([-140.0, 175.0, 170.0], [40.0, 45.0, 0.0]),
        };
        (Vec3::from_array(eye), Vec3::from_array(target))
    }
}

// --- The ground ---------------------------------------------------------------

/// The lowest open voxel of the flat meadow.
pub const GROUND: i32 = 32;

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// The house's corner of least x and z, on the meadow's ground.
const HOUSE_MIN: IVec3 = IVec3::new(46, GROUND, -21);
const HOUSE_CENTRE: Vec2 = Vec2::new(54.0, -15.0);
/// The column at the middle of the well's shaft.
const WELL: IVec2 = IVec2::new(-30, -10);
/// How far the shaft reaches from its middle column, along x and z.
const WELL_HALF_WIDTH: i32 = 3;
/// The y of the well's floor.
const WELL_BOTTOM: i32 = 6;

/// Where the ground is kept flat, so what stands there sits level: the
/// centre, the radius kept flat and the radius the meadow's bumps come back
/// by.
const FLAT_SPOTS: [(Vec2, f32, f32); 3] = [
    (HOUSE_CENTRE, 13.0, 28.0),
    (Vec2::new(WELL.x as f32, WELL.y as f32), 7.0, 20.0),
    (Vec2::new(0.0, -45.0), 14.0, 34.0),
];

/// A column of the ground: its height, below which voxels are solid, and how
/// steep it is.
#[derive(Clone, Copy)]
struct Column {
    height: f32,
    slope: f32,
}

/// The ground: a gently rolling meadow with a range of ridged peaks north of
/// z = -36, which falls away again to either side and behind, and a dome of
/// rock over the cave.
struct Terrain {
    rolling: Perlin,
    ridges: RidgedMulti<Perlin>,
}

/// The middle of the dome of rock the cave lies under.
const CAVE_DOME: Vec2 = Vec2::new(6.0, -95.0);

impl Terrain {
    fn new() -> Self {
        Self {
            rolling: Perlin::new(7),
            ridges: RidgedMulti::<Perlin>::new(11)
                .set_octaves(4)
                .set_frequency(1.0 / 130.0),
        }
    }

    /// How much of the meadow's bumps a column keeps: 0 where something
    /// stands, rising to 1 away from it.
    fn bumpiness(x: f32, z: f32) -> f32 {
        FLAT_SPOTS
            .iter()
            .map(|(centre, flat, bumpy)| {
                smoothstep(*flat, *bumpy, Vec2::new(x, z).distance(*centre))
            })
            .fold(1.0, f32::min)
    }

    fn height(&self, x: f32, z: f32) -> f32 {
        let bumps = self.rolling.get([x as f64 / 60.0, z as f64 / 60.0]) as f32;
        let meadow = GROUND as f32 + bumps * 4.0 * Self::bumpiness(x, z);
        let along = smoothstep(-36.0, -110.0, z) * smoothstep(-215.0, -145.0, z);
        let across = smoothstep(230.0, 100.0, x.abs());
        let ridges =
            (self.ridges.get([x as f64, z as f64]) as f32 * 0.5 + 0.5).clamp(0.0, 1.0);
        let range = along * across * (30.0 + 85.0 * ridges);
        let dome = 34.0
            * smoothstep(70.0, 40.0, Vec2::new(x, z).distance(CAVE_DOME))
            * smoothstep(-38.0, -56.0, z);
        meadow + range.max(dome)
    }

    fn column(&self, x: i32, z: i32) -> Column {
        let (x, z) = (x as f32, z as f32);
        let across_x = self.height(x + 1.0, z) - self.height(x - 1.0, z);
        let across_z = self.height(x, z + 1.0) - self.height(x, z - 1.0);
        Column {
            height: self.height(x, z),
            slope: across_x.abs().max(across_z.abs()) / 2.0,
        }
    }

    /// The material of a solid ground voxel at `y`: grass over dirt over
    /// stone on the meadow, bare stone on steep and high ground, and snow on
    /// the peaks.
    fn material(column: Column, y: i32) -> u8 {
        let depth = column.height - y as f32;
        let top = depth <= 1.0;
        if column.height > 104.0 {
            return if top { SNOWY_STONE } else { STONE };
        }
        if column.slope > 1.0 || column.height > 90.0 {
            return STONE;
        }
        if top {
            GRASS
        } else if depth <= 4.0 {
            DIRT
        } else {
            STONE
        }
    }
}

// --- Shapes -------------------------------------------------------------------

fn in_ellipsoid(point: Vec3, centre: Vec3, radii: Vec3) -> bool {
    ((point - centre) / radii).length_squared() < 1.0
}

fn in_capsule(point: Vec3, from: Vec3, to: Vec3, radius: f32) -> bool {
    let along = to - from;
    let t = ((point - from).dot(along) / along.length_squared()).clamp(0.0, 1.0);
    point.distance_squared(from + along * t) < radius * radius
}

/// The cave's hall: a wide tunnel along z, from its mouth in the
/// mountain's foot to a rounded end, with a floor at the meadow's ground
/// level.
///
/// Sky fill passes only through 8-voxel cells with no solid voxel in them,
/// so the hall is wide and tall enough to hold open cells from the mouth to
/// the back: `x` from -16 to 16 and `y` from 32 to 48 always stay open.
/// The hall crosses chunk boundaries at z = -64 and z = -96, and its fill
/// drops by `PathFill::whole_face_loss` scaled by the opening's narrowness at
/// each, so the hall darkens in steps that the loss settings change.
const HALL_MIDDLE: Vec2 = Vec2::new(0.0, 40.0);
/// The hall's half width and half height, before they swell and narrow
/// along its length by up to `HALL_SWELL`.
const HALL_RADII: Vec2 = Vec2::new(22.0, 12.0);
const HALL_SWELL: Vec2 = Vec2::new(1.5, 1.0);
/// How far the hall's rough walls reach past its smooth shape. They only
/// widen it, so the open cells stay open.
const HALL_ROUGHNESS: f32 = 3.0;
static HALL_WALLS: LazyLock<Perlin> = LazyLock::new(|| Perlin::new(29));
/// The hall runs from its mouth to where its end starts rounding off, and
/// the end is this long.
const HALL_MOUTH: f32 = -30.0;
const HALL_END: f32 = -104.0;
const HALL_END_LENGTH: f32 = 10.0;
/// The passage from the back of the hall to the deep chamber, straight
/// along z. Its middle is the middle of the 8-voxel cell at x 0 to 8 and
/// y 32 to 40, and its radius holds that cell open along its length.
const PASSAGE: (Vec3, Vec3) =
    (Vec3::new(4.0, 36.0, -100.0), Vec3::new(4.0, 36.0, -136.0));
const PASSAGE_RADIUS: f32 = 6.5;
/// The deep chamber, far enough behind the passage that sky light spreading
/// from the mouth never reaches it.
const CHAMBER_CENTRE: Vec3 = Vec3::new(6.0, 38.0, -146.0);
const CHAMBER_RADII: Vec3 = Vec3::new(17.0, 10.0, 15.0);

/// Whether `point` is inside the hall, whose cross-section is a rounded
/// square, the shape that holds the most whole cells for its size.
fn in_hall(point: Vec3) -> bool {
    if point.z > HALL_MOUTH {
        return false;
    }
    let swell = Vec2::new((point.z / 9.0).sin(), (point.z / 13.0 + 1.0).sin());
    let mut radii = HALL_RADII + HALL_SWELL * swell;
    if point.z < HALL_END {
        let into_end = (HALL_END - point.z) / HALL_END_LENGTH;
        if into_end >= 1.0 {
            return false;
        }
        radii *= (1.0 - into_end * into_end).sqrt();
    }
    let across = Vec2::new(point.x, point.y) - HALL_MIDDLE;
    let shape = |radii: Vec2| {
        let offset = across / radii;
        offset.x.powi(4) + offset.y.powi(4)
    };
    if shape(radii + HALL_ROUGHNESS) >= 1.0 {
        return false;
    }
    let rough = HALL_WALLS.get((point / 6.0).as_dvec3().to_array()) as f32 * 0.5 + 0.5;
    shape(radii + HALL_ROUGHNESS * rough.clamp(0.0, 1.0)) < 1.0
}

/// Whether `position` is open space in the cave, whose floor is the meadow's
/// ground level.
fn in_cave(position: IVec3) -> bool {
    if position.y < GROUND || position.z as f32 > HALL_MOUTH {
        return false;
    }
    let point = position.as_vec3() + 0.5;
    in_hall(point)
        || in_capsule(point, PASSAGE.0, PASSAGE.1, PASSAGE_RADIUS)
        || in_ellipsoid(point, CHAMBER_CENTRE, CHAMBER_RADII)
}

/// Whether `position` is open space in the well: a shaft open to the sky,
/// and a room off its bottom that only spreading sky light reaches.
fn in_well(position: IVec3) -> bool {
    let offset = position.xz() - WELL;
    let shaft = offset.abs().max_element() <= WELL_HALF_WIDTH
        && (WELL_BOTTOM..60).contains(&position.y);
    let room = (WELL_HALF_WIDTH..=WELL_HALF_WIDTH + 16).contains(&offset.x)
        && offset.y.abs() <= 2
        && (WELL_BOTTOM..WELL_BOTTOM + 5).contains(&position.y);
    shaft || room
}

/// Whether `position` is in the cobblestone wall around the well's shaft.
fn lines_well(position: IVec3) -> bool {
    let offset = position.xz() - WELL;
    offset.abs().max_element() == WELL_HALF_WIDTH + 1
        && (WELL_BOTTOM..GROUND).contains(&position.y)
        && !in_well(position)
}

/// The floating island: flat on top, tapering underneath, and wide enough
/// that sky light spreading in under it from around its edge has nearly
/// faded out by the middle of the ground below.
const ISLAND_CENTRE: Vec2 = Vec2::new(72.0, 58.0);
const ISLAND_RADII: Vec2 = Vec2::new(32.0, 28.0);
const ISLAND_TOP: i32 = 108;
const ISLAND_DEPTH: f32 = 20.0;

/// The arch: a ring of rock standing on the meadow, facing along z.
const ARCH_CENTRE: Vec3 = Vec3::new(-88.0, 30.0, 22.0);
/// The radius of the ring, to the middle of its rock.
const ARCH_RADIUS: f32 = 19.0;
/// The radius of the ring's rock around that middle.
const ARCH_THICKNESS: f32 = 6.5;

impl Terrain {
    fn island(&self, position: IVec3) -> Option<u8> {
        if !(ISLAND_TOP - 26..ISLAND_TOP).contains(&position.y) {
            return None;
        }
        let point = Vec2::new(position.x as f32, position.z as f32) + 0.5;
        let offset = (point - ISLAND_CENTRE) / ISLAND_RADII;
        if offset.abs().max_element() > 1.2 {
            return None;
        }
        let ragged =
            self.rolling
                .get([point.x as f64 / 9.0, point.y as f64 / 9.0]) as f32;
        let reach = offset.length_squared() + ragged * 0.15;
        if reach >= 1.0 {
            return None;
        }
        let bottom = ISLAND_TOP as f32 - 3.0 - ISLAND_DEPTH * (1.0 - reach).powf(0.7);
        let y = position.y as f32;
        if y < bottom {
            return None;
        }
        let depth = ISLAND_TOP - position.y;
        Some(match depth {
            1 => GRASS,
            2..=3 => DIRT,
            _ => STONE,
        })
    }

    fn in_arch(&self, position: IVec3) -> bool {
        let offset = position.as_vec3() + 0.5 - ARCH_CENTRE;
        let outer = ARCH_RADIUS + ARCH_THICKNESS + 1.5;
        if offset.y < -4.0 || offset.z.abs() > outer || offset.xy().length() > outer {
            return false;
        }
        let around_ring = Vec2::new(offset.xy().length() - ARCH_RADIUS, offset.z);
        let point = position.as_dvec3() / 5.0;
        let rough = self.rolling.get([point.x, point.y, point.z]) as f32;
        around_ring.length() < ARCH_THICKNESS + rough * 1.5
    }
}

// --- What's built -------------------------------------------------------------

/// The voxels placed on top of the ground: the house, the well's rim, the
/// lamps and the trees.
#[derive(Default)]
struct Blueprint {
    voxels: HashMap<IVec3, WorldVoxel>,
}

impl Blueprint {
    fn set(&mut self, at: IVec3, material: u8) {
        self.voxels.insert(at, WorldVoxel::Solid(material));
    }

    /// Fills the box from `min` to `max`, both inclusive.
    fn fill(&mut self, min: IVec3, max: IVec3, material: u8) {
        for x in min.x..=max.x {
            for y in min.y..=max.y {
                for z in min.z..=max.z {
                    self.set(IVec3::new(x, y, z), material);
                }
            }
        }
    }
}

static BLUEPRINT: LazyLock<HashMap<IVec3, WorldVoxel>> = LazyLock::new(|| {
    let terrain = Terrain::new();
    let mut blueprint = Blueprint::default();
    build_house(&mut blueprint);
    build_well(&mut blueprint);
    place_lamps(&mut blueprint);
    plant_trees(&mut blueprint, &terrain);
    blueprint.voxels
});

/// A cottage of planks on a stone footing, with a door and windows to let sky
/// light in, a lantern inside on a table and one outside by the door.
fn build_house(b: &mut Blueprint) {
    const WIDTH: i32 = 16;
    const DEPTH: i32 = 12;
    const WALL_HEIGHT: i32 = 6;
    let at = |x, y, z| HOUSE_MIN + IVec3::new(x, y, z);

    b.fill(at(-1, -4, -1), at(WIDTH + 1, -1, DEPTH + 1), COBBLESTONE);
    b.fill(at(1, -1, 1), at(WIDTH - 1, -1, DEPTH - 1), FLOOR);

    let door = |x: i32, y: i32, z: i32| z == DEPTH && (7..=9).contains(&x) && y <= 2;
    let window = |x: i32, y: i32, z: i32| {
        let front = z == DEPTH && ((2..=4).contains(&x) || (12..=14).contains(&x));
        let back = z == 0 && (6..=10).contains(&x);
        let sides = (x == 0 || x == WIDTH) && (5..=7).contains(&z);
        (2..=3).contains(&y) && (front || back || sides)
    };
    for y in 0..WALL_HEIGHT {
        for x in 0..=WIDTH {
            for z in 0..=DEPTH {
                let on_x_wall = x == 0 || x == WIDTH;
                let on_z_wall = z == 0 || z == DEPTH;
                if !(on_x_wall || on_z_wall) || door(x, y, z) || window(x, y, z) {
                    continue;
                }
                let material = if on_x_wall && on_z_wall || y == WALL_HEIGHT - 1 {
                    TIMBER
                } else if y == 0 {
                    COBBLESTONE
                } else {
                    DARK_PLANKS
                };
                b.set(at(x, y, z), material);
            }
        }
    }

    // A gabled roof with its ridge along x, one voxel thick and overhanging
    // the walls by one, over gable walls at each end.
    for step in 0..=DEPTH / 2 + 1 {
        let y = WALL_HEIGHT + step;
        for x in -1..=WIDTH + 1 {
            b.set(at(x, y, step - 1), ROOF);
            b.set(at(x, y, DEPTH + 1 - step), ROOF);
        }
        for z in step..=DEPTH - step {
            b.set(at(0, y, z), DARK_PLANKS);
            b.set(at(WIDTH, y, z), DARK_PLANKS);
        }
    }

    b.set(at(13, 0, 2), TIMBER);
    b.set(at(13, 1, 2), LANTERN);
    b.fill(at(5, 0, DEPTH + 2), at(5, 1, DEPTH + 2), COBBLESTONE);
    b.set(at(5, 2, DEPTH + 2), LANTERN);
}

/// A rim of cobblestone around the top of the well's shaft.
fn build_well(b: &mut Blueprint) {
    let rim = WELL_HALF_WIDTH + 1;
    for x in -rim..=rim {
        for z in -rim..=rim {
            if x.abs() == rim || z.abs() == rim {
                let (x, z) = (WELL.x + x, WELL.y + z);
                b.fill(
                    IVec3::new(x, GROUND - 1, z),
                    IVec3::new(x, GROUND + 1, z),
                    COBBLESTONE,
                );
            }
        }
    }
}

/// Lanterns on posts along the cave hall's walls, outside the open cells
/// sky fill passes through, and clusters of crystals on the deep chamber's
/// floor.
fn place_lamps(b: &mut Blueprint) {
    for (x, z) in [(-18, -62), (18, -76), (-18, -90)] {
        b.fill(
            IVec3::new(x, GROUND, z),
            IVec3::new(x, GROUND + 1, z),
            COBBLESTONE,
        );
        b.set(IVec3::new(x, GROUND + 2, z), LANTERN);
    }
    for (x, z, height) in [
        (0, -150, 3),
        (1, -150, 2),
        (0, -151, 1),
        (12, -142, 2),
        (13, -141, 1),
        (8, -154, 2),
        (16, -150, 1),
        (15, -149, 2),
        (-2, -144, 1),
    ] {
        b.fill(
            IVec3::new(x, GROUND, z),
            IVec3::new(x, GROUND + height - 1, z),
            CRYSTAL,
        );
    }
}

fn hash(x: i32, z: i32) -> u32 {
    let mut h =
        (x as u32).wrapping_mul(0x8da6_b343) ^ (z as u32).wrapping_mul(0xd816_3841);
    h ^= h >> 13;
    h = h.wrapping_mul(0x5bd1_e995);
    h ^ (h >> 15)
}

/// Whether a tree at `x`, `z` stays out of the way of the places, the walker's
/// path and the cameras.
fn room_for_tree(x: i32, z: i32) -> bool {
    let at = Vec2::new(x as f32, z as f32);
    let cave_front = x.abs() < 24 && z < -28;
    let path = x.abs() < 5 && (-50..40).contains(&z);
    let near_camera = Place::ALL
        .iter()
        .any(|place| place.view().0.xz().distance(at) < 9.0);
    // The island and the ground under it, seen from the island's place.
    let island_view = at.distance(ISLAND_CENTRE) < 44.0
        || in_capsule(
            at.extend(0.0),
            Place::Island.view().0.xz().extend(0.0),
            ISLAND_CENTRE.extend(0.0),
            30.0,
        );
    at.distance(HOUSE_CENTRE) > 18.0
        && at.distance(WELL.as_vec2()) > 10.0
        && at.distance(ARCH_CENTRE.xz()) > 30.0
        && !cave_front
        && !path
        && !near_camera
        && !island_view
}

/// Spruces scattered over the meadow and the mountain's lower slopes, and
/// three on the floating island.
fn plant_trees(b: &mut Blueprint, terrain: &Terrain) {
    const CELL: i32 = 10;
    for cell_x in -22..=22 {
        for cell_z in -22..=22 {
            let h = hash(cell_x, cell_z);
            if h % 100 >= 38 {
                continue;
            }
            let x = cell_x * CELL + (h >> 8) as i32 % 7;
            let z = cell_z * CELL + (h >> 12) as i32 % 7;
            if !room_for_tree(x, z) {
                continue;
            }
            let column = terrain.column(x, z);
            if column.slope > 0.8 || column.height > 80.0 {
                continue;
            }
            let base = IVec3::new(x, column.height.ceil() as i32, z);
            spruce(b, terrain, base, 7 + (h >> 16) as i32 % 6);
        }
    }
    for (x, z, height) in [(72, 58, 13), (62, 49, 9), (83, 66, 10)] {
        spruce(b, terrain, IVec3::new(x, ISLAND_TOP, z), height);
    }
}

/// A spruce: a trunk under tiers of leaves that narrow to a point.
fn spruce(b: &mut Blueprint, terrain: &Terrain, base: IVec3, height: i32) {
    b.fill(base, base + IVec3::Y * (height - 3), BARK);
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
                // Leaves on a slope stop at the ground.
                if ((base.y + y) as f32) >= terrain.height(x as f32, z as f32) {
                    b.set(IVec3::new(x, base.y + y, z), LEAVES);
                }
            }
        }
    }
}

// --- The world ----------------------------------------------------------------

/// Looks up the world's voxels, keeping each column of ground it works out.
pub struct Scene {
    terrain: Terrain,
    columns: HashMap<(i32, i32), Column>,
}

impl Default for Scene {
    fn default() -> Self {
        Self {
            terrain: Terrain::new(),
            columns: HashMap::new(),
        }
    }
}

impl Scene {
    /// The voxel at `position`: what's built first, then the cave and the
    /// well, the island and the arch, and the ground.
    pub fn voxel(&mut self, position: IVec3) -> WorldVoxel {
        if let Some(voxel) = BLUEPRINT.get(&position) {
            return *voxel;
        }
        if in_cave(position) || in_well(position) {
            return WorldVoxel::Air;
        }
        if let Some(material) = self.terrain.island(position) {
            return WorldVoxel::Solid(material);
        }
        if self.terrain.in_arch(position) {
            let top = !self.terrain.in_arch(position + IVec3::Y);
            let high = position.y as f32 > ARCH_CENTRE.y + ARCH_RADIUS * 0.5;
            return WorldVoxel::Solid(if top && high { GRASS } else { STONE });
        }
        let terrain = &self.terrain;
        let column = *self
            .columns
            .entry((position.x, position.z))
            .or_insert_with(|| terrain.column(position.x, position.z));
        if position.y as f32 >= column.height {
            return WorldVoxel::Air;
        }
        let above = position + IVec3::Y;
        if in_cave(above) || in_well(above) {
            return WorldVoxel::Solid(GRAVEL);
        }
        if lines_well(position) {
            return WorldVoxel::Solid(COBBLESTONE);
        }
        WorldVoxel::Solid(Terrain::material(column, position.y))
    }

    /// The y of the first open voxel above a solid one, looking down the
    /// column at `x`, `z` from `from`.
    pub fn floor(&mut self, x: i32, z: i32, from: i32) -> i32 {
        let mut open_above = false;
        for y in (0..=from).rev() {
            let solid = self.voxel(IVec3::new(x, y, z)).is_solid();
            if solid && open_above {
                return y + 1;
            }
            open_above = !solid;
        }
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_camera_stands_in_the_open() {
        let mut scene = Scene::default();
        for place in Place::ALL {
            let eye = place.view().0.floor().as_ivec3();
            assert!(!scene.voxel(eye).is_solid(), "{place:?}");
        }
    }

    #[test]
    fn the_passage_joins_the_hall_and_the_chamber() {
        assert!(in_hall(PASSAGE.0));
        assert!(in_ellipsoid(PASSAGE.1, CHAMBER_CENTRE, CHAMBER_RADII));
    }

    /// Every voxel of the 8-voxel cells at `x`, `y` and `z` from `min` to
    /// `max`, in steps of a cell, is open.
    fn cells_are_open(scene: &mut Scene, min: IVec3, max: IVec3) {
        for x in min.x..max.x {
            for y in min.y..max.y {
                for z in min.z..max.z {
                    let at = IVec3::new(x, y, z);
                    assert!(!scene.voxel(at).is_solid(), "{at}");
                }
            }
        }
    }

    #[test]
    fn sky_fill_has_open_cells_from_the_mouth_to_the_chamber() {
        let mut scene = Scene::default();
        cells_are_open(
            &mut scene,
            IVec3::new(-16, GROUND, HALL_END as i32),
            IVec3::new(16, GROUND + 16, HALL_MOUTH as i32),
        );
        cells_are_open(
            &mut scene,
            IVec3::new(0, GROUND, -136),
            IVec3::new(8, GROUND + 8, -100),
        );
    }

    #[test]
    fn the_lamps_stand_in_the_cave() {
        let mut scene = Scene::default();
        for (position, voxel) in BLUEPRINT.iter() {
            let lamp = matches!(voxel, WorldVoxel::Solid(LANTERN | CRYSTAL));
            if lamp && position.z < -40 {
                assert!(in_cave(*position), "{position}");
            }
        }
        assert!(!scene.voxel(IVec3::new(0, GROUND, -60)).is_solid());
    }
}
