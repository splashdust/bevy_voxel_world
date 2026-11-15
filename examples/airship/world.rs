//! The world the airship flies through, and the airship as it stands on its
//! dock before its first lift.
//!
//! West of x = 100 lie open hills, with a valley along z = 50 that runs east
//! to a rock mass, a mesa, between x = 100 and x = 260. A tunnel enters its
//! west cliff in that valley, opens into a hall in the middle of the rock
//! and leaves through the east cliff. The hall is too far from either mouth
//! for sky light to reach, so only the lamp voxels on its pillars light it.
//! North of the tunnel a ledge juts out of the west cliff, with an overhang
//! under it, and the dock is a pad of cobblestone on top of it.
//!
//! Everything comes from one function of the voxel position, so the cave,
//! the hall and the ledge are general voxels, not a height field.

use bevy::prelude::*;
use noise::{Fbm, MultiFractal, NoiseFn, Perlin};

// Materials.
pub const GRASS: u8 = 0;
pub const DIRT: u8 = 1;
pub const STONE: u8 = 2;
pub const COBBLESTONE: u8 = 3;
pub const DARK_PLANKS: u8 = 4;
pub const TIMBER: u8 = 5;
pub const ROOF: u8 = 6;
pub const LANTERN: u8 = 7;
pub const FLOOR: u8 = 8;
pub const GRAVEL: u8 = 9;

/// The texture the materials' tiles are in, and how many tiles it has.
pub const TEXTURE: &str = "mountain_house_texture.png";
pub const TILE_COUNT: u32 = 18;

/// The tiles of `TEXTURE` a material shows on its top, sides and bottom.
pub fn texture_indices(material: u8) -> [u32; 3] {
    match material {
        GRASS => [0, 1, 2],
        DIRT => [2; 3],
        COBBLESTONE => [6; 3],
        DARK_PLANKS => [7; 3],
        TIMBER => [9, 8, 9],
        ROOF => [10; 3],
        LANTERN => [13; 3],
        FLOOR => [16; 3],
        GRAVEL => [15; 3],
        _ => [3; 3],
    }
}

/// Whether the airship's lift takes voxels of `material`: the airship is
/// built of these, and the dock and the world around it of none of them.
pub fn is_ship_material(material: u8) -> bool {
    matches!(material, DARK_PLANKS | TIMBER | ROOF | LANTERN | FLOOR)
}

// --- The airship -----------------------------------------------------------

/// The size of the box holding the airship's voxels, from (0, 0, 0). Its
/// bow points along +z.
pub const SHIP_SIZE: IVec3 = IVec3::new(19, 24, 42);

/// The point the autopilot steers, in the airship's own voxels: the middle of
/// its length and width, between the gondola and the envelope.
pub const SHIP_CENTRE: Vec3 = Vec3::new(9.5, 12.0, 21.0);

/// The envelope: a hollow ellipsoid one voxel thick above the gondola.
const ENVELOPE_CENTRE: Vec3 = Vec3::new(9.5, 16.5, 21.0);
const ENVELOPE_RADII: Vec3 = Vec3::new(9.0, 7.0, 20.5);
const ENVELOPE_WALL: f32 = 1.3;

/// The gondola hanging under the envelope, from its lowest corner inclusive
/// to its highest exclusive, with walls, floor and roof one voxel thick.
const GONDOLA_MIN: IVec3 = IVec3::new(5, 0, 4);
const GONDOLA_MAX: IVec3 = IVec3::new(14, 8, 38);

/// The cabin is the gondola's inside. Its only opening is the door through
/// the wall at the lowest x, 3 voxels wide and 4 high, on the floor near the
/// bow. Sky light coming in at the door fades along the cabin's length, so
/// under `SubGridSky::Enclosed` the stern end is dark.
pub const CABIN_MIN: IVec3 = IVec3::new(6, 1, 5);
pub const CABIN_MAX: IVec3 = IVec3::new(13, 7, 37);
pub const DOOR_MIN: IVec3 = IVec3::new(5, 1, 31);
pub const DOOR_MAX: IVec3 = IVec3::new(6, 5, 34);

/// The cabin's lamp, under its roof towards the bow, too far from the stern
/// end for its light to reach there.
pub const CABIN_LAMP: IVec3 = IVec3::new(9, 6, 28);

fn within(position: IVec3, min: IVec3, max: IVec3) -> bool {
    position.cmpge(min).all() && position.cmplt(max).all()
}

/// The material of the airship's voxel at `position` on its own grid, or
/// `None` where it holds air.
pub fn ship_voxel(position: IVec3) -> Option<u8> {
    if !within(position, IVec3::ZERO, SHIP_SIZE) {
        return None;
    }
    if position == CABIN_LAMP {
        return Some(LANTERN);
    }
    if within(position, GONDOLA_MIN, GONDOLA_MAX) {
        if within(position, CABIN_MIN, CABIN_MAX) || within(position, DOOR_MIN, DOOR_MAX)
        {
            return None;
        }
        return Some(if position.y == 0 { FLOOR } else { DARK_PLANKS });
    }
    let IVec3 { x, y, z } = position;
    // Four struts hold the gondola's roof to the envelope.
    let strut = (x == 6 || x == 12) && (z == 9 || z == 33) && (8..12).contains(&y);
    if strut {
        return Some(TIMBER);
    }
    let centre = position.as_vec3() + 0.5 - ENVELOPE_CENTRE;
    let inside = |radii: Vec3| (centre / radii).length_squared() <= 1.0;
    if inside(ENVELOPE_RADII) {
        return (!inside(ENVELOPE_RADII - ENVELOPE_WALL)).then_some(ROOF);
    }
    // Fins at the stern: a level pair, and one above and one below.
    let fin = (1..6).contains(&z) && (y == 16 || x == 9 && (9..24).contains(&y));
    fin.then_some(TIMBER)
}

// --- The dock --------------------------------------------------------------

/// Where the airship's voxel (0, 0, 0) is in the world while it stands on
/// its dock. Its gondola's floor rests on the dock's top at y = 43.
pub const SHIP_AT_DOCK: IVec3 = IVec3::new(63, 44, -42);

/// The airship's centre in the world while it stands on its dock.
pub fn dock_centre() -> Vec3 {
    SHIP_AT_DOCK.as_vec3() + SHIP_CENTRE
}

/// The dock's pad of cobblestone, at y = 43 on top of the ledge.
const DOCK_MIN: IVec3 = IVec3::new(59, 43, -46);
const DOCK_MAX: IVec3 = IVec3::new(87, 44, 4);

/// How far past the airship's box on each side the lift box reaches, so
/// voxels the player adds to the airship while it is docked lift with it.
const LIFT_MARGIN: i32 = 3;

/// The box the dock's lift takes the airship from, from its lowest corner
/// inclusive to its highest exclusive.
pub fn dock_lift_box() -> (IVec3, IVec3) {
    (
        SHIP_AT_DOCK - LIFT_MARGIN,
        SHIP_AT_DOCK + SHIP_SIZE + LIFT_MARGIN,
    )
}

// --- The landscape ---------------------------------------------------------

/// The tunnel's axis runs along x at this height and z, through both
/// cliffs and the hall between them.
const TUNNEL_Y: f32 = 34.0;
const TUNNEL_Z: f32 = 50.0;
const TUNNEL_RADIUS: f32 = 15.0;

/// The hall: a rounded box around this centre, with a flat floor.
const HALL_CENTRE: Vec3 = Vec3::new(182.0, 34.0, 50.0);
const HALL_HALF: Vec3 = Vec3::new(36.0, 22.0, 34.0);
const HALL_FLOOR: i32 = 13;

/// The hall's lamps, each 2 by 2 voxels. Beside the tunnel's line, they
/// stand on pillars in two rows. Over the line they hang from the ceiling on
/// stone columns, and under it they stand on short posts, so the airship
/// passes between lamps above and below.
const LAMP_XS: [i32; 5] = [158, 170, 182, 194, 206];
const PILLAR_ZS: [i32; 2] = [33, 66];
const PILLAR_LAMP_Y: i32 = 33;
const LINE_Z: i32 = 49;
const HANGING_LAMP_Y: i32 = 47;
/// Above every hanging lamp the hall's ceiling is lower than this.
const HALL_CEILING_ABOVE_LAMPS: i32 = 60;
const POST_LAMP_Y: i32 = 19;

/// The mesa's cliffs, before their noise, and its flat top.
const WEST_CLIFF: f32 = 100.0;
const EAST_CLIFF: f32 = 260.0;
const NORTH_CLIFF: f32 = -60.0;
const SOUTH_CLIFF: f32 = 160.0;
const MESA_TOP: f32 = 108.0;

/// The ledge's top is the dock's level, and it reaches west from the cliff
/// to here.
const LEDGE_TOP: i32 = 43;
const LEDGE_WEST: f32 = 50.0;
const LEDGE_Z: (f32, f32) = (-52.0, 10.0);

/// The world's voxels.
pub struct Generator {
    heights: Fbm<Perlin>,
    detail: Perlin,
}

impl Default for Generator {
    fn default() -> Self {
        Self {
            heights: Fbm::<Perlin>::new(7).set_octaves(4),
            detail: Perlin::new(11),
        }
    }
}

impl Generator {
    /// The height of the open ground at column (`x`, `z`), before the 3D
    /// noise that adds overhangs to it. It flattens into the valley along
    /// the tunnel's line, so the airship can fly low to the cave.
    pub fn ground_height(&self, x: i32, z: i32) -> f32 {
        let sample = self.heights.get([x as f64 / 180.0, z as f64 / 180.0]) as f32;
        let hills = 14.0 + 10.0 * sample;
        let valley = smoothstep(12.0, 48.0, (z as f32 - TUNNEL_Z).abs());
        3.0 + valley * (hills - 3.0)
    }

    /// The world's voxel at `position`: the airship on its dock, or the
    /// landscape. `ground_height` is [`Generator::ground_height`] at its
    /// column, which a caller can cache.
    pub fn voxel(&self, position: IVec3, ground_height: f32) -> Option<u8> {
        ship_voxel(position - SHIP_AT_DOCK)
            .or_else(|| self.landscape(position, ground_height))
    }

    /// The landscape's voxel at `position`, the world without the airship.
    pub fn landscape(&self, position: IVec3, ground_height: f32) -> Option<u8> {
        if position.y < 0 {
            return Some(STONE);
        }
        if let Some(material) = hall_lamp(position) {
            return Some(material);
        }
        let p = position.as_vec3() + 0.5;
        if in_hall(p) || in_tunnel(p) {
            return None;
        }
        // The hall's floor is gravel.
        if in_hall(p + Vec3::Y) {
            return Some(GRAVEL);
        }
        if let Some(material) = self.mesa(position, p) {
            return Some(material);
        }
        if let Some(material) = self.ledge(position, p) {
            return Some(material);
        }
        self.ground(p, ground_height)
    }

    /// Noise from about -1 to 1 that varies over `scale` voxels.
    fn noise(&self, p: Vec3, scale: f32) -> f32 {
        let p = p / scale;
        self.detail.get([p.x as f64, p.y as f64, p.z as f64]) as f32
    }

    /// The open ground: grass over dirt over stone, under a surface that the
    /// 3D noise lifts and dips, so it overhangs in places.
    fn ground(&self, p: Vec3, height: f32) -> Option<u8> {
        if p.y > height + 5.0 {
            return None;
        }
        let surface = height + 4.0 * self.noise(p, 20.0);
        if p.y >= surface {
            None
        } else if p.y >= surface - 1.0 {
            Some(GRASS)
        } else if p.y >= surface - 4.0 {
            Some(DIRT)
        } else {
            Some(STONE)
        }
    }

    /// The mesa: stone between cliffs that the noise roughens, grass on top.
    fn mesa(&self, position: IVec3, p: Vec3) -> Option<u8> {
        let rough = |cliff: f32, along: f32, offset: f32| {
            let near = (along - cliff).abs() < 4.0;
            if near {
                cliff + 3.0 * self.noise(p + offset, 14.0)
            } else {
                cliff
            }
        };
        let inside = p.x >= rough(WEST_CLIFF, p.x, 0.0)
            && p.x < rough(EAST_CLIFF, p.x, 100.0)
            && p.z >= rough(NORTH_CLIFF, p.z, 200.0)
            && p.z < rough(SOUTH_CLIFF, p.z, 300.0);
        if !inside {
            return None;
        }
        let top = MESA_TOP + 5.0 * self.noise(Vec3::new(p.x, 0.0, p.z), 50.0);
        if p.y >= top {
            None
        } else if position.y as f32 >= top - 1.0 {
            Some(GRASS)
        } else {
            Some(STONE)
        }
    }

    /// The ledge: a shelf out of the west cliff with a flat top at the dock's
    /// level, thinning towards its tip, so its underside overhangs the
    /// valley side.
    fn ledge(&self, position: IVec3, p: Vec3) -> Option<u8> {
        let across = p.z >= LEDGE_Z.0 && p.z < LEDGE_Z.1;
        if !across
            || p.x < LEDGE_WEST
            || p.x >= WEST_CLIFF + 4.0
            || position.y > LEDGE_TOP
        {
            return None;
        }
        let underside = 24.0 + (WEST_CLIFF - p.x) * 0.32 + 1.5 * self.noise(p, 8.0);
        if p.y < underside {
            return None;
        }
        if within(position, DOCK_MIN, DOCK_MAX) {
            Some(COBBLESTONE)
        } else if position.y == LEDGE_TOP {
            Some(GRASS)
        } else {
            Some(STONE)
        }
    }
}

fn in_tunnel(p: Vec3) -> bool {
    let off_axis = Vec2::new(p.y - TUNNEL_Y, p.z - TUNNEL_Z) / TUNNEL_RADIUS;
    (WEST_CLIFF - 12.0..EAST_CLIFF + 12.0).contains(&p.x)
        && off_axis.length_squared() < 1.0
}

fn in_hall(p: Vec3) -> bool {
    let d = ((p - HALL_CENTRE) / HALL_HALF).abs();
    p.y >= HALL_FLOOR as f32 && d.x.powi(4) + d.y.powi(4) + d.z.powi(4) < 1.0
}

/// The hall's lamps, with the stone they stand on or hang from.
fn hall_lamp(position: IVec3) -> Option<u8> {
    let IVec3 { x, y, z } = position;
    if !LAMP_XS.iter().any(|lx| (*lx..lx + 2).contains(&x)) {
        return None;
    }
    let lamp_or_stone = |lamp_y: i32| Some(if y == lamp_y { LANTERN } else { STONE });
    if PILLAR_ZS.iter().any(|pz| (*pz..pz + 2).contains(&z)) {
        return if (HALL_FLOOR..=PILLAR_LAMP_Y).contains(&y) {
            lamp_or_stone(PILLAR_LAMP_Y)
        } else {
            None
        };
    }
    if !(LINE_Z..LINE_Z + 2).contains(&z) {
        return None;
    }
    if (HALL_FLOOR..=POST_LAMP_Y).contains(&y) {
        lamp_or_stone(POST_LAMP_Y)
    } else if (HANGING_LAMP_Y..HALL_CEILING_ABOVE_LAMPS).contains(&y) {
        lamp_or_stone(HANGING_LAMP_Y)
    } else {
        None
    }
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Every position in the box from `min` inclusive to `max` exclusive.
#[cfg(test)]
pub fn positions_in(min: IVec3, max: IVec3) -> impl Iterator<Item = IVec3> {
    (min.x..max.x).flat_map(move |x| {
        (min.y..max.y).flat_map(move |y| (min.z..max.z).map(move |z| IVec3::new(x, y, z)))
    })
}

/// The box around the hall's air, from its lowest corner to its highest.
#[cfg(test)]
pub fn hall_box() -> (Vec3, Vec3) {
    (HALL_CENTRE - HALL_HALF, HALL_CENTRE + HALL_HALF)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn voxel(generator: &Generator, position: IVec3) -> Option<u8> {
        generator.voxel(position, generator.ground_height(position.x, position.z))
    }

    #[test]
    fn the_dock_lift_picks_the_whole_airship_and_nothing_else() {
        let generator = Generator::default();
        let (min, max) = dock_lift_box();
        let mut picked = 0;
        for position in positions_in(min, max) {
            let ship = ship_voxel(position - SHIP_AT_DOCK);
            let world = voxel(&generator, position);
            let pick = world.is_some_and(is_ship_material);
            assert_eq!(pick, ship.is_some(), "at {position}");
            if pick {
                assert_eq!(world, ship, "at {position}");
                picked += 1;
            }
        }
        let ship_voxels = positions_in(IVec3::ZERO, SHIP_SIZE)
            .filter(|position| ship_voxel(*position).is_some())
            .count();
        assert_eq!(picked, ship_voxels);
    }

    #[test]
    fn the_dock_under_the_airship_is_whole_stone_the_lift_leaves() {
        let generator = Generator::default();
        let footprint = positions_in(IVec3::ZERO, SHIP_SIZE.with_y(1))
            .filter(|position| ship_voxel(*position).is_some());
        let mut count = 0;
        for position in footprint {
            let below = SHIP_AT_DOCK + position - IVec3::Y;
            assert_eq!(
                voxel(&generator, below),
                Some(COBBLESTONE),
                "under {position}"
            );
            count += 1;
        }
        // The gondola's floor, 9 by 34 voxels.
        assert_eq!(count, 9 * 34);
    }

    /// The air reachable from the cabin's middle through air, within the
    /// airship's box and one voxel around it, with `blocked` counted solid.
    fn reachable_from_cabin(blocked: &dyn Fn(IVec3) -> bool) -> HashSet<IVec3> {
        let (min, max) = (IVec3::NEG_ONE, SHIP_SIZE + 1);
        let start = (CABIN_MIN + CABIN_MAX) / 2;
        let mut seen = HashSet::from([start]);
        let mut queue = vec![start];
        while let Some(position) = queue.pop() {
            for step in [
                IVec3::X,
                IVec3::NEG_X,
                IVec3::Y,
                IVec3::NEG_Y,
                IVec3::Z,
                IVec3::NEG_Z,
            ] {
                let next = position + step;
                let open = within(next, min, max)
                    && ship_voxel(next).is_none()
                    && !blocked(next);
                if open && seen.insert(next) {
                    queue.push(next);
                }
            }
        }
        seen
    }

    #[test]
    fn the_cabin_opens_to_the_sky_only_through_its_door() {
        let outside = IVec3::NEG_ONE;
        let through_door = reachable_from_cabin(&|_| false);
        assert!(through_door.contains(&outside));
        let door_shut =
            reachable_from_cabin(&|position| within(position, DOOR_MIN, DOOR_MAX));
        assert!(!door_shut.contains(&outside));
        assert!(door_shut
            .iter()
            .all(|position| within(*position, CABIN_MIN, CABIN_MAX)));
    }

    #[test]
    fn the_cabin_holds_the_airships_only_lamp() {
        let lamps: Vec<IVec3> = positions_in(IVec3::ZERO, SHIP_SIZE)
            .filter(|position| ship_voxel(*position) == Some(LANTERN))
            .collect();
        assert_eq!(lamps, vec![CABIN_LAMP]);
        assert!(within(CABIN_LAMP, CABIN_MIN, CABIN_MAX));
    }

    #[test]
    fn the_hall_has_rock_above_it_and_lamps_inside() {
        let generator = Generator::default();
        let (min, max) = hall_box();
        let mut lamps = 0;
        for x in (min.x as i32..max.x as i32).step_by(4) {
            for z in (min.z as i32..max.z as i32).step_by(4) {
                let roofed = (max.y as i32..MESA_TOP as i32 - 10)
                    .any(|y| voxel(&generator, IVec3::new(x, y, z)).is_some());
                assert!(roofed, "column {x}, {z}");
            }
        }
        for position in positions_in(min.as_ivec3(), max.as_ivec3()) {
            if voxel(&generator, position) == Some(LANTERN) {
                lamps += 1;
            }
        }
        assert_eq!(lamps, LAMP_XS.len() * (PILLAR_ZS.len() + 2) * 4);
    }
}
