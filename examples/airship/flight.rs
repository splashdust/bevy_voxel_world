//! The airship's flight: the autopilot's loop, and the trips off and back
//! onto the dock.
//!
//! The airship starts as world voxels on its dock. Once light has settled,
//! it is lifted out of the world into a sub-grid with the dock's box and a
//! pick that takes only the airship's materials, so the dock's stone stays.
//! When `Lifted` arrives, the lift has landed and the autopilot takes over:
//! it rises off the dock and flies the loop, over the hills, through the
//! cave and the lamp-lit hall, and back. Each frame it drives the sub-grid's
//! `Transform`, which is against the world's root.
//!
//! F sends the airship along the loop to its start, backing up when that is
//! the shorter way, and down to hover over its dock. While it hovers, the gizmos draw `stamp_preview`'s
//! ghost: the snapped bounds outlined, with a box at each voxel where a solid
//! voxel would land on a solid one. Enter eases the airship to the preview's
//! pose, then stamps it into the world in the same frame, so no snap shows.
//! Once `Stamped` arrives the airship is world voxels again, and F lifts it
//! off the dock as at the start.

use std::f32::consts::PI;

use bevy::prelude::*;
use bevy_voxel_world::prelude::*;

use crate::{
    pieces::Piece,
    world::{dock_centre, dock_lift_box, is_ship_material, SHIP_CENTRE},
    AirshipWorld, Status,
};

/// The points the loop passes through, as the airship's centre, in order.
/// The first is just above the dock, and the loop closes back on it.
const WAYPOINTS: [Vec3; 20] = [
    Vec3::new(72.5, 60.0, -21.0),
    Vec3::new(72.0, 60.0, 30.0),
    Vec3::new(64.0, 56.0, 95.0),
    Vec3::new(30.0, 53.0, 135.0),
    Vec3::new(-40.0, 51.0, 140.0),
    Vec3::new(-95.0, 49.0, 105.0),
    Vec3::new(-110.0, 46.0, 70.0),
    Vec3::new(-70.0, 38.0, 50.0),
    // Along the valley, through the west cliff's mouth, the hall and out
    // through the east cliff.
    Vec3::new(0.0, 35.0, 50.0),
    Vec3::new(60.0, 34.0, 50.0),
    Vec3::new(120.0, 34.0, 50.0),
    Vec3::new(182.0, 33.0, 50.0),
    Vec3::new(244.0, 34.0, 50.0),
    Vec3::new(300.0, 36.0, 50.0),
    Vec3::new(335.0, 46.0, 5.0),
    Vec3::new(315.0, 56.0, -85.0),
    Vec3::new(240.0, 64.0, -112.0),
    Vec3::new(140.0, 66.0, -112.0),
    Vec3::new(62.0, 64.0, -95.0),
    Vec3::new(48.0, 62.0, -55.0),
];

/// How many points the loop is sampled at between two waypoints.
const SAMPLES_PER_WAYPOINT: usize = 64;

/// The autopilot's speed along the loop, and its speed back to the dock, in
/// voxels per second. Streaming and light follow the camera, which follows
/// the airship, so these stay low enough for both to keep up.
const CRUISE_SPEED: f32 = 12.0;
const RETURN_SPEED: f32 = 18.0;
/// How fast the speed changes, in voxels per second per second.
const ACCELERATION: f32 = 4.0;

/// How the airship leans and tips with the loop: its roll per radian of
/// turn per voxel, and the most it rolls and pitches.
const BANK_PER_TURN: f32 = 10.0;
const MAX_BANK: f32 = 0.25;
const MAX_PITCH: f32 = 0.15;

/// How long the trips between the dock and the loop take, and the ease onto
/// the stamp's pose, in seconds.
const DEPART_SECS: f32 = 4.0;
const SETTLE_SECS: f32 = 4.0;
const STAMP_EASE_SECS: f32 = 1.0;

/// How the airship hovers over its dock: how high above it, how far it bobs,
/// and how far it sways, in radians. The pose stays within half a voxel and
/// a few degrees of the dock, so the stamp snaps back onto the dock.
const HOVER_HEIGHT: f32 = 0.35;
const HOVER_BOB: f32 = 0.1;
const HOVER_YAW: f32 = 0.05;
const HOVER_ROLL: f32 = 0.03;

/// How many frames light must have been settled before the first lift, so
/// the airship is seen standing on its dock first.
const SETTLED_FRAMES_BEFORE_FIRST_LIFT: u32 = 90;

/// Where the airship's centre is, and which way it faces.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pose {
    pub position: Vec3,
    pub rotation: Quat,
}

impl Pose {
    /// The airship's pose standing on its dock.
    pub fn docked() -> Self {
        Self {
            position: dock_centre(),
            rotation: Quat::IDENTITY,
        }
    }

    /// The pose of a sub-grid with `transform`, whose grid has the
    /// airship's centre at `centre`.
    fn of(transform: &Transform, centre: Vec3) -> Self {
        Self {
            position: transform.transform_point(centre),
            rotation: transform.rotation,
        }
    }

    /// The `Transform` that puts the point `centre` of a sub-grid's grid at
    /// this pose.
    pub fn transform(&self, centre: Vec3) -> Transform {
        Transform {
            translation: self.position - self.rotation * centre,
            rotation: self.rotation,
            scale: Vec3::ONE,
        }
    }

    /// The pose `t` of the way from this one to `to`, easing in and out.
    fn ease(self, to: Self, t: f32) -> Self {
        let t = smoothstep(t.clamp(0.0, 1.0));
        Self {
            position: self.position.lerp(to.position, t),
            rotation: self.rotation.slerp(to.rotation, t),
        }
    }
}

fn smoothstep(t: f32) -> f32 {
    t * t * (3.0 - 2.0 * t)
}

/// The pose of the airship hovering over its dock, `t` seconds into its
/// hover.
fn hover_pose(t: f32) -> Pose {
    let docked = Pose::docked();
    Pose {
        position: docked.position
            + Vec3::Y * (HOVER_HEIGHT + HOVER_BOB * (t * 1.3).sin()),
        rotation: Quat::from_rotation_y(HOVER_YAW * (t * 0.7).sin())
            * Quat::from_rotation_z(HOVER_ROLL * (t * 0.9).sin()),
    }
}

/// The autopilot's loop: a closed curve through [`WAYPOINTS`], measured
/// along its length.
pub struct Loop {
    points: Vec<Vec3>,
    /// The distance along the loop of each of `points`, from the first.
    distances: Vec<f32>,
}

impl Default for Loop {
    fn default() -> Self {
        let count = WAYPOINTS.len();
        let waypoint = |i: usize| WAYPOINTS[i % count];
        let mut points = Vec::with_capacity(count * SAMPLES_PER_WAYPOINT + 1);
        for i in 0..count {
            let [p0, p1, p2, p3] =
                [count - 1, 0, 1, 2].map(|offset| waypoint(i + offset));
            for sample in 0..SAMPLES_PER_WAYPOINT {
                let t = sample as f32 / SAMPLES_PER_WAYPOINT as f32;
                points.push(catmull_rom(p0, p1, p2, p3, t));
            }
        }
        points.push(WAYPOINTS[0]);
        let mut distances = vec![0.0];
        for pair in points.windows(2) {
            distances.push(distances.last().unwrap() + pair[0].distance(pair[1]));
        }
        Self { points, distances }
    }
}

/// The point `t` of the way from `p1` to `p2` on the uniform Catmull-Rom
/// curve through `p0` to `p3`.
fn catmull_rom(p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3, t: f32) -> Vec3 {
    let (t2, t3) = (t * t, t * t * t);
    0.5 * (2.0 * p1
        + (p2 - p0) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
        + (3.0 * p1 - p0 - 3.0 * p2 + p3) * t3)
}

impl Loop {
    pub fn length(&self) -> f32 {
        *self.distances.last().unwrap()
    }

    /// The point `distance` along the loop, from its start above the dock.
    pub fn position(&self, distance: f32) -> Vec3 {
        let distance = distance.rem_euclid(self.length());
        let after = self.distances.partition_point(|d| *d <= distance);
        let after = after.clamp(1, self.points.len() - 1);
        let (d0, d1) = (self.distances[after - 1], self.distances[after]);
        let t = (distance - d0) / (d1 - d0).max(f32::EPSILON);
        self.points[after - 1].lerp(self.points[after], t)
    }

    /// Which way the airship faces at `distance` along the loop, about the
    /// vertical axis, with 0 facing +z.
    fn heading(&self, distance: f32) -> f32 {
        let tangent = self.position(distance + 3.0) - self.position(distance - 3.0);
        tangent.x.atan2(tangent.z)
    }

    /// The airship's pose `distance` along the loop: facing along it,
    /// tipped by its climb and rolled into its turns.
    pub fn pose(&self, distance: f32) -> Pose {
        let tangent =
            (self.position(distance + 3.0) - self.position(distance - 3.0)).normalize();
        let heading = self.heading(distance);
        let pitch = tangent.y.asin().clamp(-MAX_PITCH, MAX_PITCH);
        let turn =
            wrap_angle(self.heading(distance + 6.0) - self.heading(distance - 6.0))
                / 12.0;
        let bank = (-BANK_PER_TURN * turn).clamp(-MAX_BANK, MAX_BANK);
        Pose {
            position: self.position(distance),
            rotation: Quat::from_rotation_y(heading)
                * Quat::from_rotation_x(-pitch)
                * Quat::from_rotation_z(bank),
        }
    }
}

fn wrap_angle(angle: f32) -> f32 {
    (angle + PI).rem_euclid(2.0 * PI) - PI
}

/// The airship's sub-grid. The world's voxels hold the airship instead while
/// it is docked, and none of its pieces carry this.
#[derive(Component)]
pub struct Ship;

/// What the airship is doing.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Phase {
    /// It is world voxels on the dock.
    Docked,
    /// It has been lifted, and waits for `Lifted`.
    Lifting,
    /// It rises from `from` to the loop's start.
    Departing {
        from: Pose,
        elapsed: f32,
    },
    /// The autopilot flies it along the loop.
    Flying,
    /// It flies along the loop to the start, the way `direction` gives.
    Returning {
        direction: f32,
    },
    /// It eases down from the loop's start to hover over the dock.
    Settling {
        elapsed: f32,
    },
    Hovering {
        elapsed: f32,
    },
    /// It eases from `from` to the stamp's pose `to`, then is stamped.
    Easing {
        from: Pose,
        to: Transform,
        elapsed: f32,
    },
    /// It has been stamped, and waits for `Stamped`.
    Stamping,
}

impl Phase {
    /// A few words on what the airship is doing, for the panel.
    pub fn describe(&self) -> &'static str {
        match self {
            Phase::Docked => "docked, as world voxels",
            Phase::Lifting => "lifting off the dock",
            Phase::Departing { .. } => "leaving the dock",
            Phase::Flying => "flying the loop",
            Phase::Returning { .. } => "flying back to the dock",
            Phase::Settling { .. } => "settling over the dock",
            Phase::Hovering { .. } => "hovering over the dock",
            Phase::Easing { .. } | Phase::Stamping => "stamping onto the dock",
        }
    }
}

/// The airship's flight.
#[derive(Resource)]
pub struct Flight {
    pub route: Loop,
    pub phase: Phase,
    /// The airship's sub-grid, while it is one.
    pub ship: Option<Entity>,
    /// Where the airship's [`SHIP_CENTRE`] is on its sub-grid's grid.
    pub centre: Vec3,
    /// How far along the loop it is, and how fast it moves along it.
    pub distance: f32,
    speed: f32,
    /// Whether the autopilot holds the airship still on the loop.
    pub paused: bool,
    /// Where the airship is this frame.
    pub pose: Pose,
    /// Whether the airship has been lifted yet, and for how many frames light
    /// has been settled before that.
    lifted_once: bool,
    settled_frames: u32,
}

impl Default for Flight {
    fn default() -> Self {
        Self {
            route: Loop::default(),
            phase: Phase::Docked,
            ship: None,
            centre: SHIP_CENTRE,
            distance: 0.0,
            speed: 0.0,
            paused: false,
            pose: Pose::docked(),
            lifted_once: false,
            settled_frames: 0,
        }
    }
}

impl Flight {
    /// Lifts the airship off its dock into a new sub-grid, with every voxel
    /// of the airship's materials in the dock's box.
    ///
    /// As with `pieces::lift_piece`, a system calling this declares
    /// `voxel_world` before `commands`.
    fn lift(
        &mut self,
        voxel_world: &mut VoxelWorld<AirshipWorld>,
        commands: &mut Commands,
        status: &mut Status,
    ) {
        let (min, max) = dock_lift_box();
        let picked = |_, voxel| matches!(voxel, WorldVoxel::Solid(material) if is_ship_material(material));
        match voxel_world.lift(Grid::World, min, max, picked) {
            Ok(ship) => {
                commands.entity(ship).insert(Ship);
                self.ship = Some(ship);
                self.phase = Phase::Lifting;
                self.lifted_once = true;
            }
            Err(error) => status.set(format!("Nothing to lift: {error}")),
        }
    }
}

/// Lifts the airship off its dock once light has settled around it, the
/// first time only.
pub fn first_lift(
    mut flight: ResMut<Flight>,
    mut voxel_world: VoxelWorld<AirshipWorld>,
    mut commands: Commands,
    mut status: ResMut<Status>,
) {
    if flight.lifted_once || flight.phase != Phase::Docked {
        return;
    }
    flight.settled_frames = if voxel_world.light_is_settled() {
        flight.settled_frames + 1
    } else {
        0
    };
    if flight.settled_frames >= SETTLED_FRAMES_BEFORE_FIRST_LIFT {
        flight.lift(&mut voxel_world, &mut commands, &mut status);
    }
}

/// Starts the flight when the airship's lift lands, and docks it when its
/// stamp lands.
pub fn follow_lift_and_stamp(
    mut lifted: MessageReader<Lifted<AirshipWorld>>,
    mut stamped: MessageReader<Stamped<AirshipWorld>>,
    mut flight: ResMut<Flight>,
    transforms: Query<&Transform>,
    mut status: ResMut<Status>,
) {
    for lift in lifted.read() {
        if Some(lift.sub_grid) != flight.ship {
            continue;
        }
        // The lift spawned the airship at an offset with no rotation, so its
        // centre is where it was on the dock.
        if let Ok(transform) = transforms.get(lift.sub_grid) {
            flight.centre = dock_centre() - transform.translation;
        }
        flight.phase = Phase::Departing {
            from: Pose::docked(),
            elapsed: 0.0,
        };
        status.set("Lifted the airship off its dock".into());
    }
    for stamp in stamped.read() {
        if Some(stamp.sub_grid) == flight.ship {
            flight.ship = None;
            flight.phase = Phase::Docked;
            flight.pose = Pose::docked();
            status.set("Stamped the airship onto its dock".into());
        }
    }
}

/// F sends the airship to its dock, back onto the loop, or off the dock;
/// Enter stamps it while it hovers; Space pauses the autopilot.
pub fn keys(
    keys: Res<ButtonInput<KeyCode>>,
    mut flight: ResMut<Flight>,
    mut voxel_world: VoxelWorld<AirshipWorld>,
    mut commands: Commands,
    pieces: Query<(), With<Piece>>,
    mut status: ResMut<Status>,
) {
    if keys.just_pressed(KeyCode::Space) {
        flight.paused = !flight.paused;
    }
    if keys.just_pressed(KeyCode::KeyF) {
        let pose = flight.pose;
        match flight.phase {
            Phase::Docked => flight.lift(&mut voxel_world, &mut commands, &mut status),
            Phase::Flying => {
                // Back along the loop to its start, whichever way is shorter.
                let forward = flight.route.length() - flight.distance;
                let direction = if forward < flight.distance { 1.0 } else { -1.0 };
                flight.phase = Phase::Returning { direction };
            }
            Phase::Returning { .. } => flight.phase = Phase::Flying,
            Phase::Settling { .. } | Phase::Hovering { .. } => {
                flight.phase = Phase::Departing {
                    from: pose,
                    elapsed: 0.0,
                };
            }
            _ => {}
        }
    }
    if keys.just_pressed(KeyCode::Enter) {
        let (Phase::Hovering { .. }, Some(ship)) = (flight.phase, flight.ship) else {
            status
                .set("The airship stamps only while it hovers over its dock (F)".into());
            return;
        };
        if !pieces.is_empty() {
            status.set("Stamp the pieces back onto the airship first (V)".into());
            return;
        }
        if let Some(preview) = voxel_world.stamp_preview(ship, Grid::World) {
            flight.phase = Phase::Easing {
                from: flight.pose,
                to: preview.pose,
                elapsed: 0.0,
            };
        }
    }
}

/// Moves the airship for this frame, and stamps it when it reaches the
/// stamp's pose.
pub fn fly(
    time: Res<Time>,
    mut flight: ResMut<Flight>,
    mut voxel_world: VoxelWorld<AirshipWorld>,
    mut ships: Query<&mut Transform, With<Ship>>,
    mut status: ResMut<Status>,
) {
    let dt = time.delta_secs();
    let flight = &mut *flight;
    let Some(ship) = flight.ship else {
        return;
    };
    let Ok(mut transform) = ships.get_mut(ship) else {
        return;
    };
    let length = flight.route.length();
    match &mut flight.phase {
        Phase::Docked | Phase::Lifting | Phase::Stamping => return,
        Phase::Departing { from, elapsed } => {
            *elapsed += dt;
            flight.pose = from.ease(flight.route.pose(0.0), *elapsed / DEPART_SECS);
            if *elapsed >= DEPART_SECS {
                flight.distance = 0.0;
                flight.speed = 0.0;
                flight.phase = Phase::Flying;
            }
        }
        Phase::Flying => {
            let target = if flight.paused { 0.0 } else { CRUISE_SPEED };
            flight.speed = approach(flight.speed, target, dt);
            flight.distance = (flight.distance + flight.speed * dt).rem_euclid(length);
            flight.pose = flight.route.pose(flight.distance);
        }
        Phase::Returning { direction } => {
            let direction = *direction;
            let left = if direction > 0.0 {
                length - flight.distance
            } else {
                flight.distance
            };
            // Slows down so as to stop at the start.
            let target = if flight.paused {
                0.0
            } else {
                direction * RETURN_SPEED.min((2.0 * ACCELERATION * left).sqrt())
            };
            flight.speed = approach(flight.speed, target, dt);
            let step = flight.speed * dt;
            if step * direction >= left || left < 0.05 {
                flight.distance = 0.0;
                flight.speed = 0.0;
                flight.phase = Phase::Settling { elapsed: 0.0 };
            } else {
                flight.distance = (flight.distance + step).clamp(0.0, length);
            }
            flight.pose = flight.route.pose(flight.distance);
        }
        Phase::Settling { elapsed } => {
            *elapsed += dt;
            flight.pose = flight
                .route
                .pose(0.0)
                .ease(hover_pose(0.0), *elapsed / SETTLE_SECS);
            if *elapsed >= SETTLE_SECS {
                flight.phase = Phase::Hovering { elapsed: 0.0 };
            }
        }
        Phase::Hovering { elapsed } => {
            *elapsed += dt;
            flight.pose = hover_pose(*elapsed);
        }
        Phase::Easing { from, to, elapsed } => {
            *elapsed += dt;
            if *elapsed < STAMP_EASE_SECS {
                flight.pose =
                    from.ease(Pose::of(to, flight.centre), *elapsed / STAMP_EASE_SECS);
            } else {
                // The pose and the stamp in the same frame: the stamp snaps
                // from last frame's pose, which is next to this one, to the
                // same result.
                *transform = *to;
                flight.pose = Pose::of(to, flight.centre);
                match voxel_world.stamp(ship, Grid::World, incoming_replaces) {
                    Ok(()) => flight.phase = Phase::Stamping,
                    Err(error) => {
                        status.set(format!("Stamped nothing: {error}"));
                        flight.phase = Phase::Hovering { elapsed: 0.0 };
                    }
                }
                return;
            }
        }
    }
    *transform = flight.pose.transform(flight.centre);
}

/// `speed` moved towards `target` by at most this frame's acceleration.
fn approach(speed: f32, target: f32, dt: f32) -> f32 {
    let step = ACCELERATION * dt;
    speed + (target - speed).clamp(-step, step)
}

/// Draws `stamp_preview`'s ghost while the airship hovers over its dock: the
/// snapped bounds outlined, and a box at each overlap.
pub fn draw_stamp_preview(
    flight: Res<Flight>,
    voxel_world: VoxelWorld<AirshipWorld>,
    mut gizmos: Gizmos,
) {
    if !matches!(flight.phase, Phase::Hovering { .. } | Phase::Easing { .. }) {
        return;
    }
    let Some(ship) = flight.ship else {
        return;
    };
    let Some(preview) = voxel_world.stamp_preview(ship, Grid::World) else {
        return;
    };
    let Some((min, max)) = voxel_world
        .grid(Grid::SubGrid(ship))
        .and_then(|grid| grid.bounds())
    else {
        return;
    };
    let bounds = Transform::from_translation((min + max).as_vec3() / 2.0)
        .with_scale((max - min).as_vec3());
    gizmos.cube(
        preview.pose.mul_transform(bounds),
        Color::srgb(0.3, 0.9, 1.0),
    );
    for overlap in &preview.overlaps {
        let voxel = Transform::from_translation(overlap.as_vec3() + 0.5)
            .with_scale(Vec3::splat(1.05));
        gizmos.cube(voxel, Color::srgb(1.0, 0.2, 0.2));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::{hall_box, positions_in, ship_voxel, Generator, SHIP_SIZE};

    /// The world positions the airship's voxels' centres fall in at `pose`.
    fn ship_voxels_at(pose: Pose) -> impl Iterator<Item = IVec3> {
        let transform = pose.transform(SHIP_CENTRE);
        positions_in(IVec3::ZERO, SHIP_SIZE)
            .filter(|position| ship_voxel(*position).is_some())
            .map(move |position| {
                transform
                    .transform_point(position.as_vec3() + 0.5)
                    .floor()
                    .as_ivec3()
            })
    }

    fn clear_of_the_landscape(generator: &Generator, pose: Pose) -> Result<(), IVec3> {
        for position in ship_voxels_at(pose) {
            let height = generator.ground_height(position.x, position.z);
            if generator.landscape(position, height).is_some() {
                return Err(position);
            }
        }
        Ok(())
    }

    #[test]
    fn the_loop_never_touches_the_landscape() {
        let generator = Generator::default();
        let route = Loop::default();
        let mut touching = Vec::new();
        let mut distance = 0.0;
        while distance < route.length() {
            let pose = route.pose(distance);
            if let Err(position) = clear_of_the_landscape(&generator, pose) {
                touching.push(format!(
                    "{distance} at {:?} touches {position}",
                    pose.position
                ));
            }
            distance += 2.0;
        }
        assert!(
            touching.is_empty(),
            "of {}:\n{}",
            route.length(),
            touching.join("\n")
        );
    }

    #[test]
    fn the_trips_between_the_dock_and_the_loop_never_touch_the_landscape() {
        let generator = Generator::default();
        let start = Loop::default().pose(0.0);
        for step in 0..=20 {
            let t = step as f32 / 20.0;
            for pose in [
                Pose::docked().ease(start, t),
                start.ease(hover_pose(0.0), t),
            ] {
                assert_eq!(clear_of_the_landscape(&generator, pose), Ok(()), "{t}");
            }
        }
        for step in 0..100 {
            assert_eq!(
                clear_of_the_landscape(&generator, hover_pose(step as f32 * 0.1)),
                Ok(())
            );
        }
    }

    #[test]
    fn the_loop_flies_through_the_hall() {
        let route = Loop::default();
        let (min, max) = hall_box();
        let inside = (0..route.length() as i32)
            .map(|distance| route.position(distance as f32))
            .filter(|position| position.cmpge(min).all() && position.cmple(max).all())
            .count();
        assert!(inside as f32 >= 0.8 * (max.x - min.x), "{inside}");
    }

    #[test]
    fn the_hover_stays_within_a_snap_of_the_dock() {
        for step in 0..200 {
            let pose = hover_pose(step as f32 * 0.05);
            let offset = pose.position - Pose::docked().position;
            assert!(offset.abs().max_element() < 0.5, "{offset}");
            assert!(pose.rotation.angle_between(Quat::IDENTITY) < PI / 8.0);
        }
    }
}
