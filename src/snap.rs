//! The 24 rotations of a grid, and the snap that lines a stamped sub-grid up
//! with the grid it's stamped into.
//!
//! A sub-grid moves freely, so its voxels rarely sit on the target grid's
//! voxels. The snap finds the nearest pose where they do: the nearest of the 24
//! [`GridRotation`]s, the nearest voxel and scale 1. It turns the sub-grid
//! about the centre of its solids' bounds rather than its grid's origin, so a
//! long body's far end doesn't swing when the turn is taken out.
//!
//! The snap reads only the pose and the solids' bounds, never the voxels, so
//! its cost doesn't grow with the sub-grid's size.

use std::ops::Mul;

use bevy::prelude::*;

/// One of the 24 rotations that turn a grid's axes onto its axes, so a grid
/// position always lands on a grid position.
///
/// [`WorldVoxel`](crate::prelude::WorldVoxel) has no orientation. A game that
/// gives voxels a facing, such as stairs or logs, keeps it in the material
/// index, and a stamp rule turns that facing with the stamp's `GridRotation`.
///
/// `a * b` turns by `b`, then by `a`, as `Quat` does.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GridRotation {
    /// Where the rotation turns the x, y and z axes, each a unit vector.
    columns: [[i8; 3]; 3],
}

impl Default for GridRotation {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl GridRotation {
    /// The rotation that leaves every position where it is.
    pub const IDENTITY: Self = Self {
        columns: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    };

    /// All 24 rotations, the identity first.
    ///
    /// The order never changes between releases, so an index into it can be
    /// stored, for example as a facing in a material index.
    pub const ALL: [Self; 24] = Self::all();

    /// Lists each choice of where the x and y axes go, with the z axis
    /// following from them.
    const fn all() -> [Self; 24] {
        let mut all = [Self::IDENTITY; 24];
        let mut n = 0;
        let mut x_axis = 0;
        while x_axis < 3 {
            let mut y_axis = 0;
            while y_axis < 3 {
                if y_axis != x_axis {
                    let mut signs = 0;
                    while signs < 4 {
                        let x = unit(x_axis, if signs & 1 == 0 { 1 } else { -1 });
                        let y = unit(y_axis, if signs & 2 == 0 { 1 } else { -1 });
                        all[n] = Self {
                            columns: [x, y, cross(x, y)],
                        };
                        n += 1;
                        signs += 1;
                    }
                }
                y_axis += 1;
            }
            x_axis += 1;
        }
        all
    }

    /// Turns a grid position about the grid's origin.
    ///
    /// It turns the voxel's corner at `position`, not the voxel: voxel `v`
    /// fills `[v, v + 1)`, so a turned voxel's minimum corner is generally not
    /// `apply(v)`.
    pub fn apply(self, position: IVec3) -> IVec3 {
        let [x, y, z] = self
            .columns
            .map(|c| IVec3::new(c[0].into(), c[1].into(), c[2].into()));
        x * position.x + y * position.y + z * position.z
    }

    /// The same rotation as a `Quat`.
    pub fn to_quat(self) -> Quat {
        let [x, y, z] = self
            .columns
            .map(|c| Vec3::new(c[0].into(), c[1].into(), c[2].into()));
        Quat::from_mat3(&Mat3::from_cols(x, y, z))
    }
}

/// The unit vector along `axis`, pointing the way `sign` says.
const fn unit(axis: usize, sign: i8) -> [i8; 3] {
    let mut v = [0; 3];
    v[axis] = sign;
    v
}

const fn cross(a: [i8; 3], b: [i8; 3]) -> [i8; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

impl Mul for GridRotation {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        let column = |c: [i8; 3]| {
            let v = self.apply(IVec3::new(c[0].into(), c[1].into(), c[2].into()));
            [v.x as i8, v.y as i8, v.z as i8]
        };
        Self {
            columns: rhs.columns.map(column),
        }
    }
}

/// The nearest pose at which a sub-grid's voxels line up with a target grid's,
/// and what [`snap`] took away to reach it.
#[allow(dead_code)]
pub(crate) struct Snap {
    /// The nearest of the 24 rotations to the pose's.
    pub rotation: GridRotation,
    /// Where the sub-grid's origin goes in the target grid, after `rotation`.
    /// [`Snap::target_position`] adds it.
    pub shift: IVec3,
    /// The snapped pose against the target grid, with scale 1. It takes each
    /// voxel's centre to the centre of the target voxel that
    /// [`Snap::target_position`] names.
    pub pose: Transform,
    /// The angle, in radians, between the pose's rotation and `rotation`.
    pub angle: f32,
    /// How far the snap moves the centre of the solids' bounds, in target
    /// voxels. Each component is at most half a voxel.
    pub offset: Vec3,
    /// The pose's scale, which the snapped pose sets to 1.
    pub scale: Vec3,
}

#[allow(dead_code)]
impl Snap {
    /// The target grid position that sub-grid position `position` lands on.
    pub fn target_position(&self, position: IVec3) -> IVec3 {
        self.rotation.apply(position) + self.shift
    }
}

/// Snaps a sub-grid's pose against a target grid to the nearest pose where its
/// voxels sit on the target's voxels.
///
/// `pose` takes the sub-grid's grid coordinates to the target's, and its solids
/// fill the box from `min` (inclusive) to `max` (exclusive). The snap takes the
/// nearest of the 24 rotations and scale 1, keeping the centre of that box
/// where it was, then moves the box by at most half a voxel along each axis so
/// every voxel's centre lands on a target voxel's centre. It works from voxel
/// centres because a voxel's minimum corner, turned, isn't the turned voxel's
/// minimum corner.
///
/// A pose halfway between two rotations, to within float error, snaps to the
/// one listed first in [`GridRotation::ALL`]. A centre halfway between two
/// voxels along an axis snaps towards positive on that axis.
#[allow(dead_code)]
pub(crate) fn snap(pose: Transform, min: IVec3, max: IVec3) -> Snap {
    let turn = pose.rotation.normalize();
    let rotation = nearest_rotation(turn);
    let snapped = rotation.to_quat();

    let centre = (min + max).as_vec3() / 2.0;
    let target_centre = pose.transform_point(centre);
    // `snapped` takes voxel `v`'s centre, `v + 0.5`, to
    // `rotation.apply(v) + half_turned`, so a translation of
    // `shift + 0.5 - half_turned` puts it on target voxel `shift + apply(v)`'s
    // centre.
    let half_turned = rotation.apply(IVec3::ONE).as_vec3() / 2.0;
    let shift = (target_centre - snapped * centre + half_turned)
        .floor()
        .as_ivec3();
    let translation = shift.as_vec3() + Vec3::splat(0.5) - half_turned;
    let snapped_pose = Transform::from_translation(translation).with_rotation(snapped);

    Snap {
        rotation,
        shift,
        pose: snapped_pose,
        angle: angle_between(turn, snapped),
        offset: snapped_pose.transform_point(centre) - target_centre,
        scale: pose.scale,
    }
}

/// The nearest of the 24 rotations to `turn`, the first listed on a tie. A
/// `turn` that isn't a number snaps to the identity.
fn nearest_rotation(turn: Quat) -> GridRotation {
    // Closeness is the cosine of half the angle between them, so for a `turn`
    // halfway between two rotations it differs only by float error.
    const TIE: f32 = 1e-6;
    let closeness = GridRotation::ALL.map(|r| r.to_quat().dot(turn).abs());
    let nearest = closeness.iter().copied().fold(0.0, f32::max);
    let index = closeness
        .iter()
        .position(|&c| c >= nearest - TIE)
        .unwrap_or(0);
    GridRotation::ALL[index]
}

/// The angle between two unit rotations, in radians. Unlike
/// `Quat::angle_between`, it stays exact for rotations a hair apart.
fn angle_between(a: Quat, b: Quat) -> f32 {
    let difference = b.inverse() * a;
    2.0 * difference.xyz().length().atan2(difference.w.abs())
}

#[cfg(test)]
mod tests {
    use std::f32::consts::{FRAC_PI_2, FRAC_PI_4};

    use super::*;

    #[test]
    fn there_are_24_distinct_rotations_including_the_identity() {
        for (i, a) in GridRotation::ALL.iter().enumerate() {
            for b in &GridRotation::ALL[i + 1..] {
                assert_ne!(a, b);
            }
        }
        assert!(GridRotation::ALL.contains(&GridRotation::IDENTITY));
    }

    #[test]
    fn composing_two_rotations_gives_one_of_the_24() {
        for a in GridRotation::ALL {
            for b in GridRotation::ALL {
                assert!(GridRotation::ALL.contains(&(a * b)), "{a:?} * {b:?}");
                let v = IVec3::new(3, -7, 11);
                assert_eq!((a * b).apply(v), a.apply(b.apply(v)));
            }
        }
    }

    #[test]
    fn apply_agrees_with_to_quat_on_integer_vectors() {
        let vectors = [
            IVec3::X,
            IVec3::Y,
            IVec3::Z,
            IVec3::new(3, -7, 11),
            IVec3::new(-40, 2, 5),
        ];
        for rotation in GridRotation::ALL {
            let quat = rotation.to_quat();
            assert!(quat.is_normalized(), "{rotation:?}");
            for v in vectors {
                let turned = quat * v.as_vec3();
                assert!(
                    turned.abs_diff_eq(rotation.apply(v).as_vec3(), 1e-4),
                    "{rotation:?} turns {v} to {turned}, apply gives {}",
                    rotation.apply(v)
                );
            }
        }
    }

    #[test]
    fn the_quarter_turn_about_y_is_one_of_the_24() {
        let quarter = Quat::from_rotation_y(FRAC_PI_2);
        let found = GridRotation::ALL
            .into_iter()
            .find(|r| r.to_quat().dot(quarter).abs() > 1.0 - 1e-6)
            .unwrap();
        assert_eq!(found.apply(IVec3::X), IVec3::NEG_Z);
        assert_eq!(found.apply(IVec3::Z), IVec3::X);
    }

    /// The one of the 24 rotations that `quat` turns by.
    fn rotation_of(quat: Quat) -> GridRotation {
        GridRotation::ALL
            .into_iter()
            .find(|r| r.to_quat().dot(quat).abs() > 1.0 - 1e-6)
            .unwrap()
    }

    #[test]
    fn a_pose_a_few_degrees_off_snaps_to_that_rotation() {
        let quarter = Quat::from_rotation_y(FRAC_PI_2);
        let pose =
            Transform::from_rotation(quarter * Quat::from_rotation_x(3f32.to_radians()));
        let snapped = snap(pose, IVec3::ZERO, IVec3::splat(4));
        assert_eq!(snapped.rotation, rotation_of(quarter));
        assert!(snapped.pose.rotation.dot(quarter).abs() > 1.0 - 1e-6);
        assert!(
            (snapped.angle - 3f32.to_radians()).abs() < 1e-3,
            "{}",
            snapped.angle
        );
    }

    #[test]
    fn an_aligned_pose_snaps_to_itself_and_removes_nothing() {
        for rotation in GridRotation::ALL {
            let pose =
                Transform::from_xyz(4., -9., 17.).with_rotation(rotation.to_quat());
            let snapped = snap(pose, IVec3::new(-2, 0, 1), IVec3::new(3, 5, 4));
            assert_eq!(snapped.rotation, rotation);
            assert!(snapped.angle < 1e-5, "{rotation:?}: {}", snapped.angle);
            assert!(
                snapped.offset.abs_diff_eq(Vec3::ZERO, 1e-4),
                "{rotation:?}: {}",
                snapped.offset
            );
            assert!(snapped.pose.translation.abs_diff_eq(pose.translation, 1e-4));
        }
    }

    #[test]
    fn a_pose_past_halfway_snaps_to_the_next_rotation() {
        let quarter = Quat::from_rotation_y(FRAC_PI_2);
        let past = snap(
            Transform::from_rotation(Quat::from_rotation_y(50f32.to_radians())),
            IVec3::ZERO,
            IVec3::splat(4),
        );
        assert_eq!(past.rotation, rotation_of(quarter));
        assert!(
            (past.angle - 40f32.to_radians()).abs() < 1e-3,
            "{}",
            past.angle
        );

        let short = snap(
            Transform::from_rotation(Quat::from_rotation_y(40f32.to_radians())),
            IVec3::ZERO,
            IVec3::splat(4),
        );
        assert_eq!(short.rotation, GridRotation::IDENTITY);
        assert!(
            (short.angle - 40f32.to_radians()).abs() < 1e-3,
            "{}",
            short.angle
        );
    }

    #[test]
    fn a_pose_halfway_between_rotations_snaps_to_the_one_listed_first() {
        let listed = |r| GridRotation::ALL.iter().position(|x| *x == r).unwrap();
        for base in GridRotation::ALL {
            for axis in [Vec3::X, Vec3::Y, Vec3::Z] {
                for sign in [1., -1.] {
                    let next =
                        base * rotation_of(Quat::from_axis_angle(axis, sign * FRAC_PI_2));
                    let expected = if listed(base) < listed(next) {
                        base
                    } else {
                        next
                    };
                    let halfway =
                        base.to_quat() * Quat::from_axis_angle(axis, sign * FRAC_PI_4);
                    for _ in 0..2 {
                        let snapped = snap(
                            Transform::from_rotation(halfway),
                            IVec3::ZERO,
                            IVec3::ONE,
                        );
                        assert_eq!(snapped.rotation, expected, "{base:?} {axis} {sign}");
                    }
                }
            }
        }
    }

    #[test]
    fn a_translation_off_the_grid_snaps_to_the_nearest_voxel() {
        let pose = Transform::from_xyz(0.3, -0.2, 2.7);
        let snapped = snap(pose, IVec3::ZERO, IVec3::splat(2));
        assert_eq!(snapped.rotation, GridRotation::IDENTITY);
        assert_eq!(snapped.shift, IVec3::new(0, 0, 3));
        assert!(snapped
            .pose
            .translation
            .abs_diff_eq(Vec3::new(0., 0., 3.), 1e-5));
        assert!(
            snapped.offset.abs_diff_eq(Vec3::new(-0.3, 0.2, 0.3), 1e-5),
            "{}",
            snapped.offset
        );
    }

    #[test]
    fn a_translation_exactly_halfway_between_voxels_snaps_up() {
        for x in [0.5f32, -0.5] {
            let snapped =
                snap(Transform::from_xyz(x, 0., 0.), IVec3::ZERO, IVec3::splat(2));
            assert_eq!(snapped.shift.x, (x + 0.5) as i32, "{x}");
        }
    }

    #[test]
    fn a_scale_other_than_one_snaps_to_one() {
        let pose = Transform::from_scale(Vec3::splat(1.25));
        let snapped = snap(pose, IVec3::ZERO, IVec3::splat(2));
        assert_eq!(snapped.pose.scale, Vec3::ONE);
        assert_eq!(snapped.scale, Vec3::splat(1.25));
        assert_eq!(snapped.shift, IVec3::ZERO);
        assert!(
            snapped.offset.abs_diff_eq(Vec3::splat(-0.25), 1e-5),
            "{}",
            snapped.offset
        );
    }

    #[test]
    fn a_long_body_turns_about_the_centre_of_its_solids() {
        let (min, max) = (IVec3::ZERO, IVec3::new(1, 1, 64));
        let centre = Vec3::new(0.5, 0.5, 32.);
        let placed = Vec3::new(100.5, 10.5, 50.);
        let turn = Quat::from_rotation_y(4f32.to_radians());
        let pose =
            Transform::from_translation(placed - turn * centre).with_rotation(turn);

        let snapped = snap(pose, min, max);
        assert_eq!(snapped.rotation, GridRotation::IDENTITY);
        let moved = snapped.pose.transform_point(centre) - placed;
        assert!(moved.abs().max_element() <= 0.5, "{moved}");
        assert!(snapped.offset.abs_diff_eq(moved, 1e-4));

        // Taking the turn out about the grid's origin instead would carry the
        // centre along the arc the body's length swings through.
        let about_origin = Transform::from_translation(pose.translation.round());
        let swung = about_origin.transform_point(centre) - placed;
        assert!(swung.abs().max_element() > 1., "{swung}");
    }

    #[test]
    fn every_solid_lands_on_the_centre_of_the_voxel_its_mapping_names() {
        let (min, max) = (IVec3::new(-3, 1, 0), IVec3::new(5, 4, 7));
        let poses = [
            Transform::IDENTITY,
            Transform::from_xyz(5.3, -2.6, 11.9),
            Transform::from_xyz(-7.5, 0.2, 3.49)
                .with_rotation(Quat::from_euler(EulerRot::YXZ, 0.6, -1.9, 0.1))
                .with_scale(Vec3::splat(1.1)),
            Transform::from_xyz(0.1, 40.8, -9.6)
                .with_rotation(Quat::from_euler(EulerRot::XYZ, 2.9, 0.7, -2.2))
                .with_scale(Vec3::new(0.9, 1.05, 1.2)),
        ];
        for pose in poses {
            let snapped = snap(pose, min, max);
            for x in min.x..max.x {
                for y in min.y..max.y {
                    for z in min.z..max.z {
                        let v = IVec3::new(x, y, z);
                        let lands = snapped.pose.transform_point(v.as_vec3() + 0.5);
                        let target = snapped.target_position(v).as_vec3() + 0.5;
                        assert!(
                            lands.abs_diff_eq(target, 1e-4),
                            "{pose:?} {v}: {lands} vs {target}"
                        );
                    }
                }
            }
        }
    }
}
