//! Stamping a piece back into the world once it comes to rest, without the
//! snap showing.
//!
//! A stamp snaps a piece to the nearest pose that lines up with the world's
//! voxels: a quarter turn and a whole voxel. Stamped from where it rests, a
//! piece would jump to that pose in the frame its stamp lands. So when avian
//! puts a piece's body to sleep, it leaves the simulation as a kinematic
//! body, `stamp_preview` gives the pose the stamp will snap it to, and its
//! `Transform` eases there over [`EASE_SECONDS`]. The stamp is called once
//! it arrives, and lands with the piece already where its voxels go.
//!
//! The preview's pose is against the world's root, and a piece is the
//! root's child once it falls, so the pose is its `Transform`. The stamp
//! snaps the pose shown last frame, a moment short of the preview's, which
//! snaps to the same result.

use avian3d::prelude::*;
use bevy::prelude::*;
use bevy_voxel_world::prelude::*;

/// How long a resting piece takes to ease to its stamped pose, in seconds.
pub const EASE_SECONDS: f32 = 0.3;

/// A piece lifted out of a grid, from its `Lifted` until it despawns.
#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub enum Piece {
    /// It falls as a dynamic body.
    Falling,
    /// It rests and eases from `from` to `to`, the pose its stamp gives it.
    Easing {
        from: Transform,
        to: Transform,
        elapsed: f32,
    },
    /// Its stamp has been called, and it despawns when the stamp lands.
    Stamping,
}

/// How many pieces have been stamped back into the world.
#[derive(Resource, Default)]
pub struct Pieces {
    pub stamped: u32,
}

/// `from` moved a fraction `t` of the way to `to`, starting and ending
/// slowly. From a `t` of 1 on, it is exactly `to`.
pub fn ease(from: Transform, to: Transform, t: f32) -> Transform {
    if t <= 0.0 {
        return from;
    }
    if t >= 1.0 {
        return to;
    }
    let t = t * t * (3.0 - 2.0 * t);
    Transform {
        translation: from.translation.lerp(to.translation, t),
        rotation: from.rotation.slerp(to.rotation, t),
        scale: from.scale.lerp(to.scale, t),
    }
}

/// Takes each piece that fell asleep out of the simulation, eases it to its
/// stamped pose, and stamps it into the world when it gets there.
pub fn ease_resting<C: VoxelWorldConfig>(
    time: Res<Time>,
    mut voxel_world: VoxelWorld<C>,
    mut pieces: Query<(Entity, &mut Piece, &mut Transform, Has<Sleeping>)>,
    mut commands: Commands,
) {
    for (entity, mut piece, mut transform, sleeping) in &mut pieces {
        match *piece {
            Piece::Falling if sleeping => {
                let Some(preview) = voxel_world.stamp_preview(entity, Grid::World) else {
                    continue;
                };
                commands.entity(entity).try_insert((
                    RigidBody::Kinematic,
                    LinearVelocity::ZERO,
                    AngularVelocity::ZERO,
                ));
                *piece = Piece::Easing {
                    from: *transform,
                    to: preview.pose,
                    elapsed: 0.0,
                };
            }
            Piece::Easing { from, to, elapsed } => {
                let elapsed = elapsed + time.delta_secs();
                *transform = ease(from, to, elapsed / EASE_SECONDS);
                if elapsed < EASE_SECONDS {
                    *piece = Piece::Easing { from, to, elapsed };
                    continue;
                }
                // Stamping into the world fails only for a piece that is no
                // longer a live sub-grid, which nothing is left to stamp.
                if let Err(error) =
                    voxel_world.stamp(entity, Grid::World, incoming_replaces)
                {
                    warn!("A resting piece wasn't stamped: {error}");
                }
                *piece = Piece::Stamping;
            }
            _ => {}
        }
    }
}

/// Counts the stamps that landed. The crate despawns each stamped piece in
/// this frame's `PostUpdate`.
pub fn count_stamps<C: VoxelWorldConfig>(
    mut stamped: MessageReader<Stamped<C>>,
    mut pieces: ResMut<Pieces>,
) {
    pieces.stamped += stamped.read().count() as u32;
}

/// Below this height a piece has fallen past every collider, off the edge
/// of the chunks spawned around the camera.
const LOST_BELOW: f32 = -256.0;

/// Despawns each falling piece that has no voxels left, carved away, or
/// that fell past every collider.
pub fn drop_lost<C: VoxelWorldConfig>(
    mut voxel_world: VoxelWorld<C>,
    pieces: Query<(Entity, &Piece, &GlobalTransform)>,
) {
    for (entity, piece, pose) in &pieces {
        if *piece != Piece::Falling {
            continue;
        }
        let empty = voxel_world
            .grid(Grid::SubGrid(entity))
            .is_none_or(|grid| grid.bounds().is_none());
        if empty || pose.translation().y < LOST_BELOW {
            voxel_world.despawn_sub_grid(entity);
        }
    }
}
