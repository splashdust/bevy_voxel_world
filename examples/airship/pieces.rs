//! Pieces lifted off the airship, and stamped back onto it.
//!
//! X lifts every solid voxel in a small box around the voxel under the
//! cursor, out of the airship's grid into a new sub-grid, a piece. The lift
//! spawns the piece as the airship's child, so it rides along. Once its lift
//! lands, it slides a few voxels out along the face the cursor was on.
//!
//! V slides the piece under the cursor, or the newest piece, back to where
//! it was lifted from, then stamps it into the airship's grid in the same
//! frame. Back there its grid lines up with the airship's again, so the stamp
//! doesn't snap, and the handover shows no frame with the voxels in both
//! grids or in neither.

use bevy::prelude::*;
use bevy_voxel_world::prelude::*;

use crate::{flight::Flight, AirshipWorld, Cursor, Status};

/// How far the lift box reaches from the voxel under the cursor along each
/// axis.
const PIECE_REACH: i32 = 2;
/// How far a piece slides out, in voxels, and how long it takes either way,
/// in seconds.
const SLIDE_OUT: f32 = 3.0;
const SLIDE_SECS: f32 = 0.8;

/// A piece lifted off the airship.
#[derive(Component)]
pub struct Piece {
    /// The way it slides out, on the airship's grid.
    out: Vec3,
    /// Where it was lifted from: its translation against the airship when
    /// its lift landed.
    home: Vec3,
    state: PieceState,
    /// The order pieces were lifted in, so V finds the newest.
    order: u32,
}

#[derive(Clone, Copy, PartialEq)]
enum PieceState {
    /// It waits for `Lifted`.
    Lifting,
    /// It slides out, or has.
    Out { elapsed: f32 },
    /// It slides back from `from`, then is stamped.
    Back { from: Vec3, elapsed: f32 },
    /// It waits for its stamp to land.
    Stamping,
}

/// How many pieces have been lifted.
#[derive(Resource, Default)]
pub struct PieceCount(u32);

/// X lifts a piece off the airship around the voxel under the cursor.
pub fn lift_key(
    keys: Res<ButtonInput<KeyCode>>,
    cursor: Cursor,
    flight: Res<Flight>,
    mut voxel_world: VoxelWorld<AirshipWorld>,
    mut count: ResMut<PieceCount>,
    mut commands: Commands,
    mut status: ResMut<Status>,
) {
    if keys.just_pressed(KeyCode::KeyX) {
        let hit = cursor.hit(&voxel_world);
        let on_ship = hit.filter(|hit| {
            flight
                .ship
                .is_some_and(|ship| hit.grid == Grid::SubGrid(ship))
        });
        let Some(hit) = on_ship else {
            status.set("Point at the airship to lift a piece off it".into());
            return;
        };
        let normal = hit.normal.unwrap_or(IVec3::Y);
        if lift_piece(
            &mut voxel_world,
            &mut commands,
            &mut count,
            hit.grid,
            hit.position,
            normal,
        )
        .is_some()
        {
            status.set("Lifted a piece off the airship".into());
        }
    }
}

/// V sends the piece under the cursor back to be stamped, or the newest
/// piece if the cursor is on none.
pub fn send_back_key(
    keys: Res<ButtonInput<KeyCode>>,
    cursor: Cursor,
    voxel_world: VoxelWorld<AirshipWorld>,
    mut pieces: Query<(Entity, &mut Piece, &Transform)>,
    mut status: ResMut<Status>,
) {
    if keys.just_pressed(KeyCode::KeyV) {
        let hovered = cursor.hit(&voxel_world).and_then(|hit| match hit.grid {
            Grid::SubGrid(entity) => Some(entity),
            Grid::World => None,
        });
        let out = |piece: &Piece| matches!(piece.state, PieceState::Out { .. });
        let chosen = hovered
            .filter(|entity| pieces.get(*entity).is_ok_and(|(_, piece, _)| out(piece)))
            .or_else(|| {
                pieces
                    .iter()
                    .filter(|(_, piece, _)| out(piece))
                    .max_by_key(|(_, piece, _)| piece.order)
                    .map(|(entity, _, _)| entity)
            });
        let Some((_, mut piece, transform)) =
            chosen.and_then(|entity| pieces.get_mut(entity).ok())
        else {
            status.set("No piece to stamp back".into());
            return;
        };
        piece.send_back(transform.translation);
    }
}

/// Lifts every solid voxel in a small box around `position` out of `grid`
/// into a piece that slides out along `normal` once its lift lands.
///
/// The lift spawns the piece through `voxel_world`'s commands, so a system
/// calling this declares `voxel_world` before `commands`: a system's
/// parameters apply their commands in order, and the piece must exist
/// before it takes its `Piece`.
pub fn lift_piece(
    voxel_world: &mut VoxelWorld<AirshipWorld>,
    commands: &mut Commands,
    count: &mut PieceCount,
    grid: Grid,
    position: IVec3,
    normal: IVec3,
) -> Option<Entity> {
    let reach = IVec3::splat(PIECE_REACH);
    let piece = voxel_world
        .lift(grid, position - reach, position + reach + 1, |_, voxel| {
            voxel.is_solid()
        })
        .ok()?;
    count.0 += 1;
    commands.entity(piece).insert(Piece {
        out: normal.as_vec3(),
        home: Vec3::ZERO,
        state: PieceState::Lifting,
        order: count.0,
    });
    Some(piece)
}

impl Piece {
    /// Starts sliding the piece back from its `translation` now, to be
    /// stamped once it is home.
    pub fn send_back(&mut self, translation: Vec3) {
        self.state = PieceState::Back {
            from: translation,
            elapsed: 0.0,
        };
    }
}

/// Lets each piece slide out once its lift lands.
pub fn follow_lifts(
    mut lifted: MessageReader<Lifted<AirshipWorld>>,
    mut pieces: Query<(&mut Piece, &Transform)>,
) {
    for lift in lifted.read() {
        if let Ok((mut piece, transform)) = pieces.get_mut(lift.sub_grid) {
            piece.home = transform.translation;
            piece.state = PieceState::Out { elapsed: 0.0 };
        }
    }
}

/// Slides each piece out, or back, and stamps a piece that is back home
/// into the grid it was lifted from.
pub fn slide(
    time: Res<Time>,
    flight: Res<Flight>,
    mut voxel_world: VoxelWorld<AirshipWorld>,
    mut pieces: Query<(Entity, &mut Piece, &mut Transform)>,
    mut status: ResMut<Status>,
) {
    let dt = time.delta_secs();
    for (entity, mut piece, mut transform) in &mut pieces {
        let home = piece.home;
        let out = home + piece.out * SLIDE_OUT;
        match &mut piece.state {
            PieceState::Lifting | PieceState::Stamping => {}
            PieceState::Out { elapsed } => {
                *elapsed += dt;
                transform.translation = home.lerp(out, ease(*elapsed / SLIDE_SECS));
            }
            PieceState::Back { from, elapsed } => {
                *elapsed += dt;
                if *elapsed < SLIDE_SECS {
                    transform.translation = from.lerp(home, ease(*elapsed / SLIDE_SECS));
                    continue;
                }
                transform.translation = home;
                let Some(ship) = flight.ship else {
                    continue;
                };
                match voxel_world.stamp(entity, Grid::SubGrid(ship), incoming_replaces) {
                    Ok(()) => {
                        piece.state = PieceState::Stamping;
                        status.set("Stamped a piece back onto the airship".into());
                    }
                    Err(error) => {
                        status.set(format!("Stamped nothing: {error}"));
                        piece.state = PieceState::Out { elapsed: 0.0 };
                    }
                }
            }
        }
    }
}

fn ease(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}
