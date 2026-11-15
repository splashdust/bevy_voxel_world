//! Colliders built from the voxels of every grid's chunks, and a rigid body
//! for each lifted piece.
//!
//! A chunk's collider is rebuilt each time the chunk publishes, from its
//! published voxels rather than its mesh, so it works with any mesher. It
//! goes on the chunk's entity, and despawns with the chunk. A world chunk is
//! a static body. A sub-grid's chunks are children of the sub-grid, so avian
//! folds their colliders into the sub-grid's body.
//!
//! A lift's chunks, in both grids, publish in the frame its handover lands,
//! which is also the frame `Lifted` arrives. Both systems here run then,
//! before the frame's physics step, so the hole's colliders and the piece's
//! swap in the same step and the piece never falls through the hole it came
//! from.

use avian3d::prelude::*;
use bevy::prelude::*;
use bevy_voxel_world::{custom_meshing::CHUNK_SIZE_I, prelude::*};

use super::rest::Piece;

/// The level-0 chunks' voxel size, and so each collider's.
const VOXEL: Vec3 = Vec3::ONE;

/// The positions of a level-0 chunk's own solid voxels, in its entity's
/// frame, from `voxel_at`, which reads the chunk's voxel at a voxel
/// position. `first` is the chunk's lowest voxel.
///
/// A chunk's entity sits one voxel below its first voxel, as its padded data
/// does, so its own voxels are at 1 to `CHUNK_SIZE` on each axis. The padding
/// around a chunk repeats its neighbours' voxels, which their own colliders
/// hold, so it is left out.
pub fn collider_voxels<I: PartialEq>(
    first: IVec3,
    voxel_at: impl Fn(IVec3) -> Option<WorldVoxel<I>>,
) -> Vec<IVec3> {
    let mut voxels = Vec::new();
    for z in 0..CHUNK_SIZE_I {
        for y in 0..CHUNK_SIZE_I {
            for x in 0..CHUNK_SIZE_I {
                let offset = IVec3::new(x, y, z);
                if voxel_at(first + offset).is_some_and(|voxel| voxel.is_solid()) {
                    voxels.push(offset + IVec3::ONE);
                }
            }
        }
    }
    voxels
}

/// Rebuilds the collider of each chunk published this frame, in any grid,
/// and makes each world chunk a static body.
pub fn rebuild_colliders<C: VoxelWorldConfig>(
    mut published: MessageReader<ChunkPublished<C>>,
    voxel_world: VoxelWorld<C>,
    bodies: Query<(), With<RigidBody>>,
    mut commands: Commands,
) {
    for message in published.read() {
        let Some(chunk) = voxel_world
            .grid(message.grid)
            .and_then(|grid| grid.get_chunk_data(message.chunk_key))
        else {
            continue;
        };
        let Ok(mut entity) = commands.get_entity(message.entity) else {
            continue;
        };
        // A chunk above LOD level 0 samples the world coarsely, and an empty
        // one holds no solid voxel, padding included.
        let voxels = if chunk.lod_level() != 0 || chunk.is_empty() {
            Vec::new()
        } else {
            collider_voxels(chunk.key().min_voxel(), |position| {
                chunk.get_voxel_at_world_position(position)
            })
        };
        if voxels.is_empty() {
            entity.try_remove::<Collider>();
            continue;
        }
        entity.try_insert(Collider::voxels(VOXEL, &voxels));
        if message.grid == Grid::World && !bodies.contains(message.entity) {
            entity.try_insert(RigidBody::Static);
        }
    }
}

/// Gives each piece whose lift has landed a dynamic body, moving at the
/// speed its part of the source moved.
///
/// A piece spawns as a child of the grid it was lifted from. One cut from a
/// falling piece would ride along with it, and despawn with it if that piece
/// is stamped, so it moves under the world's root first, keeping its pose.
pub fn start_falling<C: VoxelWorldConfig>(
    mut lifted: MessageReader<Lifted<C>>,
    parents: Query<&ChildOf>,
    sub_grids: Query<(), With<SubGrid<C>>>,
    bodies: Query<(
        &LinearVelocity,
        &AngularVelocity,
        &ComputedCenterOfMass,
        &GlobalTransform,
    )>,
    poses: Query<&GlobalTransform>,
    voxel_world: VoxelWorld<C>,
    mut commands: Commands,
) {
    for message in lifted.read() {
        let piece = message.sub_grid;
        let Ok(&ChildOf(parent)) = parents.get(piece) else {
            continue;
        };
        if sub_grids.contains(parent) {
            let root = parents
                .iter_ancestors(piece)
                .find(|ancestor| !sub_grids.contains(*ancestor));
            if let Some(root) = root {
                commands.entity(piece).set_parent_in_place(root);
            }
        }

        // A point on a turning body moves at its body's speed plus the turn
        // about the body's centre of mass.
        let mut velocity = (Vec3::ZERO, Vec3::ZERO);
        let centre = voxel_world
            .grid(Grid::SubGrid(piece))
            .and_then(|grid| grid.bounds())
            .zip(poses.get(piece).ok())
            .map(|((min, max), pose)| pose.transform_point((min + max).as_vec3() / 2.0));
        if let (Grid::SubGrid(source), Some(centre)) = (message.source, centre) {
            if let Ok((linear, angular, mass_centre, pose)) = bodies.get(source) {
                let turn = angular
                    .0
                    .cross(centre - pose.transform_point(mass_centre.0));
                velocity = (linear.0 + turn, angular.0);
            }
        }
        commands.entity(piece).try_insert((
            RigidBody::Dynamic,
            LinearVelocity(velocity.0),
            AngularVelocity(velocity.1),
            Piece::Falling,
        ));
    }
}
