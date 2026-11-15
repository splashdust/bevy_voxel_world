//! Carving a sphere of air where the cursor points, in whichever grid the
//! cursor's ray meets first: the world or a falling piece.

use bevy::prelude::*;
use bevy_voxel_world::prelude::*;

use super::panel::OverPanel;

/// How far the cursor's ray reaches, in voxels.
const RAY_LENGTH: f32 = 1000.0;

/// The radius of the sphere a click carves, in voxels.
#[derive(Resource)]
pub struct CarveRadius(pub i32);

/// Sets air in every voxel of `grid` within `radius` voxels of `centre`, in
/// that grid's own voxel positions. In a piece, air removes its voxels.
pub fn carve_sphere<C: VoxelWorldConfig>(
    voxel_world: &mut VoxelWorld<C>,
    grid: Grid,
    centre: IVec3,
    radius: i32,
) {
    let Some(mut grid) = voxel_world.grid_mut(grid) else {
        return;
    };
    grid.set_voxels_in(
        centre - IVec3::splat(radius),
        centre + IVec3::splat(radius + 1),
        |position| {
            ((position - centre).length_squared() <= radius * radius)
                .then_some(WorldVoxel::Air)
        },
    );
}

/// A left click carves a sphere around the voxel under the cursor, unless
/// the pointer is over the panel.
pub fn carve_on_click<C: VoxelWorldConfig>(
    buttons: Res<ButtonInput<MouseButton>>,
    over_panel: OverPanel,
    window: Query<&Window>,
    camera: Query<(&Camera, &GlobalTransform), With<VoxelWorldCamera<C>>>,
    radius: Res<CarveRadius>,
    mut voxel_world: VoxelWorld<C>,
) {
    if !buttons.just_pressed(MouseButton::Left) || over_panel.get() {
        return;
    }
    let (Ok(window), Ok((camera, camera_pose))) = (window.single(), camera.single())
    else {
        return;
    };
    let Some(ray) = window
        .cursor_position()
        .and_then(|cursor| camera.viewport_to_world(camera_pose, cursor).ok())
    else {
        return;
    };
    let Some(hit) =
        voxel_world.raycast_grids(ray, RAY_LENGTH, &|_| true, &|_, _, _| true)
    else {
        return;
    };
    carve_sphere(&mut voxel_world, hit.grid, hit.position, radius.0);
}
