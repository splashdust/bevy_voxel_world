//! Tests of debug draw on sub-grids: chunk boxes read from each chunk's own
//! grid, and voxel gizmos that follow their sub-grid and go with it.

use bevy::ecs::system::RunSystemOnce;
use bevy::gizmos::{config::DefaultGizmoConfigGroup, AppGizmoBuilder, GizmoAsset};
use bevy::prelude::*;

use crate::chunk::{Chunk, ChunkData};
use crate::chunk_map::ChunkMap;
use crate::debug::{VoxelGizmo, VoxelWorldDebugDraw, VoxelWorldDebugDrawPlugin};
use crate::debug_draw::{
    chunk_box, for_each_chunk_box, VoxelGizmos, SUB_GRID_CHUNK_COLOUR,
};
use crate::grid::GridKey;
use crate::prelude::*;
use crate::test::{test_plugins, update_until, FloorWorld};

/// A world with no camera, so it streams no chunks of its own, whose
/// transforms propagate and whose voxel gizmos draw.
fn app() -> App {
    let mut app = App::new();
    // Gizmos draw into assets, which a headless app needs the asset plugin
    // for.
    app.add_plugins((test_plugins(), AssetPlugin::default(), TransformPlugin))
        .init_asset::<GizmoAsset>()
        .init_gizmo_group::<DefaultGizmoConfigGroup>()
        .add_plugins((
            VoxelWorldPlugin::<FloorWorld>::minimal(),
            VoxelWorldDebugDrawPlugin::<FloorWorld>::default(),
        ));
    app.update();
    app
}

/// A quarter turn about +Y, scaled by two and moved, so a misplaced box or
/// circle can't land in the right place by chance.
fn pose() -> Transform {
    Transform::from_translation(Vec3::new(10.0, 20.0, 30.0))
        .with_rotation(Quat::from_rotation_y(std::f32::consts::FRAC_PI_2))
        .with_scale(Vec3::splat(2.0))
}

fn spawn_sub_grid(app: &mut App, transform: Transform, solid: Option<IVec3>) -> Entity {
    app.world_mut()
        .run_system_once(move |mut voxel_world: VoxelWorld<FloorWorld>| {
            let entity = voxel_world.spawn_sub_grid(transform);
            if let Some(position) = solid {
                voxel_world
                    .grid_mut(Grid::SubGrid(entity))
                    .unwrap()
                    .set_voxel(position, WorldVoxel::Solid(1));
            }
            entity
        })
        .unwrap()
}

fn debug_draw(
    app: &mut App,
    f: impl FnOnce(&VoxelWorldDebugDraw<FloorWorld>) + Send + Sync + 'static,
) {
    let mut f = Some(f);
    app.world_mut()
        .run_system_once(move |debug_draw: VoxelWorldDebugDraw<FloorWorld>| {
            (f.take().unwrap())(&debug_draw)
        })
        .unwrap();
}

/// The centres of the circles the voxel gizmo draw places, with each
/// sub-grid's pose read by `pose_of`.
fn circle_centres(
    app: &App,
    pose_of: impl Fn(Entity) -> Option<GlobalTransform>,
) -> Vec<Vec3> {
    app.world()
        .resource::<VoxelGizmos<FloorWorld>>()
        .placed_circle_centres(pose_of)
}

/// The centres of the circles the draw places for the sub-grids still alive.
fn live_circle_centres(app: &App) -> Vec<Vec3> {
    circle_centres(app, |entity| {
        app.world()
            .get_entity(entity)
            .ok()
            .filter(|entity| entity.contains::<SubGrid<FloorWorld>>())
            .and_then(|entity| entity.get::<GlobalTransform>().copied())
    })
}

/// Asserts that `centres` are the six face centres of the voxel at `pos`
/// under `transform`, and nothing else.
fn assert_on_voxel(centres: &[Vec3], pos: IVec3, transform: Transform) {
    let transform = GlobalTransform::from(transform);
    let voxel_centre = pos.as_vec3() + 0.5;
    let expected: Vec<Vec3> = [
        Vec3::X,
        Vec3::NEG_X,
        Vec3::Y,
        Vec3::NEG_Y,
        Vec3::Z,
        Vec3::NEG_Z,
    ]
    .into_iter()
    .map(|normal| transform.transform_point(voxel_centre + normal * 0.5))
    .collect();
    assert_eq!(centres.len(), 6, "{centres:?}");
    for centre in expected {
        assert!(
            centres.iter().any(|c| c.abs_diff_eq(centre, 1e-3)),
            "no circle at {centre} in {centres:?}"
        );
    }
}

#[test]
fn a_sub_grid_chunks_box_reads_the_sub_grids_data_not_the_worlds() {
    let mut app = app();
    let sub_grid = spawn_sub_grid(&mut app, pose(), Some(IVec3::new(5, 5, 5)));
    update_until(&mut app, "the sub-grid's chunk to be ready", |app| {
        app.world_mut()
            .run_system_once(move |voxel_world: VoxelWorld<FloorWorld>| {
                voxel_world
                    .grid(Grid::SubGrid(sub_grid))
                    .unwrap()
                    .chunk_state(ChunkKey::new(IVec3::ZERO, 0))
            })
            .unwrap()
            == Some(ChunkStatus::Ready)
    });

    // The world has an empty chunk at the same key, which draws no box.
    let origin = ChunkKey::new(IVec3::ZERO, 0);
    app.world()
        .resource::<ChunkMap<FloorWorld, u8>>()
        .get_map()
        .write()
        .unwrap()
        .insert(
            GridKey::world(origin),
            ChunkData {
                key: origin,
                ..ChunkData::new()
            },
        );

    let boxes = app
        .world_mut()
        .run_system_once(
            |chunks: Query<(&Chunk<FloorWorld>, &GlobalTransform)>,
             voxel_world: VoxelWorld<FloorWorld>| {
                let mut boxes = Vec::new();
                for_each_chunk_box(chunks, &voxel_world, |placement, colour| {
                    boxes.push((placement, colour))
                });
                boxes
            },
        )
        .unwrap();

    // The sub-grid's chunk sits one voxel below its minimum corner, as the
    // world's chunks do under the root.
    let chunk_transform = pose() * Transform::from_translation(Vec3::splat(-1.0));
    assert_eq!(boxes.len(), 1, "{boxes:?}");
    let (placement, colour) = boxes[0];
    assert_eq!(colour, SUB_GRID_CHUNK_COLOUR);
    let expected = chunk_box(&GlobalTransform::from(chunk_transform), 0);
    assert!(
        placement.affine().abs_diff_eq(expected.affine(), 1e-3),
        "{placement:?} isn't {expected:?}"
    );
}

#[test]
fn a_sub_grids_voxel_gizmo_follows_it_and_goes_when_it_despawns() {
    let mut app = app();
    let at = IVec3::new(1, 2, 3);
    let sub_grid = spawn_sub_grid(&mut app, pose(), None);
    debug_draw(&mut app, move |debug_draw| {
        debug_draw.set_grid_voxel_gizmo(
            Grid::SubGrid(sub_grid),
            VoxelGizmo {
                color: Srgba::RED,
                pos: at,
            },
        );
    });
    app.update();
    assert_on_voxel(&live_circle_centres(&app), at, pose());

    // Moving the sub-grid moves its gizmo in the same frame.
    let moved = pose().with_translation(Vec3::new(-4.0, 0.0, 8.0));
    app.world_mut().entity_mut(sub_grid).insert(moved);
    app.update();
    assert_on_voxel(&live_circle_centres(&app), at, moved);

    // Once the draw has found the sub-grid gone, its gizmo is gone too, even
    // for a caller that would still give the entity a pose.
    app.world_mut().entity_mut(sub_grid).despawn();
    app.update();
    assert!(circle_centres(&app, |_| Some(GlobalTransform::IDENTITY)).is_empty());

    // A later sub-grid doesn't show it.
    spawn_sub_grid(&mut app, Transform::IDENTITY, None);
    app.update();
    assert!(live_circle_centres(&app).is_empty());
}

#[test]
fn clearing_a_sub_grids_gizmo_leaves_the_worlds_and_clearing_all_clears_both() {
    let mut app = app();
    let at = IVec3::new(1, 2, 3);
    let sub_grid = spawn_sub_grid(&mut app, pose(), None);
    let gizmo = move || VoxelGizmo {
        color: Srgba::RED,
        pos: at,
    };
    debug_draw(&mut app, move |debug_draw| {
        debug_draw.set_voxel_gizmo(gizmo());
        debug_draw.set_grid_voxel_gizmo(Grid::SubGrid(sub_grid), gizmo());
        debug_draw.clear_grid_voxel_gizmo(Grid::SubGrid(sub_grid), at);
    });
    app.update();
    assert_on_voxel(&live_circle_centres(&app), at, Transform::IDENTITY);

    // `Grid::World` names the gizmos the world's own methods set.
    debug_draw(&mut app, move |debug_draw| {
        debug_draw.clear_grid_voxel_gizmo_fn()(Grid::World, at);
    });
    assert!(live_circle_centres(&app).is_empty());

    debug_draw(&mut app, move |debug_draw| {
        debug_draw.set_grid_voxel_gizmo_fn()(Grid::World, gizmo());
        debug_draw.set_grid_voxel_gizmo_fn()(Grid::SubGrid(sub_grid), gizmo());
    });
    assert_eq!(live_circle_centres(&app).len(), 12);
    debug_draw(&mut app, |debug_draw| {
        debug_draw.clear_all_voxel_gizmos_fn()()
    });
    assert!(circle_centres(&app, |_| Some(GlobalTransform::IDENTITY)).is_empty());
}
