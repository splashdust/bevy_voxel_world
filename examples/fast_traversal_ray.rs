use bevy::color::palettes::css;
use bevy::prelude::*;
use smooth_bevy_cameras::{
    controllers::unreal::{
        UnrealCameraBundle, UnrealCameraController, UnrealCameraPlugin,
    },
    LookTransformPlugin,
};
use std::sync::Arc;

use bevy_voxel_world::{prelude::*, traversal_alg::*};

// Declare materials as consts for convenience
const SNOWY_BRICK: u8 = 0;
const FULL_BRICK: u8 = 1;
const GRASS: u8 = 2;

#[derive(Resource, Clone, Default)]
struct MyMainWorld;

#[derive(Resource, Clone, Default)]
struct VoxelTrace {
    start: Option<Vec3>,
    end: Vec3,
}

impl VoxelWorldConfig for MyMainWorld {
    type MaterialIndex = u8;
    type ChunkUserBundle = ();

    fn texture_index_mapper(&self) -> Arc<dyn Fn(u8) -> [u32; 3] + Send + Sync> {
        Arc::new(|vox_mat: u8| match vox_mat {
            SNOWY_BRICK => [0, 1, 2],
            FULL_BRICK => [2, 2, 2],
            _ => [3, 3, 3],
        })
    }

    fn voxel_texture(&self) -> Option<(String, u32)> {
        Some(("example_voxel_texture.png".into(), 4))
    }
}

/// Controls:
/// - Left-click to place a voxel somewhere
/// - Ctrl + Left click to place the source of a voxel line trace
///     - Hold Ctrl to see the trace end follow the cursor
///     - Hold Ctrl and hit E to erase all solid voxels intersecting the trace
///     - Release Ctrl to stop tracing
/// - Unreal controls to move the camera around
fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        // We can specify a custom texture when initializing the plugin.
        // This should just be a path to an image in your assets folder.
        .add_plugins((
            VoxelWorldPlugin::with_config(MyMainWorld),
            LookTransformPlugin,
            UnrealCameraPlugin::default(),
        ))
        .init_resource::<VoxelTrace>()
        .add_systems(Startup, (setup, create_voxel_scene))
        .add_systems(Update, (inputs, update_cursor_cube, draw_trace))
        .run();
}

#[derive(Component)]
struct CursorCube {
    voxel_pos: IVec3,
    voxel_mat: u8,
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // Cursor cube
    commands.spawn((
        Transform::from_xyz(0.0, -10.0, 0.0),
        Mesh3d(meshes.add(Mesh::from(Cuboid {
            half_size: Vec3::splat(0.5),
        }))),
        MeshMaterial3d(materials.add(Color::srgba_u8(124, 144, 255, 128))),
        CursorCube {
            voxel_pos: IVec3::new(0, -10, 0),
            voxel_mat: FULL_BRICK,
        },
    ));

    // Camera
    commands
        .spawn((
            Camera3d::default(),
            // This tells bevy_voxel_world to use this cameras transform to calculate spawning area
            VoxelWorldCamera::<MyMainWorld>::default(),
        ))
        .insert(UnrealCameraBundle::new(
            UnrealCameraController::default(),
            Vec3::new(10.0, 10.0, 10.0),
            Vec3::ZERO,
            Vec3::Y,
        ));

    // light
    commands.spawn((
        PointLight {
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(4.0, 8.0, 4.0),
    ));
}

fn create_voxel_scene(mut voxel_world: VoxelWorld<MyMainWorld>) {
    // Then we can use the `u8` consts to specify the type of voxel

    // 20 by 20 floor
    for x in -10..10 {
        for z in -10..10 {
            voxel_world.set_voxel(IVec3::new(x, -1, z), WorldVoxel::Solid(GRASS));
            // Grassy floor
        }
    }

    // Some bricks
    voxel_world.set_voxel(IVec3::new(0, 0, 0), WorldVoxel::Solid(SNOWY_BRICK));
    voxel_world.set_voxel(IVec3::new(1, 0, 0), WorldVoxel::Solid(SNOWY_BRICK));
    voxel_world.set_voxel(IVec3::new(0, 0, 1), WorldVoxel::Solid(SNOWY_BRICK));
    voxel_world.set_voxel(IVec3::new(0, 0, -1), WorldVoxel::Solid(SNOWY_BRICK));
    voxel_world.set_voxel(IVec3::new(-1, 0, 0), WorldVoxel::Solid(FULL_BRICK));
    voxel_world.set_voxel(IVec3::new(-2, 0, 0), WorldVoxel::Solid(FULL_BRICK));
    voxel_world.set_voxel(IVec3::new(-1, 1, 0), WorldVoxel::Solid(SNOWY_BRICK));
    voxel_world.set_voxel(IVec3::new(-2, 1, 0), WorldVoxel::Solid(SNOWY_BRICK));
    voxel_world.set_voxel(IVec3::new(0, 1, 0), WorldVoxel::Solid(SNOWY_BRICK));
}

fn update_cursor_cube(
    voxel_world_raycast: VoxelWorld<MyMainWorld>,
    mut trace: ResMut<VoxelTrace>,
    camera_info: Query<(&Camera, &GlobalTransform), With<VoxelWorldCamera<MyMainWorld>>>,
    mut cursor_evr: EventReader<CursorMoved>,
    mut cursor_cube: Query<(&mut Transform, &mut CursorCube)>,
) {
    for ev in cursor_evr.read() {
        // Get a ray from the cursor position into the world
        let (camera, cam_gtf) = camera_info.single();
        let Ok(ray) = camera.viewport_to_world(cam_gtf, ev.position) else {
            return;
        };

        if let Some(result) = voxel_world_raycast.raycast(ray, &|(_pos, _vox)| true) {
            let (mut transform, mut cursor_cube) = cursor_cube.single_mut();

            // Camera could end up inside geometry - in that case just ignore the trace
            if let Some(normal) = result.normal {
                // Move the cursor cube to the position of the voxel we hit
                let voxel_pos = result.position + normal;
                transform.translation = voxel_pos + Vec3::splat(VOXEL_SIZE / 2.);
                cursor_cube.voxel_pos = voxel_pos.as_ivec3();

                // Update current trace end to the cursor cube position
                trace.end = transform.translation;
            }
        }
    }
}

fn draw_trace(trace: Res<VoxelTrace>, mut gizmos: Gizmos) {
    if let Some(trace_start) = trace.start {
        gizmos.line(trace_start, trace.end, css::RED);

        voxel_line_traversal(trace_start, trace.end, |voxel_coord, time, face| {
            let voxel_center = voxel_coord.as_vec3() + Vec3::splat(VOXEL_SIZE / 2.);

            gizmos.cuboid(
                Transform::from_translation(voxel_center)
                    .with_scale(Vec3::splat(VOXEL_SIZE)),
                css::PINK,
            );

            if let Ok(normal) = face.try_into() {
                gizmos.circle(
                    Isometry3d::new(
                        voxel_center + (normal * VOXEL_SIZE / 2.),
                        Quat::from_rotation_arc(Vec3::Z, normal),
                    ),
                    0.8 * VOXEL_SIZE / 2.,
                    css::RED.with_alpha(0.5),
                );

                gizmos.sphere(
                    Isometry3d::new(
                        trace_start + (trace.end - trace_start) * time,
                        Quat::IDENTITY,
                    ),
                    0.1,
                    Color::BLACK,
                );
            }

            // Keep drawing until trace has finished visiting all voxels along the way
            const NEVER_STOP: bool = true;
            NEVER_STOP
        });
    }
}

fn inputs(
    buttons: Res<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
    mut voxel_world: VoxelWorld<MyMainWorld>,
    mut trace: ResMut<VoxelTrace>,
    cursor_cube: Query<&CursorCube>,
) {
    if keys.just_released(KeyCode::ControlLeft) {
        trace.start = None;
    } else if keys.pressed(KeyCode::ControlLeft) && keys.just_pressed(KeyCode::KeyE) {
        let cursor = cursor_cube.single();
        let trace_end = cursor.voxel_pos.as_vec3() + Vec3::splat(VOXEL_SIZE / 2.);

        voxel_line_traversal(
            trace.start.unwrap(),
            trace_end,
            |voxel_coords, _time, _face| {
                voxel_world.set_voxel(voxel_coords, WorldVoxel::Air);

                // Keep erasing until trace has finished visiting all voxels along the way
                const NEVER_STOP: bool = true;
                NEVER_STOP
            },
        );
    }

    if buttons.just_pressed(MouseButton::Left) {
        let cursor = cursor_cube.single();

        if keys.pressed(KeyCode::ControlLeft) {
            trace.start = Some(cursor.voxel_pos.as_vec3() + Vec3::splat(VOXEL_SIZE / 2.))
        } else {
            let vox_pos = cursor.voxel_pos;
            voxel_world.set_voxel(vox_pos, WorldVoxel::Solid(cursor.voxel_mat));
        }
    }
}
