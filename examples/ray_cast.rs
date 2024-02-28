use bevy::prelude::*;
use bevy_voxel_world::prelude::*;
use std::sync::Arc;

// Declare materials as consts for convenience
const SNOWY_BRICK: u8 = 0;
const FULL_BRICK: u8 = 1;
const GRASS: u8 = 2;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        // We can specify a custom texture when initializing the plugin.
        // This should just be a path to an image in your assets folder.
        .add_plugins(VoxelWorldPlugin::default().with_voxel_texture(
            "example_voxel_texture.png",
            4, // number of indexes in the texture
        ))
        .add_systems(Startup, (setup, create_voxel_scene))
        .add_systems(Update, (update_cursor_cube, mouse_button_input))
        .run();
}

#[derive(Component)]
struct CursorCube {
    voxel_pos: IVec3,
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    commands.insert_resource(VoxelWorldConfiguration {
        // To specify how the texture map to different kind of voxels we add this mapping callback
        // For each material type, we specify the texture coordinates for the top, side and bottom faces.
        texture_index_mapper: Arc::new(|vox_mat: u8| match vox_mat {
            SNOWY_BRICK => [0, 1, 2],
            FULL_BRICK => [2, 2, 2],
            GRASS | _ => [3, 3, 3],
        }),
        ..Default::default()
    });

    // Cursor cube
    commands.spawn((
        PbrBundle {
            mesh: meshes.add(Mesh::from(shape::Cube { size: 1.0 })),
            material: materials.add(Color::rgba_u8(124, 144, 255, 128)),
            transform: Transform::from_xyz(0.0, -10.0, 0.0),
            ..default()
        },
        CursorCube {
            voxel_pos: IVec3::new(0, -10, 0),
        },
    ));

    // Camera
    commands.spawn((
        Camera3dBundle {
            transform: Transform::from_xyz(10.0, 10.0, 10.0).looking_at(Vec3::ZERO, Vec3::Y),
            ..default()
        },
        // This tells bevy_voxel_world tos use this cameras transform to calculate spawning area
        VoxelWorldCamera,
    ));

    // light
    commands.spawn(PointLightBundle {
        point_light: PointLight {
            shadows_enabled: true,
            ..default()
        },
        transform: Transform::from_xyz(4.0, 8.0, 4.0),
        ..default()
    });
}

fn create_voxel_scene(mut voxel_world: VoxelWorld) {
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
    voxel_world_raycast: VoxelWorldRaycast,
    camera_info: Query<(&Camera, &GlobalTransform), With<VoxelWorldCamera>>,
    mut cursor_evr: EventReader<CursorMoved>,
    mut cursor_cube: Query<(&mut Transform, &mut CursorCube)>,
) {
    for ev in cursor_evr.read() {
        // Get a ray from the cursor position into the world
        let (camera, cam_gtf) = camera_info.single();
        let Some(ray) = camera.viewport_to_world(cam_gtf, ev.position) else {
            return;
        };

        if let Some(result) = voxel_world_raycast.raycast(ray, &|(_pos, _vox)| true) {
            let (mut transform, mut cursor_cube) = cursor_cube.single_mut();
            // Move the cursor cube to the position of the voxel we hit
            let voxel_pos = result.position + result.normal;
            transform.translation = voxel_pos + Vec3::new(0.5, 0.5, 0.5);
            cursor_cube.voxel_pos = voxel_pos.as_ivec3();
        }
    }
}

fn mouse_button_input(
    buttons: Res<ButtonInput<MouseButton>>,
    mut voxel_world: VoxelWorld,
    cursor_cube: Query<&CursorCube>,
) {
    if buttons.just_pressed(MouseButton::Left) {
        let vox_pos = cursor_cube.single().voxel_pos;
        voxel_world.set_voxel(vox_pos, WorldVoxel::Solid(FULL_BRICK));
    }
}
