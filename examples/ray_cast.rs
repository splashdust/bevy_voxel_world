use std::sync::Arc;

use bevy::prelude::*;

use bevy_voxel_world::prelude::*;

// Declare materials as consts for convenience
const SNOWY_BRICK: u8 = 0;
const FULL_BRICK: u8 = 1;
const GRASS: u8 = 2;

#[derive(Resource, Clone, Default)]
struct MyMainWorld;

impl VoxelWorldConfig for MyMainWorld {
    type MaterialIndex = u8;

    fn texture_index_mapper(&self) -> Arc<dyn Fn(Self::MaterialIndex) -> [u32; 3] + Send + Sync> {
        Arc::new(|vox_mat: u8| match vox_mat {
            SNOWY_BRICK => [0, 1, 2],
            FULL_BRICK => [2, 2, 2],
            GRASS | _ => [3, 3, 3],
        })
    }

    fn voxel_texture(&self) -> Option<(String, u32)> {
        Some(("example_voxel_texture.png".into(), 4))
    }
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        // We can specify a custom texture when initializing the plugin.
        // This should just be a path to an image in your assets folder.
        .add_plugins(VoxelWorldPlugin::with_config(MyMainWorld))
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
    // Cursor cube
    commands.spawn((
        Transform::from_xyz(0.0, -10.0, 0.0),
        MeshMaterial3d(materials.add(Color::srgba_u8(124, 144, 255, 128))),
        Mesh3d(meshes.add(Mesh::from(Cuboid {
            half_size: Vec3::splat(0.5),
        }))),
        CursorCube {
            voxel_pos: IVec3::new(0, -10, 0),
        },
    ));

    // Camera
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(10.0, 10.0, 10.0).looking_at(Vec3::ZERO, Vec3::Y),
        // This tells bevy_voxel_world to use this cameras transform to calculate spawning area
        VoxelWorldCamera::<MyMainWorld>::default(),
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
            // Move the cursor cube to the position of the voxel we hit
            // Camera is by construction not in a solid voxel, so result.normal must be Some(...)
            let voxel_pos = result.position + result.normal.unwrap();
            transform.translation = voxel_pos + Vec3::new(0.5, 0.5, 0.5);
            cursor_cube.voxel_pos = voxel_pos.as_ivec3();
        }
    }
}

fn mouse_button_input(
    buttons: Res<ButtonInput<MouseButton>>,
    mut voxel_world: VoxelWorld<MyMainWorld>,
    cursor_cube: Query<&CursorCube>,
) {
    if buttons.just_pressed(MouseButton::Left) {
        let vox_pos = cursor_cube.single().voxel_pos;
        voxel_world.set_voxel(vox_pos, WorldVoxel::Solid(FULL_BRICK));
    }
}
