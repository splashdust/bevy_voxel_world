// Example of a 3D voxel world using Bevy Northstar
// left click to move player
// g to rebuild the navigation grid
// orbit camera via middle mouse
// scroll wheel to zoom

use bevy::{light::CascadeShadowConfigBuilder, prelude::*};
use bevy_northstar::prelude::*;
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
use bevy_voxel_world::custom_meshing::{
    generate_chunk_mesh, PaddedChunkShape, CHUNK_SIZE_U,
};
use bevy_voxel_world::prelude::*;
use ndshape::ConstShape;
use std::sync::Arc;
// Declare materials as consts for convenience
const SNOWY_BRICK: u8 = 0;
const FULL_BRICK: u8 = 1;
const GRASS: u8 = 2;

// Animation tuning
const LERP_SPEED: f32 = 20.0;
const POSITION_TOLERANCE: f32 = 0.01;
const FLOOR_SIZE: u32 = 64;
const PLAYER_OFFSET: f32 = 0.5;

#[derive(Resource, Clone, Default)]
struct MyMainWorld;

#[derive(Component, Clone)]
struct ChunkNav {
    passable: Vec<UVec3>,
}

#[derive(Bundle, Clone)]
struct ChunkNavBundle {
    nav: ChunkNav,
}

impl VoxelWorldConfig for MyMainWorld {
    type MaterialIndex = u8;
    type ChunkUserBundle = ChunkNavBundle;

    fn texture_index_mapper(
        &self,
    ) -> Arc<dyn Fn(Self::MaterialIndex) -> [u32; 3] + Send + Sync> {
        Arc::new(|vox_mat: u8| match vox_mat {
            SNOWY_BRICK => [0, 1, 2],
            FULL_BRICK => [2, 2, 2],
            GRASS => [3, 3, 3],
            _ => [3, 3, 3],
        })
    }

    fn voxel_lookup_delegate(&self) -> VoxelLookupDelegate<Self::MaterialIndex> {
        Box::new(move |_chunk_pos, _lod, _previous| create_voxel_floor())
    }

    fn chunk_meshing_delegate(
        &self,
    ) -> ChunkMeshingDelegate<Self::MaterialIndex, Self::ChunkUserBundle> {
        Some(Box::new(
            |pos: IVec3, _lod, _data_shape, _mesh_shape, _previous| {
                Box::new(
                    move |voxels,
                          _data_shape_in,
                          _mesh_shape_in,
                          texture_index_mapper| {
                        // Use the crate's default meshing to build the mesh.
                        let mesh = generate_chunk_mesh(
                            voxels.clone(),
                            pos,
                            texture_index_mapper,
                        );

                        // Build per-chunk navigation: cells are passable when Air above Solid.
                        let mut passable = Vec::new();
                        let base = (pos * CHUNK_SIZE_U as i32).as_uvec3();
                        for x in 0..CHUNK_SIZE_U {
                            for z in 0..CHUNK_SIZE_U {
                                for y in 0..CHUNK_SIZE_U {
                                    let above_local = UVec3::new(x + 1, y + 1, z + 1);
                                    let below_local = UVec3::new(x + 1, y, z + 1);
                                    let vox_above = voxels[PaddedChunkShape::linearize(
                                        above_local.to_array(),
                                    )
                                        as usize];
                                    let vox_below = voxels[PaddedChunkShape::linearize(
                                        below_local.to_array(),
                                    )
                                        as usize];
                                    if vox_above.is_air() && vox_below.is_solid() {
                                        passable.push(base + UVec3::new(x, y, z));
                                    }
                                }
                            }
                        }

                        (
                            mesh,
                            Some(ChunkNavBundle {
                                nav: ChunkNav { passable },
                            }),
                        )
                    },
                )
            },
        ))
    }
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        // We can specify a custom texture when initializing the plugin.
        // This should just be a path to an image in your assets folder.
        .add_plugins(VoxelWorldPlugin::with_config(MyMainWorld))
        .add_plugins(PanOrbitCameraPlugin)
        .add_message::<AnimationWaitEvent>()
        .add_systems(Startup, (setup, create_voxel_scene))
        .add_systems(PreUpdate, move_pathfinders)
        .add_systems(
            Update,
            (
                update_cursor_cube,
                mouse_button_input,
                player_input_3d,
                animate_move,
                pathfind_error,
                apply_chunk_nav_on_spawn,
            ),
        )
        .add_plugins(NorthstarPlugin::<OrdinalNeighborhood3d>::default())
        .add_plugins(NorthstarDebugPlugin::<OrdinalNeighborhood3d>::default())
        .run();
}

#[derive(Component)]
struct CursorCube {
    voxel_pos: IVec3,
}
// Player marker
#[derive(Component)]
pub struct Player;

// Event that lets other systems know to wait until animations are completed.
#[derive(Debug, Message)]
pub struct AnimationWaitEvent;

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
        Transform::from_xyz(-5.0, 16.0, -5.0)
            .looking_at(Vec3::new(16.0, 0.0, 16.0), Vec3::Y),
        // This tells bevy_voxel_world to use this cameras transform to calculate spawning area
        VoxelWorldCamera::<MyMainWorld>::default(),
        PanOrbitCamera {
            pan_sensitivity: 0.0,
            focus: Vec3::new(16.0, 0.0, 16.0),
            button_orbit: MouseButton::Middle,
            ..default()
        },
    ));

    // Sun
    let cascade_shadow_config = CascadeShadowConfigBuilder { ..default() }.build();
    commands.spawn((
        DirectionalLight {
            color: Color::srgb(0.98, 0.95, 0.82),
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(0.0, 0.0, 0.0)
            .looking_at(Vec3::new(-0.15, -0.1, 0.15), Vec3::Y),
        cascade_shadow_config,
    ));

    // Ambient light, same color as sun
    commands.insert_resource(GlobalAmbientLight {
        color: Color::srgb(0.98, 0.95, 0.82),
        brightness: 100.0,
        affects_lightmapped_meshes: true,
    });

    //Debug grid
    // Build the grid settings: cover a larger area and keep flat z-depth.
    // Adjust sizes as needed; this should roughly cover the visible/interactive terrain.
    let grid_settings = GridSettingsBuilder::new_3d(FLOOR_SIZE, 16, FLOOR_SIZE)
        .chunk_size(8)
        // For 2.5D you will likely want a chunk depth greater than 1.
        // This will allow short paths to use direct A* to create more natural paths to height changes.
        .chunk_depth(8)
        .enable_diagonal_connections()
        .default_impassable()
        // This is a great example of when to use a neighbor filter.
        // Since we're Y Sorting, we don't want to allow the player to move diagonally around walls as the sprite will z transition through the wall.
        // We use `NoCornerCuttingFlat` here instead of `NoCornerCutting` because we want to allow diagonal movement to other height levels.
        // .add_neighbor_filter(filter::NoCornerCutting)
        .build();
    // Call `build()` to return the component.

    let debug_grid = DebugGridBuilder::new(2, 2)
        .set_depth(2)
        .tilemap_type(DebugTilemapType::Square)
        .enable_chunks()
        .enable_entrances()
        .build();
    // Spawn the grid component
    let mut grid_ec = commands.spawn(OrdinalGrid3d::new(&grid_settings));
    grid_ec.with_child(debug_grid);
    let grid_entity = grid_ec.id();

    // player
    commands.spawn((
        Transform::from_xyz(
            9.0 + PLAYER_OFFSET,
            1.0 + PLAYER_OFFSET,
            9.0 + PLAYER_OFFSET,
        ),
        Player,
        Mesh3d(meshes.add(Cuboid::new(1.0, 1.0, 1.0))),
        MeshMaterial3d(materials.add(Color::srgb(1.0, 0.0, 0.0))),
        // Northstar agent setup
        AgentPos(UVec3::new(9, 1, 9)),
        AgentOfGrid(grid_entity),
        DebugPath::new(Color::srgb(1.0, 0.0, 0.0)),
    ));
}

fn create_voxel_scene(mut voxel_world: VoxelWorld<MyMainWorld>) {
    // Some bricks
    voxel_world.set_voxel(IVec3::new(16, 1, 16), WorldVoxel::Solid(SNOWY_BRICK));
    voxel_world.set_voxel(IVec3::new(17, 1, 16), WorldVoxel::Solid(SNOWY_BRICK));
    voxel_world.set_voxel(IVec3::new(16, 1, 17), WorldVoxel::Solid(SNOWY_BRICK));
    voxel_world.set_voxel(IVec3::new(16, 1, 15), WorldVoxel::Solid(SNOWY_BRICK));
    voxel_world.set_voxel(IVec3::new(15, 1, 16), WorldVoxel::Solid(FULL_BRICK));
    voxel_world.set_voxel(IVec3::new(14, 1, 16), WorldVoxel::Solid(FULL_BRICK));
    voxel_world.set_voxel(IVec3::new(15, 2, 16), WorldVoxel::Solid(SNOWY_BRICK));
    voxel_world.set_voxel(IVec3::new(14, 2, 16), WorldVoxel::Solid(SNOWY_BRICK));
    voxel_world.set_voxel(IVec3::new(16, 2, 16), WorldVoxel::Solid(SNOWY_BRICK));
}

fn create_voxel_floor(
) -> Box<dyn FnMut(IVec3, Option<WorldVoxel>) -> WorldVoxel + Send + Sync> {
    Box::new(move |pos: IVec3, _previous| {
        if pos.x > 0
            && pos.z > 0
            && pos.x < FLOOR_SIZE as i32
            && pos.z < FLOOR_SIZE as i32
        {
            if pos.y < 1 && pos.y > -3 {
                WorldVoxel::Solid(GRASS)
            } else {
                WorldVoxel::Air
            }
        } else {
            WorldVoxel::Unset
        }
    })
}

fn update_cursor_cube(
    voxel_world_raycast: VoxelWorld<MyMainWorld>,
    camera_info: Query<(&Camera, &GlobalTransform), With<VoxelWorldCamera<MyMainWorld>>>,
    mut cursor_evr: MessageReader<CursorMoved>,
    mut cursor_cube: Query<(
        &mut Transform,
        &mut CursorCube,
        &mut MeshMaterial3d<StandardMaterial>,
    )>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    for ev in cursor_evr.read() {
        // Get a ray from the cursor position into the world
        let (camera, cam_gtf) = camera_info.single().unwrap();
        let Ok(ray) = camera.viewport_to_world(cam_gtf, ev.position) else {
            return;
        };

        if let Some(result) = voxel_world_raycast.raycast(ray, &|(_pos, _vox)| true) {
            let (mut transform, mut cursor_cube, material_handle) =
                cursor_cube.single_mut().unwrap();
            // Move the cursor cube to the position of the voxel we hit
            // Camera is by construction not in a solid voxel, so result.normal must be Some(...)
            let voxel_pos = result.position + result.normal.unwrap();
            transform.translation = voxel_pos + Vec3::new(0.5, 0.5, 0.5);
            cursor_cube.voxel_pos = voxel_pos.as_ivec3();
            if cursor_cube.voxel_pos.x < 0
                || cursor_cube.voxel_pos.y < 0
                || cursor_cube.voxel_pos.z < 0
            {
                //indicate that this voxel position is not ment to spawn a cube
                if let Some(mat) = materials.get_mut(&material_handle.0) {
                    mat.base_color = Color::srgba_u8(255, 144, 124, 128);
                }
            } else if let Some(mat) = materials.get_mut(&material_handle.0) {
                mat.base_color = Color::srgba_u8(124, 144, 255, 128);
            }
        }
    }
}

fn mouse_button_input(
    buttons: Res<ButtonInput<MouseButton>>,
    mut voxel_world: VoxelWorld<MyMainWorld>,
    cursor_cube: Query<&CursorCube>,
    grid: Single<&mut OrdinalGrid3d>,
) {
    if buttons.just_pressed(MouseButton::Right) {
        let vox = cursor_cube.single().unwrap();
        //don't allow the player to spawn bricks that will not be navigatable with northstar
        if vox.voxel_pos.x < 0 || vox.voxel_pos.y < 0 || vox.voxel_pos.z < 0 {
            return;
        }

        voxel_world.set_voxel(vox.voxel_pos, WorldVoxel::Solid(FULL_BRICK));
        let mut grid = grid.into_inner();
        let above = vox.voxel_pos + IVec3::new(0, 1, 0);
        if grid.in_bounds(above.as_uvec3()) && voxel_world.get_voxel(above).is_air() {
            let above = above.as_uvec3();
            grid.set_nav(above, Nav::Passable(1));
            grid.set_nav(vox.voxel_pos.as_uvec3(), Nav::Impassable);
            grid.build();
        }
    }
}

// Handle click-to-move using the cursor cube's current voxel position
fn player_input_3d(
    buttons: Res<ButtonInput<MouseButton>>,
    player_q: Query<Entity, With<Player>>,
    cursor_q: Query<&CursorCube>,
    mut commands: Commands,
) {
    if buttons.just_pressed(MouseButton::Left) {
        if let (Ok(player), Ok(cursor)) = (player_q.single(), cursor_q.single()) {
            let pos_i = cursor.voxel_pos;

            // Disallow invalid negative targets
            if pos_i.x < 0 || pos_i.y < 0 || pos_i.z < 0 {
                return;
            }
            let pos = pos_i.as_uvec3();
            info!("Player movement target via click: {:?}", pos);
            commands
                .entity(player)
                .insert(Pathfind::new_3d(pos.x, pos.y, pos.z));
        }
    }
}

// Advance agent logical position along the computed path when not animating
fn move_pathfinders(
    mut commands: Commands,
    mut query: Query<(Entity, &mut AgentPos, &NextPos)>,
    animation_reader: MessageReader<AnimationWaitEvent>,
) {
    if !animation_reader.is_empty() {
        return;
    }

    for (entity, mut position, next) in query.iter_mut() {
        position.0 = next.0;
        commands.entity(entity).remove::<NextPos>();
    }
}

// Smoothly animate the player mesh toward the center of its current voxel
fn animate_move(
    mut query: Query<(&AgentPos, &mut Transform)>,
    time: Res<Time>,
    mut ev_wait: MessageWriter<AnimationWaitEvent>,
) {
    for (position, mut transform) in query.iter_mut() {
        let target = Vec3::new(
            position.0.x as f32 + PLAYER_OFFSET,
            position.0.y as f32 + PLAYER_OFFSET,
            position.0.z as f32 + PLAYER_OFFSET,
        );

        let d = (target - transform.translation).length();
        let animating = if d > POSITION_TOLERANCE {
            transform.translation = transform
                .translation
                .lerp(target, LERP_SPEED * time.delta_secs());
            true
        } else {
            transform.translation = target;
            false
        };

        if animating {
            ev_wait.write(AnimationWaitEvent);
        }
    }
}

// Handle pathfinding failures cleanly
fn pathfind_error(query: Query<Entity, With<PathfindingFailed>>, mut commands: Commands) {
    for entity in query.iter() {
        error!("Pathfinding failed for entity: {:?}", entity);
        commands
            .entity(entity)
            .remove::<PathfindingFailed>()
            .remove::<Pathfind>()
            .remove::<NextPos>();
    }
}

// Apply newly generated per-chunk navigation data to the grid when chunks are spawned/remeshed
fn apply_chunk_nav_on_spawn(
    mut grid: Single<&mut OrdinalGrid3d>,
    nav_q: Query<&ChunkNav>,
    mut ev_spawn: MessageReader<ChunkWillSpawn<MyMainWorld>>,
) {
    let mut changed = false;
    for evt in ev_spawn.read() {
        if let Ok(nav) = nav_q.get(evt.entity) {
            for &pos in &nav.passable {
                if grid.in_bounds(pos) {
                    grid.set_nav(pos, Nav::Passable(1));
                    changed = true;
                }
            }
        }
    }
    if changed {
        grid.build();
    }
}
