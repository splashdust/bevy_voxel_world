use std::{sync::Arc, time::Duration};

use bevy::{pbr::CascadeShadowConfigBuilder, prelude::*};
use bevy_voxel_world::prelude::*;

// Pretend we are loading for this duraction
const SIMULATED_LOADING_DURATION_MILLIS: u64 = 2000;

const SURFACE_Y: i32 = 0;
const MAP_DIMENSION: i32 = 128;
const MAP_HALF_DIMENSION: i32 = MAP_DIMENSION / 2;

#[derive(States, Hash, Clone, PartialEq, Eq, Debug, Default)]
pub enum LoadingState {
    #[default]
    Loading,
    Ready,
}

/// This timer will simulate the time it takes to load the game before generating any voxels
#[derive(Resource)]
struct SimulatedLoadingTimer(Timer);

#[derive(Resource, Clone, Default)]
struct MainWorld;

impl VoxelWorldConfig for MainWorld {
    type MaterialIndex = u8;
    type ChunkUserBundle = ();

    fn spawning_distance(&self) -> u32 {
        16
    }

    fn voxel_lookup_delegate(&self) -> VoxelLookupDelegate<Self::MaterialIndex> {
        Box::new(move |_chunk_pos| get_voxel_fn())
    }

    // Key element to this example: this defers execution of any
    // bevy_voxel_world systems until after this condition evaluates to true
    fn get_run_if_condition(&self) -> impl Condition<()> {
        in_state(LoadingState::Ready)
    }

    fn texture_index_mapper(
        &self,
    ) -> Arc<dyn Fn(Self::MaterialIndex) -> [u32; 3] + Send + Sync> {
        Arc::new(|mat| match mat {
            0 => [0, 0, 0],
            1 => [1, 1, 1],
            2 => [2, 2, 2],
            3 => [3, 3, 3],
            _ => [0, 0, 0],
        })
    }
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        // Initialize a loading state that we will wait on
        .init_state::<LoadingState>()
        // Setup a timer that we'll use to pretend we are loading
        .insert_resource(SimulatedLoadingTimer(Timer::new(
            Duration::from_millis(SIMULATED_LOADING_DURATION_MILLIS),
            TimerMode::Once,
        )))
        .add_plugins(VoxelWorldPlugin::with_config(MainWorld))
        .add_systems(Startup, setup)
        .add_systems(Update, move_camera)
        .add_systems(Update, simulate_loading_duration)
        .run();
}

fn simulate_loading_duration(
    time: Res<Time>,
    mut timer: ResMut<SimulatedLoadingTimer>,
    mut next_state: ResMut<NextState<LoadingState>>,
) {
    timer.0.tick(time.delta());

    if timer.0.finished() {
        next_state.set(LoadingState::Ready);
    }
}

fn setup(mut commands: Commands) {
    // camera
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(-200.0, 180.0, -200.0).looking_at(Vec3::ZERO, Vec3::Y),
        // This tells bevy_voxel_world to use this cameras transform to calculate spawning area
        VoxelWorldCamera::<MainWorld>::default(),
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
    commands.insert_resource(AmbientLight {
        color: Color::srgb(0.98, 0.95, 0.82),
        brightness: 100.0,
        affects_lightmapped_meshes: true,
    });
}

fn get_voxel_fn() -> Box<dyn FnMut(IVec3) -> WorldVoxel + Send + Sync> {
    Box::new(move |pos: IVec3| {
        if pos.y == SURFACE_Y {
            let tile_x = pos.x + MAP_HALF_DIMENSION;
            let tile_z = pos.z + MAP_HALF_DIMENSION;

            let material_index = if (tile_x + tile_z) % 2 == 0 { 0 } else { 1 };
            WorldVoxel::Solid(material_index)
        } else {
            WorldVoxel::Air
        }
    })
}

fn move_camera(
    time: Res<Time>,
    mut cam_transform: Query<&mut Transform, With<VoxelWorldCamera<MainWorld>>>,
) {
    let Ok(mut transform) = cam_transform.get_single_mut() else {
        return;
    };
    transform.translation.x += time.delta_secs() * 30.0;
    transform.translation.z += time.delta_secs() * 60.0;
}
