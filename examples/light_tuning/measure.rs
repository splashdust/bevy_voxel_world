//! The `measure` mode: renders offscreen, changes a few settings one step and
//! back once the world has settled, and prints how many frames and
//! milliseconds each change took to settle, at 60 frames a second.

use bevy::{
    camera::RenderTarget, platform::time::Instant, prelude::*,
    render::render_resource::TextureFormat, window::ExitCondition,
};
use bevy_voxel_world::prelude::*;

use crate::{
    add_lights, camera_transform,
    scene::Place,
    settings::{setting_named, Tuning, SETTINGS},
    TuningWorld,
};

/// The settings the `measure` mode changes one step and back, by label.
const MEASURED: [&str; 10] = [
    "Sky",
    "Sky reach",
    "Lamp reach",
    "Shape",
    "Narrow loss",
    "Loss per face",
    "Precision",
    "Slots",
    "Brightness",
    "Lantern level",
];

/// Frames the world must stay settled, with every chunk Ready, before a
/// measured change is over.
const QUIET_FRAMES: u32 = 30;

#[derive(Resource, Default)]
struct Measuring {
    frame: u32,
    /// The last frame on which light hadn't settled or a chunk wasn't Ready.
    last_busy: u32,
    /// How many changes have been made: each setting in `MEASURED` one step
    /// and then back.
    changes: usize,
    /// The frame and time the change being measured was made.
    started: Option<(u32, Instant)>,
    /// The time of the last busy frame.
    last_busy_at: Option<Instant>,
    /// The values before a setting was changed, to change it back to.
    before: Option<Tuning>,
}

pub fn measure() {
    App::new()
        .add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: None,
                    exit_condition: ExitCondition::DontExit,
                    ..default()
                })
                .disable::<bevy::winit::WinitPlugin>()
                .add(bevy::app::ScheduleRunnerPlugin::run_loop(
                    std::time::Duration::from_secs_f64(1.0 / 60.0),
                )),
        )
        .add_plugins(VoxelWorldPlugin::with_config(TuningWorld::default()))
        .init_resource::<Measuring>()
        .add_systems(Startup, setup_measure)
        .add_systems(Update, change_and_measure)
        .run();
}

fn setup_measure(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
    let image = Image::new_target_texture(1280, 720, TextureFormat::Rgba8UnormSrgb, None);
    commands.spawn((
        Camera3d::default(),
        RenderTarget::Image(images.add(image).into()),
        camera_transform(Place::Meadow),
        VoxelWorldCamera::<TuningWorld>::default(),
    ));
    add_lights(&mut commands);
}

/// Waits for the world to settle, then changes each measured setting one step
/// and back, reporting how long each change took to settle.
fn change_and_measure(
    mut measuring: ResMut<Measuring>,
    mut world: ResMut<TuningWorld>,
    chunks: Query<&Chunk<TuningWorld>>,
    voxel_world: VoxelWorld<TuningWorld>,
    mut exit: MessageWriter<AppExit>,
) {
    measuring.frame += 1;
    let frame = measuring.frame;
    let busy = !voxel_world.light_is_settled()
        || chunks.iter().any(|chunk| {
            voxel_world.chunk_state(chunk.key()) != Some(ChunkStatus::Ready)
        });
    if busy {
        measuring.last_busy = frame;
        measuring.last_busy_at = Some(Instant::now());
    }
    if frame - measuring.last_busy < QUIET_FRAMES {
        return;
    }
    let setting_of = |change: usize| &SETTINGS[setting_named(MEASURED[change / 2])];
    if let Some((started, at)) = measuring.started.take() {
        let setting = setting_of(measuring.changes - 1);
        let (frames, ms) = match measuring.last_busy_at {
            Some(busy_at) if measuring.last_busy >= started => (
                measuring.last_busy + 1 - started,
                (busy_at - at).as_secs_f64() * 1000.0,
            ),
            _ => (1, 0.0),
        };
        report!(
            "{} to {}: settled in {frames} frames, {ms:.0} ms, with {} chunks",
            setting.rust,
            setting.show(&world.tuning),
            chunks.iter().count()
        );
    } else if measuring.changes == 0 {
        report!(
            "measure: the world settled after {frame} frames, with {} chunks",
            chunks.iter().count()
        );
    }
    if measuring.changes == MEASURED.len() * 2 {
        exit.write(AppExit::Success);
        return;
    }
    if let Some(before) = measuring.before.take() {
        world.tuning = before;
    } else {
        let mut tuning = world.tuning.clone();
        setting_of(measuring.changes).nudge(&mut tuning);
        measuring.before = Some(std::mem::replace(&mut world.tuning, tuning));
    }
    measuring.changes += 1;
    measuring.started = Some((frame, Instant::now()));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_measured_setting_exists() {
        for label in MEASURED {
            setting_named(label);
        }
    }
}
