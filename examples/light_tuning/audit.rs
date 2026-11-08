//! The `audit` mode: checks that every setting that relights changes the
//! light where the panel says it shows best.
//!
//! It renders offscreen and casts rays through each place's view to find the
//! open voxels in front of what the camera sees. Then it sets each relighting
//! setting to its lowest and highest value, or to each other option, waits
//! for the light to settle, and prints what share of its place's voxels
//! changed by a level or more, and which place changed most. A setting
//! changing less than [`SHOWS`] of its place's view at every value is listed
//! at the end: the world or the place needs changing.
//!
//! Levels change more than a picture does: a level of sky light in shade is
//! a small change of a dark colour. Look at a setting flagged as showing
//! before trusting it.
//!
//! Settings that apply only under one sky fill model are audited with that
//! model picked.

use bevy::{
    camera::RenderTarget, platform::collections::HashSet, prelude::*,
    render::render_resource::TextureFormat, window::ExitCondition,
};
use bevy_voxel_world::prelude::*;

use crate::{
    add_lights, camera_transform,
    scene::{Place, Scene},
    settings::{Applies, Control, FillModel, Tuning, Value, SETTINGS},
    TuningWorld,
};

/// The share of a place's open voxels in view a setting must change by a
/// level to count as showing there. Some settings change one patch of a
/// view, such as the middle of the shade under the island, which is about 4%
/// of its voxels.
const SHOWS: f32 = 0.03;

/// Frames the world must stay settled before its light is read.
const QUIET_FRAMES: u32 = 90;

/// A tuning to read the light under.
struct Run {
    /// The setting changed, by index in [`SETTINGS`], or `None` for a
    /// baseline.
    setting: Option<usize>,
    tuning: Tuning,
    /// The run whose light this one is compared with.
    base: usize,
}

#[derive(Resource)]
struct Auditing {
    frame: u32,
    last_busy: u32,
    runs: Vec<Run>,
    run: usize,
    /// Each place's open voxels in view.
    samples: Vec<(Place, IVec3)>,
    /// The sky and voxel levels at each sample, by run.
    levels: Vec<Vec<(f32, f32)>>,
    /// The settings that barely show at their place, and the most of it
    /// they change.
    unseen: Vec<(usize, f32)>,
}

pub fn audit() {
    let mut scene = Scene::default();
    let samples = Place::ALL
        .into_iter()
        .filter(|place| *place != Place::Overview)
        .flat_map(|place| {
            visible(&mut scene, place)
                .into_iter()
                .map(move |at| (place, at))
        })
        .collect();
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
        .insert_resource(Auditing {
            frame: 0,
            last_busy: 0,
            runs: runs(),
            run: 0,
            samples,
            levels: Vec::new(),
            unseen: Vec::new(),
        })
        .add_systems(Startup, setup_audit)
        .add_systems(Update, change_and_compare)
        .run();
}

/// The open voxels in front of the first solid voxel along rays through a
/// 64 × 36 grid over `place`'s view, with Bevy's default field of view at
/// 16:9.
fn visible(scene: &mut Scene, place: Place) -> Vec<IVec3> {
    let camera = camera_transform(place);
    let (forward, right, up) = (camera.forward(), camera.right(), camera.up());
    let half_height = (std::f32::consts::FRAC_PI_4 / 2.0).tan();
    let half_width = half_height * 16.0 / 9.0;
    let mut seen = HashSet::new();
    for row in 0..36 {
        for column in 0..64 {
            let across = (column as f32 + 0.5) / 32.0 - 1.0;
            let down = 1.0 - (row as f32 + 0.5) / 18.0;
            let ray =
                (*forward + *right * across * half_width + *up * down * half_height)
                    .normalize();
            let mut open = camera.translation.floor().as_ivec3();
            let mut distance = 0.0;
            while distance < 400.0 {
                let at = (camera.translation + ray * distance).floor().as_ivec3();
                if at != open && scene.voxel(at).is_solid() {
                    seen.insert(open);
                    break;
                }
                open = at;
                distance += 0.2;
            }
        }
    }
    seen.into_iter().collect()
}

/// The default tuning and the default under openness, then each relighting
/// setting at each value worth trying.
fn runs() -> Vec<Run> {
    let openness = Tuning {
        fill: FillModel::Openness,
        ..default()
    };
    let mut runs = vec![
        Run {
            setting: None,
            tuning: Tuning::default(),
            base: 0,
        },
        Run {
            setting: None,
            tuning: openness.clone(),
            base: 1,
        },
    ];
    for (index, setting) in SETTINGS.iter().enumerate() {
        if setting.applies != Applies::Relights {
            continue;
        }
        let (base, tuning) = if (setting.shown)(&Tuning::default()) {
            (0, Tuning::default())
        } else {
            (1, openness.clone())
        };
        let values: Vec<Value> = match &setting.control {
            Control::Slider { min, max, .. } => {
                vec![Value::Number(*min), Value::Number(*max)]
            }
            Control::Choice { options, get, .. } => (0..options.len())
                .filter(|option| *option != get(&tuning))
                .map(Value::Option)
                .collect(),
        };
        for value in values {
            let mut changed = tuning.clone();
            setting.set(&mut changed, value);
            runs.push(Run {
                setting: Some(index),
                tuning: changed,
                base,
            });
        }
    }
    runs
}

fn setup_audit(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
    let image = Image::new_target_texture(1280, 720, TextureFormat::Rgba8UnormSrgb, None);
    commands.spawn((
        Camera3d::default(),
        RenderTarget::Image(images.add(image).into()),
        camera_transform(Place::Meadow),
        VoxelWorldCamera::<TuningWorld>::default(),
    ));
    add_lights(&mut commands);
}

/// Waits for the light to settle under each run's tuning, then reads it and
/// reports how much each place's view changed from the run's baseline.
fn change_and_compare(
    mut auditing: ResMut<Auditing>,
    mut world: ResMut<TuningWorld>,
    chunks: Query<&Chunk<TuningWorld>>,
    voxel_world: VoxelWorld<TuningWorld>,
    mut exit: MessageWriter<AppExit>,
) {
    auditing.frame += 1;
    let frame = auditing.frame;
    let busy = !voxel_world.light_is_settled()
        || chunks.iter().any(|chunk| {
            voxel_world.chunk_state(chunk.key()) != Some(ChunkStatus::Ready)
        });
    if busy || auditing.frame == 1 {
        auditing.last_busy = frame;
    }
    if frame - auditing.last_busy < QUIET_FRAMES {
        return;
    }
    let levels: Vec<(f32, f32)> = auditing
        .samples
        .iter()
        .map(|&(_, at)| {
            voxel_world
                .light_at(at)
                .map_or((0.0, 0.0), |levels| (levels.sky, levels.voxel))
        })
        .collect();
    let run = &auditing.runs[auditing.run];
    if let Some(index) = run.setting {
        let setting = &SETTINGS[index];
        let base = &auditing.levels[run.base];
        let share = |place: Place| {
            let (mut count, mut changed) = (0, 0);
            for (sample, &(at, _)) in auditing.samples.iter().enumerate() {
                if at != place {
                    continue;
                }
                count += 1;
                let (sky, voxel) = levels[sample];
                let (base_sky, base_voxel) = base[sample];
                if (sky - base_sky).abs().max((voxel - base_voxel).abs()) >= 1.0 {
                    changed += 1;
                }
            }
            changed as f32 / count.max(1) as f32
        };
        let at_place = share(setting.place);
        let (most, most_share) = Place::ALL
            .into_iter()
            .filter(|place| *place != Place::Overview)
            .map(|place| (place, share(place)))
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .expect("there are places");
        report!(
            "audit: {} = {}: {:.0}% of {} changes; most at {} ({:.0}%)",
            setting.label,
            setting.show(&run.tuning),
            at_place * 100.0,
            setting.place.name(),
            most.name(),
            most_share * 100.0
        );
        let worst = auditing
            .unseen
            .iter()
            .position(|(unseen, _)| *unseen == index);
        match worst {
            Some(at) => auditing.unseen[at].1 = auditing.unseen[at].1.max(at_place),
            None => auditing.unseen.push((index, at_place)),
        }
    }
    auditing.levels.push(levels);
    auditing.run += 1;
    if auditing.run == auditing.runs.len() {
        let unseen: Vec<String> = auditing
            .unseen
            .iter()
            .filter(|(_, share)| *share < SHOWS)
            .map(|(index, share)| {
                let setting = &SETTINGS[*index];
                format!(
                    "{} at {} ({:.0}%)",
                    setting.label,
                    setting.place.name(),
                    share * 100.0
                )
            })
            .collect();
        if unseen.is_empty() {
            report!("audit: every relighting setting shows at its place");
        } else {
            report!(
                "audit: these barely show at their place: {}",
                unseen.join(", ")
            );
        }
        exit.write(AppExit::Success);
        return;
    }
    world.tuning = auditing.runs[auditing.run].tuning.clone();
    auditing.last_busy = frame;
}
