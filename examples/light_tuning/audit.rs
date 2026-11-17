//! The `audit` mode: checks that every setting that relights changes the
//! light where the panel says it shows best.
//!
//! It renders offscreen and casts rays through each place's view to find the
//! open voxels in front of what the camera sees. Then it sets each
//! relighting setting to its lowest and highest value, or to each other
//! option, waits for the light to settle, and prints what share of its
//! place's voxels changed by a level or more, or in light colour by
//! [`COLOR_SHOWS`], and which place changed most. A setting changing less
//! than [`SHOWS`] of its place's view at every value is listed at the end:
//! the world or the place needs changing.
//!
//! Last, it checks that `skip_unreached_colors` changes no voxel's light.
//! The setting relights nothing, so it is turned off together with changes
//! of shape that relight every chunk, and the light under each shape must
//! then match a run with that shape and the setting on, at every voxel in
//! view.
//!
//! Levels change more than a picture does: a level of sky light in shade is
//! a small change of a dark colour. Look at a setting flagged as showing
//! before trusting it.
//!
//! Settings that apply only under one sky fill model are audited with that
//! model picked.
//!
//! The world's places are traced through the world's voxel lookup before
//! the app starts. The cabin is a sub-grid, which `raycast` and the voxel
//! lookup don't see, so its place is traced with `raycast_grids` once it has
//! spawned, and its open voxels are read through `grid(g).light_at` on the
//! cabin's grid.

use bevy::{
    camera::RenderTarget, platform::collections::HashSet, prelude::*,
    render::render_resource::TextureFormat, window::ExitCondition,
};
use bevy_voxel_world::prelude::*;

use crate::{
    add_lights, camera_transform,
    scene::{spawn_cabin, spawn_plinth, Cabin, Place, Scene},
    settings::{setting_named, Applies, Control, FillModel, Tuning, Value, SETTINGS},
    settling, Chunks, TuningWorld,
};

/// The share of a place's open voxels in view a setting must change by a
/// level to count as showing there. Some settings change one patch of a
/// view, such as the middle of the shade under the island, which is about 4%
/// of its voxels.
const SHOWS: f32 = 0.03;

/// How far one component of a voxel's light colour must move for the voxel
/// to count as changed, where its level doesn't.
const COLOR_SHOWS: f32 = 0.1;

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
    /// Whether this run's light must match its base's exactly, rather than
    /// show a change.
    same_light: bool,
}

/// The light at a voxel as the audit compares it: the sky and voxel levels,
/// and the voxel light's colour.
type Light = (f32, f32, LinearRgba);

#[derive(Resource)]
struct Auditing {
    frame: u32,
    last_busy: u32,
    runs: Vec<Run>,
    run: usize,
    /// Each place's open voxels in view, on the grid they are read from.
    /// The cabin's are added once it has spawned.
    samples: Vec<(Place, Grid, IVec3)>,
    /// The light at each sample, by run.
    levels: Vec<Vec<Light>>,
    /// The settings that barely show at their place, and the most of it
    /// they change.
    unseen: Vec<(usize, f32)>,
    /// The settings that must never change the light and did.
    changed_light: Vec<&'static str>,
}

pub fn audit() {
    let mut scene = Scene::default();
    let samples = Place::ALL
        .into_iter()
        .filter(|place| !matches!(place, Place::Overview | Place::Cabin))
        .flat_map(|place| {
            visible(&mut scene, place)
                .into_iter()
                .map(move |at| (place, Grid::World, at))
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
            changed_light: Vec::new(),
        })
        .add_systems(Startup, (setup_audit, spawn_cabin, spawn_plinth))
        .add_systems(Update, change_and_compare)
        .run();
}

/// The directions of rays through a 64 × 36 grid over `place`'s view, with
/// Bevy's default field of view at 16:9.
fn rays(place: Place) -> impl Iterator<Item = Vec3> {
    let camera = camera_transform(place);
    let (forward, right, up) = (camera.forward(), camera.right(), camera.up());
    let half_height = (std::f32::consts::FRAC_PI_4 / 2.0).tan();
    let half_width = half_height * 16.0 / 9.0;
    (0..36).flat_map(move |row| {
        (0..64).map(move |column| {
            let across = (column as f32 + 0.5) / 32.0 - 1.0;
            let down = 1.0 - (row as f32 + 0.5) / 18.0;
            (*forward + *right * across * half_width + *up * down * half_height)
                .normalize()
        })
    })
}

/// The open voxels in front of the first solid voxel of the world along each
/// of `place`'s [`rays`].
fn visible(scene: &mut Scene, place: Place) -> Vec<IVec3> {
    let eye = camera_transform(place).translation;
    let mut seen = HashSet::new();
    for ray in rays(place) {
        let mut open = eye.floor().as_ivec3();
        let mut distance = 0.0;
        while distance < 400.0 {
            let at = (eye + ray * distance).floor().as_ivec3();
            if at != open && scene.voxel(at).is_solid() {
                seen.insert(open);
                break;
            }
            open = at;
            distance += 0.2;
        }
    }
    seen.into_iter().collect()
}

/// The open voxels on the cabin's grid in front of the faces the cabin
/// place's [`rays`] hit first. Rays that miss the cabin add nothing.
fn visible_in_cabin(voxel_world: &VoxelWorld<TuningWorld>, cabin: Grid) -> Vec<IVec3> {
    let eye = camera_transform(Place::Cabin).translation;
    let mut seen = HashSet::new();
    for ray in rays(Place::Cabin) {
        let Ok(direction) = Dir3::new(ray) else {
            continue;
        };
        let hit = voxel_world.raycast_grids(
            Ray3d::new(eye, direction),
            400.0,
            &|grid| grid == cabin,
            &|_, _, _| true,
        );
        if let Some(open) =
            hit.and_then(|hit| hit.normal.map(|normal| hit.position + normal))
        {
            seen.insert(open);
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
            same_light: false,
        },
        Run {
            setting: None,
            tuning: openness.clone(),
            base: 1,
            same_light: false,
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
                same_light: false,
            });
        }
    }
    // A run with every chunk lit in diamonds, then the setting turned off
    // together with each change of shape, which relights every chunk: back
    // to round, which must match the default, and to diamonds again, which
    // must match the first.
    let diamond = Tuning {
        light: LightPropagation {
            shape: LightShape::Diamond,
            ..Tuning::default().light
        },
        ..default()
    };
    let skip_off = |tuning: &Tuning| {
        let mut off = tuning.clone();
        off.light.skip_unreached_colors = false;
        off
    };
    let skip = setting_named("Skip colours");
    let diamond_run = runs.len();
    runs.extend([
        Run {
            setting: None,
            tuning: diamond.clone(),
            base: diamond_run,
            same_light: false,
        },
        Run {
            setting: Some(skip),
            tuning: skip_off(&Tuning::default()),
            base: 0,
            same_light: true,
        },
        Run {
            setting: Some(skip),
            tuning: skip_off(&diamond),
            base: diamond_run,
            same_light: true,
        },
    ]);
    runs
}

/// Whether two readings of a voxel's light differ by a level, or by
/// [`COLOR_SHOWS`] in a component of its colour.
fn shows(a: Light, b: Light) -> bool {
    let colour = (a.2.to_vec3() - b.2.to_vec3()).abs().max_element();
    (a.0 - b.0).abs().max((a.1 - b.1).abs()) >= 1.0 || colour >= COLOR_SHOWS
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
    chunks: Chunks,
    voxel_world: VoxelWorld<TuningWorld>,
    cabin: Res<Cabin>,
    mut exit: MessageWriter<AppExit>,
) {
    auditing.frame += 1;
    let frame = auditing.frame;
    let busy = settling(&voxel_world, &chunks);
    if busy || auditing.frame == 1 {
        auditing.last_busy = frame;
    }
    if frame - auditing.last_busy < QUIET_FRAMES {
        return;
    }
    if auditing.run == 0 {
        let cabin = Grid::SubGrid(cabin.0);
        let in_cabin = visible_in_cabin(&voxel_world, cabin);
        auditing
            .samples
            .extend(in_cabin.into_iter().map(|at| (Place::Cabin, cabin, at)));
    }
    let levels: Vec<Light> = auditing
        .samples
        .iter()
        .map(|&(_, grid, at)| {
            voxel_world
                .grid(grid)
                .and_then(|grid| grid.light_at(at))
                .map_or((0.0, 0.0, LinearRgba::WHITE), |levels| {
                    (levels.sky, levels.voxel, levels.voxel_color)
                })
        })
        .collect();
    let run = &auditing.runs[auditing.run];
    if run.same_light {
        let base = &auditing.levels[run.base];
        let differ = levels
            .iter()
            .zip(base)
            .filter(|(light, base)| light != base)
            .count();
        let setting = &SETTINGS[run.setting.expect("a run checks a setting")];
        let shape = format!("{:?}", run.tuning.light.shape);
        if differ == 0 {
            report!(
                "audit: {} = {} under {shape}: the same light at all {} voxels",
                setting.label,
                setting.show(&run.tuning),
                levels.len()
            );
        } else {
            report!(
                "audit: {} = {} under {shape}: different light at {differ} of {} voxels",
                setting.label,
                setting.show(&run.tuning),
                levels.len()
            );
            auditing.changed_light.push(setting.label);
        }
    } else if let Some(index) = run.setting {
        let setting = &SETTINGS[index];
        let base = &auditing.levels[run.base];
        let share = |place: Place| {
            let (mut count, mut changed) = (0, 0);
            for (sample, &(at, _, _)) in auditing.samples.iter().enumerate() {
                if at != place {
                    continue;
                }
                count += 1;
                if shows(levels[sample], base[sample]) {
                    changed += 1;
                }
            }
            changed as f32 / count.max(1) as f32
        };
        let at_place = share(setting.place);
        // The setting's own place wins a tie, so another place is named
        // only where the setting changes more.
        let (most, most_share) = Place::ALL
            .into_iter()
            .filter(|place| *place != Place::Overview)
            .map(|place| (place, share(place)))
            .fold((setting.place, at_place), |most, next| {
                if next.1 > most.1 {
                    next
                } else {
                    most
                }
            });
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
        if !auditing.changed_light.is_empty() {
            report!(
                "audit: these changed the light, which they never should: {}",
                auditing.changed_light.join(", ")
            );
        }
        exit.write(AppExit::Success);
        return;
    }
    world.tuning = auditing.runs[auditing.run].tuning.clone();
    auditing.last_busy = frame;
}
