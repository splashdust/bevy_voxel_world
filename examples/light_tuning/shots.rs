//! The `shots` mode: renders offscreen and saves a picture of each place
//! where coloured light shows, to the directory given after `shots`.
//!
//! The deep chamber comes first, by default, with tinted glow and with
//! stepped smoothing, then the cave and the house as they look by default,
//! then the cave's lanterns in red. Last, it sets coloured crystals as voxel
//! edits under the floating island, under the arch and at the cave's mouth,
//! and shoots the pools they light there, with tinted glow so each crystal
//! shows its colour.

use std::path::PathBuf;

use bevy::{
    camera::RenderTarget,
    prelude::*,
    render::{
        render_resource::TextureFormat,
        view::screenshot::{save_to_disk, Screenshot, ScreenshotCaptured},
    },
    window::ExitCondition,
};
use bevy_voxel_world::prelude::*;

use crate::{
    add_lights,
    scene::{
        spawn_cabin, spawn_plinth, Place, Scene, BLUE_CRYSTAL, GREEN_CRYSTAL, RED_CRYSTAL,
    },
    settings::{Tuning, RED},
    settling, Chunks, TuningWorld,
};

/// Frames the world must stay settled before a shot is taken.
const QUIET_FRAMES: u32 = 60;

/// A picture to take: its file name, where the camera stands and looks, and
/// the tuning. Shots with `lamps` are taken with the extra crystals set.
struct Shot {
    name: &'static str,
    view: (Vec3, Vec3),
    tuning: Tuning,
    lamps: bool,
}

fn shots() -> Vec<Shot> {
    let tinted = {
        let mut tuning = Tuning::default();
        tuning.light.look.tinted_glow = true;
        tuning
    };
    let stepped = {
        let mut tuning = Tuning::default();
        tuning.light.look.smoothing = LightSmoothing::Stepped;
        tuning
    };
    let red_lanterns = Tuning {
        lantern_color: RED,
        ..default()
    };
    let floor = (Vec3::new(7.0, 42.0, -138.0), Vec3::new(7.0, 32.0, -150.0));
    let plinth = (Vec3::new(13.0, 38.0, -141.0), Vec3::new(7.5, 33.0, -149.0));
    let shot = |name, view, tuning: &Tuning, lamps| Shot {
        name,
        view,
        tuning: tuning.clone(),
        lamps,
    };
    let default = Tuning::default();
    vec![
        shot("chamber", Place::Chamber.view(), &default, false),
        shot("chamber_tinted", Place::Chamber.view(), &tinted, false),
        shot("chamber_floor", floor, &default, false),
        shot("chamber_floor_stepped", floor, &stepped, false),
        shot("plinth", plinth, &default, false),
        shot("plinth_tinted", plinth, &tinted, false),
        shot("cave", Place::Cave.view(), &default, false),
        shot("house", Place::House.view(), &default, false),
        shot(
            "cave_red_lanterns",
            Place::Cave.view(),
            &red_lanterns,
            false,
        ),
        shot(
            "cave_mouth_lamps",
            (Vec3::new(6.0, 39.0, -26.0), Vec3::new(-10.0, 33.0, -48.0)),
            &tinted,
            true,
        ),
        shot(
            "island_lamps",
            (Vec3::new(95.0, 42.0, 85.0), Vec3::new(68.0, 36.0, 54.0)),
            &tinted,
            true,
        ),
        shot(
            "arch_lamps",
            (Vec3::new(-70.0, 38.0, 42.0), Vec3::new(-88.0, 36.0, 22.0)),
            &tinted,
            true,
        ),
    ]
}

/// The extra crystals: on the ground under the floating island and hanging
/// from its underside, on the ground under the arch and hanging from its
/// top, and on the cave's floor just inside its mouth.
fn lamps() -> Vec<(IVec3, u8)> {
    let mut scene = Scene::default();
    let mut lamps = Vec::new();
    // Each looks down for the ground from below the island or the arch.
    for (x, z, below, material) in [
        (72, 58, 80, RED_CRYSTAL),
        (60, 50, 80, GREEN_CRYSTAL),
        (-88, 22, 38, BLUE_CRYSTAL),
    ] {
        let ground = scene.floor(x, z, below);
        lamps.push((IVec3::new(x, ground, z), material));
        // The first open voxel under what's solid above the ground.
        let under = (ground..200)
            .find(|y| scene.voxel(IVec3::new(x, y + 1, z)).is_solid())
            .expect("a roof above");
        let hanging = match material {
            RED_CRYSTAL => BLUE_CRYSTAL,
            GREEN_CRYSTAL => RED_CRYSTAL,
            _ => GREEN_CRYSTAL,
        };
        lamps.push((IVec3::new(x, under, z), hanging));
    }
    let mouth = scene.floor(-12, -46, 40);
    lamps.push((IVec3::new(-12, mouth, -46), BLUE_CRYSTAL));
    lamps
}

#[derive(Resource)]
struct Shooting {
    directory: PathBuf,
    shots: Vec<Shot>,
    shot: usize,
    frame: u32,
    last_busy: u32,
    /// Whether the shot being taken has been asked for, and whether it has
    /// been saved.
    asked: bool,
    saved: bool,
    lamps_set: bool,
    target: Handle<Image>,
}

pub fn shoot(directory: PathBuf) {
    std::fs::create_dir_all(&directory).expect("the shots directory can be made");
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
        .insert_resource(Shooting {
            directory,
            shots: shots(),
            shot: 0,
            frame: 0,
            last_busy: 0,
            asked: false,
            saved: false,
            lamps_set: false,
            target: Handle::default(),
        })
        .add_systems(Startup, (setup_shots, spawn_cabin, spawn_plinth))
        .add_systems(Update, take_shots)
        .run();
}

fn setup_shots(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut shooting: ResMut<Shooting>,
) {
    let image = Image::new_target_texture(1280, 720, TextureFormat::Rgba8UnormSrgb, None);
    shooting.target = images.add(image);
    let (eye, target) = shooting.shots[0].view;
    commands.spawn((
        Camera3d::default(),
        RenderTarget::Image(shooting.target.clone().into()),
        Transform::from_translation(eye).looking_at(target, Vec3::Y),
        VoxelWorldCamera::<TuningWorld>::default(),
    ));
    add_lights(&mut commands);
    commands.insert_resource(ClearColor(crate::SKY));
}

/// Sets each shot's tuning and view, waits for the light to settle, and
/// saves the picture.
#[allow(clippy::too_many_arguments)]
fn take_shots(
    mut commands: Commands,
    mut shooting: ResMut<Shooting>,
    mut world: ResMut<TuningWorld>,
    mut voxel_world: VoxelWorld<TuningWorld>,
    chunks: Chunks,
    mut cameras: Query<&mut Transform, With<Camera3d>>,
    mut exit: MessageWriter<AppExit>,
) {
    shooting.frame += 1;
    let frame = shooting.frame;
    if shooting.saved {
        shooting.saved = false;
        shooting.asked = false;
        shooting.shot += 1;
        if shooting.shot == shooting.shots.len() {
            exit.write(AppExit::Success);
            return;
        }
    }
    let shot = &shooting.shots[shooting.shot];
    let (name, (eye, target), lamps_wanted) = (shot.name, shot.view, shot.lamps);
    if world.tuning != shot.tuning {
        world.tuning = shot.tuning.clone();
    }
    for mut transform in &mut cameras {
        let view = Transform::from_translation(eye).looking_at(target, Vec3::Y);
        if *transform != view {
            *transform = view;
            shooting.last_busy = frame;
        }
    }
    if lamps_wanted && !shooting.lamps_set {
        for (at, material) in lamps() {
            voxel_world.set_voxel(at, WorldVoxel::Solid(material));
        }
        shooting.lamps_set = true;
    }
    if settling(&voxel_world, &chunks) || frame == 1 {
        shooting.last_busy = frame;
    }
    if frame - shooting.last_busy < QUIET_FRAMES || shooting.asked {
        return;
    }
    shooting.asked = true;
    let path = shooting.directory.join(format!("{name}.png"));
    commands
        .spawn(Screenshot::image(shooting.target.clone()))
        .observe(save_to_disk(path))
        .observe(
            |_: On<ScreenshotCaptured>, mut shooting: ResMut<Shooting>| {
                shooting.saved = true;
            },
        );
}
