//! Tests of changing a lit world's light settings while it runs, through
//! `ResMut` of its config: the world relights and settles on the reference for
//! the new settings, with no restart. Sky fill is off in all of them.

use std::sync::{Arc, Mutex, OnceLock};

use bevy::ecs::system::RunSystemOnce;
use bevy::log::tracing::field::{Field, Visit};
use bevy::log::tracing::{Event, Level, Subscriber};
use bevy::log::tracing_subscriber::layer::{Context, SubscriberExt};
use bevy::log::tracing_subscriber::{registry, Layer};
use bevy::prelude::*;

use super::edits::{assert_matches, cavern, relights_started, settle};
use super::reference::Reference;
use super::sky_light::{light_is_settled, sky_light};
use super::voxel_light::{app, LampWorld, DIM_LAMP, LAMP};
use crate::light::LightSettings;
use crate::light_bench::light_stats;
use crate::prelude::*;

/// A lamp set into the top of the stone of [`cavern`], lighting the air above
/// it, where every open voxel has direct sky.
const SURFACE_LAMP: IVec3 = IVec3::new(4, 39, 26);

/// [`cavern`] with a lamp on its sunlit surface.
fn scene(p: IVec3) -> WorldVoxel {
    if p == SURFACE_LAMP {
        WorldVoxel::Solid(LAMP)
    } else {
        cavern(p)
    }
}

/// The scene with the default sky light at reach 30, voxel light at reach 14,
/// and its chunks streamed far enough out that the chunk above the origin has
/// its whole box loaded.
fn world() -> LampWorld {
    LampWorld {
        scene: Arc::new(scene),
        light: Some(sky_light(Sky::default(), 30)),
        spawning_distance: 3,
        ..default()
    }
}

/// The origin chunk, under the cavern's roof, and the chunk above it, which
/// holds the sunlit surface and its lamp.
const CHUNKS: [IVec3; 2] = [IVec3::ZERO, IVec3::Y];

/// The reference sky light and voxel light of the scene under `world`'s light
/// settings and emissive list, in a box reaching far enough past `CHUNKS`
/// that light from outside it can't reach them.
fn reference(world: &LampWorld) -> (Reference, Reference) {
    let light = world.light.as_ref().unwrap();
    let top = match light.precision {
        LightPrecision::Steps16 => 15,
        LightPrecision::Steps256 => 255,
    };
    let margin = IVec3::splat(32);
    let min = IVec3::ZERO - margin;
    let max = IVec3::new(32, 64, 32) + margin;
    let sky = Reference::sky_light_in_steps(
        top,
        min,
        max,
        light.sky_reach.clamp(1, 32) as u32,
        light.shape,
        &light.sky,
        scene,
    );
    let emissive = world.emissive.clone();
    let voxel = Reference::voxel_light_in_steps(
        top,
        min,
        max,
        light.voxel_reach.clamp(1, 32) as u32,
        light.shape,
        scene,
        move |material| {
            emissive
                .iter()
                .find(|(emissive, _)| *emissive == material)
                .map(|(_, emission)| emission.whole_level() as f32)
        },
    );
    (sky, voxel)
}

/// Changes the world's config through `ResMut`, as a game's system would.
fn change(app: &mut App, change: fn(&mut LampWorld)) {
    app.world_mut()
        .run_system_once(move |mut world: ResMut<LampWorld>| change(&mut world))
        .unwrap();
}

/// Fails unless every voxel of `chunks` reads the reference light for the
/// world's current settings.
fn matches_reference(app: &mut App, chunks: &[IVec3], when: &str) {
    let reference = reference(app.world().resource::<LampWorld>());
    assert_matches(app, chunks, &reference, when);
}

/// Streams `world` and lets its light settle on the reference, then makes
/// each change in `changes` in turn. After each, light must stop being
/// settled, relights must start, and light must settle on the reference for
/// the world's new settings. Returns the app.
fn changes_settle_on_the_reference(
    world: LampWorld,
    changes: &[fn(&mut LampWorld)],
) -> App {
    let mut app = app(world);
    settle(&mut app, "the world to stream in");
    matches_reference(&mut app, &CHUNKS, "before the change");
    for (index, edit) in changes.iter().enumerate() {
        let relights = relights_started(&app);
        change(&mut app, *edit);
        app.update();
        assert!(!light_is_settled(&mut app), "right after change {index}");
        settle(&mut app, "light to settle after the change");
        assert!(
            relights_started(&app) > relights,
            "change {index} relit nothing"
        );
        matches_reference(&mut app, &CHUNKS, &format!("after change {index}"));
    }
    app
}

fn light(world: &mut LampWorld) -> &mut LightPropagation {
    world.light.as_mut().unwrap()
}

#[test]
fn changing_precision_at_runtime_settles_on_the_references_light_for_the_new_precision() {
    let _app = changes_settle_on_the_reference(
        world(),
        &[
            |world| light(world).precision = LightPrecision::Steps256,
            |world| light(world).precision = LightPrecision::Steps16,
        ],
    );
}

#[test]
fn changing_slot_capacity_at_runtime_settles_on_the_references_light() {
    let _app = changes_settle_on_the_reference(
        world(),
        &[
            |world| light(world).slot_capacity = 4,
            |world| light(world).slot_capacity = 1024,
        ],
    );
}

#[test]
fn after_streaming_settles_at_256_levels_every_fully_loaded_chunk_matches_the_reference()
{
    let mut world = world();
    light(&mut world).precision = LightPrecision::Steps256;
    let reference = reference(&world);
    let mut app = super::voxel_light::app(world);
    settle(&mut app, "the world to stream in");
    assert_matches(&mut app, &CHUNKS, &reference, "at 256 levels");
}

#[test]
fn light_at_at_256_levels_reads_on_the_same_0_to_15_scale() {
    for precision in [LightPrecision::Steps16, LightPrecision::Steps256] {
        let mut world = world();
        light(&mut world).precision = precision;
        let mut app = super::voxel_light::app(world);
        settle(&mut app, "the world to stream in");
        let per_level = match precision {
            LightPrecision::Steps16 => 1.0,
            LightPrecision::Steps256 => 17.0,
        };
        let mut fractions = 0;
        for chunk in CHUNKS {
            let light = super::sky_light::chunk_light(&mut app, chunk).unwrap();
            for (p, levels) in light {
                for level in [levels.sky, levels.voxel] {
                    let steps = level * per_level;
                    assert!(
                        (steps - steps.round()).abs() < 1e-3
                            && (0.0..=15.0).contains(&level),
                        "{level} at {p} isn't a step of {precision:?}"
                    );
                    fractions += (level.fract() != 0.0) as u32;
                }
                assert_eq!(levels.direct_sky(), levels.sky == 15.0, "at {p}");
            }
        }
        // The lamp's own voxel reads its declared level, and the voxel beside
        // direct sky reads 14 at 16 levels and 247/17 at 256.
        let lamp = super::sky_light::levels(&mut app, SURFACE_LAMP);
        assert_eq!(lamp.voxel, 15.0);
        match precision {
            LightPrecision::Steps16 => assert_eq!(fractions, 0),
            LightPrecision::Steps256 => assert!(fractions > 0),
        }
    }
}

#[test]
fn changing_sky_reach_settles_on_the_reference_for_the_new_reach_with_no_restart() {
    let _app =
        changes_settle_on_the_reference(world(), &[|world| light(world).sky_reach = 16]);
}

#[test]
fn changing_voxel_reach_settles_on_the_reference_for_the_new_reach_with_no_restart() {
    let _app =
        changes_settle_on_the_reference(world(), &[|world| light(world).voxel_reach = 8]);
}

#[test]
fn changing_shape_settles_on_the_reference_for_the_new_shape_with_no_restart() {
    let _app = changes_settle_on_the_reference(
        world(),
        &[|world| light(world).shape = LightShape::Diamond],
    );
}

#[test]
fn changing_the_sky_rule_settles_on_the_reference_for_the_new_rule_with_no_restart() {
    let mut app = changes_settle_on_the_reference(
        world(),
        &[
            |world| light(world).sky = Sky::Off,
            |world| light(world).sky = Sky::Ceiling { y: 100 },
        ],
    );

    // A change from one sky to another keeps the record's masks.
    let entries = light_stats::<LampWorld>(app.world())
        .unwrap()
        .record_entries;
    change(&mut app, |world| {
        light(world).sky = Sky::Clearance { chunks: 2 }
    });
    app.update();
    let kept = light_stats::<LampWorld>(app.world())
        .unwrap()
        .record_entries;
    assert!(
        kept >= entries,
        "the record went from {entries} to {kept} entries"
    );
    settle(&mut app, "light to settle after the change");
    matches_reference(&mut app, &CHUNKS, "after the change to a clearance");
}

#[test]
fn changing_the_emissive_list_settles_on_the_reference_for_the_new_list_with_no_restart()
{
    let _app = changes_settle_on_the_reference(
        world(),
        &[
            |world| world.emissive = vec![(LAMP, Emission::level(8.0))],
            |world| {
                world.emissive = vec![
                    (LAMP, Emission::level(15.0)),
                    (DIM_LAMP, Emission::level(8.0)),
                ]
            },
        ],
    );
}

#[test]
fn a_lamp_on_a_sunlit_surface_relights_after_a_change_of_voxel_reach() {
    let mut app = app(world());
    settle(&mut app, "the world to stream in");
    let above = SURFACE_LAMP + IVec3::new(0, 6, 0);
    let before = super::sky_light::levels(&mut app, above);
    assert_eq!(before.sky, 15.0);
    assert!(before.voxel > 0.0);

    change(&mut app, |world| light(world).voxel_reach = 4);
    settle(&mut app, "light to settle after the change");
    let after = super::sky_light::levels(&mut app, above);
    assert_eq!(after.sky, 15.0);
    assert_eq!(
        after.voxel, 0.0,
        "the lamp's light reaches past its new reach"
    );
    matches_reference(&mut app, &[IVec3::Y], "after");
}

#[test]
fn setting_a_value_equal_to_the_current_one_starts_no_relight() {
    let mut app = app(world());
    settle(&mut app, "the world to stream in");
    let relights = relights_started(&app);
    // Each sets every field to what it was, which still marks the config
    // changed.
    change(&mut app, |world| {
        world.light = world.light.clone();
        world.emissive = world.emissive.clone();
    });
    change(&mut app, |world| light(world).sky_reach = 30);
    for _ in 0..5 {
        app.update();
        assert!(light_is_settled(&mut app));
    }
    assert_eq!(relights_started(&app), relights);
}

#[test]
fn sky_reach_with_sky_light_off_starts_no_relight() {
    let mut app = app(LampWorld {
        light: Some(sky_light(Sky::Off, 30)),
        ..world()
    });
    settle(&mut app, "the world to stream in");
    let relights = relights_started(&app);
    change(&mut app, |world| light(world).sky_reach = 12);
    for _ in 0..5 {
        app.update();
        assert!(light_is_settled(&mut app));
    }
    assert_eq!(relights_started(&app), relights);
}

/// The warnings logged anywhere in the test process since it started.
fn warnings() -> &'static Mutex<Vec<String>> {
    static WARNINGS: OnceLock<Mutex<Vec<String>>> = OnceLock::new();
    WARNINGS.get_or_init(|| {
        struct Warnings;
        impl<S: Subscriber> Layer<S> for Warnings {
            fn on_event(&self, event: &Event<'_>, _: Context<'_, S>) {
                if *event.metadata().level() != Level::WARN {
                    return;
                }
                struct Message(String);
                impl Visit for Message {
                    fn record_debug(
                        &mut self,
                        field: &Field,
                        value: &dyn std::fmt::Debug,
                    ) {
                        if field.name() == "message" {
                            self.0 = format!("{value:?}");
                        }
                    }
                }
                let mut message = Message(String::new());
                event.record(&mut message);
                warnings().lock().unwrap().push(message.0);
            }
        }
        // Only this test sets the global subscriber.
        bevy::log::tracing::subscriber::set_global_default(registry().with(Warnings))
            .expect("no other global subscriber");
        Mutex::new(Vec::new())
    })
}

fn warned(about: &str) -> usize {
    warnings()
        .lock()
        .unwrap()
        .iter()
        .filter(|message| message.contains(about))
        .count()
}

#[test]
fn changing_none_or_shown_logs_a_warning_once_and_changes_nothing() {
    warnings();
    let mut app = app(LampWorld {
        spawning_distance: 2,
        ..world()
    });
    settle(&mut app, "the world to stream in");
    let relights = relights_started(&app);
    let applied = app
        .world()
        .resource::<LightSettings<LampWorld>>()
        .config
        .clone();

    for _ in 0..2 {
        change(&mut app, |world| light(world).shown = false);
        app.update();
        change(&mut app, |world| light(world).shown = true);
        app.update();
    }
    for _ in 0..2 {
        change(&mut app, |world| world.light = None);
        app.update();
    }
    for _ in 0..5 {
        app.update();
        assert!(light_is_settled(&mut app));
    }
    assert_eq!(relights_started(&app), relights);
    let settings = app.world().resource::<LightSettings<LampWorld>>();
    assert_eq!(settings.config, applied);
    assert_eq!(warned("`shown` is read once"), 1);
    assert_eq!(warned("returned None after"), 1);
}

#[test]
fn an_out_of_range_reach_set_at_runtime_clamps() {
    let mut app = changes_settle_on_the_reference(
        world(),
        &[|world| {
            light(world).voxel_reach = 0;
            light(world).sky_reach = 200;
        }],
    );
    let config = &app.world().resource::<LightSettings<LampWorld>>().config;
    assert_eq!((config.voxel_reach, config.sky_reach), (1, 32));
    // A clamped value equal to the applied one changes nothing more.
    let relights = relights_started(&app);
    change(&mut app, |world| light(world).sky_reach = 250);
    for _ in 0..3 {
        app.update();
    }
    assert_eq!(relights_started(&app), relights);
}

#[test]
fn light_is_settled_goes_false_on_a_settings_change_and_true_once_everything_lands() {
    let mut app = app(LampWorld {
        spawning_distance: 2,
        ..world()
    });
    settle(&mut app, "the world to stream in");
    assert!(light_is_settled(&mut app));
    change(&mut app, |world| light(world).voxel_reach = 10);
    app.update();
    assert!(!light_is_settled(&mut app));
    let mut frames = 0;
    while !light_is_settled(&mut app) {
        app.update();
        frames += 1;
        assert!(frames < 2000, "light never settled");
    }
    assert!(relights_started(&app) > 0);
}
