//! Printing the tuned config as Rust, naming only what differs from the
//! defaults, ready to paste into a world's `VoxelWorldConfig`.

use bevy::prelude::*;
use bevy_voxel_world::prelude::*;

use crate::{
    scene::{CRYSTAL, LANTERN},
    settings::{curves, preset_of, Tuning},
};

/// Each line of `text` indented by four spaces.
fn indent(text: &str) -> String {
    text.lines().map(|line| format!("    {line}\n")).collect()
}

/// A struct literal of `name` with `fields`, and the rest from its default.
fn literal(name: &str, fields: Vec<String>) -> String {
    if fields.is_empty() {
        return format!("{name}::default()");
    }
    let body: String = fields
        .iter()
        .map(|field| indent(&format!("{field},")))
        .collect();
    format!("{name} {{\n{body}    ..default()\n}}")
}

/// `name: value` for each field of `$value` that differs from `$default`,
/// with the value written by `$rust`.
macro_rules! differing {
    ($fields:ident, $value:expr, $default:expr, $($field:ident => $rust:expr),* $(,)?) => {
        $(
            if $value.$field != $default.$field {
                let rust: fn(&_) -> String = $rust;
                $fields.push(format!("{}: {}", stringify!($field), rust(&$value.$field)));
            }
        )*
    };
}

fn debug(value: &impl std::fmt::Debug) -> String {
    format!("{value:?}")
}

/// The Rust that makes `curve`, which must be one of [`curves`].
fn curve_rust(curve: &LevelCurve) -> String {
    let index =
        preset_of(curve).expect("every curve the panel picks is one of `curves()`");
    curves()[index].0.into()
}

fn sun_rust(sun: &SunGate) -> String {
    match sun {
        SunGate::ByLevel(curve) => format!("SunGate::ByLevel({})", curve_rust(curve)),
        direct => format!("SunGate::{direct:?}"),
    }
}

fn duration_rust(duration: &std::time::Duration) -> String {
    if duration.subsec_micros().is_multiple_of(1000) {
        format!("std::time::Duration::from_millis({})", duration.as_millis())
    } else {
        format!("std::time::Duration::from_micros({})", duration.as_micros())
    }
}

fn bytes_rust(bytes: &usize) -> String {
    const MIB: usize = 1024 * 1024;
    if bytes.is_multiple_of(MIB) {
        format!("{} * 1024 * 1024", bytes / MIB)
    } else {
        bytes.to_string()
    }
}

fn sky_fill_rust(fill: &Option<SkyFill>) -> String {
    match fill {
        None => "None".into(),
        Some(SkyFill::Paths(paths)) => {
            let default = PathFill::default();
            let mut fields = Vec::new();
            differing!(fields, paths, default,
                beside_open_sky => debug,
                whole_face_loss => debug,
                opening_power => debug,
            );
            format!("Some(SkyFill::Paths({}))", literal("PathFill", fields))
        }
        Some(SkyFill::Openness(openness)) => {
            let default = OpenFill::default();
            let mut fields = Vec::new();
            differing!(fields, openness, default,
                sun_at => debug,
                gain => debug,
                rays => debug,
                reach => debug,
            );
            format!("Some(SkyFill::Openness({}))", literal("OpenFill", fields))
        }
    }
}

fn look_rust(look: &LightLook) -> String {
    let default = LightLook::default();
    let mut fields = Vec::new();
    differing!(fields, look, default,
        smoothing => |smoothing| format!("LightSmoothing::{smoothing:?}"),
        sky_falloff => curve_rust,
        voxel_falloff => curve_rust,
        sun => sun_rust,
        voxel_brightness => debug,
        emissive_glow => debug,
        ao => |ao| format!("LightAo::{ao:?}"),
        minimum_brightness => debug,
    );
    literal("LightLook", fields)
}

/// `light` as a Rust expression, naming only the fields that differ from
/// the default. It compiles where the crate's prelude and Bevy's are in
/// scope.
pub fn light_rust(light: &LightPropagation) -> String {
    let default = LightPropagation::default();
    let mut fields = Vec::new();
    differing!(fields, light, default,
        shown => debug,
        sky => |sky| format!("Sky::{sky:?}"),
        sky_reach => debug,
        voxel_reach => debug,
        shape => |shape| format!("LightShape::{shape:?}"),
        sky_fill => sky_fill_rust,
        precision => |precision| format!("LightPrecision::{precision:?}"),
        slot_capacity => debug,
        look => look_rust,
        sky_record_cap => bytes_rust,
        wasm_relight_budget => duration_rust,
    );
    literal("LightPropagation", fields)
}

/// The two config methods a world pastes into its `VoxelWorldConfig`, and
/// the app's ambient light.
pub fn config_rust(tuning: &Tuning) -> String {
    let lamps = [(LANTERN, tuning.lantern), (CRYSTAL, tuning.crystal)]
        .map(|(material, level)| format!("({material}, Emission::level({level:?}))"))
        .join(", ");
    format!(
        "// In your world's `impl VoxelWorldConfig`:\n\
         fn light_propagation(&self) -> Option<LightPropagation> {{\n{}}}\n\n\
         fn emissive_materials(&self) -> Vec<(u8, Emission)> {{\n    vec![{lamps}]\n}}\n\n\
         // In your app: GlobalAmbientLight {{ brightness: {:?}, ..default() }}",
        indent(&format!("Some({})", light_rust(&tuning.light()))),
        tuning.ambient,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::settings::SETTINGS;

    /// An expression and its source text.
    macro_rules! code {
        ($expression:expr) => {
            ($expression, stringify!($expression))
        };
    }

    fn without_spaces(text: &str) -> String {
        text.chars().filter(|c| !c.is_whitespace()).collect()
    }

    #[test]
    fn a_curve_a_last_bit_off_a_preset_reads_as_that_preset() {
        let mut table = [0.0; 16];
        for (level, entry) in table.iter_mut().enumerate() {
            *entry = LevelCurve::gamma(2.2).at(level as f32) * (1.0 + f32::EPSILON);
        }
        assert_eq!(
            curve_rust(&LevelCurve::from_table(table)),
            "LevelCurve::gamma(2.2)"
        );
    }

    #[test]
    fn the_default_config_prints_as_its_default() {
        assert_eq!(
            light_rust(&LightPropagation::default()),
            "LightPropagation::default()"
        );
    }

    #[test]
    fn a_config_prints_as_the_rust_that_builds_it() {
        let (light, rust) = code!(LightPropagation {
            shown: false,
            sky: Sky::Ceiling { y: 128 },
            sky_reach: 16,
            shape: LightShape::Diamond,
            sky_fill: Some(SkyFill::Openness(OpenFill {
                sun_at: 0.9,
                rays: 16,
                ..default()
            })),
            precision: LightPrecision::Steps256,
            slot_capacity: 343,
            look: LightLook {
                smoothing: LightSmoothing::Stepped,
                sky_falloff: LevelCurve::gamma(2.2),
                sun: SunGate::ByLevel(LevelCurve::linear()),
                voxel_brightness: 2000.0,
                ao: LightAo::BaseColor,
                minimum_brightness: 0.05,
                ..default()
            },
            sky_record_cap: 4 * 1024 * 1024,
            wasm_relight_budget: std::time::Duration::from_millis(8),
            ..default()
        });
        assert_eq!(without_spaces(&light_rust(&light)), without_spaces(rust));
    }

    #[test]
    fn sky_fill_prints_each_model_and_off() {
        for (light, rust) in [
            code!(LightPropagation {
                sky_fill: None,
                ..default()
            }),
            code!(LightPropagation {
                sky_fill: Some(SkyFill::Paths(PathFill {
                    opening_power: 1.0,
                    ..default()
                })),
                ..default()
            }),
            code!(LightPropagation {
                sky_fill: Some(SkyFill::Openness(OpenFill::default())),
                ..default()
            }),
        ] {
            assert_eq!(without_spaces(&light_rust(&light)), without_spaces(rust));
        }
    }

    #[test]
    fn every_option_of_every_setting_prints() {
        for setting in SETTINGS.iter() {
            let mut tuning = Tuning::default();
            setting.nudge(&mut tuning);
            assert!(!config_rust(&tuning).is_empty());
            if let crate::settings::Control::Choice { options, set, .. } =
                &setting.control
            {
                for index in 0..options.len() {
                    set(&mut tuning, index);
                    assert!(!config_rust(&tuning).is_empty());
                }
            }
        }
    }
}
