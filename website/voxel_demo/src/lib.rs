//! The live demo on the website: the `light_mountain_house` example, running in
//! a canvas on the page.
//!
//! The scene and its controls come from the example's `MountainHousePlugin`.
//! This crate adds the window and asset settings a page needs, and tells the
//! page when the demo starts drawing and whether its canvas is on screen.

use std::{
    sync::atomic::{AtomicBool, Ordering},
    time::Duration,
};

use bevy::{
    asset::AssetMetaCheck,
    log::{Level, LogPlugin},
    prelude::*,
    winit::{UpdateMode, WinitSettings},
};
use wasm_bindgen::prelude::*;

// The example's `main` goes unused here.
#[allow(dead_code)]
#[path = "../../../examples/light_mountain_house.rs"]
mod light_mountain_house;

use light_mountain_house::MountainHousePlugin;

#[wasm_bindgen(inline_js = "export function announce_ready() {
    window.dispatchEvent(new Event('voxel-demo-ready'));
}")]
extern "C" {
    /// Tells the page that the demo has started drawing, so it can show the canvas.
    fn announce_ready();
}

static CANVAS_VISIBLE: AtomicBool = AtomicBool::new(true);

/// Tells the demo whether its canvas is on screen.
///
/// Off screen, the demo updates once a second, which leaves the main thread to
/// the rest of the page.
#[wasm_bindgen]
pub fn set_canvas_visible(visible: bool) {
    CANVAS_VISIBLE.store(visible, Ordering::Relaxed);
}

#[wasm_bindgen(start)]
fn start() {
    App::new()
        .add_plugins(
            DefaultPlugins
                .set(ImagePlugin::default_nearest())
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        canvas: Some("#voxel-demo".into()),
                        fit_canvas_to_parent: true,
                        // A focused canvas would scroll the page to itself on startup.
                        focused: false,
                        // Keeps Space from scrolling the page and a right click
                        // from opening the browser's menu while the canvas has
                        // focus.
                        prevent_default_event_handling: true,
                        ..default()
                    }),
                    // The mouse stays free until a click on the canvas captures it.
                    primary_cursor_options: Some(default()),
                    ..default()
                })
                .set(AssetPlugin {
                    // The voxel texture sits beside the demo's WebAssembly, and
                    // the page is one folder down from it.
                    file_path: "../voxel_demo".into(),
                    // No asset has a `.meta` file, so asking for them only adds
                    // failed requests.
                    meta_check: AssetMetaCheck::Never,
                    ..default()
                })
                .set(LogPlugin {
                    level: Level::ERROR,
                    ..default()
                }),
        )
        .add_plugins(MountainHousePlugin)
        .add_systems(Update, announce_ready.run_if(run_once))
        .add_systems(Update, follow_canvas_visibility)
        .run();
}

fn follow_canvas_visibility(mut settings: ResMut<WinitSettings>) {
    let mode = if CANVAS_VISIBLE.load(Ordering::Relaxed) {
        UpdateMode::Continuous
    } else {
        UpdateMode::reactive_low_power(Duration::from_secs(1))
    };
    settings.focused_mode = mode;
    settings.unfocused_mode = mode;
}
