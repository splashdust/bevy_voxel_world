//! Touch controls
//!
//! This module lets a phone or tablet drive the demo through the controls that
//! the page draws over the canvas. The page reports each touch through the
//! functions this module exports. `apply_touch` turns what the page reported
//! into the keyboard and mouse input that the example listens for, so the
//! example walks, looks around and edits voxels just as it does for a keyboard
//! and a mouse. The example itself knows nothing of touch.
//!
//! The example only turns the camera and edits voxels while the mouse is
//! captured, and a touch screen has no mouse to capture. So while touch drives
//! the demo, `apply_touch` marks the cursor captured without telling the
//! window, and `keep_cursor_free` stops the window from capturing the real
//! cursor when the example asks it to. A real capture would fail on a phone,
//! and on a laptop with a touch screen it would lock the mouse.

use std::{collections::HashSet, hash::Hash, sync::Mutex};

use bevy::{
    input::{mouse::AccumulatedMouseMotion, InputSystems},
    prelude::*,
    window::{CursorGrabMode, CursorOptions, PrimaryWindow},
};
use wasm_bindgen::prelude::*;

/// Adds the systems that drive the example from the controls on the page.
pub struct TouchControlsPlugin;

impl Plugin for TouchControlsPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(PreUpdate, apply_touch.after(InputSystems))
            .add_systems(PostUpdate, keep_cursor_free);
    }
}

/// A key or mouse button that the example listens for.
#[derive(Clone, Copy)]
enum Input {
    Key(KeyCode),
    Mouse(MouseButton),
}

/// The buttons on the page, by the name the page gives them, with the key or
/// mouse button that each one presses.
const BUTTONS: [(&str, Input); 10] = [
    ("jump", Input::Key(KeyCode::Space)),
    ("place", Input::Mouse(MouseButton::Left)),
    ("remove", Input::Mouse(MouseButton::Right)),
    ("stone", Input::Key(KeyCode::Digit1)),
    ("timber", Input::Key(KeyCode::Digit2)),
    ("lantern", Input::Key(KeyCode::Digit3)),
    ("crystal", Input::Key(KeyCode::Digit4)),
    ("sun_down", Input::Key(KeyCode::BracketLeft)),
    ("sun_up", Input::Key(KeyCode::BracketRight)),
    ("carry_lantern", Input::Key(KeyCode::KeyL)),
];

/// The keys that the example walks with, and the direction each one walks in,
/// as right and forward.
const WALK_KEYS: [(KeyCode, Vec2); 4] = [
    (KeyCode::KeyW, Vec2::Y),
    (KeyCode::KeyS, Vec2::NEG_Y),
    (KeyCode::KeyA, Vec2::NEG_X),
    (KeyCode::KeyD, Vec2::X),
];

/// How close the walking direction must be to the direction of a walk key for
/// that key to be held, as the cosine of the angle between them.
///
/// The cosine of 67.5 degrees splits the circle into eight equal parts. A
/// direction within 22.5 degrees of forward holds W alone, and one between 22.5
/// and 67.5 degrees to the right holds W and D, which walks diagonally.
const WALK_KEY_COSINE: f32 = 0.38;

/// How far a drag of one CSS pixel moves the mouse, in the units of
/// `AccumulatedMouseMotion`.
///
/// The example turns the camera 0.003 radians for each unit of mouse motion, so
/// 2.2 turns it about 0.0066 radians for each pixel. A drag across a phone
/// screen 390 pixels wide turns the camera about 150 degrees.
const LOOK_SPEED: f32 = 2.2;

/// A button on the page.
#[derive(Clone, Copy)]
struct Button {
    held: bool,
    /// Whether the button went down since the last frame. A tap shorter than a
    /// frame goes down and up between two frames, and this keeps it from being
    /// lost.
    pressed: bool,
}

/// What the page has reported since the last frame.
struct Touch {
    /// Whether touch drives the demo, rather than a keyboard and a mouse.
    active: bool,
    /// The direction of the walking control, as right and forward.
    walk: Vec2,
    run: bool,
    /// The drag that the camera hasn't turned by yet, in CSS pixels.
    look: Vec2,
    buttons: [Button; BUTTONS.len()],
}

/// No control in use, and touch not driving the demo.
const IDLE: Touch = Touch {
    active: false,
    walk: Vec2::ZERO,
    run: false,
    look: Vec2::ZERO,
    buttons: [Button {
        held: false,
        pressed: false,
    }; BUTTONS.len()],
};

/// The page writes this between frames and `apply_touch` reads it in each
/// frame. Both run on the one thread of the page, so the lock never waits.
static TOUCH: Mutex<Touch> = Mutex::new(IDLE);

/// Tells the demo whether touch drives it, rather than a keyboard and a mouse.
///
/// A change lets go of every control. The page hides its touch controls when a
/// mouse is pressed, and a control under a finger at that moment never
/// reports that it was let go.
#[wasm_bindgen]
pub fn set_touch_controls(active: bool) {
    let mut touch = TOUCH.lock().unwrap();
    if touch.active != active {
        *touch = Touch { active, ..IDLE };
    }
}

/// Points the walking control `right` and `forward`, each from -1 to 1, and
/// runs instead of walking while `run` is true. Zero for both stands still.
#[wasm_bindgen]
pub fn touch_walk(right: f32, forward: f32, run: bool) {
    let mut touch = TOUCH.lock().unwrap();
    touch.walk = Vec2::new(right, forward);
    touch.run = run;
}

/// Turns the camera by a drag of `dx` and `dy` CSS pixels.
#[wasm_bindgen]
pub fn touch_look(dx: f32, dy: f32) {
    TOUCH.lock().unwrap().look += Vec2::new(dx, dy);
}

/// Holds or releases the button that the page calls `name`, one of the names
/// in `BUTTONS`. The demo ignores a name it doesn't know.
#[wasm_bindgen]
pub fn touch_button(name: &str, held: bool) {
    let mut touch = TOUCH.lock().unwrap();
    let Some(index) = BUTTONS.iter().position(|(button, _)| *button == name) else {
        return;
    };
    let button = &mut touch.buttons[index];
    button.held = held;
    button.pressed |= held;
}

/// The keys and mouse buttons that `apply_touch` holds, so that it releases
/// only those, and never a key held on a real keyboard.
#[derive(Default)]
struct Held {
    keys: HashSet<KeyCode>,
    mouse: HashSet<MouseButton>,
    /// Whether touch drove the demo in the last frame.
    active: bool,
}

/// Presses and releases the keys and mouse buttons that the touch controls
/// stand for, adds their drag to the mouse motion, and marks the cursor
/// captured while touch drives the demo.
///
/// It runs right after Bevy reads the real keyboard and mouse in each frame,
/// so that the example sees both in the same frame.
///
/// The help text of the example names keys and mouse buttons, and covers much
/// of a phone screen. So when touch starts to drive the demo, this taps H to
/// hide the help text, and it taps H again to show the help text once a mouse
/// takes over.
fn apply_touch(
    mut keys: ResMut<ButtonInput<KeyCode>>,
    mut mouse: ResMut<ButtonInput<MouseButton>>,
    mut motion: ResMut<AccumulatedMouseMotion>,
    mut cursor: Single<&mut CursorOptions, With<PrimaryWindow>>,
    mut held: Local<Held>,
) {
    let mut touch = TOUCH.lock().unwrap();
    let Held {
        keys: held_keys,
        mouse: held_mouse,
        active: was_active,
    } = &mut *held;
    let active = touch.active;
    let started_or_stopped = active != *was_active;
    *was_active = active;
    hold(
        &mut keys,
        held_keys,
        KeyCode::KeyH,
        false,
        started_or_stopped,
    );

    if active {
        cursor.bypass_change_detection().grab_mode = CursorGrabMode::Locked;
    } else if started_or_stopped {
        cursor.bypass_change_detection().grab_mode = CursorGrabMode::None;
    }

    let direction = touch.walk.normalize_or_zero();
    for (key, towards) in WALK_KEYS {
        let walking = active && direction.dot(towards) > WALK_KEY_COSINE;
        hold(&mut keys, held_keys, key, walking, false);
    }
    let running = active && touch.run && direction != Vec2::ZERO;
    hold(&mut keys, held_keys, KeyCode::ShiftLeft, running, false);

    for (button, (_, input)) in touch.buttons.iter_mut().zip(BUTTONS) {
        let (down, pressed) = (active && button.held, active && button.pressed);
        match input {
            Input::Key(key) => hold(&mut keys, held_keys, key, down, pressed),
            Input::Mouse(mouse_button) => {
                hold(&mut mouse, held_mouse, mouse_button, down, pressed)
            }
        }
        button.pressed = false;
    }

    if active {
        motion.delta += touch.look * LOOK_SPEED;
    }
    touch.look = Vec2::ZERO;
}

/// Holds `button` in `input` while `down` is true, and releases it once
/// `down` is false. `pressed` presses it again in this frame even while it is
/// still held, so that a second tap reads as a second press.
fn hold<T: Copy + Eq + Hash + Send + Sync + 'static>(
    input: &mut ButtonInput<T>,
    held: &mut HashSet<T>,
    button: T,
    down: bool,
    pressed: bool,
) {
    if pressed {
        input.release(button);
        input.press(button);
        held.insert(button);
    } else if down {
        if !input.pressed(button) {
            input.press(button);
            held.insert(button);
        }
    } else if held.remove(&button) {
        input.release(button);
    }
}

/// Frees the cursor again whenever the example captures it while touch drives
/// the demo, before the window sees the change.
///
/// The example captures the cursor on every left click, and the place button
/// is a left click. The window only captures or frees the cursor when the
/// cursor options change, so a change back to free keeps the real cursor free.
/// `apply_touch` marks the cursor captured again in the next frame, without a
/// change that the window would act on.
fn keep_cursor_free(mut cursor: Single<&mut CursorOptions, With<PrimaryWindow>>) {
    if TOUCH.lock().unwrap().active && cursor.is_changed() {
        **cursor = CursorOptions::default();
    }
}
