//! The settings panel on the right, the bar of places to look from at the
//! bottom, and keeping both in step with the values being tuned.
//!
//! The values in [`TuningWorld`] are the only state: a widget's change
//! writes them, and the `show_` systems write every widget back from them,
//! so resetting a setting or printing the config never disagrees with what
//! the panel shows.

use bevy::{
    feathers::{
        constants::icons,
        controls::{
            list_rows_from_strings, ButtonVariant, FeathersButton, FeathersSelect,
            FeathersSlider, FeathersToolButton, OptionIndex,
        },
        dark_theme::create_dark_theme,
        display::{caption, icon, label, label_dim, label_small},
        rounded_corners::RoundedCorners,
        theme::{ThemeBackgroundColor, UiTheme},
        tokens, FeathersPlugins,
    },
    picking::hover::Hovered,
    platform::time::Instant,
    prelude::*,
    ui::Selected,
    ui_widgets::{
        Activate, SetSelected, SliderPrecision, SliderStep, SliderValue, ValueChange,
    },
};
use bevy_panorbit_camera::PanOrbitCamera;
use bevy_voxel_world::prelude::*;

use crate::{
    move_camera, place_in_view,
    rust::config_rust,
    scene::Place,
    settings::{texture_mib, Applies, Control, Tab, Value, SETTINGS},
    TuningWorld,
};

pub struct PanelPlugin;

impl Plugin for PanelPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(FeathersPlugins)
            .insert_resource(UiTheme(create_dark_theme()))
            .init_resource::<Panel>()
            .add_systems(Startup, spawn_panel)
            .add_systems(
                Update,
                (
                    keys,
                    apply_dragged,
                    time_settling,
                    explain_hovered,
                    (show_sliders, show_options, show_changes)
                        .run_if(resource_changed::<TuningWorld>),
                    show_tab,
                    show_places,
                    show_texts,
                    block_camera,
                )
                    .chain(),
            )
            .add_observer(slider_moved)
            .add_observer(option_picked)
            .add_observer(segment_pressed)
            .add_observer(reset_pressed)
            .add_observer(tab_pressed)
            .add_observer(place_pressed)
            .add_observer(reset_all_pressed)
            .add_observer(print_pressed);
    }
}

/// What the panel shows, apart from the values themselves.
#[derive(Resource, Default)]
pub struct Panel {
    tab: Tab,
    /// The setting the pointer was last over, which the box under the
    /// settings explains.
    explained: Option<usize>,
    hidden: bool,
    /// The setting whose slider is being dragged, which `show_sliders` leaves
    /// alone.
    dragging: Option<usize>,
    /// A setting and the value its slider was dragged to, waiting for the
    /// last relight to settle.
    pending: Option<(usize, f32)>,
    /// The last change, and how long its light took to settle.
    change: Option<Change>,
    /// When the config was last printed.
    printed: Option<Instant>,
}

/// A change of a setting, and when its light settled.
struct Change {
    setting: usize,
    at: Instant,
    frames: u32,
    settled: Option<(f32, u32)>,
}

// --- Marker components ----------------------------------------------------------

/// The panel or the places bar: H hides them, and the camera doesn't follow
/// the mouse over them.
#[derive(Component, Default, Clone)]
struct Overlay;

/// The row of the setting at this index in [`SETTINGS`].
#[derive(Component, Default, Clone, Copy)]
struct RowOf(usize);

/// The dot that shows the setting differs from its default.
#[derive(Component, Default, Clone, Copy)]
struct DotOf(usize);

/// The button that resets the setting to its default.
#[derive(Component, Default, Clone, Copy)]
struct ResetOf(usize);

#[derive(Component, Default, Clone, Copy)]
struct SliderOf(usize);

#[derive(Component, Default, Clone, Copy)]
struct SelectOf(usize);

/// One of a setting's side-by-side option buttons: the setting, then the
/// option.
#[derive(Component, Default, Clone, Copy)]
struct SegmentOf(usize, usize);

/// A tab button, by index in [`Tab::ALL`].
#[derive(Component, Default, Clone, Copy)]
struct TabButton(usize);

/// A place button, by index in [`Place::ALL`].
#[derive(Component, Default, Clone, Copy)]
struct PlaceButton(usize);

/// The line under a place button, shown when the explained setting is best
/// seen from that place.
#[derive(Component, Default, Clone, Copy)]
struct HintOf(usize);

#[derive(Component, Default, Clone)]
struct ResetAllButton;

#[derive(Component, Default, Clone)]
struct PrintButton;

/// A text in the panel that changes, by what it shows.
#[derive(Component, Default, Clone, Copy, PartialEq, Debug)]
enum PanelText {
    /// What the chosen tab's settings are about.
    #[default]
    Subtitle,
    /// The light texture's GPU memory, on the Memory tab only.
    GpuMemory,
    /// The explained setting's name.
    Name,
    /// The field it sets, as written in Rust.
    Rust,
    /// What it does.
    About,
    /// Its default, and where to see it.
    Where,
    /// When a change to it shows.
    Applies,
    /// How long the last change took to settle.
    Status,
}

/// The coloured box behind [`PanelText::Applies`].
#[derive(Component, Default, Clone)]
struct Badge;

// --- Building the panel ---------------------------------------------------------

const PANEL_WIDTH: f32 = 330.0;
const LABEL_WIDTH: f32 = 96.0;
const CHANGED: Color = Color::srgb(0.98, 0.72, 0.25);

fn spawn_panel(mut commands: Commands, world: Res<TuningWorld>) {
    commands.spawn_scene(panel(&world.tuning));
    commands.spawn_scene(places_bar());
}

fn panel(tuning: &crate::settings::Tuning) -> impl Scene {
    let tabs: Vec<Box<dyn SceneList>> = Tab::ALL
        .iter()
        .enumerate()
        .map(|(index, tab)| -> Box<dyn SceneList> {
            let text = tab.label();
            let corners = segment_corners(index, Tab::ALL.len());
            let variant = ButtonVariant::selected(index == 0);
            bsn! {
                @FeathersButton {
                    @caption: bsn! { @caption(text) },
                    @corners: {corners},
                    @variant: {variant},
                }
                Node { flex_grow: 1.0 }
                TabButton(index)
            }
            .into()
        })
        .collect();
    let tabs: Box<dyn SceneList> = Box::new(tabs);
    let rows: Vec<Box<dyn SceneList>> = (0..SETTINGS.len())
        .map(|index| setting_row(index, tuning))
        .collect();
    let rows: Box<dyn SceneList> = Box::new(rows);
    let subtitle = Tab::Sky.subtitle();
    bsn! {
        Node {
            position_type: PositionType::Absolute,
            top: px(8),
            right: px(8),
            width: px(PANEL_WIDTH),
            flex_direction: FlexDirection::Column,
            row_gap: px(6),
            padding: px(10),
            border_radius: px(6),
        }
        ThemeBackgroundColor(tokens::WINDOW_BG)
        Overlay
        Hovered
        Children [
            Node {
                flex_direction: FlexDirection::Row,
                align_items: AlignItems::Center,
                justify_content: JustifyContent::SpaceBetween,
            }
            Children [
                @label("Light settings")
                --
                @label_dim("H hides the panel")
            ]
            --
            Node { flex_direction: FlexDirection::Row, column_gap: px(1) }
            Children [ {tabs} ]
            --
            @label_dim(subtitle) PanelText::Subtitle
            --
            Node { flex_direction: FlexDirection::Column, row_gap: px(4) }
            Children [ {rows} ]
            --
            @label_dim("") PanelText::GpuMemory
            --
            Node {
                flex_direction: FlexDirection::Column,
                row_gap: px(3),
                min_height: px(132),
                padding: UiRect::top(px(8)),
                border: UiRect::top(px(1)),
            }
            BorderColor::all(Color::srgba(1.0, 1.0, 1.0, 0.12))
            Children [
                Node {
                    flex_direction: FlexDirection::Row,
                    align_items: AlignItems::Center,
                    column_gap: px(8),
                }
                Children [
                    @label("") PanelText::Name
                    --
                    Node {
                        padding: UiRect::axes(px(6), px(1)),
                        border_radius: px(3),
                    }
                    BackgroundColor(Color::NONE)
                    Badge
                    Children [ @label_small("") PanelText::Applies ]
                ]
                --
                @label_dim("") PanelText::Rust
                --
                @label_dim("") PanelText::About
                --
                @label_small("") PanelText::Where
            ]
            --
            Node {
                flex_direction: FlexDirection::Row,
                column_gap: px(6),
                padding: UiRect::top(px(8)),
                border: UiRect::top(px(1)),
            }
            BorderColor::all(Color::srgba(1.0, 1.0, 1.0, 0.12))
            Children [
                @FeathersButton { @caption: bsn! { @caption("Reset all") } }
                Node { flex_grow: 1.0 }
                ResetAllButton
                --
                @FeathersButton {
                    @caption: bsn! { @caption("Print as Rust (P)") },
                    @variant: ButtonVariant::Primary,
                }
                Node { flex_grow: 1.0 }
                PrintButton
            ]
            --
            @label_small("") PanelText::Status
        ]
    }
}

/// Rounds the outer corners of a row of `count` buttons side by side.
fn segment_corners(index: usize, count: usize) -> RoundedCorners {
    match index {
        0 => RoundedCorners::Left,
        last if last + 1 == count => RoundedCorners::Right,
        _ => RoundedCorners::None,
    }
}

/// A setting's row: a dot when it differs from its default, its name, its
/// control and a button to reset it.
fn setting_row(index: usize, tuning: &crate::settings::Tuning) -> Box<dyn SceneList> {
    let setting = &SETTINGS[index];
    let name = setting.label;
    let control: Box<dyn SceneList> = match &setting.control {
        Control::Slider {
            min,
            max,
            step,
            decimals,
            get,
            ..
        } => {
            let (min, max, step, decimals, value) =
                (*min, *max, *step, *decimals, get(tuning));
            bsn! {
                @FeathersSlider { @min: {min}, @max: {max} }
                SliderValue(value)
                SliderStep(step)
                SliderPrecision(decimals)
                SliderOf(index)
            }
            .into()
        }
        Control::Choice { options, get, .. } if options.len() <= 3 => {
            let chosen = get(tuning);
            let buttons: Vec<Box<dyn SceneList>> = options
                .iter()
                .enumerate()
                .map(|(option, text)| -> Box<dyn SceneList> {
                    let text = *text;
                    let corners = segment_corners(option, options.len());
                    let variant = ButtonVariant::selected(option == chosen);
                    bsn! {
                        @FeathersButton {
                            @caption: bsn! { @caption(text) },
                            @corners: {corners},
                            @variant: {variant},
                        }
                        Node { flex_grow: 1.0 }
                        SegmentOf(index, option)
                    }
                    .into()
                })
                .collect();
            let buttons: Box<dyn SceneList> = Box::new(buttons);
            bsn! {
                Node { flex_grow: 1.0, column_gap: px(1) }
                Children [ {buttons} ]
            }
            .into()
        }
        Control::Choice { options, get, .. } => {
            let rows = list_rows_from_strings(options.iter(), Some(get(tuning)));
            bsn! {
                @FeathersSelect { @options: {rows} }
                Node { flex_grow: 1.0 }
                SelectOf(index)
            }
            .into()
        }
    };
    bsn! {
        Node {
            flex_direction: FlexDirection::Row,
            align_items: AlignItems::Center,
            column_gap: px(6),
        }
        RowOf(index)
        Hovered
        Children [
            Node {
                width: px(6),
                height: px(6),
                flex_shrink: 0.0,
                border_radius: BorderRadius::MAX,
            }
            BackgroundColor(Color::NONE)
            DotOf(index)
            --
            @label(name)
            Node { width: px(LABEL_WIDTH), flex_shrink: 0.0 }
            --
            Node { flex_grow: 1.0, flex_direction: FlexDirection::Row }
            Children [ {control} ]
            --
            @FeathersToolButton {
                @caption: bsn! { @icon(icons::X) },
                @variant: ButtonVariant::Plain,
            }
            Visibility::Hidden
            ResetOf(index)
        ]
    }
    .into()
}

fn places_bar() -> impl Scene {
    let buttons: Vec<Box<dyn SceneList>> = Place::ALL
        .iter()
        .enumerate()
        .map(|(index, place)| -> Box<dyn SceneList> {
            let text = format!("{} {}", place.key(), place.name());
            bsn! {
                Node { flex_direction: FlexDirection::Column, row_gap: px(3) }
                Children [
                    @FeathersButton { @caption: bsn! { @caption(text) } }
                    PlaceButton(index)
                    --
                    Node { height: px(3), border_radius: px(2) }
                    ThemeBackgroundColor(tokens::BUTTON_PRIMARY_BG)
                    Visibility::Hidden
                    HintOf(index)
                ]
            }
            .into()
        })
        .collect();
    let buttons: Box<dyn SceneList> = Box::new(buttons);
    bsn! {
        Node {
            position_type: PositionType::Absolute,
            left: px(8),
            bottom: px(8),
            flex_direction: FlexDirection::Column,
            row_gap: px(6),
            padding: px(8),
            border_radius: px(6),
        }
        ThemeBackgroundColor(tokens::WINDOW_BG)
        Overlay
        Hovered
        Children [
            @label_dim("Places to look from. Right-drag orbits, middle-drag pans, scroll zooms.")
            --
            Node { flex_direction: FlexDirection::Row, column_gap: px(4) }
            Children [ {buttons} ]
        ]
    }
}

// --- Turning settings -----------------------------------------------------------

/// Sets the setting at `index` to `value`, if that changes anything, and
/// starts timing how long its light takes to settle.
fn apply(world: &mut TuningWorld, panel: &mut Panel, index: usize, value: Value) {
    let mut tuning = world.tuning.clone();
    SETTINGS[index].set(&mut tuning, value);
    // Writing only on a real change keeps the config resource unchanged, so
    // the crate does nothing.
    if tuning != world.tuning {
        world.tuning = tuning;
        panel.change = Some(Change {
            setting: index,
            at: Instant::now(),
            frames: 0,
            settled: None,
        });
    }
}

/// Applies a slider's value at once if it only changes the look, and when the
/// drag ends or the last relight has settled otherwise, so dragging relights
/// as fast as the world keeps up and no faster.
fn slider_moved(
    change: On<ValueChange<f32>>,
    sliders: Query<&SliderOf>,
    mut panel: ResMut<Panel>,
    mut world: ResMut<TuningWorld>,
    mut commands: Commands,
) {
    let Ok(&SliderOf(index)) = sliders.get(change.event_target()) else {
        return;
    };
    let Control::Slider { min, max, step, .. } = SETTINGS[index].control else {
        return;
    };
    let value = (min + ((change.value - min) / step).round() * step).clamp(min, max);
    commands
        .entity(change.event_target())
        .insert(SliderValue(value));
    panel.explained = Some(index);
    let at_once = matches!(SETTINGS[index].applies, Applies::Instant | Applies::App);
    if at_once || change.is_final {
        panel.dragging = None;
        panel.pending = None;
        apply(&mut world, &mut panel, index, Value::Number(value));
    } else {
        panel.dragging = Some(index);
        panel.pending = Some((index, value));
    }
}

fn apply_dragged(
    mut panel: ResMut<Panel>,
    mut world: ResMut<TuningWorld>,
    voxel_world: VoxelWorld<TuningWorld>,
) {
    if panel.pending.is_none() || !voxel_world.light_is_settled() {
        return;
    }
    if let Some((index, value)) = panel.pending.take() {
        apply(&mut world, &mut panel, index, Value::Number(value));
    }
}

fn option_picked(
    change: On<ValueChange<Entity>>,
    selects: Query<&SelectOf>,
    options: Query<&OptionIndex>,
    mut panel: ResMut<Panel>,
    mut world: ResMut<TuningWorld>,
) {
    let (Ok(&SelectOf(index)), Ok(&OptionIndex(option))) = (
        selects.get(change.event_target()),
        options.get(change.value),
    ) else {
        return;
    };
    panel.explained = Some(index);
    apply(&mut world, &mut panel, index, Value::Option(option));
}

fn segment_pressed(
    activate: On<Activate>,
    segments: Query<&SegmentOf>,
    mut panel: ResMut<Panel>,
    mut world: ResMut<TuningWorld>,
) {
    if let Ok(&SegmentOf(index, option)) = segments.get(activate.event_target()) {
        panel.explained = Some(index);
        apply(&mut world, &mut panel, index, Value::Option(option));
    }
}

fn reset_pressed(
    activate: On<Activate>,
    resets: Query<&ResetOf>,
    mut panel: ResMut<Panel>,
    mut world: ResMut<TuningWorld>,
) {
    if let Ok(&ResetOf(index)) = resets.get(activate.event_target()) {
        panel.explained = Some(index);
        let value = SETTINGS[index].value(&default());
        apply(&mut world, &mut panel, index, value);
    }
}

fn reset_all_pressed(
    activate: On<Activate>,
    buttons: Query<(), With<ResetAllButton>>,
    mut world: ResMut<TuningWorld>,
) {
    if buttons.contains(activate.event_target()) && world.tuning != default() {
        world.tuning = default();
    }
}

fn print_pressed(
    activate: On<Activate>,
    buttons: Query<(), With<PrintButton>>,
    mut panel: ResMut<Panel>,
    world: Res<TuningWorld>,
) {
    if buttons.contains(activate.event_target()) {
        print_config(&mut panel, &world);
    }
}

fn print_config(panel: &mut Panel, world: &TuningWorld) {
    report!("{}", config_rust(&world.tuning));
    panel.printed = Some(Instant::now());
}

fn tab_pressed(
    activate: On<Activate>,
    tabs: Query<&TabButton>,
    mut panel: ResMut<Panel>,
) {
    if let Ok(&TabButton(index)) = tabs.get(activate.event_target()) {
        panel.tab = Tab::ALL[index];
    }
}

fn place_pressed(
    activate: On<Activate>,
    places: Query<&PlaceButton>,
    mut cameras: Query<(&mut Transform, &mut PanOrbitCamera)>,
) {
    if let Ok(&PlaceButton(index)) = places.get(activate.event_target()) {
        move_camera(Place::ALL[index], &mut cameras);
    }
}

/// H hides the panel, P prints the config, and the number keys move the
/// camera to the places.
fn keys(
    keys: Res<ButtonInput<KeyCode>>,
    mut panel: ResMut<Panel>,
    world: Res<TuningWorld>,
    mut cameras: Query<(&mut Transform, &mut PanOrbitCamera)>,
) {
    if keys.just_pressed(KeyCode::KeyH) {
        panel.hidden = !panel.hidden;
    }
    if keys.just_pressed(KeyCode::KeyP) {
        print_config(&mut panel, &world);
    }
    const DIGITS: [KeyCode; 8] = [
        KeyCode::Digit1,
        KeyCode::Digit2,
        KeyCode::Digit3,
        KeyCode::Digit4,
        KeyCode::Digit5,
        KeyCode::Digit6,
        KeyCode::Digit7,
        KeyCode::Digit8,
    ];
    for (place, key) in Place::ALL.into_iter().zip(DIGITS) {
        if keys.just_pressed(key) {
            move_camera(place, &mut cameras);
        }
    }
}

/// Times the last change until the world's light has settled.
fn time_settling(mut panel: ResMut<Panel>, voxel_world: VoxelWorld<TuningWorld>) {
    let Some(change) = panel.change.as_mut() else {
        return;
    };
    if change.settled.is_some() {
        return;
    }
    change.frames += 1;
    if voxel_world.light_is_settled() {
        let ms = change.at.elapsed().as_secs_f32() * 1000.0;
        change.settled = Some((ms, change.frames));
    }
}

fn explain_hovered(
    rows: Query<(&RowOf, &Hovered), Changed<Hovered>>,
    mut panel: ResMut<Panel>,
) {
    for (&RowOf(index), hovered) in &rows {
        if hovered.get() && panel.explained != Some(index) {
            panel.explained = Some(index);
        }
    }
}

// --- Showing the values ---------------------------------------------------------

/// Writes every slider back from the values being tuned, apart from one being
/// dragged.
fn show_sliders(
    world: Res<TuningWorld>,
    panel: Res<Panel>,
    sliders: Query<(Entity, &SliderOf, &SliderValue)>,
    mut commands: Commands,
) {
    for (entity, &SliderOf(index), shown) in &sliders {
        if panel.dragging == Some(index) {
            continue;
        }
        if let Value::Number(value) = SETTINGS[index].value(&world.tuning) {
            if shown.0 != value {
                commands.entity(entity).insert(SliderValue(value));
            }
        }
    }
}

/// Highlights the chosen option of every setting picked from options.
fn show_options(
    world: Res<TuningWorld>,
    mut segments: Query<(&SegmentOf, &mut ButtonVariant)>,
    selects: Query<(Entity, &SelectOf)>,
    children: Query<&Children>,
    list_rows: Query<(&OptionIndex, Has<Selected>)>,
    mut commands: Commands,
) {
    for (&SegmentOf(index, option), mut variant) in &mut segments {
        let chosen = SETTINGS[index].value(&world.tuning) == Value::Option(option);
        variant.set_if_neq(ButtonVariant::selected(chosen));
    }
    for (select, &SelectOf(index)) in &selects {
        let Value::Option(option) = SETTINGS[index].value(&world.tuning) else {
            continue;
        };
        let unselected_row = children.iter_descendants(select).find(|row| {
            list_rows
                .get(*row)
                .is_ok_and(|(&OptionIndex(of), selected)| of == option && !selected)
        });
        if let Some(row) = unselected_row {
            commands.trigger(SetSelected {
                entity: select,
                row,
            });
        }
    }
}

/// Shows the dot and the reset button of every setting changed from its
/// default.
fn show_changes(
    world: Res<TuningWorld>,
    mut dots: Query<(&DotOf, &mut BackgroundColor)>,
    mut resets: Query<(&ResetOf, &mut Visibility)>,
) {
    for (&DotOf(index), mut colour) in &mut dots {
        let changed = SETTINGS[index].changed(&world.tuning);
        colour.set_if_neq(BackgroundColor(if changed { CHANGED } else { Color::NONE }));
    }
    for (&ResetOf(index), mut visibility) in &mut resets {
        visibility.set_if_neq(if SETTINGS[index].changed(&world.tuning) {
            Visibility::Inherited
        } else {
            Visibility::Hidden
        });
    }
}

/// Shows or hides a node, writing it only when that changes, so the UI isn't
/// laid out again every frame.
fn show_node(mut node: Mut<Node>, shown: bool) {
    let display = if shown { Display::Flex } else { Display::None };
    if node.display != display {
        node.display = display;
    }
}

/// Shows the panel unless it's hidden, and in it the chosen tab's settings.
fn show_tab(
    panel: Res<Panel>,
    world: Res<TuningWorld>,
    mut overlays: Query<&mut Node, (With<Overlay>, Without<RowOf>)>,
    mut rows: Query<(&RowOf, &mut Node)>,
    mut tabs: Query<(&TabButton, &mut ButtonVariant)>,
) {
    for node in &mut overlays {
        show_node(node, !panel.hidden);
    }
    for (&RowOf(index), node) in &mut rows {
        let setting = &SETTINGS[index];
        show_node(
            node,
            setting.tab == panel.tab && (setting.shown)(&world.tuning),
        );
    }
    for (&TabButton(index), mut variant) in &mut tabs {
        variant.set_if_neq(ButtonVariant::selected(Tab::ALL[index] == panel.tab));
    }
}

/// Lights the button of the place the camera is at, and shows the line under
/// the one the explained setting is best seen from.
fn show_places(
    panel: Res<Panel>,
    cameras: Query<&PanOrbitCamera>,
    mut buttons: Query<(&PlaceButton, &mut ButtonVariant)>,
    mut hints: Query<(&HintOf, &mut Visibility)>,
) {
    let in_view = cameras.iter().find_map(place_in_view);
    for (&PlaceButton(index), mut variant) in &mut buttons {
        variant.set_if_neq(ButtonVariant::selected(in_view == Some(Place::ALL[index])));
    }
    let best = panel.explained.map(|index| SETTINGS[index].place);
    for (&HintOf(index), mut visibility) in &mut hints {
        visibility.set_if_neq(if best == Some(Place::ALL[index]) {
            Visibility::Inherited
        } else {
            Visibility::Hidden
        });
    }
}

/// Writes the panel's texts: the explanation of the hovered setting and the
/// status line.
fn show_texts(
    panel: Res<Panel>,
    world: Res<TuningWorld>,
    mut texts: Query<(&PanelText, &mut Text, &mut Node)>,
    mut badge: Single<&mut BackgroundColor, With<Badge>>,
) {
    let explained = panel.explained.map(|index| &SETTINGS[index]);
    badge.set_if_neq(BackgroundColor(
        explained.map_or(Color::NONE, |setting| setting.applies.colour()),
    ));
    for (part, mut text, node) in &mut texts {
        let value = match (part, explained) {
            (PanelText::Subtitle, _) => panel.tab.subtitle().to_string(),
            (PanelText::GpuMemory, _) => format!(
                "The light texture takes {:.0} MiB of GPU memory with every slot in use.",
                texture_mib(&world.tuning.light())
            ),
            (PanelText::Name, Some(setting)) => setting.label.to_string(),
            (PanelText::Rust, Some(setting)) => setting.rust.to_string(),
            (PanelText::About, Some(setting)) => setting.about.to_string(),
            (PanelText::Where, Some(setting)) => format!(
                "Default {}. Best seen at {} (key {}).",
                setting.show(&default()),
                setting.place.name(),
                setting.place.key()
            ),
            (PanelText::Applies, Some(setting)) => setting.applies.label().to_string(),
            (PanelText::Name, None) => "Hover over a setting".to_string(),
            (PanelText::About, None) => "to see what it does and where to look. A dot \
                                         marks a setting changed from its default."
                .to_string(),
            (PanelText::Rust | PanelText::Where | PanelText::Applies, None) => {
                String::new()
            }
            (PanelText::Status, _) => status(&panel),
        };
        if text.0 != value {
            text.0 = value;
        }
        if *part == PanelText::GpuMemory {
            show_node(node, panel.tab == Tab::Memory);
        }
    }
}

/// The status line: how long the last change took to settle, or that the
/// config was just printed.
fn status(panel: &Panel) -> String {
    if panel
        .printed
        .is_some_and(|at| at.elapsed().as_secs_f32() < 3.0)
    {
        return "Printed the config as Rust to the console.".into();
    }
    let Some(change) = &panel.change else {
        return "Change a setting to time how long its light takes to settle.".into();
    };
    let name = SETTINGS[change.setting].label;
    match change.settled {
        None => format!("{name}: relighting, {} frames so far", change.frames),
        Some((_, 1)) => format!("{name}: showed on the next frame"),
        Some((ms, frames)) => {
            format!("{name}: light settled in {ms:.0} ms, {frames} frames")
        }
    }
}

/// Stops the camera following the mouse while the pointer is over the panel or
/// the places bar.
fn block_camera(
    blockers: Query<&Hovered, With<Overlay>>,
    mut cameras: Query<&mut PanOrbitCamera>,
) {
    let over_ui = blockers.iter().any(|hovered| hovered.get());
    for mut camera in &mut cameras {
        if camera.enabled == over_ui {
            camera.enabled = !over_ui;
        }
    }
}
