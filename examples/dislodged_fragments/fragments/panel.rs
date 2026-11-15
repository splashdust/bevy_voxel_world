//! The panel in the top right: the detection settings on one tab, the
//! controls on the other, and a line on what the pieces are doing.
//!
//! The settings live in the world config's [`FragmentDetection`], which the
//! panel's buttons write and read back. The crate reads them as each fragment
//! check starts, so a change takes effect at the next check.

use std::marker::PhantomData;

use bevy::{
    ecs::system::SystemParam,
    feathers::{
        controls::{ButtonVariant, FeathersButton},
        dark_theme::create_dark_theme,
        display::{caption, label, label_dim, label_small},
        rounded_corners::RoundedCorners,
        theme::{ThemeBackgroundColor, UiTheme},
        tokens, FeathersPlugins,
    },
    picking::hover::Hovered,
    prelude::*,
    ui_widgets::Activate,
};
use bevy_panorbit_camera::PanOrbitCamera;
use bevy_voxel_world::prelude::*;

use super::{outline::Outlines, rest::Piece, FragmentsConfig, Pieces, SettingPlaces};

/// A setting the panel changes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Setting {
    Connectivity,
    Limit,
    AutoLift,
}

/// The limits the panel offers: a smaller and a larger one around the
/// crate's default of 32,768.
const LIMITS: [u32; 3] = [4_096, 32_768, 131_072];

impl Setting {
    pub const ALL: [Setting; 3] =
        [Setting::Connectivity, Setting::Limit, Setting::AutoLift];

    fn label(self) -> &'static str {
        match self {
            Setting::Connectivity => "Connectivity",
            Setting::Limit => "Limit",
            Setting::AutoLift => "Auto-lift",
        }
    }

    fn options(self) -> &'static [&'static str] {
        match self {
            Setting::Connectivity => &["Faces", "Edges", "Corners"],
            Setting::Limit => &["4k", "32k", "128k"],
            Setting::AutoLift => &["On", "Off"],
        }
    }

    /// Which of its options `detection` has, or `None` for a value the panel
    /// doesn't offer.
    pub fn option<I>(self, detection: &FragmentDetection<I>) -> Option<usize> {
        match self {
            Setting::Connectivity => Some(match detection.connectivity {
                Connectivity::Faces => 0,
                Connectivity::Edges => 1,
                Connectivity::Corners => 2,
            }),
            Setting::Limit => LIMITS.iter().position(|&limit| limit == detection.limit),
            Setting::AutoLift => Some(if detection.auto_lift { 0 } else { 1 }),
        }
    }

    /// Sets `detection` to the option at `option`.
    pub fn set<I>(self, detection: &mut FragmentDetection<I>, option: usize) {
        match self {
            Setting::Connectivity => {
                detection.connectivity = [
                    Connectivity::Faces,
                    Connectivity::Edges,
                    Connectivity::Corners,
                ][option];
            }
            Setting::Limit => detection.limit = LIMITS[option],
            Setting::AutoLift => detection.auto_lift = option == 0,
        }
    }

    /// The field it sets, and what that does.
    fn about(self) -> &'static str {
        match self {
            Setting::Connectivity => {
                "connectivity: which neighbours hold a voxel up. Faces is the 6 \
                 sharing a face. Edges adds the 12 sharing an edge, and Corners \
                 the 8 sharing a corner, so a piece hanging by an edge or a \
                 corner stays."
            }
            Setting::Limit => {
                "limit: how many voxels a check floods through before it counts \
                 a piece as attached. A larger limit drops larger pieces, and \
                 each cut into anything attached costs more."
            }
            Setting::AutoLift => {
                "auto_lift: whether the crate lifts each fragment into a sub-grid, \
                 which falls here. Off, fragments are only reported, outlined \
                 here, and stay in place."
            }
        }
    }

    fn place(self, places: &SettingPlaces) -> &'static str {
        match self {
            Setting::Connectivity => places.connectivity,
            Setting::Limit => places.limit,
            Setting::AutoLift => places.auto_lift,
        }
    }
}

/// Builds the panel and keeps it in step with the settings.
pub struct PanelPlugin<C> {
    places: SettingPlaces,
    _world: PhantomData<fn() -> C>,
}

impl<C> PanelPlugin<C> {
    pub fn new(places: SettingPlaces) -> Self {
        Self {
            places,
            _world: PhantomData,
        }
    }
}

impl<C: FragmentsConfig> Plugin for PanelPlugin<C> {
    fn build(&self, app: &mut App) {
        app.add_plugins(FeathersPlugins)
            .insert_resource(UiTheme(create_dark_theme()))
            .insert_resource(Panel {
                places: self.places,
                tab: Tab::Settings,
                explained: None,
                hidden: false,
            })
            .init_resource::<Outlines>()
            .add_systems(Startup, spawn_panel)
            .add_systems(
                Update,
                (
                    keys,
                    explain_hovered,
                    show_options::<C>.run_if(resource_changed::<C>),
                    show_tab,
                    show_texts::<C>,
                    block_camera,
                )
                    .chain(),
            )
            .add_observer(segment_pressed::<C>)
            .add_observer(tab_pressed);
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Tab {
    Settings,
    Controls,
}

impl Tab {
    const ALL: [Tab; 2] = [Tab::Settings, Tab::Controls];

    fn label(self) -> &'static str {
        match self {
            Tab::Settings => "Settings",
            Tab::Controls => "Controls",
        }
    }
}

#[derive(Resource)]
struct Panel {
    places: SettingPlaces,
    tab: Tab,
    /// The setting the pointer was last over, which the panel explains.
    explained: Option<Setting>,
    hidden: bool,
}

/// The panel: H hides it, and clicks and the camera ignore the mouse over it.
#[derive(Component, Default, Clone)]
struct Overlay;

/// Whether the pointer is over the panel.
#[derive(SystemParam)]
pub struct OverPanel<'w, 's> {
    overlays: Query<'w, 's, &'static Hovered, With<Overlay>>,
}

impl OverPanel<'_, '_> {
    pub fn get(&self) -> bool {
        self.overlays.iter().any(|hovered| hovered.get())
    }
}

/// The row of the setting at this index in [`Setting::ALL`].
#[derive(Component, Default, Clone, Copy)]
struct RowOf(usize);

/// One of a setting's option buttons: the setting's index, then the option.
#[derive(Component, Default, Clone, Copy)]
struct SegmentOf(usize, usize);

/// A tab button, by index in [`Tab::ALL`].
#[derive(Component, Default, Clone, Copy)]
struct TabButton(usize);

/// The contents of the tab at this index in [`Tab::ALL`].
#[derive(Component, Default, Clone, Copy)]
struct PageOf(usize);

/// A text in the panel that changes, by what it shows.
#[derive(Component, Default, Clone, Copy, PartialEq)]
enum PanelText {
    /// The explained setting's name.
    #[default]
    Name,
    /// What it sets and does.
    About,
    /// Its default, and where to see it.
    Where,
    /// What the pieces are doing.
    Status,
}

const PANEL_WIDTH: f32 = 300.0;
const LABEL_WIDTH: f32 = 84.0;

const CONTROLS: &str = "Left click carves a sphere in the world or in a \
                        falling piece.\nRight-drag orbits, middle-drag pans, \
                        the wheel zooms.\nH hides the panel.";

/// Rounds the outer corners of a row of `count` buttons side by side.
fn segment_corners(index: usize, count: usize) -> RoundedCorners {
    match index {
        0 => RoundedCorners::Left,
        last if last + 1 == count => RoundedCorners::Right,
        _ => RoundedCorners::None,
    }
}

fn spawn_panel(mut commands: Commands) {
    commands.spawn_scene(panel());
}

fn panel() -> impl Scene {
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
    let rows: Vec<Box<dyn SceneList>> =
        (0..Setting::ALL.len()).map(setting_row).collect();
    let rows: Box<dyn SceneList> = Box::new(rows);
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
                @label("Dislodged fragments")
                --
                @label_dim("H hides")
            ]
            --
            Node { flex_direction: FlexDirection::Row, column_gap: px(1) }
            Children [ {tabs} ]
            --
            Node { flex_direction: FlexDirection::Column, row_gap: px(6) }
            PageOf(0)
            Children [
                Node { flex_direction: FlexDirection::Column, row_gap: px(4) }
                Children [ {rows} ]
                --
                Node {
                    flex_direction: FlexDirection::Column,
                    row_gap: px(3),
                    min_height: px(96),
                    padding: UiRect::top(px(8)),
                    border: UiRect::top(px(1)),
                }
                BorderColor::all(Color::srgba(1.0, 1.0, 1.0, 0.12))
                Children [
                    @label("") PanelText::Name
                    --
                    @label_dim("") PanelText::About
                    --
                    @label_small("") PanelText::Where
                ]
            ]
            --
            Node { flex_direction: FlexDirection::Column }
            PageOf(1)
            Children [ @label_dim(CONTROLS) ]
            --
            @label_small("") PanelText::Status
        ]
    }
}

/// A setting's row: its name and a button for each of its options.
fn setting_row(index: usize) -> Box<dyn SceneList> {
    let setting = Setting::ALL[index];
    let name = setting.label();
    let options = setting.options();
    let buttons: Vec<Box<dyn SceneList>> = options
        .iter()
        .enumerate()
        .map(|(option, text)| -> Box<dyn SceneList> {
            let text = *text;
            let corners = segment_corners(option, options.len());
            bsn! {
                @FeathersButton {
                    @caption: bsn! { @caption(text) },
                    @corners: {corners},
                }
                Node { flex_grow: 1.0 }
                SegmentOf(index, option)
            }
            .into()
        })
        .collect();
    let buttons: Box<dyn SceneList> = Box::new(buttons);
    bsn! {
        Node {
            flex_direction: FlexDirection::Row,
            align_items: AlignItems::Center,
            column_gap: px(6),
        }
        RowOf(index)
        Hovered
        Children [
            @label(name)
            Node { width: px(LABEL_WIDTH), flex_shrink: 0.0 }
            --
            Node { flex_grow: 1.0, flex_direction: FlexDirection::Row, column_gap: px(1) }
            Children [ {buttons} ]
        ]
    }
    .into()
}

fn segment_pressed<C: FragmentsConfig>(
    activate: On<Activate>,
    segments: Query<&SegmentOf>,
    mut panel: ResMut<Panel>,
    mut config: ResMut<C>,
) {
    if let Ok(&SegmentOf(index, option)) = segments.get(activate.event_target()) {
        let setting = Setting::ALL[index];
        panel.explained = Some(setting);
        // Writing only on a real change leaves the config resource unchanged
        // otherwise.
        if setting.option(config.detection()) != Some(option) {
            setting.set(config.detection_mut(), option);
        }
    }
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

fn keys(keys: Res<ButtonInput<KeyCode>>, mut panel: ResMut<Panel>) {
    if keys.just_pressed(KeyCode::KeyH) {
        panel.hidden = !panel.hidden;
    }
}

fn explain_hovered(
    rows: Query<(&RowOf, &Hovered), Changed<Hovered>>,
    mut panel: ResMut<Panel>,
) {
    for (&RowOf(index), hovered) in &rows {
        let setting = Setting::ALL[index];
        if hovered.get() && panel.explained != Some(setting) {
            panel.explained = Some(setting);
        }
    }
}

/// Lights each setting's chosen option.
fn show_options<C: FragmentsConfig>(
    config: Res<C>,
    mut segments: Query<(&SegmentOf, &mut ButtonVariant)>,
) {
    for (&SegmentOf(index, option), mut variant) in &mut segments {
        let chosen = Setting::ALL[index].option(config.detection()) == Some(option);
        variant.set_if_neq(ButtonVariant::selected(chosen));
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

/// Shows the panel unless it's hidden, and in it the chosen tab.
fn show_tab(
    panel: Res<Panel>,
    mut overlays: Query<&mut Node, (With<Overlay>, Without<PageOf>)>,
    mut pages: Query<(&PageOf, &mut Node)>,
    mut tabs: Query<(&TabButton, &mut ButtonVariant)>,
) {
    for node in &mut overlays {
        show_node(node, !panel.hidden);
    }
    for (&PageOf(index), node) in &mut pages {
        show_node(node, Tab::ALL[index] == panel.tab);
    }
    for (&TabButton(index), mut variant) in &mut tabs {
        variant.set_if_neq(ButtonVariant::selected(Tab::ALL[index] == panel.tab));
    }
}

/// Writes the explanation of the hovered setting and the status line.
fn show_texts<C: FragmentsConfig>(
    panel: Res<Panel>,
    config: Res<C>,
    pieces: Query<&Piece>,
    stamped: Res<Pieces>,
    outlines: Res<Outlines>,
    mut texts: Query<(&PanelText, &mut Text)>,
) {
    let explained = panel.explained;
    for (part, mut text) in &mut texts {
        let value = match (part, explained) {
            (PanelText::Name, Some(setting)) => setting.label().to_string(),
            (PanelText::About, Some(setting)) => setting.about().to_string(),
            (PanelText::Where, Some(setting)) => {
                let default = FragmentDetection::<C::MaterialIndex>::default();
                let default = setting
                    .option(&default)
                    .map_or("", |option| setting.options()[option]);
                format!("Crate default {default}. {}", setting.place(&panel.places))
            }
            (PanelText::Name, None) => "Hover over a setting".to_string(),
            (PanelText::About, None) => "to see what it does and where to look. A \
                                         change takes effect at the next check."
                .to_string(),
            (PanelText::Where, None) => String::new(),
            (PanelText::Status, _) => {
                if config.detection().auto_lift {
                    let falling = pieces
                        .iter()
                        .filter(|piece| **piece == Piece::Falling)
                        .count();
                    let resting = pieces.iter().count() - falling;
                    format!(
                        "{falling} falling, {resting} coming to rest, {} stamped",
                        stamped.stamped
                    )
                } else {
                    format!("{} fragments outlined", outlines.len())
                }
            }
        };
        if text.0 != value {
            text.0 = value;
        }
    }
}

/// Stops the camera following the mouse while the pointer is over the panel.
fn block_camera(over_panel: OverPanel, mut cameras: Query<&mut PanOrbitCamera>) {
    let over = over_panel.get();
    for mut camera in &mut cameras {
        if camera.enabled == over {
            camera.enabled = !over;
        }
    }
}
