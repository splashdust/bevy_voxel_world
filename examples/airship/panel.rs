//! The help in the top left corner: one row per control, an explanation of
//! the row under the pointer only, and a line on what the airship is doing.

use bevy::{picking::hover::Hovered, prelude::*};

use crate::{
    flight::{Flight, Phase},
    world::{DARK_PLANKS, LANTERN},
    AirshipWorld, Brush, Status,
};

pub struct PanelPlugin;

impl Plugin for PanelPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, spawn_panel)
            .add_systems(Update, (hide, show_rows, explain_hovered, show_status));
    }
}

/// The panel, which edits ignore the cursor over.
#[derive(Component)]
pub struct Panel;

/// The row of the control at this index in [`CONTROLS`].
#[derive(Component)]
struct Row(usize);

/// The text of a row that says what its control does.
#[derive(Component)]
struct Does(usize);

/// The box under the rows that explains the hovered one.
#[derive(Component)]
struct Explanation;

#[derive(Component)]
struct StatusLine;

/// A control: its keys, what it does in a few words, and what it shows.
struct Control {
    keys: &'static str,
    does: &'static str,
    explains: &'static str,
}

const CONTROLS: [Control; 10] = [
    Control {
        keys: "Click, Shift+click",
        does: "dig, place",
        explains: "Edits whichever grid the cursor's ray hits first: the world, the \
                   airship or a piece. An edit to the moving airship lands on its own grid.",
    },
    Control {
        keys: "1 2 3",
        does: "brush",
        explains: "What Shift + click places: 1 planks, 2 stone, 3 a lamp. A lamp on the airship lights only the \
                   airship, wherever it flies.",
    },
    Control {
        keys: "F",
        does: "dock, fly on",
        explains: "Sends the airship back along its loop to hover over the dock, or \
                   onto the loop again. Docked, it is world voxels, and F lifts it out of \
                   the world: the pick takes the airship's materials and leaves the \
                   dock's stone.",
    },
    Control {
        keys: "Enter",
        does: "stamp on the dock",
        explains: "While the airship hovers, the outline is where a stamp would snap \
                   it, and each red box a world voxel it would land on. Enter eases the \
                   airship onto that pose and stamps it into the world.",
    },
    Control {
        keys: "X",
        does: "lift a piece off",
        explains: "Lifts a small box around the voxel under the cursor off the airship. \
                   The piece rides along as the airship's child, and slides out.",
    },
    Control {
        keys: "V",
        does: "stamp a piece back",
        explains: "Slides the piece under the cursor, or the newest, back where it came \
                   from, and stamps it into the airship's grid.",
    },
    Control {
        keys: "E",
        does: "sky",
        explains: "Switches SubGridSky. Enclosed: the airship's voxels shade it from the \
                   sky, so its cabin is dark but for the lamp and the doorway. World: \
                   every face takes the sky at its place. Look into the cabin's door.",
    },
    Control {
        keys: "Space",
        does: "pause autopilot",
        explains: "Holds the airship still on its loop, in the cave or the hall say, to \
                   edit it there.",
    },
    Control {
        keys: "Right drag, wheel",
        does: "turn, zoom",
        explains: "The camera follows the airship, and comes in front of rock that \
                   stands in the way.",
    },
    Control {
        keys: "H",
        does: "hide the help",
        explains: "",
    },
];

const TEXT_SIZE: f32 = 13.0;
const PANEL_WIDTH: f32 = 310.0;
/// The width of the column of keys.
const KEYS_WIDTH: f32 = 150.0;

fn text(line: impl Into<String>, color: Color) -> impl Bundle {
    (
        Text::new(line),
        TextFont {
            font_size: FontSize::Px(TEXT_SIZE),
            ..default()
        },
        TextColor(color),
    )
}

fn spawn_panel(mut commands: Commands) {
    commands
        .spawn((
            Panel,
            Hovered::default(),
            Node {
                position_type: PositionType::Absolute,
                top: Val::Px(8.0),
                left: Val::Px(8.0),
                width: Val::Px(PANEL_WIDTH),
                flex_direction: FlexDirection::Column,
                row_gap: Val::Px(2.0),
                padding: UiRect::all(Val::Px(8.0)),
                ..default()
            },
            BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.6)),
        ))
        .with_children(|panel| {
            for (index, control) in CONTROLS.iter().enumerate() {
                panel
                    .spawn((
                        Row(index),
                        Hovered::default(),
                        Node {
                            flex_direction: FlexDirection::Row,
                            ..default()
                        },
                    ))
                    .with_children(|row| {
                        row.spawn((
                            text(control.keys, Color::srgb(1.0, 0.9, 0.6)),
                            Node {
                                width: Val::Px(KEYS_WIDTH),
                                flex_shrink: 0.0,
                                ..default()
                            },
                        ));
                        row.spawn((Does(index), text("", Color::WHITE)));
                    });
            }
            panel.spawn((
                Explanation,
                text("", Color::srgb(0.8, 0.85, 0.9)),
                Node {
                    display: Display::None,
                    margin: UiRect::top(Val::Px(4.0)),
                    ..default()
                },
            ));
            panel.spawn((
                StatusLine,
                text("", Color::srgb(1.0, 0.9, 0.6)),
                Node {
                    margin: UiRect::top(Val::Px(4.0)),
                    ..default()
                },
            ));
        });
}

fn hide(
    keys: Res<ButtonInput<KeyCode>>,
    mut panels: Query<&mut Visibility, With<Panel>>,
) {
    if keys.just_pressed(KeyCode::KeyH) {
        for mut visibility in &mut panels {
            visibility.toggle_visible_hidden();
        }
    }
}

/// Writes what each control does, with the brush's material and the sky
/// setting where their rows name them.
fn show_rows(
    config: Res<AirshipWorld>,
    brush: Res<Brush>,
    mut texts: Query<(&Does, &mut Text)>,
) {
    let material = match brush.0 {
        DARK_PLANKS => "planks",
        LANTERN => "lamp",
        _ => "stone",
    };
    for (Does(index), mut text) in &mut texts {
        let does = CONTROLS[*index].does;
        let line = match *index {
            1 => format!("{does} ({material})"),
            6 => format!("{does}: {:?}", config.sub_grid_sky),
            _ => does.to_string(),
        };
        if text.0 != line {
            text.0 = line;
        }
    }
}

/// Shows the explanation of the row under the pointer, and nothing when the
/// pointer is over no row.
fn explain_hovered(
    rows: Query<(&Row, &Hovered)>,
    mut explanation: Query<(&mut Text, &mut Node), With<Explanation>>,
) {
    let Ok((mut text, mut node)) = explanation.single_mut() else {
        return;
    };
    let hovered = rows
        .iter()
        .find(|(_, hovered)| hovered.get())
        .map(|(Row(index), _)| CONTROLS[*index].explains)
        .filter(|explains| !explains.is_empty());
    let display = if hovered.is_some() {
        Display::Flex
    } else {
        Display::None
    };
    if node.display != display {
        node.display = display;
    }
    if let Some(explains) = hovered {
        if text.0 != explains {
            text.0 = explains.into();
        }
    }
}

fn show_status(
    flight: Res<Flight>,
    status: Res<Status>,
    mut lines: Query<&mut Text, With<StatusLine>>,
) {
    let Ok(mut text) = lines.single_mut() else {
        return;
    };
    let paused =
        flight.paused && matches!(flight.phase, Phase::Flying | Phase::Returning { .. });
    let line = format!(
        "Airship {}{}\n{}",
        flight.phase.describe(),
        if paused { ", paused" } else { "" },
        status.0
    );
    if text.0 != line {
        text.0 = line;
    }
}
