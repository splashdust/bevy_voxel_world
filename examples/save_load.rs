//! Saves the voxel edits to a file and loads them back, with the terrain seed.
//!
//! A save is every edited chunk's full delta: `edited_chunks` lists the
//! chunks, and `full_delta` gives each one's edits at its edit sequence. A
//! load forgets the world's edits with `forget_edits`, then applies each
//! saved full delta with `apply_delta`, so the saved sequences become the
//! world's. If a save exists when the example starts, it is loaded before any
//! chunk spawns, so every chunk takes its saved edits as it spawns.
//!
//! Controls:
//!   - Left click to carve a small sphere, right click (or Shift + left
//!     click) to lay a blob of brick
//!   - S to save, L to load the save again, dropping edits made since
//!   - N for a new world: a random seed and no edits
//!   - C to clear every edit
//!   - Arrow keys to move around
//!   - On a trackpad: scroll with two fingers to orbit, with Shift held to
//!     pan, and pinch to zoom
//!   - With a mouse: drag with the middle button to orbit, with Shift held to
//!     pan, and scroll to zoom
//!
//! C and N look alike but differ underneath. A clear is a decision about the
//! world: every edited chunk moves its edit sequence on and reports an empty
//! full delta, so other worlds and saves see the edits go, and a save made
//! after it restores the clear. A new world forgets: the edits and their
//! sequences are dropped without a word, as when switching to another world,
//! and a save made after it holds no chunks.
//!
//! The save file is one format version byte, then an `lz4_flex` block of the
//! `postcard` encoding of the seed and the full deltas. `EditDelta` carries no
//! version of its own, so the version lives here, in the file around it.
//!
//! Run with `cargo run --example save_load --features serde`. In the browser
//! the save is kept in memory, so save and load work until the page reloads.

use std::{collections::HashSet, sync::Arc};

use bevy::{
    light::CascadeShadowConfigBuilder, platform::collections::HashMap, prelude::*,
};
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin, TrackpadBehavior};
use bevy_voxel_world::prelude::*;
use noise::{HybridMulti, NoiseFn, Perlin};
use serde::{Deserialize, Serialize};

// Materials, and the layers of `example_voxel_texture.png` they show.
const SNOW: u8 = 0;
const BRICK: u8 = 1;
const GRASS: u8 = 2;

const CARVE_RADIUS: i32 = 6;
const BLOB_RADIUS: i32 = 4;

/// The height of the flat ground the hills rise from. It lies inside the
/// level-0 chunks at y = 0, which spawn, rather than on top of the chunks
/// below them, which are hinted as solid and never spawn, so never show
/// their top face.
const FLOOR: f64 = 16.0;

/// How fast the arrow keys move the camera, in voxels per second.
const PAN_SPEED: f32 = 60.0;

/// The voxel world, whose terrain comes from `seed`.
///
/// The voxel lookup delegate is fetched from this config each time a chunk
/// task starts, so a new seed reaches every chunk generated after it changes.
#[derive(Resource, Clone)]
struct SaveLoadWorld {
    seed: u32,
}

impl Default for SaveLoadWorld {
    fn default() -> Self {
        Self { seed: 1234 }
    }
}

impl VoxelWorldConfig for SaveLoadWorld {
    type MaterialIndex = u8;
    type ChunkUserBundle = ();

    fn spawning_distance(&self) -> u32 {
        32
    }

    fn voxel_lookup_delegate(&self) -> VoxelLookupDelegate<Self::MaterialIndex> {
        let seed = self.seed;
        Box::new(move |_chunk_pos, _lod, _previous| terrain(seed))
    }

    // Below y = 0 everything is solid, and the hills end below y = 64.
    fn chunk_fill_hint(&self, key: ChunkKey) -> FillHint {
        if key.position.y + key.extent() <= 0 {
            FillHint::Uniform(WorldVoxel::Solid(GRASS))
        } else if key.position.y >= 2 {
            FillHint::Uniform(WorldVoxel::Air)
        } else {
            FillHint::Unknown
        }
    }

    fn texture_index_mapper(&self) -> Arc<dyn Fn(u8) -> [u32; 3] + Send + Sync> {
        Arc::new(|material| match material {
            SNOW => [0, 0, 0],
            BRICK => [2, 2, 2],
            _ => [3, 3, 3],
        })
    }

    fn voxel_texture(&self) -> Option<(String, u32)> {
        Some(("example_voxel_texture.png".into(), 4))
    }
}

fn main() {
    App::new()
        .add_plugins((DefaultPlugins, PanOrbitCameraPlugin))
        .add_plugins(VoxelWorldPlugin::with_config(SaveLoadWorld::default()))
        .init_resource::<SaveState>()
        .add_systems(Startup, (setup, load_at_startup))
        // A save reads committed edits only, so it runs before the brushes:
        // it holds everything committed up to this frame, and this frame's
        // brush strokes commit in the next frame's `PreUpdate`. A load or a
        // clear made here commits then too.
        .add_systems(Update, (track_unsaved, keys, brushes, update_panel).chain())
        .add_systems(Update, move_with_arrow_keys)
        .run();
}

/// What the file holds: the terrain seed, and the full delta of each edited
/// chunk, in chunk position order.
#[derive(Serialize, Deserialize, PartialEq, Debug)]
struct Save {
    seed: u32,
    deltas: Vec<EditDelta<u8>>,
}

impl Save {
    /// The save of `deltas`, sorted by chunk position. `edited_chunks` lists
    /// chunks in no particular order, and sorting them makes the same world
    /// encode to the same bytes.
    fn new(seed: u32, mut deltas: Vec<EditDelta<u8>>) -> Self {
        deltas.sort_by_key(|delta| delta.chunk.to_array());
        Self { seed, deltas }
    }
}

/// The version of the file layout, written as the file's first byte. Bump it
/// when `Save` or the crate's `EditDelta` layout changes, and keep reading
/// the old versions a game has shipped.
const FORMAT_VERSION: u8 = 1;

/// The file bytes for `save`, and the size of its `postcard` encoding before
/// compression.
fn encode(save: &Save) -> (Vec<u8>, usize) {
    let encoded = postcard::to_allocvec(save).expect("a save always encodes");
    let mut bytes = vec![FORMAT_VERSION];
    bytes.extend(lz4_flex::compress_prepend_size(&encoded));
    (bytes, encoded.len())
}

/// The save in `bytes`, or why they hold none. It decodes the whole file
/// before anything touches the world, so a file that fails leaves the world
/// as it was.
fn decode(bytes: &[u8]) -> Result<Save, String> {
    match bytes.split_first() {
        None => Err("the file is empty".into()),
        Some((&FORMAT_VERSION, body)) => {
            let encoded = lz4_flex::decompress_size_prepended(body)
                .map_err(|error| format!("corrupt file: {error}"))?;
            postcard::from_bytes(&encoded)
                .map_err(|error| format!("corrupt file: {error}"))
        }
        Some((version, _)) => Err(format!("unknown format version {version}")),
    }
}

/// Where the save is kept: a file in the system temp directory.
#[cfg(not(target_arch = "wasm32"))]
mod storage {
    use std::{io::ErrorKind, path::PathBuf};

    fn path() -> PathBuf {
        std::env::temp_dir().join("bevy_voxel_world_save_load.bin")
    }

    pub fn place() -> String {
        path().display().to_string()
    }

    // These block the frame while the file is read or written, which a small
    // save allows. A real game does it in an IO task.

    pub fn write(bytes: &[u8]) -> Result<(), String> {
        std::fs::write(path(), bytes).map_err(|error| error.to_string())
    }

    /// The saved bytes, or `None` when nothing was saved.
    pub fn read() -> Result<Option<Vec<u8>>, String> {
        match std::fs::read(path()) {
            Ok(bytes) => Ok(Some(bytes)),
            Err(error) if error.kind() == ErrorKind::NotFound => Ok(None),
            Err(error) => Err(error.to_string()),
        }
    }
}

/// Where the save is kept in the browser: in memory, until the page reloads.
#[cfg(target_arch = "wasm32")]
mod storage {
    use std::sync::Mutex;

    static SLOT: Mutex<Option<Vec<u8>>> = Mutex::new(None);

    pub fn place() -> String {
        "in memory".into()
    }

    pub fn write(bytes: &[u8]) -> Result<(), String> {
        *SLOT.lock().unwrap() = Some(bytes.to_vec());
        Ok(())
    }

    pub fn read() -> Result<Option<Vec<u8>>, String> {
        Ok(SLOT.lock().unwrap().clone())
    }
}

/// Whether the world differs from the save, and the line the panel shows
/// about the last thing that happened.
#[derive(Resource, Default)]
struct SaveState {
    /// Each chunk's edit sequence at the last save or load.
    saved: HashMap<IVec3, u64>,
    /// The chunks that committed at a sequence other than the saved one.
    unsaved: HashSet<IVec3>,
    /// Set by a new world, which changes the seed and forgets the edits
    /// without committing anything.
    new_world: bool,
    last_event: String,
}

impl SaveState {
    /// Marks the world as matching the save whose chunks had `saved`.
    fn mark_saved(&mut self, saved: HashMap<IVec3, u64>) {
        self.saved = saved;
        self.unsaved.clear();
        self.new_world = false;
    }

    /// Shows `event` in the panel, and logs it.
    fn report(&mut self, event: String) {
        info!("{event}");
        self.last_event = event;
    }
}

/// Loads the save before any chunk has spawned, if there is one. In the
/// browser nothing is saved yet when the page starts.
fn load_at_startup(
    mut config: ResMut<SaveLoadWorld>,
    mut voxel_world: VoxelWorld<SaveLoadWorld>,
    mut state: ResMut<SaveState>,
) {
    load(&mut config, &mut voxel_world, &mut state);
}

/// Reads the save and loads it into the world: sets the seed, forgets every
/// edit, then applies each saved full delta. It regenerates the chunks only
/// when the seed changes; forgetting already regenerates the chunks that
/// showed an edit.
fn load(
    config: &mut SaveLoadWorld,
    voxel_world: &mut VoxelWorld<SaveLoadWorld>,
    state: &mut SaveState,
) {
    let bytes = match storage::read() {
        Ok(Some(bytes)) => bytes,
        Ok(None) => {
            state.report(format!("No save yet ({})", storage::place()));
            return;
        }
        Err(error) => {
            state.report(format!("Can't read the save: {error}"));
            return;
        }
    };
    let save = match decode(&bytes) {
        Ok(save) => save,
        Err(error) => {
            state.report(format!(
                "Rejected the save, {error}; the world is unchanged"
            ));
            return;
        }
    };

    let seed_changed = config.seed != save.seed;
    config.seed = save.seed;
    voxel_world.forget_edits();
    let mut saved = HashMap::new();
    for delta in save.deltas {
        let (chunk, seq) = (delta.chunk, delta.seq);
        // A full delta always continues a forgotten chunk; apply refuses only
        // one whose voxels are malformed.
        if voxel_world.apply_delta(delta).is_ok() {
            saved.insert(chunk, seq);
        }
    }
    if seed_changed {
        voxel_world.regenerate_chunks();
    }
    state.report(format!("Loaded {} chunks, seed {}", saved.len(), save.seed));
    state.mark_saved(saved);
}

/// Writes every edited chunk's full delta, with the seed. A chunk whose edits
/// were all cleared is saved too, with none, so that the clear survives a
/// load.
///
/// It rewrites the whole file each time. A game with a larger world writes
/// only the unsaved chunks, to a store with a record per chunk or region.
fn save(seed: u32, voxel_world: &VoxelWorld<SaveLoadWorld>, state: &mut SaveState) {
    let edited = voxel_world.edited_chunks();
    let deltas = edited
        .iter()
        .map(|&(chunk, _)| voxel_world.full_delta(chunk))
        .collect();
    let (bytes, encoded_len) = encode(&Save::new(seed, deltas));
    if let Err(error) = storage::write(&bytes) {
        state.report(format!("Can't write the save: {error}"));
        return;
    }
    state.report(format!(
        "Saved {} chunks: {} bytes, {} compressed ({})",
        edited.len(),
        encoded_len,
        bytes.len(),
        storage::place()
    ));
    state.mark_saved(edited.into_iter().collect());
}

/// Marks each chunk that commits at a sequence other than the saved one as
/// unsaved, and one that commits back at it as saved.
///
/// It compares sequences rather than counting messages: a load's own commits
/// also write `EditsCommitted`, at exactly the saved sequences.
fn track_unsaved(
    mut messages: MessageReader<EditsCommitted<SaveLoadWorld>>,
    mut state: ResMut<SaveState>,
) {
    for EditsCommitted { delta } in messages.read() {
        if state.saved.get(&delta.chunk) == Some(&delta.seq) {
            state.unsaved.remove(&delta.chunk);
        } else {
            state.unsaved.insert(delta.chunk);
        }
    }
}

fn keys(
    keys: Res<ButtonInput<KeyCode>>,
    mut config: ResMut<SaveLoadWorld>,
    mut voxel_world: VoxelWorld<SaveLoadWorld>,
    mut state: ResMut<SaveState>,
) {
    if keys.just_pressed(KeyCode::KeyS) {
        save(config.seed, &voxel_world, &mut state);
    }
    if keys.just_pressed(KeyCode::KeyL) {
        load(&mut config, &mut voxel_world, &mut state);
    }
    if keys.just_pressed(KeyCode::KeyN) {
        config.seed = rand::random();
        voxel_world.forget_edits();
        voxel_world.regenerate_chunks();
        state.saved.clear();
        state.unsaved.clear();
        state.new_world = true;
        state.report(format!("New world, seed {}", config.seed));
    }
    if keys.just_pressed(KeyCode::KeyC) {
        voxel_world.clear_edits();
        state.report("Cleared every edit".into());
    }
}

/// Left click carves a sphere of air where the cursor points, and right
/// click, or Shift + left click, lays a blob of brick on the surface there. Each sets a box of
/// voxels in one `set_voxels_in` call, leaving those outside the sphere alone,
/// so a few clicks give a chunk enough edits to be saved in the dense form.
fn brushes(
    buttons: Res<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
    window: Query<&Window>,
    camera: Query<(&Camera, &GlobalTransform), With<VoxelWorldCamera<SaveLoadWorld>>>,
    mut voxel_world: VoxelWorld<SaveLoadWorld>,
) {
    let shift = keys.any_pressed([KeyCode::ShiftLeft, KeyCode::ShiftRight]);
    let lay = buttons.just_pressed(MouseButton::Right)
        || (shift && buttons.just_pressed(MouseButton::Left));
    if !lay && !buttons.just_pressed(MouseButton::Left) {
        return;
    }
    let (Ok(window), Ok((camera, camera_transform))) = (window.single(), camera.single())
    else {
        return;
    };
    let Some(ray) = window
        .cursor_position()
        .and_then(|cursor| camera.viewport_to_world(camera_transform, cursor).ok())
    else {
        return;
    };
    let Some(hit) = voxel_world.raycast(ray, &|_| true) else {
        return;
    };

    let (center, radius, voxel) = if lay {
        let normal = hit.voxel_normal().unwrap_or(IVec3::Y);
        (
            hit.voxel_pos() + normal * (BLOB_RADIUS / 2),
            BLOB_RADIUS,
            WorldVoxel::Solid(BRICK),
        )
    } else {
        (hit.voxel_pos(), CARVE_RADIUS, WorldVoxel::Air)
    };
    voxel_world.set_voxels_in(
        center - IVec3::splat(radius),
        center + IVec3::splat(radius + 1),
        |position| {
            ((position - center).length_squared() <= radius * radius).then_some(voxel)
        },
    );
}

#[derive(Component)]
struct Panel;

fn update_panel(
    config: Res<SaveLoadWorld>,
    voxel_world: VoxelWorld<SaveLoadWorld>,
    state: Res<SaveState>,
    mut panel: Query<&mut Text, With<Panel>>,
) {
    let Ok(mut text) = panel.single_mut() else {
        return;
    };
    let saved = if state.new_world {
        "Unsaved: new world".to_string()
    } else if state.unsaved.is_empty() {
        "Saved".to_string()
    } else {
        format!("Unsaved: {} chunks", state.unsaved.len())
    };
    text.0 = format!(
        "S save  L load  N new world  C clear\n\
         Click carve, right-click brick\n\
         Seed {}, {} edited chunks\n{}\n{}",
        config.seed,
        voxel_world.edited_chunks().len(),
        saved,
        state.last_event
    );
}

fn setup(mut commands: Commands) {
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(-60.0, FLOOR as f32 + 50.0, -60.0)
            .looking_at(Vec3::new(0.0, FLOOR as f32, 0.0), Vec3::Y),
        PanOrbitCamera {
            focus: Vec3::new(0.0, FLOOR as f32, 0.0),
            // Both mouse buttons are brushes, so the camera takes the middle
            // button, and a trackpad's two-finger scroll and pinch.
            button_orbit: MouseButton::Middle,
            button_pan: MouseButton::Middle,
            modifier_pan: Some(KeyCode::ShiftLeft),
            trackpad_behavior: TrackpadBehavior::BlenderLike {
                modifier_pan: Some(KeyCode::ShiftLeft),
                modifier_zoom: Some(KeyCode::ControlLeft),
            },
            trackpad_pinch_to_zoom_enabled: true,
            ..default()
        },
        VoxelWorldCamera::<SaveLoadWorld>::default(),
    ));

    let cascade_shadow_config = CascadeShadowConfigBuilder {
        maximum_distance: 400.0,
        ..default()
    }
    .build();
    commands.spawn((
        DirectionalLight {
            color: Color::srgb(0.98, 0.95, 0.82),
            shadow_maps_enabled: true,
            ..default()
        },
        Transform::default().looking_at(Vec3::new(-0.15, -0.1, 0.15), Vec3::Y),
        cascade_shadow_config,
    ));
    commands.insert_resource(GlobalAmbientLight {
        color: Color::srgb(0.98, 0.95, 0.82),
        brightness: 100.0,
        affects_lightmapped_meshes: true,
    });

    commands.spawn((
        Text::default(),
        TextFont {
            font_size: FontSize::Px(14.0),
            ..default()
        },
        TextColor(Color::WHITE),
        BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.5)),
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(8.0),
            left: Val::Px(8.0),
            padding: UiRect::all(Val::Px(6.0)),
            ..default()
        },
        Panel,
    ));
}

/// Moves the camera's focus along the ground with the arrow keys, up the
/// screen for the up arrow, whichever way the camera faces.
fn move_with_arrow_keys(
    keys: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
    mut camera: Query<(&mut PanOrbitCamera, &Transform)>,
) {
    let Ok((mut camera, transform)) = camera.single_mut() else {
        return;
    };
    let forward = transform.forward().with_y(0.0).normalize_or_zero();
    let right = transform.right().with_y(0.0).normalize_or_zero();
    let mut direction = Vec3::ZERO;
    for (key, step) in [
        (KeyCode::ArrowUp, forward),
        (KeyCode::ArrowDown, -forward),
        (KeyCode::ArrowRight, right),
        (KeyCode::ArrowLeft, -right),
    ] {
        if keys.pressed(key) {
            direction += step;
        }
    }
    camera.target_focus += direction.normalize_or_zero() * PAN_SPEED * time.delta_secs();
}

/// Rolling hills from `seed` on flat ground, grass with snow on the tops.
fn terrain(
    seed: u32,
) -> Box<dyn FnMut(IVec3, Option<WorldVoxel>) -> WorldVoxel + Send + Sync> {
    let mut noise = HybridMulti::<Perlin>::new(seed);
    noise.octaves = 4;
    noise.frequency = 1.1;
    noise.lacunarity = 2.8;
    noise.persistence = 0.4;

    // The height only depends on x and z, so cache it per column.
    let mut heights = HashMap::<(i32, i32), f64>::new();

    Box::new(move |pos: IVec3, _previous| {
        if pos.y < 0 {
            return WorldVoxel::Solid(GRASS);
        }
        let height = *heights.entry((pos.x, pos.z)).or_insert_with(|| {
            let sample = noise.get([pos.x as f64 / 300.0, pos.z as f64 / 300.0]);
            // The noise dips below the floor about half the time, which
            // leaves wide flat stretches between the hills. Capped below
            // y = 64, where the chunks hinted as air begin.
            (FLOOR + sample * 48.0).clamp(FLOOR, 63.0)
        });
        let y = pos.y as f64;
        if y >= height {
            WorldVoxel::Air
        } else if y > FLOOR + 28.0 {
            WorldVoxel::Solid(SNOW)
        } else {
            WorldVoxel::Solid(GRASS)
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_voxel_world::edits::{Mask, Voxels};

    fn sparse(chunk: IVec3, seq: u64) -> EditDelta<u8> {
        EditDelta {
            chunk,
            seq,
            full: true,
            voxels: Voxels::Sparse(vec![
                (3, WorldVoxel::Air),
                (70, WorldVoxel::Solid(BRICK)),
            ]),
        }
    }

    fn dense(chunk: IVec3, seq: u64) -> EditDelta<u8> {
        let mut mask = Mask::default();
        for index in 0..4000 {
            mask.insert(index);
        }
        EditDelta {
            chunk,
            seq,
            full: true,
            voxels: Voxels::Dense {
                mask,
                voxels: vec![WorldVoxel::Air; 4000],
            },
        }
    }

    fn example_save() -> Save {
        Save::new(
            77,
            vec![
                dense(IVec3::new(1, 0, -2), 9000),
                sparse(IVec3::new(-1, 0, 0), 4),
                // A cleared chunk: no edits, at the sequence the clear left.
                EditDelta {
                    voxels: Voxels::Sparse(vec![]),
                    ..sparse(IVec3::new(0, 0, 0), 12)
                },
            ],
        )
    }

    #[test]
    fn a_save_decodes_to_what_was_encoded() {
        let (bytes, _) = encode(&example_save());
        assert_eq!(decode(&bytes), Ok(example_save()));
    }

    #[test]
    fn the_file_is_a_version_byte_then_lz4_of_postcard() {
        let save = example_save();
        let (bytes, encoded_len) = encode(&save);
        assert_eq!(bytes[0], FORMAT_VERSION);
        let encoded = lz4_flex::decompress_size_prepended(&bytes[1..]).unwrap();
        assert_eq!(encoded, postcard::to_allocvec(&save).unwrap());
        assert_eq!(encoded.len(), encoded_len);
        assert!(bytes.len() < encoded_len, "the dense delta compresses");
    }

    #[test]
    fn chunks_are_sorted_so_the_same_world_gives_the_same_bytes() {
        let mut deltas = vec![
            sparse(IVec3::new(2, 0, 0), 1),
            sparse(IVec3::new(-3, 1, 0), 2),
            sparse(IVec3::new(0, -1, 5), 3),
        ];
        let forward = encode(&Save::new(5, deltas.clone())).0;
        deltas.reverse();
        assert_eq!(encode(&Save::new(5, deltas)).0, forward);
    }

    #[test]
    fn an_unknown_version_is_rejected() {
        let (mut bytes, _) = encode(&example_save());
        bytes[0] = FORMAT_VERSION + 1;
        assert!(decode(&bytes)
            .unwrap_err()
            .contains("unknown format version"));
    }

    #[test]
    fn a_corrupt_or_empty_file_is_rejected() {
        let (bytes, _) = encode(&example_save());
        assert!(decode(&bytes[..bytes.len() / 2]).is_err());
        assert!(decode(&[FORMAT_VERSION, 1, 2, 3]).is_err());
        assert!(decode(&[]).is_err());
    }
}
