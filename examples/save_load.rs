//! Saves the voxel edits to a file and loads them back, with the terrain seed
//! and the sub-grids lifted out of the world.
//!
//! A save is every edited chunk's full delta: `edited_chunks` lists the
//! chunks, and `full_delta` gives each one's edits at its edit sequence. A
//! load forgets the world's edits with `forget_edits`, then applies each
//! saved full delta with `apply_delta`, so the saved sequences become the
//! world's. If a save exists when the example starts, it is loaded before any
//! chunk spawns, so every chunk takes its saved edits as it spawns.
//!
//! Space lifts the solid voxels around the cursor into a sub-grid, which
//! the example calls a piece. A piece rises a few voxels and turns slowly,
//! so a save catches it at a pose off the world's grid. Beside the world's
//! edits, the save holds for each piece the example's own id for it, its
//! `pose()`, the id of the piece it was lifted from, its spin, and the full
//! delta of each chunk in its `grid(g).edited_chunks()`. A piece with no
//! edits saves as its pose alone. A piece whose stamp has been called but
//! hasn't landed stays out of the save: its voxels are in the world's edits
//! already, and saved as its pose alone it would load as an empty piece.
//!
//! A load runs in one frame, in the order the crate's `edits` module docs
//! give:
//!
//! 1. `forget_edits`.
//! 2. Despawn every sub-grid. `forget_edits` leaves sub-grids alone, so a
//!    piece lifted since the save would otherwise outlive the load.
//! 3. Apply the world's full deltas.
//! 4. For each saved piece, `spawn_sub_grid` at its saved pose, then apply
//!    its full deltas through `grid_mut`, so the saved sequences become the
//!    piece's. `spawn_sub_grid` puts every sub-grid under the world's root,
//!    so a piece lifted from another one then moves under that piece,
//!    keeping its pose.
//!
//! Controls:
//!   - Left click to carve a small sphere, right click (or Shift + left
//!     click) to lay a blob of brick, in the world or in a piece
//!   - Space to lift the voxels around the cursor, out of the world or out
//!     of a piece
//!   - K to stamp the piece under the cursor back into the world
//!   - S to save, L to load the save again, dropping edits made since
//!   - N for a new world: a random seed, no edits and no pieces
//!   - C to clear every edit of the world
//!
//! C and N look alike but differ underneath. A clear is a decision about the
//! world: every edited chunk moves its edit sequence on and reports an empty
//! full delta, so other worlds and saves see the edits go, and a save made
//! after it restores the clear. A new world forgets: the edits and their
//! sequences are dropped without a word, as when switching to another world,
//! and a save made after it holds no chunks. Neither changes the pieces'
//! edits, but a new world despawns the pieces.
//!
//! The save file is one format version byte, then an `lz4_flex` block of the
//! `postcard` encoding of the seed, the world's full deltas and the pieces.
//! This is version 2. Version 1 held only the seed and the world's full
//! deltas, and still loads, as a world with no pieces. `EditDelta` carries no
//! version of its own, so the version lives here, in the file around it.
//! Poses are saved as arrays, since glam's serde support isn't enabled.
//!
//! Run with `cargo run --example save_load --features serde,sub_grids`. In
//! the browser the save is kept in memory, so save and load work until the
//! page reloads.

use std::{collections::HashSet, sync::Arc};

use bevy::{
    ecs::system::SystemParam, light::CascadeShadowConfigBuilder,
    platform::collections::HashMap, prelude::*,
};
use bevy_voxel_world::prelude::*;
use noise::{HybridMulti, NoiseFn, Perlin};
use serde::{de::DeserializeOwned, Deserialize, Serialize};

// Materials, and the layers of `example_voxel_texture.png` they show.
const SNOW: u8 = 0;
const BRICK: u8 = 1;
const GRASS: u8 = 2;

const CARVE_RADIUS: i32 = 6;
const BLOB_RADIUS: i32 = 4;

/// How far a lift reaches from the voxel under the cursor along each axis.
const LIFT_REACH: i32 = 3;
/// How many voxels a lifted piece rises, and how fast, in voxels per second.
const RISE: f32 = 4.0;
const RISE_SPEED: f32 = 2.0;
/// How fast a piece turns, in radians per second.
const TURN_RATE: f32 = 0.3;

/// How far the cursor's ray reaches, in voxels.
const RAY_LENGTH: f32 = 1000.0;

/// The height of the flat ground the low hills rise from. It lies inside
/// the level-0 chunks at y = 0, which spawn, rather than on top of the chunks
/// below them, which are hinted as solid and never spawn, so never show
/// their top face.
const FLOOR: f64 = 16.0;

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
        .add_plugins(DefaultPlugins)
        .add_plugins(VoxelWorldPlugin::with_config(SaveLoadWorld::default()))
        .init_resource::<SaveState>()
        .init_resource::<Stamping>()
        .init_resource::<NextPieceId>()
        .add_systems(Startup, (setup, load_at_startup))
        // A save reads committed edits only, so it runs before the brushes,
        // the lift and the stamp: it holds everything committed up to this
        // frame, and this frame's edits commit in the next frame's
        // `PreUpdate`. A load or a clear made here commits then too. Landed
        // stamps are read before the save, so it never holds a piece that
        // despawns this frame.
        .add_systems(
            Update,
            (
                track_unsaved,
                track_stamps,
                keys,
                brushes,
                lift_and_stamp,
                move_pieces,
                update_panel,
            )
                .chain(),
        )
        .run();
}

/// What the file holds: the terrain seed, the full delta of each edited
/// chunk of the world in chunk position order, and the pieces in id order.
#[derive(Serialize, Deserialize, PartialEq, Debug)]
struct Save {
    seed: u32,
    deltas: Vec<EditDelta<u8>>,
    pieces: Vec<SavedPiece>,
}

impl Save {
    /// The save of `deltas` and `pieces`, each sorted. `edited_chunks` lists
    /// chunks in no particular order, and sorting them makes the same world
    /// encode to the same bytes. Sorting the pieces by id puts each piece
    /// after the piece it was lifted from, which was lifted earlier, so a
    /// load spawns parents first.
    fn new(
        seed: u32,
        mut deltas: Vec<EditDelta<u8>>,
        mut pieces: Vec<SavedPiece>,
    ) -> Self {
        sort_by_chunk(&mut deltas);
        pieces.sort_by_key(|piece| piece.id);
        for piece in &mut pieces {
            sort_by_chunk(&mut piece.deltas);
        }
        Self {
            seed,
            deltas,
            pieces,
        }
    }
}

fn sort_by_chunk(deltas: &mut [EditDelta<u8>]) {
    deltas.sort_by_key(|delta| delta.chunk.to_array());
}

/// A piece in the save.
#[derive(Serialize, Deserialize, PartialEq, Debug)]
struct SavedPiece {
    /// The piece's `PieceId`.
    id: u32,
    /// The id of the piece it hangs under, or `None` under the world's root.
    parent: Option<u32>,
    /// Its `pose()`, against the world's root.
    pose: SavedPose,
    spin: Spin,
    /// The full delta of each chunk in its `edited_chunks`, in chunk
    /// position order.
    deltas: Vec<EditDelta<u8>>,
}

/// A pose as the arrays of its translation, rotation and scale.
#[derive(Serialize, Deserialize, PartialEq, Debug)]
struct SavedPose {
    translation: [f32; 3],
    rotation: [f32; 4],
    scale: [f32; 3],
}

impl From<&GlobalTransform> for SavedPose {
    fn from(pose: &GlobalTransform) -> Self {
        let (scale, rotation, translation) = pose.to_scale_rotation_translation();
        Self {
            translation: translation.to_array(),
            rotation: rotation.to_array(),
            scale: scale.to_array(),
        }
    }
}

impl SavedPose {
    fn transform(&self) -> Transform {
        Transform {
            translation: Vec3::from_array(self.translation),
            rotation: Quat::from_array(self.rotation),
            scale: Vec3::from_array(self.scale),
        }
    }
}

/// What a version 1 file holds: the seed and the world's full deltas.
#[derive(Deserialize)]
struct SaveV1 {
    seed: u32,
    deltas: Vec<EditDelta<u8>>,
}

/// The version of the file layout, written as the file's first byte. Bump it
/// when `Save` or the crate's `EditDelta` layout changes, and keep reading
/// the old versions a game has shipped.
const FORMAT_VERSION: u8 = 2;

/// The file bytes for `save`, and the size of its `postcard` encoding before
/// compression.
fn encode(save: &Save) -> (Vec<u8>, usize) {
    let encoded = postcard::to_allocvec(save).expect("a save always encodes");
    let mut bytes = vec![FORMAT_VERSION];
    bytes.extend(lz4_flex::compress_prepend_size(&encoded));
    (bytes, encoded.len())
}

/// The save in `bytes`, or why they hold none. A version 1 file decodes as a
/// save with no pieces. It decodes the whole file before anything touches
/// the world, so a file that fails leaves the world as it was.
fn decode(bytes: &[u8]) -> Result<Save, String> {
    match bytes.split_first() {
        None => Err("the file is empty".into()),
        Some((&FORMAT_VERSION, body)) => decode_body(body),
        Some((1, body)) => decode_body::<SaveV1>(body)
            .map(|save| Save::new(save.seed, save.deltas, Vec::new())),
        Some((version, _)) => Err(format!("unknown format version {version}")),
    }
}

/// The `postcard` value in the `lz4_flex` block `body`.
fn decode_body<T: DeserializeOwned>(body: &[u8]) -> Result<T, String> {
    let encoded = lz4_flex::decompress_size_prepended(body)
        .map_err(|error| format!("corrupt file: {error}"))?;
    postcard::from_bytes(&encoded).map_err(|error| format!("corrupt file: {error}"))
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

/// The example's own id for a piece, which names it in the save. The crate
/// gives sub-grids no id that holds across a save and a load.
#[derive(Component, Clone, Copy, PartialEq, Eq, Debug)]
struct PieceId(u32);

/// The id the next lifted piece gets.
#[derive(Resource, Default)]
struct NextPieceId(u32);

/// How a piece moves, saved as its game data: it rises `rise_left` more
/// voxels, and turns at `rate` radians per second about the vertical axis
/// through the centre of its solid voxels.
#[derive(Component, Serialize, Deserialize, Clone, Copy, PartialEq, Debug)]
struct Spin {
    rate: f32,
    rise_left: f32,
}

/// The pieces whose stamp has been called and whose `Stamped` hasn't been
/// read yet. From the call on, their voxels are in the world's edits, so a
/// save leaves them out.
#[derive(Resource, Default)]
struct Stamping(HashSet<Entity>);

/// The pieces, and what saving, loading, lifting and stamping change about
/// them beside the voxel world.
#[derive(SystemParam)]
struct Pieces<'w, 's> {
    commands: Commands<'w, 's>,
    pieces: Query<'w, 's, (Entity, &'static PieceId, &'static Spin, &'static ChildOf)>,
    sub_grids: Query<'w, 's, Entity, With<SubGrid<SaveLoadWorld>>>,
    stamping: ResMut<'w, Stamping>,
    next_id: ResMut<'w, NextPieceId>,
}

impl Pieces<'_, '_> {
    /// Despawns every sub-grid of the world, and forgets the stamps not yet
    /// landed, since a sub-grid despawned before its stamp lands gets no
    /// `Stamped`.
    fn despawn_all(&mut self) {
        for sub_grid in &self.sub_grids {
            self.commands.entity(sub_grid).try_despawn();
        }
        self.stamping.0.clear();
    }
}

/// Whether the world differs from the save, and the line the panel shows
/// about the last thing that happened.
#[derive(Resource, Default)]
struct SaveState {
    /// Each chunk's edit sequence at the last save or load, by grid.
    saved: HashMap<(Grid, IVec3), u64>,
    /// The chunks that committed at a sequence other than the saved one.
    unsaved: HashSet<(Grid, IVec3)>,
    /// Set by a new world, which changes the seed and forgets the edits
    /// without committing anything.
    new_world: bool,
    last_event: String,
}

impl SaveState {
    /// Marks the world as matching the save whose chunks had `saved`.
    fn mark_saved(&mut self, saved: HashMap<(Grid, IVec3), u64>) {
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
    mut pieces: Pieces,
) {
    load(&mut config, &mut voxel_world, &mut state, &mut pieces);
}

/// Reads the save and loads it into the world in the order the module docs
/// give. It regenerates the chunks only when the seed changes; forgetting
/// already regenerates the chunks that showed an edit.
///
/// A piece lifted from another piece moves under it with commands that
/// replace the root parent `spawn_sub_grid` gives it. The commands of a
/// system's parameters apply in the parameters' order, so a system calling
/// this declares `voxel_world` before `pieces`.
fn load(
    config: &mut SaveLoadWorld,
    voxel_world: &mut VoxelWorld<SaveLoadWorld>,
    state: &mut SaveState,
    pieces: &mut Pieces,
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
    pieces.despawn_all();
    let mut saved = HashMap::new();
    let world_chunks = save.deltas.len();
    for delta in save.deltas {
        let (chunk, seq) = (delta.chunk, delta.seq);
        // A full delta always continues a forgotten chunk; apply refuses only
        // one whose voxels are malformed.
        if voxel_world.apply_delta(delta).is_ok() {
            saved.insert((Grid::World, chunk), seq);
        }
    }

    let piece_count = save.pieces.len();
    let mut loaded = HashMap::<u32, (Entity, GlobalTransform)>::new();
    for piece in save.pieces {
        let pose = piece.pose.transform();
        let sub_grid = voxel_world.spawn_sub_grid(pose);
        let mut grid = voxel_world
            .grid_mut(Grid::SubGrid(sub_grid))
            .expect("a spawned sub-grid is live at once");
        for delta in piece.deltas {
            let (chunk, seq) = (delta.chunk, delta.seq);
            // A fresh sub-grid has no chunks, so every full delta continues
            // one.
            if grid.apply_delta(delta).is_ok() {
                saved.insert((Grid::SubGrid(sub_grid), chunk), seq);
            }
        }
        let mut entity = pieces.commands.entity(sub_grid);
        entity.insert((PieceId(piece.id), piece.spin));
        // A parent missing from the save leaves the piece under the root.
        let parent = piece.parent.and_then(|id| loaded.get(&id));
        if let Some(&(parent, parent_pose)) = parent {
            entity.insert((
                ChildOf(parent),
                GlobalTransform::from(pose).reparented_to(&parent_pose),
            ));
        }
        loaded.insert(piece.id, (sub_grid, pose.into()));
    }
    pieces.next_id.0 = loaded.keys().max().map_or(0, |id| id + 1);

    if seed_changed {
        voxel_world.regenerate_chunks();
    }
    state.report(format!(
        "Loaded {world_chunks} chunks and {piece_count} pieces, seed {}",
        save.seed
    ));
    state.mark_saved(saved);
}

/// Writes the seed, every edited chunk's full delta, and each piece but those
/// being stamped. A chunk whose edits were all cleared is saved too, with
/// none, so that the clear survives a load.
///
/// It rewrites the whole file each time. A game with a larger world writes
/// only the unsaved chunks, to a store with a record per chunk or region.
fn save(
    seed: u32,
    voxel_world: &VoxelWorld<SaveLoadWorld>,
    state: &mut SaveState,
    pieces: &Pieces,
) {
    let mut saved = HashMap::new();
    let world = voxel_world.grid(Grid::World).expect("the world is a grid");
    let deltas = full_deltas(&world, Grid::World, &mut saved);
    let world_chunks = deltas.len();

    let saving = |entity: &Entity| !pieces.stamping.0.contains(entity);
    let ids: HashMap<Entity, u32> = pieces
        .pieces
        .iter()
        .filter(|(entity, ..)| saving(entity))
        .map(|(entity, id, ..)| (entity, id.0))
        .collect();
    let mut records = Vec::new();
    for (entity, id, spin, child_of) in &pieces.pieces {
        let name = Grid::SubGrid(entity);
        let Some(grid) = voxel_world.grid(name).filter(|_| saving(&entity)) else {
            continue;
        };
        records.push(SavedPiece {
            id: id.0,
            // A piece under one being stamped saves under the root, where
            // the stamp moves it when it lands.
            parent: ids.get(&child_of.parent()).copied(),
            pose: SavedPose::from(&grid.pose()),
            spin: *spin,
            deltas: full_deltas(&grid, name, &mut saved),
        });
    }

    let piece_count = records.len();
    let (bytes, encoded_len) = encode(&Save::new(seed, deltas, records));
    if let Err(error) = storage::write(&bytes) {
        state.report(format!("Can't write the save: {error}"));
        return;
    }
    state.report(format!(
        "Saved {world_chunks} chunks and {piece_count} pieces: {encoded_len} bytes, {} compressed ({})",
        bytes.len(),
        storage::place()
    ));
    state.mark_saved(saved);
}

/// The full delta of each edited chunk of `grid`, named `name`, recording
/// each chunk's sequence in `saved`.
fn full_deltas(
    grid: &GridVoxels<SaveLoadWorld>,
    name: Grid,
    saved: &mut HashMap<(Grid, IVec3), u64>,
) -> Vec<EditDelta<u8>> {
    grid.edited_chunks()
        .into_iter()
        .map(|(chunk, seq)| {
            saved.insert((name, chunk), seq);
            grid.full_delta(chunk)
        })
        .collect()
}

/// Marks each chunk, of the world or of a piece, that commits at a sequence
/// other than the saved one as unsaved, and one that commits back at it as
/// saved.
///
/// It compares sequences rather than counting messages: a load's own commits
/// also write `EditsCommitted` and `SubGridEditsCommitted`, at exactly the
/// saved sequences.
fn track_unsaved(
    mut world_commits: MessageReader<EditsCommitted<SaveLoadWorld>>,
    mut sub_grid_commits: MessageReader<SubGridEditsCommitted<SaveLoadWorld>>,
    mut state: ResMut<SaveState>,
) {
    let world = world_commits
        .read()
        .map(|commit| (Grid::World, &commit.delta));
    let sub_grids = sub_grid_commits
        .read()
        .map(|commit| (Grid::SubGrid(commit.sub_grid), &commit.delta));
    for (grid, delta) in world.chain(sub_grids) {
        let key = (grid, delta.chunk);
        if state.saved.get(&key) == Some(&delta.seq) {
            state.unsaved.remove(&key);
        } else {
            state.unsaved.insert(key);
        }
    }
}

/// Forgets each piece whose stamp has landed. Its voxels show in the world
/// from this frame, and the crate despawns it in this frame's `PostUpdate`,
/// so removing its `PieceId` keeps a save made before then from holding it.
fn track_stamps(
    mut stamped: MessageReader<Stamped<SaveLoadWorld>>,
    mut stamping: ResMut<Stamping>,
    mut commands: Commands,
) {
    for stamp in stamped.read() {
        stamping.0.remove(&stamp.sub_grid);
        commands
            .entity(stamp.sub_grid)
            .try_remove::<(PieceId, Spin)>();
    }
}

fn keys(
    keys: Res<ButtonInput<KeyCode>>,
    mut config: ResMut<SaveLoadWorld>,
    mut voxel_world: VoxelWorld<SaveLoadWorld>,
    mut state: ResMut<SaveState>,
    mut pieces: Pieces,
) {
    if keys.just_pressed(KeyCode::KeyS) {
        save(config.seed, &voxel_world, &mut state, &pieces);
    }
    if keys.just_pressed(KeyCode::KeyL) {
        load(&mut config, &mut voxel_world, &mut state, &mut pieces);
    }
    if keys.just_pressed(KeyCode::KeyN) {
        config.seed = rand::random();
        voxel_world.forget_edits();
        pieces.despawn_all();
        voxel_world.regenerate_chunks();
        state.saved.clear();
        state.unsaved.clear();
        state.new_world = true;
        state.report(format!("New world, seed {}", config.seed));
    }
    if keys.just_pressed(KeyCode::KeyC) {
        voxel_world.clear_edits();
        state.report("Cleared every edit of the world".into());
    }
}

/// The ray from the camera through the cursor.
#[derive(SystemParam)]
struct Cursor<'w, 's> {
    window: Query<'w, 's, &'static Window>,
    camera: Query<
        'w,
        's,
        (&'static Camera, &'static GlobalTransform),
        With<VoxelWorldCamera<SaveLoadWorld>>,
    >,
}

impl Cursor<'_, '_> {
    /// The nearest voxel the cursor points at, in the world or in a piece.
    fn hit(&self, voxel_world: &VoxelWorld<SaveLoadWorld>) -> Option<GridRaycastHit> {
        let (window, (camera, camera_transform)) =
            (self.window.single().ok()?, self.camera.single().ok()?);
        let cursor = window.cursor_position()?;
        let ray = camera.viewport_to_world(camera_transform, cursor).ok()?;
        voxel_world.raycast_grids(ray, RAY_LENGTH, &|_| true, &|_, _, _| true)
    }
}

/// Left click carves a sphere of air where the cursor points, and right
/// click, or Shift + left click, lays a blob of brick on the surface there.
/// Each edits the grid the cursor's ray hits, the world or a piece, around
/// the hit's position on that grid. Each sets a box of voxels in one
/// `set_voxels_in` call, leaving those outside the sphere alone, so a few
/// clicks give a chunk enough edits to be saved in the dense form.
fn brushes(
    buttons: Res<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
    cursor: Cursor,
    mut voxel_world: VoxelWorld<SaveLoadWorld>,
) {
    let shift = keys.any_pressed([KeyCode::ShiftLeft, KeyCode::ShiftRight]);
    let lay = buttons.just_pressed(MouseButton::Right)
        || (shift && buttons.just_pressed(MouseButton::Left));
    if !lay && !buttons.just_pressed(MouseButton::Left) {
        return;
    }
    let Some(hit) = cursor.hit(&voxel_world) else {
        return;
    };

    let (center, radius, voxel) = if lay {
        let normal = hit.normal.unwrap_or(IVec3::Y);
        (
            hit.position + normal * (BLOB_RADIUS / 2),
            BLOB_RADIUS,
            WorldVoxel::Solid(BRICK),
        )
    } else {
        (hit.position, CARVE_RADIUS, WorldVoxel::Air)
    };
    let Some(mut grid) = voxel_world.grid_mut(hit.grid) else {
        return;
    };
    grid.set_voxels_in(
        center - IVec3::splat(radius),
        center + IVec3::splat(radius + 1),
        |position| {
            ((position - center).length_squared() <= radius * radius).then_some(voxel)
        },
    );
}

/// Space lifts every solid voxel in a small box around the voxel under the
/// cursor into a new piece, out of the grid holding that voxel. A piece
/// lifted out of another piece is its child, so it turns with it. K stamps
/// the piece under the cursor into the world, at the nearest pose that
/// lines up with the world's voxels.
fn lift_and_stamp(
    keys: Res<ButtonInput<KeyCode>>,
    cursor: Cursor,
    mut voxel_world: VoxelWorld<SaveLoadWorld>,
    mut pieces: Pieces,
    mut state: ResMut<SaveState>,
) {
    let lift = keys.just_pressed(KeyCode::Space);
    if !lift && !keys.just_pressed(KeyCode::KeyK) {
        return;
    }
    let Some(hit) = cursor.hit(&voxel_world) else {
        return;
    };

    if lift {
        let reach = IVec3::splat(LIFT_REACH);
        let lifted = voxel_world.lift(
            hit.grid,
            hit.position - reach,
            hit.position + reach + 1,
            |_, voxel| voxel.is_solid(),
        );
        match lifted {
            Ok(sub_grid) => {
                let id = pieces.next_id.0;
                pieces.next_id.0 += 1;
                // Neighbouring pieces turn opposite ways, so they read apart.
                let rate = if id.is_multiple_of(2) {
                    TURN_RATE
                } else {
                    -TURN_RATE
                };
                pieces.commands.entity(sub_grid).insert((
                    PieceId(id),
                    Spin {
                        rate,
                        rise_left: RISE,
                    },
                ));
                state.report(format!("Lifted piece {id}"));
            }
            Err(error) => state.report(format!("Lifted nothing: {error}")),
        }
    } else if let Grid::SubGrid(sub_grid) = hit.grid {
        if pieces.stamping.0.contains(&sub_grid) {
            return;
        }
        match voxel_world.stamp(sub_grid, Grid::World, incoming_replaces) {
            Ok(()) => {
                pieces.stamping.0.insert(sub_grid);
                state.report("Stamped a piece into the world".into());
            }
            Err(error) => state.report(format!("Stamped nothing: {error}")),
        }
    }
}

/// Raises each piece by what is left of its rise, and turns it about the
/// vertical axis through the centre of its solid voxels. A piece with no
/// solid voxel, such as one being stamped, holds still.
fn move_pieces(
    time: Res<Time>,
    voxel_world: VoxelWorld<SaveLoadWorld>,
    mut pieces: Query<(Entity, &mut Transform, &mut Spin)>,
) {
    let elapsed = time.delta_secs();
    for (entity, mut transform, mut spin) in &mut pieces {
        let Some((min, max)) = voxel_world
            .grid(Grid::SubGrid(entity))
            .and_then(|grid| grid.bounds())
        else {
            continue;
        };
        let rise = spin.rise_left.min(RISE_SPEED * elapsed);
        spin.rise_left -= rise;
        transform.translation.y += rise;
        let centre = transform.transform_point((min + max).as_vec3() / 2.0);
        transform.rotate_around(centre, Quat::from_rotation_y(spin.rate * elapsed));
    }
}

#[derive(Component)]
struct Panel;

fn update_panel(
    config: Res<SaveLoadWorld>,
    voxel_world: VoxelWorld<SaveLoadWorld>,
    state: Res<SaveState>,
    pieces: Query<(), With<PieceId>>,
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
         Space lift  K stamp piece\n\
         Seed {}, {} edited chunks, {} pieces\n{}\n{}",
        config.seed,
        voxel_world.edited_chunks().len(),
        pieces.iter().count(),
        saved,
        state.last_event
    );
}

fn setup(mut commands: Commands) {
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(-50.0, FLOOR as f32 + 40.0, -50.0)
            .looking_at(Vec3::new(0.0, FLOOR as f32, 0.0), Vec3::Y),
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

/// Low hills from `seed` on flat ground, grass with snow on the tops.
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
            // leaves wide flat stretches between the hills, where edits are
            // easiest to see.
            (FLOOR + sample * 12.0).clamp(FLOOR, FLOOR + 8.0)
        });
        let y = pos.y as f64;
        if y >= height {
            WorldVoxel::Air
        } else if y > FLOOR + 5.0 {
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

    fn world_deltas() -> Vec<EditDelta<u8>> {
        vec![
            dense(IVec3::new(1, 0, -2), 9000),
            sparse(IVec3::new(-1, 0, 0), 4),
            // A cleared chunk: no edits, at the sequence the clear left.
            EditDelta {
                voxels: Voxels::Sparse(vec![]),
                ..sparse(IVec3::new(0, 0, 0), 12)
            },
        ]
    }

    fn example_pieces() -> Vec<SavedPiece> {
        let turned = Transform::from_xyz(12.5, 20.25, -3.0)
            .with_rotation(Quat::from_rotation_y(0.7));
        vec![
            SavedPiece {
                id: 4,
                parent: Some(1),
                pose: SavedPose::from(&GlobalTransform::from(
                    turned.with_scale(Vec3::splat(0.5)),
                )),
                spin: Spin {
                    rate: -0.25,
                    rise_left: 1.5,
                },
                deltas: vec![sparse(IVec3::ZERO, 2)],
            },
            SavedPiece {
                id: 1,
                parent: None,
                pose: SavedPose::from(&GlobalTransform::from(turned)),
                spin: Spin {
                    rate: 0.3,
                    rise_left: 0.0,
                },
                deltas: vec![sparse(IVec3::new(0, 1, 0), 7), dense(IVec3::ZERO, 3)],
            },
            // An empty piece saves as its pose alone.
            SavedPiece {
                id: 2,
                parent: None,
                pose: SavedPose::from(&GlobalTransform::IDENTITY),
                spin: Spin {
                    rate: 0.3,
                    rise_left: 0.0,
                },
                deltas: vec![],
            },
        ]
    }

    fn example_save() -> Save {
        Save::new(77, world_deltas(), example_pieces())
    }

    #[test]
    fn a_save_decodes_to_what_was_encoded() {
        let (bytes, _) = encode(&example_save());
        assert_eq!(decode(&bytes), Ok(example_save()));
    }

    #[test]
    fn a_saved_piece_keeps_its_id_parent_pose_spin_and_deltas() {
        let (bytes, _) = encode(&example_save());
        let save = decode(&bytes).unwrap();
        let child = save.pieces.iter().find(|piece| piece.id == 4).unwrap();
        assert_eq!(child.parent, Some(1));
        assert_eq!(child.pose.translation, [12.5, 20.25, -3.0]);
        assert_eq!(child.pose.scale, [0.5; 3]);
        let pose = child.pose.transform();
        assert!(pose.rotation.angle_between(Quat::from_rotation_y(0.7)) < 1e-6);
        assert_eq!(child.spin.rate, -0.25);
        assert_eq!(child.spin.rise_left, 1.5);
        assert_eq!(child.deltas, vec![sparse(IVec3::ZERO, 2)]);
        let empty = save.pieces.iter().find(|piece| piece.id == 2).unwrap();
        assert!(empty.deltas.is_empty());
    }

    #[test]
    fn pieces_are_saved_by_id_so_a_parent_loads_before_its_children() {
        let ids: Vec<u32> = example_save().pieces.iter().map(|piece| piece.id).collect();
        assert_eq!(ids, [1, 2, 4]);
        let root = &example_save().pieces[0];
        let chunks: Vec<IVec3> = root.deltas.iter().map(|delta| delta.chunk).collect();
        assert_eq!(chunks, [IVec3::ZERO, IVec3::new(0, 1, 0)]);
    }

    #[test]
    fn a_version_1_file_loads_as_the_world_alone() {
        // Version 1 held the seed and the world's full deltas, in that order.
        let encoded = postcard::to_allocvec(&(77u32, world_deltas())).unwrap();
        let mut bytes = vec![1];
        bytes.extend(lz4_flex::compress_prepend_size(&encoded));
        assert_eq!(decode(&bytes), Ok(Save::new(77, world_deltas(), vec![])));
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
        let forward = encode(&Save::new(5, deltas.clone(), example_pieces())).0;
        deltas.reverse();
        let mut pieces = example_pieces();
        pieces.reverse();
        assert_eq!(encode(&Save::new(5, deltas, pieces)).0, forward);
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
