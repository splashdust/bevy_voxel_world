//! A server and two clients in one app, keeping their voxel edits and their
//! sub-grids in sync through messages sent over a fake in-memory network.
//!
//! The server is a `VoxelWorldPlugin::minimal()` world with no camera: it holds
//! the edits and their edit sequences, commits them and writes
//! `EditsCommitted`, but streams no chunks. A server that needs collision or
//! other voxel queries around its players gives each player a
//! `VoxelWorldCamera` of the server world instead. Clients A and B render the
//! same terrain from two angles, A on the left half of the window and B on the
//! right.
//!
//! Hover a voxel and it gets a white outline. Left click asks the server to dig
//! it, right click to place the client's own material beside it (A lays brick,
//! B lays snow). Clicks edit whichever grid the cursor's ray hits through
//! `raycast_grids`, the world or a piece. Clients show confirmed edits only: a
//! yellow outline marks a request until a delta carrying that voxel arrives,
//! and a red one a request the server refused. The server accepts edits inside
//! the ring drawn on the ground. The log at the bottom narrates everything that
//! is not a plain delivery.
//!
//! A piece is the game's name for a sub-grid it syncs, and each piece has a
//! game id the server gives out. Requests and messages name pieces by that id,
//! and each world maps it to its own entity:
//!
//! - `L` asks the server to lift the solid voxels in a small box around the
//!   hovered voxel, out of the grid holding it. The server raises the new
//!   piece, then drifts it on a slow circle through the corner of the ring
//!   that client B's interest leaves out. A piece lifted out of another piece
//!   rises with it and doesn't drift on its own.
//! - `K` asks the server to stamp the hovered piece into the world where it
//!   hangs.
//!
//! The lane to client A is clean. The lane to client B misbehaves on request:
//!
//! - `1` delays every message to and from B.
//! - `2` drops some messages to B.
//! - `3` reorders messages to B.
//! - `J` makes B forget its edits, despawn its pieces and join again.
//! - `R` makes the server clear every edit of the world and despawn every
//!   piece.
//! - `V` cycles the view: split, A only, B only.
//! - `Tab` switches the client panels between chunks and pieces.
//!
//! The server sends each client one message per frame, holding the frame's
//! piece spawns, piece poses, world deltas, piece deltas by game id, piece
//! despawns, the heartbeat and refused requests. So a client applies all of
//! one server commit in one system, and lands each handover in it as the
//! server did. Every message is `postcard` bytes, so swapping the lane for a
//! real transport changes only `Lane`. The lane assumes what a real reliable,
//! ordered channel gives, unless a fault key is on. The drop fault never drops
//! a message carrying a spawn, a full delta or a refused request: nothing
//! later would repair it.
//!
//! How a client recovers, as game messages built on the crate's API:
//!
//! - **Gap**: `apply_delta` returns an `EditGap` when a delta doesn't continue
//!   from the client's edit sequence for its chunk. The client asks for the
//!   chunk's `full_delta`, naming the world or the piece's game id.
//! - **Join**: the server sends the `full_delta` of every chunk in
//!   `edited_chunks()` that the client's interest covers, and every piece in
//!   the client's interest whole. A client joining again calls
//!   `forget_edits` and despawns its pieces first.
//! - **Heartbeat**: now and then the server sends each client its
//!   `edited_chunks()`, and each piece the client should hold with its
//!   `grid(g).edited_chunks()`. The client asks for the full delta of any
//!   chunk whose sequence it is behind on, asks to join a piece it lacks, and
//!   despawns one the server no longer sends it. Only this catches a dropped
//!   last delta or despawn, since nothing after it shows the gap.
//! - **Reset**: `clear_edits` on the server reaches clients as empty full
//!   deltas, through the same forwarding as any other edit, and each piece
//!   as a despawn.
//!
//! Lift and stamp travel as their parts, never as calls for a client to
//! repeat. A lift is the piece's spawn and pose, its deltas and the source
//! grid's deltas setting air, in one message. A stamp is the world's deltas,
//! the piece's deltas emptying it and its despawn, in one message; the client
//! routes the despawn through `despawn_sub_grid`, so the piece shows until the
//! world's chunks taking its voxels do. Pieces under a stamped piece move
//! under the world's root at once, keeping their pose.
//!
//! Interest takes a piece whole: a piece and every piece under it are in a
//! client's interest while the chunk column holding the centre of the top
//! piece's voxels is. A piece that leaves a client's interest despawns there,
//! and coming back is a fresh join of it: its spawn, its pose and the full
//! delta of every chunk in its `edited_chunks()`, empty ones included.
//!
//! Run with `cargo run --example multiplayer_loopback --features
//! serde,sub_grids`. With `-- audit` it plays a script instead of taking
//! input. Through each fault, a re-join and a reset, both clients edit the
//! world, lift, edit and stamp pieces, and a piece drifts across the edge of
//! client B's interest. Each step ends by holding every piece still until
//! the lane settles. Then the audit checks both clients against the server,
//! and the run exits with an error if any differ:
//!
//! - The world's edits, and the voxel `get_voxel` reads wherever the script
//!   set or lifted one.
//! - Each piece by game id: the same set of pieces within the client's
//!   interest, and for each the same parent and pose, the same full delta of
//!   every chunk in its `edited_chunks()`, the same voxel read wherever the
//!   script set or lifted one, and the same `raycast_grids` hit on a ray cast
//!   down onto it.
//! - On client A only, whose lane is clean: that every chunk a server
//!   message's deltas touch, padding neighbours included, publishes or
//!   retires in one frame when the message holds a handover. B's faults can
//!   split a handover into parts until the heartbeat heals it, so B is only
//!   checked to converge.
//!
//! `-- audit <file.png>` also saves a frame of both views partway through,
//! with a lifted piece hanging in both.

use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    marker::PhantomData,
    sync::Arc,
};

use bevy::{
    app::{HierarchyPropagatePlugin, Propagate},
    camera::{visibility::RenderLayers, Viewport},
    ecs::system::SystemParam,
    light::CascadeShadowConfigBuilder,
    platform::collections::{HashMap, HashSet},
    prelude::*,
    render::view::screenshot::{save_to_disk, Screenshot},
    ui::UiTargetCamera,
    window::PrimaryWindow,
};
use bevy_voxel_world::prelude::*;
use noise::{HybridMulti, NoiseFn, Perlin};
use serde::{Deserialize, Serialize};

// Materials. The default texture atlas holds grass, brick, stone and snow,
// from the bottom.
const GRASS: u8 = 0;
const STONE: u8 = 1;
const BRICK: u8 = 2;
const SNOW: u8 = 3;

/// The centre and radius of the ring the server accepts edits inside.
const ARENA: Vec3 = Vec3::ZERO;
const REACH: f32 = 64.0;
/// Both clients care about chunks within this many chunks of the origin,
/// horizontally, except the corner client B leaves out.
const INTEREST_CHUNKS: i32 = 3;
/// The point both cameras look at.
const LOOK_AT: Vec3 = Vec3::new(0.0, 2.0, 0.0);
/// How far a ray from the cursor reaches.
const RAY_LENGTH: f32 = 500.0;

/// How far a lift reaches from the hovered voxel along each axis.
const LIFT_REACH: i32 = 2;
/// How many voxels a lifted piece rises, and how far it rises each frame.
const RISE: f32 = 10.0;
const RISE_STEP: f32 = 0.05;
/// How far a drifting piece moves each frame, in voxels.
const DRIFT_STEP: f32 = 0.1;
/// A point inside the corner client B's interest leaves out. A piece drifts
/// on the circle whose diameter runs from where it was lifted to here, so the
/// circle crosses the edge of B's interest.
const DRIFT_THROUGH: Vec2 = Vec2::new(48.0, 48.0);

/// How many frames the delay fault adds to a message.
const DELAY_FRAMES: u32 = 45;
/// Up to how many frames the reorder fault adds to a message.
const REORDER_FRAMES: u32 = 12;
/// The share of messages the drop fault loses.
const DROP_PERCENT: u32 = 30;
/// How often the server sends each client its edited chunks and pieces.
const HEARTBEAT_FRAMES: u32 = 90;
const FLASH_FRAMES: u32 = 40;
const LOG_LINES: usize = 7;

// ---------------------------------------------------------------------------
// Three worlds: one server, two clients.

#[derive(Resource, Clone, Default)]
struct Server;
#[derive(Resource, Clone, Default)]
struct ClientA;
#[derive(Resource, Clone, Default)]
struct ClientB;

#[derive(Default, Reflect, GizmoConfigGroup)]
struct GizmosA;
#[derive(Default, Reflect, GizmoConfigGroup)]
struct GizmosB;

/// What tells the two clients apart: a lane slot, a render layer, a gizmo
/// group drawn on that layer, a camera position and the material it places.
trait Client: VoxelWorldConfig<MaterialIndex = u8> + Default {
    type Gizmos: GizmoConfigGroup + Default;
    const ID: usize;
    const LAYER: usize;
    const NAME: &'static str;
    const CAMERA: Vec3;
    const PLACES: u8;
    const PLACES_NAME: &'static str;
}

impl Client for ClientA {
    type Gizmos = GizmosA;
    const ID: usize = 0;
    const LAYER: usize = 1;
    const NAME: &'static str = "client A";
    const CAMERA: Vec3 = Vec3::new(-44.0, 30.0, -44.0);
    const PLACES: u8 = BRICK;
    const PLACES_NAME: &'static str = "brick (red)";
}

impl Client for ClientB {
    type Gizmos = GizmosB;
    const ID: usize = 1;
    const LAYER: usize = 2;
    const NAME: &'static str = "client B";
    const CAMERA: Vec3 = Vec3::new(44.0, 30.0, -44.0);
    const PLACES: u8 = SNOW;
    const PLACES_NAME: &'static str = "snow (white)";
}

const CLIENTS: [usize; 2] = [ClientA::ID, ClientB::ID];

fn name(client: usize) -> &'static str {
    [ClientA::NAME, ClientB::NAME][client]
}

/// Marks the root entity of the world `C`, so pieces can move back under it.
#[derive(Component)]
struct Root<C>(PhantomData<fn() -> C>);

macro_rules! world_config {
    ($world:ty, $layer:expr) => {
        impl VoxelWorldConfig for $world {
            type MaterialIndex = u8;
            type ChunkUserBundle = ();

            fn spawning_distance(&self) -> u32 {
                5
            }

            fn voxel_lookup_delegate(&self) -> VoxelLookupDelegate<u8> {
                Box::new(|_chunk_pos, _lod, _previous| generator())
            }

            fn texture_index_mapper(&self) -> Arc<dyn Fn(u8) -> [u32; 3] + Send + Sync> {
                Arc::new(|mat| match mat {
                    GRASS => [0; 3],
                    STONE => [2; 3],
                    BRICK => [1; 3],
                    _ => [3; 3],
                })
            }

            fn init_root(&self, mut commands: Commands, root: Entity) {
                let mut root = commands.entity(root);
                root.insert(Root::<$world>(PhantomData));
                // Chunks and pieces are children of the root, so one
                // component on the root puts every chunk of this world on its
                // client's layer, a piece's chunks included.
                let layer: Option<usize> = $layer;
                if let Some(layer) = layer {
                    root.insert(Propagate(RenderLayers::layer(layer)));
                }
            }
        }
    };
}

world_config!(Server, None);
world_config!(ClientA, Some(ClientA::LAYER));
world_config!(ClientB, Some(ClientB::LAYER));

// ---------------------------------------------------------------------------
// The messages and the fake network.

/// A grid as messages name it: the world, or a piece by its game id.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum GridName {
    World,
    Piece(u32),
}

/// A piece's `Transform` relative to its parent. Positions are arrays because
/// glam's own serde support is a feature this example doesn't enable.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
struct Pose {
    translation: [f32; 3],
    rotation: [f32; 4],
    scale: [f32; 3],
}

impl From<Transform> for Pose {
    fn from(transform: Transform) -> Self {
        Self {
            translation: transform.translation.to_array(),
            rotation: transform.rotation.to_array(),
            scale: transform.scale.to_array(),
        }
    }
}

impl From<Pose> for Transform {
    fn from(pose: Pose) -> Self {
        Transform {
            translation: Vec3::from_array(pose.translation),
            rotation: Quat::from_array(pose.rotation),
            scale: Vec3::from_array(pose.scale),
        }
    }
}

/// Where a piece is: under the world's root (`parent: None`) or under another
/// piece, and its pose relative to that parent.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
struct PieceAt {
    piece: u32,
    parent: Option<u32>,
    pose: Pose,
}

/// A message from a client to the server.
#[derive(Serialize, Deserialize)]
enum ToServer {
    /// Asks the server to set a voxel of a grid. The server checks it first.
    Set {
        grid: GridName,
        pos: [i32; 3],
        voxel: WorldVoxel<u8>,
    },
    /// Asks to lift the solid voxels in the box from `min` inclusive to
    /// `max` exclusive out of a grid.
    Lift {
        grid: GridName,
        min: [i32; 3],
        max: [i32; 3],
    },
    /// Asks to stamp a piece into the world.
    Stamp { piece: u32 },
    /// Asks for a chunk's full delta, after a gap or a heartbeat showed the
    /// client is behind on it.
    RequestFull { grid: GridName, chunk: [i32; 3] },
    /// Asks for every edited chunk and every piece the client's interest
    /// covers.
    Join,
    /// Asks for one piece whole, after a heartbeat listed a piece the client
    /// lacks.
    JoinPiece { piece: u32 },
}

/// The heartbeat: the server's edited chunks in the client's interest, and
/// the pieces the client should hold with their edited chunks, each chunk
/// with its edit sequence.
#[derive(Serialize, Deserialize, Default)]
struct Heartbeat {
    world: EditedChunks,
    pieces: Vec<(u32, EditedChunks)>,
}

/// Chunk positions with their edit sequences.
type EditedChunks = Vec<([i32; 3], u64)>;

/// Everything the server sends one client in one frame. A client applies the
/// parts in the order of the fields.
#[derive(Serialize, Deserialize, Default)]
struct ServerFrame {
    /// The server's frame, which orders poses, spawns and despawns that the
    /// reorder fault mixed up.
    frame: u32,
    /// Pieces joining the client, parents first.
    spawns: Vec<PieceAt>,
    deltas: Vec<(GridName, EditDelta<u8>)>,
    /// The pose of every piece the client holds.
    poses: Vec<PieceAt>,
    despawns: Vec<u32>,
    heartbeat: Option<Heartbeat>,
    /// Voxels whose requests the server refused.
    rejected: Vec<(GridName, [i32; 3])>,
}

impl ServerFrame {
    fn is_empty(&self) -> bool {
        self.spawns.is_empty()
            && self.deltas.is_empty()
            && self.poses.is_empty()
            && self.despawns.is_empty()
            && self.heartbeat.is_none()
            && self.rejected.is_empty()
    }

    /// Whether losing this message would leave something no later message
    /// repairs: a spawn, a full delta or a refusal.
    fn must_arrive(&self) -> bool {
        !self.spawns.is_empty()
            || !self.rejected.is_empty()
            || self.deltas.iter().any(|(_, delta)| delta.full)
    }
}

/// One message on its way, as the bytes a transport would carry.
struct Packet {
    deliver_at: u32,
    bytes: Vec<u8>,
}

/// The in-memory network between the server and the clients, indexed by
/// `Client::ID`, with the faults on the lane to client B, the server's
/// messages for this frame, and a log of what happened on it.
#[derive(Resource, Default)]
struct Lane {
    frame: u32,
    to_server: [VecDeque<Packet>; 2],
    to_client: [Vec<Packet>; 2],
    /// What the server sends each client at the end of this frame.
    outbox: [ServerFrame; 2],
    delay: bool,
    drop: bool,
    reorder: bool,
    stats: Stats,
    log: VecDeque<String>,
}

#[derive(Default)]
struct Stats {
    deltas_sent: [u32; 2],
    dropped: u32,
    rejected: u32,
    gaps: [u32; 2],
    full_deltas: [u32; 2],
}

impl Lane {
    fn note(&mut self, msg: String) {
        info!("{msg}");
        self.log.push_back(format!("{:>5}  {msg}", self.frame));
        while self.log.len() > LOG_LINES {
            self.log.pop_front();
        }
    }

    fn delay(&self, client: usize) -> u32 {
        if client == ClientB::ID && self.delay {
            DELAY_FRAMES
        } else {
            0
        }
    }

    /// Requests travel reliably and in order, but as slowly as client B's
    /// messages when delay is on, so B sees its own edit after a full round
    /// trip.
    fn send_to_server(&mut self, client: usize, msg: &ToServer) {
        let deliver_at = self.frame + 1 + self.delay(client);
        let bytes = postcard::to_allocvec(msg).unwrap();
        self.to_server[client].push_back(Packet { deliver_at, bytes });
    }

    fn take_due_requests(&mut self, client: usize) -> Vec<ToServer> {
        let mut due = Vec::new();
        while let Some(packet) = self.to_server[client].front() {
            if packet.deliver_at > self.frame {
                break;
            }
            let bytes = self.to_server[client].pop_front().unwrap().bytes;
            due.push(postcard::from_bytes(&bytes).unwrap());
        }
        due
    }

    /// Sends the client the server's message for this frame, unless it is
    /// empty or the drop fault loses it.
    fn send_frame(&mut self, client: usize) {
        let mut msg = std::mem::take(&mut self.outbox[client]);
        if msg.is_empty() {
            return;
        }
        msg.frame = self.frame;
        let flaky = client == ClientB::ID;
        let mut deliver_at = self.frame + 1 + self.delay(client);
        if flaky
            && self.drop
            && !msg.must_arrive()
            && rand::random_range(0..100) < DROP_PERCENT
        {
            self.stats.dropped += 1;
            // A message holding only poses is lost every few frames while
            // drop is on, so only one with more in it is worth a line.
            if !msg.deltas.is_empty() || !msg.despawns.is_empty() {
                self.note(format!(
                    "lane dropped a message to client B: {} deltas, {} despawns",
                    msg.deltas.len(),
                    msg.despawns.len()
                ));
            }
            return;
        }
        if flaky && self.reorder {
            deliver_at += rand::random_range(0..REORDER_FRAMES);
        }
        self.stats.deltas_sent[client] += msg.deltas.len() as u32;
        let bytes = postcard::to_allocvec(&msg).unwrap();
        self.to_client[client].push(Packet { deliver_at, bytes });
    }

    fn take_due(&mut self, client: usize) -> Vec<ServerFrame> {
        let frame = self.frame;
        let (due, keep): (Vec<_>, Vec<_>) = self.to_client[client]
            .drain(..)
            .partition(|p| p.deliver_at <= frame);
        self.to_client[client] = keep;
        due.iter()
            .map(|p| postcard::from_bytes(&p.bytes).unwrap())
            .collect()
    }
}

/// Which chunks each client gets deltas for, by `Client::ID`. A real game
/// derives this from the chunks each client has loaded; the crate never holds
/// it.
#[derive(Resource)]
struct Interest([HashSet<IVec3>; 2]);

impl Default for Interest {
    /// Client B leaves out a corner of the ring, so an edit there reaches A
    /// only.
    fn default() -> Self {
        let mut interest_a = HashSet::new();
        for x in -INTEREST_CHUNKS..=INTEREST_CHUNKS {
            for z in -INTEREST_CHUNKS..=INTEREST_CHUNKS {
                for y in -1..=1 {
                    interest_a.insert(IVec3::new(x, y, z));
                }
            }
        }
        let interest_b = interest_a
            .iter()
            .copied()
            .filter(|c| !(c.x >= 1 && c.z >= 1))
            .collect();
        Interest([interest_a, interest_b])
    }
}

impl Interest {
    /// Whether a piece whose top piece's voxels are centred at `centre`, in
    /// the root's frame, is in the client's interest: whether the client's
    /// interest holds the chunk column there.
    fn covers_piece(&self, client: usize, centre: Vec3) -> bool {
        let (chunk, _) = get_chunk_voxel_position(centre.floor().as_ivec3());
        self.0[client].contains(&IVec3::new(chunk.x, 0, chunk.z))
    }
}

// ---------------------------------------------------------------------------
// The server's pieces and the clients' copies of them.

/// The server's pieces by game id, and what each client has been sent.
#[derive(Resource, Default)]
struct ServerPieces {
    next_id: u32,
    pieces: BTreeMap<u32, ServerPiece>,
    /// Pieces whose stamp has been called and whose `Stamped` hasn't been
    /// read, with their game id and which clients held them. Their entities
    /// live on until the crate despawns them.
    stamping: HashMap<Entity, (u32, [bool; 2])>,
    /// The pieces each client holds, as far as the server knows.
    known: [BTreeSet<u32>; 2],
    /// Set to have the server clear the world's edits and despawn every
    /// piece in its next frame.
    reset: bool,
    /// Holds every piece still, so a check can compare poses.
    paused: bool,
}

struct ServerPiece {
    entity: Entity,
    parent: Option<u32>,
    /// The piece's `Transform` relative to its parent, which the server
    /// keeps on the entity and sends every frame.
    local: Transform,
    motion: Motion,
}

/// How the server moves a piece: up by `RISE` from `home`, then along its
/// drift circle, if it has one.
struct Motion {
    home: Vec3,
    risen: f32,
    drift: Option<Drift>,
}

/// A circle in the root's horizontal plane through the point `start`, where
/// the piece's voxels were centred when it finished rising.
struct Drift {
    start: Vec2,
    centre: Vec2,
    angle: f32,
}

impl Motion {
    /// A piece lifted with its voxels centred at `centre`. Only a piece under
    /// the world's root drifts, on the circle whose diameter runs from there
    /// to `DRIFT_THROUGH`.
    fn new(home: Vec3, centre: Vec3, drifts: bool) -> Self {
        let start = centre.xz();
        let mid = (start + DRIFT_THROUGH) / 2.0;
        let drift = (drifts && start.distance(mid) >= 1.0).then(|| Drift {
            start,
            centre: mid,
            angle: (start - mid).to_angle(),
        });
        Self {
            home,
            risen: 0.0,
            drift,
        }
    }

    /// Moves the piece one frame on, and returns its translation.
    fn step(&mut self) -> Vec3 {
        if self.risen < RISE {
            self.risen = (self.risen + RISE_STEP).min(RISE);
        } else if let Some(drift) = &mut self.drift {
            drift.angle += DRIFT_STEP / drift.start.distance(drift.centre);
        }
        let along = self.drift.as_ref().map_or(Vec2::ZERO, |drift| {
            let radius = drift.start.distance(drift.centre);
            drift.centre + Vec2::from_angle(drift.angle) * radius - drift.start
        });
        self.home + Vec3::new(along.x, self.risen, along.y)
    }
}

impl ServerPieces {
    fn grid(&self, name: GridName) -> Option<Grid> {
        match name {
            GridName::World => Some(Grid::World),
            GridName::Piece(id) => self.pieces.get(&id).map(|p| Grid::SubGrid(p.entity)),
        }
    }

    /// The topmost piece above `id`, or `id` itself.
    fn top(&self, mut id: u32) -> u32 {
        for _ in 0..self.pieces.len() {
            match self.pieces.get(&id).and_then(|p| p.parent) {
                Some(parent) => id = parent,
                None => break,
            }
        }
        id
    }

    fn at(&self, id: u32) -> PieceAt {
        let piece = &self.pieces[&id];
        PieceAt {
            piece: id,
            parent: piece.parent,
            pose: piece.local.into(),
        }
    }
}

/// The centre of a sub-grid's solid voxels in the root's frame, or its
/// origin if it holds none.
fn voxel_centre<C: VoxelWorldConfig>(voxels: &GridVoxels<C>) -> Vec3 {
    let pose = voxels.pose();
    voxels.bounds().map_or(pose.translation(), |(min, max)| {
        pose.transform_point((min + max).as_vec3() / 2.0)
    })
}

/// Each client's pieces, by `Client::ID`.
#[derive(Resource, Default)]
struct Replicas([Replica; 2]);

/// A client's map between game ids and its own entities.
#[derive(Default)]
struct Replica {
    pieces: HashMap<u32, Replicated>,
    ids: HashMap<Entity, u32>,
    /// The server frame of the newest spawn or despawn applied for each
    /// game id, so an older one that arrives later is ignored.
    existence: HashMap<u32, u32>,
}

struct Replicated {
    entity: Entity,
    parent: Option<u32>,
    /// The server frame of the newest pose applied.
    pose_frame: u32,
}

impl Replica {
    /// The entity to put a piece under: the world's root, or the parent
    /// piece's entity if the client holds it.
    fn parent_entity(&self, parent: Option<u32>, root: Entity) -> Option<Entity> {
        match parent {
            None => Some(root),
            Some(id) => self.pieces.get(&id).map(|p| p.entity),
        }
    }

    fn name(&self, grid: Grid) -> Option<GridName> {
        match grid {
            Grid::World => Some(GridName::World),
            Grid::SubGrid(entity) => self.ids.get(&entity).copied().map(GridName::Piece),
        }
    }

    /// Removes the piece and the pieces under it, which despawn with it, at
    /// server frame `frame`. Returns the removed pieces' entities.
    fn remove(&mut self, id: u32, frame: u32) -> Vec<Entity> {
        let mut removed = Vec::new();
        let mut queue = vec![id];
        while let Some(id) = queue.pop() {
            self.existence.insert(id, frame);
            let Some(piece) = self.pieces.remove(&id) else {
                continue;
            };
            self.ids.remove(&piece.entity);
            removed.push(piece.entity);
            queue.extend(
                self.pieces
                    .iter()
                    .filter(|(_, p)| p.parent == Some(id))
                    .map(|(child, _)| *child),
            );
        }
        removed
    }
}

/// What the outlines show, by `Client::ID`.
#[derive(Resource, Default)]
struct Feedback {
    /// Voxels each client asked to set and has not seen a delta for.
    pending: [Vec<(GridName, IVec3)>; 2],
    /// Voxels the server refused, until the frame their flash ends.
    rejected: [Vec<(GridName, IVec3, u32)>; 2],
    /// The voxel under the cursor, in one client's view.
    hover: Option<Hover>,
}

#[derive(Clone, Copy)]
struct Hover {
    client: usize,
    grid: GridName,
    dig: IVec3,
    place: Option<IVec3>,
}

/// The ring on the ground, drawn in both views.
#[derive(Resource)]
struct Ring(Vec<Vec3>);

#[derive(Resource, Clone, Copy, PartialEq)]
enum ViewMode {
    Split,
    Only(usize),
}

/// What the client panels list.
#[derive(Resource, Clone, Copy, PartialEq, Default)]
enum HudTab {
    #[default]
    Chunks,
    Pieces,
}

#[derive(Component)]
struct ViewCamera(usize);

#[derive(Component)]
struct ClientHud(usize);

#[derive(Component)]
struct BottomHud;

// ---------------------------------------------------------------------------

/// The lane, the server's and the clients' pieces, and the systems that
/// sync the three worlds, without any input, rendering or panel.
struct LoopbackSync;

impl Plugin for LoopbackSync {
    fn build(&self, app: &mut App) {
        app.init_resource::<Lane>()
            .init_resource::<Feedback>()
            .init_resource::<Interest>()
            .init_resource::<ServerPieces>()
            .init_resource::<Replicas>()
            // Everything that touches edits runs before the worlds' own sets
            // in `PreUpdate`: clients apply what arrived, then the server
            // handles what it was asked and moves its pieces. Each world
            // commits in its set, and the server sends each client its
            // frame's message right after. Systems that only read the
            // worlds, like input, panels and outlines, run in `Update`.
            .add_systems(
                PreUpdate,
                (
                    (
                        tick_lane,
                        client_receive::<ClientA>,
                        client_receive::<ClientB>,
                        server_handle_requests,
                        move_pieces,
                    )
                        .chain()
                        .before(VoxelWorldSet::<Server>::default())
                        .before(VoxelWorldSet::<ClientA>::default())
                        .before(VoxelWorldSet::<ClientB>::default()),
                    server_send.after(VoxelWorldSet::<Server>::default()),
                ),
            );
    }
}

fn main() -> AppExit {
    let audit = std::env::args().nth(1).as_deref() == Some("audit");
    let mut app = App::new();
    app.add_plugins(DefaultPlugins)
        .add_plugins(HierarchyPropagatePlugin::<RenderLayers>::new(PostUpdate))
        .add_plugins(VoxelWorldPlugin::<Server>::minimal())
        .add_plugins(VoxelWorldPlugin::with_config(ClientA))
        .add_plugins(VoxelWorldPlugin::with_config(ClientB))
        .add_plugins(LoopbackSync)
        .init_gizmo_group::<GizmosA>()
        .init_gizmo_group::<GizmosB>()
        .insert_resource(ViewMode::Split)
        .init_resource::<HudTab>()
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
                layout_viewports,
                update_hud,
                draw_markers::<ClientA>,
                draw_markers::<ClientB>,
            ),
        );
    if audit {
        app.init_resource::<Audit>()
            .add_systems(Update, (record_landings, script));
    } else {
        app.add_systems(Update, (player_input, keys).chain());
    }
    // Returned, so an audit that fails exits with an error.
    app.run()
}

fn setup(mut commands: Commands, mut gizmo_configs: ResMut<GizmoConfigStore>) {
    // Each camera streams and sees only its own client's world.
    fn camera<C: Client>(commands: &mut Commands) -> Entity {
        commands
            .spawn((
                Camera3d::default(),
                Camera {
                    order: C::ID as isize,
                    ..default()
                },
                Transform::from_translation(C::CAMERA).looking_at(LOOK_AT, Vec3::Y),
                ViewCamera(C::ID),
                VoxelWorldCamera::<C>::default(),
                RenderLayers::layer(C::LAYER),
            ))
            .id()
    }
    let cam_a = camera::<ClientA>(&mut commands);
    let cam_b = camera::<ClientB>(&mut commands);

    // Each client's gizmo group draws on its layer, so its outlines stay in
    // its view.
    fn gizmos_on<G: GizmoConfigGroup>(configs: &mut GizmoConfigStore, layer: usize) {
        let config = configs.config_mut::<G>().0;
        config.render_layers = RenderLayers::layer(layer);
        config.line.width = 3.0;
    }
    gizmos_on::<GizmosA>(&mut gizmo_configs, ClientA::LAYER);
    gizmos_on::<GizmosB>(&mut gizmo_configs, ClientB::LAYER);

    // One light per client: a light on both layers would let A's chunks cast
    // shadows into B's view, so an edit would show there as a shadow first.
    for layer in [ClientA::LAYER, ClientB::LAYER] {
        commands.spawn((
            DirectionalLight {
                color: Color::srgb(0.98, 0.95, 0.82),
                shadow_maps_enabled: true,
                ..default()
            },
            Transform::from_xyz(0.0, 0.0, 0.0)
                .looking_at(Vec3::new(-0.15, -0.1, 0.15), Vec3::Y),
            CascadeShadowConfigBuilder::default().build(),
            RenderLayers::layer(layer),
        ));
    }
    commands.insert_resource(GlobalAmbientLight {
        color: Color::srgb(0.98, 0.95, 0.82),
        brightness: 100.0,
        affects_lightmapped_meshes: true,
    });

    // The ring follows the ground, so it reads as a boundary, not a horizon.
    let mut generate = generator();
    let ring = (0..=256)
        .map(|i| {
            let angle = i as f32 / 256.0 * std::f32::consts::TAU;
            let x = ARENA.x + REACH * angle.cos();
            let z = ARENA.z + REACH * angle.sin();
            let y = surface(&mut generate, x.floor() as i32, z.floor() as i32);
            Vec3::new(x, y as f32 + 1.1, z)
        })
        .collect();
    commands.insert_resource(Ring(ring));

    // Each client's panel is UI on its own camera, so it sits in its
    // viewport. A separate UI camera over the whole window would blank
    // client A's half.
    let font = TextFont {
        font_size: FontSize::Px(13.0),
        ..default()
    };
    for (id, cam) in [(ClientA::ID, cam_a), (ClientB::ID, cam_b)] {
        commands.spawn((
            Text::new(""),
            font.clone(),
            Node {
                position_type: PositionType::Absolute,
                top: Val::Px(8.0),
                left: Val::Px(8.0),
                ..default()
            },
            UiTargetCamera(cam),
            ClientHud(id),
        ));
    }
    commands.spawn((
        Text::new(""),
        font,
        Node {
            position_type: PositionType::Absolute,
            bottom: Val::Px(8.0),
            left: Val::Px(8.0),
            ..default()
        },
        UiTargetCamera(cam_a),
        BottomHud,
    ));
}

// ---------------------------------------------------------------------------
// The sync systems.

fn tick_lane(mut lane: ResMut<Lane>, mut feedback: ResMut<Feedback>) {
    lane.frame += 1;
    let frame = lane.frame;
    for rejected in &mut feedback.rejected {
        rejected.retain(|(_, _, until)| *until > frame);
    }
}

/// Applies the messages that arrived for client `C`, and asks the server for
/// a chunk's full delta when a delta leaves a gap or the heartbeat shows the
/// client behind. The client's own `EditsCommitted` and
/// `SubGridEditsCommitted` messages are ignored: they only repeat what the
/// server sent.
///
/// `world` comes before `commands`, so the parent and pose given to a piece
/// here replace the ones `spawn_sub_grid` gives it: a system's commands
/// apply in the order of its parameters.
///
/// During an audit, client A notes the chunks each message's deltas touch
/// in the `Audit`, by the server frame that sent it.
#[allow(clippy::too_many_arguments)]
fn client_receive<C: Client>(
    mut world: VoxelWorld<C>,
    mut commands: Commands,
    root: Single<Entity, With<Root<C>>>,
    mut lane: ResMut<Lane>,
    mut feedback: ResMut<Feedback>,
    mut replicas: ResMut<Replicas>,
    mut audit: Option<ResMut<Audit>>,
) {
    let replica = &mut replicas.0[C::ID];
    let root = *root;
    // The client's edit sequences once this frame's deltas commit. A delta
    // applied this frame commits after this system, so `edited_chunks` alone
    // would still show the old sequence.
    let mut local: HashMap<(GridName, IVec3), u64> = world
        .edited_chunks()
        .into_iter()
        .map(|(chunk, seq)| ((GridName::World, chunk), seq))
        .collect();
    for (id, piece) in &replica.pieces {
        if let Some(grid) = world.grid(Grid::SubGrid(piece.entity)) {
            for (chunk, seq) in grid.edited_chunks() {
                local.insert((GridName::Piece(*id), chunk), seq);
            }
        }
    }

    for msg in lane.take_due(C::ID) {
        let frame = msg.frame;
        let mut spawned = HashSet::new();
        for at in msg.spawns {
            let id = at.piece;
            if replica.existence.get(&id).is_some_and(|f| *f > frame) {
                continue;
            }
            let Some(parent) = replica.parent_entity(at.parent, root) else {
                continue;
            };
            replica.existence.insert(id, frame);
            let pose = Transform::from(at.pose);
            // A piece the client still holds, such as one the server sent
            // again after a heartbeat, takes the spawn in place: the full
            // deltas that follow replace its edits.
            let held = replica
                .pieces
                .get(&id)
                .map(|p| p.entity)
                .filter(|e| world.grid(Grid::SubGrid(*e)).is_some());
            let entity = held.unwrap_or_else(|| world.spawn_sub_grid(pose));
            commands.entity(entity).try_insert((ChildOf(parent), pose));
            replica.ids.insert(entity, id);
            replica.pieces.insert(
                id,
                Replicated {
                    entity,
                    parent: at.parent,
                    pose_frame: frame,
                },
            );
            spawned.insert(id);
        }
        if !spawned.is_empty() {
            let mut ids: Vec<_> = spawned.iter().collect();
            ids.sort();
            lane.note(format!("{} joins pieces {ids:?}", C::NAME));
        }

        // The chunks this message's deltas touch, by game name and by grid.
        let mut touched: Vec<(GridName, Grid, IVec3)> = Vec::new();
        for (grid, delta) in msg.deltas {
            let (chunk, seq) = (delta.chunk, delta.seq);
            if delta.full {
                lane.stats.full_deltas[C::ID] += 1;
                if grid == GridName::World {
                    lane.note(format!(
                        "{} got full delta for chunk {chunk} ({} edits, seq {seq})",
                        C::NAME,
                        delta.len(),
                    ));
                }
            }
            // A full delta carries every edit of its chunk, so one sent
            // after the request carries the requested voxel too, and one
            // sent before leaves it pending.
            let carried: Vec<(GridName, IVec3)> = feedback.pending[C::ID]
                .iter()
                .copied()
                .filter(|(g, p)| *g == grid && delta.iter().any(|(pos, _)| pos == *p))
                .collect();
            let applied = match grid {
                GridName::World => {
                    touched.push((grid, Grid::World, chunk));
                    world.apply_delta(delta)
                }
                GridName::Piece(id) => {
                    // Deltas for a piece the client doesn't hold belong to
                    // one that left its interest, or whose spawn is still on
                    // its way: the join brings its full deltas.
                    let Some((local, mut voxels)) = replica
                        .pieces
                        .get(&id)
                        .map(|p| Grid::SubGrid(p.entity))
                        .and_then(|g| Some((g, world.grid_mut(g)?)))
                    else {
                        continue;
                    };
                    touched.push((grid, local, chunk));
                    voxels.apply_delta(delta)
                }
            };
            match applied {
                Ok(()) => {
                    let reached = local.entry((grid, chunk)).or_default();
                    *reached = (*reached).max(seq);
                    feedback.pending[C::ID].retain(|p| !carried.contains(p));
                }
                Err(gap) => {
                    lane.stats.gaps[C::ID] += 1;
                    lane.note(format!(
                        "{} gap in {grid:?} chunk {chunk}: has seq {}, got {}; asks for full",
                        C::NAME,
                        gap.local,
                        gap.received
                    ));
                    let chunk = chunk.to_array();
                    lane.send_to_server(C::ID, &ToServer::RequestFull { grid, chunk });
                }
            }
        }

        for at in msg.poses {
            let Some(piece) = replica.pieces.get(&at.piece) else {
                continue;
            };
            if piece.pose_frame > frame {
                continue;
            }
            let (entity, old_parent) = (piece.entity, piece.parent);
            let mut entity_commands = commands.entity(entity);
            entity_commands.try_insert(Transform::from(at.pose));
            if old_parent != at.parent {
                let Some(parent) = replica.parent_entity(at.parent, root) else {
                    continue;
                };
                entity_commands.try_insert(ChildOf(parent));
            }
            let piece = replica.pieces.get_mut(&at.piece).unwrap();
            piece.pose_frame = frame;
            piece.parent = at.parent;
        }

        let mut despawned = HashSet::new();
        for id in msg.despawns {
            if replica.existence.get(&id).is_some_and(|f| *f > frame) {
                continue;
            }
            despawned.insert(id);
            // Through `despawn_sub_grid`, so a stamped piece shows until the
            // deltas emptying it, applied above, land.
            for entity in replica.remove(id, frame) {
                world.despawn_sub_grid(entity);
            }
        }

        if let Some(beat) = msg.heartbeat {
            for (chunk, seq) in beat.world {
                let at = IVec3::from_array(chunk);
                let has = local.get(&(GridName::World, at)).copied().unwrap_or(0);
                if has < seq {
                    lane.note(format!(
                        "heartbeat: {} behind in chunk {at} ({has} < {seq}); asks for full",
                        C::NAME
                    ));
                    let grid = GridName::World;
                    lane.send_to_server(C::ID, &ToServer::RequestFull { grid, chunk });
                }
            }
            let mut listed = HashSet::new();
            for (id, chunks) in beat.pieces {
                listed.insert(id);
                if !replica.pieces.contains_key(&id) {
                    if replica.existence.get(&id).is_none_or(|f| *f < frame) {
                        lane.note(format!(
                            "heartbeat: {} lacks piece {id}; asks to join it",
                            C::NAME
                        ));
                        lane.send_to_server(C::ID, &ToServer::JoinPiece { piece: id });
                    }
                    continue;
                }
                let grid = GridName::Piece(id);
                for (chunk, seq) in chunks {
                    let at = IVec3::from_array(chunk);
                    let has = local.get(&(grid, at)).copied().unwrap_or(0);
                    if has < seq {
                        lane.note(format!(
                            "heartbeat: {} behind in piece {id} chunk {at}; asks for full",
                            C::NAME
                        ));
                        lane.send_to_server(
                            C::ID,
                            &ToServer::RequestFull { grid, chunk },
                        );
                    }
                }
            }
            // A piece the server no longer sends lost its despawn.
            let stale: Vec<u32> = replica
                .pieces
                .keys()
                .copied()
                .filter(|id| !listed.contains(id))
                .filter(|id| replica.existence.get(id).is_none_or(|f| *f < frame))
                .collect();
            for id in stale {
                lane.note(format!(
                    "heartbeat: {} holds piece {id} the server dropped; despawns it",
                    C::NAME
                ));
                for entity in replica.remove(id, frame) {
                    world.despawn_sub_grid(entity);
                }
            }
        }

        for (grid, pos) in msg.rejected {
            let pos = IVec3::from_array(pos);
            feedback.pending[C::ID].retain(|p| *p != (grid, pos));
            feedback.rejected[C::ID].push((grid, pos, lane.frame + FLASH_FRAMES));
        }

        if let Some(audit) = audit.as_mut().filter(|_| C::ID == ClientA::ID) {
            // A message fills a piece it spawns, or empties one it
            // despawns, when it carries that piece's deltas.
            let handover = touched.iter().any(|(name, ..)| {
                matches!(name, GridName::Piece(id) if spawned.contains(id) || despawned.contains(id))
            });
            let chunks = touched.into_iter().map(|(_, grid, chunk)| (grid, chunk));
            audit.note_commit(frame, lane.frame, handover, chunks);
        }
    }
}

/// Handles what clients asked for: sets the voxels that pass the server's
/// check, lifts and stamps, answers full delta requests and joins, and runs
/// a reset.
///
/// `server` comes before `commands`, so a piece lifted out of a piece and
/// moved under the world's root by a stamp in the same frame ends up under
/// the root, not under the parent the lift gave it.
fn server_handle_requests(
    mut server: VoxelWorld<Server>,
    mut commands: Commands,
    root: Single<Entity, With<Root<Server>>>,
    mut lane: ResMut<Lane>,
    mut pieces: ResMut<ServerPieces>,
    interest: Res<Interest>,
) {
    if std::mem::take(&mut pieces.reset) {
        server.clear_edits();
        for piece in pieces.pieces.values() {
            commands.entity(piece.entity).try_despawn();
        }
        for entity in pieces.stamping.keys() {
            commands.entity(*entity).try_despawn();
        }
        for client in CLIENTS {
            let known = std::mem::take(&mut pieces.known[client]);
            lane.outbox[client].despawns.extend(known);
        }
        pieces.pieces.clear();
        pieces.stamping.clear();
        lane.note("server clears every edit and despawns every piece".into());
    }

    for client in CLIENTS {
        for msg in lane.take_due_requests(client) {
            match msg {
                ToServer::Set { grid, pos, voxel } => {
                    let at = IVec3::from_array(pos);
                    let target = pieces.grid(grid);
                    if !target.is_some_and(|g| in_ring(&server, g, at.as_vec3() + 0.5)) {
                        refuse(&mut lane, client, grid, at, "set");
                        continue;
                    }
                    if let Some(mut voxels) = target.and_then(|g| server.grid_mut(g)) {
                        voxels.set_voxel(at, voxel);
                    }
                }
                ToServer::Lift { grid, min, max } => {
                    let (min, max) = (IVec3::from_array(min), IVec3::from_array(max));
                    let centre = (min + max).as_vec3() / 2.0;
                    let Some(source) =
                        pieces.grid(grid).filter(|g| in_ring(&server, *g, centre))
                    else {
                        refuse(
                            &mut lane,
                            client,
                            grid,
                            centre.floor().as_ivec3(),
                            "lift",
                        );
                        continue;
                    };
                    match server.lift(source, min, max, |_, voxel| voxel.is_solid()) {
                        Ok(entity) => {
                            let id = lift(&server, &mut pieces, grid, source, entity);
                            lane.note(format!(
                                "server lifts piece {id} out of {grid:?} for {}",
                                name(client)
                            ));
                        }
                        Err(error) => lane.note(format!("server lifts nothing: {error}")),
                    }
                }
                ToServer::Stamp { piece: id } => {
                    let Some(piece) = pieces.pieces.get(&id) else {
                        continue;
                    };
                    let entity = piece.entity;
                    if let Err(error) =
                        server.stamp(entity, Grid::World, incoming_replaces)
                    {
                        lane.note(format!("server stamps nothing: {error}"));
                        continue;
                    }
                    stamp(&server, &mut commands, *root, &mut lane, &mut pieces, id);
                    lane.note(format!("server stamps piece {id} for {}", name(client)));
                }
                ToServer::RequestFull { grid, chunk } => {
                    // A piece the client no longer holds gets nothing: its
                    // despawn is on the way.
                    let holds = match grid {
                        GridName::World => true,
                        GridName::Piece(id) => pieces.known[client].contains(&id),
                    };
                    let full = pieces
                        .grid(grid)
                        .filter(|_| holds)
                        .and_then(|g| server.grid(g))
                        .map(|voxels| voxels.full_delta(IVec3::from_array(chunk)));
                    if let Some(full) = full {
                        lane.outbox[client].deltas.push((grid, full));
                    }
                }
                ToServer::Join => {
                    let chunks: Vec<IVec3> = server
                        .edited_chunks()
                        .into_iter()
                        .map(|(chunk, _)| chunk)
                        .filter(|chunk| interest.0[client].contains(chunk))
                        .collect();
                    lane.note(format!(
                        "server answers {}'s join with {} full deltas",
                        name(client),
                        chunks.len()
                    ));
                    for chunk in chunks {
                        let full = server.full_delta(chunk);
                        lane.outbox[client].deltas.push((GridName::World, full));
                    }
                    // `server_send` sends every piece in its interest whole.
                    pieces.known[client].clear();
                }
                ToServer::JoinPiece { piece } => {
                    pieces.known[client].remove(&piece);
                }
            }
        }
    }
}

/// Whether `point` on `grid` lies inside the ring: the server's one rule. It
/// could also run the world's generator here, for example to refuse digging
/// into the bottom layer.
fn in_ring(server: &VoxelWorld<Server>, grid: Grid, point: Vec3) -> bool {
    let Some(voxels) = server.grid(grid) else {
        return false;
    };
    let at = voxels.pose().transform_point(point);
    at.xz().distance(ARENA.xz()) <= REACH
}

fn refuse(lane: &mut Lane, client: usize, grid: GridName, at: IVec3, what: &str) {
    lane.stats.rejected += 1;
    lane.note(format!(
        "server rejected {}'s {what} at {at}: outside the ring",
        name(client)
    ));
    lane.outbox[client].rejected.push((grid, at.to_array()));
}

/// Gives the piece the lift just spawned its game id and its motion, and
/// returns the id. It sits where the lift put it: under its source, the
/// world's root or a piece.
fn lift(
    server: &VoxelWorld<Server>,
    pieces: &mut ServerPieces,
    source_name: GridName,
    source: Grid,
    entity: Entity,
) -> u32 {
    let id = pieces.next_id;
    pieces.next_id += 1;
    let piece = server.grid(Grid::SubGrid(entity)).expect("lifted at once");
    let source_pose = server.grid(source).expect("lifted from it").pose();
    let local = piece.pose().reparented_to(&source_pose);
    let centre = voxel_centre(&piece);
    let parent = match source_name {
        GridName::World => None,
        GridName::Piece(parent) => Some(parent),
    };
    let motion = Motion::new(local.translation, centre, parent.is_none());
    pieces.pieces.insert(
        id,
        ServerPiece {
            entity,
            parent,
            local,
            motion,
        },
    );
    id
}

/// Forgets the stamped piece `id`, sends its despawn to the clients holding
/// it, and moves the pieces under it under the world's root, keeping their
/// pose, so they outlive it on every world.
fn stamp(
    server: &VoxelWorld<Server>,
    commands: &mut Commands,
    root: Entity,
    lane: &mut Lane,
    pieces: &mut ServerPieces,
    id: u32,
) {
    let piece = pieces.pieces.remove(&id).expect("stamped a live piece");
    let held = CLIENTS.map(|client| pieces.known[client].remove(&id));
    for client in CLIENTS.into_iter().filter(|c| held[*c]) {
        lane.outbox[client].despawns.push(id);
    }
    pieces.stamping.insert(piece.entity, (id, held));
    let Some(pose) = server.grid(Grid::SubGrid(piece.entity)).map(|g| g.pose()) else {
        return;
    };
    for child in pieces.pieces.values_mut().filter(|p| p.parent == Some(id)) {
        child.parent = None;
        child.local = pose.mul_transform(child.local).compute_transform();
        child.motion.home = child.local.translation;
        child.motion.risen = RISE;
        commands
            .entity(child.entity)
            .insert((ChildOf(root), child.local));
    }
}

/// Moves each piece one frame on and puts its new `Transform` on its entity.
/// A piece whose entity has no `Transform` yet, one lifted this frame, starts
/// next frame.
fn move_pieces(mut pieces: ResMut<ServerPieces>, mut transforms: Query<&mut Transform>) {
    if pieces.paused {
        return;
    }
    for piece in pieces.pieces.values_mut() {
        let Ok(mut transform) = transforms.get_mut(piece.entity) else {
            continue;
        };
        piece.local.translation = piece.motion.step();
        *transform = piece.local;
    }
}

/// Builds and sends each client its message for this frame: spawns and
/// despawns for the pieces that entered or left its interest, the deltas the
/// server committed in its interest, every pose, and now and then the
/// heartbeat. Answers to requests are in the message already.
#[allow(clippy::too_many_arguments)]
fn server_send(
    server: VoxelWorld<Server>,
    interest: Res<Interest>,
    mut lane: ResMut<Lane>,
    mut pieces: ResMut<ServerPieces>,
    mut world_commits: MessageReader<EditsCommitted<Server>>,
    mut piece_commits: MessageReader<SubGridEditsCommitted<Server>>,
    mut stamped: MessageReader<Stamped<Server>>,
) {
    let pieces = &mut *pieces;
    for stamp in stamped.read() {
        pieces.stamping.remove(&stamp.sub_grid);
    }

    // Interest takes a piece whole, by the top piece above it.
    let centre = |id: u32| {
        let voxels = server.grid(Grid::SubGrid(pieces.pieces[&id].entity))?;
        Some(voxel_centre(&voxels))
    };
    let centres: HashMap<u32, Option<Vec3>> = pieces
        .pieces
        .keys()
        .map(|id| (*id, centre(pieces.top(*id))))
        .collect();
    let mut fresh: [BTreeSet<u32>; 2] = default();
    for client in CLIENTS {
        let wanted: BTreeSet<u32> = centres
            .iter()
            .filter(|(_, c)| c.is_some_and(|c| interest.covers_piece(client, c)))
            .map(|(id, _)| *id)
            .collect();
        let gone: Vec<u32> = pieces.known[client].difference(&wanted).copied().collect();
        if !gone.is_empty() {
            lane.note(format!(
                "pieces {gone:?} left {}'s interest; server despawns them there",
                name(client)
            ));
        }
        fresh[client] = wanted.difference(&pieces.known[client]).copied().collect();
        lane.outbox[client].despawns.extend(gone);
        pieces.known[client] = wanted;
    }

    for msg in world_commits.read() {
        let chunk = msg.delta.chunk;
        let to: Vec<usize> = CLIENTS
            .into_iter()
            .filter(|client| interest.0[*client].contains(&chunk))
            .collect();
        if to.len() == 1 {
            lane.note(format!(
                "server sent chunk {chunk} to {} alone",
                name(to[0])
            ));
        }
        for client in to {
            let delta = msg.delta.clone();
            lane.outbox[client].deltas.push((GridName::World, delta));
        }
    }

    // A piece joining a client this frame gets its full deltas below in
    // place of this commit's.
    let ids: HashMap<Entity, u32> = pieces
        .pieces
        .iter()
        .map(|(id, piece)| (piece.entity, *id))
        .collect();
    for msg in piece_commits.read() {
        for client in CLIENTS {
            let (id, to) = if let Some(id) = ids.get(&msg.sub_grid) {
                let holds = pieces.known[client].contains(id);
                (*id, holds && !fresh[client].contains(id))
            } else if let Some((id, held)) = pieces.stamping.get(&msg.sub_grid) {
                (*id, held[client])
            } else {
                continue;
            };
            if to {
                let delta = msg.delta.clone();
                lane.outbox[client]
                    .deltas
                    .push((GridName::Piece(id), delta));
            }
        }
    }

    for client in CLIENTS {
        for id in &fresh[client] {
            lane.outbox[client].spawns.push(pieces.at(*id));
            let voxels = server
                .grid(Grid::SubGrid(pieces.pieces[id].entity))
                .expect("a piece is live");
            let mut chunks = voxels.edited_chunks();
            chunks.sort_by_key(|(chunk, _)| chunk.to_array());
            for (chunk, _) in chunks {
                let full = voxels.full_delta(chunk);
                lane.outbox[client]
                    .deltas
                    .push((GridName::Piece(*id), full));
            }
        }
        for id in pieces.known[client].difference(&fresh[client]) {
            lane.outbox[client].poses.push(pieces.at(*id));
        }
    }

    if lane.frame.is_multiple_of(HEARTBEAT_FRAMES) {
        let edited = server.edited_chunks();
        for client in CLIENTS {
            let world = edited
                .iter()
                .filter(|(chunk, _)| interest.0[client].contains(chunk))
                .map(|(chunk, seq)| (chunk.to_array(), *seq))
                .collect();
            let held = pieces.known[client].iter().filter_map(|id| {
                let voxels = server.grid(Grid::SubGrid(pieces.pieces[id].entity))?;
                let chunks = voxels.edited_chunks().into_iter();
                Some((*id, chunks.map(|(c, seq)| (c.to_array(), seq)).collect()))
            });
            lane.outbox[client].heartbeat = Some(Heartbeat {
                world,
                pieces: held.collect(),
            });
        }
    }

    for client in CLIENTS {
        lane.send_frame(client);
    }
}

// ---------------------------------------------------------------------------
// Player actions.

/// A client asks the server to set a voxel of a grid, and outlines it until a
/// delta carrying it arrives.
fn request(
    lane: &mut Lane,
    feedback: &mut Feedback,
    client: usize,
    grid: GridName,
    pos: IVec3,
    voxel: WorldVoxel<u8>,
) {
    feedback.pending[client].push((grid, pos));
    let pos = pos.to_array();
    lane.send_to_server(client, &ToServer::Set { grid, pos, voxel });
}

/// Client B forgets its edits, despawns its pieces and joins again, as a
/// client reconnecting would. It never clears instead: a clear moves its edit
/// sequences past the server's, and the server's next deltas would be skipped
/// as already applied.
fn rejoin_b(
    lane: &mut Lane,
    feedback: &mut Feedback,
    world_b: &mut VoxelWorld<ClientB>,
    replica_b: &mut Replica,
    commands: &mut Commands,
) {
    world_b.forget_edits();
    // The join brings each piece whole, so nothing of the old ones need
    // show until it lands.
    for piece in replica_b.pieces.values() {
        commands.entity(piece.entity).try_despawn();
    }
    *replica_b = Replica::default();
    // Whatever was on its way belonged to the old connection.
    lane.to_client[ClientB::ID].clear();
    feedback.pending[ClientB::ID].clear();
    lane.note("client B forgets its edits and pieces and joins again".into());
    lane.send_to_server(ClientB::ID, &ToServer::Join);
}

/// Asks the server to clear the world's edits and despawn every piece in its
/// next frame. Clients get the cleared chunks as empty full deltas and the
/// pieces as despawns, in that frame's messages.
fn reset(lane: &mut Lane, feedback: &mut Feedback, pieces: &mut ServerPieces) {
    pieces.reset = true;
    feedback.pending = default();
    lane.note("server is asked to reset".into());
}

/// Tracks the voxel under the cursor in whichever view it is over, in the
/// world or a piece. Left click asks to dig it, right click asks to place the
/// client's material beside it.
#[allow(clippy::too_many_arguments)]
fn player_input(
    buttons: Res<ButtonInput<MouseButton>>,
    window: Single<&Window, With<PrimaryWindow>>,
    cameras: Query<(&Camera, &GlobalTransform, &ViewCamera)>,
    world_a: VoxelWorld<ClientA>,
    world_b: VoxelWorld<ClientB>,
    replicas: Res<Replicas>,
    mut lane: ResMut<Lane>,
    mut feedback: ResMut<Feedback>,
) {
    feedback.hover = None;
    let Some(cursor) = window.cursor_position() else {
        return;
    };
    for (camera, cam_gtf, view) in &cameras {
        let Some(rect) = camera.logical_viewport_rect() else {
            continue;
        };
        if !camera.is_active || !rect.contains(cursor) {
            continue;
        }
        // `viewport_to_world` takes window coordinates: it subtracts the
        // viewport's origin itself.
        let Ok(ray) = camera.viewport_to_world(cam_gtf, cursor) else {
            continue;
        };
        let replica = &replicas.0[view.0];
        let synced = |grid: Grid| replica.name(grid).is_some();
        let hit = if view.0 == ClientA::ID {
            world_a.raycast_grids(ray, RAY_LENGTH, &synced, &|_, _, _| true)
        } else {
            world_b.raycast_grids(ray, RAY_LENGTH, &synced, &|_, _, _| true)
        };
        let Some((hit, grid)) = hit.and_then(|h| Some((h, replica.name(h.grid)?))) else {
            continue;
        };
        let hover = Hover {
            client: view.0,
            grid,
            dig: hit.position,
            place: hit.normal.map(|n| hit.position + n),
        };
        feedback.hover = Some(hover);
        let material = [ClientA::PLACES, ClientB::PLACES][view.0];
        if buttons.just_pressed(MouseButton::Left) {
            let air = WorldVoxel::Air;
            request(&mut lane, &mut feedback, view.0, grid, hover.dig, air);
        } else if let Some(place) = hover
            .place
            .filter(|_| buttons.just_pressed(MouseButton::Right))
        {
            let solid = WorldVoxel::Solid(material);
            request(&mut lane, &mut feedback, view.0, grid, place, solid);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn keys(
    keys: Res<ButtonInput<KeyCode>>,
    mut world_b: VoxelWorld<ClientB>,
    mut commands: Commands,
    mut lane: ResMut<Lane>,
    mut feedback: ResMut<Feedback>,
    mut view: ResMut<ViewMode>,
    mut tab: ResMut<HudTab>,
    mut pieces: ResMut<ServerPieces>,
    mut replicas: ResMut<Replicas>,
) {
    if keys.just_pressed(KeyCode::Digit1) {
        lane.delay = !lane.delay;
    }
    if keys.just_pressed(KeyCode::Digit2) {
        lane.drop = !lane.drop;
    }
    if keys.just_pressed(KeyCode::Digit3) {
        lane.reorder = !lane.reorder;
    }
    if keys.just_pressed(KeyCode::KeyV) {
        *view = match *view {
            ViewMode::Split => ViewMode::Only(ClientA::ID),
            ViewMode::Only(id) if id == ClientA::ID => ViewMode::Only(ClientB::ID),
            ViewMode::Only(_) => ViewMode::Split,
        };
    }
    if keys.just_pressed(KeyCode::Tab) {
        *tab = match *tab {
            HudTab::Chunks => HudTab::Pieces,
            HudTab::Pieces => HudTab::Chunks,
        };
    }
    if let Some(hover) = feedback.hover {
        if keys.just_pressed(KeyCode::KeyL) {
            let reach = IVec3::splat(LIFT_REACH);
            let min = (hover.dig - reach).to_array();
            let max = (hover.dig + reach + 1).to_array();
            let grid = hover.grid;
            lane.send_to_server(hover.client, &ToServer::Lift { grid, min, max });
        }
        if let (GridName::Piece(piece), true) =
            (hover.grid, keys.just_pressed(KeyCode::KeyK))
        {
            lane.send_to_server(hover.client, &ToServer::Stamp { piece });
        }
    }
    if keys.just_pressed(KeyCode::KeyJ) {
        let replica_b = &mut replicas.0[ClientB::ID];
        rejoin_b(
            &mut lane,
            &mut feedback,
            &mut world_b,
            replica_b,
            &mut commands,
        );
    }
    if keys.just_pressed(KeyCode::KeyR) {
        reset(&mut lane, &mut feedback, &mut pieces);
    }
}

// ---------------------------------------------------------------------------
// Views and panels.

fn layout_viewports(
    window: Single<&Window, With<PrimaryWindow>>,
    view: Res<ViewMode>,
    mut cameras: Query<(Entity, &mut Camera, &ViewCamera)>,
    mut bottom: Single<&mut UiTargetCamera, With<BottomHud>>,
) {
    let w = window.physical_width().max(2);
    let h = window.physical_height().max(1);
    for (entity, mut camera, ViewCamera(id)) in &mut cameras {
        let (active, x, width) = match *view {
            ViewMode::Split => (true, *id as u32 * (w / 2), w / 2),
            ViewMode::Only(shown) => (shown == *id, 0, w),
        };
        // The bottom panel sits in the leftmost view shown.
        if active && x == 0 && bottom.0 != entity {
            bottom.0 = entity;
        }
        camera.is_active = active;
        camera.viewport = Some(Viewport {
            physical_position: UVec2::new(x, 0),
            physical_size: UVec2::new(width, h),
            ..default()
        });
    }
}

/// Outlines in one client's view: the hovered voxel (white), its requests
/// still waiting for a delta (yellow), its refused requests (red), and the
/// ring. A voxel of a piece is outlined where the piece shows it.
fn draw_markers<C: Client>(
    feedback: Res<Feedback>,
    ring: Res<Ring>,
    replicas: Res<Replicas>,
    world: VoxelWorld<C>,
    mut gizmos: Gizmos<C::Gizmos>,
) {
    let replica = &replicas.0[C::ID];
    let mut outline = |grid: GridName, pos: IVec3, color: Color| {
        let pose = match grid {
            GridName::World => Some(GlobalTransform::IDENTITY),
            GridName::Piece(id) => replica
                .pieces
                .get(&id)
                .and_then(|p| world.grid(Grid::SubGrid(p.entity)))
                .map(|voxels| voxels.pose()),
        };
        if let Some(pose) = pose {
            let centre = pose.transform_point(pos.as_vec3() + 0.5);
            gizmos.primitive_3d(
                &Cuboid::from_length(1.04),
                Isometry3d::new(centre, pose.rotation()),
                color,
            );
        }
    };
    if let Some(hover) = feedback.hover.filter(|h| h.client == C::ID) {
        outline(hover.grid, hover.dig, Color::WHITE);
    }
    for (grid, pos) in &feedback.pending[C::ID] {
        outline(*grid, *pos, Color::srgb(1.0, 0.85, 0.1));
    }
    for (grid, pos, _) in &feedback.rejected[C::ID] {
        outline(*grid, *pos, Color::srgb(1.0, 0.15, 0.1));
    }
    gizmos.linestrip(ring.0.iter().copied(), Color::srgba(1.0, 1.0, 1.0, 0.7));
}

/// Each client's panel lists, on its chunks tab, the server's edited chunks
/// with the client's edit sequence against the server's, and on its pieces
/// tab, the server's pieces with the client's edited chunks of each against
/// the server's, marking where they differ.
#[allow(clippy::too_many_arguments)]
fn update_hud(
    lane: Res<Lane>,
    feedback: Res<Feedback>,
    interest: Res<Interest>,
    tab: Res<HudTab>,
    pieces: Res<ServerPieces>,
    replicas: Res<Replicas>,
    server: VoxelWorld<Server>,
    world_a: VoxelWorld<ClientA>,
    world_b: VoxelWorld<ClientB>,
    cameras: Query<(&Camera, &ViewCamera)>,
    mut huds: Query<(&mut Text, &mut Node, &ClientHud)>,
    mut bottom: Single<&mut Text, (With<BottomHud>, Without<ClientHud>)>,
) {
    let mut server_chunks = server.edited_chunks();
    server_chunks.sort_by_key(|(c, _)| (c.y, c.z, c.x));
    let sorted = |mut chunks: Vec<(IVec3, u64)>| {
        chunks.sort_by_key(|(c, _)| c.to_array());
        chunks
    };
    for (mut text, mut node, ClientHud(id)) in &mut huds {
        let active = cameras.iter().any(|(c, v)| v.0 == *id && c.is_active);
        node.display = if active { Display::Flex } else { Display::None };
        let (name, places) = [
            (ClientA::NAME, ClientA::PLACES_NAME),
            (ClientB::NAME, ClientB::PLACES_NAME),
        ][*id];
        let grid = |grid: Grid| {
            if *id == ClientA::ID {
                world_a.grid(grid).map(|g| sorted(g.edited_chunks()))
            } else {
                world_b.grid(grid).map(|g| sorted(g.edited_chunks()))
            }
        };
        let mut s = format!(
            "{name}: left click digs, right click lays {places}\n\
             waiting {}   in flight {}   gaps {}   full deltas {}\n",
            feedback.pending[*id].len(),
            lane.to_client[*id].len(),
            lane.stats.gaps[*id],
            lane.stats.full_deltas[*id],
        );
        match *tab {
            HudTab::Chunks => {
                let local = grid(Grid::World).unwrap_or_default();
                if !server_chunks.is_empty() {
                    s.push_str("\nchunk          edit sequence here / server\n");
                }
                for (chunk, seq) in server_chunks.iter().take(6) {
                    let at = format!("{:>3} {:>3} {:>3}", chunk.x, chunk.y, chunk.z);
                    if !interest.0[*id].contains(chunk) {
                        s.push_str(&format!("  {at}   outside interest\n"));
                        continue;
                    }
                    let here = local
                        .iter()
                        .find(|(c, _)| c == chunk)
                        .map_or(0, |(_, s)| *s);
                    let mark = if here == *seq { " " } else { "!" };
                    s.push_str(&format!("{mark} {at}   {here:>4} / {seq:>4}\n"));
                }
            }
            HudTab::Pieces => {
                if !pieces.pieces.is_empty() {
                    s.push_str("\npiece   edited chunks here / server\n");
                }
                for (piece, held) in pieces.pieces.iter().take(6) {
                    let there = server
                        .grid(Grid::SubGrid(held.entity))
                        .map(|g| sorted(g.edited_chunks()))
                        .unwrap_or_default();
                    if !pieces.known[*id].contains(piece) {
                        s.push_str(&format!("  {piece:>4}   outside interest\n"));
                        continue;
                    }
                    let here = replicas.0[*id]
                        .pieces
                        .get(piece)
                        .and_then(|p| grid(Grid::SubGrid(p.entity)));
                    let mark = if here.as_ref() == Some(&there) {
                        " "
                    } else {
                        "!"
                    };
                    let count = here.map_or("-".into(), |h| h.len().to_string());
                    s.push_str(&format!(
                        "{mark} {piece:>4}   {count:>4} / {:>4}\n",
                        there.len()
                    ));
                }
            }
        }
        **text = s;
    }

    let on = |b: bool| if b { "ON " } else { "off" };
    let mut s = String::new();
    for line in &lane.log {
        s.push_str(line);
        s.push('\n');
    }
    s.push_str(&format!(
        "\nlane to B:  [1] delay {}  [2] drop {}  [3] reorder {}   \
         [J] B joins again  [R] server reset  [V] view\n\
         [L] lift  [K] stamp piece  [Tab] chunks / pieces\n",
        on(lane.delay),
        on(lane.drop),
        on(lane.reorder)
    ));
    s.push_str(&format!(
        "deltas to A {}  to B {}   dropped {}   rejected {}   edited chunks {}   pieces {}",
        lane.stats.deltas_sent[ClientA::ID],
        lane.stats.deltas_sent[ClientB::ID],
        lane.stats.dropped,
        lane.stats.rejected,
        server_chunks.len(),
        pieces.pieces.len(),
    ));
    ***bottom = s;
}

// ---------------------------------------------------------------------------
// The scripted audit, run with `-- audit`.

/// What the audit keeps between checks: every voxel the script asked to set
/// or lift, by grid, so the audit can read each back on every world, how
/// many checks failed, and the server commits client A applied with the
/// frames its chunks landed in.
#[derive(Resource, Default)]
struct Audit {
    touched: HashSet<(GridName, IVec3)>,
    failures: u32,
    /// The server commits client A applied, by the server frame whose
    /// message held them.
    commits: BTreeMap<u32, ServerCommit>,
    landed: Landings,
}

/// The frames in which client A published or retired each chunk, by A's
/// grid and the chunk's position.
type Landings = HashMap<(Grid, IVec3), Vec<u32>>;

/// One server commit as client A applied it.
#[derive(Default)]
struct ServerCommit {
    /// The frame A applied the first part of it in.
    applied: u32,
    /// Whether it lands as one handover on A: it fills a piece A spawns or
    /// empties one A despawns.
    handover: bool,
    /// The chunks its deltas touch, by A's grid.
    deltas: HashSet<(Grid, IVec3)>,
    /// The neighbours of those chunks, which the deltas make stale where
    /// they change a voxel in the neighbour's padding.
    beside: HashSet<(Grid, IVec3)>,
}

impl Audit {
    /// Notes the chunks a message from server frame `server_frame` touched
    /// on client A, which applied it in frame `applied`, and whether it holds
    /// a handover. Parts of one server commit that arrive apart add to one
    /// `ServerCommit`.
    fn note_commit(
        &mut self,
        server_frame: u32,
        applied: u32,
        handover: bool,
        chunks: impl IntoIterator<Item = (Grid, IVec3)>,
    ) {
        let mut chunks = chunks.into_iter().peekable();
        if chunks.peek().is_none() {
            return;
        }
        let commit = self
            .commits
            .entry(server_frame)
            .or_insert_with(|| ServerCommit {
                applied,
                ..default()
            });
        commit.handover |= handover;
        for (grid, chunk) in chunks {
            commit.deltas.insert((grid, chunk));
            for x in -1..=1 {
                for y in -1..=1 {
                    for z in -1..=1 {
                        commit.beside.insert((grid, chunk + IVec3::new(x, y, z)));
                    }
                }
            }
        }
        let ServerCommit { deltas, beside, .. } = commit;
        beside.retain(|chunk| !deltas.contains(chunk));
    }
}

impl ServerCommit {
    /// The frames the handover's chunks landed in: one frame if it landed
    /// whole.
    ///
    /// Each chunk of `deltas` counts in the first frame it landed in from
    /// `applied` on. A chunk of `beside` counts only if it landed between
    /// `applied` and the last of those, since one that landed later holds a
    /// later edit, or wasn't made stale. A chunk that never landed is one A
    /// doesn't have, outside its spawning distance.
    fn landed_in(&self, landed: &Landings) -> BTreeSet<u32> {
        let first = |chunk: &(Grid, IVec3)| {
            landed
                .get(chunk)?
                .iter()
                .copied()
                .find(|frame| *frame >= self.applied)
        };
        let mut frames: BTreeSet<u32> = self.deltas.iter().filter_map(first).collect();
        let Some(last) = frames.last().copied() else {
            return frames;
        };
        frames.extend(self.beside.iter().filter_map(first).filter(|f| *f <= last));
        frames
    }
}

impl Audit {
    /// Asks the server to set a voxel of a grid, as `client`.
    fn set(
        &mut self,
        lane: &mut Lane,
        feedback: &mut Feedback,
        client: usize,
        grid: GridName,
        pos: IVec3,
        voxel: WorldVoxel<u8>,
    ) {
        self.touched.insert((grid, pos));
        request(lane, feedback, client, grid, pos, voxel);
    }

    /// Asks the server to lift the box `LIFT_REACH` around `at` out of a
    /// grid, as `client` pressing `L` there would.
    fn lift(&mut self, lane: &mut Lane, client: usize, grid: GridName, at: IVec3) {
        let reach = IVec3::splat(LIFT_REACH);
        let (min, max) = (at - reach, at + reach + 1);
        for x in min.x..max.x {
            for y in min.y..max.y {
                for z in min.z..max.z {
                    self.touched.insert((grid, IVec3::new(x, y, z)));
                }
            }
        }
        let (min, max) = (min.to_array(), max.to_array());
        lane.send_to_server(client, &ToServer::Lift { grid, min, max });
    }
}

/// The worlds and the sync state the audit reads and the script drives.
#[derive(SystemParam)]
struct Audited<'w, 's> {
    server: VoxelWorld<'w, 's, Server>,
    world_a: VoxelWorld<'w, 's, ClientA>,
    world_b: VoxelWorld<'w, 's, ClientB>,
    interest: Res<'w, Interest>,
    pieces: ResMut<'w, ServerPieces>,
    replicas: ResMut<'w, Replicas>,
    placed: Query<'w, 's, (&'static Transform, &'static ChildOf)>,
    live_a: Query<'w, 's, Entity, With<SubGrid<ClientA>>>,
    live_b: Query<'w, 's, Entity, With<SubGrid<ClientB>>>,
}

/// Records the frame of each `ChunkPublished` and `ChunkWillDespawn` of
/// client A, for the handover check. The chunk lifecycle writes them in
/// `PreUpdate`, so they count in the frame that wrote them.
fn record_landings(
    lane: Res<Lane>,
    mut audit: ResMut<Audit>,
    mut published: MessageReader<ChunkPublished<ClientA>>,
    mut retired: MessageReader<ChunkWillDespawn<ClientA>>,
) {
    let published = published.read().map(|m| (m.grid, m.chunk_key.position));
    let retired = retired.read().map(|m| (m.grid, m.chunk_key.position));
    for chunk in published.chain(retired) {
        audit.landed.entry(chunk).or_default().push(lane.frame);
    }
}

// The game ids of the pieces the script lifts. The server gives out ids in
// the order it lifts, and the script asks for one lift at a time. The first
// lift, game id 0, is the piece that drifts across the edge of client B's
// interest.
const FIRST: u32 = 1;
const CHILD: u32 = 2;
const SECOND: u32 = 3;
const THIRD: u32 = 4;

/// World edits and pieces from both clients through each fault, then a
/// re-join and a reset, auditing both clients against the server after each
/// step. Frames count from the first `PreUpdate`.
///
/// Under drop and reorder, client A lifts a piece in the corner client B's
/// interest leaves out, and it drifts into B's interest. B lifts `FIRST`, A
/// lifts `CHILD` out of it, both clients dig and lay in it, and B stamps it.
/// Under delay, the first piece drifts out of B's interest again, A lifts
/// `SECOND`, both clients edit it, and A stamps `CHILD`. After the re-join,
/// B lifts `THIRD` and both clients edit it and `SECOND`. Each step ends by
/// holding every piece still while the lane settles, so poses can be
/// compared.
#[allow(clippy::too_many_arguments)]
fn script(
    mut commands: Commands,
    mut lane: ResMut<Lane>,
    mut feedback: ResMut<Feedback>,
    mut audit: ResMut<Audit>,
    mut w: Audited,
    cameras: Query<(&Camera, &GlobalTransform, &ViewCamera)>,
    mut exit: MessageWriter<AppExit>,
    mut generate: Local<Option<Generator>>,
) {
    let generate = generate.get_or_insert_with(generator);
    let (lane, feedback, audit) = (&mut *lane, &mut *feedback, &mut *audit);
    let (a, b) = (ClientA::ID, ClientB::ID);
    let world = GridName::World;
    let frame = lane.frame;
    // The box around a piece's solid voxels on the server.
    let bounds = |w: &Audited, id: u32| {
        let piece = w.pieces.pieces.get(&id)?;
        w.server.grid(Grid::SubGrid(piece.entity))?.bounds()
    };
    // Client A digs down a column of the piece, and B lays three voxels on
    // its top.
    let edit_piece = |w: &Audited,
                      lane: &mut Lane,
                      feedback: &mut Feedback,
                      audit: &mut Audit,
                      id: u32| {
        let Some((min, max)) = bounds(w, id) else {
            audit.failures += 1;
            error!("audit: piece {id} isn't on the server to edit");
            return;
        };
        let grid = GridName::Piece(id);
        for y in min.y..max.y {
            let pos = IVec3::new(max.x - 1, y, max.z - 1);
            audit.set(lane, feedback, a, grid, pos, WorldVoxel::Air);
        }
        for x in 0..3 {
            let pos = IVec3::new(min.x + x, max.y, min.z);
            audit.set(lane, feedback, b, grid, pos, WorldVoxel::Solid(SNOW));
        }
    };
    match frame {
        120 => {
            lane.drop = true;
            lane.reorder = true;
            audit.lift(lane, a, world, ground(generate, 33, 33));
        }
        // A pit one layer per frame, so several incremental deltas travel per
        // chunk and the drop fault has some to lose.
        130..=135 => {
            let depth = (frame - 130) as i32;
            for dx in 3..=9 {
                for dz in 3..=9 {
                    let y = surface(generate, dx, dz) - depth;
                    let pos = IVec3::new(dx, y, dz);
                    audit.set(lane, feedback, a, world, pos, WorldVoxel::Air);
                }
            }
        }
        140..=145 => {
            let h = (frame - 140) as i32;
            for z in -4..4 {
                let pos = ground(generate, -8, z) + IVec3::Y * (1 + h);
                audit.set(lane, feedback, a, world, pos, WorldVoxel::Solid(BRICK));
            }
        }
        150..=159 => {
            let pos = ground(generate, (frame - 150) as i32 - 5, -12) + IVec3::Y;
            audit.set(lane, feedback, b, world, pos, WorldVoxel::Solid(SNOW));
        }
        // In the corner client B's interest leaves out.
        165 => {
            let pos = ground(generate, 40, 40) + IVec3::Y;
            audit.set(lane, feedback, a, world, pos, WorldVoxel::Solid(BRICK));
        }
        // Outside the ring: the server refuses it.
        166 => {
            let pos = ground(generate, 0, 70) + IVec3::Y;
            audit.set(lane, feedback, a, world, pos, WorldVoxel::Solid(BRICK));
        }
        180 => audit.lift(lane, b, world, ground(generate, -16, -8)),
        // A corner of `FIRST`.
        200 => match bounds(&w, FIRST) {
            Some((min, _)) => audit.lift(lane, a, GridName::Piece(FIRST), min),
            None => {
                audit.failures += 1;
                error!("audit: piece {FIRST} isn't on the server to lift from");
            }
        },
        220 => edit_piece(&w, lane, feedback, audit, FIRST),
        260 => lane.send_to_server(b, &ToServer::Stamp { piece: FIRST }),
        420 => settle(lane, &mut w.pieces),
        600 => {
            check(audit, "drop and reorder", &w, generate);
            w.pieces.paused = false;
            lane.delay = true;
        }
        610 => audit.lift(lane, a, world, ground(generate, -4, 10)),
        640 => edit_piece(&w, lane, feedback, audit, SECOND),
        660 => lane.send_to_server(a, &ToServer::Stamp { piece: CHILD }),
        700..=705 => {
            let pos = ground(generate, 12, (frame - 700) as i32 - 2);
            audit.set(lane, feedback, b, world, pos, WorldVoxel::Air);
        }
        // With a path after `audit`, a frame of both views goes there, with
        // `SECOND` hanging in both.
        780 => {
            if let Some(path) = std::env::args().nth(2) {
                commands
                    .spawn(Screenshot::primary_window())
                    .observe(save_to_disk(path));
            }
        }
        800 => settle(lane, &mut w.pieces),
        920 => {
            check(audit, "delay", &w, generate);
            w.pieces.paused = false;
            let Audited {
                world_b, replicas, ..
            } = &mut w;
            let replica_b = &mut replicas.0[b];
            rejoin_b(lane, feedback, world_b, replica_b, &mut commands);
        }
        940 => audit.lift(lane, b, world, ground(generate, 10, -24)),
        960 => edit_piece(&w, lane, feedback, audit, SECOND),
        970 => edit_piece(&w, lane, feedback, audit, THIRD),
        1000 => settle(lane, &mut w.pieces),
        1040 => {
            check(audit, "join again", &w, generate);
            w.pieces.paused = false;
            reset(lane, feedback, &mut w.pieces);
        }
        1160 => {
            check(audit, "reset", &w, generate);
            // Both views look at `LOOK_AT`, so a ray through the centre of
            // each viewport lands near it if the cursor maths holds for both
            // halves of the window.
            let every = |_: Grid| true;
            let solid = |_: Grid, _: IVec3, _: WorldVoxel<u8>| true;
            for (camera, gtf, view) in &cameras {
                let hit = camera
                    .logical_viewport_rect()
                    .and_then(|rect| camera.viewport_to_world(gtf, rect.center()).ok())
                    .and_then(|ray| match view.0 {
                        0 => w.world_a.raycast_grids(ray, RAY_LENGTH, &every, &solid),
                        _ => w.world_b.raycast_grids(ray, RAY_LENGTH, &every, &solid),
                    })
                    .map(|hit| hit.point);
                let near = hit.is_some_and(|h| h.xz().distance(LOOK_AT.xz()) < 12.0);
                if !near {
                    audit.failures += 1;
                    error!(
                        "audit: {} centre ray hits {hit:?}, far from {LOOK_AT}",
                        name(view.0)
                    );
                }
            }
            let msg = format!("audit: done, {} failures", audit.failures);
            if audit.failures == 0 {
                info!("{msg}");
                exit.write(AppExit::Success);
            } else {
                error!("{msg}");
                exit.write(AppExit::error());
            }
        }
        _ => {}
    }
}

/// Ends a step's faults and holds every piece still, so what is in flight
/// lands, the heartbeat heals what was lost, and the clients' poses catch up
/// with the server's.
fn settle(lane: &mut Lane, pieces: &mut ServerPieces) {
    lane.drop = false;
    lane.reorder = false;
    lane.delay = false;
    pieces.paused = true;
}

/// The top solid voxel of the world's column at `x`, `z`.
fn ground(
    generate: &mut dyn FnMut(IVec3, Option<WorldVoxel<u8>>) -> WorldVoxel<u8>,
    x: i32,
    z: i32,
) -> IVec3 {
    IVec3::new(x, surface(generate, x, z), z)
}

/// Checks both clients against the server after a step, and counts what
/// differs as failures: their world edits, their pieces, and the handovers
/// client A applied since the last check.
fn check(
    audit: &mut Audit,
    step: &str,
    w: &Audited,
    generate: &mut dyn FnMut(IVec3, Option<WorldVoxel<u8>>) -> WorldVoxel<u8>,
) {
    let wrong =
        check_world(audit, w, generate) + check_pieces(audit, w) + check_handovers(audit);
    info!("audit after {step}: {wrong} differ");
    audit.failures += wrong;
}

/// Compares each world chunk's full delta, and the voxel `get_voxel` reads
/// at every position of the world the script set or lifted. Returns how many
/// differ.
fn check_world(
    audit: &Audit,
    w: &Audited,
    generate: &mut dyn FnMut(IVec3, Option<WorldVoxel<u8>>) -> WorldVoxel<u8>,
) -> u32 {
    let (server, world_a, world_b) = (&w.server, &w.world_a, &w.world_b);
    let interest = &w.interest;
    let mut wrong = 0;
    let mut chunks: HashSet<IVec3> =
        server.edited_chunks().into_iter().map(|(c, _)| c).collect();
    chunks.extend(world_a.edited_chunks().into_iter().map(|(c, _)| c));
    chunks.extend(world_b.edited_chunks().into_iter().map(|(c, _)| c));
    for client in CLIENTS {
        for chunk in &chunks {
            // A client hears nothing of a chunk outside its interest, so it
            // has no edits there and its sequence stays 0.
            let (seq, edits) = if interest.0[client].contains(chunk) {
                let full = server.full_delta(*chunk);
                (full.seq, full.iter().collect::<Vec<_>>())
            } else {
                (0, Vec::new())
            };
            let got = if client == ClientA::ID {
                world_a.full_delta(*chunk)
            } else {
                world_b.full_delta(*chunk)
            };
            if got.seq != seq || !got.iter().eq(edits.iter().copied()) {
                wrong += 1;
                error!(
                    "audit: {} chunk {chunk}: seq {} with {} edits, expected seq {seq} with {}",
                    name(client),
                    got.seq,
                    got.len(),
                    edits.len()
                );
            }
        }
    }
    let touched: Vec<IVec3> = audit
        .touched
        .iter()
        .filter(|(grid, _)| *grid == GridName::World)
        .map(|(_, pos)| *pos)
        .collect();
    for pos in &touched {
        let (chunk, _) = get_chunk_voxel_position(*pos);
        let edited = server
            .full_delta(chunk)
            .iter()
            .find(|(p, _)| p == pos)
            .map(|(_, v)| v);
        let generated = generate(*pos, None);
        let (a, b) = (world_a.get_voxel(*pos), world_b.get_voxel(*pos));
        let edited_b = edited.filter(|_| interest.0[ClientB::ID].contains(&chunk));
        // An edit reads back whether or not its chunk is spawned. Without
        // one, a voxel outside a client's view reads `Unset`, since no chunk
        // there has published.
        let shows = |got: WorldVoxel<u8>, edit: Option<WorldVoxel<u8>>| {
            got == edit.unwrap_or(generated)
                || (edit.is_none() && got == WorldVoxel::Unset)
        };
        if !shows(a, edited) || !shows(b, edited_b) {
            wrong += 1;
            error!("audit: {pos}: server edit {edited:?}, generated {generated:?}, A {a:?}, B {b:?}");
        }
    }
    info!(
        "audit: {} world chunks and {} world voxels checked",
        chunks.len(),
        touched.len()
    );
    wrong
}

/// A piece as one world holds it, for comparing across worlds by game id.
#[derive(Debug, PartialEq)]
struct PieceView {
    parent: Option<u32>,
    /// Its `Transform` relative to its parent.
    pose: Transform,
    /// The full delta of every chunk in its `grid(g).edited_chunks()`.
    deltas: Vec<EditDelta<u8>>,
    /// What `get_voxel` reads at each position of it the script set or
    /// lifted.
    reads: Vec<(IVec3, WorldVoxel<u8>)>,
    /// What `raycast_grids` hits on a ray cast straight down onto the centre
    /// of its voxels on the server, past the world: a piece's game id and
    /// the voxel's position on it.
    ray: Option<(GridName, IVec3)>,
}

/// How far above a piece's voxels the audit's ray down onto it starts.
const RAY_DROP: f32 = 64.0;

/// What one world shows the audit of its pieces.
struct Seen<'a, 'w, 's, C: VoxelWorldConfig> {
    world: &'a VoxelWorld<'w, 's, C>,
    touched: &'a HashSet<(GridName, IVec3)>,
    /// The ray down onto each piece, by game id.
    rays: &'a BTreeMap<u32, Ray3d>,
    /// The game id of each piece's entity on this world.
    ids: &'a HashMap<Entity, u32>,
}

impl<C: VoxelWorldConfig<MaterialIndex = u8>> Seen<'_, '_, '_, C> {
    /// The piece `id` held as `entity` on this world, under `parent` at
    /// `pose`.
    fn view(
        &self,
        id: u32,
        entity: Entity,
        parent: Option<u32>,
        pose: Transform,
    ) -> PieceView {
        let voxels = self.world.grid(Grid::SubGrid(entity));
        let mut deltas = Vec::new();
        if let Some(voxels) = &voxels {
            let mut chunks = voxels.edited_chunks();
            chunks.sort_by_key(|(chunk, _)| chunk.to_array());
            deltas = chunks.iter().map(|(c, _)| voxels.full_delta(*c)).collect();
        }
        let mut reads: Vec<_> = self
            .touched
            .iter()
            .filter(|(grid, _)| *grid == GridName::Piece(id))
            .map(|(_, pos)| {
                let read = voxels.as_ref().map(|v| v.get_voxel(*pos));
                (*pos, read.unwrap_or(WorldVoxel::Unset))
            })
            .collect();
        reads.sort_by_key(|(pos, _)| pos.to_array());
        let ray = self.rays.get(&id).and_then(|ray| {
            let past_world = |grid: Grid| grid != Grid::World;
            let solid = |_: Grid, _: IVec3, _: WorldVoxel<u8>| true;
            let hit =
                self.world
                    .raycast_grids(*ray, 2.0 * RAY_DROP, &past_world, &solid)?;
            let Grid::SubGrid(hit_entity) = hit.grid else {
                return None;
            };
            Some((GridName::Piece(*self.ids.get(&hit_entity)?), hit.position))
        });
        PieceView {
            parent,
            pose,
            deltas,
            reads,
            ray,
        }
    }
}

/// Compares each client's pieces with the server's by game id: the same set
/// within the client's interest, each with the same parent, pose, voxels
/// and ray hit, and no other sub-grid live on the client. Returns how many
/// pieces differ.
fn check_pieces(audit: &Audit, w: &Audited) -> u32 {
    let pieces = &w.pieces.pieces;
    let centre_of = |id: u32| {
        let voxels = w.server.grid(Grid::SubGrid(pieces.get(&id)?.entity))?;
        Some(voxel_centre(&voxels))
    };
    let rays: BTreeMap<u32, Ray3d> = pieces
        .keys()
        .filter_map(|id| {
            let above = centre_of(*id)? + Vec3::Y * RAY_DROP;
            Some((*id, Ray3d::new(above, -Dir3::Y)))
        })
        .collect();
    let server_ids: HashMap<Entity, u32> =
        pieces.iter().map(|(id, p)| (p.entity, *id)).collect();
    let seen = Seen {
        world: &w.server,
        touched: &audit.touched,
        rays: &rays,
        ids: &server_ids,
    };
    let server: BTreeMap<u32, PieceView> = pieces
        .iter()
        .map(|(id, p)| (*id, seen.view(*id, p.entity, p.parent, p.local)))
        .collect();

    let mut wrong = 0;
    for client in CLIENTS {
        let replica = &w.replicas.0[client];
        // Interest takes a piece whole, by the top piece above it.
        let wanted: BTreeSet<u32> = pieces
            .keys()
            .copied()
            .filter(|id| {
                let centre = centre_of(w.pieces.top(*id));
                centre.is_some_and(|c| w.interest.covers_piece(client, c))
            })
            .collect();
        let held: BTreeSet<u32> = replica.pieces.keys().copied().collect();
        let view = |id: u32, entity: Entity| {
            let (pose, child_of) = w.placed.get(entity).ok()?;
            let parent = replica.ids.get(&child_of.parent()).copied();
            let (touched, rays, ids) = (&audit.touched, &rays, &replica.ids);
            Some(if client == ClientA::ID {
                let seen = Seen {
                    world: &w.world_a,
                    touched,
                    rays,
                    ids,
                };
                seen.view(id, entity, parent, *pose)
            } else {
                let seen = Seen {
                    world: &w.world_b,
                    touched,
                    rays,
                    ids,
                };
                seen.view(id, entity, parent, *pose)
            })
        };
        for id in wanted.union(&held) {
            let expected = wanted.contains(id).then(|| &server[id]);
            let got = replica.pieces.get(id).and_then(|p| view(*id, p.entity));
            if expected != got.as_ref() {
                wrong += 1;
                error!(
                    "audit: {} piece {id}: expected {expected:?}, holds {got:?}",
                    name(client)
                );
            }
        }
        let live: Vec<Entity> = if client == ClientA::ID {
            w.live_a.iter().collect()
        } else {
            w.live_b.iter().collect()
        };
        let unnamed = live
            .iter()
            .filter(|e| !replica.ids.contains_key(*e))
            .count();
        if unnamed > 0 {
            wrong += unnamed as u32;
            error!(
                "audit: {} has {unnamed} sub-grids it holds no game id for",
                name(client)
            );
        }
        info!(
            "audit: {} holds pieces {held:?}, expected {wanted:?}",
            name(client)
        );
    }
    wrong
}

/// Checks that each server commit client A applied as a handover landed
/// whole, every chunk it touches publishing or retiring in one frame. Then
/// forgets the commits and landings recorded so far, so the script leaves
/// every handover time to land before a check. Returns how many handovers
/// landed apart.
fn check_handovers(audit: &mut Audit) -> u32 {
    let commits = std::mem::take(&mut audit.commits);
    let landed = std::mem::take(&mut audit.landed);
    let mut wrong = 0;
    let handovers: Vec<_> = commits.iter().filter(|(_, c)| c.handover).collect();
    for (frame, commit) in &handovers {
        let frames = commit.landed_in(&landed);
        if frames.len() != 1 {
            wrong += 1;
            error!("audit: client A landed server frame {frame}'s handover in frames {frames:?}");
        }
    }
    info!("audit: {} handovers on client A checked", handovers.len());
    wrong
}

// ---------------------------------------------------------------------------

/// Decides each voxel of the terrain from its position.
type Generator =
    Box<dyn FnMut(IVec3, Option<WorldVoxel<u8>>) -> WorldVoxel<u8> + Send + Sync>;

/// Gently rolling terrain: grass over stone, stone from `y = -3` down.
fn generator() -> Generator {
    let mut noise = HybridMulti::<Perlin>::new(1234);
    noise.octaves = 4;
    noise.frequency = 1.0;
    noise.lacunarity = 2.2;
    noise.persistence = 0.5;
    let mut cache = HashMap::<(i32, i32), f64>::new();
    Box::new(move |pos: IVec3, _previous| {
        if pos.y < -2 {
            return WorldVoxel::Solid(STONE);
        }
        let [x, y, z] = pos.as_dvec3().to_array();
        let sample = *cache
            .entry((pos.x, pos.z))
            .or_insert_with(|| noise.get([x / 400.0, z / 400.0]) * 6.0);
        if y < sample - 1.0 {
            WorldVoxel::Solid(STONE)
        } else if y < sample {
            WorldVoxel::Solid(GRASS)
        } else {
            WorldVoxel::Air
        }
    })
}

/// The height of the top solid voxel in the column at `x`, `z`.
fn surface(
    generate: &mut dyn FnMut(IVec3, Option<WorldVoxel<u8>>) -> WorldVoxel<u8>,
    x: i32,
    z: i32,
) -> i32 {
    (-2..24)
        .rev()
        .find(|y| matches!(generate(IVec3::new(x, *y, z), None), WorldVoxel::Solid(_)))
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use bevy::ecs::system::RunSystemOnce;

    use super::*;

    /// The three worlds and their sync, with no window, camera or input.
    fn app() -> App {
        let mut app = App::new();
        app.add_plugins((
            MinimalPlugins,
            TransformPlugin,
            VoxelWorldPlugin::<Server>::minimal(),
            VoxelWorldPlugin::<ClientA>::minimal(),
            VoxelWorldPlugin::<ClientB>::minimal(),
            LoopbackSync,
        ));
        app.update();
        app
    }

    fn run(app: &mut App, frames: u32) {
        for _ in 0..frames {
            app.update();
        }
    }

    /// Runs `app` until `done` holds, failing the test after `frames`.
    fn run_until(
        app: &mut App,
        frames: u32,
        what: &str,
        mut done: impl FnMut(&mut App) -> bool,
    ) {
        for _ in 0..frames {
            app.update();
            if done(app) {
                return;
            }
        }
        panic!("not after {frames} frames: {what}");
    }

    fn ask(app: &mut App, client: usize, msg: ToServer) {
        app.world_mut()
            .resource_mut::<Lane>()
            .send_to_server(client, &msg);
    }

    fn pause(app: &mut App) {
        app.world_mut().resource_mut::<ServerPieces>().paused = true;
    }

    /// The top solid voxel of the world's column at `x`, `z`.
    fn ground(x: i32, z: i32) -> IVec3 {
        IVec3::new(x, surface(&mut generator(), x, z), z)
    }

    /// Asks, as client A, to lift the box `LIFT_REACH` around `at` out of
    /// `grid`, and runs until the server has the piece. Returns its game id.
    fn lift_at(app: &mut App, grid: GridName, at: IVec3) -> u32 {
        let id = app.world().resource::<ServerPieces>().next_id;
        let reach = IVec3::splat(LIFT_REACH);
        let (min, max) = ((at - reach).to_array(), (at + reach + 1).to_array());
        ask(app, ClientA::ID, ToServer::Lift { grid, min, max });
        run_until(app, 5, "the server lifts", |app| {
            app.world()
                .resource::<ServerPieces>()
                .pieces
                .contains_key(&id)
        });
        id
    }

    fn holds(app: &App, client: usize, id: u32) -> bool {
        app.world().resource::<Replicas>().0[client]
            .pieces
            .contains_key(&id)
    }

    /// A piece as one world holds it: its parent, its `Transform` and the
    /// full delta of every chunk in its `edited_chunks()`.
    #[derive(Debug, PartialEq)]
    struct Held {
        parent: Option<u32>,
        pose: Transform,
        deltas: Vec<EditDelta<u8>>,
    }

    fn deltas<C: VoxelWorldConfig<MaterialIndex = u8>>(
        world: &VoxelWorld<C>,
        grid: Grid,
    ) -> Vec<EditDelta<u8>> {
        let Some(voxels) = world.grid(grid) else {
            return Vec::new();
        };
        let mut chunks = voxels.edited_chunks();
        chunks.sort_by_key(|(chunk, _)| chunk.to_array());
        chunks
            .into_iter()
            .map(|(chunk, _)| voxels.full_delta(chunk))
            .collect()
    }

    fn server_pieces(app: &mut App) -> BTreeMap<u32, Held> {
        app.world_mut()
            .run_system_once(|server: VoxelWorld<Server>, pieces: Res<ServerPieces>| {
                pieces
                    .pieces
                    .iter()
                    .map(|(id, piece)| {
                        let held = Held {
                            parent: piece.parent,
                            pose: piece.local,
                            deltas: deltas(&server, Grid::SubGrid(piece.entity)),
                        };
                        (*id, held)
                    })
                    .collect::<BTreeMap<_, _>>()
            })
            .unwrap()
    }

    fn client_pieces<C: Client>(app: &mut App) -> BTreeMap<u32, Held> {
        app.world_mut()
            .run_system_once(
                |world: VoxelWorld<C>,
                 replicas: Res<Replicas>,
                 placed: Query<(&Transform, &ChildOf)>| {
                    let replica = &replicas.0[C::ID];
                    replica
                        .pieces
                        .iter()
                        .map(|(id, piece)| {
                            let (pose, child_of) = placed.get(piece.entity).unwrap();
                            let held = Held {
                                parent: replica.ids.get(&child_of.parent()).copied(),
                                pose: *pose,
                                deltas: deltas(&world, Grid::SubGrid(piece.entity)),
                            };
                            (*id, held)
                        })
                        .collect::<BTreeMap<_, _>>()
                },
            )
            .unwrap()
    }

    /// The world's full delta of every chunk any of the three worlds edited,
    /// as each world holds it: the server's, A's and B's.
    fn world_deltas(app: &mut App) -> [Vec<EditDelta<u8>>; 3] {
        app.world_mut()
            .run_system_once(
                |server: VoxelWorld<Server>,
                 a: VoxelWorld<ClientA>,
                 b: VoxelWorld<ClientB>| {
                    let mut chunks: BTreeSet<[i32; 3]> = BTreeSet::new();
                    for edited in
                        [server.edited_chunks(), a.edited_chunks(), b.edited_chunks()]
                    {
                        chunks.extend(edited.into_iter().map(|(c, _)| c.to_array()));
                    }
                    let all = |full: &dyn Fn(IVec3) -> EditDelta<u8>| {
                        chunks
                            .iter()
                            .map(|c| full(IVec3::from_array(*c)))
                            .collect::<Vec<_>>()
                    };
                    [
                        all(&|c| server.full_delta(c)),
                        all(&|c| a.full_delta(c)),
                        all(&|c| b.full_delta(c)),
                    ]
                },
            )
            .unwrap()
    }

    fn sub_grid_count<C: VoxelWorldConfig>(app: &mut App) -> usize {
        app.world_mut()
            .query_filtered::<(), With<SubGrid<C>>>()
            .iter(app.world())
            .count()
    }

    /// Checks that A holds every server piece, and B the ones in `b_holds`,
    /// each with the server's parent, pose and voxels.
    fn pieces_converged(app: &mut App, b_holds: &[u32]) {
        let server = server_pieces(app);
        let a = client_pieces::<ClientA>(app);
        let b = client_pieces::<ClientB>(app);
        assert_eq!(a, server, "client A's pieces against the server's");
        let server_b: BTreeMap<u32, Held> = server
            .into_iter()
            .filter(|(id, _)| b_holds.contains(id))
            .collect();
        assert_eq!(b, server_b, "client B's pieces against the server's");
    }

    /// Checks both clients' world edits against the server's. Every world
    /// edit in these tests is inside both clients' interest.
    fn world_converged(app: &mut App) {
        let [server, a, b] = world_deltas(app);
        assert_eq!(a, server, "client A's world against the server's");
        assert_eq!(b, server, "client B's world against the server's");
    }

    /// The messages on their way to `client`, decoded.
    fn in_flight(app: &App, client: usize) -> Vec<ServerFrame> {
        app.world().resource::<Lane>().to_client[client]
            .iter()
            .map(|p| postcard::from_bytes(&p.bytes).unwrap())
            .collect()
    }

    #[test]
    fn a_lift_reaches_each_client_in_one_message_with_the_worlds_air() {
        let mut app = app();
        pause(&mut app);
        let id = lift_at(&mut app, GridName::World, ground(0, 0));

        for client in CLIENTS {
            let messages = in_flight(&app, client);
            let [msg] = messages.as_slice() else {
                panic!("one message to {}, not {}", name(client), messages.len());
            };
            assert_eq!(msg.spawns.len(), 1);
            assert_eq!(msg.spawns[0].piece, id);
            let solids_in = |grid: GridName, solid: bool| -> usize {
                msg.deltas
                    .iter()
                    .filter(|(g, _)| *g == grid)
                    .flat_map(|(_, d)| d.iter().collect::<Vec<_>>())
                    .filter(|(_, v)| v.is_solid() == solid)
                    .count()
            };
            let piece_solids = solids_in(GridName::Piece(id), true);
            assert!(piece_solids > 0, "the piece's voxels travel with its spawn");
            assert_eq!(
                solids_in(GridName::World, false),
                piece_solids,
                "the world's air travels with them"
            );
        }

        run(&mut app, 3);
        pieces_converged(&mut app, &[id]);
        world_converged(&mut app);
        assert_eq!(sub_grid_count::<ClientA>(&mut app), 1);
        assert_eq!(sub_grid_count::<ClientB>(&mut app), 1);
    }

    #[test]
    fn a_lifted_piece_rises_and_drifts_in_both_clients() {
        let mut app = app();
        let id = lift_at(&mut app, GridName::World, ground(-20, 0));
        let start = server_pieces(&mut app)[&id].pose.translation;
        run(&mut app, (RISE / RISE_STEP) as u32 + 60);
        let server = server_pieces(&mut app)[&id].pose.translation;
        assert!(server.y >= start.y + RISE - 0.1, "it rose");
        assert!(server.xz().distance(start.xz()) > 1.0, "it drifts");
        // The lane takes a frame, so a client shows the pose the server
        // sent a frame ago, a step behind.
        for held in [
            client_pieces::<ClientA>(&mut app),
            client_pieces::<ClientB>(&mut app),
        ] {
            let shown = held[&id].pose.translation;
            assert!(
                shown.distance(server) <= 2.0 * DRIFT_STEP,
                "{shown} far from {server}"
            );
        }
    }

    /// Moves a piece on the server, as the game moving its entity.
    fn move_piece(app: &mut App, id: u32, translation: Vec3) {
        let mut pieces = app.world_mut().resource_mut::<ServerPieces>();
        let piece = pieces.pieces.get_mut(&id).unwrap();
        piece.local.translation = translation;
        let (entity, local) = (piece.entity, piece.local);
        *app.world_mut().get_mut::<Transform>(entity).unwrap() = local;
    }

    #[test]
    fn a_piece_leaving_bs_interest_despawns_there_and_comes_back_whole() {
        let mut app = app();
        pause(&mut app);
        let id = lift_at(&mut app, GridName::World, ground(0, 0));
        run(&mut app, 3);
        let home = server_pieces(&mut app)[&id].pose.translation;

        // Chunk column (2, 2) is in the corner client B's interest leaves
        // out.
        move_piece(&mut app, id, Vec3::new(70.0, home.y, 70.0));
        run(&mut app, 4);
        assert!(holds(&app, ClientA::ID, id));
        assert!(!holds(&app, ClientB::ID, id), "B despawns it");
        assert_eq!(sub_grid_count::<ClientB>(&mut app), 0);

        // An edit B can't hear of while the piece is away.
        let pos = IVec3::new(1, 2 * LIFT_REACH + 2, 1).to_array();
        let (grid, voxel) = (GridName::Piece(id), WorldVoxel::Solid(BRICK));
        ask(&mut app, ClientA::ID, ToServer::Set { grid, pos, voxel });
        run(&mut app, 3);

        move_piece(&mut app, id, home);
        run(&mut app, 4);
        assert!(holds(&app, ClientB::ID, id), "B joins it again");
        pieces_converged(&mut app, &[id]);
    }

    #[test]
    fn clicks_edit_a_piece_in_every_world_by_its_game_id() {
        let mut app = app();
        pause(&mut app);
        let id = lift_at(&mut app, GridName::World, ground(0, 0));
        run(&mut app, 3);
        let before = server_pieces(&mut app)[&id].deltas.clone();
        // Lays a voxel on top of the piece's box.
        let top = IVec3::new(1, 2 * LIFT_REACH + 2, 1);
        let (grid, pos, voxel) =
            (GridName::Piece(id), top.to_array(), WorldVoxel::Solid(SNOW));
        ask(&mut app, ClientB::ID, ToServer::Set { grid, pos, voxel });
        run(&mut app, 4);
        assert_ne!(
            server_pieces(&mut app)[&id].deltas,
            before,
            "the server's piece took the edit"
        );
        pieces_converged(&mut app, &[id]);
        let read = app
            .world_mut()
            .run_system_once(
                move |a: VoxelWorld<ClientA>,
                      b: VoxelWorld<ClientB>,
                      replicas: Res<Replicas>| {
                    CLIENTS.map(|client| {
                        let grid = Grid::SubGrid(replicas.0[client].pieces[&id].entity);
                        if client == ClientA::ID {
                            a.grid(grid).unwrap().get_voxel(top)
                        } else {
                            b.grid(grid).unwrap().get_voxel(top)
                        }
                    })
                },
            )
            .unwrap();
        assert_eq!(read, [voxel; 2]);
    }

    #[test]
    fn a_stamp_moves_the_piece_into_the_world_in_all_three_worlds() {
        let mut app = app();
        let id = lift_at(&mut app, GridName::World, ground(0, 0));
        run(&mut app, 30);
        ask(&mut app, ClientB::ID, ToServer::Stamp { piece: id });
        run_until(&mut app, 600, "every world despawns the piece", |app| {
            sub_grid_count::<Server>(app) == 0
                && sub_grid_count::<ClientA>(app) == 0
                && sub_grid_count::<ClientB>(app) == 0
        });
        world_converged(&mut app);
        let [server, ..] = world_deltas(&mut app);
        let solids = server
            .iter()
            .flat_map(|d| d.iter().collect::<Vec<_>>())
            .filter(|(_, v)| v.is_solid())
            .count();
        assert!(solids > 0, "the stamped voxels are world edits");
    }

    #[test]
    fn a_stamp_moves_the_pieces_under_it_under_the_root() {
        let mut app = app();
        pause(&mut app);
        let parent = lift_at(&mut app, GridName::World, ground(0, 0));
        run(&mut app, 3);
        let child = lift_at(&mut app, GridName::Piece(parent), IVec3::splat(1));
        run(&mut app, 3);
        assert_eq!(server_pieces(&mut app)[&child].parent, Some(parent));
        pieces_converged(&mut app, &[parent, child]);

        ask(&mut app, ClientA::ID, ToServer::Stamp { piece: parent });
        run(&mut app, 3);
        let server = server_pieces(&mut app);
        assert_eq!(server.keys().copied().collect::<Vec<_>>(), [child]);
        assert_eq!(server[&child].parent, None);
        pieces_converged(&mut app, &[child]);
    }

    #[test]
    fn reset_leaves_no_world_edits_and_no_pieces() {
        let mut app = app();
        lift_at(&mut app, GridName::World, ground(0, 0));
        lift_at(&mut app, GridName::World, ground(-20, -20));
        let pos = (ground(10, 10) + IVec3::Y).to_array();
        let (grid, voxel) = (GridName::World, WorldVoxel::Solid(BRICK));
        ask(&mut app, ClientA::ID, ToServer::Set { grid, pos, voxel });
        run(&mut app, 5);

        app.world_mut().resource_mut::<ServerPieces>().reset = true;
        run(&mut app, 5);
        for (world, deltas) in ["server", "A", "B"].iter().zip(world_deltas(&mut app)) {
            assert!(deltas.iter().all(|d| d.is_empty()), "{world} keeps edits");
        }
        assert_eq!(sub_grid_count::<Server>(&mut app), 0);
        assert_eq!(sub_grid_count::<ClientA>(&mut app), 0);
        assert_eq!(sub_grid_count::<ClientB>(&mut app), 0);
        world_converged(&mut app);
    }

    fn rejoin(app: &mut App) {
        app.world_mut()
            .run_system_once(
                |mut world_b: VoxelWorld<ClientB>,
                 mut commands: Commands,
                 mut lane: ResMut<Lane>,
                 mut feedback: ResMut<Feedback>,
                 mut replicas: ResMut<Replicas>| {
                    let replica_b = &mut replicas.0[ClientB::ID];
                    rejoin_b(
                        &mut lane,
                        &mut feedback,
                        &mut world_b,
                        replica_b,
                        &mut commands,
                    );
                },
            )
            .unwrap();
    }

    #[test]
    fn a_rejoin_despawns_bs_pieces_and_brings_each_back_whole() {
        let mut app = app();
        pause(&mut app);
        let first = lift_at(&mut app, GridName::World, ground(0, 0));
        let second = lift_at(&mut app, GridName::World, ground(-20, 0));
        // Empties one of the first piece's chunks, so the join carries an
        // empty full delta too.
        let at = server_pieces(&mut app)[&first].deltas[0].clone();
        let grid = GridName::Piece(first);
        for (pos, _) in at.iter() {
            let (pos, voxel) = (pos.to_array(), WorldVoxel::Air);
            ask(&mut app, ClientA::ID, ToServer::Set { grid, pos, voxel });
        }
        run(&mut app, 3);
        let chunks = server_pieces(&mut app)[&first].deltas.len();
        assert!(
            server_pieces(&mut app)[&first]
                .deltas
                .iter()
                .any(|d| d.is_empty()),
            "an emptied chunk stays listed"
        );
        let old = app.world().resource::<Replicas>().0[ClientB::ID].pieces[&first].entity;

        rejoin(&mut app);
        app.update();
        assert_eq!(
            sub_grid_count::<ClientB>(&mut app),
            0,
            "B despawned its pieces"
        );
        run(&mut app, 4);
        let new = app.world().resource::<Replicas>().0[ClientB::ID].pieces[&first].entity;
        assert_ne!(old, new, "B joined the piece fresh");
        pieces_converged(&mut app, &[first, second]);
        world_converged(&mut app);
        assert_eq!(
            client_pieces::<ClientB>(&mut app)[&first].deltas.len(),
            chunks
        );
    }

    fn faults(app: &mut App, on: bool) {
        let mut lane = app.world_mut().resource_mut::<Lane>();
        lane.delay = on;
        lane.drop = on;
        lane.reorder = on;
    }

    #[test]
    fn b_converges_under_delay_drop_and_reorder() {
        let mut app = app();
        faults(&mut app, true);
        let id = lift_at(&mut app, GridName::World, ground(0, 0));
        for step in 0..20 {
            let voxel = WorldVoxel::Solid(SNOW);
            let grid = GridName::Piece(id);
            let pos =
                IVec3::new(1 + step % 5, 2 * LIFT_REACH + 2, 1 + step / 5).to_array();
            ask(&mut app, ClientA::ID, ToServer::Set { grid, pos, voxel });
            let pos = (ground(-10 + step, 10) + IVec3::Y).to_array();
            let grid = GridName::World;
            ask(&mut app, ClientA::ID, ToServer::Set { grid, pos, voxel });
            app.update();
        }
        pause(&mut app);
        faults(&mut app, false);
        // Heartbeats and their round trips heal whatever was lost.
        run(&mut app, 3 * HEARTBEAT_FRAMES);
        pieces_converged(&mut app, &[id]);
        world_converged(&mut app);
    }

    /// Loses every message on its way to client B, as a lane dropping the
    /// last delta of a chunk, which no later delta shows the gap of.
    fn lose_bs_messages(app: &mut App) {
        app.world_mut().resource_mut::<Lane>().to_client[ClientB::ID].clear();
    }

    #[test]
    fn the_heartbeat_heals_a_lost_last_piece_delta_by_game_id() {
        let mut app = app();
        pause(&mut app);
        let id = lift_at(&mut app, GridName::World, ground(0, 0));
        run(&mut app, 3);
        let pos = IVec3::new(1, 2 * LIFT_REACH + 2, 1).to_array();
        let (grid, voxel) = (GridName::Piece(id), WorldVoxel::Solid(SNOW));
        ask(&mut app, ClientA::ID, ToServer::Set { grid, pos, voxel });
        // The server sets and sends it in the next frame.
        app.update();
        lose_bs_messages(&mut app);
        run(&mut app, 3);
        let b = client_pieces::<ClientB>(&mut app);
        assert_ne!(b, server_pieces(&mut app), "B missed the edit");

        run(&mut app, HEARTBEAT_FRAMES + 3);
        pieces_converged(&mut app, &[id]);
    }

    #[test]
    fn the_heartbeat_heals_a_lost_despawn() {
        let mut app = app();
        pause(&mut app);
        let id = lift_at(&mut app, GridName::World, ground(0, 0));
        run(&mut app, 3);
        let home = server_pieces(&mut app)[&id].pose.translation;
        move_piece(&mut app, id, Vec3::new(70.0, home.y, 70.0));
        // The move shows, and so reaches `pose()`, after one frame, and the
        // server sends the despawn in the next.
        run(&mut app, 2);
        lose_bs_messages(&mut app);
        run(&mut app, 3);
        assert!(holds(&app, ClientB::ID, id), "B missed the despawn");

        run(&mut app, HEARTBEAT_FRAMES + 3);
        assert!(!holds(&app, ClientB::ID, id));
        assert_eq!(sub_grid_count::<ClientB>(&mut app), 0);
    }

    /// How many pieces the audit finds a client holding wrong.
    fn audit_pieces(app: &mut App) -> u32 {
        app.world_mut()
            .run_system_once(|audit: Res<Audit>, w: Audited| check_pieces(&audit, &w))
            .unwrap()
    }

    #[test]
    fn the_audit_counts_each_piece_a_client_holds_wrong() {
        let mut app = app();
        app.init_resource::<Audit>();
        pause(&mut app);
        let id = lift_at(&mut app, GridName::World, ground(0, 0));
        run(&mut app, 3);
        assert_eq!(audit_pieces(&mut app), 0);

        // A pose neither client has heard of yet.
        let home = server_pieces(&mut app)[&id].pose.translation;
        move_piece(&mut app, id, home + Vec3::Y);
        assert_eq!(audit_pieces(&mut app), 2);
        run(&mut app, 3);
        assert_eq!(audit_pieces(&mut app), 0);

        // Out of client B's interest, B is right not to hold it.
        move_piece(&mut app, id, Vec3::new(70.0, home.y, 70.0));
        run(&mut app, 4);
        assert!(!holds(&app, ClientB::ID, id));
        assert_eq!(audit_pieces(&mut app), 0);

        // A sub-grid B holds no game id for.
        app.world_mut()
            .run_system_once(|mut world_b: VoxelWorld<ClientB>| {
                world_b.spawn_sub_grid(Transform::default());
            })
            .unwrap();
        assert_eq!(audit_pieces(&mut app), 1);
    }

    #[test]
    fn a_handover_lands_in_the_frame_its_chunks_first_land_after_it() {
        let piece = Grid::SubGrid(Entity::from_raw_u32(7).unwrap());
        let (world_chunk, piece_chunk) =
            ((Grid::World, IVec3::ZERO), (piece, IVec3::ZERO));
        let beside = (Grid::World, IVec3::X);
        let record = ServerCommit {
            applied: 10,
            handover: true,
            deltas: [world_chunk, piece_chunk].into(),
            beside: [beside, (Grid::World, IVec3::Y)].into(),
        };
        let landed = |at: &[((Grid, IVec3), &[u32])]| -> Landings {
            at.iter()
                .map(|(chunk, frames)| (*chunk, frames.to_vec()))
                .collect()
        };

        // The world chunk landed before the commit too, and the chunk above
        // never lands: client A doesn't have it.
        let together = landed(&[
            (world_chunk, &[4, 12]),
            (piece_chunk, &[12]),
            (beside, &[12]),
        ]);
        assert_eq!(record.landed_in(&together), BTreeSet::from([12]));

        let apart = landed(&[(world_chunk, &[13]), (piece_chunk, &[12])]);
        assert_eq!(record.landed_in(&apart), BTreeSet::from([12, 13]));

        // A neighbour landing between the commit and the rest of the
        // handover landed apart from it. One landing after it shows a later
        // edit, or a neighbour whose padding the commit left alone.
        let early =
            landed(&[(world_chunk, &[12]), (piece_chunk, &[12]), (beside, &[11])]);
        assert_eq!(record.landed_in(&early), BTreeSet::from([11, 12]));
        let later =
            landed(&[(world_chunk, &[12]), (piece_chunk, &[12]), (beside, &[15])]);
        assert_eq!(record.landed_in(&later), BTreeSet::from([12]));
    }

    #[test]
    fn the_drop_fault_spares_spawns_and_full_deltas() {
        let mut lane = Lane {
            drop: true,
            ..default()
        };
        let delta = |full| EditDelta {
            chunk: IVec3::ZERO,
            seq: 1,
            full,
            voxels: bevy_voxel_world::edits::Voxels::Sparse(vec![(0, WorldVoxel::Air)]),
        };
        let pose = PieceAt {
            piece: 0,
            parent: None,
            pose: Transform::IDENTITY.into(),
        };
        let sent = |lane: &mut Lane, msg: ServerFrame| {
            let before = lane.to_client[ClientB::ID].len();
            lane.outbox[ClientB::ID] = msg;
            lane.send_frame(ClientB::ID);
            lane.to_client[ClientB::ID].len() > before
        };
        for _ in 0..200 {
            let spawn = ServerFrame {
                spawns: vec![pose],
                poses: vec![pose],
                ..default()
            };
            assert!(sent(&mut lane, spawn), "a spawn arrives");
            let full = ServerFrame {
                deltas: vec![(GridName::Piece(0), delta(true))],
                poses: vec![pose],
                ..default()
            };
            assert!(sent(&mut lane, full), "a full delta arrives");
        }
        let dropped = (0..200)
            .filter(|_| {
                let incremental = ServerFrame {
                    deltas: vec![(GridName::World, delta(false))],
                    poses: vec![pose],
                    ..default()
                };
                !sent(&mut lane, incremental)
            })
            .count();
        assert!(dropped > 0, "the fault drops other messages");
    }
}
