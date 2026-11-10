//! A server and two clients in one app, keeping their voxel edits in sync
//! through edit deltas sent over a fake in-memory network.
//!
//! The server is a `VoxelWorldPlugin::minimal()` world with no camera: it holds
//! the edits and their edit sequences, commits them and writes
//! `EditsCommitted`, but spawns no chunks. A server that needs collision or
//! other voxel queries around its players gives each player a
//! `VoxelWorldCamera` of the server world instead. Clients A and B render the
//! same terrain from two angles, A on the left half of the window and B on the
//! right.
//!
//! Hover a voxel and it gets a white outline. Left click asks the server to dig
//! it, right click to place the client's own material beside it (A lays brick,
//! B lays snow). Clients show confirmed edits only: a yellow outline marks a
//! request until a delta carrying that voxel arrives, and a red one a request
//! the server refused. The server accepts edits inside the ring drawn on the
//! ground. The log at the bottom narrates everything that is not a plain
//! delivery.
//!
//! The lane to client A is clean. The lane to client B misbehaves on request:
//!
//! - `1` delays every message to and from B.
//! - `2` drops some incremental deltas to B.
//! - `3` reorders deltas to B.
//! - `J` makes B forget its edits and join again.
//! - `R` makes the server clear every edit.
//! - `V` cycles the view: split, A only, B only.
//!
//! Every message on the lane is `postcard` bytes, so swapping the lane for a
//! real transport changes only `Lane`. The lane assumes what a real reliable,
//! ordered channel gives, unless a fault key is on.
//!
//! How a client recovers, as game messages built on the crate's API:
//!
//! - **Gap**: `apply_delta` returns an `EditGap` when a delta doesn't continue
//!   from the client's edit sequence for its chunk. The client asks for the
//!   chunk's `full_delta`.
//! - **Join**: the server sends the `full_delta` of every chunk in
//!   `edited_chunks()` that the client's interest covers. A client joining
//!   again calls `forget_edits` first.
//! - **Heartbeat**: now and then the server sends each client its
//!   `edited_chunks()`, and the client asks for the full delta of any chunk
//!   whose sequence it is behind on. Only this catches a dropped last delta,
//!   since no later delta shows the gap.
//! - **Reset**: `clear_edits` on the server reaches clients as empty full
//!   deltas, through the same forwarding as any other edit.
//!
//! Run with `cargo run --example multiplayer_loopback --features serde`. With
//! `-- audit` it plays a script instead of taking input: it edits through each
//! fault, re-join and reset, checks both client worlds against the server after
//! each, and exits with an error if any differ. `-- audit <file.png>` also
//! saves a frame of both views partway through.

use std::{collections::VecDeque, sync::Arc};

use bevy::{
    app::{HierarchyPropagatePlugin, Propagate},
    camera::{visibility::RenderLayers, Viewport},
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

/// How many frames the delay fault adds to a message.
const DELAY_FRAMES: u32 = 45;
/// Up to how many frames the reorder fault adds to a delta.
const REORDER_FRAMES: u32 = 12;
/// The share of incremental deltas the drop fault loses.
const DROP_PERCENT: u32 = 30;
/// How often the server sends each client its edited chunks.
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

fn name(client: usize) -> &'static str {
    [ClientA::NAME, ClientB::NAME][client]
}

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
                // Chunks are children of the root, so one component on the
                // root puts every chunk of this world on its client's layer.
                let layer: Option<usize> = $layer;
                if let Some(layer) = layer {
                    commands
                        .entity(root)
                        .insert(Propagate(RenderLayers::layer(layer)));
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

/// A message from a client to the server. Positions are arrays because glam's
/// own serde support is a feature this example doesn't enable.
#[derive(Serialize, Deserialize)]
enum ToServer {
    /// Asks the server to set a voxel. The server checks it first.
    Set {
        pos: [i32; 3],
        voxel: WorldVoxel<u8>,
    },
    /// Asks for a chunk's full delta, after a gap or a heartbeat showed the
    /// client is behind on it.
    RequestFull { chunk: [i32; 3] },
    /// Asks for every edited chunk the client's interest covers.
    Join,
}

/// A message from the server to a client.
#[derive(Serialize, Deserialize)]
enum ToClient {
    Delta(EditDelta<u8>),
    /// The heartbeat: the server's edited chunks in the client's interest,
    /// each with its edit sequence.
    EditedChunks(Vec<([i32; 3], u64)>),
    /// The server refused the client's request to set this voxel.
    Rejected {
        pos: [i32; 3],
    },
}

/// One message on its way, as the bytes a transport would carry.
struct Packet {
    deliver_at: u32,
    bytes: Vec<u8>,
}

/// The in-memory network between the server and the clients, indexed by
/// `Client::ID`, with the faults on the lane to client B and a log of what
/// happened on it.
#[derive(Resource, Default)]
struct Lane {
    frame: u32,
    to_server: [VecDeque<Packet>; 2],
    to_client: [Vec<Packet>; 2],
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
    /// deltas when delay is on, so B sees its own edit after a full round trip.
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

    /// Drop and reorder touch deltas only, and drop only incremental ones: a
    /// lost full delta would leave nothing for the recovery paths to repair.
    fn send_to_client(&mut self, client: usize, msg: &ToClient) {
        let flaky = client == ClientB::ID;
        let mut deliver_at = self.frame + 1 + self.delay(client);
        if let ToClient::Delta(delta) = msg {
            if flaky
                && self.drop
                && !delta.full
                && rand::random_range(0..100) < DROP_PERCENT
            {
                self.stats.dropped += 1;
                self.note(format!(
                    "lane dropped delta to client B: chunk {} seq {}",
                    delta.chunk, delta.seq
                ));
                return;
            }
            if flaky && self.reorder {
                deliver_at += rand::random_range(0..REORDER_FRAMES);
            }
            self.stats.deltas_sent[client] += 1;
        }
        let bytes = postcard::to_allocvec(msg).unwrap();
        self.to_client[client].push(Packet { deliver_at, bytes });
    }

    fn take_due(&mut self, client: usize) -> Vec<ToClient> {
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

/// What the outlines show, by `Client::ID`.
#[derive(Resource, Default)]
struct Feedback {
    /// Voxels each client asked to set and has not seen a delta for.
    pending: [Vec<IVec3>; 2],
    /// Voxels the server refused, until the frame their flash ends.
    rejected: [Vec<(IVec3, u32)>; 2],
    /// The voxel under the cursor, in one client's view.
    hover: Option<Hover>,
}

#[derive(Clone, Copy)]
struct Hover {
    client: usize,
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

#[derive(Component)]
struct ViewCamera(usize);

#[derive(Component)]
struct ClientHud(usize);

#[derive(Component)]
struct BottomHud;

// ---------------------------------------------------------------------------

fn main() {
    let audit = std::env::args().nth(1).as_deref() == Some("audit");
    let mut app = App::new();
    app.add_plugins(DefaultPlugins)
        .add_plugins(HierarchyPropagatePlugin::<RenderLayers>::new(PostUpdate))
        .add_plugins(VoxelWorldPlugin::<Server>::minimal())
        .add_plugins(VoxelWorldPlugin::with_config(ClientA))
        .add_plugins(VoxelWorldPlugin::with_config(ClientB))
        .init_gizmo_group::<GizmosA>()
        .init_gizmo_group::<GizmosB>()
        .init_resource::<Lane>()
        .init_resource::<Feedback>()
        .insert_resource(ViewMode::Split)
        .add_systems(Startup, setup)
        // Everything that touches edits runs before the worlds' own sets in
        // `PreUpdate`: clients apply what arrived, then the server sets what it
        // was asked to. Each world commits in its set, and the server forwards
        // its commit's deltas right after. Systems that only read the worlds,
        // like input, panels and outlines, run in `Update`.
        .add_systems(
            PreUpdate,
            (
                (
                    tick_lane,
                    client_receive::<ClientA>,
                    client_receive::<ClientB>,
                    server_handle_requests,
                    server_heartbeat,
                )
                    .chain()
                    .before(VoxelWorldSet::<Server>::default())
                    .before(VoxelWorldSet::<ClientA>::default())
                    .before(VoxelWorldSet::<ClientB>::default()),
                forward_deltas.after(VoxelWorldSet::<Server>::default()),
            ),
        )
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
        app.init_resource::<Audit>().add_systems(Update, script);
    } else {
        app.add_systems(Update, (player_input, keys));
    }
    app.run();
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

    // Client B leaves out a corner of the ring, so an edit there reaches A
    // only.
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
    commands.insert_resource(Interest([interest_a, interest_b]));

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
        rejected.retain(|(_, until)| *until > frame);
    }
}

/// Applies the deltas that arrived for client `C`, and asks the server for a
/// chunk's full delta when a delta leaves a gap or the heartbeat shows the
/// client behind. The client's own `EditsCommitted` messages are ignored: they
/// only repeat what the server sent.
fn client_receive<C: Client>(
    mut lane: ResMut<Lane>,
    mut feedback: ResMut<Feedback>,
    mut world: VoxelWorld<C>,
) {
    // The client's edit sequences once this frame's deltas commit. A delta
    // applied this frame commits after this system, so `edited_chunks` alone
    // would still show the old sequence.
    let mut local: HashMap<IVec3, u64> = world.edited_chunks().into_iter().collect();
    for msg in lane.take_due(C::ID) {
        match msg {
            ToClient::Delta(delta) => {
                let (chunk, seq) = (delta.chunk, delta.seq);
                if delta.full {
                    lane.stats.full_deltas[C::ID] += 1;
                    lane.note(format!(
                        "{} got full delta for chunk {chunk} ({} edits, seq {seq})",
                        C::NAME,
                        delta.len(),
                    ));
                }
                // A full delta carries every edit of its chunk, so one sent
                // after the request carries the requested voxel too, and one
                // sent before leaves it pending.
                let carried: Vec<IVec3> = feedback.pending[C::ID]
                    .iter()
                    .copied()
                    .filter(|p| delta.iter().any(|(pos, _)| pos == *p))
                    .collect();
                match world.apply_delta(delta) {
                    Ok(()) => {
                        let reached = local.entry(chunk).or_default();
                        *reached = (*reached).max(seq);
                        feedback.pending[C::ID].retain(|p| !carried.contains(p));
                    }
                    Err(gap) => {
                        lane.stats.gaps[C::ID] += 1;
                        lane.note(format!(
                            "{} gap in chunk {chunk}: has seq {}, got {}; asks for full",
                            C::NAME,
                            gap.local,
                            gap.received
                        ));
                        let chunk = chunk.to_array();
                        lane.send_to_server(C::ID, &ToServer::RequestFull { chunk });
                    }
                }
            }
            ToClient::EditedChunks(chunks) => {
                for (chunk, seq) in chunks {
                    let at = IVec3::from_array(chunk);
                    let has = local.get(&at).copied().unwrap_or(0);
                    if has < seq {
                        lane.note(format!(
                            "heartbeat: {} behind in chunk {at} ({has} < {seq}); asks for full",
                            C::NAME
                        ));
                        lane.send_to_server(C::ID, &ToServer::RequestFull { chunk });
                    }
                }
            }
            ToClient::Rejected { pos } => {
                let pos = IVec3::from_array(pos);
                feedback.pending[C::ID].retain(|p| *p != pos);
                feedback.rejected[C::ID].push((pos, lane.frame + FLASH_FRAMES));
            }
        }
    }
}

/// Sets the voxels clients asked for, if they pass the server's check, and
/// answers requests for full deltas.
fn server_handle_requests(
    mut lane: ResMut<Lane>,
    mut server: VoxelWorld<Server>,
    interest: Res<Interest>,
) {
    for client in [ClientA::ID, ClientB::ID] {
        for msg in lane.take_due_requests(client) {
            match msg {
                ToServer::Set { pos, voxel } => {
                    let at = IVec3::from_array(pos);
                    // The server's one rule. It could also run the world's
                    // generator here, for example to refuse digging into the
                    // bottom layer.
                    if (at.as_vec3() + 0.5).xz().distance(ARENA.xz()) > REACH {
                        lane.stats.rejected += 1;
                        lane.note(format!(
                            "server rejected {} at {at}: outside the ring",
                            name(client)
                        ));
                        lane.send_to_client(client, &ToClient::Rejected { pos });
                        continue;
                    }
                    server.set_voxel(at, voxel);
                }
                ToServer::RequestFull { chunk } => {
                    let full = server.full_delta(IVec3::from_array(chunk));
                    lane.send_to_client(client, &ToClient::Delta(full));
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
                        lane.send_to_client(
                            client,
                            &ToClient::Delta(server.full_delta(chunk)),
                        );
                    }
                }
            }
        }
    }
}

/// Now and then sends each client the server's edited chunks in its interest,
/// with their edit sequences, so a client can notice a lost last delta.
fn server_heartbeat(
    mut lane: ResMut<Lane>,
    server: VoxelWorld<Server>,
    interest: Res<Interest>,
) {
    if !lane.frame.is_multiple_of(HEARTBEAT_FRAMES) {
        return;
    }
    let edited = server.edited_chunks();
    for client in [ClientA::ID, ClientB::ID] {
        let chunks: Vec<_> = edited
            .iter()
            .filter(|(chunk, _)| interest.0[client].contains(chunk))
            .map(|(chunk, seq)| (chunk.to_array(), *seq))
            .collect();
        if !chunks.is_empty() {
            lane.send_to_client(client, &ToClient::EditedChunks(chunks));
        }
    }
}

/// Sends each delta the server committed to every client whose interest
/// covers its chunk.
fn forward_deltas(
    mut committed: MessageReader<EditsCommitted<Server>>,
    interest: Res<Interest>,
    mut lane: ResMut<Lane>,
) {
    for msg in committed.read() {
        let chunk = msg.delta.chunk;
        let to: Vec<usize> = [ClientA::ID, ClientB::ID]
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
            lane.send_to_client(client, &ToClient::Delta(msg.delta.clone()));
        }
    }
}

// ---------------------------------------------------------------------------
// Player actions.

/// A client asks the server to set a voxel, and outlines it until a delta
/// carrying it arrives.
fn request(
    lane: &mut Lane,
    feedback: &mut Feedback,
    client: usize,
    pos: IVec3,
    voxel: WorldVoxel<u8>,
) {
    feedback.pending[client].push(pos);
    let pos = pos.to_array();
    lane.send_to_server(client, &ToServer::Set { pos, voxel });
}

/// Client B forgets its edits and joins again, as a client reconnecting
/// would. It never clears instead: a clear moves its edit sequences past the
/// server's, and the server's next deltas would be skipped as already
/// applied.
fn rejoin_b(lane: &mut Lane, feedback: &mut Feedback, world_b: &mut VoxelWorld<ClientB>) {
    world_b.forget_edits();
    // Whatever was on its way belonged to the old connection.
    lane.to_client[ClientB::ID].clear();
    feedback.pending[ClientB::ID].clear();
    lane.note("client B forgets its edits and joins again".into());
    lane.send_to_server(ClientB::ID, &ToServer::Join);
}

/// The server clears every edit. Clients get the cleared chunks as empty full
/// deltas through `forward_deltas`, with nothing else to wire.
fn reset(lane: &mut Lane, feedback: &mut Feedback, server: &mut VoxelWorld<Server>) {
    server.clear_edits();
    feedback.pending = default();
    lane.note("server clears every edit".into());
}

/// Tracks the voxel under the cursor in whichever view it is over. Left click
/// asks to dig it, right click asks to place the client's material beside it.
fn player_input(
    buttons: Res<ButtonInput<MouseButton>>,
    window: Single<&Window, With<PrimaryWindow>>,
    cameras: Query<(&Camera, &GlobalTransform, &ViewCamera)>,
    world_a: VoxelWorld<ClientA>,
    world_b: VoxelWorld<ClientB>,
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
        let hit = if view.0 == ClientA::ID {
            world_a.raycast(ray, &|_| true)
        } else {
            world_b.raycast(ray, &|_| true)
        };
        let Some(hit) = hit else {
            continue;
        };
        let hover = Hover {
            client: view.0,
            dig: hit.voxel_pos(),
            place: hit.voxel_normal().map(|n| hit.voxel_pos() + n),
        };
        feedback.hover = Some(hover);
        let material = [ClientA::PLACES, ClientB::PLACES][view.0];
        if buttons.just_pressed(MouseButton::Left) {
            request(&mut lane, &mut feedback, view.0, hover.dig, WorldVoxel::Air);
        } else if let Some(place) = hover
            .place
            .filter(|_| buttons.just_pressed(MouseButton::Right))
        {
            request(
                &mut lane,
                &mut feedback,
                view.0,
                place,
                WorldVoxel::Solid(material),
            );
        }
    }
}

fn keys(
    keys: Res<ButtonInput<KeyCode>>,
    mut lane: ResMut<Lane>,
    mut feedback: ResMut<Feedback>,
    mut view: ResMut<ViewMode>,
    mut server: VoxelWorld<Server>,
    mut world_b: VoxelWorld<ClientB>,
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
    if keys.just_pressed(KeyCode::KeyJ) {
        rejoin_b(&mut lane, &mut feedback, &mut world_b);
    }
    if keys.just_pressed(KeyCode::KeyR) {
        reset(&mut lane, &mut feedback, &mut server);
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
/// ring.
fn draw_markers<C: Client>(
    feedback: Res<Feedback>,
    ring: Res<Ring>,
    mut gizmos: Gizmos<C::Gizmos>,
) {
    let mut outline = |pos: IVec3, color: Color| {
        gizmos.primitive_3d(
            &Cuboid::from_length(1.04),
            Isometry3d::from_translation(pos.as_vec3() + 0.5),
            color,
        );
    };
    if let Some(hover) = feedback.hover.filter(|h| h.client == C::ID) {
        outline(hover.dig, Color::WHITE);
    }
    for pos in &feedback.pending[C::ID] {
        outline(*pos, Color::srgb(1.0, 0.85, 0.1));
    }
    for (pos, _) in &feedback.rejected[C::ID] {
        outline(*pos, Color::srgb(1.0, 0.15, 0.1));
    }
    gizmos.linestrip(ring.0.iter().copied(), Color::srgba(1.0, 1.0, 1.0, 0.7));
}

/// Each client's panel lists the server's edited chunks with the client's
/// edit sequence against the server's, marking the chunks where they differ.
#[allow(clippy::too_many_arguments)]
fn update_hud(
    lane: Res<Lane>,
    feedback: Res<Feedback>,
    interest: Res<Interest>,
    server: VoxelWorld<Server>,
    world_a: VoxelWorld<ClientA>,
    world_b: VoxelWorld<ClientB>,
    cameras: Query<(&Camera, &ViewCamera)>,
    mut huds: Query<(&mut Text, &mut Node, &ClientHud)>,
    mut bottom: Single<&mut Text, (With<BottomHud>, Without<ClientHud>)>,
) {
    let mut server_chunks = server.edited_chunks();
    server_chunks.sort_by_key(|(c, _)| (c.y, c.z, c.x));
    for (mut text, mut node, ClientHud(id)) in &mut huds {
        let active = cameras.iter().any(|(c, v)| v.0 == *id && c.is_active);
        node.display = if active { Display::Flex } else { Display::None };
        let (name, places, local) = if *id == ClientA::ID {
            (ClientA::NAME, ClientA::PLACES_NAME, world_a.edited_chunks())
        } else {
            (ClientB::NAME, ClientB::PLACES_NAME, world_b.edited_chunks())
        };
        let mut s = format!(
            "{name}: left click digs, right click lays {places}\n\
             waiting {}   in flight {}   gaps {}   full deltas {}\n",
            feedback.pending[*id].len(),
            lane.to_client[*id].len(),
            lane.stats.gaps[*id],
            lane.stats.full_deltas[*id],
        );
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
         [J] B joins again  [R] server reset  [V] view\n",
        on(lane.delay),
        on(lane.drop),
        on(lane.reorder)
    ));
    s.push_str(&format!(
        "deltas to A {}  to B {}   dropped {}   rejected {}   edited chunks {}",
        lane.stats.deltas_sent[ClientA::ID],
        lane.stats.deltas_sent[ClientB::ID],
        lane.stats.dropped,
        lane.stats.rejected,
        server_chunks.len(),
    ));
    ***bottom = s;
}

// ---------------------------------------------------------------------------
// The scripted audit, run with `-- audit`.

/// Every voxel the script asked for, so the audit can check those whose edit
/// is gone show the generated voxel again, and how many checks failed.
#[derive(Resource, Default)]
struct Audit {
    touched: HashSet<IVec3>,
    failures: u32,
}

/// Edits from both clients through each fault, then a re-join and a reset,
/// auditing both client worlds against the server after each step. Frames
/// count from the first `PreUpdate`.
#[allow(clippy::too_many_arguments)]
fn script(
    mut lane: ResMut<Lane>,
    mut feedback: ResMut<Feedback>,
    mut audit: ResMut<Audit>,
    interest: Res<Interest>,
    mut server: VoxelWorld<Server>,
    world_a: VoxelWorld<ClientA>,
    mut world_b: VoxelWorld<ClientB>,
    cameras: Query<(&Camera, &GlobalTransform, &ViewCamera)>,
    mut exit: MessageWriter<AppExit>,
    mut commands: Commands,
    mut generate: Local<Option<Generator>>,
) {
    let generate = generate.get_or_insert_with(generator);
    let mut ask = |lane: &mut Lane, client: usize, pos: IVec3, voxel: WorldVoxel<u8>| {
        audit.touched.insert(pos);
        request(lane, &mut feedback, client, pos, voxel);
    };
    let frame = lane.frame;
    match frame {
        120 => {
            lane.drop = true;
            lane.reorder = true;
        }
        // A pit one layer per frame, so several incremental deltas travel per
        // chunk and the drop fault has some to lose.
        130..=135 => {
            let depth = (frame - 130) as i32;
            for dx in 3..=9 {
                for dz in 3..=9 {
                    let y = surface(generate, dx, dz) - depth;
                    ask(
                        &mut lane,
                        ClientA::ID,
                        IVec3::new(dx, y, dz),
                        WorldVoxel::Air,
                    );
                }
            }
        }
        140..=145 => {
            let h = (frame - 140) as i32;
            for z in -4..4 {
                let y = surface(generate, -8, z) + 1 + h;
                ask(
                    &mut lane,
                    ClientA::ID,
                    IVec3::new(-8, y, z),
                    WorldVoxel::Solid(BRICK),
                );
            }
        }
        150..=159 => {
            let x = (frame - 150) as i32 - 5;
            let y = surface(generate, x, -12) + 1;
            ask(
                &mut lane,
                ClientB::ID,
                IVec3::new(x, y, -12),
                WorldVoxel::Solid(SNOW),
            );
        }
        // In the corner client B's interest leaves out.
        165 => {
            let y = surface(generate, 40, 40) + 1;
            ask(
                &mut lane,
                ClientA::ID,
                IVec3::new(40, y, 40),
                WorldVoxel::Solid(BRICK),
            );
        }
        // Outside the ring: the server refuses it.
        166 => {
            let y = surface(generate, 0, 70) + 1;
            ask(
                &mut lane,
                ClientA::ID,
                IVec3::new(0, y, 70),
                WorldVoxel::Solid(BRICK),
            );
        }
        400 => {
            check(
                &mut audit,
                "drop and reorder",
                &interest,
                &server,
                &world_a,
                &world_b,
                generate,
            );
            lane.drop = false;
            lane.reorder = false;
            lane.delay = true;
        }
        410..=415 => {
            let z = (frame - 410) as i32 - 2;
            let y = surface(generate, 12, z);
            ask(
                &mut lane,
                ClientB::ID,
                IVec3::new(12, y, z),
                WorldVoxel::Air,
            );
        }
        // With a path after `audit`, a frame of both views goes there.
        590 => {
            if let Some(path) = std::env::args().nth(2) {
                commands
                    .spawn(Screenshot::primary_window())
                    .observe(save_to_disk(path));
            }
        }
        600 => {
            check(
                &mut audit, "delay", &interest, &server, &world_a, &world_b, generate,
            );
            lane.delay = false;
            rejoin_b(&mut lane, &mut feedback, &mut world_b);
        }
        720 => {
            check(
                &mut audit,
                "join again",
                &interest,
                &server,
                &world_a,
                &world_b,
                generate,
            );
            reset(&mut lane, &mut feedback, &mut server);
        }
        880 => {
            check(
                &mut audit, "reset", &interest, &server, &world_a, &world_b, generate,
            );
            // Both views look at `LOOK_AT`, so a ray through the centre of
            // each viewport lands near it if the cursor maths holds for both
            // halves of the window.
            for (camera, gtf, view) in &cameras {
                let hit = camera
                    .logical_viewport_rect()
                    .and_then(|rect| camera.viewport_to_world(gtf, rect.center()).ok())
                    .and_then(|ray| match view.0 {
                        0 => world_a.raycast(ray, &|_| true),
                        _ => world_b.raycast(ray, &|_| true),
                    })
                    .map(|hit| hit.voxel_pos());
                let near =
                    hit.is_some_and(|h| h.as_vec3().xz().distance(LOOK_AT.xz()) < 12.0);
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

/// Checks both client worlds against the server: each chunk's full delta, and
/// the voxel `get_voxel` reads at every position the script edited.
fn check(
    audit: &mut Audit,
    step: &str,
    interest: &Interest,
    server: &VoxelWorld<Server>,
    world_a: &VoxelWorld<ClientA>,
    world_b: &VoxelWorld<ClientB>,
    generate: &mut dyn FnMut(IVec3, Option<WorldVoxel<u8>>) -> WorldVoxel<u8>,
) {
    let mut wrong = 0;
    let mut chunks: HashSet<IVec3> =
        server.edited_chunks().into_iter().map(|(c, _)| c).collect();
    chunks.extend(world_a.edited_chunks().into_iter().map(|(c, _)| c));
    chunks.extend(world_b.edited_chunks().into_iter().map(|(c, _)| c));
    for client in [ClientA::ID, ClientB::ID] {
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
    for pos in &audit.touched {
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
        "audit after {step}: {} chunks and {} voxels checked, {wrong} differ",
        chunks.len(),
        audit.touched.len()
    );
    audit.failures += wrong;
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
