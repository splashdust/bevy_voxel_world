//! A server and two clients in one app, keeping their voxel edits and their
//! sub-grids in sync through messages sent over a fake in-memory network.
//!
//! The server is a `VoxelWorldPlugin::minimal()` world with no camera: it
//! holds the edits, commits them and writes `EditsCommitted`, but streams no
//! chunks. Clients A and B render the same terrain from two angles, A on the
//! left half of the window and B on the right. Clients only ever show what
//! the server sent back.
//!
//! Left click asks the server to dig the voxel under the cursor, right click
//! to place the client's own material beside it (A lays brick, B lays snow).
//! `L` asks to lift the solid voxels around it out of the world into a piece,
//! which the server raises. `K` asks to stamp the piece under the cursor back
//! into the world.
//!
//! A piece is the game's name for a sub-grid it syncs. The server gives each
//! piece a game id, and each world maps that id to its own entity. The server
//! sends each client one message per frame: the frame's piece spawns, its
//! deltas, every piece's pose and its piece despawns, applied in that order.
//! So a lift lands on a client in one frame, as the piece's spawn, its deltas
//! and the world's deltas setting air. A stamp lands as the world's deltas,
//! the piece's deltas emptying it and its despawn, which the client routes
//! through `despawn_sub_grid` so the piece shows until the world's chunks
//! taking its voxels do.
//!
//! Every message is `postcard` bytes, so swapping the lane for a real
//! transport changes only `Lane`. The lane is reliable and ordered, which is
//! what deltas need. A real game also checks each request on the server,
//! sends each client only the chunks and pieces near it, and recovers from
//! what its channel lets through, with the gap, join and heartbeat flows the
//! `edits` module docs describe.
//!
//! Run with `cargo run --example multiplayer_loopback --features
//! serde,sub_grids`.

use std::sync::Arc;

use bevy::{
    app::{HierarchyPropagatePlugin, Propagate},
    camera::{visibility::RenderLayers, Viewport},
    platform::collections::HashMap,
    prelude::*,
    ui::UiTargetCamera,
    window::PrimaryWindow,
};
use bevy_voxel_world::prelude::*;
use serde::{Deserialize, Serialize};

// Materials. The default texture atlas holds grass, brick, stone and snow,
// from the bottom.
const GRASS: u8 = 0;
const STONE: u8 = 1;
const BRICK: u8 = 2;
const SNOW: u8 = 3;

/// How far a ray from the cursor reaches.
const RAY_LENGTH: f32 = 500.0;
/// How far a lift reaches from the hovered voxel along each axis.
const LIFT_REACH: i32 = 2;
/// How many voxels a lifted piece rises, and how far it rises each frame.
const RISE: f32 = 10.0;
const RISE_STEP: f32 = 0.05;

// ---------------------------------------------------------------------------
// Three worlds: one server, two clients.

#[derive(Resource, Clone, Default)]
struct Server;
#[derive(Resource, Clone, Default)]
struct ClientA;
#[derive(Resource, Clone, Default)]
struct ClientB;

/// What tells the two clients apart: a lane slot and render layer, a camera
/// position and the material it places.
trait Client: VoxelWorldConfig<MaterialIndex = u8> + Default {
    const ID: usize;
    const CAMERA: Vec3;
    const PLACES: u8;
}

impl Client for ClientA {
    const ID: usize = 0;
    const CAMERA: Vec3 = Vec3::new(-44.0, 30.0, -44.0);
    const PLACES: u8 = BRICK;
}

impl Client for ClientB {
    const ID: usize = 1;
    const CAMERA: Vec3 = Vec3::new(44.0, 30.0, -44.0);
    const PLACES: u8 = SNOW;
}

const CLIENTS: [usize; 2] = [ClientA::ID, ClientB::ID];

/// The render layer of a client's chunks and camera.
fn layer(client: usize) -> RenderLayers {
    RenderLayers::layer(client + 1)
}

macro_rules! world_config {
    ($world:ty, $client:expr) => {
        impl VoxelWorldConfig for $world {
            type MaterialIndex = u8;
            type ChunkUserBundle = ();

            fn spawning_distance(&self) -> u32 {
                5
            }

            /// Flat ground: grass over stone.
            fn voxel_lookup_delegate(&self) -> VoxelLookupDelegate<u8> {
                Box::new(|_chunk_pos, _lod, _previous| {
                    Box::new(|pos, _previous| match pos.y {
                        ..-2 => WorldVoxel::Solid(STONE),
                        ..0 => WorldVoxel::Solid(GRASS),
                        _ => WorldVoxel::Air,
                    })
                })
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
                // Chunks and sub-grids are children of the root, so one
                // component on the root puts every chunk of this world on its
                // client's layer, a sub-grid's chunks included.
                let client: Option<usize> = $client;
                if let Some(client) = client {
                    commands.entity(root).insert(Propagate(layer(client)));
                }
            }
        }
    };
}

world_config!(Server, None);
world_config!(ClientA, Some(ClientA::ID));
world_config!(ClientB, Some(ClientB::ID));

// ---------------------------------------------------------------------------
// The messages and the fake network.

/// A grid as messages name it: the world, or a piece by its game id.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum GridName {
    World,
    Piece(u32),
}

/// A piece and its translation under the world's root, as an array because
/// glam's own serde support is a feature this example doesn't enable. Pieces
/// never turn here, so the rotation isn't sent.
#[derive(Serialize, Deserialize, Clone, Copy)]
struct PieceAt {
    piece: u32,
    translation: [f32; 3],
}

impl PieceAt {
    fn new(piece: u32, translation: Vec3) -> Self {
        let translation = translation.to_array();
        Self { piece, translation }
    }
}

/// A message from a client to the server.
#[derive(Serialize, Deserialize)]
enum ToServer {
    Set {
        grid: GridName,
        pos: [i32; 3],
        voxel: WorldVoxel<u8>,
    },
    /// Asks to lift the solid voxels of the world in the box from `min`
    /// inclusive to `max` exclusive into a new piece.
    Lift {
        min: [i32; 3],
        max: [i32; 3],
    },
    Stamp {
        piece: u32,
    },
}

/// Everything the server sends one client in one frame. A client applies the
/// parts in the order of the fields.
#[derive(Serialize, Deserialize, Default)]
struct ServerFrame {
    spawns: Vec<PieceAt>,
    deltas: Vec<(GridName, EditDelta<u8>)>,
    poses: Vec<PieceAt>,
    despawns: Vec<u32>,
}

/// The in-memory network between the server and the clients, indexed by
/// `Client::ID`. Each message arrives the frame after it was sent, in order.
#[derive(Resource, Default)]
struct Lane {
    to_server: [Vec<Vec<u8>>; 2],
    to_client: [Vec<Vec<u8>>; 2],
    /// What the server sends each client at the end of this frame.
    outbox: [ServerFrame; 2],
}

impl Lane {
    fn send_to_server(&mut self, client: usize, msg: &ToServer) {
        self.to_server[client].push(postcard::to_allocvec(msg).unwrap());
    }

    fn take_requests(&mut self, client: usize) -> Vec<ToServer> {
        let packets = std::mem::take(&mut self.to_server[client]);
        packets
            .iter()
            .map(|p| postcard::from_bytes(p).unwrap())
            .collect()
    }

    fn send_frame(&mut self, client: usize) {
        let msg = std::mem::take(&mut self.outbox[client]);
        self.to_client[client].push(postcard::to_allocvec(&msg).unwrap());
    }

    fn take_frames(&mut self, client: usize) -> Vec<ServerFrame> {
        let packets = std::mem::take(&mut self.to_client[client]);
        packets
            .iter()
            .map(|p| postcard::from_bytes(p).unwrap())
            .collect()
    }
}

// ---------------------------------------------------------------------------
// The server's pieces and the clients' copies of them.

#[derive(Resource, Default)]
struct ServerPieces {
    next_id: u32,
    pieces: HashMap<u32, ServerPiece>,
    /// The game id of each piece entity, kept for a stamped piece until its
    /// `Stamped` is read, so the deltas emptying it still reach the clients.
    ids: HashMap<Entity, u32>,
}

struct ServerPiece {
    entity: Entity,
    /// The piece's translation, which the server keeps on the entity and
    /// sends every frame.
    translation: Vec3,
    risen: f32,
}

/// Each client's map between game ids and its own entities, by `Client::ID`.
#[derive(Resource, Default)]
struct Replicas([Replica; 2]);

#[derive(Default)]
struct Replica {
    entities: HashMap<u32, Entity>,
    ids: HashMap<Entity, u32>,
}

impl Replica {
    fn name(&self, grid: Grid) -> Option<GridName> {
        match grid {
            Grid::World => Some(GridName::World),
            Grid::SubGrid(entity) => self.ids.get(&entity).copied().map(GridName::Piece),
        }
    }
}

// ---------------------------------------------------------------------------

#[derive(Component)]
struct ViewCamera(usize);

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(HierarchyPropagatePlugin::<RenderLayers>::new(PostUpdate))
        .add_plugins(VoxelWorldPlugin::<Server>::minimal())
        .add_plugins(VoxelWorldPlugin::with_config(ClientA))
        .add_plugins(VoxelWorldPlugin::with_config(ClientB))
        .init_resource::<Lane>()
        .init_resource::<ServerPieces>()
        .init_resource::<Replicas>()
        .add_systems(Startup, setup)
        // Everything that touches edits runs before the worlds' own sets in
        // `PreUpdate`: clients apply what arrived, then the server handles
        // what it was asked and moves its pieces. Each world commits in its
        // set, and the server sends each client its frame's message right
        // after.
        .add_systems(
            PreUpdate,
            (
                (
                    client_receive::<ClientA>,
                    client_receive::<ClientB>,
                    server_handle_requests,
                    raise_pieces,
                )
                    .chain()
                    .before(VoxelWorldSet::<Server>::default())
                    .before(VoxelWorldSet::<ClientA>::default())
                    .before(VoxelWorldSet::<ClientB>::default()),
                server_send.after(VoxelWorldSet::<Server>::default()),
            ),
        )
        .add_systems(Update, (input, layout_viewports))
        .run();
}

fn setup(mut commands: Commands) {
    // Each camera streams and sees only its own client's world.
    fn camera<C: Client>(commands: &mut Commands) -> Entity {
        commands
            .spawn((
                Camera3d::default(),
                Camera {
                    order: C::ID as isize,
                    ..default()
                },
                Transform::from_translation(C::CAMERA).looking_at(Vec3::ZERO, Vec3::Y),
                ViewCamera(C::ID),
                VoxelWorldCamera::<C>::default(),
                layer(C::ID),
            ))
            .id()
    }
    let cam_a = camera::<ClientA>(&mut commands);
    camera::<ClientB>(&mut commands);

    // One light on both layers. It casts no shadows: A's chunks would
    // shadow B's view, so an edit would show there as a shadow first.
    commands.spawn((
        DirectionalLight::default(),
        Transform::default().looking_at(Vec3::new(-0.15, -0.1, 0.15), Vec3::Y),
        layer(ClientA::ID).union(&layer(ClientB::ID)),
    ));

    // The help text is UI on client A's camera, so it sits in A's half. A
    // separate UI camera over the whole window would blank that half.
    commands.spawn((
        Text::new(
            "left click digs, right click places   [L] lift   [K] stamp piece\n\
             A lays brick, B lays snow; each shows only what the server sent",
        ),
        Node {
            position_type: PositionType::Absolute,
            bottom: Val::Px(8.0),
            left: Val::Px(8.0),
            ..default()
        },
        UiTargetCamera(cam_a),
    ));
}

// ---------------------------------------------------------------------------
// The sync systems.

/// Applies the messages that arrived for client `C`. The client's own
/// `EditsCommitted` messages are ignored: they only repeat what the server
/// sent.
///
/// `world` comes before `commands`, so a pose given here replaces the one
/// `spawn_sub_grid` gives: a system's commands apply in the order of its
/// parameters.
fn client_receive<C: Client>(
    mut world: VoxelWorld<C>,
    mut commands: Commands,
    mut lane: ResMut<Lane>,
    mut replicas: ResMut<Replicas>,
) {
    let replica = &mut replicas.0[C::ID];
    for msg in lane.take_frames(C::ID) {
        for at in msg.spawns {
            let pose = Transform::from_translation(at.translation.into());
            let entity = world.spawn_sub_grid(pose);
            replica.entities.insert(at.piece, entity);
            replica.ids.insert(entity, at.piece);
        }

        for (grid, delta) in msg.deltas {
            let applied = match grid {
                GridName::World => world.apply_delta(delta),
                GridName::Piece(id) => {
                    let entity = replica.entities[&id];
                    let mut voxels = world.grid_mut(Grid::SubGrid(entity)).unwrap();
                    voxels.apply_delta(delta)
                }
            };
            applied.expect("a reliable, ordered lane leaves no gaps");
        }

        for at in msg.poses {
            let pose = Transform::from_translation(at.translation.into());
            commands.entity(replica.entities[&at.piece]).insert(pose);
        }

        for id in msg.despawns {
            // Through `despawn_sub_grid`, so a stamped piece shows until the
            // deltas emptying it, applied above, land.
            let entity = replica.entities.remove(&id).unwrap();
            replica.ids.remove(&entity);
            world.despawn_sub_grid(entity);
        }
    }
}

/// Handles what the clients asked for: sets voxels, lifts and stamps. Lifts
/// and stamps send their spawns and despawns here; their deltas follow in
/// `server_send`.
fn server_handle_requests(
    mut server: VoxelWorld<Server>,
    mut lane: ResMut<Lane>,
    mut pieces: ResMut<ServerPieces>,
) {
    for client in CLIENTS {
        for msg in lane.take_requests(client) {
            match msg {
                ToServer::Set { grid, pos, voxel } => {
                    let grid = match grid {
                        GridName::World => Some(Grid::World),
                        GridName::Piece(id) => {
                            pieces.pieces.get(&id).map(|p| Grid::SubGrid(p.entity))
                        }
                    };
                    if let Some(mut voxels) = grid.and_then(|g| server.grid_mut(g)) {
                        voxels.set_voxel(IVec3::from_array(pos), voxel);
                    }
                }
                ToServer::Lift { min, max } => {
                    let (min, max) = (IVec3::from_array(min), IVec3::from_array(max));
                    let Ok(entity) =
                        server.lift(Grid::World, min, max, |_, v| v.is_solid())
                    else {
                        continue;
                    };
                    let id = pieces.next_id;
                    pieces.next_id += 1;
                    let voxels = server.grid(Grid::SubGrid(entity)).unwrap();
                    let translation = voxels.pose().translation();
                    let piece = ServerPiece {
                        entity,
                        translation,
                        risen: 0.0,
                    };
                    pieces.pieces.insert(id, piece);
                    pieces.ids.insert(entity, id);
                    for client in CLIENTS {
                        let at = PieceAt::new(id, translation);
                        lane.outbox[client].spawns.push(at);
                    }
                }
                ToServer::Stamp { piece: id } => {
                    let Some(piece) = pieces.pieces.get(&id) else {
                        continue;
                    };
                    if server
                        .stamp(piece.entity, Grid::World, incoming_replaces)
                        .is_ok()
                    {
                        pieces.pieces.remove(&id);
                        for client in CLIENTS {
                            lane.outbox[client].despawns.push(id);
                        }
                    }
                }
            }
        }
    }
}

/// Raises each lifted piece `RISE` voxels and puts its new translation on
/// its entity. A piece whose entity has no `Transform` yet, one lifted this
/// frame, starts next frame.
fn raise_pieces(mut pieces: ResMut<ServerPieces>, mut transforms: Query<&mut Transform>) {
    for piece in pieces.pieces.values_mut() {
        let Ok(mut transform) = transforms.get_mut(piece.entity) else {
            continue;
        };
        let step = RISE_STEP.min(RISE - piece.risen);
        piece.risen += step;
        piece.translation.y += step;
        transform.translation = piece.translation;
    }
}

/// Adds the deltas the server committed this frame and every piece's pose to
/// each client's message, and sends it.
fn server_send(
    mut lane: ResMut<Lane>,
    mut pieces: ResMut<ServerPieces>,
    mut world_commits: MessageReader<EditsCommitted<Server>>,
    mut piece_commits: MessageReader<SubGridEditsCommitted<Server>>,
    mut stamped: MessageReader<Stamped<Server>>,
) {
    for msg in world_commits.read() {
        for client in CLIENTS {
            let delta = msg.delta.clone();
            lane.outbox[client].deltas.push((GridName::World, delta));
        }
    }
    for msg in piece_commits.read() {
        let id = pieces.ids[&msg.sub_grid];
        for client in CLIENTS {
            let delta = msg.delta.clone();
            lane.outbox[client]
                .deltas
                .push((GridName::Piece(id), delta));
        }
    }
    for stamp in stamped.read() {
        pieces.ids.remove(&stamp.sub_grid);
    }

    for client in CLIENTS {
        for (id, piece) in &pieces.pieces {
            let at = PieceAt::new(*id, piece.translation);
            lane.outbox[client].poses.push(at);
        }
        lane.send_frame(client);
    }
}

// ---------------------------------------------------------------------------
// Player input and the view.

/// Turns clicks, `L` and `K` into requests about the voxel under the cursor,
/// from the client whose view it is over.
#[allow(clippy::too_many_arguments)]
fn input(
    buttons: Res<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
    window: Single<&Window, With<PrimaryWindow>>,
    cameras: Query<(&Camera, &GlobalTransform, &ViewCamera)>,
    world_a: VoxelWorld<ClientA>,
    world_b: VoxelWorld<ClientB>,
    replicas: Res<Replicas>,
    mut lane: ResMut<Lane>,
) {
    let Some(cursor) = window.cursor_position() else {
        return;
    };
    for (camera, cam_gtf, ViewCamera(client)) in &cameras {
        if !camera
            .logical_viewport_rect()
            .is_some_and(|r| r.contains(cursor))
        {
            continue;
        }
        // `viewport_to_world` takes window coordinates: it subtracts the
        // viewport's origin itself.
        let Ok(ray) = camera.viewport_to_world(cam_gtf, cursor) else {
            continue;
        };
        let replica = &replicas.0[*client];
        // Only grids the server knows by name: a piece already despawned on
        // the server but still showing here isn't one.
        let synced = |grid: Grid| replica.name(grid).is_some();
        let hit = if *client == ClientA::ID {
            world_a.raycast_grids(ray, RAY_LENGTH, &synced, &|_, _, _| true)
        } else {
            world_b.raycast_grids(ray, RAY_LENGTH, &synced, &|_, _, _| true)
        };
        let Some((hit, grid)) = hit.and_then(|h| Some((h, replica.name(h.grid)?))) else {
            continue;
        };

        let place = hit.normal.map(|n| (hit.position + n).to_array());
        let msg = if buttons.just_pressed(MouseButton::Left) {
            let pos = hit.position.to_array();
            ToServer::Set {
                grid,
                pos,
                voxel: WorldVoxel::Air,
            }
        } else if let (true, Some(pos)) =
            (buttons.just_pressed(MouseButton::Right), place)
        {
            let material = [ClientA::PLACES, ClientB::PLACES][*client];
            ToServer::Set {
                grid,
                pos,
                voxel: WorldVoxel::Solid(material),
            }
        } else if let (true, GridName::World) = (keys.just_pressed(KeyCode::KeyL), grid) {
            let reach = IVec3::splat(LIFT_REACH);
            let (min, max) = (hit.position - reach, hit.position + reach + 1);
            ToServer::Lift {
                min: min.to_array(),
                max: max.to_array(),
            }
        } else if let (true, GridName::Piece(piece)) =
            (keys.just_pressed(KeyCode::KeyK), grid)
        {
            ToServer::Stamp { piece }
        } else {
            continue;
        };
        lane.send_to_server(*client, &msg);
    }
}

/// Splits the window between the two cameras.
fn layout_viewports(
    window: Single<&Window, With<PrimaryWindow>>,
    mut cameras: Query<(&mut Camera, &ViewCamera)>,
) {
    let half = window.physical_width().max(2) / 2;
    let height = window.physical_height().max(1);
    for (mut camera, ViewCamera(client)) in &mut cameras {
        camera.viewport = Some(Viewport {
            physical_position: UVec2::new(*client as u32 * half, 0),
            physical_size: UVec2::new(half, height),
            ..default()
        });
    }
}
