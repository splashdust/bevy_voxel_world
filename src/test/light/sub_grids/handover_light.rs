//! Tests of handovers in a lit world: a lift or stamp lands with its light,
//! every chunk's mesh and light changing in the same frame.

use std::collections::BTreeSet;
use std::sync::Arc;

use bevy::diagnostic::FrameCount;
use bevy::ecs::system::RunSystemOnce;
use bevy::mesh::MeshTag;
use bevy::prelude::*;

use crate::chunk::Chunk;
use crate::light::route::{LightRoute, HAS_SLOT};
use crate::light::LightSettings;
use crate::mesh_cache::MeshRef;
use crate::prelude::*;
use crate::test::light::edits::{paced_update, settle};
use crate::test::light::mixing::GREEN_LAMP;
use crate::test::light::sky_light::sky_light;
use crate::test::light::voxel_light::{app, Hold, LampWorld, LAMP, STONE};
use crate::test::{record_chunk_messages, recorded, update_until};
use crate::voxel_world_internal::WorldRoot;

/// The lamp in the underside of the slab of [`overhang`].
const SLAB_LAMP: IVec3 = IVec3::new(12, 36, 12);

/// The corners of the piece of the slab a lift takes, holding [`SLAB_LAMP`]:
/// from `PIECE.0` inclusive to `PIECE.1` exclusive.
const PIECE: (IVec3, IVec3) = (IVec3::new(8, 36, 8), IVec3::new(16, 40, 16));

/// Ground below y = 8, and a slab of stone from y = 36 to 40 over the chunks
/// around the origin, holding a lamp in its underside. The slab lies inside
/// the chunks at y = 1, so taking a piece of it remeshes none of the chunks
/// below, whose columns it opens to the sky.
fn overhang(p: IVec3) -> WorldVoxel {
    let slab =
        (36..40).contains(&p.y) && (-24..56).contains(&p.x) && (-24..56).contains(&p.z);
    if p == SLAB_LAMP {
        WorldVoxel::Solid(LAMP)
    } else if p.y < 8 || slab {
        WorldVoxel::Solid(STONE)
    } else {
        WorldVoxel::Air
    }
}

fn overhang_world() -> LampWorld {
    LampWorld {
        scene: Arc::new(overhang),
        light: Some(sky_light(Sky::default(), 30)),
        ..default()
    }
}

/// What a chunk showed in one frame: its tag, its slot's bytes if it has a
/// slot, and its mesh.
#[derive(Clone, PartialEq)]
struct Shown {
    tag: Option<u32>,
    bytes: Option<Vec<u8>>,
    mesh: Option<AssetId<Mesh>>,
}

impl Shown {
    /// The sky and voxel light levels the chunk shows at `local`, on the 0 to
    /// 15 scale.
    fn at(&self, local: IVec3) -> Option<(u8, u8)> {
        let tag = self.tag?;
        let byte = if tag & HAS_SLOT == 0 {
            tag as u8
        } else {
            let cell = local + IVec3::ONE;
            self.bytes.as_ref()?[(cell.z + cell.x * 34 + cell.y * 34 * 34) as usize]
        };
        Some((byte & 0xF, byte >> 4))
    }
}

/// What each chunk showed in each frame, by frame, with the chunk's parent
/// and position, and whether light was settled at the end of the frame.
#[derive(Resource, Default)]
struct ShownLog {
    frames: Vec<(u32, Vec<ShownChunk>, bool)>,
}

/// What a chunk showed in a frame, with its parent and position.
type ShownChunk = (Entity, IVec3, Shown);

/// The chunks [`record_shown`] reads.
type ShownChunks<'w, 's> = Query<
    'w,
    's,
    (
        &'static Chunk<LampWorld>,
        &'static ChildOf,
        Option<&'static MeshTag>,
        Option<&'static MeshRef>,
    ),
>;

fn record_shown(
    mut log: ResMut<ShownLog>,
    frame: Res<FrameCount>,
    chunks: ShownChunks,
    route: Res<LightRoute<LampWorld>>,
    voxel_world: VoxelWorld<LampWorld>,
) {
    let shown = chunks
        .iter()
        .filter(|(chunk, ..)| chunk.key().lod_level == 0)
        .map(|(chunk, child_of, tag, mesh)| {
            let tag = tag.map(|tag| tag.value);
            let bytes = tag
                .filter(|tag| tag & HAS_SLOT != 0)
                .and_then(|tag| route.queued_upload(tag & 0xFF_FFFF))
                .map(<[u8]>::to_vec);
            let shown = Shown {
                tag,
                bytes,
                mesh: mesh.map(|mesh| mesh.0.id()),
            };
            (child_of.parent(), chunk.key().position, shown)
        })
        .collect();
    log.frames
        .push((frame.0, shown, voxel_world.light_is_settled()));
}

/// Each `Stamped` an app read, with the frame it arrived in.
#[derive(Resource, Default)]
struct StampedLog(Vec<u32>);

fn record_stamped(
    mut log: ResMut<StampedLog>,
    mut stamped: MessageReader<Stamped<LampWorld>>,
    frame: Res<FrameCount>,
) {
    log.0.extend(stamped.read().map(|_| frame.0));
}

/// An app of `world`, settled, recording what its chunks show, the frame
/// each chunk publishes in and the frame of each `Stamped`. Its relights
/// run on the main thread, as on WASM, if `main_thread`.
fn recording_app(world: LampWorld, main_thread: bool) -> App {
    let mut app = app(world);
    app.world_mut()
        .resource_mut::<LightSettings<LampWorld>>()
        .main_thread = main_thread;
    record_chunk_messages::<ChunkPublished<LampWorld>>(&mut app);
    app.init_resource::<ShownLog>()
        .init_resource::<StampedLog>()
        .add_systems(PostUpdate, record_shown)
        .add_systems(Update, record_stamped);
    settle(&mut app, "the world to settle");
    app
}

/// Runs `f` on the voxel world in a system of its own, whose commands apply
/// when it returns.
fn run<T: 'static>(
    app: &mut App,
    mut f: impl FnMut(&mut VoxelWorld<LampWorld>) -> T + Send + Sync + 'static,
) -> T {
    app.world_mut()
        .run_system_once(move |mut voxel_world: VoxelWorld<LampWorld>| {
            f(&mut voxel_world)
        })
        .unwrap()
}

fn root(app: &mut App) -> Entity {
    app.world_mut()
        .query_filtered::<Entity, With<WorldRoot<LampWorld>>>()
        .single(app.world())
        .unwrap()
}

/// The frame the next update runs.
fn next_frame(app: &App) -> u32 {
    app.world().resource::<FrameCount>().0
}

/// The frames, from `since` on, in which a chunk of `grid` published.
fn published_since(app: &App, since: u32, grid: Grid) -> Vec<(IVec3, u32)> {
    recorded::<ChunkPublished<LampWorld>>(app)
        .messages
        .iter()
        .filter(|(frame, message)| *frame >= since && message.grid == grid)
        .map(|(frame, message)| (message.chunk_key.position, *frame))
        .collect()
}

/// What the chunk of `parent` at `position` showed in each frame from
/// `since` on.
fn shown_since(
    app: &App,
    since: u32,
    parent: Entity,
    position: IVec3,
) -> Vec<(u32, Shown)> {
    app.world()
        .resource::<ShownLog>()
        .frames
        .iter()
        .filter(|(frame, ..)| *frame >= since)
        .filter_map(|(frame, chunks, _)| {
            chunks
                .iter()
                .find(|(p, at, _)| *p == parent && *at == position)
                .map(|(.., shown)| (*frame, shown.clone()))
        })
        .collect()
}

/// What the chunk of `parent` at `position` shows now.
fn shown_now(app: &App, parent: Entity, position: IVec3) -> Shown {
    let frames = &app.world().resource::<ShownLog>().frames;
    let (_, chunks, _) = frames.last().unwrap();
    chunks
        .iter()
        .find(|(p, at, _)| *p == parent && *at == position)
        .map(|(.., shown)| shown.clone())
        .unwrap_or_else(|| panic!("no chunk at {position}"))
}

/// Whether light was settled at the end of each frame from `since` on.
fn settled_since(app: &App, since: u32) -> Vec<(u32, bool)> {
    app.world()
        .resource::<ShownLog>()
        .frames
        .iter()
        .filter(|(frame, ..)| *frame >= since)
        .map(|(frame, _, settled)| (*frame, *settled))
        .collect()
}

/// Lifts the piece of the slab holding its lamp. Returns the new sub-grid
/// and the frame its commit runs in.
fn lift_piece(app: &mut App) -> (Entity, u32) {
    let frame = next_frame(app);
    let sub_grid = run(app, |voxel_world| {
        voxel_world
            .lift(Grid::World, PIECE.0, PIECE.1, |_, _| true)
            .unwrap()
    });
    (sub_grid, frame)
}

/// A voxel under the piece, in its chunk, and one in the chunk below it,
/// whose columns the lift opens to the sky.
const UNDER_PIECE: IVec3 = IVec3::new(12, 34, 12);
const BELOW: IVec3 = IVec3::new(12, 20, 12);

/// The chunk position holding `p`, and `p`'s place in it.
fn place_of(p: IVec3) -> (IVec3, IVec3) {
    let chunk = p.div_euclid(IVec3::splat(32));
    (chunk, p - chunk * 32)
}

#[test]
fn a_lift_out_of_a_lit_slab_lands_with_the_holes_light_and_the_pieces_lamp_in_one_frame()
{
    let _ = lift_lands_in_one_frame(overhang_world());
}

/// The position of a green lamp in the underside of the slab of [`overhang`],
/// just outside the piece that a lift takes.
const GREEN_SLAB_LAMP: IVec3 = IVec3::new(20, 36, 12);

#[test]
fn a_lift_beside_coloured_lamps_lands_with_the_pieces_red_lamp_in_one_frame() {
    let (mut app, sub_grid) = lift_lands_in_one_frame(LampWorld {
        scene: Arc::new(|p| match p {
            GREEN_SLAB_LAMP => WorldVoxel::Solid(GREEN_LAMP),
            p => overhang(p),
        }),
        emissive: vec![
            (LAMP, Emission::level(15.0).with_color(LinearRgba::RED)),
            (
                GREEN_LAMP,
                Emission::level(15.0).with_color(LinearRgba::GREEN),
            ),
        ],
        ..overhang_world()
    });
    // Read the voxel under the red lamp of the piece. Red light outshines the
    // light of the green lamp there, so the voxel shows red with little green.
    let levels = run(&mut app, move |voxel_world| {
        voxel_world
            .grid(Grid::SubGrid(sub_grid))
            .unwrap()
            .light_at(IVec3::new(5, 0, 5))
    })
    .unwrap();
    let color = levels.voxel_color;
    assert_eq!(levels.voxel, 14.0);
    assert!(
        color.red == 1.0 && color.green < 0.2 && color.blue == 0.0,
        "{color:?}"
    );
}

/// Lifts the piece of the slab of `world` that holds [`SLAB_LAMP`], and
/// returns the app and the new sub-grid.
///
/// The mesh of the new sub-grid is held back for a while. The test fails
/// unless the light of the hole, the new mesh of the chunk the piece left and
/// the lamp of the piece all land in the same frame.
fn lift_lands_in_one_frame(world: LampWorld) -> (App, Entity) {
    let _holding = Hold::exclusive();
    let mut app = recording_app(world, false);
    let root = root(&mut app);
    let (source, under) = place_of(UNDER_PIECE);
    let (below, deep) = place_of(BELOW);
    let before = (shown_now(&app, root, source), shown_now(&app, root, below));
    assert!(
        before.0.at(under).unwrap().0 < 15,
        "the slab shades the source"
    );
    assert!(
        before.1.at(deep).unwrap().0 < 15,
        "the slab shades the chunk below"
    );

    // The new sub-grid's only chunk is at the origin of its grid. Meshing
    // there waits, so the handover can't land until the hold is released.
    let hold = Hold::new(IVec3::ZERO);
    app.world_mut().resource_mut::<LampWorld>().mesh_hold = Some(hold.clone());
    let (sub_grid, commit) = lift_piece(&mut app);
    assert!(!run(&mut app, |voxel_world| voxel_world.light_is_settled()));
    for _ in 0..20 {
        paced_update(&mut app);
    }
    assert!(
        published_since(&app, commit, Grid::World).is_empty(),
        "the handover waits for the sub-grid's mesh"
    );
    hold.release();
    update_until(&mut app, "the handover to land", |app| {
        !published_since(app, commit, Grid::SubGrid(sub_grid)).is_empty()
    });
    app.update();

    let mut published = published_since(&app, commit, Grid::World);
    assert_eq!(
        published
            .iter()
            .map(|(p, _)| p.to_array())
            .collect::<BTreeSet<_>>(),
        BTreeSet::from([source.to_array()]),
    );
    published.extend(published_since(&app, commit, Grid::SubGrid(sub_grid)));
    let frames: BTreeSet<u32> = published.iter().map(|(_, frame)| *frame).collect();
    assert_eq!(frames.len(), 1, "{published:?}");
    let landed = *frames.first().unwrap();

    for (frame, shown) in shown_since(&app, commit, root, source) {
        // Its slot's shell copies its neighbours' light, so the relight
        // ripple from the hole showing early shows here too.
        if frame < landed {
            assert!(shown == before.0, "the source changed in frame {frame}");
        } else if frame == landed {
            assert_ne!(shown.mesh, before.0.mesh, "the source's new mesh");
            assert_eq!(shown.at(under).unwrap().0, 15, "the hole's sky");
        }
    }
    for (frame, shown) in shown_since(&app, commit, root, below) {
        if frame < landed {
            assert!(
                shown == before.1,
                "the chunk below changed in frame {frame}"
            );
        } else if frame == landed {
            assert_eq!(shown.mesh, before.1.mesh, "the chunk below keeps its mesh");
            assert_eq!(shown.at(deep).unwrap().0, 15, "the sky below the hole");
        }
    }
    // The lamp sits at (5, 1, 5) in the sub-grid, over its air.
    let piece = shown_since(&app, landed, sub_grid, IVec3::ZERO);
    let (frame, shown) = &piece[0];
    assert_eq!(*frame, landed);
    assert!(shown.mesh.is_some());
    assert_eq!(
        shown.at(IVec3::new(5, 0, 5)).unwrap().1,
        14,
        "the lamp's light"
    );
    for (frame, settled) in settled_since(&app, commit) {
        if frame < landed {
            assert!(!settled, "light settled in frame {frame}");
        }
    }
    (app, sub_grid)
}

/// Where the stamped lamp lands: the middle of the world's chunk at (1, 1, 1),
/// in the open air over the slab, so its light stays inside that chunk.
const STAMPED_LAMP: IVec3 = IVec3::splat(48);

/// Stones placed in the sky-lit air over the slab, in the chunks nearest the
/// camera, beside the stamp.
const BESIDE: [IVec3; 4] = [
    IVec3::new(5, 45, 5),
    IVec3::new(-5, 45, 5),
    IVec3::new(5, 45, -5),
    IVec3::new(-5, 45, -5),
];

#[test]
fn a_stamped_lamps_pool_shows_in_the_frame_its_chunk_publishes_and_stamped_arrives() {
    // On the main thread, at the shortest budget, relights start about one
    // a frame, nearest the camera first, so the stamped lamp's chunk relights
    // after the other chunks of the commit and well after its chunk task.
    let mut app = recording_app(
        LampWorld {
            light: Some(LightPropagation {
                wasm_relight_budget: std::time::Duration::from_micros(100),
                ..sky_light(Sky::default(), 30)
            }),
            ..overhang_world()
        },
        true,
    );
    let root = root(&mut app);
    let sub_grid = run(&mut app, |voxel_world| {
        let entity = voxel_world
            .spawn_sub_grid(Transform::from_translation(STAMPED_LAMP.as_vec3()));
        let mut grid = voxel_world.grid_mut(Grid::SubGrid(entity)).unwrap();
        grid.set_voxel(IVec3::ZERO, WorldVoxel::Solid(LAMP));
        entity
    });
    settle(&mut app, "the sub-grid to settle");
    let (target, lamp) = place_of(STAMPED_LAMP);
    let above = lamp + IVec3::Y;
    let before = shown_now(&app, root, target);
    assert_eq!(before.at(above).unwrap().1, 0, "no lamp in the world yet");

    let commit = next_frame(&app);
    run(&mut app, move |voxel_world| {
        voxel_world
            .stamp(sub_grid, Grid::World, incoming_replaces)
            .unwrap();
        for position in BESIDE {
            voxel_world.set_voxel(position, WorldVoxel::Solid(STONE));
        }
    });
    update_until(&mut app, "the stamp to land", |app| {
        !app.world().resource::<StampedLog>().0.is_empty()
    });
    app.update();

    let stamped = app.world().resource::<StampedLog>().0.clone();
    assert_eq!(stamped.len(), 1);
    let landed = stamped[0];
    let published = published_since(&app, commit, Grid::World);
    assert!(published.contains(&(target, landed)), "{published:?}");
    assert!(
        published.iter().all(|(_, frame)| *frame == landed),
        "{published:?}"
    );
    for (frame, shown) in shown_since(&app, commit, root, target) {
        if frame < landed {
            assert!(shown == before, "the target changed in frame {frame}");
        } else if frame == landed {
            assert_eq!(shown.at(above).unwrap().1, 14, "the lamp's pool");
        }
    }
}

#[test]
fn a_lit_lift_lands_while_its_source_chunk_is_edited_every_frame() {
    let mut app = recording_app(overhang_world(), false);
    let (sub_grid, commit) = lift_piece(&mut app);
    let mut solid = false;
    update_until(&mut app, "the handover to land", |app| {
        solid = !solid;
        let voxel = if solid {
            WorldVoxel::Solid(STONE)
        } else {
            WorldVoxel::Air
        };
        run(app, move |voxel_world| {
            voxel_world.set_voxel(IVec3::new(24, 50, 24), voxel)
        });
        !published_since(app, commit, Grid::SubGrid(sub_grid)).is_empty()
    });
    let (source, _) = place_of(UNDER_PIECE);
    let landed = published_since(&app, commit, Grid::SubGrid(sub_grid))[0].1;
    assert!(published_since(&app, commit, Grid::World).contains(&(source, landed)));
    settle(&mut app, "light to settle once the edits stop");
}

#[test]
fn a_sub_grid_despawned_mid_handover_in_a_lit_world_lets_the_hole_land_with_its_light() {
    let _holding = Hold::exclusive();
    let mut app = recording_app(overhang_world(), false);
    let root = root(&mut app);
    let (source, under) = place_of(UNDER_PIECE);
    // Meshing the sub-grid's only chunk waits until the end of the test.
    let hold = Hold::new(IVec3::ZERO);
    app.world_mut().resource_mut::<LampWorld>().mesh_hold = Some(hold.clone());
    let (sub_grid, commit) = lift_piece(&mut app);
    for _ in 0..3 {
        paced_update(&mut app);
    }
    assert!(published_since(&app, commit, Grid::World).is_empty());
    app.world_mut().entity_mut(sub_grid).despawn();

    update_until(&mut app, "the world's chunks to land", |app| {
        !published_since(app, commit, Grid::World).is_empty()
    });
    app.update();
    let landed = published_since(&app, commit, Grid::World)[0].1;
    let shown = shown_since(&app, landed, root, source);
    assert_eq!(shown[0].1.at(under).unwrap().0, 15, "the hole's sky");
    hold.release();
    settle(&mut app, "light to settle");
}

#[test]
fn a_lit_handover_whose_chunks_all_despawn_before_its_relights_solve_still_lands() {
    // On the main thread, at the shortest budget, the lift's relights solve
    // about one a frame, so they are still running when its chunks go.
    let mut app = recording_app(
        LampWorld {
            light: Some(LightPropagation {
                wasm_relight_budget: std::time::Duration::from_micros(100),
                ..sky_light(Sky::default(), 30)
            }),
            ..overhang_world()
        },
        true,
    );
    let root = root(&mut app);
    let (source, under) = place_of(UNDER_PIECE);
    let (below, deep) = place_of(BELOW);
    let (sub_grid, _) = lift_piece(&mut app);
    app.update();
    // The handover holds the source chunk and the sub-grid's chunk, and
    // waits for the relights around the hole, the chunk below included.
    let source_chunk = app
        .world_mut()
        .query::<(Entity, &Chunk<LampWorld>, &ChildOf)>()
        .iter(app.world())
        .find(|(_, chunk, child_of)| {
            chunk.key() == ChunkKey::new(source, 0) && child_of.parent() == root
        })
        .map(|(entity, ..)| entity)
        .unwrap();
    assert!(!run(&mut app, |voxel_world| voxel_world.light_is_settled()));
    app.world_mut().entity_mut(sub_grid).despawn();
    app.world_mut().entity_mut(source_chunk).despawn();

    settle(&mut app, "light to settle once the handover lands");
    assert_eq!(shown_now(&app, root, below).at(deep).unwrap().0, 15);
    assert_eq!(shown_now(&app, root, source).at(under).unwrap().0, 15);
}
