//! Tests of voxel light from emissive voxels at LOD level 0, read through
//! `light_at`. Sky light is off in all of them.

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use bevy::ecs::system::RunSystemOnce;
use bevy::prelude::*;

use super::super::{chunk_status, meshing_app, spawn_camera, update_until};
use super::reference::Reference;
use crate::chunk_map::ChunkMap;
use crate::light::LightSettings;
use crate::prelude::*;

pub(super) const STONE: u8 = 1;
pub(super) const LAMP: u8 = 2;
pub(super) const DIM_LAMP: u8 = 3;

pub(super) type Scene = Arc<dyn Fn(IVec3) -> WorldVoxel + Send + Sync>;

/// A world's fill hint for each level-0 chunk position.
pub(super) type Hints = Arc<dyn Fn(IVec3) -> FillHint + Send + Sync>;

/// Holds back the chunk task of one chunk position: its lookups of voxels
/// outside the chunk, in its padding, wait until the hold is released.
///
/// A held task blocks a thread of the async compute pool, which every test in
/// the process shares and which may have only two threads, so tests take
/// [`Hold::exclusive`] to hold one chunk at a time between them.
pub(super) struct Hold {
    pub chunk: IVec3,
    pub released: AtomicBool,
    /// Whether the lookups held are those inside the chunk position, made by
    /// sky probes and by walks of the chunks below, instead of its chunk
    /// task's padding.
    pub inside: bool,
    /// How many lookups have waited on the hold.
    pub waited: AtomicU32,
}

static HOLDING: Mutex<()> = Mutex::new(());

impl Hold {
    pub fn new(chunk: IVec3) -> Arc<Self> {
        Arc::new(Self {
            chunk,
            released: AtomicBool::new(false),
            inside: false,
            waited: AtomicU32::new(0),
        })
    }

    /// Holds back every lookup of a voxel inside the chunk position `chunk`
    /// made for that position: its sky probes and the walks through it of the
    /// chunks below. The position must not spawn a chunk.
    pub fn inside(chunk: IVec3) -> Arc<Self> {
        Arc::new(Self {
            inside: true,
            ..Arc::into_inner(Self::new(chunk)).unwrap()
        })
    }

    /// Waits until no other test holds a chunk, and keeps them from holding
    /// one until the guard drops.
    pub fn exclusive() -> MutexGuard<'static, ()> {
        HOLDING
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    pub fn release(&self) {
        self.released.store(true, Ordering::Release);
    }

    fn wait(&self) {
        self.waited.fetch_add(1, Ordering::AcqRel);
        let deadline = Instant::now() + Duration::from_secs(20);
        while !self.released.load(Ordering::Acquire) && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(1));
        }
    }
}

/// A lit world whose voxels come from a scene, with lamps at level 15 and dim
/// lamps at level 8.
#[derive(Resource, Clone)]
pub(super) struct LampWorld {
    pub scene: Scene,
    pub light: Option<LightPropagation>,
    pub emissive: Vec<(u8, Emission)>,
    pub hold: Option<Arc<Hold>>,
    pub spawning_distance: u32,
    pub hints: Option<Hints>,
    /// Holds back meshing the chunk at its position until it is released.
    pub mesh_hold: Option<Arc<Hold>>,
}

impl Default for LampWorld {
    fn default() -> Self {
        Self {
            scene: Arc::new(|_| WorldVoxel::Air),
            light: Some(lit(LightShape::Round)),
            emissive: vec![
                (LAMP, Emission::level(15.0)),
                (DIM_LAMP, Emission::level(8.0)),
            ],
            hold: None,
            spawning_distance: 2,
            hints: None,
            mesh_hold: None,
        }
    }
}

impl VoxelWorldConfig for LampWorld {
    type MaterialIndex = u8;
    type ChunkUserBundle = ();

    fn spawning_distance(&self) -> u32 {
        self.spawning_distance
    }

    fn chunk_spawn_strategy(&self) -> ChunkSpawnStrategy {
        ChunkSpawnStrategy::Close
    }

    fn chunk_despawn_strategy(&self) -> ChunkDespawnStrategy {
        ChunkDespawnStrategy::FarAway
    }

    fn chunk_fill_hint(&self, key: ChunkKey) -> FillHint {
        match (&self.hints, key.lod_level) {
            (Some(hints), 0) => hints(key.position),
            _ => FillHint::Unknown,
        }
    }

    fn voxel_lookup_delegate(&self) -> VoxelLookupDelegate<Self::MaterialIndex> {
        let scene = self.scene.clone();
        let hold = self.hold.clone();
        Box::new(move |chunk, _, _| {
            let scene = scene.clone();
            let hold = hold.clone().filter(|hold| hold.chunk == chunk);
            Box::new(move |position, _| {
                // A chunk task generates its padding first, while a walk of a
                // chunk below reads only voxels inside the chunks above it, so
                // holding back the padding holds back the chunk's task alone.
                let inside = position.div_euclid(IVec3::splat(32)) == chunk;
                if let Some(hold) = hold.as_ref().filter(|hold| hold.inside == inside) {
                    hold.wait();
                }
                scene(position)
            })
        })
    }

    fn chunk_meshing_delegate(
        &self,
    ) -> ChunkMeshingDelegate<Self::MaterialIndex, Self::ChunkUserBundle> {
        let hold = self.mesh_hold.clone()?;
        Some(Box::new(
            move |position, lod, data_shape, mesh_shape, previous| {
                let mut mesh = crate::configuration::default_chunk_meshing_delegate(
                    position, lod, data_shape, mesh_shape, previous,
                );
                let hold = Some(hold.clone()).filter(|hold| hold.chunk == position);
                Box::new(move |voxels, data_shape, mesh_shape, mapper| {
                    if let Some(hold) = &hold {
                        hold.wait();
                    }
                    mesh(voxels, data_shape, mesh_shape, mapper)
                })
            },
        ))
    }

    fn light_propagation(&self) -> Option<LightPropagation> {
        self.light.clone()
    }

    fn emissive_materials(&self) -> Vec<(Self::MaterialIndex, Emission)> {
        self.emissive.clone()
    }
}

/// Voxel light only, at the default voxel reach of 14.
pub(super) fn lit(shape: LightShape) -> LightPropagation {
    LightPropagation {
        sky: Sky::Off,
        sky_fill: None,
        shape,
        ..default()
    }
}

/// The middle voxel of the chunk at the origin, where the camera is.
const MIDDLE: IVec3 = IVec3::splat(16);

pub(super) fn app(world: LampWorld) -> App {
    let mut app = meshing_app();
    app.add_plugins(VoxelWorldPlugin::with_config(world))
        .add_systems(Startup, spawn_camera::<LampWorld>);
    app
}

/// A world of air, apart from `lamps`.
fn lamps_in_air(lamps: Vec<(IVec3, u8)>, shape: LightShape) -> LampWorld {
    LampWorld {
        scene: Arc::new(move |position| {
            match lamps.iter().find(|(lamp, _)| *lamp == position) {
                Some((_, material)) => WorldVoxel::Solid(*material),
                None => WorldVoxel::Air,
            }
        }),
        light: Some(lit(shape)),
        ..default()
    }
}

/// An app of `world`, run until the chunk at the origin is ready.
fn origin_ready(world: LampWorld) -> App {
    let mut app = app(world);
    update_until(&mut app, "the origin chunk to be ready", |app| {
        chunk_status::<LampWorld>(app, IVec3::ZERO) == Some(ChunkStatus::Ready)
    });
    app
}

pub(super) fn light_at(app: &mut App, position: IVec3) -> Option<LightLevels> {
    app.world_mut()
        .run_system_once(move |voxel_world: VoxelWorld<LampWorld>| {
            voxel_world.light_at(position)
        })
        .unwrap()
}

fn voxel_light(app: &mut App, position: IVec3) -> f32 {
    light_at(app, position)
        .unwrap_or_else(|| panic!("no light at {position}"))
        .voxel
}

/// How many voxels from `from` along `step` voxel light reaches.
fn reach_along(app: &mut App, from: IVec3, step: IVec3) -> i32 {
    (1..32)
        .take_while(|&k| voxel_light(app, from + step * k) > 0.0)
        .count() as i32
}

/// Every voxel of the chunk at the origin.
pub(super) fn origin_voxels() -> impl Iterator<Item = IVec3> {
    (0..32).flat_map(|z| {
        (0..32).flat_map(move |y| (0..32).map(move |x| IVec3::new(x, y, z)))
    })
}

#[test]
fn a_lamp_reads_its_declared_level_and_its_neighbours_one_step_less() {
    let dim = MIDDLE + IVec3::new(0, 0, 8);
    let mut app = origin_ready(lamps_in_air(
        vec![(MIDDLE, LAMP), (dim, DIM_LAMP)],
        LightShape::Round,
    ));

    assert_eq!(voxel_light(&mut app, MIDDLE), 15.0);
    assert_eq!(voxel_light(&mut app, dim), 8.0);
    for step in [IVec3::X, IVec3::NEG_X, IVec3::Y, IVec3::NEG_Y, IVec3::NEG_Z] {
        assert_eq!(
            voxel_light(&mut app, MIDDLE + step),
            14.0,
            "beside the lamp"
        );
    }
    for step in [IVec3::X, IVec3::NEG_X, IVec3::Y, IVec3::NEG_Y, IVec3::Z] {
        assert_eq!(
            voxel_light(&mut app, dim + step),
            7.0,
            "beside the dim lamp"
        );
    }
    assert_eq!(light_at(&mut app, MIDDLE + IVec3::X).unwrap().sky, 0.0);
}

#[test]
fn a_lamp_at_8_reaches_about_half_as_far_as_one_at_15() {
    // The dim lamp sits in the middle of the next chunk along x.
    let dim = MIDDLE + IVec3::new(32, 0, 0);
    let mut app = app(lamps_in_air(
        vec![(MIDDLE, LAMP), (dim, DIM_LAMP)],
        LightShape::Round,
    ));
    update_until(&mut app, "both lamps' chunks to be ready", |app| {
        [IVec3::ZERO, IVec3::X].iter().all(|chunk| {
            chunk_status::<LampWorld>(app, *chunk) == Some(ChunkStatus::Ready)
        })
    });

    let bright = reach_along(&mut app, MIDDLE, IVec3::Y);
    let dim = reach_along(&mut app, dim, IVec3::Y);
    assert_eq!(bright, 14);
    let ratio = dim as f32 / bright as f32;
    assert!((0.4..=0.6).contains(&ratio), "{dim} against {bright}");
}

#[test]
fn light_doesnt_pass_between_two_solids() {
    // A room split by a wall one voxel thick running diagonally across it, so
    // its two halves touch only where two solids meet at an edge.
    let room =
        |p: IVec3| p.cmpge(IVec3::splat(4)).all() && p.cmplt(IVec3::splat(20)).all();
    let lamp = IVec3::new(17, 6, 12);
    let mut app = origin_ready(LampWorld {
        scene: Arc::new(move |p| {
            if p == lamp {
                WorldVoxel::Solid(LAMP)
            } else if room(p) && p.x != p.y {
                WorldVoxel::Air
            } else {
                WorldVoxel::Solid(STONE)
            }
        }),
        ..default()
    });

    assert!(voxel_light(&mut app, lamp + IVec3::Y) > 0.0);
    for p in origin_voxels().filter(|p| room(*p) && p.x < p.y) {
        assert_eq!(voxel_light(&mut app, p), 0.0, "at {p}");
    }
}

#[test]
fn a_round_pool_reaches_reach_along_each_axis_and_is_nearly_circular() {
    let mut app = origin_ready(lamps_in_air(vec![(MIDDLE, LAMP)], LightShape::Round));

    for step in [
        IVec3::X,
        IVec3::NEG_X,
        IVec3::Y,
        IVec3::NEG_Y,
        IVec3::Z,
        IVec3::NEG_Z,
    ] {
        assert_eq!(reach_along(&mut app, MIDDLE, step), 14, "along {step}");
    }
    for step in [
        IVec3::new(1, 0, 1),
        IVec3::new(1, 1, 0),
        IVec3::new(-1, 0, 1),
    ] {
        let diagonal =
            reach_along(&mut app, MIDDLE, step) as f32 * step.as_vec3().length();
        let ratio = diagonal / 14.0;
        assert!(
            (0.85..=1.15).contains(&ratio),
            "along {step} the pool reaches {diagonal} voxels against 14 along an axis"
        );
    }
}

#[test]
fn a_diamond_pool_is_one_face_step_per_round() {
    let mut app = origin_ready(lamps_in_air(vec![(MIDDLE, LAMP)], LightShape::Diamond));

    for z in -15..=15 {
        for y in -15..=15 {
            for x in -15..=15 {
                let offset = IVec3::new(x, y, z);
                let steps = x.abs() + y.abs() + z.abs();
                let want = if steps <= 14 { 15 - steps } else { 0 };
                assert_eq!(
                    voxel_light(&mut app, MIDDLE + offset),
                    want as f32,
                    "{offset} from the lamp"
                );
            }
        }
    }
}

#[test]
fn the_brightest_of_two_lamps_wins() {
    let bright = IVec3::new(6, 16, 16);
    let dim = IVec3::new(16, 16, 16);
    let mut app = origin_ready(lamps_in_air(
        vec![(bright, LAMP), (dim, DIM_LAMP)],
        LightShape::Diamond,
    ));

    // 6 steps from the bright lamp, and 4 past the dim lamp's start of 7.
    assert_eq!(voxel_light(&mut app, IVec3::new(12, 16, 16)), 9.0);
    // Beside the dim lamp, 13 steps around it from the bright one.
    assert_eq!(voxel_light(&mut app, IVec3::new(17, 16, 16)), 7.0);
}

#[test]
fn a_cave_with_no_lamp_reads_0_in_voxel_light() {
    let cave =
        |p: IVec3| p.cmpge(IVec3::splat(4)).all() && p.cmplt(IVec3::splat(12)).all();
    let room =
        |p: IVec3| p.cmpge(IVec3::splat(14)).all() && p.cmplt(IVec3::splat(28)).all();
    let lamp = IVec3::new(14, 14, 14);
    let mut app = origin_ready(LampWorld {
        scene: Arc::new(move |p| {
            if p == lamp {
                WorldVoxel::Solid(LAMP)
            } else if cave(p) || room(p) {
                WorldVoxel::Air
            } else {
                WorldVoxel::Solid(STONE)
            }
        }),
        ..default()
    });

    assert!(voxel_light(&mut app, lamp + IVec3::X) > 0.0);
    for p in origin_voxels().filter(|p| cave(*p)) {
        assert_eq!(voxel_light(&mut app, p), 0.0, "at {p}");
    }
}

#[test]
fn light_at_reads_0_in_a_solid_and_none_with_light_off() {
    let stone = MIDDLE + IVec3::X;
    let scene: Scene = Arc::new(move |p| match p {
        p if p == MIDDLE => WorldVoxel::Solid(LAMP),
        p if p == stone => WorldVoxel::Solid(STONE),
        _ => WorldVoxel::Air,
    });
    let mut app = origin_ready(LampWorld {
        scene: scene.clone(),
        ..default()
    });
    assert_eq!(light_at(&mut app, stone), Some(LightLevels::new(0, 0)));
    assert!(light_at(&mut app, IVec3::new(100_000, 0, 0)).is_none());

    let mut unlit = origin_ready(LampWorld {
        scene,
        light: None,
        ..default()
    });
    assert_eq!(light_at(&mut unlit, MIDDLE), None);
    assert_eq!(light_at(&mut unlit, stone), None);
}

pub(super) fn emissive_scans(app: &App) -> u64 {
    app.world()
        .resource::<LightSettings<LampWorld>>()
        .counters()
        .emissive_scans
        .load(Ordering::Relaxed)
}

#[test]
fn an_empty_emissive_list_scans_nothing() {
    let world = lamps_in_air(vec![(MIDDLE, LAMP)], LightShape::Round);
    let mut app = origin_ready(LampWorld {
        emissive: Vec::new(),
        ..world.clone()
    });
    assert_eq!(emissive_scans(&app), 0);
    assert_eq!(voxel_light(&mut app, MIDDLE + IVec3::X), 0.0);

    let listed = origin_ready(world);
    assert!(emissive_scans(&listed) > 0);
}

fn hash3(p: IVec3) -> u32 {
    let mut h = (p.x as u32).wrapping_mul(0x8da6_b343)
        ^ (p.y as u32).wrapping_mul(0xd816_3841)
        ^ (p.z as u32).wrapping_mul(0xcb1a_b31f);
    h ^= h >> 13;
    h = h.wrapping_mul(0x5bd1_e995);
    h ^ (h >> 15)
}

/// Blocks of stone 4 voxels wide among open space, with a few lamps and dim
/// lamps in the stone.
pub(super) fn caves(p: IVec3) -> WorldVoxel {
    if !hash3(p.div_euclid(IVec3::splat(4))).is_multiple_of(3) {
        return WorldVoxel::Air;
    }
    match hash3(p) % 1500 {
        0 => WorldVoxel::Solid(LAMP),
        1 => WorldVoxel::Solid(DIM_LAMP),
        _ => WorldVoxel::Solid(STONE),
    }
}

fn reference(min: IVec3, max: IVec3) -> Reference {
    caves_reference(min, max, 14)
}

/// Voxel light in the box from `min` to `max` of the world of [`caves`] at
/// `reach`, with round spread.
pub(super) fn caves_reference(min: IVec3, max: IVec3, reach: u32) -> Reference {
    let emission = LampWorld::default().emissive;
    Reference::voxel_light(min, max, reach, LightShape::Round, caves, |material| {
        emission
            .iter()
            .find(|(emissive, _)| *emissive == material)
            .map(|(_, emission)| emission.whole_level() as f32)
    })
}

/// The voxel light of every voxel of the level-0 chunk at `chunk`, or `None`
/// if it has none published.
pub(super) fn chunk_voxel_light(
    app: &mut App,
    chunk: IVec3,
) -> Option<Vec<(IVec3, f32)>> {
    app.world_mut()
        .run_system_once(
            move |settings: Res<LightSettings<LampWorld>>,
                  chunk_map: Res<ChunkMap<LampWorld, u8>>| {
                let map = chunk_map.get_read_lock();
                // What `light_at` reads, looked up once for the whole chunk.
                let data = map.get(&ChunkKey::new(chunk, 0))?;
                let light = settings.view.read().get(&chunk)?.light.clone();
                origin_voxels()
                    .map(|local| {
                        let p = chunk * 32 + local;
                        let voxel = data.get_voxel_at_world_position(p)?;
                        let levels = light.levels_at(
                            local.as_uvec3(),
                            voxel,
                            &settings.task.emission,
                        );
                        Some((p, levels.voxel))
                    })
                    .collect::<Option<Vec<_>>>()
            },
        )
        .unwrap()
}

#[derive(Resource, Default)]
struct OriginPublishes(usize);

fn count_origin_publishes(
    mut published: MessageReader<ChunkPublished<LampWorld>>,
    mut count: ResMut<OriginPublishes>,
) {
    count.0 += published
        .read()
        .filter(|message| message.chunk_key == ChunkKey::new(IVec3::ZERO, 0))
        .count();
}

#[test]
fn a_chunk_whose_task_starts_after_its_whole_box_has_published_matches_the_reference() {
    let _holding = Hold::exclusive();
    let hold = Hold::new(IVec3::ZERO);
    let mut app = app(LampWorld {
        scene: Arc::new(caves),
        hold: Some(hold.clone()),
        ..default()
    });
    app.init_resource::<OriginPublishes>()
        .add_systems(Update, count_origin_publishes);

    let neighbours: Vec<IVec3> = (-1..=1)
        .flat_map(|z| {
            (-1..=1).flat_map(move |y| (-1..=1).map(move |x| IVec3::new(x, y, z)))
        })
        .filter(|offset| *offset != IVec3::ZERO)
        .collect();
    update_until(&mut app, "the held chunk's neighbours to be ready", |app| {
        neighbours.iter().all(|chunk| {
            chunk_status::<LampWorld>(app, *chunk) == Some(ChunkStatus::Ready)
        })
    });
    // An edit while the held task runs starts a new task once it lands, after
    // every neighbour has published.
    let edited = IVec3::new(5, 5, 5);
    app.world_mut()
        .run_system_once(move |mut voxel_world: VoxelWorld<LampWorld>| {
            voxel_world.set_voxel(edited, caves(edited))
        })
        .unwrap();
    app.update();
    hold.release();
    update_until(&mut app, "the follow-up task to publish", |app| {
        app.world().resource::<OriginPublishes>().0 >= 2
    });

    let reference = reference(IVec3::splat(-32), IVec3::splat(64));
    let light = chunk_voxel_light(&mut app, IVec3::ZERO).expect("the chunk's light");
    let lit = light.iter().filter(|(_, level)| *level > 0.0).count();
    assert!(lit > 1000, "only {lit} voxels are lit");
    for (p, level) in light {
        assert_eq!(level, reference.voxel_at(p) as f32, "at {p}");
    }
}

#[test]
fn no_reading_is_brighter_than_the_reference() {
    let mut app = app(LampWorld {
        scene: Arc::new(caves),
        ..default()
    });
    let checked: Vec<IVec3> = (-1..=1)
        .flat_map(|z| {
            (-1..=1).flat_map(move |y| (-1..=1).map(move |x| IVec3::new(x, y, z)))
        })
        .collect();
    // Far enough past the checked chunks that light from outside it can't
    // reach them.
    let reference = reference(IVec3::splat(-32 - 14), IVec3::splat(64 + 14));

    let mut ready_frames = 0;
    let deadline = Instant::now() + Duration::from_secs(60);
    while ready_frames < 5 {
        assert!(Instant::now() < deadline, "timed out streaming");
        app.update();
        for chunk in &checked {
            let Some(light) = chunk_voxel_light(&mut app, *chunk) else {
                continue;
            };
            for (p, level) in light {
                let truth = reference.voxel_at(p) as f32;
                assert!(level <= truth, "{level} at {p} is brighter than {truth}");
            }
        }
        let all_ready = checked.iter().all(|chunk| {
            chunk_status::<LampWorld>(&mut app, *chunk) == Some(ChunkStatus::Ready)
        });
        ready_frames = if all_ready { ready_frames + 1 } else { 0 };
    }
    for chunk in &checked {
        assert!(
            chunk_voxel_light(&mut app, *chunk).is_some(),
            "{chunk} has no light"
        );
    }
}
