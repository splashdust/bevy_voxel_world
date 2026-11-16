//! Tests that the level-0 reader reads what the screen shows: the voxel edit,
//! else the published level-0 chunk, else its fill hint, else the world's
//! voxel lookup run on the caller's thread.

use std::sync::atomic::{AtomicU32, AtomicU8, Ordering};
use std::sync::Arc;

use bevy::ecs::system::RunSystemOnce;
use bevy::prelude::*;

use super::{chunk_status, chunk_status_of, spawn_camera, test_plugins, update_until};
use crate::chunk::{CHUNK_SIZE_F, CHUNK_SIZE_I};
use crate::prelude::*;

/// A chunk position far from the camera that the world hints is all of
/// [`HINTED_VOXEL`], though its lookup gives air there.
const HINTED: IVec3 = IVec3::new(40, 40, 40);
const HINTED_VOXEL: WorldVoxel = WorldVoxel::Solid(9);

/// A voxel below ground far beyond the spawning distance.
const FAR: IVec3 = IVec3::new(3000, -5, 3000);

/// A voxel below ground in the chunk below the camera.
const LOADED: IVec3 = IVec3::new(3, -5, 3);

/// Ground below y = 0 of material `material`, air above it, and hinted
/// [`HINTED_VOXEL`] at [`HINTED`]. It counts the fill hints it is asked for
/// and the lookups it makes. With `stripes`, the ground from y = -40 up is
/// solid only where x is odd, and chunks beyond the one under the camera
/// stream at level 1.
#[derive(Resource, Clone)]
struct ReaderWorld {
    material: Arc<AtomicU8>,
    hints: Arc<AtomicU32>,
    lookups: Arc<AtomicU32>,
    stripes: bool,
}

impl Default for ReaderWorld {
    fn default() -> Self {
        Self {
            material: Arc::new(AtomicU8::new(1)),
            hints: Arc::default(),
            lookups: Arc::default(),
            stripes: false,
        }
    }
}

impl VoxelWorldConfig for ReaderWorld {
    type MaterialIndex = u8;
    type ChunkUserBundle = ();

    fn spawning_distance(&self) -> u32 {
        3
    }

    fn chunk_spawn_strategy(&self) -> ChunkSpawnStrategy {
        ChunkSpawnStrategy::Close
    }

    fn chunk_despawn_strategy(&self) -> ChunkDespawnStrategy {
        ChunkDespawnStrategy::FarAway
    }

    fn voxel_lookup_delegate(&self) -> VoxelLookupDelegate<Self::MaterialIndex> {
        let (material, lookups, stripes) =
            (self.material.clone(), self.lookups.clone(), self.stripes);
        Box::new(move |_, _, _| {
            lookups.fetch_add(1, Ordering::Relaxed);
            let material = material.clone();
            Box::new(move |position, _| {
                let striped =
                    !stripes || position.y < -40 || position.x.rem_euclid(2) == 1;
                let ground = position.y < 0 && striped;
                if ground {
                    WorldVoxel::Solid(material.load(Ordering::Relaxed))
                } else {
                    WorldVoxel::Air
                }
            })
        })
    }

    fn chunk_fill_hint(&self, key: ChunkKey) -> FillHint<Self::MaterialIndex> {
        self.hints.fetch_add(1, Ordering::Relaxed);
        if key == ChunkKey::new(HINTED, 0) {
            FillHint::Uniform(HINTED_VOXEL)
        } else {
            FillHint::Unknown
        }
    }

    fn chunk_lod(
        &self,
        chunk_position: IVec3,
        _previous_lod: Option<LodLevel>,
        camera_position: Vec3,
    ) -> LodLevel {
        let camera_chunk = (camera_position / CHUNK_SIZE_F).floor().as_ivec3();
        if !self.stripes || chunk_position.distance_squared(camera_chunk) < 4 {
            0
        } else {
            1
        }
    }
}

fn reader_app(world: ReaderWorld) -> App {
    let mut app = App::new();
    app.add_plugins((test_plugins(), VoxelWorldPlugin::<ReaderWorld>::minimal()))
        .insert_resource(world)
        .add_systems(Startup, spawn_camera::<ReaderWorld>);
    app
}

/// A `ReaderWorld` app run until the chunk holding [`LOADED`] is ready.
fn loaded_app() -> App {
    let mut app = reader_app(ReaderWorld::default());
    update_until(&mut app, "the chunk below the camera to be ready", |app| {
        chunk_status::<ReaderWorld>(app, IVec3::NEG_Y) == Some(ChunkStatus::Ready)
    });
    app
}

/// What one read session of the level-0 reader gives at each of `positions`.
fn read(app: &mut App, positions: &[IVec3]) -> Vec<WorldVoxel> {
    let positions = positions.to_vec();
    app.world_mut()
        .run_system_once(move |voxel_world: VoxelWorld<ReaderWorld>| {
            let mut reader = voxel_world.level_0_reader();
            positions
                .iter()
                .map(|p| reader.read(*p))
                .collect::<Vec<_>>()
        })
        .unwrap()
}

fn config(app: &App) -> &ReaderWorld {
    app.world().resource::<ReaderWorld>()
}

#[test]
fn an_edit_wins_over_the_chunk_and_the_generator() {
    let mut app = loaded_app();
    app.world_mut()
        .run_system_once(|mut voxel_world: VoxelWorld<ReaderWorld>| {
            voxel_world.set_voxel(LOADED, WorldVoxel::Solid(7));
            voxel_world.set_voxel(FAR, WorldVoxel::Air);
            voxel_world.set_voxel(HINTED * CHUNK_SIZE_I, WorldVoxel::Solid(8));
        })
        .unwrap();

    assert_eq!(
        read(&mut app, &[LOADED, FAR, HINTED * CHUNK_SIZE_I]),
        [WorldVoxel::Solid(7), WorldVoxel::Air, WorldVoxel::Solid(8)]
    );
}

#[test]
fn in_a_loaded_area_it_reads_the_published_level_0_chunk() {
    let mut app = loaded_app();
    config(&app).material.store(2, Ordering::Relaxed);

    // The chunk was generated with material 1, and nothing regenerated it.
    assert_eq!(
        read(&mut app, &[LOADED, FAR]),
        [WorldVoxel::Solid(1), WorldVoxel::Solid(2)]
    );
}

#[test]
fn a_chunk_position_hinted_uniform_reads_the_hinted_voxel() {
    let mut app = reader_app(ReaderWorld::default());
    app.update();

    assert_eq!(
        read(&mut app, &[HINTED * CHUNK_SIZE_I + IVec3::new(5, 17, 30)]),
        [HINTED_VOXEL]
    );
}

#[test]
fn a_position_far_beyond_the_spawning_distance_reads_the_generator() {
    let mut app = loaded_app();

    let (read, got) = app
        .world_mut()
        .run_system_once(|voxel_world: VoxelWorld<ReaderWorld>| {
            let mut reader = voxel_world.level_0_reader();
            (reader.read(FAR), voxel_world.get_voxel(FAR))
        })
        .unwrap();
    assert_eq!(read, WorldVoxel::Solid(1));
    assert_eq!(got, WorldVoxel::Unset, "no chunk covers the voxel");
}

#[test]
fn a_position_only_a_coarse_chunk_covers_reads_the_generators_level_0_voxel() {
    let mut app = reader_app(ReaderWorld {
        stripes: true,
        ..default()
    });
    // A level-1 chunk two chunks from the camera, below ground.
    let coarse = ChunkKey::new(IVec3::new(2, -2, 0), 1);
    update_until(&mut app, "the coarse chunk to be ready", |app| {
        chunk_status_of::<ReaderWorld>(app, coarse) == Some(ChunkStatus::Ready)
    });
    // Odd x in the stripes, where level 0 is solid, but the coarse chunk's
    // sample at the even x before it is air.
    let striped = coarse.min_voxel() + IVec3::new(5, 44, 6);

    let (read, got) = app
        .world_mut()
        .run_system_once(move |voxel_world: VoxelWorld<ReaderWorld>| {
            let mut reader = voxel_world.level_0_reader();
            (reader.read(striped), voxel_world.get_voxel(striped))
        })
        .unwrap();
    assert_eq!(read, WorldVoxel::Solid(1));
    assert_eq!(got, WorldVoxel::Air, "get_voxel reads the coarse sample");
}

#[test]
fn the_hint_and_the_lookup_run_once_per_chunk_in_a_read_session() {
    // No camera, so no chunk task asks for hints or lookups meanwhile.
    let mut app = App::new();
    app.add_plugins((test_plugins(), VoxelWorldPlugin::<ReaderWorld>::minimal()));
    app.update();
    let (hints, lookups) = (config(&app).hints.clone(), config(&app).lookups.clone());
    let (hints_before, lookups_before) = (
        hints.load(Ordering::Relaxed),
        lookups.load(Ordering::Relaxed),
    );

    let mut positions: Vec<IVec3> = (0..20).map(|y| FAR - IVec3::Y * y).collect();
    positions.extend(positions.clone());
    let read = read(&mut app, &positions);

    assert!(read.iter().all(|voxel| *voxel == WorldVoxel::Solid(1)));
    assert_eq!(hints.load(Ordering::Relaxed) - hints_before, 1);
    assert_eq!(lookups.load(Ordering::Relaxed) - lookups_before, 1);
}
