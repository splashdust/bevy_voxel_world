//! Tests of light in coarse chunks: chunks above LOD level 0, and level-0
//! chunks with a data shape other than the default.

use std::sync::Arc;

use bevy::ecs::system::RunSystemOnce;
use bevy::mesh::MeshTag;
use bevy::prelude::*;

use super::super::{
    chunk_status_of, meshing_app, move_cameras, spawn_camera, update_until,
};
use super::voxel_light::Scene;
use crate::chunk::CHUNK_SIZE_F;
use crate::prelude::*;

const HAS_SLOT: u32 = 1 << 31;
const STONE: u8 = 1;

/// A lit world of `scene`, at level 0 within `near` chunks of the camera
/// and at level 1 beyond, whose level-0 chunks hold `data` data voxels a
/// side.
#[derive(Resource, Clone)]
struct CoarseWorld {
    scene: Scene,
    near: i32,
    data: u32,
}

impl Default for CoarseWorld {
    fn default() -> Self {
        Self {
            scene: Arc::new(slab_and_shaft),
            near: 2,
            data: 32,
        }
    }
}

impl VoxelWorldConfig for CoarseWorld {
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
        let scene = self.scene.clone();
        Box::new(move |_, _, _| {
            let scene = scene.clone();
            Box::new(move |position, _| scene(position))
        })
    }

    fn chunk_lod(
        &self,
        chunk_position: IVec3,
        _previous_lod: Option<LodLevel>,
        camera_position: Vec3,
    ) -> LodLevel {
        let camera_chunk = (camera_position / CHUNK_SIZE_F).floor().as_ivec3();
        if chunk_position.distance_squared(camera_chunk) < self.near * self.near {
            0
        } else {
            1
        }
    }

    fn chunk_data_shape(&self, lod_level: LodLevel) -> UVec3 {
        if lod_level == 0 {
            padded_chunk_shape_uniform(self.data)
        } else {
            padded_chunk_shape_uniform(32)
        }
    }

    fn light_propagation(&self) -> Option<LightPropagation> {
        Some(LightPropagation {
            sky_fill: None,
            ..default()
        })
    }
}

/// Ground below y = 0 with a shaft 8 voxels wide down to y = -20, under a
/// slab floating at y = 10 that shades part of a level-1 chunk. The camera of
/// [`app`] is at level 0 within two chunks of the origin.
fn slab_and_shaft(p: IVec3) -> WorldVoxel {
    let slab =
        (64..96).contains(&p.x) && (10..12).contains(&p.y) && (0..64).contains(&p.z);
    let shaft =
        (104..112).contains(&p.x) && (-20..0).contains(&p.y) && (24..32).contains(&p.z);
    if slab || (p.y < 0 && !shaft) {
        WorldVoxel::Solid(STONE)
    } else {
        WorldVoxel::Air
    }
}

const SHADED: IVec3 = IVec3::new(80, 4, 30);
const SUNLIT: IVec3 = IVec3::new(110, 4, 30);
const IN_SHAFT: IVec3 = IVec3::new(106, -10, 26);

/// The level-1 chunks above and below y = 0 that the slab and the shaft lie
/// in.
const ABOVE: ChunkKey = ChunkKey {
    position: IVec3::new(2, 0, 0),
    lod_level: 1,
};
const BELOW: ChunkKey = ChunkKey {
    position: IVec3::new(2, -2, 0),
    lod_level: 1,
};

fn app(scene: Scene, data: u32) -> App {
    let mut app = meshing_app();
    app.add_plugins(VoxelWorldPlugin::with_config(CoarseWorld {
        scene,
        near: 2,
        data,
    }))
    .add_systems(Startup, spawn_camera::<CoarseWorld>);
    app
}

fn light_at(app: &mut App, position: IVec3) -> Option<LightLevels> {
    app.world_mut()
        .run_system_once(move |voxel_world: VoxelWorld<CoarseWorld>| {
            voxel_world.light_at(position)
        })
        .unwrap()
}

fn sky_at(app: &mut App, position: IVec3) -> f32 {
    light_at(app, position)
        .unwrap_or_else(|| panic!("no light at {position}"))
        .sky
}

fn tag_of(app: &mut App, key: ChunkKey) -> Option<u32> {
    app.world_mut()
        .query::<(&Chunk<CoarseWorld>, &MeshTag)>()
        .iter(app.world())
        .find(|(chunk, _)| chunk.key() == key)
        .map(|(_, tag)| tag.value)
}

fn coarse_lit(app: &mut App, positions: &[IVec3]) -> bool {
    positions
        .iter()
        .all(|position| light_at(app, *position).is_some())
}

/// The slab-and-shaft world, run until both level-1 chunks show their meshes
/// and their light has landed.
fn slab_app() -> App {
    let mut app = app(Arc::new(slab_and_shaft), 32);
    update_until(&mut app, "the coarse chunks to show their light", |app| {
        tag_of(app, ABOVE).is_some()
            && tag_of(app, BELOW).is_some()
            && coarse_lit(app, &[SHADED, SUNLIT, IN_SHAFT])
    });
    app
}

#[test]
fn a_coarse_chunks_tag_carries_its_lod_level_beside_its_slot() {
    let mut app = slab_app();
    update_until(&mut app, "the shaded coarse chunk to take a slot", |app| {
        tag_of(app, ABOVE).is_some_and(|tag| tag & HAS_SLOT != 0)
    });

    let tag = tag_of(&mut app, ABOVE).unwrap();
    assert_eq!(tag >> 24 & 15, 1, "the LOD level in bits 24-27 of {tag:#x}");
    assert_eq!(tag & 0x7000_0000, 0, "bits 28-30 of {tag:#x}");
}

#[test]
fn light_at_above_level_0_reads_a_slot_cell_and_a_tag_only_chunk() {
    let mut app = slab_app();

    let shaded = sky_at(&mut app, SHADED);
    assert!(
        shaded > 0.0 && shaded < 15.0,
        "under the slab sky light comes from its sides, {shaded}"
    );
    assert_eq!(sky_at(&mut app, SUNLIT), 15.0);
    assert_eq!(
        sky_at(&mut app, IN_SHAFT),
        15.0,
        "sky light falls down the shaft"
    );

    // The flat ground on the other side of the camera: every face sees
    // direct sky, so the chunk shows its tag alone and reads its level.
    let flat = ChunkKey::new(IVec3::new(-2, -2, 2), 1);
    update_until(&mut app, "the flat coarse chunk to show", |app| {
        tag_of(app, flat).is_some()
    });
    assert_eq!(tag_of(&mut app, flat), Some(15), "no slot, sky 15, voxel 0");
    let in_flat = IVec3::new(-50, -1, 80);
    assert_eq!(light_at(&mut app, in_flat), Some(LightLevels::new(15, 0)));
}

#[test]
fn reading_across_a_split_stays_plausible() {
    let mut app = slab_app();
    let coarse = [SHADED, SUNLIT].map(|position| sky_at(&mut app, position));

    // Into the shaded chunk, which splits into level-0 chunks.
    move_cameras::<CoarseWorld>(&mut app, Transform::from_xyz(80.0, 20.0, 30.0));
    update_until(&mut app, "the shaded chunk to split and settle", |app| {
        chunk_status_of::<CoarseWorld>(app, ABOVE).is_none()
            && coarse_lit(app, &[SHADED, SUNLIT])
            && light_is_settled(app)
    });
    let fine = [SHADED, SUNLIT].map(|position| sky_at(&mut app, position));
    for (coarse, fine) in coarse.iter().zip(&fine) {
        assert!(
            (coarse - fine).abs() <= 3.0,
            "coarse {coarse:?} and fine {fine:?} differ by more than the coarse rule allows"
        );
    }
}

fn light_is_settled(app: &mut App) -> bool {
    app.world_mut()
        .run_system_once(|voxel_world: VoxelWorld<CoarseWorld>| {
            voxel_world.light_is_settled()
        })
        .unwrap()
}

#[test]
fn an_edit_outside_a_coarse_chunks_padding_changes_nothing_there_until_its_task_runs_again(
) {
    let mut app = slab_app();
    assert_eq!(sky_at(&mut app, IN_SHAFT), 15.0);

    // A cap far above the shaft, in the chunk above, outside the shaft's
    // chunk and its padding.
    let cap = IVec3::new(96, 40, 16)..IVec3::new(120, 42, 40);
    app.world_mut()
        .run_system_once(move |mut voxel_world: VoxelWorld<CoarseWorld>| {
            for x in cap.start.x..cap.end.x {
                for y in cap.start.y..cap.end.y {
                    for z in cap.start.z..cap.end.z {
                        voxel_world
                            .set_voxel(IVec3::new(x, y, z), WorldVoxel::Solid(STONE));
                    }
                }
            }
        })
        .unwrap();
    let capped = IVec3::new(106, 40, 26);
    update_until(&mut app, "the chunk above to show the cap", |app| {
        app.world_mut()
            .run_system_once(move |voxel_world: VoxelWorld<CoarseWorld>| {
                voxel_world
                    .get_chunk_data(ABOVE)
                    .and_then(|data| data.get_voxel_at_world_position(capped))
            })
            .unwrap()
            == Some(WorldVoxel::Solid(STONE))
    });
    for _ in 0..10 {
        app.update();
    }
    assert_eq!(
        sky_at(&mut app, IN_SHAFT),
        15.0,
        "the shaft's chunk didn't run again"
    );

    // An edit inside the shaft's chunk runs its task again, whose walk finds
    // the cap.
    app.world_mut()
        .run_system_once(|mut voxel_world: VoxelWorld<CoarseWorld>| {
            voxel_world.set_voxel(IVec3::new(70, -40, 10), WorldVoxel::Air);
        })
        .unwrap();
    update_until(&mut app, "the shaft to lose its direct sky", |app| {
        light_at(app, IN_SHAFT).is_some_and(|light| light.sky < 15.0)
    });
}

/// Ground below y = 0 with a roof over part of the chunk at the origin.
fn roofed(p: IVec3) -> WorldVoxel {
    let roof = (0..16).contains(&p.x) && (8..10).contains(&p.y) && (0..32).contains(&p.z);
    if roof || p.y < 0 {
        WorldVoxel::Solid(STONE)
    } else {
        WorldVoxel::Air
    }
}

#[test]
fn level_0_chunks_with_other_data_shapes_get_plausible_light() {
    for data in [16, 24, 64] {
        let mut app = app(Arc::new(roofed), data);
        let (under_roof, open) = (IVec3::new(4, 2, 16), IVec3::new(24, 2, 16));
        update_until(&mut app, "the origin chunk's light to land", |app| {
            coarse_lit(app, &[under_roof, open])
        });

        assert_eq!(sky_at(&mut app, open), 15.0, "{data} data voxels: open sky");
        let shaded = sky_at(&mut app, under_roof);
        assert!(
            shaded > 0.0 && shaded < 15.0,
            "{data} data voxels: under the roof, lit from its side, read {shaded}"
        );
    }
}
