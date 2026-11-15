//! Carve the world, and whatever a cut frees falls, with only physics added
//! by the game.
//!
//! The world comes from 3D noise, so it has overhangs, arches and floating
//! islands, and no light propagation. The world opts in to dislodged
//! fragments with `auto_lift` on, so the crate lifts each dislodged fragment
//! into a sub-grid, which the example calls a piece, and the example gives
//! the piece a body to fall with in avian3d. A floating island cut through
//! its neck drops its smaller half. The physics, carving, stamping and panel
//! are in the `fragments` module, which another world can reuse.
//!
//! Controls:
//!   - Left click carves a sphere where the cursor points, in the world or in
//!     a falling piece. Carving a falling piece cuts it in two, and the
//!     smaller part falls on its own.
//!   - The right mouse button orbits the camera, the middle one pans it, and
//!     the wheel zooms.
//!   - The panel sets the connectivity, the limit and auto-lift, and H hides
//!     it. With auto-lift off, each fragment is outlined and stays in place.
//!
//! Colliders. Each chunk's collider is rebuilt on `ChunkPublished`, from
//! the chunk's voxels as `grid(message.grid)?.get_chunk_data(key)` gives
//! them, never from its mesh, so any mesher works. A chunk's published data
//! holds a voxel of padding around it from its neighbours, so a collider is
//! built from the chunk's own voxels only. The collider goes on the chunk's
//! entity: a world chunk is a static body, and a sub-grid's chunks are its
//! children, so avian folds their colliders into the sub-grid's body. A
//! collider despawns with its chunk. A lift's chunks, in the world and in
//! the piece, publish in the same frame, so the hole's colliders and the
//! piece's swap in the same physics step and a piece never falls through
//! the hole it came from.
//!
//! Re-parenting on `Lifted`. A lift spawns its piece as a child of the grid
//! it came from, so a piece cut from a falling piece starts out riding along
//! with it, and would despawn with it if that piece were stamped. When its
//! `Lifted` arrives, the piece moves under the world's root, keeping its
//! pose, before it gets its `RigidBody::Dynamic`, and it starts out moving
//! as its part of the piece it was cut from moved.
//!
//! Easing to the preview before stamping. When avian puts a piece to sleep,
//! it becomes a kinematic body, and `stamp_preview(piece, Grid::World)`
//! gives the pose its stamp will snap it to: the nearest quarter turn and
//! whole voxel. Its `Transform` eases there, and once it arrives,
//! `stamp(piece, Grid::World, incoming_replaces)` writes its voxels into the
//! world. The stamp lands as a handover, so the piece's chunks and the
//! world's change in the same frame, with the piece already where its voxels
//! go, and the snap never shows.
//!
//! Raycasts read voxel edits as soon as they commit, so the cursor can hit a
//! stamped piece's voxels in the world a few frames before they show there.
//!
//! Run with `cargo run --example dislodged_fragments --features
//! sub_grids,dislodged_fragments`.

use std::sync::Arc;

use avian3d::prelude::*;
use bevy::prelude::*;
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
use bevy_voxel_world::prelude::*;
use noise::{NoiseFn, Perlin};

mod fragments;

use fragments::{FragmentsConfig, FragmentsPlugin, SettingPlaces};

// Materials, and the layers of `example_voxel_texture.png` they show.
const GRASS: u8 = 0;
const STONE: u8 = 1;

/// The radius of the sphere a click carves, in voxels.
const CARVE_RADIUS: i32 = 3;

/// Where the camera looks when the example starts, and where it looks from.
const FOCUS: Vec3 = Vec3::new(0.0, 26.0, 0.0);
const EYE: Vec3 = Vec3::new(60.0, 38.0, 70.0);

const SKY: Color = Color::srgb(0.58, 0.74, 0.92);

/// The voxel world: the terrain from [`Terrain`], and the fragment
/// detection settings the panel changes.
#[derive(Resource, Clone)]
struct FragmentsWorld {
    detection: FragmentDetection<u8>,
}

impl Default for FragmentsWorld {
    fn default() -> Self {
        Self {
            detection: FragmentDetection {
                auto_lift: true,
                ..default()
            },
        }
    }
}

impl FragmentsConfig for FragmentsWorld {
    fn detection(&self) -> &FragmentDetection<u8> {
        &self.detection
    }

    fn detection_mut(&mut self) -> &mut FragmentDetection<u8> {
        &mut self.detection
    }
}

impl VoxelWorldConfig for FragmentsWorld {
    type MaterialIndex = u8;
    type ChunkUserBundle = ();

    fn spawning_distance(&self) -> u32 {
        6
    }

    // Every chunk within reach spawns and stays while the camera turns, so
    // a piece that lands out of view still lands on a collider.
    fn chunk_spawn_strategy(&self) -> ChunkSpawnStrategy {
        ChunkSpawnStrategy::Close
    }

    fn chunk_despawn_strategy(&self) -> ChunkDespawnStrategy {
        ChunkDespawnStrategy::FarAway
    }

    fn voxel_lookup_delegate(&self) -> VoxelLookupDelegate<u8> {
        Box::new(|_chunk_position, _lod, _previous| {
            let terrain = Terrain::new(SEED);
            Box::new(move |position, _previous| terrain.voxel(position))
        })
    }

    fn chunk_fill_hint(&self, key: ChunkKey) -> FillHint<u8> {
        let (min, max) = (key.min_voxel().y, key.min_voxel().y + key.voxel_extent());
        if max <= Terrain::SOLID_BELOW {
            FillHint::Uniform(WorldVoxel::Solid(STONE))
        } else if min >= Terrain::AIR_FROM {
            FillHint::Uniform(WorldVoxel::Air)
        } else {
            FillHint::Unknown
        }
    }

    fn texture_index_mapper(&self) -> Arc<dyn Fn(u8) -> [u32; 3] + Send + Sync> {
        // One texture on every face, so a piece stamped on its side looks
        // the same before and after its voxels join the world.
        Arc::new(|material| match material {
            GRASS => [3, 3, 3],
            _ => [0, 0, 0],
        })
    }

    fn voxel_texture(&self) -> Option<(String, u32)> {
        Some(("example_voxel_texture.png".into(), 4))
    }

    fn detect_dislodged_fragments(&self) -> Option<FragmentDetection<u8>> {
        Some(self.detection.clone())
    }
}

const SEED: u32 = 7;

/// Rock from 3D noise: ground whose surface folds into overhangs and
/// arches, and islands floating above it. The voxels with air above them
/// are grass.
struct Terrain {
    ground: Perlin,
    detail: Perlin,
    islands: Perlin,
}

impl Terrain {
    /// Every voxel below this height is rock.
    const SOLID_BELOW: i32 = -32;
    /// Every voxel from this height up is air.
    const AIR_FROM: i32 = 96;
    /// The height the ground's surface wanders around.
    const GROUND: f64 = 0.0;
    /// The height the islands float around, and how far they reach above
    /// and below it.
    const ISLANDS: f64 = 56.0;
    const ISLAND_REACH: f64 = 12.0;

    fn new(seed: u32) -> Self {
        Self {
            ground: Perlin::new(seed),
            detail: Perlin::new(seed.wrapping_add(1)),
            islands: Perlin::new(seed.wrapping_add(2)),
        }
    }

    /// How deep `position` is inside rock: above 0 inside, below 0 in air.
    fn density(&self, position: IVec3) -> f64 {
        let [x, y, z] = position.as_dvec3().to_array();
        // Noise stretched less upwards than sideways folds the surface over
        // itself, which is what makes overhangs and arches.
        let ground = self.ground.get([x / 40.0, y / 16.0, z / 40.0])
            + 0.35 * self.detail.get([x / 14.0, y / 14.0, z / 14.0])
            - (y - Self::GROUND) / 24.0;
        let height = (y - Self::ISLANDS) / Self::ISLAND_REACH;
        let island =
            self.islands.get([x / 26.0, y / 13.0, z / 26.0]) - 0.3 - height * height;
        ground.max(island)
    }

    fn voxel(&self, position: IVec3) -> WorldVoxel<u8> {
        if position.y < Self::SOLID_BELOW {
            return WorldVoxel::Solid(STONE);
        }
        if position.y >= Self::AIR_FROM || self.density(position) <= 0.0 {
            return WorldVoxel::Air;
        }
        if self.density(position + IVec3::Y) <= 0.0 {
            WorldVoxel::Solid(GRASS)
        } else {
            WorldVoxel::Solid(STONE)
        }
    }
}

fn main() {
    #[cfg(target_arch = "wasm32")]
    let plugins = DefaultPlugins.set(WindowPlugin {
        primary_window: Some(Window {
            canvas: Some("#bevy".into()),
            ..default()
        }),
        ..default()
    });
    #[cfg(not(target_arch = "wasm32"))]
    let plugins = DefaultPlugins;
    App::new()
        .add_plugins((plugins, PanOrbitCameraPlugin, PhysicsPlugins::default()))
        .add_plugins(VoxelWorldPlugin::with_config(FragmentsWorld::default()))
        .add_plugins(FragmentsPlugin::<FragmentsWorld>::new(
            CARVE_RADIUS,
            SettingPlaces {
                connectivity: "Carve around an overhang until it hangs by an \
                               edge or a corner: with Faces it falls, with \
                               Edges or Corners it stays.",
                limit: "Cut a large floating island in two: at 4k, halves \
                        larger than the limit both stay up.",
                auto_lift: "This example starts with it On. Turn it Off and \
                            cut through an island's neck: the piece is \
                            outlined and stays where it hung.",
            },
        ))
        .insert_resource(ClearColor(SKY))
        .add_systems(Startup, setup)
        .run();
}

fn setup(mut commands: Commands) {
    let offset = EYE - FOCUS;
    let radius = offset.length();
    commands.spawn((
        Camera3d::default(),
        Transform::from_translation(EYE).looking_at(FOCUS, Vec3::Y),
        PanOrbitCamera {
            focus: FOCUS,
            radius: Some(radius),
            yaw: Some(offset.x.atan2(offset.z)),
            pitch: Some((offset.y / radius).asin()),
            button_orbit: MouseButton::Right,
            button_pan: MouseButton::Middle,
            ..default()
        },
        VoxelWorldCamera::<FragmentsWorld>::default(),
    ));
    commands.spawn((
        DirectionalLight {
            color: Color::srgb(0.98, 0.95, 0.82),
            shadow_maps_enabled: true,
            ..default()
        },
        Transform::default().looking_at(Vec3::new(-0.3, -1.0, 0.45), Vec3::Y),
    ));
    commands.insert_resource(GlobalAmbientLight {
        color: Color::srgb(0.98, 0.95, 0.82),
        brightness: 300.0,
        affects_lightmapped_meshes: true,
    });
}

#[cfg(test)]
mod tests {
    use std::collections::{HashSet, VecDeque};

    use super::*;

    /// How far around the camera's focus the tests look, sideways.
    const NEAR: i32 = 48;

    /// The face-joined groups of solid voxels above the highest the ground
    /// reaches, near the focus: the floating islands.
    fn islands(terrain: &Terrain) -> Vec<Vec<IVec3>> {
        let focus = FOCUS.as_ivec3();
        let (low, high) = (36, Terrain::AIR_FROM);
        let near = |p: IVec3| {
            (p.x - focus.x).abs() <= NEAR
                && (p.z - focus.z).abs() <= NEAR
                && p.y >= low
                && p.y < high
        };
        let mut seen = HashSet::new();
        let mut islands = Vec::new();
        for z in -NEAR..=NEAR {
            for x in -NEAR..=NEAR {
                for y in low..high {
                    let start = focus * IVec3::new(1, 0, 1) + IVec3::new(x, y, z);
                    if seen.contains(&start) || !terrain.voxel(start).is_solid() {
                        continue;
                    }
                    let mut island = Vec::new();
                    let mut queue = VecDeque::from([start]);
                    seen.insert(start);
                    while let Some(voxel) = queue.pop_front() {
                        island.push(voxel);
                        for step in [IVec3::X, IVec3::Y, IVec3::Z] {
                            for next in [voxel + step, voxel - step] {
                                if near(next)
                                    && terrain.voxel(next).is_solid()
                                    && seen.insert(next)
                                {
                                    queue.push_back(next);
                                }
                            }
                        }
                    }
                    islands.push(island);
                }
            }
        }
        islands
    }

    #[test]
    fn the_camera_starts_in_open_air() {
        let terrain = Terrain::new(SEED);
        let eye = EYE.as_ivec3();
        for z in -3..=3 {
            for y in -3..=3 {
                for x in -3..=3 {
                    assert!(!terrain.voxel(eye + IVec3::new(x, y, z)).is_solid());
                }
            }
        }
    }

    #[test]
    fn floating_islands_hang_near_the_camera_focus() {
        let terrain = Terrain::new(SEED);
        let islands = islands(&terrain);
        for island in &islands {
            let min = island.iter().fold(IVec3::MAX, |a, b| a.min(*b));
            let max = island.iter().fold(IVec3::MIN, |a, b| a.max(*b));
            println!("island of {} voxels from {min} to {max}", island.len());
        }
        let sized = islands
            .iter()
            .filter(|island| (200..20_000).contains(&island.len()))
            .count();
        assert!(sized >= 2, "{sized} islands of a size to cut");
    }

    #[test]
    fn the_ground_near_the_camera_focus_overhangs() {
        let terrain = Terrain::new(SEED);
        let mut overhanging = 0;
        let mut columns = 0;
        for z in -NEAR..=NEAR {
            for x in -NEAR..=NEAR {
                // Rock with air under it and rock again further down, below
                // the islands.
                let mut solid_below = false;
                let mut air_below = false;
                let mut overhangs = false;
                for y in Terrain::SOLID_BELOW..36 {
                    let solid = terrain.voxel(IVec3::new(x, y, z)).is_solid();
                    if solid && air_below && solid_below {
                        overhanging += 1;
                        overhangs = true;
                    }
                    solid_below |= solid;
                    air_below = !solid && solid_below;
                }
                columns += usize::from(overhangs);
            }
        }
        println!("{overhanging} overhanging voxels in {columns} columns");
        assert!(columns >= 200, "{columns} columns overhang");
    }
}
