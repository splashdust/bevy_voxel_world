//! Cut branches off a tree at dusk, and watch them fall through lamp light.
//!
//! A meadow holds one large tree and two small ones, and lamp posts stand
//! around and under them. The sun has set, so the light that carries the
//! scene is voxel light from the lamps, which the world declares emissive,
//! with no Bevy point lights. Dusk is set through the sun and the light
//! look: a dim, low sun, little ambient light, and a `voxel_brightness` high
//! enough for the lamps to light the meadow around them. The world opts in
//! to light propagation and to dislodged fragments with `auto_lift` on, so
//! a branch cut off a tree is lifted into a sub-grid, which the example
//! calls a piece, and falls in avian3d.
//!
//! The trees are built so that the default `Connectivity::Faces` finds each
//! branch on its own. A branch is a run of voxels that steps along one axis
//! at a time, so its voxels share faces, and no leaf shares a face with
//! another branch's wood or leaves. A thin branch falls with its leaves
//! after one click, and never takes a neighbour whose leaves touch its own
//! at an edge. The trunk takes several clicks, and cutting through it fells
//! the tree, which leans enough to tip off its stump.
//!
//! A falling piece's faces read the world's light at the piece's world
//! position, so a branch brightens as it falls past a lamp and darkens as it
//! leaves it, and a tipped branch's faces take sky light from the world's
//! up, not the branch's. Once a piece comes to rest, it is stamped into the
//! world, and its voxels join the world's light: its light changes as they
//! do, and it starts casting shadows in the light around it.
//!
//! The physics, carving, easing, stamping and panel are the `fragments`
//! module of the `dislodged_fragments` example, which this example shares.
//! That example's docs explain how they work.
//!
//! Controls:
//!   - Left click carves a sphere about a branch wide where the cursor
//!     points, in the world or in a falling piece. Carving a falling branch
//!     cuts it in two.
//!   - The right mouse button orbits the camera, the middle one pans it, and
//!     the wheel zooms.
//!   - The panel sets the connectivity, the limit and auto-lift, and H hides
//!     it. With auto-lift off, each fragment is outlined and stays in place.
//!
//! Run with `cargo run --example dislodged_fragments_tree --features
//! sub_grids,dislodged_fragments,light_propagation`.

use std::sync::Arc;

use avian3d::prelude::*;
use bevy::prelude::*;
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
use bevy_voxel_world::prelude::*;

#[path = "../dislodged_fragments/fragments/mod.rs"]
mod fragments;
mod scene;

use fragments::{FragmentsConfig, FragmentsPlugin, SettingPlaces};
use scene::Scene;

/// The radius of the sphere a click carves, in voxels: a little wider than a
/// thin branch, so one click through it cuts it off.
const CARVE_RADIUS: i32 = 2;

/// Where the camera looks when the example starts, and where it looks from.
const FOCUS: Vec3 = Vec3::new(0.0, 14.0, 0.0);
const EYE: Vec3 = Vec3::new(50.0, 24.0, -52.0);

/// The sky just after sunset.
const DUSK: Color = Color::srgb(0.13, 0.14, 0.27);

/// The voxel world: the meadow from [`Scene`], and the fragment detection
/// settings the panel changes.
#[derive(Resource, Clone)]
struct TreeWorld {
    scene: Arc<Scene>,
    detection: FragmentDetection<u8>,
}

impl Default for TreeWorld {
    fn default() -> Self {
        Self {
            scene: Arc::new(Scene::new()),
            detection: FragmentDetection {
                auto_lift: true,
                ..default()
            },
        }
    }
}

impl FragmentsConfig for TreeWorld {
    fn detection(&self) -> &FragmentDetection<u8> {
        &self.detection
    }

    fn detection_mut(&mut self) -> &mut FragmentDetection<u8> {
        &mut self.detection
    }
}

impl VoxelWorldConfig for TreeWorld {
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
        let scene = self.scene.clone();
        Box::new(move |_chunk_position, _lod, _previous| {
            let scene = scene.clone();
            Box::new(move |position, _previous| scene.voxel(position))
        })
    }

    fn chunk_fill_hint(&self, key: ChunkKey) -> FillHint<u8> {
        let (min, max) = (key.min_voxel().y, key.min_voxel().y + key.voxel_extent());
        if max <= Scene::SOLID_BELOW {
            FillHint::Uniform(WorldVoxel::Solid(scene::STONE))
        } else if min >= Scene::AIR_FROM {
            FillHint::Uniform(WorldVoxel::Air)
        } else {
            FillHint::Unknown
        }
    }

    fn texture_index_mapper(&self) -> Arc<dyn Fn(u8) -> [u32; 3] + Send + Sync> {
        // One tile on every face, so a branch stamped on its side looks the
        // same before and after its voxels join the world. The tiles are
        // those of `mountain_house_texture.png`, from the top.
        Arc::new(|material| match material {
            scene::GRASS => [0; 3],
            scene::DIRT => [2; 3],
            scene::BARK => [12; 3],
            scene::LEAVES => [11; 3],
            scene::POST => [8; 3],
            scene::LAMP => [13; 3],
            _ => [3; 3],
        })
    }

    fn voxel_texture(&self) -> Option<(String, u32)> {
        Some(("mountain_house_texture.png".into(), 18))
    }

    fn light_propagation(&self) -> Option<LightPropagation> {
        Some(LightPropagation {
            // Nothing stands above `AIR_FROM`, so sky light enters every
            // column open up to there.
            sky: Sky::Ceiling {
                y: Scene::AIR_FROM - 1,
            },
            // Brighter voxel light than the default, so the lamps light the
            // meadow around them under a sky this dark.
            look: LightLook {
                voxel_brightness: 4000.0,
                ..default()
            },
            ..default()
        })
    }

    fn emissive_materials(&self) -> Vec<(u8, Emission)> {
        vec![(scene::LAMP, Emission::level(15.0))]
    }

    fn detect_dislodged_fragments(&self) -> Option<FragmentDetection<u8>> {
        Some(self.detection.clone())
    }
}

fn main() {
    let window = WindowPlugin {
        #[cfg(target_arch = "wasm32")]
        primary_window: Some(Window {
            canvas: Some("#bevy".into()),
            ..default()
        }),
        ..default()
    };
    App::new()
        .add_plugins((
            DefaultPlugins
                .set(ImagePlugin::default_nearest())
                .set(window),
            PanOrbitCameraPlugin,
            PhysicsPlugins::default(),
        ))
        .add_plugins(VoxelWorldPlugin::with_config(TreeWorld::default()))
        .add_plugins(FragmentsPlugin::<TreeWorld>::new(
            CARVE_RADIUS,
            SettingPlaces {
                connectivity: "Cut a thin branch whose leaves touch another \
                               branch's: with Faces it falls with its own \
                               leaves, with Edges or Corners the leaves where \
                               they touch hold it up.",
                limit: "Cut through the large tree's trunk near its foot: at \
                        4k the tree is over the limit and stays standing on \
                        nothing.",
                auto_lift: "This example starts with it On. Turn it Off and \
                            cut a branch: it is outlined and stays where it \
                            grew.",
            },
        ))
        .insert_resource(ClearColor(DUSK))
        .add_systems(Startup, setup)
        .run();
}

/// The camera and the last of the daylight. The lamps need no Bevy lights:
/// all their light is voxel light.
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
        VoxelWorldCamera::<TreeWorld>::default(),
    ));
    // A dim sun low in the sky, which lights little more than the tops of
    // things.
    commands.spawn((
        DirectionalLight {
            color: Color::srgb(1.0, 0.62, 0.42),
            illuminance: 600.0,
            shadow_maps_enabled: true,
            ..default()
        },
        Transform::default().looking_at(Vec3::new(1.0, -0.35, 0.3), Vec3::Y),
    ));
    commands.insert_resource(GlobalAmbientLight {
        color: Color::srgb(0.55, 0.6, 0.95),
        brightness: 250.0,
        affects_lightmapped_meshes: true,
    });
}

#[cfg(test)]
mod tests {
    use bevy::ecs::system::RunSystemOnce;

    use super::*;

    #[test]
    fn the_camera_starts_in_open_air() {
        let scene = Scene::new();
        let eye = EYE.as_ivec3();
        for z in -3..=3 {
            for y in -3..=3 {
                for x in -3..=3 {
                    assert!(!scene.voxel(eye + IVec3::new(x, y, z)).is_solid());
                }
            }
        }
    }

    #[test]
    fn the_scene_spawns_no_point_or_spot_light() {
        let mut world = World::new();
        world.run_system_once(setup).unwrap();
        let mut point = world.query::<&PointLight>();
        let mut spot = world.query::<&SpotLight>();
        assert_eq!(point.iter(&world).count(), 0);
        assert_eq!(spot.iter(&world).count(), 0);
        let mut cameras = world.query::<&Camera3d>();
        assert_eq!(cameras.iter(&world).count(), 1);
    }
}
