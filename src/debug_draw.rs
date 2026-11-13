use bevy::{ecs::system::SystemParam, prelude::*};
use std::sync::{Arc, RwLock};

use crate::{
    chunk::{Chunk, ChunkKey},
    configuration::{LodLevel, VoxelWorldConfig},
    prelude::VoxelWorld,
};

#[derive(Default)]
pub struct VoxelWorldDebugDrawPlugin<C: VoxelWorldConfig> {
    _marker: std::marker::PhantomData<C>,
}

impl<C: VoxelWorldConfig> Plugin for VoxelWorldDebugDrawPlugin<C> {
    fn build(&self, app: &mut App) {
        app.init_gizmo_group::<ChunkGizmos>()
            .add_systems(Startup, setup::<C>)
            .add_systems(Update, (draw_voxel_gizmos::<C>, draw_ray_gizmos::<C>));
    }
}

pub struct VoxelGizmo {
    pub color: Srgba,
    pub pos: IVec3,
}

#[derive(Resource)]
struct VoxelGizmos<C: VoxelWorldConfig> {
    gizmos: Arc<RwLock<Vec<VoxelGizmo>>>,
    _marker: std::marker::PhantomData<C>,
}

pub struct RayGizmo {
    pub ray: Ray3d,
    pub color: Srgba,
}

#[derive(Resource)]
struct RayGizmos<C: VoxelWorldConfig> {
    gizmos: Arc<RwLock<Vec<RayGizmo>>>,
    _marker: std::marker::PhantomData<C>,
}

#[derive(SystemParam)]
pub struct VoxelWorldDebugDraw<'w, C: VoxelWorldConfig> {
    voxel_gizmos: Res<'w, VoxelGizmos<C>>,
    ray_gizmos: Res<'w, RayGizmos<C>>,
}

impl<C: VoxelWorldConfig> VoxelWorldDebugDraw<'_, C> {
    pub fn set_voxel_gizmo(&self, gizmo: VoxelGizmo) {
        self.set_voxel_gizmo_fn()(gizmo);
    }

    pub fn set_voxel_gizmo_fn(&self) -> Arc<dyn Fn(VoxelGizmo) + Send + Sync> {
        let gizmos = self.voxel_gizmos.gizmos.clone();
        Arc::new(move |gizmo| {
            gizmos.write().unwrap().push(gizmo);
        })
    }

    pub fn clear_voxel_gizmo(&self, pos: IVec3) {
        self.clear_voxel_gizmo_fn()(pos);
    }

    pub fn clear_voxel_gizmo_fn(&self) -> Arc<dyn Fn(IVec3) + Send + Sync> {
        let gizmos = self.voxel_gizmos.gizmos.clone();
        Arc::new(move |pos: IVec3| {
            gizmos.write().unwrap().retain(|gizmo| gizmo.pos != pos);
        })
    }

    pub fn clear_all_voxel_gizmos(&self) {
        self.clear_all_voxel_gizmos_fn()();
    }

    pub fn clear_all_voxel_gizmos_fn(&self) -> Arc<dyn Fn() + Send + Sync> {
        let gizmos = self.voxel_gizmos.gizmos.clone();
        Arc::new(move || {
            gizmos.write().unwrap().clear();
        })
    }

    pub fn set_ray_gizmo(&self, gizmo: RayGizmo) {
        self.set_ray_gizmo_fn()(gizmo);
    }

    pub fn set_ray_gizmo_fn(&self) -> Arc<dyn Fn(RayGizmo) + Send + Sync> {
        let gizmos = self.ray_gizmos.gizmos.clone();
        Arc::new(move |gizmo| {
            gizmos.write().unwrap().push(gizmo);
        })
    }

    pub fn clear_ray_gizmo(&self, ray: Ray3d) {
        self.clear_ray_gizmo_fn()(ray);
    }

    pub fn clear_ray_gizmo_fn(&self) -> Arc<dyn Fn(Ray3d) + Send + Sync> {
        let gizmos = self.ray_gizmos.gizmos.clone();
        Arc::new(move |ray| {
            gizmos.write().unwrap().retain(|gizmo| gizmo.ray != ray);
        })
    }

    pub fn clear_all_ray_gizmos(&self) {
        self.clear_all_ray_gizmos_fn()();
    }

    pub fn clear_all_ray_gizmos_fn(&self) -> Arc<dyn Fn() + Send + Sync> {
        let gizmos = self.ray_gizmos.gizmos.clone();
        Arc::new(move || {
            gizmos.write().unwrap().clear();
        })
    }
}

fn setup<C: VoxelWorldConfig>(mut commands: Commands) {
    commands.insert_resource(VoxelGizmos {
        gizmos: Arc::new(RwLock::new(Vec::new())),
        _marker: std::marker::PhantomData::<C>,
    });
    commands.insert_resource(RayGizmos {
        gizmos: Arc::new(RwLock::new(Vec::new())),
        _marker: std::marker::PhantomData::<C>,
    });
}

fn draw_voxel_gizmos<C: VoxelWorldConfig>(
    mut gizmos: Gizmos,
    voxel_gizmos: Res<VoxelGizmos<C>>,
) {
    for gizmo in voxel_gizmos.gizmos.read().unwrap().iter() {
        for circle in voxel_gizmo_circles(gizmo.pos, &GlobalTransform::IDENTITY) {
            gizmos.circle(
                Isometry3d::new(
                    circle.centre,
                    Quat::from_rotation_arc(Vec3::Z, circle.normal),
                ),
                circle.radius,
                gizmo.color,
            );
        }
    }
}

/// How far a voxel gizmo's circle reaches across its face, in voxels.
const GIZMO_CIRCLE_RADIUS: f32 = 0.45;

/// One circle of a voxel gizmo, lying flat on one face of the voxel.
#[derive(Clone, Copy, Debug)]
struct GizmoCircle {
    centre: Vec3,
    /// The face's outward normal, of unit length.
    normal: Vec3,
    radius: f32,
}

/// Places the six circles of a voxel gizmo, one on each face of the voxel
/// at `pos`, which fills `pos` to `pos + 1`.
///
/// `transform` takes the voxel's grid into world space. Centres go through it
/// as points and normals as normals, so the circles stay on the faces when it
/// rotates, translates or scales. Each radius scales with the shorter of its
/// face's two edges, so a circle never reaches past its face.
fn voxel_gizmo_circles(pos: IVec3, transform: &GlobalTransform) -> [GizmoCircle; 6] {
    let affine = transform.affine();
    let normal_matrix = affine.matrix3.inverse().transpose();
    let voxel_centre = pos.as_vec3() + 0.5;
    let face = |normal: Vec3| {
        let (edge_a, edge_b) = normal.any_orthonormal_pair();
        let shorter_edge = affine
            .transform_vector3(edge_a)
            .length()
            .min(affine.transform_vector3(edge_b).length());
        GizmoCircle {
            centre: affine.transform_point3(voxel_centre + normal * 0.5),
            normal: Vec3::from(normal_matrix * Vec3A::from(normal)).normalize(),
            radius: GIZMO_CIRCLE_RADIUS * shorter_edge,
        }
    };
    [
        face(Vec3::NEG_X),
        face(Vec3::X),
        face(Vec3::NEG_Y),
        face(Vec3::Y),
        face(Vec3::NEG_Z),
        face(Vec3::Z),
    ]
}

fn draw_ray_gizmos<C: VoxelWorldConfig>(
    mut gizmos: Gizmos,
    ray_gizmos: Res<RayGizmos<C>>,
) {
    for gizmo in ray_gizmos.gizmos.read().unwrap().iter() {
        gizmos.line(gizmo.ray.origin, gizmo.ray.get_point(10.0), gizmo.color);
    }
}

#[derive(Default, Reflect, GizmoConfigGroup)]
pub struct ChunkGizmos;

/// Add this system to your app to draw cuboid gizmos for non-empy chunks
pub fn debug_draw_chunks<C: VoxelWorldConfig>(
    mut gizmos: Gizmos<ChunkGizmos>,
    chunks: Query<(&Chunk<C>, &GlobalTransform)>,
    voxel_world: VoxelWorld<C>,
) {
    for (chunk, transform) in chunks.iter() {
        let color = Srgba::new(0.0, 1.0, 0.0, 1.0);

        let Some(chunk_data) = voxel_world.get_chunk_data(chunk.key()) else {
            continue;
        };

        if chunk_data.is_empty() {
            continue;
        }

        gizmos.cube(chunk_box(transform, chunk.lod_level()), color);
    }
}

/// Places the box `debug_draw_chunks` draws around a chunk, as the transform
/// that takes a unit cube centred on the origin onto the chunk's voxels.
///
/// A chunk's transform sits one voxel below its minimum corner, so in the
/// chunk's own space its voxels fill 1 to `1 + voxel_extent` on each axis, at
/// every LOD level. The box is placed in that space and then taken through
/// the chunk's whole `GlobalTransform`, so it turns, scales and moves with
/// the chunk.
fn chunk_box(transform: &GlobalTransform, lod_level: LodLevel) -> GlobalTransform {
    let extent = ChunkKey::new(IVec3::ZERO, lod_level).voxel_extent() as f32;
    *transform
        * Transform::from_translation(Vec3::splat(1.0 + extent / 2.0))
            .with_scale(Vec3::splat(extent))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_near(actual: Vec3, expected: Vec3) {
        assert!(
            actual.abs_diff_eq(expected, 1e-4),
            "expected {expected}, got {actual}"
        );
    }

    /// Finds the circle whose centre is nearest `centre`, so tests don't
    /// depend on the order the circles come in.
    fn circle_at(circles: &[GizmoCircle], centre: Vec3) -> GizmoCircle {
        *circles
            .iter()
            .min_by(|a, b| {
                a.centre
                    .distance(centre)
                    .total_cmp(&b.centre.distance(centre))
            })
            .unwrap()
    }

    fn assert_one_circle_per_face(circles: &[GizmoCircle], min_spacing: f32) {
        for (i, a) in circles.iter().enumerate() {
            for b in &circles[i + 1..] {
                assert!(
                    a.centre.distance(b.centre) > min_spacing,
                    "two circles on one face"
                );
            }
        }
    }

    #[test]
    fn voxel_gizmo_circles_sit_on_the_six_faces_of_the_voxel() {
        // Voxel (2, -3, 5) fills (2, -3, 5) to (3, -2, 6).
        let circles =
            voxel_gizmo_circles(IVec3::new(2, -3, 5), &GlobalTransform::IDENTITY);
        let faces = [
            (Vec3::new(2.0, -2.5, 5.5), Vec3::NEG_X),
            (Vec3::new(3.0, -2.5, 5.5), Vec3::X),
            (Vec3::new(2.5, -3.0, 5.5), Vec3::NEG_Y),
            (Vec3::new(2.5, -2.0, 5.5), Vec3::Y),
            (Vec3::new(2.5, -2.5, 5.0), Vec3::NEG_Z),
            (Vec3::new(2.5, -2.5, 6.0), Vec3::Z),
        ];
        for (centre, normal) in faces {
            let circle = circle_at(&circles, centre);
            assert_near(circle.centre, centre);
            assert_near(circle.normal, normal);
            assert!((circle.radius - 0.45).abs() < 1e-4);
        }
        assert_one_circle_per_face(&circles, 0.5);
    }

    #[test]
    fn voxel_gizmo_circles_follow_a_rotated_translated_and_scaled_transform() {
        // A quarter turn about +Y maps +X to -Z and +Z to +X.
        let transform = GlobalTransform::from(
            Transform::from_translation(Vec3::new(10.0, 20.0, 30.0))
                .with_rotation(Quat::from_rotation_y(std::f32::consts::FRAC_PI_2))
                .with_scale(Vec3::splat(2.0)),
        );
        let circles = voxel_gizmo_circles(IVec3::ZERO, &transform);
        // The voxel's centre (0.5, 0.5, 0.5) lands at (11, 21, 29).
        let faces = [
            (Vec3::new(11.0, 21.0, 28.0), Vec3::NEG_Z), // the voxel's +X face
            (Vec3::new(11.0, 21.0, 30.0), Vec3::Z),     // its -X face
            (Vec3::new(11.0, 22.0, 29.0), Vec3::Y),
            (Vec3::new(11.0, 20.0, 29.0), Vec3::NEG_Y),
            (Vec3::new(12.0, 21.0, 29.0), Vec3::X), // its +Z face
            (Vec3::new(10.0, 21.0, 29.0), Vec3::NEG_X), // its -Z face
        ];
        for (centre, normal) in faces {
            let circle = circle_at(&circles, centre);
            assert_near(circle.centre, centre);
            assert_near(circle.normal, normal);
            assert!((circle.radius - 0.9).abs() < 1e-4);
        }
        assert_one_circle_per_face(&circles, 1.0);
    }

    /// The eight corners of the box `chunk_box` draws, in world space.
    fn box_corners(transform: &GlobalTransform, lod_level: LodLevel) -> Vec<Vec3> {
        let placement = chunk_box(transform, lod_level);
        (0..8)
            .map(|corner: i32| {
                let unit = IVec3::new(corner & 1, (corner >> 1) & 1, corner >> 2);
                placement.transform_point(unit.as_vec3() - 0.5)
            })
            .collect()
    }

    fn assert_same_corners(actual: Vec<Vec3>, xs: [f32; 2], ys: [f32; 2], zs: [f32; 2]) {
        for x in xs {
            for y in ys {
                for z in zs {
                    let expected = Vec3::new(x, y, z);
                    assert!(
                        actual.iter().any(|c| c.abs_diff_eq(expected, 1e-3)),
                        "no corner at {expected} in {actual:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn chunk_box_follows_a_rotated_and_scaled_chunk_transform() {
        // A level-0 chunk's voxels fill chunk-local 1 to 33. A quarter turn
        // about +Y maps (x, y, z) to (z, y, -x), then the scale doubles it
        // and the translation moves it, so local 1 lands at (12, 22, 28) and
        // local 33 at (76, 86, -36).
        let transform = GlobalTransform::from(
            Transform::from_translation(Vec3::new(10.0, 20.0, 30.0))
                .with_rotation(Quat::from_rotation_y(std::f32::consts::FRAC_PI_2))
                .with_scale(Vec3::splat(2.0)),
        );
        assert_same_corners(
            box_corners(&transform, 0),
            [12.0, 76.0],
            [22.0, 86.0],
            [28.0, -36.0],
        );
    }

    #[test]
    fn chunk_box_under_the_identity_matches_the_chunk_extent() {
        // The level-0 chunk at (1, 0, -2) sits at (31, -1, -65), one voxel
        // below its minimum corner, and fills (32, 0, -64) to (64, 32, -32).
        let level_0 = GlobalTransform::from_translation(Vec3::new(31.0, -1.0, -65.0));
        assert_same_corners(
            box_corners(&level_0, 0),
            [32.0, 64.0],
            [0.0, 32.0],
            [-64.0, -32.0],
        );

        // The level-2 chunk at (4, 0, 0) fills (128, 0, 0) to (256, 128, 128).
        let level_2 = GlobalTransform::from_translation(Vec3::new(127.0, -1.0, -1.0));
        assert_same_corners(
            box_corners(&level_2, 2),
            [128.0, 256.0],
            [0.0, 128.0],
            [0.0, 128.0],
        );
    }
}
