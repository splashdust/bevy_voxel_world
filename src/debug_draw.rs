use bevy::{ecs::system::SystemParam, prelude::*};
use std::sync::{Arc, RwLock};

#[cfg(feature = "sub_grids")]
use bevy::platform::collections::HashMap;

#[cfg(feature = "sub_grids")]
use crate::sub_grid::{Grid, SubGrid};
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
            .add_systems(Update, draw_ray_gizmos::<C>)
            .add_systems(
                PostUpdate,
                draw_voxel_gizmos::<C>.after(TransformSystems::Propagate),
            );
    }
}

pub struct VoxelGizmo {
    pub color: Srgba,
    pub pos: IVec3,
}

#[derive(Resource)]
pub(crate) struct VoxelGizmos<C: VoxelWorldConfig> {
    gizmos: Arc<RwLock<VoxelGizmoStore>>,
    _marker: std::marker::PhantomData<C>,
}

impl<C: VoxelWorldConfig> VoxelGizmos<C> {
    /// The centres of the circles the voxel gizmo draw places, with each
    /// sub-grid's pose read by `pose_of`.
    #[cfg(all(test, feature = "sub_grids"))]
    pub(crate) fn placed_circle_centres(
        &self,
        pose_of: impl Fn(Entity) -> Option<GlobalTransform>,
    ) -> Vec<Vec3> {
        let mut centres = Vec::new();
        self.gizmos
            .read()
            .unwrap()
            .place(pose_of, |circle, _| centres.push(circle.centre));
        centres
    }
}

/// The voxel gizmos of every grid of one world.
#[derive(Default)]
struct VoxelGizmoStore {
    world: Vec<VoxelGizmo>,
    /// Each sub-grid's gizmos, by the sub-grid's entity.
    #[cfg(feature = "sub_grids")]
    sub_grids: HashMap<Entity, Vec<VoxelGizmo>>,
}

impl VoxelGizmoStore {
    #[cfg(feature = "sub_grids")]
    fn set(&mut self, grid: Grid, gizmo: VoxelGizmo) {
        match grid {
            Grid::World => self.world.push(gizmo),
            Grid::SubGrid(entity) => {
                self.sub_grids.entry(entity).or_default().push(gizmo)
            }
        }
    }

    fn clear_world(&mut self, pos: IVec3) {
        self.world.retain(|gizmo| gizmo.pos != pos);
    }

    #[cfg(feature = "sub_grids")]
    fn clear(&mut self, grid: Grid, pos: IVec3) {
        match grid {
            Grid::World => self.clear_world(pos),
            Grid::SubGrid(entity) => {
                if let Some(gizmos) = self.sub_grids.get_mut(&entity) {
                    gizmos.retain(|gizmo| gizmo.pos != pos);
                    if gizmos.is_empty() {
                        self.sub_grids.remove(&entity);
                    }
                }
            }
        }
    }

    /// Drops the gizmos of each sub-grid `is_live` says is gone.
    #[cfg(feature = "sub_grids")]
    fn prune(&mut self, is_live: impl Fn(Entity) -> bool) {
        self.sub_grids.retain(|entity, _| is_live(*entity));
    }

    /// Places the six circles of every gizmo, passing each to `draw` with
    /// its gizmo's colour.
    ///
    /// The world's gizmos are placed with the identity transform. A
    /// sub-grid's go through the pose `pose_of` gives its entity, and are
    /// skipped when it gives none.
    fn place(
        &self,
        #[cfg(feature = "sub_grids")] pose_of: impl Fn(Entity) -> Option<GlobalTransform>,
        mut draw: impl FnMut(GizmoCircle, Srgba),
    ) {
        for gizmo in &self.world {
            for circle in voxel_gizmo_circles(gizmo.pos, &GlobalTransform::IDENTITY) {
                draw(circle, gizmo.color);
            }
        }
        #[cfg(feature = "sub_grids")]
        for (entity, gizmos) in &self.sub_grids {
            let Some(pose) = pose_of(*entity) else {
                continue;
            };
            for gizmo in gizmos {
                for circle in voxel_gizmo_circles(gizmo.pos, &pose) {
                    draw(circle, gizmo.color);
                }
            }
        }
    }
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
            gizmos.write().unwrap().world.push(gizmo);
        })
    }

    pub fn clear_voxel_gizmo(&self, pos: IVec3) {
        self.clear_voxel_gizmo_fn()(pos);
    }

    pub fn clear_voxel_gizmo_fn(&self) -> Arc<dyn Fn(IVec3) + Send + Sync> {
        let gizmos = self.voxel_gizmos.gizmos.clone();
        Arc::new(move |pos: IVec3| {
            gizmos.write().unwrap().clear_world(pos);
        })
    }

    /// Clears the voxel gizmos of every grid: the world's and, with the
    /// `sub_grids` feature, each sub-grid's.
    pub fn clear_all_voxel_gizmos(&self) {
        self.clear_all_voxel_gizmos_fn()();
    }

    /// Returns a function that does what
    /// [`clear_all_voxel_gizmos`](Self::clear_all_voxel_gizmos) does.
    pub fn clear_all_voxel_gizmos_fn(&self) -> Arc<dyn Fn() + Send + Sync> {
        let gizmos = self.voxel_gizmos.gizmos.clone();
        Arc::new(move || {
            *gizmos.write().unwrap() = VoxelGizmoStore::default();
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

/// Voxel gizmos on any grid.
///
/// A sub-grid's gizmos are drawn through its `GlobalTransform` after
/// transform propagation in `PostUpdate`, so they follow the pose drawn this
/// frame and stay on their voxels as the sub-grid moves. They are dropped
/// once the draw finds the sub-grid's entity despawned, or without its
/// [`SubGrid<C>`]. [`Grid::World`] names the gizmos that the methods taking no
/// grid set and clear.
#[cfg(feature = "sub_grids")]
impl<C: VoxelWorldConfig> VoxelWorldDebugDraw<'_, C> {
    /// Marks the voxel at `gizmo.pos` on `grid`.
    pub fn set_grid_voxel_gizmo(&self, grid: Grid, gizmo: VoxelGizmo) {
        self.set_grid_voxel_gizmo_fn()(grid, gizmo);
    }

    /// Returns a function that does what
    /// [`set_grid_voxel_gizmo`](Self::set_grid_voxel_gizmo) does.
    pub fn set_grid_voxel_gizmo_fn(&self) -> Arc<dyn Fn(Grid, VoxelGizmo) + Send + Sync> {
        let gizmos = self.voxel_gizmos.gizmos.clone();
        Arc::new(move |grid, gizmo| {
            gizmos.write().unwrap().set(grid, gizmo);
        })
    }

    /// Clears the gizmos at `pos` on `grid`, and leaves other grids' gizmos
    /// at the same position.
    pub fn clear_grid_voxel_gizmo(&self, grid: Grid, pos: IVec3) {
        self.clear_grid_voxel_gizmo_fn()(grid, pos);
    }

    /// Returns a function that does what
    /// [`clear_grid_voxel_gizmo`](Self::clear_grid_voxel_gizmo) does.
    pub fn clear_grid_voxel_gizmo_fn(&self) -> Arc<dyn Fn(Grid, IVec3) + Send + Sync> {
        let gizmos = self.voxel_gizmos.gizmos.clone();
        Arc::new(move |grid, pos| {
            gizmos.write().unwrap().clear(grid, pos);
        })
    }
}

fn setup<C: VoxelWorldConfig>(mut commands: Commands) {
    commands.insert_resource(VoxelGizmos {
        gizmos: Arc::new(RwLock::new(VoxelGizmoStore::default())),
        _marker: std::marker::PhantomData::<C>,
    });
    commands.insert_resource(RayGizmos {
        gizmos: Arc::new(RwLock::new(Vec::new())),
        _marker: std::marker::PhantomData::<C>,
    });
}

/// Draws every grid's voxel gizmos, after dropping those of sub-grids that
/// are gone.
fn draw_voxel_gizmos<C: VoxelWorldConfig>(
    mut gizmos: Gizmos,
    voxel_gizmos: Res<VoxelGizmos<C>>,
    #[cfg(feature = "sub_grids")] sub_grids: Query<&GlobalTransform, With<SubGrid<C>>>,
) {
    #[cfg(feature = "sub_grids")]
    voxel_gizmos
        .gizmos
        .write()
        .unwrap()
        .prune(|entity| sub_grids.contains(entity));
    voxel_gizmos.gizmos.read().unwrap().place(
        #[cfg(feature = "sub_grids")]
        |entity| sub_grids.get(entity).ok().copied(),
        |circle, color| {
            gizmos.circle(
                Isometry3d::new(
                    circle.centre,
                    Quat::from_rotation_arc(Vec3::Z, circle.normal),
                ),
                circle.radius,
                color,
            );
        },
    );
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

/// The colour of a box `debug_draw_chunks` draws around a chunk of the
/// world's streamed voxels.
pub(crate) const WORLD_CHUNK_COLOUR: Srgba = Srgba::new(0.0, 1.0, 0.0, 1.0);

/// The colour of a box `debug_draw_chunks` draws around a sub-grid's chunk.
#[cfg(feature = "sub_grids")]
pub(crate) const SUB_GRID_CHUNK_COLOUR: Srgba = Srgba::new(1.0, 0.0, 1.0, 1.0);

/// Draws a box around each chunk, of every grid, whose published data holds
/// voxels other than air. Add it to your app to see the chunks.
///
/// Schedule it in `PostUpdate` after transform propagation
/// (`TransformSystems::Propagate`), so each box follows the chunk as it is
/// drawn this frame. A box goes through its chunk's whole `GlobalTransform`,
/// so it turns, scales and moves with the world's root or a sub-grid. A
/// sub-grid's chunks are drawn in a second colour. The cost grows with the
/// number of spawned chunks.
pub fn debug_draw_chunks<C: VoxelWorldConfig>(
    mut gizmos: Gizmos<ChunkGizmos>,
    chunks: Query<(&Chunk<C>, &GlobalTransform)>,
    voxel_world: VoxelWorld<C>,
) {
    for_each_chunk_box(chunks, &voxel_world, |placement, color| {
        gizmos.cube(placement, color);
    });
}

/// Passes the placement and colour of each box [`debug_draw_chunks`] draws
/// to `draw`.
///
/// Each chunk's published data is read from its own grid, so a sub-grid's
/// chunk shows the sub-grid's voxels, never those of the world's chunk at the
/// same key.
pub(crate) fn for_each_chunk_box<'a, C: VoxelWorldConfig>(
    chunks: impl IntoIterator<Item = (&'a Chunk<C>, &'a GlobalTransform)>,
    voxel_world: &VoxelWorld<C>,
    mut draw: impl FnMut(GlobalTransform, Srgba),
) {
    let chunk_map = voxel_world.chunk_map().get_read_lock();
    for (chunk, transform) in chunks {
        let key = chunk.grid_key();
        let Some(chunk_data) = chunk_map.get_in(&key) else {
            continue;
        };
        if chunk_data.is_empty() {
            continue;
        }
        #[cfg(feature = "sub_grids")]
        let color = if key.is_world() {
            WORLD_CHUNK_COLOUR
        } else {
            SUB_GRID_CHUNK_COLOUR
        };
        #[cfg(not(feature = "sub_grids"))]
        let color = WORLD_CHUNK_COLOUR;
        draw(chunk_box(transform, chunk.lod_level()), color);
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
pub(crate) fn chunk_box(
    transform: &GlobalTransform,
    lod_level: LodLevel,
) -> GlobalTransform {
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
    fn a_sub_grid_chunks_box_follows_a_rotated_translated_and_scaled_pose() {
        // The sub-grid's chunk at (1, 0, 0) sits at (31, -1, -1) on its grid
        // and fills (32, 0, 0) to (64, 32, 32) there. The pose turns a quarter
        // about +Y, mapping (x, y, z) to (z, y, -x), doubles and moves it, so
        // grid (32, 0, 0) lands at (10, 20, -34) and (64, 32, 32) at
        // (74, 84, -98).
        let pose = Transform::from_translation(Vec3::new(10.0, 20.0, 30.0))
            .with_rotation(Quat::from_rotation_y(std::f32::consts::FRAC_PI_2))
            .with_scale(Vec3::splat(2.0));
        let chunk = GlobalTransform::from(
            pose * Transform::from_translation(Vec3::new(31.0, -1.0, -1.0)),
        );
        assert_same_corners(
            box_corners(&chunk, 0),
            [10.0, 74.0],
            [20.0, 84.0],
            [-34.0, -98.0],
        );
    }

    #[cfg(feature = "sub_grids")]
    #[test]
    fn a_sub_grids_gizmo_circles_sit_on_its_voxels_faces_through_its_pose() {
        let sub_grid = Entity::from_raw_u32(7).unwrap();
        let mut store = VoxelGizmoStore::default();
        store.set(
            Grid::SubGrid(sub_grid),
            VoxelGizmo {
                color: Srgba::RED,
                pos: IVec3::new(1, 0, 0),
            },
        );
        store.set(
            Grid::World,
            VoxelGizmo {
                color: Srgba::BLUE,
                pos: IVec3::new(1, 0, 0),
            },
        );
        // A quarter turn about +Y maps (x, y, z) to (z, y, -x), then the
        // scale doubles it and the translation moves it, so the voxel's
        // centre (1.5, 0.5, 0.5) lands at (11, 21, 27).
        let pose = GlobalTransform::from(
            Transform::from_translation(Vec3::new(10.0, 20.0, 30.0))
                .with_rotation(Quat::from_rotation_y(std::f32::consts::FRAC_PI_2))
                .with_scale(Vec3::splat(2.0)),
        );
        let mut on_sub_grid = Vec::new();
        let mut on_world = Vec::new();
        store.place(
            |entity| (entity == sub_grid).then_some(pose),
            |circle, color| {
                if color == Srgba::RED {
                    on_sub_grid.push(circle);
                } else {
                    on_world.push(circle);
                }
            },
        );

        let faces = [
            (Vec3::new(11.0, 21.0, 26.0), Vec3::NEG_Z), // the voxel's +X face
            (Vec3::new(11.0, 21.0, 28.0), Vec3::Z),     // its -X face
            (Vec3::new(11.0, 22.0, 27.0), Vec3::Y),
            (Vec3::new(11.0, 20.0, 27.0), Vec3::NEG_Y),
            (Vec3::new(12.0, 21.0, 27.0), Vec3::X), // its +Z face
            (Vec3::new(10.0, 21.0, 27.0), Vec3::NEG_X), // its -Z face
        ];
        assert_eq!(on_sub_grid.len(), 6);
        for (centre, normal) in faces {
            let circle = circle_at(&on_sub_grid, centre);
            assert_near(circle.centre, centre);
            assert_near(circle.normal, normal);
        }
        assert_one_circle_per_face(&on_sub_grid, 1.0);

        // The world's gizmo at the same position stays on the world's voxel.
        assert_eq!(on_world.len(), 6);
        assert_near(
            circle_at(&on_world, Vec3::new(2.0, 0.5, 0.5)).centre,
            Vec3::new(2.0, 0.5, 0.5),
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
