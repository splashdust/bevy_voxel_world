///
/// Streaming policy
/// Decides, from the cameras and the chunks that exist, which chunks to spawn,
/// which to give a new LOD level and which to retire. It reads no Bevy world
/// state, so it can be tested with plain values.
///
use std::collections::VecDeque;

use bevy::{math::Affine3A, platform::collections::HashSet, prelude::*};
use rand::{Rng, RngExt};

use crate::{
    chunk::{CHUNK_SIZE_F, CHUNK_SIZE_I},
    configuration::{
        ChunkDespawnStrategy, ChunkSpawnStrategy, LodLevel, VoxelWorldConfig,
    },
};

/// What the streaming policy needs to know about one camera. View and
/// projection are kept apart, as Bevy's camera does, so that points far from
/// the origin keep their precision.
#[derive(Clone, Copy, Debug)]
pub(crate) struct CameraSnapshot {
    position: Vec3,
    world_from_view: Affine3A,
    view_from_world: Affine3A,
    clip_from_view: Mat4,
    view_from_clip: Mat4,
    /// In physical pixels; zero while the camera has no render target.
    viewport_size: UVec2,
}

impl CameraSnapshot {
    pub fn new(
        world_from_view: Affine3A,
        clip_from_view: Mat4,
        viewport_size: UVec2,
    ) -> Self {
        Self {
            position: world_from_view.translation.into(),
            world_from_view,
            view_from_world: world_from_view.inverse(),
            clip_from_view,
            view_from_clip: clip_from_view.inverse(),
            viewport_size,
        }
    }

    fn chunk_position(&self) -> IVec3 {
        self.position.as_ivec3() / CHUNK_SIZE_I
    }

    /// The ray from the near plane through a point of the viewport, given in
    /// physical pixels from its top left corner.
    fn viewport_ray(&self, point: Vec2) -> Option<Ray3d> {
        if self.viewport_size.cmpeq(UVec2::ZERO).any() {
            return None;
        }
        let mut ndc = point / self.viewport_size.as_vec2() * 2.0 - Vec2::ONE;
        ndc.y = -ndc.y;

        // Using EPSILON because an NDC with Z = 0 returns NaNs.
        let near = self.view_from_clip.project_point3a(ndc.extend(1.0).into());
        let far = self
            .view_from_clip
            .project_point3a(ndc.extend(f32::EPSILON).into());
        let origin = self.world_from_view.transform_point3a(near).into();
        let direction = self.world_from_view.transform_vector3a(far - near).into();
        Dir3::new(direction)
            .ok()
            .map(|direction| Ray3d { origin, direction })
    }

    fn world_to_ndc(&self, point: Vec3) -> Option<Vec3> {
        let view_point = self.view_from_world.transform_point3a(point.into());
        let ndc = self.clip_from_view.project_point3a(view_point);
        (!ndc.is_nan()).then_some(ndc.into())
    }

    /// Whether the chunk contains the camera, or its centre or one of its
    /// corners is in view, widened by `ndc_margin`.
    fn sees_chunk(&self, chunk_position: IVec3, ndc_margin: f32) -> bool {
        let chunk_min = chunk_position.as_vec3() * CHUNK_SIZE_F;
        let chunk_max = chunk_min + Vec3::splat(CHUNK_SIZE_F);

        let cam_pos = self.position;
        if cam_pos.cmpge(chunk_min).all() && cam_pos.cmple(chunk_max).all() {
            return true;
        }

        let limit = 1.0 + ndc_margin;
        let point_in_ndc = |point: Vec3| -> bool {
            if let Some(ndc) = self.world_to_ndc(point) {
                ndc.x >= -limit
                    && ndc.x <= limit
                    && ndc.y >= -limit
                    && ndc.y <= limit
                    && ndc.z >= -ndc_margin
                    && ndc.z <= 1.0 + ndc_margin
            } else {
                false
            }
        };

        if point_in_ndc((chunk_min + chunk_max) * 0.5) {
            return true;
        }

        for &x in &[chunk_min.x, chunk_max.x] {
            for &y in &[chunk_min.y, chunk_max.y] {
                for &z in &[chunk_min.z, chunk_max.z] {
                    if point_in_ndc(Vec3::new(x, y, z)) {
                        return true;
                    }
                }
            }
        }

        false
    }
}

/// What the streaming policy needs to know about a chunk position.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Occupancy {
    /// No chunk exists here.
    Missing,
    /// A chunk exists here but has not been published.
    Spawned,
    /// A chunk exists here and has been published.
    Published,
    /// A chunk exists here and its published voxels are all solid.
    Full,
}

/// A chunk that exists in the world.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ExistingChunk {
    pub position: IVec3,
    pub lod_level: LodLevel,
    /// Whether the renderer saw the chunk last frame; `None` for a chunk that
    /// never had a mesh.
    pub view_visible: Option<bool>,
}

/// The chunks to spawn, give a new LOD level and retire this frame.
#[derive(Debug, Default)]
pub(crate) struct StreamingPlan {
    pub spawn: Vec<(IVec3, LodLevel)>,
    pub change_lod: Vec<(IVec3, LodLevel)>,
    pub retire: Vec<IVec3>,
}

/// Plans this frame's spawns, LOD changes and retirements. `occupancy` tells
/// what is at a chunk position, and `rng` picks where to cast spawning rays.
pub(crate) fn plan<C: VoxelWorldConfig>(
    config: &C,
    cameras: &[CameraSnapshot],
    chunks: impl IntoIterator<Item = ExistingChunk>,
    occupancy: impl Fn(IVec3) -> Occupancy,
    rng: &mut impl Rng,
) -> StreamingPlan {
    let mut plan = StreamingPlan::default();
    if cameras.is_empty() {
        return plan;
    }

    plan.spawn = chunks_to_spawn(config, cameras, occupancy, rng);

    for chunk in chunks {
        let camera_position = closest_camera_position(cameras, chunk.position);
        let target_lod =
            config.chunk_lod(chunk.position, Some(chunk.lod_level), camera_position);
        if target_lod != chunk.lod_level {
            plan.change_lod.push((chunk.position, target_lod));
        }

        if should_retire(config, cameras, &chunk) {
            plan.retire.push(chunk.position);
        }
    }

    plan
}

fn chunks_to_spawn<C: VoxelWorldConfig>(
    config: &C,
    cameras: &[CameraSnapshot],
    occupancy: impl Fn(IVec3) -> Occupancy,
    rng: &mut impl Rng,
) -> Vec<(IVec3, LodLevel)> {
    let spawning_distance = config.spawning_distance() as i32;
    let spawning_distance_squared = spawning_distance.pow(2);
    let spawn_strategy = config.chunk_spawn_strategy();

    let visibility_margin = 0.0f32;
    let protected_chunk_radius_sq = (config.min_despawn_distance() as i32).pow(2);

    let mut spawn = Vec::new();
    let mut visited = HashSet::new();
    let mut chunks_deque =
        VecDeque::with_capacity(config.spawning_rays() * spawning_distance as usize);

    // Shoots a ray from the given point, and queue all (non-published) chunks intersecting the ray
    let queue_chunks_intersecting_ray_from_point =
        |camera: &CameraSnapshot, point: Vec2, queue: &mut VecDeque<IVec3>| {
            let Some(ray) = camera.viewport_ray(point) else {
                return;
            };
            let mut current = ray.origin;
            let mut t = 0.0;
            while t < (spawning_distance * CHUNK_SIZE_I) as f32 {
                let chunk_pos = current.as_ivec3() / CHUNK_SIZE_I;
                match occupancy(chunk_pos) {
                    // If we hit a full chunk, we can stop the ray early
                    Occupancy::Full => break,
                    Occupancy::Published => {}
                    Occupancy::Missing | Occupancy::Spawned => queue.push_back(chunk_pos),
                }
                t += CHUNK_SIZE_F;
                current = ray.origin + ray.direction * t;
            }
        };

    // Each frame we pick some random points on the screen
    let m = config.spawning_ray_margin();
    for _ in 0..config.spawning_rays() {
        let camera = &cameras[rng.random_range(0..cameras.len())];
        let viewport_size = camera.viewport_size;
        let random_point_in_viewport = {
            let x = rng.random::<f32>() * (viewport_size.x + m * 2) as f32 - m as f32;
            let y = rng.random::<f32>() * (viewport_size.y + m * 2) as f32 - m as f32;
            Vec2::new(x, y)
        };

        // Then, for each point, we cast a ray, picking up any unspawned chunks along the ray
        queue_chunks_intersecting_ray_from_point(
            camera,
            random_point_in_viewport,
            &mut chunks_deque,
        );
    }

    // We also queue the chunks closest to every camera to make sure they will always spawn early.
    let protected_distance = config.min_despawn_distance() as i32;
    for camera in cameras {
        for x in -protected_distance..=protected_distance {
            for y in -protected_distance..=protected_distance {
                for z in -protected_distance..=protected_distance {
                    let queue_pos = camera.chunk_position() + IVec3::new(x, y, z);
                    chunks_deque.push_back(queue_pos);
                }
            }
        }
    }

    // Then, when we have a queue of chunks, we can set them up for spawning
    while let Some(chunk_position) = chunks_deque.pop_front() {
        if visited.contains(&chunk_position)
            || chunks_deque.len() > config.max_spawn_per_frame()
        {
            continue;
        }
        visited.insert(chunk_position);

        if !chunk_is_close_to_any_camera(
            cameras,
            chunk_position,
            spawning_distance_squared,
        ) {
            continue;
        }

        if spawn_strategy == ChunkSpawnStrategy::CloseAndInView
            && !chunk_visible_to_any_camera(cameras, chunk_position, visibility_margin)
            && !chunk_is_close_to_any_camera(
                cameras,
                chunk_position,
                protected_chunk_radius_sq,
            )
        {
            continue;
        }

        if occupancy(chunk_position) != Occupancy::Missing {
            continue;
        }
        let camera_position = closest_camera_position(cameras, chunk_position);
        let lod_level = config.chunk_lod(chunk_position, None, camera_position);
        spawn.push((chunk_position, lod_level));

        if spawn_strategy != ChunkSpawnStrategy::Close {
            continue;
        }

        // If we get here, we queue the neighbors
        for x in -1..=1 {
            for y in -1..=1 {
                for z in -1..=1 {
                    let queue_pos = chunk_position + IVec3::new(x, y, z);
                    if queue_pos == chunk_position {
                        continue;
                    }
                    chunks_deque.push_back(queue_pos);
                }
            }
        }
    }

    spawn
}

fn should_retire<C: VoxelWorldConfig>(
    config: &C,
    cameras: &[CameraSnapshot],
    chunk: &ExistingChunk,
) -> bool {
    let spawning_distance_squared = (config.spawning_distance() as i32).pow(2);

    let should_be_culled = match config.chunk_despawn_strategy() {
        ChunkDespawnStrategy::FarAway => false,
        ChunkDespawnStrategy::FarAwayOrOutOfView => {
            let frustum_culled =
                !chunk_visible_to_any_camera(cameras, chunk.position, 0.0);
            if let Some(visible) = chunk.view_visible {
                !visible || frustum_culled
            } else {
                frustum_culled
            }
        }
    };
    let near_camera = chunk_is_close_to_any_camera(
        cameras,
        chunk.position,
        (CHUNK_SIZE_I * config.min_despawn_distance() as i32).pow(2),
    );

    (should_be_culled && !near_camera)
        || !chunk_is_close_to_any_camera(
            cameras,
            chunk.position,
            spawning_distance_squared + 1,
        )
}

fn chunk_is_close_to_any_camera(
    cameras: &[CameraSnapshot],
    chunk_position: IVec3,
    distance_squared: i32,
) -> bool {
    cameras.iter().any(|camera| {
        chunk_position.distance_squared(camera.chunk_position()) <= distance_squared
    })
}

fn closest_camera_position(cameras: &[CameraSnapshot], chunk_position: IVec3) -> Vec3 {
    let chunk_world_position = chunk_position.as_vec3() * CHUNK_SIZE_F;
    cameras
        .iter()
        .min_by(|a, b| {
            a.position
                .distance_squared(chunk_world_position)
                .total_cmp(&b.position.distance_squared(chunk_world_position))
        })
        .map(|camera| camera.position)
        .unwrap_or(Vec3::ZERO)
}

fn chunk_visible_to_any_camera(
    cameras: &[CameraSnapshot],
    chunk_position: IVec3,
    ndc_margin: f32,
) -> bool {
    cameras
        .iter()
        .any(|camera| camera.sees_chunk(chunk_position, ndc_margin))
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::{camera::CameraProjection, platform::collections::HashMap};
    use rand::{rngs::StdRng, SeedableRng};

    #[derive(Resource, Clone)]
    struct TestWorld {
        spawning_distance: u32,
        min_despawn_distance: u32,
        /// `CloseAndInView` spawning when set, `Close` otherwise.
        spawn_in_view: bool,
        /// `FarAwayOrOutOfView` despawning when set, `FarAway` otherwise.
        despawn_out_of_view: bool,
    }

    impl Default for TestWorld {
        fn default() -> Self {
            Self {
                spawning_distance: 3,
                min_despawn_distance: 1,
                spawn_in_view: false,
                despawn_out_of_view: false,
            }
        }
    }

    impl VoxelWorldConfig for TestWorld {
        type MaterialIndex = u8;
        type ChunkUserBundle = ();

        fn spawning_distance(&self) -> u32 {
            self.spawning_distance
        }

        fn min_despawn_distance(&self) -> u32 {
            self.min_despawn_distance
        }

        fn chunk_spawn_strategy(&self) -> ChunkSpawnStrategy {
            if self.spawn_in_view {
                ChunkSpawnStrategy::CloseAndInView
            } else {
                ChunkSpawnStrategy::Close
            }
        }

        fn chunk_despawn_strategy(&self) -> ChunkDespawnStrategy {
            if self.despawn_out_of_view {
                ChunkDespawnStrategy::FarAwayOrOutOfView
            } else {
                ChunkDespawnStrategy::FarAway
            }
        }

        /// LOD level 1 for chunks whose corner is more than two chunks from the
        /// camera, 0 otherwise.
        fn chunk_lod(
            &self,
            chunk_position: IVec3,
            _previous_lod: Option<LodLevel>,
            camera_position: Vec3,
        ) -> LodLevel {
            let corner = chunk_position.as_vec3() * CHUNK_SIZE_F;
            (corner.distance(camera_position) > 2.0 * CHUNK_SIZE_F) as LodLevel
        }
    }

    /// Far enough from the origin that no test position is negative.
    const C: IVec3 = IVec3::splat(10);

    /// A camera at the centre of `chunk`, looking along `direction`.
    fn camera(chunk: IVec3, direction: Vec3) -> CameraSnapshot {
        let position = (chunk.as_vec3() + 0.5) * CHUNK_SIZE_F;
        let world_from_view = Transform::from_translation(position)
            .looking_to(direction, Vec3::Y)
            .compute_affine();
        let projection = PerspectiveProjection {
            aspect_ratio: 16.0 / 9.0,
            ..default()
        };
        CameraSnapshot::new(
            world_from_view,
            projection.get_clip_from_view(),
            UVec2::new(1280, 720),
        )
    }

    fn existing(position: IVec3, lod_level: LodLevel) -> ExistingChunk {
        ExistingChunk {
            position,
            lod_level,
            view_visible: None,
        }
    }

    /// Plans with `occupied` as the only chunk positions that are not missing.
    fn plan_with(
        config: &TestWorld,
        cameras: &[CameraSnapshot],
        occupied: &HashMap<IVec3, Occupancy>,
        chunks: Vec<ExistingChunk>,
    ) -> StreamingPlan {
        plan(
            config,
            cameras,
            chunks,
            |position| {
                occupied
                    .get(&position)
                    .copied()
                    .unwrap_or(Occupancy::Missing)
            },
            &mut StdRng::seed_from_u64(1),
        )
    }

    /// Every chunk position within `distance` chunks of `center`.
    fn within(center: IVec3, distance: i32) -> HashSet<IVec3> {
        let range = -distance..=distance;
        let mut positions = HashSet::new();
        for x in range.clone() {
            for y in range.clone() {
                for z in range.clone() {
                    let offset = IVec3::new(x, y, z);
                    if offset.length_squared() <= distance.pow(2) {
                        positions.insert(center + offset);
                    }
                }
            }
        }
        positions
    }

    /// The positions a plan spawns, checking none is spawned twice.
    fn spawned(plan: &StreamingPlan) -> HashSet<IVec3> {
        let positions: HashSet<_> =
            plan.spawn.iter().map(|(position, _)| *position).collect();
        assert_eq!(
            positions.len(),
            plan.spawn.len(),
            "a chunk was spawned twice"
        );
        positions
    }

    #[test]
    fn nothing_is_planned_without_cameras() {
        let plan = plan_with(
            &TestWorld::default(),
            &[],
            &HashMap::new(),
            vec![existing(C, 0)],
        );

        assert!(plan.spawn.is_empty());
        assert!(plan.change_lod.is_empty());
        assert!(plan.retire.is_empty());
    }

    #[test]
    fn the_flood_fill_spawns_every_chunk_within_spawning_distance() {
        let plan = plan_with(
            &TestWorld::default(),
            &[camera(C, Vec3::NEG_Z)],
            &HashMap::new(),
            vec![],
        );

        assert_eq!(spawned(&plan), within(C, 3));
    }

    #[test]
    fn chunks_that_exist_are_not_spawned_again() {
        let occupied = within(C, 1)
            .into_iter()
            .map(|position| (position, Occupancy::Published))
            .collect();

        let plan = plan_with(
            &TestWorld::default(),
            &[camera(C, Vec3::NEG_Z)],
            &occupied,
            vec![],
        );

        let expected: HashSet<_> =
            within(C, 3).difference(&within(C, 1)).copied().collect();
        assert_eq!(spawned(&plan), expected);
    }

    #[test]
    fn in_view_spawning_skips_chunks_behind_the_camera_outside_the_protected_radius() {
        let config = TestWorld {
            spawn_in_view: true,
            despawn_out_of_view: true,
            ..default()
        };

        let plan = plan_with(&config, &[camera(C, Vec3::NEG_Z)], &HashMap::new(), vec![]);

        let spawned = spawned(&plan);
        assert!(
            within(C, 1).is_subset(&spawned),
            "chunks in the protected radius spawn"
        );
        assert!(
            spawned.contains(&(C + IVec3::new(0, 0, -2))),
            "a chunk ahead of the camera spawns"
        );
        // Queued as part of the cube around the camera, but outside the
        // protected radius.
        assert!(
            !spawned.contains(&(C + IVec3::ONE)),
            "a chunk behind the camera does not spawn"
        );
    }

    #[test]
    fn chunks_retire_beyond_spawning_distance_plus_a_margin() {
        let kept = C + IVec3::new(3, 1, 0);
        let retired = C + IVec3::new(3, 1, 1);
        let occupied = [C, kept, retired]
            .into_iter()
            .map(|position| (position, Occupancy::Published))
            .collect();

        let plan = plan_with(
            &TestWorld::default(),
            &[camera(C, Vec3::NEG_Z)],
            &occupied,
            vec![existing(C, 0), existing(kept, 0), existing(retired, 0)],
        );

        assert_eq!(plan.retire, vec![retired]);
    }

    #[test]
    fn each_camera_spawns_and_keeps_its_own_chunks() {
        let config = TestWorld {
            spawning_distance: 2,
            ..default()
        };
        let a = C;
        let b = C + IVec3::new(20, 0, 0);
        let near_b = b + IVec3::X;
        let between = a + IVec3::new(5, 0, 0);
        let occupied = [near_b, between]
            .into_iter()
            .map(|position| (position, Occupancy::Published))
            .collect();

        let plan = plan_with(
            &config,
            &[camera(a, Vec3::NEG_Z), camera(b, Vec3::NEG_Z)],
            &occupied,
            vec![existing(near_b, 0), existing(between, 0)],
        );

        let mut expected: HashSet<_> =
            within(a, 2).union(&within(b, 2)).copied().collect();
        expected.remove(&near_b);
        assert_eq!(spawned(&plan), expected);
        assert_eq!(plan.retire, vec![between]);
    }

    #[test]
    fn lod_changes_are_planned_for_chunks_whose_level_should_change() {
        let config = TestWorld::default();
        let near = C + IVec3::X;
        let far = C + IVec3::new(3, 0, 0);
        let coarse_but_near = C + IVec3::Y;
        let occupied = [near, far, coarse_but_near]
            .into_iter()
            .map(|position| (position, Occupancy::Published))
            .collect();
        let camera = camera(C, Vec3::NEG_Z);

        let plan = plan_with(
            &config,
            &[camera],
            &occupied,
            vec![
                existing(near, 0),
                existing(far, 0),
                existing(coarse_but_near, 1),
            ],
        );

        let changes: HashSet<_> = plan.change_lod.iter().copied().collect();
        let expected: HashSet<_> = [(far, 1), (coarse_but_near, 0)].into_iter().collect();
        assert_eq!(changes, expected);
        for (position, lod_level) in &plan.spawn {
            assert_eq!(
                *lod_level,
                config.chunk_lod(*position, None, camera.position),
                "{position} spawns at the LOD level chunk_lod gives it"
            );
        }
    }
}
