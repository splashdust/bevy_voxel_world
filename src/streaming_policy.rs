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
    chunk::{chunk_containing, CHUNK_SIZE_F, CHUNK_SIZE_I},
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
        chunk_containing(self.position.floor().as_ivec3())
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
    /// A chunk exists here, but its voxels aren't published yet.
    Unpublished,
    /// A chunk exists here and its published voxels are all air.
    Empty,
    /// A chunk exists here and its published voxels are solid in places.
    Mixed,
    /// A chunk exists here and its published voxels are all solid.
    Full,
}

/// How many missing chunks a spawning ray queues past the published chunks in
/// its way, in open air: the next one, and one behind it in case that is air
/// too. Queueing more would spawn more chunks underground wherever a ray meets
/// the ground before the ground chunk publishes.
const RAY_LOOKAHEAD: usize = 2;

/// The most rays cast in a frame, as a multiple of `spawning_rays`, while the
/// lifecycle has room for more chunks than the rays have found.
const MAX_RAYS_PER_SPAWNING_RAY: usize = 16;

/// How far from a point where a ray found chunks the rays cast near it land, as
/// a fraction of the viewport: a few chunks either way at the far end of the
/// view, where a chunk covers around a hundredth of the viewport.
const RAY_JITTER: f32 = 1.0 / 16.0;

/// A chunk that exists in the world.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ExistingChunk {
    pub position: IVec3,
    pub lod_level: LodLevel,
}

/// The chunks to spawn, give a new LOD level and retire this frame.
#[derive(Debug, Default)]
pub(crate) struct StreamingPlan {
    pub spawn: Vec<(IVec3, LodLevel)>,
    pub change_lod: Vec<(IVec3, LodLevel)>,
    pub retire: Vec<IVec3>,
}

/// Plans this frame's spawns, LOD changes and retirements. `occupancy` tells
/// what is at a chunk position, `wanted` how many chunks the lifecycle could
/// start right away beyond those already waiting, and `rng` picks where to cast
/// spawning rays.
pub(crate) fn plan<C: VoxelWorldConfig>(
    config: &C,
    cameras: &[CameraSnapshot],
    chunks: impl IntoIterator<Item = ExistingChunk>,
    occupancy: impl Fn(IVec3) -> Occupancy,
    wanted: usize,
    rng: &mut impl Rng,
) -> StreamingPlan {
    let mut plan = StreamingPlan::default();
    if cameras.is_empty() {
        return plan;
    }

    plan.spawn = chunks_to_spawn(config, cameras, occupancy, wanted, rng);

    for chunk in chunks {
        let camera_position = closest_camera_position(cameras, chunk.position);
        let target_lod =
            config.chunk_lod(chunk.position, Some(chunk.lod_level), camera_position);
        if target_lod != chunk.lod_level {
            plan.change_lod.push((chunk.position, target_lod));
        }

        if should_retire(config, cameras, chunk.position) {
            plan.retire.push(chunk.position);
        }
    }

    plan
}

fn chunks_to_spawn<C: VoxelWorldConfig>(
    config: &C,
    cameras: &[CameraSnapshot],
    occupancy: impl Fn(IVec3) -> Occupancy,
    wanted: usize,
    rng: &mut impl Rng,
) -> Vec<(IVec3, LodLevel)> {
    let spawning_distance = config.spawning_distance() as i32;
    let spawning_distance_squared = spawning_distance.pow(2);
    let spawn_strategy = config.chunk_spawn_strategy();
    // Out of view, only chunks in the protected radius are spawned: when spawning
    // in view, and when despawning out of view, since any other chunk would be
    // retired as soon as it existed.
    let in_view_only = spawn_strategy == ChunkSpawnStrategy::CloseAndInView
        || config.chunk_despawn_strategy() == ChunkDespawnStrategy::FarAwayOrOutOfView;

    let mut spawn = Vec::new();
    let mut visited = HashSet::new();
    // Positions queued by the protected cubes and the rays, so a ray can tell
    // whether it found a chunk no other ray had.
    let mut queued = HashSet::new();
    let mut chunks_deque =
        VecDeque::with_capacity(config.spawning_rays() * RAY_LOOKAHEAD);

    // Every camera's protected cube is queued first, nearest chunks first, so
    // they spawn before any other chunk.
    let protected_distance = config.min_despawn_distance() as i32;
    let mut protected_offsets = Vec::new();
    for x in -protected_distance..=protected_distance {
        for y in -protected_distance..=protected_distance {
            for z in -protected_distance..=protected_distance {
                protected_offsets.push(IVec3::new(x, y, z));
            }
        }
    }
    protected_offsets.sort_by_key(|offset| offset.length_squared());
    for offset in protected_offsets {
        for camera in cameras {
            let chunk_pos = camera.chunk_position() + offset;
            if queued.insert(chunk_pos) {
                chunks_deque.push_back(chunk_pos);
            }
        }
    }

    // Shoots a ray from the given point, and queues the first missing chunks past
    // the published chunks in its way, returning how many no ray had queued. The
    // ray walks through chunks whose published voxels have air in them. It stops
    // at a full chunk, since nothing behind it can be seen, and at a chunk that
    // isn't published yet, since that chunk decides whether the ones behind it
    // can be. So chunks spawn from the cameras outwards as the chunks in front of
    // them publish, and chunks hidden underground are hardly ever spawned.
    let queue_chunks_intersecting_ray_from_point =
        |camera: &CameraSnapshot,
         point: Vec2,
         queue: &mut VecDeque<IVec3>,
         queued: &mut HashSet<IVec3>| {
            let Some(ray) = camera.viewport_ray(point) else {
                return 0;
            };
            let mut found = 0;
            let mut lookahead = RAY_LOOKAHEAD;
            let mut current = ray.origin;
            let mut t = 0.0;
            while t < (spawning_distance * CHUNK_SIZE_I) as f32 {
                let chunk_pos = chunk_containing(current.floor().as_ivec3());
                match occupancy(chunk_pos) {
                    Occupancy::Empty => {}
                    // Past a chunk with solid voxels, the next one is likely solid.
                    Occupancy::Mixed => lookahead = lookahead.min(1),
                    Occupancy::Full | Occupancy::Unpublished => break,
                    Occupancy::Missing => {
                        if queued.insert(chunk_pos) {
                            queue.push_back(chunk_pos);
                            found += 1;
                        }
                        lookahead -= 1;
                        if lookahead == 0 {
                            break;
                        }
                    }
                }
                t += CHUNK_SIZE_F;
                current = ray.origin + ray.direction * t;
            }
            found
        };

    // Each frame some random points on the screen get a ray. While the lifecycle
    // could start more chunks than the rays have found, more rays are cast, up to
    // a limit, as long as they keep finding chunks: a ray that ends at a chunk
    // that isn't published yet finds nothing, and once every ray does, the chunks
    // found so far must publish before more can be found. The chunks to find lie
    // in patches of the view, such as the sky above a far ridge, so every other
    // extra ray is cast near a point where a ray found chunks this frame.
    let base_rays = config.spawning_rays();
    let max_rays = base_rays.saturating_mul(MAX_RAYS_PER_SPAWNING_RAY);
    let m = config.spawning_ray_margin();
    let mut rays = 0;
    let mut found = 0;
    let mut misses = 0;
    let mut hits: Vec<(usize, Vec2)> = Vec::new();
    while rays < base_rays || (found < wanted && misses < base_rays && rays < max_rays) {
        rays += 1;
        let (camera_index, point) =
            if rays > base_rays && !hits.is_empty() && rng.random() {
                let (camera_index, hit) = hits[rng.random_range(0..hits.len())];
                let jitter =
                    Vec2::new(rng.random_range(-1.0..=1.0), rng.random_range(-1.0..=1.0))
                        * cameras[camera_index].viewport_size.as_vec2()
                        * RAY_JITTER;
                (camera_index, hit + jitter)
            } else {
                let camera_index = rng.random_range(0..cameras.len());
                let viewport_size = cameras[camera_index].viewport_size;
                let x = rng.random::<f32>() * (viewport_size.x + m * 2) as f32 - m as f32;
                let y = rng.random::<f32>() * (viewport_size.y + m * 2) as f32 - m as f32;
                (camera_index, Vec2::new(x, y))
            };

        let found_by_ray = queue_chunks_intersecting_ray_from_point(
            &cameras[camera_index],
            point,
            &mut chunks_deque,
            &mut queued,
        );
        if found_by_ray == 0 {
            misses += 1;
        } else {
            misses = 0;
            found += found_by_ray;
            hits.push((camera_index, point));
        }
    }

    // Then, when we have a queue of chunks, we can set them up for spawning
    while let Some(chunk_position) = chunks_deque.pop_front() {
        if spawn.len() >= config.max_spawn_per_frame() {
            break;
        }
        if !visited.insert(chunk_position) {
            continue;
        }

        if !chunk_is_close_to_any_camera(
            cameras,
            chunk_position,
            spawning_distance_squared,
        ) {
            continue;
        }

        if in_view_only && out_of_view_and_unprotected(config, cameras, chunk_position) {
            continue;
        }

        if occupancy(chunk_position) != Occupancy::Missing {
            continue;
        }
        let camera_position = closest_camera_position(cameras, chunk_position);
        let lod_level = config.chunk_lod(chunk_position, None, camera_position);
        let distance_squared =
            camera_position.distance_squared(chunk_centre(chunk_position));
        spawn.push((distance_squared, chunk_position, lod_level));

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

    // Nearest a camera first, which is the order the chunks spawn in.
    spawn.sort_by(|a, b| a.0.total_cmp(&b.0));
    spawn
        .into_iter()
        .map(|(_, position, lod_level)| (position, lod_level))
        .collect()
}

/// Chunks are retired only beyond the spawning distance plus this margin, in
/// squared chunks, leaving a thin band where existing chunks are kept but none
/// are spawned.
const RETIRE_MARGIN_SQUARED: i32 = 1;

fn should_retire<C: VoxelWorldConfig>(
    config: &C,
    cameras: &[CameraSnapshot],
    chunk_position: IVec3,
) -> bool {
    let spawning_distance_squared = (config.spawning_distance() as i32).pow(2);
    if !chunk_is_close_to_any_camera(
        cameras,
        chunk_position,
        spawning_distance_squared + RETIRE_MARGIN_SQUARED,
    ) {
        return true;
    }

    config.chunk_despawn_strategy() == ChunkDespawnStrategy::FarAwayOrOutOfView
        && out_of_view_and_unprotected(config, cameras, chunk_position)
}

/// Whether no camera sees the chunk and it is outside every camera's
/// protected radius.
fn out_of_view_and_unprotected<C: VoxelWorldConfig>(
    config: &C,
    cameras: &[CameraSnapshot],
    chunk_position: IVec3,
) -> bool {
    let protected_radius_squared = (config.min_despawn_distance() as i32).pow(2);
    !chunk_is_close_to_any_camera(cameras, chunk_position, protected_radius_squared)
        && !chunk_visible_to_any_camera(cameras, chunk_position, 0.0)
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

pub(crate) fn chunk_centre(chunk_position: IVec3) -> Vec3 {
    (chunk_position.as_vec3() + 0.5) * CHUNK_SIZE_F
}

/// The position of the camera nearest the chunk's centre.
fn closest_camera_position(cameras: &[CameraSnapshot], chunk_position: IVec3) -> Vec3 {
    let chunk_centre = chunk_centre(chunk_position);
    cameras
        .iter()
        .min_by(|a, b| {
            a.position
                .distance_squared(chunk_centre)
                .total_cmp(&b.position.distance_squared(chunk_centre))
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

    #[derive(Resource, Clone, Debug)]
    struct TestWorld {
        spawning_distance: u32,
        min_despawn_distance: u32,
        /// `CloseAndInView` spawning when set, `Close` otherwise.
        spawn_in_view: bool,
        /// `FarAwayOrOutOfView` despawning when set, `FarAway` otherwise.
        despawn_out_of_view: bool,
        max_spawn_per_frame: usize,
        spawning_rays: usize,
    }

    impl Default for TestWorld {
        fn default() -> Self {
            Self {
                spawning_distance: 3,
                min_despawn_distance: 1,
                spawn_in_view: false,
                despawn_out_of_view: false,
                max_spawn_per_frame: 10_000,
                spawning_rays: 100,
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

        fn max_spawn_per_frame(&self) -> usize {
            self.max_spawn_per_frame
        }

        fn spawning_rays(&self) -> usize {
            self.spawning_rays
        }

        /// LOD level 1 for chunks whose centre is more than one and a half
        /// chunks from the camera, 0 otherwise.
        fn chunk_lod(
            &self,
            chunk_position: IVec3,
            _previous_lod: Option<LodLevel>,
            camera_position: Vec3,
        ) -> LodLevel {
            let centre = (chunk_position.as_vec3() + 0.5) * CHUNK_SIZE_F;
            (centre.distance(camera_position) > 1.5 * CHUNK_SIZE_F) as LodLevel
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
        }
    }

    /// Plans with `occupied` as the only chunk positions that are not missing,
    /// with no chunks wanted beyond what `spawning_rays` rays find.
    fn plan_with(
        config: &TestWorld,
        cameras: &[CameraSnapshot],
        occupied: &HashMap<IVec3, Occupancy>,
        chunks: Vec<ExistingChunk>,
    ) -> StreamingPlan {
        plan_wanting(config, cameras, occupied, chunks, 0)
    }

    fn plan_wanting(
        config: &TestWorld,
        cameras: &[CameraSnapshot],
        occupied: &HashMap<IVec3, Occupancy>,
        chunks: Vec<ExistingChunk>,
        wanted: usize,
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
            wanted,
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
            .map(|position| (position, Occupancy::Empty))
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

        let first = spawned(&plan);
        assert!(
            within(C, 1).is_subset(&first),
            "chunks in the protected radius spawn"
        );
        // Queued as part of the cube around the camera, but outside the
        // protected radius.
        assert!(
            !first.contains(&(C + IVec3::ONE)),
            "a chunk behind the camera does not spawn"
        );

        // Once the chunks around the camera are published, rays walk out of them.
        let plan = plan_with(&config, &[camera(C, Vec3::NEG_Z)], &air_around(C), vec![]);

        assert!(
            spawned(&plan).contains(&(C + IVec3::new(0, 0, -2))),
            "a chunk ahead of the camera spawns"
        );
    }

    #[test]
    fn chunks_retire_beyond_spawning_distance_plus_a_margin() {
        let kept = C + IVec3::new(3, 1, 0);
        let retired = C + IVec3::new(3, 1, 1);
        let occupied = [C, kept, retired]
            .into_iter()
            .map(|position| (position, Occupancy::Unpublished))
            .collect();

        let plan = plan_with(
            &TestWorld::default(),
            &[camera(C, Vec3::NEG_Z)],
            &occupied,
            vec![existing(C, 0), existing(kept, 0), existing(retired, 0)],
        );

        assert_eq!(plan.retire, vec![retired]);
    }

    /// Plans for a camera at `C` looking along -z, with chunks behind it just
    /// inside and just outside the protected radius, and one ahead of it.
    fn plan_with_chunks_behind_the_camera(config: &TestWorld) -> (StreamingPlan, IVec3) {
        let protected = C + IVec3::new(0, 0, 1);
        let unprotected = C + IVec3::new(0, 0, 2);
        let ahead = C + IVec3::new(0, 0, -2);
        let occupied = [C, protected, unprotected, ahead]
            .into_iter()
            .map(|position| (position, Occupancy::Unpublished))
            .collect();

        let plan = plan_with(
            config,
            &[camera(C, Vec3::NEG_Z)],
            &occupied,
            vec![
                existing(C, 0),
                existing(protected, 0),
                existing(unprotected, 0),
                existing(ahead, 0),
            ],
        );
        (plan, unprotected)
    }

    #[test]
    fn out_of_view_despawning_retires_chunks_out_of_view_outside_the_protected_radius() {
        let config = TestWorld {
            despawn_out_of_view: true,
            ..default()
        };

        let (plan, unprotected) = plan_with_chunks_behind_the_camera(&config);

        assert_eq!(plan.retire, vec![unprotected]);
    }

    #[test]
    fn far_away_despawning_keeps_chunks_out_of_view() {
        let (plan, _) = plan_with_chunks_behind_the_camera(&TestWorld::default());

        assert!(plan.retire.is_empty(), "retired {:?}", plan.retire);
    }

    #[test]
    fn a_chunk_the_plan_spawns_is_not_retired_by_the_next_plan() {
        let mut rng = StdRng::seed_from_u64(48);
        for case in 0..200 {
            let config = TestWorld {
                spawning_distance: rng.random_range(1..=4),
                min_despawn_distance: rng.random_range(0..=2),
                spawn_in_view: rng.random(),
                despawn_out_of_view: rng.random(),
                ..default()
            };
            let cameras: Vec<_> = (0..rng.random_range(1..=2))
                .map(|_| {
                    let chunk = C + IVec3::new(
                        rng.random_range(-4..=4),
                        rng.random_range(-4..=4),
                        rng.random_range(-4..=4),
                    );
                    let direction = Vec3::new(
                        rng.random_range(-1.0..=1.0),
                        rng.random_range(-0.5..=0.5),
                        rng.random_range(-1.0..=1.0),
                    );
                    camera(chunk, direction.try_normalize().unwrap_or(Vec3::NEG_Z))
                })
                .collect();

            let first = plan_with(&config, &cameras, &HashMap::new(), vec![]);
            let occupied = first
                .spawn
                .iter()
                .map(|(position, _)| (*position, Occupancy::Unpublished))
                .collect();
            let chunks = first
                .spawn
                .iter()
                .map(|(position, lod_level)| existing(*position, *lod_level))
                .collect();
            let next = plan_with(&config, &cameras, &occupied, chunks);

            assert!(
                next.retire.is_empty(),
                "case {case}, {config:?}: retired chunks it had just spawned: {:?}",
                next.retire
            );
        }
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
            .map(|position| (position, Occupancy::Unpublished))
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
            .map(|position| (position, Occupancy::Unpublished))
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

    #[test]
    fn lod_levels_are_chosen_with_the_camera_nearest_the_chunk_centre() {
        // `a` is nearer the chunk's corner, `b` nearer its centre.
        let a = camera(C - IVec3::ONE, Vec3::NEG_Z);
        let b = camera(C + IVec3::X, Vec3::NEG_Z);
        let occupied = [(C, Occupancy::Unpublished)].into_iter().collect();

        let plan = plan_with(
            &TestWorld::default(),
            &[a, b],
            &occupied,
            vec![existing(C, 1)],
        );

        assert_eq!(plan.change_lod, vec![(C, 0)]);
    }

    #[test]
    fn at_most_max_spawn_per_frame_chunks_spawn_nearest_the_cameras_first() {
        let config = TestWorld {
            max_spawn_per_frame: 14,
            ..default()
        };
        let a = C;
        let b = C + IVec3::new(20, 0, 0);

        let plan = plan_with(
            &config,
            &[camera(a, Vec3::NEG_Z), camera(b, Vec3::NEG_Z)],
            &HashMap::new(),
            vec![],
        );

        let expected: HashSet<_> = within(a, 1).union(&within(b, 1)).copied().collect();
        assert_eq!(spawned(&plan), expected);
    }

    /// A chunk whose centre is at negative coordinates on every axis.
    const NEGATIVE: IVec3 = IVec3::splat(-1);

    #[test]
    fn a_camera_in_negative_coordinates_protects_its_own_chunk() {
        let config = TestWorld {
            spawn_in_view: true,
            ..default()
        };

        let plan = plan_with(
            &config,
            &[camera(NEGATIVE, Vec3::NEG_Z)],
            &HashMap::new(),
            vec![],
        );

        let spawned = spawned(&plan);
        let missing: Vec<_> = within(NEGATIVE, 1).difference(&spawned).copied().collect();
        assert!(
            missing.is_empty(),
            "protected chunks did not spawn: {missing:?}"
        );
    }

    #[test]
    fn rays_find_chunks_ahead_of_a_camera_in_negative_coordinates() {
        let config = TestWorld {
            spawn_in_view: true,
            ..default()
        };

        let plan = plan_with(
            &config,
            &[camera(NEGATIVE, Vec3::NEG_Z)],
            &air_around(NEGATIVE),
            vec![],
        );

        assert!(spawned(&plan).contains(&(NEGATIVE + IVec3::new(0, 0, -2))));
    }

    /// The chunks around a camera's chunk, published as air, so rays walk out
    /// of them.
    fn air_around(chunk: IVec3) -> HashMap<IVec3, Occupancy> {
        within(chunk, 1)
            .into_iter()
            .map(|position| (position, Occupancy::Empty))
            .collect()
    }

    #[test]
    fn rays_walk_through_published_chunks_with_air_and_stop_at_the_others() {
        let config = TestWorld {
            spawning_distance: 4,
            spawn_in_view: true,
            ..default()
        };
        let ahead = C + IVec3::new(0, 0, -2);
        let behind_it = C + IVec3::new(0, 0, -3);

        for (occupancy, walks_through) in [
            (Occupancy::Empty, true),
            (Occupancy::Mixed, true),
            (Occupancy::Full, false),
            (Occupancy::Unpublished, false),
        ] {
            let mut occupied = air_around(C);
            occupied.insert(ahead, occupancy);

            let plan = plan_with(&config, &[camera(C, Vec3::NEG_Z)], &occupied, vec![]);

            assert_eq!(
                spawned(&plan).contains(&behind_it),
                walks_through,
                "the chunk behind a {occupancy:?} chunk"
            );
        }
    }

    #[test]
    fn rays_spawn_two_chunks_past_the_published_ones_and_one_past_a_solid_one() {
        let config = TestWorld {
            spawning_distance: 6,
            spawn_in_view: true,
            ..default()
        };
        let ahead = |distance: i32| C + IVec3::new(0, 0, -distance);

        // What is two chunks ahead, and how far ahead the farthest spawn is.
        for (occupancy, farthest) in [
            (Occupancy::Missing, 3),
            (Occupancy::Empty, 4),
            (Occupancy::Mixed, 3),
        ] {
            let mut occupied = air_around(C);
            if occupancy != Occupancy::Missing {
                occupied.insert(ahead(2), occupancy);
            }

            let plan = plan_with(&config, &[camera(C, Vec3::NEG_Z)], &occupied, vec![]);

            let spawned = spawned(&plan);
            assert!(
                spawned.contains(&ahead(farthest)),
                "{occupancy:?} two chunks ahead: {farthest} ahead did not spawn"
            );
            assert!(
                !spawned.contains(&ahead(farthest + 1)),
                "{occupancy:?} two chunks ahead: {} ahead spawned",
                farthest + 1
            );
        }
    }

    #[test]
    fn more_rays_are_cast_while_more_chunks_are_wanted() {
        let config = TestWorld {
            spawning_distance: 8,
            spawn_in_view: true,
            spawning_rays: 4,
            ..default()
        };
        let air: HashMap<_, _> = within(C, 5)
            .into_iter()
            .map(|position| (position, Occupancy::Empty))
            .collect();
        let camera = camera(C, Vec3::NEG_Z);

        let unwanted = plan_wanting(&config, &[camera], &air, vec![], 0);
        let wanted = plan_wanting(&config, &[camera], &air, vec![], 1000);

        assert!(
            spawned(&unwanted).len() <= 4 * RAY_LOOKAHEAD,
            "{} rays found {} chunks",
            config.spawning_rays,
            unwanted.spawn.len()
        );
        assert!(
            spawned(&wanted).len() > 2 * unwanted.spawn.len(),
            "wanting more found {} chunks, not wanting {}",
            wanted.spawn.len(),
            unwanted.spawn.len()
        );
        assert!(
            spawned(&wanted).len() <= 4 * MAX_RAYS_PER_SPAWNING_RAY * RAY_LOOKAHEAD,
            "{} chunks found with at most {} rays",
            wanted.spawn.len(),
            4 * MAX_RAYS_PER_SPAWNING_RAY
        );
    }

    #[test]
    fn spawns_are_listed_nearest_a_camera_first() {
        let a = camera(C, Vec3::NEG_Z);
        let b = camera(C + IVec3::new(20, 0, 0), Vec3::NEG_Z);

        let plan = plan_with(&TestWorld::default(), &[a, b], &HashMap::new(), vec![]);

        let distances: Vec<f32> = plan
            .spawn
            .iter()
            .map(|(position, _)| {
                let centre = chunk_centre(*position);
                a.position.distance(centre).min(b.position.distance(centre))
            })
            .collect();
        assert!(distances.len() > 2);
        assert!(
            distances.windows(2).all(|pair| pair[0] <= pair[1]),
            "{distances:?}"
        );
    }
}
