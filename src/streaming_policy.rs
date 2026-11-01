///
/// Streaming policy
/// Decides, from the cameras and the chunks that exist, which chunks to spawn
/// and which to retire, and at which LOD level. It reads no Bevy world state, so
/// it can be tested with plain values.
///
use std::collections::VecDeque;

use bevy::{math::Affine3A, platform::collections::HashSet, prelude::*};
use rand::{Rng, RngExt};

use crate::{
    chunk::{chunk_containing, ChunkData, ChunkKey, CHUNK_SIZE_I, MAX_LOD_LEVEL},
    configuration::{
        ChunkDespawnStrategy, ChunkSpawnStrategy, FillHint, LodLevel, VoxelWorldConfig,
    },
    voxel::WorldVoxel,
};

/// What the streaming policy needs to know about one camera.
///
/// The view and projection matrices are kept apart, as in Bevy's camera, so
/// that points far from the origin keep their precision.
#[derive(Clone, Copy, Debug)]
pub(crate) struct CameraSnapshot {
    position: Vec3,
    /// The level-0 chunk the camera is in. Computed once, since every chunk's
    /// distance to the camera is measured from it.
    chunk_position: IVec3,
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
        let position: Vec3 = world_from_view.translation.into();
        Self {
            position,
            chunk_position: chunk_containing(position.floor().as_ivec3()),
            world_from_view,
            view_from_world: world_from_view.inverse(),
            clip_from_view,
            view_from_clip: clip_from_view.inverse(),
            viewport_size,
        }
    }

    /// The ray from the near plane through a point of the viewport.
    ///
    /// `point` is in physical pixels from the viewport's top left corner.
    /// Returns `None` while the camera has no render target.
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
    /// corners is in view. `ndc_margin` widens the view on every side, in NDC
    /// units.
    fn sees_chunk(&self, key: ChunkKey, ndc_margin: f32) -> bool {
        let chunk_min = key.world_position();
        let chunk_max = chunk_min + Vec3::splat(key.voxel_extent() as f32);

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

/// What the streaming policy needs to know about a chunk key.
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

impl Occupancy {
    /// Whether a chunk exists here with its voxels published.
    fn is_published(self) -> bool {
        matches!(self, Occupancy::Empty | Occupancy::Mixed | Occupancy::Full)
    }

    /// The occupancy of a chunk with published `data`.
    ///
    /// Every solid voxel counts as opaque, so a ray sees nothing behind a full
    /// chunk. Supporting translucent voxels means changing this and
    /// [`Occupancy::of_uniform`] together.
    pub fn of_data<I>(data: &ChunkData<I>) -> Self {
        if data.is_full {
            Occupancy::Full
        } else if data.is_empty {
            Occupancy::Empty
        } else {
            Occupancy::Mixed
        }
    }

    /// The occupancy a chunk filled with `voxel` throughout will have once
    /// published, matching [`Occupancy::of_data`].
    pub fn of_uniform<I: PartialEq>(voxel: WorldVoxel<I>) -> Self {
        if voxel.is_solid() {
            Occupancy::Full
        } else {
            Occupancy::Empty
        }
    }
}

/// What the streaming policy needs to know about the chunks that exist and the
/// voxel edits.
pub(crate) trait ChunkLookup {
    /// The finest chunk covering the level-0 chunk at `chunk_position`, and its
    /// occupancy. Returns `None` when no chunk covers it.
    fn covering(&self, chunk_position: IVec3) -> Option<(ChunkKey, Occupancy)>;

    /// The occupancy of the chunk at `key`.
    fn occupancy(&self, key: ChunkKey) -> Occupancy;

    /// The coarsest LOD level of any chunk, or 0 when no chunks exist.
    fn coarsest_level(&self) -> LodLevel;

    /// Whether a voxel edit lies in the extent of the chunk at `key`, or in the
    /// level-0 chunks its padding samples.
    fn edits_reach(&self, key: ChunkKey) -> bool;
}

/// The most chunks that have not been spawned yet that one spawning ray queues
/// before it stops.
///
/// Two covers the next chunk along the ray, and one behind it in case the first
/// turns out to be air. A larger value spawns more chunks hidden behind solid
/// ones, wherever a ray reaches solid voxels before their chunk is published.
const RAY_LOOKAHEAD: usize = 2;

/// The most rays cast in one frame, as a multiple of `spawning_rays`.
///
/// Rays beyond `spawning_rays` are cast only while the chunk lifecycle could
/// start more chunks than the rays have found.
const MAX_RAYS_PER_SPAWNING_RAY: usize = 16;

/// How far an extra ray may land from an earlier ray that found chunks, as a
/// fraction of the viewport size.
///
/// At the far end of the view a chunk covers around a hundredth of the
/// viewport, so this is a few chunks either way.
const RAY_JITTER: f32 = 1.0 / 16.0;

/// The margin past the spawning distance, squared and in chunks, before a chunk
/// is retired.
///
/// It leaves a thin band where existing chunks are kept but none are spawned.
const RETIRE_MARGIN_SQUARED: i32 = 1;

/// The chunks to spawn and retire this frame.
#[derive(Debug, Default)]
pub(crate) struct StreamingPlan {
    pub spawn: Vec<ChunkKey>,
    pub retire: Vec<ChunkKey>,
}

/// Plans this frame's spawns and retirements.
///
/// `chunks` are the chunks that exist, and `lookup` gives the occupancy of any
/// chunk key. `wanted` is how many chunks the chunk lifecycle could start right
/// away beyond those already Pending. `rng` picks where to cast spawning rays.
pub(crate) fn plan<C: VoxelWorldConfig>(
    config: &C,
    cameras: &[CameraSnapshot],
    chunks: impl IntoIterator<Item = ChunkKey>,
    lookup: &impl ChunkLookup,
    wanted: usize,
    rng: &mut impl Rng,
) -> StreamingPlan {
    let mut plan = StreamingPlan::default();
    if cameras.is_empty() {
        return plan;
    }
    let planner = Planner::new(config, cameras, lookup);

    plan.spawn = planner.chunks_to_spawn(wanted, rng);
    if planner.coarsest == 0 && lookup.coarsest_level() == 0 {
        // Every chunk is at level 0 and level 0 is wanted everywhere, so only
        // range and view retire chunks, and the per-chunk LOD checks are skipped.
        for key in chunks {
            if planner.out_of_range_or_view(key) {
                plan.retire.push(key);
            }
        }
    } else {
        for key in chunks {
            if planner.retires(key) {
                plan.retire.push(key);
            }
        }
    }

    plan
}

/// One frame's view of the configuration, the cameras and the chunks that
/// exist, and the rules that follow from them.
struct Planner<'a, C, L> {
    config: &'a C,
    cameras: &'a [CameraSnapshot],
    lookup: &'a L,
    spawning_distance: i32,
    spawning_distance_squared: i32,
    /// The protected radius, squared.
    protected_distance_squared: i32,
    /// Whether chunks out of view are spawned only inside the protected radius.
    /// True with `CloseAndInView` spawning, and also with `FarAwayOrOutOfView`
    /// despawning, since a chunk spawned out of view would retire right away.
    in_view_only: bool,
    /// The coarsest LOD level the policy plans, which is the level `chunk_lod`
    /// wants at the spawning distance.
    coarsest: LodLevel,
}

impl<'a, C: VoxelWorldConfig, L: ChunkLookup> Planner<'a, C, L> {
    fn new(config: &'a C, cameras: &'a [CameraSnapshot], lookup: &'a L) -> Self {
        let spawning_distance = config.spawning_distance() as i32;
        let in_view_only = config.chunk_spawn_strategy()
            == ChunkSpawnStrategy::CloseAndInView
            || config.chunk_despawn_strategy()
                == ChunkDespawnStrategy::FarAwayOrOutOfView;

        let mut coarsest = 0;
        for camera in cameras {
            for axis in [
                IVec3::X,
                IVec3::Y,
                IVec3::Z,
                IVec3::NEG_X,
                IVec3::NEG_Y,
                IVec3::NEG_Z,
            ] {
                let far_chunk = camera.chunk_position + axis * spawning_distance;
                coarsest =
                    coarsest.max(config.chunk_lod(far_chunk, None, camera.position));
            }
        }

        Self {
            config,
            cameras,
            lookup,
            spawning_distance,
            spawning_distance_squared: spawning_distance.pow(2),
            protected_distance_squared: (config.min_despawn_distance() as i32).pow(2),
            in_view_only,
            coarsest: coarsest.min(MAX_LOD_LEVEL),
        }
    }

    // --- LOD levels ---------------------------------------------------------
    //
    // The LOD level at a position is chosen from the top down. Starting from
    // the coarsest chunk containing the position, a chunk is split while
    // `chunk_lod` wants a finer level for it than it has.
    //
    // `chunk_lod` never wants finer detail farther away. So the chunk wanted at
    // a position is the coarsest one, up to `coarsest`, for which `chunk_lod`
    // wants its own level or coarser. That holds for a chunk but not for its
    // parent exactly when the chunk is the wanted one.

    /// The level `chunk_lod` wants for the chunk.
    ///
    /// `chunk_lod` is asked about the chunk's level-0 chunk nearest the nearest
    /// camera. `previous` is the level of the chunk covering that position now,
    /// passed on as `chunk_lod`'s `previous_lod`.
    fn asked(&self, key: ChunkKey, previous: Option<LodLevel>) -> LodLevel {
        let (camera, nearest) = self
            .cameras
            .iter()
            .map(|camera| (camera, nearest_chunk(key, camera.chunk_position)))
            .min_by_key(|(camera, nearest)| {
                nearest.distance_squared(camera.chunk_position)
            })
            .expect("the policy plans with at least one camera");
        self.config.chunk_lod(nearest, previous, camera.position)
    }

    /// Whether the chunk is the one wanted at its position.
    fn wants(&self, key: ChunkKey, previous: Option<LodLevel>) -> bool {
        if key.lod_level > self.coarsest {
            return false;
        }
        if key.lod_level > 0 && self.asked(key, previous) < key.lod_level {
            return false;
        }
        key.lod_level == self.coarsest || {
            let parent = key.parent();
            self.asked(parent, previous) < parent.lod_level
        }
    }

    /// The key of the chunk wanted at a level-0 chunk position. `previous` is
    /// the level of the chunk covering the position now.
    fn wanted_key(&self, chunk_position: IVec3, previous: Option<LodLevel>) -> ChunkKey {
        let mut lod_level = 0;
        while lod_level < self.coarsest {
            let above = ChunkKey::containing_chunk(chunk_position, lod_level + 1);
            if self.asked(above, previous) < above.lod_level {
                break;
            }
            lod_level += 1;
        }
        ChunkKey::containing_chunk(chunk_position, lod_level)
    }

    /// The key of the chunk wanted at a level-0 chunk position, given what
    /// covers the position now.
    fn wanted_key_at(&self, chunk_position: IVec3) -> ChunkKey {
        if self.coarsest == 0 {
            return ChunkKey::new(chunk_position, 0);
        }
        let previous = self
            .lookup
            .covering(chunk_position)
            .map(|(key, _)| key.lod_level);
        self.wanted_key(chunk_position, previous)
    }

    // --- Fill hints ---------------------------------------------------------

    /// The occupancy a missing chunk will have once published, according to
    /// its fill hint.
    ///
    /// Returns `None` when the world gives no hint, or when a voxel edit
    /// reaches the chunk, since the hint doesn't know about voxel edits.
    fn hinted(&self, key: ChunkKey) -> Option<Occupancy> {
        match self.config.chunk_fill_hint(key) {
            FillHint::Unknown => None,
            FillHint::Uniform(voxel) => {
                (!self.lookup.edits_reach(key)).then(|| Occupancy::of_uniform(voxel))
            }
        }
    }

    /// Whether a missing chunk's fill hint is used instead of spawning it.
    ///
    /// Such a chunk is never spawned, and covers its extent just as a published
    /// chunk would. This holds when the chunk is [`hinted`](Self::hinted) and
    /// outside every camera's protected radius. Chunks inside the protected
    /// radius spawn whatever their fill hint.
    fn hint_stands_for(&self, key: ChunkKey) -> bool {
        self.hinted(key).is_some()
            && !self.is_close_to_any_camera(key, self.protected_distance_squared)
    }

    // --- Range and view -----------------------------------------------------

    /// Whether the chunk's nearest point is within `distance_squared` chunks of
    /// a camera.
    fn is_close_to_any_camera(&self, key: ChunkKey, distance_squared: i32) -> bool {
        self.cameras.iter().any(|camera| {
            nearest_chunk(key, camera.chunk_position)
                .distance_squared(camera.chunk_position)
                <= distance_squared
        })
    }

    /// Whether no camera sees the chunk and it is outside every camera's
    /// protected radius.
    fn out_of_view_and_unprotected(&self, key: ChunkKey) -> bool {
        !self.is_close_to_any_camera(key, self.protected_distance_squared)
            && !self
                .cameras
                .iter()
                .any(|camera| camera.sees_chunk(key, 0.0))
    }

    /// Whether the policy wants no chunk at the key. True when it is beyond the
    /// spawning distance, or out of view while `in_view_only` is set.
    fn unwanted_region(&self, key: ChunkKey) -> bool {
        !self.is_close_to_any_camera(key, self.spawning_distance_squared)
            || (self.in_view_only && self.out_of_view_and_unprotected(key))
    }

    // --- Retiring -----------------------------------------------------------

    /// Whether an existing chunk retires this frame.
    ///
    /// It retires when it is out of range or view. It also retires when a
    /// different level is wanted at its position, but only once chunks at that
    /// level cover its extent, so that no hole opens where it was.
    fn retires(&self, key: ChunkKey) -> bool {
        self.out_of_range_or_view(key)
            || (!self.wants(key, Some(key.lod_level)) && self.covered(key))
    }

    /// Whether an existing chunk is beyond the spawning distance plus
    /// `RETIRE_MARGIN_SQUARED`, or out of view and unprotected under
    /// `FarAwayOrOutOfView` despawning.
    fn out_of_range_or_view(&self, key: ChunkKey) -> bool {
        if !self.is_close_to_any_camera(
            key,
            self.spawning_distance_squared + RETIRE_MARGIN_SQUARED,
        ) {
            return true;
        }
        self.config.chunk_despawn_strategy() == ChunkDespawnStrategy::FarAwayOrOutOfView
            && self.out_of_view_and_unprotected(key)
    }

    /// Whether other chunks cover the chunk's extent, so it can retire.
    ///
    /// The extent is covered by a published chunk above it, or by published
    /// chunks below it wherever the policy wants a chunk at all. A missing
    /// chunk whose fill hint is used instead of spawning it counts as
    /// published.
    fn covered(&self, key: ChunkKey) -> bool {
        for lod_level in key.lod_level + 1..=self.coarsest {
            let above = ChunkKey::containing_chunk(key.position, lod_level);
            if self.published_or_hinted(above) {
                return true;
            }
        }
        key.lod_level > 0
            && key
                .children()
                .all(|child| self.covered_below(child, key.lod_level))
    }

    /// Whether the region of the chunk at `key` is covered by published or
    /// hinted chunks at its level or finer, or no chunk is wanted there.
    /// `retiring` is the level of the chunk about to retire, which still covers
    /// the region.
    fn covered_below(&self, key: ChunkKey, retiring: LodLevel) -> bool {
        if self.published_or_hinted(key) || self.unwanted_region(key) {
            return true;
        }
        if key.lod_level == 0 || self.asked(key, Some(retiring)) >= key.lod_level {
            return false;
        }
        key.children()
            .all(|child| self.covered_below(child, retiring))
    }

    /// Whether the chunk at `key` is published, or missing with a fill hint used
    /// in its place. A chunk that is spawned but not yet published is waited
    /// for, even when it has a fill hint.
    fn published_or_hinted(&self, key: ChunkKey) -> bool {
        match self.lookup.occupancy(key) {
            Occupancy::Missing => self.hint_stands_for(key),
            occupancy => occupancy.is_published(),
        }
    }

    // --- Spawning -----------------------------------------------------------

    fn chunks_to_spawn(&self, wanted: usize, rng: &mut impl Rng) -> Vec<ChunkKey> {
        let config = self.config;
        let cameras = self.cameras;
        let spawn_strategy = config.chunk_spawn_strategy();
        let camera_positions: Vec<CameraPosition> =
            cameras.iter().map(CameraPosition::from).collect();

        let mut spawn = Vec::new();
        let mut visited = HashSet::new();
        // Every chunk queued so far, so a ray can tell whether it found a chunk
        // that nothing else had queued.
        let mut queued = HashSet::new();
        let mut chunks_deque =
            VecDeque::with_capacity(config.spawning_rays() * RAY_LOOKAHEAD);

        // The cube of chunks within the protected radius of each camera is
        // queued first, nearest chunks first, so they spawn before any other.
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
                let key = self.wanted_key_at(camera.chunk_position + offset);
                if queued.insert(key) {
                    chunks_deque.push_back(key);
                }
            }
        }

        // Each frame, `spawning_rays` rays are cast through random points of the
        // viewports. While the chunk lifecycle could start more chunks than the
        // rays have found, extra rays are cast, up to `MAX_RAYS_PER_SPAWNING_RAY`
        // times as many, until `spawning_rays` rays in a row find nothing. A ray
        // that stops at an unpublished chunk finds nothing, so once every ray
        // does, the chunks found so far must publish before more can be found.
        //
        // Missing chunks tend to lie in patches of the view, such as the sky
        // above a distant hill, so every other extra ray is cast near a point
        // where an earlier ray found chunks this frame.
        let base_rays = config.spawning_rays();
        let max_rays = base_rays.saturating_mul(MAX_RAYS_PER_SPAWNING_RAY);
        let m = config.spawning_ray_margin();
        let mut rays = 0;
        let mut found = 0;
        let mut misses = 0;
        let mut hits: Vec<(usize, Vec2)> = Vec::new();
        while rays < base_rays
            || (found < wanted && misses < base_rays && rays < max_rays)
        {
            rays += 1;
            let (camera_index, point) = if rays > base_rays
                && !hits.is_empty()
                && rng.random()
            {
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

            let found_by_ray = self.queue_chunks_intersecting_ray_from_point(
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
        while let Some(key) = chunks_deque.pop_front() {
            if spawn.len() >= config.max_spawn_per_frame() {
                break;
            }
            if !visited.insert(key) {
                continue;
            }

            if !self.is_close_to_any_camera(key, self.spawning_distance_squared) {
                continue;
            }

            if self.in_view_only && self.out_of_view_and_unprotected(key) {
                continue;
            }

            if self.lookup.occupancy(key) != Occupancy::Missing {
                continue;
            }
            if !self.hint_stands_for(key) {
                spawn.push((
                    StartOrder::of(key, &camera_positions, config.min_despawn_distance()),
                    key,
                ));
            }

            if spawn_strategy != ChunkSpawnStrategy::Close {
                continue;
            }

            // If we get here, we queue the neighbors, at whatever level is
            // wanted where they are. The flood fill also continues past chunks
            // whose fill hint is used instead of spawning them.
            let extent = key.extent();
            for x in -1..=1 {
                for y in -1..=1 {
                    for z in -1..=1 {
                        let offset = IVec3::new(x, y, z);
                        if offset == IVec3::ZERO {
                            continue;
                        }
                        chunks_deque.push_back(
                            self.wanted_key_at(key.position + offset * extent),
                        );
                    }
                }
            }
        }

        // In start order, which is the order the chunks spawn in.
        spawn.sort_by_key(|&(order, _)| order);
        spawn.into_iter().map(|(_, key)| key).collect()
    }

    /// Shoots a ray from the given point and queues yet unpublished chunks.
    /// Returns how many were added to the queue.
    ///
    /// Each step takes the ray into the next chunk along its path. Far from the
    /// camera, where `chunk_lod` asks for large chunks, the steps are large too,
    /// so the ray crosses open air in a few steps.
    ///
    /// The ray passes through chunks that contain air. It stops at a chunk that
    /// is completely solid, because nothing behind it can be seen. It also stops
    /// at a chunk that is spawned but not yet published, because what lies
    /// behind it is unknown until it has generated. As a result, chunks spawn
    /// from the camera outwards as the chunks in front of them publish, and
    /// chunks hidden behind solid ones are rarely spawned.
    ///
    /// When the ray reaches a chunk that has not been spawned yet, it queues it
    /// and continues. It queues at most `RAY_LOOKAHEAD` such chunks, then stops.
    ///
    /// A chunk with a fill hint is never queued. The ray passes through a chunk
    /// hinted empty and stops at a chunk hinted solid, just as if the chunk
    /// existed. A world that hints the air above its terrain therefore lets the
    /// ray reach the terrain in one go.
    fn queue_chunks_intersecting_ray_from_point(
        &self,
        camera: &CameraSnapshot,
        point: Vec2,
        queue: &mut VecDeque<ChunkKey>,
        queued: &mut HashSet<ChunkKey>,
    ) -> usize {
        let Some(ray) = camera.viewport_ray(point) else {
            return 0;
        };
        let mut found = 0;
        let mut lookahead = RAY_LOOKAHEAD;
        let mut current = ray.origin;
        let mut t = 0.0;
        while t < (self.spawning_distance * CHUNK_SIZE_I) as f32 {
            let chunk_pos = chunk_containing(current.floor().as_ivec3());
            // The chunk wanted here and its occupancy. `behind` is the
            // occupancy of the chunk covering this position now, if any. While
            // the wanted chunk is missing, it decides whether the ray goes on.
            let (key, mut occupancy, behind) = if self.coarsest == 0 {
                // Only level 0 is planned, so the wanted chunk is the level-0
                // chunk here.
                let key = ChunkKey::new(chunk_pos, 0);
                (key, self.lookup.occupancy(key), Occupancy::Empty)
            } else {
                let covering = self.lookup.covering(chunk_pos);
                let key =
                    self.wanted_key(chunk_pos, covering.map(|(key, _)| key.lod_level));
                let occupancy = match covering {
                    Some((covering, occupancy)) if covering == key => occupancy,
                    // Wanted coarser than the finest chunk here: it may exist.
                    Some((covering, _)) if key.lod_level > covering.lod_level => {
                        self.lookup.occupancy(key)
                    }
                    _ => Occupancy::Missing,
                };
                let behind =
                    covering.map_or(Occupancy::Empty, |(_, occupancy)| occupancy);
                (key, occupancy, behind)
            };
            if occupancy == Occupancy::Missing {
                match self.hinted(key) {
                    Some(hinted) => occupancy = hinted,
                    None => {
                        if queued.insert(key) {
                            queue.push_back(key);
                            found += 1;
                        }
                        lookahead -= 1;
                        if lookahead == 0 {
                            break;
                        }
                        occupancy = behind;
                    }
                }
            }
            match occupancy {
                Occupancy::Missing | Occupancy::Empty => {}
                // Past a chunk with solid voxels, the next one is likely solid.
                Occupancy::Mixed => lookahead = lookahead.min(1),
                Occupancy::Full | Occupancy::Unpublished => break,
            }
            t += key.voxel_extent() as f32;
            current = ray.origin + ray.direction * t;
        }
        found
    }
}

/// The level-0 chunk of the chunk at `key` nearest to `chunk_position`.
fn nearest_chunk(key: ChunkKey, chunk_position: IVec3) -> IVec3 {
    // A level-0 chunk is its own nearest chunk. This runs for every existing
    // chunk every frame, so it skips the clamp there.
    if key.lod_level == 0 {
        return key.position;
    }
    chunk_position.clamp(key.position, key.position + (key.extent() - 1))
}

// --- Start order ------------------------------------------------------------

/// A camera's position, as [`StartOrder`] measures distances from it.
#[derive(Clone, Copy, Debug)]
pub(crate) struct CameraPosition {
    position: Vec3,
    /// The level-0 chunk the camera is in.
    chunk_position: IVec3,
}

impl CameraPosition {
    #[inline]
    pub fn new(position: Vec3) -> Self {
        Self {
            position,
            chunk_position: chunk_containing(position.floor().as_ivec3()),
        }
    }
}

impl From<&CameraSnapshot> for CameraPosition {
    fn from(camera: &CameraSnapshot) -> Self {
        Self {
            position: camera.position,
            chunk_position: camera.chunk_position,
        }
    }
}

/// A chunk's place in the order chunks spawn and their chunk tasks start in.
///
/// Chunks within a camera's protected radius come first, nearest first. The
/// rest follow by their size on screen, largest first. Size on screen goes as
/// extent over distance, so a chunk at level `L` ranks the same as a chunk at
/// level `L - 1` half as far away. In a world with several LOD levels, far
/// coarse chunks therefore fill in alongside near fine ones, not after them.
///
/// It is a single integer, because every Pending chunk is ordered every frame.
/// The value is the squared distance from the nearest camera to the chunk's
/// centre, divided by the chunk's squared extent. That non-negative `f32` is
/// stored as its bits, which sort in the same order as the float. The sign bit
/// is set for chunks outside the protected radius, so they sort last.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct StartOrder(u32);

impl StartOrder {
    /// The start order of the chunk at `key`. `protected_distance` is the
    /// protected radius in chunks, as `min_despawn_distance` gives it.
    #[inline]
    pub fn of(
        key: ChunkKey,
        cameras: &[CameraPosition],
        protected_distance: u32,
    ) -> Self {
        let centre = key.centre();
        let extent = key.voxel_extent() as f32;
        let distance_squared = cameras
            .iter()
            .map(|camera| camera.position.distance_squared(centre))
            .fold(f32::INFINITY, f32::min);
        let distance_squared_in_extents = distance_squared / (extent * extent);
        debug_assert!(distance_squared_in_extents.is_sign_positive());

        // A chunk inside the protected radius has a level-0 chunk within that
        // radius of the camera's, so its centre is at most half its diagonal
        // plus the radius away, in extents. That cheap check runs first, and
        // only chunks passing it take the exact test.
        let protected_distance_squared = (protected_distance as i32).pow(2);
        let near_enough = (HALF_DIAGONAL + protected_distance as f32).powi(2);
        let protected = distance_squared_in_extents <= near_enough
            && cameras.iter().any(|camera| {
                nearest_chunk(key, camera.chunk_position)
                    .distance_squared(camera.chunk_position)
                    <= protected_distance_squared
            });
        Self(distance_squared_in_extents.to_bits() | ((!protected as u32) << 31))
    }
}

/// Half the diagonal of a cube of side 1 (`√3 / 2`, rounded up).
const HALF_DIAGONAL: f32 = 0.87;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk::CHUNK_SIZE_F;
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
        /// The LOD level of chunks nearer than each distance, in chunks, and of
        /// those farther than every distance.
        lod_bands: Vec<(i32, LodLevel)>,
        far_lod: LodLevel,
        /// Whether `chunk_lod` keeps the level a position has.
        sticky_lod: bool,
        /// The fill hints of particular chunks, and of every other chunk.
        hinted: HashMap<ChunkKey, FillHint<u8>>,
        hint_elsewhere: FillHint<u8>,
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
                lod_bands: Vec::new(),
                far_lod: 0,
                sticky_lod: false,
                hinted: HashMap::new(),
                hint_elsewhere: FillHint::Unknown,
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

        fn chunk_lod(
            &self,
            chunk_position: IVec3,
            previous_lod: Option<LodLevel>,
            camera_position: Vec3,
        ) -> LodLevel {
            if self.sticky_lod {
                if let Some(previous) = previous_lod {
                    return previous;
                }
            }
            let camera_chunk = chunk_containing(camera_position.floor().as_ivec3());
            let distance_squared = chunk_position.distance_squared(camera_chunk);
            self.lod_bands
                .iter()
                .find(|(within, _)| distance_squared < within * within)
                .map_or(self.far_lod, |(_, lod_level)| *lod_level)
        }

        fn chunk_fill_hint(&self, key: ChunkKey) -> FillHint<u8> {
            self.hinted
                .get(&key)
                .copied()
                .unwrap_or(self.hint_elsewhere)
        }
    }

    /// Far enough from the origin that no test position is negative, and the
    /// corner of a level-1 chunk.
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

    /// The key of the level-0 chunk at `position`.
    fn existing(position: IVec3) -> ChunkKey {
        ChunkKey::new(position, 0)
    }

    /// The chunks that exist, with what is at each.
    type Occupied = HashMap<ChunkKey, Occupancy>;

    impl ChunkLookup for Occupied {
        fn covering(&self, chunk_position: IVec3) -> Option<(ChunkKey, Occupancy)> {
            (0..=MAX_LOD_LEVEL).find_map(|lod_level| {
                let key = ChunkKey::containing_chunk(chunk_position, lod_level);
                self.get(&key).map(|occupancy| (key, *occupancy))
            })
        }

        fn occupancy(&self, key: ChunkKey) -> Occupancy {
            self.get(&key).copied().unwrap_or(Occupancy::Missing)
        }

        fn coarsest_level(&self) -> LodLevel {
            self.keys().map(|key| key.lod_level).max().unwrap_or(0)
        }

        fn edits_reach(&self, _key: ChunkKey) -> bool {
            false
        }
    }

    /// The chunks that exist, and the chunks that voxel edits reach.
    struct WithEdits<'a> {
        occupied: &'a Occupied,
        edited: Vec<ChunkKey>,
    }

    impl ChunkLookup for WithEdits<'_> {
        fn covering(&self, chunk_position: IVec3) -> Option<(ChunkKey, Occupancy)> {
            self.occupied.covering(chunk_position)
        }

        fn occupancy(&self, key: ChunkKey) -> Occupancy {
            self.occupied.occupancy(key)
        }

        fn coarsest_level(&self) -> LodLevel {
            self.occupied.coarsest_level()
        }

        fn edits_reach(&self, key: ChunkKey) -> bool {
            self.edited.contains(&key)
        }
    }

    /// Level-0 chunks at `positions`, all with `occupancy`.
    fn level_0(
        positions: impl IntoIterator<Item = IVec3>,
        occupancy: Occupancy,
    ) -> Occupied {
        positions
            .into_iter()
            .map(|position| (existing(position), occupancy))
            .collect()
    }

    /// Plans with `occupied` as the only chunks that exist, with no chunks
    /// wanted beyond what `spawning_rays` rays find.
    fn plan_with(
        config: &TestWorld,
        cameras: &[CameraSnapshot],
        occupied: &Occupied,
    ) -> StreamingPlan {
        plan_wanting(config, cameras, occupied, 0)
    }

    fn plan_wanting(
        config: &TestWorld,
        cameras: &[CameraSnapshot],
        occupied: &Occupied,
        wanted: usize,
    ) -> StreamingPlan {
        plan(
            config,
            cameras,
            occupied.keys().copied(),
            occupied,
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

    /// The positions a plan spawns, checking none is spawned twice and all are
    /// at level 0.
    fn spawned(plan: &StreamingPlan) -> HashSet<IVec3> {
        let positions: HashSet<_> = plan.spawn.iter().map(|key| key.position).collect();
        assert_eq!(
            positions.len(),
            plan.spawn.len(),
            "a chunk was spawned twice"
        );
        assert!(
            plan.spawn.iter().all(|key| key.lod_level == 0),
            "a chunk above level 0 spawned: {:?}",
            plan.spawn
        );
        positions
    }

    #[test]
    fn nothing_is_planned_without_cameras() {
        let plan = plan_with(
            &TestWorld::default(),
            &[],
            &level_0([C], Occupancy::Unpublished),
        );

        assert!(plan.spawn.is_empty());
        assert!(plan.retire.is_empty());
    }

    #[test]
    fn the_flood_fill_spawns_every_chunk_within_spawning_distance() {
        let plan = plan_with(
            &TestWorld::default(),
            &[camera(C, Vec3::NEG_Z)],
            &Occupied::new(),
        );

        assert_eq!(spawned(&plan), within(C, 3));
    }

    #[test]
    fn chunks_that_exist_are_not_spawned_again() {
        let plan = plan_with(
            &TestWorld::default(),
            &[camera(C, Vec3::NEG_Z)],
            &level_0(within(C, 1), Occupancy::Empty),
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

        let plan = plan_with(&config, &[camera(C, Vec3::NEG_Z)], &Occupied::new());

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

        // Once the chunks around the camera are published, rays pass through them.
        let plan = plan_with(&config, &[camera(C, Vec3::NEG_Z)], &air_around(C));

        assert!(
            spawned(&plan).contains(&(C + IVec3::new(0, 0, -2))),
            "a chunk ahead of the camera spawns"
        );
    }

    #[test]
    fn chunks_retire_beyond_spawning_distance_plus_a_margin() {
        let kept = C + IVec3::new(3, 1, 0);
        let retired = C + IVec3::new(3, 1, 1);

        let plan = plan_with(
            &TestWorld::default(),
            &[camera(C, Vec3::NEG_Z)],
            &level_0([C, kept, retired], Occupancy::Unpublished),
        );

        assert_eq!(plan.retire, vec![existing(retired)]);
    }

    /// Plans for a camera at `C` looking along -z, with chunks behind it just
    /// inside and just outside the protected radius, and one ahead of it.
    fn plan_with_chunks_behind_the_camera(config: &TestWorld) -> (StreamingPlan, IVec3) {
        let protected = C + IVec3::new(0, 0, 1);
        let unprotected = C + IVec3::new(0, 0, 2);
        let ahead = C + IVec3::new(0, 0, -2);

        let plan = plan_with(
            config,
            &[camera(C, Vec3::NEG_Z)],
            &level_0([C, protected, unprotected, ahead], Occupancy::Unpublished),
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

        assert_eq!(plan.retire, vec![existing(unprotected)]);
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
                lod_bands: vec![(rng.random_range(1..=3), 0)],
                far_lod: rng.random_range(0..=2),
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

            let first = plan_with(&config, &cameras, &Occupied::new());
            let occupied = first
                .spawn
                .iter()
                .map(|key| (*key, Occupancy::Unpublished))
                .collect();
            let next = plan_with(&config, &cameras, &occupied);

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

        let plan = plan_with(
            &config,
            &[camera(a, Vec3::NEG_Z), camera(b, Vec3::NEG_Z)],
            &level_0([near_b, between], Occupancy::Unpublished),
        );

        let mut expected: HashSet<_> =
            within(a, 2).union(&within(b, 2)).copied().collect();
        expected.remove(&near_b);
        assert_eq!(spawned(&plan), expected);
        assert_eq!(plan.retire, vec![existing(between)]);
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
            &Occupied::new(),
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

        let plan = plan_with(&config, &[camera(NEGATIVE, Vec3::NEG_Z)], &Occupied::new());

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
        );

        assert!(spawned(&plan).contains(&(NEGATIVE + IVec3::new(0, 0, -2))));
    }

    /// The chunks around a camera's chunk, published as air, so rays pass out
    /// through them.
    fn air_around(chunk: IVec3) -> Occupied {
        level_0(within(chunk, 1), Occupancy::Empty)
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
            occupied.insert(existing(ahead), occupancy);

            let plan = plan_with(&config, &[camera(C, Vec3::NEG_Z)], &occupied);

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
                occupied.insert(existing(ahead(2)), occupancy);
            }

            let plan = plan_with(&config, &[camera(C, Vec3::NEG_Z)], &occupied);

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

    // --- Fill hints ---------------------------------------------------------

    /// The chunk `distance` chunks ahead of a camera at `C` looking along -z.
    fn ahead(distance: i32) -> IVec3 {
        C + IVec3::new(0, 0, -distance)
    }

    /// Spawning in view to six chunks, with the chunk two ahead of the camera
    /// hinted to be `voxel` throughout.
    fn hinting_two_ahead(voxel: WorldVoxel) -> TestWorld {
        TestWorld {
            spawning_distance: 6,
            spawn_in_view: true,
            hinted: [(existing(ahead(2)), FillHint::Uniform(voxel))]
                .into_iter()
                .collect(),
            ..default()
        }
    }

    #[test]
    fn rays_walk_through_a_chunk_hinted_empty_and_stop_at_one_hinted_solid_spawning_neither(
    ) {
        for (voxel, walks_through) in [
            (WorldVoxel::Air, true),
            (WorldVoxel::Unset, true),
            (WorldVoxel::Solid(1), false),
        ] {
            let config = hinting_two_ahead(voxel);

            let plan = plan_with(&config, &[camera(C, Vec3::NEG_Z)], &air_around(C));

            let spawned = spawned(&plan);
            assert!(
                !spawned.contains(&ahead(2)),
                "the chunk hinted {voxel:?} spawned"
            );
            // The two chunks past the hinted one, as past a published one.
            for distance in [3, 4] {
                assert_eq!(
                    spawned.contains(&ahead(distance)),
                    walks_through,
                    "{distance} chunks ahead, past a chunk hinted {voxel:?}"
                );
            }
            assert!(
                !spawned.contains(&ahead(5)),
                "past a chunk hinted {voxel:?}"
            );
        }
    }

    #[test]
    fn a_chunk_a_voxel_edit_reaches_spawns_whatever_its_hint() {
        let config = hinting_two_ahead(WorldVoxel::Air);
        let occupied = air_around(C);
        let lookup = WithEdits {
            occupied: &occupied,
            edited: vec![existing(ahead(2))],
        };

        let plan = plan(
            &config,
            &[camera(C, Vec3::NEG_Z)],
            occupied.keys().copied(),
            &lookup,
            0,
            &mut StdRng::seed_from_u64(1),
        );

        assert!(spawned(&plan).contains(&ahead(2)));
    }

    #[test]
    fn where_every_chunk_is_hinted_only_the_protected_radius_spawns() {
        let config = TestWorld {
            hint_elsewhere: FillHint::Uniform(WorldVoxel::Air),
            ..default()
        };

        let plan = plan_with(&config, &[camera(C, Vec3::NEG_Z)], &Occupied::new());

        assert_eq!(spawned(&plan), within(C, 1));
    }

    #[test]
    fn the_flood_fill_continues_through_hinted_chunks_without_spawning_them() {
        // Behind the camera, where no ray goes, a wall of hinted chunks two
        // chunks back, which only the flood fill crosses.
        let wall: Vec<IVec3> = (-3..=3)
            .flat_map(|x| (-3..=3).map(move |y| C + IVec3::new(x, y, 2)))
            .collect();
        let config = TestWorld {
            hinted: wall
                .iter()
                .map(|&position| (existing(position), FillHint::Uniform(WorldVoxel::Air)))
                .collect(),
            ..default()
        };

        let plan = plan_with(&config, &[camera(C, Vec3::NEG_Z)], &Occupied::new());

        let spawned = spawned(&plan);
        assert!(
            spawned.contains(&(C + IVec3::new(0, 0, 3))),
            "behind the wall"
        );
        assert!(wall.iter().all(|position| !spawned.contains(position)));
    }

    #[test]
    fn a_coarse_chunk_retires_once_the_finer_chunks_replacing_it_are_published_or_hinted()
    {
        let config = TestWorld {
            hint_elsewhere: FillHint::Uniform(WorldVoxel::Air),
            ..two_levels()
        };
        let parent = parent_at_c();
        let protected: Vec<_> = parent
            .children()
            .filter(|child| child.position.distance_squared(C) <= 1)
            .collect();
        assert_eq!(protected.len(), 4);

        // The level-1 chunk is published; level 0 is wanted here. Of its
        // children only those in the protected radius spawn, and until they
        // publish the parent stays.
        let mut occupied: Occupied = [(parent, Occupancy::Empty)].into_iter().collect();
        let plan = plan_with(&config, &[camera(C, Vec3::NEG_Z)], &occupied);
        assert_eq!(spawned(&plan), within(C, 1));
        assert!(plan.retire.is_empty(), "retired {:?}", plan.retire);

        for child in &protected {
            occupied.insert(*child, Occupancy::Empty);
        }
        let plan = plan_with(&config, &[camera(C, Vec3::NEG_Z)], &occupied);
        assert!(
            parent.children().all(|child| !plan.spawn.contains(&child)),
            "spawned {:?}",
            plan.spawn
        );
        assert_eq!(plan.retire, vec![parent]);
    }

    #[test]
    fn chunks_are_merged_at_once_when_the_coarser_chunk_wanted_in_their_place_is_hinted()
    {
        let config = TestWorld {
            spawning_distance: 5,
            far_lod: 1,
            hint_elsewhere: FillHint::Uniform(WorldVoxel::Air),
            ..default()
        };
        // Outside the protected radius, so the parent isn't spawned for being
        // close to the camera.
        let parent = ChunkKey::new(C + IVec3::new(2, 0, 0), 1);
        let children: Vec<_> = parent.children().collect();

        let occupied: Occupied = children
            .iter()
            .map(|child| (*child, Occupancy::Mixed))
            .collect();
        let plan = plan_with(&config, &[camera(C, Vec3::NEG_Z)], &occupied);

        assert!(!plan.spawn.contains(&parent), "spawned {:?}", plan.spawn);
        let retired: HashSet<_> = plan.retire.iter().copied().collect();
        assert_eq!(retired, children.iter().copied().collect());
    }

    #[test]
    fn more_rays_are_cast_while_more_chunks_are_wanted() {
        let config = TestWorld {
            spawning_distance: 8,
            spawn_in_view: true,
            spawning_rays: 4,
            ..default()
        };
        let air = level_0(within(C, 5), Occupancy::Empty);
        let camera = camera(C, Vec3::NEG_Z);

        let unwanted = plan_wanting(&config, &[camera], &air, 0);
        let wanted = plan_wanting(&config, &[camera], &air, 1000);

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
    fn spawns_are_listed_in_the_protected_radius_first_then_nearest_a_camera_first() {
        let a = camera(C, Vec3::NEG_Z);
        let b = camera(C + IVec3::new(20, 0, 0), Vec3::NEG_Z);
        let config = TestWorld::default();

        let plan = plan_with(&config, &[a, b], &Occupied::new());

        let protected_distance_squared = (config.min_despawn_distance as i32).pow(2);
        let order: Vec<(bool, f32)> = plan
            .spawn
            .iter()
            .map(|key| {
                let unprotected = [a, b].iter().all(|camera| {
                    key.position.distance_squared(camera.chunk_position)
                        > protected_distance_squared
                });
                let centre = key.centre();
                let distance =
                    a.position.distance(centre).min(b.position.distance(centre));
                (unprotected, distance)
            })
            .collect();
        assert!(order.iter().filter(|(unprotected, _)| *unprotected).count() > 2);
        assert!(order.iter().filter(|(unprotected, _)| !unprotected).count() > 2);
        assert!(order.windows(2).all(|pair| pair[0] <= pair[1]), "{order:?}");
    }

    #[test]
    fn chunks_start_within_the_protected_radius_first_then_largest_on_screen() {
        let cameras = [CameraPosition::new((C.as_vec3() + 0.5) * CHUNK_SIZE_F)];
        let order = |key: ChunkKey| StartOrder::of(key, &cameras, 2);

        // Two chunks from the camera, at the edge of the protected radius.
        let protected = ChunkKey::new(C + IVec3::new(2, 0, 0), 0);
        // Level 5, 32 chunks on a side, 22 chunks away: a chunk 1.4 times its
        // extent away, which looks larger than the level-0 chunk 3 chunks away.
        let coarse = ChunkKey::containing_chunk(C + IVec3::new(32, 0, 0), 5);
        let fine = ChunkKey::new(C + IVec3::new(3, 0, 0), 0);
        // Level 1, 2 chunks on a side, 6.5 chunks away: 3.3 extents.
        let middling = ChunkKey::containing_chunk(C + IVec3::new(6, 0, 0), 1);
        let far = ChunkKey::new(C + IVec3::new(0, 0, -4), 0);

        let mut keys = vec![far, middling, fine, coarse, protected];
        keys.sort_by_key(|&key| order(key));
        assert_eq!(keys, vec![protected, coarse, fine, middling, far]);
    }

    // --- LOD levels ---------------------------------------------------------

    /// Level 0 nearer than 3 chunks, level 1 beyond.
    fn two_levels() -> TestWorld {
        TestWorld {
            spawning_distance: 5,
            lod_bands: vec![(3, 0)],
            far_lod: 1,
            ..default()
        }
    }

    /// The level-1 chunk around `C`, and its eight level-0 children.
    fn parent_at_c() -> ChunkKey {
        ChunkKey::containing_chunk(C, 1)
    }

    #[test]
    fn chunks_spawn_at_the_level_chunk_lod_wants_where_they_are() {
        let plan = plan_with(&two_levels(), &[camera(C, Vec3::NEG_Z)], &Occupied::new());

        assert!(plan.spawn.contains(&existing(C)), "level 0 at the camera");
        let far = C + IVec3::new(4, 0, 0);
        assert!(
            plan.spawn.contains(&ChunkKey::containing_chunk(far, 1)),
            "level 1 four chunks away: {:?}",
            plan.spawn
        );
        assert!(!plan.spawn.contains(&existing(far)));

        // The spawned chunks tile the space within the spawning distance.
        let mut covered: HashMap<IVec3, ChunkKey> = HashMap::new();
        for key in &plan.spawn {
            for x in 0..key.extent() {
                for y in 0..key.extent() {
                    for z in 0..key.extent() {
                        let position = key.position + IVec3::new(x, y, z);
                        if let Some(other) = covered.insert(position, *key) {
                            panic!("{key} and {other} overlap at {position}");
                        }
                    }
                }
            }
        }
        for position in within(C, 5) {
            assert!(covered.contains_key(&position), "{position} is not covered");
        }
    }

    #[test]
    fn range_is_measured_to_the_nearest_point_of_a_chunk() {
        let config = TestWorld {
            spawning_distance: 3,
            lod_bands: vec![(2, 0)],
            far_lod: 2,
            ..default()
        };
        // Four chunks wide; its nearest point is two chunks from the camera and
        // its centre four.
        let near_corner = ChunkKey::new(IVec3::new(12, 8, 8), 2);
        let far_corner = ChunkKey::new(IVec3::new(16, 8, 8), 2);

        let plan = plan_with(&config, &[camera(C, Vec3::NEG_Z)], &Occupied::new());
        assert!(
            plan.spawn.contains(&near_corner),
            "the chunk whose corner is in range spawns: {:?}",
            plan.spawn
        );
        assert!(!plan.spawn.contains(&far_corner));

        let occupied: Occupied = [near_corner, far_corner]
            .into_iter()
            .map(|key| (key, Occupancy::Empty))
            .collect();
        let plan = plan_with(&config, &[camera(C, Vec3::NEG_Z)], &occupied);
        assert_eq!(plan.retire, vec![far_corner]);
    }

    #[test]
    fn a_chunk_is_split_once_the_finer_chunks_wanted_in_its_place_are_published() {
        let config = two_levels();
        let parent = parent_at_c();
        let children: Vec<_> = parent.children().collect();

        // The level-1 chunk is published; level 0 is wanted here.
        let mut occupied: Occupied = [(parent, Occupancy::Empty)].into_iter().collect();
        let plan = plan_with(&config, &[camera(C, Vec3::NEG_Z)], &occupied);
        for child in &children {
            assert!(
                plan.spawn.contains(child),
                "{child} spawns: {:?}",
                plan.spawn
            );
        }
        assert!(plan.retire.is_empty(), "retired {:?}", plan.retire);

        // One child isn't published yet.
        for (i, child) in children.iter().enumerate() {
            let occupancy = if i == 3 {
                Occupancy::Unpublished
            } else {
                Occupancy::Empty
            };
            occupied.insert(*child, occupancy);
        }
        let plan = plan_with(&config, &[camera(C, Vec3::NEG_Z)], &occupied);
        assert!(
            children.iter().all(|child| !plan.spawn.contains(child)),
            "spawned {:?}",
            plan.spawn
        );
        assert!(plan.retire.is_empty(), "retired {:?}", plan.retire);

        // Every child is published.
        occupied.insert(children[3], Occupancy::Mixed);
        let plan = plan_with(&config, &[camera(C, Vec3::NEG_Z)], &occupied);
        assert_eq!(plan.retire, vec![parent]);
    }

    #[test]
    fn a_finer_chunk_the_policy_does_not_want_counts_as_covered() {
        let config = TestWorld {
            spawning_distance: 2,
            lod_bands: vec![(4, 0)],
            far_lod: 1,
            ..default()
        };
        // Level 0 is wanted everywhere in range; of this chunk's children, only
        // the one at its near corner is within the spawning distance.
        let parent = ChunkKey::new(C + IVec3::new(2, 0, 0), 1);
        let near_child = ChunkKey::new(C + IVec3::new(2, 0, 0), 0);

        let occupied: Occupied = [(parent, Occupancy::Empty)].into_iter().collect();
        let plan = plan_with(&config, &[camera(C, Vec3::NEG_Z)], &occupied);
        assert!(plan.spawn.contains(&near_child));
        assert!(
            parent
                .children()
                .filter(|child| plan.spawn.contains(child))
                .count()
                == 1,
            "only the child in range spawns: {:?}",
            plan.spawn
        );
        assert!(plan.retire.is_empty(), "retired {:?}", plan.retire);

        let occupied: Occupied =
            [(parent, Occupancy::Empty), (near_child, Occupancy::Empty)]
                .into_iter()
                .collect();
        let plan = plan_with(&config, &[camera(C, Vec3::NEG_Z)], &occupied);
        assert_eq!(plan.retire, vec![parent]);
    }

    #[test]
    fn chunks_are_merged_once_the_coarser_chunk_wanted_in_their_place_is_published() {
        let config = TestWorld {
            far_lod: 1,
            ..default()
        };
        let parent = parent_at_c();
        let children: Vec<_> = parent.children().collect();

        // The level-0 chunks are published; level 1 is wanted everywhere.
        let mut occupied: Occupied = children
            .iter()
            .map(|child| (*child, Occupancy::Mixed))
            .collect();
        let plan = plan_with(&config, &[camera(C, Vec3::NEG_Z)], &occupied);
        assert!(plan.spawn.contains(&parent), "spawned {:?}", plan.spawn);
        assert!(plan.retire.is_empty(), "retired {:?}", plan.retire);

        occupied.insert(parent, Occupancy::Unpublished);
        let plan = plan_with(&config, &[camera(C, Vec3::NEG_Z)], &occupied);
        assert!(plan.retire.is_empty(), "retired {:?}", plan.retire);

        occupied.insert(parent, Occupancy::Mixed);
        let plan = plan_with(&config, &[camera(C, Vec3::NEG_Z)], &occupied);
        let retired: HashSet<_> = plan.retire.iter().copied().collect();
        assert_eq!(retired, children.iter().copied().collect());
    }

    #[test]
    fn chunk_lod_is_told_the_level_a_position_has() {
        let config = TestWorld {
            sticky_lod: true,
            ..two_levels()
        };
        let parent = parent_at_c();

        // Level 0 would be wanted here, but `chunk_lod` keeps the level the
        // chunk has, so nothing replaces it.
        let occupied: Occupied = [(parent, Occupancy::Empty)].into_iter().collect();
        let plan = plan_with(&config, &[camera(C, Vec3::NEG_Z)], &occupied);
        assert!(
            plan.spawn
                .iter()
                .all(|key| ChunkKey::containing_chunk(key.position, 1) != parent),
            "spawned inside the kept chunk: {:?}",
            plan.spawn
        );
        assert!(plan.retire.is_empty(), "retired {:?}", plan.retire);
    }

    #[test]
    fn rays_find_chunks_at_the_level_wanted_where_they_are() {
        let config = TestWorld {
            spawning_distance: 6,
            spawn_in_view: true,
            lod_bands: vec![(2, 0)],
            far_lod: 1,
            ..default()
        };

        let plan = plan_with(&config, &[camera(C, Vec3::NEG_Z)], &air_around(C));

        // Two chunks ahead is beyond the level-0 band, but the level-1 chunk
        // there reaches back to one chunk ahead, so it is split.
        let ahead = C + IVec3::new(0, 0, -2);
        assert!(plan.spawn.contains(&existing(ahead)), "{:?}", plan.spawn);
        // The level-1 chunk four chunks ahead is all beyond the band.
        let far_ahead = C + IVec3::new(0, 0, -4);
        assert!(
            plan.spawn
                .contains(&ChunkKey::containing_chunk(far_ahead, 1)),
            "the level-1 chunk four chunks ahead spawns: {:?}",
            plan.spawn
        );
        assert!(!plan.spawn.contains(&existing(far_ahead)));
    }
}
