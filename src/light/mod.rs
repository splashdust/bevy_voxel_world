//! Light propagation: light from sky and emissive voxels spreading through a
//! world's open voxels. Compiled only with the `light_propagation` feature.
//!
//! Each level-0 chunk's light is worked out by the [`solver`] in the chunk's
//! own task, after its voxels are generated, and lands in the world's
//! [world view](view) on the main thread. Chunks arriving later around it
//! [relight](relight) it. A chunk above LOD level 0 works out its light from
//! its own cells alone and never relights: see [`coarse`].
//!
//! A world that shows its light draws its chunks with its material wrapped in
//! the crate's own extension ([`material`]), and gets its light to the shader
//! through a texture of slots and a tag on each chunk ([`route`]).

mod coarse;
pub(crate) mod config;
mod edit;
mod faces;
mod inputs;
mod kernel;
mod levels;
pub(crate) mod material;
pub(crate) mod probe;
pub(crate) mod record;
pub(crate) mod relight;
pub(crate) mod route;
pub(crate) mod sky;
pub(crate) mod slot;
pub(crate) mod solver;
pub(crate) mod view;

#[doc(hidden)]
pub mod bench;

use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use bevy::prelude::*;

use crate::chunk::ChunkKey;
use crate::chunk_lifecycle::ChunkLifecycle;
use crate::chunk_map::ChunkMapData;
use crate::configuration::{VoxelLookupDelegate, VoxelWorldConfig};
use crate::voxel_edits::VoxelEdits;
use crate::voxel_world::VoxelWorldSet;
use coarse::{takes_coarse_path, CoarseSolved, CoarseTask, CoarseWalk};
use config::{Emission, LightPropagation, Sky};
use levels::LevelScale;
pub use levels::LightLevels;
use probe::SkyProbes;
use record::SkyRecord;
use relight::{LightWork, Relights};
use sky::{Above, SkyFound, SkyRule, SkyWalk};
use solver::{KeptLight, KeptTask, LightCounters, LightTask, LightTaskSettings, Solved};
use view::LightView;

/// The light settings world `C` returned at plugin build, with out-of-range
/// values clamped. Only a world that returned a light config has one.
#[derive(Resource)]
pub(crate) struct LightSettings<C: VoxelWorldConfig> {
    #[cfg_attr(not(test), expect(dead_code, reason = "nothing compares settings yet"))]
    pub config: LightPropagation,
    #[cfg_attr(not(test), expect(dead_code, reason = "nothing compares settings yet"))]
    pub emissive_materials: Vec<(C::MaterialIndex, Emission)>,
    /// What chunk tasks and relight tasks read to solve light.
    pub task: Arc<LightTaskSettings<C::MaterialIndex>>,
    /// Where sky light enters, or `None` with sky light off.
    pub sky_rule: Option<SkyRule>,
    pub view: LightView,
    /// The world's sky record, used only with sky light on.
    pub record: Mutex<SkyRecord>,
    /// Set by `reset_light` until the light work picks it up.
    pub reset: AtomicBool,
    /// How many frames have committed voxel edits. A chunk task or relight
    /// notes it when it starts, so a relight's light can wait for a mesh that
    /// shows the edits it read.
    pub edit_frame: AtomicU64,
    _world: PhantomData<C>,
}

impl<C: VoxelWorldConfig> LightSettings<C> {
    pub fn counters(&self) -> &LightCounters {
        &self.task.counters
    }

    /// What the task of the chunk `entity` with `key` needs to solve its
    /// light: a level-0 chunk's solve, or a coarse chunk's for one above level
    /// 0 or with another data shape.
    ///
    /// A level-0 chunk whose light the world view holds for its current voxel
    /// edits is being re-meshed after an edit, which relights it without its
    /// task: the task keeps that light and skips its solve.
    ///
    /// With sky light on, the task gets the world's lookup `delegate` to walk
    /// the chunk positions above the chunk, and the fill hints and voxel edits
    /// of those positions, read now.
    #[allow(clippy::too_many_arguments)]
    pub fn task_for(
        &self,
        key: ChunkKey,
        entity: Entity,
        data_shape: UVec3,
        world: &C,
        edits: &VoxelEdits<C, C::MaterialIndex>,
        delegate: VoxelLookupDelegate<C::MaterialIndex>,
    ) -> Option<TaskLight<C::MaterialIndex>> {
        if takes_coarse_path(key, data_shape) {
            return Some(TaskLight::Coarse(CoarseTask {
                settings: self.task.clone(),
                walk: self
                    .sky_rule
                    .map(|rule| CoarseWalk::new(key, rule, world, edits, delegate)),
            }));
        }
        let own_edits = edits.edit_count(key.position);
        let kept = self
            .view
            .read()
            .get(&key.position)
            .filter(|chunk| chunk.entity == entity && chunk.edits == own_edits)
            .map(|chunk| chunk.light.clone());
        if let Some(light) = kept {
            return Some(TaskLight::Kept(KeptTask {
                settings: self.task.clone(),
                light,
                edit_frame: self.edit_frame.load(Ordering::Relaxed),
            }));
        }
        let walk = self.sky_rule.map(|rule| {
            let window = rule.window(key.position.y);
            let positions =
                (1..=sky::walkable(window) as i32).map(|y| key.position + IVec3::Y * y);
            let counts: Vec<u64> = positions
                .clone()
                .map(|position| edits.edit_count(position))
                .collect();
            let mut record = self.record.lock().unwrap();
            let above = positions
                .clone()
                .zip(edits.buckets(positions))
                .zip(&counts)
                .map(|((position, edits), count)| {
                    Above::new(
                        world.chunk_fill_hint(ChunkKey::new(position, 0)),
                        record.get(position, *count),
                        edits,
                    )
                })
                .collect();
            SkyWalk {
                position: key.position,
                window,
                above,
                delegate,
                found: SkyFound {
                    edits: counts,
                    own_edits: edits.edit_count(key.position),
                    generation: record.generation(),
                    ..default()
                },
            }
        });
        Some(TaskLight::Level0(LightTask {
            settings: self.task.clone(),
            view: self.view.clone(),
            walk,
            own_edits,
            edit_frame: self.edit_frame.load(Ordering::Relaxed),
        }))
    }

    /// The light at the voxel at `position`, from the finest published chunk
    /// covering it whose light has landed, or `None` if none has.
    ///
    /// A level-0 chunk reads its light in the world view and its published
    /// voxels. A coarse chunk reads the cell holding the voxel in the copy of
    /// its slot it keeps in `route`, or else the levels its tag holds, from
    /// `work`.
    pub fn light_at(
        &self,
        chunk_map: &ChunkMapData<C::MaterialIndex>,
        work: &LightWork<C>,
        route: Option<&route::LightRoute<C>>,
        position: IVec3,
    ) -> Option<LightLevels> {
        chunk_map.covering(position).find_map(|chunk| {
            let key = chunk.key;
            if !takes_coarse_path(key, chunk.data_shape) {
                let light = self.view.read().get(&key.position)?.light.clone();
                let voxel = chunk.get_voxel_at_world_position(position)?;
                let local = (position - key.min_voxel()).as_uvec3();
                return Some(light.levels_at(local, voxel, &self.task.emission));
            }
            let width = 1 << key.lod_level;
            let cell = ((position - key.min_voxel()) / width + 1).as_uvec3();
            let texel = route
                .and_then(|route| route.coarse_cell(chunk.entity, cell))
                .or_else(|| work.coarse_levels(chunk.entity))?;
            Some(LightLevels::of_texel(texel))
        })
    }
}

/// What a chunk task needs to solve its chunk's light.
pub(crate) enum TaskLight<I> {
    Level0(LightTask<I>),
    Kept(KeptTask<I>),
    Coarse(CoarseTask<I>),
}

impl<I: Copy + PartialEq> TaskLight<I> {
    pub fn solve(self, chunk: &crate::chunk::ChunkData<I>) -> SolvedLight {
        match self {
            TaskLight::Level0(task) => SolvedLight::Level0(task.solve(chunk)),
            TaskLight::Kept(task) => SolvedLight::Kept(task.solve()),
            TaskLight::Coarse(task) => SolvedLight::Coarse(task.solve(chunk)),
        }
    }
}

/// The light a chunk task solved.
#[expect(
    clippy::large_enum_variant,
    reason = "one per chunk task, moved twice on its way to landing"
)]
pub(crate) enum SolvedLight {
    Level0(Solved),
    Kept(KeptLight),
    Coarse(CoarseSolved),
}

/// Reads world `C`'s light settings into a [`LightSettings`]. A world returning
/// no light config gets nothing, so it costs what it cost without light.
pub(crate) fn build<C: VoxelWorldConfig>(app: &mut App, world: &C) {
    let Some(config) = world.light_propagation() else {
        return;
    };
    let config = config.clamped();
    let emissive_materials: Vec<_> = world
        .emissive_materials()
        .into_iter()
        .map(|(material, emission)| (material, emission.clamped()))
        .collect();
    let routed = app.world().contains_resource::<route::LightRoute<C>>();
    let task = Arc::new(task_settings::<C>(&config, &emissive_materials, routed));
    let sky_rule = SkyRule::of(&config.sky);
    if let Some(rule) = sky_rule {
        app.insert_resource(SkyProbes::<C>::new(rule));
    }
    let record = Mutex::new(SkyRecord::new(config.sky_record_cap));
    app.insert_resource(LightSettings::<C> {
        sky_rule,
        config,
        emissive_materials,
        task,
        view: LightView::default(),
        record,
        reset: AtomicBool::new(false),
        edit_frame: AtomicU64::new(0),
        _world: PhantomData,
    })
    .init_resource::<LightWork<C>>()
    .add_systems(
        PreUpdate,
        (
            (
                Relights::<C>::forget_despawned,
                Relights::<C>::collect,
                edit::relight_edits::<C>,
                SkyProbes::<C>::collect.run_if(resource_exists::<SkyProbes<C>>),
                probe::reset_light::<C>,
                Relights::<C>::land,
            )
                .chain()
                .after(ChunkLifecycle::<C>::collect_chunk_tasks)
                .before(crate::voxel_world_internal::Internals::<C>::stream_chunks),
            // Relights start before chunk tasks, which get the room left in
            // the pool budget.
            Relights::<C>::start
                .after(ChunkLifecycle::<C>::apply_requests)
                .before(ChunkLifecycle::<C>::start_chunk_tasks),
            // Sky probes get the room chunk tasks leave.
            SkyProbes::<C>::start
                .run_if(resource_exists::<SkyProbes<C>>)
                .after(ChunkLifecycle::<C>::start_chunk_tasks),
        )
            .in_set(VoxelWorldSet::<C>::default()),
    );
}

/// The level of sky light a chunk shows before its light lands: direct sky,
/// as a column nothing has settled shows, or 0 with sky light off.
pub(crate) fn placeholder_sky(config: &LightPropagation) -> u8 {
    if config.sky == Sky::Off {
        0
    } else {
        15
    }
}

/// What chunk tasks read of `config`, which must be clamped.
fn task_settings<C: VoxelWorldConfig>(
    config: &LightPropagation,
    emissive_materials: &[(C::MaterialIndex, Emission)],
    routed: bool,
) -> LightTaskSettings<C::MaterialIndex> {
    LightTaskSettings {
        voxel_scale: LevelScale::new(config.voxel_reach as u32),
        sky_scale: (config.sky != Sky::Off)
            .then(|| LevelScale::new(config.sky_reach as u32)),
        shape: config.shape,
        routed,
        emission: emissive_materials
            .iter()
            .map(|(material, emission)| (*material, emission.whole_level()))
            .collect(),
        counters: Arc::new(LightCounters::default()),
    }
}
