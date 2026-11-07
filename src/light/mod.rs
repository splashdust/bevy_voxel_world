//! Light propagation: light from sky and emissive voxels spreading through a
//! world's open voxels. Compiled only with the `light_propagation` feature.
//!
//! Each level-0 chunk's light is worked out by the [`solver`] in the chunk's
//! own task, after its voxels are generated, and lands in the world's
//! [world view](view) on the main thread. Chunks arriving later around it
//! [relight](relight) it. A chunk above LOD level 0 works out its light from
//! its own cells alone and never relights: see [`coarse`].
//!
//! With sky fill on, sky light also reaches level-0 chunks past where it
//! spreads from direct sky, from levels a field on the main thread works out
//! per 8-voxel cell: see [`fill`].
//!
//! A world that shows its light draws its chunks with its material wrapped in
//! the crate's own extension ([`material`]), and gets its light to the shader
//! through a texture of slots and a tag on each chunk ([`route`]).

mod coarse;
pub(crate) mod config;
mod edit;
mod faces;
pub(crate) mod fill;
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

use bevy::asset::AssetEventSystems;
use bevy::log::warn_once;
use bevy::prelude::*;

use bevy::platform::collections::HashSet;

use crate::chunk::{ChunkKey, FillType};
use crate::chunk_lifecycle::ChunkLifecycle;
use crate::chunk_map::{ChunkMap, ChunkMapData};
use crate::configuration::{VoxelLookupDelegate, VoxelWorldConfig};
use crate::pool_budget::{JobKind, PoolBudget};
use crate::voxel_edits::VoxelEdits;
use crate::voxel_world::VoxelWorldSet;
use coarse::{takes_coarse_path, CoarseSolved, CoarseTask, CoarseWalk};
use config::{Emission, LightPropagation, Sky};
use fill::driver::SkyFillWork;
use fill::FillCells;
use inputs::Occupancy;
use levels::LevelScale;
pub use levels::LightLevels;
use probe::SkyProbes;
use record::SkyRecord;
use relight::{LightWork, Relights};
use sky::{Above, ColumnAnswer, SkyFound, SkyRule, SkyWalk};
use solver::{KeptLight, KeptTask, LightCounters, LightTask, LightTaskSettings, Solved};
use view::LightView;

/// The light settings of world `C` as last applied, with out-of-range values
/// clamped. Only a world that returned a light config at plugin build has
/// one.
#[derive(Resource)]
pub(crate) struct LightSettings<C: VoxelWorldConfig> {
    /// The config as last applied. `shown`, `precision` and `slot_capacity`
    /// keep the values read at plugin build.
    pub config: LightPropagation,
    pub emissive_materials: Vec<(C::MaterialIndex, Emission)>,
    /// What chunk tasks and relight tasks read to solve light.
    pub task: Arc<LightTaskSettings<C::MaterialIndex>>,
    /// Where sky light enters, or `None` with sky light off.
    pub sky_rule: Option<SkyRule>,
    pub view: LightView,
    /// The world's sky record, used only with sky light on.
    pub record: Mutex<SkyRecord>,
    /// The cells of the world's sky fill field, which stay empty with sky
    /// fill off.
    pub fill_cells: FillCells,
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
    let emissive_materials = clamped_emission(world);
    let routed = app.world().contains_resource::<route::LightRoute<C>>();
    let fill_cells = FillCells::default();
    let field = fill::driver::field_of(&config);
    let task = Arc::new(task_settings::<C>(
        &config,
        &emissive_materials,
        routed,
        field.as_ref().map(|_| fill_cells.clone()),
    ));
    let sky_rule = SkyRule::of(&config.sky);
    if let Some(rule) = sky_rule {
        let mut probes = SkyProbes::<C>::new(rule);
        probes.reads_cells = field.is_some();
        app.insert_resource(probes);
    }
    let mut record = SkyRecord::new(config.sky_record_cap);
    let fill_on = field.is_some();
    if let Some(field) = field {
        record.set_keeps_cells(true);
        app.insert_resource(SkyFillWork::<C>::new(field, fill_cells.clone()));
    }
    let record = Mutex::new(record);
    let mut work = LightWork::<C>::default();
    work.tracks_fill = fill_on;
    app.insert_resource(LightSettings::<C> {
        sky_rule,
        config,
        emissive_materials,
        task,
        view: LightView::default(),
        record,
        fill_cells,
        reset: AtomicBool::new(false),
        edit_frame: AtomicU64::new(0),
        _world: PhantomData,
    })
    .insert_resource(work)
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
                SkyFillWork::<C>::follow.run_if(resource_exists::<SkyFillWork<C>>),
            )
                .chain()
                .after(ChunkLifecycle::<C>::collect_chunk_tasks)
                .before(crate::voxel_world_internal::Internals::<C>::stream_chunks),
            // Relights start before chunk tasks, which get the room left in
            // the pool budget.
            Relights::<C>::start
                .after(ChunkLifecycle::<C>::apply_requests)
                .before(ChunkLifecycle::<C>::start_chunk_tasks),
            // Relights for a change of sky fill, then for a settings change,
            // get the room chunk tasks leave, sky probes the room left after
            // them, and fill probes what is left after that.
            (
                Relights::<C>::start_fill,
                Relights::<C>::start_settings,
                SkyProbes::<C>::start.run_if(resource_exists::<SkyProbes<C>>),
                SkyFillWork::<C>::start_probes.run_if(resource_exists::<SkyFillWork<C>>),
            )
                .chain()
                .after(ChunkLifecycle::<C>::start_chunk_tasks),
        )
            .in_set(VoxelWorldSet::<C>::default()),
    )
    .add_systems(
        PostUpdate,
        follow_config::<C>
            .run_if(resource_changed::<C>)
            .before(AssetEventSystems),
    );
}

/// Compares world `C`'s light config and emissive list with what it last
/// applied, on frames where the world's config resource changed, and does
/// the least the difference needs.
///
/// A changed look is stored, for the world's material to write into its
/// uniform on the same frame, and a changed sky record cap applies at once.
/// A change of reach, shape, the sky rule, sky fill or the emissive list
/// runs the settings relight: see [`relight_settings`]. A change of sky fill
/// or of the sky rule first builds the sky fill field again, at once, from
/// the loaded chunks and the cells the sky record holds, so the relights
/// read the new fill. Values equal to those last applied start nothing.
///
/// `shown`, and returning `None` after `Some`, are read once at plugin
/// build: changing them logs a warning once and changes nothing.
#[allow(clippy::too_many_arguments)]
pub(crate) fn follow_config<C: VoxelWorldConfig>(
    world: Res<C>,
    mut settings: ResMut<LightSettings<C>>,
    mut work: ResMut<LightWork<C>>,
    mut probes: Option<ResMut<SkyProbes<C>>>,
    mut fill_work: Option<ResMut<SkyFillWork<C>>>,
    route: Option<ResMut<route::LightRoute<C>>>,
    mut budget: ResMut<PoolBudget<C>>,
    chunk_map: Res<ChunkMap<C, C::MaterialIndex>>,
    edits: Res<VoxelEdits<C, C::MaterialIndex>>,
    mut commands: Commands,
) {
    let Some(config) = world.light_propagation() else {
        warn_once!(
            "light propagation: a world's light_propagation() returned None after \
             returning a config at plugin build, which changes nothing"
        );
        return;
    };
    if config.shown != settings.config.shown {
        warn_once!(
            "light propagation: `shown` is read once at plugin build, so changing it \
             changes nothing"
        );
    }
    let config = config.clamped();
    let emissive = clamped_emission(world.as_ref());
    let applied = &settings.config;
    let sky_changed = config.sky != applied.sky;
    let emission_changed = emissive != settings.emissive_materials;
    // Sky reach changes nothing with sky light off.
    let sky_reach_changed =
        config.sky_reach != applied.sky_reach && config.sky != Sky::Off;
    let fill_changed = config.sky_fill != applied.sky_fill;
    let relights = sky_changed
        || fill_changed && config.sky != Sky::Off
        || emission_changed
        || sky_reach_changed
        || config.voxel_reach != applied.voxel_reach
        || config.shape != applied.shape;
    let cap_changed = config.sky_record_cap != applied.sky_record_cap;

    if config.look != settings.config.look {
        settings.config.look = config.look.clone();
    }
    if cap_changed {
        settings.config.sky_record_cap = config.sky_record_cap;
        settings
            .record
            .lock()
            .unwrap()
            .set_cap(config.sky_record_cap);
    }
    settings.config.wasm_relight_budget = config.wasm_relight_budget;
    if config.sky_reach != settings.config.sky_reach && !sky_reach_changed {
        settings.config.sky_reach = config.sky_reach;
    }
    if fill_changed && config.sky == Sky::Off {
        settings.config.sky_fill = config.sky_fill.clone();
    }
    if !relights {
        return;
    }
    let settings = settings.as_mut();
    settings.config.sky = config.sky.clone();
    settings.config.sky_reach = config.sky_reach;
    settings.config.voxel_reach = config.voxel_reach;
    settings.config.shape = config.shape;
    settings.config.sky_fill = config.sky_fill.clone();
    settings.emissive_materials = emissive;
    let field = fill::driver::field_of(&settings.config);
    let old = settings.task.clone();
    let generation = old.generation + 1;
    settings.task = Arc::new(LightTaskSettings {
        counters: old.counters.clone(),
        generation,
        sky_generation: if sky_changed {
            generation
        } else {
            old.sky_generation
        },
        emission_generation: if emission_changed {
            generation
        } else {
            old.emission_generation
        },
        ..task_settings::<C>(
            &settings.config,
            &settings.emissive_materials,
            old.routed,
            field.as_ref().map(|_| settings.fill_cells.clone()),
        )
    });

    let mut sky_changed_at = HashSet::new();
    let mut new_probes: Option<SkyProbes<C>> = None;
    if sky_changed {
        settings.sky_rule = SkyRule::of(&settings.config.sky);
        if let Some(mut route) = route {
            route.set_unlit_sky(placeholder_sky(&settings.config));
        }
        let mut record = settings.record.lock().unwrap();
        match (settings.sky_rule, probes.as_mut()) {
            (None, probes) => {
                if probes.is_some() {
                    commands.remove_resource::<SkyProbes<C>>();
                    budget.set_running(JobKind::SkyProbe, 0);
                }
                let lit: Vec<(IVec3, Entity)> = settings
                    .view
                    .read()
                    .iter()
                    .filter(|(_, chunk)| chunk.light.inputs.columns.has_sky())
                    .map(|(position, chunk)| (*position, chunk.entity))
                    .collect();
                for (position, entity) in lit {
                    let none = ColumnAnswer::none();
                    probe::replace_columns(&settings.view, position, entity, none);
                    sky_changed_at.insert(position);
                }
            }
            (Some(rule), Some(probes)) => {
                probes.set_rule(rule);
                sky_changed_at.extend(probe::answer_loaded_again(
                    probes,
                    &settings.view,
                    &mut record,
                    &chunk_map,
                    &edits,
                    world.as_ref(),
                ));
            }
            (Some(rule), None) => {
                record.clear();
                let mut probes = SkyProbes::<C>::new(rule);
                sky_changed_at.extend(probe::answer_loaded_again(
                    &mut probes,
                    &settings.view,
                    &mut record,
                    &chunk_map,
                    &edits,
                    world.as_ref(),
                ));
                new_probes = Some(probes);
            }
        }
    }
    if sky_changed || fill_changed {
        work.tracks_fill = field.is_some();
        work.take_fill_touched();
        let mut record = settings.record.lock().unwrap();
        record.set_keeps_cells(field.is_some());
        if let Some(probes) = new_probes.as_mut().or(probes.as_deref_mut()) {
            probes.reads_cells = field.is_some();
            probes.take_touched();
        }
        let probes = new_probes.as_ref().or(probes.as_deref());
        match (field, settings.sky_rule) {
            (Some(field), Some(rule)) => {
                let learn = |work: &mut SkyFillWork<C>, record: &mut SkyRecord| {
                    work.learn_everything(
                        &settings.view,
                        record,
                        probes,
                        rule,
                        &edits,
                        world.as_ref(),
                    )
                };
                match fill_work.as_mut() {
                    Some(work) => {
                        work.reset(field);
                        learn(work, &mut record);
                    }
                    None => {
                        let mut work =
                            SkyFillWork::<C>::new(field, settings.fill_cells.clone());
                        learn(&mut work, &mut record);
                        commands.insert_resource(work);
                    }
                }
            }
            _ => {
                if fill_work.is_some() {
                    commands.remove_resource::<SkyFillWork<C>>();
                    budget.set_running(JobKind::FillProbe, 0);
                }
                settings.fill_cells.write().clear();
            }
        }
    }
    if let Some(probes) = new_probes {
        commands.insert_resource(probes);
    }
    relight_settings(
        settings,
        &mut work,
        &chunk_map,
        &sky_changed_at,
        emission_changed,
    );
}

/// The settings relight: marks for a relight every chunk with an open voxel
/// touching a solid by a face, an edge or a corner, counting the solids in
/// its padding, and every chunk whose direct sky changed. Old light shows
/// until new light lands.
///
/// A chunk with open and solid voxels always has such an open voxel, and a
/// chunk of air has one exactly when a solid lies in its padding. A chunk of
/// air with no solid in its padding keeps its light, which may then be a
/// setting behind: within reach of nothing, only its direct sky could change,
/// and that relights it. A solid chunk has no open voxel, and relights only
/// when `emission_changed`, to find its emissive voxels again for its
/// neighbours' solves.
fn relight_settings<C: VoxelWorldConfig>(
    settings: &LightSettings<C>,
    work: &mut LightWork<C>,
    chunk_map: &ChunkMap<C, C::MaterialIndex>,
    sky_changed_at: &HashSet<IVec3>,
    emission_changed: bool,
) {
    let chunk_map = chunk_map.get_read_lock();
    let marked: Vec<IVec3> = settings
        .view
        .read()
        .iter()
        .filter(|(position, chunk)| {
            sky_changed_at.contains(*position)
                || match &chunk.light.inputs.occupancy {
                    Occupancy::Mixed(_) => true,
                    Occupancy::Solid => emission_changed,
                    Occupancy::Open => chunk_map
                        .get(&ChunkKey::new(**position, 0))
                        .is_some_and(|data| matches!(data.fill_type, FillType::Mixed)),
                }
        })
        .map(|(position, _)| *position)
        .collect();
    for position in marked {
        work.trigger_settings(position);
    }
}

/// World `C`'s emissive list, with each level clamped.
fn clamped_emission<C: VoxelWorldConfig>(world: &C) -> Vec<(C::MaterialIndex, Emission)> {
    world
        .emissive_materials()
        .into_iter()
        .map(|(material, emission)| (material, emission.clamped()))
        .collect()
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
    fill: Option<FillCells>,
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
        generation: 0,
        sky_generation: 0,
        emission_generation: 0,
        fill,
    }
}
