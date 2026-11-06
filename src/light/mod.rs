//! Light propagation: light from sky and emissive voxels spreading through a
//! world's open voxels. Compiled only with the `light_propagation` feature.
//!
//! Each level-0 chunk's light is worked out by the [`solver`] in the chunk's
//! own task, after its voxels are generated, and published with its voxels.

pub(crate) mod config;
mod inputs;
mod kernel;
mod levels;
pub(crate) mod solver;

#[doc(hidden)]
pub mod bench;

use std::marker::PhantomData;
use std::sync::Arc;
use std::time::Instant;

use bevy::log::warn_once;
use bevy::prelude::*;

use crate::chunk::{ChunkKey, PADDED_CHUNK_SIZE};
use crate::chunk_map::ChunkMapData;
use crate::configuration::VoxelWorldConfig;
use config::{Emission, LightPropagation, Sky};
use kernel::MAX_ONE_WORD_REACH;
use levels::LevelScale;
pub use levels::LightLevels;
use solver::{LightCounters, LightTask, LightTaskSettings};

/// The light settings world `C` returned at plugin build, with out-of-range
/// values clamped. Only a world that returned a light config has one.
#[derive(Resource)]
pub(crate) struct LightSettings<C: VoxelWorldConfig> {
    #[cfg_attr(not(test), expect(dead_code, reason = "nothing compares settings yet"))]
    pub config: LightPropagation,
    #[cfg_attr(not(test), expect(dead_code, reason = "nothing compares settings yet"))]
    pub emissive_materials: Vec<(C::MaterialIndex, Emission)>,
    /// What chunk tasks read to solve light.
    pub task: Arc<LightTaskSettings<C::MaterialIndex>>,
    _world: PhantomData<C>,
}

impl<C: VoxelWorldConfig> LightSettings<C> {
    pub fn counters(&self) -> &LightCounters {
        &self.task.counters
    }

    /// What the task of the chunk with `key` needs to solve its light, read
    /// from the chunks published around it now. `None` for a chunk that
    /// doesn't get light: one above level 0, or with another data shape.
    pub fn task_for(
        &self,
        key: ChunkKey,
        data_shape: UVec3,
        chunk_map: &ChunkMapData<C::MaterialIndex>,
    ) -> Option<LightTask<C::MaterialIndex>> {
        if key.lod_level != 0 || data_shape != UVec3::splat(PADDED_CHUNK_SIZE) {
            return None;
        }
        let started = Instant::now();
        let neighbours = std::array::from_fn(|index| {
            let index = index as i32;
            let offset = IVec3::new(index % 3, index / 3 % 3, index / 9) - IVec3::ONE;
            if offset == IVec3::ZERO {
                return None;
            }
            let neighbour = chunk_map.get(&ChunkKey::new(key.position + offset, 0))?;
            Some(neighbour.light.as_ref()?.inputs.clone())
        });
        self.counters().snapshot_nanos.fetch_add(
            started.elapsed().as_nanos() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        Some(LightTask {
            settings: self.task.clone(),
            neighbours,
        })
    }

    /// The light at the voxel at `position`, from the published data of the
    /// level-0 chunk holding it, or `None` if it has no light.
    pub fn light_at(
        &self,
        chunk_map: &ChunkMapData<C::MaterialIndex>,
        position: IVec3,
    ) -> Option<LightLevels> {
        let chunk = chunk_map.get(&ChunkKey::containing(position, 0))?;
        let light = chunk.light.as_ref()?;
        let voxel = chunk.get_voxel_at_world_position(position)?;
        let local = (position - chunk.key.min_voxel()).as_uvec3();
        Some(light.levels_at(local, voxel, &self.task.emission))
    }
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
    let task = Arc::new(task_settings::<C>(&config, &emissive_materials));
    app.insert_resource(LightSettings::<C> {
        config,
        emissive_materials,
        task,
        _world: PhantomData,
    });
}

fn task_settings<C: VoxelWorldConfig>(
    config: &LightPropagation,
    emissive_materials: &[(C::MaterialIndex, Emission)],
) -> LightTaskSettings<C::MaterialIndex> {
    let mut voxel_reach = config.voxel_reach as u32;
    if voxel_reach > MAX_ONE_WORD_REACH {
        warn_once!(
            "light propagation: voxel_reach {voxel_reach} isn't supported yet, using \
             {MAX_ONE_WORD_REACH}"
        );
        voxel_reach = MAX_ONE_WORD_REACH;
    }
    LightTaskSettings {
        voxel_scale: LevelScale::new(voxel_reach),
        shape: config.shape,
        sky: if config.sky == Sky::Off { 0 } else { 15 },
        emission: emissive_materials
            .iter()
            .map(|(material, emission)| (*material, emission.whole_level()))
            .collect(),
        counters: Arc::new(LightCounters::default()),
    }
}
