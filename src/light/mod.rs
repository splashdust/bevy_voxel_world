//! Light propagation: light from sky and emissive voxels spreading through a
//! world's open voxels. Compiled only with the `light_propagation` feature.

pub(crate) mod config;

use std::marker::PhantomData;

use bevy::prelude::*;

use crate::configuration::VoxelWorldConfig;
use config::{Emission, LightPropagation};

/// The light settings world `C` returned at plugin build, with out-of-range
/// values clamped. Only a world that returned a light config has one.
#[derive(Resource)]
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "nothing acts on the light settings yet")
)]
pub(crate) struct LightSettings<C: VoxelWorldConfig> {
    pub config: LightPropagation,
    pub emissive_materials: Vec<(C::MaterialIndex, Emission)>,
    _world: PhantomData<C>,
}

/// Reads world `C`'s light settings into a [`LightSettings`]. A world returning
/// no light config gets nothing, so it costs what it cost without light.
pub(crate) fn build<C: VoxelWorldConfig>(app: &mut App, world: &C) {
    let Some(config) = world.light_propagation() else {
        return;
    };
    let emissive_materials = world
        .emissive_materials()
        .into_iter()
        .map(|(material, emission)| (material, emission.clamped()))
        .collect();
    app.insert_resource(LightSettings::<C> {
        config: config.clamped(),
        emissive_materials,
        _world: PhantomData,
    });
}
