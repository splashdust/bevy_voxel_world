//! The material of a lit world that shows its light: the world's material
//! wrapped in the crate's own extension, [`VoxelWorldBindings`].
//!
//! The extension holds what the shader needs to read light, the world's light
//! texture and its look, at bindings the crate keeps for itself, 200 to 219.
//! It has no shader of its own, so the world's material keeps its shaders, and
//! a shader that wants light imports the `bevy_voxel_world::light` module,
//! which declares those bindings. The extension adds the `VOXEL_LIGHT` shader
//! def, under which the crate's vertex shader outputs what that module reads.
//!
//! `ExtendedMaterial` takes `enable_prepass` and `enable_shadows` from the
//! extension alone, so the extension passes on the wrapped material's. It
//! takes the opaque render method from the wrapped material, so the extension
//! can't force forward rendering. Light adds to the emissive channel only in
//! the forward pass, so a lit chunk material must render forward.
//!
//! An unlit world, or one with `shown: false`, keeps its material unwrapped.
//!
//! The sun gate: by default Bevy's directional lights shine only on faces
//! with direct sky. Bevy's shadow map can't be trusted for this, because it
//! holds only spawned chunks, and under the default streaming rock out of
//! view isn't spawned, so the sun would shine into caves through it. Direct
//! sky falls straight down, so the sun stops at an overhang's drip line.
//! A face reads light blended between voxels, so the gate is a step between
//! direct sky and the level beside it, worked out from sky reach and
//! precision.

use std::marker::PhantomData;

use bevy::{
    asset::uuid_handle,
    mesh::MeshVertexBufferLayoutRef,
    pbr::{
        ExtendedMaterial, MaterialExtension, MaterialExtensionKey,
        MaterialExtensionPipeline,
    },
    prelude::*,
    reflect::TypePath,
    render::render_resource::{
        AsBindGroup, RenderPipelineDescriptor, ShaderType, SpecializedMeshPipelineError,
    },
    shader::{Shader, ShaderDefVal},
};

use super::config::{
    LevelCurve, LightAo, LightPrecision, LightPropagation, LightSmoothing, SunGate,
};
use super::route::{self, LightRoute};
use crate::configuration::VoxelWorldConfig;
use crate::plugin::{add_material_path, WorldMaterial};

/// The material of a lit world showing its light: the world's material `M`
/// with [`VoxelWorldBindings`].
pub type VoxelWorldMaterial<M> = ExtendedMaterial<M, VoxelWorldBindings<M>>;

/// The crate's extension of a lit world's material: the world's light texture
/// and how light looks, for the `bevy_voxel_world::light` shader module.
///
/// It has no shader of its own. It passes on `enable_prepass` and
/// `enable_shadows` from `M`, and adds the `VOXEL_LIGHT` shader def.
#[derive(Asset, AsBindGroup, TypePath, Debug, Clone)]
pub struct VoxelWorldBindings<M: Material> {
    #[texture(200, dimension = "3d", sample_type = "u_int")]
    light_texture: Handle<Image>,
    #[uniform(201)]
    look: LookUniform,
    _material: PhantomData<fn() -> M>,
}

impl<M: Material> MaterialExtension for VoxelWorldBindings<M> {
    fn enable_prepass() -> bool {
        M::enable_prepass()
    }

    fn enable_shadows() -> bool {
        M::enable_shadows()
    }

    fn specialize(
        _pipeline: &MaterialExtensionPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        _layout: &MeshVertexBufferLayoutRef,
        _key: MaterialExtensionKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        let def = ShaderDefVal::Bool("VOXEL_LIGHT".into(), true);
        descriptor.vertex.shader_defs.push(def.clone());
        if let Some(fragment) = descriptor.fragment.as_mut() {
            fragment.shader_defs.push(def);
        }
        Ok(())
    }
}

/// A world's [`LightLook`](super::config::LightLook) as the shader reads it, the `LightLook` struct of
/// the `bevy_voxel_world::light` module.
///
/// Each curve is 16 levels as 4 `vec4`s, with level 0 forced to 0, so a
/// face with no light stays black whatever the curve says.
#[derive(Clone, Copy, Debug, ShaderType)]
pub(crate) struct LookUniform {
    pub sky_falloff: [Vec4; 4],
    pub voxel_falloff: [Vec4; 4],
    /// How much of Bevy's directional lights reaches each sky light level,
    /// when `sun_step` is 0.
    pub sun_falloff: [Vec4; 4],
    /// With [`SunGate::DirectSkyOnly`], the sky light level from which the sun
    /// shines fully, and below which it doesn't shine. 0 otherwise.
    pub sun_step: f32,
    pub voxel_brightness: f32,
    pub emissive_glow: f32,
    pub minimum_brightness: f32,
    /// 1 for [`LightSmoothing::Smoothed`], 0 for stepped.
    pub smoothed: u32,
    /// 0 for [`LightAo::IndirectLight`], 1 for the base colour, 2 for off.
    pub ao: u32,
}

impl LookUniform {
    /// The uniform for `config`'s look, which must be clamped.
    pub fn new(config: &LightPropagation) -> Self {
        let look = &config.look;
        let (sun, sun_step) = match &look.sun {
            SunGate::DirectSkyOnly => {
                let mut brightness = [0.0; 16];
                brightness[15] = 1.0;
                (LevelCurve { brightness }, direct_sky_step(config))
            }
            SunGate::ByLevel(curve) => (curve.clone(), 0.0),
        };
        Self {
            sky_falloff: table(&look.sky_falloff),
            voxel_falloff: table(&look.voxel_falloff),
            sun_falloff: table(&sun),
            sun_step,
            voxel_brightness: look.voxel_brightness,
            emissive_glow: look.emissive_glow,
            minimum_brightness: look.minimum_brightness,
            smoothed: (look.smoothing == LightSmoothing::Smoothed) as u32,
            ao: match look.ao {
                LightAo::IndirectLight => 0,
                LightAo::BaseColor => 1,
                LightAo::Off => 2,
            },
        }
    }
}

/// The sky light level halfway between direct sky, at 15, and the brightest
/// level an open voxel without direct sky holds, one step from it.
///
/// A face reads light blended between the voxels in front of it, so at the edge
/// of direct sky it reads levels between the two. The sun shines from this
/// level up, so it stops halfway across the edge. At 16 levels the voxel beside
/// direct sky holds at most 14. At 256 levels it holds `255 × reach / (reach +
/// 1)` in 255ths, which comes closer to 15 the longer the reach, so the step
/// follows the reach.
fn direct_sky_step(config: &LightPropagation) -> f32 {
    let t = 1.0 - 1.0 / (config.sky_reach as f32 + 1.0);
    let beside = match config.precision {
        LightPrecision::Steps16 => (15.0 * t).round().clamp(1.0, 14.0),
        LightPrecision::Steps256 => (255.0 * t).round().clamp(1.0, 254.0) / 17.0,
    };
    (beside + 15.0) / 2.0
}

/// A curve as the shader's table, with level 0 at 0.
fn table(curve: &LevelCurve) -> [Vec4; 4] {
    let mut brightness = curve.brightness;
    brightness[0] = 0.0;
    std::array::from_fn(|row| Vec4::from_slice(&brightness[row * 4..row * 4 + 4]))
}

/// The `bevy_voxel_world::light` shader module.
pub(crate) const LIGHT_SHADER_HANDLE: Handle<Shader> =
    uuid_handle!("3c5e9a1d-7b24-4f86-a0d3-58e1f2b7c946");

/// Adds the `bevy_voxel_world::light` shader module to `shaders`.
pub(crate) fn add_light_shader(shaders: &mut Assets<Shader>) {
    // An embedded path gives the module its import path.
    let light = Shader::from_wesl(
        include_str!("../shaders/light.wesl"),
        "embedded://bevy_voxel_world/light.wesl",
    );
    shaders.insert(LIGHT_SHADER_HANDLE.id(), light).unwrap();
}

/// Sets up giving the chunks of lit world `C` its material `base` wrapped in
/// [`VoxelWorldBindings`], with the world's light texture, and the crate's tag.
pub(crate) fn add_lit_material_path<C: VoxelWorldConfig, M: Material>(
    app: &mut App,
    base: M,
    config: &LightPropagation,
) where
    <VoxelWorldMaterial<M> as AsBindGroup>::Data:
        PartialEq + Eq + std::hash::Hash + Clone,
{
    if app
        .get_added_plugins::<MaterialPlugin<VoxelWorldMaterial<M>>>()
        .is_empty()
    {
        app.add_plugins(MaterialPlugin::<VoxelWorldMaterial<M>>::default());
    }

    let config = config.clamped();
    let route = LightRoute::<C>::new(
        &config,
        super::placeholder_sky(&config),
        &mut app.world_mut().resource_mut::<Assets<Image>>(),
    );
    let material = ExtendedMaterial {
        base,
        extension: VoxelWorldBindings {
            light_texture: route.texture.clone(),
            look: LookUniform::new(&config),
            _material: PhantomData,
        },
    };
    app.insert_resource(route);
    route::add_render_systems::<C>(app);

    let handle = app
        .world_mut()
        .resource_mut::<Assets<VoxelWorldMaterial<M>>>()
        .add(material);
    add_material_path::<C, VoxelWorldMaterial<M>>(app, Some(handle));
    app.world_mut()
        .resource_mut::<WorldMaterial<C, VoxelWorldMaterial<M>>>()
        .insert_chunk_components = route::tag_chunk::<C>;
}

#[cfg(test)]
mod tests {
    use super::super::config::LightLook;
    use super::*;

    #[test]
    fn the_looks_uniform_fits_webgl2s_uniform_block_and_level_0_is_black() {
        assert!(LookUniform::min_size().get() <= 16 * 1024);

        let mut config = LightPropagation::default();
        config.look.sky_falloff.brightness[0] = 0.5;
        config.look.voxel_falloff.brightness[0] = 0.5;
        let uniform = LookUniform::new(&config);
        assert_eq!(uniform.sky_falloff[0].x, 0.0);
        assert_eq!(uniform.voxel_falloff[0].x, 0.0);
        assert_eq!(uniform.sun_falloff[0].x, 0.0);
        assert_eq!(uniform.voxel_falloff[3].w, 1.0);
        assert!((uniform.voxel_falloff[3].z - 0.7).abs() < 1e-6);
    }

    #[test]
    fn the_sun_steps_between_direct_sky_and_the_brightest_level_beside_it() {
        let step = |sky_reach, precision| {
            LookUniform::new(&LightPropagation {
                sky_reach,
                precision,
                ..default()
            })
            .sun_step
        };
        // At 16 levels the voxel beside direct sky holds 14 at any reach above
        // a few voxels, and level 15 is direct sky.
        assert_eq!(step(30, LightPrecision::Steps16), 14.5);
        assert_eq!(step(14, LightPrecision::Steps16), 14.5);
        // At 256 levels it holds 247/255 at reach 30, and 238/255 at reach 14.
        let beside_30 = 247.0 / 17.0;
        assert!(
            (step(30, LightPrecision::Steps256) - (beside_30 + 15.0) / 2.0).abs() < 1e-5
        );
        let beside_14 = 238.0 / 17.0;
        assert!(
            (step(14, LightPrecision::Steps256) - (beside_14 + 15.0) / 2.0).abs() < 1e-5
        );

        let by_level = LookUniform::new(&LightPropagation {
            look: LightLook {
                sun: SunGate::ByLevel(LevelCurve::default()),
                ..default()
            },
            ..default()
        });
        assert_eq!(by_level.sun_step, 0.0);
    }
}
