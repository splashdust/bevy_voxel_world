//! Lit chunk materials
//!
//! This module wraps the material of a lit world that shows its light in
//! [`VoxelWorldBindings`], the extension the crate adds, and gives the wrapped
//! material to the chunks of the world. The extension binds the light texture, the look, the colour texture
//! and, with the `sub_grids` feature, the tag map, for the shader to read
//! light from. The systems here keep those bindings current as the textures
//! are made, replaced or grown, and as the look or the light colours of the
//! world change.
//!
//! # The material wrap
//!
//! A world names one material, `M`, with `with_material`, or gets the default
//! material of the crate. When the world is lit and shows its light, its
//! chunks get [`VoxelWorldMaterial<M>`] in place of `M`. That type is
//! `ExtendedMaterial<M, VoxelWorldBindings<M>>`, and each lit world has one.
//! The Rust code of the world doesn't change, and only its shader imports
//! light. A world whose material is added by hand, as
//! `VoxelWorldMaterialHandle<M>`, gets a wrapped copy of that material. So two
//! lit worlds naming the same handle each bind their own light texture. An
//! unlit world, a world with `shown: false` and a `VoxelWorldPlugin::minimal()`
//! world keep their material unwrapped, or get none.
//!
//! The extension holds what the shader needs to read light, the light texture
//! of the world and its look. They sit at bindings the crate keeps for itself,
//! 200 to 219. The extension has no shader of its own, so the world material
//! keeps its shaders. A shader that wants light imports the
//! `bevy_voxel_world::light` module, which declares those bindings. The one
//! exception is a world whose material is exactly `StandardMaterial`. The Bevy
//! shader for it can't read light, so the extension picks the lit PBR shader
//! of the crate, which can. The extension adds the `VOXEL_LIGHT` shader def,
//! under which the vertex shaders of the crate also output the tag of the
//! chunk. The light module reads the tag together with the local position and
//! normal, which those shaders output in every world.
//!
//! With the `sub_grids` feature the extension also binds the tag map of the
//! world at 202. Until the world has a sub-grid, it binds a placeholder of one
//! empty entry. The extension also adds the `SUB_GRIDS` shader def, under
//! which the vertex shaders of the crate also output the position of the face
//! in the grid of the world. The shader reads the world light at the faces of
//! a sub-grid from both: see [Sub-grids](#sub-grids). The tag map gets a new
//! texture each time it grows. The materials of the world bind the new one in
//! the same frame, as they bind the light texture once it is made.
//!
//! The extension also binds the colour texture of the world at 203. Until a
//! chunk with a slot shows a light colour other than white, it binds a
//! placeholder of one white texel. When the emission table of the world holds
//! such a colour, the extension adds the `LIGHT_COLOR` shader def. The def
//! travels in the material key, [`VoxelWorldBindingsKey`], since `specialize`
//! sees only the key and not the material. Under the def, the shader reads the
//! light colour of each chunk from its tag and from its entry in the colour
//! texture. Without the def, every colour in the shader is white, so the
//! shader of a white world does no colour work at all.
//!
//! # Changing the colour def
//!
//! A world coloured at startup makes its material with `LIGHT_COLOR` already
//! set, so its chunks draw with it from the first frame. A running world sets
//! the def when its emission table gains its first light colour other than
//! white, and clears it when the table loses its last one. Either change
//! changes the material key. Bevy then takes the chunks of the world out of
//! their bins for the main pass, the prepass and shadows. It draws them again
//! only once their pipelines for the new key have compiled. Where Bevy
//! compiles pipelines in the background, as on native Linux and Windows, the
//! chunks would vanish for the frames the compile takes.
//!
//! So in each frame the def changes, the render world waits for those
//! pipelines, as the upscaling pass of Bevy waits for its own. The frame takes
//! longer instead of showing the world without its chunks. Bevy keeps the
//! pipelines it compiled for both values of the def, so a change back to a
//! value it already compiled costs the wait nothing.
//!
//! # Forward rendering
//!
//! `ExtendedMaterial` takes `enable_prepass` and `enable_shadows` from the
//! extension alone, so the extension passes on those of the wrapped material.
//! It takes the opaque render method from the wrapped material, so the
//! extension can't force forward rendering. Light adds to the emissive output
//! only in the forward pass, so a lit chunk material must render forward.
//!
//! # Reading light on any mesh
//!
//! Light comes from voxels, and the shader finds the voxels of a face from the
//! position and normal of the face alone, so a mesher emits nothing for light.
//! A face on the voxel grid, as every cube face is, reads the voxel in front
//! of it, half a voxel along its normal. A face off the grid, such as a slope
//! or a rounded corner of a marching cubes mesh, can cut into a solid voxel.
//! The voxel half a voxel along its normal may then be solid and read black.
//! Such a face reads the 8 voxels around that point instead. It leaves out
//! those holding no light, and those not joined to the nearest of the 8
//! through voxels it counts, so light doesn't pass through a corner between
//! two solids. A face of a cube mesh is always on the grid, so it never reads
//! the 8 voxels.
//!
//! A face glows when the voxel half a voxel behind it is an emissive voxel.
//! The cell of an emissive voxel holds the top voxel light level, which no
//! open voxel holds, and holds the level of the emissive voxel in its sky
//! half, which a solid never uses. A face off the grid can have an open voxel
//! in front of it lit by the emissive voxel behind it, as a rounded inside
//! corner beside an emissive voxel does. That face doesn't glow, because only
//! the cell of an emissive voxel holds the top level.
//!
//! # The sun gate
//!
//! By default the Bevy directional lights shine only on faces with direct sky.
//! The Bevy shadow map can't be trusted for this, because it holds only
//! spawned chunks. Under the default streaming, solid chunks out of view
//! aren't spawned, so the sun would shine into caves through them. Direct sky
//! falls straight down, so the sun stops at the drip line of an overhang. A
//! face reads light blended between voxels, so the gate is a step between
//! direct sky and the level beside it. The step is worked out from the sky
//! light reach and the light precision.
//!
//! # Sub-grids
//!
//! With the `sub_grids` feature, a chunk of a sub-grid holds light of its own,
//! which the light of the world never sees, and its tag has bit 30 set. For a
//! face of such a chunk, the shader function `sample_face_light` reads the
//! light of the sub-grid with `sample_light`. It reads the world light at the
//! face with `world_light`, from the position and normal of the face in the
//! grid of the world. The face shows the smaller of the two sky light levels,
//! so the enclosure of a sub-grid caps the sky of the world.
//!
//! Under [`SubGridSky::Enclosed`](super::config::SubGridSky) with sky light on,
//! the `enclosure_caps_voxel` field of the look caps the world voxel light by
//! the same sky light level. So the emissive voxels of the world don't light
//! the inside of a closed cabin through its walls. The face shows the larger
//! of that capped level and the voxel light level of the sub-grid, so the
//! emissive voxels of the sub-grid add to those of the world. Under
//! `LIGHT_COLOR`, the face shows the light colour that the two voxel lights
//! mix to. The sun gate reads the capped sky. A face of a world chunk reads
//! light as `sample_light` reads it.
//!
//! `world_light` reads as `sample_light` would for a world face at the same
//! place. A face on the grid of the world reads the voxel in front of it, and
//! any other face reads the 8 voxels around the point in front of it. Every
//! face of a sub-grid at a pose that a stamp keeps lies on the grid of the
//! world. So a sub-grid at such a pose, with sky of its own at the top level
//! and no emissive voxels, looks as its voxels would look set in the world.
//!
//! `world_light` looks up one world chunk in the tag map of the world, which
//! the `tag_map` module keeps. That chunk is the one holding the voxel in
//! front of the face, the nearest of those it reads. `world_light` reads every
//! voxel from the slot of that chunk, whose shell holds the cells across its
//! faces. It reads the entry of a coarse chunk at LOD level `L` in the cells
//! of that chunk, each `1 << L` voxels wide. A position the tag map doesn't
//! hold reads as open sky with no emissive voxels, so a sub-grid far from any
//! camera never goes black.
//!
//! The shader takes the Bevy world position as a position in the grid of the
//! world, so it assumes the `WorldRoot` of the world is at the identity. A
//! custom shader that still calls `sample_light` lights the faces of a
//! sub-grid from the light of the sub-grid alone.
//!
//! What it costs grows with:
//!
//! - On the GPU, each fragment of a face of a sub-grid. It costs one lookup in
//!   the hash table of the tag map, and the cells of one more read of light. A
//!   face of a world chunk costs nothing more, and each vertex of a lit world
//!   carries one more output.
//! - On the CPU, the world chunks that the boxes of sub-grids touch. A
//!   tumbling sub-grid has a box that touches more of them.
//! - Slots, for the chunks of air near sub-grids whose light varies.
//! - Texture memory, which takes 16 bytes for each entry of the tag map.
//! - Column answers, one for each position of hinted air a sub-grid comes to
//!   touch.
//!
//! Nothing hides the places where the light of a sub-grid and the light of the
//! world differ. The emissive voxels of a sub-grid light only the sub-grid.
//! The voxels of a sub-grid cast no sky shadow on the world, though the Bevy
//! shadow map still draws their shadow from the sun. An emissive voxel that
//! should light the world around it can carry a Bevy point light as a child of
//! the sub-grid entity, which moves with the sub-grid:
//!
//! ```
//! use bevy::prelude::*;
//!
//! /// Lights the world around the lamp at `lamp`, a voxel position on the
//! /// grid of the sub-grid `sub_grid`.
//! fn light_the_world_around(commands: &mut Commands, sub_grid: Entity, lamp: IVec3) {
//!     commands.entity(sub_grid).with_child((
//!         PointLight {
//!             color: Color::srgb(1.0, 0.8, 0.5),
//!             range: 14.0,
//!             ..default()
//!         },
//!         // The middle of the lamp's voxel.
//!         Transform::from_translation(lamp.as_vec3() + Vec3::splat(0.5)),
//!     ));
//! }
//! ```
//!
//! # References
//!
//! - [`BlockMeshPart` in Terasology](https://github.com/MovingBlocks/Terasology/blob/develop/engine/src/main/java/org/terasology/engine/world/block/shapes/BlockMeshPart.java):
//!   lights a vertex from 8 points around it, so a vertex anywhere gets light.
//! - [Lighting in Cubyz](https://github.com/PixelGuys/Cubyz/blob/master/src/renderer/lighting.zig):
//!   blends the light of the 8 cells around a vertex off the grid.
//! - [Smooth lighting in Luanti](https://github.com/luanti-org/luanti/blob/master/src/client/content_mapblock.cpp):
//!   averages the cells around a corner, leaving out cells that don't carry
//!   light.
//! - [Light propagation in tesserae](https://github.com/mkappworks-dev/tesserae/pull/3):
//!   sky light dims the Bevy ambient light and sun, and voxel light adds
//!   through the emissive output.

use std::any::TypeId;
use std::marker::PhantomData;

use bevy::{
    asset::{uuid_handle, AssetEventSystems, UntypedAssetId},
    mesh::MeshVertexBufferLayoutRef,
    pbr::{
        ExtendedMaterial, MaterialExtension, MaterialExtensionKey,
        MaterialExtensionPipeline, RenderMaterialInstances,
        SpecializedMaterialPipelineCache, SpecializedPrepassMaterialPipelineCache,
        SpecializedShadowMaterialPipelineCache,
    },
    prelude::*,
    reflect::TypePath,
    render::{
        render_resource::{
            AsBindGroup, CachedRenderPipelineId, PipelineCache, RenderPipelineDescriptor,
            ShaderType, SpecializedMeshPipelineError,
        },
        sync_world::MainEntity,
        Extract, ExtractSchedule, Render, RenderApp, RenderSystems,
    },
    shader::{Shader, ShaderDefVal, ShaderRef},
};

use super::config::{
    LevelCurve, LightAo, LightPrecision, LightPropagation, LightSmoothing, SunGate,
};
use super::route::{self, LightRoute};
use super::LightSettings;
use crate::configuration::VoxelWorldConfig;
use crate::plugin::{add_material_path, VoxelWorldMaterialHandle, WorldMaterial};
use crate::voxel_world_internal::Internals;

/// The material of a lit world showing its light: the world's material `M`
/// with [`VoxelWorldBindings`].
///
/// The chunks of such a world get this material in place of `M`, one per lit
/// world. Reach your own material through
/// `VoxelWorldMaterialHandle<VoxelWorldMaterial<M>>` and its `.base`. A world
/// whose material is added by hand, with `init_custom_materials` returning
/// false, gets a wrapped copy of the material its
/// `VoxelWorldMaterialHandle<M>` names, so later edits go through the wrapped
/// handle.
///
/// Lit chunk materials are forward only. The wrap can't force forward
/// rendering, since `ExtendedMaterial` takes that from `M`, so a custom `M`
/// must render forward itself.
pub type VoxelWorldMaterial<M> = ExtendedMaterial<M, VoxelWorldBindings<M>>;

/// The extension the crate adds to the material of a lit world: the light
/// texture of the world and how light looks, for the `bevy_voxel_world::light`
/// shader module.
///
/// It passes on `enable_prepass` and `enable_shadows` from `M`, and adds the
/// `VOXEL_LIGHT` shader def. It has no shader of its own, except when `M` is
/// exactly `StandardMaterial`, which gets the lit PBR shader of the crate.
/// That shader draws any mesh, with or without the texture index of the crate,
/// and vertex colours tint the base colour as they do in the Bevy shader. The
/// [`LightAo`] setting of the look doesn't apply to it, since the vertex
/// colour belongs to the mesher. Any other `M`, including
/// `ExtendedMaterial<StandardMaterial, _>`, reads light in its own shader by
/// importing the `bevy_voxel_world::light` module, which declares the
/// bindings.
///
/// Bindings 200 to 219 belong to the crate, and the extension uses these:
///
/// - Binding 200 holds the light texture of the world, and binding 201 holds
///   the look.
/// - With the `sub_grids` feature, binding 202 holds the tag map of the world,
///   and the extension adds the `SUB_GRIDS` shader def.
/// - Binding 203 holds the colour texture of the world. The extension adds the
///   `LIGHT_COLOR` shader def when an emissive material of the world gives a
///   light colour other than white.
///
/// A custom shader can test `LIGHT_COLOR`. Without it, every light colour in
/// the `LightLevels` of the shader module is white. When a running world
/// gains its first light colour other than white, or loses its last one, the
/// def changes. The frame it changes in takes longer while Bevy compiles the
/// shaders for the new def, and the chunks keep drawing.
///
/// # Meshes and vertex shaders
///
/// Light comes from voxels, not from the mesh. A shaft the mesh rounds away
/// still lets light through, and a wall the mesh doesn't draw still blocks it.
/// The shader finds the voxels of a face from its local position and normal.
/// So a custom mesh is built in the padded local coordinates of the chunk, as
/// the [`rendering`](crate::rendering) module docs describe them, with normals
/// pointing out of the solid. A surface sunk more than about half a voxel into
/// solid voxels can read black along convex edges, where it lies more than a
/// voxel from open air.
///
/// A custom vertex shader outputs what the [`rendering`](crate::rendering)
/// module docs list. Light reads these outputs:
///
/// - Location 9 holds the local position, and location 10 the local normal.
/// - Location 11 holds the `MeshTag` of the chunk, which light reads only
///   under the `VOXEL_LIGHT` shader def this extension adds.
/// - Location 12 holds the position in the grid of the world, which light
///   reads only under `SUB_GRIDS`.
///
/// # Sub-grids
///
/// With the `sub_grids` feature, a fragment shader lights any face, of a
/// world chunk or of a sub-grid, with `sample_face_light` from the
/// `bevy_voxel_world::light` module. A face of a sub-grid shows the light of
/// the world where it is, capped by the enclosure of the sub-grid, with the
/// light of its own emissive voxels on top. A shader that still calls
/// `sample_light` lights the faces of a sub-grid from the light of the
/// sub-grid alone. Those faces then don't darken as the sub-grid enters a
/// cave, or brighten as it passes an emissive voxel of the world. The shader
/// reads the Bevy world position as a position in the grid of the world, so
/// it assumes the `WorldRoot` of the world is at the identity.
#[derive(Asset, AsBindGroup, TypePath, Debug, Clone)]
#[bind_group_data(VoxelWorldBindingsKey)]
pub struct VoxelWorldBindings<M: Material> {
    #[texture(200, dimension = "3d", sample_type = "u_int")]
    light_texture: Handle<Image>,
    #[uniform(201)]
    look: LookUniform,
    /// The world's tag map, or a placeholder of one empty entry until the
    /// world has a sub-grid.
    #[cfg(feature = "sub_grids")]
    #[texture(202, sample_type = "u_int")]
    tag_map: Handle<Image>,
    /// The colour texture of the world, or a placeholder of one white texel
    /// until a chunk with a slot shows a light colour other than white.
    #[texture(203, dimension = "3d", sample_type = "u_int")]
    light_colors: Handle<Image>,
    /// Whether the emission table of the world holds a light colour other
    /// than white. It reaches `specialize` through [`VoxelWorldBindingsKey`],
    /// and sets the `LIGHT_COLOR` shader def there.
    light_color: bool,
    _material: PhantomData<fn() -> M>,
}

/// The part of [`VoxelWorldBindings`] that picks its pipeline: whether the
/// `LIGHT_COLOR` shader def is set.
///
/// Bevy gives `specialize` this key and not the material, so a shader def
/// that depends on the world has to travel in it. The type is `pub` because
/// it is the bind group data of the public [`VoxelWorldBindings`], but the
/// crate doesn't re-export it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct VoxelWorldBindingsKey {
    pub(crate) light_color: bool,
}

impl<M: Material> From<&VoxelWorldBindings<M>> for VoxelWorldBindingsKey {
    fn from(bindings: &VoxelWorldBindings<M>) -> Self {
        Self {
            light_color: bindings.light_color,
        }
    }
}

impl<M: Material> VoxelWorldBindings<M> {
    /// The look the shader reads.
    #[cfg(test)]
    pub(crate) fn look(&self) -> &LookUniform {
        &self.look
    }

    /// The colour texture the shader reads.
    #[cfg(test)]
    pub(crate) fn light_colors(&self) -> &Handle<Image> {
        &self.light_colors
    }

    /// The tag map the shader reads.
    #[cfg(all(test, feature = "sub_grids"))]
    pub(crate) fn tag_map(&self) -> &Handle<Image> {
        &self.tag_map
    }
}

impl<M: Material> MaterialExtension for VoxelWorldBindings<M> {
    fn vertex_shader() -> ShaderRef {
        lit_pbr_shader::<M>()
    }

    fn fragment_shader() -> ShaderRef {
        lit_pbr_shader::<M>()
    }

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
        key: MaterialExtensionKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        let mut defs = vec![ShaderDefVal::Bool("VOXEL_LIGHT".into(), true)];
        if cfg!(feature = "sub_grids") {
            defs.push(ShaderDefVal::Bool("SUB_GRIDS".into(), true));
        }
        if key.bind_group_data.light_color {
            defs.push(ShaderDefVal::Bool("LIGHT_COLOR".into(), true));
        }
        descriptor.vertex.shader_defs.extend(defs.iter().cloned());
        if let Some(fragment) = descriptor.fragment.as_mut() {
            fragment.shader_defs.extend(defs);
        }
        Ok(())
    }
}

/// The crate's lit PBR shader if `M` is exactly `StandardMaterial`, whose own
/// shader can't read light, or else `M`'s own shader.
fn lit_pbr_shader<M: Material>() -> ShaderRef {
    if TypeId::of::<M>() == TypeId::of::<StandardMaterial>() {
        LIT_PBR_SHADER_HANDLE.into()
    } else {
        ShaderRef::Default
    }
}

/// A world's [`LightLook`](super::config::LightLook) as the shader reads it, the `LightLook` struct of
/// the `bevy_voxel_world::light` module.
///
/// Each curve is 16 levels as 4 `vec4`s, with level 0 forced to 0, so a
/// face with no light stays black whatever the curve says.
#[derive(Clone, Copy, Debug, PartialEq, ShaderType)]
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
    /// 1 at [`LightPrecision::Steps256`], whose texels and tags hold 8 bits
    /// each for sky light and voxel light, and 0 at
    /// [`LightPrecision::Steps16`].
    pub steps_256: u32,
    /// 1 when a sub-grid's enclosure caps the world's voxel light its faces
    /// show, as it caps the world's sky. 0 without the `sub_grids` feature.
    pub enclosure_caps_voxel: u32,
    /// 1 when the faces of an emissive voxel glow in its light colour, and 0
    /// when they glow white.
    pub tinted_glow: u32,
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
            steps_256: (config.precision == LightPrecision::Steps256) as u32,
            #[cfg(feature = "sub_grids")]
            enclosure_caps_voxel: config.enclosure_caps_voxel_light() as u32,
            #[cfg(not(feature = "sub_grids"))]
            enclosure_caps_voxel: 0,
            tinted_glow: look.tinted_glow as u32,
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

/// The crate's lit PBR shader, for lit worlds whose material is plain
/// `StandardMaterial`.
const LIT_PBR_SHADER_HANDLE: Handle<Shader> =
    uuid_handle!("8d2f4b6a-1c3e-4a57-b9d0-e2f4a6c8b013");

/// Adds the `bevy_voxel_world::light` shader module and the lit PBR shader to
/// `shaders`.
pub(crate) fn add_light_shader(shaders: &mut Assets<Shader>) {
    // An embedded path gives the module its import path.
    let light = Shader::from_wesl(
        include_str!("../shaders/light.wesl"),
        "embedded://bevy_voxel_world/light.wesl",
    );
    shaders.insert(LIGHT_SHADER_HANDLE.id(), light).unwrap();
    let lit_pbr = Shader::from_wesl(
        include_str!("../shaders/lit_pbr.wesl"),
        "embedded://bevy_voxel_world/lit_pbr.wesl",
    );
    shaders.insert(LIT_PBR_SHADER_HANDLE.id(), lit_pbr).unwrap();
}

/// Sets up giving the chunks of lit world `C` its material `base`, wrapped in
/// [`VoxelWorldBindings`] with the light texture of the world, and giving each
/// chunk its tag.
///
/// The route reads the light colours of `world` before the material is made,
/// so a world coloured at startup makes its material with the `LIGHT_COLOR`
/// shader def already in its key.
///
/// With no `base`, the material of the world is added by hand. Each time
/// `VoxelWorldMaterialHandle<M>` names another material, the world wraps a
/// copy of it. Returns the handle of the wrapped material when there is a
/// `base`.
pub(crate) fn add_lit_material_path<C: VoxelWorldConfig, M: Material>(
    app: &mut App,
    base: Option<M>,
    world: &C,
    config: &LightPropagation,
) -> Option<Handle<VoxelWorldMaterial<M>>>
where
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
    let mut route = LightRoute::<C>::new(
        &config,
        super::placeholder_sky(&config),
        &mut app.world_mut().resource_mut::<Assets<Image>>(),
    );
    super::read_light_colors(&mut route, world);
    let look = LookUniform::new(&config);
    #[cfg(feature = "sub_grids")]
    let tag_map =
        TagMapBinding::<C>::new(&mut app.world_mut().resource_mut::<Assets<Image>>());
    let textures = Textures::of(
        &route,
        #[cfg(feature = "sub_grids")]
        &tag_map,
    );
    let color_binding = ColorBinding::<C>::new(&route);
    let handle = base.map(|base| {
        let material = wrap(base, textures, look);
        app.world_mut()
            .resource_mut::<Assets<VoxelWorldMaterial<M>>>()
            .add(material)
    });
    app.insert_resource(route);
    app.insert_resource(color_binding);
    #[cfg(feature = "sub_grids")]
    app.insert_resource(tag_map);
    route::add_render_systems::<C>(app);
    add_first_flip_wait::<C>(app);

    // A hand-added material is wrapped by `wrap_hand_added_material`, not
    // followed: the wrapped handle names one world's copy only.
    add_material_path::<C, VoxelWorldMaterial<M>>(app, handle.clone(), false);
    if handle.is_none() {
        app.insert_resource(HandAddedMaterial::<C, M> {
            wrapped_from: None,
            _world: PhantomData,
        })
        .add_systems(
            Update,
            wrap_hand_added_material::<C, M>
                .before(Internals::<C>::assign_material::<VoxelWorldMaterial<M>>),
        );
    }
    app.add_systems(
        PostUpdate,
        (
            release_texture::<C, M>
                .run_if(|route: Res<LightRoute<C>>| route.has_released())
                .after(super::follow_config::<C>)
                .before(AssetEventSystems),
            make_texture::<C, M>
                .run_if(|route: Res<LightRoute<C>>| route.texture_wanted())
                .after(release_texture::<C, M>),
            route::make_color_texture::<C>
                .run_if(|route: Res<LightRoute<C>>| route.color_texture_wanted())
                .after(make_texture::<C, M>),
            bind_colors::<C, M>
                .after(super::follow_config::<C>)
                .after(route::make_color_texture::<C>)
                .before(AssetEventSystems),
            apply_look::<C, M>
                .run_if(resource_changed::<C>)
                .after(super::follow_config::<C>)
                .after(release_texture::<C, M>)
                .before(AssetEventSystems),
        ),
    );
    // The tag map reads this frame's poses, and the tags every system before
    // it set. A slot it gives a chunk can be the first, which makes the
    // texture. It reads the live sub-grids after those despawning this
    // frame have ended.
    #[cfg(feature = "sub_grids")]
    app.add_systems(
        PostUpdate,
        (
            super::sub_grid::tag_map::upkeep::<C>
                .run_if(super::sub_grid::tag_map::has_upkeep::<C>)
                .after(crate::sub_grid::SubGrids::<C>::snapshot_poses)
                .after(crate::sub_grid::handover::Handovers::<C>::despawn_landed)
                .after(super::follow_config::<C>)
                .before(make_texture::<C, M>),
            super::sub_grid::tag_map::set_changed_tags::<C>
                .run_if(super::sub_grid::tag_map::has_upkeep::<C>)
                .after(super::sub_grid::tag_map::upkeep::<C>),
            bind_tag_map::<C, M>
                .after(super::sub_grid::tag_map::upkeep::<C>)
                .before(AssetEventSystems),
        ),
    );
    app.world_mut()
        .resource_mut::<WorldMaterial<C, VoxelWorldMaterial<M>>>()
        .insert_chunk_components = route::tag_chunk::<C>;
    handle
}

/// The textures the material of a world binds for the shader to read light
/// from, and whether the shader reads light colours.
struct Textures {
    light: Handle<Image>,
    #[cfg(feature = "sub_grids")]
    tag_map: Handle<Image>,
    light_colors: Handle<Image>,
    light_color: bool,
}

impl Textures {
    /// Returns what a material made now binds, for the world that `route`
    /// belongs to.
    fn of<C: VoxelWorldConfig>(
        route: &LightRoute<C>,
        #[cfg(feature = "sub_grids")] tag_map: &TagMapBinding<C>,
    ) -> Self {
        Self {
            light: route.texture(),
            #[cfg(feature = "sub_grids")]
            tag_map: tag_map.bound(route),
            light_colors: route.color_texture(),
            light_color: route.is_colored(),
        }
    }
}

/// `base` wrapped with a world's textures and look.
fn wrap<M: Material>(
    base: M,
    textures: Textures,
    look: LookUniform,
) -> VoxelWorldMaterial<M> {
    ExtendedMaterial {
        base,
        extension: VoxelWorldBindings {
            light_texture: textures.light,
            look,
            #[cfg(feature = "sub_grids")]
            tag_map: textures.tag_map,
            light_colors: textures.light_colors,
            light_color: textures.light_color,
            _material: PhantomData,
        },
    }
}

/// What lit world `C` needs to wrap a material added by hand.
#[derive(Resource)]
struct HandAddedMaterial<C, M: Material> {
    /// The hand-added material the world's wrapped copy was made from.
    wrapped_from: Option<AssetId<M>>,
    _world: PhantomData<C>,
}

/// Gives lit world `C` a wrapped copy of the material
/// `VoxelWorldMaterialHandle<M>` names, once it's loaded, and again whenever
/// the handle names another material.
///
/// Each lit world makes its own copy, so worlds sharing a hand-added handle
/// each bind their own light texture. The copy's handle is also inserted as
/// `VoxelWorldMaterialHandle<VoxelWorldMaterial<M>>`, and later edits go
/// through it: an edit to the hand-added material doesn't reach the copy.
#[allow(clippy::too_many_arguments)]
fn wrap_hand_added_material<C: VoxelWorldConfig, M: Material>(
    mut commands: Commands,
    hand_added: Option<Res<VoxelWorldMaterialHandle<M>>>,
    materials: Option<Res<Assets<M>>>,
    mut wrapped: ResMut<Assets<VoxelWorldMaterial<M>>>,
    mut state: ResMut<HandAddedMaterial<C, M>>,
    mut world_material: ResMut<WorldMaterial<C, VoxelWorldMaterial<M>>>,
    route: Res<LightRoute<C>>,
    settings: Res<LightSettings<C>>,
    #[cfg(feature = "sub_grids")] tag_map: Res<TagMapBinding<C>>,
) {
    let Some(hand_added) = hand_added else {
        return;
    };
    let id = hand_added.handle.id();
    if state.wrapped_from == Some(id) {
        return;
    }
    let Some(base) = materials.and_then(|materials| materials.get(id).cloned()) else {
        return;
    };
    let look = LookUniform::new(&settings.config);
    let textures = Textures::of(
        &route,
        #[cfg(feature = "sub_grids")]
        &tag_map,
    );
    let handle = wrapped.add(wrap(base, textures, look));
    state.wrapped_from = Some(id);
    world_material.handle = Some(handle.clone());
    commands.insert_resource(VoxelWorldMaterialHandle { handle });
}

/// Writes world `C`'s look into the uniform of each of its materials whose
/// look differs from the one the world last applied.
///
/// A world's materials are those binding its light texture, or its
/// placeholder before the texture is made: its material, and any earlier copy
/// of a hand-added material that chunks still show.
///
/// It runs only on frames where the world's config changed, and writes a
/// material only when its look changed, since a write makes Bevy prepare the
/// material's bind group again. It runs before Bevy sends this frame's asset
/// events, so the frame renders the new look.
fn apply_look<C: VoxelWorldConfig, M: Material>(
    settings: Res<LightSettings<C>>,
    route: Res<LightRoute<C>>,
    mut materials: ResMut<Assets<VoxelWorldMaterial<M>>>,
) where
    <VoxelWorldMaterial<M> as AsBindGroup>::Data:
        PartialEq + Eq + std::hash::Hash + Clone,
{
    let look = LookUniform::new(&settings.config);
    let texture = route.texture().id();
    let changed: Vec<_> = materials
        .iter()
        .filter(|(_, material)| {
            material.extension.light_texture.id() == texture
                && material.extension.look != look
        })
        .map(|(id, _)| id)
        .collect();
    for id in changed {
        if let Some(mut material) = materials.get_mut(id) {
            material.extension.look = look;
        }
    }
}

/// Gives world `C`'s materials its placeholder in place of a light texture a
/// change of precision or slot capacity replaced, so the old texture is
/// dropped. The new one is made when a chunk next takes a slot.
fn release_texture<C: VoxelWorldConfig, M: Material>(
    mut route: ResMut<LightRoute<C>>,
    mut materials: ResMut<Assets<VoxelWorldMaterial<M>>>,
) where
    <VoxelWorldMaterial<M> as AsBindGroup>::Data:
        PartialEq + Eq + std::hash::Hash + Clone,
{
    let released: Vec<AssetId<Image>> =
        route.take_released().iter().map(Handle::id).collect();
    let ours: Vec<_> = materials
        .iter()
        .filter(|(_, material)| released.contains(&material.extension.light_texture.id()))
        .map(|(id, _)| id)
        .collect();
    for id in ours {
        if let Some(mut material) = materials.get_mut(id) {
            material.extension.light_texture = route.placeholder.clone();
        }
    }
}

/// Makes world `C`'s light texture, once a chunk first takes a slot, and
/// gives it to the world's material in place of the placeholder.
fn make_texture<C: VoxelWorldConfig, M: Material>(world: &mut World)
where
    <VoxelWorldMaterial<M> as AsBindGroup>::Data:
        PartialEq + Eq + std::hash::Hash + Clone,
{
    let route = world.resource::<LightRoute<C>>();
    let image = route.texture_image();
    let placeholder = route.placeholder.id();
    let texture = world.resource_mut::<Assets<Image>>().add(image);
    world
        .resource_mut::<LightRoute<C>>()
        .texture_made(texture.clone());
    // Each world has its own placeholder, so the materials binding it are this
    // world's: its material, and any earlier copy of a hand-added material
    // that chunks still show.
    let mut materials = world.resource_mut::<Assets<VoxelWorldMaterial<M>>>();
    let ours: Vec<_> = materials
        .iter()
        .filter(|(_, material)| material.extension.light_texture.id() == placeholder)
        .map(|(id, _)| id)
        .collect();
    // Changing a material makes Bevy prepare its bind group again, with the
    // new texture.
    for id in ours {
        if let Some(mut material) = materials.get_mut(id) {
            material.extension.light_texture = texture.clone();
        }
    }
}

/// What the materials of world `C` last bound as its colour texture, and
/// whether they carry the `LIGHT_COLOR` shader def.
#[derive(Resource)]
struct ColorBinding<C> {
    bound: AssetId<Image>,
    light_color: bool,
    /// The materials whose `LIGHT_COLOR` def changed this frame. The render
    /// world waits on their new pipelines.
    first_flip: Vec<UntypedAssetId>,
    _world: PhantomData<C>,
}

impl<C: VoxelWorldConfig> ColorBinding<C> {
    fn new(route: &LightRoute<C>) -> Self {
        Self {
            bound: route.color_texture().id(),
            light_color: route.is_colored(),
            first_flip: Vec::new(),
            _world: PhantomData,
        }
    }
}

/// Returns the materials of world `C` whose `LIGHT_COLOR` shader def changed
/// this frame.
#[cfg(test)]
pub(crate) fn first_flip_materials<C: VoxelWorldConfig>(
    world: &World,
) -> &[UntypedAssetId] {
    &world.resource::<ColorBinding<C>>().first_flip
}

/// Gives the materials of world `C` its current colour texture and
/// `LIGHT_COLOR` shader def, where either differs from what they bind.
///
/// The colour texture changes when it is made or replaced. The def changes
/// when the emission table of the world gains its first light colour other
/// than white, or loses its last one. A world coloured at startup makes its
/// material with the def already set, so only a change of emission while the
/// world runs changes the def here.
///
/// Changing the def changes the material key, so Bevy specializes the
/// pipelines of the material again. The system notes each material whose def
/// changed in `first_flip`, for the render world to wait on. It runs before
/// Bevy sends the asset events of this frame, so the frame renders with the
/// new binding.
fn bind_colors<C: VoxelWorldConfig, M: Material>(
    route: Res<LightRoute<C>>,
    mut binding: ResMut<ColorBinding<C>>,
    mut materials: ResMut<Assets<VoxelWorldMaterial<M>>>,
) where
    <VoxelWorldMaterial<M> as AsBindGroup>::Data:
        PartialEq + Eq + std::hash::Hash + Clone,
{
    binding.first_flip.clear();
    let texture = route.color_texture();
    let light_color = route.is_colored();
    if texture.id() == binding.bound && light_color == binding.light_color {
        return;
    }
    let old = std::mem::replace(&mut binding.bound, texture.id());
    // Every change of the def waits, not only the first. A view or pipeline
    // key that arrived since the last change has never been compiled with the
    // new def. Pipelines already compiled cost the wait nothing.
    let first_flip = light_color != binding.light_color;
    binding.light_color = light_color;
    // Each world has its own placeholder and colour texture, so the materials
    // binding the old texture belong to this world. They are its material, and
    // any earlier copy of a hand-added material that chunks still show.
    let ours: Vec<_> = materials
        .iter()
        .filter(|(_, material)| {
            let bound = material.extension.light_colors.id();
            let ours = bound == old || bound == texture.id();
            ours && (bound != texture.id()
                || material.extension.light_color != light_color)
        })
        .map(|(id, _)| id)
        .collect();
    for id in ours {
        if let Some(mut material) = materials.get_mut(id) {
            if first_flip && material.extension.light_color != light_color {
                binding.first_flip.push(id.untyped());
            }
            material.extension.light_colors = texture.clone();
            material.extension.light_color = light_color;
        }
    }
}

/// The materials of every lit world whose `LIGHT_COLOR` shader def changed
/// this frame, as the render world holds them.
#[derive(Resource, Default)]
struct FirstFlipMaterials(Vec<UntypedAssetId>);

/// Sets up the render world to wait on the new pipelines of the chunks of
/// world `C`, in each frame the `LIGHT_COLOR` shader def of the world changes.
///
/// One wait system serves every lit world, so the first world to call this
/// adds it. Each world adds its own system to extract its materials.
fn add_first_flip_wait<C: VoxelWorldConfig>(app: &mut App) {
    let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
        return;
    };
    if !render_app.world().contains_resource::<FirstFlipMaterials>() {
        render_app
            .init_resource::<FirstFlipMaterials>()
            .add_systems(
                Render,
                wait_on_first_flip
                    .in_set(RenderSystems::Queue)
                    .after(RenderSystems::Specialize),
            );
    }
    render_app.add_systems(ExtractSchedule, extract_first_flip::<C>);
}

/// Copies into the render world the materials of world `C` whose
/// `LIGHT_COLOR` def changed this frame.
fn extract_first_flip<C: VoxelWorldConfig>(
    binding: Extract<Option<Res<ColorBinding<C>>>>,
    mut flipped: ResMut<FirstFlipMaterials>,
) {
    if let Some(binding) = binding.as_ref() {
        flipped.0.extend(binding.first_flip.iter().copied());
    }
}

/// Waits for the pipelines of the main pass, the prepass and shadows that
/// draw the chunks of a material whose `LIGHT_COLOR` def changed this frame.
///
/// The system runs after Bevy specializes the chunks for their new key. It
/// takes the pipeline ids from the Bevy caches of specialized pipelines, for
/// every view.
///
/// On macOS and WASM, Bevy compiles a pipeline in the frame it is queued, so
/// the wait changes nothing there. The gap the wait closes comes from Bevy
/// compiling pipelines in the background on native Linux and Windows.
fn wait_on_first_flip(
    mut flipped: ResMut<FirstFlipMaterials>,
    instances: Res<RenderMaterialInstances>,
    main_pass: Res<SpecializedMaterialPipelineCache>,
    prepass: Res<SpecializedPrepassMaterialPipelineCache>,
    shadows: Res<SpecializedShadowMaterialPipelineCache>,
    mut pipelines: ResMut<PipelineCache>,
) {
    if flipped.0.is_empty() {
        return;
    }
    let flipped_material = |entity: &MainEntity| {
        instances
            .instances
            .get(entity)
            .is_some_and(|instance| flipped.0.contains(&instance.asset_id))
    };
    let main_pass = main_pass.values().flat_map(|view| view.iter());
    let prepass = prepass
        .values()
        .flat_map(|view| view.iter().map(|(entity, (_, id, _))| (entity, id)));
    let shadows = shadows
        .values()
        .flat_map(|view| view.iter().map(|(entity, (id, _, _))| (entity, id)));
    let mut waits: Vec<CachedRenderPipelineId> = main_pass
        .chain(prepass)
        .chain(shadows)
        .filter(|(entity, _)| flipped_material(entity))
        .map(|(_, id)| *id)
        .collect();
    waits.sort_unstable_by_key(|id| id.id());
    waits.dedup();
    for id in waits {
        pipelines.block_on_render_pipeline(id);
    }
    flipped.0.clear();
}

/// What world `C`'s materials bind as its tag map: its placeholder until the
/// world has a sub-grid, then its tag map's texture, a new one each time the
/// tag map grows.
#[cfg(feature = "sub_grids")]
#[derive(Resource)]
struct TagMapBinding<C> {
    /// A texture of one empty entry, which the shader reads as a tag map
    /// where every lookup misses. It has no upkeep.
    placeholder: Handle<Image>,
    /// The texture the world's materials bind now.
    bound: AssetId<Image>,
    _world: PhantomData<C>,
}

#[cfg(feature = "sub_grids")]
impl<C: VoxelWorldConfig> TagMapBinding<C> {
    fn new(images: &mut Assets<Image>) -> Self {
        let placeholder = images.add(super::sub_grid::tag_map::placeholder_image());
        Self {
            bound: placeholder.id(),
            placeholder,
            _world: PhantomData,
        }
    }

    /// The texture a material made now binds: the tag map's, once `route`
    /// has one, or else the placeholder.
    fn bound(&self, route: &LightRoute<C>) -> Handle<Image> {
        route
            .tag_map()
            .map(|map| map.texture().clone())
            .unwrap_or_else(|| self.placeholder.clone())
    }
}

/// Gives world `C`'s materials its tag map's texture in place of the one
/// they bind, when the world makes its tag map and each time the tag map
/// grows into a new texture. The texture they bound is dropped once no
/// material binds it.
///
/// It runs after the tag map's upkeep and before Bevy sends this frame's
/// asset events, so the frame that extracts the new texture's entries
/// renders with it.
#[cfg(feature = "sub_grids")]
fn bind_tag_map<C: VoxelWorldConfig, M: Material>(
    route: Res<LightRoute<C>>,
    mut binding: ResMut<TagMapBinding<C>>,
    mut materials: ResMut<Assets<VoxelWorldMaterial<M>>>,
) where
    <VoxelWorldMaterial<M> as AsBindGroup>::Data:
        PartialEq + Eq + std::hash::Hash + Clone,
{
    let texture = binding.bound(&route);
    if texture.id() == binding.bound {
        return;
    }
    let old = std::mem::replace(&mut binding.bound, texture.id());
    // Each world has its own placeholder and tag map, so the materials binding
    // the old texture are this world's: its material, and any earlier copy
    // of a hand-added material that chunks still show.
    let ours: Vec<_> = materials
        .iter()
        .filter(|(_, material)| material.extension.tag_map.id() == old)
        .map(|(id, _)| id)
        .collect();
    for id in ours {
        if let Some(mut material) = materials.get_mut(id) {
            material.extension.tag_map = texture.clone();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::config::LightLook;
    use super::*;
    use crate::voxel_material::DefaultVoxelMaterial;

    fn is_lit_pbr(shader: ShaderRef) -> bool {
        matches!(shader, ShaderRef::Handle(handle) if handle == LIT_PBR_SHADER_HANDLE)
    }

    #[test]
    fn only_plain_standard_material_gets_the_lit_pbr_shader() {
        type Plain = VoxelWorldBindings<StandardMaterial>;
        assert!(is_lit_pbr(Plain::vertex_shader()));
        assert!(is_lit_pbr(Plain::fragment_shader()));
        assert!(is_lit_pbr(
            VoxelWorldMaterial::<StandardMaterial>::fragment_shader()
        ));

        // An extended `StandardMaterial` keeps its own shaders.
        type Extended = VoxelWorldMaterial<DefaultVoxelMaterial>;
        assert!(!is_lit_pbr(Extended::vertex_shader()));
        assert!(!is_lit_pbr(Extended::fragment_shader()));
        assert!(matches!(
            VoxelWorldBindings::<DefaultVoxelMaterial>::fragment_shader(),
            ShaderRef::Default
        ));
    }

    /// A material with no prepass and no shadows.
    #[derive(Asset, AsBindGroup, TypePath, Debug, Clone)]
    struct Flat {}

    impl Material for Flat {
        fn enable_prepass() -> bool {
            false
        }

        fn enable_shadows() -> bool {
            false
        }
    }

    #[test]
    fn the_wrap_passes_on_prepass_and_shadows_from_the_wrapped_material() {
        assert!(!VoxelWorldMaterial::<Flat>::enable_prepass());
        assert!(!VoxelWorldMaterial::<Flat>::enable_shadows());
        assert!(VoxelWorldMaterial::<StandardMaterial>::enable_prepass());
        assert!(VoxelWorldMaterial::<StandardMaterial>::enable_shadows());
    }

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
        assert!((uniform.sky_falloff[3].z - 0.7).abs() < 1e-6);
        // The default voxel falloff at level 14 is
        // `(0.7 - 0.7^14) / (1 - 0.7^14)`.
        assert!((uniform.voxel_falloff[3].z - 0.697_951).abs() < 1e-6);
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
