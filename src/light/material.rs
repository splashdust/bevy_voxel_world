//! The material of a lit world that shows its light: the world's material
//! wrapped in the crate's own extension, [`VoxelWorldBindings`].
//!
//! # The material wrap
//!
//! A world names one material, `M`, with `with_material`, or gets the
//! crate's default. When the world is lit and shows its light, its chunks
//! don't get `M` but [`VoxelWorldMaterial<M>`],
//! `ExtendedMaterial<M, VoxelWorldBindings<M>>`, one per lit world. The
//! world's Rust code doesn't change, and only its shader imports light. A
//! world whose material is added by hand, as `VoxelWorldMaterialHandle<M>`,
//! gets a wrapped copy of that material, so two lit worlds naming the same
//! handle each bind their own light texture. An unlit world, one with
//! `shown: false`, and a `VoxelWorldPlugin::minimal()` world keep their
//! material unwrapped, or get none.
//!
//! The extension holds what the shader needs to read light, the world's light
//! texture and its look, at bindings the crate keeps for itself, 200 to 219.
//! It has no shader of its own, so the world's material keeps its shaders, and
//! a shader that wants light imports the `bevy_voxel_world::light` module,
//! which declares those bindings. The one exception is a world whose material
//! is exactly `StandardMaterial`: Bevy's shader for it can't read light, so
//! the extension picks the crate's lit PBR shader, which can. The extension
//! adds the `VOXEL_LIGHT` shader def, under which the crate's vertex shaders
//! also output the chunk's tag. That module reads the tag with the local
//! position and normal they output in every world.
//!
//! With the `sub_grids` feature the extension also binds the world's tag map
//! at 202, a placeholder of one empty entry until the world has a sub-grid,
//! and adds the `SUB_GRIDS` shader def, under which the crate's vertex
//! shaders also output the face's position in the world's grid. The shader
//! reads the world's light at a sub-grid's faces from both: see
//! [Sub-grids](#sub-grids). The tag map's texture is replaced each time it
//! grows, and the world's materials bind the new one in the same frame, as
//! they do the light texture once it's made.
//!
//! `ExtendedMaterial` takes `enable_prepass` and `enable_shadows` from the
//! extension alone, so the extension passes on the wrapped material's. It
//! takes the opaque render method from the wrapped material, so the extension
//! can't force forward rendering. Light adds to the emissive channel only in
//! the forward pass, so a lit chunk material must render forward.
//!
//! # Reading light on any mesh
//!
//! Light comes from voxels, and the shader finds a face's voxels from the
//! face's position and normal alone, so a mesher emits nothing for light. A
//! face on the voxel grid, as every cube face is, reads the voxel in front of
//! it, half a voxel along its normal. A face off the grid, such as a slope or
//! a rounded corner of a marching cubes mesh, can cut into a solid voxel, so
//! the voxel half a voxel along its normal may be solid and read black. It
//! reads the 8 voxels around that point instead, leaving out those holding no
//! light and those not joined through counted voxels to the nearest of the 8,
//! so light doesn't pass through a corner between two solids. A cube world
//! reads exactly as before.
//!
//! A face glows when the voxel half a voxel behind it is a lamp. A lamp's own
//! cell holds the top voxel level, which no open voxel holds, with the lamp's
//! level in its sky half, which a solid never uses. A face off the grid can
//! have an open voxel lit by the lamp behind it, as a rounded inside corner
//! beside a lamp does, and it doesn't glow, because only a lamp's cell holds
//! the top level.
//!
//! # The sun gate
//!
//! By default Bevy's directional lights shine only on faces
//! with direct sky. Bevy's shadow map can't be trusted for this, because it
//! holds only spawned chunks, and under the default streaming rock out of
//! view isn't spawned, so the sun would shine into caves through it. Direct
//! sky falls straight down, so the sun stops at an overhang's drip line.
//! A face reads light blended between voxels, so the gate is a step between
//! direct sky and the level beside it, worked out from sky reach and
//! precision.
//!
//! # Sub-grids
//!
//! With the `sub_grids` feature, a sub-grid's chunk holds light of its own,
//! which the world's light never sees, and its tag has bit 30 set. The
//! shader's `sample_face_light` reads such a face's own light with
//! `sample_light`, and the world's light at the face with `world_light`,
//! from its position and normal in the world's grid. The face shows the smaller of the two sky levels, so a sub-grid's
//! enclosure caps the world's sky, and the larger of the two voxel levels, so
//! its own lamps add to the world's. The sun gate reads that capped sky. A
//! world chunk's face reads as `sample_light` reads it.
//!
//! `world_light` reads as `sample_light` would for a world face at the same
//! place: a face on the world's grid, as every face of a sub-grid at a pose
//! a stamp keeps is, reads the voxel in front of it, and any other face the
//! 8 voxels around the point in front of it. So a sub-grid at such a pose,
//! with sky of its own at the top level and no lamps, looks as its voxels
//! would set in the world. It looks up one world chunk in the world's tag
//! map, kept by the `tag_map` module: the chunk holding the voxel in front
//! of the face, the nearest of those it reads. It reads every voxel from
//! that chunk's slot, whose shell holds the cells across its faces. A
//! coarse chunk's entry is read in its own cells, `1 << L` voxels wide. A position the tag map doesn't hold reads
//! open sky with no lamps, so a sub-grid far from any camera never goes
//! black.
//!
//! The shader takes Bevy's world position as a position in the world's
//! grid, so it assumes the world's `WorldRoot` is at the identity. A custom
//! shader that still calls `sample_light` lights a sub-grid's faces from the
//! sub-grid's own light alone.
//!
//! What it costs grows with:
//!
//! - On the GPU, each fragment of a sub-grid's face: one lookup in the tag
//!   map's hash table, and one more read's cells. A world chunk's face costs
//!   nothing more, and each vertex of a lit world carries one more output.
//! - On the CPU, the world chunks that sub-grids' boxes touch, which a
//!   tumbling sub-grid's box touches more of.
//! - Slots, for the chunks of air near sub-grids whose light varies.
//! - Texture memory, 16 bytes for each entry of the tag map.
//! - One column answer for each position of hinted air a sub-grid comes to
//!   touch.
//!
//! Nothing hides where a sub-grid's light and the world's differ. A
//! sub-grid's lamps light only the sub-grid, and its voxels cast no sky
//! shadow on the world, though Bevy's shadow map still draws its shadow
//! from the sun. A lamp that should light the world around it can carry a
//! Bevy point light as a child of the sub-grid's entity, which moves with
//! it:
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
//! - [Terasology's `BlockMeshPart`](https://github.com/MovingBlocks/Terasology/blob/develop/engine/src/main/java/org/terasology/engine/world/block/shapes/BlockMeshPart.java):
//!   lights a vertex from 8 points around it, so a vertex anywhere gets light.
//! - [Cubyz's lighting](https://github.com/PixelGuys/Cubyz/blob/master/src/renderer/lighting.zig):
//!   blends the light of the 8 cells around a vertex off the grid.
//! - [Luanti's smooth lighting](https://github.com/luanti-org/luanti/blob/master/src/client/content_mapblock.cpp):
//!   averages the cells around a corner, leaving out cells that don't carry
//!   light.
//! - [tesserae's light propagation](https://github.com/mkappworks-dev/tesserae/pull/3):
//!   sky light dims Bevy's ambient light and sun, and voxel light adds through
//!   the emissive channel.

use std::any::TypeId;
use std::marker::PhantomData;

use bevy::{
    asset::{uuid_handle, AssetEventSystems},
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

/// The crate's extension of a lit world's material: the world's light texture
/// and how light looks, for the `bevy_voxel_world::light` shader module.
///
/// It passes on `enable_prepass` and `enable_shadows` from `M`, and adds the
/// `VOXEL_LIGHT` shader def. It has no shader of its own, except when `M` is
/// exactly `StandardMaterial`, which gets the crate's lit PBR shader. That
/// shader draws any mesh, with or without the crate's texture index, and
/// vertex colours tint the base colour as in Bevy's own shader. The look's
/// [`LightAo`] setting doesn't apply to it, since the vertex colour belongs
/// to the mesher. Any other `M`, `ExtendedMaterial<StandardMaterial, _>`
/// included, reads light in its own shader by importing the
/// `bevy_voxel_world::light` module, which declares bindings 200 and 201.
/// With the `sub_grids` feature it also binds the world's tag map at 202 and
/// adds the `SUB_GRIDS` shader def. Bindings 200 to 219 are the crate's.
///
/// # Meshes and vertex shaders
///
/// Light comes from voxels, not from the mesh: a shaft the mesh rounds away
/// still lets light through, and a wall it doesn't draw still blocks it. The
/// shader finds a face's voxels from its local position and normal, so a
/// custom mesh is built in the chunk's padded local coordinates, as the
/// [`rendering`](crate::rendering) module docs describe them, with normals
/// pointing out of the solid. A surface sunk more than about half a voxel
/// into solid voxels can read black along convex edges, where it lies more
/// than a voxel from open air.
///
/// A custom vertex shader outputs what the [`rendering`](crate::rendering)
/// module docs list. Light reads the local position at location 9, the local
/// normal at 10 and the chunk's `MeshTag` at 11, the last only under the
/// `VOXEL_LIGHT` shader def this extension adds, and under `SUB_GRIDS` the
/// position in the world's grid at 12.
///
/// # Sub-grids
///
/// With the `sub_grids` feature, a fragment shader lights any face, a world
/// chunk's or a sub-grid's, with the `bevy_voxel_world::light` module's
/// `sample_face_light`. A sub-grid's face shows the world's light where it
/// is, with the world's sky capped by its own and its own lamps on top. A
/// shader that still calls `sample_light` lights a sub-grid's faces from the
/// sub-grid's own light alone, so they don't darken as the sub-grid enters a
/// cave or brighten as it passes a lamp of the world. The shader reads Bevy's
/// world position as a position in the world's grid, so it assumes the
/// world's `WorldRoot` is at the identity.
#[derive(Asset, AsBindGroup, TypePath, Debug, Clone)]
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
    _material: PhantomData<fn() -> M>,
}

impl<M: Material> VoxelWorldBindings<M> {
    /// The look the shader reads.
    #[cfg(test)]
    pub(crate) fn look(&self) -> &LookUniform {
        &self.look
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
        _key: MaterialExtensionKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        let mut defs = vec![ShaderDefVal::Bool("VOXEL_LIGHT".into(), true)];
        if cfg!(feature = "sub_grids") {
            defs.push(ShaderDefVal::Bool("SUB_GRIDS".into(), true));
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
    /// per channel, 0 at 16 levels.
    pub steps_256: u32,
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

/// Sets up giving the chunks of lit world `C` its material `base` wrapped in
/// [`VoxelWorldBindings`], with the world's light texture, and the crate's tag.
///
/// With no `base`, the world's material is added by hand: each time
/// `VoxelWorldMaterialHandle<M>` names another material, the world wraps a
/// copy of it. Returns the wrapped material's handle when there's a `base`.
pub(crate) fn add_lit_material_path<C: VoxelWorldConfig, M: Material>(
    app: &mut App,
    base: Option<M>,
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
    let route = LightRoute::<C>::new(
        &config,
        super::placeholder_sky(&config),
        &mut app.world_mut().resource_mut::<Assets<Image>>(),
    );
    let look = LookUniform::new(&config);
    #[cfg(feature = "sub_grids")]
    let tag_map =
        TagMapBinding::<C>::new(&mut app.world_mut().resource_mut::<Assets<Image>>());
    let textures = Textures {
        light: route.texture(),
        #[cfg(feature = "sub_grids")]
        tag_map: tag_map.bound(&route),
    };
    let handle = base.map(|base| {
        let material = wrap(base, textures, look);
        app.world_mut()
            .resource_mut::<Assets<VoxelWorldMaterial<M>>>()
            .add(material)
    });
    app.insert_resource(route);
    #[cfg(feature = "sub_grids")]
    app.insert_resource(tag_map);
    route::add_render_systems::<C>(app);

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
            apply_look::<C, M>
                .run_if(resource_changed::<C>)
                .after(super::follow_config::<C>)
                .after(release_texture::<C, M>)
                .before(AssetEventSystems),
        ),
    );
    // The tag map reads this frame's poses, and the tags every system before
    // it set. A slot it gives a chunk can be the first, which makes the
    // texture.
    #[cfg(feature = "sub_grids")]
    app.add_systems(
        PostUpdate,
        (
            super::tag_map::upkeep::<C>
                .after(crate::sub_grid::SubGrids::<C>::snapshot_poses)
                .after(super::follow_config::<C>)
                .before(make_texture::<C, M>),
            bind_tag_map::<C, M>
                .after(super::tag_map::upkeep::<C>)
                .before(AssetEventSystems),
        ),
    );
    app.world_mut()
        .resource_mut::<WorldMaterial<C, VoxelWorldMaterial<M>>>()
        .insert_chunk_components = route::tag_chunk::<C>;
    handle
}

/// The textures a world's material binds for the shader to read light.
struct Textures {
    light: Handle<Image>,
    #[cfg(feature = "sub_grids")]
    tag_map: Handle<Image>,
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
    let textures = Textures {
        light: route.texture(),
        #[cfg(feature = "sub_grids")]
        tag_map: tag_map.bound(&route),
    };
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
        let placeholder = images.add(super::tag_map::placeholder_image());
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
