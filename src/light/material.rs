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
//! output what that module reads.
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

use std::any::TypeId;
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
    shader::{Shader, ShaderDefVal, ShaderRef},
};

use super::config::{
    LevelCurve, LightAo, LightLook, LightPropagation, LightSmoothing, SunGate,
};
use super::route::{self, LightRoute};
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
/// Bindings 200 to 219 are the crate's.
///
/// # Meshes and vertex shaders
///
/// Light comes from voxels, not from the mesh: a shaft the mesh rounds away
/// still lets light through, and a wall it doesn't draw still blocks it. The
/// shader finds a face's voxels from its position and normal, so a custom
/// mesh uses the chunk's padded local coordinates, where voxel `i` spans
/// `i..i + 1`, as the crate's mesher does, with normals pointing out of the
/// solid. A surface sunk more than about half a voxel into solid voxels can
/// read black along convex edges, where it lies more than a voxel from open
/// air.
///
/// A custom vertex shader must output, under the `VOXEL_LIGHT` shader def,
/// the vertex's local position at location 9 with
/// `@interpolate(perspective, centroid)`, its local normal at 10, and the
/// chunk's `MeshTag` at 11 with `@interpolate(flat)`, as
/// `bevy_voxel_world::voxel_io::VoxelVertexOutput` does. The position must be
/// centroid-sampled because with MSAA a pixel centre can lie outside its
/// triangle, and the position there would read the light of a voxel beyond
/// the face.
#[derive(Asset, AsBindGroup, TypePath, Debug, Clone)]
pub struct VoxelWorldBindings<M: Material> {
    #[texture(200, dimension = "3d", sample_type = "u_int")]
    light_texture: Handle<Image>,
    #[uniform(201)]
    look: LookUniform,
    _material: PhantomData<fn() -> M>,
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
        let def = ShaderDefVal::Bool("VOXEL_LIGHT".into(), true);
        descriptor.vertex.shader_defs.push(def.clone());
        if let Some(fragment) = descriptor.fragment.as_mut() {
            fragment.shader_defs.push(def);
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

/// A world's [`LightLook`] as the shader reads it, the `LightLook` struct of
/// the `bevy_voxel_world::light` module.
///
/// Each curve is 16 levels as 4 `vec4`s, with level 0 forced to 0, so a
/// face with no light stays black whatever the curve says.
#[derive(Clone, Copy, Debug, ShaderType)]
pub(crate) struct LookUniform {
    pub sky_falloff: [Vec4; 4],
    pub voxel_falloff: [Vec4; 4],
    /// How much of Bevy's directional lights reaches each sky light level.
    pub sun_falloff: [Vec4; 4],
    pub voxel_brightness: f32,
    pub emissive_glow: f32,
    pub minimum_brightness: f32,
    /// 1 for [`LightSmoothing::Smoothed`], 0 for stepped.
    pub smoothed: u32,
    /// 0 for [`LightAo::IndirectLight`], 1 for the base colour, 2 for off.
    pub ao: u32,
}

impl LookUniform {
    pub fn new(look: &LightLook) -> Self {
        let sun = match &look.sun {
            SunGate::DirectSkyOnly => {
                let mut brightness = [0.0; 16];
                brightness[15] = 1.0;
                LevelCurve { brightness }
            }
            SunGate::ByLevel(curve) => curve.clone(),
        };
        Self {
            sky_falloff: table(&look.sky_falloff),
            voxel_falloff: table(&look.voxel_falloff),
            sun_falloff: table(&sun),
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
    let look = LookUniform::new(&config.look);
    let handle = base.map(|base| {
        let material = wrap(base, route.texture(), look);
        app.world_mut()
            .resource_mut::<Assets<VoxelWorldMaterial<M>>>()
            .add(material)
    });
    app.insert_resource(route);
    route::add_render_systems::<C>(app);

    // A hand-added material is wrapped by `wrap_hand_added_material`, not
    // followed: the wrapped handle names one world's copy only.
    add_material_path::<C, VoxelWorldMaterial<M>>(app, handle.clone(), false);
    if handle.is_none() {
        app.insert_resource(HandAddedMaterial::<C, M> {
            look,
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
        make_texture::<C, M>.run_if(|route: Res<LightRoute<C>>| route.texture_wanted()),
    );
    app.world_mut()
        .resource_mut::<WorldMaterial<C, VoxelWorldMaterial<M>>>()
        .insert_chunk_components = route::tag_chunk::<C>;
    handle
}

/// `base` wrapped with a world's light texture and look.
fn wrap<M: Material>(
    base: M,
    light_texture: Handle<Image>,
    look: LookUniform,
) -> VoxelWorldMaterial<M> {
    ExtendedMaterial {
        base,
        extension: VoxelWorldBindings {
            light_texture,
            look,
            _material: PhantomData,
        },
    }
}

/// What lit world `C` needs to wrap a material added by hand.
#[derive(Resource)]
struct HandAddedMaterial<C, M: Material> {
    look: LookUniform,
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
fn wrap_hand_added_material<C: VoxelWorldConfig, M: Material>(
    mut commands: Commands,
    hand_added: Option<Res<VoxelWorldMaterialHandle<M>>>,
    materials: Option<Res<Assets<M>>>,
    mut wrapped: ResMut<Assets<VoxelWorldMaterial<M>>>,
    mut state: ResMut<HandAddedMaterial<C, M>>,
    mut world_material: ResMut<WorldMaterial<C, VoxelWorldMaterial<M>>>,
    route: Res<LightRoute<C>>,
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
    let handle = wrapped.add(wrap(base, route.texture(), state.look));
    state.wrapped_from = Some(id);
    world_material.handle = Some(handle.clone());
    commands.insert_resource(VoxelWorldMaterialHandle { handle });
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

#[cfg(test)]
mod tests {
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

        let mut look = LightLook::default();
        look.sky_falloff.brightness[0] = 0.5;
        look.voxel_falloff.brightness[0] = 0.5;
        let uniform = LookUniform::new(&look);
        assert_eq!(uniform.sky_falloff[0].x, 0.0);
        assert_eq!(uniform.voxel_falloff[0].x, 0.0);
        assert_eq!(uniform.sun_falloff[0].x, 0.0);
        assert_eq!(uniform.voxel_falloff[3].w, 1.0);
        assert!((uniform.voxel_falloff[3].z - 0.7).abs() < 1e-6);
    }
}
