use std::marker::PhantomData;
use std::sync::Arc;

#[cfg(feature = "light_propagation")]
use bevy::render::render_resource::AsBindGroup;
use bevy::{asset::load_internal_asset, ecs::system::EntityCommands, prelude::*};

use crate::{
    chunk_lifecycle::ChunkLifecycle,
    configuration::{DefaultWorld, VoxelWorldConfig},
    voxel_material::{
        add_shaders, default_material, DefaultVoxelMaterial, VOXEL_TEXTURE_SHADER_HANDLE,
    },
    voxel_world::*,
    voxel_world_internal::{Internals, SpawnMeshes},
};

/// The handle of a custom material of type `M`.
///
/// A world whose `VoxelWorldConfig::init_custom_materials` returns false gives
/// its chunks this handle, which you add yourself. All such worlds with material
/// type `M` share it.
///
/// For any other world with a custom material, `VoxelWorldPlugin` creates the
/// material and adds this handle for you to read. The chunks of that world keep
/// the material created for them, and if several such worlds have material type
/// `M`, this is the handle of the last plugin built.
///
/// With the `light_propagation` feature, a lit world that shows its light
/// gives its chunks `VoxelWorldMaterial<M>`, its material wrapped in the
/// crate's light bindings, and adds `VoxelWorldMaterialHandle` of that type,
/// whose `.base` is the world's material. A lit world whose material is added
/// by hand wraps a copy of the material named here and adds the copy's handle
/// the same way, so later edits go through that handle.
#[derive(Resource)]
pub struct VoxelWorldMaterialHandle<M: Material> {
    pub handle: Handle<M>,
}

/// The material the chunks of world `C` get, and what else they get with it.
#[derive(Resource)]
pub(crate) struct WorldMaterial<C, M: Material> {
    /// The material handle. A world that doesn't init its custom material has
    /// none until `VoxelWorldMaterialHandle<M>` is added, and then follows it.
    pub handle: Option<Handle<M>>,
    /// Inserts extra components on a chunk entity alongside its `Mesh3d` and
    /// `MeshMaterial3d`. It inserts nothing unless something sets it for this
    /// world.
    pub insert_chunk_components: fn(&mut EntityCommands),
    _world: PhantomData<C>,
}

impl<C, M: Material> WorldMaterial<C, M> {
    fn new(handle: Option<Handle<M>>) -> Self {
        Self {
            handle,
            insert_chunk_components: |_| {},
            _world: PhantomData,
        }
    }

    /// Takes the handle of `VoxelWorldMaterialHandle<M>`, for a world whose
    /// material is added by hand.
    fn follow_hand_added_material(
        mut world_material: ResMut<Self>,
        hand_added: Option<Res<VoxelWorldMaterialHandle<M>>>,
    ) where
        C: Send + Sync + 'static,
    {
        let Some(hand_added) = hand_added else {
            return;
        };
        if world_material.handle.as_ref() != Some(&hand_added.handle) {
            world_material.handle = Some(hand_added.handle.clone());
        }
    }
}

/// Sets up giving the chunks of world `C` material `M`.
///
/// With no `handle`, the chunks wait for a handle: with `follow_hand_added`,
/// the one `VoxelWorldMaterialHandle<M>` names, which worlds naming the same
/// handle share; without it, one the caller sets on `WorldMaterial<C, M>`.
pub(crate) fn add_material_path<C: VoxelWorldConfig, M: Material>(
    app: &mut App,
    handle: Option<Handle<M>>,
    follow_hand_added: bool,
) {
    let assign_material = Internals::<C>::assign_material::<M>;
    if handle.is_some() || !follow_hand_added {
        app.add_systems(Update, assign_material);
    } else {
        app.add_systems(
            Update,
            (
                WorldMaterial::<C, M>::follow_hand_added_material,
                assign_material,
            )
                .chain(),
        );
    }
    app.insert_resource(WorldMaterial::<C, M>::new(handle));
}

/// Sets up giving the chunks of world `C` the default material, wrapped in
/// the crate's light bindings if the world shows its light.
fn add_default_material_path<C: VoxelWorldConfig>(app: &mut App, config: &C) {
    let material = default_material::<C>(app, config);
    #[cfg(feature = "light_propagation")]
    if let Some(light) = config.light_propagation().filter(|light| light.shown) {
        crate::light::material::add_lit_material_path::<C, _>(
            app,
            Some(material),
            &light,
        );
        return;
    }
    if app
        .get_added_plugins::<MaterialPlugin<DefaultVoxelMaterial>>()
        .is_empty()
    {
        app.add_plugins(MaterialPlugin::<DefaultVoxelMaterial>::default());
    }
    let handle = app
        .world_mut()
        .resource_mut::<Assets<DefaultVoxelMaterial>>()
        .add(material);
    add_material_path::<C, DefaultVoxelMaterial>(app, Some(handle), false);
}

/// Sets up giving the chunks of world `C` its custom material `material`.
///
/// A world that inits its custom material adds it and inserts its handle; any
/// other waits for `VoxelWorldMaterialHandle<M>` to be added by hand.
fn add_custom_material_path<C: VoxelWorldConfig, M: Material>(
    app: &mut App,
    config: &C,
    material: &M,
) {
    let handle = config.init_custom_materials().then(|| {
        let handle = app
            .world_mut()
            .resource_mut::<Assets<M>>()
            .add(material.clone());
        app.insert_resource(VoxelWorldMaterialHandle {
            handle: handle.clone(),
        });
        handle
    });
    add_material_path::<C, M>(app, handle, true);
}

/// [`add_custom_material_path`], wrapping the material in the crate's light
/// bindings if the world shows its light.
#[cfg(feature = "light_propagation")]
fn add_custom_material_path_or_lit<C: VoxelWorldConfig, M: Material>(
    app: &mut App,
    config: &C,
    material: &M,
) where
    <crate::light::material::VoxelWorldMaterial<M> as AsBindGroup>::Data:
        PartialEq + Eq + std::hash::Hash + Clone,
{
    let Some(light) = config.light_propagation().filter(|light| light.shown) else {
        add_custom_material_path(app, config, material);
        return;
    };
    let base = config.init_custom_materials().then(|| material.clone());
    let handle = crate::light::material::add_lit_material_path::<C, M>(app, base, &light);
    if let Some(handle) = handle {
        app.insert_resource(VoxelWorldMaterialHandle { handle });
    }
}

/// The main plugin for the voxel world. This plugin sets up the voxel world and its dependencies.
/// The type parameter `C` is used to differentiate between different voxel worlds with different configs.
pub struct VoxelWorldPlugin<C, M = StandardMaterial>
where
    C: VoxelWorldConfig,
    M: Material,
{
    spawn_meshes: bool,
    use_custom_material: bool,
    config: C,
    material: M,
}

impl<C> VoxelWorldPlugin<C, StandardMaterial>
where
    C: VoxelWorldConfig,
{
    pub fn with_config(config: C) -> Self {
        Self {
            config,
            spawn_meshes: true,
            use_custom_material: false,
            material: StandardMaterial::default(),
        }
    }

    pub fn minimal() -> Self {
        Self {
            spawn_meshes: false,
            use_custom_material: false,
            config: C::default(),
            material: StandardMaterial::default(),
        }
    }

    /// [`minimal`](Self::minimal) with `config` in place of the default.
    #[cfg(all(test, feature = "light_propagation"))]
    pub(crate) fn minimal_with_config(config: C) -> Self {
        Self {
            config,
            ..Self::minimal()
        }
    }
}

impl<C, M> VoxelWorldPlugin<C, M>
where
    C: VoxelWorldConfig,
    M: Material,
{
    /// Use this to tell `bevy_voxel_world` to use a custom material. This can be any material that
    /// implements `bevy::pbr::Material`. You can use this to create custom shaders for your voxel
    /// world. You can set this up like any other material in Bevy.
    ///
    /// `bevy_voxel_world` will add the material as an asset, so you can query for it later using
    /// `Res<Assets<MyCustomVoxelMaterialType>>`.
    pub fn with_material<CustomMaterial: Material>(
        self,
        material: CustomMaterial,
    ) -> VoxelWorldPlugin<C, CustomMaterial> {
        VoxelWorldPlugin {
            spawn_meshes: self.spawn_meshes,
            use_custom_material: true,
            config: self.config,
            material,
        }
    }
}

impl Default for VoxelWorldPlugin<DefaultWorld, StandardMaterial> {
    fn default() -> Self {
        Self {
            spawn_meshes: true,
            use_custom_material: false,
            config: DefaultWorld,
            material: StandardMaterial::default(),
        }
    }
}

#[cfg(not(feature = "light_propagation"))]
impl<C, M> Plugin for VoxelWorldPlugin<C, M>
where
    C: VoxelWorldConfig,
    M: Material,
{
    fn build(&self, app: &mut App) {
        self.build_world(app, add_custom_material_path::<C, M>);
    }
}

/// With light, a world's custom material `M` may be wrapped in the crate's
/// light bindings, and the wrapped material's bind group data must be
/// hashable like any material's. That holds when `M`'s is, and is `Copy`.
#[cfg(feature = "light_propagation")]
impl<C, M> Plugin for VoxelWorldPlugin<C, M>
where
    C: VoxelWorldConfig,
    M: Material,
    <crate::light::material::VoxelWorldMaterial<M> as AsBindGroup>::Data:
        PartialEq + Eq + std::hash::Hash + Clone,
{
    fn build(&self, app: &mut App) {
        self.build_world(app, add_custom_material_path_or_lit::<C, M>);
    }
}

impl<C, M> VoxelWorldPlugin<C, M>
where
    C: VoxelWorldConfig,
    M: Material,
{
    /// Builds the world, setting up a custom material with
    /// `add_custom_material_path` if it has one.
    fn build_world(&self, app: &mut App, add_custom_material_path: fn(&mut App, &C, &M)) {
        app.init_resource::<C>()
            .insert_resource(ConfigCopy(Arc::new(self.config.clone())))
            .add_systems(First, ConfigCopy::<C>::follow.run_if(resource_changed::<C>))
            // A config inserted after the plugin replaces the one it was built
            // with, so the copy follows it before any startup system runs.
            .add_systems(PreStartup, (ConfigCopy::<C>::follow, Internals::<C>::setup))
            .insert_resource(SpawnMeshes::<C>::new(self.spawn_meshes))
            .add_systems(
                PreUpdate,
                (
                    ChunkLifecycle::<C>::forget_despawned_chunks,
                    Internals::<C>::commit_voxel_edits,
                    // Finished tasks are published before the chunks are streamed, so
                    // spawning rays see this frame's results, and before tasks start,
                    // so a chunk whose task finished gets its follow-up task, and the
                    // task pool its next tasks, without waiting for the next frame.
                    ChunkLifecycle::<C>::collect_chunk_tasks,
                    (
                        ChunkLifecycle::<C>::flush_chunk_map_buffers,
                        ChunkLifecycle::<C>::flush_mesh_cache_buffers,
                    ),
                    Internals::<C>::stream_chunks,
                    ChunkLifecycle::<C>::apply_requests,
                    // Starting tasks needs the chunks spawned above. No other step
                    // depends on an earlier step's commands, so this is the only sync
                    // point the set needs.
                    ApplyDeferred,
                    ChunkLifecycle::<C>::start_chunk_tasks,
                    // Flushing again removes the despawned chunks from the chunk map
                    // this frame.
                    (
                        ChunkLifecycle::<C>::despawn_retired_chunks,
                        ChunkLifecycle::<C>::flush_chunk_map_buffers,
                    )
                        .chain(),
                )
                    .chain_ignore_deferred()
                    .in_set(VoxelWorldSet::<C>::default()),
            )
            .add_message::<ChunkWillSpawn<C>>()
            .add_message::<ChunkPublished<C>>()
            .add_message::<ChunkWillDespawn<C>>()
            .add_message::<ChunkWillRemesh<C>>()
            .add_message::<ChunkWillUpdate<C>>();

        // Spawning of meshes is optional, mainly to simplify testing.
        // This makes voxel_world work with a MinimalPlugins setup.
        if self.spawn_meshes {
            load_internal_asset!(
                app,
                VOXEL_TEXTURE_SHADER_HANDLE,
                "shaders/voxel_texture.wesl",
                Shader::from_wesl
            );
            add_shaders(&mut app.world_mut().resource_mut::<Assets<Shader>>());

            if self.use_custom_material {
                add_custom_material_path(app, &self.config, &self.material);
            } else {
                add_default_material_path(app, &self.config);
            }
        }

        #[cfg(feature = "light_propagation")]
        crate::light::build(app, &self.config);

        app.insert_resource(self.config.clone());
    }
}
