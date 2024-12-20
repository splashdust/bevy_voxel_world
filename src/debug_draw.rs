use bevy::{ecs::system::SystemParam, prelude::*};
use std::sync::{Arc, RwLock};

use crate::{
    chunk::{Chunk, CHUNK_SIZE_F},
    configuration::VoxelWorldConfig,
    prelude::VoxelWorld,
};

#[derive(Default)]
pub struct VoxelWorldDebugDrawPlugin<C: VoxelWorldConfig> {
    _marker: std::marker::PhantomData<C>,
}

impl<C: VoxelWorldConfig> Plugin for VoxelWorldDebugDrawPlugin<C> {
    fn build(&self, app: &mut App) {
        app.init_gizmo_group::<ChunkGizmos>()
            .add_systems(Startup, setup::<C>)
            .add_systems(Update, (draw_voxel_gizmos::<C>, draw_ray_gizmos::<C>));
    }
}

pub struct VoxelGizmo {
    pub color: Srgba,
    pub pos: IVec3,
}

#[derive(Resource)]
struct VoxelGizmos<C: VoxelWorldConfig> {
    gizmos: Arc<RwLock<Vec<VoxelGizmo>>>,
    _marker: std::marker::PhantomData<C>,
}

pub struct RayGizmo {
    pub ray: Ray3d,
    pub color: Srgba,
}

#[derive(Resource)]
struct RayGizmos<C: VoxelWorldConfig> {
    gizmos: Arc<RwLock<Vec<RayGizmo>>>,
    _marker: std::marker::PhantomData<C>,
}

#[derive(SystemParam)]
pub struct VoxelWorldDebugDraw<'w, C: VoxelWorldConfig> {
    voxel_gizmos: Res<'w, VoxelGizmos<C>>,
    ray_gizmos: Res<'w, RayGizmos<C>>,
}

impl<C: VoxelWorldConfig> VoxelWorldDebugDraw<'_, C> {
    pub fn set_voxel_gizmo(&self, gizmo: VoxelGizmo) {
        self.set_voxel_gizmo_fn()(gizmo);
    }

    pub fn set_voxel_gizmo_fn(&self) -> Arc<dyn Fn(VoxelGizmo) + Send + Sync> {
        let gizmos = self.voxel_gizmos.gizmos.clone();
        Arc::new(move |gizmo| {
            gizmos.write().unwrap().push(gizmo);
        })
    }

    pub fn clear_voxel_gizmo(&self, pos: IVec3) {
        self.clear_voxel_gizmo_fn()(pos);
    }

    pub fn clear_voxel_gizmo_fn(&self) -> Arc<dyn Fn(IVec3) + Send + Sync> {
        let gizmos = self.voxel_gizmos.gizmos.clone();
        Arc::new(move |pos: IVec3| {
            gizmos.write().unwrap().retain(|gizmo| gizmo.pos != pos);
        })
    }

    pub fn clear_all_voxel_gizmos(&self) {
        self.clear_all_voxel_gizmos_fn()();
    }

    pub fn clear_all_voxel_gizmos_fn(&self) -> Arc<dyn Fn() + Send + Sync> {
        let gizmos = self.voxel_gizmos.gizmos.clone();
        Arc::new(move || {
            gizmos.write().unwrap().clear();
        })
    }

    pub fn set_ray_gizmo(&self, gizmo: RayGizmo) {
        self.set_ray_gizmo_fn()(gizmo);
    }

    pub fn set_ray_gizmo_fn(&self) -> Arc<dyn Fn(RayGizmo) + Send + Sync> {
        let gizmos = self.ray_gizmos.gizmos.clone();
        Arc::new(move |gizmo| {
            gizmos.write().unwrap().push(gizmo);
        })
    }

    pub fn clear_ray_gizmo(&self, ray: Ray3d) {
        self.clear_ray_gizmo_fn()(ray);
    }

    pub fn clear_ray_gizmo_fn(&self) -> Arc<dyn Fn(Ray3d) + Send + Sync> {
        let gizmos = self.ray_gizmos.gizmos.clone();
        Arc::new(move |ray| {
            gizmos.write().unwrap().retain(|gizmo| gizmo.ray != ray);
        })
    }

    pub fn clear_all_ray_gizmos(&self) {
        self.clear_all_ray_gizmos_fn()();
    }

    pub fn clear_all_ray_gizmos_fn(&self) -> Arc<dyn Fn() + Send + Sync> {
        let gizmos = self.ray_gizmos.gizmos.clone();
        Arc::new(move || {
            gizmos.write().unwrap().clear();
        })
    }
}

fn setup<C: VoxelWorldConfig>(mut commands: Commands) {
    commands.insert_resource(VoxelGizmos {
        gizmos: Arc::new(RwLock::new(Vec::new())),
        _marker: std::marker::PhantomData::<C>,
    });
    commands.insert_resource(RayGizmos {
        gizmos: Arc::new(RwLock::new(Vec::new())),
        _marker: std::marker::PhantomData::<C>,
    });
}

fn draw_voxel_gizmos<C: VoxelWorldConfig>(mut gizmos: Gizmos, voxel_gizmos: Res<VoxelGizmos<C>>) {
    for gizmo in voxel_gizmos.gizmos.read().unwrap().iter() {
        let pos = gizmo.pos.as_vec3();
        let radius = 0.45;
        let color = gizmo.color;

        Vec3::AXES.iter().for_each(|&axis| {
            gizmos.circle(
                Isometry3d::new(
                    pos - Vec3::ONE * 0.5,
                    Quat::from_rotation_arc(Vec3::Z, axis),
                ),
                radius,
                color,
            );
            gizmos.circle(
                Isometry3d::new(
                    pos + Vec3::ONE * 0.5,
                    Quat::from_rotation_arc(Vec3::Z, -axis),
                ),
                radius,
                color,
            );
        });
    }
}

fn draw_ray_gizmos<C: VoxelWorldConfig>(mut gizmos: Gizmos, ray_gizmos: Res<RayGizmos<C>>) {
    for gizmo in ray_gizmos.gizmos.read().unwrap().iter() {
        gizmos.line(gizmo.ray.origin, gizmo.ray.get_point(10.0), gizmo.color);
    }
}

#[derive(Default, Reflect, GizmoConfigGroup)]
pub struct ChunkGizmos;

/// Add this system to your app to draw cuboid gizmos for non-empy chunks
pub fn debug_draw_chunks<C: VoxelWorldConfig>(
    mut gizmos: Gizmos<ChunkGizmos>,
    chunks: Query<(&Chunk<C>, &GlobalTransform)>,
    voxel_world: VoxelWorld<C>,
) {
    for (chunk, transform) in chunks.iter() {
        let size = Vec3::ONE * CHUNK_SIZE_F;
        let color = Srgba::new(0.0, 1.0, 0.0, 1.0);

        let Some(chunk_data) = voxel_world.get_chunk_data(chunk.position) else {
            continue;
        };

        if chunk_data.is_empty() {
            continue;
        }

        gizmos.cuboid(
            Transform::from(*transform)
                .with_scale(size)
                .with_translation(transform.translation() + 15.0),
            color,
        );
    }
}
