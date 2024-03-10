use std::marker::PhantomData;

use bevy::{prelude::*, render::primitives::Aabb};

use crate::chunk::Chunk;

pub struct VoxelWorldGizmoPlugin<C>(PhantomData<C>);

impl<C: Send + Sync + 'static> Plugin for VoxelWorldGizmoPlugin<C> {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, draw_aabbs::<C>);
    }
}

#[derive(Component, Default)]
pub struct ChunkAabbGizmo {
    pub color: Option<Color>,
}

fn draw_aabbs<C: Send + Sync + 'static>(
    query: Query<(&Chunk<C>, &GlobalTransform, &ChunkAabbGizmo)>,
    mut gizmos: Gizmos,
) {
    for (chunk, &transform, gizmo) in &query {
        let color = gizmo.color.unwrap_or(Color::WHITE);
        gizmos.cuboid(aabb_transform(chunk.aabb(), transform), color);
    }
}

fn aabb_transform(aabb: Aabb, transform: GlobalTransform) -> GlobalTransform {
    transform
        * GlobalTransform::from(
            Transform::from_translation(aabb.center.into())
                .with_scale((aabb.half_extents * 2.).into()),
        )
}
