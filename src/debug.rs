use std::marker::PhantomData;

use bevy::{prelude::*, render::primitives::Aabb};

use crate::chunk::Chunk;

pub struct VoxelWorldGizmoPlugin<I>(PhantomData<I>);

impl<I: Send + Sync + 'static> Plugin for VoxelWorldGizmoPlugin<I> {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, draw_aabbs::<I>);
    }
}

#[derive(Component, Default)]
pub struct ChunkAabbGizmo {
    pub color: Option<Color>,
}

fn draw_aabbs<I: Send + Sync + 'static>(
    query: Query<(&Chunk<I>, &GlobalTransform, &ChunkAabbGizmo)>,
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
