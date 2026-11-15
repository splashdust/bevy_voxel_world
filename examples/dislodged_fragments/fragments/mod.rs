//! What a game adds to a world that lifts its dislodged fragments: physics
//! for the pieces, carving, easing a resting piece into the world, outlines
//! when nothing lifts, and a panel for the detection settings.
//!
//! The fragments examples share this module. Each brings its own world, whose
//! config implements [`FragmentsConfig`], and adds [`FragmentsPlugin`].
//! [`PiecesPlugin`] is the part with no window: colliders, physics and
//! stamping, which the tests run on their own.

use std::marker::PhantomData;

use bevy::{ecs::component::Mutable, prelude::*};
use bevy_voxel_world::prelude::*;

mod carve;
mod outline;
mod panel;
mod physics;
mod rest;

#[cfg(test)]
mod tests;

use rest::Pieces;

/// A world config that keeps its [`FragmentDetection`] settings where the
/// panel can change them. Its `detect_dislodged_fragments` returns them.
pub trait FragmentsConfig: VoxelWorldConfig + Component<Mutability = Mutable> {
    fn detection(&self) -> &FragmentDetection<Self::MaterialIndex>;
    fn detection_mut(&mut self) -> &mut FragmentDetection<Self::MaterialIndex>;
}

/// Where each setting shows best in an example's world, as the panel
/// explains it.
#[derive(Clone, Copy)]
pub struct SettingPlaces {
    pub connectivity: &'static str,
    pub limit: &'static str,
    pub auto_lift: &'static str,
}

/// Physics for the pieces, carving on a left click, outlines, and the panel.
pub struct FragmentsPlugin<C> {
    carve_radius: i32,
    places: SettingPlaces,
    _world: PhantomData<fn() -> C>,
}

impl<C> FragmentsPlugin<C> {
    /// A click carves a sphere of `carve_radius` voxels, and the panel names
    /// `places` as where each setting shows.
    pub fn new(carve_radius: i32, places: SettingPlaces) -> Self {
        Self {
            carve_radius,
            places,
            _world: PhantomData,
        }
    }
}

impl<C: FragmentsConfig> Plugin for FragmentsPlugin<C> {
    fn build(&self, app: &mut App) {
        app.add_plugins(PiecesPlugin::<C>::default())
            .insert_resource(carve::CarveRadius(self.carve_radius))
            .add_plugins(panel::PanelPlugin::<C>::new(self.places))
            .add_systems(
                Update,
                (
                    carve::carve_on_click::<C>,
                    outline::keep_outlines::<C>,
                    outline::draw_outlines::<C>,
                )
                    .chain(),
            );
    }
}

/// Colliders for every grid's chunks, a body for each lifted piece, and
/// easing and stamping a piece once it rests.
pub struct PiecesPlugin<C>(PhantomData<fn() -> C>);

impl<C> Default for PiecesPlugin<C> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<C: VoxelWorldConfig> Plugin for PiecesPlugin<C> {
    fn build(&self, app: &mut App) {
        app.init_resource::<Pieces>()
            // Both run before this frame's physics step, in the frame the
            // chunks publish, so a lift's colliders swap in one step. A
            // lifted piece gets its body before its chunks get colliders, so
            // each collider joins that body as it is added.
            .add_systems(
                PreUpdate,
                (physics::start_falling::<C>, physics::rebuild_colliders::<C>)
                    .chain()
                    .after(VoxelWorldSet::<C>::default()),
            )
            .add_systems(
                Update,
                (
                    rest::ease_resting::<C>,
                    rest::count_stamps::<C>,
                    rest::drop_lost::<C>,
                ),
            );
    }
}
