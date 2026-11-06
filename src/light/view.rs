//! The world view: each loaded level-0 chunk's light and what its neighbours'
//! solves read of it, shared with the tasks that solve light.
//!
//! Chunk tasks and relight tasks read their neighbours from it off the main
//! thread, the way chunk tasks read the chunk map. Only the main thread writes
//! it, when light lands and when chunks despawn.

use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};

use bevy::platform::collections::HashMap;
use bevy::prelude::*;

use super::faces::{Face, FaceHashes};
use super::solver::{neighbour_index, ChunkLight, Neighbourhood};

/// A loaded chunk's entry in the world view.
#[derive(Clone)]
pub(crate) struct ViewChunk {
    pub entity: Entity,
    /// The chunk's light, which holds what its neighbours' solves read.
    pub light: Arc<ChunkLight>,
    pub faces: FaceHashes,
    /// Changes whenever `light` does, and never goes back, so a task can tell
    /// whether the light it compared against is still there when it lands.
    pub version: u64,
}

/// The world view of one world, by level-0 chunk position.
#[derive(Clone, Default)]
pub(crate) struct LightView(Arc<RwLock<HashMap<IVec3, ViewChunk>>>);

impl LightView {
    pub fn read(&self) -> RwLockReadGuard<'_, HashMap<IVec3, ViewChunk>> {
        self.0.read().unwrap()
    }

    pub fn write(&self) -> RwLockWriteGuard<'_, HashMap<IVec3, ViewChunk>> {
        self.0.write().unwrap()
    }
}

/// What a solve read of the chunks around it: their inputs for the solver, and
/// the light of its face neighbours for comparing with its own.
pub(crate) struct Snapshot {
    pub neighbours: Neighbourhood,
    /// Each face neighbour's light and its version, by [`Face`].
    pub faces: [Option<(Arc<ChunkLight>, u64)>; 6],
}

impl Snapshot {
    /// Reads the chunks around the chunk at `position` from `view`, leaving
    /// the chunk's own inputs out.
    pub fn read(view: &HashMap<IVec3, ViewChunk>, position: IVec3) -> Self {
        let neighbours = std::array::from_fn(|index| {
            let index = index as i32;
            let offset = IVec3::new(index % 3, index / 3 % 3, index / 9) - IVec3::ONE;
            if offset == IVec3::ZERO {
                return None;
            }
            Some(view.get(&(position + offset))?.light.inputs.clone())
        });
        let faces = Face::ALL.map(|face| {
            let chunk = view.get(&(position + face.offset()))?;
            Some((chunk.light.clone(), chunk.version))
        });
        debug_assert!(neighbours[neighbour_index(IVec3::ZERO)].is_none());
        Self { neighbours, faces }
    }
}
