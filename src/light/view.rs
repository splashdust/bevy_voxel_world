//! The world view: each loaded level-0 chunk's light and what its neighbours'
//! solves read of it, shared with the tasks that solve light. It also holds
//! the light entries and the inputs of the hinted air around them: see
//! [`hinted`](super::hinted).
//!
//! Chunk tasks and relight tasks read their neighbours from it off the main
//! thread, the way chunk tasks read the chunk map. Only the main thread writes
//! it, when light lands and when chunks despawn.

use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};

use bevy::platform::collections::HashMap;
use bevy::prelude::*;

use super::faces::{Face, FaceHashes};
use super::inputs::LightInputs;
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
    /// The edit count of the chunk's position its inputs reflect.
    pub edits: u64,
    /// The [emission generation](super::solver::LightTaskSettings::emission_generation)
    /// its inputs' emissive voxels were found under.
    pub emission: u64,
}

/// What a light entry has in place of a chunk entity in the world view, in
/// [`LightWork`](super::relight::LightWork) and in the sky probes: a light
/// entry has no entity, and there is at most one at a position.
pub(crate) const ENTRY: Entity = Entity::PLACEHOLDER;

impl ViewChunk {
    /// Whether this is a light entry, the light of hinted air, not a chunk.
    /// See [`hinted`](super::hinted).
    pub fn is_entry(&self) -> bool {
        self.entity == ENTRY
    }
}

/// Hinted air that holds no light entry, as the solves around it read it.
#[derive(Clone)]
pub(crate) struct AirInputs {
    /// Its voxels, every one open with the voxel edits over it, and its column
    /// answer.
    pub inputs: Arc<LightInputs>,
    /// The edit count of the position its inputs reflect.
    pub edits: u64,
}

/// What the world view holds: each loaded chunk and light entry by level-0
/// position, which it derefs to, and the hinted air around them that holds no
/// light entry.
#[derive(Default)]
pub(crate) struct ViewMaps {
    pub chunks: HashMap<IVec3, ViewChunk>,
    pub air: HashMap<IVec3, AirInputs>,
}

impl ViewMaps {
    /// The inputs a solve reads at `position`: a chunk's or light entry's, or
    /// hinted air's.
    pub fn inputs(&self, position: IVec3) -> Option<&Arc<LightInputs>> {
        match self.chunks.get(&position) {
            Some(chunk) => Some(&chunk.light.inputs),
            None => self.air.get(&position).map(|air| &air.inputs),
        }
    }

    /// Whether `position` is hinted air holding no light entry.
    pub fn is_air(&self, position: IVec3) -> bool {
        !self.chunks.contains_key(&position) && self.air.contains_key(&position)
    }
}

impl std::ops::Deref for ViewMaps {
    type Target = HashMap<IVec3, ViewChunk>;

    fn deref(&self) -> &Self::Target {
        &self.chunks
    }
}

impl std::ops::DerefMut for ViewMaps {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.chunks
    }
}

/// The world view of one world.
#[derive(Clone, Default)]
pub(crate) struct LightView(Arc<RwLock<ViewMaps>>);

impl LightView {
    pub fn read(&self) -> RwLockReadGuard<'_, ViewMaps> {
        self.0.read().unwrap()
    }

    pub fn write(&self) -> RwLockWriteGuard<'_, ViewMaps> {
        self.0.write().unwrap()
    }
}

/// What a solve read of the chunks around it: their inputs for the solver, and
/// the light of its face neighbours for comparing with its own.
pub(crate) struct Snapshot {
    /// The position of the chunk the snapshot is for.
    pub position: IVec3,
    pub neighbours: Neighbourhood,
    /// Each face neighbour's light and its version, by [`Face`].
    pub faces: [Option<(Arc<ChunkLight>, u64)>; 6],
}

impl Snapshot {
    /// Reads the chunks around the chunk at `position` from `view`, leaving
    /// the chunk's own inputs out. Hinted air reads as its inputs, and is a
    /// face neighbour to compare light with only where it holds a light entry.
    pub fn read(view: &ViewMaps, position: IVec3) -> Self {
        let neighbours = std::array::from_fn(|index| {
            let index = index as i32;
            let offset = IVec3::new(index % 3, index / 3 % 3, index / 9) - IVec3::ONE;
            if offset == IVec3::ZERO {
                return None;
            }
            view.inputs(position + offset).cloned()
        });
        let faces = Face::ALL.map(|face| {
            let chunk = view.get(&(position + face.offset()))?;
            Some((chunk.light.clone(), chunk.version))
        });
        debug_assert!(neighbours[neighbour_index(IVec3::ZERO)].is_none());
        Self {
            position,
            neighbours,
            faces,
        }
    }
}
