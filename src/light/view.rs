//! The world view: each loaded level-0 chunk's light and what its neighbours'
//! solves read of it, shared with the tasks that solve light. It also holds
//! the light entries and the inputs of the hinted air around them: see
//! [`hinted`](super::hinted).
//!
//! Chunk tasks and relight tasks read their neighbours from it off the main
//! thread, the way chunk tasks read the chunk map. Only the main thread writes
//! it, when light lands and when chunks despawn.

use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard, Weak};

use bevy::platform::collections::HashMap;
use bevy::prelude::*;

use super::faces::{Face, FaceHashes};
use super::inputs::LightInputs;
use super::solver::{neighbour_index, ChunkLight, Neighbourhood};
use crate::chunk::CHUNK_SIZE_F;
use crate::grid::{GridId, GridKey};

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
    /// The edit stamp of the chunk's position its inputs reflect.
    pub edits: u64,
    /// The [emission generation](super::solver::LightTaskSettings::emission_generation)
    /// its inputs' emissive voxels were found under.
    pub emission: u64,
    /// For a chunk that takes sky fill, the light a solve landed and the hash
    /// of the sky fill seeds it was solved from. Anything that puts other
    /// light in `light` leaves this behind, so it holds only while `light` is
    /// the light it names.
    pub fill_seeds: Option<(Weak<ChunkLight>, u64)>,
}

impl ViewChunk {
    /// The hash of the sky fill seeds the chunk's light was solved from, if
    /// known.
    pub fn solved_from_seeds(&self) -> Option<u64> {
        let (light, hash) = self.fill_seeds.as_ref()?;
        std::ptr::eq(light.as_ptr(), Arc::as_ptr(&self.light)).then_some(*hash)
    }
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
    /// The edit stamp of the position its inputs reflect.
    pub edits: u64,
}

/// What the world view holds: each loaded level-0 chunk and light entry by
/// grid key, and the hinted air around them that holds no light entry.
///
/// Only the world has light entries and hinted air. The parts of light that
/// only the world has, such as the sky record and sky fill, keep level-0 chunk
/// positions, and find the world's chunks here through [`in_world`],
/// [`get`](Self::get) and [`world_chunks`](Self::world_chunks).
#[derive(Default)]
pub(crate) struct ViewMaps {
    pub chunks: HashMap<GridKey, ViewChunk>,
    pub air: HashMap<GridKey, AirInputs>,
}

impl ViewMaps {
    /// The inputs a solve reads at `key`: a chunk's or light entry's, or
    /// hinted air's.
    #[inline]
    pub fn inputs(&self, key: GridKey) -> Option<&Arc<LightInputs>> {
        match self.chunks.get(&key) {
            Some(chunk) => Some(&chunk.light.inputs),
            None => self.air.get(&key).map(|air| &air.inputs),
        }
    }

    /// Whether `key` is hinted air holding no light entry.
    #[inline]
    pub fn is_air(&self, key: GridKey) -> bool {
        !self.chunks.contains_key(&key) && self.air.contains_key(&key)
    }

    /// The world's chunk or light entry at the level-0 chunk position
    /// `position`.
    #[inline]
    pub fn get(&self, position: &IVec3) -> Option<&ViewChunk> {
        self.chunks.get(&in_world(*position))
    }

    /// Whether the world has a chunk or light entry at the level-0 chunk
    /// position `position`.
    #[inline]
    pub fn contains_key(&self, position: &IVec3) -> bool {
        self.chunks.contains_key(&in_world(*position))
    }

    /// Each of the world's chunks and light entries, with its level-0 chunk
    /// position, in no particular order. Its cost grows with every grid's
    /// chunks in the view.
    pub fn world_chunks(&self) -> impl Iterator<Item = (IVec3, &ViewChunk)> {
        self.chunks
            .iter()
            .filter_map(|(key, chunk)| Some((world_position(key)?, chunk)))
    }

    /// The world's hinted air holding no light entry, with its level-0 chunk
    /// position, in no particular order.
    pub fn world_air(&self) -> impl Iterator<Item = (IVec3, &AirInputs)> {
        self.air
            .iter()
            .filter_map(|(key, air)| Some((world_position(key)?, air)))
    }
}

/// The grid key of the world's level-0 chunk at `position`.
#[inline]
pub(crate) fn in_world(position: IVec3) -> GridKey {
    GridKey::level_0(GridId::WORLD, position)
}

/// The level-0 chunk position of the chunk at `key`, if it is the world's.
#[inline]
pub(crate) fn world_position(key: &GridKey) -> Option<IVec3> {
    (key.grid() == GridId::WORLD).then(|| key.position())
}

/// The middle of the level-0 chunk at `key`, placed by `pose`, the pose of
/// its grid against the world's root.
///
/// The world's own chunks take [`GlobalTransform::IDENTITY`], which leaves
/// the middle where the chunk sits in the world.
pub(crate) fn chunk_centre(key: GridKey, pose: &GlobalTransform) -> Vec3 {
    pose.transform_point((key.position().as_vec3() + 0.5) * CHUNK_SIZE_F)
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

/// How a solve reads a chunk position around its chunk that the world view
/// holds nothing at.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Missing {
    /// Every voxel is solid: no light passes through it. A world solve
    /// always reads missing chunks this way.
    Solid,
    /// Every voxel is open, with no direct sky and no emissive voxel, so
    /// light passes through it but nothing starts there.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "no solve reads missing chunks as air yet")
    )]
    Air,
}

/// What a solve read of the chunks around it: their inputs for the solver, and
/// the light of its face neighbours for comparing with its own.
pub(crate) struct Snapshot {
    pub neighbours: Neighbourhood,
    /// Each face neighbour's light and its version, by [`Face`].
    pub faces: [Option<(Arc<ChunkLight>, u64)>; 6],
    /// How the solve reads the positions of `neighbours` that hold nothing.
    pub missing: Missing,
}

impl Snapshot {
    /// Reads the chunks around the chunk at `key` from `view`, on its own
    /// grid, leaving the chunk's own inputs out. Hinted air reads as its
    /// inputs, and is a face neighbour to compare light with only where it
    /// holds a light entry. A position holding nothing reads as `missing`
    /// says.
    pub fn read(view: &ViewMaps, key: GridKey, missing: Missing) -> Self {
        let neighbours = std::array::from_fn(|index| {
            let index = index as i32;
            let offset = IVec3::new(index % 3, index / 3 % 3, index / 9) - IVec3::ONE;
            if offset == IVec3::ZERO {
                return None;
            }
            view.inputs(key + offset).cloned()
        });
        let faces = Face::ALL.map(|face| {
            let chunk = view.chunks.get(&(key + face.offset()))?;
            Some((chunk.light.clone(), chunk.version))
        });
        debug_assert!(neighbours[neighbour_index(IVec3::ZERO)].is_none());
        Self {
            neighbours,
            faces,
            missing,
        }
    }
}

#[cfg(all(test, feature = "sub_grids"))]
mod tests {
    use super::*;
    use crate::light::config::LightPrecision;
    use crate::light::hinted::known_light;
    use crate::light::inputs::Occupancy;
    use crate::light::sky::ColumnAnswer;

    fn chunk(occupancy: Occupancy) -> ViewChunk {
        let inputs = Arc::new(LightInputs {
            occupancy,
            emissive: Vec::new(),
            columns: ColumnAnswer::none(),
            ring: false,
        });
        ViewChunk {
            entity: Entity::PLACEHOLDER,
            light: Arc::new(known_light(&inputs, LightPrecision::Steps16)),
            faces: FaceHashes {
                sky: [0; 6],
                voxel: [0; 6],
            },
            version: 0,
            edits: 0,
            emission: 0,
            fill_seeds: None,
        }
    }

    fn offsets() -> impl Iterator<Item = IVec3> {
        (0..27).map(|i| IVec3::new(i % 3, i / 3 % 3, i / 9) - IVec3::ONE)
    }

    #[test]
    fn a_grid_keys_neighbours_stay_on_its_grid() {
        let key = GridKey::level_0(GridId::new(1), IVec3::new(2, -3, 4));
        for offset in [IVec3::X, IVec3::NEG_Y, IVec3::new(1, 1, -1)] {
            let beside = key + offset;
            assert_eq!(beside.grid(), GridId::new(1));
            assert_eq!(beside.position(), IVec3::new(2, -3, 4) + offset);
        }
        assert_eq!(world_position(&key), None);
        let world = in_world(key.position());
        assert_eq!(
            world_position(&(world + IVec3::X)),
            Some(IVec3::new(3, -3, 4))
        );
    }

    #[test]
    fn a_sub_grid_chunks_snapshot_reads_no_world_chunk_beside_it() {
        let position = IVec3::new(2, -3, 4);
        let mut view = ViewMaps::default();
        // The world's chunks fill every place around the sub-grid chunk's
        // position, and the sub-grid has one chunk beside it, across +X.
        for offset in offsets() {
            view.chunks
                .insert(in_world(position + offset), chunk(Occupancy::Solid));
        }
        let own = GridKey::level_0(GridId::new(1), position);
        view.chunks.insert(own + IVec3::X, chunk(Occupancy::Open));

        let snapshot = Snapshot::read(&view, own, Missing::Solid);

        let read: Vec<IVec3> = offsets()
            .filter(|offset| snapshot.neighbours[neighbour_index(*offset)].is_some())
            .collect();
        assert_eq!(read, vec![IVec3::X]);
        let beside = snapshot.neighbours[neighbour_index(IVec3::X)].as_ref();
        assert!(matches!(beside.unwrap().occupancy, Occupancy::Open));
        let faces: Vec<Face> = Face::ALL
            .into_iter()
            .filter(|face| snapshot.faces[*face as usize].is_some())
            .collect();
        assert_eq!(faces, vec![Face::PosX]);
    }
}
