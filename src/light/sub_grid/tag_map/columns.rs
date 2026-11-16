//! The column answers of the hinted air a tag map holds: whether the column
//! above the middle voxel of each such position is open to the sky, walked
//! in a task as `light_at` walks it on request (see
//! [`request`](crate::light::request)).
//!
//! The walks run under [`JobKind::HintedColumn`](crate::pool_budget::JobKind)
//! on the threads no other job keeps busy, and while every thread is busy,
//! one at a time. A position with no answer yet reads as open sky meanwhile.
//!
//! An answer stays while its position stays in the tag map, and goes once
//! the position leaves. Answers are stale whenever `light_at`'s walk cache
//! empties, at any voxel edit of the world, a `reset_light` or a change of
//! light settings: each position walks again, and keeps showing its old
//! answer until the new one lands.

use bevy::platform::collections::{HashMap, HashSet};
use bevy::prelude::*;
use bevy::tasks::Task;
use futures_lite::future;

use crate::chunk::CHUNK_SIZE_I;

/// The voxel whose column the walk of the chunk position `position` reads:
/// the chunk's middle voxel.
pub(crate) fn walked_voxel(position: IVec3) -> IVec3 {
    position * CHUNK_SIZE_I + CHUNK_SIZE_I / 2
}

/// A walk running in a task.
struct Walk {
    position: IVec3,
    /// The walk cache's generation as the walk started.
    generation: u64,
    task: Task<bool>,
}

/// What hinted air the tag map holds, and how its positions read.
pub(crate) struct Hinting<'a> {
    /// Whether the fill hint of the level-0 chunk position says air.
    pub is_hinted_air: &'a dyn Fn(IVec3) -> bool,
    /// The level-0 chunk position of each camera.
    pub cameras: Vec<IVec3>,
    /// The world's spawning distance, in chunks.
    pub spawning_distance: u32,
    /// Whether sky light is on.
    pub sky: bool,
    /// The generation of `light_at`'s walk cache.
    pub generation: u64,
}

impl Hinting<'_> {
    /// Whether `position` is hinted air within the spawning distance of a
    /// camera. Beyond it, nothing is known about a position.
    pub fn holds(&self, position: IVec3) -> bool {
        let reach = (self.spawning_distance as i32).pow(2);
        self.cameras
            .iter()
            .any(|camera| camera.distance_squared(position) <= reach)
            && (self.is_hinted_air)(position)
    }
}

/// The column answers of a tag map's hinted air, and the walks working them
/// out. See the module docs.
#[derive(Default)]
pub(crate) struct HintedColumns {
    /// Whether the walked column of each position is open.
    answers: HashMap<IVec3, bool>,
    /// Positions waiting for a walk to start.
    wanted: HashSet<IVec3>,
    running: Vec<Walk>,
    /// The positions of `running`.
    walking: HashSet<IVec3>,
    /// The walk cache's generation the answers were last checked against.
    generation: u64,
}

impl HintedColumns {
    /// Whether the walked column at `position` is open, if a walk has
    /// answered. With no answer and no walk running, the position waits for
    /// one.
    pub fn answer(&mut self, position: IVec3) -> Option<bool> {
        let answer = self.answers.get(&position).copied();
        if answer.is_none() && !self.walking.contains(&position) {
            self.wanted.insert(position);
        }
        answer
    }

    /// Notes the walk cache's `generation`. Once it moves, every answer is
    /// stale and its position waits for another walk.
    pub fn follow(&mut self, generation: u64) {
        if generation == self.generation {
            return;
        }
        self.generation = generation;
        let stale = self
            .answers
            .keys()
            .filter(|position| !self.walking.contains(*position));
        self.wanted.extend(stale);
    }

    /// Takes the answers of finished walks of the positions `holds` accepts,
    /// pushing each position whose answer changed onto `changed`. A walk
    /// started before the walk cache's generation moved still gives its
    /// answer, and its position walks again. Returns how many walks
    /// finished.
    pub fn collect(
        &mut self,
        holds: impl Fn(IVec3) -> bool,
        changed: &mut Vec<IVec3>,
    ) -> usize {
        let mut finished = 0;
        let mut index = 0;
        while index < self.running.len() {
            let Some(open) =
                future::block_on(future::poll_once(&mut self.running[index].task))
            else {
                index += 1;
                continue;
            };
            finished += 1;
            let walk = self.running.swap_remove(index);
            self.walking.remove(&walk.position);
            if !holds(walk.position) {
                continue;
            }
            if walk.generation != self.generation {
                self.wanted.insert(walk.position);
            }
            if self.answers.insert(walk.position, open) != Some(open) {
                changed.push(walk.position);
            }
        }
        finished
    }

    /// Forgets the answers of `positions`, which the tag map no longer
    /// holds.
    pub fn release(&mut self, positions: impl Iterator<Item = IVec3>) {
        for position in positions {
            self.answers.remove(&position);
            self.wanted.remove(&position);
        }
    }

    /// Starts walks for up to `count` waiting positions, each in the task
    /// `spawn` makes for it.
    pub fn start(&mut self, count: usize, mut spawn: impl FnMut(IVec3) -> Task<bool>) {
        let started: Vec<IVec3> = self.wanted.iter().take(count).copied().collect();
        for position in started {
            self.wanted.remove(&position);
            self.walking.insert(position);
            self.running.push(Walk {
                position,
                generation: self.generation,
                task: spawn(position),
            });
        }
    }

    /// Whether any position waits for a walk to start.
    pub fn has_wanted(&self) -> bool {
        !self.wanted.is_empty()
    }

    /// How many walks are running.
    pub fn running(&self) -> usize {
        self.running.len()
    }
}

#[cfg(test)]
impl Hinting<'static> {
    /// What a world with no hinted air holds.
    pub fn none() -> Self {
        fn no_air(_: IVec3) -> bool {
            false
        }
        Self {
            is_hinted_air: &no_air,
            cameras: Vec::new(),
            spawning_distance: 0,
            sky: true,
            generation: 0,
        }
    }
}
