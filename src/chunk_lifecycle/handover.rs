//! Handovers: how voxels move between grids without a frame that shows them
//! twice or not at all.
//!
//! A commit that gives a sub-grid its first solid voxel or takes its last,
//! or one an operation marks with [`ChunkRequests::mark_handover`], lands as
//! one handover. It holds every chunk the commit spawns or makes stale, in
//! every grid, padding neighbours included. A held chunk keeps showing what
//! it showed: its old mesh and published voxel data, or nothing for a chunk
//! the commit spawned. A held chunk the commit retires keeps showing too,
//! rather than retiring at once.
//!
//! Each held chunk waits for a finished chunk task that includes the commit:
//! one started after it. The task already running at the commit is
//! superseded by it, so the chunk waits for that task's follow-up. A task
//! superseded by a later edit started after the commit, so it counts, and an
//! edit made every frame can't hold a handover. The newest finished task of
//! each held chunk is kept unpublished. Once every chunk of a handover has
//! one, they all publish in the same frame, and the chunks it retires
//! despawn in that frame at once, with their `ChunkWillDespawn`. Colliders
//! rebuilt on `ChunkPublished` and removed on `ChunkWillDespawn` swap in one
//! frame.
//!
//! In a lit world, a handover also waits until the relights its commit
//! started have solved, in every grid: the first solves of the sub-grid
//! chunks it spawns, and the relights of the chunks within reach of the
//! voxels it changed, around a lift's hole and under a stamp. The light of
//! those chunks and of the chunks it holds is held until it lands, and
//! lands in the frame its chunks publish, so meshes and light change
//! together, a relit chunk with no new mesh included. It doesn't wait for
//! the relight ripple that light calls for, which starts as it lands and
//! lands later, as after any edit. So a lit handover lasts as long as its
//! slowest relight, which grows with reach. A relight superseded by a later
//! edit counts once it has solved, as a superseded task does. See the light
//! module's `handover` module.
//!
//! Handovers that hold a chunk in common fold into one and land together.
//! So lifting from the same chunks every frame holds the folded handover
//! until the lifts stop. A sub-grid that flips between empty and not every
//! frame, beside chunks edited every frame, keeps folding handovers too.
//!
//! A chunk that leaves mid-handover drops out of it, and the others land
//! when they are ready: a despawned grid's chunks, a world chunk the
//! streaming policy retires, a chunk a game despawns.
//!
//! A handover carries the [`Lifted`] or [`Stamped`] message of each lift and
//! stamp whose commit it holds, and folded handovers carry them all. They
//! are written once it lands, leaving out those whose sub-grid is gone by
//! then, and each stamped sub-grid despawns in that frame's `PostUpdate`.
//!
//! A stamp marks its commit, since a lift and a stamp of the new sub-grid in
//! one frame leave it as empty after the commit as before. A world applying
//! another world's edit deltas gets no such mark: it sees only the deltas of
//! the grid lifted from and the grid stamped into, which fill or empty no
//! sub-grid, so there the two grids' chunks can publish a frame apart.
//!
//! A world with no handover in flight pays a check per lifecycle system for
//! it. A handover's cost grows with the chunks it holds.

use std::sync::Mutex;

use super::*;
use crate::sub_grid::{Lifted, Stamped, SubGrids};

/// A lift or stamp whose commit a handover holds, announced once it lands.
pub(crate) enum Operation<C: VoxelWorldConfig> {
    Lift(Lifted<C>),
    Stamp(Stamped<C>),
}

/// What a held chunk waits for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Wait {
    /// The task running at the commit to finish, and then the task after it.
    RunningTask,
    /// The chunk's next finished task, which includes the commit.
    NextTask,
    /// Nothing: a finished task including the commit is kept.
    Done,
    /// Nothing: the chunk despawns when the handover lands.
    Despawn,
}

impl Wait {
    /// What a chunk the commit makes stale waits for, given whether its task
    /// was running at the commit.
    pub fn stale(generating: bool) -> Self {
        if generating {
            Self::RunningTask
        } else {
            Self::NextTask
        }
    }

    fn is_ready(self) -> bool {
        matches!(self, Self::Done | Self::Despawn)
    }
}

/// A chunk a handover holds.
struct Held<C: VoxelWorldConfig> {
    key: GridKey,
    handover: u32,
    wait: Wait,
    /// The chunk's newest finished task, kept unpublished.
    kept: Option<Finished<C>>,
}

/// The handovers in flight in the voxel world `C`.
#[derive(Resource)]
pub(crate) struct Handovers<C: VoxelWorldConfig> {
    held: HashMap<Entity, Held<C>>,
    /// How many of `held` each grid has, for grids with any.
    held_per_grid: HashMap<GridId, usize>,
    /// The chunks of each handover, by its id.
    handovers: HashMap<u32, Vec<Entity>>,
    /// The operations of each handover holding any, by its id.
    operations: HashMap<u32, Vec<Operation<C>>>,
    /// Operations whose commit is still to come.
    queued: Mutex<Vec<Operation<C>>>,
    /// Operations whose handover has landed, to announce.
    landed: Vec<Operation<C>>,
    next_id: u32,
    /// Sub-grids to despawn once no handover holds their chunks, each with
    /// whether a `PostUpdate` has passed since `despawn_sub_grid`.
    despawns: Mutex<Vec<(Entity, bool)>>,
    /// Sub-grids `despawn_landed` ended this frame, for `apply_despawns` to
    /// despawn.
    despawning: Vec<Entity>,
    /// Sub-grids stamped and still live, which fragment checks landing from
    /// the stamp on count as gone.
    #[cfg(feature = "dislodged_fragments")]
    stamped_away: Mutex<HashSet<Entity>>,
    stats: HandoverStats,
    /// In a lit world, the chunks whose light each handover holds, by its
    /// id: those whose relights its commits started, and its own.
    #[cfg(feature = "light_propagation")]
    light: HashMap<u32, Vec<GridKey>>,
    /// The handover holding the light of each chunk in `light`.
    #[cfg(feature = "light_propagation")]
    lit_by: HashMap<GridKey, u32>,
    /// The handover the last commit formed, whose commit's relights are
    /// still to be noted.
    #[cfg(feature = "light_propagation")]
    formed: Option<u32>,
}

impl<C: VoxelWorldConfig> Default for Handovers<C> {
    fn default() -> Self {
        Self {
            held: HashMap::new(),
            held_per_grid: HashMap::new(),
            handovers: HashMap::new(),
            operations: HashMap::new(),
            queued: Mutex::default(),
            landed: Vec::new(),
            next_id: 0,
            despawns: Mutex::default(),
            despawning: Vec::new(),
            #[cfg(feature = "dislodged_fragments")]
            stamped_away: Mutex::default(),
            stats: HandoverStats::default(),
            #[cfg(feature = "light_propagation")]
            light: HashMap::new(),
            #[cfg(feature = "light_propagation")]
            lit_by: HashMap::new(),
            #[cfg(feature = "light_propagation")]
            formed: None,
        }
    }
}

/// The chunks this frame's requests touch, gathered by `apply_requests` for
/// the handovers.
pub(crate) struct Joining {
    /// Whether this frame's commit lands as one handover.
    forming: bool,
    /// Whether to gather anything: a handover forms, or one is in flight.
    watching: bool,
    /// Each chunk with what it waits for. A later entry for a chunk wins.
    chunks: Vec<(Entity, GridKey, Wait)>,
}

impl Joining {
    /// Notes that the commit made the chunk `entity` at `key` stale, and
    /// whether its task was running.
    #[inline]
    pub fn stale(&mut self, entity: Entity, key: GridKey, generating: bool) {
        if self.watching {
            self.chunks.push((entity, key, Wait::stale(generating)));
        }
    }

    /// Notes that a commit spawned the sub-grid chunk `entity` at `key`.
    #[inline]
    pub fn spawned(&mut self, entity: Entity, key: GridKey) {
        if self.forming {
            self.chunks.push((entity, key, Wait::NextTask));
        }
    }
}

impl<C: VoxelWorldConfig> Handovers<C> {
    /// Starts gathering this frame's chunks, `forming` a handover if this
    /// frame's commit lands as one.
    pub fn joining(&self, forming: bool) -> Joining {
        Joining {
            forming,
            watching: forming || !self.held.is_empty(),
            chunks: Vec::new(),
        }
    }

    /// Whether the retire request for the sub-grid chunk `entity` at `key`
    /// is held, rather than retiring the chunk: in a frame that forms a
    /// handover, and for a chunk a handover holds. A held retire despawns
    /// the chunk when its handover lands.
    pub fn holds_retire(
        &self,
        joining: &mut Joining,
        entity: Entity,
        key: GridKey,
    ) -> bool {
        if !joining.forming && !self.held.contains_key(&entity) {
            return false;
        }
        joining.chunks.push((entity, key, Wait::Despawn));
        true
    }

    /// Takes in the chunks `joining` gathered, and the lifts and stamps
    /// whose edits this frame's commit took. A forming handover holds them
    /// all, and folds in each handover holding one of the chunks. Otherwise
    /// only chunks already held change: a retire held for one of them, or
    /// one held for despawning that the commit needs again.
    ///
    /// A lift's commit fills its sub-grid, so it forms a handover unless the
    /// sub-grid was emptied or despawned before it, and then nothing moved
    /// and the lift is dropped. A forming handover with no chunks lands at
    /// once. A stamp marks its commit, so it always forms one.
    pub fn join(&mut self, joining: Joining) {
        let operations = std::mem::take(self.queued.get_mut().unwrap());
        if joining.chunks.is_empty() {
            if joining.forming {
                self.landed.extend(operations);
            }
            return;
        }
        if !joining.forming {
            for (entity, _, wait) in joining.chunks {
                if let Some(held) = self.held.get_mut(&entity) {
                    if wait == Wait::Despawn || held.wait == Wait::Despawn {
                        held.set(wait);
                    }
                }
            }
            return;
        }
        let id = self.next_id;
        self.next_id = self.next_id.wrapping_add(1);
        self.handovers.insert(id, Vec::new());
        #[cfg(feature = "light_propagation")]
        {
            self.formed = Some(id);
        }
        if !operations.is_empty() {
            self.operations.insert(id, operations);
        }
        for (entity, key, wait) in joining.chunks {
            match self.held.get_mut(&entity) {
                Some(held) => {
                    let from = held.handover;
                    held.set(wait);
                    if from != id {
                        self.fold(from, id);
                    }
                }
                None => {
                    *self.held_per_grid.entry(key.grid()).or_default() += 1;
                    self.held.insert(
                        entity,
                        Held {
                            key,
                            handover: id,
                            wait,
                            kept: None,
                        },
                    );
                    self.handovers.get_mut(&id).unwrap().push(entity);
                }
            }
        }
    }

    /// Moves every chunk of the handover `from` into the handover `into`.
    fn fold(&mut self, from: u32, into: u32) {
        let Some(chunks) = self.handovers.remove(&from) else {
            return;
        };
        for entity in &chunks {
            if let Some(held) = self.held.get_mut(entity) {
                held.handover = into;
            }
        }
        self.handovers.get_mut(&into).unwrap().extend(chunks);
        if let Some(operations) = self.operations.remove(&from) {
            self.operations.entry(into).or_default().extend(operations);
        }
        #[cfg(feature = "light_propagation")]
        if let Some(keys) = self.light.remove(&from) {
            for key in &keys {
                self.lit_by.insert(*key, into);
            }
            self.light.entry(into).or_default().extend(keys);
        }
    }

    /// Makes the handover the last commit formed wait for the relights that
    /// commit started, and hold the light of those chunks and of its own,
    /// which would otherwise show as it lands. Handovers holding the light of
    /// a chunk in common fold into one.
    ///
    /// The relights are noted as the commit is taken into light, after its
    /// handover forms, so this runs at the next landing check.
    #[cfg(feature = "light_propagation")]
    fn wait_for_light(&mut self, light_work: &mut crate::light::relight::LightWork<C>) {
        let relights = light_work.take_handover_relights();
        let Some(id) = self
            .formed
            .take()
            .filter(|id| self.handovers.contains_key(id))
        else {
            for key in relights {
                light_work.release_for_handover(key);
            }
            return;
        };
        let mut keys: HashSet<GridKey> = relights.into_iter().collect();
        for entity in &self.handovers[&id] {
            let Some(held) = self.held.get(entity) else {
                continue;
            };
            // A chunk the handover retires despawns as it lands, so the
            // handover waits for no relight of it, but holds its light.
            if held.wait == Wait::Despawn {
                light_work.release_for_handover(held.key);
            }
            light_work.hold_for_handover(held.key);
            keys.insert(held.key);
        }
        for key in &keys {
            if let Some(from) = self.lit_by.insert(*key, id) {
                if from != id && self.handovers.contains_key(&from) {
                    self.fold(from, id);
                }
            }
        }
        self.light.entry(id).or_default().extend(keys);
    }

    /// Keeps `finished` if a handover holds its chunk, and returns it to
    /// publish otherwise. Called after the chunk's state has moved on, so a
    /// superseded task's follow-up starts either way.
    #[inline]
    pub fn keep(&mut self, finished: Finished<C>) -> Option<Finished<C>> {
        if self.held.is_empty() {
            return Some(finished);
        }
        let Some(held) = self.held.get_mut(&finished.entity) else {
            return Some(finished);
        };
        match held.wait {
            Wait::Despawn => {}
            Wait::RunningTask => {
                held.kept = Some(finished);
                held.wait = Wait::NextTask;
            }
            Wait::NextTask | Wait::Done => {
                held.kept = Some(finished);
                held.wait = Wait::Done;
            }
        }
        None
    }

    /// Notes that the task of the chunk `entity` finished without a result,
    /// since its cached mesh expired: its next task starts after the commit.
    #[inline]
    pub fn task_dropped(&mut self, entity: Entity) {
        if let Some(held) = self.held.get_mut(&entity) {
            if held.wait == Wait::RunningTask {
                held.wait = Wait::NextTask;
            }
        }
    }

    /// Drops the chunks that left their handover, and lands each handover
    /// whose chunks are all ready: publishes their kept tasks, despawns the
    /// chunks it retires and queues the messages of its lifts and stamps.
    /// The data they publish is readable once the next
    /// `flush_chunk_map_buffers` has run, in this frame.
    #[allow(clippy::too_many_arguments)]
    pub fn land_ready(
        &mut self,
        index: &mut ChunkIndex<C>,
        commands: &mut Commands,
        chunk_map_update_buffer: &mut ChunkMapUpdateBuffer<C, C::MaterialIndex>,
        chunk_map_remove_buffer: &mut ChunkMapRemoveBuffer<C>,
        ev_chunk_will_despawn: &mut MessageWriter<ChunkWillDespawn<C>>,
        #[cfg(feature = "light_propagation")] mut light_work: Option<
            &mut crate::light::relight::LightWork<C>,
        >,
    ) {
        #[cfg(feature = "light_propagation")]
        if let Some(light_work) = light_work.as_deref_mut() {
            self.wait_for_light(light_work);
        }
        // A lit handover whose chunks have all left still waits for its
        // relights, and holds their light until it lands.
        if self.held.is_empty() && self.handovers.is_empty() {
            return;
        }
        let Self {
            held,
            held_per_grid,
            handovers,
            operations,
            landed,
            stats,
            #[cfg(feature = "light_propagation")]
            light,
            #[cfg(feature = "light_propagation")]
            lit_by,
            ..
        } = self;
        held.retain(|entity, chunk| {
            let stays = index.keys.get(entity) == Some(&chunk.key)
                && !index.retiring.contains(entity);
            if !stays {
                drop_held(held_per_grid, chunk.key.grid());
            }
            stays
        });
        handovers.retain(|id, chunks| {
            chunks.retain(|entity| held.contains_key(entity));
            let ready =
                |entity: &Entity| held.get(entity).is_some_and(|c| c.wait.is_ready());
            if !chunks.iter().all(ready) {
                return true;
            }
            // In a lit world, the relights its commits started must have
            // solved too. The light held for its chunks lands in this frame.
            #[cfg(feature = "light_propagation")]
            if let Some(light_work) = light_work.as_deref_mut() {
                let keys = light.get(id).map_or(&[][..], Vec::as_slice);
                if !light_work.handover_solved(keys) {
                    return true;
                }
                for key in light.remove(id).into_iter().flatten() {
                    light_work.release_for_handover(key);
                    lit_by.remove(&key);
                }
            }
            landed.extend(operations.remove(id).into_iter().flatten());
            if chunks.is_empty() {
                return false;
            }
            stats.handovers += 1;
            stats.chunks += chunks.len() as u64;
            for entity in chunks.drain(..) {
                let Some(chunk) = held.remove(&entity) else {
                    continue;
                };
                drop_held(held_per_grid, chunk.key.grid());
                if chunk.wait == Wait::Despawn {
                    if let Some(message) = index.message(chunk.key, entity) {
                        ev_chunk_will_despawn.write(message);
                    }
                    commands.entity(entity).try_despawn();
                    if let Some(key) = index.despawned(entity) {
                        chunk_map_remove_buffer.push((key, entity));
                    }
                } else if let Some(finished) = chunk.kept {
                    publish(
                        finished,
                        commands,
                        chunk_map_update_buffer,
                        #[cfg(feature = "light_propagation")]
                        light_work.as_deref_mut(),
                    );
                    chunk_map_update_buffer.must_land = true;
                }
            }
            false
        });
    }

    /// Notes a lift, whose edits commit in the next `commit_voxel_edits`,
    /// to announce with `lifted` once that commit's handover lands.
    pub fn lifted(&self, lifted: Lifted<C>) {
        self.queued.lock().unwrap().push(Operation::Lift(lifted));
    }

    /// Notes a stamp, whose edits commit in the next `commit_voxel_edits`,
    /// to announce with `stamped` once that commit's handover lands.
    pub fn stamped(&self, stamped: Stamped<C>) {
        #[cfg(feature = "dislodged_fragments")]
        self.stamped_away.lock().unwrap().insert(stamped.sub_grid);
        self.queued.lock().unwrap().push(Operation::Stamp(stamped));
    }

    /// Whether the sub-grid `entity` was stamped: from the call on, a
    /// fragment check of it finds nothing to report, since its voxels are
    /// the target's.
    #[cfg(feature = "dislodged_fragments")]
    pub fn is_stamped_away(&self, entity: Entity) -> bool {
        self.stamped_away.lock().unwrap().contains(&entity)
    }

    /// Whether [`write_landed`](Self::write_landed) has anything to do: a
    /// landed lift or stamp to announce, or a stamped sub-grid to forget.
    pub fn has_landed(handovers: Res<Self>) -> bool {
        #[cfg(feature = "dislodged_fragments")]
        if !handovers.stamped_away.lock().unwrap().is_empty() {
            return true;
        }
        !handovers.landed.is_empty()
    }

    /// Whether [`despawn_landed`](Self::despawn_landed) has a sub-grid
    /// waiting to despawn.
    pub fn has_despawns(handovers: Res<Self>) -> bool {
        !handovers.despawns.lock().unwrap().is_empty()
    }

    /// Writes the [`Lifted`] or [`Stamped`] of each lift and stamp whose
    /// handover has landed, in call order, except those whose sub-grid is no
    /// longer live. Each stamped sub-grid despawns in this frame's
    /// `PostUpdate`.
    pub fn write_landed(
        mut handovers: ResMut<Self>,
        sub_grids: Res<SubGrids<C>>,
        mut lifted: MessageWriter<Lifted<C>>,
        mut stamped: MessageWriter<Stamped<C>>,
    ) {
        #[cfg(feature = "dislodged_fragments")]
        {
            let stamped_away = handovers.stamped_away.get_mut().unwrap();
            if !stamped_away.is_empty() {
                stamped_away.retain(|&entity| sub_grids.live(entity).is_some());
            }
        }
        if handovers.landed.is_empty() {
            return;
        }
        let Self {
            landed, despawns, ..
        } = &mut *handovers;
        for operation in landed.drain(..) {
            match operation {
                Operation::Lift(message) => {
                    if sub_grids.live(message.sub_grid).is_some() {
                        lifted.write(message);
                    }
                }
                Operation::Stamp(message) => {
                    if sub_grids.live(message.sub_grid).is_some() {
                        stamped.write(message);
                        // The handover holding its chunks has landed, so it
                        // despawns in this frame's `despawn_landed`.
                        despawns.get_mut().unwrap().push((message.sub_grid, true));
                    }
                }
            }
        }
    }

    /// Whether a handover holds a chunk of `grid`.
    pub fn holds_grid(&self, grid: GridId) -> bool {
        self.held_per_grid.contains_key(&grid)
    }

    /// Despawns the sub-grid `entity` once no handover holds its chunks: in
    /// the `PostUpdate` of the frame the handover holding them lands. Edits
    /// of the sub-grid not yet committed count as held, since their commit
    /// may form a handover.
    pub fn despawn_after_landing(&self, entity: Entity) {
        self.despawns.lock().unwrap().push((entity, false));
    }

    /// Ends each sub-grid waiting for a handover that no handover holds any
    /// more in the record of live sub-grids, and queues it for
    /// [`apply_despawns`](Self::apply_despawns) to despawn.
    ///
    /// It has no `Commands`, so a system ordered after it, such as one that
    /// reads the live sub-grids, waits for no sync point.
    pub fn despawn_landed(
        mut handovers: ResMut<Self>,
        sub_grids: Res<crate::sub_grid::SubGrids<C>>,
        edits: Res<VoxelEdits<C, C::MaterialIndex>>,
    ) {
        let mut despawns = std::mem::take(handovers.despawns.get_mut().unwrap());
        if despawns.is_empty() {
            return;
        }
        despawns.retain_mut(|(entity, waited)| {
            let Some((grid, _)) = sub_grids.live(*entity) else {
                return false;
            };
            let committing = !*waited && edits.has_pending_in(grid);
            *waited = true;
            if committing || handovers.holds_grid(grid) {
                return true;
            }
            sub_grids.end_despawned(*entity, &edits);
            handovers.despawning.push(*entity);
            false
        });
        handovers.despawns.get_mut().unwrap().extend(despawns);
    }

    /// Whether [`apply_despawns`](Self::apply_despawns) has a sub-grid to
    /// despawn.
    pub fn has_despawning(handovers: Res<Self>) -> bool {
        !handovers.despawning.is_empty()
    }

    /// Despawns the sub-grids [`despawn_landed`](Self::despawn_landed) ended
    /// this frame.
    ///
    /// No system is ordered after it, so its despawns need no sync point and
    /// apply by the end of `PostUpdate`: a sub-grid despawned in the frame its
    /// handover lands is never drawn.
    pub fn apply_despawns(mut commands: Commands, mut handovers: ResMut<Self>) {
        for entity in handovers.despawning.drain(..) {
            commands.entity(entity).try_despawn();
        }
    }
}

/// Counts one chunk of `grid` fewer in `held_per_grid`, dropping the grid
/// once it has none.
fn drop_held(held_per_grid: &mut HashMap<GridId, usize>, grid: GridId) {
    if let Some(count) = held_per_grid.get_mut(&grid) {
        *count -= 1;
        if *count == 0 {
            held_per_grid.remove(&grid);
        }
    }
}

impl<C: VoxelWorldConfig> Held<C> {
    /// Makes the chunk wait for `wait`. A chunk held for despawning keeps no
    /// task.
    fn set(&mut self, wait: Wait) {
        self.wait = wait;
        if wait == Wait::Despawn {
            self.kept = None;
        }
    }
}

/// What the handovers of a world have done since it started.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HandoverStats {
    /// Handovers landed, each folded handover counting once.
    pub handovers: u64,
    /// Chunks the landed handovers held, in every grid, counting those that
    /// despawned as they landed. Divided by `handovers`, the chunks per
    /// handover.
    pub chunks: u64,
}

/// What the handovers of world `C` have done, or `None` if it has no
/// sub-grid support.
pub fn handover_stats<C: VoxelWorldConfig>(world: &World) -> Option<HandoverStats> {
    Some(world.get_resource::<Handovers<C>>()?.stats)
}
