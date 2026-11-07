///
/// Pool budget
/// How many jobs of each kind a voxel world keeps running on the async compute
/// task pool, and which of its Pending chunks start first.
///
/// Bevy's task pool runs tasks in the order they start and has no priorities,
/// so a world decides only when it starts work: only as far as keeps the pool
/// busy until the next frame. Work started later can then still go ahead of
/// work that is waiting.
///
use std::marker::PhantomData;

use bevy::{prelude::*, tasks::AsyncComputeTaskPool};

/// The fewest jobs kept running per task pool thread.
///
/// More run once the pool finishes more than half that many in a frame, since
/// the target is twice the number finished in the last frame.
const MIN_TASKS_PER_THREAD: usize = 4;

/// A kind of job a voxel world runs on the task pool.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum JobKind {
    /// A relight of a chunk's light, in a world with light propagation.
    #[cfg(feature = "light_propagation")]
    Relight,
    /// A sky probe, in a world with light propagation.
    #[cfg(feature = "light_propagation")]
    SkyProbe,
    Chunk,
    /// A stand-in for a second kind of job, in tests.
    #[cfg(test)]
    Test,
}

impl JobKind {
    const COUNT: usize =
        1 + 2 * cfg!(feature = "light_propagation") as usize + cfg!(test) as usize;

    fn index(self) -> usize {
        self as usize
    }
}

/// How many jobs of each kind the voxel world `C` runs, and how many more it
/// may start.
///
/// The jobs of every kind share one running target, so jobs of one kind
/// running leave fewer slots for the others.
#[derive(Resource)]
pub(crate) struct PoolBudget<C> {
    running: [usize; JobKind::COUNT],
    finished_last_frame: [usize; JobKind::COUNT],
    /// The task pool's thread count, if set by a test instead of read from the
    /// pool.
    threads: Option<usize>,
    _marker: PhantomData<C>,
}

impl<C> Default for PoolBudget<C> {
    fn default() -> Self {
        Self {
            running: [0; JobKind::COUNT],
            finished_last_frame: [0; JobKind::COUNT],
            threads: None,
            _marker: PhantomData,
        }
    }
}

impl<C> PoolBudget<C> {
    #[cfg(test)]
    fn with_threads(threads: usize) -> Self {
        Self {
            threads: Some(threads),
            ..default()
        }
    }

    /// Sets how many jobs of `kind` are running.
    pub fn set_running(&mut self, kind: JobKind, count: usize) {
        self.running[kind.index()] = count;
    }

    /// Counts a job of `kind` that just started.
    pub fn started(&mut self, kind: JobKind) {
        self.running[kind.index()] += 1;
    }

    /// Sets how many jobs of `kind` finished in the last frame.
    pub fn set_finished_last_frame(&mut self, kind: JobKind, count: usize) {
        self.finished_last_frame[kind.index()] = count;
    }

    fn threads(&self) -> usize {
        self.threads
            .unwrap_or_else(|| AsyncComputeTaskPool::get().thread_num())
            .max(1)
    }

    /// How many jobs of all kinds to keep running.
    ///
    /// The target is enough to keep the task pool busy until the next frame,
    /// judged by how many jobs finished in the last one.
    fn running_target(&self) -> usize {
        let finished: usize = self.finished_last_frame.iter().sum();
        (2 * finished).max(MIN_TASKS_PER_THREAD * self.threads())
    }

    /// How many task pool threads no job of any kind keeps busy.
    #[cfg(feature = "light_propagation")]
    pub fn idle(&self) -> usize {
        let running: usize = self.running.iter().sum();
        self.threads().saturating_sub(running)
    }

    /// Whether another job of `kind` may start with at most `cap` of its kind
    /// running, regardless of the running target.
    pub fn has_room(&self, kind: JobKind, cap: usize) -> bool {
        self.running[kind.index()] < cap
    }

    /// How many more jobs of `kind` to start now, with at most `cap` of its
    /// kind running.
    pub fn slots(&self, kind: JobKind, cap: usize) -> usize {
        let running: usize = self.running.iter().sum();
        let target = self.running_target().saturating_sub(running);
        target.min(cap.saturating_sub(self.running[kind.index()]))
    }

    /// Starts chunk tasks for the Pending `chunks`, in start order.
    ///
    /// Priority chunks start first, oldest first. They are few and something
    /// waits on them, so they start whenever `cap` allows. The others start in
    /// their order, only as many as the running target leaves slots for.
    pub fn start_chunk_tasks<P: PendingChunks>(&mut self, cap: usize, chunks: &mut P) {
        while self.has_room(JobKind::Chunk, cap) {
            let Some(entity) = chunks.pop_priority() else {
                break;
            };
            if chunks.start(entity) {
                self.started(JobKind::Chunk);
            }
        }

        let slots = self.slots(JobKind::Chunk, cap);
        if slots == 0 {
            return;
        }
        for entity in first_in_start_order(chunks.others(), slots) {
            if chunks.start(entity) {
                self.started(JobKind::Chunk);
            }
        }
    }

    /// How many more chunks could start a chunk task right away, beyond the
    /// `waiting` chunks already Pending, with at most `cap` chunk tasks
    /// running.
    pub fn spawn_budget(&self, cap: usize, waiting: usize) -> usize {
        self.slots(JobKind::Chunk, cap).saturating_sub(waiting)
    }
}

/// The Pending chunks of a voxel world, as [`PoolBudget::start_chunk_tasks`]
/// sees them.
pub(crate) trait PendingChunks {
    /// What orders the chunks without priority. The smallest starts first.
    type Order: Ord + Copy;

    /// Takes the oldest priority chunk off its queue. The chunk may since
    /// have moved on, in which case starting it does nothing.
    fn pop_priority(&mut self) -> Option<Entity>;

    /// The Pending chunks without priority, each with its order.
    fn others(&self) -> Vec<(Self::Order, Entity)>;

    /// Starts the chunk's task. Returns whether it started.
    fn start(&mut self, entity: Entity) -> bool;
}

/// The `count` candidates first in start order, in the order to start them.
///
/// The task pool runs tasks in the order they start, so the earliest in start
/// order starts first.
pub(crate) fn first_in_start_order<O: Ord + Copy, T>(
    mut candidates: Vec<(O, T)>,
    count: usize,
) -> Vec<T> {
    if count == 0 {
        return Vec::new();
    }
    if candidates.len() > count {
        candidates.select_nth_unstable_by_key(count - 1, |&(order, _)| order);
        candidates.truncate(count);
    }
    candidates.sort_unstable_by_key(|&(order, _)| order);
    candidates.into_iter().map(|(_, item)| item).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    struct World;

    fn budget(threads: usize) -> PoolBudget<World> {
        PoolBudget::with_threads(threads)
    }

    #[test]
    fn with_only_chunk_tasks_the_running_target_is_twice_the_finished_or_four_per_thread()
    {
        let mut budget = budget(2);
        assert_eq!(budget.running_target(), 8);
        budget.set_finished_last_frame(JobKind::Chunk, 3);
        assert_eq!(budget.running_target(), 8);
        budget.set_finished_last_frame(JobKind::Chunk, 10);
        assert_eq!(budget.running_target(), 20);
    }

    #[test]
    fn with_only_chunk_tasks_the_slots_and_spawn_budget_are_the_capped_target_less_the_running(
    ) {
        let mut budget = budget(2);
        budget.set_finished_last_frame(JobKind::Chunk, 10);
        budget.set_running(JobKind::Chunk, 5);
        assert_eq!(budget.slots(JobKind::Chunk, 100), 15);
        assert_eq!(budget.slots(JobKind::Chunk, 12), 7);
        assert_eq!(budget.spawn_budget(12, 4), 3);
        assert_eq!(budget.spawn_budget(12, 9), 0);
        assert_eq!(budget.spawn_budget(100, 4), 11);

        budget.set_running(JobKind::Chunk, 14);
        assert_eq!(budget.slots(JobKind::Chunk, 12), 0);
        assert!(!budget.has_room(JobKind::Chunk, 12));
        assert!(budget.has_room(JobKind::Chunk, 15));
    }

    #[test]
    fn a_started_job_takes_a_slot() {
        let mut budget = budget(1);
        budget.set_running(JobKind::Chunk, 2);
        budget.started(JobKind::Chunk);
        assert_eq!(budget.slots(JobKind::Chunk, 100), 1);
        assert!(!budget.has_room(JobKind::Chunk, 3));
    }

    #[test]
    fn jobs_of_another_kind_running_reduce_how_many_chunk_tasks_start() {
        let mut budget = budget(2);
        budget.set_running(JobKind::Chunk, 2);
        budget.set_running(JobKind::Test, 3);
        assert_eq!(budget.slots(JobKind::Chunk, 100), 3);
        assert_eq!(budget.spawn_budget(100, 1), 2);

        budget.set_running(JobKind::Test, 10);
        assert_eq!(budget.slots(JobKind::Chunk, 100), 0);
        assert_eq!(budget.spawn_budget(100, 0), 0);
    }

    #[test]
    fn jobs_of_another_kind_running_leave_the_chunk_cap_alone() {
        let mut budget = budget(2);
        budget.set_running(JobKind::Chunk, 2);
        budget.set_running(JobKind::Test, 10);
        assert!(budget.has_room(JobKind::Chunk, 3));
        assert!(!budget.has_room(JobKind::Chunk, 2));
    }

    #[test]
    fn jobs_of_every_kind_finished_last_frame_raise_the_target() {
        let mut budget = budget(1);
        budget.set_finished_last_frame(JobKind::Chunk, 4);
        budget.set_finished_last_frame(JobKind::Test, 6);
        assert_eq!(budget.running_target(), 20);
    }

    /// Pending chunks that record the order their tasks start in.
    struct Chunks {
        priority: Vec<Entity>,
        others: Vec<(u32, Entity)>,
        started: Vec<Entity>,
    }

    impl Chunks {
        fn new(priority: &[u32], others: &[(u32, u32)]) -> Self {
            Self {
                priority: priority.iter().rev().map(|&i| entity(i)).collect(),
                others: others
                    .iter()
                    .map(|&(order, i)| (order, entity(i)))
                    .collect(),
                started: Vec::new(),
            }
        }

        fn started(&self) -> Vec<u32> {
            self.started
                .iter()
                .map(|entity| entity.index_u32())
                .collect()
        }
    }

    impl PendingChunks for Chunks {
        type Order = u32;

        fn pop_priority(&mut self) -> Option<Entity> {
            self.priority.pop()
        }

        fn others(&self) -> Vec<(u32, Entity)> {
            self.others.clone()
        }

        fn start(&mut self, entity: Entity) -> bool {
            self.others.retain(|&(_, other)| other != entity);
            self.started.push(entity);
            true
        }
    }

    fn entity(index: u32) -> Entity {
        Entity::from_raw_u32(index).unwrap()
    }

    #[test]
    fn priority_chunks_start_first_oldest_first_then_the_others_in_their_order() {
        let mut budget = budget(1);
        let mut chunks = Chunks::new(&[7, 8], &[(30, 3), (10, 1), (20, 2), (40, 4)]);
        budget.start_chunk_tasks(100, &mut chunks);
        assert_eq!(chunks.started(), vec![7, 8, 1, 2]);
    }

    #[test]
    fn priority_chunks_start_up_to_the_cap_past_the_running_target() {
        let mut budget = budget(1);
        let priority: Vec<u32> = (1..=6).collect();
        let mut chunks = Chunks::new(&priority, &[(1, 10)]);
        budget.start_chunk_tasks(5, &mut chunks);
        assert_eq!(chunks.started(), vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn jobs_of_another_kind_running_start_fewer_chunk_tasks() {
        let mut budget = budget(1);
        budget.set_running(JobKind::Test, 3);
        let mut chunks = Chunks::new(&[], &[(3, 3), (1, 1), (2, 2)]);
        budget.start_chunk_tasks(100, &mut chunks);
        assert_eq!(chunks.started(), vec![1]);
    }

    #[test]
    fn the_first_in_start_order_come_out_in_that_order() {
        let candidates = vec![(5, 'e'), (1, 'a'), (4, 'd'), (2, 'b'), (3, 'c')];
        assert_eq!(
            first_in_start_order(candidates.clone(), 3),
            vec!['a', 'b', 'c']
        );
        assert_eq!(
            first_in_start_order(candidates, 10),
            vec!['a', 'b', 'c', 'd', 'e']
        );
    }
}
