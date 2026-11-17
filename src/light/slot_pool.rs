//! The bookkeeping of a texture of slots: which slots are free, which chunk
//! holds each one, and which holder gives way when every slot is taken.
//!
//! A [`SlotPool`] hands out slots by index and knows nothing of what a slot
//! holds or what its holder shows without one. An adapter, a type
//! implementing [`Holders`], supplies those: how far a holder is from the
//! cameras, and what it shows once it gives its slot up. The route's light
//! slots are one such adapter.
//!
//! The pool hands out slots given back first, the last one first, and
//! otherwise the next slot never handed out. The texture's shape is set
//! when the first slot is taken, and the slot count is clamped then to the
//! largest 3D texture the device makes, with a warning. When every slot is
//! taken, the holder farthest from the cameras gives way to a nearer chunk,
//! never to a farther one, and the pool warns once. Finding that holder
//! costs a pass over every holder, so it grows with the slot count.

use bevy::log::warn_once;
use bevy::platform::collections::HashMap;
use bevy::prelude::*;

use super::route::SLOT_SIDE;

/// What a pool's slots are for, supplied to the pool by the route that owns
/// it: how far a holder is, and what a holder shows once it gives its slot
/// up.
pub(crate) trait Holders<H> {
    /// What the pool's slots are called in its warnings, such as
    /// "light slots".
    const SLOTS: &'static str;

    /// What a holder shows once it gives its slot up, as said in the
    /// pool's warning that every slot is taken.
    const WITHOUT_SLOT: &'static str;

    /// How far `holder` is from the nearest camera.
    fn distance(&self, holder: H) -> f32;

    /// Makes `holder` show what it shows without a slot, after it gave its
    /// slot up to a nearer chunk.
    fn gave_way(&mut self, holder: H);
}

/// The slots of one texture, and the holder of each slot taken.
pub(crate) struct SlotPool<H> {
    /// The slot count, clamped to the device once the texture's shape is set.
    capacity: u32,
    /// How many slots the texture holds along each axis, once the first slot
    /// is taken.
    shape: Option<UVec3>,
    /// The device's largest 3D texture side, if there's a device.
    max_side: Option<u32>,
    /// The lowest slot never handed out.
    next: u32,
    /// Slots given back, the last one first.
    free: Vec<u32>,
    holders: HashMap<u32, H>,
    /// How many times the pool warned that every slot is taken.
    full_warnings: u32,
}

impl<H: Copy> SlotPool<H> {
    /// A pool of `capacity` slots, none of them taken.
    pub fn new(capacity: u32) -> Self {
        Self {
            capacity,
            shape: None,
            max_side: None,
            next: 0,
            free: Vec::new(),
            holders: HashMap::default(),
            full_warnings: 0,
        }
    }

    /// Notes the device's largest 3D texture side, if there is a device,
    /// which the slot count is clamped to when the texture's shape is set.
    /// Once it's set this does nothing.
    pub fn set_device(&mut self, max_side: Option<u32>) {
        if self.shape.is_none() {
            self.max_side = max_side;
        }
    }

    /// Empties the pool and makes it `capacity` slots, without telling the
    /// holders. The texture's shape is set again when a slot is next taken,
    /// and the pool warns again when every slot is taken.
    pub fn reset(&mut self, capacity: u32) {
        *self = Self {
            max_side: self.max_side,
            ..Self::new(capacity)
        };
    }

    /// How many slots the texture holds along each axis, once the first slot
    /// is taken.
    pub fn shape(&self) -> Option<UVec3> {
        self.shape
    }

    /// A slot for `holder`, `near` from the cameras: a free one, or else the
    /// slot of the holder farthest from the cameras if it's farther than
    /// `near`, which `holders` then shows without a slot. Returns `None` if
    /// every holder is as near as `near` or nearer.
    pub fn take<A: Holders<H>>(
        &mut self,
        holder: H,
        near: f32,
        holders: &mut A,
    ) -> Option<u32> {
        if self.shape.is_none() {
            self.set_shape::<A>();
        }
        let slot = match self.free.pop() {
            Some(slot) => slot,
            None if self.next < self.capacity => {
                self.next += 1;
                self.next - 1
            }
            None => self.take_farthest(near, holders)?,
        };
        self.holders.insert(slot, holder);
        Some(slot)
    }

    /// Gives `slot` back, so it's handed out again.
    pub fn give_back(&mut self, slot: u32) {
        if self.holders.remove(&slot).is_some() {
            self.free.push(slot);
        }
    }

    /// Whether a slot can be taken without a holder giving way.
    pub fn has_free(&self) -> bool {
        !self.free.is_empty() || self.next < self.capacity
    }

    /// How many slots are held.
    pub fn held(&self) -> usize {
        self.holders.len()
    }

    /// Whether the pool has warned that every slot is taken.
    pub fn warned_full(&self) -> bool {
        self.full_warnings > 0
    }

    /// Sets the texture's shape, clamping the slot count to the device.
    fn set_shape<A: Holders<H>>(&mut self) {
        let (capacity, axes) = texture_shape(self.capacity, self.max_side);
        if capacity < self.capacity {
            warn_once!(
                "light propagation: slot_capacity {} is more than this device's 3D \
                 textures hold, using {capacity} {}",
                self.capacity,
                A::SLOTS,
            );
        }
        self.capacity = capacity;
        self.shape = Some(axes);
    }

    /// Takes the slot of the holder farthest from the cameras, if it's
    /// farther than `near`, and has `holders` show it without a slot.
    fn take_farthest<A: Holders<H>>(
        &mut self,
        near: f32,
        holders: &mut A,
    ) -> Option<u32> {
        if self.full_warnings == 0 {
            self.full_warnings += 1;
            warn!(
                "light propagation: every one of the {} {} is taken, so the chunks \
                 farthest from the camera {}; raise slot_capacity to give them slots",
                self.capacity,
                A::SLOTS,
                A::WITHOUT_SLOT,
            );
        }
        let (slot, holder, far) = self
            .holders
            .iter()
            .map(|(slot, holder)| (*slot, *holder, holders.distance(*holder)))
            .max_by(|a, b| a.2.total_cmp(&b.2))?;
        if far <= near {
            return None;
        }
        self.holders.remove(&slot);
        holders.gave_way(holder);
        Some(slot)
    }

    /// How many times the pool warned that every slot is taken.
    #[cfg(test)]
    fn full_warnings(&self) -> u32 {
        self.full_warnings
    }
}

/// How many slots the texture holds along each of its axes, for `capacity`
/// slots on a device whose 3D textures are at most `max_side` texels on a
/// side: the smallest axis doubles, x first, until they hold them all.
///
/// Returns the slot count the texture holds, which is less than `capacity`
/// only if the device can't make a texture that large. Then each axis holds
/// as many slots as fit.
pub(crate) fn texture_shape(capacity: u32, max_side: Option<u32>) -> (u32, UVec3) {
    let limit = max_side.map_or(u32::MAX, |side| (side / SLOT_SIDE).max(1));
    let mut axes = UVec3::ONE;
    while axes.element_product() < capacity {
        let Some(axis) = (0..3)
            .filter(|&axis| axes[axis] * 2 <= limit)
            .min_by_key(|&axis| axes[axis])
        else {
            let axes = UVec3::splat(limit);
            return (axes.element_product().min(capacity), axes);
        };
        axes[axis] *= 2;
    }
    (capacity, axes)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Holders numbered by their distance from the cameras, noting which
    /// gave way.
    #[derive(Default)]
    struct ByDistance {
        gave_way: Vec<u32>,
    }

    impl Holders<u32> for ByDistance {
        const SLOTS: &'static str = "test slots";
        const WITHOUT_SLOT: &'static str = "show nothing";

        fn distance(&self, holder: u32) -> f32 {
            holder as f32
        }

        fn gave_way(&mut self, holder: u32) {
            self.gave_way.push(holder);
        }
    }

    #[test]
    fn a_slot_given_back_is_handed_out_again() {
        let mut holders = ByDistance::default();
        let mut pool = SlotPool::new(2);
        let first = pool.take(1, 1.0, &mut holders).unwrap();
        let second = pool.take(2, 2.0, &mut holders).unwrap();
        assert_ne!(first, second);
        assert!(!pool.has_free());

        pool.give_back(first);
        assert!(pool.has_free());
        assert_eq!(pool.take(3, 3.0, &mut holders), Some(first));
        assert_eq!(pool.held(), 2);
        assert!(holders.gave_way.is_empty());
    }

    #[test]
    fn when_full_the_farthest_holder_gives_way_to_a_nearer_chunk_but_never_to_a_farther_one(
    ) {
        let mut holders = ByDistance::default();
        let mut pool = SlotPool::new(2);
        let near = pool.take(10, 10.0, &mut holders).unwrap();
        let far = pool.take(30, 30.0, &mut holders).unwrap();

        assert_eq!(pool.take(40, 40.0, &mut holders), None);
        assert_eq!(pool.take(30, 30.0, &mut holders), None);
        assert!(holders.gave_way.is_empty());

        assert_eq!(pool.take(20, 20.0, &mut holders), Some(far));
        assert_eq!(holders.gave_way, [30]);
        assert_eq!(pool.take(15, 15.0, &mut holders), Some(far));
        assert_eq!(holders.gave_way, [30, 20]);
        assert_eq!(pool.take(12, 12.0, &mut holders), Some(far));
        assert_eq!(holders.gave_way, [30, 20, 15]);
        assert_eq!(pool.held(), 2);

        // The nearest holder keeps its slot until a nearer chunk comes.
        pool.give_back(far);
        assert_eq!(pool.take(5, 5.0, &mut holders), Some(far));
        assert_eq!(pool.take(1, 1.0, &mut holders), Some(near));
        assert_eq!(holders.gave_way, [30, 20, 15, 10]);
    }

    #[test]
    fn the_devices_limit_clamps_the_slot_count() {
        let mut holders = ByDistance::default();
        let mut pool = SlotPool::new(1024);
        // A side of 256 texels holds 7 slots of 34 texels along each axis.
        pool.set_device(Some(256));
        let mut taken = 0;
        while pool.has_free() {
            pool.take(taken, taken as f32, &mut holders).unwrap();
            taken += 1;
        }
        assert_eq!(taken, 343);
        assert_eq!(pool.shape(), Some(UVec3::splat(7)));

        // The device is read only until the shape is set.
        pool.set_device(Some(4096));
        assert!(!pool.has_free());
    }

    #[test]
    fn the_pool_warns_once_that_every_slot_is_taken() {
        let mut holders = ByDistance::default();
        let mut pool = SlotPool::new(1);
        pool.take(10, 10.0, &mut holders).unwrap();
        assert!(!pool.warned_full());

        for near in [20.0, 5.0, 2.0, 30.0] {
            pool.take(near as u32, near, &mut holders);
        }
        assert!(pool.warned_full());
        assert_eq!(pool.full_warnings(), 1);

        pool.reset(1);
        assert!(!pool.warned_full());
    }
}
