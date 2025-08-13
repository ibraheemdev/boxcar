#![allow(clippy::declare_interior_mutable_const)]

use core::mem::{self, MaybeUninit};
use core::ops::Index;
use core::panic::RefUnwindSafe;
use core::ptr;

use crate::buckets::{self, buckets_for_index_bits, Buckets, MaybeZeroable};
use crate::loom::atomic::{AtomicBool, AtomicUsize, Ordering};
use crate::loom::cell::UnsafeCell;
use crate::loom::AtomicMut;

/// A lock-free, append-only vector.
pub struct Vec<T> {
    /// A counter used to retrieve a unique index to push to.
    ///
    /// Stores a `buckets::Index` that may be out-of-bounds on
    /// the positive side, but never overflows.
    ///
    /// This value may be more than the true length as it will
    /// be incremented before values are actually stored.
    inflight: AtomicUsize,

    /// Buckets of length 32, 64 .. 2^62.
    buckets: Buckets<Entry<T>, BUCKETS>,

    /// The number of initialized elements in this vector.
    count: AtomicUsize,
}

impl<T> Vec<T> {
    /// Create an empty vector.
    #[cfg(not(loom))]
    pub const fn new() -> Vec<T> {
        let Some(zero) = <buckets::Index<BUCKETS>>::new(0) else {
            unreachable!();
        };

        Vec {
            inflight: AtomicUsize::new(zero.into_raw().get()),
            buckets: Buckets::new(),
            count: AtomicUsize::new(0),
        }
    }

    /// Create an empty vector.
    #[cfg(loom)]
    pub fn new() -> Vec<T> {
        Vec {
            inflight: AtomicUsize::new(<buckets::Index<BUCKETS>>::new(0).unwrap().into_raw().get()),
            buckets: Buckets::new(),
            count: AtomicUsize::new(0),
        }
    }

    /// Constructs a new, empty `Vec<T>` with the specified capacity.
    #[inline]
    pub fn with_capacity(capacity: usize) -> Vec<T> {
        let mut this = Self::new();

        if let Some(highest_index) = capacity.checked_sub(1) {
            this.buckets
                .reserve_mut(buckets::Index::new_saturating(highest_index));
        }

        this
    }

    /// Returns the number of elements in the vector.
    #[inline]
    pub fn count(&self) -> usize {
        // The `Acquire` here synchronizes with the `Release` increment
        // when an entry is added to the vector.
        self.count.load(Ordering::Acquire)
    }

    /// Returns a reference to the element at the given index.
    #[inline]
    pub fn get(&self, index: usize) -> Option<&T> {
        // If the index is out-of-bounds, or the bucket is not allocated, return `None`.
        let entry = self.buckets.get(buckets::Index::new(index)?)?;

        if entry.active.load(Ordering::Acquire) {
            // Safety: The entry is active.
            unsafe { return Some(entry.value_unchecked()) }
        }

        // The entry is uninitialized.
        None
    }

    /// Returns a reference to the element at the given index.
    ///
    /// # Safety
    ///
    /// The entry at `index` must be initialized.
    #[inline]
    pub unsafe fn get_unchecked(&self, index: usize) -> &T {
        // Safety: Caller guarantees the bucket and entry are initialized.
        unsafe {
            let entry = self
                .buckets
                .get_unchecked(buckets::Index::new_unchecked(index));
            (*entry).value_unchecked()
        }
    }

    /// Returns a mutable reference to the element at the given index.
    #[inline]
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        // If the index is out-of-bounds, or the bucket is not allocated, return `None`.
        let entry = self.buckets.get_mut(buckets::Index::new(index)?)?;

        if entry.active.read_mut() {
            // Safety: The entry is active.
            unsafe { return Some(entry.value_unchecked_mut()) }
        }

        // The entry is uninitialized.
        None
    }

    ///
    /// # Safety
    ///
    /// The entry at `index` must be initialized.
    #[inline]
    pub unsafe fn get_unchecked_mut(&mut self, index: usize) -> &mut T {
        // Safety: Caller guarantees the bucket and entry are initialized.
        unsafe {
            let entry = self
                .buckets
                .get_unchecked_mut(buckets::Index::new_unchecked(index));
            (*entry).value_unchecked_mut()
        }
    }

    /// Returns a unique index for insertion.
    #[inline]
    fn next_index(&self) -> buckets::Index<BUCKETS> {
        // The inflight counter cannot exceed `isize::MAX`, allowing it to catch capacity overflow.
        //
        // Note that the `Relaxed` ordering here is sufficient, as we only care about
        // the index being unique and do not use it for synchronization.
        let index = self.inflight.fetch_add(1, Ordering::Relaxed);

        // Safety: We cannot overflow.
        let Some(index) = (unsafe { buckets::Index::from_raw_checked_above(index) }) else {
            self.next_index_overflow();
        };

        index
    }

    #[cold]
    #[inline(never)]
    fn next_index_overflow(&self) -> ! {
        // We could alternatively abort here, as `Arc` does. But we decrement and panic instead
        // to keep in line with `Vec`'s behavior. Assuming that `isize::MAX` concurrent threads
        // don't call this method, it is still impossible for it to overflow.
        self.inflight.fetch_sub(1, Ordering::Relaxed);
        panic!("capacity overflow");
    }

    /// Appends the element returned from the closure to the back of the vector
    /// at the index represented by the `usize` passed to closure.
    ///
    /// This allows for use of the would-be index to be utilized within the
    /// element.
    #[inline]
    pub fn push_with<F>(&self, f: F) -> usize
    where
        F: FnOnce(usize) -> T,
    {
        // Acquire a unique index to insert into.
        let index = self.next_index();
        let value = f(index.get());

        // Safety: `next_index` is always in-bounds and unique.
        unsafe { self.write(index, value) }
    }

    /// Appends an element to the back of the vector.
    #[inline]
    pub fn push(&self, value: T) -> usize {
        // Safety: `next_index` is always in-bounds and unique.
        unsafe { self.write(self.next_index(), value) }
    }

    /// Write an element at the given index.
    ///
    /// # Safety
    ///
    /// The index must be unique.
    #[inline]
    unsafe fn write(&self, index: buckets::Index<BUCKETS>, value: T) -> usize {
        let entry = self.buckets.get_or_alloc(index);

        unsafe {
            // Safety: We have unique access to this entry (ensured by
            // the caller).
            //
            // 1. It is impossible for another thread to attempt a `push`
            // to this location as we retrieved it from `next_index`.
            //
            // 2. Any thread trying to `get` this entry will see `!active`
            // and will not try to access it.
            entry
                .slot
                .with_mut(|slot| slot.write(MaybeUninit::new(value)));

            // Let other threads know that this entry is active.
            //
            // Note that this `Release` write synchronizes with the `Acquire`
            // load in `Vec::get`.
            entry.active.store(true, Ordering::Release);
        }

        // Increase the element count.
        //
        // The `Release` here is not strictly necessary, but does
        // allow users to use `count` for some sort of synchronization
        // in terms of the number of initialized elements.
        self.count.fetch_add(1, Ordering::Release);

        // Return the index of the entry that we initialized.
        index.get()
    }

    /// Reserves capacity for at least `additional` more elements to be inserted in
    /// the vector.
    ///
    /// The collection may reserve more space to avoid frequent reallocations.
    pub fn reserve(&self, additional: usize) {
        let len = self.count.load(Ordering::Acquire);

        if let Some(highest_index) = len.saturating_add(additional).checked_sub(1) {
            self.buckets
                .reserve(buckets::Index::new_saturating(highest_index));
        }
    }

    /// Iterates over shared references to values in the vector.
    #[inline]
    pub fn iter(&self) -> Iter<'_, T> {
        Iter {
            inner: self.buckets.iter(),
            // Snapshot the number of potentially active entries when the iterator is
            // created, so we know how many entries we have to check. The alternative
            // is to always check every single entry and/or bucket. Note that yielding
            // `self.count` entries is not enough; we may end up yielding the *wrong*
            // `self.count` entries, as the gaps in-between the entries we want to yield
            // may get filled concurrently.
            // Safety: We cannot overflow.
            after_inflight: unsafe {
                buckets::Index::from_raw_checked_above(self.inflight.load(Ordering::Relaxed))
            },
        }
    }

    /// Iterates over owned values in the vector.
    #[inline]
    pub fn into_iter(mut self) -> IntoIter<T> {
        IntoIter {
            inner: self.buckets.into_iter(),
            remaining: self.count.read_mut(),
        }
    }

    /// Clear every element in the vector.
    #[inline]
    pub fn clear(&mut self) {
        // Consume and reset every entry in the vector.
        self.buckets
            .iter_mut()
            .filter_map(|(_, entry)| entry.take())
            .take(self.count.read_mut())
            .for_each(drop);

        // Reset the count.
        self.count.write_mut(0);
        self.inflight
            .write_mut(<buckets::Index<BUCKETS>>::new(0).unwrap().into_raw().get());
    }
}

impl<T> Index<usize> for Vec<T> {
    type Output = T;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        #[cold]
        #[inline(never)]
        fn assert_failed(index: usize) -> ! {
            panic!("index {index} is uninitialized");
        }

        match self.get(index) {
            Some(value) => value,
            None => assert_failed(index),
        }
    }
}

/// A possibly uninitialized entry in the vector.
struct Entry<T> {
    /// A flag indicating whether or not this entry is initialized.
    active: AtomicBool,

    /// The entry's value.
    slot: UnsafeCell<MaybeUninit<T>>,
}

impl<T> Default for Entry<T> {
    fn default() -> Self {
        Self {
            active: AtomicBool::new(false),
            slot: UnsafeCell::new(MaybeUninit::uninit()),
        }
    }
}

// Safety: `AtomicBool` and `MaybeUninit` are all zeroable outside Loom.
unsafe impl<T> MaybeZeroable for Entry<T> {
    fn zeroable() -> bool {
        // Loom's types are not zeroable; thus we fall back to `Default`.
        cfg!(not(loom))
    }
}

// Safety: An `Entry` is owned and owns its data, so sending an
// entry only sends that data, hence `T: Send`.
unsafe impl<T: Send> Send for Entry<T> {}

// Safety: Sharing an `Entry` exposes shared access to the
// data inside, hence `T: Sync`. Additionally, the `Entry`
// may act as a channel, exposing owned access of elements
// to other threads, hence we also require `T: Send`.
unsafe impl<T: Send + Sync> Sync for Entry<T> {}

// In an ideal world, we might require `T: UnwindSafe + RefUnwindSafe`
// here, with similar reasoning to above. But this is here for backward
// compatibility. Also, who ever worried about unwind safety :P
impl<T> RefUnwindSafe for Entry<T> {}

impl<T> Entry<T> {
    /// Returns a reference to the value in this entry.
    ///
    /// # Safety
    ///
    /// The value must be initialized.
    #[inline]
    unsafe fn value_unchecked(&self) -> &T {
        // Safety: Guaranteed by caller.
        self.slot.with(|slot| unsafe { (*slot).assume_init_ref() })
    }

    /// Returns a mutable reference to the value in this entry.
    ///
    // # Safety
    //
    // The value must be initialized.
    #[inline]
    unsafe fn value_unchecked_mut(&mut self) -> &mut T {
        // Safety: Guaranteed by caller.
        self.slot
            .with_mut(|slot| unsafe { (*slot).assume_init_mut() })
    }

    /// Takes the value in the entry, if there is one.
    fn take(&mut self) -> Option<T> {
        if self.active.read_mut() {
            // Mark the entry as uninitialized so it is not accessed after this.
            self.active.write_mut(false);

            let value = mem::replace(&mut self.slot, UnsafeCell::new(MaybeUninit::uninit()));

            // Safety: We just verified that the entry is initialized.
            Some(unsafe { value.into_inner().assume_init() })
        } else {
            None
        }
    }
}

impl<T> Drop for Entry<T> {
    fn drop(&mut self) {
        if self.active.read_mut() {
            // Safety: We have `&mut self` and verifid that the value is initialized.
            unsafe { ptr::drop_in_place(self.slot.with_mut(|slot| (*slot).as_mut_ptr())) }
        }
    }
}

/// The number of buckets in a vector.
const BUCKETS: usize = buckets_for_index_bits(usize::BITS - 1);

pub struct Iter<'a, T> {
    inner: buckets::Iter<'a, Entry<T>, BUCKETS>,
    after_inflight: Option<buckets::Index<BUCKETS>>,
}

impl<T> Clone for Iter<'_, T> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            after_inflight: self.after_inflight,
        }
    }
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = (usize, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let (index, entry) = self.inner.next()?;

            if self
                .after_inflight
                .is_some_and(|after_inflight| index >= after_inflight)
            {
                return None;
            }

            if entry.active.load(Ordering::Acquire) {
                // Safety: We just verified that the entry is initialized.
                return Some((index.get(), unsafe { entry.value_unchecked() }));
            }
        }
    }
}

pub struct IntoIter<T> {
    inner: buckets::IntoIter<Entry<T>, BUCKETS>,
    remaining: usize,
}

impl<T> Iterator for IntoIter<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            self.remaining = self.remaining.checked_sub(1)?;

            let (_, mut entry) = self.inner.next()?;
            if let Some(value) = entry.take() {
                return Some(value);
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<T> ExactSizeIterator for IntoIter<T> {
    fn len(&self) -> usize {
        self.remaining
    }
}
