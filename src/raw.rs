#![allow(clippy::declare_interior_mutable_const)]

use core::mem::{self, MaybeUninit};
use core::ops::Index;
use core::{ptr, slice};

use crate::loom::atomic::{AtomicBool, AtomicPtr, AtomicUsize, Ordering};
use crate::loom::cell::UnsafeCell;
use crate::loom::AtomicMut;

use alloc::boxed::Box;

#[cfg(target_has_atomic = "64")]
type Inflight = crate::loom::atomic::AtomicU64;

#[cfg(not(target_has_atomic = "64"))]
type Inflight = crate::loom::atomic::AtomicUsize;

/// A lock-free, append-only vector.
pub struct Vec<T> {
    /// A counter used to retrieve a unique index to push to.
    ///
    /// This value may be more than the true length as it will
    /// be incremented before values are actually stored.
    inflight: Inflight,

    /// Buckets of length 32, 64 .. 2^63.
    buckets: [Bucket<T>; BUCKETS],

    /// The number of initialized elements in this vector.
    count: AtomicUsize,
}

/// Safety: A `Vec` owns its elements, so sending a
/// vector also sends its elements, hence `T: Send`.
unsafe impl<T: Send> Send for Vec<T> {}

/// Safety: Sharing a `Vec` only exposes shared access
/// to the elements inside, so we only require `T: Sync`.
unsafe impl<T: Sync> Sync for Vec<T> {}

impl<T> Vec<T> {
    /// An empty vector.
    #[cfg(not(loom))]
    const EMPTY: Vec<T> = Vec {
        inflight: Inflight::new(0),
        buckets: [Bucket::EMPTY; BUCKETS],
        count: AtomicUsize::new(0),
    };

    /// Create an empty vector.
    #[cfg(not(loom))]
    pub const fn new() -> Vec<T> {
        Vec::EMPTY
    }

    /// Create an empty vector.
    #[cfg(loom)]
    pub fn new() -> Vec<T> {
        Vec {
            inflight: Inflight::new(0),
            buckets: [0; BUCKETS].map(|_| Bucket {
                entries: AtomicPtr::new(ptr::null_mut()),
            }),
            count: AtomicUsize::new(0),
        }
    }

    /// Constructs a new, empty `Vec<T>` with the specified capacity.
    #[inline]
    pub fn with_capacity(capacity: usize) -> Vec<T> {
        let init = match capacity {
            0 => 0,
            // Initialize enough buckets for the n'th element to be inserted.
            n => Location::of(n - 1).bucket,
        };

        let mut vec = Vec::new();
        for (i, bucket) in vec.buckets[..=init].iter_mut().enumerate() {
            // Initialize each bucket.
            let len = Location::bucket_capacity(i);
            *bucket = Bucket::from_ptr(Bucket::alloc(len));
        }

        vec
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
        let location = Location::of(index);

        // Safety: `location.bucket` is always in bounds.
        //
        // The `Acquire` load here synchronizes with the `Release`
        // store in `Vec::get_or_alloc`.
        let entries = unsafe {
            self.buckets
                .get_unchecked(location.bucket)
                .entries
                .load(Ordering::Acquire)
        };

        // The bucket is uninitialized.
        if entries.is_null() {
            return None;
        }

        // Safety: `location.entry` is always in bounds for its bucket.
        let entry = unsafe { &*entries.add(location.entry) };

        if entry.active.load(Ordering::Acquire) {
            // Safety: The entry is active. Additionally, the `Acquire`
            // load synchronizes with the `Release` store in `Vec::write`,
            // ensuring the initialization happens-before this read.
            unsafe { return Some(entry.value_unchecked()) }
        }

        // The entry is uninitialized.
        None
    }

    /// Returns a reference to the element at the given index.
    #[inline]
    pub fn get_or_default(&self, index: usize) -> &T
    where
        T: Default,
    {
        let location = Location::of(index);

        // Safety: `location.bucket` is always in bounds.
        let bucket = unsafe { self.buckets.get_unchecked(location.bucket) };

        // The `Acquire` load here synchronizes with the `Release`
        // store in `Vec::get_or_alloc`.
        let mut entries = bucket.entries.load(Ordering::Acquire);

        // The bucket is uninitialized.
        if entries.is_null() {
            entries = Vec::get_or_alloc_default(bucket, location.bucket_len);
        }

        // Safety: `location.entry` is always in bounds for its bucket.
        let entry = unsafe { &*entries.add(location.entry) };

        if !entry.active.load(Ordering::Relaxed) {
            // Let other threads know that this entry is active.
            entry.active.store(true, Ordering::Release);
        }

        // Safety: The entry is active. Additionally, the `Acquire`
        // load synchronizes with the `Release` store in `Vec::write`,
        // ensuring the initialization happens-before this read.
        unsafe { return entry.value_unchecked() }
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
            let location = Location::of_unchecked(index);

            // The `Acquire` load here synchronizes with the `Release`
            // store in `Vec::get_or_alloc`.
            let entry = self
                .buckets
                .get_unchecked(location.bucket)
                .entries
                .load(Ordering::Acquire)
                .add(location.entry);

            (*entry).value_unchecked()
        }
    }

    /// Returns a mutable reference to the element at the given index.
    #[inline]
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        let location = Location::of(index);

        // Safety: `location.bucket` is always in bounds.
        let entries = unsafe {
            self.buckets
                .get_unchecked_mut(location.bucket)
                .entries
                .read_mut()
        };

        // The bucket is uninitialized.
        if entries.is_null() {
            return None;
        }

        // Safety: `location.entry` is always in bounds for its bucket.
        let entry = unsafe { &mut *entries.add(location.entry) };

        if entry.active.read_mut() {
            // Safety: The entry is active.
            unsafe { return Some(entry.value_unchecked_mut()) }
        }

        // The entry is uninitialized.
        None
    }

    /// Returns a mutable reference to the element at the given index.
    ///
    /// # Safety
    ///
    /// The entry at `index` must be initialized.
    #[inline]
    pub unsafe fn get_unchecked_mut(&mut self, index: usize) -> &mut T {
        // Safety: Caller guarantees the bucket and entry are initialized.
        unsafe {
            let location = Location::of_unchecked(index);

            let entry = self
                .buckets
                .get_unchecked_mut(location.bucket)
                .entries
                .read_mut()
                .add(location.entry);

            (*entry).value_unchecked_mut()
        }
    }

    /// Returns a unique index for insertion.
    #[inline]
    fn next_index(&self) -> usize {
        #[cfg(target_has_atomic = "64")]
        {
            // The inflight counter is an `AtomicU64`, allowing it to catch capacity overflow.
            //
            // Note that the `Relaxed` ordering here is sufficient, as we only care about
            // the index being unique and do not use it for synchronization.
            let index = self.inflight.fetch_add(1, Ordering::Relaxed);
            assert!(index <= (MAX_INDEX as u64), "capacity overflow");
            index as usize
        }

        #[cfg(not(target_has_atomic = "64"))]
        {
            // We must use a CAS loop to avoid wrapping on overflow.
            //
            // The `Relaxed` ordering is sufficient for the same reason as above.
            self.inflight
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |index| {
                    assert!(index <= MAX_INDEX, "capacity overflow");
                    Some(index + 1)
                })
                .unwrap()
        }
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
        let value = f(index);

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
    /// The index must be unique and in-bounds.
    #[inline]
    unsafe fn write(&self, index: usize, value: T) -> usize {
        // Safety: Caller guarantees the entry is initialized.
        let location = unsafe { Location::of_unchecked(index) };

        // Eagerly allocate the next bucket if we are close to the end of this one.
        if index == (location.bucket_len - (location.bucket_len >> 3)) {
            if let Some(next_bucket) = self.buckets.get(location.bucket + 1) {
                Vec::get_or_alloc(next_bucket, location.bucket_len << 1);
            }
        }

        // Safety: `location.bucket` is always in bounds.
        let bucket = unsafe { self.buckets.get_unchecked(location.bucket) };

        // The `Acquire` load here synchronizes with the `Release`
        // store in `Vec::get_or_alloc`.
        let mut entries = bucket.entries.load(Ordering::Acquire);

        // The bucket has not been allocated yet.
        if entries.is_null() {
            entries = Vec::get_or_alloc(bucket, location.bucket_len);
        }

        unsafe {
            // Safety: We loaded the entries pointer with `Acquire` ordering,
            // ensuring that it's initialization happens-before our access.
            //
            // Additionally, `location.entry` is always in bounds for its bucket.
            let entry = &*entries.add(location.entry);

            // Safety: We have unique access to this entry.
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
        index
    }

    /// Race to initialize a bucket.
    ///
    /// The returned pointer is guaranteed to be valid for access.
    ///
    /// Note that we avoid contention on bucket allocation by having a specified
    /// writer eagerly allocate the next bucket.
    #[cold]
    #[inline(never)]
    fn get_or_alloc(bucket: &Bucket<T>, len: usize) -> *mut Entry<T> {
        let entries = Bucket::alloc(len);

        match bucket.entries.compare_exchange(
            ptr::null_mut(),
            entries,
            // Establish synchronization with `Acquire` loads of the pointer.
            Ordering::Release,
            // If we lose the race, ensure that we synchronize with the initialization
            // of the bucket that won.
            Ordering::Acquire,
        ) {
            // We won the race.
            Ok(_) => entries,

            // We lost the race, deallocate our bucket and return the bucket that won.
            Err(found) => unsafe {
                Bucket::dealloc(entries, len);
                found
            },
        }
    }

    /// Race to initialize a bucket.
    ///
    /// The returned pointer is guaranteed to be valid for access.
    ///
    /// Note that we avoid contention on bucket allocation by having a specified
    /// writer eagerly allocate the next bucket.
    #[cold]
    #[inline(never)]
    fn get_or_alloc_default(bucket: &Bucket<T>, len: usize) -> *mut Entry<T>
    where
        T: Default,
    {
        let entries = Bucket::alloc_default(len);

        match bucket.entries.compare_exchange(
            ptr::null_mut(),
            entries,
            // Establish synchronization with `Acquire` loads of the pointer.
            Ordering::Release,
            // If we lose the race, ensure that we synchronize with the initialization
            // of the bucket that won.
            Ordering::Acquire,
        ) {
            // We won the race.
            Ok(_) => entries,

            // We lost the race, deallocate our bucket and return the bucket that won.
            Err(found) => unsafe {
                Bucket::dealloc(entries, len);
                found
            },
        }
    }

    /// Reserves capacity for at least `additional` more elements to be inserted in
    /// the vector.
    ///
    /// The collection may reserve more space to avoid frequent reallocations.
    pub fn reserve(&self, additional: usize) {
        let len = self.count.load(Ordering::Acquire);
        let mut location = Location::of(len.checked_add(additional).unwrap_or(MAX_INDEX));

        // Allocate buckets starting from the bucket at `len + additional` and
        // working our way backwards.
        loop {
            // Safety: `location.bucket` is always in bounds.
            let bucket = unsafe { self.buckets.get_unchecked(location.bucket) };

            // Reached an initialized bucket, we're done.
            //
            // `Relaxed` is sufficient here as we never access the bucket
            // and only ensure that it is initialized.
            if !bucket.entries.load(Ordering::Relaxed).is_null() {
                break;
            }

            // Allocate the bucket.
            Vec::get_or_alloc(bucket, location.bucket_len);

            // Reached the first bucket, we're done.
            if location.bucket == 0 {
                break;
            }

            location.bucket -= 1;
            location.bucket_len = Location::bucket_capacity(location.bucket);
        }
    }

    // Returns an iterator over the vector.
    #[inline]
    pub fn iter(&self) -> Iter {
        Iter {
            index: 0,
            yielded: 0,
            location: Location {
                bucket: 0,
                entry: 0,
                bucket_len: Location::bucket_capacity(0),
            },
        }
    }

    /// Clear every element in the vector.
    #[inline]
    pub fn clear(&mut self) {
        let mut iter = self.iter();

        // Consume and reset every entry in the vector.
        while iter.next_owned(self).is_some() {}

        // Reset the count.
        self.count.store(0, Ordering::Relaxed);
        self.inflight.store(0, Ordering::Relaxed);
    }
}

impl<T> Index<usize> for Vec<T> {
    type Output = T;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        self.get(index)
            .unwrap_or_else(|| panic!("index `{index}` is uninitialized"))
    }
}

impl<T> Drop for Vec<T> {
    fn drop(&mut self) {
        for (i, bucket) in self.buckets.iter_mut().enumerate() {
            let entries = bucket.entries.read_mut();

            if entries.is_null() {
                break;
            }

            let len = Location::bucket_capacity(i);

            // Safety: We have `&mut self` and verified that this bucket is
            // initialized.
            unsafe { Bucket::dealloc(entries, len) }
        }
    }
}

/// A bucket of entries.
struct Bucket<T> {
    entries: AtomicPtr<Entry<T>>,
}

impl<T> Bucket<T> {
    /// An empty bucket.
    #[cfg(not(loom))]
    const EMPTY: Bucket<T> = Bucket {
        entries: AtomicPtr::new(ptr::null_mut()),
    };
}

/// A possibly uninitialized entry in the vector.
struct Entry<T> {
    /// A flag indicating whether or not this entry is initialized.
    active: AtomicBool,

    /// The entry's value.
    slot: UnsafeCell<MaybeUninit<T>>,
}

impl<T> Bucket<T> {
    /// Create a `Bucket` from the given `Entry` pointer.
    fn from_ptr(entries: *mut Entry<T>) -> Bucket<T> {
        Bucket {
            entries: AtomicPtr::new(entries),
        }
    }

    /// Allocate a bucket of the specified capacity.
    fn alloc(len: usize) -> *mut Entry<T> {
        let entries = (0..len)
            .map(|_| Entry::<T> {
                slot: UnsafeCell::new(MaybeUninit::uninit()),
                active: AtomicBool::new(false),
            })
            .collect::<Box<[Entry<_>]>>();

        Box::into_raw(entries) as _
    }

    /// Allocate a bucket of the specified capacity.
    fn alloc_default(len: usize) -> *mut Entry<T>
    where
        T: Default,
    {
        let entries = (0..len)
            .map(|_| Entry::<T> {
                slot: UnsafeCell::new(MaybeUninit::new(T::default())),
                active: AtomicBool::new(false),
            })
            .collect::<Box<[Entry<_>]>>();

        Box::into_raw(entries) as _
    }

    /// Deallocate a bucket of the specified capacity.
    ///
    /// # Safety
    ///
    /// The safety requirements of `slice::from_raw_parts_mut` and
    /// `Box::from_raw`. The pointer must be a valid, owned pointer
    /// to an array of entries of the provided length.
    unsafe fn dealloc(entries: *mut Entry<T>, len: usize) {
        // Safety: Guaranteed by caller.
        unsafe { drop(Box::from_raw(slice::from_raw_parts_mut(entries, len))) }
    }
}

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
}

impl<T> Drop for Entry<T> {
    fn drop(&mut self) {
        if self.active.read_mut() {
            // Safety: We have `&mut self` and verifid that the value is initialized.
            unsafe { ptr::drop_in_place(self.slot.with_mut(|slot| (*slot).as_mut_ptr())) }
        }
    }
}

/// The location of an entry in the bucket array.
#[derive(Debug, Clone)]
struct Location {
    /// The index of the bucket.
    bucket: usize,

    /// The length of the bucket at the above index.
    bucket_len: usize,

    /// The index of the entry in the bucket.
    entry: usize,
}

/// The number of entries that are skipped from the start of a vector.
///
/// Index calculations assume that buckets are of sizes `[2^0, 2^1, ..., 2^63]`.
/// To skip shorter buckets and avoid unnecessary location, the zeroeth entry
/// index is remapped to a larger index (`2^0 + ... + 2^4 = 31`).
const ZERO_ENTRY: usize = 31;

/// The number of buckets that are skipped from the start of a vector.
///
/// This is the index that the zeroeth bucket index is remapped to (`5`).
const ZERO_BUCKET: usize = (usize::BITS - ZERO_ENTRY.leading_zeros()) as usize;

/// The number of buckets in a vector.
const BUCKETS: usize = (usize::BITS as usize) - ZERO_BUCKET;

/// The maximum index of an element in the vector.
///
/// Note that capacity of the vector is:
/// `2^ZERO_BUCKET + ... + 2^63 = usize::MAX - ZERO_INDEX`.
const MAX_INDEX: usize = usize::MAX - ZERO_ENTRY - 1;

impl Location {
    /// Returns the location of a given entry in a vector.
    ///
    /// This function will panic if the entry is greater than `MAX_INDEX`.
    #[inline]
    fn of(index: usize) -> Location {
        if index > MAX_INDEX {
            panic!("index out of bounds");
        }

        Location::of_raw(index + ZERO_ENTRY)
    }

    /// Returns the location of a given entry in a vector, without bounds checks.
    ///
    /// # Safety
    ///
    /// The index must be in-bounds.
    #[inline]
    unsafe fn of_unchecked(index: usize) -> Location {
        // Note: This can lead to unsoundness if it wraps.
        Location::of_raw(index + ZERO_ENTRY)
    }

    /// Returns the location of the entry at index `index - ZERO_INDEX` in a vector.
    #[inline]
    fn of_raw(index: usize) -> Location {
        // Calculate the bucket index based on ⌊log2(index)⌋.
        let bucket = BUCKETS - ((index + 1).leading_zeros() as usize) - 1;
        let bucket_len = Location::bucket_capacity(bucket);

        // Offset the absolute index by the capacity of the preceding buckets.
        let entry = index - (bucket_len - 1);

        Location {
            bucket,
            bucket_len,
            entry,
        }
    }

    /// Returns the capacity of the bucket at the given index.
    #[inline]
    fn bucket_capacity(bucket: usize) -> usize {
        1 << (bucket + ZERO_BUCKET)
    }
}

/// An iterator over the elements of a [`Vec<T>`].
#[derive(Clone)]
pub struct Iter {
    location: Location,
    yielded: usize,
    index: usize,
}

impl Iter {
    /// Returns a pointer to the next entry in the iterator.
    ///
    /// Any returned entries are guaranteed to be initialized.
    #[inline]
    fn next<T>(&mut self, vec: &Vec<T>) -> Option<(usize, *mut Entry<T>)> {
        // We returned every entry in the vector, we're done.
        if self.yielded == vec.count() {
            return None;
        }

        // It is possible that the the length was incremented due to an element
        // being stored in a bucket that we have already iterated over, so we
        // still have to check that we are in bounds.
        while self.location.bucket < BUCKETS {
            // Safety: Bounds checked above.
            let entries = unsafe {
                vec.buckets
                    .get_unchecked(self.location.bucket)
                    .entries
                    .load(Ordering::Acquire)
            };

            // Despite this bucket not being initialized, it is possible, but rare,
            // that a subsequent bucket was initialized before this one. Thus we
            // have to continue checking every bucket until we yield `vec.count()`
            // elements.
            if !entries.is_null() {
                while self.location.entry < self.location.bucket_len {
                    // Safety: Bounds checked above.
                    let entry = unsafe { entries.add(self.location.entry) };
                    let index = self.index;

                    self.location.entry += 1;
                    self.index += 1;

                    // Continue even after we find an uninitialized entry for the same
                    // reason as uninitialized buckets.
                    //
                    // Note that the `Acquire` ordering ensures that the initialization
                    // of the value happens-before this read if we yield the entry.
                    if unsafe { (*entry).active.load(Ordering::Acquire) } {
                        self.yielded += 1;
                        return Some((index, entry));
                    }
                }
            }

            // We iterated over an entire bucket, move on to the next one.
            self.location.entry = 0;
            self.location.bucket += 1;
            if self.location.bucket < BUCKETS {
                self.location.bucket_len = Location::bucket_capacity(self.location.bucket);
            }
        }

        None
    }

    /// Returns a shared reference to the next entry in the iterator.
    #[inline]
    pub fn next_shared<'v, T>(&mut self, vec: &'v Vec<T>) -> Option<(usize, &'v T)> {
        self.next(vec)
            // Safety: `Iter::next` guarantees that the entry is initialized.
            .map(|(index, entry)| (index, unsafe { (*entry).value_unchecked() }))
    }

    /// Returns an owned reference to the next entry in the iterator, resetting the entry in
    /// the vector.
    #[inline]
    pub fn next_owned<T>(&mut self, vec: &mut Vec<T>) -> Option<T> {
        self.next(vec).map(|(_, entry)| {
            // Safety: `Iter::next` guarantees that the entry is initialized, and we have `&mut Vec<T>`.
            let entry = unsafe { &mut *entry };

            // Mark the entry as uninitialized so it is not accessed after this.
            entry.active.write_mut(false);

            // Safety: `Iter::next` only yields initialized entries.
            unsafe {
                let slot = entry.slot.with_mut(|slot| &mut *slot);
                mem::replace(slot, MaybeUninit::uninit()).assume_init()
            }
        })
    }

    /// Returns the number of elements that have been yielded by this iterator.
    #[inline]
    pub fn yielded(&self) -> usize {
        self.yielded
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate std;

    #[test]
    fn location() {
        assert_eq!(Location::bucket_capacity(0), 32);
        for i in 0..32 {
            let loc = Location::of(i);
            assert_eq!(loc.bucket_len, 32);
            assert_eq!(loc.bucket, 0);
            assert_eq!(loc.entry, i);
        }

        assert_eq!(Location::bucket_capacity(1), 64);
        for i in 33..96 {
            let loc = Location::of(i);
            assert_eq!(loc.bucket_len, 64);
            assert_eq!(loc.bucket, 1);
            assert_eq!(loc.entry, i - 32);
        }

        assert_eq!(Location::bucket_capacity(2), 128);
        for i in 96..224 {
            let loc = Location::of(i);
            assert_eq!(loc.bucket_len, 128);
            assert_eq!(loc.bucket, 2);
            assert_eq!(loc.entry, i - 96);
        }
    }

    #[test]
    fn max_entries() {
        let mut entries = 0;
        for i in 0..BUCKETS {
            entries += Location::bucket_capacity(i);
        }
        assert_eq!(entries, MAX_INDEX + 1);

        let max = Location::of(MAX_INDEX);
        assert_eq!(max.bucket, BUCKETS - 1);
        assert_eq!(max.bucket_len, 1 << (usize::BITS - 1));
        assert_eq!(max.entry, (1 << (usize::BITS - 1)) - 1);

        let panic = std::panic::catch_unwind(|| Location::of(MAX_INDEX + 1)).unwrap_err();
        let panic = *panic.downcast_ref::<&'static str>().unwrap();
        assert_eq!(panic, "index out of bounds");
    }
}
