#![allow(clippy::declare_interior_mutable_const)]

use core::cell::UnsafeCell;
use core::mem::{self, MaybeUninit};
use core::ops::Index;
use core::sync::atomic::{AtomicBool, AtomicPtr, AtomicUsize, Ordering};
use core::{ptr, slice};

use alloc::boxed::Box;

const BUCKETS: usize = (usize::BITS as usize) - SKIP_BUCKET;
const MAX_ENTRY: usize = usize::MAX - SKIP;

#[cfg(target_has_atomic = "64")]
type Inflight = core::sync::atomic::AtomicU64;

#[cfg(not(target_has_atomic = "64"))]
type Inflight = core::sync::atomic::AtomicUsize;

// A lock-free, append-only vector.
pub struct Vec<T> {
    // A counter used to retrieve a unique index to push to.
    //
    // This value may be more than the true length as it will
    // be incremented before values are actually stored.
    inflight: Inflight,

    // Buckets of length 32, 64 .. 2^63.
    buckets: [Bucket<T>; BUCKETS],

    // The number of initialized elements in this vector.
    count: AtomicUsize,
}

unsafe impl<T: Send> Send for Vec<T> {}
unsafe impl<T: Sync> Sync for Vec<T> {}

impl<T> Vec<T> {
    pub const EMPTY: Vec<T> = Vec {
        inflight: Inflight::new(0),
        buckets: [Bucket::EMPTY; BUCKETS],
        count: AtomicUsize::new(0),
    };

    /// Constructs a new, empty `Vec<T>` with the specified capacity.
    pub fn with_capacity(capacity: usize) -> Vec<T> {
        let init = match capacity {
            0 => 0,
            // Initialize enough buckets for `capacity` elements.
            n => Location::of(n).bucket,
        };

        let mut buckets = [Bucket::EMPTY; BUCKETS];

        for (i, bucket) in buckets[..=init].iter_mut().enumerate() {
            let len = Location::bucket_capacity(i);
            *bucket = Bucket::from_ptr(Bucket::alloc(len));
        }

        Vec {
            buckets,
            inflight: Inflight::new(0),
            count: AtomicUsize::new(0),
        }
    }

    /// Returns the number of elements in the vector.
    pub fn count(&self) -> usize {
        self.count.load(Ordering::Acquire)
    }

    /// Returns a reference to the element at the given index.
    pub fn get(&self, index: usize) -> Option<&T> {
        let location = Location::of(index);

        // Safety: `location.bucket` is always in bounds.
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
    /// Entry at `index` must be initialized.
    pub unsafe fn get_unchecked(&self, index: usize) -> &T {
        // Safety: Caller guarantees the entry is initialized.
        unsafe {
            let location = Location::of_unchecked(index);

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
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        let location = Location::of(index);

        // Safety: `location.bucket` is always in bounds.
        let entries = unsafe {
            self.buckets
                .get_unchecked_mut(location.bucket)
                .entries
                .get_mut()
        };

        // The bucket is uninitialized.
        if entries.is_null() {
            return None;
        }

        // Safety: `location.entry` is always in bounds for its bucket.
        let entry = unsafe { &mut *entries.add(location.entry) };

        if *entry.active.get_mut() {
            // safety: the entry is active
            unsafe { return Some(entry.value_unchecked_mut()) }
        }

        // The entry is uninitialized.
        None
    }

    /// Returns a mutable reference to the element at the given index.
    ///
    /// # Safety
    ///
    /// Entry at `index` must be initialized.
    pub unsafe fn get_unchecked_mut(&mut self, index: usize) -> &mut T {
        // Safety: Caller guarantees the entry is initialized.
        unsafe {
            let location = Location::of_unchecked(index);

            let entry = self
                .buckets
                .get_unchecked_mut(location.bucket)
                .entries
                .get_mut()
                .add(location.entry);

            (*entry).value_unchecked_mut()
        }
    }

    /// Returns a unique index for insertion.
    fn next_index(&self) -> usize {
        #[cfg(target_has_atomic = "64")]
        {
            // The inflight counter is an `AtomicU64`, allowing it to catch capacity overflow.
            let index = self.inflight.fetch_add(1, Ordering::Relaxed);
            assert!(index <= (MAX_ENTRY as u64), "capacity overflow");
            index as usize
        }

        #[cfg(not(target_has_atomic = "64"))]
        {
            // We must use a CAS loop to avoid wrapping on overflow.
            self.inflight
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |index| {
                    assert!(index <= MAX_ENTRY, "capacity overflow");
                    Some(index + 1)
                })
                .unwrap()
        }
    }

    /// Appends an element returned from the closure to the back of the vector
    /// at the index represented by the `usize` passed to closure.
    ///
    /// This allows for use of the would-be index to be utilized within the
    /// element.
    pub fn push_with<F>(&self, f: F) -> usize
    where
        F: FnOnce(usize) -> T,
    {
        let index = self.next_index();
        let value = f(index);

        // Safety: `next_index` is always in-bounds.
        unsafe { self.write(index, value) }
    }

    /// Appends an element to the back of the vector.
    pub fn push(&self, value: T) -> usize {
        // Safety: `next_index` is always in-bounds.
        unsafe { self.write(self.next_index(), value) }
    }

    /// Write an element at the given index.
    ///
    /// # Safety
    ///
    /// The index must be in-bounds.
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
        let mut entries = bucket.entries.load(Ordering::Acquire);

        // The bucket has not been allocated yet.
        if entries.is_null() {
            entries = Vec::get_or_alloc(bucket, location.bucket_len);
        }

        unsafe {
            // Safety: `location.entry` is always in bounds for its bucket.
            let entry = &*entries.add(location.entry);

            // Safety: We have unique access to this entry.
            //
            // 1. It is impossible for another thread to attempt a `push`
            // to this location as we retrieved it from `next_index`.
            //
            // 2. Any thread trying to `get` this entry will see `!active`
            // and will not try to access it.
            entry.slot.get().write(MaybeUninit::new(value));

            // Let other threads know that this entry is active.
            entry.active.store(true, Ordering::Release);
        }

        // Increase the element count.
        self.count.fetch_add(1, Ordering::Release);
        index
    }

    // Race to initialize a bucket.
    //
    // Note that we avoid contention on bucket allocation by having a specified
    // writer eagerly allocate the next bucket.
    fn get_or_alloc(bucket: &Bucket<T>, len: usize) -> *mut Entry<T> {
        let entries = Bucket::alloc(len);

        match bucket.entries.compare_exchange(
            ptr::null_mut(),
            entries,
            Ordering::Release,
            Ordering::Acquire,
        ) {
            Ok(_) => entries,
            Err(found) => unsafe {
                Bucket::dealloc(entries, len);
                found
            },
        }
    }

    // Reserves capacity for at least `additional` more elements to be inserted in
    // the vector. The collection may reserve more space to avoid frequent reallocations.
    pub fn reserve(&self, additional: usize) {
        let len = self.count.load(Ordering::Acquire);
        let mut location = Location::of(len.checked_add(additional).unwrap_or(MAX_ENTRY));

        // Allocate buckets starting from the bucket at `len + additional` and
        // working our way backwards.
        loop {
            // Safety: `location.bucket` is always in bounds.
            let bucket = unsafe { self.buckets.get_unchecked(location.bucket) };

            // Reached an initialized bucket, we're done.
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

    fn index(&self, index: usize) -> &Self::Output {
        self.get(index).expect("no element found at index {index}")
    }
}

impl<T> Drop for Vec<T> {
    fn drop(&mut self) {
        for (i, bucket) in self.buckets.iter_mut().enumerate() {
            let entries = *bucket.entries.get_mut();

            if entries.is_null() {
                break;
            }

            let len = Location::bucket_capacity(i);

            // Safety: We have `&mut self`.
            unsafe { Bucket::dealloc(entries, len) }
        }
    }
}

/// A bucket of entries.
struct Bucket<T> {
    entries: AtomicPtr<Entry<T>>,
}

impl<T> Bucket<T> {
    const EMPTY: Bucket<T> = Bucket::from_ptr(ptr::null_mut());
}

/// A possibly uninitialized entry in the vector.
struct Entry<T> {
    slot: UnsafeCell<MaybeUninit<T>>,
    active: AtomicBool,
}

impl<T> Bucket<T> {
    /// Cast a pointer to an `Entry` to a bucket pointer.
    const fn from_ptr(entries: *mut Entry<T>) -> Bucket<T> {
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

    /// Deallocate a bucket of the specified capacity.
    unsafe fn dealloc(entries: *mut Entry<T>, len: usize) {
        unsafe { drop(Box::from_raw(slice::from_raw_parts_mut(entries, len))) }
    }
}

impl<T> Entry<T> {
    // # Safety
    //
    // The value must be initialized.
    unsafe fn value_unchecked(&self) -> &T {
        // Safety: Guaranteed by caller.
        unsafe { (*self.slot.get()).assume_init_ref() }
    }

    // # Safety
    //
    // The value must be initialized.
    unsafe fn value_unchecked_mut(&mut self) -> &mut T {
        // Safety: Guaranteed by caller.
        unsafe { self.slot.get_mut().assume_init_mut() }
    }
}

impl<T> Drop for Entry<T> {
    fn drop(&mut self) {
        if *self.active.get_mut() {
            unsafe { ptr::drop_in_place((*self.slot.get()).as_mut_ptr()) }
        }
    }
}

#[derive(Debug, Clone)]
struct Location {
    // The index of the bucket.
    bucket: usize,
    // The length of `bucket`.
    bucket_len: usize,
    // The index of the entry in the bucket.
    entry: usize,
}

// Skip shorter buckets to avoid unnecessary allocations.
const SKIP: usize = 32;
const SKIP_BUCKET: usize = ((usize::BITS - SKIP.leading_zeros()) as usize) - 1;

impl Location {
    /// Returns the location of a given entry in a vector.
    fn of(index: usize) -> Location {
        let skipped = index.checked_add(SKIP).expect("index out of bounds");
        Location::of_raw(skipped)
    }

    /// Returns the location of a given entry in a vector, without bounds checks.
    ///
    /// # Safety
    ///
    /// The index must be in-bounds.
    unsafe fn of_unchecked(index: usize) -> Location {
        // Note: This can lead to unsoundness if it wraps.
        Location::of_raw(index + SKIP)
    }

    /// Returns the location of a given entry in a vector, without offsetting.
    fn of_raw(index: usize) -> Location {
        let bucket = usize::BITS - index.leading_zeros();
        let bucket = (bucket as usize) - (SKIP_BUCKET + 1);
        let bucket_len = Location::bucket_capacity(bucket);
        let entry = index ^ bucket_len;

        Location {
            bucket,
            bucket_len,
            entry,
        }
    }

    /// Returns the capacity of the bucket at the given index.
    fn bucket_capacity(bucket: usize) -> usize {
        1 << (bucket + SKIP_BUCKET)
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
                    if unsafe { (*entry).active.load(Ordering::Acquire) } {
                        self.yielded += 1;
                        return Some((index, entry));
                    }
                }
            }

            self.location.entry = 0;
            self.location.bucket += 1;

            if self.location.bucket < BUCKETS {
                self.location.bucket_len = Location::bucket_capacity(self.location.bucket);
            }
        }

        None
    }

    /// Returns a shared reference to the next entry in the iterator.
    pub fn next_shared<'v, T>(&mut self, vec: &'v Vec<T>) -> Option<(usize, &'v T)> {
        self.next(vec)
            .map(|(index, entry)| (index, unsafe { (*entry).value_unchecked() }))
    }

    /// Returns an owned reference to the next entry in the iterator, resetting the entry in
    /// the vector.
    pub fn next_owned<T>(&mut self, vec: &mut Vec<T>) -> Option<T> {
        self.next(vec).map(|(_, entry)| {
            // Safety: We have `&mut Vec<T>`.
            let entry = unsafe { &mut *entry };
            *entry.active.get_mut() = false;

            // Safety: `Iter::next` only yields initialized entries.
            unsafe { mem::replace(&mut *entry.slot.get(), MaybeUninit::uninit()).assume_init() }
        })
    }

    /// Returns the number of elements that have been yielded by this iterator.
    pub fn yielded(&self) -> usize {
        self.yielded
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_eq!(entries, MAX_ENTRY + 1);

        let max = Location::of(MAX_ENTRY);
        assert_eq!(max.bucket, BUCKETS - 1);
        assert_eq!(max.bucket_len, 1 << (usize::BITS - 1));
        assert_eq!(max.entry, (1 << (usize::BITS - 1)) - 1);
    }
}
