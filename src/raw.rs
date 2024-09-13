#![allow(clippy::declare_interior_mutable_const)]

use core::cell::UnsafeCell;
use core::mem::{self, MaybeUninit};
use core::ops::Index;
use core::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, AtomicUsize, Ordering};
use core::{ptr, slice};

use alloc::boxed::Box;

const BUCKETS: usize = (usize::BITS as usize) - SKIP_BUCKET;
const MAX_ENTRIES: usize = usize::MAX - SKIP;

// A lock-free, append-only vector.
pub struct Vec<T> {
    // a counter used to retrieve a unique index to push to.
    //
    // this value may be more than the true length as it will
    // be incremented before values are actually stored.
    inflight: AtomicU64,
    // buckets of length 32, 64 .. 2^63
    buckets: [Bucket<T>; BUCKETS],
    // the number of initialized elements in this vector
    count: AtomicUsize,
}

unsafe impl<T: Send> Send for Vec<T> {}
unsafe impl<T: Sync> Sync for Vec<T> {}

impl<T> Vec<T> {
    pub const EMPTY: Vec<T> = Vec {
        inflight: AtomicU64::new(0),
        buckets: [Bucket::EMPTY; BUCKETS],
        count: AtomicUsize::new(0),
    };

    // Constructs a new, empty `Vec<T>` with the specified capacity.
    pub fn with_capacity(capacity: usize) -> Vec<T> {
        let init = match capacity {
            0 => 0,
            // initialize enough buckets for `capacity` elements
            n => Location::of(n).bucket,
        };

        let mut buckets = [Bucket::EMPTY; BUCKETS];

        for (i, bucket) in buckets[..=init].iter_mut().enumerate() {
            let len = Location::bucket_len(i);
            *bucket = Bucket::from_ptr(Bucket::alloc(len));
        }

        Vec {
            buckets,
            inflight: AtomicU64::new(0),
            count: AtomicUsize::new(0),
        }
    }

    // Returns the number of elements in the vector.
    pub fn count(&self) -> usize {
        self.count.load(Ordering::Acquire)
    }

    // Returns a reference to the element at the given index.
    pub fn get(&self, index: usize) -> Option<&T> {
        let location = Location::of(index);

        // safety: `location.bucket` is always in bounds
        let entries = unsafe {
            self.buckets
                .get_unchecked(location.bucket)
                .entries
                .load(Ordering::Acquire)
        };

        // bucket is uninitialized
        if entries.is_null() {
            return None;
        }

        // safety: `location.entry` is always in bounds for it's bucket
        let entry = unsafe { &*entries.add(location.entry) };

        if entry.active.load(Ordering::Acquire) {
            // safety: the entry is active
            unsafe { return Some(entry.value_unchecked()) }
        }

        // entry is uninitialized
        None
    }

    // Returns a reference to the element at the given index.
    //
    // # Safety
    //
    // Entry at `index` must be initialized.
    pub unsafe fn get_unchecked(&self, index: usize) -> &T {
        let location = Location::of(index);

        // safety: caller guarantees the entry is initialized
        unsafe {
            let entry = self
                .buckets
                .get_unchecked(location.bucket)
                .entries
                .load(Ordering::Acquire)
                .add(location.entry);

            (*entry).value_unchecked()
        }
    }

    // Returns a mutable reference to the element at the given index.
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        let location = Location::of(index);

        // safety: `location.bucket` is always in bounds
        let entries = unsafe {
            self.buckets
                .get_unchecked_mut(location.bucket)
                .entries
                .get_mut()
        };

        // bucket is uninitialized
        if entries.is_null() {
            return None;
        }

        // safety: `location.entry` is always in bounds for it's bucket
        let entry = unsafe { &mut *entries.add(location.entry) };

        if *entry.active.get_mut() {
            // safety: the entry is active
            unsafe { return Some(entry.value_unchecked_mut()) }
        }

        // entry is uninitialized
        None
    }

    // Returns a mutable reference to the element at the given index.
    //
    // # Safety
    //
    // Entry at `index` must be initialized.
    pub unsafe fn get_unchecked_mut(&mut self, index: usize) -> &mut T {
        let location = Location::of(index);

        // safety: caller guarantees the entry is initialized
        unsafe {
            let entry = self
                .buckets
                .get_unchecked_mut(location.bucket)
                .entries
                .get_mut()
                .add(location.entry);

            (*entry).value_unchecked_mut()
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
        let value = f(index - 1);
        self.push_to(index, value)
    }

    // Appends an element to the back of the vector.
    pub fn push(&self, value: T) -> usize {
        self.push_to(self.next_index(), value)
    }

    fn next_index(&self) -> usize {
        let index = self.inflight.fetch_add(1, Ordering::Relaxed);
        // the inflight counter is a `u64` to catch overflows of the vector'scapacity
        let index: usize = index.try_into().expect("overflowed maximum capacity");
        index
    }

    fn push_to(&self, index: usize, value: T) -> usize {
        let location = Location::of(index);
        // eagerly allocate the next bucket if we are close to the end of this one
        if index == (location.bucket_len - (location.bucket_len >> 3)) {
            if let Some(next_bucket) = self.buckets.get(location.bucket + 1) {
                Vec::get_or_alloc(next_bucket, location.bucket_len << 1);
            }
        }

        // safety: `location.bucket` is always in bounds
        let bucket = unsafe { self.buckets.get_unchecked(location.bucket) };
        let mut entries = bucket.entries.load(Ordering::Acquire);

        // the bucket has not been allocated yet
        if entries.is_null() {
            entries = Vec::get_or_alloc(bucket, location.bucket_len);
        }

        unsafe {
            // safety: `location.entry` is always in bounds for it's bucket
            let entry = &*entries.add(location.entry);

            // safety: we have unique access to this entry.
            //
            // 1. it is impossible for another thread to attempt a `push`
            // to this location as we retreived it from `inflight.fetch_add`
            //
            // 2. any thread trying to `get` this entry will see `active == false`,
            // and will not try to access it
            entry.slot.get().write(MaybeUninit::new(value));

            // let other threads know that this entry is active
            entry.active.store(true, Ordering::Release);
        }

        // increase the true count
        self.count.fetch_add(1, Ordering::Release);
        index
    }

    // race to intialize a bucket
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

    // Reserves capacity for at least `additional` more elements to be inserted
    // in the given `Vec<T>`. The collection may reserve more space to avoid
    // frequent reallocations.
    pub fn reserve(&self, additional: usize) {
        let len = self.count.load(Ordering::Acquire);
        let mut location = Location::of(len.checked_add(additional).unwrap_or(MAX_ENTRIES));

        // allocate buckets starting from the bucket at `len + additional` and
        // working our way backwards
        loop {
            // safety: `location.bucket` is always in bounds
            let bucket = unsafe { self.buckets.get_unchecked(location.bucket) };

            // reached an initalized bucket, we're done
            if !bucket.entries.load(Ordering::Acquire).is_null() {
                break;
            }

            // someone allocated before us
            if !bucket.entries.load(Ordering::Relaxed).is_null() {
                break;
            }

            // allocate the bucket
            Vec::get_or_alloc(bucket, location.bucket_len);

            // reached the first bucket
            if location.bucket == 0 {
                break;
            }

            location.bucket -= 1;
            location.bucket_len = Location::bucket_len(location.bucket);
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
                bucket_len: Location::bucket_len(0),
            },
        }
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

            let len = Location::bucket_len(i);
            // safety: in drop
            unsafe { Bucket::dealloc(entries, len) }
        }
    }
}

#[derive(Clone)]
pub struct Iter {
    location: Location,
    yielded: usize,
    index: usize,
}

impl Iter {
    fn next<'v, T>(&mut self, vec: &'v Vec<T>) -> Option<(usize, &'v Entry<T>)> {
        if self.yielded() == vec.count() {
            return None;
        }

        // it is possible that the the length was incremented due to an element
        // being stored in a bucket that we have already iterated over, so we
        // still have to check that we are in bounds
        while self.location.bucket < BUCKETS {
            // safety: bounds checked above
            let entries = unsafe {
                vec.buckets
                    .get_unchecked(self.location.bucket)
                    .entries
                    .load(Ordering::Acquire)
            };

            // just because this bucket is not initialized doesn't mean all
            // subsequent buckets aren't. a push may have acquired an index
            // in a new bucket before a previous push finished storing, so
            // we have to continue checking every bucket until we yield
            // `vec.count()` elements
            if !entries.is_null() {
                while self.location.entry < self.location.bucket_len {
                    // safety: bounds checked above
                    let entry = unsafe { &*entries.add(self.location.entry) };
                    let index = self.index;

                    self.location.entry += 1;
                    self.index += 1;

                    // we have to continue checking entries even after we find an
                    // uninitialized one for the same reason as uninitialized buckets
                    if entry.active.load(Ordering::Acquire) {
                        self.yielded += 1;
                        return Some((index, entry));
                    }
                }
            }

            self.location.entry = 0;
            self.location.bucket += 1;

            if self.location.bucket < BUCKETS {
                self.location.bucket_len = Location::bucket_len(self.location.bucket);
            }
        }

        None
    }

    pub fn next_shared<'v, T>(&mut self, vec: &'v Vec<T>) -> Option<(usize, &'v T)> {
        self.next(vec)
            .map(|(index, entry)| (index, unsafe { entry.value_unchecked() }))
    }

    pub unsafe fn next_owned<T>(&mut self, vec: &mut Vec<T>) -> Option<T> {
        self.next(vec).map(|(_, entry)| unsafe {
            entry.active.store(false, Ordering::Relaxed);
            // safety: `next` only yields initialized entries
            let value = mem::replace(&mut *entry.slot.get(), MaybeUninit::uninit());
            value.assume_init()
        })
    }

    pub fn yielded(&self) -> usize {
        self.yielded
    }
}

struct Bucket<T> {
    entries: AtomicPtr<Entry<T>>,
}

impl<T> Bucket<T> {
    const EMPTY: Bucket<T> = Bucket::from_ptr(ptr::null_mut());
}

struct Entry<T> {
    slot: UnsafeCell<MaybeUninit<T>>,
    active: AtomicBool,
}

impl<T> Bucket<T> {
    const fn from_ptr(entries: *mut Entry<T>) -> Bucket<T> {
        Bucket {
            entries: AtomicPtr::new(entries),
        }
    }

    fn alloc(len: usize) -> *mut Entry<T> {
        let entries = (0..len)
            .map(|_| Entry::<T> {
                slot: UnsafeCell::new(MaybeUninit::uninit()),
                active: AtomicBool::new(false),
            })
            .collect::<Box<[Entry<_>]>>();

        Box::into_raw(entries) as _
    }

    unsafe fn dealloc(entries: *mut Entry<T>, len: usize) {
        unsafe { drop(Box::from_raw(slice::from_raw_parts_mut(entries, len))) }
    }
}

impl<T> Entry<T> {
    // # Safety
    //
    // Value must be initialized.
    unsafe fn value_unchecked(&self) -> &T {
        // safety: guaranteed by caller
        unsafe { (*self.slot.get()).assume_init_ref() }
    }

    // # Safety
    //
    // Value must be initialized.
    unsafe fn value_unchecked_mut(&mut self) -> &mut T {
        // safety: guaranteed by caller
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
    // the index of the bucket
    bucket: usize,
    // the length of `bucket`
    bucket_len: usize,
    // the index of the entry in `bucket`
    entry: usize,
}

// skip the shorter buckets to avoid unnecessary allocations.
// this also reduces the maximum capacity of a vector.
const SKIP: usize = 32;
const SKIP_BUCKET: usize = ((usize::BITS - SKIP.leading_zeros()) as usize) - 1;

impl Location {
    fn of(index: usize) -> Location {
        let skipped = index.checked_add(SKIP).expect("exceeded maximum length");
        let bucket = usize::BITS - skipped.leading_zeros();
        let bucket = (bucket as usize) - (SKIP_BUCKET + 1);
        let bucket_len = Location::bucket_len(bucket);
        let entry = skipped ^ bucket_len;

        Location {
            bucket,
            bucket_len,
            entry,
        }
    }

    fn bucket_len(bucket: usize) -> usize {
        1 << (bucket + SKIP_BUCKET)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn location() {
        assert_eq!(Location::bucket_len(0), 32);
        for i in 0..32 {
            let loc = Location::of(i);
            assert_eq!(loc.bucket_len, 32);
            assert_eq!(loc.bucket, 0);
            assert_eq!(loc.entry, i);
        }

        assert_eq!(Location::bucket_len(1), 64);
        for i in 33..96 {
            let loc = Location::of(i);
            assert_eq!(loc.bucket_len, 64);
            assert_eq!(loc.bucket, 1);
            assert_eq!(loc.entry, i - 32);
        }

        assert_eq!(Location::bucket_len(2), 128);
        for i in 96..224 {
            let loc = Location::of(i);
            assert_eq!(loc.bucket_len, 128);
            assert_eq!(loc.bucket, 2);
            assert_eq!(loc.entry, i - 96);
        }

        let max = Location::of(MAX_ENTRIES);
        assert_eq!(max.bucket, BUCKETS - 1);
        assert_eq!(max.bucket_len, 1 << 63);
        assert_eq!(max.entry, (1 << 63) - 1);
    }
}
