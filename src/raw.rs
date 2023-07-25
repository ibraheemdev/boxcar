use std::cell::UnsafeCell;
use std::mem::{self, MaybeUninit};
use std::ops::Index;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicUsize, Ordering};
use std::sync::Mutex;
use std::{ptr, slice};

const BUCKETS: usize = (usize::BITS + 1) as _;
const MAX_ENTRIES: usize = usize::MAX;

// A lock-free, append-only vector.
pub struct Vec<T> {
    // buckets of length 1, 1, 2, 4, 8 .. 2^63
    buckets: [Bucket<T>; BUCKETS],
    // the number of elements in this vector
    count: AtomicUsize,
    // a counter used to retrieve a unique index to push to.
    // this value may be more than the true length as it will
    // be incremented before values are actually stored.
    inflight: AtomicUsize,
}

unsafe impl<T: Send> Send for Vec<T> {}
unsafe impl<T: Sync> Sync for Vec<T> {}

impl<T> Vec<T> {
    // Constructs a new, empty `Vec<T>` with the specified capacity.
    pub fn with_capacity(capacity: usize) -> Vec<T> {
        let init = match capacity {
            0 => 0,
            // initialize enough buckets for `capacity` elements
            n => Location::of(n).bucket,
        };

        let mut buckets = [ptr::null_mut(); BUCKETS];

        for (i, bucket) in buckets[..=init].iter_mut().enumerate() {
            let len = Location::bucket_len(i);
            *bucket = Bucket::alloc(len);
        }

        Vec {
            buckets: buckets.map(Bucket::from_raw),
            inflight: AtomicUsize::new(0),
            count: AtomicUsize::new(0),
        }
    }

    // Reserves capacity for at least `additional` more elements to be inserted
    // in the given `Vec<T>`. The collection may reserve more space to avoid
    // frequent reallocations.
    pub fn reserve(&self, additional: usize) {
        let len = self.count.load(Ordering::Acquire);
        let location = Location::of(len.checked_add(additional).unwrap_or(MAX_ENTRIES));

        let mut bucket_index = location.bucket;
        let mut bucket_len = location.bucket_len;

        // allocate buckets starting from the bucket at `len + additional` and
        // working our way backwards
        loop {
            // SAFETY: we have enough buckets for `usize::MAX` entries
            let bucket = unsafe { self.buckets.get_unchecked(bucket_index) };

            // reached an initalized bucket, we're done
            if !bucket.entries.load(Ordering::Acquire).is_null() {
                break;
            }

            // guard against concurrent allocations
            let _allocating = bucket.lock.lock().unwrap();

            // someone allocated before us
            if !bucket.entries.load(Ordering::Relaxed).is_null() {
                break;
            }

            // otherwise, allocate the bucket
            let new_entries = Bucket::alloc(bucket_len);
            bucket.entries.store(new_entries, Ordering::Release);

            if bucket_index == 0 {
                break;
            }

            bucket_index -= 1;
            bucket_len = Location::bucket_len(bucket_index);
        }
    }

    // Appends an element to the back of the vector.
    pub fn push(&self, value: T) -> usize {
        let index = self.inflight.fetch_add(1, Ordering::Relaxed);
        let location = Location::of(index);

        // SAFETY: we have enough buckets for usize::MAX entries.
        // we assume that `inflight` cannot realistically overflow.
        let bucket = unsafe { self.buckets.get_unchecked(location.bucket) };
        let mut entries = bucket.entries.load(Ordering::Acquire);

        // the bucket has not been allocated yet
        if entries.is_null() {
            // guard against concurrent allocations
            let _allocating = bucket.lock.lock().unwrap();

            let new_entries = bucket.entries.load(Ordering::Acquire);
            if !new_entries.is_null() {
                // someone allocated before us
                entries = new_entries;
            } else {
                // otherwise allocate the bucket
                let alloc = Bucket::alloc(location.bucket_len);
                bucket.entries.store(alloc, Ordering::Release);
                entries = alloc;
            }
        }

        unsafe {
            // SAFETY: `location.entry` is always in bounds for `location.bucket`
            let entry = &*entries.add(location.entry);

            // SAFETY: we have unique access to this entry.
            //
            // 1. it is impossible for another thread to attempt
            // a `push` to this location as we retreived it with
            // a `inflight.fetch_add`.
            //
            // 2. any thread trying to `get` this entry will see
            // `active == false`, and will not try to access it
            entry.slot.get().write(MaybeUninit::new(value));

            // let other threads know that this slot
            // is active
            entry.active.store(true, Ordering::Release);
        }

        self.count.fetch_add(1, Ordering::Release);

        location.index
    }

    // Returns the number of elements in the vector.
    pub fn count(&self) -> usize {
        self.count.load(Ordering::Acquire)
    }

    // Returns a reference to the element at the given index.
    pub fn get(&self, index: usize) -> Option<&T> {
        let location = Location::of(index);

        // SAFETY: we have enough buckets for `usize::MAX` entries
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

        // SAFETY: `location.entry` is always in bounds for `location.bucket`
        let entry = unsafe { &*entries.add(location.entry) };

        if entry.active.load(Ordering::Acquire) {
            // SAFETY: the entry is active
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

        // SAFETY: caller guarantees the index is in bounds and
        // the entry is present.
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

    // Returns an iterator over the vector.
    pub fn iter(&self) -> Iter {
        Iter {
            location: Location {
                bucket: 0,
                bucket_len: 1,
                entry: 0,
                index: 0,
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
            // SAFETY: we have &mut self
            unsafe { drop(Box::from_raw(slice::from_raw_parts_mut(entries, len))) }
        }
    }
}

pub struct Iter {
    location: Location,
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
            // SAFETY: bounds checked above
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
                    // SAFETY: bounds checked above
                    let entry = unsafe { &*entries.add(self.location.entry) };
                    let index = self.location.index;
                    self.location.entry += 1;

                    // we have to continue checking entries even after we find an
                    // uninitialized one for the same reason as uninitialized buckets
                    if entry.active.load(Ordering::Acquire) {
                        self.location.index += 1;
                        return Some((index, entry));
                    }
                }
            }

            self.location.entry = 0;
            self.location.bucket += 1;
            self.location.bucket_len = Location::bucket_len(self.location.bucket);
        }

        None
    }

    pub fn next_shared<'v, T>(&mut self, vec: &'v Vec<T>) -> Option<(usize, &'v T)> {
        self.next(vec)
            .map(|(index, entry)| (index, unsafe { entry.value_unchecked() }))
    }

    pub unsafe fn next_owned<T>(&mut self, vec: &mut Vec<T>) -> Option<T> {
        self.next(vec).map(|(index, entry)| unsafe {
            entry.active.store(false, Ordering::Relaxed);
            // SAFETY: `next` only yields initialized entries
            let value = mem::replace(&mut *entry.slot.get(), MaybeUninit::uninit());
            value.assume_init()
        })
    }

    pub fn yielded(&self) -> usize {
        self.location.index
    }
}

struct Bucket<T> {
    lock: Mutex<()>,
    entries: AtomicPtr<Entry<T>>,
}

struct Entry<T> {
    slot: UnsafeCell<MaybeUninit<T>>,
    active: AtomicBool,
}

impl<T> Bucket<T> {
    fn alloc(len: usize) -> *mut Entry<T> {
        let entries = (0..len)
            .map(|_| Entry::<T> {
                slot: UnsafeCell::new(MaybeUninit::uninit()),
                active: AtomicBool::new(false),
            })
            .collect::<Box<[Entry<_>]>>();

        Box::into_raw(entries) as _
    }

    fn from_raw(entries: *mut Entry<T>) -> Bucket<T> {
        Bucket {
            lock: Mutex::new(()),
            entries: AtomicPtr::new(entries),
        }
    }
}

impl<T> Drop for Entry<T> {
    fn drop(&mut self) {
        if *self.active.get_mut() {
            unsafe {
                let _ = ptr::drop_in_place((*self.slot.get()).as_mut_ptr());
            }
        }
    }
}

impl<T> Entry<T> {
    // # Safety
    //
    // Value must be initialized.
    unsafe fn value_unchecked(&self) -> &T {
        // SAFETY: guaranteed by caller
        unsafe { (*self.slot.get()).assume_init_ref() }
    }
}

#[derive(Debug)]
struct Location {
    // the index of the element in the vector
    index: usize,
    // the index of the bucket
    bucket: usize,
    // the length of `bucket`
    bucket_len: usize,
    // the index of the entry in `bucket`
    entry: usize,
}

impl Location {
    fn of(index: usize) -> Location {
        let bucket = (usize::BITS - index.leading_zeros()) as usize;
        let bucket_len = Location::bucket_len(bucket);
        let entry = if index == 0 { 0 } else { index ^ bucket_len };

        Location {
            index,
            bucket,
            bucket_len,
            entry,
        }
    }

    fn bucket_len(bucket: usize) -> usize {
        1 << bucket.saturating_sub(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn location() {
        let min = Location::of(0);
        assert_eq!(min.bucket, 0);

        let max = Location::of(MAX_ENTRIES);
        assert_eq!(max.bucket, BUCKETS - 1);
    }
}
