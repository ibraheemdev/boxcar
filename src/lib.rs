#![deny(unsafe_op_in_unsafe_fn)]

use std::cell::UnsafeCell;
use std::mem::{self, MaybeUninit};
use std::ops::Index;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicUsize, Ordering};
use std::{fmt, iter, ptr, slice};

const BUCKETS: usize = (usize::BITS + 1) as _;
const MAX_ENTRIES: usize = usize::MAX;

/// Creates a [`Vec`] containing the arguments.
///
/// `vec!` allows `Vec`s to be defined with the same syntax as array expressions.
/// There are two forms of this macro:
///
/// - Create a [`Vec`] containing a given list of elements:
///
/// ```
/// let v = vec![1, 2, 3];
/// assert_eq!(v[0], 1);
/// assert_eq!(v[1], 2);
/// assert_eq!(v[2], 3);
/// ```
///
/// - Create a [`Vec`] from a given element and size:
///
/// ```
/// let v = vec![1; 3];
/// assert_eq!(v, [1, 1, 1]);
/// ```
#[macro_export]
macro_rules! vec {
    () => {
        $crate::Vec::new()
    };
    ($elem:expr; $n:expr) => {{
        let vec = $crate::Vec::with_capacity($n);
        vec.extend(::core::iter::repeat($elem).take($n));
        vec
    }};
    ($($x:expr),+ $(,)?) => (
        <$crate::Vec<_> as core::iter::FromIterator<_>>::from_iter([$($x),+])
    );
}

/// A lock-free, append-only vector.
///
/// See [the crate documentation](crate) for details.
pub struct Vec<T> {
    // buckets of length 1, 1, 2, 4, 8 .. 2^63
    buckets: Box<[Bucket<T>; BUCKETS]>,
    // the number of elements in this vector
    len: AtomicUsize,
    // a counter used to retrieve a unique index
    // to push to. this value may be more than
    // the true length as it will be incremented
    // before values are actually stored
    inflight: AtomicUsize,
}

unsafe impl<T: Send> Send for Vec<T> {}
unsafe impl<T: Send> Sync for Vec<T> {}

impl<T> Default for Vec<T> {
    fn default() -> Vec<T> {
        Vec::new()
    }
}

impl<T> Vec<T> {
    /// Constructs a new, empty `Vec<T>`.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![allow(unused_mut)]
    /// let mut vec: boxcar::Vec<i32> = boxcar::Vec::new();
    /// ```
    pub fn new() -> Vec<T> {
        Vec::with_capacity(0)
    }

    /// Constructs a new, empty `Vec<T>` with the specified capacity.
    ///
    /// The vector will be able to hold at least `capacity` elements
    /// without reallocating.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut vec = boxcar::Vec::with_capacity(10);
    ///
    /// for i in 0..10 {
    ///     // will not allocate
    ///     vec.push(i);
    /// }
    ///
    /// // may allocate
    /// vec.push(11);
    /// ```
    pub fn with_capacity(capacity: usize) -> Vec<T> {
        let init = match capacity {
            0 => 0,
            // intialize enough buckets for `capacity` elements
            n => Location::of(n).bucket,
        };

        let mut buckets = iter::repeat_with(|| Bucket::from_raw(ptr::null_mut()))
            .take(BUCKETS)
            .collect::<Box<[_]>>();

        let mut bucket_len = 1_usize;

        for (i, bucket) in buckets[..=init].iter_mut().enumerate() {
            *bucket = Bucket::from_raw(Bucket::alloc(bucket_len));
            bucket_len = Location::bucket_len(i);
        }

        Vec {
            buckets: buckets.try_into().unwrap_or_else(|_| unreachable!()),
            inflight: AtomicUsize::new(0),
            len: AtomicUsize::new(0),
        }
    }

    /// Reserves capacity for at least `additional` more elements to be inserted
    /// in the given `Vec<T>`. The collection may reserve more space to avoid
    /// frequent reallocations.
    ///
    /// Does nothing if capacity is already sufficient.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut vec = boxcar::Vec::new();
    /// vec.reserve(10);
    ///
    /// for i in 0..10 {
    ///     // will not allocate
    ///     vec.push(i);
    /// }
    ///
    /// // may allocate
    /// vec.push(11);
    /// ```
    pub fn reserve(&self, additional: usize) {
        let len = self.len.load(Ordering::Acquire);
        let location = Location::of(len.checked_add(additional).unwrap_or(MAX_ENTRIES));

        let mut bucket_index = location.bucket;
        let mut bucket_len = location.bucket_len;

        // allocate buckets starting from the bucket
        // at `len + additional` and working our way
        // backwards
        loop {
            // SAFETY: we have enough buckets for `usize::MAX` entries
            let bucket = unsafe { self.buckets.get_unchecked(bucket_index) };

            // reached an initalized bucket, we're done
            if !bucket.entries.load(Ordering::Acquire).is_null() {
                break;
            }

            let alloc = Bucket::alloc(bucket_len);

            if bucket
                .entries
                .compare_exchange(ptr::null_mut(), alloc, Ordering::AcqRel, Ordering::Relaxed)
                .is_err()
            {
                // SAFETY: someone else stored before us,
                // so we have unique access to our allocation
                unsafe {
                    let _ = Box::from_raw(alloc);
                }
            }

            if bucket_index == 0 {
                break;
            }

            bucket_index -= 1;
            bucket_len = Location::bucket_len(bucket_index);
        }
    }

    /// Appends an element to the back of the vector.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut vec = boxcar::vec![1, 2];
    /// vec.push(3);
    /// assert_eq!(vec, [1, 2, 3]);
    /// ```
    pub fn push(&self, value: T) -> usize {
        let index = self.inflight.fetch_add(1, Ordering::Relaxed);
        let location = Location::of(index);

        // SAFETY: we have enough buckets for usize::MAX entries.
        // technically `inflight` could overflow, but that would
        // require pushing `usize::MAX + 1` times
        let bucket = unsafe { self.buckets.get_unchecked(location.bucket) };
        let mut entries = bucket.entries.load(Ordering::Acquire);

        if entries.is_null() {
            let alloc = Bucket::alloc(location.bucket_len);

            match bucket.entries.compare_exchange(
                ptr::null_mut(),
                alloc,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => entries = alloc,
                // we lost the race.
                Err(found) => {
                    // SAFETY: someone else stored before us,
                    // so we have unique access to our allocation
                    unsafe {
                        let _ = Box::from_raw(alloc);
                    }

                    // use the bucket allocated by
                    // the other thread
                    entries = found;
                }
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
            // `initialized == false`, and will not try to access it
            entry.value.get().write(MaybeUninit::new(value));

            // let other threads know that this entry
            // has been initialized
            entry.initialized.store(true, Ordering::Release);
        }

        self.len.fetch_add(1, Ordering::Release);

        location.index
    }

    /// Returns the number of elements in the vector.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut a = boxcar::Vec::new();
    /// assert_eq!(a.len(), 0);
    /// a.push(1);
    /// a.push(2);
    /// assert_eq!(a.len(), 2);
    /// ```
    #[inline]
    pub fn len(&self) -> usize {
        self.len.load(Ordering::Acquire)
    }

    /// Returns `true` if the vector contains no elements.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut v = boxcar::Vec::new();
    /// assert!(v.is_empty());
    ///
    /// v.push(1);
    /// assert!(!v.is_empty());
    /// ```
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the first element of the slice, or `None` if it is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// let v = boxcar::vec![10, 40, 30];
    /// assert_eq!(Some(&10), v.first());
    ///
    /// let w: boxcar::Vec<i32> = boxcar::Vec::new();
    /// assert_eq!(None, w.first());
    /// ```
    #[inline]
    pub fn first(&self) -> Option<&T> {
        self.get(0)
    }

    /// Returns the last element of the slice, or `None` if it is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// let v = boxcar::vec![10, 40, 30];
    /// assert_eq!(Some(&30), v.last());
    ///
    /// let w: boxcar::Vec<i32> = boxcar::Vec::new();
    /// assert_eq!(None, w.last());
    /// ```
    #[inline]
    pub fn last(&self) -> Option<&T> {
        self.get(self.len().saturating_sub(1))
    }

    /// Returns a reference to the element at the given index.
    ///
    /// # Examples
    ///
    /// ```
    /// let v = boxcar::vec![10, 40, 30];
    /// assert_eq!(Some(&40), v.get(1));
    /// assert_eq!(None, v.get(3));
    /// ```
    pub fn get(&self, index: usize) -> Option<&T> {
        self._get(index).0.ok()
    }

    fn _get(&self, index: usize) -> (Result<&T, bool>, usize) {
        let len = self.len();

        // out of bounds
        if index >= len {
            return (Err(false), len);
        }

        let location = Location::of(index);

        // SAFETY: we have enough buckets for `usize::MAX` entries
        let entries = unsafe {
            self.buckets
                .get_unchecked(location.bucket)
                .entries
                .load(Ordering::Acquire)
        };

        // SAFETY:
        //
        // - entries must be a valid pointer because we checked that
        // `index < len` above, and `len` is only incremented *after*
        // a value is stored.
        // - `location.entry` is always in bounds for `location.bucket`
        let entry = unsafe { &*entries.add(location.entry) };

        if entry.initialized.load(Ordering::Acquire) {
            // SAFETY: the entry is initialized
            unsafe { return (Ok(entry.value_unchecked()), len) }
        }

        (Err(true), len)
    }

    /// Returns an iterator over the slice.
    ///
    /// # Examples
    ///
    /// ```
    /// let x = boxcar::vec![1, 2, 4];
    /// let mut iterator = x.iter();
    ///
    /// assert_eq!(iterator.next(), Some(&1));
    /// assert_eq!(iterator.next(), Some(&2));
    /// assert_eq!(iterator.next(), Some(&4));
    /// assert_eq!(iterator.next(), None);
    /// ```
    pub fn iter(&self) -> Iter<'_, T> {
        Iter {
            vec: self,
            raw: RawIter::new(),
        }
    }
}

impl<T> Index<usize> for Vec<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        let (result, len) = self._get(index);
        match result {
            Ok(value) => value,
            // in bounds
            Err(true) => panic!("attempted to access value during initialization"),
            // out of bounds
            Err(false) => panic!("index out of bounds: the len is {len} but the index is {index}"),
        }
    }
}

impl<T> Drop for Vec<T> {
    fn drop(&mut self) {
        let mut bucket_len = 1;

        for (i, bucket) in self.buckets.iter_mut().enumerate() {
            let entries = *bucket.entries.get_mut();

            if entries.is_null() {
                break;
            }

            // SAFETY: we have &mut self
            unsafe {
                let _ = Box::from_raw(slice::from_raw_parts_mut(entries, bucket_len));
            }

            bucket_len = Location::bucket_len(i);
        }
    }
}

impl<T> IntoIterator for Vec<T> {
    type Item = T;
    type IntoIter = IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        IntoIter {
            vec: self,
            raw: RawIter::new(),
        }
    }
}

impl<'a, T> IntoIterator for &'a Vec<T> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// An iterator that moves out of a vector.
///
/// This struct is created by the `into_iter` method on [`Vec`]
/// (provided by the [`IntoIterator`] trait).
pub struct IntoIter<T> {
    vec: Vec<T>,
    raw: RawIter,
}

impl<T> Iterator for IntoIter<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.raw.next(&self.vec).map(|entry| unsafe {
            entry.initialized.store(false, Ordering::Relaxed);
            // SAFETY: RawIter only yields initialized entries
            mem::replace(&mut *entry.value.get(), MaybeUninit::uninit()).assume_init()
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.vec.len() - self.raw.location.index;
        (len, Some(len))
    }
}

/// An iterator over the elements of a [`Vec<T>`].
///
/// See [`Vec::iter`] for details.
pub struct Iter<'a, T> {
    vec: &'a Vec<T>,
    raw: RawIter,
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        self.raw
            .next(self.vec)
            // SAFETY: RawIter only yields initialized entries
            .map(|entry| unsafe { entry.value_unchecked() })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.vec.len() - self.raw.location.index, None)
    }
}

struct RawIter {
    location: Location,
}

impl RawIter {
    fn new() -> RawIter {
        RawIter {
            location: Location {
                bucket: 0,
                bucket_len: 1,
                entry: 0,
                index: 0,
            },
        }
    }

    fn next<'v, T>(&mut self, vec: &'v Vec<T>) -> Option<&'v Entry<T>> {
        while self.location.bucket < BUCKETS {
            // SAFETY: bounds checked above
            let entries = unsafe {
                vec.buckets
                    .get_unchecked(self.location.bucket)
                    .entries
                    .load(Ordering::Acquire)
            };

            // this bucket has not yet been initialized
            if entries.is_null() {
                return None;
            }

            if self.location.entry < self.location.bucket_len {
                // SAFETY: bounds checked above
                let entry = unsafe { &*entries.add(self.location.entry) };

                if entry.initialized.load(Ordering::Acquire) {
                    self.location.entry += 1;
                    self.location.index += 1;
                    return Some(entry);
                }

                return None;
            }

            self.location.entry = 0;
            self.location.bucket += 1;
            self.location.bucket_len = Location::bucket_len(self.location.bucket);
        }

        None
    }
}

impl<T> FromIterator<T> for Vec<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let iter = iter.into_iter();

        let (lower, _) = iter.size_hint();
        let vec = Vec::with_capacity(lower);

        for value in iter {
            vec.push(value);
        }

        vec
    }
}

impl<T> Extend<T> for Vec<T> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        let iter = iter.into_iter();

        let (lower, _) = iter.size_hint();
        let vec = self.reserve(lower);

        for value in iter {
            self.push(value);
        }

        vec
    }
}

impl<T: Clone> Clone for Vec<T> {
    fn clone(&self) -> Vec<T> {
        self.iter().cloned().collect()
    }
}

impl<T: fmt::Debug> fmt::Debug for Vec<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl<T: PartialEq> PartialEq for Vec<T> {
    fn eq(&self, other: &Self) -> bool {
        if self.len() != other.len() {
            return false;
        }

        self.iter().zip(other).all(|(a, b)| a == b)
    }
}

impl<A, T> PartialEq<A> for Vec<T>
where
    A: AsRef<[T]>,
    T: PartialEq,
{
    fn eq(&self, other: &A) -> bool {
        let other = other.as_ref();

        if self.len() != other.len() {
            return false;
        }

        self.iter().zip(other).all(|(a, b)| a == b)
    }
}

impl<T: Eq> Eq for Vec<T> {}

struct Bucket<T> {
    entries: AtomicPtr<Entry<T>>,
}

struct Entry<T> {
    value: UnsafeCell<MaybeUninit<T>>,
    initialized: AtomicBool,
}

impl<T> Bucket<T> {
    fn alloc(len: usize) -> *mut Entry<T> {
        let entries = (0..len)
            .map(|_| Entry::<T> {
                value: UnsafeCell::new(MaybeUninit::uninit()),
                initialized: AtomicBool::new(false),
            })
            .collect::<Box<[Entry<_>]>>();

        Box::into_raw(entries) as _
    }

    fn from_raw(entries: *mut Entry<T>) -> Bucket<T> {
        Bucket {
            entries: AtomicPtr::new(entries),
        }
    }
}

impl<T> Drop for Entry<T> {
    fn drop(&mut self) {
        if *self.initialized.get_mut() {
            unsafe {
                let _ = ptr::drop_in_place((*self.value.get()).as_mut_ptr());
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
        unsafe { (*self.value.get()).assume_init_ref() }
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
mod test {
    use super::*;

    #[test]
    fn location() {
        let location = Location::of(0);
        assert_eq!(location.index, 0);
        assert_eq!(location.bucket, 0);
        assert_eq!(location.bucket_len, 1);
        assert_eq!(location.entry, 0);

        let location = Location::of(1);
        assert_eq!(location.index, 1);
        assert_eq!(location.bucket, 1);
        assert_eq!(location.bucket_len, 1);
        assert_eq!(location.entry, 0);

        let location = Location::of(2);
        assert_eq!(location.index, 2);
        assert_eq!(location.bucket, 2);
        assert_eq!(location.bucket_len, 2);
        assert_eq!(location.entry, 0);

        let location = Location::of(3);
        assert_eq!(location.index, 3);
        assert_eq!(location.bucket, 2);
        assert_eq!(location.bucket_len, 2);
        assert_eq!(location.entry, 1);

        let location = Location::of(24);
        assert_eq!(location.index, 24);
        assert_eq!(location.bucket, 5);
        assert_eq!(location.bucket_len, 16);
        assert_eq!(location.entry, 8);

        assert_eq!(Location::of(usize::MAX).bucket + 1, BUCKETS);
        assert_eq!(Location::of(usize::MAX).bucket_len, 1 << 63);
        assert_eq!(Location::bucket_len(64), 1 << 63);
    }
}
