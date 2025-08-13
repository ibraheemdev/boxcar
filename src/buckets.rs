//! The low-level primitive behind [`Vec`](crate::Vec): a lazily-initialized array implemented as a
//! sequence of buckets with sizes of increasing powers of two.

use ::alloc::alloc::{self, alloc_zeroed, handle_alloc_error};
use ::alloc::boxed::Box;
use ::alloc::vec::{self, Vec};
use core::cmp;
use core::fmt::{self, Debug, Formatter};
use core::hint::unreachable_unchecked;
use core::iter::FusedIterator;
use core::mem::size_of;
use core::num::NonZeroUsize;
use core::panic::{RefUnwindSafe, UnwindSafe};
use core::ptr;
use core::slice;
use core::sync::atomic;

use crate::loom::atomic::AtomicPtr;
use crate::loom::AtomicMut as _;

/// The low-level primitive behind [`Vec`](crate::Vec): a lazily-initialized array implemented as a
/// sequence of buckets with sizes of increasing powers of two.
///
/// The `BUCKETS` generic parameter controls the maximum capacity, and the inline size, of the
/// type. See [`buckets_for_index_bits`] for a convenient way to calculate its desired value.
pub struct Buckets<T, const BUCKETS: usize> {
    buckets: [AtomicPtr<T>; BUCKETS],
}

// Safety:
// - `T: Send` is required since we drop our `T`s when the type is dropped.
// - `T: Sync` is not required since we own our `T`s wholly.
unsafe impl<T: Send, const BUCKETS: usize> Send for Buckets<T, BUCKETS> {}

// Safety:
// - `T: Send` is not required because one can neither put values into the `Buckets` nor take values
//   out of the `Buckets` from only a shared reference. One can force new values to be added to the
//   `Buckets`, however since those values are zeroed it doesn't matter which thread the zeroing
//   happened on.
// - `T: Sync` is required because we provide shared access to the data in the buckets from a
//   shared reference.
unsafe impl<T: Sync, const BUCKETS: usize> Sync for Buckets<T, BUCKETS> {}

// Since we act like we own a `T`, we inherit its unwind-safety-ness.
impl<T: UnwindSafe, const BUCKETS: usize> UnwindSafe for Buckets<T, BUCKETS> {}
impl<T: RefUnwindSafe, const BUCKETS: usize> RefUnwindSafe for Buckets<T, BUCKETS> {}

impl<T, const BUCKETS: usize> Buckets<T, BUCKETS> {
    #[cfg(not(loom))]
    #[allow(clippy::declare_interior_mutable_const)]
    const NULL_PTR: AtomicPtr<T> = AtomicPtr::new(ptr::null_mut());

    /// Construct a new, empty, `Buckets`.
    #[cfg(not(loom))]
    pub const fn new() -> Self {
        Self {
            buckets: [Self::NULL_PTR; BUCKETS],
        }
    }

    #[cfg(loom)]
    pub fn new() -> Self {
        Self {
            buckets: [(); BUCKETS].map(|_| AtomicPtr::new(ptr::null_mut())),
        }
    }

    /// Get the bucket at the given index.
    fn bucket(&self, i: BucketIndex<BUCKETS>) -> &AtomicPtr<T> {
        // Safety: Ensured by the invariant of `BucketIndex`.
        unsafe { self.buckets.get_unchecked(i.0) }
    }

    /// Get a unique reference to the bucket at the given index.
    fn bucket_mut(&mut self, i: BucketIndex<BUCKETS>) -> &mut AtomicPtr<T> {
        // Safety: Ensured by the invariant of `BucketIndex`.
        unsafe { self.buckets.get_unchecked_mut(i.0) }
    }

    /// Take ownership over the bucket at the given index.
    fn take_bucket(&mut self, i: BucketIndex<BUCKETS>) -> Option<Box<[T]>> {
        // Take the pointer, replacing it with null.
        let bucket = self.bucket_mut(i);
        let ptr = bucket.read_mut();
        bucket.write_mut(ptr::null_mut());

        // Ensure the pointer is non-null.
        ptr::NonNull::new(ptr)?;

        // Safety: Guaranteed by our invariants.
        // We know we won't double-free because we just set the pointer to null.
        Some(unsafe { Box::from_raw(ptr::slice_from_raw_parts_mut(ptr, i.len().get())) })
    }

    /// Retrieve the value at the specified index, or `None` if it has not been allocated yet.
    ///
    /// # Examples
    ///
    /// ```
    /// # use boxcar::buckets::{self, Buckets};
    /// let mut buckets = <Buckets<u8, 5>>::new();
    /// let index = buckets::Index::new(20).unwrap();
    ///
    /// assert_eq!(buckets.get(index), None);
    /// buckets.get_or_alloc(index);
    /// assert_eq!(*buckets.get(index).unwrap(), 0);
    /// ```
    pub fn get(&self, index: Index<BUCKETS>) -> Option<&T> {
        let location = index.location();
        let bucket = self.bucket(location.bucket);

        // Acquire is necessary because we access the bucket afterward.
        let ptr = bucket.load(atomic::Ordering::Acquire);
        if ptr.is_null() {
            return None;
        }

        // Safety:
        // - By our invariants, the index is in bounds.
        // - We loaded the bucket pointer with `Acquire`, allowing us to access the allocation.
        Some(unsafe { &*ptr.add(location.entry) })
    }

    /// Retrieve a unique reference to the value at the specified index, or `None` if it has not
    /// been allocated yet.
    ///
    /// # Examples
    ///
    /// ```
    /// # use boxcar::buckets::{self, Buckets};
    /// let mut buckets = <Buckets<u8, 5>>::new();
    /// let index = buckets::Index::new(20).unwrap();
    ///
    /// assert_eq!(buckets.get_mut(index), None);
    /// buckets.get_or_alloc(index);
    /// assert_eq!(*buckets.get_mut(index).unwrap(), 0);
    /// ```
    pub fn get_mut(&mut self, index: Index<BUCKETS>) -> Option<&mut T> {
        let location = index.location();
        let bucket = self.bucket_mut(location.bucket);

        let ptr = bucket.read_mut();
        if ptr.is_null() {
            return None;
        }

        // Safety: By our invariants, the index is in bounds and the pointer is valid.
        Some(unsafe { &mut *ptr.add(location.entry) })
    }

    /// Retrieve the value at the specified index, without performing bounds checking.
    ///
    /// # Safety
    ///
    /// The element must be allocated.
    ///
    /// # Examples
    ///
    /// ```
    /// # use boxcar::buckets::{self, Buckets};
    /// let mut buckets = <Buckets<u8, 5>>::new();
    /// let index = buckets::Index::new(20).unwrap();
    /// buckets.reserve_mut(index);
    /// assert_eq!(*unsafe { buckets.get_unchecked(index) }, 0);
    /// ```
    pub unsafe fn get_unchecked(&self, index: Index<BUCKETS>) -> &T {
        let location = index.location();
        let bucket = self.bucket(location.bucket);

        // We only need `Relaxed`, because the caller guarantees that the bucket is already allocated.
        // In theory, we could get away with an unsynchronized read here, but there's an open question
        // as to whether or not unsynchronized reads race with failing RMWs:
        // https://github.com/rust-lang/unsafe-code-guidelines/issues/355
        let ptr = bucket.load(atomic::Ordering::Relaxed);

        // Safety:
        // - By our invariants, the index is in bounds.
        // - The caller ensures that we can access the allocation.
        unsafe { &*ptr.add(location.entry) }
    }

    /// Retrieve unique reference to the value at the specified index, without performing bounds
    /// checking.
    ///
    /// # Safety
    ///
    /// The element must be allocated.
    ///
    /// # Examples
    ///
    /// ```
    /// # use boxcar::buckets::{self, Buckets};
    /// let mut buckets = <Buckets<u8, 5>>::new();
    /// let index = buckets::Index::new(20).unwrap();
    /// buckets.reserve_mut(index);
    /// assert_eq!(*unsafe { buckets.get_unchecked_mut(index) }, 0);
    /// ```
    pub unsafe fn get_unchecked_mut(&mut self, index: Index<BUCKETS>) -> &mut T {
        let location = index.location();
        let bucket = self.bucket_mut(location.bucket);

        let ptr = bucket.read_mut();

        // Safety:
        // - By our invariants, the index is in bounds.
        // - The caller ensures that we can access the allocation.
        unsafe { &mut *ptr.add(location.entry) }
    }

    /// Retrieve the value at the specified index, or allocate the bucket if it hasn't been
    /// allocated yet.
    ///
    /// # Example
    ///
    /// ```
    /// # use boxcar::buckets::{self, Buckets};
    /// let mut buckets = <Buckets<u8, 5>>::new();
    ///
    /// let index = buckets::Index::new(48).unwrap();
    /// assert_eq!(*buckets.get_or_alloc(index), 0);
    ///
    /// *buckets.get_mut(index).unwrap() += 3;
    /// assert_eq!(*buckets.get_or_alloc(index), 3);
    ///
    /// // Prior indices are not necessary allocated.
    /// assert_eq!(buckets.get(buckets::Index::new(0).unwrap()), None);
    /// ```
    ///
    /// # Panics
    ///
    /// May panic if `index` is too large or allocation fails.
    pub fn get_or_alloc(&self, index: Index<BUCKETS>) -> &T
    where
        T: MaybeZeroable,
    {
        let location = index.location();

        // If we are close to the end of this bucket, we eagerly allocate the next one to reduce
        // later contention.
        if location.entry == (location.bucket_len.get() - (location.bucket_len.get() >> 3)) {
            self.alloc_bucket_after(index);
        }

        let bucket = self.bucket(location.bucket);

        // Acquire is necessary because we access the bucket afterward.
        let mut ptr = bucket.load(atomic::Ordering::Acquire) as *const T;
        if ptr.is_null() {
            // Panics: Let `i` be `index.get()`. To avoid panics, the condition is:
            //
            //   location.bucket_len * size_of::<T>() < isize::MAX + 1
            // ⇔ 2 ^ floor(log2(i + SKIPPED_ENTRIES + 1)) < (isize::MAX + 1) / size_of::<T>()
            // ⇔ i + SKIPPED_ENTRIES + 1 < ((isize::MAX + 1) / size_of::<T>()).next_power_of_two()
            // ⇔ i < ((isize::MAX + 1) / size_of::<T>()).next_power_of_two() - SKIPPED_ENTRIES - 1
            //
            // Since `SKIPPED_ENTRIES` is an implementation detail, the caller can't enforce this.
            // But the formula may be useful anyway.
            ptr = allocate_race_and_get(bucket, location.bucket_len);
        }

        // Safety:
        // - The pointer is non-null.
        // - By our invariants, the index is in bounds.
        // - We loaded the bucket pointer with `Acquire`, allowing us to access the allocation.
        unsafe { &*ptr.add(location.entry) }
    }

    /// Eagerly allocate the bucket after the bucket containing the provided index.
    #[cold]
    #[inline(never)]
    fn alloc_bucket_after(&self, index: Index<BUCKETS>)
    where
        T: MaybeZeroable,
    {
        if let Some(new_index) = index.after_bucket().advance() {
            allocate_race(self.bucket(new_index), new_index.len());
        }
    }

    /// Retrieve a unique reference to the value at the specified index, or allocate the bucket if
    /// it hasn't been allocated yet.
    ///
    /// # Example
    ///
    /// ```
    /// # use boxcar::buckets::{self, Buckets};
    /// let mut buckets = <Buckets<u8, 5>>::new();
    ///
    /// let index = buckets::Index::new(48).unwrap();
    /// assert_eq!(*buckets.get_or_alloc_mut(index), 0);
    ///
    /// *buckets.get_or_alloc_mut(index) += 3;
    /// assert_eq!(*buckets.get(index).unwrap(), 3);
    ///
    /// // Prior indices are not necessary allocated.
    /// assert_eq!(buckets.get(buckets::Index::new(0).unwrap()), None);
    /// ```
    ///
    /// # Panics
    ///
    /// May panic if `index` is too large or allocation fails.
    pub fn get_or_alloc_mut(&mut self, index: Index<BUCKETS>) -> &mut T
    where
        T: MaybeZeroable,
    {
        let location = index.location();
        let bucket = self.bucket_mut(location.bucket);

        let mut ptr = bucket.read_mut();
        if ptr.is_null() {
            // See `get_or_alloc` for the panicking conditions.
            ptr = Box::into_raw(allocate_slice::<T>(location.bucket_len)).cast::<T>();
            bucket.write_mut(ptr);
        }

        // Safety: By our invariants, the index is in bounds and the pointer is valid.
        unsafe { &mut *ptr.add(location.entry) }
    }

    /// Reserve capacity up to and including the provided index.
    ///
    /// After calling this method, [`.get_or_alloc(n)`](Self::get_or_alloc) is guaranteed not to
    /// allocate.
    ///
    /// # Examples
    ///
    /// ```
    /// # use boxcar::buckets::{self, Buckets};
    /// let buckets = <Buckets<u8, 5>>::new();
    /// let index = buckets::Index::new(20).unwrap();
    /// assert_eq!(buckets.get(index), None);
    ///
    /// buckets.reserve(index);
    /// assert_eq!(*buckets.get(index).unwrap(), 0);
    /// ```
    ///
    /// # Panics
    ///
    /// May panic if `index` is too large or allocation fails.
    pub fn reserve(&self, index: Index<BUCKETS>)
    where
        T: MaybeZeroable,
    {
        // Start at the current bucket and work our way backwards.
        let mut cursor = index.after_bucket();
        while let Some(index) = cursor.retreat() {
            let bucket = self.bucket(index);

            // If the bucket is allocated, we're done. It's technically possible that a later
            // bucket got allocated but all the threads racing to allocate the earlier bucket
            // died, or no such threads existed. This case is rare, and quitting early anyway is
            // benign.
            //
            // Since we only check if the bucket exists and don't access its data, `Relaxed` is
            // sufficient.
            if !bucket.load(atomic::Ordering::Relaxed).is_null() {
                break;
            }

            // Otherwise, race to allocate the bucket.
            allocate_race(bucket, index.len());
        }
    }

    /// Reserve capacity up to and including the provided index.
    ///
    /// Unlike [`reserve`](Self::reserve), this method takes a mutable reference to the `Buckets`,
    /// avoiding synchronization.
    ///
    /// # Examples
    ///
    /// ```
    /// # use boxcar::buckets::{self, Buckets};
    /// let mut buckets = <Buckets<u8, 5>>::new();
    /// let index = buckets::Index::new(20).unwrap();
    ///
    /// buckets.reserve_mut(index);
    /// assert_eq!(*buckets.get(index).unwrap(), 0);
    /// ```
    ///
    /// # Panics
    ///
    /// May panic if `index` is too large or allocation fails.
    pub fn reserve_mut(&mut self, index: Index<BUCKETS>)
    where
        T: MaybeZeroable,
    {
        // The same algorithm as in `reserve`.
        let mut cursor = index.after_bucket();
        while let Some(index) = cursor.retreat() {
            let bucket = self.bucket_mut(index);

            if !bucket.read_mut().is_null() {
                break;
            }

            let ptr = Box::into_raw(allocate_slice::<T>(index.len()));
            bucket.write_mut(ptr.cast::<T>());
        }
    }

    /// Truncate the `Buckets`, keeping at least the first `n` elements.
    ///
    /// This method truncates to the smallest capacity that preserves any items before
    /// `n`, but may include subsequent elements due to the bucket layout.
    ///
    /// # Examples
    ///
    /// ```
    /// # use boxcar::buckets::{self, Buckets};
    /// let mut buckets = <Buckets<u8, 16>>::new();
    ///
    /// // We reserve capacity for 1000 elements, so all elements will be present.
    /// buckets.reserve_mut(buckets::Index::new(1000).unwrap());
    /// assert_eq!(*buckets.get(buckets::Index::new(12).unwrap()).unwrap(), 0);
    /// assert_eq!(*buckets.get(buckets::Index::new(1000).unwrap()).unwrap(), 0);
    ///
    /// // If we truncate to a smaller capacity, the later elements will not be present.
    /// buckets.truncate(buckets::Index::new(13).unwrap());
    /// assert_eq!(*buckets.get(buckets::Index::new(12).unwrap()).unwrap(), 0);
    /// assert_eq!(buckets.get(buckets::Index::new(1000).unwrap()), None);
    /// ```
    pub fn truncate(&mut self, n: Index<BUCKETS>) {
        let mut cursor = n.after_lower_buckets();
        while let Some(bucket) = cursor.advance() {
            self.take_bucket(bucket);
        }
    }

    /// Iterate over `(index, &value)` pairs of the `Buckets`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use boxcar::buckets::{self, Buckets};
    /// let mut buckets = <Buckets<u8, 5>>::new();
    /// buckets.reserve_mut(buckets::Index::new(25).unwrap());
    /// assert!(25 < buckets.iter().count());
    /// ```
    pub fn iter(&self) -> Iter<'_, T, BUCKETS> {
        self.into_iter()
    }

    /// Iterate over `(index, &mut value)` pairs of the `Buckets`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use boxcar::buckets::{self, Buckets};
    /// let mut buckets = <Buckets<u8, 5>>::new();
    /// buckets.reserve_mut(buckets::Index::new(25).unwrap());
    /// assert!(25 < buckets.iter_mut().count());
    ///
    /// for (i, (_, val)) in buckets.iter_mut().enumerate() {
    ///     *val = i as u8;
    /// }
    /// assert_eq!(*buckets.get(buckets::Index::new(10).unwrap()).unwrap(), 10);
    /// ```
    pub fn iter_mut(&mut self) -> IterMut<'_, T, BUCKETS> {
        self.into_iter()
    }
}

impl<T, const BUCKETS: usize> Drop for Buckets<T, BUCKETS> {
    fn drop(&mut self) {
        self.truncate(Index::new(0).unwrap());
    }
}

impl<T, const BUCKETS: usize> Default for Buckets<T, BUCKETS> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Debug, const BUCKETS: usize> Debug for Buckets<T, BUCKETS> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_map().entries(self.iter()).finish()
    }
}

/// Types with a [`Default`] implementation that may or may not additionally support safe
/// initialization with all zero bytes.
///
/// The provided implementations of this trait are only for example purposes, as it is
/// expected to be implemented manually for user-defined types.
///
/// # Safety
///
/// If [`zeroable`](Self::zeroable) returns true, the all-zeros bit pattern must be a valid instance
/// of this type.
pub unsafe trait MaybeZeroable: Default {
    /// Returns `true` if the all-zeros bit pattern is a valid instance of the type.
    fn zeroable() -> bool;
}

unsafe impl MaybeZeroable for u8 {
    fn zeroable() -> bool {
        true
    }
}

unsafe impl MaybeZeroable for u16 {
    fn zeroable() -> bool {
        true
    }
}

/// Race to allocate a bucket.
///
/// # Panics
///
/// `len * size_of::<T>()` must not overflow an `isize`.
#[cold]
#[inline(never)]
#[must_use]
fn allocate_race_and_get<T: MaybeZeroable>(bucket: &AtomicPtr<T>, len: NonZeroUsize) -> *const T {
    // Panics: Ensured by caller.
    let ptr = Box::into_raw(allocate_slice::<T>(len));

    match bucket.compare_exchange(
        ptr::null_mut(),
        ptr.cast::<T>(),
        // `Release` is necessary to allow accesses to the allocation on other threads to
        // happen-after the allocation. We load from the pointer with `Acquire` as necessary.
        atomic::Ordering::Release,
        // `Acquire` is necessary to allow the caller to use the returned pointer.
        atomic::Ordering::Acquire,
    ) {
        Ok(_) => ptr.cast::<T>(),
        // If we fail the race, just deallocate our now-useless bucket and continue.
        Err(new_ptr) => {
            drop(unsafe { Box::from_raw(ptr) });
            new_ptr
        }
    }
}

/// Like [`allocate_race_and_get`], but doesn’t return a pointer.
///
/// # Panics
///
/// `len * size_of::<T>()` must not overflow an `isize`.
fn allocate_race<T: MaybeZeroable>(bucket: &AtomicPtr<T>, len: NonZeroUsize) {
    // Panics: Ensured by caller.
    let ptr = Box::into_raw(allocate_slice::<T>(len));

    match bucket.compare_exchange(
        ptr::null_mut(),
        ptr.cast::<T>(),
        // `Release` is necessary to allow accesses to the allocation on other threads to
        // happen-after the allocation. We load from the pointer with `Acquire` as necessary.
        atomic::Ordering::Release,
        // Unlike in `allocate_race_and_get`, we don't need the returned pointer, so just use
        // `Relaxed` here.
        atomic::Ordering::Relaxed,
    ) {
        Ok(_) => {}
        // If we fail the race, just deallocate our now-useless bucket and continue.
        Err(_) => drop(unsafe { Box::from_raw(ptr) }),
    }
}

/// Allocate a zeroed array.
///
/// # Panics
///
/// `len * size_of::<T>()` must not overflow an `isize`.
fn allocate_slice<T: MaybeZeroable>(len: NonZeroUsize) -> Box<[T]> {
    if size_of::<T>() == 0 {
        return Box::new([]);
    }

    if T::zeroable() {
        // Panics: Ensured by caller.
        let layout = alloc::Layout::array::<T>(len.get()).unwrap();

        // Safety: `len` is a `NonZeroUsize`, and we just ensured that `T` is not zero-sized.
        // Therefore, `layout` has a non-zero size.
        let ptr = unsafe { alloc_zeroed(layout) }.cast::<T>();

        if ptr.is_null() {
            handle_alloc_error(layout);
        }

        unsafe { Box::from_raw(ptr::slice_from_raw_parts_mut(ptr, len.get())) }
    } else {
        let mut vec = Vec::new();
        vec.resize_with(len.get(), T::default);
        vec.into_boxed_slice()
    }
}

/// An valid index into [`Buckets`].
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Index<const BUCKETS: usize> {
    /// The original index plus `SKIPPED_ENTRIES` plus one.
    ///
    /// Invariant: `SKIPPED_ENTRIES < inner ≤ ENTRIES_WITH_SKIPPED`.
    inner: NonZeroUsize,
}

/// To avoid small allocations, we skip a number of small buckets at the start.
///
/// Note that some code relies on `SKIPPED_BUCKETS` being at least 1, e.g. `Index::into_raw` and
/// `Index::next_buckets`. Other than that, it can be anything.
const SKIPPED_BUCKETS: usize = 5;
const SKIPPED_ENTRIES: usize = 2_usize.pow(SKIPPED_BUCKETS as u32) - 1;

impl<const BUCKETS: usize> Index<BUCKETS> {
    /// If there were in fact `BUCKETS + SKIPPED_BUCKETS` buckets, this is the number of entries
    /// there would be.
    ///
    /// This is equal to `2 ^ (BUCKETS + SKIPPED_BUCKETS) - 1`, but calculated in a way that avoids
    /// overflow.
    const ENTRIES_WITH_SKIPPED: usize = {
        let mut total = 0;
        let mut i = 0;
        while i < BUCKETS + SKIPPED_BUCKETS {
            total += 2_usize.pow(i as u32);
            i += 1;
        }
        total
    };

    /// The real number of entries is just the above figure with the number of skipped entries
    /// subtracted.
    const ENTRIES: usize = Self::ENTRIES_WITH_SKIPPED - SKIPPED_ENTRIES;

    /// Construct a new `Index`.
    ///
    /// Returns `None` if the index is out of bounds. Note that [`buckets_for_index_bits`] can be used to guarantee
    /// that this function returns `None` for any values greater than `2 ^ bits`, but it may also do so for smaller
    /// values.
    pub const fn new(i: usize) -> Option<Self> {
        if i < Self::ENTRIES {
            // Safety: What we just checked.
            Some(unsafe { Self::new_unchecked(i) })
        } else {
            None
        }
    }

    /// Construct a new `Index` without bounds-checking.
    ///
    /// # Safety
    ///
    /// `Index::new(i)` must return `Some`.
    pub const unsafe fn new_unchecked(i: usize) -> Self {
        // Safety: Ensured by the caller.
        // We use this to elide bounds checks below.
        if i >= Self::ENTRIES {
            unsafe { unreachable_unchecked() };
        }

        // Panics: This addition never overflows:
        //   i < ENTRIES
        // ⇒ i < ENTRIES_WITH_SKIPPED - SKIPPED_ENTRIES ≤ usize::MAX - SKIPPED_ENTRIES
        // ⇒ i + SKIPPED_ENTRIES < usize::MAX
        // ⇒ i + SKIPPED_ENTRIES + 1 ≤ usize::MAX
        // The compiler can tell this, so we just panic.
        let Some(inner) = i.checked_add(SKIPPED_ENTRIES + 1) else {
            unreachable!()
        };

        // Panics: `inner` is always nonzero, since we just added one to it and didn't overflow.
        // Because we use `checked_add`, the compiler can tell this, so we just panic.
        let Some(inner) = NonZeroUsize::new(inner) else {
            unreachable!()
        };

        Self { inner }
    }

    /// Construct a new `Index`, returning the maximum index if it is out-of-bounds.
    pub fn new_saturating(i: usize) -> Self {
        // Panics: cmp::min(i, Self::ENTRIES - 1) ≤ Self::ENTRIES - 1 < Self::ENTRIES
        Self::new(cmp::min(i, Self::ENTRIES - 1)).unwrap()
    }

    /// Get the index passed into [`Index::new`].
    pub const fn get(self) -> usize {
        self.inner.get() - (SKIPPED_ENTRIES + 1)
    }

    /// Convert an `Index` into its raw representation.
    ///
    /// The returned value is guaranteed not to be equal to one. Additional range guarantees are
    /// provided if [`buckets_for_index_bits`] is used.
    ///
    /// Indices are guaranteed to be represented contiguously.
    pub const fn into_raw(self) -> NonZeroUsize {
        // Assert the invariants of the type.
        debug_assert!(SKIPPED_ENTRIES < self.inner.get());
        debug_assert!(self.inner.get() <= Self::ENTRIES_WITH_SKIPPED);

        // Since `1 ≤ SKIPPED_ENTRIES`, we know that `2 ≤ SKIPPED_ENTRIES + 1 ≤ inner`.
        self.inner
    }

    /// Reconstruct an `Index` from its raw representation.
    ///
    /// # Safety
    ///
    /// The given index must be at an in-bounds offset from an index previously returned by
    /// [`into_raw`](Self::into_raw).
    pub const unsafe fn from_raw_unchecked(inner: usize) -> Self {
        debug_assert!(SKIPPED_ENTRIES < inner);
        debug_assert!(inner <= Self::ENTRIES_WITH_SKIPPED);

        // Panics: Ensured by caller.
        let inner = unsafe { NonZeroUsize::new_unchecked(inner) };
        Self { inner }
    }

    /// Reconstruct an `Index` from its raw representation, failing if the index is out of bounds
    /// on the positive side.
    ///
    /// # Safety
    ///
    /// The given index must be at a positive offset from an index previously returned by
    /// [`into_raw`](Self::into_raw).
    pub const unsafe fn from_raw_checked_above(inner: usize) -> Option<Self> {
        if inner <= Self::ENTRIES_WITH_SKIPPED {
            // Safety: The lower bound is ensured by the caller, and we just checked the upper one.
            Some(unsafe { Self::from_raw_unchecked(inner) })
        } else {
            None
        }
    }

    /// Reconstruct an `Index` from its raw representation, failing if the index is out of bounds.
    pub const fn from_raw_checked(inner: usize) -> Option<Self> {
        if SKIPPED_ENTRIES < inner {
            // Safety: We just checked the lower bound.
            unsafe { Self::from_raw_checked_above(inner) }
        } else {
            None
        }
    }

    /// Split apart this index into the underlying bucket index, bucket length, and offset into the
    /// bucket.
    fn location(self) -> Location<BUCKETS> {
        // We need to calculate:
        // - the bucket index `0 ≤ bucket < BUCKETS`, and
        // - the entry index `0 ≤ entry < 2 ^ (bucket + SKIPPED_BUCKETS)`.
        //
        // Let `i` be the index that includes skipped entries, i.e. `self.inner - 1`.
        // Let `b` be the bucket index that includes skipped buckets,
        // i.e. `bucket + SKIPPED_BUCKETS`.
        // Note that the bucket at index-including-skipped-buckets `j` has length `2 ^ j`.
        // The following equivalent equalities must be satisfied:
        //
        //   i = (0..b).map(|j| 2 ^ j).sum() + entry
        // ⇔ i = 2 ^ b - 1 + entry
        // ⇔ i + 1 - entry = 2 ^ b
        // ⇔ b = log2(i + 1 - entry)
        //     = floor(log2(i + 1))
        //
        // To justify the last step, observe that, for a given power of two `x` and real number `y`,
        // `x ≤ y < 2 * x` iff `floor(log2(y)) = log2(x)`. Since we have a power of two
        // `i + 1 - entry`, we can prove this works:
        //
        //   i + 1 - entry ≤ i + 1 < 2 * (i + 1 - entry)
        // ⇔ -entry ≤ 0 < i + 1 - 2 * entry
        // ⇔ 0 ≤ entry and entry < i + 1 - entry = 2 ^ b
        //
        // Which holds by definition of `entry`. We can thus set `b = floor(log2(i + 1))
        //
        //= floor(log2(self.inner)) = self.inner.ilog2()`.
        let b = self.inner.ilog2();

        // The compiler can tell that this conversion never fails, so we just unwrap.
        let b_usize = usize::try_from(b).unwrap();

        // The bucket length is `2 ^ b` a.k.a. `1 << b`.
        //
        // Since `b` is the result of an `ilog2`, we know this can never overflow. The compiler
        // knows this, so we just unwrap. Since the operation doesn't overflow and starts with `1`,
        // we also know the result is nonzero. The compiler also knows this, so we just unwrap.
        let bucket_len = NonZeroUsize::new(1_usize.checked_shl(b).unwrap()).unwrap();

        // Recall that: i + 1 - entry = 2 ^ b
        // Therefore: entry = i + 1 - 2 ^ b = self.inner - bucket_len
        //
        // We now prove that `0 ≤ entry < 2 ^ b`:
        //
        //   0 ≤ entry < 2 ^ b
        // ⇔ 2 ^ b ≤ i + 1 < 2 ^ (b + 1)
        // ⇔ 2 ^ floor(log2(i + 1)) ≤ i + 1 < 2 ^ (floor(log2(i + 1)) + 1)
        //
        // Which is obviously true: `i + 1` is contained between the powers of two beneath and
        // above it.
        let entry = self.inner.get() - bucket_len.get();

        // By definition, `b = bucket + SKIPPED_BUCKETS`, so `bucket = b - SKIPPED_BUCKETS`.
        //
        // We prove that `0 ≤ bucket < BUCKETS`:
        //
        //   0 ≤ bucket < BUCKETS
        // ⇔ SKIPPED_BUCKETS ≤ b < BUCKETS + SKIPPED_BUCKETS
        // ⇔ SKIPPED_BUCKETS ≤ floor(log2(i + 1)) < BUCKETS + SKIPPED_BUCKETS
        // ⇔ SKIPPED_ENTRIES + 1 ≤ 2 ^ floor(log2(i + 1)) < ENTRIES_WITH_SKIPPED + 1
        //
        // And this holds by `inner`'s invariant: since the value `i + 1` itself is between those
        // two powers of two, the power of two beneath it will also be contained in the range.
        let bucket = BucketIndex(b_usize - SKIPPED_BUCKETS);

        Location {
            bucket,
            bucket_len,
            entry,
        }
    }

    /// Returns `true` if this index is the first in its bucket.
    pub fn is_first_in_bucket(self) -> bool {
        self.location().entry == 0
    }
}

impl<const BUCKETS: usize> Debug for Index<BUCKETS> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        <usize as Debug>::fmt(&self.into_raw().get(), f)
    }
}

/// The location of a value in a `Buckets`.
#[derive(Clone)]
struct Location<const BUCKETS: usize> {
    /// Which bucket the value is in.
    bucket: BucketIndex<BUCKETS>,
    /// How many entries are in this bucket.
    bucket_len: NonZeroUsize,
    /// The index of the entry within the bucket's array. Less than `bucket_len`.
    entry: usize,
}

/// The index of a bucket.
///
/// Invariant: The inner index is less than `BUCKETS`.
#[derive(Clone, Copy)]
struct BucketIndex<const BUCKETS: usize>(usize);

impl<const BUCKETS: usize> BucketIndex<BUCKETS> {
    /// Get the index of the first entry in this bucket.
    fn first(self) -> Index<BUCKETS> {
        // The desired index `inner - 1` satisfies these equalities:
        //
        //   inner - 1 = (0..bucket + SKIPPED_BUCKETS).map(|j| 2 ^ j).sum() + 0
        //             = 2 ^ (bucket + SKIPPED_BUCKETS) - 1
        // ⇔ inner = 2 ^ (bucket + SKIPPED_BUCKETS)

        // Safety: This is the invariant of the type.
        if self.0 >= BUCKETS {
            unsafe { unreachable_unchecked() };
        }

        // All of these operations can be proven by the compiler not to overflow because of the
        // above assert; hence, we just unwrap freely knowing that it'll be eliminated.
        let b = self.0.checked_add(SKIPPED_BUCKETS).unwrap();
        let inner = 1_usize.checked_shl(b.try_into().unwrap()).unwrap();
        Index::from_raw_checked(inner).unwrap()
    }

    /// Get the length of the bucket.
    fn len(self) -> NonZeroUsize {
        // self.first() computes `2 ^ (bucket + SKIPPED_BUCKETS)`, which is also the bucket length.
        self.first().into_raw()
    }
}

/// Calculate the largest value of the `BUCKETS` generic parameter such that every element is given
/// an index less than `2 ^ bits` (though not every index less than `2 ^ bits` may be given an
/// element).
///
/// This ensures both that [`Index::new`] will reject all values greater than `2 ^ bits` and that
/// [`Index::into_raw`] will return values less than `2 ^ bits`.
///
/// # Panics
///
/// `bits` may not exceed `usize::BITS`.
pub const fn buckets_for_index_bits(bits: u32) -> usize {
    assert!(bits <= usize::BITS);

    // We want the largest value of `BUCKETS` such that `<Index<BUCKETS>>::ENTRIES ≤ 2 ^ bits`:
    //
    //   <Index<BUCKETS>>::ENTRIES ≤ 2 ^ bits
    // ⇔ 2 ^ (BUCKETS + SKIPPED_BUCKETS) - 2 ^ SKIPPED_BUCKETS ≤ 2 ^ bits
    // ⇔ BUCKETS + SKIPPED_BUCKETS ≤ log2(2 ^ bits + 2 ^ SKIPPED_BUCKETS)
    // ⇔ BUCKETS ≤ log2(2 ^ bits + 2 ^ SKIPPED_BUCKETS) - SKIPPED_BUCKETS
    //           ≤ ceil(log2(2 ^ bits + 2 ^ SKIPPED_BUCKETS)) - SKIPPED_BUCKETS
    //           = max(bits, SKIPPED_BUCKETS) - SKIPPED_BUCKETS
    //           = max(bits - SKIPPED_BUCKETS, 0)
    //
    // This `as` never overflows, since we know `bits ≤ usize::BITS < usize::MAX`.
    (bits as usize).saturating_sub(SKIPPED_BUCKETS)
}

impl<const BUCKETS: usize> Index<BUCKETS> {
    /// Get the bucket cursor located between the bucket this index is in and the next bucket.
    fn after_bucket(self) -> BucketCursor<BUCKETS> {
        let location = self.location();

        // We can add one without overflowing because `bucket < BUCKETS`.
        BucketCursor(location.bucket.0 + 1)
    }

    /// Get the bucket cursor located after every bucket that has `Index`es lower than this one (and
    /// before all buckets whose every `Index` is greater than or equal to this one).
    ///
    /// In other words, if the index is the first entry in the bucket, get the bucket cursor before
    /// the current bucket, otherwise get the bucket cursor after the current bucket.
    fn after_lower_buckets(self) -> BucketCursor<BUCKETS> {
        // Every bucket with index `b - SKIPPED_BUCKETS` whose every `Index` is greater than or
        // equal to this one satisfies the following equivalent properties:
        //
        //   ∀ i ≥ (0..b).map(|j| 2 ^ j).sum(), i ≥ self.inner - 1
        // ⇔ self.inner - 1 ≤ (0..b).map(|j| 2 ^ j).sum()
        //                  = 2 ^ b - 1
        // ⇔ log2(self.inner) ≤ b
        //
        // Thus, the first such bucket is the smallest integer `b` satisfying the inequality, i.e.
        // `ceil(log2(self.inner))`, which may be alternately expressed as
        // `floor(log2(self.inner - 1)) + 1`. Note that these formulae diverge in the case where
        // `self.inner = 1`; however, since `1 ≤ SKIPPED_BUCKETS` this cannot happen.

        // Panics: Can always subtract one from a `NonZero`.
        //
        // The compiler knows this, so we just unwrap.
        let inner_minus_one = self.into_raw().get().checked_sub(1).unwrap();

        // Calculate the bucket cursor index, which is the index of the first non-lower bucket.
        //
        // bucket = b - SKIPPED_BUCKETS
        //        = floor(log2(self.inner - 1)) + 1 - SKIPPED_BUCKETS
        //
        // We do the calculation in this order to ensure that overflow does not occur,
        // again relying on `1 ≤ SKIPPED_BUCKETS`:
        //
        //   SKIPPED_BUCKETS = log2(SKIPPED_ENTRIES + 1)
        //                   ≤ log2(self.inner)
        //                   ≤ ceil(log2(self.inner))
        //                   = floor(log2(self.inner - 1)) + 1
        // ⇒ 0 ≤ floor(log2(self.inner - 1)) + 1 - SKIPPED_BUCKETS
        //
        // Panics (ilog2): `SKIPPED_BUCKETS` is at least 1, meaning that `2 ≤ self.inner`.
        // The compiler can tell that `ilog2` always gives a valid `usize`, so we just unwrap.
        let bucket = usize::try_from(inner_minus_one.ilog2()).unwrap() + 1 - SKIPPED_BUCKETS;

        BucketCursor(bucket)
    }
}

/// A point between two buckets – if each bucket is a fence, this represents a fencepost.
///
/// Invariant: `self.0 ≤ BUCKETS`.
#[derive(Clone, Default)]
struct BucketCursor<const BUCKETS: usize>(usize);

impl<const BUCKETS: usize> BucketCursor<BUCKETS> {
    /// Advance this cursor and return the bucket index that was advanced over.
    fn advance(&mut self) -> Option<BucketIndex<BUCKETS>> {
        if self.0 >= BUCKETS {
            return None;
        }

        // This is okay because we just ensured `self.0 < BUCKETS`.
        let index = BucketIndex(self.0);
        self.0 += 1;
        Some(index)
    }

    /// Move this cursor backward and return the bucket index that was advanced over.
    fn retreat(&mut self) -> Option<BucketIndex<BUCKETS>> {
        self.0 = self.0.checked_sub(1)?;

        // This is okay because our invariant is that `self.0 ≤ BUCKETS`, therefore
        // `self.0 - 1 < BUCKETS`.
        Some(BucketIndex(self.0))
    }
}

impl<'a, T, const BUCKETS: usize> IntoIterator for &'a Buckets<T, BUCKETS> {
    type Item = (Index<BUCKETS>, &'a T);
    type IntoIter = Iter<'a, T, BUCKETS>;

    fn into_iter(self) -> Self::IntoIter {
        Iter {
            buckets: self,
            bucket: BucketCursor::default(),
            iter: [].iter(),
            index: 0,
        }
    }
}

/// An iterator over shared references to the elements in a [`Buckets`].
#[must_use]
pub struct Iter<'a, T, const BUCKETS: usize> {
    buckets: &'a Buckets<T, BUCKETS>,
    bucket: BucketCursor<BUCKETS>,
    iter: slice::Iter<'a, T>,
    /// The `Index<BUCKETS>` of the next element if there is one, or an invalid index otherwise.
    index: usize,
}

impl<'a, T, const BUCKETS: usize> Iterator for Iter<'a, T, BUCKETS> {
    type Item = (Index<BUCKETS>, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // If we have more elements in the current bucket, yield them first.
            if let Some(item) = self.iter.next() {
                // Safety: Invariant of the `index` field.
                let index = unsafe { Index::from_raw_unchecked(self.index) };
                // This might overflow, which is okay because `self.iter` would be empty.
                self.index = self.index.wrapping_add(1);
                return Some((index, item));
            }

            let bucket_index = self.bucket.advance()?;
            let bucket = self.buckets.bucket(bucket_index);

            // `Acquire` is necessary because we access the pointer afterward.
            let ptr = bucket.load(atomic::Ordering::Acquire);

            // Skip over unallocated buckets. We need to check every bucket since it's possible
            // later buckets got allocated first.
            if !ptr.is_null() {
                // Safety: Guaranteed by the invariants of `Buckets`.
                let slice = unsafe { slice::from_raw_parts(ptr, bucket_index.len().get()) };
                self.iter = slice.iter();
                self.index = bucket_index.first().into_raw().get();
            }
        }
    }
}

// Valid since `BucketIter` is fused.
impl<T, const BUCKETS: usize> FusedIterator for Iter<'_, T, BUCKETS> {}

impl<T, const BUCKETS: usize> Clone for Iter<'_, T, BUCKETS> {
    fn clone(&self) -> Self {
        Self {
            buckets: self.buckets,
            bucket: self.bucket.clone(),
            iter: self.iter.clone(),
            index: self.index,
        }
    }
}

impl<'a, T, const BUCKETS: usize> IntoIterator for &'a mut Buckets<T, BUCKETS> {
    type Item = (Index<BUCKETS>, &'a mut T);
    type IntoIter = IterMut<'a, T, BUCKETS>;

    fn into_iter(self) -> Self::IntoIter {
        IterMut {
            buckets: self,
            bucket: BucketCursor::default(),
            iter: [].iter_mut(),
            index: 0,
        }
    }
}

/// An iterator over unique references to the elements in a [`Buckets`].
#[must_use]
pub struct IterMut<'a, T, const BUCKETS: usize> {
    buckets: &'a mut Buckets<T, BUCKETS>,
    bucket: BucketCursor<BUCKETS>,
    iter: slice::IterMut<'a, T>,
    /// The index of the next element if there is one, or an invalid index otherwise.
    index: usize,
}

impl<'a, T, const BUCKETS: usize> Iterator for IterMut<'a, T, BUCKETS> {
    type Item = (Index<BUCKETS>, &'a mut T);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // If we have more elements in the current bucket, yield them first.
            if let Some(item) = self.iter.next() {
                // Safety: Invariant of the `index` field.
                let index = unsafe { Index::from_raw_unchecked(self.index) };
                // This might overflow, which is okay because `self.iter` would be empty.
                self.index = self.index.wrapping_add(1);
                return Some((index, item));
            }

            let bucket_index = self.bucket.advance()?;
            let bucket = self.buckets.bucket_mut(bucket_index);

            let ptr = bucket.read_mut();

            // Skip over unallocated buckets. We need to check every bucket since it's possible
            // later buckets got allocated first.
            if !ptr.is_null() {
                // Safety: Guaranteed by the invariants of `Buckets`.
                let slice = unsafe { slice::from_raw_parts_mut(ptr, bucket_index.len().get()) };
                self.iter = slice.iter_mut();
                self.index = bucket_index.first().into_raw().get();
            }
        }
    }
}

// Valid since `BucketIter` is fused.
impl<T, const BUCKETS: usize> FusedIterator for IterMut<'_, T, BUCKETS> {}

impl<T, const BUCKETS: usize> IntoIterator for Buckets<T, BUCKETS> {
    type Item = (Index<BUCKETS>, T);
    type IntoIter = IntoIter<T, BUCKETS>;

    fn into_iter(self) -> Self::IntoIter {
        IntoIter {
            buckets: self,
            bucket: BucketCursor::default(),
            iter: Vec::new().into_iter(),
            index: 0,
        }
    }
}

/// An iterator over the elements in a [`Buckets`].
#[must_use]
pub struct IntoIter<T, const BUCKETS: usize> {
    buckets: Buckets<T, BUCKETS>,
    bucket: BucketCursor<BUCKETS>,
    iter: vec::IntoIter<T>,
    /// The index of the next element if there is one, or an invalid index otherwise.
    index: usize,
}

impl<T, const BUCKETS: usize> Iterator for IntoIter<T, BUCKETS> {
    type Item = (Index<BUCKETS>, T);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // If we have more elements in the current bucket, yield them first.
            if let Some(item) = self.iter.next() {
                // Safety: Invariant of the `index` field.
                let index = unsafe { Index::from_raw_unchecked(self.index) };
                // This might overflow, which is okay because `self.iter` would be empty.
                self.index = self.index.wrapping_add(1);
                return Some((index, item));
            }

            let bucket_index = self.bucket.advance()?;

            // Skip over unallocated buckets. We need to check every bucket since it's possible
            // later buckets got allocated first.
            if let Some(bucket) = self.buckets.take_bucket(bucket_index) {
                self.iter = Vec::from(bucket).into_iter();
                self.index = bucket_index.first().into_raw().get();
            }
        }
    }
}

// Valid since `BucketIter` is fused.
impl<T, const BUCKETS: usize> FusedIterator for IntoIter<T, BUCKETS> {}

#[cfg(test)]
mod tests {
    use super::buckets_for_index_bits;
    use super::Buckets;
    use super::Index;
    use super::MaybeZeroable;
    use crate::buckets::SKIPPED_BUCKETS;
    use crate::buckets::SKIPPED_ENTRIES;
    use alloc::vec::Vec;
    use core::cell::Cell;

    std::thread_local!(static COUNTER: Cell<usize> = const { Cell::new(0) });

    #[derive(Default)]
    struct Helper;
    unsafe impl MaybeZeroable for Helper {
        fn zeroable() -> bool {
            true
        }
    }
    impl Drop for Helper {
        fn drop(&mut self) {
            COUNTER.with(|c| c.set(c.get() + 1));
        }
    }

    fn drops_with(f: impl FnOnce()) -> usize {
        COUNTER.with(|c| c.set(0));
        f();
        COUNTER.with(|c| c.get())
    }

    fn drops<const BUCKETS: usize>(buckets: Buckets<Helper, BUCKETS>) -> usize {
        drops_with(|| drop(buckets))
    }

    #[test]
    fn new() {
        assert_eq!(drops(<Buckets<_, 1>>::new()), 0);
        assert_eq!(
            drops(<Buckets<_, { buckets_for_index_bits(usize::BITS) }>>::new()),
            0
        );
    }

    #[test]
    fn reserve() {
        let buckets = <Buckets<Helper, 8>>::new();
        buckets.reserve(Index::new(0).unwrap());
        assert_eq!(drops(buckets), SKIPPED_ENTRIES + 1);

        let buckets = <Buckets<Helper, 8>>::new();
        buckets.reserve(Index::new(SKIPPED_ENTRIES).unwrap());
        assert_eq!(drops(buckets), SKIPPED_ENTRIES + 1);

        let buckets = <Buckets<Helper, 8>>::new();
        buckets.reserve(Index::new(SKIPPED_ENTRIES + 1).unwrap());
        assert_eq!(drops(buckets), (SKIPPED_ENTRIES + 1) * 3);

        let buckets = <Buckets<Helper, 5>>::new();
        let total = (1 << (5 + SKIPPED_BUCKETS)) - SKIPPED_ENTRIES - 1;
        assert_eq!(<Index<5>>::new(total), None);
        buckets.reserve(Index::new(total - 1).unwrap());
        assert_eq!(drops(buckets), total);
    }

    #[test]
    fn truncate_exact() {
        let mut buckets = <Buckets<Helper, 5>>::new();
        let first_in_second_bucket = Index::new(SKIPPED_ENTRIES + 1).unwrap();

        buckets.reserve(first_in_second_bucket);

        assert_eq!(
            drops_with(|| buckets.truncate(Index::new(SKIPPED_ENTRIES + 2).unwrap())),
            0,
        );
        assert_eq!(
            drops_with(|| buckets.truncate(first_in_second_bucket)),
            (SKIPPED_ENTRIES + 1) * 2
        );
        assert!(buckets.get(first_in_second_bucket).is_none());
        assert_eq!(drops(buckets), SKIPPED_ENTRIES + 1);
    }

    #[test]
    fn get_or_alloc() {
        let buckets = <Buckets<Helper, 8>>::new();
        buckets.get_or_alloc(Index::new(48).unwrap());
        assert_eq!(drops(buckets), 64);
    }

    #[test]
    fn iter() {
        let mut buckets = <Buckets<u16, 4>>::new();

        // buckets go 32, 64, 128, 256; we allocate the second and fourth
        buckets.get_or_alloc(Index::new(48).unwrap());
        buckets.get_or_alloc(Index::new(225).unwrap());

        let indices = buckets.iter().map(|(i, _)| i).collect::<Vec<_>>();
        assert_eq!(indices.len(), 320);
        for &i in &indices {
            assert_eq!(*buckets.get(i).unwrap(), 0);
        }

        for (i, (index, val)) in buckets.iter_mut().enumerate() {
            assert_eq!(indices[i], index);
            *val = i as u16;
        }

        for (i, (index, val)) in buckets.into_iter().enumerate() {
            assert_eq!(indices[i], index);
            assert_eq!(val, i as u16);
        }
    }

    #[test]
    fn location() {
        let index = <Index<12>>::new(0).unwrap();
        assert_eq!(index.location().bucket.0, 0);
        assert_eq!(index.location().bucket_len.get(), 32);
        assert_eq!(index.location().entry, 0);

        let index = <Index<12>>::new(31).unwrap();
        assert_eq!(index.location().bucket.0, 0);
        assert_eq!(index.location().bucket_len.get(), 32);
        assert_eq!(index.location().entry, 31);

        let index = <Index<12>>::new(34).unwrap();
        assert_eq!(index.location().bucket.0, 1);
        assert_eq!(index.location().bucket_len.get(), 64);
        assert_eq!(index.location().entry, 2);

        let max = usize::MAX - SKIPPED_ENTRIES - 1;
        assert_eq!(
            <Index<{ buckets_for_index_bits(usize::BITS) }>>::new(max + 1),
            None
        );
        let index = <Index<{ buckets_for_index_bits(usize::BITS) }>>::new(max).unwrap();
        assert_eq!(
            index.location().bucket.0,
            (usize::BITS as usize) - SKIPPED_BUCKETS - 1
        );
        assert_eq!(index.location().bucket_len.get(), usize::MAX / 2 + 1);
        assert_eq!(index.location().entry, usize::MAX / 2);
    }
}
