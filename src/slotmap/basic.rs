use core::convert::Infallible;
use core::fmt::{self, Debug, Formatter};
use core::iter::FusedIterator;
use core::marker::PhantomData;
use core::mem::{ManuallyDrop, MaybeUninit};
use core::ops::{Index, IndexMut};

use super::{buckets_index_as_u32, buckets_index_from_u32, BUCKETS, FIRST_OOB_INDEX, ZERO_INDEX};
use super::{DefaultKey, Key, KeyData};
use crate::buckets::{self, Buckets, MaybeZeroable};
use crate::loom::atomic::{self, AtomicU32, AtomicUsize};
use crate::loom::cell::UnsafeCell;
use crate::loom::AtomicMut as _;
use core::panic::{RefUnwindSafe, UnwindSafe};

/// A concurrent, append-only [slot map](https://docs.rs/slotmap).
///
/// # Notes
///
/// The bucket array is stored inline, meaning that the `SlotMap<K, V>` type
/// is quite large on the stack. It is expected that you store it behind
/// an [`Arc`](std::sync::Arc) or similar.
pub struct SlotMap<K: Key, V> {
    buckets: Buckets<Entry<V>, BUCKETS>,
    meta: Meta,
    _key: PhantomData<fn() -> K>,
}

/// The parts of the `SlotMap` that excludes `Buckets`.
///
/// This is used in `Drain` to store a single reference that can access both these fields – it also
/// makes a couple other methods a little more convenient to write. `&SlotMap<K, V>` cannot be used
/// since that would borrow the `Buckets` field.
struct Meta {
    /// The index of the head of the free-list as a `buckets::Index`, or any value greater than
    /// `FIRST_OOB_INDEX` if the `SlotMap` is completely full.
    ///
    /// This is an `AtomicUsize` because on 64-bit systems we want to allow 2³² elements in a
    /// slotmap while allowing overflow. On 32-bit systems, we only allow 2³¹ elements in a
    /// slotmap, so overflow is naturally accounted for in the top bit.
    ///
    /// We just use `Relaxed` on this atomic, since it's only a counter and data does not depend on
    /// it.
    next_free: AtomicUsize,

    /// The number of initialized elements.
    ///
    /// We generally use `Acquire`/`Release` ordering for this field, even though this is not
    /// necessary for soundness, to give guarantees of sane behaviour (i.e. if you read a certain
    /// length, you will definitely see at least that many initialized values when iterating over
    /// the `SlotMap`).
    len: AtomicU32,
}

struct Entry<T> {
    /// Even if vacant (or yet to be filled), odd if occupied.
    version: AtomicU32,

    /// The value itself. Initialized and available for reads iff the version is odd. If the
    /// version is even, either this entry is in the free-list or it was popped and is now owned by
    /// the popping thread.
    value: UnsafeCell<MaybeUninit<T>>,

    /// The next free slot in the free-list – equivalent to [`NextFree`] but stored more compactly.
    ///
    /// Unlike in normal `SlotMap`, we can't store this in a union with the value, because threads
    /// need to be able to access it even after the head points elsewhere.
    next_free: NextFreePacked,
}

/// A link from one element of the free-list to the next.
enum NextFree {
    /// The free-list consists of all slots from the current one to the end.
    ///
    /// This value is represented as zero in `NextFreePacked`, which is important because when we
    /// allocate a new bucket, all the entries will have this value.
    AllRemainingEntries,

    /// The map is full and this is the end of the free-list.
    ///
    /// Equivalent to when `Meta::next_free` stores an out-of-bounds index.
    None,

    /// The next element of the free-list is located at the given index.
    At(buckets::Index<BUCKETS>),
}

impl<V> SlotMap<DefaultKey, V> {
    /// Construct a new, empty `SlotMap` with the the default key type.
    #[cfg(not(loom))]
    pub const fn new() -> Self {
        Self::with_key()
    }

    #[cfg(loom)]
    pub fn new() -> Self {
        Self::with_key()
    }
}

impl<K: Key, V> SlotMap<K, V> {
    /// Construct a new, empty `SlotMap` with a custom key type.
    #[cfg(not(loom))]
    pub const fn with_key() -> Self {
        Self {
            buckets: Buckets::new(),
            meta: Meta {
                next_free: AtomicUsize::new(ZERO_INDEX.into_raw().get()),
                len: AtomicU32::new(0),
            },
            _key: PhantomData,
        }
    }

    #[cfg(loom)]
    pub fn with_key() -> Self {
        Self {
            buckets: Buckets::new(),
            meta: Meta {
                next_free: AtomicUsize::new(ZERO_INDEX.into_raw().get()),
                len: AtomicU32::new(0),
            },
            _key: PhantomData,
        }
    }

    /// Construct a new, empty, `SlotMap`, with enough allocated capacity to hold `capacity`
    /// elements without reallocating.
    pub fn with_capacity(capacity: usize) -> Self {
        let mut this = Self::with_key();
        this.reserve_mut(capacity);
        this
    }

    /// Get the number of entries in the `SlotMap`.
    ///
    /// If you call this and then iterate over the map, it is guaranteed you will see at least that
    /// many elements, but you may see more if more are added in the meantime.
    ///
    /// # Examples
    ///
    /// ```
    /// # use boxcar::slotmap::*;
    /// let mut map = SlotMap::new();
    /// assert_eq!(map.len(), 0);
    ///
    /// map.insert(0);
    /// assert_eq!(map.len(), 1);
    ///
    /// map.insert(0);
    /// let (key, _) = map.insert(0);
    /// assert_eq!(map.len(), 3);
    ///
    /// map.remove(key);
    /// assert_eq!(map.len(), 2);
    ///
    /// map.clear();
    /// assert_eq!(map.len(), 0);
    /// ```
    #[must_use]
    pub fn len(&self) -> usize {
        // See the doc comment of the field for ordering considerations.
        self.meta.len.load(atomic::Ordering::Acquire) as usize
    }

    /// Query whether the `SlotMap` is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// # use boxcar::slotmap::*;
    /// let mut map = SlotMap::new();
    /// assert!(map.is_empty());
    ///
    /// let (key, _) = map.insert(0);
    /// assert!(!map.is_empty());
    ///
    /// map.remove(key);
    /// assert!(map.is_empty());
    /// ```
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Reserve space for at least `additional` new elements to be added to the `SlotMap` without
    /// allocating.
    pub fn reserve(&self, additional: usize) {
        let len = self.len();

        if let Some(max_index) = len.saturating_add(additional).checked_sub(1) {
            self.buckets
                .reserve(buckets::Index::new_saturating(max_index));
        }
    }

    /// Like [`Self::reserve`], but taking `&mut`. This can avoid synchronization sometimes.
    ///
    /// # Panics
    ///
    /// May panic if more elements are requested than this `SlotMap` can possibly hold, or if
    /// allocation fails.
    pub fn reserve_mut(&mut self, additional: usize) {
        let len = self.meta.len.read_mut() as usize;

        if let Some(max_index) = len.saturating_add(additional).checked_sub(1) {
            self.buckets
                .reserve(buckets::Index::new_saturating(max_index));
        }
    }

    /// Insert a value into the `SlotMap`, returning its key and a reference to the value.
    ///
    /// # Examples
    ///
    /// ```
    /// # use boxcar::slotmap::*;
    /// let map = SlotMap::new();
    ///
    /// let (k, v) = map.insert("hello".to_owned());
    /// assert_eq!(v, "hello");
    /// assert_eq!(map[k], "hello");
    /// ```
    pub fn insert(&self, value: V) -> (K, &V) {
        let (key, value, ()) = self.insert_with_updater(value, |_, _| {});
        (key, value)
    }

    /// Insert a value into the `SlotMap`, running a function on the value once it has been
    /// inserted.
    ///
    /// Since the function accepts the key the value will be stored at, this can be used to insert
    /// values that store their own key.
    ///
    /// The value will only be made visible to other threads after `updater` has completed, either
    /// by returning successfully or by panicking.
    ///
    /// # Examples
    ///
    /// ```
    /// # use boxcar::slotmap::*;
    /// let map = SlotMap::new();
    ///
    /// let (key, val, ()) = map.insert_with_updater(DefaultKey::null(), |val, key| *val = key);
    /// assert_eq!(key, *val);
    /// assert_eq!(map[key], key);
    /// ```
    pub fn insert_with_updater<O>(
        &self,
        value: V,
        updater: impl FnOnce(&mut V, K) -> O,
    ) -> (K, &V, O) {
        // See the doc comment of the field for ordering considerations.
        let mut next_free_raw = self.meta.next_free.load(atomic::Ordering::Relaxed);

        let (index, entry) = loop {
            // Since `next_free` can overflow and be out-of-bounds, we check whether it's in bounds.
            // If it's not, we've run out of space, so we just panic.
            //
            // Safety: `next_free` is always assigned a valid index and added to, and it cannot
            // overflow its data type (`usize`).
            let next_free =
                unsafe { buckets::Index::from_raw_checked_above(next_free_raw) }.unwrap();

            let entry = self.buckets.get_or_alloc(next_free);

            let new_next_free = match entry.next_free.get() {
                NextFree::AllRemainingEntries => {
                    // We avoid looping by incrementing `next_free` directly.
                    let slot = self.meta.next_free.fetch_add(1, atomic::Ordering::Relaxed);

                    // In the simple case, there were no concurrent pops of the free list.
                    if slot == next_free_raw {
                        break (next_free, entry);
                    }

                    // Otherwise, we check for overflow.
                    let Some(slot) = (unsafe { buckets::Index::from_raw_checked_above(slot) })
                    else {
                        // If overflow occurred, we undo our `fetch_add` and panic. This ensures
                        // that overflow of the full `usize` cannot happen unless there are
                        // `isize::MAX` concurent threads, which we assume is impossible.
                        self.meta.next_free.fetch_sub(1, atomic::Ordering::Relaxed);
                        panic!();
                    };

                    break (slot, self.buckets.get_or_alloc(slot));
                }
                NextFree::None => FIRST_OOB_INDEX,
                NextFree::At(index) => index.into_raw().get(),
            };

            // Pop from the free list, attempting to take ownership over the slot.
            match self.meta.next_free.compare_exchange(
                next_free_raw,
                new_next_free,
                atomic::Ordering::Relaxed,
                atomic::Ordering::Relaxed,
            ) {
                Ok(_) => break (next_free, entry),
                // If we fail, try again.
                Err(updated) => next_free_raw = updated,
            }
        };

        // Safety: The only time the version is written to is by this thread, when the value is
        // written. Therefore, we can load this without synchronization.
        let version = unsafe { entry.version.unsync_load() };

        // Safety: Slots can only be in the free list if they are empty.
        let key = K::from(unsafe { KeyData::pack_vacant(index, version).unwrap_unchecked() });

        // Safety: We have taken ownership of the slot in the above loop.
        unsafe { entry.value.with_mut(|v| *v = MaybeUninit::new(value)) };

        // A guard that marks the entry as initialized on drop. This is used to ensure that the
        // entry gets marked initialized even if the updater panics.
        struct Guard<'a, V>(&'a Meta, &'a Entry<V>, u32);

        impl<V> Drop for Guard<'_, V> {
            fn drop(&mut self) {
                // Two things are getting `Release`d here: both the fact we updated the head,
                // and the value we just wrote.
                self.1.version.store(
                    KeyData::version_make_occupied(self.2),
                    atomic::Ordering::Release,
                );

                // See the doc comment of the field for ordering considerations.
                self.0.len.fetch_add(1, atomic::Ordering::Release);
            }
        }

        let output = {
            let _guard = Guard(&self.meta, entry, version);

            // Safety: As above, we took ownership of the slot. The `assume_init_mut()` is safe
            // since we just initialized it.
            unsafe {
                entry
                    .value
                    .with_mut(|v| updater((*v).assume_init_mut(), key))
            }
        };

        // Safety: We just initialized thre slot.
        (key, unsafe { entry.get_unchecked() }, output)
    }

    /// Insert a value into the `SlotMap` using a fallible function of the key.
    ///
    /// `make(state)` will be called to attempt construction of the value. If this fails or panics,
    /// the operation will immediately abort; otherwise, we will attempt to insert the value with
    /// the given key. However, in contended scenarios this may not always succeed, so in such
    /// situations `fail(value)` will be called and the operation restarted until it does.
    ///
    /// If possible, you should prefer using the infallible [`Self::insert_with_updater`] API.
    ///
    /// # Examples
    ///
    /// ```
    /// # use boxcar::slotmap::*;
    /// let map = SlotMap::new();
    ///
    /// let (key, value) = map.try_insert_with::<_, ()>((), |(), key| Ok(key), |_| {}).unwrap();
    /// assert_eq!(*value, key);
    /// assert_eq!(map[key], key);
    ///
    /// map.try_insert_with((), |_, _| Err(()), |_| {}).unwrap_err();
    /// ```
    pub fn try_insert_with<S, E>(
        &self,
        mut state: S,
        mut make: impl FnMut(S, K) -> Result<V, E>,
        mut fail: impl FnMut(V) -> S,
    ) -> Result<(K, &V), E> {
        // See the doc comment of the field for ordering considerations.
        let mut next_free_raw = self.meta.next_free.load(atomic::Ordering::Relaxed);

        loop {
            // Since `next_free` can overflow and be out-of-bounds, we check whether it's in bounds.
            // If it's not, we've run out of space, so we just panic.
            //
            // Safety: `next_free` is always assigned a valid index and added to, and it cannot
            // overflow its data type (`usize`).
            let next_free =
                unsafe { buckets::Index::from_raw_checked_above(next_free_raw) }.unwrap();

            let entry = self.buckets.get_or_alloc(next_free);

            // `Acquire` ensures if that `Key::pack_vacant` fails (which happens if the current slot
            // just had a value written to it), we are able to read the updated value of `next_free`
            // so we don't end up spinlooping.
            let version = entry.version.load(atomic::Ordering::Acquire);

            // `Key::pack_vacant` can fail as described above. If it does, read the updated
            // `next_free` head and try again.
            let Some(key) = KeyData::pack_vacant(next_free, version) else {
                next_free_raw = self.meta.next_free.load(atomic::Ordering::Relaxed);
                continue;
            };

            // Compute the value to put in this slot. If this panics or fails, it doesn't matter,
            // since we haven't done any writes yet.
            let val = make(state, K::from(key))?;

            // Find where the next free slot is.
            let new_next_free = match entry.next_free.get() {
                // This might become out of bounds, which is okay, but it won't ever overflow a
                // `usize`.
                NextFree::AllRemainingEntries => next_free_raw + 1,
                NextFree::None => FIRST_OOB_INDEX,
                NextFree::At(index) => index.into_raw().get(),
            };

            // Pop from the free list, attempting to take ownership over the slot.
            match self.meta.next_free.compare_exchange(
                next_free_raw,
                new_next_free,
                atomic::Ordering::Relaxed,
                atomic::Ordering::Relaxed,
            ) {
                // If we succeeded, we've now taken ownership over this slot. Thus we can write our
                // value and return.
                Ok(_) => {
                    // Safety: We just took ownership over this slot.
                    unsafe { entry.value.with_mut(|v| *v = MaybeUninit::new(val)) };

                    // Two things are getting `Release`d here: both the fact we updated the head,
                    // and the value we just wrote.
                    entry.version.store(
                        KeyData::version_make_occupied(version),
                        atomic::Ordering::Release,
                    );

                    // See the doc comment of the field for ordering considerations.
                    self.meta.len.fetch_add(1, atomic::Ordering::Release);

                    // Safety: We just initialized the slot.
                    break Ok((K::from(key), unsafe { entry.get_unchecked() }));
                }
                // Otherwise, the head has moved since we last checked. Try again with the updated
                // head.
                Err(updated) => {
                    state = fail(val);
                    next_free_raw = updated;
                }
            }
        }
    }

    /// Insert a value into the `SlotMap`, returning its key and a unique reference to the value.
    ///
    /// This can avoid some overhead in comparison to [`Self::insert`].
    ///
    /// # Examples
    ///
    /// ```
    /// # use boxcar::slotmap::*;
    /// let mut map = SlotMap::new();
    ///
    /// let (key, value) = map.insert_mut("hello".to_owned());
    /// value.push_str(" world");
    ///
    /// assert_eq!(map[key], "hello world");
    /// ```
    pub fn insert_mut(&mut self, value: V) -> (K, &mut V) {
        self.insert_mut_with(|_| value)
    }

    /// Insert a value into the `SlotMap` by running a closure that computes it from its key,
    /// and return a unique reference to the value.
    ///
    /// This can avoid some overhead in comparison to [`Self::insert_with_updater`].
    ///
    /// # Examples
    ///
    /// ```
    /// # use boxcar::slotmap::*;
    /// let mut map = SlotMap::new();
    ///
    /// let (key, value) = map.insert_mut_with(|key| key);
    /// assert_eq!(*value, key);
    /// assert_eq!(map[key], key);
    /// ```
    pub fn insert_mut_with(&mut self, f: impl FnOnce(K) -> V) -> (K, &mut V) {
        match self.try_insert_mut_with::<Infallible>(|key| Ok(f(key))) {
            Ok(t) => t,
            Err(e) => match e {},
        }
    }

    /// Insert a value into the `SlotMap` by running a fallible closure that computes it from its
    /// key, and return a unique reference to the value.
    ///
    /// This can avoid some overhead in comparison to [`Self::try_insert_with`].
    pub fn try_insert_mut_with<E>(
        &mut self,
        f: impl FnOnce(K) -> Result<V, E>,
    ) -> Result<(K, &mut V), E> {
        let next_free_raw = self.meta.next_free.read_mut();

        // Since `next_free` can overflow and be out-of-bounds, we check whether it's in bounds.
        // If it's not, we've run out of space, so we just panic.
        //
        // Safety: `next_free` is always assigned a valid index and added to, and it cannot
        // overflow its data type (`usize`).
        let next_free = unsafe { buckets::Index::from_raw_checked_above(next_free_raw) }.unwrap();

        let entry = self.buckets.get_or_alloc_mut(next_free);

        let version = entry.version.read_mut();

        // Panics: Slots can only be in the free-list if they are empty.
        let key = K::from(KeyData::pack_vacant(next_free, version).unwrap());

        let value = f(key)?;

        // Pop from the free-list, advancing its head.
        self.meta.next_free.write_mut(match entry.next_free.get() {
            // This might become out of bounds, which is okay, but it won't ever overflow a `usize`.
            NextFree::AllRemainingEntries => next_free_raw + 1,
            NextFree::None => FIRST_OOB_INDEX,
            NextFree::At(index) => index.into_raw().get(),
        });

        entry.value = UnsafeCell::new(MaybeUninit::new(value));
        entry
            .version
            .write_mut(KeyData::version_make_occupied(version));

        let len = self.meta.len.read_mut();
        self.meta.len.write_mut(len + 1);

        // Safety: We just initialized the slot.
        Ok((key, unsafe { entry.get_unchecked_mut() }))
    }

    /// Get a value in the `SlotMap` by its key.
    ///
    /// Returns [`None`] if no value at this key exists.
    ///
    /// # Examples
    ///
    /// ```
    /// # use boxcar::slotmap::*;
    /// let mut map = SlotMap::new();
    ///
    /// let (key, _) = map.insert("hello".to_owned());
    /// assert_eq!(map.get(DefaultKey::null()), None);
    /// assert_eq!(map.get(key).unwrap(), "hello");
    ///
    /// // The slot might be reused, but the key won't be.
    /// map.remove(key);
    /// map.insert("world".to_owned());
    /// assert_eq!(map.get(key), None);
    /// ```
    pub fn get(&self, key: K) -> Option<&V> {
        let (index, version) = key.data().unpack();
        let entry = self.buckets.get(index)?;
        if version != entry.version.load(atomic::Ordering::Acquire) {
            return None;
        }
        Some(unsafe { entry.get_unchecked() })
    }

    /// Get a unique reference to a value in the `SlotMap` by its key.
    ///
    /// Returns [`None`] if no value at this key exists.
    ///
    /// # Examples
    ///
    /// ```
    /// # use boxcar::slotmap::*;
    /// let mut map = SlotMap::new();
    ///
    /// let (key, _) = map.insert("hello".to_owned());
    /// assert_eq!(map.get_mut(DefaultKey::null()), None);
    /// assert_eq!(map.get_mut(key).unwrap(), "hello");
    ///
    /// // The slot might be reused, but the key won't be.
    /// map.remove(key);
    /// map.insert("world".to_owned());
    /// assert_eq!(map.get_mut(key), None);
    /// ```
    pub fn get_mut(&mut self, key: K) -> Option<&mut V> {
        let (index, version) = key.data().unpack();
        let entry = self.buckets.get_mut(index)?;
        if version != entry.version.read_mut() {
            return None;
        }
        Some(unsafe { entry.get_unchecked_mut() })
    }

    /// Like [`get`](Self::get), but without bounds checks.
    ///
    /// # Examples
    ///
    /// ```
    /// # use boxcar::slotmap::*;
    /// let mut map = SlotMap::new();
    /// let (key, _) = map.insert("hello".to_owned());
    /// assert_eq!(unsafe { map.get_unchecked(key) }, "hello");
    /// ```
    ///
    /// # Safety
    ///
    /// `key` must refer to a valid entry in the `SlotMap`.
    pub unsafe fn get_unchecked(&self, key: K) -> &V {
        let (index, _) = key.data().unpack();
        let entry = unsafe { self.buckets.get_unchecked(index) };
        unsafe { entry.get_unchecked() }
    }

    /// Like [`get_mut`](Self::get_mut), but without bounds checks.
    ///
    /// # Examples
    ///
    /// ```
    /// # use boxcar::slotmap::*;
    /// let mut map = SlotMap::new();
    /// let (key, _) = map.insert("hello".to_owned());
    /// assert_eq!(unsafe { map.get_unchecked_mut(key) }, "hello");
    /// ```
    ///
    /// # Safety
    ///
    /// `key` must refer to a valid entry in the `SlotMap`.
    pub unsafe fn get_unchecked_mut(&mut self, key: K) -> &mut V {
        let (index, _) = key.data().unpack();
        let entry = unsafe { self.buckets.get_unchecked_mut(index) };
        unsafe { entry.get_unchecked_mut() }
    }

    /// Query whether the `SlotMap` contains a value at the given key.
    ///
    /// # Examples
    ///
    /// ```
    /// # use boxcar::slotmap::*;
    /// let map = SlotMap::new();
    /// let (key, _) = map.insert(0);
    /// assert!(map.contains_key(key));
    /// ```
    pub fn contains_key(&self, key: K) -> bool {
        self.get(key).is_some()
    }

    /// Iterate over `(key, &value)` pairs in the `SlotMap`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use boxcar::slotmap::*;
    /// let map = SlotMap::new();
    /// let (key1, _) = map.insert(0);
    /// let (key2, _) = map.insert(1);
    ///
    /// let mut iter = map.iter();
    /// assert_eq!(iter.next().unwrap(), (key1, &0));
    /// assert_eq!(iter.next().unwrap(), (key2, &1));
    /// assert_eq!(iter.next(), None);
    /// ```
    pub fn iter(&self) -> Iter<'_, K, V> {
        self.into_iter()
    }

    /// Iterate over `(key, &mut value)` pairs in the `SlotMap`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use boxcar::slotmap::*;
    /// let mut map = SlotMap::new();
    /// let (key1, _) = map.insert(0);
    /// let (key2, _) = map.insert(1);
    ///
    /// let mut iter = map.iter_mut();
    /// assert_eq!(iter.next().unwrap(), (key1, &mut 0));
    /// assert_eq!(iter.next().unwrap(), (key2, &mut 1));
    /// assert_eq!(iter.next(), None);
    /// ```
    pub fn iter_mut(&mut self) -> IterMut<'_, K, V> {
        self.into_iter()
    }

    /// Iterate over the keys of the `SlotMap`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use boxcar::slotmap::*;
    /// let map = SlotMap::new();
    /// let (key1, _) = map.insert(0);
    /// let (key2, _) = map.insert(1);
    ///
    /// let mut iter = map.keys();
    /// assert_eq!(iter.next().unwrap(), key1);
    /// assert_eq!(iter.next().unwrap(), key2);
    /// assert_eq!(iter.next(), None);
    /// ```
    pub fn keys(&self) -> Keys<'_, K, V> {
        Keys(self.iter())
    }

    /// Iterate over the values of the `SlotMap`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use boxcar::slotmap::*;
    /// let map = SlotMap::new();
    /// map.insert(0);
    /// map.insert(1);
    ///
    /// let mut iter = map.values();
    /// assert_eq!(*iter.next().unwrap(), 0);
    /// assert_eq!(*iter.next().unwrap(), 1);
    /// assert_eq!(iter.next(), None);
    /// ```
    pub fn values(&self) -> Values<'_, K, V> {
        Values(self.iter())
    }

    /// Iterate over unique references ot the values in the `SlotMap`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use boxcar::slotmap::*;
    /// let mut map = SlotMap::new();
    /// map.insert(0);
    /// map.insert(1);
    ///
    /// let mut iter = map.values_mut();
    /// assert_eq!(*iter.next().unwrap(), 0);
    /// assert_eq!(*iter.next().unwrap(), 1);
    /// assert_eq!(iter.next(), None);
    /// ```
    pub fn values_mut(&mut self) -> ValuesMut<'_, K, V> {
        ValuesMut(self.iter_mut())
    }

    /// Remove a value from the `SlotMap` given its key.
    ///
    /// Returns the old value, if there was one, and [`None`] otherwise.
    ///
    /// # Examples
    ///
    /// ```
    /// # use boxcar::slotmap::*;
    /// let mut map = SlotMap::new();
    /// let (key, _) = map.insert("hello".to_owned());
    /// assert_eq!(map.remove(key).unwrap(), "hello");
    /// assert_eq!(map.remove(key), None);
    /// ```
    pub fn remove(&mut self, key: K) -> Option<V> {
        let (index, version) = key.data().unpack();
        let entry = self.buckets.get_mut(index)?;
        if version != entry.version.read_mut() {
            return None;
        }
        // Safety: We just checked that `entry` exists and is initialized.
        Some(unsafe { Self::remove_inner(&mut self.meta, index, entry) })
    }

    /// Run a function on every entry in the slot map, only keeping those entries for which the
    /// function returns true.
    ///
    /// # Examples
    ///
    /// ```
    /// # use boxcar::slotmap::*;
    /// let mut map = SlotMap::new();
    /// map.insert(25);
    /// map.insert(2);
    /// map.insert(13);
    /// map.insert(9);
    /// map.retain(|_, &mut v| v >= 10);
    /// assert_eq!(map.values().copied().collect::<Vec<_>>(), [25, 13]);
    /// ```
    pub fn retain(&mut self, mut f: impl FnMut(K, &mut V) -> bool) {
        for (index, entry) in self.buckets.iter_mut().take(self.meta.len.read_mut() as usize) {
            if let Some((key, value)) = entry.get_mut(index) {
                if !f(K::from(key), value) {
                    // Safety: We just checked that `entry` exists and is initialized.
                    unsafe { Self::remove_inner(&mut self.meta, index, entry) };
                }
            }
        }
    }

    /// Remove an entry, adding it to the free-list.
    ///
    /// # Safety
    ///
    /// - `index` must be the index of `entry` inside a `SlotMap`.
    /// - `meta` must be the `meta` field of the same `SlotMap`.
    /// - `entry` must be initialized.
    unsafe fn remove_inner(
        meta: &mut Meta,
        index: buckets::Index<BUCKETS>,
        entry: &mut Entry<V>,
    ) -> V {
        // Copy out the value, then increment the version to make sure that it doesn't get
        // double-dropped. Like regular `SlotMap`, we wrap version numbers on the principle that
        // reusing indices is probably less bad than aborting or causing a memory leak.
        let value = unsafe { entry.read_unchecked() };
        let version = KeyData::version_make_vacant(entry.version.read_mut());
        entry.version.write_mut(version);

        // Update the free list, pushing this entry to the start.
        entry.next_free = NextFreePacked::new(
            match unsafe {
                <buckets::Index<BUCKETS>>::from_raw_checked_above(meta.next_free.read_mut())
            } {
                // If there is a normal slot after this one, that will be the next free slot.
                Some(i) => NextFree::At(i),
                // Otherwise, the index is out-of-bounds, meaning the `SlotMap` was full.
                None => NextFree::None,
            },
        );
        meta.next_free.write_mut(index.into_raw().get());

        // Update the length of the list.
        let len = meta.len.read_mut() - 1;
        meta.len.write_mut(len);

        value
    }

    /// Remove all elements from the `SlotMap`, but keep any allocated capacity.
    pub fn clear(&mut self) {
        drop(self.drain());
    }

    /// Clear the `SlotMap`, returning all key-value pairs in arbitrary order in an iterator, but
    /// keeping any allocated capacity.
    ///
    /// When the returned iterator is dropped, all elements in the map will be removed, even if the
    /// iterator was not fully consumed. If the iterator is instead leaked, only the elements that
    /// were iterated over will be removed, and no memory is otherwise leaked.
    ///
    /// # Examples
    ///
    /// ```
    /// # use boxcar::slotmap::*;
    /// let mut map = SlotMap::new();
    /// let (key1, _) = map.insert("hello".to_owned());
    /// let (key2, _) = map.insert("world".to_owned());
    ///
    /// let mut iter = map.drain();
    /// assert_eq!(iter.next().unwrap(), (key1, "hello".to_owned()));
    /// drop(iter);
    ///
    /// assert_eq!(map.get(key2), None);
    /// assert!(map.is_empty());
    /// ```
    pub fn drain(&mut self) -> Drain<'_, K, V> {
        Drain {
            entries: self.buckets.iter_mut(),
            meta: &mut self.meta,
            _key: PhantomData,
        }
    }

    /// Construct a `SlotMap` from all its underlying (index, version, value) tuples.
    ///
    /// Panics if the iterator contains the same index multiple times. Results in dud slots if some
    /// but not all indices in a bucket are given.
    fn from_all_entries_raw<E>(
        iter: impl IntoIterator<Item = Result<(buckets::Index<BUCKETS>, u32, Option<V>), E>>,
    ) -> Result<Self, E> {
        let mut buckets = <Buckets<Entry<V>, BUCKETS>>::new();
        let mut len = 0;
        let mut free_list_head_tail = None::<(buckets::Index<BUCKETS>, buckets::Index<BUCKETS>)>;
        let mut highest_index = None::<buckets::Index<BUCKETS>>;

        for res in iter {
            let (index, version, value) = res?;

            highest_index = Some(match highest_index {
                Some(highest) => highest.max(index),
                None => index,
            });

            let new_entry = buckets.get_or_alloc_mut(index);
            assert!(!KeyData::version_is_occupied(new_entry.version.read_mut()));
            assert_eq!(KeyData::version_is_occupied(version), value.is_some());

            if let Some(value) = value {
                // If there is a value to add, we write to it and increment the length.
                new_entry.value = UnsafeCell::new(MaybeUninit::new(value));
                len += 1;
            } else {
                // Otherwise, we add to the start of the free-list.
                let tail = match free_list_head_tail {
                    Some((head, tail)) => {
                        new_entry.next_free = NextFreePacked::new(NextFree::At(head));
                        tail
                    }
                    // If the free-list is currently empty, we become the tail. We don't assign to
                    // `next_free` because it's assigned to later.
                    None => index,
                };
                free_list_head_tail = Some((index, tail));
            }

            new_entry.version.write_mut(version);
        }

        // We calculate `after_highest`, which is `highest_index + 1`, or `None` if the entire
        // `SlotMap` is full.
        let after_highest = match highest_index {
            // Safety: We can become out-of-bounds, but this addition will never overflow a
            // `usize`, because our indices never overflow usizes.
            Some(highest_index) => unsafe {
                buckets::Index::from_raw_checked_above(highest_index.into_raw().get() + 1)
            },
            None => Some(buckets::Index::new(0).unwrap()),
        };

        let head = match free_list_head_tail {
            // If the free-list is not empty, then we have to update the tail to point to the
            // actual end of the free-list (i.e. `after_highest`).
            Some((head, tail)) => {
                buckets.get_mut(tail).unwrap().next_free =
                    NextFreePacked::new(match after_highest {
                        Some(after_highest) => NextFree::At(after_highest),
                        None => NextFree::None,
                    });
                head.into_raw().get()
            }
            // If the free-list is empty, then we directly point to `after_highest`.
            None => match after_highest {
                Some(after_highest) => after_highest.into_raw().get(),
                None => FIRST_OOB_INDEX,
            },
        };

        Ok(Self {
            buckets,
            meta: Meta {
                next_free: AtomicUsize::new(head),
                len: AtomicU32::new(len),
            },
            _key: PhantomData,
        })
    }
}

impl<K: Key, V> Default for SlotMap<K, V> {
    fn default() -> Self {
        Self::with_key()
    }
}

impl<K: Key, V: Clone> Clone for SlotMap<K, V> {
    fn clone(&self) -> Self {
        // We can't rely on the existing free-list or length, since concurrent modifications would
        // result in dud slots (slots that are neither initialized nor in the free-list) or an
        // incorrect length. Instead we iterate over the map and rebuild the free-list ourselves.

        let res = Self::from_all_entries_raw(self.buckets.iter().map(|(index, entry)| {
            // `Acquire` is necessary because we read the value after.
            let version = entry.version.load(atomic::Ordering::Acquire);

            let value = KeyData::version_is_occupied(version)
                .then(|| unsafe { entry.get_unchecked() }.clone());

            Ok::<_, Infallible>((index, version, value))
        }));

        match res {
            Ok(map) => map,
            Err(e) => match e {},
        }
    }
}

impl<K: Key, V: Debug> Debug for SlotMap<K, V> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_map().entries(self.iter()).finish()
    }
}

impl<K: Key, V> Index<K> for SlotMap<K, V> {
    type Output = V;
    fn index(&self, index: K) -> &Self::Output {
        self.get(index).unwrap()
    }
}

impl<K: Key, V> IndexMut<K> for SlotMap<K, V> {
    fn index_mut(&mut self, index: K) -> &mut Self::Output {
        self.get_mut(index).unwrap()
    }
}

impl<T> Default for Entry<T> {
    fn default() -> Self {
        Self {
            version: AtomicU32::new(0),
            value: UnsafeCell::new(MaybeUninit::uninit()),
            next_free: NextFreePacked::new(NextFree::AllRemainingEntries),
        }
    }
}

// Safety: `AtomicU32`, `MaybeUninit` and `NextFreePacked` are all zeroable outside Loom.
//
// Note that when zeroed, `next_free` will be `AllRemainingEntries`, which is necessary.
unsafe impl<T> MaybeZeroable for Entry<T> {
    fn zeroable() -> bool {
        // Loom's types are not zeroable; thus we fall back to `Default`.
        cfg!(not(loom))
    }
}

// Safety:
// - `T: Send` is required since we drop our `T`s when the type is dropped.
// - `T: Sync` is not required since we own our `T`s wholly.
unsafe impl<T: Send> Send for Entry<T> {}

// Safety:
// - `T: Send` is required because one can insert into a `SlotMap` from a shared reference.
// - `T: Sync` is required because we provide shared access to the data in the entry from a
//   shared reference.
unsafe impl<T: Send + Sync> Sync for Entry<T> {}

// Same logic as above.
impl<T: UnwindSafe> UnwindSafe for Entry<T> {}
impl<T: UnwindSafe + RefUnwindSafe> RefUnwindSafe for Entry<T> {}

impl<T> Entry<T> {
    unsafe fn get_unchecked(&self) -> &T {
        self.value.with(|v| unsafe { (*v).assume_init_ref() })
    }
    unsafe fn get_unchecked_mut(&mut self) -> &mut T {
        self.value.with_mut(|v| unsafe { (*v).assume_init_mut() })
    }
    unsafe fn read_unchecked(&mut self) -> T {
        self.value.with(|v| unsafe { (*v).assume_init_read() })
    }
    fn get(&self, index: buckets::Index<BUCKETS>) -> Option<(KeyData, &T)> {
        // `Acquire` is necessary because we read the value after.
        let version = self.version.load(atomic::Ordering::Acquire);
        let key = KeyData::pack_occupied(index, version)?;
        Some((key, unsafe { self.get_unchecked() }))
    }
    fn get_mut(&mut self, index: buckets::Index<BUCKETS>) -> Option<(KeyData, &mut T)> {
        let version = self.version.read_mut();
        let key = KeyData::pack_occupied(index, version)?;
        Some((key, unsafe { self.get_unchecked_mut() }))
    }
    fn into_inner(mut self, index: buckets::Index<BUCKETS>) -> Option<(KeyData, T)> {
        let version = self.version.read_mut();
        let key = KeyData::pack_occupied(index, version)?;
        let mut this = ManuallyDrop::new(self);
        Some((key, unsafe { this.read_unchecked() }))
    }
}

impl<T> Drop for Entry<T> {
    fn drop(&mut self) {
        if KeyData::version_is_occupied(self.version.read_mut()) {
            unsafe { self.value.with_mut(|v| (*v).assume_init_drop()) }
        }
    }
}

/// [`NextFree`], but packed into a `u32`.
#[derive(Clone, Copy)]
#[repr(transparent)]
struct NextFreePacked(u32);

impl NextFreePacked {
    fn new(next_free: NextFree) -> Self {
        Self(match next_free {
            NextFree::AllRemainingEntries => 0,
            NextFree::None => 1,
            // This works since `buckets::Index::into_raw` guarantees we never get a value equal to
            // one.
            NextFree::At(index) => buckets_index_as_u32(index).get(),
        })
    }
    fn get(self) -> NextFree {
        match self.0 {
            0 => NextFree::AllRemainingEntries,
            1 => NextFree::None,
            _ => NextFree::At(unsafe { buckets_index_from_u32(self.0) }),
        }
    }
}

impl<'map, K: Key, V> IntoIterator for &'map SlotMap<K, V> {
    type Item = (K, &'map V);
    type IntoIter = Iter<'map, K, V>;
    fn into_iter(self) -> Self::IntoIter {
        Iter {
            entries: self.buckets.iter(),
            min_len: self.len(),
            _key: PhantomData,
        }
    }
}

/// Iterator returned by [`SlotMap::iter`].
#[must_use]
pub struct Iter<'map, K: Key, V> {
    entries: buckets::Iter<'map, Entry<V>, BUCKETS>,
    min_len: usize,
    _key: PhantomData<fn() -> K>,
}

impl<'map, K: Key, V> Iterator for Iter<'map, K, V> {
    type Item = (K, &'map V);
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let (index, entry) = self.entries.next()?;
            if let Some((key, value)) = entry.get(index) {
                self.min_len = self.min_len.saturating_sub(1);
                return Some((K::from(key), value));
            }
        }
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.min_len, None)
    }
}

impl<K: Key, V> FusedIterator for Iter<'_, K, V> {}

impl<K: Key, V> Clone for Iter<'_, K, V> {
    fn clone(&self) -> Self {
        Self {
            entries: self.entries.clone(),
            min_len: self.min_len,
            _key: PhantomData,
        }
    }
}

impl<'map, K: Key, V> IntoIterator for &'map mut SlotMap<K, V> {
    type Item = (K, &'map mut V);
    type IntoIter = IterMut<'map, K, V>;
    fn into_iter(self) -> Self::IntoIter {
        IterMut {
            entries: self.buckets.iter_mut(),
            len: self.meta.len.read_mut() as usize,
            _key: PhantomData,
        }
    }
}

/// Iterator returned by [`SlotMap::iter_mut`].
#[must_use]
pub struct IterMut<'map, K: Key, V> {
    entries: buckets::IterMut<'map, Entry<V>, BUCKETS>,
    len: usize,
    _key: PhantomData<fn() -> K>,
}

impl<'map, K: Key, V> Iterator for IterMut<'map, K, V> {
    type Item = (K, &'map mut V);
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let (index, entry) = self.entries.next()?;
            if let Some((key, value)) = entry.get_mut(index) {
                self.len -= 1;
                return Some((K::from(key), value));
            }
        }
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len, Some(self.len))
    }
}

impl<K: Key, V> ExactSizeIterator for IterMut<'_, K, V> {
    fn len(&self) -> usize {
        self.len
    }
}

impl<K: Key, V> FusedIterator for IterMut<'_, K, V> {}

impl<K: Key, V> IntoIterator for SlotMap<K, V> {
    type Item = (K, V);
    type IntoIter = IntoIter<K, V>;
    fn into_iter(mut self) -> Self::IntoIter {
        IntoIter {
            entries: self.buckets.into_iter(),
            len: self.meta.len.read_mut() as usize,
            _key: PhantomData,
        }
    }
}

/// Iterator returned by [`SlotMap::into_iter`].
#[must_use]
pub struct IntoIter<K, V> {
    entries: buckets::IntoIter<Entry<V>, BUCKETS>,
    len: usize,
    _key: PhantomData<fn() -> K>,
}

impl<K: Key, V> Iterator for IntoIter<K, V> {
    type Item = (K, V);
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let (index, entry) = self.entries.next()?;
            if let Some((key, value)) = entry.into_inner(index) {
                self.len -= 1;
                return Some((K::from(key), value));
            }
        }
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len, Some(self.len))
    }
}

impl<K: Key, V> ExactSizeIterator for IntoIter<K, V> {
    fn len(&self) -> usize {
        self.len
    }
}

impl<K: Key, V> FusedIterator for IntoIter<K, V> {}

/// Iterator returned by [`SlotMap::drain`].
#[must_use = "To remove all the elements from a `SlotMap`, use `.clear()`."]
pub struct Drain<'map, K: Key, V> {
    entries: buckets::IterMut<'map, Entry<V>, BUCKETS>,
    meta: &'map mut Meta,
    _key: PhantomData<fn() -> K>,
}

impl<K: Key, V> Iterator for Drain<'_, K, V> {
    type Item = (K, V);
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let (index, entry) = self.entries.next()?;
            if let Some((key, _)) = entry.get_mut(index) {
                // Safety: We just ensured that `entry` is initialized.
                let value = unsafe { <SlotMap<K, V>>::remove_inner(self.meta, index, entry) };
                return Some((K::from(key), value));
            }
        }
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }
}

impl<K: Key, V> ExactSizeIterator for Drain<'_, K, V> {
    fn len(&self) -> usize {
        // Safety: `len` cannot be atomically stored to while a `Drain` exists.
        (unsafe { self.meta.len.unsync_load() }) as usize
    }
}

impl<K: Key, V> FusedIterator for Drain<'_, K, V> {}

impl<K: Key, V> Drop for Drain<'_, K, V> {
    fn drop(&mut self) {
        // This will cause the free list and length to be updated every time. In some sense, this
        // is unnecessary extra work, but it also means that if any destructor panics we don't end
        // up with a bunch of dud slots.
        self.for_each(drop);
    }
}

/// Iterator returned by [`SlotMap::keys`].
pub struct Keys<'map, K: Key, V>(Iter<'map, K, V>);

impl<K: Key, V> Iterator for Keys<'_, K, V> {
    type Item = K;
    fn next(&mut self) -> Option<Self::Item> {
        Some(self.0.next()?.0)
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}

impl<K: Key, V> FusedIterator for Keys<'_, K, V> {}

impl<K: Key, V> Clone for Keys<'_, K, V> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

/// Iterator returned by [`SlotMap::values`].
pub struct Values<'map, K: Key, V>(Iter<'map, K, V>);

impl<'map, K: Key, V> Iterator for Values<'map, K, V> {
    type Item = &'map V;
    fn next(&mut self) -> Option<Self::Item> {
        Some(self.0.next()?.1)
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}

impl<K: Key, V> FusedIterator for Values<'_, K, V> {}

impl<K: Key, V> Clone for Values<'_, K, V> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

/// Iterator returned by [`SlotMap::values_mut`].
pub struct ValuesMut<'map, K: Key, V>(IterMut<'map, K, V>);

impl<'map, K: Key, V> Iterator for ValuesMut<'map, K, V> {
    type Item = &'map mut V;
    fn next(&mut self) -> Option<Self::Item> {
        Some(self.0.next()?.1)
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}

impl<K: Key, V> ExactSizeIterator for ValuesMut<'_, K, V> {
    fn len(&self) -> usize {
        self.0.len()
    }
}

impl<K: Key, V> FusedIterator for ValuesMut<'_, K, V> {}

#[cfg(feature = "serde")]
impl<K: Key + serde::Serialize, V: serde::Serialize> serde::Serialize for SlotMap<K, V> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        // Note that we include empty slots in the serialization. This is for two reasons:
        // 1. To preserve their versions, so that keys are not reused even through serde.
        // 2. To prevent malicious payloads from including one entry in each bucket, causing memory
        //    exhaustion in the parser from a very short serialized representation.
        serializer.collect_seq(self.buckets.iter().map(
            |(index, entry)| -> (usize, u32, Option<&V>) {
                // `Acquire` is necessary because we read the value after.
                let version = entry.version.load(atomic::Ordering::Acquire);
                let value =
                    KeyData::version_is_occupied(version).then(|| unsafe { entry.get_unchecked() });
                (index.into_raw().get(), version, value)
            },
        ))
    }
}

#[cfg(feature = "serde")]
impl<'de, K: Key, V> serde::Deserialize<'de> for SlotMap<K, V>
where
    K: serde::Deserialize<'de>,
    V: serde::Deserialize<'de>,
{
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct Visitor<K, V>(core::marker::PhantomData<(K, V)>);
        impl<'de, K: Key, V> serde::de::Visitor<'de> for Visitor<K, V>
        where
            K: serde::Deserialize<'de>,
            V: serde::Deserialize<'de>,
        {
            type Value = SlotMap<K, V>;
            fn expecting(&self, f: &mut Formatter<'_>) -> fmt::Result {
                f.write_str("a slot map")
            }
            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: serde::de::SeqAccess<'de>,
            {
                // The expected index of the next entry in the map. `None` if the slot map is full.
                let mut expected_next = Some(buckets::Index::new(0).unwrap());

                let iter = core::iter::from_fn(|| {
                    seq.next_element::<(usize, u32, Option<V>)>().transpose()
                });

                let map = SlotMap::from_all_entries_raw(iter.map(|res| {
                    let (index, version, value) = res?;

                    let expected = expected_next
                        .ok_or_else(|| serde::de::Error::custom("map is too large"))?;

                    let index = buckets::Index::from_raw_checked(index)
                        .ok_or_else(|| serde::de::Error::custom("invalid index"))?;

                    if index != expected
                        // Since buckets can technically not be contiguous, we allow
                        // skipping entire buckets.
                        && !(index.is_first_in_bucket()
                            && expected.is_first_in_bucket()
                            && expected < index)
                    {
                        return Err(serde::de::Error::custom("indices not contiguous"));
                    }

                    let occupied = KeyData::version_is_occupied(version);

                    if occupied != value.is_some() {
                        return Err(serde::de::Error::custom("inconsistent version"));
                    }

                    expected_next = buckets::Index::from_raw_checked(index.into_raw().get() + 1);

                    Ok((index, version, value))
                }))?;

                if expected_next.is_some_and(|index| !index.is_first_in_bucket()) {
                    return Err(serde::de::Error::custom("missing entries from map"));
                }

                Ok(map)
            }
        }
        deserializer.deserialize_seq(Visitor(core::marker::PhantomData))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::borrow::ToOwned;
    use std::boxed::Box;
    use std::panic::catch_unwind;
    use std::string::String;
    use std::string::ToString;
    use std::sync::Barrier;
    use std::thread;
    use std::vec::Vec;

    #[test]
    fn insert_with_updater_panics() {
        let map = SlotMap::new();

        let _ = catch_unwind(|| map.insert_with_updater("hello".to_owned(), |_, _| panic!()));
        assert_eq!(map.iter().next().unwrap().1, "hello");
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn slot_reuse() {
        let mut map = SlotMap::new();

        for i in 0..=7 {
            map.insert(Box::new(i));
        }
        map.retain(|_, v| **v % 3 == 0);

        let (key, _) = map.insert(Box::new(37));
        map.insert(Box::new(29));
        map.insert(Box::new(48));

        assert_eq!(
            map.values().map(|i| **i).collect::<Vec<_>>(),
            [0, 3, 48, 29, 6, 37],
        );

        map.insert(Box::new(49));
        map.insert(Box::new(50));
        map.remove(key);
        map.insert(Box::new(51));
        map.insert(Box::new(52));
        map.insert(Box::new(53));

        assert_eq!(
            map.values().map(|i| **i).collect::<Vec<_>>(),
            [0, 50, 49, 3, 48, 29, 6, 51, 52, 53],
        );
    }

    #[test]
    fn empty_clone() {
        let map = <SlotMap<_, String>>::new();
        assert!(map.clone().is_empty());
    }

    #[test]
    fn cloning() {
        let mut map = SlotMap::new();

        for i in 0..=7 {
            map.insert((i, i.to_string()));
        }
        map.retain(|_, &mut (v, _)| v % 3 == 0);

        let cloned = map.clone();
        assert_eq!(
            cloned.values().map(|(i, _)| *i).collect::<Vec<_>>(),
            [0, 3, 6]
        );
        assert_eq!(cloned.len(), 3);

        // We have to insert quite a few values to allocate a new bucket and thus start reusing
        // slots.
        for i in 0..31 {
            cloned.insert((48 + i, String::new()));
        }
        assert_eq!(cloned.len(), 3 + 31);
        assert_eq!(
            cloned.values().map(|(i, _)| *i).collect::<Vec<_>>(),
            [
                0, 76, 75, 3, 74, 73, 6, 72, 71, 70, 69, 68, 67, 66, 65, 64, 63, 62, 61, 60, 59,
                58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 77, 78
            ],
        );
    }

    #[test]
    fn stress() {
        let map = SlotMap::new();
        let barrier = Barrier::new(5);
        let n = if cfg!(miri) { 100 } else { 1000 };

        thread::scope(|s| {
            for t in 0..4 {
                let (barrier, map) = (&barrier, &map);
                s.spawn(move || {
                    barrier.wait();
                    for i in 0..n {
                        map.insert(t * n + i);
                    }
                });
            }
            s.spawn(|| {
                barrier.wait();
                for (key, &x) in map.iter() {
                    assert!(x < 4 * n);
                    assert_eq!(map[key], x);
                }
            });
        });

        assert_eq!(map.len(), 4 * n);
        let mut sorted = map.values().copied().collect::<Vec<_>>();
        sorted.sort();
        assert_eq!(sorted, (0..4 * n).collect::<Vec<_>>());
    }

    #[test]
    fn stress_reuse() {
        let mut map = SlotMap::new();
        let n = if cfg!(miri) { 100 } else { 1000 };
        for _ in 0..4 * n {
            map.insert(0);
        }
        map.clear();
        let barrier = Barrier::new(5);

        thread::scope(|s| {
            for t in 0..4 {
                let (barrier, map) = (&barrier, &map);
                s.spawn(move || {
                    barrier.wait();
                    for i in 0..n {
                        map.insert(t * n + i);
                    }
                });
            }
            s.spawn(|| {
                barrier.wait();
                for (key, &x) in map.iter() {
                    assert!(x < 4 * n);
                    assert_eq!(map[key], x);
                }
            });
        });

        assert_eq!(map.len(), 4 * n);
        let mut sorted = map.values().copied().collect::<Vec<_>>();
        sorted.sort();
        assert_eq!(sorted, (0..4 * n).collect::<Vec<_>>());
    }

    #[test]
    #[cfg(feature = "serde")]
    fn serde() {
        let mut map = SlotMap::new();
        for i in 0..=7 {
            map.insert(i);
        }
        map.retain(|_, &mut v| v % 3 == 0);
        map.insert(12);

        let json = serde_json::to_string(&map).unwrap();
        assert_eq!(
            json,
            "[[32,1,0],[33,2,null],[34,2,null],[35,1,3],[36,2,null],[37,2,null],[38,1,6],[39,3,12],[40,0,null],[41,0,null],[42,0,null],[43,0,null],[44,0,null],[45,0,null],[46,0,null],[47,0,null],[48,0,null],[49,0,null],[50,0,null],[51,0,null],[52,0,null],[53,0,null],[54,0,null],[55,0,null],[56,0,null],[57,0,null],[58,0,null],[59,0,null],[60,0,null],[61,0,null],[62,0,null],[63,0,null]]",
        );

        let new_map = serde_json::from_str::<SlotMap<DefaultKey, i32>>(&json).unwrap();
        assert_eq!(serde_json::to_string(&new_map).unwrap(), json);
    }

    #[test]
    #[cfg(feature = "serde")]
    fn serde_empty() {
        let map = <SlotMap<DefaultKey, u32>>::new();
        assert_eq!(serde_json::to_string(&map).unwrap(), "[]");

        let new_map = serde_json::from_str::<SlotMap<DefaultKey, u32>>("[]").unwrap();
        assert!(new_map.is_empty());
        assert_eq!(serde_json::to_string(&new_map).unwrap(), "[]");
    }

    #[test]
    #[cfg(feature = "serde")]
    fn serde_failures() {
        fn msg(json: &str) -> String {
            serde_json::from_str::<SlotMap<DefaultKey, i32>>(json)
                .unwrap_err()
                .to_string()
                .split_once(" at ")
                .unwrap()
                .0
                .to_owned()
        }
        assert_eq!(msg("[[31,0,null]]"), "invalid index");
        assert_eq!(msg("[[32,0,null]]"), "missing entries from map");
        assert_eq!(msg("[[32,1,null]]"), "inconsistent version");
        assert_eq!(msg("[[32,0,5]]"), "inconsistent version");
        assert_eq!(msg("[[32,0,null],[34,0,null]]"), "indices not contiguous");
    }
}
