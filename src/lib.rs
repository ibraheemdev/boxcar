#![doc = include_str!("../README.md")]
#![deny(unsafe_op_in_unsafe_fn)]
#![no_std]

extern crate alloc;

mod raw;

use core::fmt;
use core::ops::Index;

/// Creates a [`Vec`] containing the given elements.
///
/// `vec!` allows `Vec`s to be defined with the same syntax as array expressions.
/// There are two forms of this macro:
///
/// - Create a [`Vec`] containing a given list of elements:
///
/// ```
/// let vec = vec![1, 2, 3];
/// assert_eq!(vec[0], 1);
/// assert_eq!(vec[1], 2);
/// assert_eq!(vec[2], 3);
/// ```
///
/// - Create a [`Vec`] from a given element and size:
///
/// ```
/// let vec = vec![1; 3];
/// assert_eq!(vec, [1, 1, 1]);
/// ```
#[macro_export]
macro_rules! vec {
    () => {
        $crate::Vec::new()
    };
    ($elem:expr; $n:expr) => {{
        let n = $n;
        let mut vec = $crate::Vec::with_capacity(n);
        let iter = ::core::iter::Iterator::take(::core::iter::repeat($elem), n);
        ::core::iter::Extend::extend(&mut vec, iter);
        vec
    }};
    ($($x:expr),+ $(,)?) => (
        <$crate::Vec<_> as ::core::iter::FromIterator<_>>::from_iter([$($x),+])
    );
}

/// A concurrent, append-only vector.
///
/// See [the crate documentation](crate) for details.
///
/// # Notes
///
/// The bucket array is stored inline, meaning that the
/// `Vec<T>` is quite large. It is expected that you
/// store it behind an [`Arc`](std::sync::Arc) or similar.
pub struct Vec<T> {
    raw: raw::Vec<T>,
}

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
    /// let vec: boxcar::Vec<i32> = boxcar::Vec::new();
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
    /// let vec = boxcar::Vec::with_capacity(10);
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
        Vec {
            raw: raw::Vec::with_capacity(capacity),
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
    /// let vec = boxcar::Vec::new();
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
        self.raw.reserve(additional)
    }

    /// Appends an element to the back of the vector,
    /// returning the index it was inserted into.
    ///
    /// # Examples
    ///
    /// ```
    /// let vec = boxcar::vec![1, 2];
    /// assert_eq!(vec.push(3), 2);
    /// assert_eq!(vec, [1, 2, 3]);
    /// ```
    pub fn push(&self, value: T) -> usize {
        self.raw.push(value)
    }

    /// Returns the number of elements in the vector.
    ///
    /// Note that due to concurrent writes, it is not guaranteed
    /// that all elements `0..vec.count()` are initialized.
    ///
    /// # Examples
    ///
    /// ```
    /// let vec = boxcar::Vec::new();
    /// assert_eq!(vec.count(), 0);
    /// vec.push(1);
    /// vec.push(2);
    /// assert_eq!(vec.count(), 2);
    /// ```
    #[inline]
    pub fn count(&self) -> usize {
        self.raw.count()
    }

    /// Returns `true` if the vector contains no elements.
    ///
    /// # Examples
    ///
    /// ```
    /// let vec = boxcar::Vec::new();
    /// assert!(vec.is_empty());
    ///
    /// vec.push(1);
    /// assert!(!vec.is_empty());
    /// ```
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.count() == 0
    }

    /// Returns a reference to the element at the given index.
    ///
    /// # Examples
    ///
    /// ```
    /// let vec = boxcar::vec![10, 40, 30];
    /// assert_eq!(Some(&40), vec.get(1));
    /// assert_eq!(None, vec.get(3));
    /// ```
    pub fn get(&self, index: usize) -> Option<&T> {
        self.raw.get(index)
    }

    /// Returns a mutable reference to the element at the given index.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut vec = boxcar::vec![10, 40, 30];
    /// assert_eq!(Some(&mut 40), vec.get_mut(1));
    /// assert_eq!(None, vec.get_mut(3));
    /// ```
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        self.raw.get_mut(index)
    }

    /// Returns a reference to an element, without doing bounds
    /// checking or verifying that the element is fully initialized.
    ///
    /// For a safe alternative see [`get`](Vec::get).
    ///
    /// # Safety
    ///
    /// Calling this method with an out-of-bounds index, or for an element that
    /// is being concurrently initialized is **undefined behavior**, even if
    /// the resulting reference is not used.
    ///
    /// # Examples
    ///
    /// ```
    /// let vec = boxcar::vec![1, 2, 4];
    ///
    /// unsafe {
    ///     assert_eq!(vec.get_unchecked(1), &2);
    /// }
    /// ```
    pub unsafe fn get_unchecked(&self, index: usize) -> &T {
        // SAFETY: guaranteed by caller
        unsafe { self.raw.get_unchecked(index) }
    }

    /// Returns a mutable reference to an element, without doing bounds
    /// checking or verifying that the element is fully initialized.
    ///
    /// For a safe alternative see [`get`](Vec::get).
    ///
    /// # Safety
    ///
    /// Calling this method with an out-of-bounds index is **undefined
    /// behavior**, even if the resulting reference is not used.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut vec = boxcar::vec![1, 2, 4];
    ///
    /// unsafe {
    ///     assert_eq!(vec.get_unchecked_mut(1), &mut 2);
    /// }
    /// ```
    pub unsafe fn get_unchecked_mut(&mut self, index: usize) -> &mut T {
        // SAFETY: guaranteed by caller
        unsafe { self.raw.get_unchecked_mut(index) }
    }

    /// Returns an iterator over the vector.
    ///
    /// Values are yielded in the form `(index, value)`. The vector may
    /// have in-progress concurrent writes that create gaps, so `index`
    /// may not be strictly sequential.
    ///
    /// # Examples
    ///
    /// ```
    /// let vec = boxcar::vec![1, 2, 4];
    /// let mut iterator = vec.iter();
    ///
    /// assert_eq!(iterator.next(), Some((0, &1)));
    /// assert_eq!(iterator.next(), Some((1, &2)));
    /// assert_eq!(iterator.next(), Some((2, &4)));
    /// assert_eq!(iterator.next(), None);
    /// ```
    pub fn iter(&self) -> Iter<'_, T> {
        Iter {
            vec: &self.raw,
            raw: self.raw.iter(),
        }
    }
}

impl<T> Index<usize> for Vec<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.raw[index]
    }
}

impl<T> IntoIterator for Vec<T> {
    type Item = T;
    type IntoIter = IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        IntoIter {
            raw: self.raw.iter(),
            vec: self.raw,
        }
    }
}

impl<'a, T> IntoIterator for &'a Vec<T> {
    type Item = (usize, &'a T);
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
    vec: raw::Vec<T>,
    raw: raw::Iter,
}

impl<T> Iterator for IntoIter<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        unsafe { self.raw.next_owned(&mut self.vec) }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.raw.yielded(), Some(self.raw.yielded()))
    }
}

/// An iterator over the elements of a [`Vec<T>`].
///
/// See [`Vec::iter`] for details.
pub struct Iter<'a, T> {
    vec: &'a raw::Vec<T>,
    raw: raw::Iter,
}

impl<'a, T> Clone for Iter<'a, T> {
    fn clone(&self) -> Iter<'a, T> {
        Iter {
            vec: self.vec,
            raw: self.raw.clone(),
        }
    }
}
impl<'a, T> Iterator for Iter<'a, T> {
    type Item = (usize, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        self.raw.next_shared(self.vec)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.vec.count() - self.raw.yielded(), None)
    }
}
impl<'a, T: fmt::Debug> fmt::Debug for Iter<'a, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        struct Contents<'a, T>(&'a T);
        impl<T> fmt::Debug for Contents<'_, T>
        where
            T: Iterator + Clone,
            T::Item: fmt::Debug,
        {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.debug_list().entries(self.0.clone()).finish()
            }
        }
        f.debug_tuple("Iter").field(&Contents(self)).finish()
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
        self.reserve(lower);

        for value in iter {
            self.push(value);
        }
    }
}

impl<T: Clone> Clone for Vec<T> {
    fn clone(&self) -> Vec<T> {
        self.iter().map(|(_, x)| x).cloned().collect()
    }
}

impl<T: fmt::Debug> fmt::Debug for Vec<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl<T: PartialEq> PartialEq for Vec<T> {
    fn eq(&self, other: &Self) -> bool {
        if self.count() != other.count() {
            return false;
        }

        // ensure indexes are checked along with values to handle gaps in the vector
        for (index, value) in self.iter() {
            if other.get(index) != Some(value) {
                return false;
            }
        }

        true
    }
}

impl<A, T> PartialEq<A> for Vec<T>
where
    A: AsRef<[T]>,
    T: PartialEq,
{
    fn eq(&self, other: &A) -> bool {
        let other = other.as_ref();

        if self.count() != other.len() {
            return false;
        }

        // ensure indexes are checked along with values to handle gaps in the vector
        for (index, value) in self.iter() {
            if other.get(index) != Some(value) {
                return false;
            }
        }

        true
    }
}

impl<T: Eq> Eq for Vec<T> {}
