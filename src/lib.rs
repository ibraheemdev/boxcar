#![doc = include_str!("../README.md")]
#![deny(unsafe_op_in_unsafe_fn)]

mod raw;

use std::fmt;
use std::ops::Index;

/// Creates a [`Vec`] containing the given elements.
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

/// A concurrent, append-only vector.
///
/// See [the crate documentation](crate) for details.
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
        self.raw.reserve(additional)
    }

    /// Appends an element to the back of the vector,
    /// returning the index it was inserted into.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut vec = boxcar::vec![1, 2];
    /// assert_eq!(vec.push(3), 2);
    /// assert_eq!(vec, [1, 2, 3]);
    /// ```
    pub fn push(&self, value: T) -> usize {
        self.raw.push(value)
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
        self.raw.len()
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
        self.raw.get(index)
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

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        self.raw.next_shared(self.vec)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.vec.len() - self.raw.yielded(), None)
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