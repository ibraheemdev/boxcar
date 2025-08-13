//! A concurrent, append-only [slot map](https://docs.rs/slotmap).
//!
//! See [`SlotMap`] for details.

use crate::buckets;
use core::fmt;
use core::fmt::Debug;
use core::fmt::Formatter;
use core::hash::Hash;
use core::hash::Hasher;
use core::num::NonZeroU32;

mod basic;
pub use basic::*;

// The below code is largely adapted from:
// <https://github.com/orlp/slotmap/blob/c905b6ced490551476cb7c37778eb8128bdea7ba/src/lib.rs>

/// Key used to access stored values in a slot map.
///
/// Do not use a key from one slot map in another. The behavior is safe but
/// non-sensical (and might panic in case of out-of-bounds).
///
/// To prevent this, it is suggested to have a unique key type for each slot
/// map. You can create new key types using [`new_key_type!`], which makes a
/// new type identical to [`DefaultKey`], just with a different name.
///
/// # Safety
///
/// This trait is intended to be a thin wrapper around [`KeyData`], and all
/// methods must behave exactly as if we're operating on a [`KeyData`] directly.
/// The internal unsafe code relies on this, therefore this trait is `unsafe` to
/// implement. It is strongly suggested to simply use [`new_key_type!`] instead
/// of implementing this trait yourself.
pub unsafe trait Key:
    From<KeyData>
    + Copy
    + Clone
    + Default
    + Eq
    + PartialEq
    + Ord
    + PartialOrd
    + core::hash::Hash
    + core::fmt::Debug
{
    /// Creates a new key that is always invalid and distinct from any non-null
    /// key. A null key can only be created through this method (or default
    /// initialization of keys made with [`new_key_type!`], which calls this
    /// method).
    ///
    /// A null key is always invalid, but an invalid key (that is, a key that
    /// has been removed from the slot map) does not become a null key. A null
    /// is safe to use with any safe method of any slot map instance.
    ///
    /// # Examples
    ///
    /// ```
    /// # use boxcar::slotmap::*;
    /// let mut sm = SlotMap::new();
    /// let (k, _) = sm.insert(42);
    /// let nk = DefaultKey::null();
    /// assert!(nk.is_null());
    /// assert!(k != nk);
    /// assert_eq!(sm.get(nk), None);
    /// ```
    fn null() -> Self {
        KeyData::default().into()
    }

    /// Checks if a key is null. There is only a single null key, that is
    /// `a.is_null() && b.is_null()` implies `a == b`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use boxcar::slotmap::*;
    /// new_key_type! { struct MyKey; }
    /// let a = MyKey::null();
    /// let b = MyKey::default();
    /// assert_eq!(a, b);
    /// assert!(a.is_null());
    /// ```
    fn is_null(&self) -> bool {
        self.data() == KeyData::default()
    }

    /// Gets the [`KeyData`] stored in this key.
    ///
    /// # Examples
    ///
    /// ```
    /// # use boxcar::slotmap::*;
    /// new_key_type! { struct MyKey; }
    /// let dk = DefaultKey::null();
    /// let mk = MyKey::null();
    /// assert_eq!(dk.data(), mk.data());
    /// ```
    fn data(&self) -> KeyData;
}

/// A helper macro to create new key types for use in a [`SlotMap`].
/// If you use a new key type for each slot map you create you can
/// entirely prevent using the wrong key on the wrong slot map.
///
/// The type constructed by this macro is defined exactly as [`DefaultKey`],
/// but is a distinct type for the type checker and does not implicitly convert.
///
/// # Examples
///
/// ```
/// # use boxcar::slotmap::*;
/// new_key_type! {
///     // A private key type.
///     struct RocketKey;
///
///     // A public key type with a doc comment.
///     /// Key for the user slot map.
///     pub struct UserKey;
/// }
///
/// fn main() {
///     let mut users = SlotMap::with_key();
///     let mut rockets = SlotMap::with_key();
///     let (bob, _): (UserKey, _) = users.insert("bobby");
///     let (apollo, _): (RocketKey, _) = rockets.insert("apollo");
///     // Now this is a type error because rockets.get expects an RocketKey:
///     // rockets.get(bob);
///
///     // If for some reason you do end up needing to convert (e.g. storing
///     // keys of multiple slot maps in the same data structure without
///     // boxing), you can use KeyData as an intermediate representation. This
///     // does mean that once again you are responsible for not using the wrong
///     // key on the wrong slot map.
///     let keys = vec![bob.data(), apollo.data()];
///     println!("{} likes rocket {}",
///              users[keys[0].into()], rockets[keys[1].into()]);
/// }
/// ```
#[macro_export(local_inner_macros)]
macro_rules! new_key_type {
    ( $(#[$outer:meta])* $vis:vis struct $name:ident; $($rest:tt)* ) => {
        $(#[$outer])*
        #[derive(Clone, Copy, Default, Eq, PartialEq, Ord, PartialOrd, Hash, Debug)]
        #[repr(transparent)]
        $vis struct $name($crate::slotmap::KeyData);

        impl ::core::convert::From<$crate::slotmap::KeyData> for $name {
            fn from(k: $crate::slotmap::KeyData) -> Self {
                $name(k)
            }
        }

        unsafe impl $crate::slotmap::Key for $name {
            fn data(&self) -> $crate::slotmap::KeyData {
                self.0
            }
        }

        $crate::__serialize_key!($name);

        $crate::new_key_type!($($rest)*);
    };

    () => {}
}

pub use crate::new_key_type;

new_key_type! {
    /// The default slot map key type.
    pub struct DefaultKey;
}

#[cfg(feature = "serde")]
#[doc(hidden)]
#[macro_export]
macro_rules! __serialize_key {
    ( $name:ty ) => {
        impl $crate::slotmap::__impl::Serialize for $name {
            fn serialize<S>(&self, serializer: S) -> ::core::result::Result<S::Ok, S::Error>
            where
                S: $crate::slotmap::__impl::Serializer,
            {
                $crate::slotmap::Key::data(self).serialize(serializer)
            }
        }

        impl<'de> $crate::slotmap::__impl::Deserialize<'de> for $name {
            fn deserialize<D>(deserializer: D) -> ::core::result::Result<Self, D::Error>
            where
                D: $crate::slotmap::__impl::Deserializer<'de>,
            {
                let key_data: $crate::slotmap::KeyData =
                    $crate::slotmap::__impl::Deserialize::deserialize(deserializer)?;
                ::core::result::Result::Ok(::core::convert::From::from(key_data))
            }
        }
    };
}

#[cfg(not(feature = "serde"))]
#[doc(hidden)]
#[macro_export]
macro_rules! __serialize_key {
    ( $name:ty ) => {};
}

// Used internally by the macro. Not public API.
#[doc(hidden)]
pub mod __impl {
    #[cfg(feature = "serde")]
    pub use serde::*;
}

/// The underlying data held by a [`Key`].
///
/// This implements [`Ord`] so keys can be stored in e.g. [`BTreeMap`](std::collections::BTreeMap),
/// but the order of keys is unspecified.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct KeyData {
    // Invariant: A valid `buckets::Index<BUCKETS>`.
    index: NonZeroU32,
    version: NonZeroU32,
}

impl KeyData {
    /// Returns the key data as a 64-bit integer. No guarantees about its value
    /// are made other than that passing it to [`from_ffi`](Self::from_ffi)
    /// will return a key equal to the original.
    ///
    /// With this you can easily pass slot map keys as opaque handles to foreign
    /// code. After you get them back you can confidently use them in your slot
    /// map without worrying about unsafe behavior as you would with passing and
    /// receiving back references or pointers.
    ///
    /// This is not a substitute for proper serialization; use `serde` for
    /// that. If you are not doing FFI, you almost surely do not need this
    /// function.
    #[must_use]
    pub fn as_ffi(self) -> u64 {
        (u64::from(self.index.get()) << 32) | u64::from(self.version.get())
    }

    /// If `value` is a value received from `k.as_ffi()`, returns a key equal
    /// to `k`. Otherwise the behavior is safe, unspecified, and non-panicking.
    #[must_use]
    pub fn from_ffi(value: u64) -> Self {
        let index = (value >> 32) as usize;
        let version = value as u32;
        buckets::Index::from_raw_checked(index)
            .and_then(|i| Self::pack_occupied(i, version))
            .unwrap_or_default()
    }

    fn pack_vacant(index: buckets::Index<BUCKETS>, version: u32) -> Option<Self> {
        if Self::version_is_occupied(version) {
            return None;
        }
        Some(Self {
            index: buckets_index_as_u32(index),
            version: NonZeroU32::new(Self::version_make_occupied(version)).unwrap(),
        })
    }

    fn pack_occupied(index: buckets::Index<BUCKETS>, version: u32) -> Option<Self> {
        if !Self::version_is_occupied(version) {
            return None;
        }
        Some(Self {
            index: buckets_index_as_u32(index),
            version: NonZeroU32::new(version).unwrap(),
        })
    }

    fn unpack(self) -> (buckets::Index<BUCKETS>, u32) {
        // Safety: Invariant of the type.
        let index = unsafe { buckets_index_from_u32(self.index.get()) };
        (index, self.version.get())
    }

    fn version_is_occupied(version: u32) -> bool {
        version & 1 != 0
    }
    fn version_make_occupied(version: u32) -> u32 {
        version | 1
    }
    fn version_make_vacant(version: u32) -> u32 {
        version.wrapping_add(1)
    }
}

impl Default for KeyData {
    fn default() -> Self {
        Self {
            index: buckets_index_as_u32(buckets::Index::new(0).unwrap()),
            // Since we use an even (vacant) version, this cannot compare equal to any other key.
            version: NonZeroU32::new(2).unwrap(),
        }
    }
}

impl Hash for KeyData {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Like in regular `SlotMap`, we write a single u64, which is beneficial for some hashers.
        state.write_u64(self.as_ffi());
    }
}

#[cfg(feature = "serde")]
impl serde::Serialize for KeyData {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_u64(self.as_ffi())
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for KeyData {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Ok(Self::from_ffi(u64::deserialize(deserializer)?))
    }
}

impl Debug for KeyData {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let (index, version) = self.unpack();
        write!(f, "({index:?}, {version})")
    }
}

// On 32-bit systems, we use indices up to 2³¹ since higher indices are not possible due to memory
// limitations anyway. Importantly, avoiding the top bit means we can assume `usize` does not
// overflow.
//
// On 64-bit systems, we want indices using all 32 bits to allow the full u32 range to be used.
const INDEX_BITS: u32 = if usize::BITS <= 32 {
    usize::BITS - 1
} else {
    32
};

const BUCKETS: usize = buckets::buckets_for_index_bits(INDEX_BITS);

/// The first out-of-bounds index. Might overflow a `u32`, but won't overflow a `usize`.
const FIRST_OOB_INDEX: usize = 1 << INDEX_BITS;

const ZERO_INDEX: buckets::Index<BUCKETS> = match buckets::Index::new(0) {
    Some(i) => i,
    None => unreachable!(),
};

fn buckets_index_as_u32(index: buckets::Index<BUCKETS>) -> NonZeroU32 {
    // Safety: `buckets_for_index_bits` is used with a value that is at most 32, therefore
    // indices fit in 32 bits, therefore zeroness is preserved.
    unsafe { NonZeroU32::new_unchecked(index.into_raw().get() as u32) }
}

unsafe fn buckets_index_from_u32(inner: u32) -> buckets::Index<BUCKETS> {
    // We can cast to usize because `buckets_for_index_bits` is used with a value that is at most
    // `usize::BITS`, therefore indices fit in a usize.
    unsafe { buckets::Index::from_raw_unchecked(inner as usize) }
}
