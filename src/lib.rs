#![doc = include_str!("../README.md")]
#![deny(unsafe_op_in_unsafe_fn)]
#![allow(clippy::needless_doctest_main)]
#![no_std]

extern crate alloc;

#[cfg(test)]
extern crate std;

mod loom;

pub mod buckets;
pub mod vec;

#[doc(inline)]
pub use buckets::Buckets;

#[doc(inline)]
pub use vec::Vec;

// Reexports for backward compatibility.
#[doc(hidden)]
pub use vec::IntoIter;
#[doc(hidden)]
pub use vec::Iter;
