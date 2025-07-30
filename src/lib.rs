#![doc = include_str!("../README.md")]
#![deny(unsafe_op_in_unsafe_fn)]
#![allow(clippy::needless_doctest_main)]
#![no_std]

extern crate alloc;

#[cfg(test)]
extern crate std;

mod loom;

pub mod buckets;
pub use buckets::Buckets;

pub mod vec;
pub use vec::Vec;

// Reexports for backward compatibility.
#[doc(hidden)]
pub use vec::IntoIter;
#[doc(hidden)]
pub use vec::Iter;
