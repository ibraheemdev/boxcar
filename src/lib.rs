#![doc = include_str!("../README.md")]
#![deny(unsafe_op_in_unsafe_fn)]
#![allow(clippy::needless_doctest_main)]
#![no_std]

extern crate alloc;

mod loom;

pub mod vec;
pub use vec::Vec;
