pub use self::inner::*;

pub trait AtomicMut<T> {
    #[inline(always)]
    fn read_mut(&mut self) -> T {
        unsafe { self.unsync_load() }
    }
    unsafe fn unsync_load(&self) -> T;
    fn write_mut(&mut self, value: T);
}

#[cfg(loom)]
mod inner {
    pub use loom::{cell, sync::atomic};

    impl super::AtomicMut<bool> for atomic::AtomicBool {
        unsafe fn unsync_load(&self) -> bool {
            unsafe { self.unsync_load() }
        }

        fn write_mut(&mut self, value: bool) {
            self.store(value, atomic::Ordering::Relaxed)
        }
    }

    impl super::AtomicMut<u32> for atomic::AtomicU32 {
        unsafe fn unsync_load(&self) -> u32 {
            unsafe { self.unsync_load() }
        }

        fn write_mut(&mut self, value: u32) {
            self.store(value, atomic::Ordering::Relaxed)
        }
    }

    impl super::AtomicMut<usize> for atomic::AtomicUsize {
        unsafe fn unsync_load(&self) -> usize {
            unsafe { self.unsync_load() }
        }

        fn write_mut(&mut self, value: usize) {
            self.store(value, atomic::Ordering::Relaxed)
        }
    }

    impl<T> super::AtomicMut<*mut T> for atomic::AtomicPtr<T> {
        unsafe fn unsync_load(&self) -> *mut T {
            unsafe { self.unsync_load() }
        }

        fn write_mut(&mut self, value: *mut T) {
            self.store(value, atomic::Ordering::Relaxed)
        }
    }
}

#[cfg(not(loom))]
mod inner {
    pub use core::sync::atomic;

    impl super::AtomicMut<bool> for atomic::AtomicBool {
        #[inline(always)]
        unsafe fn unsync_load(&self) -> bool {
            unsafe { *self.as_ptr() }
        }

        #[inline(always)]
        fn write_mut(&mut self, value: bool) {
            *self.get_mut() = value;
        }
    }

    impl super::AtomicMut<u32> for atomic::AtomicU32 {
        #[inline(always)]
        unsafe fn unsync_load(&self) -> u32 {
            unsafe { *self.as_ptr() }
        }

        #[inline(always)]
        fn write_mut(&mut self, value: u32) {
            *self.get_mut() = value;
        }
    }

    impl super::AtomicMut<usize> for atomic::AtomicUsize {
        #[inline(always)]
        unsafe fn unsync_load(&self) -> usize {
            unsafe { *self.as_ptr() }
        }

        #[inline(always)]
        fn write_mut(&mut self, value: usize) {
            *self.get_mut() = value;
        }
    }

    impl<T> super::AtomicMut<*mut T> for atomic::AtomicPtr<T> {
        #[inline(always)]
        unsafe fn unsync_load(&self) -> *mut T {
            unsafe { *self.as_ptr() }
        }

        #[inline(always)]
        fn write_mut(&mut self, value: *mut T) {
            *self.get_mut() = value;
        }
    }

    pub mod cell {
        pub struct UnsafeCell<T>(core::cell::UnsafeCell<T>);

        impl<T> UnsafeCell<T> {
            #[inline(always)]
            pub fn new(val: T) -> Self {
                Self(core::cell::UnsafeCell::new(val))
            }

            #[inline(always)]
            pub fn into_inner(self) -> T {
                self.0.into_inner()
            }

            #[inline(always)]
            pub fn with<F, R>(&self, f: F) -> R
            where
                F: FnOnce(*const T) -> R,
            {
                f(self.0.get())
            }

            #[inline(always)]
            pub fn with_mut<F, R>(&self, f: F) -> R
            where
                F: FnOnce(*mut T) -> R,
            {
                f(self.0.get())
            }
        }
    }
}
