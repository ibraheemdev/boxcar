use std::sync::atomic::{AtomicUsize, Ordering};
use std::{sync::Barrier, thread};

#[test]
fn simple() {
    let vec = boxcar::vec![0, 1, 2];
    assert_eq!(vec[0], 0);
    assert_eq!(vec[1], 1);
    assert_eq!(vec[2], 2);

    for x in 3..1000 {
        let i = vec.push(x);
        assert_eq!(vec[i], x);
    }

    for i in 0..1000 {
        assert_eq!(vec[i], i);
    }

    for (i, &x) in vec.iter() {
        assert_eq!(i, x);
    }

    for (i, x) in vec.into_iter().enumerate() {
        assert_eq!(i, x);
    }
}

#[test]
fn simple_boxed() {
    let vec = boxcar::vec![Box::new(0), Box::new(1), Box::new(2)];
    assert_eq!(vec[0], Box::new(0));
    assert_eq!(vec[1], Box::new(1));
    assert_eq!(vec[2], Box::new(2));

    for x in 3..1000 {
        let i = vec.push(Box::new(x));
        assert_eq!(*vec[i], x);
    }

    for i in 0..1000 {
        assert_eq!(*vec[i], i);
    }

    for (i, x) in vec.iter() {
        assert_eq!(i, **x);
    }

    for (i, x) in vec.into_iter().enumerate() {
        assert_eq!(i, *x);
    }
}

#[test]
fn clear() {
    struct T<'a>(&'a AtomicUsize);
    impl Drop for T<'_> {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
    }

    let drops = AtomicUsize::new(0);

    let mut vec = boxcar::Vec::new();
    vec.push(T(&drops));
    vec.push(T(&drops));

    let first_ptr: *const _ = vec.iter().next().unwrap().1 as _;

    vec.clear();
    assert_eq!(vec.count(), 0);
    assert_eq!(vec.iter().count(), 0);
    assert_eq!(drops.swap(0, Ordering::Relaxed), 2);

    vec.clear();
    assert_eq!(vec.count(), 0);
    assert_eq!(vec.iter().count(), 0);
    assert_eq!(drops.load(Ordering::Relaxed), 0);

    vec.push(T(&drops));
    let ptr: *const _ = vec.iter().next().unwrap().1 as _;
    assert_eq!(ptr, first_ptr);

    drop(vec);
    assert_eq!(drops.load(Ordering::Relaxed), 1);
}

#[test]
fn fused_iterator() {
    let vec = boxcar::vec![0, 1, 2];
    assert_eq!(vec.iter().collect::<Vec<_>>(), [(0, &0), (1, &1), (2, &2)]);

    let iter = vec.iter();
    vec.push(3);
    assert_eq!(iter.collect::<Vec<_>>(), [(0, &0), (1, &1), (2, &2)]);
}

#[test]
fn stress() {
    let vec = boxcar::Vec::new();
    let barrier = Barrier::new(6);

    thread::scope(|s| {
        s.spawn(|| {
            barrier.wait();
            for i in 0..1000 {
                vec.push(i);
            }
        });

        s.spawn(|| {
            barrier.wait();
            for i in 1000..2000 {
                vec.push(i);
            }
        });

        s.spawn(|| {
            barrier.wait();
            for i in 2000..3000 {
                vec.push(i);
            }
        });

        s.spawn(|| {
            barrier.wait();
            for i in 3000..4000 {
                vec.push(i);
            }
        });

        s.spawn(|| {
            barrier.wait();
            for i in 0..10_000 {
                if let Some(&x) = vec.get(i) {
                    assert!(x < 4000);
                }
            }
        });

        s.spawn(|| {
            barrier.wait();
            for (i, &x) in vec.iter() {
                assert!(x < 4000);
                assert!(vec[i] < 4000);
            }
        });
    });

    assert_eq!(vec.count(), 4000);
    let mut sorted = vec.into_iter().collect::<Vec<_>>();
    sorted.sort();
    assert_eq!(sorted, (0..4000).collect::<Vec<_>>());
}
