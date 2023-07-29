use std::{sync::Barrier, thread};

#[test]
fn simple() {
    let vec = boxcar::vec![0, 1, 2];
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
