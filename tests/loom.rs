#![cfg(loom)]

use loom::thread;
use std::sync::Arc;

#[test]
fn write_write() {
    loom::model(|| {
        let vec = Arc::new(boxcar::Vec::new());
        let v1 = vec.clone();
        let v2 = vec.clone();

        let t1 = thread::spawn(move || v1.push(1));
        let t2 = thread::spawn(move || v2.push(2));

        let i1 = t1.join().unwrap();
        let i2 = t2.join().unwrap();

        assert_eq!(vec[i1], 1);
        assert_eq!(vec[i2], 2);
        assert_eq!(vec.count(), 2);
    });
}

#[test]
fn read_write() {
    loom::model(|| {
        let vec = Arc::new(boxcar::Vec::new());
        let v1 = vec.clone();
        let v2 = vec.clone();

        let t1 = thread::spawn(move || v1.push(1));
        let t2 = thread::spawn(move || loop {
            let Some(&v) = v2.get(0) else {
                thread::yield_now();
                continue;
            };

            break v;
        });

        t1.join().unwrap();
        let val = t2.join().unwrap();

        assert_eq!(val, 1);
        assert_eq!(vec.count(), 1);
    });
}

#[test]
fn iter_linearizability() {
    loom::model(|| {
        let vec = Arc::new(boxcar::Vec::new());
        let v1 = vec.clone();
        let v2 = vec.clone();

        let t1 = thread::spawn(move || v1.push(1));
        let t2 = thread::spawn(move || {
            v2.push(2);
            assert!(v2.iter().find(|&(_, x)| *x == 2).is_some());
        });

        t1.join().unwrap();
        t2.join().unwrap();
    });
}

#[test]
fn mixed() {
    loom::model(|| {
        let vec = Arc::new(boxcar::Vec::new());
        let v1 = vec.clone();
        let v2 = vec.clone();
        let v3 = vec.clone();

        let t1 = thread::spawn(move || {
            v1.push(0);
        });

        let t2 = thread::spawn(move || {
            v2.push(1);
        });

        let t3 = thread::spawn(move || {
            for i in 0..2 {
                if let Some(&v) = v3.get(i) {
                    assert!(v == 0 || v == 1);
                };
            }
        });

        t1.join().unwrap();
        t2.join().unwrap();
        t3.join().unwrap();

        assert_eq!(vec.count(), 2);

        let mut values = vec.iter().map(|(_, &v)| v).collect::<Vec<_>>();
        values.sort();
        assert_eq!(values, (0..2).collect::<Vec<_>>());
    });
}
