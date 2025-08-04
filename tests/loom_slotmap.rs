#![cfg(loom)]

use loom::thread;
use std::sync::Arc;

use boxcar::slotmap::{self, Key as _, SlotMap};

#[test]
fn write_write() {
    loom::model(|| {
        let map = Arc::new(SlotMap::new());
        let m1 = map.clone();
        let m2 = map.clone();

        let t1 = thread::spawn(move || m1.insert(1).0);
        let t2 = thread::spawn(move || m2.insert(2).0);

        let k1 = t1.join().unwrap();
        let k2 = t2.join().unwrap();

        assert_eq!(map[k1], 1);
        assert_eq!(map[k2], 2);
        assert_eq!(map.len(), 2);
    });
}

#[test]
fn read_write() {
    loom::model(|| {
        let mut map = SlotMap::new();
        let (key, _) = map.insert(0);
        map.remove(key);

        // Guess the key by incrementing the version. This is entirely not stable API.
        let key = slotmap::DefaultKey::from(slotmap::KeyData::from_ffi(key.data().as_ffi() + 2));

        let map = Arc::new(map);
        let m1 = map.clone();
        let m2 = map.clone();

        let t1 = thread::spawn(move || m1.insert(1).0);
        let t2 = thread::spawn(move || loop {
            if let Some(&v) = m2.get(key) {
                break v;
            }
            thread::yield_now();
        });

        t1.join().unwrap();
        let val = t2.join().unwrap();

        assert_eq!(val, 1);
        assert_eq!(map.len(), 1);
    });
}

#[test]
fn write_iter() {
    loom::model(|| {
        let map = Arc::new(SlotMap::new());
        let m1 = map.clone();
        let m2 = map.clone();

        let t1 = thread::spawn(move || m1.insert(1).0);
        let t2 = thread::spawn(move || m2.values().copied().collect::<Vec<_>>());

        t1.join().unwrap();
        let vals = t2.join().unwrap();

        assert!(vals == [1] || vals == [], "{vals:?}");
    });
}
