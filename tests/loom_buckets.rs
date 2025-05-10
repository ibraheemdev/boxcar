#![cfg(loom)]

use std::sync::Arc;

use boxcar::buckets;
use boxcar::Buckets;
use loom::thread;

#[test]
fn get_or_alloc() {
    loom::model(|| {
        let buckets = Arc::new(<Buckets<u8, 5>>::new());
        let b1 = buckets.clone();
        let b2 = buckets.clone();
        let b3 = buckets.clone();

        let index = buckets::Index::new(0).unwrap();
        let t1 = thread::spawn(move || _ = b1.get_or_alloc(index));
        let t2 = thread::spawn(move || _ = b2.get_or_alloc(index));
        let t3 = thread::spawn(move || buckets.iter().for_each(drop));

        t1.join().unwrap();
        t2.join().unwrap();
        t3.join().unwrap();
    });
}
