# `boxcar`

[<img alt="crates.io" src="https://img.shields.io/crates/v/boxcar?style=for-the-badge" height="25">](https://crates.io/crates/boxcar)
[<img alt="github" src="https://img.shields.io/badge/github-boxcar-blue?style=for-the-badge" height="25">](https://github.com/ibraheemdev/boxcar)
[<img alt="docs.rs" src="https://img.shields.io/docsrs/boxcar?style=for-the-badge" height="25">](https://docs.rs/boxcar)

A concurrent, append-only vector.

The vector provided by this crate supports lock-free `get` and `push` operations.

## Examples

Appending an element to a vector and retrieving it:

```rust
let vec = boxcar::Vec::new();
vec.push(42);
assert_eq!(vec[0], 42);
```

The vector can be shared across threads with an `Arc`:

```rust
use std::sync::Arc;

fn main() {
    let vec = Arc::new(boxcar::Vec::new());

    // spawn 6 threads that append to the vec
    let threads = (0..6)
        .map(|i| {
            let vec = vec.clone();

            std::thread::spawn(move || {
                vec.push(i); // push through `&Vec`
            })
        })
        .collect::<Vec<_>>();

    // wait for the threads to finish
    for thread in threads {
        thread.join().unwrap();
    }

    for i in 0..6 {
        assert!(vec.iter().any(|(_, &x)| x == i));
    }
}
```

Elements can be mutated through fine-grained locking:

```rust
use std::sync::{Mutex, Arc};

fn main() {
    let vec = Arc::new(boxcar::Vec::new());

    // insert an element
    vec.push(Mutex::new(1));

    let thread = std::thread::spawn({
        let vec = vec.clone();
        move || {
            // mutate through the mutex
            *vec[0].lock().unwrap() += 1;
        }
    });

    thread.join().unwrap();

    let x = vec[0].lock().unwrap();
    assert_eq!(*x, 2);
}
```
