// adapted from: https://github.com/hawkw/sharded-slab/blob/main/benches/bench.rs
// which carries the following MIT license:
//
// Copyright (c) 2019 Eliza Weisman
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::{
    sync::{Arc, Barrier, RwLock},
    thread,
    time::{Duration, Instant},
};

#[derive(Clone)]
struct Harness<T> {
    start: Arc<Barrier>,
    end: Arc<Barrier>,
    vec: Arc<T>,
}

const THREADS: usize = 12;

impl<T> Harness<T>
where
    T: Send + Sync + 'static,
{
    fn new(vec: Arc<T>) -> Self {
        Self {
            start: Arc::new(Barrier::new(THREADS + 1)),
            end: Arc::new(Barrier::new(THREADS + 1)),
            vec,
        }
    }

    fn run(&self, f: impl FnOnce(&T) + Send + Copy + 'static) -> Duration {
        for _ in 0..THREADS {
            let start = self.start.clone();
            let end = self.end.clone();
            let vec = self.vec.clone();
            thread::spawn(move || {
                start.wait();
                f(&*vec);
                end.wait();
            });
        }

        self.start.wait();
        let t0 = Instant::now();
        self.end.wait();
        t0.elapsed()
    }
}

fn write_read(c: &mut Criterion) {
    const WORKLOAD: &[usize] = &[100, 1000, 10_000, 50_000, 100_000];

    let mut group = c.benchmark_group("push_get");
    group.measurement_time(Duration::from_secs(15));

    for i in WORKLOAD {
        group.bench_with_input(BenchmarkId::new("boxcar::Vec<_>", i), i, |b, &i| {
            b.iter_custom(|iters| {
                let mut total = Duration::from_secs(0);
                for _ in 0..iters {
                    let bench = Harness::new(Arc::new(boxcar::Vec::new()));

                    let elapsed = bench.run(move |vec: &boxcar::Vec<bool>| {
                        let v: Vec<_> = (0..i).map(|_| vec.push(true)).collect();
                        for i in v {
                            assert!(vec.get(i).unwrap());
                        }
                    });

                    total += elapsed;
                }
                total
            })
        });

        group.bench_with_input(BenchmarkId::new("RwLock<Vec<_>>", i), i, |b, &i| {
            b.iter_custom(|iters| {
                let mut total = Duration::from_secs(0);
                for _ in 0..iters {
                    let bench = Harness::new(Arc::new(RwLock::new(Vec::new())));

                    let elapsed = bench.run(move |vec: &RwLock<Vec<bool>>| {
                        let v: Vec<_> = (0..i)
                            .map(|_| {
                                let mut vec = vec.write().unwrap();
                                vec.push(true);
                                vec.len() - 1
                            })
                            .collect();
                        for i in v {
                            assert!(vec.read().unwrap().get(i).unwrap());
                        }
                    });

                    total += elapsed;
                }
                total
            })
        });
    }

    group.finish();
}

criterion_group!(benches, write_read);
criterion_main!(benches);
