//! Recall latency vs store size, to measure where exact brute-force search stops
//! being "fine" and justify the v1.1 HNSW milestone.
//!
//! Run: `cargo bench`

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use memnon::{Budget, HashEmbedder, MemoryStore, Query, Scope};
use std::hint::black_box;

fn setup(n: usize) -> (tempfile::TempDir, MemoryStore) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bench.db");
    let store = MemoryStore::open(&path, HashEmbedder::new(384)).unwrap();
    let items: Vec<String> = (0..n)
        .map(|i| format!("memory number {i} about topic {}", i % 100))
        .collect();
    let refs: Vec<&str> = items.iter().map(String::as_str).collect();
    store.add(refs, Scope::LongTerm).unwrap();
    (dir, store)
}

fn bench_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("search_tokens_2000");
    for &n in &[1_000usize, 10_000, 50_000] {
        let (_dir, store) = setup(n);
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                store
                    .search(
                        black_box("memory number 42 about topic 42"),
                        Query::new(),
                        Budget::Tokens(2000),
                    )
                    .unwrap()
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_search);
criterion_main!(benches);
