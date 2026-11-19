//! Tests for the Embedder trait and the HashEmbedder dev double.

use memnon::{Embedder, HashEmbedder};

#[test]
fn hash_embedder_is_deterministic_and_shaped() {
    let e = HashEmbedder::new(64);
    let a = e.embed(&["hello world"]).unwrap();
    let b = e.embed(&["hello world"]).unwrap();

    assert_eq!(a, b, "same input must produce the same vector");
    assert_eq!(a.len(), 1, "one input -> one vector");
    assert_eq!(a[0].len(), 64, "vector length == dimensions()");
    assert_eq!(e.dimensions(), 64);
    assert_eq!(e.identity(), "hash-v1-d64");
}

#[test]
fn hash_embedder_batch_is_one_to_one() {
    let e = HashEmbedder::new(8);
    let out = e.embed(&["a", "b", "c"]).unwrap();
    assert_eq!(out.len(), 3);
}

#[test]
fn hash_embedder_handles_empty_string_without_nan() {
    let e = HashEmbedder::new(8);
    let out = e.embed(&[""]).unwrap();

    assert_eq!(out[0].len(), 8);
    assert!(
        out[0].iter().all(|x| x.is_finite()),
        "empty text must not produce NaN/inf"
    );
    assert!(
        out[0].iter().all(|&x| x == 0.0),
        "empty text normalizes to an all-zero vector"
    );
}
