//! The `Embedder` trait and the `HashEmbedder` dev double.
//!
//! Implemented test-first. See `tests/embedder.rs`.

use crate::error::Result;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Turns text into vectors. This is the pluggable seam: memnon's core stays
/// model-agnostic, and callers supply a real embedder (examples ship a local
/// on-device one and an OpenAI-compatible one).
///
/// `Send + Sync` so a single embedder can be shared across threads behind the
/// `Arc` the store holds.
pub trait Embedder: Send + Sync {
    /// Embed a batch of texts. The returned outer `Vec` is 1:1 with `texts`.
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;

    /// Vector dimensionality. Recorded in the store file and checked on reopen.
    fn dimensions(&self) -> usize;

    /// Stable identity of the model (e.g. `"bge-small-en-v1.5"`). Pinned to the
    /// store so a mismatched reopen is rejected rather than silently misranking.
    fn identity(&self) -> &str;
}

/// A deterministic, **non-semantic** embedder for tests and local development.
///
/// It hashes whitespace-separated tokens into `dims` buckets and L2-normalizes,
/// so identical text always yields the identical vector — but texts that *mean*
/// the same thing do **not** land near each other. Real semantic recall needs a
/// real model (see the `examples/`). `HashEmbedder` exists so the whole pipeline
/// can be tested without a model download or network.
pub struct HashEmbedder {
    dims: usize,
    // Stored, not computed on the fly: `identity()` returns `&str`, which can't
    // borrow a temporary `String`.
    identity: String,
}

impl HashEmbedder {
    /// Create an embedder producing `dims`-dimensional vectors.
    pub fn new(dims: usize) -> Self {
        Self {
            dims,
            identity: format!("hash-v1-d{dims}"),
        }
    }

    /// Embed a single text into a normalized bag-of-hashed-tokens vector.
    fn embed_one(&self, text: &str) -> Vec<f32> {
        let mut v = vec![0.0f32; self.dims];
        for token in text.split_whitespace() {
            let bucket = (hash_token(token) % self.dims as u64) as usize;
            v[bucket] += 1.0;
        }
        l2_normalize(v)
    }
}

impl Embedder for HashEmbedder {
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|t| self.embed_one(t)).collect())
    }

    fn dimensions(&self) -> usize {
        self.dims
    }

    fn identity(&self) -> &str {
        &self.identity
    }
}

/// Deterministic hash of a token. `DefaultHasher::new()` uses fixed seeds (not the
/// randomized `RandomState`), so results are stable across runs.
fn hash_token(token: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    token.hash(&mut hasher);
    hasher.finish()
}

/// Scale a vector to unit length. A zero vector (e.g. from empty text) is returned
/// unchanged — never divided by zero, so no `NaN`/`inf` ever reaches the index.
fn l2_normalize(mut v: Vec<f32>) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}
