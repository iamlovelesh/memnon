//! On-device semantic [`Embedder`] backed by `fastembed` (ONNX Runtime).
//!
//! Behind the `fastembed` feature so the core crate stays pure-Rust and light.
//! The model (all-MiniLM-L6-v2, 384-dim) downloads once on first construction and
//! is cached; afterwards it runs fully offline, on-device.

use crate::embedder::Embedder;
use crate::error::{Error, Result};
use fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions};
use std::sync::Mutex;

const MODEL: EmbeddingModel = EmbeddingModel::AllMiniLML6V2;
const DIMENSIONS: usize = 384;
const IDENTITY: &str = "fastembed:all-MiniLM-L6-v2";

/// A local, on-device semantic embedder. Construct once and share it (it's
/// `Send + Sync`); inference is serialized internally.
pub struct FastEmbedEmbedder {
    // `TextEmbedding::embed` takes `&mut self`; a Mutex gives us the `&self`
    // the `Embedder` trait requires and makes the type `Sync`.
    model: Mutex<TextEmbedding>,
}

impl FastEmbedEmbedder {
    /// Build the embedder, downloading the model on first use (then cached).
    pub fn new() -> Result<Self> {
        let options = TextInitOptions::new(MODEL).with_show_download_progress(false);
        let model = TextEmbedding::try_new(options).map_err(|e| Error::Embed(e.to_string()))?;
        Ok(Self {
            model: Mutex::new(model),
        })
    }
}

impl Embedder for FastEmbedEmbedder {
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let mut model = self.model.lock().unwrap_or_else(|e| e.into_inner());
        model
            .embed(texts, None)
            .map_err(|e| Error::Embed(e.to_string()))
    }

    fn dimensions(&self) -> usize {
        DIMENSIONS
    }

    fn identity(&self) -> &str {
        IDENTITY
    }
}
