//! On-device semantic [`Embedder`] backed by `fastembed` (ONNX Runtime).
//!
//! Behind the `fastembed` feature so the core crate stays pure-Rust and light.
//! The model downloads once on first construction and is cached; afterwards it
//! runs fully offline, on-device.

use crate::embedder::Embedder;
use crate::error::{Error, Result};
use fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions};
use std::sync::Mutex;

/// The bundled default model. To change it, edit this one line — dimensions and
/// identity are derived from fastembed's model info, so they can't drift out of
/// sync (which would silently defeat the store's embedder pin).
const MODEL: EmbeddingModel = EmbeddingModel::AllMiniLML6V2;

/// A local, on-device semantic embedder. Construct once and share it (it's
/// `Send + Sync`); inference is serialized internally.
pub struct FastEmbedEmbedder {
    // `TextEmbedding::embed` takes `&mut self`; a Mutex gives us the `&self` the
    // `Embedder` trait requires and makes the type `Sync`.
    model: Mutex<TextEmbedding>,
    dimensions: usize,
    identity: String,
}

impl FastEmbedEmbedder {
    /// Build the embedder, downloading the model on first use (then cached).
    pub fn new() -> Result<Self> {
        // Derive dimensions + a canonical identity from fastembed's model info so
        // they always match the actual model (never hand-kept constants).
        let info =
            TextEmbedding::get_model_info(&MODEL).map_err(|e| Error::Embed(e.to_string()))?;
        let dimensions = info.dim;
        let identity = format!("fastembed:{}", info.model_code);

        let options = TextInitOptions::new(MODEL).with_show_download_progress(false);
        let model = TextEmbedding::try_new(options).map_err(|e| Error::Embed(e.to_string()))?;
        Ok(Self {
            model: Mutex::new(model),
            dimensions,
            identity,
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
        self.dimensions
    }

    fn identity(&self) -> &str {
        &self.identity
    }
}
