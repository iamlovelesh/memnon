//! OpenAI-compatible BPE [`Tokenizer`] backed by `tiktoken-rs`.
//!
//! Behind the `tiktoken` feature. Gives exact token counts for OpenAI models
//! instead of the crate's default heuristic.

use crate::error::{Error, Result};
use crate::tokenizer::Tokenizer;
use tiktoken_rs::{CoreBPE, bpe_for_model, o200k_base};

/// A [`Tokenizer`] backed by `tiktoken-rs`'s BPE implementation.
pub struct TiktokenTokenizer {
    bpe: CoreBPE,
}

impl TiktokenTokenizer {
    /// Build a tokenizer using the default `o200k_base` encoding (GPT-4o and
    /// newer OpenAI models).
    pub fn new() -> Self {
        // `o200k_base()` only fails if the vocab compiled into `tiktoken-rs` is
        // corrupt, which is unreachable for a crate we depend on at a pinned
        // version — this is the crate's one sanctioned `.expect()`.
        let bpe = o200k_base().expect("embedded o200k vocab");
        Self { bpe }
    }

    /// Build a tokenizer for a specific OpenAI model name (e.g. `"gpt-4o"`).
    ///
    /// Returns `Err(Error::Tokenizer)` if the model name isn't recognized.
    pub fn with_model(model: &str) -> Result<Self> {
        let bpe = bpe_for_model(model)
            .map_err(|e| Error::Tokenizer(e.to_string()))?
            .clone();
        Ok(Self { bpe })
    }
}

impl Default for TiktokenTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Tokenizer for TiktokenTokenizer {
    fn count_tokens(&self, text: &str) -> usize {
        // `encode_ordinary`: no special-token handling, matching how we count
        // plain memory content (never model control tokens).
        self.bpe.encode_ordinary(text).len()
    }
}
