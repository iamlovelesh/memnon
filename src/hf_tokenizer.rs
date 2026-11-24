//! HuggingFace-compatible [`Tokenizer`] backed by a local `tokenizer.json`.
//!
//! Behind the `hf-tokenizer` feature. Covers any model family that ships a
//! `tokenizer.json` (BERT WordPiece, Llama/Mistral/Gemini/Claude-style byte-BPE,
//! etc.) without requiring a network fetch — callers load their own file.

use std::path::Path;

use crate::error::{Error, Result};
use crate::tokenizer::{HeuristicTokenizer, Tokenizer};

/// A [`Tokenizer`] backed by a local HuggingFace `tokenizer.json` file.
pub struct HfTokenizer {
    inner: tokenizers::Tokenizer,
}

impl HfTokenizer {
    /// Load a tokenizer from a `tokenizer.json` file on disk.
    ///
    /// Returns `Err(Error::Tokenizer)` if the file is missing, unreadable, or
    /// not a valid tokenizer definition.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let inner =
            tokenizers::Tokenizer::from_file(path).map_err(|e| Error::Tokenizer(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Load a tokenizer from in-memory `tokenizer.json` bytes.
    ///
    /// Returns `Err(Error::Tokenizer)` if the bytes aren't a valid tokenizer
    /// definition.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let inner = tokenizers::Tokenizer::from_bytes(bytes)
            .map_err(|e| Error::Tokenizer(e.to_string()))?;
        Ok(Self { inner })
    }
}

impl Tokenizer for HfTokenizer {
    fn count_tokens(&self, text: &str) -> usize {
        // `add_special_tokens = false`: we're counting plain memory content,
        // never model control tokens (mirrors `TiktokenTokenizer`'s
        // `encode_ordinary`).
        //
        // The `unwrap_or_else` fallback is belt-and-suspenders: `encode` on a
        // valid `&str` against an already-loaded tokenizer effectively never
        // fails in practice, but we still avoid a panic/`unwrap()` in a
        // production code path by degrading to the heuristic instead.
        self.inner
            .encode(text, false)
            .map(|e| e.len())
            .unwrap_or_else(|_| HeuristicTokenizer.count_tokens(text))
    }
}
