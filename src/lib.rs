//! # memnon
//!
//! A token-budget-first, local-first memory engine for AI agents.
//!
//! This crate is being built test-first, module by module, following the plan in
//! `docs/plans/`. Public items are re-exported from here as each module lands.

pub mod embedder;
pub mod error;
pub mod query;
pub mod store;
pub mod tokenizer;
pub mod types;

pub(crate) mod hnsw;
pub(crate) mod index;
pub(crate) mod storage;

#[cfg(feature = "fastembed")]
mod fastembed_embedder;
#[cfg(feature = "fastembed")]
pub use fastembed_embedder::FastEmbedEmbedder;

#[cfg(feature = "tiktoken")]
mod tiktoken_tokenizer;
#[cfg(feature = "tiktoken")]
pub use tiktoken_tokenizer::TiktokenTokenizer;

pub use embedder::{Embedder, HashEmbedder};
pub use error::{Error, Result};
pub use query::{Budget, Query};
pub use store::MemoryStore;
pub use tokenizer::{HeuristicTokenizer, Tokenizer};
pub use types::{Entry, Hit, Memory, MemoryId, Scope};
