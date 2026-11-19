//! # memnon
//!
//! A token-budget-first, local-first memory engine for AI agents.
//!
//! This crate is being built test-first, module by module, following the plan in
//! `docs/plans/`. Public items are re-exported from here as each module lands.

pub mod embedder;
pub mod error;
pub mod query;
pub mod types;

pub use embedder::{Embedder, HashEmbedder};
pub use error::{Error, Result};
pub use query::Query;
pub use types::{Entry, MemoryId, Scope};
