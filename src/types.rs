//! Core data types for memnon.
//!
//! Implemented test-first. See `tests/types.rs`.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

/// Opaque identifier for a stored memory. The inner value is private (minted only
/// inside the crate) so callers can't forge ids, and so a future switch to
/// content-hash ids isn't a breaking change.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct MemoryId(u64);

impl MemoryId {
    /// Mint an id. Crate-internal: storage allocates these; tests construct them.
    pub(crate) fn new(raw: u64) -> Self {
        MemoryId(raw)
    }

    /// The underlying value, for the storage layer to key on.
    pub(crate) fn raw(self) -> u64 {
        self.0
    }
}

impl fmt::Display for MemoryId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// A write-side item: content plus optional metadata. `String`/`&str` convert
/// via [`From`] for the no-metadata path.
#[derive(Debug, Clone)]
pub struct Entry {
    content: String,
    metadata: HashMap<String, String>,
}

impl Entry {
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            metadata: HashMap::new(),
        }
    }
    /// Attach a metadata key/value (chainable).
    pub fn meta(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
    pub fn content(&self) -> &str {
        &self.content
    }
    pub fn metadata(&self) -> &HashMap<String, String> {
        &self.metadata
    }
    /// Consume the entry into its parts, so callers can store them without cloning.
    pub(crate) fn into_parts(self) -> (String, HashMap<String, String>) {
        (self.content, self.metadata)
    }
}

impl From<String> for Entry {
    fn from(s: String) -> Self {
        Entry::new(s)
    }
}

impl From<&str> for Entry {
    fn from(s: &str) -> Self {
        Entry::new(s)
    }
}

/// Where a memory lives. `Scope` is a *general key*: `Session` and `LongTerm` are
/// the v1 conveniences, and `Custom` lets a subagent carve out its own partition
/// (`Scope::Custom("subagent:7".into())`).
///
/// `#[non_exhaustive]` so adding variants later is not a breaking change for
/// downstream `match`es. Serialized by variant *name* (serde's default), so
/// adding or reordering variants never remaps existing data.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Scope {
    /// Working memory / recent context.
    Session,
    /// Durable, long-lived facts.
    LongTerm,
    /// Any other partition, keyed by an arbitrary string.
    Custom(String),
}

/// A stored memory. `created_at` is fixed at insertion; `updated_at` advances
/// when a memory is superseded by consolidation (its content/embedding replaced
/// in place while the id is preserved).
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Memory {
    pub id: MemoryId,
    pub scope: Scope,
    pub content: String,
    // Not part of the public field API (read it via `embedding()`), so the
    // internal representation can change without a breaking change.
    pub(crate) embedding: Vec<f32>,
    pub metadata: HashMap<String, String>,
    /// Unix epoch milliseconds. Fixed at insertion time.
    pub created_at: u64,
    /// Unix epoch milliseconds. Advances when consolidation supersedes this memory.
    pub updated_at: u64,
}

impl Memory {
    /// The embedding vector produced by the store's pinned embedder.
    pub fn embedding(&self) -> &[f32] {
        &self.embedding
    }
}

/// A recall result: the memory plus the cosine similarity that ranked it.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct Hit {
    pub memory: Memory,
    pub score: f32,
}
