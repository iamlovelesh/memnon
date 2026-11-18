//! # memnon
//!
//! A **token-budget-first**, local-first memory engine for AI agents. Hand it
//! your model's context-window budget and it returns exactly the memories that
//! fit — most-relevant first, with scores — offline and in-process.
//!
//! **Status: API skeleton.** Every public item is the *shape* decided in the
//! design doc (`docs/specs/2026-11-18-local-first-memory-engine-design.md`,
//! revised after staff review). Bodies are `todo!()`; this file compiles so the
//! API can be reviewed and type-checked. Implementation follows TDD per the plan.
//!
//! ```ignore
//! let store = MemoryStore::open("./memory.db", embedder)?;   // embedder required here
//! store.add([Entry::new("user prefers dark mode").meta("source", "settings")], Scope::LongTerm)?;
//! let hits = store.search(
//!     "what theme does the user like?",
//!     Query::new().scope(Scope::LongTerm).min_score(0.2),
//!     Budget::Tokens(2000),
//! )?;
//! for Hit { memory, score } in &hits { /* ... */ }
//! ```

#![allow(dead_code, unused_variables)]

use std::collections::HashMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Result alias used throughout the crate.
pub type Result<T> = std::result::Result<T, Error>;

/// Errors returned by the memory engine. `#[non_exhaustive]` so new variants are
/// not breaking changes. (Implementation may use `thiserror` for `source()`.)
#[non_exhaustive]
#[derive(Debug)]
pub enum Error {
    /// Storage/IO failure.
    Io(String),
    /// The file is unreadable/damaged (bad envelope, deserialization failure).
    Corrupt(String),
    /// The file was written by a newer, unsupported schema version.
    UnsupportedSchema { found: u32, max_supported: u32 },
    /// The supplied embedder does not match the one pinned to the file.
    EmbedderMismatch { expected: String, found: String },
    /// An embedder returned a vector whose length ≠ `dimensions()`.
    DimensionMismatch { expected: usize, found: usize },
    /// The embedder failed (e.g. a BYO network/API embedder).
    Embed(String),
    /// No memory exists with the given id.
    NotFound(MemoryId),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(m) => write!(f, "io error: {m}"),
            Error::Corrupt(m) => write!(f, "corrupt store: {m}"),
            Error::UnsupportedSchema {
                found,
                max_supported,
            } => write!(
                f,
                "unsupported schema version {found} (this build supports up to {max_supported})"
            ),
            Error::EmbedderMismatch { expected, found } => write!(
                f,
                "embedder mismatch: file was built with `{expected}`, got `{found}`"
            ),
            Error::DimensionMismatch { expected, found } => {
                write!(f, "dimension mismatch: expected {expected}, got {found}")
            }
            Error::Embed(m) => write!(f, "embedder failed: {m}"),
            Error::NotFound(id) => write!(f, "memory not found: {}", id.0),
        }
    }
}

impl std::error::Error for Error {}

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// Opaque identifier for a stored memory. The inner value is private so a future
/// move to content-hash ids is not a breaking on-disk format change.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MemoryId(u64);

/// Memory partition. A subagent is `Scope::Custom("subagent:<id>")`.
/// `#[non_exhaustive]`; serialized by name, never by discriminant index.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Scope {
    /// Working memory / recent context.
    Session,
    /// Durable, long-lived facts.
    LongTerm,
    /// Any other partition, keyed by an arbitrary string.
    Custom(String),
}

/// A stored memory.
#[derive(Debug, Clone)]
pub struct Memory {
    pub id: MemoryId,
    pub scope: Scope,
    pub content: String,
    /// Embedding produced by the store's pinned [`Embedder`].
    pub embedding: Vec<f32>,
    pub metadata: HashMap<String, String>,
    /// Unix epoch milliseconds.
    pub created_at: u64,
    /// Unix epoch milliseconds. Equals `created_at` in v1 (append-only).
    pub updated_at: u64,
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

/// A recall result: the memory plus the cosine similarity that ranked it.
#[derive(Debug, Clone)]
pub struct Hit {
    pub memory: Memory,
    pub score: f32,
}

// ---------------------------------------------------------------------------
// Embedder
// ---------------------------------------------------------------------------

/// Turns text into vectors. Required at store construction; its [`identity`] and
/// [`dimensions`] are pinned to the file. Examples ship a local on-device
/// embedder and an OpenAI-compatible one.
///
/// [`identity`]: Embedder::identity
/// [`dimensions`]: Embedder::dimensions
pub trait Embedder: Send + Sync {
    /// Embed a batch of texts; the outer `Vec` is 1:1 with `texts`.
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;
    /// Vector dimensionality; recorded in the file and checked on reopen.
    fn dimensions(&self) -> usize;
    /// Stable model identity (e.g. `"bge-small-en-v1.5"`); pinned to the file.
    fn identity(&self) -> &str;
}

// ---------------------------------------------------------------------------
// Query + budget
// ---------------------------------------------------------------------------

/// How much to return from [`MemoryStore::search`]. `Tokens` is the
/// context-management path: the most-similar memories that fit the budget.
#[non_exhaustive]
#[derive(Debug, Clone, Copy)]
pub enum Budget {
    /// Return the top `n` most similar memories.
    TopK(usize),
    /// Return the most similar memories that together fit this many tokens.
    Tokens(usize),
}

/// Retrieval constraints, built fluently so `search` stays stable as filters grow.
#[derive(Debug, Clone, Default)]
pub struct Query {
    scope: Option<Scope>,
    filters: Vec<(String, String)>,
    min_score: Option<f32>,
}

impl Query {
    pub fn new() -> Self {
        Self::default()
    }
    /// Restrict to a single scope. Omit to search **all** scopes.
    pub fn scope(mut self, scope: Scope) -> Self {
        self.scope = Some(scope);
        self
    }
    /// Require `metadata[key] == value` (equality in v1).
    pub fn filter(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.filters.push((key.into(), value.into()));
        self
    }
    /// Drop matches scoring below this cosine similarity.
    pub fn min_score(mut self, score: f32) -> Self {
        self.min_score = Some(score);
        self
    }

    pub fn selected_scope(&self) -> Option<&Scope> {
        self.scope.as_ref()
    }
    pub fn metadata_filters(&self) -> &[(String, String)] {
        &self.filters
    }
    pub fn min_score_floor(&self) -> Option<f32> {
        self.min_score
    }
}

// ---------------------------------------------------------------------------
// MemoryStore
// ---------------------------------------------------------------------------

/// A handle to a memory store backed by a single embedded file. Cheap to clone
/// and safe to share across threads (`Clone + Send + Sync`). The embedder is
/// fixed at construction and never mutated, so there is no identity-pin race.
#[derive(Clone)]
pub struct MemoryStore {
    inner: Arc<Inner>,
}

struct Inner {
    // Placeholder — the real `Storage` handle lands during implementation.
    path: PathBuf,
    embedder: Arc<dyn Embedder>,
}

impl MemoryStore {
    /// Open (or create) a store at `path`, pinning `embedder` to the file. On an
    /// existing file whose pinned `(identity, dimensions)` differ from `embedder`,
    /// returns [`Error::EmbedderMismatch`]; on a newer format,
    /// [`Error::UnsupportedSchema`].
    pub fn open<E: Embedder + 'static>(path: impl AsRef<Path>, embedder: E) -> Result<Self> {
        todo!("implemented during the TDD phase")
    }

    /// Append memories to `scope`. Each item may carry metadata; `String`/`&str`
    /// convert via [`Entry`]. Ids are returned 1:1 with input order, all written
    /// in one transaction. Exact `(scope, content)` duplicates return the existing
    /// id. Empty input returns `Ok(vec![])`.
    pub fn add<I, E>(&self, items: I, scope: Scope) -> Result<Vec<MemoryId>>
    where
        I: IntoIterator<Item = E>,
        E: Into<Entry>,
    {
        todo!("implemented during the TDD phase")
    }

    /// Recall memories for `query` under `constraints`, limited by `budget`,
    /// returning scored [`Hit`]s most-relevant first. v1 uses exact brute-force
    /// cosine.
    pub fn search(&self, query: &str, constraints: Query, budget: Budget) -> Result<Vec<Hit>> {
        todo!("implemented during the TDD phase")
    }

    /// Fetch a single memory by id.
    pub fn get(&self, id: MemoryId) -> Result<Option<Memory>> {
        todo!("implemented during the TDD phase")
    }

    /// Every memory in a scope, unranked.
    pub fn get_all(&self, scope: Scope) -> Result<Vec<Memory>> {
        todo!("implemented during the TDD phase")
    }

    /// Delete one memory by id (`Error::NotFound` if absent).
    pub fn delete(&self, id: MemoryId) -> Result<()> {
        todo!("implemented during the TDD phase")
    }

    /// Remove every memory in a scope (session/subagent teardown); returns the
    /// number removed.
    pub fn clear(&self, scope: &Scope) -> Result<u64> {
        todo!("implemented during the TDD phase")
    }

    /// Number of memories in a scope.
    pub fn count(&self, scope: &Scope) -> Result<usize> {
        todo!("implemented during the TDD phase")
    }

    /// All scopes currently present in the store.
    pub fn list_scopes(&self) -> Result<Vec<Scope>> {
        todo!("implemented during the TDD phase")
    }
}
