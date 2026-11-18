//! Core data types for memnon.
//!
//! Implemented test-first. See `tests/types.rs`.

use std::collections::HashMap;
use std::fmt;

/// Opaque identifier for a stored memory. The inner value is private (minted only
/// inside the crate) so callers can't forge ids, and so a future switch to
/// content-hash ids isn't a breaking change.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MemoryId(u64);

// NOTE: the `new`/`raw` constructors arrive in Task 4, when the storage layer
// mints ids and keys on them. Adding them now would be untested dead code.

impl fmt::Display for MemoryId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

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
    pub fn content(&self) -> &str {
        &self.content
    }
    pub fn metadata(&self) -> &HashMap<String, String> {
        &self.metadata
    }
    pub fn meta(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
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
/// downstream `match`es.
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
