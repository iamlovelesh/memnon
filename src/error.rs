//! The crate error type and `Result` alias.
//!
//! Implemented test-first. See `tests/error.rs`.

use crate::types::MemoryId;
use std::fmt;

/// Result alias used throughout the crate.
pub type Result<T> = std::result::Result<T, Error>;

/// Everything that can go wrong in memnon.
///
/// `#[non_exhaustive]` so new variants can be added without breaking downstream
/// `match`es. (A later pass may adopt `thiserror` for `source()` chaining; the
/// hand-written `Display`/`Error` impls keep the crate dependency-free for now.)
#[non_exhaustive]
#[derive(Debug)]
pub enum Error {
    /// Storage/IO failure (opening, reading, or writing the file).
    Io(String),
    /// The file is unreadable or damaged (bad envelope, deserialization failure).
    Corrupt(String),
    /// The file was written by a newer, unsupported schema version.
    UnsupportedSchema { found: u32, max_supported: u32 },
    /// The supplied embedder does not match the one pinned to the file.
    EmbedderMismatch { expected: String, found: String },
    /// An embedder returned a vector whose length differs from `dimensions()`.
    DimensionMismatch { expected: usize, found: usize },
    /// The embedder itself failed (e.g. a bring-your-own network/API embedder).
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
            Error::NotFound(id) => write!(f, "memory not found: {id}"),
        }
    }
}

impl std::error::Error for Error {}
