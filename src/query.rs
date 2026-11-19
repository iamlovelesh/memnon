//! The recall query builder (`Query`) and retrieval budget (`Budget`).
//!
//! Implemented test-first. See `tests/query.rs`.

use crate::types::Scope;

/// How much to return from a recall. `Tokens` is the headline context-management
/// path: the most-similar memories that *fit* the budget, not a fixed `k`.
#[non_exhaustive]
#[derive(Debug, Clone, Copy)]
pub enum Budget {
    /// Return the top `n` most similar memories.
    TopK(usize),
    /// Return the most similar memories that together fit within this many tokens.
    Tokens(usize),
}

/// Constraints for a recall, built fluently. Kept as a builder so `search`'s
/// signature stays stable as richer filters are added later.
///
/// ```ignore
/// Query::new().scope(Scope::LongTerm).filter("source", "docs").min_score(0.2)
/// ```
#[derive(Debug, Clone, Default)]
pub struct Query {
    scope: Option<Scope>,
    filters: Vec<(String, String)>,
    min_score: Option<f32>,
}

impl Query {
    /// An unconstrained query: all scopes, no filters, no score floor.
    pub fn new() -> Self {
        // `Default` already gives None / empty Vec / None — no need to repeat them.
        Self::default()
    }

    /// Restrict recall to a single scope. Omit to search every scope.
    pub fn scope(mut self, scope: Scope) -> Self {
        self.scope = Some(scope);
        self
    }

    /// Require `metadata[key] == value`. Multiple filters accumulate (all must match).
    pub fn filter(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.filters.push((key.into(), value.into()));
        self
    }

    /// Drop matches scoring below this cosine similarity.
    pub fn min_score(mut self, score: f32) -> Self {
        self.min_score = Some(score);
        self
    }

    /// The selected scope, if any. Borrows the inner `Scope` rather than moving it.
    pub fn selected_scope(&self) -> Option<&Scope> {
        self.scope.as_ref()
    }

    /// The accumulated metadata filters. `&Vec<_>` coerces to a slice `&[_]`.
    pub fn metadata_filters(&self) -> &[(String, String)] {
        &self.filters
    }

    /// The score floor, if set. `f32` is `Copy`, so it's returned by value.
    pub fn min_score_floor(&self) -> Option<f32> {
        self.min_score
    }
}
