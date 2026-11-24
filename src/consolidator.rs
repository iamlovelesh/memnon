//! The `Consolidator` trait, its ops/outcome types.
//!
//! Implemented test-first. See tests below.

use crate::error::Result;
use crate::types::MemoryId;

/// An existing memory found to be related to incoming content, surfaced to a
/// [`Consolidator`] so it can decide whether the incoming content should be
/// added fresh, merged into one of these, or should supersede/delete one.
///
/// The store builds these from its recall path; external `Consolidator`
/// implementations only ever read the fields. `#[non_exhaustive]` still
/// permits that read-only usage — it just blocks struct-literal construction
/// outside the crate, which is why `new` is `pub(crate)`.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct RelatedMemory {
    /// Identity of the existing memory.
    pub id: MemoryId,
    /// The existing memory's stored content.
    pub content: String,
    /// Similarity score that surfaced this memory as related (same scale as
    /// [`crate::types::Hit::score`]).
    pub score: f32,
}

impl RelatedMemory {
    /// Build a related-memory view. Crate-internal: only the store assembles
    /// these (from recall results), so an external `Consolidator` can't forge
    /// fake "related" entries with ids that don't exist.
    ///
    /// Not yet called from production code — the store wires this in once
    /// consolidation is plumbed through `MemoryStore` (a later task in this
    /// feature). Allowed dead for now so clippy doesn't block this seam from
    /// landing independently.
    #[allow(dead_code)]
    pub(crate) fn new(id: MemoryId, content: String, score: f32) -> Self {
        Self { id, content, score }
    }
}

/// A single reconciliation decision produced by a [`Consolidator`].
///
/// `#[non_exhaustive]` at the enum level still lets external `Consolidator`
/// implementations construct any of the existing variants (it only blocks
/// exhaustive `match`es without a `_` arm) — the same shape as
/// `std::io::ErrorKind`. That's intentional: new variants may be added later
/// without breaking callers who already construct `Add`/`Update`/`Delete`/`Noop`.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub enum ConsolidationOp {
    /// Store the incoming content as a brand-new memory.
    Add,
    /// Overwrite an existing memory's content in place.
    Update {
        /// The memory to update.
        id: MemoryId,
        /// Its new content.
        content: String,
    },
    /// Remove an existing memory (e.g. the incoming fact supersedes it and
    /// nothing should remain of the old version).
    Delete {
        /// The memory to delete.
        id: MemoryId,
    },
    /// Do nothing — the incoming content is redundant with what's already
    /// stored.
    Noop,
}

/// Decides how incoming content should be reconciled against related,
/// already-stored memories. This is the pluggable seam: memnon's core stays
/// model-agnostic, and callers supply the reconciliation policy — anything
/// from a simple heuristic to an LLM call that reads `related` and proposes
/// edits.
///
/// `Send + Sync` so a single consolidator can be shared across threads behind
/// the `Arc` the store holds, mirroring [`crate::tokenizer::Tokenizer`] and
/// [`crate::embedder::Embedder`].
pub trait Consolidator: Send + Sync {
    /// Decide what to do with `incoming` given the `related` memories already
    /// in the store. Returns the ops to apply; an empty `Vec` is equivalent to
    /// `[ConsolidationOp::Noop]`.
    fn reconcile(&self, incoming: &str, related: &[RelatedMemory]) -> Result<Vec<ConsolidationOp>>;
}

/// Tally of what a consolidation pass actually did, for callers that want to
/// report or log the outcome.
#[non_exhaustive]
#[derive(Debug, Default)]
pub struct ConsolidationOutcome {
    /// Ids of newly added memories.
    pub added: Vec<MemoryId>,
    /// Ids of memories whose content was updated.
    pub updated: Vec<MemoryId>,
    /// Ids of memories that were deleted.
    pub deleted: Vec<MemoryId>,
    /// Count of ops that were no-ops.
    pub noops: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::MemoryId;

    // A closure-backed double proves the trait is implementable and ops construct.
    struct Rule<F>(F);
    impl<F: Fn(&str, &[RelatedMemory]) -> Vec<ConsolidationOp> + Send + Sync> Consolidator for Rule<F> {
        fn reconcile(
            &self,
            incoming: &str,
            related: &[RelatedMemory],
        ) -> crate::error::Result<Vec<ConsolidationOp>> {
            Ok((self.0)(incoming, related))
        }
    }

    #[test]
    fn double_returns_ops_and_reads_related() {
        let r = RelatedMemory::new(MemoryId::new(7), "old".into(), 0.9);
        let llm = Rule(|_in: &str, rel: &[RelatedMemory]| {
            vec![ConsolidationOp::Update {
                id: rel[0].id,
                content: "new".into(),
            }]
        });
        let ops = llm.reconcile("incoming", std::slice::from_ref(&r)).unwrap();
        assert!(matches!(ops[0], ConsolidationOp::Update { .. }));
        assert_eq!(r.score, 0.9);
    }

    #[test]
    fn outcome_default_is_empty() {
        let o = ConsolidationOutcome::default();
        assert!(o.added.is_empty() && o.noops == 0);
    }
}
