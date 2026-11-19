//! Recall math: similarity, token estimation, ranking, and budget packing.
//!
//! Pure functions, unit-tested below. These are crate-internal; the store
//! (Task 5) wires them into `search`.

// These are consumed by the store in Task 5. Until then only the unit tests
// below exercise them in a non-`cfg(test)` build. Remove when the store lands.
#![allow(dead_code)]

use crate::query::Budget;
use crate::types::MemoryId;
use std::collections::HashMap;

/// Cosine similarity of two vectors, in `[-1.0, 1.0]`.
///
/// Returns **`0.0`** when either vector has zero norm (e.g. the embedding of
/// empty text) instead of dividing by zero — so `NaN` never enters a ranking.
pub(crate) fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

/// Rough token estimate: one token per ~4 Unicode scalar values, rounded up.
///
/// Counts `chars()` (scalar values), not bytes, so multi-byte text (CJK, emoji)
/// isn't wildly mis-estimated. Documented as approximate; a real tokenizer is a
/// later milestone.
pub(crate) fn approx_tokens(s: &str) -> usize {
    s.chars().count().div_ceil(4)
}

/// Rank candidates against a query embedding, most-similar first.
///
/// Each candidate is `(id, embedding, created_at)`. Ties on score break toward
/// the more recent memory (higher `created_at`).
pub(crate) fn rank(
    query: &[f32],
    candidates: &[(MemoryId, Vec<f32>, u64)],
) -> Vec<(MemoryId, f32)> {
    let mut scored: Vec<(MemoryId, f32, u64)> = candidates
        .iter()
        .map(|(id, embedding, created_at)| (*id, cosine(query, embedding), *created_at))
        .collect();

    // Sort by score descending; break ties by created_at descending (recency).
    // `total_cmp` gives a total order over f32 (handles any stray NaN deterministically).
    scored.sort_by(|a, b| b.1.total_cmp(&a.1).then(b.2.cmp(&a.2)));

    scored
        .into_iter()
        .map(|(id, score, _)| (id, score))
        .collect()
}

/// Select which ranked ids to return under a budget.
///
/// Drops anything below `min_score` first. For `TopK(k)`, takes the first `k`.
/// For `Tokens(max)`, walks the ranked list and keeps each memory whose tokens
/// still fit the running total (empty if even the top memory exceeds `max`).
pub(crate) fn pack(
    ranked: &[(MemoryId, f32)],
    tokens: &HashMap<MemoryId, usize>,
    budget: Budget,
    min_score: Option<f32>,
) -> Vec<MemoryId> {
    let eligible = ranked
        .iter()
        .filter(|(_, score)| min_score.is_none_or(|floor| *score >= floor));

    match budget {
        Budget::TopK(k) => eligible.take(k).map(|(id, _)| *id).collect(),
        Budget::Tokens(max) => {
            let mut used = 0usize;
            let mut out = Vec::new();
            for (id, _) in eligible {
                let cost = *tokens.get(id).unwrap_or(&0);
                if used + cost <= max {
                    used += cost;
                    out.push(*id);
                }
            }
            out
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_identical_is_one() {
        assert!((cosine(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_orthogonal_is_zero() {
        assert!(cosine(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6);
    }

    #[test]
    fn cosine_zero_norm_is_zero() {
        assert_eq!(cosine(&[0.0, 0.0], &[1.0, 0.0]), 0.0);
    }

    #[test]
    fn approx_tokens_rounds_up() {
        assert_eq!(approx_tokens("abcde"), 2); // 5 chars -> ceil(5/4)
    }

    #[test]
    fn approx_tokens_counts_unicode_scalars_not_bytes() {
        assert_eq!(approx_tokens("日本語"), 1); // 3 scalar values, not 9 bytes
    }

    #[test]
    fn rank_orders_by_similarity_desc() {
        let query = vec![1.0, 0.0];
        let candidates = vec![
            (MemoryId::new(1), vec![0.0, 1.0], 0u64),
            (MemoryId::new(2), vec![1.0, 0.0], 0u64),
        ];
        let ranked = rank(&query, &candidates);
        assert_eq!(ranked[0].0, MemoryId::new(2));
    }

    #[test]
    fn rank_breaks_ties_by_recency() {
        let query = vec![1.0, 0.0];
        // Identical embeddings => equal score => higher created_at wins.
        let candidates = vec![
            (MemoryId::new(1), vec![1.0, 0.0], 100u64),
            (MemoryId::new(2), vec![1.0, 0.0], 200u64),
        ];
        let ranked = rank(&query, &candidates);
        assert_eq!(ranked[0].0, MemoryId::new(2));
    }

    #[test]
    fn pack_topk_limits_count() {
        let ranked = vec![
            (MemoryId::new(1), 0.9),
            (MemoryId::new(2), 0.8),
            (MemoryId::new(3), 0.7),
        ];
        assert_eq!(
            pack(&ranked, &HashMap::new(), Budget::TopK(2), None).len(),
            2
        );
    }

    #[test]
    fn pack_tokens_stops_at_budget() {
        let ranked = vec![(MemoryId::new(1), 0.9), (MemoryId::new(2), 0.8)];
        let tokens = HashMap::from([(MemoryId::new(1), 30usize), (MemoryId::new(2), 30usize)]);
        assert_eq!(
            pack(&ranked, &tokens, Budget::Tokens(40), None),
            vec![MemoryId::new(1)]
        );
    }

    #[test]
    fn pack_tokens_empty_when_top_exceeds_budget() {
        let ranked = vec![(MemoryId::new(1), 0.9)];
        let tokens = HashMap::from([(MemoryId::new(1), 100usize)]);
        assert!(pack(&ranked, &tokens, Budget::Tokens(10), None).is_empty());
    }

    #[test]
    fn pack_min_score_drops_weak_matches() {
        let ranked = vec![(MemoryId::new(1), 0.9), (MemoryId::new(2), 0.1)];
        assert_eq!(
            pack(&ranked, &HashMap::new(), Budget::TopK(10), Some(0.5)),
            vec![MemoryId::new(1)]
        );
    }
}
