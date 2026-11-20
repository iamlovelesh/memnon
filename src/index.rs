//! Recall math: similarity, token estimation, ranking, and budget packing.
//!
//! Pure functions, unit-tested below. These are crate-internal; the store
//! wires them into `search`.

use crate::query::Budget;
use crate::types::MemoryId;
use std::collections::HashMap;

/// Cosine similarity of two vectors, in `[-1.0, 1.0]`.
///
/// Returns **`0.0`** when either vector has zero norm (e.g. the embedding of
/// empty text) instead of dividing by zero. This guards zero-norm only; callers
/// reject non-finite (`NaN`/`inf`) embeddings at the boundary (`MemoryStore::add`
/// / `search`), so such values never reach here.
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
/// This is a crude heuristic tuned for English/Latin text. It counts `chars()`
/// (scalar values), not bytes, but real tokenizers emit roughly 1–2+ tokens *per*
/// CJK character, so this **underestimates** token-dense scripts (CJK, etc.) by
/// several times. `Budget::Tokens` can therefore under-count for non-Latin
/// content — a real tokenizer is a planned milestone. Documented as approximate.
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
        .map(|(id, embedding, created_at)| {
            let score = cosine(query, embedding);
            // Defense-in-depth: a non-finite score (e.g. legacy data written before
            // the ingest guard) sorts LAST, not first — `total_cmp` would otherwise
            // place NaN above every real score.
            let score = if score.is_finite() {
                score
            } else {
                f32::NEG_INFINITY
            };
            (*id, score, *created_at)
        })
        .collect();

    // Sort by score desc, then created_at desc (recency), then id asc as a final
    // deterministic tiebreak so results don't depend on candidate iteration order
    // (which differs between the brute-force and HNSW paths). `total_cmp` gives a
    // total order over f32.
    scored.sort_by(|a, b| b.1.total_cmp(&a.1).then(b.2.cmp(&a.2)).then(a.0.cmp(&b.0)));

    scored
        .into_iter()
        .map(|(id, score, _)| (id, score))
        .collect()
}

/// Select which ranked ids to return under a budget.
///
/// Drops anything below `min_score` first. For `TopK(k)`, takes the first `k`.
///
/// For `Tokens(max)`, this is a **greedy fill** in ranked order: each memory is
/// included if it still fits the running token total, and one that does *not* fit
/// is **skipped** so a later, smaller memory can still be included (it does not
/// stop at the first overflow). This maximizes use of the budget, at the cost of
/// inclusion not being strictly monotonic in rank — a more-relevant large memory
/// can be skipped while a less-relevant small one is kept. Empty if even the top
/// eligible memory exceeds `max`.
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
    fn rank_tiebreak_is_deterministic_by_id() {
        let query = vec![1.0, 0.0];
        // Identical vectors AND created_at => tie on score and recency => id asc.
        let candidates = vec![
            (MemoryId::new(3), vec![1.0, 0.0], 5u64),
            (MemoryId::new(1), vec![1.0, 0.0], 5u64),
            (MemoryId::new(2), vec![1.0, 0.0], 5u64),
        ];
        let ids: Vec<_> = rank(&query, &candidates)
            .into_iter()
            .map(|(id, _)| id)
            .collect();
        assert_eq!(
            ids,
            vec![MemoryId::new(1), MemoryId::new(2), MemoryId::new(3)]
        );
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
    fn pack_tokens_respects_running_total() {
        let ranked = vec![(MemoryId::new(1), 0.9), (MemoryId::new(2), 0.8)];
        let tokens = HashMap::from([(MemoryId::new(1), 30usize), (MemoryId::new(2), 30usize)]);
        // First fits (30 <= 40); second would overflow (30+30 > 40) and is dropped.
        assert_eq!(
            pack(&ranked, &tokens, Budget::Tokens(40), None),
            vec![MemoryId::new(1)]
        );
    }

    #[test]
    fn pack_tokens_skips_oversized_then_includes_smaller() {
        // Greedy fill: the top-ranked item is too big and is skipped; the smaller
        // lower-ranked item still fits and is included.
        let ranked = vec![(MemoryId::new(1), 0.9), (MemoryId::new(2), 0.8)];
        let tokens = HashMap::from([(MemoryId::new(1), 50usize), (MemoryId::new(2), 10usize)]);
        assert_eq!(
            pack(&ranked, &tokens, Budget::Tokens(40), None),
            vec![MemoryId::new(2)]
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
