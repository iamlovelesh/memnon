//! The `Tokenizer` trait and the `HeuristicTokenizer` default implementation.
//!
//! Implemented test-first. See tests below.

use crate::index::approx_tokens;

/// Counts tokens in text. This is the pluggable seam: memnon's core stays
/// model-agnostic, and callers can supply alternative tokenizers (e.g., tiktoken,
/// HuggingFace). The default heuristic is always available.
///
/// `Send + Sync` so a single tokenizer can be shared across threads behind the
/// `Arc` the store holds.
pub trait Tokenizer: Send + Sync {
    /// Count tokens in text.
    fn count_tokens(&self, text: &str) -> usize;
}

/// A rough heuristic tokenizer: one token per ~4 Unicode scalar values, rounded up.
///
/// This is a crude heuristic tuned for English/Latin text. It counts `chars()`
/// (scalar values), not bytes, but real tokenizers emit roughly 1–2+ tokens *per*
/// CJK character, so this **underestimates** token-dense scripts (CJK, etc.) by
/// several times. `Budget::Tokens` can therefore under-count for non-Latin
/// content — a real tokenizer is a planned milestone. Documented as approximate.
pub struct HeuristicTokenizer;

impl Tokenizer for HeuristicTokenizer {
    fn count_tokens(&self, text: &str) -> usize {
        approx_tokens(text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heuristic_matches_approx_tokens() {
        assert_eq!(HeuristicTokenizer.count_tokens("abcde"), 2); // ceil(5/4)
        assert_eq!(HeuristicTokenizer.count_tokens("日本語"), 1); // 3 scalars
    }

    #[test]
    fn heuristic_empty_is_zero() {
        assert_eq!(HeuristicTokenizer.count_tokens(""), 0);
    }
}
