//! A dependency-free custom [`memnon::Tokenizer`] — the bring-your-own-tokenizer
//! path, used via [`MemoryStore::open_with_tokenizer`].
//!
//! Run: `cargo run --example custom_tokenizer` (no features needed).
//!
//! `WordTokenizer` below is deliberately crude (whitespace-split word count) —
//! it exists to show the seam, not to be an accurate token counter. For real
//! token counts, prefer a bundled tokenizer instead of hand-rolling one:
//! - `tiktoken` feature → [`memnon::TiktokenTokenizer`] (OpenAI BPE, o200k/model-specific)
//! - `hf-tokenizer` feature → [`memnon::HfTokenizer`] (any HuggingFace `tokenizer.json`)
//! - `hf-hub` feature → [`memnon::HfTokenizer::from_pretrained`] (fetch by model id)
//!
//! Implementing your own `Tokenizer` is still useful when: you already have an
//! in-process tokenizer (e.g. one shipped with your inference runtime), you need
//! a provider not covered above, or you're deliberately approximating (as here).

use memnon::{Budget, Entry, HashEmbedder, MemoryStore, Query, Scope, Tokenizer};

/// Counts tokens as whitespace-separated words. Crude, dependency-free, and a
/// poor approximation of real BPE/WordPiece tokenizers — real tokenizers
/// typically split words further (subwords, punctuation), so this
/// *undercounts* relative to an actual model tokenizer. Good enough to
/// demonstrate the `Tokenizer` seam; not recommended for production budgeting.
struct WordTokenizer;

impl Tokenizer for WordTokenizer {
    fn count_tokens(&self, text: &str) -> usize {
        text.split_whitespace().count()
    }
}

fn main() -> memnon::Result<()> {
    let path = std::env::temp_dir().join("memnon-custom-tokenizer.db");
    let _ = std::fs::remove_file(&path);

    let store = MemoryStore::open_with_tokenizer(&path, HashEmbedder::new(256), WordTokenizer)?;

    store.add(
        [
            Entry::new("the user prefers dark mode").meta("source", "settings"),
            Entry::new("the user is on the pro plan"),
            Entry::new("the user lives in Bangalore"),
        ],
        Scope::LongTerm,
    )?;

    // Budget::Tokens counts via our WordTokenizer here, not the crate's
    // built-in heuristic — whatever Tokenizer the store was opened with.
    let hits = store.search(
        "the user prefers dark mode",
        Query::new().scope(Scope::LongTerm),
        Budget::Tokens(10),
    )?;

    println!("Recall for \"dark mode\" (budget 10 words via WordTokenizer):");
    for hit in &hits {
        println!("  [{:.3}] {}", hit.score, hit.memory.content);
    }

    Ok(())
}
