//! Fully offline, zero-setup quickstart.
//!
//! Run: `cargo run --example quickstart`
//!
//! NOTE: `HashEmbedder` is a **deterministic placeholder**, not a semantic model.
//! It's perfect for trying the API offline with zero setup, but it does not
//! understand meaning — recall ranks by token overlap, not semantics. For real
//! semantic recall, implement [`memnon::Embedder`] over a real model (see
//! `examples/openai_embedder.rs`, or a local model such as `fastembed`).

use memnon::{Budget, Entry, HashEmbedder, MemoryStore, Query, Scope};

fn main() -> memnon::Result<()> {
    // A throwaway temp file so re-running is clean; use a real path to persist.
    let path = std::env::temp_dir().join("memnon-quickstart.db");
    let _ = std::fs::remove_file(&path);

    let store = MemoryStore::open(&path, HashEmbedder::new(256))?;

    store.add(
        [
            Entry::new("the user prefers dark mode").meta("source", "settings"),
            Entry::new("the user is on the pro plan"),
            Entry::new("the user lives in Bangalore"),
        ],
        Scope::LongTerm,
    )?;

    // Token-budget-first recall: return what fits 1000 tokens, best-first.
    let hits = store.search(
        "the user prefers dark mode",
        Query::new().scope(Scope::LongTerm),
        Budget::Tokens(1000),
    )?;

    println!("Recall for \"dark mode\" (budget 1000 tokens):");
    for hit in &hits {
        println!("  [{:.3}] {}", hit.score, hit.memory.content);
    }

    Ok(())
}
