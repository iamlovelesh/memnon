//! Zero-setup *semantic* quickstart using the on-device fastembed embedder.
//!
//! Run: `cargo run --example semantic_quickstart --features fastembed`
//!
//! Downloads a small model (~90 MB) on first run, then works fully offline.
//! Unlike the `quickstart` example (which uses the non-semantic `HashEmbedder`),
//! this ranks by *meaning* — the query shares almost no words with the answer.

use memnon::{Budget, FastEmbedEmbedder, MemoryStore, Query, Scope};

fn main() -> memnon::Result<()> {
    let path = std::env::temp_dir().join("memnon-semantic.db");
    let _ = std::fs::remove_file(&path);

    let embedder = FastEmbedEmbedder::new()?; // downloads the model on first run
    let store = MemoryStore::open(&path, embedder)?;

    store.add(
        [
            "The user's favorite programming language is Rust.",
            "The capital of France is Paris.",
            "Mitochondria are the powerhouse of the cell.",
        ],
        Scope::LongTerm,
    )?;

    // No keyword overlap with "Rust" — recall must work on meaning.
    let hits = store.search(
        "What coding language does the user prefer?",
        Query::new(),
        Budget::TopK(1),
    )?;

    println!("Semantic recall for \"What coding language does the user prefer?\":");
    for hit in &hits {
        println!("  [{:.3}] {}", hit.score, hit.memory.content);
    }

    Ok(())
}
