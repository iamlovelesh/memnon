#![cfg(feature = "fastembed")]
//! Online test for the on-device fastembed embedder. Downloads a model and runs
//! ONNX inference, so it's `#[ignore]`d — run with:
//!   `cargo test --features fastembed --test fastembed -- --ignored`

use memnon::{Budget, FastEmbedEmbedder, MemoryStore, Query, Scope};

#[test]
#[ignore = "downloads a model and runs ONNX inference; run manually"]
fn semantic_recall_prefers_related_memory() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("m.db");
    let embedder = FastEmbedEmbedder::new().unwrap();
    let store = MemoryStore::open(&path, embedder).unwrap();

    store
        .add(
            [
                "The user loves hiking in the mountains.",
                "The recipe calls for two cups of flour.",
            ],
            Scope::LongTerm,
        )
        .unwrap();

    // Semantically related to hiking, not to the recipe, with no shared keywords.
    let hits = store
        .search(
            "outdoor activities the user enjoys",
            Query::new(),
            Budget::TopK(1),
        )
        .unwrap();
    assert_eq!(
        hits[0].memory.content,
        "The user loves hiking in the mountains."
    );
}
