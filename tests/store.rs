//! End-to-end tests for `MemoryStore` using the deterministic `HashEmbedder`.

use memnon::{Budget, Embedder, Entry, Error, HashEmbedder, MemoryStore, Query, Result, Scope};
use std::thread;

fn store() -> (tempfile::TempDir, MemoryStore) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("memory.db");
    let s = MemoryStore::open(&path, HashEmbedder::new(64)).unwrap();
    (dir, s)
}

#[test]
fn add_returns_one_id_per_item_and_get_all_sees_them() {
    let (_d, s) = store();
    let ids = s.add(["a", "b", "c"], Scope::LongTerm).unwrap();
    assert_eq!(ids.len(), 3);
    assert_eq!(s.get_all(Scope::LongTerm).unwrap().len(), 3);
    // append-only: updated_at == created_at in v1
    let mem = s.get(ids[0]).unwrap().unwrap();
    assert_eq!(mem.created_at, mem.updated_at);
}

#[test]
fn add_empty_is_ok_and_noop() {
    let (_d, s) = store();
    let ids = s.add(Vec::<&str>::new(), Scope::Session).unwrap();
    assert!(ids.is_empty());
    assert_eq!(s.count(&Scope::Session).unwrap(), 0);
}

#[test]
fn add_exact_duplicate_returns_same_id() {
    let (_d, s) = store();
    let a = s.add(["same text"], Scope::LongTerm).unwrap();
    let b = s.add(["same text"], Scope::LongTerm).unwrap();
    assert_eq!(a, b);
    assert_eq!(s.count(&Scope::LongTerm).unwrap(), 1);
}

#[test]
fn add_carries_metadata_and_search_filters_on_it() {
    let (_d, s) = store();
    s.add(
        [
            Entry::new("dark theme note").meta("source", "settings"),
            Entry::new("unrelated note"),
        ],
        Scope::LongTerm,
    )
    .unwrap();

    let hits = s
        .search(
            "dark theme note",
            Query::new().filter("source", "settings"),
            Budget::TopK(10),
        )
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].memory.content, "dark theme note");
}

#[test]
fn search_ranks_most_similar_first() {
    // HashEmbedder is deterministic on exact token overlap: identical text scores
    // highest, partial overlap next, disjoint tokens last. No model, no flake.
    let (_d, s) = store();
    s.add(
        ["alpha beta gamma", "alpha beta delta", "zeta eta theta"],
        Scope::LongTerm,
    )
    .unwrap();

    let hits = s
        .search(
            "alpha beta gamma",
            Query::new().scope(Scope::LongTerm),
            Budget::TopK(3),
        )
        .unwrap();

    assert_eq!(hits.len(), 3);
    assert_eq!(hits[0].memory.content, "alpha beta gamma"); // exact match first
    assert_eq!(hits[2].memory.content, "zeta eta theta"); // disjoint tokens last
    assert!(hits[0].score >= hits[1].score && hits[1].score >= hits[2].score);
}

#[test]
fn search_respects_scope_and_defaults_to_all() {
    let (_d, s) = store();
    s.add(["session note"], Scope::Session).unwrap();
    s.add(["longterm note"], Scope::LongTerm).unwrap();

    let scoped = s
        .search(
            "note",
            Query::new().scope(Scope::LongTerm),
            Budget::TopK(10),
        )
        .unwrap();
    assert!(scoped.iter().all(|h| h.memory.scope == Scope::LongTerm));

    let all = s.search("note", Query::new(), Budget::TopK(10)).unwrap();
    assert_eq!(all.len(), 2); // no scope => every scope
}

#[test]
fn search_min_score_drops_weak_matches() {
    let (_d, s) = store();
    s.add(["alpha beta gamma", "zeta eta theta"], Scope::LongTerm)
        .unwrap();
    // Query matches the first exactly (score ~1) and the second not at all (~0).
    let hits = s
        .search(
            "alpha beta gamma",
            Query::new().min_score(0.5),
            Budget::TopK(10),
        )
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].memory.content, "alpha beta gamma");
}

#[test]
fn search_on_empty_store_is_empty() {
    let (_d, s) = store();
    let hits = s
        .search("anything", Query::new(), Budget::TopK(10))
        .unwrap();
    assert!(hits.is_empty());
}

#[test]
fn reopen_persists_data_and_recall() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("m.db");
    {
        let s = MemoryStore::open(&path, HashEmbedder::new(64)).unwrap();
        s.add(
            [Entry::new("persisted fact").meta("k", "v")],
            Scope::LongTerm,
        )
        .unwrap();
    } // dropped: file closed

    let s = MemoryStore::open(&path, HashEmbedder::new(64)).unwrap();
    let all = s.get_all(Scope::LongTerm).unwrap();
    assert_eq!(all.len(), 1);
    assert_eq!(all[0].content, "persisted fact");
    assert_eq!(all[0].metadata.get("k"), Some(&"v".to_string()));

    let hits = s
        .search("persisted fact", Query::new(), Budget::TopK(1))
        .unwrap();
    assert_eq!(hits[0].memory.content, "persisted fact");
}

#[test]
fn reopen_with_mismatched_embedder_errors() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("m.db");
    MemoryStore::open(&path, HashEmbedder::new(16)).unwrap();
    // Different dimension => different identity => refuse.
    let result = MemoryStore::open(&path, HashEmbedder::new(32));
    assert!(matches!(result, Err(Error::EmbedderMismatch { .. })));
}

#[test]
fn concurrent_adds_have_unique_ids() {
    let (_d, s) = store();
    let mut handles = Vec::new();
    for t in 0..8 {
        let s = s.clone();
        handles.push(thread::spawn(move || {
            for i in 0..50 {
                s.add([format!("t{t}-i{i}")], Scope::Session).unwrap();
            }
        }));
    }
    for h in handles {
        h.join().unwrap();
    }

    let all = s.get_all(Scope::Session).unwrap();
    assert_eq!(all.len(), 400);
    let mut ids: Vec<_> = all.iter().map(|m| m.id).collect();
    ids.sort_by_key(|id| format!("{id}"));
    ids.dedup();
    assert_eq!(ids.len(), 400, "all ids must be unique");
}

#[test]
fn clear_scope_and_introspection() {
    let (_d, s) = store();
    s.add(["s1", "s2"], Scope::Session).unwrap();
    s.add(["l1"], Scope::LongTerm).unwrap();

    assert_eq!(s.count(&Scope::Session).unwrap(), 2);
    assert_eq!(s.list_scopes().unwrap().len(), 2);

    assert_eq!(s.clear(&Scope::Session).unwrap(), 2);
    assert_eq!(s.count(&Scope::Session).unwrap(), 0);
    assert_eq!(s.count(&Scope::LongTerm).unwrap(), 1);
}

#[test]
fn delete_removes_and_reports_missing() {
    let (_d, s) = store();
    let ids = s.add(["x"], Scope::LongTerm).unwrap();
    s.delete(ids[0]).unwrap();
    assert!(s.get(ids[0]).unwrap().is_none());
    assert!(matches!(s.delete(ids[0]), Err(Error::NotFound(_))));
}

#[test]
fn search_handles_unicode_within_token_budget() {
    let (_d, s) = store();
    s.add(["日本語 の メモ", "emoji 🎉 memory"], Scope::LongTerm)
        .unwrap();
    // Should not panic or mis-budget on multi-byte content.
    let hits = s
        .search("日本語 の メモ", Query::new(), Budget::Tokens(50))
        .unwrap();
    assert!(!hits.is_empty());
}

/// An embedder that lies about its dimensions, to exercise the guard in `add`.
struct LyingEmbedder;
impl Embedder for LyingEmbedder {
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|_| vec![0.0; 3]).collect()) // 3, not 4
    }
    fn dimensions(&self) -> usize {
        4
    }
    fn identity(&self) -> &str {
        "lying-v1"
    }
}

#[test]
fn add_rejects_embedding_with_wrong_dimension() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("m.db");
    let s = MemoryStore::open(&path, LyingEmbedder).unwrap();
    let result = s.add(["anything"], Scope::LongTerm);
    assert!(matches!(
        result,
        Err(Error::DimensionMismatch {
            expected: 4,
            found: 3
        })
    ));
}
