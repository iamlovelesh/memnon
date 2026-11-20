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
fn large_store_uses_hnsw_and_finds_exact_match() {
    let (_d, s) = store();
    // Exceed the brute-force threshold (2000) to exercise the HNSW path.
    // Topics give the vectors a gradient so the target sits in a reachable
    // cluster (an isolated outlier is a known-hard ANN case, out of scope here).
    let bulk: Vec<String> = (0..2100)
        .map(|i| format!("filler memory {i} about topic {}", i % 50))
        .collect();
    let refs: Vec<&str> = bulk.iter().map(String::as_str).collect();
    s.add(refs, Scope::LongTerm).unwrap();

    // Query a real memory verbatim: its same-topic siblings make it reachable,
    // and the exact match (cosine 1.0) ranks first after the exact re-rank.
    let target = "filler memory 1234 about topic 34";
    let hits = s.search(target, Query::new(), Budget::TopK(3)).unwrap();
    assert!(!hits.is_empty());
    assert_eq!(hits[0].memory.content, target);
}

#[test]
fn large_store_metadata_filter_is_exact() {
    // Filtered recall takes the exact-scan path even above the threshold, so it
    // never under-returns the way a post-filtered ANN pool would.
    let (_d, s) = store();
    let bulk: Vec<Entry> = (0..2100).map(|i| Entry::new(format!("doc {i}"))).collect();
    s.add(bulk, Scope::LongTerm).unwrap();
    s.add(
        [Entry::new("special tagged memory").meta("tag", "keep")],
        Scope::LongTerm,
    )
    .unwrap();

    let hits = s
        .search(
            "special tagged memory",
            Query::new().filter("tag", "keep"),
            Budget::TopK(5),
        )
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].memory.content, "special tagged memory");
}

#[test]
fn large_store_scoped_query_returns_all_in_scope() {
    // Regression for the HNSW under-return bug: a scoped query on a large,
    // mixed-scope store must return ALL in-scope matches, not a truncated pool.
    let (_d, s) = store();
    let session: Vec<String> = (0..2100).map(|i| format!("session doc {i}")).collect();
    s.add(
        session.iter().map(String::as_str).collect::<Vec<_>>(),
        Scope::Session,
    )
    .unwrap();
    let longterm: Vec<String> = (0..40).map(|i| format!("longterm fact {i}")).collect();
    s.add(
        longterm.iter().map(String::as_str).collect::<Vec<_>>(),
        Scope::LongTerm,
    )
    .unwrap();

    let hits = s
        .search(
            "longterm fact 5",
            Query::new().scope(Scope::LongTerm),
            Budget::TopK(40),
        )
        .unwrap();
    assert_eq!(hits.len(), 40, "scoped query under-returned");
    assert!(hits.iter().all(|h| h.memory.scope == Scope::LongTerm));
}

#[test]
fn large_store_tokens_budget_is_not_capped() {
    // Regression: a large token budget on a large store must not be silently
    // capped at the HNSW pool floor (128).
    let (_d, s) = store();
    // Navigable content (shared tokens form a gradient) so HNSW can reach the store.
    let items: Vec<String> = (0..2200)
        .map(|i| format!("note {i} cluster {} band {}", i % 20, i % 7))
        .collect();
    s.add(
        items.iter().map(String::as_str).collect::<Vec<_>>(),
        Scope::LongTerm,
    )
    .unwrap();

    let hits = s
        .search(
            "note 100 cluster 0 band 2",
            Query::new(),
            Budget::Tokens(1_000_000),
        )
        .unwrap();
    assert!(
        hits.len() > 1000,
        "Tokens budget silently capped: {} hits",
        hits.len()
    );
}

#[test]
fn hnsw_path_top_k_tracks_brute_force() {
    // Equivalence guard: on a large unfiltered store, the HNSW path's top-k should
    // closely track a brute-force cosine ranking (computed here from the stored
    // embeddings), and reproduce the exact match at #1.
    let (_d, s) = store();
    let embedder = HashEmbedder::new(64);
    let items: Vec<String> = (0..2200)
        .map(|i| format!("doc number {i} topic {} sub {}", i % 37, i % 11))
        .collect();
    s.add(
        items.iter().map(String::as_str).collect::<Vec<_>>(),
        Scope::LongTerm,
    )
    .unwrap();

    // Item 1234 verbatim: 1234 % 37 = 13, 1234 % 11 = 2.
    let query = "doc number 1234 topic 13 sub 2";
    let q = embedder.embed(&[query]).unwrap().pop().unwrap();

    let cos = |a: &[f32], b: &[f32]| -> f32 {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if na == 0.0 || nb == 0.0 {
            0.0
        } else {
            dot / (na * nb)
        }
    };
    let all = s.get_all(Scope::LongTerm).unwrap();
    let mut scored: Vec<(f32, &str)> = all
        .iter()
        .map(|m| (cos(&q, m.embedding()), m.content.as_str()))
        .collect();
    scored.sort_by(|a, b| b.0.total_cmp(&a.0));
    let brute_top: Vec<&str> = scored.iter().take(10).map(|(_, c)| *c).collect();

    let hits = s.search(query, Query::new(), Budget::TopK(10)).unwrap();
    assert_eq!(hits[0].memory.content, query); // exact match ranks #1
    let overlap = hits
        .iter()
        .filter(|h| brute_top.contains(&h.memory.content.as_str()))
        .count();
    assert!(
        overlap >= 7,
        "HNSW/brute top-10 overlap too low: {overlap}/10"
    );
}

#[test]
fn concurrent_add_and_search_do_not_deadlock() {
    let (_d, s) = store();
    let seed: Vec<String> = (0..100).map(|i| format!("seed {i}")).collect();
    s.add(
        seed.iter().map(String::as_str).collect::<Vec<_>>(),
        Scope::LongTerm,
    )
    .unwrap();

    let mut handles = Vec::new();
    for t in 0..4 {
        let s = s.clone();
        handles.push(thread::spawn(move || {
            for i in 0..50 {
                s.add([format!("w{t}-{i}")], Scope::LongTerm).unwrap();
                let _ = s.search("seed 1", Query::new(), Budget::TopK(3)).unwrap();
            }
        }));
    }
    for h in handles {
        h.join().unwrap();
    }
    assert!(s.count(&Scope::LongTerm).unwrap() >= 100);
}

#[test]
fn incremental_index_finds_rows_added_after_first_search() {
    // Regression for the "extended incrementally on add" path: it only runs when
    // an index is already built, which needs a search BEFORE the add on a large store.
    let (_d, s) = store();
    let bulk: Vec<String> = (0..2100)
        .map(|i| format!("note {i} group {}", i % 20))
        .collect();
    s.add(
        bulk.iter().map(String::as_str).collect::<Vec<_>>(),
        Scope::LongTerm,
    )
    .unwrap();
    // Build the HNSW index.
    let _ = s
        .search("note 5 group 5", Query::new(), Budget::TopK(3))
        .unwrap();

    // Now add a distinctive row; the incremental insert must put it in the index.
    s.add(["note 5 group 5 freshly added marker"], Scope::LongTerm)
        .unwrap();
    let hits = s
        .search(
            "note 5 group 5 freshly added marker",
            Query::new(),
            Budget::TopK(1),
        )
        .unwrap();
    assert_eq!(
        hits[0].memory.content,
        "note 5 group 5 freshly added marker"
    );
}

#[test]
fn large_store_delete_then_search_excludes_deleted() {
    let (_d, s) = store();
    // Same reachable content shape as the other large-store tests (enough tokens
    // that the exact match is distinct, not lost in a flat cluster).
    let bulk: Vec<String> = (0..2100)
        .map(|i| format!("filler memory {i} about topic {}", i % 50))
        .collect();
    s.add(
        bulk.iter().map(String::as_str).collect::<Vec<_>>(),
        Scope::LongTerm,
    )
    .unwrap();

    let target = "filler memory 1234 about topic 34"; // 1234 % 50 == 34
    let id = s
        .get_all(Scope::LongTerm)
        .unwrap()
        .into_iter()
        .find(|m| m.content == target)
        .unwrap()
        .id;

    // Build the index + confirm the target is found.
    let hits = s.search(target, Query::new(), Budget::TopK(1)).unwrap();
    assert_eq!(hits[0].memory.content, target);

    // Delete invalidates the index; the next search rebuilds without the row.
    s.delete(id).unwrap();
    let hits = s.search(target, Query::new(), Budget::TopK(3)).unwrap();
    assert!(
        hits.iter().all(|h| h.memory.id != id),
        "deleted row resurfaced"
    );
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

/// An embedder that returns a `NaN` component, to exercise the finiteness guard.
struct NanEmbedder;
impl Embedder for NanEmbedder {
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        Ok(texts
            .iter()
            .map(|_| vec![f32::NAN, 0.0, 0.0, 0.0])
            .collect())
    }
    fn dimensions(&self) -> usize {
        4
    }
    fn identity(&self) -> &str {
        "nan-v1"
    }
}

#[test]
fn add_rejects_non_finite_embedding() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("m.db");
    let s = MemoryStore::open(&path, NanEmbedder).unwrap();
    assert!(matches!(
        s.add(["x"], Scope::LongTerm),
        Err(Error::Embed(_))
    ));
}
