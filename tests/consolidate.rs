//! End-to-end tests for `MemoryStore::consolidate` — the reconciliation
//! pipeline that runs incoming facts through a pluggable `Consolidator` and
//! applies the resulting Add/Update/Delete/Noop decisions.

use memnon::{
    Budget, ConsolidationOp, Consolidator, Embedder, Error, HashEmbedder, MemoryStore, Query,
    RelatedMemory, Result, Scope,
};
use std::sync::atomic::{AtomicUsize, Ordering};

/// A closure-backed `Consolidator` double, so each test can express its own
/// reconciliation policy inline.
struct Rule<F>(F);
impl<F: Fn(&str, &[RelatedMemory]) -> Vec<ConsolidationOp> + Send + Sync> Consolidator for Rule<F> {
    fn reconcile(&self, incoming: &str, related: &[RelatedMemory]) -> Result<Vec<ConsolidationOp>> {
        Ok((self.0)(incoming, related))
    }
}

fn store() -> (tempfile::TempDir, MemoryStore) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("m.db");
    let s = MemoryStore::open(&path, HashEmbedder::new(64)).unwrap();
    (dir, s)
}

#[test]
fn supersede_updates_and_advances_updated_at() {
    let (_d, s) = store();
    s.add(["the user lives in NYC"], Scope::LongTerm).unwrap();
    // Rule: update the top related memory to the incoming content.
    let llm = Rule(|incoming: &str, rel: &[RelatedMemory]| {
        if let Some(r) = rel.first() {
            vec![ConsolidationOp::Update {
                id: r.id,
                content: incoming.to_string(),
            }]
        } else {
            vec![ConsolidationOp::Add]
        }
    });
    let out = s
        .consolidate(["the user lives in SF"], Scope::LongTerm, &llm)
        .unwrap();
    assert_eq!(out.updated.len(), 1);
    assert_eq!(s.count(&Scope::LongTerm).unwrap(), 1);
    let hits = s
        .search("where does the user live", Query::new(), Budget::TopK(1))
        .unwrap();
    assert_eq!(hits[0].memory.content, "the user lives in SF");
    let m = s.get(out.updated[0]).unwrap().unwrap();
    assert!(m.updated_at >= m.created_at);
}

#[test]
fn delete_removes_the_superseded_memory() {
    let (_d, s) = store();
    s.add(["obsolete fact"], Scope::LongTerm).unwrap();
    let llm = Rule(|_incoming: &str, rel: &[RelatedMemory]| {
        if let Some(r) = rel.first() {
            vec![ConsolidationOp::Delete { id: r.id }]
        } else {
            vec![ConsolidationOp::Noop]
        }
    });
    let out = s
        .consolidate(
            ["this fact supersedes and removes the old one"],
            Scope::LongTerm,
            &llm,
        )
        .unwrap();
    assert_eq!(out.deleted.len(), 1);
    assert_eq!(s.count(&Scope::LongTerm).unwrap(), 0);
}

#[test]
fn add_creates_a_new_memory_when_nothing_related() {
    let (_d, s) = store();
    let llm = Rule(|_incoming: &str, _rel: &[RelatedMemory]| vec![ConsolidationOp::Add]);
    let out = s
        .consolidate(["brand new fact"], Scope::LongTerm, &llm)
        .unwrap();
    assert_eq!(out.added.len(), 1);
    assert_eq!(s.count(&Scope::LongTerm).unwrap(), 1);
    let m = s.get(out.added[0]).unwrap().unwrap();
    assert_eq!(m.content, "brand new fact");
}

#[test]
fn noop_leaves_the_store_unchanged() {
    let (_d, s) = store();
    s.add(["unrelated existing fact"], Scope::LongTerm).unwrap();
    let llm = Rule(|_incoming: &str, _rel: &[RelatedMemory]| vec![ConsolidationOp::Noop]);
    let out = s
        .consolidate(["ignored incoming fact"], Scope::LongTerm, &llm)
        .unwrap();
    assert_eq!(out.noops, 1);
    assert!(out.added.is_empty() && out.updated.is_empty() && out.deleted.is_empty());
    assert_eq!(s.count(&Scope::LongTerm).unwrap(), 1);
}

#[test]
fn empty_ops_vec_is_treated_as_a_single_noop() {
    let (_d, s) = store();
    let llm = Rule(|_incoming: &str, _rel: &[RelatedMemory]| Vec::new());
    let out = s.consolidate(["x"], Scope::LongTerm, &llm).unwrap();
    assert_eq!(out.noops, 1);
    assert_eq!(s.count(&Scope::LongTerm).unwrap(), 0);
}

#[test]
fn outcome_accounts_for_a_mixed_batch() {
    let (_d, s) = store();
    let existing = s
        .add(["fact to update", "fact to delete"], Scope::LongTerm)
        .unwrap();
    let update_id = existing[0];
    let delete_id = existing[1];

    let llm = Rule(
        move |incoming: &str, _rel: &[RelatedMemory]| match incoming {
            "do update" => vec![ConsolidationOp::Update {
                id: update_id,
                content: "updated content".into(),
            }],
            "do delete" => vec![ConsolidationOp::Delete { id: delete_id }],
            "do add" => vec![ConsolidationOp::Add],
            _ => vec![ConsolidationOp::Noop],
        },
    );

    let out = s
        .consolidate(
            ["do update", "do delete", "do add", "do nothing"],
            Scope::LongTerm,
            &llm,
        )
        .unwrap();
    assert_eq!(out.updated, vec![update_id]);
    assert_eq!(out.deleted, vec![delete_id]);
    assert_eq!(out.added.len(), 1);
    assert_eq!(out.noops, 1);

    let updated = s.get(update_id).unwrap().unwrap();
    assert_eq!(updated.content, "updated content");
    assert!(s.get(delete_id).unwrap().is_none());
}

#[test]
fn empty_items_returns_default_outcome_without_calling_the_consolidator() {
    let (_d, s) = store();
    let llm = Rule(|_incoming: &str, _rel: &[RelatedMemory]| {
        panic!("must not be called for an empty batch")
    });
    let out = s
        .consolidate(Vec::<&str>::new(), Scope::LongTerm, &llm)
        .unwrap();
    assert!(out.added.is_empty());
    assert!(out.updated.is_empty());
    assert!(out.deleted.is_empty());
    assert_eq!(out.noops, 0);
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
fn non_finite_incoming_embedding_errors_before_reconcile() {
    let dir = tempfile::tempdir().unwrap();
    let s = MemoryStore::open(dir.path().join("m.db"), NanEmbedder).unwrap();
    let llm = Rule(|_incoming: &str, _rel: &[RelatedMemory]| {
        panic!("must not reach reconcile on a bad embedding")
    });
    let result = s.consolidate(["x"], Scope::LongTerm, &llm);
    assert!(matches!(result, Err(Error::Embed(_))));
}

#[test]
fn convergence_running_the_same_batch_twice_is_a_noop_the_second_time() {
    let (_d, s) = store();
    s.add(["the user lives in NYC"], Scope::LongTerm).unwrap();

    // A dedup-aware rule: if the most related memory already matches the
    // incoming content verbatim, there's nothing to do; otherwise supersede it.
    let llm = Rule(|incoming: &str, rel: &[RelatedMemory]| {
        if let Some(r) = rel.first() {
            if r.content == incoming {
                vec![ConsolidationOp::Noop]
            } else {
                vec![ConsolidationOp::Update {
                    id: r.id,
                    content: incoming.to_string(),
                }]
            }
        } else {
            vec![ConsolidationOp::Add]
        }
    });

    let first = s
        .consolidate(["the user lives in SF"], Scope::LongTerm, &llm)
        .unwrap();
    assert_eq!(first.updated.len(), 1);

    let before = s.get_all(Scope::LongTerm).unwrap();
    assert_eq!(before.len(), 1);

    let second = s
        .consolidate(["the user lives in SF"], Scope::LongTerm, &llm)
        .unwrap();
    assert_eq!(second.noops, 1);
    assert!(second.added.is_empty());
    assert!(second.updated.is_empty());
    assert!(second.deleted.is_empty());

    let after = s.get_all(Scope::LongTerm).unwrap();
    assert_eq!(after.len(), 1);
    assert_eq!(before[0].id, after[0].id);
    assert_eq!(before[0].content, after[0].content);
    assert_eq!(before[0].updated_at, after[0].updated_at);
}

/// A `Consolidator` double that errors on its Nth call (1-indexed), to exercise
/// mid-batch, fail-fast behavior.
struct ErrOnCall {
    fail_at: usize,
    calls: AtomicUsize,
}

impl Consolidator for ErrOnCall {
    fn reconcile(
        &self,
        _incoming: &str,
        _related: &[RelatedMemory],
    ) -> Result<Vec<ConsolidationOp>> {
        let n = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
        if n == self.fail_at {
            return Err(Error::Consolidate("synthetic reconcile failure".into()));
        }
        Ok(vec![ConsolidationOp::Add])
    }
}

#[test]
fn mid_batch_failure_is_fail_fast_and_earlier_items_stay_committed() {
    let (_d, s) = store();

    let llm = ErrOnCall {
        fail_at: 2,
        calls: AtomicUsize::new(0),
    };
    let result = s.consolidate(
        ["item one", "item two", "item three"],
        Scope::LongTerm,
        &llm,
    );
    assert!(matches!(result, Err(Error::Consolidate(_))));

    // Item 1 committed before the failure; item 3 was never reached.
    assert_eq!(s.count(&Scope::LongTerm).unwrap(), 1);
    let all = s.get_all(Scope::LongTerm).unwrap();
    assert_eq!(all[0].content, "item one");
    assert!(all.iter().all(|m| m.content != "item three"));

    // Re-running converges: the already-committed fact dedups, and the rest
    // get added, ending at exactly 3 distinct facts with no duplication.
    let llm2 = Rule(|_incoming: &str, _rel: &[RelatedMemory]| vec![ConsolidationOp::Add]);
    let out2 = s
        .consolidate(
            ["item one", "item two", "item three"],
            Scope::LongTerm,
            &llm2,
        )
        .unwrap();
    assert_eq!(out2.added.len(), 3);
    assert_eq!(s.count(&Scope::LongTerm).unwrap(), 3);
}
