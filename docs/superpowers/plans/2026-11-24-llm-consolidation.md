# LLM Memory Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add reconciliation-only LLM consolidation so new facts update/supersede/delete existing memories instead of blindly appending.

**Architecture:** A pure-Rust `Consolidator` trait (bring-your-own LLM, mirroring `Embedder`/`Tokenizer`) + a `MemoryStore::consolidate(items, scope, &llm)` pipeline (embed → retrieve related → LLM decides ops → pre-resolve embeddings → apply per-item in one write txn). Memories become mutable via a new atomic `storage.apply_consolidation` primitive. No on-disk format change; `add`/`search`/`open` stay non-breaking.

**Tech Stack:** Rust (edition 2024), `redb`, existing `HashEmbedder` for tests, `ureq` (dev-dep) for the OpenAI example.

**Spec:** `docs/specs/2026-11-24-llm-consolidation-design.md`

## Global Constraints

- `add`, `search`, `open`, and the on-disk format/schema are unchanged (non-breaking). The LLM is a per-call argument to `consolidate`, never store-wide state.
- `Consolidator` is a pure-Rust trait, `Send + Sync`; **no new Cargo feature**, no bundled LLM. `reconcile` is fallible → `Error::Consolidate`.
- Re-embedding on `Update` uses the store's pinned embedder; the same finite/dimension guards as `add` apply (`Error::Embed` / `Error::DimensionMismatch`), enforced **before** any write transaction opens.
- Each incoming item's ops apply in **one write transaction** (`storage.apply_consolidation`) — atomic per item.
- Items reconcile **sequentially** (a later fact sees earlier facts' effects). Batch failure is fail-fast; prior items stay committed; re-running is safe because consolidation is **convergent** (re-seen facts reconcile to `Noop`/dedup).
- No `unwrap()`/`expect()` in production paths. `cargo clippy --all-targets -- -D warnings` and `cargo fmt --check` pass; test output pristine.
- Operation set is exactly `Add` / `Update{id, content}` / `Delete{id}` / `Noop`. Default `CONSOLIDATE_TOP_K = 8`, `min_score` floor `0.0`.

## Review Focus

- **Empty `items`** — `consolidate` is a no-op returning an empty `ConsolidationOutcome`, mirroring `add`'s empty-input short-circuit (Task 3).
- **Non-finite / wrong-dimension embedding** — incoming *or* an `Update`'s re-embedded content; `consolidate` returns `Error::Embed`/`Error::DimensionMismatch` before any write (both reuse `add`'s guard; tested via a NaN embedder on the incoming path in Task 3, which is the same code path).
- **Convergence** — running the same batch twice applies nothing the second time; store state is identical (Task 3).
- **Mid-batch failure** — a `reconcile` error on item *k* returns `Err`, leaves `0..k` durably applied and `k+1..` untouched; a re-run converges (Task 3).

---

### Task 1: `Consolidator` trait, ops, outcome, error

**Files:**
- Create: `src/consolidator.rs`
- Modify: `src/error.rs` (variant + Display), `src/lib.rs` (`pub mod` + re-exports)
- Test: unit tests in `src/consolidator.rs`

**Interfaces:**
- Consumes: `crate::types::MemoryId`, `crate::error::{Error, Result}`.
- Produces:
  - `#[non_exhaustive] pub struct RelatedMemory { pub id: MemoryId, pub content: String, pub score: f32 }` with `pub(crate) fn new(id: MemoryId, content: String, score: f32) -> Self` (store builds these; external `Consolidator` impls only read the fields).
  - `#[non_exhaustive] pub enum ConsolidationOp { Add, Update { id: MemoryId, content: String }, Delete { id: MemoryId }, Noop }` (external impls construct these — enum-level `#[non_exhaustive]` still allows constructing existing variants).
  - `pub trait Consolidator: Send + Sync { fn reconcile(&self, incoming: &str, related: &[RelatedMemory]) -> Result<Vec<ConsolidationOp>>; }`
  - `#[non_exhaustive] #[derive(Debug, Default)] pub struct ConsolidationOutcome { pub added: Vec<MemoryId>, pub updated: Vec<MemoryId>, pub deleted: Vec<MemoryId>, pub noops: usize }`
  - `Error::Consolidate(String)`; `lib.rs` re-exports `consolidator::{ConsolidationOp, ConsolidationOutcome, Consolidator, RelatedMemory}`.

- [ ] **Step 1: Write the failing unit tests** in `src/consolidator.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::MemoryId;

    // A closure-backed double proves the trait is implementable and ops construct.
    struct Rule<F>(F);
    impl<F: Fn(&str, &[RelatedMemory]) -> Vec<ConsolidationOp> + Send + Sync> Consolidator for Rule<F> {
        fn reconcile(&self, incoming: &str, related: &[RelatedMemory]) -> crate::error::Result<Vec<ConsolidationOp>> {
            Ok((self.0)(incoming, related))
        }
    }

    #[test]
    fn double_returns_ops_and_reads_related() {
        let r = RelatedMemory::new(MemoryId::new(7), "old".into(), 0.9);
        let llm = Rule(|_in: &str, rel: &[RelatedMemory]| vec![ConsolidationOp::Update { id: rel[0].id, content: "new".into() }]);
        let ops = llm.reconcile("incoming", std::slice::from_ref(&r)).unwrap();
        assert!(matches!(ops[0], ConsolidationOp::Update { .. }));
        assert_eq!(r.score, 0.9);
    }

    #[test]
    fn outcome_default_is_empty() {
        let o = ConsolidationOutcome::default();
        assert!(o.added.is_empty() && o.noops == 0);
    }
}
```

- [ ] **Step 2: Run to verify failure** — `cargo test --lib consolidator` → FAIL (module/types absent).
- [ ] **Step 3: Implement** `src/consolidator.rs` (the types + trait above), add `Error::Consolidate(String)` to `src/error.rs` with Display arm `consolidation failed: {m}`, and `pub mod consolidator;` + the re-export to `src/lib.rs` (alphabetical, matching existing style).
- [ ] **Step 4: Run to verify pass** — `cargo test --lib consolidator` → PASS; `cargo clippy --all-targets -- -D warnings` + `cargo fmt --check` clean.
- [ ] **Step 5: Commit** — `feat: add Consolidator trait, ops, and ConsolidationOutcome`

---

### Task 2: Storage `apply_consolidation` (mutable memories)

**Files:**
- Modify: `src/storage.rs` (new types + method), `src/types.rs` (`updated_at` doc note)
- Test: unit tests in `src/storage.rs` `#[cfg(test)]`

**Interfaces:**
- Consumes: `MemoryId`, existing `MEMORIES`/`DEDUP` tables, `KEY_NEXT_ID`, `dedup_key`, `encode_record`/`decode_record`, `read_u64`.
- Produces:
  - `pub(crate) enum ResolvedOp { Add { content: String, metadata: HashMap<String, String>, embedding: Vec<f32> }, Update { id: MemoryId, content: String, embedding: Vec<f32> }, Delete { id: MemoryId } }`
  - `pub(crate) struct AppliedIds { pub added: Vec<MemoryId>, pub updated: Vec<MemoryId>, pub deleted: Vec<MemoryId> }`
  - `pub(crate) fn apply_consolidation(&self, scope: &Scope, ops: &[ResolvedOp], now: u64) -> Result<AppliedIds>` — one `begin_write()` txn applying every op, one `commit()`:
    - `Add` → mint next id (reuse `KEY_NEXT_ID` counter), check `DEDUP` (`dedup_key(scope, content)`): on hit push the existing id to `added` and insert nothing; else write record `{id, scope, content, embedding, metadata, created_at: now, updated_at: now}` + dedup entry, push new id to `added`.
    - `Update{id, content, embedding}` → read record by id; if absent, skip. Else remove old `dedup_key(scope, old_content)`, write record `{same id/scope/metadata/created_at, new content/embedding, updated_at: now}`, insert `dedup_key(record.scope, content)`, push id to `updated`.
    - `Delete{id}` → read record; if absent skip; else remove record + its dedup entry, push id to `deleted`.

- [ ] **Step 1: Write the failing unit tests** in `src/storage.rs` tests:

```rust
#[test]
fn apply_consolidation_add_update_delete_in_one_txn() {
    let (_d, s) = temp_store();
    // seed two rows via insert_batch
    let ids = only_ids(s.insert_batch(&[
        item(Scope::LongTerm, "lives in NYC", vec![1.0], 10),
        item(Scope::LongTerm, "dup", vec![1.0], 10),
    ]).unwrap());
    let applied = s.apply_consolidation(
        &Scope::LongTerm,
        &[
            ResolvedOp::Update { id: ids[0], content: "lives in SF".into(), embedding: vec![2.0] },
            ResolvedOp::Delete { id: ids[1] },
            ResolvedOp::Add { content: "new fact".into(), metadata: HashMap::new(), embedding: vec![3.0] },
        ],
        99,
    ).unwrap();
    assert_eq!(applied.updated, vec![ids[0]]);
    assert_eq!(applied.deleted, vec![ids[1]]);
    assert_eq!(applied.added.len(), 1);
    let m = s.get(ids[0]).unwrap().unwrap();
    assert_eq!(m.content, "lives in SF");
    assert_eq!(m.embedding, vec![2.0]);
    assert_eq!(m.created_at, 10);      // preserved
    assert_eq!(m.updated_at, 99);      // advanced
    assert!(s.get(ids[1]).unwrap().is_none());
    // dedup rewritten: old content re-adds fresh, new content dedups
    let re = only_ids(s.insert_batch(&[item(Scope::LongTerm, "lives in NYC", vec![1.0], 1)]).unwrap());
    assert_ne!(re[0], ids[0]);
}

#[test]
fn apply_consolidation_tolerates_missing_ids() {
    let (_d, s) = temp_store();
    let applied = s.apply_consolidation(
        &Scope::LongTerm,
        &[ResolvedOp::Update { id: MemoryId::new(999), content: "x".into(), embedding: vec![1.0] },
          ResolvedOp::Delete { id: MemoryId::new(1000) }],
        1,
    ).unwrap();
    assert!(applied.updated.is_empty() && applied.deleted.is_empty());
}
```

- [ ] **Step 2: Run to verify failure** — `cargo test --lib storage::tests::apply_consolidation` → FAIL.
- [ ] **Step 3: Implement** `ResolvedOp`, `AppliedIds`, and `apply_consolidation` in `src/storage.rs` per the interface. Update `src/types.rs` `Memory` doc: drop "`updated_at == created_at` in v1"; note `updated_at` advances when a memory is superseded by consolidation while `created_at` is fixed.
- [ ] **Step 4: Run to verify pass** — the two tests + full `cargo test --lib` → PASS; clippy + fmt clean.
- [ ] **Step 5: Commit** — `feat: add atomic apply_consolidation storage primitive (mutable memories)`

---

### Task 3: `MemoryStore::consolidate` pipeline

**Files:**
- Modify: `src/store.rs` (new method + consts)
- Test: `tests/consolidate.rs`

**Interfaces:**
- Consumes: `Consolidator`, `ConsolidationOp`, `RelatedMemory`, `ConsolidationOutcome` (Task 1); `ResolvedOp`, `AppliedIds`, `apply_consolidation` (Task 2); existing `embedder`, `storage.candidates`, `index::rank`, `now_millis`, `invalidate_index`, `is_finite`, `index_write`/incremental insert.
- Produces:
  - `pub fn consolidate<I, E, L>(&self, items: I, scope: Scope, llm: &L) -> Result<ConsolidationOutcome> where I: IntoIterator<Item = E>, E: Into<Entry>, L: Consolidator + ?Sized`
  - Private consts `const CONSOLIDATE_TOP_K: usize = 8;` and `const CONSOLIDATE_MIN_SCORE: f32 = 0.0;`

  Per-item body: collect `Entry`s; empty ⇒ return `ConsolidationOutcome::default()`. For each entry: embed its content (guard `is_finite` + dimension, same errors as `add`); retrieve related = `storage.candidates(Some(&scope), &[])` → `index::rank(incoming, &candidates-as-(id,emb,created_at))` → take `CONSOLIDATE_TOP_K` with score `>= CONSOLIDATE_MIN_SCORE`, map to `RelatedMemory::new(id, content, score)`; `llm.reconcile(content, &related)?` (empty ⇒ treat as `[Noop]`); build `Vec<ResolvedOp>`: `Add` → `ResolvedOp::Add { content, metadata, embedding: incoming_emb }`, `Update{id,content}` → re-embed `content` (guard finite/dim) → `ResolvedOp::Update{..}`, `Delete{id}` → `ResolvedOp::Delete{id}`, `Noop` → increment `outcome.noops`, no resolved op; call `storage.apply_consolidation(&scope, &resolved, now_millis())`; fold `AppliedIds` into the running `ConsolidationOutcome`. After all items: if any `updated`/`deleted` occurred, `self.invalidate_index()`; else incrementally insert the added ids' embeddings into a built index (or leave to next search).

- [ ] **Step 1: Write the failing integration tests** in `tests/consolidate.rs` using a closure-backed `Consolidator` double (and an erroring one). Cover: supersede (Update most-similar → content changed, `updated_at > created_at`, recall returns new not old), delete, add, noop-leaves-unchanged, outcome accounting, empty-items no-op, non-finite incoming embedding → `Error::Embed`, convergence (run same batch twice → second is all-noop, counts/store identical), mid-batch failure (erroring double on 2nd of 3 items → `Err`, item 1 applied, item 3 absent; re-run converges). Example core:

```rust
use memnon::{Budget, ConsolidationOp, Consolidator, HashEmbedder, MemoryStore, Query, RelatedMemory, Result, Scope};

struct Rule<F>(F);
impl<F: Fn(&str, &[RelatedMemory]) -> Vec<ConsolidationOp> + Send + Sync> Consolidator for Rule<F> {
    fn reconcile(&self, incoming: &str, related: &[RelatedMemory]) -> Result<Vec<ConsolidationOp>> {
        Ok((self.0)(incoming, related))
    }
}

#[test]
fn supersede_updates_and_advances_updated_at() {
    let dir = tempfile::tempdir().unwrap();
    let s = MemoryStore::open(dir.path().join("m.db"), HashEmbedder::new(64)).unwrap();
    s.add(["the user lives in NYC"], Scope::LongTerm).unwrap();
    // Rule: update the top related memory to the incoming content.
    let llm = Rule(|incoming: &str, rel: &[RelatedMemory]| {
        if let Some(r) = rel.first() { vec![ConsolidationOp::Update { id: r.id, content: incoming.to_string() }] }
        else { vec![ConsolidationOp::Add] }
    });
    let out = s.consolidate(["the user lives in SF"], Scope::LongTerm, &llm).unwrap();
    assert_eq!(out.updated.len(), 1);
    assert_eq!(s.count(&Scope::LongTerm).unwrap(), 1);
    let hits = s.search("where does the user live", Query::new(), Budget::TopK(1)).unwrap();
    assert_eq!(hits[0].memory.content, "the user lives in SF");
    let m = s.get(out.updated[0]).unwrap().unwrap();
    assert!(m.updated_at >= m.created_at);
}
```

- [ ] **Step 2: Run to verify failure** — `cargo test --test consolidate` → FAIL (`consolidate` not found).
- [ ] **Step 3: Implement** `consolidate` + the two consts in `src/store.rs` per the interface.
- [ ] **Step 4: Run to verify pass** — `cargo test --test consolidate` + full `cargo test` → PASS; clippy `--all-targets -D warnings` + fmt clean.
- [ ] **Step 5: Commit** — `feat: MemoryStore::consolidate — reconcile facts via a pluggable LLM`

---

### Task 4: OpenAI example + docs

**Files:**
- Create: `examples/openai_consolidator.rs`
- Modify: `README.md`

**Interfaces:**
- Consumes: `Consolidator`, `ConsolidationOp`, `RelatedMemory`, `consolidate` (Tasks 1,3). Uses `ureq` + `serde_json` (existing dev-deps). No feature.

- [ ] **Step 1: Write `examples/openai_consolidator.rs`** — an `OpenAiConsolidator` implementing `Consolidator` by POSTing incoming + related to an OpenAI chat-completions endpoint, prompting for a JSON array of ops, parsed into `Vec<ConsolidationOp>` (map parse/network failures to `Error::Consolidate`). `main` gated on `OPENAI_API_KEY` (print-and-return if unset), mirroring `examples/openai_embedder.rs`. Doc comment: this is bring-your-own; extraction (conversation → facts) is the caller's job.
- [ ] **Step 2: Verify it builds** — `cargo build --examples` → PASS (base job, no features).
- [ ] **Step 3: Update `README.md`** — a consolidation section (the `Consolidator` seam, `consolidate` usage with a short snippet, and the explicit "extraction is yours" boundary); check off "LLM consolidation" under v2 in the roadmap.
- [ ] **Step 4: Full verification** — `cargo fmt --check`, `cargo test`, `cargo clippy --all-targets -- -D warnings`, `cargo build --examples` → all clean.
- [ ] **Step 5: Commit** — `docs: openai consolidator example and README consolidation section`

---

## Notes for the executor

- Tasks are sequential by interface: T2 needs T1 (`MemoryId`, `Error`); T3 needs T1+T2; T4 needs T1+T3.
- The retrieval in T3 reuses the *scoped exact* path (`storage.candidates(Some(&scope), …)` + `index::rank`), consistent with how `search` always takes the brute-force path for scoped queries — the HNSW index is never consulted during consolidate's own retrieval.
- Convergence (T3's retry test) is the contract that makes per-item atomicity sufficient; if that test can't be made to pass, stop and raise it — it signals the reconcile/apply loop isn't idempotent against already-applied state.
