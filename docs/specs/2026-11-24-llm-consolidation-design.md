# LLM Memory Consolidation — Design

**Date:** 2026-11-24
**Status:** Proposed
**Milestone:** v2 (first of the v2 subsystems — the "LLM consolidation" roadmap item)

## Intent & why

memnon is append-only today: `add` stores raw text and only exact-dedups. Two
contradictory facts ("user lives in NYC", later "user lives in SF") both persist,
and recall surfaces the stale one. Every comparable memory layer (mem0, Zep,
Letta) earns the word *memory* by **reconciling** new information against what's
stored — updating, superseding, or dropping — not just accumulating it.

This phase adds **reconciliation-only consolidation**: given a discrete incoming
fact, memnon retrieves the related existing memories and a pluggable LLM decides
whether to add it, update an existing memory, delete a now-wrong one, or do
nothing. Turning raw conversation into discrete facts ("extraction") stays the
caller's job — consistent with memnon's "SQLite for agent memory" identity (the
engine stores and reconciles rows; it doesn't parse your sentences). A caller who
wants full mem0 behavior composes their own extractor with this.

## The pluggable-seam certainty

The LLM is a **pluggable trait**, exactly like `Embedder` and `Tokenizer`:
pure-Rust core, no bundled model, **no new Cargo feature**. Testing uses a
deterministic rule/closure-based double (the `HashEmbedder` of LLMs); a real
OpenAI-backed `Consolidator` ships as a bring-your-own **example**
(`examples/openai_consolidator.rs`), mirroring `examples/openai_embedder.rs`.

## Goals

- A `Consolidator` trait (bring-your-own LLM) that, given an incoming fact and the
  top-K related existing memories, returns a list of operations.
- A `MemoryStore::consolidate(items, scope, &llm)` method: retrieve → LLM decide →
  apply, returning a `ConsolidationOutcome`. `add`/`search`/`get`/etc. unchanged.
- Mutable memories: a storage `update` path that **re-embeds** changed content via
  the pinned embedder and bumps `updated_at` (so `updated_at != created_at` now
  carries meaning).
- Deterministic, model-free tests of the whole pipeline.

## Non-goals

- **No extraction** in core (raw-conversation → facts). That's the caller's, shown
  via the example only.
- No bundled LLM, no hosted anything, no new Cargo feature.
- No graph/temporal memory (separate roadmap item).
- No batch cross-fact optimization (facts reconcile sequentially in v2).
- No conflict *history*/audit trail — update overwrites; the superseded content is
  gone (the embedder pin + `updated_at` are the only provenance).

## Anti-goals

- Don't make `add`, `search`, or `open` breaking, and don't carry the LLM as
  store-wide state (it's only needed for `consolidate`).
- Don't hold a write transaction across the (slow, network) LLM call.
- Don't pull a network/LLM dependency into the core crate.

## Constraints

- `Consolidator` is `Send + Sync`. `reconcile` is fallible (network) →
  `Error::Consolidate`.
- Re-embedding on update uses the store's pinned embedder — no second embedder, no
  dimension drift.
- No on-disk *format* change (record layout already carries `updated_at`); only the
  *semantics* of `updated_at` change, plus the type doc note.
- No `unwrap()`/`expect()` in production paths.

## Approaches considered

### Interaction model (decided: reconciliation-only)
- **A. Reconciliation-only (CHOSEN).** Caller supplies discrete facts; engine
  reconciles. Minimal, deterministic-testable, matches the embedded-engine identity;
  the hard/valuable part mem0 does (the ADD/UPDATE/DELETE decision) lives in core,
  extraction composes on top.
- **B. Extraction + reconciliation.** Dump raw turns; LLM extracts then reconciles.
  More turnkey/mem0-like but pulls prompt-engineering + conversation parsing into
  the core, heavier LLM contract, hard to test deterministically. Rejected for v2.

### API shape (decided: per-call LLM method)
- **A. `consolidate(items, scope, &llm)` method (CHOSEN).** The LLM is a per-call
  dependency (like the query string to `search`); `add`/`search` stay dependency-free.
- **B. `ConsolidatingStore::new(store, llm)` wrapper.** Keeps `MemoryStore` pristine
  but adds a second handle type + ownership story. Rejected as heavier.
- **C. `open_with_consolidator`.** Every store carries an LLM even if it never
  consolidates, coupling construction to a network dep. Rejected.

## Design

### The trait & types (new module `src/consolidator.rs`)

```rust
use crate::error::Result;
use crate::types::MemoryId;

/// An existing memory offered to the consolidator as reconciliation context.
#[non_exhaustive]
pub struct RelatedMemory {
    pub id: MemoryId,
    pub content: String,
    /// Cosine similarity to the incoming fact (how the candidate was retrieved).
    pub score: f32,
}

/// What to do about an incoming fact, relative to the related memories.
#[non_exhaustive]
pub enum ConsolidationOp {
    /// Store the incoming fact as a new memory.
    Add,
    /// Replace an existing memory's content (re-embedded) — supersede.
    Update { id: MemoryId, content: String },
    /// Remove an existing memory (now wrong/redundant).
    Delete { id: MemoryId },
    /// The incoming fact adds nothing new.
    Noop,
}

/// The pluggable LLM seam for reconciliation. Bring your own (OpenAI, local, …);
/// core ships only the trait + a deterministic test double. `Send + Sync` so it
/// can be shared across threads.
pub trait Consolidator: Send + Sync {
    /// Decide how `incoming` reconciles against `related` (most-similar first).
    /// May return several ops for one fact (e.g. update one, delete a duplicate).
    /// An empty vec is treated as a single `Noop`.
    fn reconcile(&self, incoming: &str, related: &[RelatedMemory])
        -> Result<Vec<ConsolidationOp>>;
}
```

Re-exported from `lib.rs`: `pub use consolidator::{ConsolidationOp, Consolidator, RelatedMemory};`

### The outcome

```rust
#[non_exhaustive]
pub struct ConsolidationOutcome {
    pub added: Vec<MemoryId>,
    pub updated: Vec<MemoryId>,
    pub deleted: Vec<MemoryId>,
    pub noops: usize,
}
```

### Store method (`src/store.rs`)

```rust
/// Reconcile each incoming fact against existing memories via `llm`, applying the
/// ops it returns. For each item: embed it, retrieve the top related memories in
/// `scope`, ask the LLM, then apply. `add`/`search` are unaffected.
pub fn consolidate<I, E, L>(&self, items: I, scope: Scope, llm: &L)
    -> Result<ConsolidationOutcome>
where
    I: IntoIterator<Item = E>,
    E: Into<Entry>,
    L: Consolidator + ?Sized,
```

Per-item pipeline:
1. Embed the incoming content (pinned embedder); guard finite/dimension as `add` does.
2. Retrieve related: an in-scope semantic search over the incoming embedding, top
   `CONSOLIDATE_TOP_K` (const, default **8**) with a small `min_score` floor
   (const, default **0.0** → tune), reusing the existing rank path. Build
   `RelatedMemory { id, content, score }`. (No pre-LLM exact-dedup short-circuit —
   the LLM is always called; an `Add` of an exact duplicate is absorbed by
   `insert_batch`'s existing dedup. **Resolved (builder check Q5).**)
3. `llm.reconcile(incoming, &related)?` → `Vec<ConsolidationOp>` (empty ⇒ one `Noop`).
4. **Pre-resolve & validate all embeddings OUTSIDE any transaction** — for every
   `Update{id, content}`, re-embed `content` via the pinned embedder and apply the
   same finite/dimension guards as `add` (`Error::Embed` / `Error::DimensionMismatch`).
   A bad embedding fails this item *before* any write. The `Add` op reuses the
   step-1 incoming embedding. **Resolved (builder check Q1/Q2):** embeddings can't be
   computed inside a redb write txn, so they are all resolved first.
5. **Apply this item's resolved ops in ONE write transaction** via a new storage
   primitive `storage.apply_consolidation(&[ResolvedOp])` (below), so an item that
   both updates one memory and deletes a duplicate applies atomically or not at all:
   - `Add` → insert (step-1 embedding + the entry's metadata).
   - `Update{id, pre-embedded content}` → in-place update; the op is a no-op within
     the txn if the id vanished (counts nothing).
   - `Delete{id}` → remove; no-op if already gone.
   - `Noop` → nothing.
6. Accumulate ids into the running `ConsolidationOutcome`.

Items reconcile **sequentially** — a later fact sees the state left by earlier ones.
After the whole batch, invalidate the HNSW index **once** if any `Update`/`Delete`
was applied across the call; an `Add`-only call extends the index incrementally like
`add`. **Resolved (builder check Q4):** per-call granularity — consolidate's own
retrieval is scope-filtered and never consults the index, so this only governs what a
concurrent reader sees and how much incremental-add work a mid-batch invalidation discards.

### Storage (`src/storage.rs`)

A single-transaction primitive applies one item's resolved ops atomically. Every
embedding is already computed and validated by the caller (pipeline step 4), so the
txn does no embedding work and can't fail on a bad vector mid-commit:

```rust
/// A consolidation op with its embedding already resolved + validated by the store.
pub(crate) enum ResolvedOp {
    Add { content: String, metadata: HashMap<String, String>, embedding: Vec<f32> },
    Update { id: MemoryId, content: String, embedding: Vec<f32> },
    Delete { id: MemoryId },
}

/// Apply all ops for one incoming fact in a single write transaction. Mutates
/// MEMORIES + DEDUP for each op and commits once (atomic per item). Add mints a new
/// id (reusing the next-id counter + exact-dedup, like insert_batch); Update rewrites
/// content/embedding in place (preserving id/scope/metadata/created_at, bumping
/// updated_at, rewriting the dedup key) and is a no-op if the id is gone; Delete
/// removes the row + its dedup entry, no-op if gone. Returns which ids were
/// added/updated/deleted so the store can fold them into the outcome.
pub(crate) fn apply_consolidation(&self, ops: &[ResolvedOp], now: u64)
    -> Result<AppliedIds>  // { added: Vec<MemoryId>, updated: Vec<MemoryId>, deleted: Vec<MemoryId> }
```

This reuses the existing record encode/decode, the `next_id` counter, and the dedup
key helper (`dedup_key(scope, content)`); it is the consolidation-era sibling of
`insert_batch`. Dedup-collision edge (an `Update`'s new content already keys another
row in-scope): last-writer-wins on the dedup pointer — documented; the LLM is expected
to express true merges as `Update` + `Delete`. `storage.delete` already exists and is
reused as-is for non-consolidation callers; `apply_consolidation` handles Delete
inside the shared txn rather than calling it, to keep the item atomic.

### Errors (`src/error.rs`)

Add to the `#[non_exhaustive]` enum: `Consolidate(String)` with Display
`consolidation failed: {m}`. Raised by `reconcile` failures.

### Types (`src/types.rs`)

Update the `Memory` doc: drop "`updated_at == created_at` in v1"; note `updated_at`
advances when a memory is superseded by consolidation, while `created_at` is fixed.

### Concurrency / atomicity

`retrieve (read) → reconcile (no lock, network) → resolve embeddings → apply (one
write txn per item)`. Ids can vanish between retrieve and apply under concurrent
writers; apply tolerates this (missing id ⇒ that op is skipped, not an error).
Overlapping concurrent `consolidate` calls are last-writer-wins — acceptable and
documented for the embedded single-process model; no cross-call locking.

### Batch failure & convergence (builder-check Q3, resolved)

Items are committed per item, sequentially. If `reconcile` (or an embedding guard)
fails on item *k*, `consolidate` returns `Err(Error::Consolidate…)`/`Err(Error::Embed…)`
immediately; items `0..k` are **already durably applied**, item *k* applied nothing
(its embeddings are validated before its txn opens), and `k+1..` are untouched. The
signature stays `Result<ConsolidationOutcome>` — no partial-outcome-in-error type.

The caller's recovery is simply to **re-run the same batch**: consolidation is
*convergent*. A fact applied on the first run is now in the store, so on re-run it
appears in that fact's `related` set and the LLM reconciles it to `Noop` (or an
exact-dup `Add` is absorbed by dedup). Re-running therefore does not double-apply —
it re-reconciles against the now-current state and no-ops the already-done work. This
convergence is the documented contract that makes a mid-batch failure safe to retry,
and it is why per-item (not per-batch) atomicity is sufficient.

### Example & docs

- `examples/openai_consolidator.rs`: a `Consolidator` backed by an OpenAI
  chat-completions call, prompting the model to return JSON ops parsed into
  `Vec<ConsolidationOp>`. Bring-your-own; compile-checked in CI, not run (mirrors
  `openai_embedder.rs`, `ureq` dev-dep already present).
- README: document consolidation, the `Consolidator` seam, and the extraction-is-
  yours boundary; add it to the roadmap (check off "LLM consolidation" under v2).

## Testing

- **Unit (storage):** `update` round-trips (content+embedding change, `created_at`
  preserved, `updated_at` advanced, dedup rewritten so old content re-adds fresh and
  new content dedups); `update` on a missing id returns false.
- **Integration (`tests/consolidate.rs`), model-free** via a rule/closure-based
  `Consolidator` double:
  - supersede: add "lives in NYC"; consolidate "lives in SF" with a rule that
    `Update`s the most-similar → one memory, content "SF", `updated_at > created_at`,
    recall returns SF not NYC.
  - delete: a rule that `Delete`s a contradicted memory removes it.
  - add: a rule that returns `Add` (or empty-related) inserts a new memory.
  - noop: a `Noop` rule leaves the store unchanged.
  - outcome accounting: `ConsolidationOutcome` counts match applied ops.
  - vanished-id tolerance: a rule referencing a deleted id doesn't error.
  - atomicity: an item whose rule returns `Update` + `Delete` applies both (one memory
    rewritten, the duplicate gone) — exercises `apply_consolidation`'s single txn.
  - convergence/retry: running the same consolidate batch twice is a no-op the second
    time (re-reconciled facts land as `Noop`/dedup; store state identical).
  - mid-batch failure: a rule that `Err`s on the 2nd of 3 items returns `Err`, leaves
    item 1 applied and item 3 untouched; re-running then converges.
- Deterministic doubles only — no network, no `#[ignore]`.

## Left to the builder

- Final `CONSOLIDATE_TOP_K` / `min_score` defaults (start 8 / 0.0) and whether to
  expose them via a `ConsolidateOptions` now or defer (bias: defer; const defaults).
- Whether `reconcile` takes incoming metadata too (bias: no for v2 — content only).
- The test double's exact shape (closure-backed `Consolidator` impl in the test
  module vs. a shipped `FnConsolidator` helper; bias: test-local, keep core minimal).
- OpenAI example's exact prompt/JSON schema for ops.
