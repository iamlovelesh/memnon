# Design: memnon — Token-Budget-First, Local-First Agent Memory Engine (Rust)

**Status:** Approved, revised after staff-level design review (2026-11-18)
**Name:** `memnon`

> **Revision note (post-review).** A four-lens review (API/data-model, Rust
> correctness/concurrency, product/adoption, testing/ops) found several
> issues that are cheap now and irreversible later. This revision folds in the
> engineering fixes and adopts the **learning-first** framing. Material changes:
> on-disk **schema versioning**; **embedder injected at construction** (removes a
> silent-garbage race and the `NoEmbedder` class); `add` carries **metadata**;
> `search` returns **scored `Hit`s**; **transactional batch writes**; defined
> **zero-norm cosine**; **scope teardown / introspection** methods;
> `#[non_exhaustive]` public enums; and a reframed identity around
> **token-budgeted retrieval**.

---

## 0. What this is, honestly (learning-first)

memnon is **primarily a learning project** — its author is using it to learn real
Rust (ownership, concurrency, embedded storage, serialization) and real AI
engineering (context management, memory, subagents) by building the internals
rather than importing them. Its **secondary** goal is to be a genuinely useful,
star-worthy OSS library.

We are explicit about reach: a `cargo add` Rust library serves teams whose agent
is *already* in Rust — a real but small slice of the AI world (desktop/Tauri,
on-device/edge, infra/CLI). Broad, cross-language adoption ("SQLite for agent
memory" in the full sense) requires the **sidecar + language clients**, which are
a deliberate post-v1 path. **v1 is built to be sidecar-ready, and we do not claim
broad adoption it hasn't earned.**

## 1. Intent & the differentiator

Agents forget, and context windows overflow — blowing cost and reliability. A
funded market exists (Mem0, Letta, Zep), but it is **Python + cloud-first**.
memnon's wedge is being **embeddable, local-first, offline, in-process** — which
Rust enables and Python can't.

But "embedded vector store with scopes" is nearly a commodity (LanceDB,
`sqlite-vec`, `usearch` exist). So memnon's **identity is the one non-commodity
thing it does**:

> **Token-budget-first recall.** You hand memnon your model's context-window
> budget; it returns *exactly the memories that fit*, most-relevant first, with
> their scores — offline and in-process. Not "top-k," but "what fits the window."

That is a named context-engineering problem nobody packages, and it is also the
richest part of the learning goal.

### Why not just LanceDB / sqlite-vec + 20 lines of cosine?

memnon is **not a vector DB**; it is a thin, opinionated **memory layer** on top
of that idea: scoped recall, a Mem0-shaped API, token-budget packing, and
embedder-pinned files — designed to grow a sidecar so non-Rust teams can use it.
For raw ANN at scale, use LanceDB. memnon is the memory *semantics* and the
context-window packing, not the index.

## 2. Goals / Non-Goals / Anti-Goals

### Goals
- A Rust **library** (`cargo add memnon`), embeddable in-process.
- Fully functional **offline**, from a **single embedded file**.
- **Token-budgeted recall** as the headline capability.
- Mem0-shaped API: `add` / `search` / `get` / `get_all` / `delete` / `clear`.
- Layered memory via **scopes** (`Session`, `LongTerm`, `Custom`).
- Permissive license (**Apache-2.0 OR MIT**).
- **Forward-compatible on disk** — files written by v1 remain readable by v1.x.

### Non-Goals (v1)
- No cloud service / SaaS / multi-tenant hosting.
- Not a general vector database (no competition with LanceDB/Qdrant).
- No server/sidecar or language bindings yet (post-v1; v1 is *built for* them).
- No graph/temporal memory.
- No **LLM-based** consolidation (ADD/UPDATE/DELETE reasoning) — v2. v1 does
  cheap, non-LLM **exact-duplicate dedup** only (see §5).

### Anti-Goals (failure even if it "works")
- Requiring any network or external process to function. Offline or it failed.
- A mandatory C/C++ toolchain for the **core** crate (ONNX lives only in examples).
- **Silent wrong recall.** Any condition that could return garbage (mismatched
  embedder, zero-norm vector, corrupt/old file) must **error**, never mislead.
- Shipping a schema-less on-disk format that orphans early users' data.

## 3. Constraints
- Core crate: pure-Rust deps only; no mandatory native/C deps.
- On-disk format **versioned** from the first byte written.
- Public API uses `#[non_exhaustive]` on enums so new variants aren't breaking.
- `MemoryStore` is `Clone + Send + Sync`; all mutation of shared state is atomic.
- Small enough for a solo author to finish v1.

## 4. Approaches Considered

(Unchanged in conclusion; see history.) Problem: **P1 memory engine** chosen over
P2 runtime / P3 observability. Positioning: **embeddable/local-first** over
cloning incumbents. Embeddings: **pluggable `Embedder`** (approach ②) with a
real local example. Recall index: **exact brute-force cosine** in v1; HNSW is the
first post-v1 milestone. Serialization: **versioned envelope** (below) over naked
`bincode` — the review's decisive correction.

## 5. The Design (v1, post-review)

### Public API
```rust
// Construction: the embedder is REQUIRED here. Its identity + dimension are
// pinned into the file atomically on first open; a later open with a mismatched
// embedder errors. (No `NoEmbedder` runtime state; no post-open race.)
let store = MemoryStore::open("./memory.db", embedder)?;  // embedder: impl Embedder + 'static

// Write: each item may carry metadata. Append-only beyond exact-dedup.
// Returns ids 1:1 with input order. All items commit in ONE transaction.
let ids = store.add(
    [
        Entry::new("user prefers dark mode").meta("source", "settings"),
        Entry::new("user is on the pro plan"),            // &str: Into<Entry>, no metadata
    ],
    Scope::LongTerm,
)?;

// Recall: returns SCORED hits, budget-packed, optionally floored by score.
let hits: Vec<Hit> = store.search(
    "what theme does the user like?",
    Query::new().scope(Scope::LongTerm).filter("source", "settings").min_score(0.2),
    Budget::Tokens(2000),  // the memories that FIT 2000 tokens, best-first
)?;
for Hit { memory, score } in &hits { /* ... */ }

// Lifecycle / introspection
let one   = store.get(ids[0])?;             // Option<Memory>
let all   = store.get_all(Scope::LongTerm)?;
store.delete(ids[0])?;
store.clear(&Scope::Session)?;              // wipe a scope (session/subagent teardown)
let n     = store.count(&Scope::LongTerm)?; // usize
let scopes = store.list_scopes()?;          // Vec<Scope>
```

### Types
- `Entry { content, metadata }` — write-side item. `Entry::new(text).meta(k, v)`;
  `String`/`&str` implement `Into<Entry>` for the no-metadata path.
- `Hit { memory: Memory, score: f32 }` — read-side result; the cosine score is
  surfaced, never discarded.
- `Memory { id, scope, content, embedding, metadata, created_at, updated_at }` —
  `updated_at == created_at` in v1 (append-only). `MemoryId` is **opaque**
  (private field) so a v2 move to content-hash ids isn't a breaking format change.
- `Scope { Session, LongTerm, Custom(String) }` — a subagent is
  `Scope::Custom("subagent:<id>")`.
- `Budget { TopK(usize), Tokens(usize) }`; `Query` builder with `scope`, `filter`,
  `min_score`.

### Embedder (pluggable, required at construction)
```rust
pub trait Embedder: Send + Sync {
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;  // &str, non-allocating
    fn dimensions(&self) -> usize;
    fn identity(&self) -> &str;   // pinned to the file; mismatch on reopen => error
}
```
A returned vector whose length ≠ `dimensions()` is an `Error::DimensionMismatch`,
never a silent mis-rank or a slice panic.

### On-disk format (the irreversible decision)
- The `meta` table stores a `schema_version: u32` on create.
- Every record is a **versioned envelope** (a leading version tag) around the
  serialized `Memory`, so reads branch on version and future field changes
  migrate instead of corrupting. `open` rejects an unknown/newer `schema_version`
  with `Error::UnsupportedSchema`.
- `Scope` is serialized by **name**, not enum discriminant, so adding/reordering
  variants never remaps old `Custom` rows.

### Writes: transactional, deduped, append-only
- `add` of N items: allocate N ids and write N rows in **one** `redb` write
  transaction — all-or-nothing. Partial failure leaves no orphan rows.
- **Exact-dedup:** identical `(scope, content)` returns the existing `MemoryId`
  instead of inserting a duplicate (cheap, no LLM).
- **Append-only, named loudly:** conflicting facts ("likes dark" then "likes
  light") are *both* retained and ranked by similarity, with a `created_at`
  tiebreak favoring recency. memnon does **not** resolve contradictions in v1;
  LLM consolidation and latest-wins-per-key are v2. This limit is stated in the
  README so users aren't surprised.

### Recall: defined, scored, single-load
- `cosine` is defined to return **`0.0`** when either norm is `0.0` (no `NaN`).
- `rank` sorts by score desc, `created_at` desc as tiebreak (`f32::total_cmp`).
- `pack`: `TopK(k)` → first k; `Tokens(max)` → greedily include while cumulative
  approx-tokens ≤ max (empty if the top hit alone exceeds max); `min_score` drops
  weak matches first.
- Ranking reads a light `(MemoryId, Vec<f32>, created_at)` projection; full
  `Memory` rows are materialized **only** for the final packed set (no double-load).

### Errors (`#[non_exhaustive]`)
`Io`, `Corrupt`, `UnsupportedSchema`, `EmbedderMismatch`, `DimensionMismatch`,
`Embed`, `NotFound`. Typed with `source()` chaining (implementation may use
`thiserror`). There is no `NoEmbedder` — the embedder is required at construction.

### Concurrency
`MemoryStore { inner: Arc<Inner> }`; cheap clone, `Send + Sync`. The embedder is
immutable after construction (no post-open mutation, so no identity-pin race).
`redb` provides single-writer transactions; `next_id` + batch insert happen inside
one write txn, so concurrent `add`s produce unique ids.

### Concepts exercised by v1
- **Memory:** scopes + semantic recall + persistence + exact-dedup. ✓
- **Context management:** token-budgeted, score-floored retrieval. ✓ (the headline)
- **Subagents:** `Scope::Custom` + `clear(scope)` give per-subagent isolation and
  teardown; LLM-driven subagent orchestration remains v2.

## 6. Left to the Builder (non-blocking)
- Embedded store engine — lean **`redb`** (pure-Rust, single-file). Alt: SQLite.
- Dedup index representation (secondary `(scope, content-hash) → id` table vs scan).
- Tokenizer — approximate `chars/4` in v1, **documented as rough** (esp. for CJK);
  real tokenizer later.
- `thiserror` vs hand-rolled error; module layout.

## 7. Roadmap
- **v1** — library; token-budgeted scored recall; scopes; exact-dedup;
  versioned single-file store; local + API embedder examples; benchmark + CI.
- **v1.1** — HNSW index; bundled default local embedder (zero-setup offline demo);
  real tokenizer.
- **v2** — sidecar binary (REST / AG-UI) + Python/TS clients (the adoption path);
  LLM consolidation (update/supersede conflicting facts); subagent orchestration.
- **Later** — graph/temporal memory; eval harness vs LongMemEval-style benchmarks.
