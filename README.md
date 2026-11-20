# memnon

**A token-budget-first memory engine for AI agents.** Hand it your model's
context-window budget; it returns exactly the memories that fit — most-relevant
first, with scores — **offline and in-process.**

> ⚠️ **Status: design / API skeleton.** The public API in `src/lib.rs` compiles
> but is **not implemented** (`todo!()` bodies). This repo currently exists to
> review the shape. Implementation follows the TDD plan in `docs/plans/`.
>
> ℹ️ **This is a learning-first project.** Its author is using it to learn Rust
> and AI engineering by building the internals. It aims to become a genuinely
> useful OSS library, but v1 is a `cargo add` Rust library — broad, cross-language
> adoption is a deliberate later step (see *Honest scope*).

---

## What makes it different

An embedded vector store with scopes is nearly a commodity (LanceDB, `sqlite-vec`,
`usearch`). memnon's one non-commodity idea is **token-budget-first recall**:

```rust
let hits = store.search(
    "what theme does the user like?",
    Query::new().scope(Scope::LongTerm).min_score(0.2),
    Budget::Tokens(2000),   // not "top 5" — "whatever best fits 2000 tokens"
)?;
```

You don't guess `k`. You state the budget your context window has, and memnon
returns the most relevant memories that *fit it*, with their similarity scores.
That's a named context-engineering problem nobody else packages.

## Why local-first

- **Embeddable** — a Rust library you `cargo add` and call in-process.
- **Local-first** — fully offline; one embedded file; no network required.
- **No sidecars** — no Python stack, no separate vector database.
- **Private by default** — data never leaves the process unless you make it.

## Usage (v1)

```rust
use memnon::{MemoryStore, Entry, Scope, Query, Budget, Hit};

// The embedder is required at construction and pinned to the file.
let store = MemoryStore::open("./memory.db", embedder)?;

let ids = store.add(
    [
        Entry::new("user prefers dark mode").meta("source", "settings"),
        Entry::new("user is on the pro plan"),         // &str works too
    ],
    Scope::LongTerm,
)?;

let hits: Vec<Hit> = store.search(
    "what theme does the user like?",
    Query::new().scope(Scope::LongTerm),
    Budget::Tokens(2000),
)?;

store.clear(&Scope::Session)?;   // wipe a scope (session / subagent teardown)
```

**v1 exercises the target concepts end-to-end:**

- **Context management** — token-budgeted, score-floored retrieval (*the headline*).
- **Memory** — layered scopes (`Session`, `LongTerm`, `Custom`) + semantic recall
  + persistence + exact-duplicate dedup.
- **Subagents** — `Scope::Custom("subagent:…")` + `clear(scope)` give per-subagent
  isolation and teardown. LLM-driven orchestration is v2.

## Key decisions

- **Embedder required at `open`** and pinned to the file — a reopen with a
  mismatched model/dimension errors rather than returning garbage.
- **Versioned on-disk format** — files written by v1 stay readable by v1.x.
- **Scored results** (`Hit { memory, score }`) and an optional `min_score` floor.
- **Transactional, append-only writes** with exact-duplicate dedup.
- **Exact scan under ~2k memories; from-scratch HNSW index above** (approximate,
  then re-ranked exactly).
- `MemoryStore` is `Clone + Send + Sync`.

Full rationale, non-goals, and approaches:
[`docs/specs/2026-11-18-local-first-memory-engine-design.md`](docs/specs/2026-11-18-local-first-memory-engine-design.md).
Build plan: [`docs/plans/2026-11-18-memnon-v1-implementation-plan.md`](docs/plans/2026-11-18-memnon-v1-implementation-plan.md).

## How it compares

| | memnon | Mem0 / Zep / Letta | LanceDB / sqlite-vec |
|---|---|---|---|
| Runs embedded, in-process | ✅ | ❌ (service / Python stack) | ✅ |
| Fully offline, single file | ✅ | ❌ | ✅ (vector store only) |
| Memory semantics (scopes, recall API) | ✅ | ✅ | ❌ (you build it) |
| **Token-budget-first recall** | ✅ | ❌ | ❌ |
| LLM conflict resolution | ⏳ v2 | ✅ | ❌ |
| ANN index at scale | ✅ HNSW (approximate) | ✅ | ✅ |
| Language reach | Rust (sidecar → any, v2) | Python-first | Rust / SQL |

memnon isn't a vector DB and isn't trying to beat Mem0 on features — it owns the
**embeddable + local-first + token-budget-packing** corner the others don't fill.

## Benchmarks

384-dim vectors, `Budget::Tokens(2000)`, `cargo bench` on a developer laptop
(indicative). memnon auto-switches from the exact scan to the HNSW index above
~2000 memories:

| Memories | Exact scan (v1) | HNSW (v1.1) |
|---|---|---|
| 1,000 | ~0.8 ms | ~0.8 ms (scan) |
| 10,000 | ~9 ms | **~1 ms** |
| 50,000 | ~50 ms | **~0.5 ms** |

Brute force is linear (~1 ms per 1k memories); HNSW is ~logarithmic, keeping
recall sub-millisecond as the store grows. HNSW recall is **approximate**
(tunable via `ef`) and re-ranked exactly; the first search after a write pays a
one-time O(n log n) index build. Reproduce with `cargo bench`.

## Honest scope & limitations (read before adopting)

- **Rust only (v1).** memnon is embeddable from Rust today. "SQLite for agent
  memory" in the full, every-language sense needs the **sidecar + language
  clients**, which are a deliberate later step. v1 is built to be sidecar-ready.
- **Append-only; no conflict resolution.** memnon stores facts and dedups exact
  duplicates, but it does **not** reconcile contradictions — add "likes dark mode"
  then "likes light mode" and both are retained and recalled (recency breaks
  ties). LLM-based consolidation (update/supersede) is **v2**. If you need
  self-correcting memory today, memnon v1 isn't it yet.
- **Not a vector DB.** For raw ANN at scale, use LanceDB. memnon is the memory
  *semantics* and context-window packing on top.

## Roadmap

- **v1** — token-budgeted scored recall; scopes; exact-dedup; versioned single-file
  store; local + API embedder examples; benchmark + CI.
- **v1.1** — ✅ from-scratch HNSW index (shipped). Next: bundled default local
  embedder (zero-setup offline *semantic* demo), real tokenizer.
- **v2** — sidecar binary (REST / AG-UI) + Python/TS clients; LLM consolidation;
  subagent orchestration.
- **Later** — graph/temporal memory; eval harness vs LongMemEval-style benchmarks.

## License

Dual-licensed under **Apache-2.0 OR MIT**.
