# memnon

**A token-budget-first memory engine for AI agents.** Hand it your model's
context-window budget; it returns exactly the memories that fit — most-relevant
first, with scores — **offline and in-process.**

> ℹ️ **Status: implemented and tested (pre-1.0).** v1 (store, scopes, token-budget
> recall), a from-scratch HNSW index, and an optional on-device semantic embedder
> are all in and under test. Not yet published to crates.io; APIs may still change
> before 1.0.
>
> ℹ️ **This is a learning-first project.** Its author is using it to learn Rust
> and AI engineering by building the internals. It aims to become a genuinely
> useful OSS library, but today it's a `cargo add` Rust library — broad,
> cross-language adoption is a deliberate later step (see *Honest scope*).

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

## Semantic embeddings (optional)

The default build ships only the deterministic, **non-semantic** `HashEmbedder`
(for tests and offline demos — it keeps the core pure-Rust and light). For real
on-device semantic recall, enable the `fastembed` feature:

```toml
memnon = { version = "0", features = ["fastembed"] }
```

```rust
let store = MemoryStore::open("./memory.db", memnon::FastEmbedEmbedder::new()?)?;
```

It downloads a small model (~90 MB, all-MiniLM-L6-v2) once, then runs **fully
offline, on-device**. Try it:
`cargo run --example semantic_quickstart --features fastembed`.

> Note: the `fastembed` feature pulls a heavy native ONNX stack (currently a
> pre-1.0 `ort` release). Fine for applications; worth knowing if you vendor or
> audit dependencies.

(Or bring your own: implement [`Embedder`] over any model/API — see
`examples/openai_embedder.rs`.)

## Real tokenizers (optional)

`Budget::Tokens` needs a token *count*, not just a text length, to pack a
context window accurately. The default build ships only `HeuristicTokenizer`
(~1 token per 4 characters — fast, zero deps, but approximate, especially for
non-Latin scripts). For exact counts, plug in a real tokenizer via
[`Tokenizer`] and `MemoryStore::open_with_tokenizer`:

```toml
memnon = { version = "0", features = ["tiktoken"] }       # OpenAI BPE
# or
memnon = { version = "0", features = ["hf-tokenizer"] }   # any local tokenizer.json
# or
memnon = { version = "0", features = ["hf-hub"] }         # hf-tokenizer + fetch-by-model-id
```

```rust
// tiktoken: exact counts for OpenAI models (o200k_base by default).
use memnon::{MemoryStore, TiktokenTokenizer};
let store = MemoryStore::open_with_tokenizer(
    "./memory.db", embedder, TiktokenTokenizer::new(),
)?;
// or pin to a specific model's encoding:
let tok = TiktokenTokenizer::with_model("gpt-4o")?;
```

```rust
// HuggingFace: any model that ships a tokenizer.json (BERT, Llama, Mistral, ...).
use memnon::{MemoryStore, HfTokenizer};
let tok = HfTokenizer::from_file("./tokenizer.json")?;
// or, behind `hf-hub`, fetch by model id (downloads once, then cached offline):
let tok = HfTokenizer::from_pretrained("bert-base-uncased")?;
let store = MemoryStore::open_with_tokenizer("./memory.db", embedder, tok)?;
```

Don't have either? Implement [`Tokenizer`] yourself — it's one method,
`count_tokens(&self, text: &str) -> usize` — and pass it to
`open_with_tokenizer`. See `examples/custom_tokenizer.rs`.

> **Match your generation LLM, not your embedding model.** `Budget::Tokens` is
> about fitting memories into the **prompt you send to the LLM that generates
> the response** — so tokenize with *that* model's tokenizer (e.g. `tiktoken`
> for GPT-4o, the target model's `tokenizer.json` for open models), not the
> embedding model's. The embedding model's tokenizer is irrelevant to the
> budget; embedding already happened by the time you're packing a prompt.

## Key decisions

- **Embedder required at `open`** and pinned to the file — a reopen with a
  mismatched model/dimension errors rather than returning garbage.
- **Versioned on-disk format** — files written by v1 stay readable by v1.x.
- **Scored results** (`Hit { memory, score }`) and an optional `min_score` floor.
- **Transactional, append-only writes** with exact-duplicate dedup.
- **Exact scan under ~2k memories and for all scoped/filtered queries;
  from-scratch HNSW index for large unfiltered recall** (approximate, re-ranked
  exactly).
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
recall sub-millisecond as the store grows. **Scoped and metadata-filtered queries
use the exact scan** (correctness over speed) — HNSW accelerates *unfiltered*
whole-store recall, where it's approximate (tunable via `ef`) and re-ranked
exactly. The index is built on first use, **extended incrementally on `add`**, and
rebuilt after `delete`/`clear`. Reproduce with `cargo bench`.

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
- **v1.1** — ✅ from-scratch HNSW index + ✅ on-device semantic embedder
  (`fastembed` feature) + ✅ real tokenizer (`tiktoken` / `hf-tokenizer` / `hf-hub`
  features; pluggable `Tokenizer` trait).
- **v2** — sidecar binary (REST / AG-UI) + Python/TS clients; LLM consolidation;
  subagent orchestration.
- **Later** — graph/temporal memory; eval harness vs LongMemEval-style benchmarks.

## License

Dual-licensed under **Apache-2.0 OR MIT**.
