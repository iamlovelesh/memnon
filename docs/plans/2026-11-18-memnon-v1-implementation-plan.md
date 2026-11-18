# memnon v1 Implementation Plan (post-review)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a usable, local-first Rust library (`memnon`) that stores agent memories in a single versioned file and returns the most-relevant memories that fit a token budget, offline and in-process.

**Architecture:** A `Clone+Send+Sync` `MemoryStore` wraps a pure-Rust `redb` file (`storage`), a required immutable `Embedder`, and pure-function recall math (`index`). `add` embeds + appends (exact-dedup, one transaction); `search` embeds the query, brute-force cosine-ranks, and packs scored hits to a budget. Versioned on-disk format; no network, no sidecar, no HNSW in v1.

**Tech Stack:** Rust (edition 2024); `redb` (store); `serde` + `bincode` inside a versioned envelope; `thiserror` (errors); `std`. Dev/examples: `tempfile`, `criterion` (bench), `fastembed` (local example), `ureq` + `serde_json` (API example).

**Spec:** `memnon/docs/specs/2026-11-18-local-first-memory-engine-design.md`

## Global Constraints

- License **Apache-2.0 OR MIT**; edition **2024**.
- Core crate: **pure-Rust deps only**, fully **offline**, single embedded file.
- **Versioned on-disk format:** `schema_version: u32` in `meta`; each record in a versioned envelope; `Scope` serialized **by name**; unknown/newer version → `Error::UnsupportedSchema`.
- **Embedder required at `open`**; identity+dimension pinned atomically; mismatch → `Error::EmbedderMismatch`. No `NoEmbedder` state.
- `search` returns **`Vec<Hit>`** (`Hit { memory, score }`), never bare `Memory`.
- `add` carries **per-item metadata**, returns ids **1:1 with input**, commits in **one transaction**, and **exact-dedups** `(scope, content)`.
- **`cosine` returns `0.0` on a zero norm** (never `NaN`); vector len ≠ `dimensions()` → `Error::DimensionMismatch`.
- Public enums are **`#[non_exhaustive]`**.
- Token budgeting uses approximate `chars/4`, **documented as rough**.
- v1 recall is exact brute-force cosine (no HNSW); writes append-only beyond exact-dedup.

## Review Focus

Inputs the spec implies; each pinned to a test in the owning task:

1. **Reopen with a mismatched embedder** (identity or dimension) → `EmbedderMismatch`. → Task 4 + Task 5.
2. **Open a file with an unknown/newer `schema_version`** → `UnsupportedSchema`, not corruption. → Task 4.
3. **Zero-norm embedding** (e.g. `""`) → `cosine` = 0.0, never floats to the top of every search. → Task 3 (+ Task 2 embeds `""`).
4. **`Budget::Tokens(n)` smaller than the top hit** → empty; cumulative never exceeds `n`. → Task 3.
5. **Concurrent `add` from many threads** → unique ids, exactly N*M rows (counter race). → Task 5.
6. **Persistence across reopen** → content+embedding+metadata+scope survive a close/reopen and still rank correctly. → Task 5.
7. **`add` empty iterator** → `Ok(vec![])`, no write. **`search` with no scope** → all scopes; **with a scope** → excludes others. → Task 5.
8. **Unicode content** (CJK/emoji) → `approx_tokens` counts scalar values (not bytes); budget packing behaves. → Task 3 + Task 5.
9. **Semantic ordering** — the most-similar memory ranks first end-to-end (deterministic, no model). → Task 5.

---

## File Structure

- `src/lib.rs` — module decls, re-exports, crate docs.
- `src/error.rs` — `Error` (`#[non_exhaustive]`, `thiserror`), `Result`.
- `src/types.rs` — `MemoryId` (opaque), `Scope`, `Memory`, `Entry`, `Hit`.
- `src/query.rs` — `Query` (scope/filter/min_score), `Budget`.
- `src/embedder.rs` — `Embedder` trait + `HashEmbedder` dev double.
- `src/index.rs` — `cosine`, `approx_tokens`, `rank`, `pack` (pure fns).
- `src/storage.rs` — `redb` tables, versioned envelope, schema version, id counter, dedup index, batch txn.
- `src/store.rs` — `MemoryStore` orchestration.
- `examples/quickstart.rs` (fastembed, offline), `examples/openai_api.rs`.
- `benches/recall.rs`; `.github/workflows/ci.yml`; `tests/*.rs`.

---

## Task 1: Core types, errors, query, Entry/Hit, licenses

**Files:** Create `src/error.rs`, `src/types.rs`, `src/query.rs`, `LICENSE-MIT`, `LICENSE-APACHE`; Modify `src/lib.rs`, `Cargo.toml` (add `serde`, `thiserror`); Test `tests/types.rs`.

**Produces:**
- `Error` (`#[non_exhaustive]`): `Io`, `Corrupt(String)`, `UnsupportedSchema { found: u32, max_supported: u32 }`, `EmbedderMismatch { expected: String, found: String }`, `DimensionMismatch { expected: usize, found: usize }`, `Embed(String)`, `NotFound(MemoryId)`; `type Result<T>`.
- `MemoryId` — opaque (private `u64`), `Debug/Clone/Copy/PartialEq/Eq/Hash` + `Serialize/Deserialize`.
- `Scope { Session, LongTerm, Custom(String) }` `#[non_exhaustive]`; serialized **by name** (serde default is by-name for these variants — verify `Custom` carries its string).
- `Memory { id, scope, content, embedding: Vec<f32>, metadata: HashMap<String,String>, created_at: u64, updated_at: u64 }`.
- `Entry { content: String, metadata: HashMap<String,String> }` — `new(impl Into<String>)`, `meta(self, k, v) -> Self`; `From<String>` / `From<&str>`.
- `Hit { memory: Memory, score: f32 }`.
- `Budget { TopK(usize), Tokens(usize) }` `#[non_exhaustive]`.
- `Query` — `new()`, `scope(self,Scope)`, `filter(self,k,v)`, `min_score(self,f32)`; accessors `selected_scope() -> Option<&Scope>`, `metadata_filters() -> &[(String,String)]`, `min_score_floor() -> Option<f32>`.

- [ ] **Step 1: Failing test** (`tests/types.rs`): query builder accumulates scope+filters+min_score; `Entry::from("x")` has empty metadata; `Entry::new("x").meta("k","v")` carries it.
- [ ] **Step 2: Run** `cargo test --test types` → FAIL.
- [ ] **Step 3: Implement** the above; `thiserror` derives on `Error`; wire `lib.rs`; add license text.
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat: core types, errors, Entry/Hit, query builder, licenses`.

---

## Task 2: Embedder trait + HashEmbedder dev double

**Files:** Create `src/embedder.rs`; Modify `src/lib.rs`; Test `tests/embedder.rs`.

**Consumes:** `Result` (T1).
**Produces:** `trait Embedder: Send + Sync { fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>; fn dimensions(&self) -> usize; fn identity(&self) -> &str; }`; `HashEmbedder { dims }` with `new(dims)`, `identity() == "hash-v1-d{dims}"`. **Deterministic, non-semantic dev double**: hash tokens into `dims` buckets, then L2-normalize; an all-zero bucket vector normalizes to **all-zeros** (not NaN).

- [ ] **Step 1: Failing test** (`tests/embedder.rs`): determinism; `embed` 1:1 with input; `len == dimensions()`; identity string; **`embed(&[""])` returns a finite vector with no `NaN`** (Review Focus #3 feeder).
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat: Embedder trait + deterministic HashEmbedder`.

---

## Task 3: Recall math — cosine, tokens, rank, pack

**Files:** Create `src/index.rs`; Modify `src/lib.rs` (`pub(crate) mod index`); Test: `#[cfg(test)]` in `src/index.rs`.

**Produces (crate-internal):**
- `fn cosine(a: &[f32], b: &[f32]) -> f32` — **returns `0.0` if either norm is 0.0**.
- `fn approx_tokens(s: &str) -> usize` — `s.chars().count().div_ceil(4)` (scalar values, not bytes).
- `fn rank(query: &[f32], candidates: &[(MemoryId, Vec<f32>, u64)]) -> Vec<(MemoryId, f32)>` — score desc, `created_at` (the `u64`) desc tiebreak, via `f32::total_cmp`.
- `fn pack(ranked: &[(MemoryId, f32)], tokens: &HashMap<MemoryId,usize>, budget: Budget, min_score: Option<f32>) -> Vec<MemoryId>` — drop below `min_score`; `TopK(k)` → first k; `Tokens(max)` → greedy while running sum ≤ max.

- [ ] **Step 1: Failing tests**:
  - `cosine_identical_is_one`, `cosine_orthogonal_is_zero`, **`cosine_zero_norm_is_zero`** (`cosine(&[0.0,0.0], &[1.0,0.0]) == 0.0`, Review Focus #3).
  - `approx_tokens_rounds_up` (`"abcde"`→2) and **`approx_tokens_counts_unicode_scalars`** (`"日本語"`→1, i.e. 3 chars, not 9 bytes; Review Focus #8).
  - `rank_orders_by_similarity_desc`; `rank_breaks_ties_by_recency` (equal score → higher `created_at` first).
  - `pack_tokens_stops_at_budget`; **`pack_tokens_empty_when_top_exceeds_budget`** (#4); `pack_min_score_drops_weak`.
- [ ] **Step 2: Run** `cargo test index` → FAIL. **Step 3: Implement.** **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat: recall math (zero-safe cosine, tokens, rank, pack)`.

---

## Task 4: redb storage layer (versioned, transactional)

**Files:** Create `src/storage.rs`; Modify `src/lib.rs`, `Cargo.toml` (add `redb`, `bincode`; dev `tempfile`); Test `tests/storage.rs`.

**Consumes:** `Memory`, `MemoryId`, `Scope`, `Error`/`Result` (T1).
**Produces `Storage`:**
- `open(path) -> Result<Storage>` — creates/validates `schema_version` in `meta`; unknown/newer → `Error::UnsupportedSchema`.
- `embedder_identity() -> Result<Option<(String,usize)>>`; `init_embedder_identity_atomic(id,dims) -> Result<(String,usize)>` — inside **one** write txn: if absent, set and return it; if present, return the existing one (caller compares). This is the atomic compare-and-set that kills the pin race.
- `insert_batch(entries: &[(Scope, String, HashMap<String,String>, Vec<f32>, u64)]) -> Result<Vec<MemoryId>>` — allocate ids + write all rows + dedup-index updates in **one** write txn; exact-dedup on `(scope, content)` returns the existing id without inserting.
- `get(id) -> Result<Option<Memory>>`; `projection_in_scope(scope: Option<&Scope>) -> Result<Vec<(MemoryId, Vec<f32>, u64)>>` (light, for ranking); `rows(ids: &[MemoryId]) -> Result<Vec<Memory>>`.
- `get_all(scope) -> Result<Vec<Memory>>`; `delete(id) -> Result<bool>`; `clear(scope) -> Result<u64>`; `count(scope) -> Result<usize>`; `list_scopes() -> Result<Vec<Scope>>`.

**Record encoding:** value = `[version: u32][bincode(Memory)]` envelope; key = `id` big-endian. Map `redb`/`bincode` errors to `Error::Io`/`Error::Corrupt`.

- [ ] **Step 1: Failing tests** (`tests/storage.rs`, `tempfile::tempdir`): roundtrip insert→get; `insert_batch` returns monotonic ids and persists across reopen; **exact-dedup** (same `(scope,content)` twice → one row, same id); `init_embedder_identity_atomic` persists + returns existing on second call (Review Focus #1 feeder); **forged `schema_version` in `meta` → `open` returns `UnsupportedSchema`** (#2); `projection_in_scope`/`get_all` scope filtering; `clear` removes a scope and returns count; `count`/`list_scopes`; `delete` true→get None.
- [ ] **Step 2: Run** `cargo test --test storage` → FAIL. **Step 3: Implement** over `redb` (`memories`, `meta`, `dedup` tables). **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat: versioned transactional redb storage (dedup, scopes, atomic identity)`.

---

## Task 5: MemoryStore end-to-end

**Files:** Modify `src/store.rs`, `src/lib.rs`, `README.md`; Test `tests/store.rs`.

**Consumes:** `Storage` (T4), `Embedder` (T2), `cosine/rank/pack/approx_tokens` (T3), types/query (T1).
**Produces (final public API):**
- `open(path, embedder) -> Result<Self> where E: Embedder + 'static` — opens storage; `init_embedder_identity_atomic(embedder.identity(), embedder.dimensions())` then compares to the live embedder → `EmbedderMismatch` on mismatch; stores `Arc<dyn Embedder>` (immutable).
- `add<I,E>(&self, items: I, scope: Scope) -> Result<Vec<MemoryId>> where I: IntoIterator<Item = E>, E: Into<Entry>` — batch-embed all contents (validate each len == `dimensions()` → `DimensionMismatch`); one `insert_batch` txn; `created_at == updated_at`; empty input → `Ok(vec![])`.
- `search(&self, query: &str, constraints: Query, budget: Budget) -> Result<Vec<Hit>>` — embed query; `projection_in_scope(selected_scope)`; apply `metadata_filters` (needs row metadata — gather via projection carrying filters, or filter after `rows`); `rank`; `pack` (with `min_score`); `rows()` the packed ids; zip scores into `Hit`s in ranked order.
- `get(id) -> Result<Option<Memory>>`; `get_all(scope) -> Result<Vec<Memory>>`; `delete(id) -> Result<()>` (missing → `NotFound`); `clear(&Scope) -> Result<u64>`; `count(&Scope) -> Result<usize>`; `list_scopes() -> Result<Vec<Scope>>`.

**Structure:** `MemoryStore { inner: Arc<Inner> }`, `Inner { storage: Storage, embedder: Arc<dyn Embedder> }` → `Clone+Send+Sync`, no interior mutation of the embedder.

- [ ] **Step 1: Failing tests** (`tests/store.rs`, `HashEmbedder::new(16)`):
  - `add_returns_one_id_per_item` + `get_all`; `updated_at == created_at`.
  - **`add_exact_duplicate_returns_same_id`** (one row).
  - **`add_empty_is_ok_noop`** (#7); `add_dimension_mismatch_errors` (embedder lies about len).
  - **`search_ranks_most_similar_first`** — memories `"alpha beta gamma"`, `"alpha beta delta"`, `"zeta eta theta"`; query `"alpha beta gamma"`; assert exact-overlap row is `hits[0]`, disjoint row last; `hits[0].score >= hits[1].score` (Review Focus #9 — the one test that proves the value prop, model-free).
  - `search_respects_scope` + `search_no_scope_searches_all` (#7); `search_metadata_filter`; `search_min_score_floor`; `search_budget_tokens_respects_limit`; **`search_empty_store_is_empty`**.
  - **`reopen_persists_data`** — add → drop → reopen same path → `get_all` + `search` intact (#6).
  - **`reopen_mismatched_embedder_errors`** — build with `d16`, reopen with `d32` → `EmbedderMismatch` (#1).
  - **`concurrent_adds_have_unique_ids`** — 8 threads × 50 adds on a cloned store → `get_all` has 400 unique ids (#5).
  - `clear_scope_removes_only_that_scope`; `count`/`list_scopes`.
  - **`search_unicode_budget`** — CJK/emoji content with `Budget::Tokens` returns without blowing up (#8).
- [ ] **Step 2: Run** `cargo test --test store` → FAIL. **Step 3: Implement**; update `README.md` quickstart to the final API. **Step 4: Run** → PASS; then `cargo test` (all) + `cargo clippy -- -D warnings` + `cargo fmt --check` clean.
- [ ] **Step 5: Commit** `feat: MemoryStore end-to-end (scored token-budget recall, scopes, dedup)`.

---

## Task 6: Shipped embedder examples (offline quickstart + API)

**Files:** Create `examples/quickstart.rs` (fastembed), `examples/openai_api.rs`; Modify `Cargo.toml` (`[dev-dependencies]`: `fastembed`, `ureq`, `serde_json`); Test: `cargo build --examples` + one `#[ignore]` online smoke test.

**Produces:** `FastEmbedEmbedder` (local ONNX; downloads a small model once, then fully offline — the real local-first path) and `OpenAiEmbedder` (`ureq` POST to `/embeddings`). `quickstart.rs` opens a store, adds a few memories, and prints a scored `search` result.

- [ ] **Step 1:** Write both examples implementing `Embedder` against the public API.
- [ ] **Step 2:** `cargo build --examples` → success. Add one `#[ignore]` test (`tests/semantic_online.rs`) that, when run, builds a `FastEmbedEmbedder`, adds two memories, and asserts a semantically-related query out-ranks an unrelated one (the only true semantic-quality regression check; excluded from fast CI).
- [ ] **Step 3: Commit** `docs: offline fastembed quickstart + OpenAI API embedder examples`.

---

## Task 7: Benchmark, CI, and launch docs

**Files:** Create `benches/recall.rs` (criterion), `.github/workflows/ci.yml`; Modify `Cargo.toml` (`[dev-dependencies] criterion`, `[[bench]]`), `README.md` (benchmarks + comparison table + append-only caveat).

**Produces:**
- `benches/recall.rs` — insert 1k/10k/100k memories (`HashEmbedder`), measure `search` latency + peak rows scanned; **documents the row count where latency crosses ~100ms**, turning "brute force is fine" into a measured boundary and justifying the v1.1 HNSW milestone.
- `.github/workflows/ci.yml` — `cargo fmt --check`, `cargo clippy -- -D warnings`, `cargo test` (all, excludes `#[ignore]`), `cargo llvm-cov` (report; soft 80% target per repo rule).
- `README.md` — a benchmarks section (binary size, latency, footprint), a **comparison table** (vs Mem0 / LanceDB / sqlite-vec) answering "why memnon," and the explicit **append-only / no-conflict-resolution** caveat.

- [ ] **Step 1:** Add bench + CI + README sections.
- [ ] **Step 2:** `cargo bench` runs; CI green locally (`act` or manual command run).
- [ ] **Step 3: Commit** `chore: recall benchmark, CI (fmt/clippy/test/coverage), launch docs`.

---

## Notes for the executor
- Work top-to-bottom; each task's `Produces` is the next task's only context.
- `index` and `storage` are `pub(crate)`; only `error`, `types`, `query`, `embedder`, `store` re-export from `lib.rs`.
- After Task 5 the library is a complete, correct v1 (with `HashEmbedder`); Tasks 6–7 make it genuinely useful and launch-ready.
