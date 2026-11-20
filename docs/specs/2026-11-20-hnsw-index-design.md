# Design + Plan: HNSW Index (memnon v1.1)

**Status:** Approved approach (2026-11-20)
**Goal:** Make recall sub-linear at scale by adding a from-scratch HNSW
approximate-nearest-neighbor index, while keeping the public API and on-disk
format unchanged and preserving exact-correct results for small/filtered queries.

## Why

v1 recall is exact brute-force cosine: ~1 ms/1k memories, ~50 ms at 50k, crossing
~100 ms near 100k. HNSW (Hierarchical Navigable Small World) — a layered proximity
graph — turns that into ~log(n) search, the structure Qdrant/LanceDB use.

## Decisions

- **From scratch** (`src/hnsw.rs`) — the point is to learn the algorithm. A tiny
  seeded xorshift PRNG (no `rand` dep) keeps layer assignment deterministic for
  tests.
- **In-memory, rebuilt lazily.** The graph is NOT persisted; on first search it's
  built from the vectors already in storage. On-disk format is untouched. Cost:
  O(n log n) at build; acceptable to ~100k, revisit persistence later.
- **Index lifecycle:** built lazily and cached in the store; `add` inserts into
  the cached index incrementally; `delete`/`clear` invalidate it (next search
  rebuilds). HNSW deletion is hard, so invalidate-on-delete is the simple correct
  choice (deletes are rarer than adds).
- **Search stays exact-correct:** HNSW returns an *approximate* candidate pool;
  the store **re-ranks it exactly** (cosine), then applies scope + metadata filter,
  `min_score`, and `pack`. The public `search` signature is unchanged.
- **Brute-force fallback (safety + exactness):** below a size threshold
  (`BRUTE_FORCE_THRESHOLD`, ~2000 — where brute force is <2 ms anyway) or when the
  HNSW+filter pool comes back empty, fall back to the exact v1 scan. So small and
  selectively-filtered queries remain exact; large broad queries get the speedup.
- **Accepted tradeoff:** above the threshold, recall is approximate (ANN can miss
  a true nearest neighbor), tunable via `ef`. Documented honestly.

## Parameters
`M = 16`, `M0 = 32` (layer-0 max degree), `EF_CONSTRUCTION = 100`, `ml = 1/ln(M)`,
default `ef_search` derived from the request (over-fetch vs budget, floor ~64).

## Plan (TDD, two tasks)

### Task 1 — HNSW core (`src/hnsw.rs`)
Pure in-memory index over `(MemoryId, Vec<f32>)`, no storage coupling.
- `HnswIndex::new()`, `insert(id, vector)`, `search(query, k, ef) -> Vec<MemoryId>`.
- Internals: `search_layer(q, entry, layer, ef)` (the greedy graph walk with a
  candidate min-heap + result max-heap), random level, simple M-nearest neighbor
  selection with degree pruning, cosine distance (`1 - cosine`, reusing
  `index::cosine`).
- Deterministic xorshift PRNG.
- Tests: empty search → empty; single insert → found; small set → true nearest
  ranks first; **recall test** — build ~500 deterministic vectors, assert the
  brute-force nearest is within the HNSW top-k for a sample of queries.

### Task 2 — Integrate into `MemoryStore`
- Add `index: RwLock<Option<HnswIndex>>` to `Inner`; lazy `ensure_index()` builds
  from `storage.candidates(None, &[])`.
- `add` inserts new `(id, embedding)` into the cached index (if built);
  `delete`/`clear` drop the cache.
- `search`: brute-force path when store size ≤ threshold; else HNSW candidate pool
  → re-rank exactly → filter → pack; fall back to brute-force if the pool is empty.
- Re-add `storage.rows(ids)` to fetch the candidate pool's full rows.
- Tests: results match the brute-force path for small stores; a large store
  returns the exact-match row first; correctness holds across `add` then `search`
  and after `delete`. Extend `benches/recall.rs` to show the speedup.

## Non-goals (this pass)
Persisting the graph; deletion within the graph (tombstones); the neighbor-
selection *heuristic* (we start with simple M-nearest); per-scope indexes.
