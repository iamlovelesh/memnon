//! `MemoryStore` — the public handle that wires the embedder, storage, recall
//! math, and the HNSW index into `add` / `search` / lifecycle operations.

use crate::consolidator::{ConsolidationOp, ConsolidationOutcome, Consolidator, RelatedMemory};
use crate::embedder::Embedder;
use crate::error::{Error, Result};
use crate::hnsw::HnswIndex;
use crate::index::{pack, rank};
use crate::query::{Budget, Query};
use crate::storage::{MemoryInput, ResolvedOp, Storage};
use crate::tokenizer::{HeuristicTokenizer, Tokenizer};
use crate::types::{Entry, Hit, Memory, MemoryId, Scope};
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, RwLock, RwLockWriteGuard};
use std::time::{SystemTime, UNIX_EPOCH};

/// At or below this many memories, search scans exhaustively (exact and fast
/// enough that an ANN index isn't worth it). Above it, unfiltered recall uses HNSW.
const BRUTE_FORCE_THRESHOLD: usize = 2000;
/// Minimum HNSW candidate pool, so small budgets still re-rank a healthy set.
const HNSW_POOL_FLOOR: usize = 128;
/// How many related memories are surfaced to a [`Consolidator`] per incoming item.
const CONSOLIDATE_TOP_K: usize = 8;
/// Minimum similarity score for a candidate to count as "related" during
/// consolidation. `0.0` admits anything non-negatively similar.
const CONSOLIDATE_MIN_SCORE: f32 = 0.0;

/// A handle to a memory store backed by a single embedded file.
///
/// Cheap to clone and safe to share across threads (`Clone + Send + Sync`).
#[derive(Clone)]
pub struct MemoryStore {
    inner: Arc<Inner>,
}

struct Inner {
    storage: Storage,
    embedder: Arc<dyn Embedder>,
    tokenizer: Arc<dyn Tokenizer>,
    /// Cached HNSW index: lazily built, incrementally extended on `add`, and
    /// dropped (rebuilt on next use) on `delete`/`clear`.
    index: RwLock<Option<HnswIndex>>,
}

impl MemoryStore {
    /// Open (or create) a store at `path`, pinning `embedder` to the file. A
    /// reopen with a mismatched embedder returns [`Error::EmbedderMismatch`].
    ///
    /// Uses [`HeuristicTokenizer`] for token-budget counting; see
    /// [`Self::open_with_tokenizer`] to inject a different one.
    pub fn open<E: Embedder + 'static>(path: impl AsRef<Path>, embedder: E) -> Result<Self> {
        Self::open_with_tokenizer(path, embedder, HeuristicTokenizer)
    }

    /// Like [`Self::open`], but with an explicit [`Tokenizer`] for token-budget
    /// counting (e.g. a model-specific tokenizer instead of the built-in heuristic).
    /// Also performs the embedder-identity pin, same as `open`: a reopen with a
    /// mismatched embedder returns [`Error::EmbedderMismatch`].
    pub fn open_with_tokenizer<E: Embedder + 'static, T: Tokenizer + 'static>(
        path: impl AsRef<Path>,
        embedder: E,
        tokenizer: T,
    ) -> Result<Self> {
        let storage = Storage::open(path)?;
        let (pinned_identity, pinned_dims) =
            storage.init_embedder_identity_atomic(embedder.identity(), embedder.dimensions())?;
        if pinned_identity != embedder.identity() || pinned_dims != embedder.dimensions() {
            return Err(Error::EmbedderMismatch {
                expected: format!("{pinned_identity} (dim {pinned_dims})"),
                found: format!("{} (dim {})", embedder.identity(), embedder.dimensions()),
            });
        }
        Ok(Self {
            inner: Arc::new(Inner {
                storage,
                embedder: Arc::new(embedder),
                tokenizer: Arc::new(tokenizer),
                index: RwLock::new(None),
            }),
        })
    }

    /// Append memories to `scope`. One transaction, 1:1 ids, exact-dedup; empty
    /// input is a no-op. Embeddings must be finite and match the embedder's
    /// dimension, else [`Error::Embed`] / [`Error::DimensionMismatch`].
    pub fn add<I, E>(&self, items: I, scope: Scope) -> Result<Vec<MemoryId>>
    where
        I: IntoIterator<Item = E>,
        E: Into<Entry>,
    {
        let entries: Vec<Entry> = items.into_iter().map(Into::into).collect();
        if entries.is_empty() {
            return Ok(Vec::new());
        }

        let texts: Vec<&str> = entries.iter().map(Entry::content).collect();
        let embeddings = self.inner.embedder.embed(&texts)?;
        if embeddings.len() != entries.len() {
            return Err(Error::Embed(format!(
                "embedder returned {} vectors for {} inputs",
                embeddings.len(),
                entries.len()
            )));
        }

        let dims = self.inner.embedder.dimensions();
        let created_at = now_millis();
        let mut inputs: Vec<MemoryInput> = Vec::with_capacity(entries.len());
        for (entry, embedding) in entries.into_iter().zip(embeddings) {
            if embedding.len() != dims {
                return Err(Error::DimensionMismatch {
                    expected: dims,
                    found: embedding.len(),
                });
            }
            if !is_finite(&embedding) {
                return Err(Error::Embed(
                    "embedder returned a non-finite (NaN/inf) value".into(),
                ));
            }
            let (content, metadata) = entry.into_parts();
            inputs.push((scope.clone(), content, metadata, embedding, created_at));
        }

        let inserted = self.inner.storage.insert_batch(&inputs)?;

        // Keep a built index in sync incrementally (avoids an O(n) rebuild per add);
        // if none is built, the next search builds it from storage, new rows included.
        {
            let mut guard = self.index_write();
            if let Some(index) = guard.as_mut() {
                for ((id, is_new), input) in inserted.iter().zip(&inputs) {
                    if *is_new {
                        index.insert(*id, input.3.clone());
                    }
                }
            }
        }

        Ok(inserted.into_iter().map(|(id, _)| id).collect())
    }

    /// Reconcile incoming facts against what's already stored in `scope`,
    /// delegating the merge decision (add/update/delete/no-op) to `llm`.
    ///
    /// Each item is processed in order: it's embedded, the most-related
    /// existing memories in `scope` are retrieved (top [`CONSOLIDATE_TOP_K`],
    /// scored at or above [`CONSOLIDATE_MIN_SCORE`]), `llm.reconcile` decides
    /// what to do, and the result is applied in its own transaction before the
    /// next item is considered — so a later item in the same batch sees the
    /// committed effects of earlier ones. An empty `reconcile` result is
    /// treated as a single [`ConsolidationOp::Noop`].
    ///
    /// Fails fast: an embedding or `reconcile` error aborts the whole call
    /// immediately with `Err`, but items already applied before the failing
    /// one stay committed (no rollback across items).
    ///
    /// Embeddings are produced and validated (finite, correct dimension —
    /// same guards as [`Self::add`]) *before* every write, so a bad embedder
    /// output fails the item before any transaction touches storage.
    pub fn consolidate<I, E, L>(
        &self,
        items: I,
        scope: Scope,
        llm: &L,
    ) -> Result<ConsolidationOutcome>
    where
        I: IntoIterator<Item = E>,
        E: Into<Entry>,
        L: Consolidator + ?Sized,
    {
        let entries: Vec<Entry> = items.into_iter().map(Into::into).collect();
        if entries.is_empty() {
            return Ok(ConsolidationOutcome::default());
        }

        let dims = self.inner.embedder.dimensions();
        let mut outcome = ConsolidationOutcome::default();
        let mut mutated_existing = false; // any Update or Delete applied this call

        for entry in entries {
            let (content, metadata) = entry.into_parts();
            let incoming_embedding = self.embed_one_checked(&content, dims)?;

            let related = self.related_memories(&scope, &incoming_embedding)?;

            let ops = llm.reconcile(&content, &related)?;
            let ops: Vec<ConsolidationOp> = if ops.is_empty() {
                vec![ConsolidationOp::Noop]
            } else {
                ops
            };

            let mut resolved: Vec<ResolvedOp> = Vec::with_capacity(ops.len());
            for op in ops {
                match op {
                    ConsolidationOp::Add => resolved.push(ResolvedOp::Add {
                        content: content.clone(),
                        metadata: metadata.clone(),
                        embedding: incoming_embedding.clone(),
                    }),
                    ConsolidationOp::Update {
                        id,
                        content: new_content,
                    } => {
                        let embedding = self.embed_one_checked(&new_content, dims)?;
                        resolved.push(ResolvedOp::Update {
                            id,
                            content: new_content,
                            embedding,
                        });
                    }
                    ConsolidationOp::Delete { id } => {
                        resolved.push(ResolvedOp::Delete { id });
                    }
                    ConsolidationOp::Noop => outcome.noops += 1,
                    // `ConsolidationOp` is `#[non_exhaustive]` for downstream crates
                    // only; within memnon this match must stay exhaustive so a new
                    // variant is a compile error here, not a silently-ignored op.
                }
            }

            if !resolved.is_empty() {
                let applied =
                    self.inner
                        .storage
                        .apply_consolidation(&scope, &resolved, now_millis())?;
                if !applied.updated.is_empty() || !applied.deleted.is_empty() {
                    mutated_existing = true;
                }
                outcome.added.extend(applied.added);
                outcome.updated.extend(applied.updated);
                outcome.deleted.extend(applied.deleted);
            }
        }

        // Updates/deletes can shift or remove graph neighbors the HNSW index
        // doesn't support patching in place, so drop it and let the next
        // search rebuild. Add-only batches are cheaper to keep in sync: extend
        // an already-built index incrementally (mirroring `add`); if no index
        // is built yet, leave it — the next search builds one from storage,
        // new rows included.
        if mutated_existing {
            self.invalidate_index();
        } else if !outcome.added.is_empty() {
            let mut guard = self.index_write();
            if let Some(index) = guard.as_mut() {
                for id in &outcome.added {
                    if let Some(memory) = self.inner.storage.get(*id)? {
                        index.insert(*id, memory.embedding.clone());
                    }
                }
            }
        }

        Ok(outcome)
    }

    /// Embed a single piece of content, applying the same finiteness/dimension
    /// guards as [`Self::add`].
    fn embed_one_checked(&self, content: &str, dims: usize) -> Result<Vec<f32>> {
        let mut embeddings = self.inner.embedder.embed(&[content])?;
        let embedding = embeddings
            .pop()
            .ok_or_else(|| Error::Embed("embedder returned no vector for the input".into()))?;
        if embedding.len() != dims {
            return Err(Error::DimensionMismatch {
                expected: dims,
                found: embedding.len(),
            });
        }
        if !is_finite(&embedding) {
            return Err(Error::Embed(
                "embedder returned a non-finite (NaN/inf) value".into(),
            ));
        }
        Ok(embedding)
    }

    /// The top [`CONSOLIDATE_TOP_K`] memories in `scope` related to
    /// `incoming_embedding`, scored at or above [`CONSOLIDATE_MIN_SCORE`].
    fn related_memories(
        &self,
        scope: &Scope,
        incoming_embedding: &[f32],
    ) -> Result<Vec<RelatedMemory>> {
        let candidates = self.inner.storage.candidates(Some(scope), &[])?;
        let contents: HashMap<MemoryId, String> = candidates
            .iter()
            .map(|m| (m.id, m.content.clone()))
            .collect();
        let for_ranking: Vec<(MemoryId, Vec<f32>, u64)> = candidates
            .iter()
            .map(|m| (m.id, m.embedding.clone(), m.created_at))
            .collect();
        let ranked = rank(incoming_embedding, &for_ranking);

        Ok(ranked
            .into_iter()
            .filter(|(_, score)| *score >= CONSOLIDATE_MIN_SCORE)
            .take(CONSOLIDATE_TOP_K)
            .filter_map(|(id, score)| {
                contents
                    .get(&id)
                    .map(|content| RelatedMemory::new(id, content.clone(), score))
            })
            .collect())
    }

    /// Recall memories for `query` under `constraints`, limited by `budget`,
    /// returning scored [`Hit`]s most-relevant first.
    ///
    /// Scoped/filtered queries and small stores are searched exactly. Large
    /// *unfiltered* stores use the HNSW index (approximate) and re-rank the
    /// candidate pool exactly. For `Budget::Tokens`, the pool grows until the
    /// budget — not the pool — limits the result, so recall is never silently
    /// capped (even with zero-token/empty memories).
    pub fn search(&self, query: &str, constraints: Query, budget: Budget) -> Result<Vec<Hit>> {
        let mut query_embeddings = self.inner.embedder.embed(&[query])?;
        let query_embedding = query_embeddings
            .pop()
            .ok_or_else(|| Error::Embed("embedder returned no vector for the query".into()))?;
        if !is_finite(&query_embedding) {
            return Err(Error::Embed(
                "embedder returned a non-finite (NaN/inf) query vector".into(),
            ));
        }

        let scope = constraints.selected_scope();
        let filters = constraints.metadata_filters();
        let min_score = constraints.min_score_floor();

        // Exact path: scoped/filtered recall (an ANN pool is global; post-filtering
        // it under-returns) and small stores. v1.1 accelerates only unfiltered
        // whole-store recall; per-scope indexes are a future milestone.
        let total = self.inner.storage.len()?;
        if scope.is_some() || !filters.is_empty() || total <= BRUTE_FORCE_THRESHOLD {
            let candidates = self.inner.storage.candidates(scope, filters)?;
            return Ok(self.rank_and_pack(&query_embedding, candidates, budget, min_score));
        }

        // HNSW path. Start sized to the budget; for `Tokens`, grow geometrically
        // while `pack` keeps the entire returned pool (budget not yet the limiter)
        // and the index can still return more.
        let grows = matches!(budget, Budget::Tokens(_));
        let mut pool = match budget {
            Budget::TopK(k) => k.max(HNSW_POOL_FLOOR),
            Budget::Tokens(max) => max.max(HNSW_POOL_FLOOR),
        };
        loop {
            let capped = pool.min(total);
            let ids = self.hnsw_candidate_ids(&query_embedding, capped)?;
            let candidates = self.inner.storage.rows(&ids)?;
            let fetched = candidates.len();
            let hits = self.rank_and_pack(&query_embedding, candidates, budget, min_score);

            let may_have_more = fetched == capped && capped < total;
            if !grows || hits.len() < fetched || !may_have_more {
                return Ok(hits);
            }
            pool = capped.saturating_mul(2).max(capped + 1);
        }
    }

    /// Rank candidates against the query, pack to the budget, and materialize
    /// scored hits in ranked order.
    fn rank_and_pack(
        &self,
        query_embedding: &[f32],
        candidates: Vec<Memory>,
        budget: Budget,
        min_score: Option<f32>,
    ) -> Vec<Hit> {
        if candidates.is_empty() {
            return Vec::new();
        }
        let for_ranking: Vec<(MemoryId, Vec<f32>, u64)> = candidates
            .iter()
            .map(|m| (m.id, m.embedding.clone(), m.created_at))
            .collect();
        let ranked = rank(query_embedding, &for_ranking);

        let tokens: HashMap<MemoryId, usize> = candidates
            .iter()
            .map(|m| (m.id, self.inner.tokenizer.count_tokens(&m.content)))
            .collect();
        let selected = pack(&ranked, &tokens, budget, min_score);

        let scores: HashMap<MemoryId, f32> = ranked.into_iter().collect();
        let mut by_id: HashMap<MemoryId, Memory> =
            candidates.into_iter().map(|m| (m.id, m)).collect();

        selected
            .into_iter()
            .filter_map(|id| {
                let score = *scores.get(&id)?;
                by_id.remove(&id).map(|memory| Hit { memory, score })
            })
            .collect()
    }

    /// Approximate nearest ids from the cached HNSW index, building it on first use.
    fn hnsw_candidate_ids(&self, query_embedding: &[f32], pool: usize) -> Result<Vec<MemoryId>> {
        let ef = (pool * 2).max(64);
        // Fast path: index already built — a shared read lock allows concurrent searches.
        {
            let guard = self.inner.index.read().unwrap_or_else(|e| e.into_inner());
            if let Some(index) = guard.as_ref() {
                return Ok(index.search(query_embedding, pool, ef));
            }
        }
        // Build under the write lock (double-checked in case another thread raced).
        let mut guard = self.index_write();
        if guard.is_none() {
            *guard = Some(self.build_index()?);
        }
        Ok(guard
            .as_ref()
            .expect("index present")
            .search(query_embedding, pool, ef))
    }

    /// Build an HNSW index over every stored memory.
    fn build_index(&self) -> Result<HnswIndex> {
        let all = self.inner.storage.candidates(None, &[])?;
        let mut index = HnswIndex::new();
        for memory in &all {
            index.insert(memory.id, memory.embedding.clone());
        }
        Ok(index)
    }

    /// Write guard on the index, recovering from a poisoned lock rather than
    /// panicking — one transient failure shouldn't permanently brick the store.
    fn index_write(&self) -> RwLockWriteGuard<'_, Option<HnswIndex>> {
        self.inner.index.write().unwrap_or_else(|e| e.into_inner())
    }

    fn invalidate_index(&self) {
        *self.index_write() = None;
    }

    /// Fetch a single memory by id.
    pub fn get(&self, id: MemoryId) -> Result<Option<Memory>> {
        self.inner.storage.get(id)
    }

    /// Every memory in a scope, unranked.
    pub fn get_all(&self, scope: Scope) -> Result<Vec<Memory>> {
        self.inner.storage.get_all(&scope)
    }

    /// Delete one memory by id (`Error::NotFound` if absent).
    pub fn delete(&self, id: MemoryId) -> Result<()> {
        let existed = self.inner.storage.delete(id)?;
        self.invalidate_index();
        if existed {
            Ok(())
        } else {
            Err(Error::NotFound(id))
        }
    }

    /// Remove every memory in a scope; returns the number removed.
    pub fn clear(&self, scope: &Scope) -> Result<u64> {
        let removed = self.inner.storage.clear(scope)?;
        self.invalidate_index();
        Ok(removed)
    }

    /// Number of memories in a scope.
    pub fn count(&self, scope: &Scope) -> Result<usize> {
        self.inner.storage.count(scope)
    }

    /// All scopes currently present in the store.
    pub fn list_scopes(&self) -> Result<Vec<Scope>> {
        self.inner.storage.list_scopes()
    }
}

fn is_finite(v: &[f32]) -> bool {
    v.iter().all(|x| x.is_finite())
}

/// Current Unix time in milliseconds (0 if the clock is before the epoch).
fn now_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}
