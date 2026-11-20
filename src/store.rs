//! `MemoryStore` — the public handle that wires the embedder, storage, recall
//! math, and the HNSW index into `add` / `search` / lifecycle operations.

use crate::embedder::Embedder;
use crate::error::{Error, Result};
use crate::hnsw::HnswIndex;
use crate::index::{approx_tokens, pack, rank};
use crate::query::{Budget, Query};
use crate::storage::{MemoryInput, Storage};
use crate::types::{Entry, Hit, Memory, MemoryId, Scope};
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

/// At or below this many memories, search scans exhaustively (exact and fast
/// enough that an ANN index isn't worth it). Above it, the HNSW index is used.
const BRUTE_FORCE_THRESHOLD: usize = 2000;
/// Minimum HNSW candidate pool, so small budgets still re-rank a healthy set.
const HNSW_POOL_FLOOR: usize = 128;
/// Extra over-fetch factor when a metadata filter may discard candidates.
const FILTER_OVERFETCH: usize = 4;

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
    /// Lazily built, cached HNSW index; dropped (invalidated) on any write and
    /// rebuilt on the next search that needs it.
    index: RwLock<Option<HnswIndex>>,
}

impl MemoryStore {
    /// Open (or create) a store at `path`, pinning `embedder` to the file. A
    /// reopen with a mismatched embedder returns [`Error::EmbedderMismatch`].
    pub fn open<E: Embedder + 'static>(path: impl AsRef<Path>, embedder: E) -> Result<Self> {
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
                index: RwLock::new(None),
            }),
        })
    }

    /// Append memories to `scope` (see the type docs). One transaction, 1:1 ids,
    /// exact-dedup; empty input is a no-op.
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
            let (content, metadata) = entry.into_parts();
            inputs.push((scope.clone(), content, metadata, embedding, created_at));
        }

        let ids = self.inner.storage.insert_batch(&inputs)?;
        self.invalidate_index();
        Ok(ids)
    }

    /// Recall memories for `query` under `constraints`, limited by `budget`,
    /// returning scored [`Hit`]s most-relevant first.
    ///
    /// Small or filtered stores are searched exactly; large stores use the HNSW
    /// index (approximate) and re-rank the candidate pool exactly.
    pub fn search(&self, query: &str, constraints: Query, budget: Budget) -> Result<Vec<Hit>> {
        let mut query_embeddings = self.inner.embedder.embed(&[query])?;
        let query_embedding = query_embeddings
            .pop()
            .ok_or_else(|| Error::Embed("embedder returned no vector for the query".into()))?;

        let candidates = self.gather_candidates(&query_embedding, &constraints, budget)?;
        if candidates.is_empty() {
            return Ok(Vec::new());
        }

        let for_ranking: Vec<(MemoryId, Vec<f32>, u64)> = candidates
            .iter()
            .map(|m| (m.id, m.embedding.clone(), m.created_at))
            .collect();
        let ranked = rank(&query_embedding, &for_ranking);

        let tokens: HashMap<MemoryId, usize> = candidates
            .iter()
            .map(|m| (m.id, approx_tokens(&m.content)))
            .collect();
        let selected = pack(&ranked, &tokens, budget, constraints.min_score_floor());

        let scores: HashMap<MemoryId, f32> = ranked.into_iter().collect();
        let mut by_id: HashMap<MemoryId, Memory> =
            candidates.into_iter().map(|m| (m.id, m)).collect();

        let hits = selected
            .into_iter()
            .filter_map(|id| {
                let score = *scores.get(&id)?;
                by_id.remove(&id).map(|memory| Hit { memory, score })
            })
            .collect();
        Ok(hits)
    }

    /// Choose the exact scan or the HNSW path, returning filtered candidate rows.
    fn gather_candidates(
        &self,
        query_embedding: &[f32],
        constraints: &Query,
        budget: Budget,
    ) -> Result<Vec<Memory>> {
        let scope = constraints.selected_scope();
        let filters = constraints.metadata_filters();

        if self.inner.storage.len()? <= BRUTE_FORCE_THRESHOLD {
            return self.inner.storage.candidates(scope, filters);
        }

        // HNSW path: over-fetch a candidate pool, fetch rows, filter, re-rank later.
        let base = match budget {
            Budget::TopK(k) => k,
            Budget::Tokens(_) => HNSW_POOL_FLOOR,
        };
        let overfetch = if filters.is_empty() {
            1
        } else {
            FILTER_OVERFETCH
        };
        let pool = base.max(HNSW_POOL_FLOOR) * overfetch;

        let ids = self.hnsw_candidate_ids(query_embedding, pool)?;
        let filtered: Vec<Memory> = self
            .inner
            .storage
            .rows(&ids)?
            .into_iter()
            .filter(|m| candidate_matches(m, scope, filters))
            .collect();

        // Safety net: if the approximate pool left nothing after filtering, fall
        // back to an exact scan so we never silently under-return.
        if filtered.is_empty() {
            return self.inner.storage.candidates(scope, filters);
        }
        Ok(filtered)
    }

    /// Approximate nearest ids from the (lazily built) HNSW index.
    fn hnsw_candidate_ids(&self, query_embedding: &[f32], pool: usize) -> Result<Vec<MemoryId>> {
        let mut guard = self.inner.index.write().unwrap();
        if guard.is_none() {
            *guard = Some(self.build_index()?);
        }
        let index = guard.as_ref().expect("index just built");
        Ok(index.search(query_embedding, pool, (pool * 2).max(64)))
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

    fn invalidate_index(&self) {
        *self.inner.index.write().unwrap() = None;
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

/// Whether a memory passes a scope + metadata-equality filter.
fn candidate_matches(memory: &Memory, scope: Option<&Scope>, filters: &[(String, String)]) -> bool {
    scope.is_none_or(|s| &memory.scope == s)
        && filters
            .iter()
            .all(|(k, v)| memory.metadata.get(k).map(String::as_str) == Some(v.as_str()))
}

/// Current Unix time in milliseconds (0 if the clock is before the epoch).
fn now_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}
