//! `MemoryStore` — the public handle that wires the embedder, storage, and recall
//! math into `add` / `search` / lifecycle operations.

use crate::embedder::Embedder;
use crate::error::{Error, Result};
use crate::index::{approx_tokens, pack, rank};
use crate::query::{Budget, Query};
use crate::storage::{MemoryInput, Storage};
use crate::types::{Entry, Hit, Memory, MemoryId, Scope};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

/// A handle to a memory store backed by a single embedded file.
///
/// Cheap to clone and safe to share across threads (`Clone + Send + Sync`): the
/// embedder is fixed at construction and never mutated, and `redb` serializes
/// writes, so there is no identity-pin race.
#[derive(Clone)]
pub struct MemoryStore {
    inner: Arc<Inner>,
}

struct Inner {
    storage: Storage,
    embedder: Arc<dyn Embedder>,
}

impl MemoryStore {
    /// Open (or create) a store at `path`, pinning `embedder` to the file.
    ///
    /// On an existing file whose recorded `(identity, dimensions)` differ from
    /// `embedder`, returns [`Error::EmbedderMismatch`] rather than returning
    /// garbage recall.
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
            }),
        })
    }

    /// Append memories to `scope`. Each item may carry metadata; `String`/`&str`
    /// convert via [`Entry`]. Ids are returned 1:1 with input order, all in one
    /// transaction. Exact `(scope, content)` duplicates return the existing id.
    /// Empty input is a no-op returning `Ok(vec![])`.
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
        self.inner.storage.insert_batch(&inputs)
    }

    /// Recall memories for `query` under `constraints`, limited by `budget`,
    /// returning scored [`Hit`]s most-relevant first. v1 uses exact brute-force
    /// cosine over the candidate set.
    pub fn search(&self, query: &str, constraints: Query, budget: Budget) -> Result<Vec<Hit>> {
        let mut query_embeddings = self.inner.embedder.embed(&[query])?;
        let query_embedding = query_embeddings
            .pop()
            .ok_or_else(|| Error::Embed("embedder returned no vector for the query".into()))?;

        let candidates = self
            .inner
            .storage
            .candidates(constraints.selected_scope(), constraints.metadata_filters())?;
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
        if self.inner.storage.delete(id)? {
            Ok(())
        } else {
            Err(Error::NotFound(id))
        }
    }

    /// Remove every memory in a scope (session/subagent teardown); returns the
    /// number removed.
    pub fn clear(&self, scope: &Scope) -> Result<u64> {
        self.inner.storage.clear(scope)
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

/// Current Unix time in milliseconds (0 if the clock is before the epoch).
fn now_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}
