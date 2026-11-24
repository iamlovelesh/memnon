//! Single-file storage over `redb`: versioned records, a monotonic id counter,
//! exact-dedup, and an atomically-pinned embedder identity.
//!
//! Crate-internal; the store orchestrates these. Unit-tested below.

use crate::error::{Error, Result};
use crate::types::{Memory, MemoryId, Scope};
use redb::{Database, ReadableDatabase, ReadableTable, ReadableTableMetadata, TableDefinition};
use std::collections::HashMap;
use std::path::Path;

/// On-disk format version, written into `meta` on create. Opening a file written
/// by a newer version is refused rather than misread.
const SCHEMA_VERSION: u32 = 1;
/// Per-record envelope version (a leading `u32`), so a record's layout can evolve
/// independently of the schema.
const RECORD_VERSION: u32 = 1;

const MEMORIES: TableDefinition<u64, &[u8]> = TableDefinition::new("memories");
const META: TableDefinition<&str, &[u8]> = TableDefinition::new("meta");
const DEDUP: TableDefinition<&str, u64> = TableDefinition::new("dedup");

const KEY_SCHEMA: &str = "schema_version";
const KEY_NEXT_ID: &str = "next_id";
const KEY_EMB_IDENTITY: &str = "embedder_identity";
const KEY_EMB_DIMS: &str = "embedder_dims";

/// One memory to insert: `(scope, content, metadata, embedding, created_at)`.
pub(crate) type MemoryInput = (Scope, String, HashMap<String, String>, Vec<f32>, u64);

/// An opened memory store file.
pub(crate) struct Storage {
    db: Database,
}

impl Storage {
    /// Open (or create) the store at `path`, validating the schema version.
    pub(crate) fn open(path: impl AsRef<Path>) -> Result<Self> {
        let db = Database::create(path).map_err(io)?;
        let wtxn = db.begin_write().map_err(io)?;
        {
            let mut meta = wtxn.open_table(META).map_err(io)?;
            let found = meta
                .get(KEY_SCHEMA)
                .map_err(io)?
                .map(|g| read_u32(g.value()))
                .transpose()?;
            match found {
                Some(v) if v > SCHEMA_VERSION => {
                    return Err(Error::UnsupportedSchema {
                        found: v,
                        max_supported: SCHEMA_VERSION,
                    });
                }
                Some(_) => {}
                None => {
                    meta.insert(KEY_SCHEMA, SCHEMA_VERSION.to_le_bytes().as_slice())
                        .map_err(io)?;
                }
            }
        }
        // Create the data tables up front so reads on a fresh store return empty
        // instead of failing with "table does not exist".
        {
            wtxn.open_table(MEMORIES).map_err(io)?;
            wtxn.open_table(DEDUP).map_err(io)?;
        }
        wtxn.commit().map_err(io)?;
        Ok(Storage { db })
    }

    /// Atomically initialize-or-read the embedder identity in one write txn: if
    /// absent, record `(identity, dims)` and return it; if present, return the
    /// stored pair (the caller compares). This closes the first-open race.
    pub(crate) fn init_embedder_identity_atomic(
        &self,
        identity: &str,
        dims: usize,
    ) -> Result<(String, usize)> {
        let wtxn = self.db.begin_write().map_err(io)?;
        let result;
        {
            let mut meta = wtxn.open_table(META).map_err(io)?;
            let existing_identity = meta
                .get(KEY_EMB_IDENTITY)
                .map_err(io)?
                .map(|g| String::from_utf8_lossy(g.value()).into_owned());
            let existing_dims = meta
                .get(KEY_EMB_DIMS)
                .map_err(io)?
                .map(|g| read_u64(g.value()))
                .transpose()?;
            match (existing_identity, existing_dims) {
                (Some(i), Some(d)) => result = (i, d as usize),
                _ => {
                    meta.insert(KEY_EMB_IDENTITY, identity.as_bytes())
                        .map_err(io)?;
                    meta.insert(KEY_EMB_DIMS, (dims as u64).to_le_bytes().as_slice())
                        .map_err(io)?;
                    result = (identity.to_string(), dims);
                }
            }
        }
        wtxn.commit().map_err(io)?;
        Ok(result)
    }

    /// Append a batch of memories in a single transaction. Exact `(scope, content)`
    /// duplicates return the existing id instead of inserting.
    ///
    /// Returns `(id, is_new)` 1:1 with `items`: `is_new` is `false` when the item
    /// matched an existing row (so callers can keep an index in sync incrementally).
    pub(crate) fn insert_batch(&self, items: &[MemoryInput]) -> Result<Vec<(MemoryId, bool)>> {
        let wtxn = self.db.begin_write().map_err(io)?;
        let mut ids = Vec::with_capacity(items.len());
        {
            let mut memories = wtxn.open_table(MEMORIES).map_err(io)?;
            let mut meta = wtxn.open_table(META).map_err(io)?;
            let mut dedup = wtxn.open_table(DEDUP).map_err(io)?;

            let mut next = meta
                .get(KEY_NEXT_ID)
                .map_err(io)?
                .map(|g| read_u64(g.value()))
                .transpose()?
                .unwrap_or(1);

            for (scope, content, metadata, embedding, created_at) in items {
                let dkey = dedup_key(scope, content);
                if let Some(found) = dedup.get(dkey.as_str()).map_err(io)? {
                    ids.push((MemoryId::new(found.value()), false));
                    continue;
                }
                let id = next;
                next += 1;
                let mem = Memory {
                    id: MemoryId::new(id),
                    scope: scope.clone(),
                    content: content.clone(),
                    embedding: embedding.clone(),
                    metadata: metadata.clone(),
                    created_at: *created_at,
                    updated_at: *created_at,
                };
                memories
                    .insert(id, encode_record(&mem)?.as_slice())
                    .map_err(io)?;
                dedup.insert(dkey.as_str(), id).map_err(io)?;
                ids.push((MemoryId::new(id), true));
            }
            meta.insert(KEY_NEXT_ID, next.to_le_bytes().as_slice())
                .map_err(io)?;
        }
        wtxn.commit().map_err(io)?;
        Ok(ids)
    }

    /// A single memory by id.
    pub(crate) fn get(&self, id: MemoryId) -> Result<Option<Memory>> {
        let rtxn = self.db.begin_read().map_err(io)?;
        let memories = rtxn.open_table(MEMORIES).map_err(io)?;
        match memories.get(id.raw()).map_err(io)? {
            Some(v) => Ok(Some(decode_record(v.value())?)),
            None => Ok(None),
        }
    }

    /// Candidate memories for a recall: every memory in `scope` (or all scopes if
    /// `None`) matching all metadata `filters`. Full rows — the store ranks and
    /// token-packs from these without a second read.
    pub(crate) fn candidates(
        &self,
        scope: Option<&Scope>,
        filters: &[(String, String)],
    ) -> Result<Vec<Memory>> {
        let rtxn = self.db.begin_read().map_err(io)?;
        let memories = rtxn.open_table(MEMORIES).map_err(io)?;
        let mut out = Vec::new();
        for item in memories.iter().map_err(io)? {
            let (_k, v) = item.map_err(io)?;
            let mem = decode_record(v.value())?;
            let scope_ok = scope.is_none_or(|s| &mem.scope == s);
            let meta_ok = filters
                .iter()
                .all(|(k, val)| mem.metadata.get(k).map(String::as_str) == Some(val.as_str()));
            if scope_ok && meta_ok {
                out.push(mem);
            }
        }
        Ok(out)
    }

    /// Every memory in a scope.
    pub(crate) fn get_all(&self, scope: &Scope) -> Result<Vec<Memory>> {
        let rtxn = self.db.begin_read().map_err(io)?;
        let memories = rtxn.open_table(MEMORIES).map_err(io)?;
        let mut out = Vec::new();
        for item in memories.iter().map_err(io)? {
            let (_k, v) = item.map_err(io)?;
            let mem = decode_record(v.value())?;
            if &mem.scope == scope {
                out.push(mem);
            }
        }
        Ok(out)
    }

    /// Delete one memory (and its dedup entry). Returns whether it existed.
    pub(crate) fn delete(&self, id: MemoryId) -> Result<bool> {
        let wtxn = self.db.begin_write().map_err(io)?;
        let existed;
        {
            let mut memories = wtxn.open_table(MEMORIES).map_err(io)?;
            let found = memories
                .get(id.raw())
                .map_err(io)?
                .map(|v| decode_record(v.value()))
                .transpose()?;
            match found {
                Some(mem) => {
                    memories.remove(id.raw()).map_err(io)?;
                    let mut dedup = wtxn.open_table(DEDUP).map_err(io)?;
                    dedup
                        .remove(dedup_key(&mem.scope, &mem.content).as_str())
                        .map_err(io)?;
                    existed = true;
                }
                None => existed = false,
            }
        }
        wtxn.commit().map_err(io)?;
        Ok(existed)
    }

    /// Remove every memory in a scope; returns the count removed.
    pub(crate) fn clear(&self, scope: &Scope) -> Result<u64> {
        let wtxn = self.db.begin_write().map_err(io)?;
        let mut removed = 0u64;
        {
            let mut memories = wtxn.open_table(MEMORIES).map_err(io)?;
            let mut dedup = wtxn.open_table(DEDUP).map_err(io)?;
            let mut targets = Vec::new();
            {
                for item in memories.iter().map_err(io)? {
                    let (k, v) = item.map_err(io)?;
                    let mem = decode_record(v.value())?;
                    if &mem.scope == scope {
                        targets.push((k.value(), mem.content));
                    }
                }
            }
            for (id, content) in targets {
                memories.remove(id).map_err(io)?;
                dedup
                    .remove(dedup_key(scope, &content).as_str())
                    .map_err(io)?;
                removed += 1;
            }
        }
        wtxn.commit().map_err(io)?;
        Ok(removed)
    }

    /// Number of memories in a scope.
    pub(crate) fn count(&self, scope: &Scope) -> Result<usize> {
        Ok(self.get_all(scope)?.len())
    }

    /// Total number of memories across all scopes (cheap — a table length).
    pub(crate) fn len(&self) -> Result<usize> {
        let rtxn = self.db.begin_read().map_err(io)?;
        let memories = rtxn.open_table(MEMORIES).map_err(io)?;
        Ok(memories.len().map_err(io)? as usize)
    }

    /// Full rows for a set of ids, skipping any that are gone.
    pub(crate) fn rows(&self, ids: &[MemoryId]) -> Result<Vec<Memory>> {
        let rtxn = self.db.begin_read().map_err(io)?;
        let memories = rtxn.open_table(MEMORIES).map_err(io)?;
        let mut out = Vec::with_capacity(ids.len());
        for id in ids {
            if let Some(v) = memories.get(id.raw()).map_err(io)? {
                out.push(decode_record(v.value())?);
            }
        }
        Ok(out)
    }

    /// All distinct scopes currently present.
    pub(crate) fn list_scopes(&self) -> Result<Vec<Scope>> {
        let rtxn = self.db.begin_read().map_err(io)?;
        let memories = rtxn.open_table(MEMORIES).map_err(io)?;
        let mut seen: Vec<Scope> = Vec::new();
        for item in memories.iter().map_err(io)? {
            let (_k, v) = item.map_err(io)?;
            let mem = decode_record(v.value())?;
            if !seen.contains(&mem.scope) {
                seen.push(mem.scope);
            }
        }
        Ok(seen)
    }

    /// Apply a resolved consolidation plan atomically: every `Add`/`Update`/`Delete`
    /// in `ops` is applied in a single write transaction, committed once. Embeddings
    /// arrive pre-computed — this stores the bytes without touching an embedder.
    ///
    /// Not yet called from production code — the store wires this in once
    /// consolidation is plumbed through `MemoryStore` (a later task in this
    /// feature). Allowed dead for now so clippy doesn't block this seam from
    /// landing independently.
    #[allow(dead_code)]
    pub(crate) fn apply_consolidation(
        &self,
        scope: &Scope,
        ops: &[ResolvedOp],
        now: u64,
    ) -> Result<AppliedIds> {
        let wtxn = self.db.begin_write().map_err(io)?;
        let mut applied = AppliedIds::default();
        {
            let mut memories = wtxn.open_table(MEMORIES).map_err(io)?;
            let mut meta = wtxn.open_table(META).map_err(io)?;
            let mut dedup = wtxn.open_table(DEDUP).map_err(io)?;

            let mut next = meta
                .get(KEY_NEXT_ID)
                .map_err(io)?
                .map(|g| read_u64(g.value()))
                .transpose()?
                .unwrap_or(1);

            for op in ops {
                match op {
                    ResolvedOp::Add {
                        content,
                        metadata,
                        embedding,
                    } => {
                        let dkey = dedup_key(scope, content);
                        if let Some(found) = dedup.get(dkey.as_str()).map_err(io)? {
                            applied.added.push(MemoryId::new(found.value()));
                            continue;
                        }
                        let id = next;
                        next += 1;
                        let mem = Memory {
                            id: MemoryId::new(id),
                            scope: scope.clone(),
                            content: content.clone(),
                            embedding: embedding.clone(),
                            metadata: metadata.clone(),
                            created_at: now,
                            updated_at: now,
                        };
                        memories
                            .insert(id, encode_record(&mem)?.as_slice())
                            .map_err(io)?;
                        dedup.insert(dkey.as_str(), id).map_err(io)?;
                        applied.added.push(MemoryId::new(id));
                    }
                    ResolvedOp::Update {
                        id,
                        content,
                        embedding,
                    } => {
                        let existing = memories
                            .get(id.raw())
                            .map_err(io)?
                            .map(|v| decode_record(v.value()))
                            .transpose()?;
                        let Some(old) = existing else {
                            continue;
                        };
                        dedup
                            .remove(dedup_key(&old.scope, &old.content).as_str())
                            .map_err(io)?;
                        let mem = Memory {
                            id: old.id,
                            scope: old.scope.clone(),
                            content: content.clone(),
                            embedding: embedding.clone(),
                            metadata: old.metadata,
                            created_at: old.created_at,
                            updated_at: now,
                        };
                        memories
                            .insert(id.raw(), encode_record(&mem)?.as_slice())
                            .map_err(io)?;
                        dedup
                            .insert(dedup_key(&mem.scope, content).as_str(), id.raw())
                            .map_err(io)?;
                        applied.updated.push(*id);
                    }
                    ResolvedOp::Delete { id } => {
                        let existing = memories
                            .get(id.raw())
                            .map_err(io)?
                            .map(|v| decode_record(v.value()))
                            .transpose()?;
                        let Some(old) = existing else {
                            continue;
                        };
                        memories.remove(id.raw()).map_err(io)?;
                        dedup
                            .remove(dedup_key(&old.scope, &old.content).as_str())
                            .map_err(io)?;
                        applied.deleted.push(*id);
                    }
                }
            }
            meta.insert(KEY_NEXT_ID, next.to_le_bytes().as_slice())
                .map_err(io)?;
        }
        wtxn.commit().map_err(io)?;
        Ok(applied)
    }
}

/// A single resolved consolidation operation, ready to apply against storage.
/// Embeddings are pre-computed by the caller — this type (and `apply_consolidation`)
/// never invoke an embedder.
///
/// Not yet constructed from production code — see `apply_consolidation`.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub(crate) enum ResolvedOp {
    /// Insert a new memory (subject to the usual exact-dedup check).
    Add {
        content: String,
        metadata: HashMap<String, String>,
        embedding: Vec<f32>,
    },
    /// Replace an existing memory's content/embedding in place, advancing `updated_at`.
    Update {
        id: MemoryId,
        content: String,
        embedding: Vec<f32>,
    },
    /// Remove an existing memory.
    Delete { id: MemoryId },
}

/// Ids touched by one `apply_consolidation` call, grouped by the kind of change.
///
/// Not yet constructed from production code — see `apply_consolidation`.
#[allow(dead_code)]
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct AppliedIds {
    pub added: Vec<MemoryId>,
    pub updated: Vec<MemoryId>,
    pub deleted: Vec<MemoryId>,
}

// --- helpers ---------------------------------------------------------------

fn io<E: std::fmt::Display>(e: E) -> Error {
    Error::Io(e.to_string())
}

fn read_u32(b: &[u8]) -> Result<u32> {
    b.try_into()
        .map(u32::from_le_bytes)
        .map_err(|_| Error::Corrupt("expected 4-byte u32 in meta".into()))
}

fn read_u64(b: &[u8]) -> Result<u64> {
    b.try_into()
        .map(u64::from_le_bytes)
        .map_err(|_| Error::Corrupt("expected 8-byte u64 in meta".into()))
}

fn scope_key(scope: &Scope) -> String {
    match scope {
        Scope::Session => "session".to_string(),
        Scope::LongTerm => "long_term".to_string(),
        Scope::Custom(s) => format!("custom:{s}"),
    }
}

fn dedup_key(scope: &Scope, content: &str) -> String {
    // NUL separates the scope key from content.
    format!("{}\u{0}{content}", scope_key(scope))
}

/// Encode a record as `[version: u32 LE][bincode(Memory)]`.
fn encode_record(mem: &Memory) -> Result<Vec<u8>> {
    let mut out = RECORD_VERSION.to_le_bytes().to_vec();
    let body = bincode::serialize(mem).map_err(|e| Error::Corrupt(e.to_string()))?;
    out.extend_from_slice(&body);
    Ok(out)
}

fn decode_record(bytes: &[u8]) -> Result<Memory> {
    if bytes.len() < 4 {
        return Err(Error::Corrupt("record shorter than version tag".into()));
    }
    let (ver, body) = bytes.split_at(4);
    let ver = u32::from_le_bytes(ver.try_into().unwrap());
    if ver > RECORD_VERSION {
        return Err(Error::UnsupportedSchema {
            found: ver,
            max_supported: RECORD_VERSION,
        });
    }
    bincode::deserialize(body).map_err(|e| Error::Corrupt(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_store() -> (tempfile::TempDir, Storage) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("memory.db");
        let store = Storage::open(&path).unwrap();
        (dir, store)
    }

    fn item(scope: Scope, content: &str, embedding: Vec<f32>, created_at: u64) -> MemoryInput {
        (
            scope,
            content.to_string(),
            HashMap::new(),
            embedding,
            created_at,
        )
    }

    fn only_ids(inserted: Vec<(MemoryId, bool)>) -> Vec<MemoryId> {
        inserted.into_iter().map(|(id, _)| id).collect()
    }

    #[test]
    fn insert_then_get_roundtrips() {
        let (_d, s) = temp_store();
        let ids = only_ids(
            s.insert_batch(&[item(Scope::LongTerm, "hello", vec![1.0, 2.0], 7)])
                .unwrap(),
        );
        let got = s.get(ids[0]).unwrap().unwrap();
        assert_eq!(got.content, "hello");
        assert_eq!(got.embedding, vec![1.0, 2.0]);
        assert_eq!(got.scope, Scope::LongTerm);
        assert_eq!(got.created_at, 7);
        assert_eq!(got.updated_at, 7);
        assert_eq!(got.id, ids[0]);
    }

    #[test]
    fn ids_are_unique_and_persist_across_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("m.db");
        let first = {
            let s = Storage::open(&path).unwrap();
            only_ids(
                s.insert_batch(&[
                    item(Scope::Session, "a", vec![1.0], 1),
                    item(Scope::Session, "b", vec![1.0], 1),
                ])
                .unwrap(),
            )
        };
        let s = Storage::open(&path).unwrap();
        let more = only_ids(
            s.insert_batch(&[item(Scope::Session, "c", vec![1.0], 1)])
                .unwrap(),
        );
        assert_ne!(first[0], first[1]);
        assert_ne!(more[0], first[0]);
        assert_ne!(more[0], first[1]);
        assert_eq!(s.count(&Scope::Session).unwrap(), 3);
    }

    #[test]
    fn exact_duplicate_returns_same_id_without_second_row() {
        let (_d, s) = temp_store();
        let a = only_ids(
            s.insert_batch(&[item(Scope::LongTerm, "dup", vec![1.0], 1)])
                .unwrap(),
        );
        let b = only_ids(
            s.insert_batch(&[item(Scope::LongTerm, "dup", vec![1.0], 2)])
                .unwrap(),
        );
        assert_eq!(a[0], b[0]);
        assert_eq!(s.count(&Scope::LongTerm).unwrap(), 1);
    }

    #[test]
    fn intra_batch_duplicate_reuses_id_and_marks_not_new() {
        let (_d, s) = temp_store();
        let inserted = s
            .insert_batch(&[
                item(Scope::LongTerm, "same", vec![1.0], 1),
                item(Scope::LongTerm, "same", vec![1.0], 2),
            ])
            .unwrap();
        assert_eq!(inserted[0].0, inserted[1].0); // same id
        assert!(inserted[0].1, "first occurrence is new");
        assert!(!inserted[1].1, "second occurrence is a dedup hit");
        assert_eq!(s.count(&Scope::LongTerm).unwrap(), 1);
    }

    #[test]
    fn embedder_identity_inits_then_returns_existing() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("m.db");
        {
            let s = Storage::open(&path).unwrap();
            assert_eq!(
                s.init_embedder_identity_atomic("bge", 16).unwrap(),
                ("bge".to_string(), 16)
            );
        }
        let s = Storage::open(&path).unwrap();
        // A different embedder gets the STORED pair back (caller then compares).
        assert_eq!(
            s.init_embedder_identity_atomic("ada", 32).unwrap(),
            ("bge".to_string(), 16)
        );
    }

    #[test]
    fn open_rejects_newer_schema_version() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("m.db");
        drop(Storage::open(&path).unwrap());
        {
            let db = Database::create(&path).unwrap();
            let w = db.begin_write().unwrap();
            {
                let mut m = w.open_table(META).unwrap();
                m.insert(KEY_SCHEMA, 999u32.to_le_bytes().as_slice())
                    .unwrap();
            }
            w.commit().unwrap();
        }
        // `matches!` avoids requiring `Storage: Debug` (which `unwrap_err` would).
        assert!(matches!(
            Storage::open(&path),
            Err(Error::UnsupportedSchema { found: 999, .. })
        ));
    }

    #[test]
    fn record_decode_rejects_short_and_future_version() {
        let (_d, s) = temp_store();
        // Write malformed records straight into the MEMORIES table.
        {
            let w = s.db.begin_write().unwrap();
            {
                let mut t = w.open_table(MEMORIES).unwrap();
                t.insert(1u64, [1u8, 2, 3].as_slice()).unwrap(); // shorter than the version tag
                let mut future = 999u32.to_le_bytes().to_vec();
                future.extend_from_slice(&[0u8; 8]); // future record version + body
                t.insert(2u64, future.as_slice()).unwrap();
            }
            w.commit().unwrap();
        }
        assert!(matches!(s.get(MemoryId::new(1)), Err(Error::Corrupt(_))));
        assert!(matches!(
            s.get(MemoryId::new(2)),
            Err(Error::UnsupportedSchema { found: 999, .. })
        ));
    }

    #[test]
    fn projection_and_get_all_filter_by_scope() {
        let (_d, s) = temp_store();
        s.insert_batch(&[
            item(Scope::Session, "s1", vec![1.0], 1),
            item(Scope::LongTerm, "l1", vec![1.0], 1),
        ])
        .unwrap();
        assert_eq!(s.candidates(Some(&Scope::LongTerm), &[]).unwrap().len(), 1);
        assert_eq!(s.candidates(None, &[]).unwrap().len(), 2);
        assert_eq!(s.get_all(&Scope::Session).unwrap().len(), 1);
    }

    #[test]
    fn projection_filters_by_metadata() {
        let (_d, s) = temp_store();
        let md = HashMap::from([("source".to_string(), "docs".to_string())]);
        s.insert_batch(&[
            (Scope::LongTerm, "with".to_string(), md, vec![1.0], 1),
            (
                Scope::LongTerm,
                "without".to_string(),
                HashMap::new(),
                vec![1.0],
                1,
            ),
        ])
        .unwrap();
        let hits = s
            .candidates(None, &[("source".to_string(), "docs".to_string())])
            .unwrap();
        assert_eq!(hits.len(), 1);
    }

    #[test]
    fn delete_removes_row_and_clears_dedup() {
        let (_d, s) = temp_store();
        let ids = only_ids(
            s.insert_batch(&[item(Scope::LongTerm, "x", vec![1.0], 1)])
                .unwrap(),
        );
        assert!(s.delete(ids[0]).unwrap());
        assert!(s.get(ids[0]).unwrap().is_none());
        assert!(!s.delete(ids[0]).unwrap());
        // dedup cleared: re-adding the same content yields a NEW id.
        let again = only_ids(
            s.insert_batch(&[item(Scope::LongTerm, "x", vec![1.0], 1)])
                .unwrap(),
        );
        assert_ne!(again[0], ids[0]);
    }

    #[test]
    fn clear_scope_and_list_scopes() {
        let (_d, s) = temp_store();
        s.insert_batch(&[
            item(Scope::Session, "s", vec![1.0], 1),
            item(Scope::LongTerm, "l", vec![1.0], 1),
        ])
        .unwrap();
        assert_eq!(s.list_scopes().unwrap().len(), 2);
        assert_eq!(s.clear(&Scope::Session).unwrap(), 1);
        assert_eq!(s.count(&Scope::Session).unwrap(), 0);
        assert_eq!(s.count(&Scope::LongTerm).unwrap(), 1);
    }

    #[test]
    fn apply_consolidation_add_update_delete_in_one_txn() {
        let (_d, s) = temp_store();
        // seed two rows via insert_batch
        let ids = only_ids(
            s.insert_batch(&[
                item(Scope::LongTerm, "lives in NYC", vec![1.0], 10),
                item(Scope::LongTerm, "dup", vec![1.0], 10),
            ])
            .unwrap(),
        );
        let applied = s
            .apply_consolidation(
                &Scope::LongTerm,
                &[
                    ResolvedOp::Update {
                        id: ids[0],
                        content: "lives in SF".into(),
                        embedding: vec![2.0],
                    },
                    ResolvedOp::Delete { id: ids[1] },
                    ResolvedOp::Add {
                        content: "new fact".into(),
                        metadata: HashMap::new(),
                        embedding: vec![3.0],
                    },
                ],
                99,
            )
            .unwrap();
        assert_eq!(applied.updated, vec![ids[0]]);
        assert_eq!(applied.deleted, vec![ids[1]]);
        assert_eq!(applied.added.len(), 1);
        let m = s.get(ids[0]).unwrap().unwrap();
        assert_eq!(m.content, "lives in SF");
        assert_eq!(m.embedding, vec![2.0]);
        assert_eq!(m.created_at, 10); // preserved
        assert_eq!(m.updated_at, 99); // advanced
        assert!(s.get(ids[1]).unwrap().is_none());
        // dedup rewritten: old content re-adds fresh, new content dedups
        let re = only_ids(
            s.insert_batch(&[item(Scope::LongTerm, "lives in NYC", vec![1.0], 1)])
                .unwrap(),
        );
        assert_ne!(re[0], ids[0]);
    }

    #[test]
    fn apply_consolidation_tolerates_missing_ids() {
        let (_d, s) = temp_store();
        let applied = s
            .apply_consolidation(
                &Scope::LongTerm,
                &[
                    ResolvedOp::Update {
                        id: MemoryId::new(999),
                        content: "x".into(),
                        embedding: vec![1.0],
                    },
                    ResolvedOp::Delete {
                        id: MemoryId::new(1000),
                    },
                ],
                1,
            )
            .unwrap();
        assert!(applied.updated.is_empty() && applied.deleted.is_empty());
    }
}
