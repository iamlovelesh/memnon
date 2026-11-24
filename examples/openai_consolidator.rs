//! A real, OpenAI-backed [`memnon::Consolidator`] example — the
//! bring-your-own-reconciliation-policy path, mirroring how
//! `examples/openai_embedder.rs` is the bring-your-own-embedder path.
//!
//! **Scope boundary (read this first):** this example reconciles
//! already-discrete facts (e.g. `"the user moved to San Francisco"`) against
//! related memories already in the store. **Extraction — turning a raw
//! conversation or document into those discrete facts in the first place —
//! is the caller's job**, upstream of `MemoryStore::consolidate`. memnon's
//! consolidation seam starts at "I have a fact; what should happen to it?",
//! not "here's a transcript, figure out the facts."
//!
//! Run: `OPENAI_API_KEY=sk-... cargo run --example openai_consolidator`
//! (Needs network + a key. Compile-checked in CI; not run there.)

use memnon::{
    ConsolidationOp, Consolidator, Error, HashEmbedder, MemoryId, MemoryStore, RelatedMemory,
    Result, Scope,
};

/// Reconciles incoming facts against related memories via an OpenAI
/// chat-completions call. The model is asked to return a JSON array of
/// operations; see [`parse_ops`] for the schema and parsing rules.
struct OpenAiConsolidator {
    api_key: String,
    model: String,
}

impl OpenAiConsolidator {
    fn new(api_key: String) -> Self {
        Self {
            api_key,
            model: "gpt-4o-mini".to_string(),
        }
    }
}

impl Consolidator for OpenAiConsolidator {
    fn reconcile(&self, incoming: &str, related: &[RelatedMemory]) -> Result<Vec<ConsolidationOp>> {
        let related_json: Vec<serde_json::Value> = related
            .iter()
            .map(|r| serde_json::json!({ "id": r.id, "content": r.content }))
            .collect();

        let prompt = format!(
            "New fact: {incoming}\n\
             Related existing memories (JSON array of {{id, content}}): {related}\n\n\
             Decide how to reconcile the new fact against the related memories. \
             Respond with ONLY a JSON array of operations (no prose, no markdown \
             fences), where each entry is one of:\n\
             {{\"op\": \"add\"}} — the new fact is distinct; store it as-is\n\
             {{\"op\": \"update\", \"id\": <id>, \"content\": \"...\"}} — overwrite an \
             existing memory's content with the merged/superseding content\n\
             {{\"op\": \"delete\", \"id\": <id>}} — remove an existing memory that the \
             new fact supersedes outright\n\
             {{\"op\": \"noop\"}} — the new fact is already covered; do nothing\n\
             If nothing related applies, reply with exactly [{{\"op\": \"add\"}}].",
            incoming = incoming,
            related = serde_json::Value::Array(related_json),
        );

        let body = serde_json::json!({
            "model": self.model,
            "messages": [{ "role": "user", "content": prompt }],
        });
        let response = ureq::post("https://api.openai.com/v1/chat/completions")
            .set("Authorization", &format!("Bearer {}", self.api_key))
            .send_json(body)
            .map_err(|e| Error::Consolidate(e.to_string()))?;
        let json: serde_json::Value = response
            .into_json()
            .map_err(|e| Error::Consolidate(e.to_string()))?;

        let text = json["choices"][0]["message"]["content"]
            .as_str()
            .ok_or_else(|| Error::Consolidate("response had no message content".into()))?;

        parse_ops(text)
    }
}

/// Parse the model's response text into ops. Parsing is defensive: malformed
/// JSON, a non-array top level, an unrecognized `op`, or a missing required
/// field all map to `Error::Consolidate` rather than panicking — the model's
/// output is untrusted input.
fn parse_ops(text: &str) -> Result<Vec<ConsolidationOp>> {
    let value: serde_json::Value =
        serde_json::from_str(text.trim()).map_err(|e| Error::Consolidate(e.to_string()))?;
    let items = value
        .as_array()
        .ok_or_else(|| Error::Consolidate("expected a JSON array of ops".into()))?;

    items.iter().map(parse_op).collect()
}

fn parse_op(item: &serde_json::Value) -> Result<ConsolidationOp> {
    let op = item["op"]
        .as_str()
        .ok_or_else(|| Error::Consolidate("op entry missing string `op` field".into()))?;

    match op {
        "add" => Ok(ConsolidationOp::Add),
        "noop" => Ok(ConsolidationOp::Noop),
        "update" => Ok(ConsolidationOp::Update {
            id: parse_id(&item["id"])?,
            content: item["content"]
                .as_str()
                .ok_or_else(|| Error::Consolidate("`update` op missing `content`".into()))?
                .to_string(),
        }),
        "delete" => Ok(ConsolidationOp::Delete {
            id: parse_id(&item["id"])?,
        }),
        other => Err(Error::Consolidate(format!("unknown op: {other}"))),
    }
}

/// `MemoryId`'s inner value is crate-private (ids can't be forged), but it
/// derives `Deserialize`, so this is the sanctioned way to turn a related
/// memory's numeric id (as echoed back by the model) into a real `MemoryId`.
fn parse_id(value: &serde_json::Value) -> Result<MemoryId> {
    serde_json::from_value(value.clone())
        .map_err(|_| Error::Consolidate("op missing or invalid `id`".into()))
}

fn main() -> Result<()> {
    let key = std::env::var("OPENAI_API_KEY").unwrap_or_default();
    if key.is_empty() {
        eprintln!("Set OPENAI_API_KEY to run this example.");
        return Ok(());
    }

    let path = std::env::temp_dir().join("memnon-openai-consolidator.db");
    let _ = std::fs::remove_file(&path);
    let store = MemoryStore::open(&path, HashEmbedder::new(64))?;

    store.add(["the user lives in NYC"], Scope::LongTerm)?;

    let consolidator = OpenAiConsolidator::new(key);
    let outcome = store.consolidate(
        ["the user moved to San Francisco"],
        Scope::LongTerm,
        &consolidator,
    )?;

    println!("{outcome:?}");
    Ok(())
}
