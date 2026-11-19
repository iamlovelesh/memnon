//! A real, **semantic** [`memnon::Embedder`] backed by an OpenAI-compatible
//! `/embeddings` endpoint — the bring-your-own-model path.
//!
//! Run: `OPENAI_API_KEY=sk-... cargo run --example openai_embedder`
//! (Needs network + a key. Compile-checked in CI; not run there.)

use memnon::{Budget, Embedder, Error, MemoryStore, Query, Result, Scope};

struct OpenAiEmbedder {
    api_key: String,
    model: String,
    dims: usize,
    identity: String,
}

impl OpenAiEmbedder {
    fn new(api_key: String) -> Self {
        let model = "text-embedding-3-small".to_string();
        Self {
            identity: format!("openai:{model}"),
            dims: 1536,
            model,
            api_key,
        }
    }
}

impl Embedder for OpenAiEmbedder {
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let body = serde_json::json!({ "model": self.model, "input": texts });
        let response = ureq::post("https://api.openai.com/v1/embeddings")
            .set("Authorization", &format!("Bearer {}", self.api_key))
            .send_json(body)
            .map_err(|e| Error::Embed(e.to_string()))?;
        let json: serde_json::Value = response
            .into_json()
            .map_err(|e| Error::Embed(e.to_string()))?;

        let data = json["data"]
            .as_array()
            .ok_or_else(|| Error::Embed("response had no `data` array".into()))?;
        let mut out = Vec::with_capacity(data.len());
        for item in data {
            let vector = item["embedding"]
                .as_array()
                .ok_or_else(|| Error::Embed("response item had no `embedding`".into()))?;
            out.push(
                vector
                    .iter()
                    .map(|x| x.as_f64().unwrap_or(0.0) as f32)
                    .collect(),
            );
        }
        Ok(out)
    }

    fn dimensions(&self) -> usize {
        self.dims
    }

    fn identity(&self) -> &str {
        &self.identity
    }
}

fn main() -> Result<()> {
    let key = std::env::var("OPENAI_API_KEY").unwrap_or_default();
    if key.is_empty() {
        eprintln!("Set OPENAI_API_KEY to run this example.");
        return Ok(());
    }

    let path = std::env::temp_dir().join("memnon-openai.db");
    let _ = std::fs::remove_file(&path);
    let store = MemoryStore::open(&path, OpenAiEmbedder::new(key))?;

    store.add(
        [
            "Paris is the capital of France",
            "Mitochondria are the powerhouse of the cell",
        ],
        Scope::LongTerm,
    )?;

    let hits = store.search(
        "What is the capital of France?",
        Query::new(),
        Budget::TopK(1),
    )?;
    for hit in &hits {
        println!("[{:.3}] {}", hit.score, hit.memory.content);
    }

    Ok(())
}
