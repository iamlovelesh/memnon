# Pluggable Tokenizer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the hard-coded `chars/4` token heuristic with a pluggable `Tokenizer` seam, shipping accurate tiktoken (OpenAI) and HuggingFace tokenizers as optional bundled backends.

**Architecture:** A `Tokenizer` trait (always available) with `HeuristicTokenizer` as the default, injected into `MemoryStore` via a new additive `open_with_tokenizer` constructor and consumed at the single search-time call site (`rank_and_pack`). Two bundled impls sit behind independent Cargo features. Nothing is persisted, so there is no schema/format change and no pinning.

**Tech Stack:** Rust (edition 2024), `redb`, `tiktoken-rs` (optional), `tokenizers` (optional), existing `HashEmbedder` for tests.

**Spec:** `docs/specs/2026-11-24-pluggable-tokenizer-design.md`

## Global Constraints

- `MemoryStore::open(path, embedder)` signature stays unchanged (non-breaking). New capability is additive only.
- No change to on-disk format, `SCHEMA_VERSION`, or `RECORD_VERSION`. No tokenizer pinning.
- Core crate stays pure-Rust and light: `tiktoken-rs` and `tokenizers` are `optional = true`; a default build pulls neither. `tokenizers` uses `default-features = false`.
- `count_tokens(&self, text: &str) -> usize` is **infallible**; only constructors return `Result`. `search` keeps its current `Result` behavior (no new error path).
- No `unwrap()`/`expect()` in production paths except the one documented `.expect()` on tiktoken's embedded static vocab.
- `cargo clippy --all-targets -- -D warnings` must pass on the default build and on each feature (`tiktoken`, `hf-tokenizer`, `hf-hub`). `cargo fmt --check` must pass.
- Feature wiring: `tiktoken = ["dep:tiktoken-rs"]`, `hf-tokenizer = ["dep:tokenizers"]`, `hf-hub = ["hf-tokenizer", "tokenizers/http"]`. All independent of `fastembed`.
- Default tiktoken encoding: `o200k_base`. Use `tiktoken_rs::bpe_for_model` (NOT deprecated `get_bpe_from_model`).
- HF counting uses `add_special_tokens = false`.

## Review Focus

- **Empty / whitespace-only content** — `count_tokens("")` returns 0 and packing includes it at zero cost without panic or overflow (owned by Task 1; test added there).
- **CJK / multi-byte content** — a real tokenizer counts strictly **more** tokens than the `chars/4` heuristic for the same text (the core bug being fixed; owned by Tasks 3 & 4; test added there).
- **Unknown model name / malformed `tokenizer.json`** — constructors return `Error::Tokenizer`, never panic (owned by Tasks 3 & 4; test added there).
- **Default `open` path** — a store opened with `open` (no tokenizer) still counts via the heuristic, byte-for-byte unchanged from today (owned by Task 2; test added there).
- **Concurrent counting** — a single tokenizer shared across threads counts in parallel with no `Mutex` and no data race (owned by Task 2; covered by reusing the existing concurrent store test with an injected tokenizer).

---

### Task 1: `Tokenizer` trait + `HeuristicTokenizer`

**Files:**
- Create: `src/tokenizer.rs`
- Modify: `src/lib.rs` (add `pub mod tokenizer;` and re-exports), `src/index.rs` (keep `approx_tokens` as the shared heuristic body; no signature change)
- Test: unit tests in `src/tokenizer.rs`

**Interfaces:**
- Consumes: `crate::index::approx_tokens(s: &str) -> usize` (existing, `pub(crate)`).
- Produces:
  - `pub trait Tokenizer: Send + Sync { fn count_tokens(&self, text: &str) -> usize; }`
  - `pub struct HeuristicTokenizer;` implementing `Tokenizer`.
  - Re-exports from `lib.rs`: `pub use tokenizer::{HeuristicTokenizer, Tokenizer};`

- [ ] **Step 1: Write the failing tests** in `src/tokenizer.rs` `#[cfg(test)]`:

```rust
#[test]
fn heuristic_matches_approx_tokens() {
    assert_eq!(HeuristicTokenizer.count_tokens("abcde"), 2); // ceil(5/4)
    assert_eq!(HeuristicTokenizer.count_tokens("日本語"), 1); // 3 scalars
}
#[test]
fn heuristic_empty_is_zero() {
    assert_eq!(HeuristicTokenizer.count_tokens(""), 0);
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test --lib tokenizer`
Expected: FAIL (module/type not defined).

- [ ] **Step 3: Implement** the trait and `HeuristicTokenizer` in `src/tokenizer.rs`. `count_tokens` delegates to `crate::index::approx_tokens(text)` (DRY — one heuristic). Add `pub mod tokenizer;` and the re-export to `src/lib.rs`.

- [ ] **Step 4: Run to verify pass**

Run: `cargo test --lib tokenizer` → PASS. Then `cargo clippy --all-targets -- -D warnings` → clean.

- [ ] **Step 5: Commit**

```bash
git add src/tokenizer.rs src/lib.rs
git commit -m "feat: add Tokenizer trait and HeuristicTokenizer default"
```

---

### Task 2: Wire tokenizer into `MemoryStore`

**Files:**
- Modify: `src/error.rs` (add variant + Display arm), `src/store.rs` (Inner field, constructors, `rank_and_pack` swap)
- Test: `tests/store.rs`

**Interfaces:**
- Consumes: `Tokenizer`, `HeuristicTokenizer` (Task 1).
- Produces:
  - `pub fn open_with_tokenizer<E: Embedder + 'static, T: Tokenizer + 'static>(path: impl AsRef<Path>, embedder: E, tokenizer: T) -> Result<Self>`
  - `open` becomes a thin wrapper: `Self::open_with_tokenizer(path, embedder, HeuristicTokenizer)`.
  - `Error::Tokenizer(String)`.
  - `Inner` gains `tokenizer: Arc<dyn Tokenizer>`.

- [ ] **Step 1: Write the failing tests** in `tests/store.rs`. Add a deterministic stub:

```rust
// Counts one token per char — distinct from the chars/4 heuristic, so we can
// prove the injected tokenizer (not the default) drives packing.
struct PerCharTokenizer;
impl memnon::Tokenizer for PerCharTokenizer {
    fn count_tokens(&self, text: &str) -> usize { text.chars().count() }
}

#[test]
fn open_with_tokenizer_packs_by_injected_counts() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("m.db");
    let s = MemoryStore::open_with_tokenizer(&path, HashEmbedder::new(64), PerCharTokenizer).unwrap();
    s.add(["aaaa", "bb"], Scope::LongTerm).unwrap(); // 4 and 2 tokens under PerChar
    // Budget of 3 tokens: "aaaa"(4) overflows and is skipped, "bb"(2) fits.
    let hits = s.search("bb", Query::new(), Budget::Tokens(3)).unwrap();
    assert!(hits.iter().any(|h| h.memory.content == "bb"));
    assert!(hits.iter().all(|h| h.memory.content != "aaaa"));
}

#[test]
fn default_open_still_uses_heuristic() {
    // "aaaa" is 1 token under chars/4 (ceil(4/4)); with Budget::Tokens(1) it fits.
    let (_d, s) = store(); // existing helper: MemoryStore::open with HashEmbedder
    s.add(["aaaa"], Scope::LongTerm).unwrap();
    let hits = s.search("aaaa", Query::new(), Budget::Tokens(1)).unwrap();
    assert_eq!(hits.len(), 1);
}
```

Also add `Tokenizer` to the `use memnon::{...}` import line.

- [ ] **Step 2: Run to verify failure**

Run: `cargo test --test store open_with_tokenizer_packs_by_injected_counts default_open_still_uses_heuristic`
Expected: FAIL (`open_with_tokenizer` not found).

- [ ] **Step 3: Implement.**
  - `src/error.rs`: add `Tokenizer(String)` to the enum and `Error::Tokenizer(m) => write!(f, "tokenizer error: {m}")` to `Display`.
  - `src/store.rs`: add `tokenizer: Arc<dyn Tokenizer>` to `Inner`; add `open_with_tokenizer` (constructs `Inner` with `Arc::new(tokenizer)`); rewrite `open` to delegate to it with `HeuristicTokenizer`. In `rank_and_pack`, replace `approx_tokens(&m.content)` with `self.inner.tokenizer.count_tokens(&m.content)` and drop the now-unused `approx_tokens` import (keep `pack`, `rank`).

- [ ] **Step 4: Run to verify pass**

Run: `cargo test --test store` → PASS (new + all existing). Then `cargo clippy --all-targets -- -D warnings` and `cargo fmt --check` → clean.

- [ ] **Step 5: Commit**

```bash
git add src/error.rs src/store.rs tests/store.rs
git commit -m "feat: inject Tokenizer into MemoryStore via open_with_tokenizer"
```

---

### Task 3: `TiktokenTokenizer` (`tiktoken` feature)

**Files:**
- Create: `src/tiktoken_tokenizer.rs`
- Modify: `Cargo.toml` (optional dep + `tiktoken` feature), `src/lib.rs` (gated re-export), `.github/workflows/ci.yml` (feature clippy job)
- Test: `tests/tiktoken.rs` (`#![cfg(feature = "tiktoken")]`)

**Interfaces:**
- Consumes: `Tokenizer` (Task 1), `Error::Tokenizer` (Task 2).
- Produces:
  - `pub struct TiktokenTokenizer` with `pub fn new() -> Self`, `pub fn with_model(model: &str) -> Result<Self>`, `impl Default`, `impl Tokenizer`.
  - Gated re-export: `#[cfg(feature = "tiktoken")] pub use tiktoken_tokenizer::TiktokenTokenizer;`

- [ ] **Step 1: Add the dependency and feature** to `Cargo.toml`: `tiktoken-rs = { version = "<latest 0.x>", optional = true }` and `tiktoken = ["dep:tiktoken-rs"]`. Add `mod tiktoken_tokenizer;` gated in `lib.rs`.

- [ ] **Step 2: Write the failing tests** in `tests/tiktoken.rs`:

```rust
#![cfg(feature = "tiktoken")]
use memnon::{TiktokenTokenizer, Tokenizer, Error};

#[test]
fn counts_ascii_anchor() {
    // Stable small anchor for o200k_base; confirm exact value on the pinned version.
    assert_eq!(TiktokenTokenizer::new().count_tokens("hello world"), 2);
}
#[test]
fn cjk_counts_more_than_heuristic() {
    let t = TiktokenTokenizer::new();
    let text = "日本語のテキストです";
    let heuristic = text.chars().count().div_ceil(4);
    assert!(t.count_tokens(text) > heuristic);
}
#[test]
fn with_model_ok_and_unknown_errors() {
    assert!(TiktokenTokenizer::with_model("gpt-4o").is_ok());
    assert!(matches!(TiktokenTokenizer::with_model("not-a-real-model"), Err(Error::Tokenizer(_))));
}
```

- [ ] **Step 3: Run to verify failure**

Run: `cargo test --features tiktoken --test tiktoken`
Expected: FAIL (type not defined).

- [ ] **Step 4: Implement** `src/tiktoken_tokenizer.rs` per the spec: `new()` wraps `o200k_base().expect("embedded o200k vocab")`; `with_model` wraps `bpe_for_model(model).map_err(|e| Error::Tokenizer(e.to_string()))`; `count_tokens` returns `self.bpe.encode_ordinary(text).len()`; add `impl Default`. Add the gated `pub use` to `lib.rs`.

- [ ] **Step 5: Run to verify pass**

Run: `cargo test --features tiktoken --test tiktoken` → PASS. Then `cargo clippy --features tiktoken --all-targets -- -D warnings` → clean. (If the ASCII anchor differs on the pinned version, correct the expected value; the CJK property is the load-bearing assertion.)

- [ ] **Step 6: Add CI job.** In `.github/workflows/ci.yml`, extend the feature-lint job (mirroring the `fastembed` job) to also run `cargo clippy --features tiktoken --all-targets -- -D warnings`.

- [ ] **Step 7: Commit**

```bash
git add Cargo.toml Cargo.lock src/tiktoken_tokenizer.rs src/lib.rs tests/tiktoken.rs .github/workflows/ci.yml
git commit -m "feat: add TiktokenTokenizer behind tiktoken feature"
```

---

### Task 4: `HfTokenizer` local (`hf-tokenizer` feature)

**Files:**
- Create: `src/hf_tokenizer.rs`, `tests/fixtures/tokenizer.json` (small permissive fixture)
- Modify: `Cargo.toml` (optional dep + `hf-tokenizer` feature), `src/lib.rs` (gated re-export), `.github/workflows/ci.yml`
- Test: `tests/hf_tokenizer.rs` (`#![cfg(feature = "hf-tokenizer")]`)

**Interfaces:**
- Consumes: `Tokenizer` (Task 1), `Error::Tokenizer` (Task 2).
- Produces:
  - `pub struct HfTokenizer` with `pub fn from_file(path: impl AsRef<Path>) -> Result<Self>`, `pub fn from_bytes(bytes: &[u8]) -> Result<Self>`, `impl Tokenizer`.
  - Gated re-export: `#[cfg(feature = "hf-tokenizer")] pub use hf_tokenizer::HfTokenizer;`

- [ ] **Step 1 (spike — review guardrail #2): prove the dependency is light.** Add to `Cargo.toml`: `tokenizers = { version = "<latest 0.x>", optional = true, default-features = false, features = ["<minimal>"] }` and `hf-tokenizer = ["dep:tokenizers"]`. Run `cargo build --features hf-tokenizer` and `cargo tree --features hf-tokenizer -e no-dev`. Confirm it compiles with no default features and record the dep count. If `default-features = false` won't build a load+encode path, add the smallest feature that does and note it. This step gates the rest of the task.

- [ ] **Step 2: Add a tiny `tokenizer.json` fixture** under `tests/fixtures/tokenizer.json` (any small permissively-licensed tokenizer; a BERT/wordpiece or byte-BPE json is fine). Commit it with the task.

- [ ] **Step 3: Write the failing tests** in `tests/hf_tokenizer.rs`:

```rust
#![cfg(feature = "hf-tokenizer")]
use memnon::{HfTokenizer, Tokenizer, Error};

const FIXTURE: &[u8] = include_bytes!("fixtures/tokenizer.json");

#[test]
fn from_bytes_loads_and_counts() {
    let t = HfTokenizer::from_bytes(FIXTURE).unwrap();
    assert!(t.count_tokens("hello world") >= 2); // fixture-independent lower bound
}
#[test]
fn cjk_counts_more_than_heuristic() {
    let t = HfTokenizer::from_bytes(FIXTURE).unwrap();
    let text = "日本語のテキストです";
    assert!(t.count_tokens(text) > text.chars().count().div_ceil(4));
}
#[test]
fn malformed_bytes_error() {
    assert!(matches!(HfTokenizer::from_bytes(b"not json"), Err(Error::Tokenizer(_))));
}
```

- [ ] **Step 4: Run to verify failure**

Run: `cargo test --features hf-tokenizer --test hf_tokenizer`
Expected: FAIL (type not defined).

- [ ] **Step 5: Implement** `src/hf_tokenizer.rs` per the spec: `from_file`/`from_bytes` wrap `tokenizers::Tokenizer::{from_file,from_bytes}` mapping errors to `Error::Tokenizer`; `count_tokens` returns `self.inner.encode(text, false).map(|e| e.len()).unwrap_or_else(|_| HeuristicTokenizer.count_tokens(text))` (comment the fallback as belt-and-suspenders — review guardrail #4). Add gated `pub use` to `lib.rs`.

- [ ] **Step 6: Run to verify pass**

Run: `cargo test --features hf-tokenizer --test hf_tokenizer` → PASS. Then `cargo clippy --features hf-tokenizer --all-targets -- -D warnings` → clean.

- [ ] **Step 7: Add CI job** for `cargo clippy --features hf-tokenizer --all-targets -- -D warnings` (alongside the tiktoken job).

- [ ] **Step 8: Commit**

```bash
git add Cargo.toml Cargo.lock src/hf_tokenizer.rs src/lib.rs tests/hf_tokenizer.rs tests/fixtures/tokenizer.json .github/workflows/ci.yml
git commit -m "feat: add HfTokenizer (local tokenizer.json) behind hf-tokenizer feature"
```

---

### Task 5: `HfTokenizer::from_pretrained` (`hf-hub` feature)

**Files:**
- Modify: `Cargo.toml` (`hf-hub` feature), `src/hf_tokenizer.rs` (gated method), `.github/workflows/ci.yml`
- Test: `tests/hf_tokenizer.rs` (add one `#[ignore]`d online test)

**Interfaces:**
- Produces: `#[cfg(feature = "hf-hub")] pub fn from_pretrained(model_id: &str) -> Result<Self>` on `HfTokenizer`.

- [ ] **Step 1: Add the feature** to `Cargo.toml`: `hf-hub = ["hf-tokenizer", "tokenizers/http"]`.

- [ ] **Step 2: Write the ignored online test** in `tests/hf_tokenizer.rs`:

```rust
#[test]
#[cfg(feature = "hf-hub")]
#[ignore = "downloads a tokenizer.json from the HF Hub; run manually"]
fn from_pretrained_fetches_and_counts() {
    let t = HfTokenizer::from_pretrained("bert-base-uncased").unwrap();
    assert!(t.count_tokens("hello world") >= 2);
}
```

- [ ] **Step 3: Verify it is collected but skipped**

Run: `cargo test --features hf-hub --test hf_tokenizer -- --list`
Expected: the test is listed (ignored). It compiles.

- [ ] **Step 4: Implement** the gated `from_pretrained` in `src/hf_tokenizer.rs`: wrap `tokenizers::Tokenizer::from_pretrained(model_id, None).map_err(|e| Error::Tokenizer(e.to_string()))`.

- [ ] **Step 5: Verify compile + manual run**

Run: `cargo clippy --features hf-hub --all-targets -- -D warnings` → clean. Optionally `cargo test --features hf-hub --test hf_tokenizer -- --ignored` (needs network) → PASS.

- [ ] **Step 6: Add CI job** for `cargo clippy --features hf-hub --all-targets -- -D warnings`.

- [ ] **Step 7: Commit**

```bash
git add Cargo.toml Cargo.lock src/hf_tokenizer.rs tests/hf_tokenizer.rs .github/workflows/ci.yml
git commit -m "feat: add HfTokenizer::from_pretrained behind hf-hub feature"
```

---

### Task 6: Example, README, and roadmap

**Files:**
- Create: `examples/custom_tokenizer.rs`
- Modify: `README.md`, `src/query.rs` (doc comment on `Budget::Tokens` — review guardrail #3)

**Interfaces:**
- Consumes: `Tokenizer`, `open_with_tokenizer` (Tasks 1–2). Dependency-free (no feature, no new dep).

- [ ] **Step 1: Write `examples/custom_tokenizer.rs`** — a dependency-free custom `Tokenizer` (e.g. whitespace/word-based `WordTokenizer`), opened via `open_with_tokenizer`, printing a `Budget::Tokens` recall. Doc comment points to the bundled `tiktoken`/`hf-tokenizer` features and the HF `tokenizers` crate for production.

- [ ] **Step 2: Verify it builds** (base job coverage)

Run: `cargo build --examples`
Expected: PASS (no features needed).

- [ ] **Step 3: Update docs.**
  - `README.md`: check off "real tokenizer" in the roadmap; document the three features and the `Tokenizer` trait; add the "match your **generation** LLM, not the embedding model" guidance; short tiktoken + HF snippets.
  - `src/query.rs`: on `Budget::Tokens`, add one line — counts memory **content only**; callers using chat/message formats should leave headroom for per-message framing tokens (`add_special_tokens = false`).

- [ ] **Step 4: Full verification**

Run: `cargo fmt --check`, `cargo test`, `cargo clippy --all-targets -- -D warnings`, and the three feature clippy runs (`tiktoken`, `hf-tokenizer`, `hf-hub`) → all clean.

- [ ] **Step 5: Commit**

```bash
git add examples/custom_tokenizer.rs README.md src/query.rs
git commit -m "docs: tokenizer example, README features, and Budget::Tokens guidance"
```

---

## Notes for the executor

- Pin exact `tiktoken-rs` / `tokenizers` versions when adding them (latest 0.x at build time); the plan's version placeholders are intentional.
- Task 4 Step 1 (the dependency spike) is a real gate: if the `tokenizers` dep is far heavier than expected, stop and raise it before continuing — it bears on the crate's "stays light" promise.
- Tasks are mostly sequential by interface: Task 2 depends on Task 1; Tasks 3–5 depend on Tasks 1–2; Task 6 depends on Tasks 1–2. Tasks 3 and 4 are independent of each other and could be done in either order.
