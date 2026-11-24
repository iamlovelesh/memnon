# Pluggable Tokenizer — Design

**Date:** 2026-11-24
**Status:** Proposed
**Milestone:** v1.2 (the "real tokenizer" item from the v1.1 roadmap)

## Intent & why

`Budget::Tokens` is memnon's headline feature: recall the most-similar memories
that *fit* a token budget. Today the token count comes from `approx_tokens` in
`src/index.rs` — `chars().count().div_ceil(4)`. That heuristic under-counts
token-dense scripts (CJK and friends) by several times, so a `Budget::Tokens(n)`
can pack far more real tokens than the caller asked for. This is the one place
the budget-first promise leaks.

This work replaces the hard-coded heuristic with a **pluggable `Tokenizer`
seam**, mirroring the existing `Embedder` seam. The `chars/4` heuristic stays as
the always-on default, and **two** real tokenizers ship as bundled impls, each
behind its own optional feature:

- **`TiktokenTokenizer`** (`tiktoken` feature) — OpenAI BPE (GPT-4o / 4.1 / o-series
  / GPT-5, GPT-4/3.5).
- **`HfTokenizer`** (`hf-tokenizer` feature) — HuggingFace `tokenizers`, which
  covers the widely-used non-OpenAI families: Claude, Llama, Mistral, Gemini, and
  BERT/wordpiece models.

Together these cover essentially every LLM a caller is likely to budget against.
The trait keeps the seam open for anything exotic.

> **Which tokenizer to pick:** the budget is about the **LLM whose context window
> you're filling**, not the embedding model. Use tiktoken for OpenAI generation
> models; use `HfTokenizer` with that model's `tokenizer.json` for Llama/Claude/
> Mistral/Gemini/etc.

## Key property — no persistence, no pinning

Token counts are computed **only at search time**, at a single call site
(`src/store.rs`, `rank_and_pack`), and are **never written to disk**. Unlike the
embedder (whose stored vectors are meaningless under a different model, hence the
pin), the tokenizer:

- needs **no** pinning to the store file,
- needs **no** schema/record version bump,
- can be **changed or swapped freely** between opens with zero compatibility risk.

The change stays self-contained: a new trait, a default impl, two optional
bundled impls, and a one-line swap at the call site.

## Goals

- A public `Tokenizer` trait: model-agnostic, `Send + Sync`, infallible counting.
- `HeuristicTokenizer` — the current `chars/4` logic, always available, the default.
- `TiktokenTokenizer` behind the `tiktoken` feature: accurate OpenAI BPE counts.
  - Defaults to **`o200k_base`** (GPT-4o / 4.1 / o-series / GPT-5 family).
  - Also constructible **by model name** (`"gpt-4o"`, `"gpt-4"`, …) and by a named
    encoding (`cl100k_base` for GPT-4/3.5 and the `text-embedding-3-*` models).
- `HfTokenizer` behind the `hf-tokenizer` feature: HuggingFace `tokenizers`.
  - **Local construction** (`hf-tokenizer`): `from_file(path)` and `from_bytes(&[u8])`,
    loading a standard `tokenizer.json`. No network.
  - **Hub download** (`hf-hub` sub-feature): `from_pretrained(model_id)` fetches the
    `tokenizer.json` by name from the HuggingFace Hub on first use, then caches it
    (like `fastembed`'s model download). Kept in a separate sub-feature so the
    local-only path carries no HTTP stack.
- A caller supplies a tokenizer when opening the store; `MemoryStore::open` keeps
  its current signature and defaults to the heuristic (non-breaking).
- A dependency-free example demonstrating a custom `Tokenizer` impl (the raw seam).

## Non-goals

- No change to the on-disk format, schema version, or record version.
- No tokenizer identity pinning or mismatch checking (counts aren't persisted).
- No per-scope or write-time tokenization.
- Hub download (`from_pretrained`) stays **opt-in behind its own `hf-hub` sub-feature**
  — it is not pulled by the default build or by the local-only `hf-tokenizer` feature.

## Anti-goals

- Don't make `MemoryStore::open` breaking.
- Don't pull heavy/native deps into the **core** crate. Both tokenizer crates are
  pure-Rust (no ONNX/native) and stay behind their own optional features; a default
  build pulls neither. The HF `tokenizers` dep is configured inference-only
  (`default-features = false`, no trainer/onig extras) to stay as light as possible.
- Don't silently change token counts based on which Cargo features are compiled
  (the reason we chose a pluggable trait over a feature that swaps `approx_tokens`
  internally).

## Approaches considered

- **A. Pluggable `Tokenizer` trait + bundled impls (CHOSEN).** Mirrors the `Embedder`
  seam the crate already commits to. Heuristic stays the default; tiktoken and HF are
  opt-in backends; the trait stays open for anything else. Trade-off: adds public API
  surface (a trait + one store constructor). Chosen for consistency and because "which
  tokenizer matches my context window" is genuinely the caller's decision.
- **B. Pure feature swap, no API change.** A feature just makes `approx_tokens` call a
  real tokenizer internally. Simplest, zero new surface — but no bring-your-own, no way
  to pick a backend, and two builds give silently different counts. Rejected.
- **C. Builder API** (`MemoryStore::builder(path).embedder(e).tokenizer(t).open()`).
  More flexible, but a larger, more invasive public-API change than this milestone
  warrants. Rejected as premature; can come later if more open-time options accumulate.

## Design

### The trait (always available, in `src/tokenizer.rs`)

```rust
/// Counts tokens in text for `Budget::Tokens` packing. The pluggable seam that
/// lets the budget match the caller's actual LLM context window.
///
/// `Send + Sync` so one tokenizer is shared across threads behind the store's Arc.
/// Counting is infallible: a constructed tokenizer counts any `&str`.
pub trait Tokenizer: Send + Sync {
    fn count_tokens(&self, text: &str) -> usize;
}

/// The always-on default: ~1 token per 4 Unicode scalar values, rounded up.
/// Non-semantic and script-insensitive — accurate enough for Latin text, and the
/// zero-dependency fallback when no tokenizer feature is enabled.
pub struct HeuristicTokenizer;

impl Tokenizer for HeuristicTokenizer {
    fn count_tokens(&self, text: &str) -> usize {
        text.chars().count().div_ceil(4) // reuses today's approx_tokens logic
    }
}
```

### Bundled impl 1 — tiktoken (`tiktoken` feature, `src/tiktoken_tokenizer.rs`)

```rust
/// On-device OpenAI BPE token counting via `tiktoken-rs`. Behind the `tiktoken`
/// feature so the core crate stays pure-Rust and light.
pub struct TiktokenTokenizer { bpe: tiktoken_rs::CoreBPE }

impl TiktokenTokenizer {
    /// Default encoding: `o200k_base` (GPT-4o / 4.1 / o-series / GPT-5 family).
    /// Infallible: the vocab is embedded in the binary. The internal `o200k_base()`
    /// returns `Result`, but it can only fail on a corrupt *compiled-in* vocab — a
    /// truly-unreachable state — so this uses a documented `.expect()` (the one
    /// sanctioned `expect` per the Rust rules: reserved for unreachable states).
    pub fn new() -> Self { /* Self { bpe: o200k_base().expect("embedded o200k vocab") } */ }

    /// By OpenAI model name, e.g. "gpt-4o", "gpt-4". Unknown names error.
    /// Use `tiktoken_rs::bpe_for_model` (NOT the deprecated `get_bpe_from_model`).
    pub fn with_model(model: &str) -> Result<Self> { /* bpe_for_model(model) */ }
}

// Satisfies `clippy::new_without_default` (feature CI jobs run `-D warnings`).
impl Default for TiktokenTokenizer {
    fn default() -> Self { Self::new() }
}

impl Tokenizer for TiktokenTokenizer {
    fn count_tokens(&self, text: &str) -> usize {
        // `encode_ordinary` = no special tokens, matching the HF `add_special_tokens
        // = false` choice below (special tokens wrap the whole prompt, not each memory).
        self.bpe.encode_ordinary(text).len()
    }
}
```

`encode_ordinary` takes `&self`, so — unlike `FastEmbedEmbedder` — **no `Mutex` is
needed**; concurrent searches count in parallel. (`CoreBPE: Send + Sync` is required
for `Arc<dyn Tokenizer>` and is enforced at compile time; confirm against the pinned
version — if it ever regressed, the trait bound would fail to compile, not misbehave.)

### Bundled impl 2 — HuggingFace (`hf-tokenizer` feature, `src/hf_tokenizer.rs`)

```rust
/// On-device token counting via HuggingFace `tokenizers` — covers Llama, Mistral,
/// Gemini, Claude, and BERT/wordpiece models. Behind the `hf-tokenizer` feature.
/// Load a model's standard `tokenizer.json` from disk or bytes (local-first).
pub struct HfTokenizer { inner: tokenizers::Tokenizer }

impl HfTokenizer {
    /// Load from a `tokenizer.json` file. Bad path / malformed json -> Error::Tokenizer.
    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self> { /* ... */ }

    /// Load from in-memory `tokenizer.json` bytes (e.g. `include_bytes!`).
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> { /* ... */ }

    /// Fetch a model's `tokenizer.json` by Hub id (e.g. "meta-llama/Llama-3.1-8B")
    /// on first use, then cache it. Network on first call; offline thereafter.
    /// Behind the `hf-hub` sub-feature (uses the `tokenizers` crate's `http`
    /// feature, which pulls `hf-hub` internally). Errors -> Error::Tokenizer.
    #[cfg(feature = "hf-hub")]
    pub fn from_pretrained(model_id: &str) -> Result<Self> {
        /* tokenizers::Tokenizer::from_pretrained(model_id, None) */
    }
}

impl Tokenizer for HfTokenizer {
    fn count_tokens(&self, text: &str) -> usize {
        // `add_special_tokens = false`: special tokens ([CLS]/[SEP], BOS/EOS) are
        // added once around the whole prompt, not per memory, so per-memory counts
        // must exclude them. `encode` is `&self` (no Mutex). On the pathological
        // encode error, fall back to the heuristic so counting stays infallible.
        self.inner
            .encode(text, false)
            .map(|e| e.len())
            .unwrap_or_else(|_| HeuristicTokenizer.count_tokens(text))
    }
}
```

`tokenizers::Tokenizer` is `Send + Sync` with `&self` encoding — no `Mutex` needed.
The HF crate is pulled with `default-features = false` (inference only; no trainer or
`onig` extras) to minimize weight.

### Store wiring (`src/store.rs`)

- `Inner` gains `tokenizer: Arc<dyn Tokenizer>`.
- `MemoryStore::open(path, embedder)` — unchanged signature — defaults `tokenizer`
  to `Arc::new(HeuristicTokenizer)`.
- New `MemoryStore::open_with_tokenizer(path, embedder, tokenizer)` for an explicit
  tokenizer. (Non-breaking: additive. Generic over `T: Tokenizer + 'static`, mirroring
  how `open` is generic over `E: Embedder`.)
  - **Decided (staff review):** keep the two-constructor shape (`open` / `open_with_*`)
    rather than a builder or options struct — idiomatic and YAGNI while the crate is
    pre-1.0. **Trigger to revisit:** the moment a *third* open-time knob appears,
    migrate to an extensible `StoreOptions` struct (free to do pre-1.0).
- `rank_and_pack` replaces `approx_tokens(&m.content)` with
  `self.inner.tokenizer.count_tokens(&m.content)`. No new error path — counting is
  infallible, so `search` keeps its current `Result` behavior.

### Errors (`src/error.rs`)

Add a non-breaking variant (the enum is `#[non_exhaustive]`):

```rust
/// A tokenizer failed to initialize (unknown model name, bad/missing tokenizer file).
Tokenizer(String),
```

Used only by fallible *constructors* (`with_model`, `from_file`, `from_bytes`);
`count_tokens` never fails.

### `index.rs`

`approx_tokens` stays (now the body of `HeuristicTokenizer::count_tokens`, either
called through or kept as a private helper). Its existing unit tests stay green.

### Cargo (`Cargo.toml`)

```toml
[dependencies]
tiktoken-rs = { version = "<latest 0.x>", optional = true }
tokenizers  = { version = "<latest 0.x>", optional = true, default-features = false, features = ["<minimal inference set>"] }

[features]
tiktoken     = ["dep:tiktoken-rs"]
hf-tokenizer = ["dep:tokenizers"]
# Adds HfTokenizer::from_pretrained (Hub download). Enables the tokenizers crate's
# own `http` feature, which pulls hf-hub — so the local-only path stays HTTP-free.
hf-hub       = ["hf-tokenizer", "tokenizers/http"]
```

Pin exact versions and settle the minimal `tokenizers` feature set at implementation
time. All tokenizer features are independent of `fastembed`; `hf-hub` implies
`hf-tokenizer`.

### Public surface (`src/lib.rs`)

```rust
pub use tokenizer::{HeuristicTokenizer, Tokenizer};
#[cfg(feature = "tiktoken")]
pub use tiktoken_tokenizer::TiktokenTokenizer;
#[cfg(feature = "hf-tokenizer")]
pub use hf_tokenizer::HfTokenizer;
```

### Example & docs

- `examples/custom_tokenizer.rs`: a **dependency-free** custom `Tokenizer` impl (e.g.
  whitespace/word-based) showing how to wrap any counter behind the trait and pass it
  to `open_with_tokenizer`. Demonstrates the raw seam without dragging a heavy crate
  into the default build (the base CI job builds all examples with no features). The
  real backends are the *bundled* tiktoken/HF impls, so the example's only job is the
  trait shape.
- README: update the roadmap (check off "real tokenizer"); document both features, the
  trait, and the "match your generation LLM" guidance; show a short tiktoken and HF
  snippet.
- CI (`.github/workflows/ci.yml`): extend the existing feature-lint job to clippy
  `--features tiktoken`, `--features hf-tokenizer`, and `--features hf-hub` (mirroring
  the `fastembed` job). The dependency-free example is covered by the existing base
  `cargo build --examples`.

## Staff-review guardrails (fold into the plan)

1. **Spike the `tokenizers` minimal feature set FIRST.** Before building the impl,
   confirm `hf-tokenizer` compiles + encodes with `default-features = false` and a
   minimal feature set, and `cargo tree` the resulting dep weight. Cheap to learn
   early; shapes whether the "stays light" claim holds.
2. **Document the `add_special_tokens = false` assumption** in the `Budget::Tokens` /
   tokenizer docs: counts memory *content only*; callers using chat formats must
   leave headroom for per-message framing tokens.
3. **Anchor token-count tests** on short, stable strings (e.g. o200k `"hello world"
   == 2`) and the CJK-`>`-heuristic property — not large exact paragraph counts.
4. The HF encode-error → heuristic fallback is a belt-and-suspenders path (encode
   effectively never fails on valid `&str`); comment it as such so mixed counting
   isn't mistaken for intended behavior.

## Testing

- **Unit (`src/tokenizer.rs`):** `HeuristicTokenizer` matches the old `approx_tokens`
  numbers (reuse the existing cases).
- **Feature-gated unit — tiktoken (`tests/tiktoken.rs`, `#![cfg(feature = "tiktoken")]`):**
  `TiktokenTokenizer::new()` counts a known English string to tiktoken's exact token
  count; CJK counts strictly higher than `chars/4` for the same text (the core bug this
  fixes); `with_model("gpt-4o")` succeeds and an unknown model returns `Error::Tokenizer`.
  Offline (vocab embedded) — **no `#[ignore]`**.
- **Feature-gated unit — HF (`tests/hf_tokenizer.rs`, `#![cfg(feature = "hf-tokenizer")]`):**
  load a small committed `tokenizer.json` test fixture via `from_bytes`
  (`include_bytes!`), assert a known string's exact count and that a bad blob returns
  `Error::Tokenizer`. Fixture is tiny and local, so this is offline — **no `#[ignore]`**
  and no download. (Builder picks a small permissively-licensed `tokenizer.json`.)
- **Feature-gated online — Hub (`#![cfg(feature = "hf-hub")]`):** one `#[ignore]`d test
  that `from_pretrained(...)` fetches and counts (network on first run), mirroring the
  fastembed online test. Run manually with `--features hf-hub -- --ignored`.
- **Integration (`tests/store.rs`):** `open_with_tokenizer` + `Budget::Tokens` packs by
  the injected tokenizer's counts; `open` still uses the heuristic. A simple stub
  `Tokenizer` (length-based) keeps this deterministic and feature-independent.

## Left to the builder

- Exact `tiktoken-rs` and `tokenizers` versions (latest 0.x at build time) and the
  precise constructor/helper names each exposes.
- The minimal `tokenizers` feature set that supports load-from-json + encode without
  trainer/`onig` weight.
- Whether `TiktokenTokenizer::with_encoding(name)` is also public or model-name is
  enough (bias: add it — free once the model-name path exists, and it's how a caller
  selects `cl100k_base` for a `text-embedding-3` match).
- The small `tokenizer.json` fixture for the HF test (any small permissive one).
