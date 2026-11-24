#![cfg(feature = "hf-tokenizer")]
use memnon::{Error, HfTokenizer, Tokenizer};

const FIXTURE: &[u8] = include_bytes!("fixtures/tokenizer.json");

#[test]
fn from_bytes_loads_and_counts() {
    let t = HfTokenizer::from_bytes(FIXTURE).unwrap();
    assert!(t.count_tokens("hello world") >= 2); // fixture-independent lower bound
}
#[test]
fn cjk_counts_more_than_heuristic() {
    let t = HfTokenizer::from_bytes(FIXTURE).unwrap();
    // Repeated to give the real-tokenizer count a comfortable margin over the
    // `chars/4` heuristic, so the assertion stays robust to fixture changes.
    let text = "日本語のテキストです".repeat(5);
    assert!(t.count_tokens(&text) > text.chars().count().div_ceil(4));
}
#[test]
fn malformed_bytes_error() {
    assert!(matches!(
        HfTokenizer::from_bytes(b"not json"),
        Err(Error::Tokenizer(_))
    ));
}

#[test]
#[cfg(feature = "hf-hub")]
#[ignore = "downloads a tokenizer.json from the HF Hub; run manually"]
fn from_pretrained_fetches_and_counts() {
    let t = HfTokenizer::from_pretrained("bert-base-uncased").unwrap();
    assert!(t.count_tokens("hello world") >= 2);
}
