#![cfg(feature = "tiktoken")]
use memnon::{Error, TiktokenTokenizer, Tokenizer};

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
    assert!(matches!(
        TiktokenTokenizer::with_model("not-a-real-model"),
        Err(Error::Tokenizer(_))
    ));
}
