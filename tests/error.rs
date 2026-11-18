//! Tests for the crate error type.

use memnon::error::Error;

#[test]
fn error_display_is_human_readable() {
    let s = Error::EmbedderMismatch {
        expected: "bge".into(),
        found: "ada".into(),
    }
    .to_string();
    assert!(s.contains("bge"), "got: {s}");
    assert!(s.contains("ada"), "got: {s}");

    let s = Error::DimensionMismatch {
        expected: 16,
        found: 32,
    }
    .to_string();
    assert!(s.contains("16") && s.contains("32"), "got: {s}");
}

#[test]
fn error_implements_std_error_trait() {
    // Compiles only if `Error: std::error::Error`.
    fn takes_std_error<E: std::error::Error>(_: E) {}
    takes_std_error(Error::Io("disk full".into()));
}
