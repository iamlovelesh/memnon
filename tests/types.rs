//! Tests for memnon's core data types.

use memnon::types::Entry;

#[test]
fn entry_from_str_has_no_metadata() {
    let e = Entry::from("hello");
    assert_eq!(e.content(), "hello");
    assert!(e.metadata().is_empty());
}

#[test]
fn entry_meta_is_chainable_and_stored() {
    let e = Entry::new("hello")
        .meta("source", "docs")
        .meta("lang", "rust");
    assert_eq!(e.content(), "hello");
    assert_eq!(e.metadata().get("source"), Some(&"docs".to_string()));
    assert_eq!(e.metadata().get("lang"), Some(&"rust".to_string()));
}
