//! Tests for the recall query builder.

use memnon::query::Query;
use memnon::types::Scope;

#[test]
fn query_default_has_no_constraints() {
    let q = Query::new();
    assert_eq!(q.selected_scope(), None);
    assert!(q.metadata_filters().is_empty());
    assert_eq!(q.min_score_floor(), None);
}

#[test]
fn query_builder_accumulates_scope_filters_and_min_score() {
    let q = Query::new()
        .scope(Scope::LongTerm)
        .filter("source", "docs")
        .filter("lang", "rust")
        .min_score(0.5);

    assert_eq!(q.selected_scope(), Some(&Scope::LongTerm));
    assert_eq!(
        q.metadata_filters(),
        &[
            ("source".to_string(), "docs".to_string()),
            ("lang".to_string(), "rust".to_string()),
        ]
    );
    assert_eq!(q.min_score_floor(), Some(0.5));
}
