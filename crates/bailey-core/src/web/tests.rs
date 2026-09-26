use serde_json::json;

use super::{body::read_bounded, client::endpoint, parsing};

#[test]
fn language_cannot_change_the_fixed_https_host() {
    for language in [
        "FR",
        "",
        "f",
        "../fr",
        "fr:80",
        "fr.wikipedia.org",
        "fr@localhost",
        "-fr",
        "fr-",
    ] {
        assert!(endpoint(language).is_err(), "{language} should be rejected");
    }
    for language in ["fr", "en", "simple", "zh-min-nan"] {
        let url = endpoint(language).unwrap();
        assert_eq!(url.scheme(), "https");
        assert_eq!(
            url.host_str(),
            Some(format!("{language}.wikipedia.org").as_str())
        );
        assert_eq!(url.path(), "/w/api.php");
    }
}

#[test]
fn reader_detects_excess_without_a_content_length_header() {
    assert_eq!(read_bounded("12345".as_bytes(), 5).unwrap(), b"12345");
    assert!(read_bounded("123456".as_bytes(), 5).is_err());
    assert!(read_bounded("a".as_bytes(), usize::MAX).is_err());
}

#[test]
fn search_distinguishes_no_matches_from_invalid_response() {
    assert!(
        parsing::search(&json!({"query": {"search": []}}))
            .unwrap()
            .is_empty()
    );
    let valid = json!({"query": {"search": [{"ns": 0, "pageid": 42}]}});
    assert_eq!(parsing::search(&valid).unwrap(), vec![42]);
    assert!(parsing::search(&json!({"query": {}})).is_err());
    assert!(parsing::search(&json!({"query": {"search": [{"ns": 1, "pageid": 42}]}})).is_err());
}

#[test]
fn api_errors_are_never_treated_as_corpus_data() {
    let error = json!({"error": {"code": "maxlag", "info": "Waiting for replication"}});
    assert!(
        parsing::search(&error)
            .unwrap_err()
            .to_string()
            .contains("maxlag")
    );
    assert!(parsing::article(&error, 42, "fr").is_err());
}

#[test]
fn article_retains_source_revision_language_and_license() {
    let article = parsing::article(&article_fixture(), 42, "fr").unwrap();
    assert_eq!(article.id, 42);
    assert_eq!(article.revision_id, 123);
    assert_eq!(article.url, "https://fr.wikipedia.org/?curid=42");
    assert_eq!(article.title, "Mathématiques");
    assert_eq!(article.language, "fr");
    assert!(article.license.contains("CC BY-SA 4.0"));
    assert!(article.fetched_at_unix > 0);
}

#[test]
fn absent_articles_extracts_and_revisions_are_rejected() {
    assert!(parsing::article(&article_fixture(), 999, "fr").is_err());
    let mut value = article_fixture();
    value["query"]["pages"][0]["extract"] = json!("  ");
    assert!(parsing::article(&value, 42, "fr").is_err());
    let mut value = article_fixture();
    value["query"]["pages"][0]["revisions"] = json!([]);
    assert!(parsing::article(&value, 42, "fr").is_err());
    let mut value = article_fixture();
    value["query"]["pages"][0]["missing"] = json!(true);
    assert!(parsing::article(&value, 42, "fr").is_err());
}

fn article_fixture() -> serde_json::Value {
    json!({"query": {"pages": [{
        "pageid": 42, "ns": 0, "title": "Mathématiques",
        "extract": "Un texte public avec sa provenance.",
        "revisions": [{"revid": 123}]
    }]}})
}
