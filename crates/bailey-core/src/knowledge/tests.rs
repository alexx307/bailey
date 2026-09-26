use std::fs;

use super::{Library, LibraryLock, hashing};
use crate::web::Article;

#[test]
fn revisions_and_duplicate_text_are_preserved_or_deduplicated() {
    let dir = tempfile::tempdir().unwrap();
    let mut library = Library::open(dir.path(), 1_000_000).unwrap();
    let original = fixture(1, 10);
    assert!(library.insert(&original).unwrap());
    assert!(!library.insert(&original).unwrap());
    let mut same_text = fixture(2, 10);
    same_text.text = original.text.clone();
    assert!(!library.insert(&same_text).unwrap());
    assert!(library.insert(&fixture(1, 20)).unwrap());
    assert_eq!(
        Library::open(dir.path(), 1_000_000)
            .unwrap()
            .list()
            .unwrap()
            .len(),
        2
    );
    let found = library.search("article", 10).unwrap();
    assert_eq!(found.len(), 1);
    assert_eq!(found[0].revision_id, 20);
}

#[test]
fn budget_and_single_writer_lock_are_enforced() {
    let dir = tempfile::tempdir().unwrap();
    let mut library = Library::open(dir.path(), 10).unwrap();
    assert!(library.insert(&fixture(1, 1)).is_err());
    let first = LibraryLock::acquire(dir.path()).unwrap();
    assert!(LibraryLock::acquire(dir.path()).is_err());
    drop(first);
    assert!(LibraryLock::acquire(dir.path()).is_ok());
}

#[test]
fn export_is_disjoint_and_a_page_keeps_its_partition_across_revisions() {
    let dir = tempfile::tempdir().unwrap();
    let mut library = Library::open(&dir.path().join("library"), 1_000_000).unwrap();
    for id in 1..=100 {
        assert_eq!(
            hashing::partition(&fixture(id, 1)),
            hashing::partition(&fixture(id, 2))
        );
        library.insert(&fixture(id, 1)).unwrap();
    }
    library.insert(&fixture(1, 2)).unwrap();
    let out = dir.path().join("dataset");
    let summary = library.export_dataset(&out, 32).unwrap();
    assert_eq!(summary.articles, 100);
    let sets: Vec<String> = ["train", "validation", "test"]
        .iter()
        .map(|name| fs::read_to_string(out.join(format!("{name}.txt"))).unwrap())
        .collect();
    for id in 1..=100 {
        let marker = format!("# Article {id}\n");
        assert_eq!(sets.iter().filter(|text| text.contains(&marker)).count(), 1);
    }
    assert!(!sets.iter().any(|text| text.contains("page=1 revision=1.")));
    assert!(library.export_dataset(&out, 32).is_err());
}

#[test]
fn not_enough_partitions_never_moves_reserved_pages_into_training() {
    let dir = tempfile::tempdir().unwrap();
    let mut library = Library::open(&dir.path().join("library"), 1_000_000).unwrap();
    library.insert(&fixture(1, 1)).unwrap();
    let out = dir.path().join("dataset");
    assert!(library.export_dataset(&out, 32).is_err());
    assert!(!out.exists());
}

#[test]
fn tampered_article_is_rejected_on_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let article = fixture(1, 1);
    let mut library = Library::open(dir.path(), 1_000_000).unwrap();
    library.insert(&article).unwrap();
    let folder = dir
        .path()
        .join("articles")
        .join(hashing::identity(&article));
    let path = fs::read_dir(folder)
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    let mut value: serde_json::Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    value["article"]["text"] = serde_json::json!("modified");
    fs::write(path, serde_json::to_vec(&value).unwrap()).unwrap();
    assert!(Library::open(dir.path(), 1_000_000).is_err());
}

#[test]
fn publication_is_complete_and_cannot_replace_an_existing_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("immutable.json");
    super::storage::write_new(&path, b"original").unwrap();
    assert!(super::storage::write_new(&path, b"replacement").is_err());
    assert_eq!(fs::read(&path).unwrap(), b"original");
    assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 1);
}

fn fixture(id: u64, revision_id: u64) -> Article {
    Article {
        id,
        revision_id,
        title: format!("Article {id}"),
        url: format!("https://fr.wikipedia.org/?curid={id}"),
        language: "fr".into(),
        text: format!("Un document assez long sur les nombres, page={id} revision={revision_id}."),
        license: "CC BY-SA 4.0".into(),
        fetched_at_unix: 100,
    }
}
