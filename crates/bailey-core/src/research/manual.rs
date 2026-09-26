use anyhow::{Result, ensure};
use std::{path::Path, thread, time::Duration};

use crate::{
    knowledge::{Library, LibraryLock},
    web::{Article, WikiClient},
};

pub(super) const DEFAULT_LIBRARY_BYTES: u64 = 64 * 1024 * 1024;

pub fn search_and_store(
    topic: &str,
    language: &str,
    library: &Path,
    limit: usize,
) -> Result<Vec<Article>> {
    ensure!(
        (1..=50).contains(&limit),
        "Recherche manuelle : 1 a 50 pages"
    );
    let reader = WikiClient::new(language, 20, 2 * 1024 * 1024)?;
    let ids = reader.search(topic, limit, 0)?;
    let mut found = Vec::new();
    for id in ids {
        thread::sleep(Duration::from_secs(1));
        let article = reader.article(id)?;
        {
            let _lock = LibraryLock::acquire(library)?;
            Library::open(library, DEFAULT_LIBRARY_BYTES)?.insert(&article)?;
        }
        found.push(article);
    }
    Ok(found)
}

pub fn lookup(query: &str, library: &Path, limit: usize) -> Result<Vec<Article>> {
    let _lock = LibraryLock::acquire(library)?;
    Library::open(library, u64::MAX)?.search(query, limit)
}
