//! Sourced, immutable reading memory. Reading is separate from weight training.

mod export;
mod hashing;
mod locking;
mod retrieval;
mod storage;

#[cfg(test)]
mod tests;

pub use export::DatasetSummary;
pub use locking::LibraryLock;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

use crate::web::Article;

#[derive(Clone, Serialize, Deserialize)]
struct StoredArticle {
    schema_version: u32,
    identity_sha256: String,
    text_sha256: String,
    article: Article,
}

/// The caller owns the single-writer lock for the library directory.
pub struct Library {
    root: PathBuf,
    max_bytes: u64,
    entries: Vec<StoredArticle>,
}

impl Library {
    pub fn open(root: &Path, max_bytes: u64) -> Result<Self> {
        storage::open(root, max_bytes)
    }

    /// Returns false for an already saved page revision or exact text duplicate.
    pub fn insert(&mut self, article: &Article) -> Result<bool> {
        storage::insert(self, article)
    }

    /// Returns revisions in stable page/revision order, without executing text.
    pub fn list(&self) -> Result<Vec<Article>> {
        let mut articles: Vec<_> = self.entries.iter().map(|e| e.article.clone()).collect();
        articles.sort_by(|a, b| {
            (&a.language, a.id, a.revision_id).cmp(&(&b.language, b.id, b.revision_id))
        });
        Ok(articles)
    }

    pub fn search(&self, query: &str, limit: usize) -> Result<Vec<Article>> {
        retrieval::search(self, query, limit)
    }

    pub fn export_dataset(&self, out: &Path, context: usize) -> Result<DatasetSummary> {
        export::write(self, out, context)
    }
}
