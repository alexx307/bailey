use crate::web::Article;
use sha2::{Digest, Sha256};

pub fn sha256(text: &str) -> String {
    format!("{:x}", Sha256::digest(text.as_bytes()))
}

pub fn identity(article: &Article) -> String {
    sha256(&format!("{}:{}", article.language, article.id))
}

/// Page assignment is independent of text, revision, collection order and time.
pub fn partition(article: &Article) -> usize {
    let digest = Sha256::digest(format!("{}:{}", article.language, article.id).as_bytes());
    let mut prefix = [0_u8; 8];
    prefix.copy_from_slice(&digest[..8]);
    match u64::from_be_bytes(prefix) % 10 {
        0 => 2,
        1 => 1,
        _ => 0,
    }
}
