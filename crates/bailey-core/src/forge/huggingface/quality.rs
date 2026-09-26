use super::client::Document;
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};

pub fn hash(text: &str) -> String {
    format!("{:x}", Sha256::digest(text.as_bytes()))
}
pub fn partition(host: &str) -> usize {
    let bytes = Sha256::digest(host.as_bytes());
    match u64::from_be_bytes(bytes[..8].try_into().unwrap()) % 10 {
        0 => 2,
        1 => 1,
        _ => 0,
    }
}
pub fn reason(doc: &Document) -> Option<&'static str> {
    if doc.language != "fra"
        || doc.language_script != "Latn"
        || !doc.language_score.is_finite()
        || doc.language_score < 0.8
    {
        return Some("language");
    }
    let chars = doc.text.chars().count();
    if !(200..=200_000).contains(&chars) {
        return Some("length");
    }
    if doc
        .text
        .chars()
        .any(|c| c == '\u{fffd}' || (c.is_control() && !c.is_whitespace()))
    {
        return Some("corrupted_characters");
    }
    if doc.text.chars().filter(|c| c.is_alphabetic()).count() * 10 < chars * 4 {
        return Some("low_text_ratio");
    }
    if [
        "<script",
        "<html",
        crate::tokenization::USER,
        crate::tokenization::ASSISTANT,
        crate::tokenization::EOS,
    ]
    .iter()
    .any(|s| doc.text.contains(s))
    {
        return Some("markup_or_reserved_tokens");
    }
    let lines: Vec<_> = doc
        .text
        .lines()
        .map(str::trim)
        .filter(|l| l.chars().count() >= 40)
        .collect();
    if lines.len() >= 5 && lines.iter().collect::<HashSet<_>>().len() * 10 < lines.len() * 6 {
        return Some("repeated_lines");
    }
    None
}

#[derive(Default)]
pub struct Dedup {
    urls: HashSet<String>,
    texts: HashSet<String>,
    fragments: HashMap<String, usize>,
    sketches: Vec<u64>,
}
impl Dedup {
    pub fn accept(&mut self, text: &str, url: &str, split: usize) -> Result<(), &'static str> {
        let digest = hash(text);
        if self.urls.contains(url) || self.texts.contains(&digest) {
            return Err("exact_duplicate");
        }
        let sketch = simhash(text);
        if self
            .sketches
            .iter()
            .any(|old| (old ^ sketch).count_ones() <= 3)
        {
            return Err("near_duplicate_simhash");
        }
        let pieces: Vec<_> = text
            .lines()
            .chain(text.split("\n\n"))
            .map(str::trim)
            .filter(|s| s.chars().count() >= 64 && s.split_whitespace().count() >= 8)
            .map(hash)
            .collect();
        if pieces
            .iter()
            .any(|h| self.fragments.get(h).is_some_and(|old| *old != split))
        {
            return Err("cross_partition_fragment");
        }
        self.urls.insert(url.into());
        self.texts.insert(digest);
        self.sketches.push(sketch);
        for piece in pieces {
            self.fragments.insert(piece, split);
        }
        Ok(())
    }
}
fn simhash(text: &str) -> u64 {
    let mut weights = [0i64; 64];
    let words: Vec<String> = text.split_whitespace().map(|s| s.to_lowercase()).collect();
    for triple in words.windows(3) {
        let digest = Sha256::digest(triple.join(" ").as_bytes());
        let value = u64::from_le_bytes(digest[..8].try_into().unwrap());
        for (bit, w) in weights.iter_mut().enumerate() {
            *w += if value & (1 << bit) != 0 { 1 } else { -1 };
        }
    }
    weights
        .iter()
        .enumerate()
        .fold(0, |value, (i, w)| value | if *w > 0 { 1 << i } else { 0 })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn domain_split_is_stable_and_duplicate_paragraphs_cannot_leak() {
        assert_eq!(partition("example.org"), partition("example.org"));
        let common = "Ce passage contient suffisamment de mots et de caracteres pour constituer un paragraphe documentaire partage.";
        let mut d = Dedup::default();
        assert!(d.accept(common, "https://a.fr", 0).is_ok());
        assert!(
            d.accept(&format!("Un autre titre\n{common}"), "https://b.fr", 1)
                .is_err()
        );
        assert!(d.accept(common, "https://c.fr", 2).is_err());
    }
}
