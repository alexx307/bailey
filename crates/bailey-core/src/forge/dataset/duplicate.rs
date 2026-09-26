use std::collections::HashMap;

use anyhow::{Result, ensure};

use super::io::hash;

pub(super) const POLICY: &str = "Reject identical complete files across splits; also reject identical trimmed lines, blank-line blocks and <|end|>-delimited records with at least 64 Unicode characters and 8 whitespace-separated words. CRLF is normalized for fragment comparison.";

#[derive(Default)]
pub(super) struct Detector {
    fragments: HashMap<String, String>,
    files: HashMap<String, String>,
}

impl Detector {
    pub(super) fn check(&mut self, partition: &str, text: &str) -> Result<()> {
        let normalized = text.replace("\r\n", "\n");
        Self::insert(&mut self.files, hash(normalized.trim().as_bytes()), partition)?;
        for fragment in normalized.lines()
            .chain(normalized.split("\n\n"))
            .chain(normalized.split(crate::tokenization::EOS))
        {
            let fragment = fragment.trim();
            if fragment.chars().count() >= 64 && fragment.split_whitespace().count() >= 8 {
                Self::insert(&mut self.fragments, hash(fragment.as_bytes()), partition)?;
            }
        }
        Ok(())
    }

    fn insert(seen: &mut HashMap<String, String>, hash: String, partition: &str) -> Result<()> {
        if let Some(previous) = seen.get(&hash) {
            ensure!(previous == partition, "Doublon exact entre {previous} et {partition} ; corriger les partitions avant la tokenisation");
        } else {
            seen.insert(hash, partition.to_owned());
        }
        Ok(())
    }
}
