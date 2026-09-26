//! Decontamination exacte : priorite test, puis validation, puis entrainement.
use anyhow::{Result, ensure};
use sha2::{Digest, Sha256};
use std::{collections::HashSet, fs, path::Path};

fn significant(text: &str) -> bool {
    text.chars().count() >= 64 && text.split_whitespace().count() >= 8
}
fn hash(text: &str) -> String {
    format!("{:x}", Sha256::digest(text.trim().as_bytes()))
}
fn remember(text: &str, reserved: &mut HashSet<String>) {
    for part in text.lines().chain(text.split("\n\n")) {
        if significant(part.trim()) {
            reserved.insert(hash(part));
        }
    }
}
fn clean(text: &str, reserved: &HashSet<String>) -> (String, Vec<String>) {
    let mut kept = Vec::new();
    let mut removed = Vec::new();
    for block in text.split("\n\n") {
        if significant(block.trim()) && reserved.contains(&hash(block)) {
            removed.push(hash(block));
            continue;
        }
        let mut lines = Vec::new();
        for line in block.lines() {
            if significant(line.trim()) && reserved.contains(&hash(line)) {
                removed.push(hash(line));
            } else {
                lines.push(line);
            }
        }
        let block = lines.join("\n");
        if !block.trim().is_empty() {
            kept.push(block);
        }
    }
    (kept.join("\n\n"), removed)
}
pub fn curate(data: &Path, out: &Path) -> Result<()> {
    ensure!(!out.exists(), "Dossier de curation deja existant");
    let source: serde_json::Value = serde_json::from_slice(&fs::read(data.join("manifest.json"))?)?;
    let mut reserved = HashSet::new();
    let mut results = Vec::new();
    let mut audit = Vec::new();
    for name in ["test", "validation", "train"] {
        let file = data.join(format!("{name}.txt"));
        ensure!(
            fs::metadata(&file)?.len() <= 512 * 1024 * 1024,
            "Curation pilote : limite 512 Mio par partition"
        );
        let raw = fs::read_to_string(&file)?;
        let normalized = raw.replace("\r\n", "\n");
        let (text, removed) = if name == "test" {
            (raw.clone(), Vec::new())
        } else {
            clean(&normalized, &reserved)
        };
        ensure!(
            !text.trim().is_empty(),
            "Partition {name} vide apres decontamination ; collecter d'autres sources"
        );
        remember(&text.replace("\r\n", "\n"), &mut reserved);
        audit.push(serde_json::json!({"partition":name,"input_sha256":format!("{:x}",Sha256::digest(raw.as_bytes())),"output_sha256":format!("{:x}",Sha256::digest(text.as_bytes())),"input_bytes":raw.len(),"output_bytes":text.len(),"removed_fragments":removed}));
        println!(
            "{name} : {} -> {} octets, {} fragments retires",
            raw.len(),
            text.len(),
            audit.last().unwrap()["removed_fragments"]
                .as_array()
                .unwrap()
                .len()
        );
        results.push((name, text));
    }
    if let Some(parent) = out.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    fs::create_dir(out)?;
    for (name, text) in results {
        fs::write(out.join(format!("{name}.txt")), text)?;
    }
    fs::write(
        out.join("manifest.json"),
        serde_json::to_vec_pretty(
            &serde_json::json!({"source_manifest":source,"transform":"Remove exact lines/paragraphs >=64 chars and >=8 words shared with a higher-priority partition; test unchanged; validation priority over train","audit":audit,"limitations":"No near-duplicate or factual verification; metadata describes original articles plus these transformations"}),
        )?,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn removes_reserved_content_but_keeps_unrelated_training_text() {
        let common = "Un paragraphe documentaire assez long et compose de plusieurs mots pour etre reserve a la validation.";
        let mut seen = HashSet::new();
        remember(common, &mut seen);
        let (text, removed) = clean(
            &format!("Texte original.\n\n{common}\n\nFin originale."),
            &seen,
        );
        assert_eq!(text, "Texte original.\n\nFin originale.");
        assert_eq!(removed.len(), 1);
    }
}
