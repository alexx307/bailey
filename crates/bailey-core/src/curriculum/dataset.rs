use std::collections::{BTreeMap, HashSet};
use std::fs;
use std::path::Path;

use anyhow::{Context, Result, ensure};
use serde_json::json;
use sha2::{Digest, Sha256};

use super::{conversation, grammar, language, record::Unit, world};

pub(super) fn examples() -> Vec<Unit> {
    let mut units = Vec::new();
    language::append(&mut units);
    grammar::append(&mut units);
    world::append(&mut units);
    conversation::append(&mut units);
    units
}

pub(super) fn split(id: &str) -> usize {
    match Sha256::digest(id.as_bytes())[0] % 10 {
        0 => 2,
        1 => 1,
        _ => 0,
    }
}

pub fn prepare(out: &Path) -> Result<()> {
    ensure!(!out.exists(), "Le dossier {} existe déjà", out.display());
    let mut partitions: [Vec<Unit>; 3] = Default::default();
    let mut ids = HashSet::new();
    let mut texts = HashSet::new();
    for unit in examples() {
        ensure!(
            ids.insert(unit.id.clone()),
            "Identifiant répété : {}",
            unit.id
        );
        ensure!(
            texts.insert(unit.text.clone()),
            "Leçon répétée : {}",
            unit.id
        );
        partitions[split(&unit.id)].push(unit);
    }
    ensure!(
        partitions.iter().all(|part| !part.is_empty()),
        "Partition vide"
    );
    if let Some(parent) = out.parent().filter(|parent| !parent.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    fs::create_dir(out).context("Création d'un nouveau corpus de fondation")?;
    let mut summaries = Vec::new();
    for (name, units) in ["train", "validation", "test"].into_iter().zip(partitions) {
        let mut text = String::new();
        let mut topics = BTreeMap::new();
        let mut unit_ids = Vec::new();
        for unit in &units {
            text.push_str(&unit.text);
            *topics.entry(unit.topic).or_insert(0_usize) += 1;
            unit_ids.push(&unit.id);
        }
        fs::write(out.join(format!("{name}.txt")), text.as_bytes())?;
        summaries.push(json!({
            "name": name, "units": units.len(), "bytes": text.len(),
            "sha256": format!("{:x}", Sha256::digest(text.as_bytes())),
            "topics": topics, "unit_ids": unit_ids,
        }));
        println!(
            "Fondation française {name} : {} unités, {} octets",
            units.len(),
            text.len()
        );
    }
    let manifest = json!({
        "format_version": 1,
        "stage": "seed",
        "purpose": "pipeline_smoke_test",
        "not_general_language_pretraining": true,
        "language": "fr",
        "origin": "Original French prose and short dialogues authored by the coding assistant; no pretrained weights are transferred",
        "unit_count": ids.len(),
        "format": "Prose followed by <|end|>, or <|user|>{prompt}<|assistant|>{answer}<|end|>; UTF-8",
        "partition_rule": "SHA256(unit_id)[0] modulo 10: 0=test, 1=validation, other=train; assigned before tokenizer training",
        "partitions": summaries,
        "limitations": [
            "Tiny demonstration corpus: not sufficient to pretrain a 100-million-parameter language model",
            "Validation and test reserve complete authored units, but topics, grammar rules and some facts overlap",
            "Only train.txt may train the tokenizer and model; test.txt must not select checkpoints",
            "Training on statements about uncertainty does not establish reliable self-assessment",
            "Lower prediction loss does not demonstrate fluent French or general reasoning",
            "These examples need further independent editorial review before use in a substantial training corpus"
        ],
    });
    fs::write(
        out.join("manifest.json"),
        serde_json::to_vec_pretty(&manifest)?,
    )?;
    Ok(())
}
