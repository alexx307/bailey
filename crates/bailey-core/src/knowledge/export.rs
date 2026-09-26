use super::{Library, hashing, retrieval, storage};
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{fs, path::Path};

const PARTITIONS: [&str; 3] = ["train", "validation", "test"];

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct DatasetSummary {
    pub articles: usize,
    pub train_bytes: usize,
    pub validation_bytes: usize,
    pub test_bytes: usize,
}

pub fn write(library: &Library, out: &Path, context: usize) -> Result<DatasetSummary> {
    ensure!(context > 0, "Le contexte doit etre positif");
    ensure!(
        !out.exists(),
        "L'export existe deja : {} ; choisissez un nouveau dossier",
        out.display()
    );
    let mut corpus: [String; 3] = Default::default();
    let mut counts = [0_usize; 3];
    let mut sources = Vec::new();
    for article in retrieval::latest(library) {
        let split = hashing::partition(article);
        corpus[split].push_str(&format!("# {}\n\n{}\n\n", article.title, article.text));
        counts[split] += 1;
        sources.push(json!({
            "id": article.id,
            "revision_id": article.revision_id,
            "title": article.title,
            "url": article.url,
            "language": article.language,
            "license": article.license,
            "fetched_at_unix": article.fetched_at_unix,
            "identity_sha256": hashing::identity(article),
            "text_sha256": hashing::sha256(&article.text),
            "partition": PARTITIONS[split],
        }));
    }
    for (index, name) in PARTITIONS.iter().enumerate() {
        ensure!(
            counts[index] > 0 && corpus[index].chars().count() > context,
            "Partition {name} insuffisante ({} pages, {} caracteres, contexte {context}). \
             Collectez davantage de pages independantes avant d'exporter ; \
             les pages reservees ne sont jamais deplacees vers l'entrainement.",
            counts[index],
            corpus[index].chars().count()
        );
    }
    let summary = DatasetSummary {
        articles: sources.len(),
        train_bytes: corpus[0].len(),
        validation_bytes: corpus[1].len(),
        test_bytes: corpus[2].len(),
    };
    let manifest = manifest(&summary, counts, sources, context);
    if let Some(parent) = out.parent().filter(|parent| !parent.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    // create_dir fails if another process has created this destination meanwhile.
    fs::create_dir(out)?;
    for (name, text) in PARTITIONS.iter().zip(corpus.iter()) {
        storage::write_new(&out.join(format!("{name}.txt")), text.as_bytes())?;
    }
    storage::write_new(
        &out.join("manifest.json"),
        &serde_json::to_vec_pretty(&manifest)?,
    )?;
    Ok(summary)
}

fn manifest(
    summary: &DatasetSummary,
    counts: [usize; 3],
    sources: Vec<Value>,
    context: usize,
) -> Value {
    json!({
        "schema_version": 1,
        "origin": "Sourced article text stored as untrusted data; no downloaded model weights",
        "summary": summary,
        "minimum_context_characters": context,
        "group_key": "language:page_id",
        "partition_rule": "First 8 SHA256(language:page_id) bytes as unsigned big-endian integer, modulo 10: 0 test, 1 validation, all others train",
        "revision_policy": "Newest stored revision per page; all revisions of a page have the same partition",
        "format": "# title\\n\\narticle text\\n\\n",
        "counts": {"train": counts[0], "validation": counts[1], "test": counts[2]},
        "sources": sources,
        "limitations": [
            "Exact full-text duplicates are excluded globally; near duplicates, translations and overlapping quotations are not detected",
            "Train/validation/test assignment happens at page level before byte windows exist",
            "Validation can select versions; test is excluded from training and version selection",
            "Reading and exporting do not update model weights",
            "Source text is untrusted data and is never executed or treated as agent instructions",
            "A lower loss on encyclopedia text does not demonstrate programming competence",
            "Attribution and source license must accompany redistributed text; consult each source URL"
        ]
    })
}
