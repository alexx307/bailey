//! Melange borne de sources train, sans repetition pour remplir un quota.
mod config;
mod selection;
#[cfg(test)]
mod tests;

use anyhow::{Context, Result, ensure};
use config::{Config, DOMAINS};
use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{collections::HashSet, fs, path::Path};

pub(super) fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

pub fn prepare(config_path: &Path, out: &Path) -> Result<()> {
    ensure!(!out.exists(), "Dossier de melange deja utilise");
    let bytes = fs::read(config_path)?;
    let config: Config = serde_json::from_slice(&bytes)?;
    config.validate()?;
    let base = config_path.parent().unwrap_or(Path::new("."));
    let mut texts = Vec::new();
    let mut sources = Vec::new();
    let mut fingerprints = HashSet::new();
    let mut input_bytes = 0usize;
    for source in &config.sources {
        let file = base.join(&source.file).canonicalize()?;
        let size: usize = fs::metadata(&file)?.len().try_into()?;
        input_bytes = input_bytes
            .checked_add(size)
            .context("Taille hors limites")?;
        ensure!(
            input_bytes <= 512 * 1024 * 1024,
            "Pilote : maximum 512 Mio de sources en RAM"
        );
        let text = fs::read_to_string(&file)?;
        let digest = hash(text.as_bytes());
        ensure!(
            digest == source.sha256.to_ascii_lowercase(),
            "Source modifiee : {}",
            source.id
        );
        ensure!(!text.trim().is_empty(), "Source vide : {}", source.id);
        ensure!(
            fingerprints.insert(digest.clone()),
            "Source exacte repetee : {}",
            source.id
        );
        let provenance_path = base.join(&source.provenance).canonicalize()?;
        ensure!(
            fs::metadata(&provenance_path)?.len() <= 16 * 1024 * 1024,
            "Provenance trop volumineuse"
        );
        let provenance = fs::read(&provenance_path)?;
        let value: serde_json::Value = serde_json::from_slice(&provenance)?;
        ensure!(
            value["complete"].as_bool() != Some(false),
            "Source d'import incomplete : {}",
            source.id
        );
        if source.file.file_name().and_then(|s| s.to_str()) == Some("train.txt")
            && let Some(expected) = value["train_sha256"].as_str()
        {
            ensure!(
                expected == digest,
                "Provenance train perimee : {}",
                source.id
            );
        }
        sources.push(
            json!({"specification":source,"resolved_file":file,"bytes":text.len(),
            "sha256":digest,"provenance_sha256":hash(&provenance),"provenance":value}),
        );
        texts.push(text);
    }
    let mut pools = Vec::new();
    let mut available = Vec::new();
    for domain in DOMAINS {
        let mut pool = Vec::new();
        for (index, source) in config.sources.iter().enumerate() {
            if source.domain == domain {
                pool.extend(selection::fragments(index, &texts[index]));
            }
        }
        let count: usize = pool.iter().map(|f| f.bytes).sum();
        ensure!(count > 0, "Domaine {domain:?} absent du melange");
        available.push(count);
        pools.push(pool);
    }
    // Une fin de fichier par source, blocs pleins >=4093 octets et trois
    // fragments tronques de quota : reserver aussi les separateurs dans le budget.
    let separator_reserve = config.sources.len() + config.max_text_bytes.div_ceil(4093) + 3;
    let source_budget = config
        .max_text_bytes
        .checked_sub(separator_reserve)
        .context("Budget trop petit pour les sources et separateurs")?;
    // La categorie la moins fournie limite le melange ; aucune sur-repetition.
    let effective_budget = available
        .iter()
        .zip(config.percentages)
        .map(|(bytes, percentage)| bytes * 100 / percentage)
        .fold(source_budget, usize::min);
    let mut selected = Vec::new();
    let mut domains = Vec::new();
    for (index, pool) in pools.into_iter().enumerate() {
        let target = effective_budget * config.percentages[index] / 100;
        let chosen =
            selection::select(pool, &texts, target, config.seed.wrapping_add(index as u64));
        let retained: usize = chosen.iter().map(|f| f.bytes).sum();
        ensure!(
            retained > 0,
            "Budget trop petit pour conserver chaque domaine"
        );
        domains.push(json!({"domain":DOMAINS[index],"available_bytes":available[index],
            "target_bytes":target,"retained_bytes":retained,"requested_percent":config.percentages[index]}));
        selected.extend(chosen);
    }
    selected.shuffle(&mut StdRng::seed_from_u64(config.seed));
    let mut train = String::new();
    for fragment in &selected {
        train.push_str(
            &texts[fragment.source_index][fragment.offset..fragment.offset + fragment.bytes],
        );
        train.push('\n');
    }
    let total: usize = selected.iter().map(|f| f.bytes).sum();
    ensure!(
        train.len() <= config.max_text_bytes,
        "Budget de texte depasse"
    );
    for domain in &mut domains {
        domain["actual_source_percent"] =
            json!(100.0 * domain["retained_bytes"].as_u64().unwrap() as f64 / total as f64);
    }
    let manifest = json!({"version":1,"purpose":"tokenizer_candidate_training_only",
        "config_sha256":hash(&bytes),"config":config,"effective_source_budget":effective_budget,
        "domains":domains,"sources":sources,"fragments":selected,"separator_bytes":selected.len(),
        "train_bytes":train.len(),"train_sha256":hash(train.as_bytes()),"frozen":false,
        "selection":"4096-byte UTF-8 fragments shuffled without replacement; source bytes determine domain ratios; one newline separator per fragment",
        "limitations":["Pilot, not a representative final corpus","Source partition declarations are trusted; exact complete-file dedup only, no global near-duplicate check","512 MiB source RAM cap; text fragments may cut sentences or code","Tokenizer training does not train model weights; changed IDs cannot reuse old embeddings"]});
    if let Some(parent) = out.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    fs::create_dir(out)?;
    fs::write(out.join("train.txt"), train)?;
    fs::write(
        out.join("manifest.json"),
        serde_json::to_vec_pretty(&manifest)?,
    )?;
    println!(
        "Melange candidat : {total} octets sources, {} separateurs ; {}",
        selected.len(),
        out.display()
    );
    for domain in domains {
        println!(
            "{} : {} octets ({:.2} %)",
            domain["domain"],
            domain["retained_bytes"],
            domain["actual_source_percent"].as_f64().unwrap()
        );
    }
    Ok(())
}
