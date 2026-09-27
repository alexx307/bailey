//! Mesures de représentation sur des sondes de développement, sans modèle de langue.

use std::collections::{BTreeMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::Path;

use anyhow::{Context, Result, anyhow, ensure};
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use tokenizers::Tokenizer;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;

use super::{ASSISTANT, EOS, USER};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Probe {
    id: String,
    domain: String,
    text: String,
}

#[derive(Default)]
struct Counts {
    samples: usize,
    bytes: usize,
    characters: usize,
    tokens: usize,
    exact_roundtrips: usize,
    unknown_tokens: Option<usize>,
}

impl Counts {
    fn observe(&mut self, text: &str, ids: &[u32], decoded: &str, unknown: Option<u32>) {
        self.samples += 1;
        self.bytes += text.len();
        self.characters += text.chars().count();
        self.tokens += ids.len();
        self.exact_roundtrips += usize::from(text == decoded);
        if let Some(unknown) = unknown {
            *self.unknown_tokens.get_or_insert(0) +=
                ids.iter().filter(|&&id| id == unknown).count();
        }
    }

    fn report(&self) -> Value {
        json!({
            "samples": self.samples, "bytes": self.bytes, "characters": self.characters,
            "tokens": self.tokens, "bytes_per_token": ratio(self.bytes, self.tokens),
            "tokens_per_character": ratio(self.tokens, self.characters),
            "exact_roundtrips": self.exact_roundtrips,
            "all_roundtrips_exact": self.exact_roundtrips == self.samples,
            "unknown_tokens": self.unknown_tokens,
        })
    }
}

/// Écrit un nouveau fichier JSON ; refuse une destination existante.
/// Une comparaison utilise exactement les mêmes sondes pour les deux tokenizers.
pub fn audit(tokenizer: &Path, baseline: Option<&Path>, probes: &Path, out: &Path) -> Result<()> {
    ensure!(!out.exists(), "Le rapport {} existe déjà", out.display());
    let probe_bytes = read_limited(probes, 8 * 1024 * 1024)?;
    let samples: Vec<Probe> = serde_json::from_slice(&probe_bytes).context("Format des sondes")?;
    validate_probes(&samples)?;
    let candidate = measure(tokenizer, &samples)?;
    let baseline = baseline.map(|path| measure(path, &samples)).transpose()?;
    let comparison = baseline
        .as_ref()
        .map(|reference| compare(&candidate, reference));
    let report = json!({
        "format_version": 1,
        "purpose": "development_tokenizer_representation_only",
        "probes": {"path": probes.canonicalize()?, "sha256": sha256(&probe_bytes),
                   "count": samples.len()},
        "candidate": candidate, "baseline": baseline, "comparison": comparison,
        "limitations": [
            "Ces sondes de développement ne sont pas un test final réservé.",
            "Les caractères comptés sont des valeurs scalaires Unicode, pas des graphèmes.",
            "La présence des 256 symboles ByteLevel ne prouve pas seule le bon fonctionnement du pipeline.",
            "La couverture et la compression ne mesurent aucune compétence du modèle de langue.",
            "Ce rapport ne promeut pas automatiquement un tokenizer et ne migre aucun poids."
        ]
    });
    let encoded = serde_json::to_vec_pretty(&report)?;
    if let Some(parent) = out.parent().filter(|path| !path.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    let mut destination = OpenOptions::new().write(true).create_new(true).open(out)?;
    destination.write_all(&encoded)?;
    destination.sync_all()?;
    println!("Audit de {} sondes : {}", samples.len(), out.display());
    Ok(())
}

fn validate_probes(probes: &[Probe]) -> Result<()> {
    ensure!(
        !probes.is_empty() && probes.len() <= 10_000,
        "Nombre de sondes invalide"
    );
    let mut identifiers = HashSet::new();
    let mut texts = HashSet::new();
    for probe in probes {
        for field in [&probe.id, &probe.domain] {
            ensure!(
                !field.trim().is_empty() && field.trim() == field && field.len() <= 128,
                "Identifiant ou domaine invalide : {}",
                probe.id
            );
        }
        ensure!(
            !probe.text.is_empty() && probe.text.len() <= 65_536,
            "Texte de sonde invalide : {}",
            probe.id
        );
        ensure!(
            identifiers.insert(&probe.id),
            "Identifiant de sonde dupliqué : {}",
            probe.id
        );
        ensure!(
            texts.insert(&probe.text),
            "Texte de sonde dupliqué : {}",
            probe.id
        );
        ensure!(
            ![USER, ASSISTANT, EOS]
                .iter()
                .any(|marker| probe.text.contains(marker)),
            "Les marqueurs réservés sont mesurés séparément : {}",
            probe.id
        );
    }
    Ok(())
}

fn measure(path: &Path, probes: &[Probe]) -> Result<Value> {
    let file = if path.is_dir() {
        path.join("tokenizer.json")
    } else {
        path.to_owned()
    };
    let bytes = read_limited(&file, 64 * 1024 * 1024)?;
    let definition: Value = serde_json::from_slice(&bytes)?;
    let tokenizer = Tokenizer::from_bytes(&bytes).map_err(|error| anyhow!(error.to_string()))?;
    let unknown = definition
        .pointer("/model/unk_token")
        .and_then(Value::as_str)
        .and_then(|token| tokenizer.token_to_id(token))
        .or_else(|| {
            definition
                .pointer("/model/unk_id")
                .and_then(Value::as_u64)
                .and_then(|id| u32::try_from(id).ok())
        });
    let mut total = Counts::default();
    let mut domains: BTreeMap<String, Counts> = BTreeMap::new();
    let mut sample_reports = Vec::with_capacity(probes.len());
    for probe in probes {
        let encoding = tokenizer
            .encode(probe.text.as_str(), false)
            .map_err(|error| anyhow!(error.to_string()))?;
        let ids = encoding.get_ids();
        let decoded = tokenizer
            .decode(ids, false)
            .map_err(|error| anyhow!(error.to_string()))?;
        let mut counts = Counts::default();
        counts.observe(&probe.text, ids, &decoded, unknown);
        total.observe(&probe.text, ids, &decoded, unknown);
        domains.entry(probe.domain.clone()).or_default().observe(
            &probe.text,
            ids,
            &decoded,
            unknown,
        );
        sample_reports.push(json!({"id": probe.id, "domain": probe.domain,
            "text_sha256": sha256(probe.text.as_bytes()), "metrics": counts.report(),
            "decoded_on_mismatch": if decoded == probe.text { None } else { Some(decoded) }}));
    }
    let model_vocab = tokenizer.get_vocab(false);
    let mut missing: Vec<String> = ByteLevel::alphabet()
        .into_iter()
        .map(|c| c.to_string())
        .filter(|symbol| !model_vocab.contains_key(symbol))
        .collect();
    missing.sort();
    let markers = [USER, ASSISTANT, EOS].into_iter().map(|marker| {
        let encoded = tokenizer.encode(marker, false).map_err(|error| anyhow!(error.to_string()))?;
        let decoded = tokenizer.decode(encoded.get_ids(), false).map_err(|error| anyhow!(error.to_string()))?;
        let configured = tokenizer.token_to_id(marker);
        Ok(json!({"text": marker, "configured_id": configured, "encoded_ids": encoded.get_ids(),
            "atomic": configured.is_some_and(|id| encoded.get_ids() == [id]),
            "roundtrip_exact": decoded == marker}))
    }).collect::<Result<Vec<_>>>()?;
    let vocabulary: BTreeMap<_, _> = tokenizer.get_vocab(true).into_iter().collect();
    let domains: BTreeMap<_, _> = domains
        .into_iter()
        .map(|(name, count)| (name, count.report()))
        .collect();
    Ok(json!({
        "path": file.canonicalize()?, "sha256": sha256(&bytes),
        "vocab_size": tokenizer.get_vocab_size(true), "model_vocab_size": model_vocab.len(),
        "vocabulary_ids_sha256": sha256(&serde_json::to_vec(&vocabulary)?),
        "configured_unknown_id": unknown,
        "byte_level_alphabet": {"expected": 256, "present": 256 - missing.len(),
            "all_present": missing.is_empty(), "missing_symbols": missing},
        "reserved_markers": markers, "total": total.report(), "domains": domains, "samples": sample_reports
    }))
}

fn compare(candidate: &Value, baseline: &Value) -> Value {
    let comparison = |left: &Value, right: &Value| {
        let left = left["tokens"].as_u64().unwrap_or(0);
        let right = right["tokens"].as_u64().unwrap_or(0);
        json!({"candidate_tokens": left, "baseline_tokens": right,
            "token_difference": left as i64 - right as i64,
            "candidate_over_baseline_tokens": if right == 0 { None } else { Some(left as f64 / right as f64) }})
    };
    let domains: BTreeMap<_, _> = candidate["domains"]
        .as_object()
        .into_iter()
        .flatten()
        .map(|(name, value)| (name, comparison(value, &baseline["domains"][name])))
        .collect();
    json!({"same_probe_set": true, "identical_tokenizer_bytes": candidate["sha256"] == baseline["sha256"],
        "vocabulary_ids_equal": candidate["vocabulary_ids_sha256"] == baseline["vocabulary_ids_sha256"],
        "total": comparison(&candidate["total"], &baseline["total"]), "domains": domains})
}

fn ratio(numerator: usize, denominator: usize) -> Option<f64> {
    (denominator != 0).then(|| numerator as f64 / denominator as f64)
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn read_limited(path: &Path, limit: u64) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    File::open(path)
        .with_context(|| format!("Lecture de {}", path.display()))?
        .take(limit + 1)
        .read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() as u64 <= limit,
        "Fichier trop grand : {}",
        path.display()
    );
    Ok(bytes)
}

#[cfg(test)]
#[path = "audit_tests.rs"]
mod tests;
