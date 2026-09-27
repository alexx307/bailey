use std::fs::{self, File};
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, ensure};
use serde::Serialize;
use sha2::{Digest, Sha256};
use tokenizers::models::TrainerWrapper;
use tokenizers::models::bpe::{BPE, BpeTrainerBuilder};
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::{AddedToken, Tokenizer};

use super::{ASSISTANT, EOS, USER};

#[derive(Serialize)]
struct Manifest {
    format_version: u32,
    algorithm: &'static str,
    input: PathBuf,
    input_sha256: String,
    input_bytes: u64,
    requested_vocab_size: usize,
    actual_vocab_size: usize,
    tokenizer_sha256: String,
    source_manifest_sha256: Option<String>,
    status: &'static str,
    special_tokens: Vec<SpecialToken>,
    note: &'static str,
}

#[derive(Serialize)]
struct SpecialToken {
    text: &'static str,
    id: u32,
}

/// Apprend les fusions BPE à partir de ce seul `train.txt` local.
/// Les 256 valeurs d'octet restent représentables, même absentes du corpus.
pub fn train(input: &Path, out: &Path, vocab_size: usize) -> Result<()> {
    ensure!(
        input.file_name().and_then(|name| name.to_str()) == Some("train.txt"),
        "Sélectionner explicitement train.txt ; validation et test sont réservés"
    );
    ensure!(!out.exists(), "Le dossier {} existe déjà", out.display());
    ensure!(
        (259..=u32::MAX as usize).contains(&vocab_size),
        "Le vocabulaire doit contenir au moins 256 octets et 3 marqueurs"
    );
    let input = input
        .canonicalize()
        .context("Corpus d'entraînement absent")?;
    ensure!(input.is_file(), "Le corpus doit être un fichier local");
    let input_name = input
        .to_str()
        .context("Le chemin du corpus doit être UTF-8")?;
    let (input_sha256, input_bytes) = fingerprint(&input)?;
    ensure!(input_bytes > 0, "Le corpus d'entraînement est vide");
    let source_file = input.parent().unwrap().join("manifest.json");
    let source_manifest = read_source_manifest(&source_file, &input_sha256)?;

    let mut tokenizer = Tokenizer::new(BPE::default());
    let byte_level = ByteLevel::default()
        .add_prefix_space(false)
        .trim_offsets(false);
    tokenizer.with_pre_tokenizer(Some(byte_level));
    tokenizer.with_decoder(Some(byte_level));
    let mut trainer: TrainerWrapper = BpeTrainerBuilder::new()
        .vocab_size(vocab_size)
        .min_frequency(2)
        .show_progress(false)
        .initial_alphabet(ByteLevel::alphabet().into_iter().collect())
        .special_tokens(
            [USER, ASSISTANT, EOS]
                .into_iter()
                .map(|token| AddedToken::from(token, true))
                .collect(),
        )
        .build()
        .into();
    tokenizer
        .train_from_files(&mut trainer, vec![input_name.to_owned()])
        .map_err(|error| anyhow!(error.to_string()))
        .context("Apprentissage BPE sur train.txt")?;
    ensure!(
        fingerprint(&input)? == (input_sha256.clone(), input_bytes),
        "Le corpus a changé pendant l'apprentissage ; relancer sur un fichier stable"
    );
    ensure!(
        read_source_manifest(&source_file, &input_sha256)? == source_manifest,
        "La provenance a change pendant l'apprentissage"
    );
    let special_tokens = [USER, ASSISTANT, EOS]
        .into_iter()
        .map(|text| {
            Ok(SpecialToken {
                text,
                id: tokenizer
                    .token_to_id(text)
                    .with_context(|| format!("Marqueur manquant : {text}"))?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let serialized = tokenizer
        .to_string(true)
        .map_err(|error| anyhow!(error.to_string()))?;
    let manifest = Manifest {
        format_version: 1,
        algorithm: "byte-level-bpe",
        input,
        input_sha256,
        input_bytes,
        requested_vocab_size: vocab_size,
        actual_vocab_size: tokenizer.get_vocab_size(true),
        tokenizer_sha256: format!("{:x}", Sha256::digest(serialized.as_bytes())),
        source_manifest_sha256: source_manifest
            .as_ref()
            .map(|bytes| format!("{:x}", Sha256::digest(bytes))),
        status: "candidate_not_frozen",
        special_tokens,
        note: "La taille demandée est un plafond ; un petit corpus produit moins de fusions. Aucun poids de modèle de langue n'est appris ici.",
    };
    if let Some(parent) = out.parent().filter(|parent| !parent.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    // create_dir refuse aussi une destination créée pendant l'entraînement.
    fs::create_dir(out).context("Création d'une nouvelle version du tokenizer")?;
    fs::write(out.join("tokenizer.json"), serialized)?;
    if let Some(bytes) = source_manifest {
        fs::write(out.join("source-manifest.json"), bytes)?;
    }
    fs::write(
        out.join("manifest.json"),
        serde_json::to_vec_pretty(&manifest)?,
    )?;
    println!(
        "Tokenizer local : {} jetons appris (plafond demandé : {vocab_size}), {} octets de train.txt.",
        manifest.actual_vocab_size, input_bytes
    );
    Ok(())
}

fn read_source_manifest(path: &Path, input_sha256: &str) -> Result<Option<Vec<u8>>> {
    if !path.exists() {
        return Ok(None);
    }
    let mut bytes = Vec::new();
    File::open(path)?
        .take(16 * 1024 * 1024 + 1)
        .read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() <= 16 * 1024 * 1024,
        "Manifeste source trop volumineux"
    );
    let value: serde_json::Value = serde_json::from_slice(&bytes)?;
    ensure!(
        value["complete"].as_bool() != Some(false),
        "Import source incomplet"
    );
    if let Some(expected) = value["train_sha256"].as_str() {
        ensure!(
            expected == input_sha256,
            "Empreinte train incompatible avec la provenance"
        );
    }
    Ok(Some(bytes))
}

fn fingerprint(path: &Path) -> Result<(String, u64)> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut hash = Sha256::new();
    let mut buffer = [0_u8; 65_536];
    let mut length = 0_u64;
    loop {
        let count = reader.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
        length += count as u64;
    }
    Ok((format!("{:x}", hash.finalize()), length))
}
