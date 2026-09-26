use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};

use super::PARTITIONS;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FileRecord {
    pub file: String,
    pub sha256: String,
    pub bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TokenizerRecord {
    pub artifact: FileRecord,
    pub vocab_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Shard {
    pub artifact: FileRecord,
    pub tokens: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Partition {
    pub name: String,
    pub source: FileRecord,
    pub tokens: u64,
    pub shards: Vec<Shard>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub format_version: u32,
    pub encoding: String,
    pub shard_tokens: usize,
    pub tokenizer: TokenizerRecord,
    pub provenance: FileRecord,
    pub partitions: Vec<Partition>,
    pub duplicate_policy: String,
    pub limitations: Vec<String>,
}

pub(super) fn shard_name(partition: &str, index: usize) -> String {
    format!("{partition}/shard-{index:06}.u32")
}

fn fingerprint(record: &FileRecord) -> Result<()> {
    ensure!(record.bytes > 0, "Fichier vide dans le manifeste");
    ensure!(
        record.sha256.len() == 64 && record.sha256.bytes().all(|b| b.is_ascii_hexdigit()),
        "Empreinte SHA256 invalide"
    );
    Ok(())
}

impl Manifest {
    pub(super) fn validate(&self) -> Result<()> {
        ensure!(
            self.format_version == 1,
            "Version de corpus non prise en charge"
        );
        ensure!(self.encoding == "u32-le", "Encodage de shards inconnu");
        ensure!(self.shard_tokens > 0, "Taille de shard nulle");
        ensure!(self.tokenizer.vocab_size > 0, "Vocabulaire vide");
        ensure!(
            self.tokenizer.artifact.file == "tokenizer.json",
            "Chemin tokenizer invalide"
        );
        ensure!(
            self.provenance.file == "source-manifest.json",
            "Chemin provenance invalide"
        );
        fingerprint(&self.tokenizer.artifact)?;
        fingerprint(&self.provenance)?;
        ensure!(self.partitions.len() == 3, "Trois partitions sont requises");
        for (partition, name) in self.partitions.iter().zip(PARTITIONS) {
            ensure!(partition.name == name, "Ordre ou nom de partition invalide");
            ensure!(
                partition.source.file == format!("{name}.txt"),
                "Chemin source invalide"
            );
            fingerprint(&partition.source)?;
            ensure!(
                partition.tokens > 0 && !partition.shards.is_empty(),
                "Partition vide"
            );
            let mut total = 0_u64;
            for (index, shard) in partition.shards.iter().enumerate() {
                ensure!(
                    shard.artifact.file == shard_name(name, index),
                    "Chemin de shard invalide"
                );
                fingerprint(&shard.artifact)?;
                ensure!(
                    shard.tokens > 0 && shard.tokens <= self.shard_tokens as u64,
                    "Nombre de tokens invalide"
                );
                ensure!(
                    shard.tokens.checked_mul(4) == Some(shard.artifact.bytes),
                    "Taille de shard incohérente"
                );
                if index + 1 < partition.shards.len() {
                    ensure!(
                        shard.tokens == self.shard_tokens as u64,
                        "Shard intermédiaire incomplet"
                    );
                }
                total = total
                    .checked_add(shard.tokens)
                    .ok_or_else(|| anyhow::anyhow!("Nombre de tokens hors limites"))?;
            }
            ensure!(total == partition.tokens, "Total de tokens incohérent");
        }
        for (index, partition) in self.partitions.iter().enumerate() {
            ensure!(
                self.partitions[..index]
                    .iter()
                    .all(|other| other.source.sha256 != partition.source.sha256),
                "Sources identiques entre partitions"
            );
        }
        Ok(())
    }
}
