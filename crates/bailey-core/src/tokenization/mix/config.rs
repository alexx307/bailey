use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum Domain {
    French,
    English,
    Code,
}
pub const DOMAINS: [Domain; 3] = [Domain::French, Domain::English, Domain::Code];

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Source {
    pub id: String,
    pub domain: Domain,
    pub partition: String,
    pub file: PathBuf,
    pub sha256: String,
    pub provenance: PathBuf,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub seed: u64,
    pub max_text_bytes: usize,
    /// Ordre : français, anglais, code. Parts mesurees en octets UTF-8 sources.
    pub percentages: [usize; 3],
    pub sources: Vec<Source>,
}

impl Config {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            (100..=256 * 1024 * 1024).contains(&self.max_text_bytes),
            "Melange pilote : budget attendu de 100 octets a 256 Mio"
        );
        ensure!(
            self.percentages.iter().all(|&n| (1..=98).contains(&n))
                && self.percentages.iter().sum::<usize>() == 100,
            "Trois pourcentages positifs dont la somme vaut 100 sont requis"
        );
        ensure!(
            !self.sources.is_empty() && self.sources.len() <= 10000,
            "1..10000 sources explicites attendues"
        );
        let mut ids = std::collections::HashSet::new();
        for source in &self.sources {
            ensure!(
                !source.id.trim().is_empty() && ids.insert(&source.id),
                "Identifiant de source vide ou repete"
            );
            ensure!(
                source.partition == "train",
                "Seul train peut former le tokenizer"
            );
            ensure!(
                !matches!(
                    source.file.file_stem().and_then(|v| v.to_str()),
                    Some("test" | "validation")
                ),
                "Fichier reserve interdit dans le melange"
            );
            ensure!(
                source.sha256.len() == 64 && source.sha256.bytes().all(|v| v.is_ascii_hexdigit()),
                "Empreinte attendue invalide : {}",
                source.id
            );
        }
        Ok(())
    }
}
