use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct ResearchConfig {
    pub language: String,
    pub topics: Vec<String>,
    pub cycles: usize,
    pub pages_per_topic: usize,
    pub interval_seconds: u64,
    pub max_minutes: u64,
    pub max_library_bytes: u64,
}

impl Default for ResearchConfig {
    fn default() -> Self {
        Self {
            language: "fr".into(),
            topics: [
                "programmation informatique",
                "Rust langage",
                "algorithme",
                "mathematiques",
                "physique",
                "biologie",
                "histoire",
                "geographie",
                "litterature",
                "art",
            ]
            .into_iter()
            .map(str::to_owned)
            .collect(),
            cycles: 10,
            pages_per_topic: 3,
            interval_seconds: 5,
            max_minutes: 30,
            max_library_bytes: 64 * 1024 * 1024,
        }
    }
}

impl ResearchConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(!self.topics.is_empty(), "Au moins un sujet est requis");
        ensure!(self.topics.len() <= 100, "Maximum 100 sujets par session");
        ensure!(
            self.topics
                .iter()
                .all(|topic| !topic.trim().is_empty() && topic.chars().count() <= 512),
            "Chaque sujet doit contenir de 1 a 512 caracteres"
        );
        ensure!(
            self.cycles > 0 && self.cycles <= 100_000,
            "Cycles attendus : 1..100000"
        );
        ensure!(
            (1..=50).contains(&self.pages_per_topic),
            "Pages par sujet attendues : 1..50"
        );
        ensure!(
            (1..=3600).contains(&self.interval_seconds),
            "Intervalle attendu : 1..3600 secondes"
        );
        ensure!(
            (1..=10_080).contains(&self.max_minutes),
            "Duree attendue : 1..10080 minutes"
        );
        ensure!(
            self.max_library_bytes > 0,
            "Budget de bibliotheque positif requis"
        );
        crate::web::WikiClient::new(&self.language, 20, 2 * 1024 * 1024)?;
        Ok(())
    }
}
