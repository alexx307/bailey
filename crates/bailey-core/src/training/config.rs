use crate::model::CoreConfig;
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Clone, Serialize, Deserialize)]
pub struct TrainConfig {
    pub model: CoreConfig,
    pub data: PathBuf,
    pub tokenizer: PathBuf,
    pub steps: usize,
    pub sequence: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub eval_every: usize,
    pub seed: u64,
    pub init_from: Option<PathBuf>,
}

impl TrainConfig {
    pub fn validate(&self) -> Result<()> {
        self.model.validate()?;
        ensure!(
            (1..=1_000_000).contains(&self.steps),
            "Etapes attendues : 1..1000000"
        );
        ensure!(
            (2..=self.model.max_seq_len).contains(&self.sequence),
            "Sequence invalide"
        );
        ensure!((1..=64).contains(&self.batch_size), "Lot attendu : 1..64");
        ensure!(
            self.learning_rate.is_finite() && self.learning_rate > 0.0,
            "Taux d'apprentissage invalide"
        );
        ensure!(
            self.eval_every > 0,
            "Frequence d'evaluation positive requise"
        );
        Ok(())
    }
}
