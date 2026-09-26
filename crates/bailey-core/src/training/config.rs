use crate::model::CoreConfig;
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Clone, Copy, Default, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum Objective {
    #[default]
    NextToken,
    Dialogue,
}

#[derive(Clone, Copy, Default, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum DataFormat {
    #[default]
    Text,
    Shards,
}

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
    #[serde(default)]
    pub warmup_steps: usize,
    #[serde(default = "constant_rate")]
    pub min_lr_ratio: f64,
    #[serde(default)]
    pub max_grad_norm: Option<f64>,
    #[serde(default = "legacy_windows")]
    pub evaluation_windows: usize,
    #[serde(default)]
    pub objective: Objective,
    #[serde(default)]
    pub data_format: DataFormat,
}

fn constant_rate() -> f64 {
    1.0
}
fn legacy_windows() -> usize {
    4
}

impl TrainConfig {
    pub fn validate(&self) -> Result<()> {
        self.model.validate()?;
        ensure!(
            !matches!(
                (self.data_format, self.objective),
                (DataFormat::Shards, Objective::Dialogue)
            ),
            "Les shards actuels sont destines au texte continu ; le dialogue exige ses frontieres d'exemples"
        );
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
        ensure!(
            self.warmup_steps <= self.steps,
            "Warmup superieur a la seance"
        );
        ensure!(
            self.min_lr_ratio.is_finite() && (0.0..=1.0).contains(&self.min_lr_ratio),
            "Ratio final attendu : 0..1"
        );
        if let Some(norm) = self.max_grad_norm {
            ensure!(
                norm.is_finite() && norm > 0.0,
                "Norme de clipping positive requise"
            );
        }
        ensure!(
            (1..=1024).contains(&self.evaluation_windows),
            "Fenetres de validation attendues : 1..1024"
        );
        Ok(())
    }
}
