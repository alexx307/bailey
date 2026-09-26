use anyhow::{Result, ensure};
use candle_core::Var;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Hyperparameters {
    pub learning_rate: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
    pub weight_decay: f64,
}

impl Hyperparameters {
    pub fn new(learning_rate: f64) -> Self {
        Self { learning_rate, beta1: 0.9, beta2: 0.999, epsilon: 1e-8, weight_decay: 0.01 }
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(self.learning_rate.is_finite() && self.learning_rate >= 0.,
            "Taux AdamW invalide");
        ensure!(self.beta1 == 0.9 && self.beta2 == 0.999 && self.epsilon == 1e-8
            && self.weight_decay == 0.01, "Hyperparametres AdamW incompatibles");
        Ok(())
    }
}

#[derive(Serialize, Deserialize, PartialEq, Debug)]
#[serde(deny_unknown_fields)]
pub(super) struct Descriptor {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: String,
}

impl Descriptor {
    pub fn new(name: &str, variable: &Var) -> Self {
        Self {
            name: name.to_owned(),
            shape: variable.dims().to_vec(),
            dtype: format!("{:?}", variable.dtype()),
        }
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Metadata {
    pub version: u32,
    pub algorithm: String,
    pub step: usize,
    pub hyperparameters: Hyperparameters,
    pub parameters: Vec<Descriptor>,
    pub moments_sha256: String,
}

impl Metadata {
    pub fn validate(&self) -> Result<()> {
        ensure!(self.version == 1 && self.algorithm == "adamw", "Format AdamW incompatible");
        ensure!(self.step <= i32::MAX as usize, "Compteur AdamW hors limite");
        self.hyperparameters.validate()
    }
}
