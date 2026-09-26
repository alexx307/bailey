mod metadata;
mod persistence;
#[cfg(test)]
mod tests;

use anyhow::{Result, ensure};
use candle_core::{Var, backprop::GradStore};
use candle_nn::VarMap;
use metadata::Hyperparameters;

struct Parameter {
    name: String,
    weight: Var,
    first: Var,
    second: Var,
}

/// AdamW avec moments nommes, pour reprendre une trajectoire d'optimisation.
/// Les poids eux-memes sont sauvegardes par le gestionnaire de checkpoints.
pub struct StatefulAdamW {
    parameters: Vec<Parameter>,
    step: usize,
    hyperparameters: Hyperparameters,
}

impl StatefulAdamW {
    pub fn new(vars: &VarMap, learning_rate: f64) -> Result<Self> {
        let hyperparameters = Hyperparameters::new(learning_rate);
        hyperparameters.validate()?;
        let parameters = named_variables(vars)?
            .into_iter()
            .map(|(name, weight)| {
                let first = Var::zeros(weight.shape(), weight.dtype(), weight.device())?;
                let second = Var::zeros(weight.shape(), weight.dtype(), weight.device())?;
                Ok(Parameter {
                    name,
                    weight,
                    first,
                    second,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            parameters,
            step: 0,
            hyperparameters,
        })
    }

    pub fn set_learning_rate(&mut self, learning_rate: f64) -> Result<()> {
        ensure!(
            learning_rate.is_finite() && learning_rate >= 0.,
            "Taux AdamW invalide"
        );
        self.hyperparameters.learning_rate = learning_rate;
        Ok(())
    }

    pub fn step_count(&self) -> usize {
        self.step
    }
    pub fn variables(&self) -> Vec<Var> {
        self.parameters.iter().map(|p| p.weight.clone()).collect()
    }

    pub fn step(&mut self, gradients: &GradStore) -> Result<()> {
        ensure!(self.step < i32::MAX as usize, "Compteur AdamW hors limite");
        // Verifier tous les gradients avant de modifier le premier parametre.
        for parameter in &self.parameters {
            let gradient = gradients
                .get(&parameter.weight)
                .ok_or_else(|| anyhow::anyhow!("Gradient absent : {}", parameter.name))?;
            ensure!(
                gradient.shape() == parameter.weight.shape()
                    && gradient.dtype() == parameter.weight.dtype(),
                "Gradient incompatible : {}",
                parameter.name
            );
        }
        let next_step = self.step + 1;
        let p = &self.hyperparameters;
        let scale_m = 1. / (1. - p.beta1.powi(next_step as i32));
        let scale_v = 1. / (1. - p.beta2.powi(next_step as i32));
        for parameter in &self.parameters {
            let gradient = gradients.get(&parameter.weight).expect("gradient verifie");
            // Ordre des operations identique a candle-nn 0.11.0 AdamW.
            let next_m = ((parameter.first.as_tensor() * p.beta1)? + (gradient * (1. - p.beta1))?)?;
            let next_v =
                ((parameter.second.as_tensor() * p.beta2)? + (gradient.sqr()? * (1. - p.beta2))?)?;
            let m_hat = (&next_m * scale_m)?;
            let v_hat = (&next_v * scale_v)?;
            let next_weight =
                (parameter.weight.as_tensor() * (1. - p.learning_rate * p.weight_decay))?;
            let adjusted_gradient = (m_hat / (v_hat.sqrt()? + p.epsilon)?)?;
            let next_weight = (next_weight - (adjusted_gradient * p.learning_rate)?)?;
            parameter.first.set(&next_m)?;
            parameter.second.set(&next_v)?;
            parameter.weight.set(&next_weight)?;
        }
        self.step = next_step;
        Ok(())
    }
}

fn named_variables(vars: &VarMap) -> Result<Vec<(String, Var)>> {
    let data = vars
        .data()
        .lock()
        .map_err(|_| anyhow::anyhow!("VarMap verrouille apres erreur"))?;
    ensure!(!data.is_empty(), "Aucun parametre pour AdamW");
    let mut parameters = data
        .iter()
        .map(|(name, var)| (name.clone(), var.clone()))
        .collect::<Vec<_>>();
    parameters.sort_by(|left, right| left.0.cmp(&right.0));
    for (name, variable) in &parameters {
        ensure!(
            !name.is_empty() && variable.dtype().is_float() && variable.elem_count() > 0,
            "Parametre AdamW invalide : {name}"
        );
    }
    Ok(parameters)
}
