use anyhow::{Result, ensure};
use rand::{Rng, rngs::StdRng};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize, clap::Args)]
pub struct Sampling {
    /// Zero = token le plus probable.
    #[arg(long, default_value_t = 0.0)]
    pub temperature: f64,
    #[arg(long, default_value_t = 40)]
    pub top_k: usize,
    #[arg(long, default_value_t = 0.9)]
    pub top_p: f64,
    #[arg(long, default_value_t = 1.0)]
    pub repetition_penalty: f64,
    #[arg(long, default_value_t = 42)]
    pub seed: u64,
}

impl Default for Sampling {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_k: 40,
            top_p: 0.9,
            repetition_penalty: 1.0,
            seed: 42,
        }
    }
}

impl Sampling {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.temperature.is_finite() && self.temperature >= 0.0,
            "Temperature positive ou nulle requise"
        );
        ensure!(self.top_k > 0, "top-k doit etre positif");
        ensure!(
            self.top_p.is_finite() && self.top_p > 0.0 && self.top_p <= 1.0,
            "top-p attendu dans ]0,1]"
        );
        ensure!(
            self.repetition_penalty.is_finite() && self.repetition_penalty >= 1.0,
            "Penalite de repetition >= 1 requise"
        );
        Ok(())
    }
}

pub fn distribution(
    logits: &[f32],
    valid: &[u32],
    history: &[u32],
    options: &Sampling,
) -> Result<Vec<(u32, f64)>> {
    options.validate()?;
    let mut scores = Vec::with_capacity(valid.len());
    for &id in valid {
        let raw = *logits
            .get(id as usize)
            .ok_or_else(|| anyhow::anyhow!("ID hors logits"))? as f64;
        ensure!(raw.is_finite(), "Logit non fini pour le token {id}");
        let penalized = if history.contains(&id) {
            if raw < 0.0 {
                raw * options.repetition_penalty
            } else {
                raw / options.repetition_penalty
            }
        } else {
            raw
        };
        scores.push((id, penalized));
    }
    ensure!(!scores.is_empty(), "Aucun token valide");
    scores.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    if options.temperature == 0.0 {
        return Ok(vec![(scores[0].0, 1.0)]);
    }
    scores.truncate(options.top_k);
    let maximum = scores[0].1;
    for (_, value) in &mut scores {
        *value = ((*value - maximum) / options.temperature).exp();
    }
    let total: f64 = scores.iter().map(|(_, p)| p).sum();
    let mut cumulative = 0.0;
    let keep = scores
        .iter()
        .position(|(_, p)| {
            cumulative += p / total;
            cumulative >= options.top_p
        })
        .map_or(scores.len(), |i| i + 1);
    scores.truncate(keep);
    let retained: f64 = scores.iter().map(|(_, p)| p).sum();
    for (_, p) in &mut scores {
        *p /= retained;
    }
    Ok(scores)
}

pub fn choose(distribution: &[(u32, f64)], rng: &mut StdRng) -> u32 {
    let draw = rng.random::<f64>();
    let mut sum = 0.0;
    for &(id, p) in distribution {
        sum += p;
        if draw < sum {
            return id;
        }
    }
    distribution.last().expect("distribution non vide").0
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn excludes_unknown_ids_and_respects_top_k_and_top_p() -> Result<()> {
        let settings = Sampling {
            temperature: 1.0,
            top_k: 2,
            top_p: 0.9,
            ..Default::default()
        };
        let p = distribution(&[1., 3., 2., 1000.], &[0, 1, 2], &[], &settings)?;
        assert_eq!(p.iter().map(|x| x.0).collect::<Vec<_>>(), [1, 2]);
        assert!((p.iter().map(|x| x.1).sum::<f64>() - 1.).abs() < 1e-9);
        let settings = Sampling {
            top_p: 0.1,
            ..settings
        };
        assert_eq!(
            distribution(&[1., 3., 2.], &[0, 1, 2], &[], &settings)?,
            [(1, 1.)]
        );
        Ok(())
    }

    #[test]
    fn seed_is_repeatable_and_invalid_numbers_are_rejected() -> Result<()> {
        let mut a = StdRng::seed_from_u64(7);
        let mut b = StdRng::seed_from_u64(7);
        for _ in 0..30 {
            assert_eq!(
                choose(&[(1, 0.5), (2, 0.5)], &mut a),
                choose(&[(1, 0.5), (2, 0.5)], &mut b)
            );
        }
        assert!(distribution(&[f32::NAN], &[0], &[], &Sampling::default()).is_err());
        assert!(
            Sampling {
                top_p: 0.0,
                ..Default::default()
            }
            .validate()
            .is_err()
        );
        assert_eq!(
            distribution(
                &[2., 3.],
                &[0, 1],
                &[1],
                &Sampling {
                    repetition_penalty: 2.,
                    ..Default::default()
                }
            )?,
            [(0, 1.)]
        );
        Ok(())
    }
}
