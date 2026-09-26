use super::{Sampling, sampling};
use crate::tokenization::{EOS, Tokenizer};
use anyhow::Result;
use serde::Serialize;

#[derive(Serialize)]
pub struct Prediction {
    pub id: u32,
    pub token: String,
    pub probability: f64,
}

#[derive(Serialize)]
pub struct Diagnostics {
    pub probability_scope: &'static str,
    pub valid_token_count: usize,
    pub eos_probability: f64,
    pub top_tokens: Vec<Prediction>,
}

pub fn inspect(logits: &[f32], valid: &[u32], tokenizer: &Tokenizer) -> Result<Diagnostics> {
    let settings = Sampling {
        temperature: 1.0,
        top_k: valid.len(),
        top_p: 1.0,
        ..Default::default()
    };
    let distribution = sampling::distribution(logits, valid, &[], &settings)?;
    let eos = tokenizer.token_to_id(EOS);
    Ok(Diagnostics {
        probability_scope: "raw logits renormalized over valid tokenizer IDs, before sampling penalties",
        valid_token_count: valid.len(),
        eos_probability: distribution
            .iter()
            .find(|(id, _)| Some(*id) == eos)
            .map_or(0.0, |(_, p)| *p),
        top_tokens: distribution
            .into_iter()
            .take(10)
            .map(|(id, probability)| Prediction {
                id,
                probability,
                token: tokenizer.id_to_token(id).unwrap_or_default(),
            })
            .collect(),
    })
}
