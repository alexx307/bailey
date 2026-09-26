use super::{
    Sampling,
    diagnostics::{self, Diagnostics},
    sampling,
};
use crate::{
    model::{CoreConfig, CoreModel},
    tokenization::{EOS, Tokenizer},
};
use anyhow::{Result, ensure};
use candle_core::{Device, IndexOp, Tensor};
use rand::{SeedableRng, rngs::StdRng};
use serde::Serialize;

#[derive(Serialize)]
pub struct Generation {
    pub text: String,
    pub tokens_generated: usize,
    pub stop_reason: &'static str,
    pub first_prediction: Diagnostics,
}

pub fn generate(
    model: &CoreModel,
    config: &CoreConfig,
    tokenizer: &Tokenizer,
    prompt: &str,
    maximum: usize,
    device: &Device,
) -> Result<String> {
    Ok(generate_with(
        model,
        config,
        tokenizer,
        prompt,
        maximum,
        &Sampling::default(),
        device,
    )?
    .text)
}

pub fn generate_with(
    model: &CoreModel,
    config: &CoreConfig,
    tokenizer: &Tokenizer,
    prompt: &str,
    maximum: usize,
    options: &Sampling,
    device: &Device,
) -> Result<Generation> {
    options.validate()?;
    ensure!(
        (1..=2048).contains(&maximum),
        "Generation limitee a 2048 tokens"
    );
    let mut ids = tokenizer
        .encode(prompt, false)
        .map_err(|e| anyhow::anyhow!(e.to_string()))?
        .get_ids()
        .to_vec();
    ensure!(!ids.is_empty(), "Texte vide");
    let prompt_length = ids.len();
    let mut valid: Vec<u32> = tokenizer.get_vocab(true).into_values().collect();
    valid.sort_unstable();
    valid.dedup();
    let end_ids: Vec<u32> = [
        EOS,
        crate::tokenization::USER,
        crate::tokenization::ASSISTANT,
    ]
    .iter()
    .filter_map(|s| tokenizer.token_to_id(s))
    .collect();
    let mut rng = StdRng::seed_from_u64(options.seed);
    let mut first_prediction = None;
    let mut stop_reason = "token_limit";
    for _ in 0..maximum {
        let start = ids.len().saturating_sub(config.max_seq_len);
        let input = Tensor::new(&ids[start..], device)?.unsqueeze(0)?;
        let logits = model
            .forward(&input)?
            .i((0, ids.len() - start - 1, ..))?
            .to_vec1::<f32>()?;
        if first_prediction.is_none() {
            first_prediction = Some(diagnostics::inspect(&logits, &valid, tokenizer)?);
        }
        let probabilities =
            sampling::distribution(&logits, &valid, &ids[prompt_length..], options)?;
        let next = sampling::choose(&probabilities, &mut rng);
        if end_ids.contains(&next) {
            stop_reason = "end_or_role_token";
            break;
        }
        ids.push(next);
    }
    let text = tokenizer
        .decode(&ids[prompt_length..], true)
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    let text = text
        .chars()
        .map(|c| {
            if c.is_control() && c != '\n' && c != '\t' {
                '\u{fffd}'
            } else {
                c
            }
        })
        .collect();
    Ok(Generation {
        text,
        tokens_generated: ids.len() - prompt_length,
        stop_reason,
        first_prediction: first_prediction.expect("au moins une prediction"),
    })
}
