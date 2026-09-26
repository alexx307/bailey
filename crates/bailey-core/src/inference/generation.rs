use crate::{
    model::{CoreConfig, CoreModel},
    tokenization::{EOS, Tokenizer},
};
use anyhow::{Result, ensure};
use candle_core::{Device, IndexOp, Tensor};

pub fn generate(
    model: &CoreModel,
    config: &CoreConfig,
    tokenizer: &Tokenizer,
    prompt: &str,
    maximum: usize,
    device: &Device,
) -> Result<String> {
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
    let valid = tokenizer.get_vocab(true);
    let eos = tokenizer.token_to_id(EOS);
    for _ in 0..maximum {
        let start = ids.len().saturating_sub(config.max_seq_len);
        let input = Tensor::new(&ids[start..], device)?.unsqueeze(0)?;
        let logits = model
            .forward(&input)?
            .i((0, ids.len() - start - 1, ..))?
            .to_vec1::<f32>()?;
        let next = valid
            .values()
            .copied()
            .filter(|id| (*id as usize) < logits.len())
            .max_by(|a, b| {
                logits[*a as usize]
                    .total_cmp(&logits[*b as usize])
                    .then_with(|| b.cmp(a))
            })
            .ok_or_else(|| anyhow::anyhow!("Aucun token valide"))?;
        if Some(next) == eos {
            break;
        }
        ids.push(next);
    }
    let text = tokenizer
        .decode(&ids[prompt_length..], true)
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    Ok(text
        .chars()
        .map(|c| {
            if c.is_control() && c != '\n' && c != '\t' {
                '\u{fffd}'
            } else {
                c
            }
        })
        .collect())
}
