use anyhow::{Result, ensure};
use candle_core::{Device, Tensor};
use rand::{Rng, rngs::StdRng};
use std::{fs, path::Path};
use tokenizers::Tokenizer;

pub fn load(file: &Path, tokenizer: &Tokenizer, sequence: usize, vocab: usize) -> Result<Vec<u32>> {
    let text = fs::read_to_string(file)?;
    let encoding = tokenizer
        .encode(text, false)
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    let ids = encoding.get_ids().to_vec();
    ensure!(
        ids.len() > sequence,
        "{} : {} tokens, plus de {sequence} requis",
        file.display(),
        ids.len()
    );
    ensure!(
        ids.iter().all(|id| (*id as usize) < vocab),
        "Un token depasse le vocabulaire du modele"
    );
    Ok(ids)
}

pub fn random_batch(
    ids: &[u32],
    size: usize,
    sequence: usize,
    rng: &mut StdRng,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let starts = (0..size).map(|_| rng.random_range(0..ids.len() - sequence));
    batch(ids, sequence, starts, device)
}

pub fn fixed_batch(ids: &[u32], sequence: usize, device: &Device) -> Result<(Tensor, Tensor)> {
    let span = ids.len() - sequence - 1;
    batch(ids, sequence, (0..4).map(|i| i * span / 3), device)
}

fn batch(
    ids: &[u32],
    sequence: usize,
    starts: impl Iterator<Item = usize>,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let mut input = Vec::new();
    let mut target = Vec::new();
    for start in starts {
        input.extend_from_slice(&ids[start..start + sequence]);
        target.extend_from_slice(&ids[start + 1..start + sequence + 1]);
    }
    let size = input.len() / sequence;
    Ok((
        Tensor::from_vec(input, (size, sequence), device)?,
        Tensor::from_vec(target, (size * sequence,), device)?,
    ))
}
