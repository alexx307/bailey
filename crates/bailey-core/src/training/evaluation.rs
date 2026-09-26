use super::dataset;
use crate::model::CoreModel;
use anyhow::Result;
use candle_core::{Device, Tensor};

pub fn evaluate_file(
    model: &CoreModel,
    file: &std::path::Path,
    tokenizer: &tokenizers::Tokenizer,
    config: &super::TrainConfig,
    device: &Device,
) -> Result<f32> {
    super::training_data::TrainingData::load(file, tokenizer, config)?
        .evaluate(model, config, device)
}

pub fn loss(model: &CoreModel, input: &Tensor, target: &Tensor) -> Result<Tensor> {
    Ok(candle_nn::loss::cross_entropy(
        &model.forward(input)?.flatten_to(1)?,
        target,
    )?)
}

pub fn evaluate(model: &CoreModel, ids: &[u32], sequence: usize, device: &Device) -> Result<f32> {
    let (input, target) = dataset::fixed_batch(ids, sequence, device)?;
    Ok(loss(model, &input, &target)?.to_scalar::<f32>()?)
}

/// Fenetres reparties sur tout le fichier ; un lot a la fois pour borner la VRAM.
pub fn evaluate_windows(
    model: &CoreModel,
    ids: &[u32],
    sequence: usize,
    windows: usize,
    device: &Device,
) -> Result<f32> {
    anyhow::ensure!(
        ids.len() > sequence && windows > 0,
        "Validation vide ou trop courte"
    );
    let span = ids.len() - sequence - 1;
    let count = windows.min(span + 1);
    let mut sum = 0f64;
    for i in 0..count {
        let start = i * span / count.saturating_sub(1).max(1);
        let (input, target) = dataset::window(ids, sequence, start, device)?;
        sum += loss(model, &input, &target)?.to_scalar::<f32>()? as f64;
    }
    Ok((sum / count as f64) as f32)
}
