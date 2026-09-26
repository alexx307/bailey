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
    let ids = dataset::load(file, tokenizer, config.sequence, config.model.vocab_size)?;
    evaluate(model, &ids, config.sequence, device)
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
