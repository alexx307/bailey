use super::{TrainConfig, training_data::Batch};
use crate::{forge::dataset::stream::StreamingPartition, model::CoreModel};
use anyhow::{Result, ensure};
use candle_core::{Device, Tensor};
use rand::{Rng, rngs::StdRng};

fn batch(
    reader: &StreamingPartition,
    sequence: usize,
    starts: impl Iterator<Item = usize>,
    device: &Device,
) -> Result<Batch> {
    let mut input = Vec::new();
    let mut target = Vec::new();
    for start in starts {
        let ids = reader.read_window(start, sequence + 1)?;
        input.extend_from_slice(&ids[..sequence]);
        target.extend_from_slice(&ids[1..]);
    }
    let size = input.len() / sequence;
    Ok(Batch {
        input: Tensor::from_vec(input, (size, sequence), device)?,
        target: Tensor::from_vec(target, (size * sequence,), device)?,
        positions: None,
    })
}

pub fn random_batch(
    reader: &StreamingPartition,
    config: &TrainConfig,
    rng: &mut StdRng,
    device: &Device,
) -> Result<Batch> {
    // Meme distribution et meme nombre de tirages que le chargeur en RAM.
    let starts =
        (0..config.batch_size).map(|_| rng.random_range(0..reader.len() - config.sequence));
    batch(reader, config.sequence, starts, device)
}

pub fn evaluate(
    reader: &StreamingPartition,
    model: &CoreModel,
    config: &TrainConfig,
    device: &Device,
) -> Result<f32> {
    ensure!(reader.len() > config.sequence, "Partition trop courte");
    let span = reader.len() - config.sequence - 1;
    let count = config.evaluation_windows.min(span + 1);
    let mut sum = 0f64;
    for i in 0..count {
        let start = i * span / count.saturating_sub(1).max(1);
        sum += batch(reader, config.sequence, [start].into_iter(), device)?
            .loss(model)?
            .to_scalar::<f32>()? as f64;
    }
    Ok((sum / count as f64) as f32)
}
