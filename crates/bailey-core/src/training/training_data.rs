use super::{Objective, TrainConfig, dataset, dialogue::Dialogues, evaluation};
use crate::{
    forge::dataset::stream::StreamingPartition, model::CoreModel, tokenization::Tokenizer,
};
use anyhow::Result;
use candle_core::{Device, Tensor};
use rand::{Rng, rngs::StdRng};
use std::{fs, path::Path};

pub struct Batch {
    pub input: Tensor,
    pub target: Tensor,
    pub positions: Option<Tensor>,
}
impl Batch {
    pub fn loss(&self, model: &CoreModel) -> Result<Tensor> {
        let logits = model.forward(&self.input)?.flatten_to(1)?;
        let logits = if let Some(positions) = &self.positions {
            logits.index_select(positions, 0)?
        } else {
            logits
        };
        Ok(candle_nn::loss::cross_entropy(&logits, &self.target)?)
    }
}

pub enum TrainingData {
    Text(Vec<u32>),
    Dialogue(Dialogues),
    Stream(StreamingPartition),
}
impl TrainingData {
    pub fn load(file: &Path, tokenizer: &Tokenizer, config: &TrainConfig) -> Result<Self> {
        Ok(match config.objective {
            Objective::NextToken
                if matches!(config.data_format, super::DataFormat::ShardsStream) =>
            {
                let partition = file
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .ok_or_else(|| anyhow::anyhow!("Partition invalide"))?;
                println!(
                    "Verification en flux de {partition} ; cache {} Mio, aucun chargement integral en RAM.",
                    config.shard_cache_mib
                );
                let reader = StreamingPartition::open(
                    &config.data,
                    partition,
                    &config.tokenizer,
                    config.model.vocab_size,
                    config.shard_cache_mib * 1024 * 1024,
                )?;
                anyhow::ensure!(
                    reader.len() > config.sequence,
                    "Partition trop courte pour le contexte"
                );
                Self::Stream(reader)
            }
            Objective::NextToken => {
                let ids = match config.data_format {
                    super::DataFormat::Text => {
                        dataset::load(file, tokenizer, config.sequence, config.model.vocab_size)?
                    }
                    super::DataFormat::Shards => crate::forge::dataset::load_partition(
                        &config.data,
                        file.file_stem()
                            .and_then(|s| s.to_str())
                            .ok_or_else(|| anyhow::anyhow!("Partition invalide"))?,
                        &config.tokenizer,
                        config.model.vocab_size,
                    )?,
                    super::DataFormat::ShardsStream => unreachable!("branche flux traitee avant"),
                };
                anyhow::ensure!(
                    ids.len() > config.sequence,
                    "Partition trop courte pour le contexte"
                );
                Self::Text(ids)
            }
            Objective::Dialogue => Self::Dialogue(Dialogues::parse(
                &fs::read_to_string(file)?,
                tokenizer,
                config.sequence,
                config.model.vocab_size,
            )?),
        })
    }
    pub fn tokens(&self) -> usize {
        match self {
            Self::Text(ids) => ids.len(),
            Self::Dialogue(data) => data.tokens,
            Self::Stream(reader) => reader.len(),
        }
    }
    pub fn reading_stats(&self) -> serde_json::Value {
        match self {
            Self::Stream(reader) => serde_json::json!({"mode":"shards_stream","io":reader.stats()}),
            Self::Text(ids) => {
                serde_json::json!({"mode":"resident_tokens","token_bytes":ids.len()*4})
            }
            Self::Dialogue(data) => {
                serde_json::json!({"mode":"resident_dialogue","tokens":data.tokens})
            }
        }
    }
    pub fn same_content(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Text(a), Self::Text(b)) => a == b,
            (Self::Dialogue(a), Self::Dialogue(b)) => a == b,
            (Self::Stream(a), Self::Stream(b)) => a.fingerprint() == b.fingerprint(),
            _ => false,
        }
    }
    pub fn batch(&self, config: &TrainConfig, rng: &mut StdRng, device: &Device) -> Result<Batch> {
        match self {
            Self::Stream(reader) => super::streaming::random_batch(reader, config, rng, device),
            Self::Text(ids) => {
                let (input, target) =
                    dataset::random_batch(ids, config.batch_size, config.sequence, rng, device)?;
                Ok(Batch {
                    input,
                    target,
                    positions: None,
                })
            }
            Self::Dialogue(data) => {
                let indexes: Vec<usize> = (0..config.batch_size)
                    .map(|_| rng.random_range(0..data.examples.len()))
                    .collect();
                data.batch(&indexes, device)
            }
        }
    }
    pub fn evaluate(
        &self,
        model: &CoreModel,
        config: &TrainConfig,
        device: &Device,
    ) -> Result<f32> {
        match self {
            Self::Stream(reader) => super::streaming::evaluate(reader, model, config, device),
            Self::Text(ids) => evaluation::evaluate_windows(
                model,
                ids,
                config.sequence,
                config.evaluation_windows,
                device,
            ),
            Self::Dialogue(data) => {
                let count = config.evaluation_windows.min(data.examples.len());
                let mut sum = 0f64;
                let mut tokens = 0;
                for i in 0..count {
                    let index = i * (data.examples.len() - 1) / count.saturating_sub(1).max(1);
                    let batch = data.batch(&[index], device)?;
                    let n = batch.target.elem_count();
                    sum += batch.loss(model)?.to_scalar::<f32>()? as f64 * n as f64;
                    tokens += n;
                }
                Ok((sum / tokens as f64) as f32)
            }
        }
    }
}

pub fn write_io_report(train: &TrainingData, validation: &TrainingData, out: &Path) -> Result<()> {
    fs::write(
        out.join("data-reader.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "train":train.reading_stats(),"validation":validation.reading_stats(),"test_used":false,
            "note":"Cache counts token pages; tokenizer, index, file handles, batches and OS cache are separate"
        }))?,
    )?;
    Ok(())
}
