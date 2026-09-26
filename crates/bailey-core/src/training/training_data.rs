use super::{Objective, TrainConfig, dataset, dialogue::Dialogues, evaluation};
use crate::{model::CoreModel, tokenization::Tokenizer};
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

#[derive(PartialEq)]
pub enum TrainingData {
    Text(Vec<u32>),
    Dialogue(Dialogues),
}
impl TrainingData {
    pub fn load(file: &Path, tokenizer: &Tokenizer, config: &TrainConfig) -> Result<Self> {
        Ok(match config.objective {
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
        }
    }
    pub fn batch(&self, config: &TrainConfig, rng: &mut StdRng, device: &Device) -> Result<Batch> {
        match self {
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
