use candle_core::{DType, Result, Tensor, bail};
use candle_nn::{Embedding, Init, Linear, Module, VarBuilder};

use super::{CoreConfig, block::Block, normalization::RmsNorm, rotary::Rotary};

pub struct CoreModel {
    embedding: Embedding,
    blocks: Vec<Block>,
    final_norm: RmsNorm,
    output: Linear,
    rotary: Rotary,
    max_seq_len: usize,
}

impl CoreModel {
    pub fn new(config: &CoreConfig, vb: VarBuilder<'_>) -> Result<Self> {
        config.validate()?;
        config.parameter_count()?;
        let weight = vb.pp("embedding").get_with_hints(
            (config.vocab_size, config.hidden_size),
            "weight",
            Init::Randn {
                mean: 0.,
                stdev: 0.02,
            },
        )?;
        let embedding = Embedding::new(weight.clone(), config.hidden_size);
        let output = Linear::new(weight, None);
        let blocks = (0..config.num_layers)
            .map(|index| Block::new(config, vb.pp(format!("layers.{index}"))))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            embedding,
            blocks,
            final_norm: RmsNorm::new(config.hidden_size, config.rms_norm_eps, vb.pp("final_norm"))?,
            output,
            rotary: Rotary::new(
                config.hidden_size / config.num_attention_heads,
                config.max_seq_len,
                config.rope_theta,
                vb.device(),
            )?,
            max_seq_len: config.max_seq_len,
        })
    }

    /// Recalcule le contexte complet ; aucun cache KV d'inférence pour l'instant.
    pub fn forward(&self, tokens: &Tensor) -> Result<Tensor> {
        let (batch, length) = tokens.dims2()?;
        if batch == 0 || length == 0 || length > self.max_seq_len {
            bail!(
                "Lot vide ou longueur {length} hors contexte 1..={}",
                self.max_seq_len
            );
        }
        if !matches!(tokens.dtype(), DType::U32 | DType::I64) {
            bail!("Les identifiants de tokens doivent être U32 ou I64");
        }
        let mask: Vec<f32> = (0..length)
            .flat_map(|row| {
                (0..length).map(move |column| if column > row { f32::NEG_INFINITY } else { 0. })
            })
            .collect();
        let mask = Tensor::from_vec(mask, (1, 1, length, length), tokens.device())?;
        let mut hidden = self.embedding.forward(tokens)?;
        for block in &self.blocks {
            hidden = block.forward(&hidden, &self.rotary, &mask)?;
        }
        self.output.forward(&self.final_norm.forward(&hidden)?)
    }
}
