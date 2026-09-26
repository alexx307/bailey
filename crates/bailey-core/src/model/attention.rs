use candle_core::{DType, Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder};

use super::{CoreConfig, feed_forward::projection, rotary::Rotary};

pub(super) struct Attention {
    query: Linear,
    key: Linear,
    value: Linear,
    output: Linear,
    query_heads: usize,
    kv_heads: usize,
    head_dim: usize,
}

impl Attention {
    pub fn new(config: &CoreConfig, vb: VarBuilder<'_>) -> Result<Self> {
        let hidden = config.hidden_size;
        let head_dim = hidden / config.num_attention_heads;
        let kv_width = head_dim * config.num_kv_heads;
        Ok(Self {
            query: projection(hidden, hidden, vb.pp("query"))?,
            key: projection(hidden, kv_width, vb.pp("key"))?,
            value: projection(hidden, kv_width, vb.pp("value"))?,
            output: projection(hidden, hidden, vb.pp("output"))?,
            query_heads: config.num_attention_heads,
            kv_heads: config.num_kv_heads,
            head_dim,
        })
    }

    fn repeat_kv(&self, input: &Tensor) -> Result<Tensor> {
        let (batch, _, length, width) = input.dims4()?;
        input
            .unsqueeze(2)?
            .broadcast_as((
                batch,
                self.kv_heads,
                self.query_heads / self.kv_heads,
                length,
                width,
            ))?
            .contiguous()?
            .reshape((batch, self.query_heads, length, width))
    }

    pub fn forward(&self, input: &Tensor, rotary: &Rotary, mask: &Tensor) -> Result<Tensor> {
        let (batch, length, hidden) = input.dims3()?;
        let query = self
            .query
            .forward(input)?
            .reshape((batch, length, self.query_heads, self.head_dim))?
            .transpose(1, 2)?;
        let key = self
            .key
            .forward(input)?
            .reshape((batch, length, self.kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let value = self
            .value
            .forward(input)?
            .reshape((batch, length, self.kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let query = rotary.apply(&query)?.contiguous()?;
        let key = self.repeat_kv(&rotary.apply(&key)?)?;
        let value = self.repeat_kv(&value)?;
        let scores = (query.matmul(&key.transpose(2, 3)?)? / (self.head_dim as f64).sqrt())?;
        let scores = scores.to_dtype(DType::F32)?.broadcast_add(mask)?;
        let probabilities = candle_nn::ops::softmax(&scores, 3)?.to_dtype(value.dtype())?;
        let attended = probabilities
            .matmul(&value)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((batch, length, hidden))?;
        self.output.forward(&attended)
    }
}
