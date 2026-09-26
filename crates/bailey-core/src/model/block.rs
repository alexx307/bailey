use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

use super::{
    CoreConfig, attention::Attention, feed_forward::FeedForward, normalization::RmsNorm,
    rotary::Rotary,
};

pub(super) struct Block {
    attention_norm: RmsNorm,
    attention: Attention,
    feed_forward_norm: RmsNorm,
    feed_forward: FeedForward,
}

impl Block {
    pub fn new(config: &CoreConfig, vb: VarBuilder<'_>) -> Result<Self> {
        Ok(Self {
            attention_norm: RmsNorm::new(
                config.hidden_size,
                config.rms_norm_eps,
                vb.pp("attention_norm"),
            )?,
            attention: Attention::new(config, vb.pp("attention"))?,
            feed_forward_norm: RmsNorm::new(
                config.hidden_size,
                config.rms_norm_eps,
                vb.pp("feed_forward_norm"),
            )?,
            feed_forward: FeedForward::new(config, vb.pp("feed_forward"))?,
        })
    }

    pub fn forward(&self, input: &Tensor, rotary: &Rotary, mask: &Tensor) -> Result<Tensor> {
        let attended =
            self.attention
                .forward(&self.attention_norm.forward(input)?, rotary, mask)?;
        let residual = (input + attended)?;
        let transformed = self
            .feed_forward
            .forward(&self.feed_forward_norm.forward(&residual)?)?;
        residual + transformed
    }
}
