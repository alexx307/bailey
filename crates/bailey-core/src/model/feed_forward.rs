use candle_core::{Result, Tensor};
use candle_nn::{Init, Linear, Module, VarBuilder};

use super::CoreConfig;

pub(super) fn projection(input: usize, output: usize, vb: VarBuilder<'_>) -> Result<Linear> {
    let weight = vb.get_with_hints(
        (output, input),
        "weight",
        Init::Randn {
            mean: 0.,
            stdev: 0.02,
        },
    )?;
    Ok(Linear::new(weight, None))
}

pub(super) struct FeedForward {
    gate: Linear,
    up: Linear,
    down: Linear,
}

impl FeedForward {
    pub fn new(config: &CoreConfig, vb: VarBuilder<'_>) -> Result<Self> {
        let (hidden, intermediate) = (config.hidden_size, config.intermediate_size);
        Ok(Self {
            gate: projection(hidden, intermediate, vb.pp("gate"))?,
            up: projection(hidden, intermediate, vb.pp("up"))?,
            down: projection(intermediate, hidden, vb.pp("down"))?,
        })
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let gate = self.gate.forward(input)?;
        // UnaryOp::Silu fournit un backward Candle (contrairement au RoPE fusionné).
        let silu = gate.silu()?;
        self.down.forward(&(silu * self.up.forward(input)?)?)
    }
}
