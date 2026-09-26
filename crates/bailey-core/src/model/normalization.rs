use candle_core::{D, DType, Result, Tensor};
use candle_nn::{Init, VarBuilder};

/// RMSNorm exprimée en opérations différentiables ; calcul interne en F32.
pub(super) struct RmsNorm {
    weight: Tensor,
    epsilon: f64,
}

impl RmsNorm {
    pub fn new(width: usize, epsilon: f64, vb: VarBuilder<'_>) -> Result<Self> {
        Ok(Self {
            weight: vb.get_with_hints(width, "weight", Init::Const(1.))?,
            epsilon,
        })
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let float = input.to_dtype(DType::F32)?;
        let scale = (float.sqr()?.mean_keepdim(D::Minus1)? + self.epsilon)?.sqrt()?;
        float
            .broadcast_div(&scale)?
            .to_dtype(input.dtype())?
            .broadcast_mul(&self.weight)
    }
}
