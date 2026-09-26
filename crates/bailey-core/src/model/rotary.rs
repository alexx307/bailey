use candle_core::{D, DType, Device, Result, Tensor};

/// Rotation des deux moitiés de chaque tête, sans noyau réservé à l'inférence.
pub(super) struct Rotary {
    cosine: Tensor,
    sine: Tensor,
    half: usize,
}

impl Rotary {
    pub fn new(head_dim: usize, length: usize, theta: f64, device: &Device) -> Result<Self> {
        let half = head_dim / 2;
        let mut cosine = Vec::with_capacity(length * half);
        let mut sine = Vec::with_capacity(length * half);
        for position in 0..length {
            for dimension in 0..half {
                let angle = position as f64 / theta.powf((2 * dimension) as f64 / head_dim as f64);
                cosine.push(angle.cos() as f32);
                sine.push(angle.sin() as f32);
            }
        }
        Ok(Self {
            cosine: Tensor::from_vec(cosine, (1, 1, length, half), device)?,
            sine: Tensor::from_vec(sine, (1, 1, length, half), device)?,
            half,
        })
    }

    pub fn apply(&self, input: &Tensor) -> Result<Tensor> {
        let length = input.dim(2)?;
        let dtype: DType = input.dtype();
        let cosine = self.cosine.narrow(2, 0, length)?.to_dtype(dtype)?;
        let sine = self.sine.narrow(2, 0, length)?.to_dtype(dtype)?;
        let first = input.narrow(D::Minus1, 0, self.half)?;
        let second = input.narrow(D::Minus1, self.half, self.half)?;
        let left = (first.broadcast_mul(&cosine)? - second.broadcast_mul(&sine)?)?;
        let right = (first.broadcast_mul(&sine)? + second.broadcast_mul(&cosine)?)?;
        Tensor::cat(&[&left, &right], D::Minus1)
    }
}
