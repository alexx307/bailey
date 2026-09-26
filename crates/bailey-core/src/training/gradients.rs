use anyhow::{Result, ensure};
use candle_core::{Tensor, Var, backprop::GradStore};

/// Norme globale L2 sur les gradients des parametres, calculee sur leur device.
pub fn clip(grads: &mut GradStore, vars: &[Var], maximum: f64) -> Result<f64> {
    let mut squares = Vec::with_capacity(vars.len());
    for var in vars {
        let grad = grads
            .get(var)
            .ok_or_else(|| anyhow::anyhow!("Gradient de parametre absent"))?;
        squares.push(grad.sqr()?.sum_all()?);
    }
    let norm = Tensor::stack(&squares, 0)?
        .sum_all()?
        .sqrt()?
        .to_scalar::<f32>()? as f64;
    ensure!(norm.is_finite(), "Norme de gradient non finie");
    if norm > maximum {
        let scale = maximum / (norm + 1e-12);
        for var in vars {
            let grad = grads.remove(var).expect("gradient verifie");
            grads.insert(var, (grad * scale)?);
        }
    }
    Ok(norm)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn clips_joint_norm_and_keeps_direction() -> Result<()> {
        let a = Var::new(&[3f32], &Device::Cpu)?;
        let b = Var::new(&[4f32], &Device::Cpu)?;
        let loss = (a.sqr()?.sum_all()? + b.sqr()?.sum_all()?)?;
        let mut grads = loss.backward()?;
        assert!((clip(&mut grads, &[a.clone(), b.clone()], 5.)? - 10.).abs() < 1e-6);
        assert!((grads.get(&a).unwrap().to_vec1::<f32>()?[0] - 3.).abs() < 1e-6);
        assert!((grads.get(&b).unwrap().to_vec1::<f32>()?[0] - 4.).abs() < 1e-6);
        Ok(())
    }
}
