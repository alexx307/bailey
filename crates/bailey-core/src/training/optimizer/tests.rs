use super::*;
use candle_core::{DType, Device};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder};

fn variables() -> Result<VarMap> {
    let vars = VarMap::new();
    let vb = VarBuilder::from_varmap(&vars, DType::F32, &Device::Cpu);
    vb.get_with_hints((2,), "weights", candle_nn::Init::Const(0.5))?;
    Ok(vars)
}
fn gradients(vars: &VarMap) -> Result<GradStore> {
    Ok(vars.all_vars()[0].sqr()?.sum_all()?.backward()?)
}
fn values(vars: &VarMap) -> Result<Vec<f32>> {
    Ok(vars.all_vars()[0].to_vec1::<f32>()?)
}

#[test]
fn matches_candle_and_resume_preserves_moments() -> Result<()> {
    let left = variables()?;
    let right = variables()?;
    let mut custom = StatefulAdamW::new(&left, 0.003)?;
    let mut candle = AdamW::new(
        right.all_vars(),
        ParamsAdamW {
            lr: 0.003,
            weight_decay: 0.01,
            ..Default::default()
        },
    )?;
    for _ in 0..5 {
        custom.step(&gradients(&left)?)?;
        candle.step(&gradients(&right)?)?;
    }
    assert_eq!(values(&left)?, values(&right)?);
    let temp = tempfile::tempdir()?;
    let out = temp.path().join("adam");
    custom.save(&out)?;
    assert!(custom.save(&out).is_err());
    let resumed = variables()?;
    resumed.all_vars()[0].set(left.all_vars()[0].as_tensor())?;
    let mut loaded = StatefulAdamW::load(&out, &resumed)?;
    for _ in 0..3 {
        custom.step(&gradients(&left)?)?;
        loaded.step(&gradients(&resumed)?)?;
    }
    assert_eq!(values(&left)?, values(&resumed)?);
    assert_eq!(loaded.step_count(), 8);
    std::fs::write(out.join("moments.safetensors"), b"broken")?;
    assert!(StatefulAdamW::load(&out, &resumed).is_err());
    Ok(())
}
