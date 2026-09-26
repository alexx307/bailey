use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Optimizer, SGD, VarBuilder, VarMap};

use super::{CoreConfig, CoreModel, block::Block, rotary::Rotary};

fn model() -> Result<(CoreModel, VarMap, CoreConfig)> {
    let config = CoreConfig::tiny(32);
    let variables = VarMap::new();
    let builder = VarBuilder::from_varmap(&variables, DType::F32, &Device::Cpu);
    Ok((CoreModel::new(&config, builder)?, variables, config))
}

#[test]
fn configuration_matches_the_shared_head_parameter_budget() -> Result<()> {
    assert_eq!(CoreConfig::default().parameter_count()?, 100_092_672);
    let (_, variables, config) = model()?;
    let actual: usize = variables
        .all_vars()
        .iter()
        .map(|var| var.elem_count())
        .sum();
    assert_eq!(actual, config.parameter_count()?);
    let named = variables.data().lock().unwrap();
    assert_eq!(named.len(), 2 + config.num_layers * 9);
    assert!(named.contains_key("embedding.weight"));
    assert!(!named.contains_key("lm_head.weight"));
    assert!(!named.contains_key("output.weight"));
    assert!(named.keys().all(|name| !name.ends_with("bias")));
    Ok(())
}

#[test]
fn rejects_invalid_configuration_and_context() -> Result<()> {
    let valid = CoreConfig::tiny(32);
    for invalid in [
        CoreConfig {
            vocab_size: 0,
            ..valid.clone()
        },
        CoreConfig {
            hidden_size: 65,
            ..valid.clone()
        },
        CoreConfig {
            hidden_size: 12,
            ..valid.clone()
        },
        CoreConfig {
            num_kv_heads: 3,
            ..valid.clone()
        },
        CoreConfig {
            num_attention_heads: 0,
            ..valid.clone()
        },
        CoreConfig {
            rope_theta: f64::NAN,
            ..valid.clone()
        },
        CoreConfig {
            rms_norm_eps: 0.,
            ..valid.clone()
        },
    ] {
        assert!(invalid.validate().is_err());
    }
    let (model, _, config) = model()?;
    let overlong = Tensor::zeros((1, config.max_seq_len + 1), DType::U32, &Device::Cpu)?;
    assert!(model.forward(&overlong).is_err());
    let empty = Tensor::zeros((1, 0), DType::U32, &Device::Cpu)?;
    assert!(model.forward(&empty).is_err());
    let float = Tensor::zeros((1, 2), DType::F32, &Device::Cpu)?;
    assert!(model.forward(&float).is_err());
    Ok(())
}

#[test]
fn future_tokens_cannot_change_earlier_predictions() -> Result<()> {
    let (model, _, config) = model()?;
    let original = Tensor::new(&[[1u32, 2, 3, 4, 5]], &Device::Cpu)?;
    let changed = Tensor::new(&[[1u32, 2, 3, 8, 9]], &Device::Cpu)?;
    let original = model.forward(&original)?;
    let changed = model.forward(&changed)?;
    assert_eq!(original.dims(), &[1, 5, config.vocab_size]);
    let prefix_error = (original.narrow(1, 0, 3)? - changed.narrow(1, 0, 3)?)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(prefix_error < 1e-6, "Fuite causale : {prefix_error}");
    let suffix_difference = (original.narrow(1, 3, 2)? - changed.narrow(1, 3, 2)?)?
        .abs()?
        .sum_all()?
        .to_scalar::<f32>()?;
    assert!(suffix_difference > 1e-4);
    Ok(())
}

#[test]
fn every_parameter_receives_a_finite_gradient_and_an_update() -> Result<()> {
    let (model, variables, config) = model()?;
    let input = Tensor::new(&[[1u32, 2, 3, 4, 5], [6u32, 7, 8, 9, 10]], &Device::Cpu)?;
    let targets = Tensor::new(&[2u32, 3, 4, 5, 6, 7, 8, 9, 10, 11], &Device::Cpu)?;
    let logits = model.forward(&input)?.reshape((10, config.vocab_size))?;
    let loss = candle_nn::loss::cross_entropy(&logits, &targets)?;
    assert!(loss.to_scalar::<f32>()?.is_finite());
    let gradients = loss.backward()?;
    let mut saved = Vec::new();
    for (name, parameter) in variables.data().lock().unwrap().iter() {
        let gradient = gradients
            .get(parameter)
            .unwrap_or_else(|| panic!("Gradient absent : {name}"));
        let values = gradient.flatten_all()?.to_vec1::<f32>()?;
        assert!(
            values.iter().all(|value| value.is_finite()),
            "Gradient non fini : {name}"
        );
        assert!(
            values.iter().any(|value| value.abs() > 0.),
            "Gradient nul : {name}"
        );
        saved.push((
            name.clone(),
            parameter.clone(),
            parameter.flatten_all()?.to_vec1::<f32>()?,
        ));
    }
    let mut optimizer = SGD::new(variables.all_vars(), 0.1)?;
    optimizer.step(&gradients)?;
    for (name, parameter, before) in saved {
        let after = parameter.flatten_all()?.to_vec1::<f32>()?;
        assert!(
            before.iter().zip(&after).any(|(a, b)| a != b),
            "Paramètre inchangé : {name}"
        );
    }
    Ok(())
}

#[test]
fn zero_sublayers_preserve_the_residual_input() -> Result<()> {
    let config = CoreConfig::tiny(32);
    let variables = VarMap::new();
    let builder = VarBuilder::from_varmap(&variables, DType::F32, &Device::Cpu);
    let block = Block::new(&config, builder)?;
    for parameter in variables.all_vars() {
        if parameter.rank() == 2 {
            parameter.set(&parameter.zeros_like()?)?;
        }
    }
    let input = Tensor::arange(0f32, (3 * config.hidden_size) as f32, &Device::Cpu)?.reshape((
        1,
        3,
        config.hidden_size,
    ))?;
    let rotary = Rotary::new(16, 3, config.rope_theta, &Device::Cpu)?;
    let mask = Tensor::zeros((1, 1, 3, 3), DType::F32, &Device::Cpu)?;
    let result = block.forward(&input, &rotary, &mask)?;
    let error = (input - result)?.abs()?.sum_all()?.to_scalar::<f32>()?;
    assert_eq!(error, 0.);
    Ok(())
}

#[test]
fn rotary_preserves_norm_and_rotates_positions_after_zero() -> Result<()> {
    let rotary = Rotary::new(4, 2, 10_000., &Device::Cpu)?;
    let input =
        Tensor::new(&[1f32, 2., 3., 4., 1., 2., 3., 4.], &Device::Cpu)?.reshape((1, 1, 2, 4))?;
    let rotated = rotary.apply(&input)?;
    let values = rotated.flatten_all()?.to_vec1::<f32>()?;
    assert_eq!(&values[..4], &[1., 2., 3., 4.]);
    assert!((values[4] - 1.).abs() > 0.1);
    let norm_difference = (rotated.sqr()?.sum_keepdim(3)? - input.sqr()?.sum_keepdim(3)?)?
        .abs()?
        .sum_all()?
        .to_scalar::<f32>()?;
    assert!(norm_difference < 1e-5);
    Ok(())
}
