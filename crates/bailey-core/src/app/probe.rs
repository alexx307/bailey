use crate::{
    model::{CoreConfig, CoreModel},
    training::evaluation,
};
use anyhow::{Result, ensure};
use candle_core::{DType, Device, Tensor};
use candle_nn::{AdamW, Optimizer, VarBuilder, VarMap};
use serde_json::json;
use std::{fs, path::Path, time::Instant};

pub fn run(
    config: &CoreConfig,
    sequence: usize,
    steps: usize,
    report: &Path,
    device: &Device,
) -> Result<()> {
    config.validate()?;
    ensure!(
        (2..=config.max_seq_len).contains(&sequence),
        "Sequence invalide"
    );
    ensure!((1..=10).contains(&steps), "Controle limite a 10 pas");
    ensure!(!report.exists(), "Rapport deja present");
    let vars = VarMap::new();
    let started = Instant::now();
    let model = CoreModel::new(config, VarBuilder::from_varmap(&vars, DType::F32, device))?;
    let actual: usize = vars.all_vars().iter().map(|v| v.elem_count()).sum();
    ensure!(
        actual == config.parameter_count()?,
        "Comptage different du plan"
    );
    let ids: Vec<u32> = (0..=sequence)
        .map(|i| (i % config.vocab_size) as u32)
        .collect();
    let input = Tensor::new(&ids[..sequence], device)?.reshape((1, sequence))?;
    let target = Tensor::new(&ids[1..], device)?;
    let mut optimizer = AdamW::new_lr(vars.all_vars(), 0.0001)?;
    let mut losses = Vec::new();
    for step in 1..=steps {
        let loss = evaluation::loss(&model, &input, &target)?;
        let value = loss.to_scalar::<f32>()?;
        ensure!(value.is_finite(), "Perte non finie");
        let gradients = loss.backward()?;
        let all = vars.data().lock().unwrap();
        for (name, var) in all.iter() {
            ensure!(
                gradients.get(var.as_tensor()).is_some(),
                "Gradient absent : {name}"
            );
        }
        drop(all);
        optimizer.step(&gradients)?;
        losses.push(value);
        println!("Controle {step}/{steps} | {actual} parametres | perte {value:.4}");
    }
    let result = json!({"parameters":actual,"sequence":sequence,"steps":steps,
        "device":format!("{device:?}"),"precision":"FP32","all_gradients_present":true,
        "losses":losses,"elapsed_seconds":started.elapsed().as_secs_f64(),
        "purpose":"forward/backward/optimizer check only; not language pretraining"});
    if let Some(parent) = report.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    fs::write(report, serde_json::to_vec_pretty(&result)?)?;
    println!("{result}");
    Ok(())
}
