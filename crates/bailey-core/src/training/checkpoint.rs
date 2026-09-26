use super::TrainConfig;
use crate::model::CoreModel;
use anyhow::{Result, ensure};
use candle_core::{DType, Device};
use candle_nn::{VarBuilder, VarMap};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{fs, path::Path};

#[derive(Serialize, Deserialize)]
pub struct Selection {
    pub step: usize,
    pub validation_loss: f32,
}

pub fn fingerprint(file: &Path) -> Result<String> {
    Ok(format!("{:x}", Sha256::digest(fs::read(file)?)))
}

pub fn read_config(run: &Path) -> Result<TrainConfig> {
    let config: TrainConfig = serde_json::from_slice(&fs::read(run.join("config.json"))?)?;
    config.validate()?;
    Ok(config)
}

pub fn save(vars: &VarMap, run: &Path, step: usize, value: f32) -> Result<()> {
    ensure!(value.is_finite(), "Perte non finie");
    let directory = run.join("checkpoints").join(format!("step-{step:08}"));
    fs::create_dir_all(directory.parent().unwrap())?;
    fs::create_dir(&directory)?;
    vars.save(directory.join("weights.safetensors"))?;
    let temporary = run.join("best.next.json");
    fs::write(
        &temporary,
        serde_json::to_vec_pretty(&Selection {
            step,
            validation_loss: value,
        })?,
    )?;
    fs::rename(temporary, run.join("best.json"))?;
    Ok(())
}

pub fn load_weights(vars: &mut VarMap, run: &Path) -> Result<()> {
    let selected: Selection = serde_json::from_slice(&fs::read(run.join("best.json"))?)?;
    vars.load(
        run.join("checkpoints")
            .join(format!("step-{:08}", selected.step))
            .join("weights.safetensors"),
    )?;
    Ok(())
}

pub fn load(run: &Path, device: &Device) -> Result<(CoreModel, TrainConfig)> {
    let config = read_config(run)?;
    let mut vars = VarMap::new();
    let model = CoreModel::new(
        &config.model,
        VarBuilder::from_varmap(&vars, DType::F32, device),
    )?;
    load_weights(&mut vars, run)?;
    Ok((model, config))
}
