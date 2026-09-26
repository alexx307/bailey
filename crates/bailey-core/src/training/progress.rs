use super::{DataFormat, TrainConfig, checkpoint, optimizer::StatefulAdamW};
use anyhow::{Result, ensure};
use candle_nn::VarMap;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fs,
    path::{Path, PathBuf},
};

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Progress {
    pub version: u32,
    pub step: usize,
    pub best_step: usize,
    pub best_loss: f32,
    pub input_tokens: usize,
    pub target_tokens: usize,
    pub signature: String,
    pub sampling: String,
    pub weights_sha256: String,
}

pub fn signature(config: &TrainConfig) -> Result<String> {
    let mut settings = config.clone();
    settings.data = PathBuf::new();
    settings.tokenizer = PathBuf::new();
    settings.init_from = None;
    let data = match config.data_format {
        DataFormat::Text => vec![
            checkpoint::fingerprint(&config.data.join("train.txt"))?,
            checkpoint::fingerprint(&config.data.join("validation.txt"))?,
        ],
        DataFormat::Shards => vec![checkpoint::fingerprint(
            &config.data.join("forge-manifest.json"),
        )?],
    };
    let bytes = serde_json::to_vec(&(settings, data, checkpoint::fingerprint(&config.tokenizer)?))?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

pub fn sampling_seed(seed: u64, step: usize) -> u64 {
    let mut z = seed.wrapping_add((step as u64).wrapping_mul(0x9e3779b97f4a7c15));
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

pub fn directory(run: &Path, step: usize) -> PathBuf {
    run.join("checkpoints").join(format!("step-{step:08}"))
}

fn pointer(run: &Path, name: &str, value: &impl Serialize) -> Result<()> {
    let next = run.join(format!("{name}.next.json"));
    fs::write(&next, serde_json::to_vec_pretty(value)?)?;
    fs::rename(next, run.join(format!("{name}.json")))?;
    Ok(())
}

pub fn save(
    vars: &VarMap,
    optimizer: &StatefulAdamW,
    run: &Path,
    state: &mut Progress,
) -> Result<()> {
    ensure!(
        optimizer.step_count() == state.step,
        "Compteur de l'optimiseur incoherent"
    );
    let folder = directory(run, state.step);
    fs::create_dir_all(folder.parent().unwrap())?;
    fs::create_dir(&folder)?;
    vars.save(folder.join("weights.safetensors"))?;
    state.weights_sha256 = checkpoint::fingerprint(&folder.join("weights.safetensors"))?;
    optimizer.save(&folder.join("optimizer"))?;
    fs::write(
        folder.join("trainer.json"),
        serde_json::to_vec_pretty(state)?,
    )?;
    if state.best_step == state.step {
        pointer(
            run,
            "best",
            &checkpoint::Selection {
                step: state.step,
                validation_loss: state.best_loss,
            },
        )?;
    }
    pointer(run, "latest", &serde_json::json!({"step":state.step}))?;
    Ok(())
}

pub fn read(run: &Path, config: &TrainConfig) -> Result<Progress> {
    #[derive(Deserialize)]
    struct Latest {
        step: usize,
    }
    let latest: Latest =
        serde_json::from_slice(&fs::read(run.join("latest.json")).map_err(|e| {
            anyhow::anyhow!(
                "Reprise complete indisponible ({e}) ; anciens checkpoints : utiliser --init-from"
            )
        })?)?;
    let folder = directory(run, latest.step);
    let state: Progress = serde_json::from_slice(&fs::read(folder.join("trainer.json"))?)?;
    ensure!(
        state.version == 1 && state.sampling == "splitmix64-std-rng-per-step-v1",
        "Etat d'entrainement incompatible"
    );
    ensure!(
        state.step == latest.step
            && state.step <= config.steps
            && state.best_step <= state.step
            && state.best_loss.is_finite(),
        "Progression invalide"
    );
    ensure!(
        state.signature == signature(config)?,
        "Donnees, tokenizer ou parametres d'entrainement modifies depuis le checkpoint"
    );
    ensure!(
        state.weights_sha256 == checkpoint::fingerprint(&folder.join("weights.safetensors"))?,
        "Poids du checkpoint alteres"
    );
    Ok(state)
}

pub fn inherit_best(previous: &Path, out: &Path, state: &Progress) -> Result<()> {
    // latest.json est le point de commit. best.json peut avoir ete avance juste
    // avant une interruption de publication ; la selection de l'etat fait foi.
    let best = checkpoint::Selection {
        step: state.best_step,
        validation_loss: state.best_loss,
    };
    if best.step != state.step {
        let source = directory(previous, best.step);
        let best_state: Progress = serde_json::from_slice(&fs::read(source.join("trainer.json"))?)?;
        ensure!(
            best_state.step == best.step
                && best_state.weights_sha256
                    == checkpoint::fingerprint(&source.join("weights.safetensors"))?,
            "Meilleurs poids precedents alteres"
        );
        let target = directory(out, best.step);
        fs::create_dir_all(&target)?;
        fs::copy(
            directory(previous, best.step).join("weights.safetensors"),
            target.join("weights.safetensors"),
        )?;
        fs::copy(source.join("trainer.json"), target.join("trainer.json"))?;
    }
    pointer(out, "best", &best)
}
