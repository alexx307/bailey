use super::{TrainConfig, checkpoint, gradients, schedule, training_data::TrainingData};
use crate::{model::CoreModel, tokenization};
use anyhow::{Result, ensure};
use candle_core::{DType, Device};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use rand::{SeedableRng, rngs::StdRng};
use serde_json::json;
use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::Path,
    time::Instant,
};

pub fn train(config: TrainConfig, out: &Path, device: &Device) -> Result<()> {
    train_controlled(config, out, device, &|| false).map(|_| ())
}

pub fn train_controlled(
    mut config: TrainConfig,
    out: &Path,
    device: &Device,
    stop: &dyn Fn() -> bool,
) -> Result<bool> {
    if stop() {
        return Ok(false);
    }
    if config.tokenizer.is_dir() {
        config.tokenizer = config.tokenizer.join("tokenizer.json");
    }
    ensure!(!out.exists(), "Dossier deja utilise : {}", out.display());
    if let Some(previous) = &config.init_from {
        config.model = checkpoint::read_config(previous)?.model;
        ensure!(
            checkpoint::fingerprint(&config.tokenizer)?
                == checkpoint::fingerprint(&previous.join("tokenizer.json"))?,
            "Le tokenizer a change : les identifiants doivent rester identiques lors d'une reprise"
        );
    }
    config.validate()?;
    let tokenizer = tokenization::load(&config.tokenizer)?;
    ensure!(
        tokenizer.get_vocab_size(true) <= config.model.vocab_size,
        "Vocabulaire du tokenizer trop grand"
    );
    let train = TrainingData::load(&config.data.join("train.txt"), &tokenizer, &config)?;
    let validation = TrainingData::load(&config.data.join("validation.txt"), &tokenizer, &config)?;
    ensure!(train != validation, "Corpus train et validation identiques");
    if device.is_cuda() {
        device.set_seed(config.seed)?;
    }
    let mut vars = VarMap::new();
    let model = CoreModel::new(
        &config.model,
        VarBuilder::from_varmap(&vars, DType::F32, device),
    )?;
    if let Some(previous) = &config.init_from {
        checkpoint::load_weights(&mut vars, previous)?;
        println!(
            "Reprise des poids {} ; nouvel optimiseur",
            previous.display()
        );
    } else {
        println!("Premiere initialisation de cette architecture ; aucun poids externe.");
    }
    let actual: usize = vars.all_vars().iter().map(|v| v.elem_count()).sum();
    ensure!(
        actual == config.model.parameter_count()?,
        "Comptage incoherent : {actual}"
    );
    println!(
        "{actual} parametres | {} tokens train | {} validation | sequence {}",
        train.tokens(),
        validation.tokens(),
        config.sequence
    );
    if let Some(parent) = out.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    fs::create_dir(out)?;
    fs::copy(&config.tokenizer, out.join("tokenizer.json"))?;
    fs::write(out.join("config.json"), serde_json::to_vec_pretty(&config)?)?;
    fs::write(
        out.join("data-provenance.json"),
        serde_json::to_vec_pretty(&json!({
            "train_sha256":checkpoint::fingerprint(&config.data.join("train.txt"))?,
            "validation_sha256":checkpoint::fingerprint(&config.data.join("validation.txt"))?,
            "tokenizer_sha256":checkpoint::fingerprint(&config.tokenizer)?,
        "test_used":false,"backend_initialization_seeded":device.is_cuda(),"precision":"FP32",
        "device":format!("{device:?}"),"optimizer_resumed":false,"objective":config.objective
        }))?,
    )?;
    let mut log = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(out.join("metrics.jsonl"))?;
    let started = Instant::now();
    let mut best = validation.evaluate(&model, &config, device)?;
    ensure!(best.is_finite(), "Validation initiale non finie");
    checkpoint::save(&vars, out, 0, best)?;
    writeln!(log, "{}", json!({"step":0,"validation_loss":best}))?;
    println!("Validation initiale : {best:.4}");
    let parameters = vars.all_vars();
    let mut optimizer = AdamW::new(
        parameters.clone(),
        ParamsAdamW {
            lr: config.learning_rate,
            weight_decay: 0.01,
            ..Default::default()
        },
    )?;
    let mut rng = StdRng::seed_from_u64(config.seed);
    let mut compute_seconds = 0.0;
    let mut tokens_seen = 0usize;
    let mut supervised_tokens_seen = 0usize;
    println!(
        "Calcul sur {device:?} | warmup {} | clipping {:?} | validation {} elements max | objectif {}",
        config.warmup_steps,
        config.max_grad_norm,
        config.evaluation_windows,
        serde_json::to_string(&config.objective)?
    );
    for step in 1..=config.steps {
        if stop() || out.join("STOP").exists() {
            println!("Arret demande ; checkpoints conserves.");
            fs::write(
                out.join("interrupted.json"),
                format!("{{\"before_step\":{step}}}"),
            )?;
            return Ok(false);
        }
        let step_start = Instant::now();
        let rate = schedule::learning_rate(&config, step);
        optimizer.set_learning_rate(rate);
        let batch = train.batch(&config, &mut rng, device)?;
        tokens_seen += batch.input.elem_count();
        supervised_tokens_seen += batch.target.elem_count();
        let loss = batch.loss(&model)?;
        let value = loss.to_scalar::<f32>()?;
        ensure!(value.is_finite(), "Perte non finie a l'etape {step}");
        let mut grads = loss.backward()?;
        let grad_norm = config
            .max_grad_norm
            .map(|maximum| gradients::clip(&mut grads, &parameters, maximum))
            .transpose()?;
        optimizer.step(&grads)?;
        drop(grads);
        drop(loss);
        device.synchronize()?;
        compute_seconds += step_start.elapsed().as_secs_f64();
        if step % config.eval_every == 0 || step == config.steps {
            let score = validation.evaluate(&model, &config, device)?;
            ensure!(score.is_finite(), "Validation non finie");
            let selected = score < best;
            if selected {
                checkpoint::save(&vars, out, step, score)?;
                best = score;
            }
            writeln!(
                log,
                "{}",
                json!({"step":step,"train_loss":value,"validation_loss":score,"selected":selected,"elapsed_seconds":started.elapsed().as_secs_f64(),"learning_rate":rate,"gradient_norm_before_clip":grad_norm,"tokens_seen":tokens_seen,"supervised_tokens_seen":supervised_tokens_seen,"input_token_count_includes_padding":matches!(config.objective, super::Objective::Dialogue),"training_tokens_per_second":tokens_seen as f64/compute_seconds,"objective":config.objective})
            )?;
            log.flush()?;
            println!(
                "{step}/{} | train {value:.4} | validation {score:.4} | meilleur {best:.4} | lr {rate:.2e} | {:.0} tok/s",
                config.steps,
                tokens_seen as f64 / compute_seconds
            );
        }
    }
    println!(
        "Poids conserves : {} | {:.1}s",
        out.display(),
        started.elapsed().as_secs_f64()
    );
    Ok(true)
}
