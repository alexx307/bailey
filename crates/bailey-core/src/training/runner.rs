use super::{
    TrainConfig, checkpoint, gradients,
    optimizer::StatefulAdamW,
    progress::{self, Progress},
    schedule,
    training_data::TrainingData,
};
use crate::{model::CoreModel, tokenization};
use anyhow::{Result, ensure};
use candle_core::{DType, Device};
use candle_nn::{VarBuilder, VarMap};
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
pub fn train_until(
    config: TrainConfig,
    out: &Path,
    device: &Device,
    stop_after: Option<usize>,
) -> Result<bool> {
    run(config, out, device, &|| false, None, stop_after)
}
pub fn train_controlled(
    config: TrainConfig,
    out: &Path,
    device: &Device,
    stop: &dyn Fn() -> bool,
) -> Result<bool> {
    run(config, out, device, stop, None, None)
}
pub fn resume(
    previous: &Path,
    out: &Path,
    data: Option<&Path>,
    device: &Device,
    stop_after: Option<usize>,
) -> Result<bool> {
    let mut config = checkpoint::read_config(previous)?;
    config.tokenizer = previous.join("tokenizer.json");
    if let Some(data) = data {
        config.data = data.to_owned();
    }
    run(config, out, device, &|| false, Some(previous), stop_after)
}

fn run(
    mut config: TrainConfig,
    out: &Path,
    device: &Device,
    stop: &dyn Fn() -> bool,
    previous: Option<&Path>,
    stop_after: Option<usize>,
) -> Result<bool> {
    if stop() {
        return Ok(false);
    }
    ensure!(!out.exists(), "Dossier deja utilise : {}", out.display());
    if config.tokenizer.is_dir() {
        config.tokenizer = config.tokenizer.join("tokenizer.json");
    }
    config.tokenizer = fs::canonicalize(&config.tokenizer)?;
    config.data = fs::canonicalize(&config.data)?;
    if previous.is_none()
        && let Some(initial) = &config.init_from
    {
        config.model = checkpoint::read_config(initial)?.model;
        ensure!(
            checkpoint::fingerprint(&config.tokenizer)?
                == checkpoint::fingerprint(&initial.join("tokenizer.json"))?,
            "Le tokenizer a change : ne pas reutiliser ces poids avec de nouveaux IDs"
        );
    }
    config.validate()?;
    ensure!(
        stop_after.is_none_or(|n| n <= config.steps),
        "Arret demande au-dela des etapes prevues"
    );
    let tokenizer = tokenization::load(&config.tokenizer)?;
    ensure!(
        tokenizer.get_vocab_size(true) <= config.model.vocab_size,
        "Vocabulaire du tokenizer trop grand"
    );
    let train = TrainingData::load(&config.data.join("train.txt"), &tokenizer, &config)?;
    let validation = TrainingData::load(&config.data.join("validation.txt"), &tokenizer, &config)?;
    ensure!(
        !train.same_content(&validation),
        "Corpus train et validation identiques"
    );
    let restored = previous
        .map(|run| progress::read(run, &config))
        .transpose()?;
    if let Some(state) = &restored {
        ensure!(
            state.step < config.steps,
            "Cette seance est deja terminee ; --init-from commence une nouvelle seance"
        );
        ensure!(
            stop_after.is_none_or(|n| n >= state.step),
            "Arret anterieur au checkpoint"
        );
    }
    if device.is_cuda() {
        device.set_seed(config.seed)?;
    }
    let mut vars = VarMap::new();
    let model = CoreModel::new(
        &config.model,
        VarBuilder::from_varmap(&vars, DType::F32, device),
    )?;
    let mut optimizer = if let (Some(source), Some(state)) = (previous, &restored) {
        let folder = progress::directory(source, state.step);
        vars.load(folder.join("weights.safetensors"))?;
        let optimizer = StatefulAdamW::load(&folder.join("optimizer"), &vars)?;
        ensure!(
            optimizer.step_count() == state.step,
            "Compteur AdamW incoherent"
        );
        println!(
            "Reprise complete a l'etape {} : poids, moments Adam, planning et tirage des lots.",
            state.step
        );
        optimizer
    } else {
        if let Some(initial) = &config.init_from {
            checkpoint::load_weights(&mut vars, initial)?;
            println!(
                "Reprise des poids {} ; nouvel optimiseur.",
                initial.display()
            );
        } else {
            println!("Premiere initialisation ; aucun poids externe.");
        }
        StatefulAdamW::new(&vars, config.learning_rate)?
    };
    let actual: usize = vars.all_vars().iter().map(|v| v.elem_count()).sum();
    ensure!(
        actual == config.model.parameter_count()?,
        "Comptage de parametres incoherent"
    );
    println!(
        "{actual} parametres | {} tokens train | {} validation | contexte maximal {}",
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
    let started = Instant::now();
    let mut state = if let Some(state) = restored {
        state
    } else {
        let best_loss = validation.evaluate(&model, &config, device)?;
        ensure!(best_loss.is_finite(), "Validation initiale non finie");
        Progress {
            version: 1,
            step: 0,
            best_step: 0,
            best_loss,
            input_tokens: 0,
            target_tokens: 0,
            signature: progress::signature(&config)?,
            sampling: "splitmix64-std-rng-per-step-v1".into(),
            weights_sha256: String::new(),
        }
    };
    fs::write(
        out.join("data-provenance.json"),
        serde_json::to_vec_pretty(&json!({
            "signature":state.signature,"data":config.data,"data_format":config.data_format,
            "tokenizer_sha256":checkpoint::fingerprint(&config.tokenizer)?,"test_used":false,
            "precision":"FP32","device":format!("{device:?}"),"optimizer_resumed":previous.is_some(),
            "resume_from":previous,"backend_initialization_seeded":device.is_cuda(),"objective":config.objective
        }))?,
    )?;
    if let Some(previous) = previous {
        progress::inherit_best(previous, out, &state)?;
    }
    progress::save(&vars, &optimizer, out, &mut state)?;
    super::training_data::write_io_report(&train, &validation, out)?;
    let mut last_saved = state.step;
    let mut log = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(out.join("metrics.jsonl"))?;
    writeln!(
        log,
        "{}",
        json!({"step":state.step,"best_validation_loss":state.best_loss,"resumed":previous.is_some()})
    )?;
    let parameters = optimizer.variables();
    let initial_tokens = state.input_tokens;
    let mut compute_seconds = 0.0;
    println!(
        "Calcul sur {device:?} | objectif {} | validation {} elements max",
        serde_json::to_string(&config.objective)?,
        config.evaluation_windows
    );
    for step in state.step + 1..=config.steps {
        if stop() || out.join("STOP").exists() || stop_after.is_some_and(|n| step > n) {
            if last_saved != state.step {
                progress::save(&vars, &optimizer, out, &mut state)?;
            }
            fs::write(
                out.join("interrupted.json"),
                serde_json::to_vec_pretty(&json!({"after_step":state.step,"next_step":step}))?,
            )?;
            println!(
                "Arret apres {} etapes ; etat complet sauvegarde.",
                state.step
            );
            super::training_data::write_io_report(&train, &validation, out)?;
            return Ok(false);
        }
        let step_start = Instant::now();
        let rate = schedule::learning_rate(&config, step);
        optimizer.set_learning_rate(rate)?;
        let mut rng = StdRng::seed_from_u64(progress::sampling_seed(config.seed, step));
        let batch = train.batch(&config, &mut rng, device)?;
        state.input_tokens += batch.input.elem_count();
        state.target_tokens += batch.target.elem_count();
        let loss = batch.loss(&model)?;
        let value = loss.to_scalar::<f32>()?;
        ensure!(value.is_finite(), "Perte non finie a l'etape {step}");
        let mut grads = loss.backward()?;
        let grad_norm = config
            .max_grad_norm
            .map(|n| gradients::clip(&mut grads, &parameters, n))
            .transpose()?;
        optimizer.step(&grads)?;
        drop(grads);
        drop(loss);
        device.synchronize()?;
        compute_seconds += step_start.elapsed().as_secs_f64();
        state.step = step;
        if step % config.eval_every == 0 || step == config.steps {
            let score = validation.evaluate(&model, &config, device)?;
            ensure!(score.is_finite(), "Validation non finie");
            let selected = score < state.best_loss;
            if selected {
                state.best_loss = score;
                state.best_step = step;
            }
            progress::save(&vars, &optimizer, out, &mut state)?;
            super::training_data::write_io_report(&train, &validation, out)?;
            last_saved = step;
            let rate_tokens = (state.input_tokens - initial_tokens) as f64 / compute_seconds;
            writeln!(
                log,
                "{}",
                json!({"step":step,"train_loss":value,"validation_loss":score,"selected":selected,
                "best_step":state.best_step,"learning_rate":rate,"gradient_norm_before_clip":grad_norm,
                "tokens_seen":state.input_tokens,"supervised_tokens_seen":state.target_tokens,
                "input_token_count_includes_padding":matches!(config.objective,super::Objective::Dialogue),
                "training_tokens_per_second":rate_tokens,"elapsed_seconds":started.elapsed().as_secs_f64(),"objective":config.objective})
            )?;
            log.flush()?;
            println!(
                "{step}/{} | train {value:.4} | validation {score:.4} | meilleur {:.4} | {rate_tokens:.0} tok/s",
                config.steps, state.best_loss
            );
        }
    }
    println!(
        "Seance terminee : {} | {:.1}s",
        out.display(),
        started.elapsed().as_secs_f64()
    );
    Ok(true)
}
