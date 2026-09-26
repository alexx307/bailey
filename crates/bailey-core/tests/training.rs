use anyhow::Result;
use bailey_core::{
    model::CoreConfig,
    tokenization,
    training::{self, TrainConfig, checkpoint, evaluation},
};
use candle_core::Device;
use std::fs;

#[test]
fn training_saves_reloads_and_continues_without_changing_the_previous_run() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let data = temp.path().join("data");
    fs::create_dir(&data)?;
    fs::write(
        data.join("train.txt"),
        "Bonjour. Le chat dort. Le soleil brille.\n".repeat(50),
    )?;
    fs::write(
        data.join("validation.txt"),
        "Bonjour. Le chat joue. Le soleil brille.\n".repeat(10),
    )?;
    let tokens = temp.path().join("tokenizer");
    tokenization::train(&data.join("train.txt"), &tokens, 300)?;
    let tokenizer = tokenization::load(&tokens)?;
    let mut model = CoreConfig::tiny(tokenizer.get_vocab_size(true));
    model.hidden_size = 16;
    model.intermediate_size = 32;
    model.num_layers = 1;
    model.num_attention_heads = 2;
    model.num_kv_heads = 1;
    let config = TrainConfig {
        model,
        data,
        tokenizer: tokens.join("tokenizer.json"),
        steps: 40,
        sequence: 16,
        batch_size: 2,
        learning_rate: 0.005,
        eval_every: 20,
        seed: 42,
        init_from: None,
        warmup_steps: 4,
        min_lr_ratio: 0.2,
        max_grad_norm: Some(1.0),
        evaluation_windows: 8,
        objective: training::Objective::NextToken,
        data_format: training::DataFormat::Text,
    };
    let run = temp.path().join("first");
    training::train(config.clone(), &run, &Device::Cpu)?;
    let original = fs::read(run.join("metrics.jsonl"))?;
    let selected: checkpoint::Selection =
        serde_json::from_slice(&fs::read(run.join("best.json"))?)?;
    assert!(selected.step > 0);
    let (loaded, settings) = checkpoint::load(&run, &Device::Cpu)?;
    let value = evaluation::evaluate_file(
        &loaded,
        &settings.data.join("validation.txt"),
        &tokenizer,
        &settings,
        &Device::Cpu,
    )?;
    assert!((value - selected.validation_loss).abs() < 1e-5);
    assert!(training::train(config.clone(), &run, &Device::Cpu).is_err());
    let mut resumed = config;
    resumed.init_from = Some(run.clone());
    resumed.steps = 1;
    resumed.warmup_steps = 0;
    training::train(resumed.clone(), &temp.path().join("next"), &Device::Cpu)?;
    assert_eq!(fs::read(run.join("metrics.jsonl"))?, original);
    let stopped = temp.path().join("stopped");
    assert!(!training::train_controlled(
        resumed.clone(),
        &stopped,
        &Device::Cpu,
        &|| true
    )?);
    assert!(!stopped.exists());
    fs::write(&resumed.tokenizer, "changed tokenizer")?;
    assert!(training::train(resumed, &temp.path().join("invalid"), &Device::Cpu).is_err());
    assert!(!temp.path().join("invalid").exists());
    Ok(())
}
