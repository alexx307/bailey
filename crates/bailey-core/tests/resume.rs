use anyhow::Result;
use bailey_core::{
    model::CoreConfig,
    tokenization,
    training::{self, DataFormat, Objective, TrainConfig},
};
use candle_core::{Device, safetensors::BufferedSafetensors};
use std::{fs, path::Path};

fn latest_weights(run: &Path) -> Result<BufferedSafetensors> {
    let state: serde_json::Value = serde_json::from_slice(&fs::read(run.join("latest.json"))?)?;
    let step = state["step"].as_u64().unwrap();
    Ok(BufferedSafetensors::new(fs::read(run.join(format!(
        "checkpoints/step-{step:08}/weights.safetensors"
    )))?)?)
}

#[test]
fn interrupted_training_matches_uninterrupted_and_rejects_changed_data() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let data = temp.path().join("data");
    fs::create_dir(&data)?;
    fs::write(
        data.join("train.txt"),
        "Le chat dort dans la maison. Le soleil se leve.\n".repeat(30),
    )?;
    fs::write(
        data.join("validation.txt"),
        "Un chien regarde le jardin. La lune est ronde.\n".repeat(10),
    )?;
    let tokens = temp.path().join("tokens");
    tokenization::train(&data.join("train.txt"), &tokens, 300)?;
    let mut model = CoreConfig::tiny(tokenization::load(&tokens)?.get_vocab_size(true));
    model.hidden_size = 16;
    model.intermediate_size = 32;
    model.num_layers = 1;
    model.num_attention_heads = 2;
    model.num_kv_heads = 1;
    let config = TrainConfig {
        model,
        data: data.clone(),
        tokenizer: tokens,
        steps: 8,
        sequence: 12,
        batch_size: 2,
        learning_rate: 0.003,
        eval_every: 4,
        seed: 42,
        init_from: None,
        warmup_steps: 2,
        min_lr_ratio: 0.1,
        max_grad_norm: Some(1.),
        evaluation_windows: 4,
        objective: Objective::NextToken,
        data_format: DataFormat::Text,
    };
    let initial = temp.path().join("initial");
    assert!(!training::train_until(
        config,
        &initial,
        &Device::Cpu,
        Some(0)
    )?);
    let full = temp.path().join("full");
    assert!(training::resume(&initial, &full, None, &Device::Cpu, None)?);
    let part = temp.path().join("part");
    assert!(!training::resume(
        &initial,
        &part,
        None,
        &Device::Cpu,
        Some(3)
    )?);
    let continued = temp.path().join("continued");
    // Une interruption entre publication best/latest ne doit pas casser la reprise.
    fs::write(
        part.join("best.json"),
        r#"{"step":999,"validation_loss":0.0}"#,
    )?;
    assert!(training::resume(
        &part,
        &continued,
        None,
        &Device::Cpu,
        None
    )?);
    let a = latest_weights(&full)?;
    let b = latest_weights(&continued)?;
    for (name, _) in a.tensors() {
        let left = a
            .load(&name, &Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let right = b
            .load(&name, &Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_eq!(left, right, "Poids divergents : {name}");
    }
    fs::write(data.join("train.txt"), "Autre texte. ".repeat(100))?;
    assert!(
        training::resume(
            &part,
            &temp.path().join("invalid"),
            None,
            &Device::Cpu,
            None
        )
        .is_err()
    );
    assert!(!temp.path().join("invalid").exists());
    Ok(())
}
