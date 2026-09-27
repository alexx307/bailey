use anyhow::Result;
use bailey_core::{
    forge::dataset,
    model::CoreConfig,
    tokenization,
    training::{self, DataFormat, Objective, TrainConfig},
};
use candle_core::{Device, safetensors::BufferedSafetensors};
use std::{fs, path::Path};

fn latest_weights(run: &Path) -> Result<BufferedSafetensors> {
    let latest: serde_json::Value = serde_json::from_slice(&fs::read(run.join("latest.json"))?)?;
    let step = latest["step"].as_u64().unwrap();
    assert_eq!(step, 6, "La seance doit atteindre son pas global final");
    Ok(BufferedSafetensors::new(fs::read(run.join(format!(
        "checkpoints/step-{step:08}/weights.safetensors"
    )))?)?)
}

fn assert_same_weights(left: &Path, right: &Path) -> Result<()> {
    let left = latest_weights(left)?;
    let right = latest_weights(right)?;
    assert_eq!(left.tensors().len(), right.tensors().len());
    for (name, _) in left.tensors() {
        let expected = left
            .load(&name, &Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let actual = right
            .load(&name, &Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_eq!(expected, actual, "Poids divergents : {name}");
    }
    Ok(())
}

#[test]
fn streaming_matches_ram_and_resumes_without_reading_test() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let source = temp.path().join("source");
    fs::create_dir(&source)?;
    for (partition, sentence) in [
        (
            "train",
            "Le chat observe le soleil depuis la fenetre de la maison.\n",
        ),
        (
            "validation",
            "Une musicienne prepare sa partition pour jouer demain soir.\n",
        ),
        (
            "test",
            "Des voyageurs traversent plusieurs montagnes avant leur arrivee.\n",
        ),
    ] {
        fs::write(source.join(format!("{partition}.txt")), sentence.repeat(12))?;
    }
    fs::write(
        source.join("manifest.json"),
        r#"{"source":"integration-test","partitions":"independent texts"}"#,
    )?;
    let tokenizer = temp.path().join("tokenizer");
    tokenization::train(&source.join("train.txt"), &tokenizer, 300)?;
    let shards = temp.path().join("shards");
    let manifest = dataset::build(&source, &tokenizer, &shards, 17)?;
    assert!(manifest.partitions.iter().all(|p| p.shards.len() > 2));
    // Ni la lecture en RAM ni le flux d'entrainement ne doivent consulter test.
    fs::remove_file(shards.join(&manifest.partitions[2].shards[0].artifact.file))?;
    fs::write(
        shards.join(&manifest.partitions[2].shards[1].artifact.file),
        b"reserved test shard deliberately corrupted",
    )?;

    let mut model = CoreConfig::tiny(tokenization::load(&tokenizer)?.get_vocab_size(true));
    model.hidden_size = 16;
    model.intermediate_size = 32;
    model.num_layers = 1;
    model.num_attention_heads = 2;
    model.num_kv_heads = 1;
    let mut config = TrainConfig {
        model,
        data: shards.clone(),
        tokenizer,
        steps: 6,
        sequence: 12,
        batch_size: 2,
        learning_rate: 0.003,
        eval_every: 3,
        seed: 42,
        init_from: None,
        warmup_steps: 2,
        min_lr_ratio: 0.1,
        max_grad_norm: Some(1.),
        evaluation_windows: 4,
        objective: Objective::NextToken,
        data_format: DataFormat::Shards,
        shard_cache_mib: 1,
    };
    let initial = temp.path().join("initial");
    assert!(!training::train_until(
        config.clone(),
        &initial,
        &Device::Cpu,
        Some(0)
    )?);

    // Les trois essais commencent avec les memes poids et un nouvel AdamW.
    config.init_from = Some(initial);
    let ram = temp.path().join("ram");
    assert!(training::train_until(
        config.clone(),
        &ram,
        &Device::Cpu,
        None
    )?);
    config.data_format = DataFormat::ShardsStream;
    let stream = temp.path().join("stream");
    assert!(training::train_until(
        config.clone(),
        &stream,
        &Device::Cpu,
        None
    )?);
    assert_same_weights(&ram, &stream)?;

    let partial = temp.path().join("partial");
    assert!(!training::train_until(
        config,
        &partial,
        &Device::Cpu,
        Some(3)
    )?);
    let resumed = temp.path().join("resumed");
    assert!(training::resume(
        &partial,
        &resumed,
        None,
        &Device::Cpu,
        None
    )?);
    assert_same_weights(&stream, &resumed)?;

    // Une alteration de train a taille constante doit etre detectee au redemarrage.
    let shard = shards.join(&manifest.partitions[0].shards[0].artifact.file);
    let mut bytes = fs::read(&shard)?;
    bytes[0] ^= 1;
    fs::write(shard, bytes)?;
    let invalid = temp.path().join("invalid");
    assert!(training::resume(&partial, &invalid, None, &Device::Cpu, None).is_err());
    assert!(
        !invalid.exists(),
        "Aucune nouvelle seance ne doit etre creee"
    );
    Ok(())
}
