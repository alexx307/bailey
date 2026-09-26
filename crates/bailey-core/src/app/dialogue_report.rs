use crate::{
    curriculum::dialogue_course::Example,
    inference,
    tokenization::{self, ASSISTANT, USER},
    training::checkpoint,
};
use anyhow::{Result, ensure};
use candle_core::Device;
use serde_json::json;
use std::{fs, path::Path};

pub fn run(run: &Path, prompts: &Path, out: &Path, tokens: usize, device: &Device) -> Result<()> {
    ensure!(!out.exists(), "Rapport deja existant");
    ensure!(
        prompts.file_stem().and_then(|s| s.to_str()) != Some("test"),
        "Reserver test.json a l'evaluation finale ; utiliser validation.json pour le developpement"
    );
    let samples: Vec<Example> = serde_json::from_slice(&fs::read(prompts)?)?;
    ensure!(
        !samples.is_empty() && samples.len() <= 200,
        "1..200 exemples attendus"
    );
    let (model, config) = checkpoint::load(run, device)?;
    let tokenizer = tokenization::load(&run.join("tokenizer.json"))?;
    let mut results = Vec::new();
    for example in samples {
        let prompt = format!("{USER}{}{ASSISTANT}", example.question);
        let generated = inference::generate_with(
            &model,
            &config.model,
            &tokenizer,
            &prompt,
            tokens,
            &inference::Sampling::default(),
            device,
        )?;
        println!("Toi > {}\nBailey > {}", example.question, generated.text);
        results.push(
            json!({"question":example.question,"reference":example.answer,"generation":generated}),
        );
    }
    fs::write(
        out,
        serde_json::to_vec_pretty(
            &json!({"run":run,"prompts_sha256":checkpoint::fingerprint(prompts)?,"partition":"development","decoding":"greedy without repetition penalty","results":results,"warning":"qualitative diagnostic, not a general competence score"}),
        )?,
    )?;
    Ok(())
}
