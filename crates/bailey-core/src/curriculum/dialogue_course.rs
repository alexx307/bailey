use crate::{
    tokenization::{ASSISTANT, EOS, USER},
    training::checkpoint::fingerprint,
};
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{collections::HashSet, fs, path::Path};

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Example {
    pub question: String,
    pub answer: String,
}

pub fn prepare(source: &Path, out: &Path) -> Result<()> {
    ensure!(!out.exists(), "Corpus deja existant : {}", out.display());
    let mut outputs = Vec::new();
    let mut seen = HashSet::new();
    let mut provenance = Vec::new();
    for partition in ["train", "validation", "test"] {
        let mut files = if partition == "train" {
            fs::read_dir(source.join("train"))?
                .map(|entry| Ok(entry?.path()))
                .collect::<Result<Vec<_>>>()?
        } else {
            vec![source.join(format!("{partition}.json"))]
        };
        files.retain(|file| file.extension().and_then(|s| s.to_str()) == Some("json"));
        files.sort();
        let mut text = String::new();
        let mut count = 0;
        for file in files {
            let examples: Vec<Example> = serde_json::from_slice(&fs::read(&file)?)?;
            provenance.push(json!({"file":file,"sha256":fingerprint(&file)?,"partition":partition,"examples":examples.len()}));
            for example in examples {
                validate(&example, &mut seen)?;
                text.push_str(&format!(
                    "{USER}{}{ASSISTANT}{}{EOS}\n",
                    example.question.trim(),
                    example.answer.trim()
                ));
                count += 1;
            }
        }
        ensure!(count > 0, "Partition vide : {partition}");
        println!("{partition} : {count} dialogues, {} octets", text.len());
        outputs.push((partition, text));
    }
    fs::create_dir_all(out.parent().unwrap_or(Path::new(".")))?;
    fs::create_dir(out)?;
    for (partition, text) in outputs {
        fs::write(out.join(format!("{partition}.txt")), text)?;
    }
    fs::write(
        out.join("manifest.json"),
        serde_json::to_vec_pretty(
            &json!({"origin":"Original French teaching examples written and reviewed by the coding assistant","purpose":"limited dialogue learning experiment, not general pretraining","sources":provenance,"split_policy":"explicit files, normalized exact prompt duplicates rejected across all partitions; paraphrase transfer intentionally measured","tokenizer_retrained":false}),
        )?,
    )?;
    Ok(())
}

fn validate(example: &Example, seen: &mut HashSet<String>) -> Result<()> {
    ensure!(
        !example.question.trim().is_empty() && !example.answer.trim().is_empty(),
        "Dialogue vide"
    );
    for marker in [USER, ASSISTANT, EOS] {
        ensure!(
            !example.question.contains(marker) && !example.answer.contains(marker),
            "Marqueur reserve dans le contenu"
        );
    }
    let key = example
        .question
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase();
    ensure!(
        seen.insert(key),
        "Question dupliquee : {}",
        example.question
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_leaking_prompts_and_reserved_tokens() {
        let mut seen = HashSet::new();
        assert!(
            validate(
                &Example {
                    question: "Bonjour à toi".into(),
                    answer: "Salut !".into()
                },
                &mut seen
            )
            .is_ok()
        );
        assert!(
            validate(
                &Example {
                    question: "  bonjour   à toi ".into(),
                    answer: "Autre".into()
                },
                &mut seen
            )
            .is_err()
        );
        assert!(
            validate(
                &Example {
                    question: "Question".into(),
                    answer: EOS.into()
                },
                &mut seen
            )
            .is_err()
        );
    }
}
