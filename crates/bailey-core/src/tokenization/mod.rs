//! Un tokenizer appris uniquement sur un corpus local d'entraînement.

mod training;

use std::path::Path;

use anyhow::{Context, Result, anyhow, ensure};
pub use tokenizers::Tokenizer;

pub use training::train;

pub const USER: &str = "<|user|>";
pub const ASSISTANT: &str = "<|assistant|>";
pub const EOS: &str = "<|end|>";
pub const DEFAULT_VOCAB_SIZE: usize = 32_000;

/// Accepte le dossier produit par `train` ou son fichier `tokenizer.json`.
pub fn load(path: &Path) -> Result<Tokenizer> {
    let file = if path.is_dir() {
        path.join("tokenizer.json")
    } else {
        path.to_owned()
    };
    let tokenizer = Tokenizer::from_file(&file)
        .map_err(|error| anyhow!(error.to_string()))
        .with_context(|| format!("Lecture du tokenizer local {}", file.display()))?;
    for token in [USER, ASSISTANT, EOS] {
        ensure!(
            tokenizer.token_to_id(token).is_some(),
            "Le tokenizer ne contient pas le marqueur requis {token}"
        );
    }
    Ok(tokenizer)
}

#[cfg(test)]
mod tests;
