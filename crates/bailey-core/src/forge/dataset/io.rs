use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use sha2::{Digest, Sha256};

use super::FileRecord;

pub(super) fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

pub(super) fn record(file: &str, bytes: &[u8]) -> FileRecord {
    FileRecord { file: file.to_owned(), sha256: hash(bytes), bytes: bytes.len() as u64 }
}

pub(super) fn contained(root: &Path, relative: &str) -> Result<PathBuf> {
    let root = root.canonicalize()?;
    ensure!(root.is_dir(), "Le corpus doit être un dossier");
    let path = root.join(relative).canonicalize()
        .with_context(|| format!("Fichier manquant : {relative}"))?;
    ensure!(path.starts_with(&root) && path.is_file(), "Fichier hors du corpus : {relative}");
    Ok(path)
}

pub(super) fn checked_read(root: &Path, record: &FileRecord) -> Result<Vec<u8>> {
    let path = contained(root, &record.file)?;
    ensure!(fs::metadata(&path)?.len() == record.bytes, "Fichier tronqué ou taille altérée : {}", record.file);
    let bytes = fs::read(path)?;
    ensure!(bytes.len() as u64 == record.bytes && hash(&bytes) == record.sha256,
        "Empreinte incorrecte : {}", record.file);
    Ok(bytes)
}

pub(super) fn write_new(root: &Path, file: &str, bytes: &[u8]) -> Result<FileRecord> {
    let path = root.join(file);
    let mut out = OpenOptions::new().write(true).create_new(true).open(&path)?;
    out.write_all(bytes)?;
    out.sync_all()?;
    Ok(record(file, bytes))
}

pub(super) fn tokenizer_file(path: &Path) -> PathBuf {
    if path.is_dir() { path.join("tokenizer.json") } else { path.to_owned() }
}
