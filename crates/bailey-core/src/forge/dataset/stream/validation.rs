use std::collections::HashSet;
use std::fs::{File, Metadata, OpenOptions};
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::time::SystemTime;

use anyhow::{Context, Result, ensure};
use sha2::{Digest, Sha256};

use super::super::{FileRecord, Manifest, Shard, io};
use super::cache::PAGE_BYTES;

pub(super) struct OpenShard {
    pub file: File,
    pub start: u64,
    pub tokens: u64,
    pub bytes: u64,
    modified: Option<SystemTime>,
    name: String,
}

fn read_only(path: &Path) -> Result<File> {
    let mut options = OpenOptions::new();
    options.read(true);
    // Keep the validated file identity and refuse writers/replacements on Windows.
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        options.share_mode(1); // FILE_SHARE_READ
    }
    options
        .open(path)
        .with_context(|| format!("Ouverture immuable : {}", path.display()))
}

fn unchanged(metadata: &Metadata, bytes: u64, modified: Option<SystemTime>) -> bool {
    metadata.len() == bytes && metadata.modified().ok() == modified
}

fn scan(
    file: &mut File,
    record: &FileRecord,
    ids: Option<(&HashSet<u32>, usize)>,
    mut partition_digest: Option<&mut Sha256>,
) -> Result<Option<SystemTime>> {
    let before = file.metadata()?;
    ensure!(
        before.len() == record.bytes,
        "Taille de shard/fichier altérée : {}",
        record.file
    );
    let mut digest = Sha256::new();
    let mut consumed = 0_u64;
    let mut buffer = [0_u8; PAGE_BYTES];
    while consumed < record.bytes {
        let count = (record.bytes - consumed).min(PAGE_BYTES as u64) as usize;
        file.read_exact(&mut buffer[..count])?;
        digest.update(&buffer[..count]);
        if let Some(partition_digest) = partition_digest.as_deref_mut() {
            partition_digest.update(&buffer[..count]);
        }
        if let Some((valid, vocab)) = ids {
            ensure!(count.is_multiple_of(4), "Shard non aligné sur des IDs u32");
            for chunk in buffer[..count].chunks_exact(4) {
                let id = u32::from_le_bytes(chunk.try_into().unwrap());
                ensure!(
                    (id as usize) < vocab && valid.contains(&id),
                    "ID de token invalide : {}",
                    record.file
                );
            }
        }
        consumed += count as u64;
    }
    ensure!(
        file.read(&mut buffer[..1])? == 0,
        "Fichier agrandi pendant validation"
    );
    ensure!(
        unchanged(&file.metadata()?, record.bytes, before.modified().ok()),
        "Fichier modifié pendant validation"
    );
    ensure!(
        format!("{:x}", digest.finalize()) == record.sha256,
        "Empreinte incorrecte : {}",
        record.file
    );
    file.seek(SeekFrom::Start(0))?;
    Ok(before.modified().ok())
}

pub(super) fn validate_artifacts(
    root: &Path,
    manifest: &Manifest,
    tokenizer: &Path,
) -> Result<HashSet<u32>> {
    let tokenizer_bytes = io::checked_read(root, &manifest.tokenizer.artifact)?;
    ensure!(
        tokenizer_bytes == std::fs::read(io::tokenizer_file(tokenizer))?,
        "Tokenizer incompatible avec les shards"
    );
    // Load precisely the verified bytes, rather than reopening a mutable path.
    let loaded = crate::tokenization::Tokenizer::from_bytes(&tokenizer_bytes)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    ensure!(
        loaded.get_vocab_size(true) == manifest.tokenizer.vocab_size,
        "Taille de vocabulaire incohérente"
    );
    for marker in [
        crate::tokenization::USER,
        crate::tokenization::ASSISTANT,
        crate::tokenization::EOS,
    ] {
        ensure!(
            loaded.token_to_id(marker).is_some(),
            "Marqueur requis absent du tokenizer"
        );
    }
    let mut provenance = read_only(&io::contained(root, &manifest.provenance.file)?)?;
    scan(&mut provenance, &manifest.provenance, None, None)?;
    Ok(loaded.get_vocab(true).into_values().collect())
}

impl OpenShard {
    pub fn open(
        root: &Path,
        shard: &Shard,
        start: u64,
        valid: &HashSet<u32>,
        vocab: usize,
        partition_digest: &mut Sha256,
    ) -> Result<Self> {
        let path = io::contained(root, &shard.artifact.file)?;
        let mut file = read_only(&path)?;
        let modified = scan(
            &mut file,
            &shard.artifact,
            Some((valid, vocab)),
            Some(partition_digest),
        )?;
        Ok(Self {
            file,
            start,
            tokens: shard.tokens,
            bytes: shard.artifact.bytes,
            modified,
            name: shard.artifact.file.clone(),
        })
    }

    pub fn check_unchanged(&self) -> Result<()> {
        ensure!(
            unchanged(&self.file.metadata()?, self.bytes, self.modified),
            "Shard modifié après validation : {}",
            self.name
        );
        Ok(())
    }
}
