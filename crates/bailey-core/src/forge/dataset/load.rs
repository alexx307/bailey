use super::{Manifest, io};
use anyhow::{Result, ensure};
use std::{fs, path::Path};

pub fn read_manifest(root: &Path) -> Result<Manifest> {
    let manifest: Manifest =
        serde_json::from_slice(&fs::read(io::contained(root, "forge-manifest.json")?)?)?;
    manifest.validate()?;
    Ok(manifest)
}

pub fn load_partition(
    root: &Path,
    partition: &str,
    tokenizer: &Path,
    vocab: usize,
) -> Result<Vec<u32>> {
    let manifest = read_manifest(root)?;
    let part = manifest
        .partitions
        .iter()
        .find(|p| p.name == partition)
        .ok_or_else(|| anyhow::anyhow!("Partition inconnue"))?;
    let tokenizer_bytes = io::checked_read(root, &manifest.tokenizer.artifact)?;
    ensure!(
        tokenizer_bytes == fs::read(io::tokenizer_file(tokenizer))?,
        "Tokenizer incompatible avec les shards"
    );
    io::checked_read(root, &manifest.provenance)?;
    let tokenizer = crate::tokenization::load(&root.join("tokenizer.json"))?;
    ensure!(
        tokenizer.get_vocab_size(true) == manifest.tokenizer.vocab_size,
        "Taille de vocabulaire incoherente"
    );
    let valid = tokenizer
        .get_vocab(true)
        .into_values()
        .collect::<std::collections::HashSet<_>>();
    let capacity: usize = part.tokens.try_into()?;
    ensure!(
        capacity <= 134_217_728,
        "Prototype : partition limitee a 134217728 tokens en RAM"
    );
    let mut ids = Vec::with_capacity(capacity);
    for shard in &part.shards {
        let bytes = io::checked_read(root, &shard.artifact)?;
        for chunk in bytes.chunks_exact(4) {
            let id = u32::from_le_bytes(chunk.try_into().unwrap());
            ensure!(
                (id as usize) < vocab && valid.contains(&id),
                "ID de token invalide dans le shard"
            );
            ids.push(id);
        }
    }
    ensure!(ids.len() == capacity, "Nombre de tokens incorrect");
    Ok(ids)
}
