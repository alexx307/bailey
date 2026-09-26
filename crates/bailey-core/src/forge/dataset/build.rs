use super::{Manifest, PARTITIONS, Partition, Shard, TokenizerRecord, duplicate, io};
use anyhow::{Result, ensure};
use std::{fs, path::Path};

pub fn build(data: &Path, tokenizer: &Path, out: &Path, shard_tokens: usize) -> Result<Manifest> {
    ensure!(!out.exists(), "Dossier de shards deja existant");
    ensure!(
        (1..=16_000_000).contains(&shard_tokens),
        "Taille de shard attendue : 1..16000000"
    );
    let tokenizer_path = io::tokenizer_file(tokenizer);
    let tokenizer_bytes = fs::read(&tokenizer_path)?;
    let tokenizer = crate::tokenization::load(&tokenizer_path)?;
    let provenance = fs::read(data.join("manifest.json"))?;
    let _: serde_json::Value = serde_json::from_slice(&provenance)?;
    let mut detector = duplicate::Detector::default();
    let mut sources = Vec::new();
    for name in PARTITIONS {
        let path = data.join(format!("{name}.txt"));
        ensure!(
            fs::metadata(&path)?.len() <= 512 * 1024 * 1024,
            "Prototype : fichier source limite a 512 Mio ; utiliser des lots distincts"
        );
        let text = fs::read_to_string(path)?;
        ensure!(!text.trim().is_empty(), "Partition vide : {name}");
        detector.check(name, &text)?;
        sources.push(text);
    }
    if let Some(parent) = out.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    fs::create_dir(out)?;
    let mut manifest = Manifest {
        format_version:1, encoding:"u32-le".into(), shard_tokens,
        tokenizer:TokenizerRecord {artifact:io::write_new(out,"tokenizer.json",&tokenizer_bytes)?, vocab_size:tokenizer.get_vocab_size(true)},
        provenance:io::write_new(out,"source-manifest.json",&provenance)?,
        partitions:Vec::new(), duplicate_policy:duplicate::POLICY.into(),
        limitations:vec!["Prototype: source files capped at 512 MiB each; sources and one encoded partition held in RAM; loader holds requested partition in RAM".into(), "Exact fragment checks do not detect all near duplicates, translations or factual errors".into()],
    };
    for (name, text) in PARTITIONS.into_iter().zip(sources) {
        fs::create_dir(out.join(name))?;
        let encoded = tokenizer
            .encode(text.as_str(), false)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        let ids = encoded.get_ids();
        ensure!(!ids.is_empty(), "Aucun token : {name}");
        let mut partition = Partition {
            name: name.into(),
            source: io::record(&format!("{name}.txt"), text.as_bytes()),
            tokens: ids.len() as u64,
            shards: Vec::new(),
        };
        for (index, slice) in ids.chunks(shard_tokens).enumerate() {
            let bytes: Vec<u8> = slice.iter().flat_map(|id| id.to_le_bytes()).collect();
            partition.shards.push(Shard {
                artifact: io::write_new(out, &super::manifest::shard_name(name, index), &bytes)?,
                tokens: slice.len() as u64,
            });
        }
        manifest.partitions.push(partition);
    }
    manifest.validate()?;
    io::write_new(
        out,
        "forge-manifest.json",
        &serde_json::to_vec_pretty(&manifest)?,
    )?;
    Ok(manifest)
}
