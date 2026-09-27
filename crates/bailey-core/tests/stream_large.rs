//! Optional disk-scale proof: generated IDs, not a language training corpus.
use anyhow::Result;
use bailey_core::{
    forge::dataset::{
        self, FileRecord, Manifest, Partition, Shard, TokenizerRecord, stream::StreamingPartition,
    },
    tokenization,
};
use sha2::{Digest, Sha256};
use std::{fs, io::Write, path::Path, time::Instant};

const TOKENS: usize = 134_217_760;
const CACHE_BYTES: usize = 1024 * 1024;
const BUFFER_BYTES: usize = 64 * 1024;

fn small_file(root: &Path, file: &str, bytes: &[u8]) -> Result<FileRecord> {
    fs::write(root.join(file), bytes)?;
    Ok(FileRecord {
        file: file.to_owned(),
        sha256: format!("{:x}", Sha256::digest(bytes)),
        bytes: bytes.len() as u64,
    })
}

fn synthetic_shard(root: &Path, name: &str, count: usize, ids: &[u32]) -> Result<Shard> {
    let file = format!("{name}/shard-000000.u32");
    let mut output = fs::File::create(root.join(&file))?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; BUFFER_BYTES];
    let mut written = 0;
    while written < count {
        let next = (count - written).min(BUFFER_BYTES / 4);
        for (offset, chunk) in buffer[..next * 4].chunks_exact_mut(4).enumerate() {
            chunk.copy_from_slice(&ids[(written + offset) % ids.len()].to_le_bytes());
        }
        output.write_all(&buffer[..next * 4])?;
        digest.update(&buffer[..next * 4]);
        written += next;
    }
    output.sync_all()?;
    Ok(Shard {
        artifact: FileRecord {
            file,
            sha256: format!("{:x}", digest.finalize()),
            bytes: count as u64 * 4,
        },
        tokens: count as u64,
    })
}

#[test]
#[ignore = "writes and scans 512MiB fixture"]
fn bounded_cache_reads_beyond_legacy_ram_limit() -> Result<()> {
    let started = Instant::now();
    let temp = tempfile::tempdir()?;
    let source = temp.path().join("train.txt");
    fs::write(
        &source,
        "Un petit texte original pour apprendre les identifiants du test.",
    )?;
    let tokens = temp.path().join("tokens");
    tokenization::train(&source, &tokens, 300)?;
    let tokenizer = tokenization::load(&tokens)?;
    let mut vocabulary: Vec<u32> = tokenizer.get_vocab(true).into_values().collect();
    vocabulary.sort_unstable();
    assert!(vocabulary.len() >= 32);
    let pattern = &vocabulary[..16];
    let root = temp.path().join("shards");
    fs::create_dir(&root)?;
    let tokenizer_record = small_file(
        &root,
        "tokenizer.json",
        &fs::read(tokens.join("tokenizer.json"))?,
    )?;
    let provenance = small_file(
        &root,
        "source-manifest.json",
        br#"{"origin":"artificial integration fixture","purpose":"bounded reader memory only","language_corpus":false,"generator":"cyclic known token IDs, 64 KiB write buffer"}"#,
    )?;
    let mut partitions = Vec::new();
    for (name, count, ids) in [
        ("train", TOKENS, pattern),
        ("validation", 32, &vocabulary[16..24]),
        ("test", 32, &vocabulary[24..32]),
    ] {
        fs::create_dir(root.join(name))?;
        let description = format!("Artificial {name} ID sequence; not natural-language data.");
        partitions.push(Partition {
            name: name.to_owned(),
            source: small_file(&root, &format!("{name}.txt"), description.as_bytes())?,
            tokens: count as u64,
            shards: vec![synthetic_shard(&root, name, count, ids)?],
        });
    }
    let manifest = Manifest {
        format_version: 1,
        encoding: "u32-le".into(),
        shard_tokens: TOKENS,
        tokenizer: TokenizerRecord {
            artifact: tokenizer_record,
            vocab_size: tokenizer.get_vocab_size(true),
        },
        provenance,
        partitions,
        duplicate_policy: "Artificial disjoint ID patterns; no language claims".into(),
        limitations: vec![
            "Synthetic size proof only; no model training or corpus quality claim".into(),
        ],
    };
    fs::write(
        root.join("forge-manifest.json"),
        serde_json::to_vec(&manifest)?,
    )?;
    dataset::read_manifest(&root)?;
    let generation_seconds = started.elapsed().as_secs_f64();

    let error = dataset::load_partition(&root, "train", &tokens, vocabulary.len())
        .expect_err("The old RAM loader must reject its fixed token limit");
    assert!(error.to_string().contains("134217728"), "{error:#}");
    let validating = Instant::now();
    let reader = StreamingPartition::open(&root, "train", &tokens, vocabulary.len(), CACHE_BYTES)?;
    let validation_seconds = validating.elapsed().as_secs_f64();
    assert_eq!(reader.len(), TOKENS);
    assert_eq!(reader.stats().validated_bytes, TOKENS as u64 * 4);
    assert_eq!(reader.stats().cached_bytes, 0);

    let reading = Instant::now();
    let window = 37;
    // Forty spread-out windows exercise eviction, plus an explicit page boundary.
    for start in (0..40)
        .map(|index| (TOKENS - window) * index / 39)
        .chain([BUFFER_BYTES / 4 - 3, TOKENS / 2])
    {
        let observed = reader.read_window(start, window)?;
        for (offset, id) in observed.into_iter().enumerate() {
            assert_eq!(id, pattern[(start + offset) % pattern.len()]);
        }
        assert!(reader.stats().cached_bytes <= CACHE_BYTES);
        assert!(reader.stats().peak_cached_bytes <= CACHE_BYTES);
    }
    let stats = reader.stats();
    assert!(stats.cache_misses >= 40);
    assert_eq!(stats.peak_cached_bytes, CACHE_BYTES);
    assert_eq!(stats.open_shards, 1);
    println!(
        "Synthetic size proof: {} tokens / {} bytes; generation {:.3}s; integrity scan {:.3}s; 42 windows {:.3}s; stats {}. Cache measures data pages, not total process RAM.",
        TOKENS,
        TOKENS as u64 * 4,
        generation_seconds,
        validation_seconds,
        reading.elapsed().as_secs_f64(),
        serde_json::to_string(&stats)?
    );
    drop(reader);
    temp.close()?;
    Ok(())
}
