use std::fs;
use std::path::{Path, PathBuf};

use anyhow::Result;
use sha2::{Digest, Sha256};

use super::super::{Manifest, Partition, Shard, TokenizerRecord, io};
use super::{StreamingPartition, cache::PAGE_BYTES};

struct Fixture {
    _temp: tempfile::TempDir,
    root: PathBuf,
    tokenizer: PathBuf,
    expected: Vec<u32>,
    manifest: Manifest,
}

fn save_manifest(root: &Path, manifest: &Manifest) -> Result<()> {
    fs::write(
        root.join("forge-manifest.json"),
        serde_json::to_vec(manifest)?,
    )?;
    Ok(())
}

fn fixture(shard_tokens: usize, count: usize) -> Result<Fixture> {
    let temp = tempfile::tempdir()?;
    let source = temp.path().join("train.txt");
    fs::write(
        &source,
        "Bonjour, voici un texte original pour notre tokenizer.",
    )?;
    let tokenizer = temp.path().join("tokens");
    crate::tokenization::train(&source, &tokenizer, 300)?;
    let loaded = crate::tokenization::load(&tokenizer)?;
    let mut vocabulary: Vec<u32> = loaded.get_vocab(true).into_values().collect();
    vocabulary.sort_unstable();
    let expected: Vec<u32> = (0..count)
        .map(|index| vocabulary[index % vocabulary.len()])
        .collect();
    let root = temp.path().join("shards");
    fs::create_dir(&root)?;
    let tokenizer_record = io::write_new(
        &root,
        "tokenizer.json",
        &fs::read(tokenizer.join("tokenizer.json"))?,
    )?;
    let provenance = io::write_new(
        &root,
        "source-manifest.json",
        br#"{"origin":"original unit test"}"#,
    )?;
    let mut partitions = Vec::new();
    for name in ["train", "validation", "test"] {
        fs::create_dir(root.join(name))?;
        let ids = if name == "train" {
            &expected[..]
        } else {
            &expected[..3]
        };
        let mut shards = Vec::new();
        for (index, chunk) in ids.chunks(shard_tokens).enumerate() {
            let bytes: Vec<u8> = chunk.iter().flat_map(|id| id.to_le_bytes()).collect();
            let artifact = io::write_new(&root, &format!("{name}/shard-{index:06}.u32"), &bytes)?;
            shards.push(Shard {
                artifact,
                tokens: chunk.len() as u64,
            });
        }
        partitions.push(Partition {
            name: name.to_owned(),
            source: io::record(&format!("{name}.txt"), name.as_bytes()),
            tokens: ids.len() as u64,
            shards,
        });
    }
    let manifest = Manifest {
        format_version: 1,
        encoding: "u32-le".into(),
        shard_tokens,
        tokenizer: TokenizerRecord {
            artifact: tokenizer_record,
            vocab_size: loaded.get_vocab_size(true),
        },
        provenance,
        partitions,
        duplicate_policy: "synthetic independent fixtures".into(),
        limitations: Vec::new(),
    };
    save_manifest(&root, &manifest)?;
    Ok(Fixture {
        _temp: temp,
        root,
        tokenizer,
        expected,
        manifest,
    })
}

fn open(fixture: &Fixture) -> Result<StreamingPartition> {
    StreamingPartition::open(&fixture.root, "train", &fixture.tokenizer, 300, PAGE_BYTES)
}

#[test]
fn bounded_lru_windows_cross_pages_and_shards_deterministically() -> Result<()> {
    let fixture = fixture(20_000, 40_311)?;
    let reader = open(&fixture)?;
    assert_eq!(reader.len(), fixture.expected.len());
    assert!(!reader.is_empty());
    assert_eq!(reader.stats().bytes_read, 0);
    assert_eq!(reader.stats().cached_bytes, 0);
    assert_eq!(reader.stats().validated_bytes, (reader.len() * 4) as u64);
    let encoded: Vec<u8> = fixture
        .expected
        .iter()
        .flat_map(|id| id.to_le_bytes())
        .collect();
    assert_eq!(
        reader.fingerprint(),
        format!("{:x}", Sha256::digest(encoded))
    );
    for (start, len) in [
        (0, 8),
        (2, 8),
        (16_380, 9),
        (19_990, 20),
        (39_990, 321),
        (0, 8),
    ] {
        assert_eq!(
            reader.read_window(start, len)?,
            fixture.expected[start..start + len]
        );
        assert!(reader.stats().cached_bytes <= PAGE_BYTES);
        assert!(reader.stats().peak_cached_bytes <= PAGE_BYTES);
    }
    assert!(reader.stats().cache_hits > 0);
    assert!(reader.stats().cache_misses >= 5);
    let previous = reader.stats();
    assert_eq!(reader.read_window(0, 8)?, fixture.expected[..8]);
    assert_eq!(reader.stats().bytes_read, previous.bytes_read);
    assert_eq!(reader.stats().cache_hits, previous.cache_hits + 1);
    assert!(reader.read_window(reader.len(), 0)?.is_empty());
    assert!(reader.read_window(reader.len(), 1).is_err());
    assert!(reader.read_window(usize::MAX, 2).is_err());
    assert!(reader.read_window(reader.len() + 1, 0).is_err());
    Ok(())
}

#[test]
fn small_shards_share_the_byte_budget_and_test_is_not_opened() -> Result<()> {
    let fixture = fixture(13, 133)?;
    fs::write(fixture.root.join("test/shard-000000.u32"), b"broken")?;
    let reader = open(&fixture)?;
    assert_eq!(reader.read_window(0, reader.len())?, fixture.expected);
    assert_eq!(reader.stats().bytes_read, (reader.len() * 4) as u64);
    assert_eq!(reader.stats().cache_misses, 11);
    assert!(
        StreamingPartition::open(&fixture.root, "test", &fixture.tokenizer, 300, PAGE_BYTES)
            .is_err()
    );
    assert!(
        StreamingPartition::open(&fixture.root, "absent", &fixture.tokenizer, 300, PAGE_BYTES)
            .is_err()
    );
    assert!(
        StreamingPartition::open(
            &fixture.root,
            "train",
            &fixture.tokenizer,
            300,
            PAGE_BYTES - 1
        )
        .is_err()
    );
    assert!(
        StreamingPartition::open(
            &fixture.root,
            "train",
            &fixture.tokenizer,
            300,
            256 * 1024 * 1024 + 1
        )
        .is_err()
    );
    Ok(())
}

#[test]
fn eviction_keeps_the_most_recently_used_page() -> Result<()> {
    let fixture = fixture(8192, 24576)?;
    let reader = open(&fixture)?;
    reader.read_window(0, 1)?;
    reader.read_window(8192, 1)?;
    reader.read_window(0, 1)?;
    reader.read_window(16384, 1)?;
    let before = reader.stats();
    assert_eq!(before.cached_bytes, PAGE_BYTES);
    reader.read_window(0, 1)?;
    assert_eq!(reader.stats().bytes_read, before.bytes_read);
    reader.read_window(8192, 1)?;
    assert_eq!(reader.stats().bytes_read, before.bytes_read + 32768);
    assert_eq!(reader.stats().peak_cached_bytes, PAGE_BYTES);
    Ok(())
}

#[test]
fn detects_corruption_truncation_ids_tokenizer_and_provenance() -> Result<()> {
    let mut fixture = fixture(13, 29)?;
    let path = fixture.root.join("train/shard-000000.u32");
    let original = fs::read(&path)?;
    let mut corrupt = original.clone();
    corrupt[0..4].copy_from_slice(&fixture.expected[1].to_le_bytes());
    fs::write(&path, &corrupt)?;
    assert!(open(&fixture).is_err());
    fs::write(&path, &original[..original.len() - 1])?;
    assert!(open(&fixture).is_err());
    corrupt[0..4].copy_from_slice(&u32::MAX.to_le_bytes());
    fs::write(&path, &corrupt)?;
    fixture.manifest.partitions[0].shards[0].artifact =
        io::record("train/shard-000000.u32", &corrupt);
    save_manifest(&fixture.root, &fixture.manifest)?;
    assert!(open(&fixture).is_err());
    fs::write(&path, &original)?;
    fixture.manifest.partitions[0].shards[0].artifact =
        io::record("train/shard-000000.u32", &original);
    save_manifest(&fixture.root, &fixture.manifest)?;
    assert!(
        StreamingPartition::open(&fixture.root, "train", &fixture.tokenizer, 1, PAGE_BYTES)
            .is_err()
    );
    let other = fixture._temp.path().join("other-tokenizer.json");
    fs::write(&other, "{}")?;
    assert!(StreamingPartition::open(&fixture.root, "train", &other, 300, PAGE_BYTES).is_err());
    fs::write(fixture.root.join("source-manifest.json"), "{}")?;
    assert!(open(&fixture).is_err());
    Ok(())
}

#[test]
fn live_reader_prevents_or_detects_shard_mutation() -> Result<()> {
    let fixture = fixture(13, 29)?;
    let reader = open(&fixture)?;
    reader.read_window(0, 3)?;
    let path = fixture.root.join("train/shard-000000.u32");
    let mutation = fs::OpenOptions::new()
        .write(true)
        .truncate(true)
        .open(&path);
    #[cfg(windows)]
    assert!(
        mutation.is_err(),
        "Validated Windows handle must deny writes"
    );
    #[cfg(not(windows))]
    {
        drop(mutation?);
        assert!(
            reader.read_window(0, 3).is_err(),
            "Cached pages must not hide file changes"
        );
    }
    drop(reader);
    fs::write(path, b"now allowed")?;
    Ok(())
}
