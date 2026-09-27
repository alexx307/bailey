use crate::forge::dataset::stream::StreamingPartition;
use anyhow::{Result, ensure};
use clap::Args;
use rand::{Rng, SeedableRng, rngs::StdRng};
use serde_json::json;
use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::PathBuf,
    time::Instant,
};

#[derive(Args)]
pub struct StreamArgs {
    #[arg(long)]
    pub data: PathBuf,
    #[arg(long, default_value_t = 8)]
    pub cache_mib: usize,
    #[arg(long, default_value_t = 1024)]
    pub sequence: usize,
    #[arg(long, default_value_t = 256)]
    pub windows: usize,
    #[arg(long, default_value_t = 42)]
    pub seed: u64,
    #[arg(long)]
    pub out: PathBuf,
}

pub fn run(args: &StreamArgs) -> Result<()> {
    ensure!(!args.out.exists(), "Rapport deja existant");
    ensure!(
        (1..=256).contains(&args.cache_mib),
        "Cache attendu : 1..256 Mio"
    );
    ensure!(
        (1..=1_000_000).contains(&args.windows),
        "1..1000000 fenetres attendues"
    );
    ensure!(
        (2..=16384).contains(&args.sequence),
        "Sequence attendue : 2..16384"
    );
    let started = Instant::now();
    let reader = StreamingPartition::open(
        &args.data,
        "train",
        &args.data.join("tokenizer.json"),
        u32::MAX as usize,
        args.cache_mib * 1024 * 1024,
    )?;
    ensure!(reader.len() > args.sequence, "Partition trop courte");
    let validation_seconds = started.elapsed().as_secs_f64();
    let started = Instant::now();
    let mut rng = StdRng::seed_from_u64(args.seed);
    let mut checksum = 0u64;
    for _ in 0..args.windows {
        let start = rng.random_range(0..reader.len() - args.sequence);
        for id in reader.read_window(start, args.sequence + 1)? {
            checksum = checksum.wrapping_mul(31).wrapping_add(id as u64);
        }
    }
    let stats = reader.stats();
    let report = json!({"partition":"train","tokens":reader.len(),"fingerprint":reader.fingerprint(),
        "windows":args.windows,"sequence":args.sequence,"seed":args.seed,"checksum":checksum,
        "validation_seconds":validation_seconds,"window_read_seconds":started.elapsed().as_secs_f64(),
        "io":stats,"test_used":false,"model_weights_updated":false,
        "limitations":["Cache statistics count token pages, not all process RAM or operating-system cache", "Index and open handles grow with shard count", "Initial integrity scan reads all train bytes with bounded buffers"]});
    if let Some(parent) = args.out.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    let mut out = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args.out)?;
    out.write_all(&serde_json::to_vec_pretty(&report)?)?;
    println!(
        "{} tokens sur disque ; {} fenetres lues ; rapport {}",
        reader.len(),
        args.windows,
        args.out.display()
    );
    println!("Cache et lectures : {}", serde_json::to_string(&stats)?);
    Ok(())
}
