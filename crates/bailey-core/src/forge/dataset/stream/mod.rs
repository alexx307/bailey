//! Random windows from verified shards, with bounded data pages in memory.
//!
//! Opening scans only the requested partition once using a 64 KiB buffer. The
//! index and open handles scale with shard count, never with token count. Windows
//! handles deny concurrent writes/deletion; other platforms detect metadata
//! changes but require callers to preserve immutability after validation.

mod cache;
mod validation;

use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::Mutex;

use anyhow::{Result, ensure};
use sha2::{Digest, Sha256};

pub use cache::StreamStats;
use cache::{Cache, MAX_CACHE_BYTES, PAGE_BYTES};
use validation::OpenShard;

struct State {
    shards: Vec<OpenShard>,
    cache: Cache,
}

pub struct StreamingPartition {
    tokens: usize,
    fingerprint: String,
    state: Mutex<State>,
}

impl StreamingPartition {
    pub fn open(
        root: &Path,
        partition: &str,
        tokenizer: &Path,
        vocab: usize,
        cache_bytes: usize,
    ) -> Result<Self> {
        ensure!(
            (PAGE_BYTES..=MAX_CACHE_BYTES).contains(&cache_bytes),
            "Cache requis : entre 64 Kio et 256 Mio"
        );
        let manifest = super::read_manifest(root)?;
        let part = manifest
            .partitions
            .iter()
            .find(|part| part.name == partition)
            .ok_or_else(|| anyhow::anyhow!("Partition inconnue"))?;
        let valid = validation::validate_artifacts(root, &manifest, tokenizer)?;
        let mut shards = Vec::with_capacity(part.shards.len());
        let mut start = 0_u64;
        let mut digest = Sha256::new();
        for shard in &part.shards {
            shards.push(OpenShard::open(
                root,
                shard,
                start,
                &valid,
                vocab,
                &mut digest,
            )?);
            start = start
                .checked_add(shard.tokens)
                .ok_or_else(|| anyhow::anyhow!("Index de tokens hors limites"))?;
        }
        ensure!(start == part.tokens, "Total de tokens incorrect");
        let validated_bytes = start
            .checked_mul(4)
            .ok_or_else(|| anyhow::anyhow!("Taille de partition hors limites"))?;
        let cache = Cache::new(cache_bytes, validated_bytes, shards.len());
        Ok(Self {
            tokens: start.try_into()?,
            fingerprint: format!("{:x}", digest.finalize()),
            state: Mutex::new(State { shards, cache }),
        })
    }

    pub fn len(&self) -> usize {
        self.tokens
    }

    pub fn is_empty(&self) -> bool {
        self.tokens == 0
    }

    pub fn fingerprint(&self) -> &str {
        &self.fingerprint
    }

    /// Allocation equals the requested window, in addition to the bounded cache.
    pub fn read_window(&self, start: usize, len: usize) -> Result<Vec<u32>> {
        let end = start
            .checked_add(len)
            .ok_or_else(|| anyhow::anyhow!("Fenêtre hors limites"))?;
        ensure!(end <= self.tokens, "Fenêtre hors de la partition");
        if len == 0 {
            return Ok(Vec::new());
        }
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("Lecteur de shards empoisonné"))?;
        let State { shards, cache } = &mut *state;
        let mut output = Vec::new();
        output.try_reserve_exact(len)?;
        let mut cursor = start as u64;
        let mut index = shards.partition_point(|shard| shard.start <= cursor) - 1;
        while cursor < end as u64 {
            let shard = &mut shards[index];
            shard.check_unchanged()?;
            let byte_offset = (cursor - shard.start) * 4;
            let page = byte_offset / PAGE_BYTES as u64;
            let within = (byte_offset % PAGE_BYTES as u64) as usize;
            let page_start = page * PAGE_BYTES as u64;
            let size = (shard.bytes - page_start).min(PAGE_BYTES as u64) as usize;
            let key = (index, page);
            let bytes = if let Some(bytes) = cache.get(key) {
                bytes
            } else {
                cache.prepare(size);
                let mut bytes = vec![0_u8; size];
                shard.file.seek(SeekFrom::Start(page_start))?;
                shard.file.read_exact(&mut bytes)?;
                shard.check_unchanged()?;
                cache.insert(key, bytes)
            };
            let count = ((bytes.len() - within) / 4).min(end - cursor as usize);
            for chunk in bytes[within..within + count * 4].chunks_exact(4) {
                output.push(u32::from_le_bytes(chunk.try_into().unwrap()));
            }
            cursor += count as u64;
            if cursor == shard.start + shard.tokens {
                index += 1;
            }
        }
        Ok(output)
    }

    pub fn stats(&self) -> StreamStats {
        self.state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .cache
            .stats
    }
}

#[cfg(test)]
mod tests;
