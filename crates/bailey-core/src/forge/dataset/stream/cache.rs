use std::collections::{BTreeMap, HashMap};

use serde::Serialize;

pub(super) const PAGE_BYTES: usize = 64 * 1024;
pub(super) const MAX_CACHE_BYTES: usize = 256 * 1024 * 1024;
pub(super) type Key = (usize, u64);

#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct StreamStats {
    pub bytes_read: u64,
    pub validated_bytes: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub cached_bytes: usize,
    pub peak_cached_bytes: usize,
    pub cache_limit_bytes: usize,
    pub open_shards: usize,
}

struct Entry {
    bytes: Vec<u8>,
    stamp: u64,
}

pub(super) struct Cache {
    entries: HashMap<Key, Entry>,
    order: BTreeMap<u64, Key>,
    clock: u64,
    pub stats: StreamStats,
}

impl Cache {
    pub fn new(limit: usize, validated_bytes: u64, open_shards: usize) -> Self {
        Self {
            entries: HashMap::new(),
            order: BTreeMap::new(),
            clock: 0,
            stats: StreamStats {
                cache_limit_bytes: limit,
                validated_bytes,
                open_shards,
                ..Default::default()
            },
        }
    }

    pub fn get(&mut self, key: Key) -> Option<&[u8]> {
        let entry = self.entries.get_mut(&key)?;
        self.order.remove(&entry.stamp);
        self.clock += 1;
        entry.stamp = self.clock;
        self.order.insert(entry.stamp, key);
        self.stats.cache_hits += 1;
        Some(&entry.bytes)
    }

    /// Evict before allocating the incoming page, including partial shard pages.
    pub fn prepare(&mut self, size: usize) {
        self.stats.cache_misses += 1;
        while self.stats.cached_bytes + size > self.stats.cache_limit_bytes {
            let (_, oldest) = self.order.pop_first().expect("cache contains an entry");
            let entry = self.entries.remove(&oldest).expect("LRU entry exists");
            self.stats.cached_bytes -= entry.bytes.len();
        }
    }

    pub fn insert(&mut self, key: Key, bytes: Vec<u8>) -> &[u8] {
        self.clock += 1;
        self.stats.bytes_read += bytes.len() as u64;
        self.stats.cached_bytes += bytes.len();
        self.stats.peak_cached_bytes = self.stats.peak_cached_bytes.max(self.stats.cached_bytes);
        self.order.insert(self.clock, key);
        self.entries.insert(
            key,
            Entry {
                bytes,
                stamp: self.clock,
            },
        );
        &self.entries[&key].bytes
    }
}
