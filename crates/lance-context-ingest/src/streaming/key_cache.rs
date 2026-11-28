//! Applied-key cache: 128-bit identities of turns already durably committed or
//! Applied, so the producer can drop resent turns before they reach the buffer.
//!
//! Key = xxh3-128 of the shard's scope digest followed by the length-prefixed
//! primary-key values (for the generic Trace spec: `session_id`, `id`). The
//! scope is the exact table identity (table id, incarnation, relation, shard
//! bucket, spec fingerprint); it is stored in the disk tier and a DB whose
//! stored scope differs is wiped on open, so keys never cross identities. Tier 1 is a fixed-capacity RAM map with
//! FIFO eviction; tier 2 (feature `applied-key-cache`) is RocksDB on the PVC
//! with a bloom filter and a fixed LRU block cache. Writes go through to both
//! tiers; lookups check RAM, then RocksDB.
//!
//! Insertion policy (never at send):
//! - commit ACK: never inserted (a committed record can still be Rejected),
//! - Applied at a fence: never inserted directly. The lane's keys are staged
//!   with the Applied WAL index and enter the skip set only through `prove`,
//!   once a selected full body of the same scope covers that index, i.e. the
//!   keys are durable in an installed/selected manifest,
//! - warm: background scan of the canonical Turns table's key columns.
//! Owner dedup hits are not inserted: the Applied reply does not carry them
//! and fetching them is not cheap.
//!
//! A hit is only an optimization: the owner remains authoritative for
//! first-create, so a false hit would drop a turn. The 128-bit hash makes that
//! negligible; bloom false positives only cost a RocksDB read.
use std::{
    collections::{HashMap, VecDeque},
    path::PathBuf,
    sync::{
        atomic::{AtomicU64, Ordering::Relaxed},
        Mutex,
    },
};

use anyhow::{Context, Result};
use arrow_array::{RecordBatch, StringArray};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

pub type Key = u128;

/// Exact table identity of one cached shard. Keys of different scopes never
/// alias: the scope digest is mixed into every key hash.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Scope {
    pub table: Uuid,
    pub incarnation: Uuid,
    pub relation: String,
    pub bucket: u32,
    pub spec_fingerprint: [u8; 32],
}

impl Scope {
    pub fn new(
        relation: &str,
        bucket: u32,
        binding: &crate::TableBinding,
        spec: &crate::turn_ingestion::TurnIngestionSpec,
    ) -> Result<Self> {
        anyhow::ensure!(
            !binding.table.is_nil() && !binding.incarnation.is_nil() && !relation.is_empty(),
            "applied-key scope requires a bound table identity"
        );
        Ok(Self {
            table: binding.table,
            incarnation: binding.incarnation,
            relation: relation.to_owned(),
            bucket,
            spec_fingerprint: spec.fingerprint()?,
        })
    }

    fn digest(&self) -> Result<[u8; 16]> {
        let bytes = serde_json::to_vec(self)?;
        Ok(xxhash_rust::xxh3::xxh3_128(&bytes).to_le_bytes())
    }
}

/// Key of one row's primary-key values within a scope digest.
pub fn key(scope: &[u8; 16], values: &[&str]) -> Key {
    let mut hasher = xxhash_rust::xxh3::Xxh3::new();
    hasher.update(scope);
    // Length-prefix each value: separator joins are ambiguous
    // (["a\0b","c"] vs ["a","b\0c"]).
    hasher.update(&(values.len() as u64).to_le_bytes());
    for value in values {
        hasher.update(&(value.len() as u64).to_le_bytes());
        hasher.update(value.as_bytes());
    }
    hasher.digest128()
}

/// Keys of every row, from the named UTF8 key columns in order.
fn batch_keys(scope: &[u8; 16], batch: &RecordBatch, columns: &[String]) -> Result<Vec<Key>> {
    let arrays = columns
        .iter()
        .map(|name| {
            batch
                .column_by_name(name)
                .with_context(|| format!("key column {name} absent"))?
                .as_any()
                .downcast_ref::<StringArray>()
                .with_context(|| format!("key column {name} is not UTF8"))
        })
        .collect::<Result<Vec<_>>>()?;
    let mut values = Vec::with_capacity(arrays.len());
    Ok((0..batch.num_rows())
        .map(|row| {
            values.clear();
            values.extend(arrays.iter().map(|array| array.value(row)));
            key(scope, &values)
        })
        .collect())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reason {
    Commit,
    Applied,
    Proven,
    Warm,
    OwnerDedup,
}

pub struct Config {
    /// Exact identity of every cached shard, by bucket. Required.
    pub scopes: Vec<Scope>,
    /// RAM tier capacity in keys (~48 bytes each with map overhead).
    pub ram_keys: usize,
    /// RocksDB directory on the PVC; `None` keeps the cache RAM-only.
    pub disk_dir: Option<PathBuf>,
    /// RocksDB LRU block cache bytes.
    pub block_cache_bytes: usize,
    pub bloom_bits_per_key: f64,
}

#[derive(Default)]
struct Ram {
    map: HashMap<Key, ()>,
    order: VecDeque<Key>,
}

pub struct AppliedKeyCache {
    capacity: usize,
    /// Applied keys per shard awaiting a selected-body proof, in WAL order.
    staged: Mutex<std::collections::BTreeMap<u32, VecDeque<(u64, Vec<Key>)>>>,
    scopes: std::collections::BTreeMap<u32, (Scope, [u8; 16])>,
    ram: Mutex<Ram>,
    #[cfg(feature = "applied-key-cache")]
    disk: Option<rocksdb::DB>,
    metrics: CacheMetrics,
}

#[derive(Default)]
struct CacheMetrics {
    insert_commit: AtomicU64,
    insert_applied: AtomicU64,
    insert_proven: AtomicU64,
    staged_keys: AtomicU64,
    unproven_dropped: AtomicU64,
    insert_warm: AtomicU64,
    insert_owner_dedup: AtomicU64,
    removed_rejected: AtomicU64,
    ram_hits: AtomicU64,
    disk_hits: AtomicU64,
    misses: AtomicU64,
    evictions: AtomicU64,
    warm_rows: AtomicU64,
    warm_bytes: AtomicU64,
    warm_fragments: AtomicU64,
    warm_complete_shards: AtomicU64,
    warm: Mutex<std::collections::BTreeMap<u32, WarmProgress>>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct WarmProgress {
    /// Fragment ids of this shard's Turns dataset fully loaded, ascending.
    pub warmed_through_fragment: Option<u64>,
    pub fragments_total: u64,
    pub fragments_done: u64,
    pub rows: u64,
    pub complete: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct CacheSnapshot {
    pub insert_commit: u64,
    pub insert_applied: u64,
    /// Keys inserted after a selected-body proof covered their Applied index.
    pub insert_proven: u64,
    /// Applied keys currently awaiting a selected-body proof.
    pub staged_keys: u64,
    /// Applied keys dropped unproven (staging bound exceeded).
    pub unproven_dropped: u64,
    pub insert_warm: u64,
    pub insert_owner_dedup: u64,
    pub removed_rejected: u64,
    pub ram_hits: u64,
    pub disk_hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub ram_keys: u64,
    pub disk_enabled: bool,
    pub warm_rows: u64,
    pub warm_bytes: u64,
    pub warm_fragments: u64,
    pub warm_complete_shards: u64,
    pub warm: std::collections::BTreeMap<u32, WarmProgress>,
}

#[cfg(feature = "applied-key-cache")]
const WARM_PREFIX: &[u8] = b"\xffwarm/";
#[cfg(feature = "applied-key-cache")]
const SCOPE_KEY: &[u8] = b"\xffscope";

impl AppliedKeyCache {
    pub fn open(config: Config) -> Result<Self> {
        anyhow::ensure!(config.ram_keys > 0, "empty applied-key RAM tier");
        anyhow::ensure!(!config.scopes.is_empty(), "applied-key cache has no scope");
        let mut scopes = std::collections::BTreeMap::new();
        for scope in &config.scopes {
            anyhow::ensure!(
                scopes
                    .insert(scope.bucket, (scope.clone(), scope.digest()?))
                    .is_none(),
                "duplicate applied-key scope for shard {}",
                scope.bucket
            );
        }
        #[cfg(feature = "applied-key-cache")]
        let disk = match &config.disk_dir {
            Some(dir) => Some(open_disk(
                dir,
                &config,
                &serde_json::to_vec(&config.scopes)?,
            )?),
            None => None,
        };
        #[cfg(not(feature = "applied-key-cache"))]
        anyhow::ensure!(
            config.disk_dir.is_none(),
            "applied-key disk tier requires feature applied-key-cache"
        );
        Ok(Self {
            capacity: config.ram_keys,
            staged: Mutex::default(),
            scopes,
            ram: Mutex::default(),
            #[cfg(feature = "applied-key-cache")]
            disk,
            metrics: CacheMetrics::default(),
        })
    }

    fn scope(&self, bucket: u32) -> Result<&(Scope, [u8; 16])> {
        self.scopes
            .get(&bucket)
            .with_context(|| format!("applied-key cache has no scope for shard {bucket}"))
    }

    /// Scoped keys of every row of a batch routed to `bucket`.
    pub fn batch_keys(
        &self,
        bucket: u32,
        batch: &RecordBatch,
        columns: &[String],
    ) -> Result<Vec<Key>> {
        batch_keys(&self.scope(bucket)?.1, batch, columns)
    }

    /// Refuse a producer shard whose binding or spec is not this cache's scope.
    pub fn check_binding(
        &self,
        bucket: u32,
        binding: &crate::TableBinding,
        spec: &crate::turn_ingestion::TurnIngestionSpec,
    ) -> Result<()> {
        let (scope, _) = self.scope(bucket)?;
        anyhow::ensure!(
            scope.table == binding.table
                && scope.incarnation == binding.incarnation
                && scope.spec_fingerprint == spec.fingerprint()?,
            "applied-key cache scope differs from shard {bucket} binding"
        );
        Ok(())
    }

    pub fn insert(&self, keys: &[Key], reason: Reason) -> Result<()> {
        if keys.is_empty() {
            return Ok(());
        }
        let counter = match reason {
            Reason::Commit => &self.metrics.insert_commit,
            Reason::Applied => &self.metrics.insert_applied,
            Reason::Proven => &self.metrics.insert_proven,
            Reason::Warm => &self.metrics.insert_warm,
            Reason::OwnerDedup => &self.metrics.insert_owner_dedup,
        };
        counter.fetch_add(keys.len() as u64, Relaxed);
        {
            let mut ram = self.ram.lock().unwrap();
            for &key in keys {
                if ram.map.insert(key, ()).is_none() {
                    ram.order.push_back(key);
                }
            }
            let mut evicted = 0;
            while ram.map.len() > self.capacity {
                let Some(old) = ram.order.pop_front() else {
                    break;
                };
                if ram.map.remove(&old).is_some() {
                    evicted += 1;
                }
            }
            // Removed keys leave stale order slots; compact occasionally.
            if ram.order.len() > self.capacity.saturating_mul(2) {
                let Ram { map, order } = &mut *ram;
                order.retain(|key| map.contains_key(key));
            }
            self.metrics.evictions.fetch_add(evicted, Relaxed);
        }
        #[cfg(feature = "applied-key-cache")]
        if let Some(db) = &self.disk {
            let mut batch = rocksdb::WriteBatch::default();
            for key in keys {
                batch.put(key.to_be_bytes(), []);
            }
            let mut options = rocksdb::WriteOptions::default();
            // A cache: losing the tail on crash only costs owner dedup.
            options.disable_wal(true);
            db.write_opt(batch, &options)?;
        }
        Ok(())
    }

    /// Stage keys whose lane outcome at WAL `index` was Applied. They are not
    /// skips yet: Applied is not durable publication. The staging set is
    /// bounded by the RAM tier capacity; the oldest unproven keys are dropped
    /// (owner dedup stays authoritative for them).
    pub fn stage_applied(&self, shard: u32, index: u64, keys: Vec<Key>) -> Result<()> {
        self.scope(shard)?;
        if keys.is_empty() {
            return Ok(());
        }
        let mut staged = self.staged.lock().unwrap();
        let added = keys.len() as u64;
        staged.entry(shard).or_default().push_back((index, keys));
        let mut total = self.metrics.staged_keys.fetch_add(added, Relaxed) + added;
        while total > self.capacity as u64 {
            let Some(queue) = staged.values_mut().max_by_key(|q| q.len()) else {
                break;
            };
            let Some((_, old)) = queue.pop_front() else {
                break;
            };
            let n = old.len() as u64;
            total -= n;
            self.metrics.staged_keys.fetch_sub(n, Relaxed);
            self.metrics.unproven_dropped.fetch_add(n, Relaxed);
        }
        Ok(())
    }

    /// Insert staged keys of `shard` whose Applied index the verified selected
    /// full body covers. `selected` must be this shard's scope (table,
    /// incarnation, bucket); otherwise nothing is inserted. Blocking when the
    /// disk tier is enabled.
    pub fn prove(
        &self,
        shard: u32,
        selected: &crate::annotations::single_body::SelectedBody,
    ) -> Result<u64> {
        let (scope, _) = self.scope(shard)?;
        anyhow::ensure!(
            selected.table == scope.table
                && selected.incarnation == scope.incarnation
                && selected.source.shard == shard,
            "applied-key proof is not this shard's selected identity"
        );
        let proven: Vec<Key> = {
            let mut staged = self.staged.lock().unwrap();
            let Some(queue) = staged.get_mut(&shard) else {
                return Ok(0);
            };
            let mut keys = Vec::new();
            while queue
                .front()
                .is_some_and(|(index, _)| *index <= selected.through)
            {
                keys.extend(queue.pop_front().unwrap().1);
            }
            self.metrics
                .staged_keys
                .fetch_sub(keys.len() as u64, Relaxed);
            keys
        };
        self.insert(&proven, Reason::Proven)?;
        Ok(proven.len() as u64)
    }

    /// Drop staged, unproven keys of the given shards (failed fence).
    pub fn discard_staged(&self) {
        let mut staged = self.staged.lock().unwrap();
        let n: usize = staged
            .values()
            .flat_map(|q| q.iter())
            .map(|(_, k)| k.len())
            .sum();
        staged.clear();
        self.metrics.staged_keys.fetch_sub(n as u64, Relaxed);
        self.metrics.unproven_dropped.fetch_add(n as u64, Relaxed);
    }

    /// Remove keys of a record later reported Rejected.
    pub fn remove(&self, keys: &[Key]) -> Result<()> {
        self.metrics
            .removed_rejected
            .fetch_add(keys.len() as u64, Relaxed);
        {
            let mut ram = self.ram.lock().unwrap();
            for key in keys {
                ram.map.remove(key);
            }
        }
        #[cfg(feature = "applied-key-cache")]
        if let Some(db) = &self.disk {
            let mut batch = rocksdb::WriteBatch::default();
            for key in keys {
                batch.delete(key.to_be_bytes());
            }
            db.write(batch)?;
        }
        Ok(())
    }

    /// Per-key membership. Blocking when the disk tier is enabled; call from
    /// a blocking context.
    pub fn contains_many(&self, keys: &[Key]) -> Result<Vec<bool>> {
        let mut found = {
            let ram = self.ram.lock().unwrap();
            keys.iter()
                .map(|key| ram.map.contains_key(key))
                .collect::<Vec<_>>()
        };
        let ram_hits = found.iter().filter(|hit| **hit).count() as u64;
        self.metrics.ram_hits.fetch_add(ram_hits, Relaxed);
        #[allow(unused_mut)]
        let mut disk_hits = 0u64;
        #[cfg(feature = "applied-key-cache")]
        if let Some(db) = &self.disk {
            let missing: Vec<usize> = (0..keys.len()).filter(|&i| !found[i]).collect();
            if !missing.is_empty() {
                let lookups = db.multi_get(missing.iter().map(|&i| keys[i].to_be_bytes()));
                for (&i, value) in missing.iter().zip(lookups) {
                    if value?.is_some() {
                        found[i] = true;
                        disk_hits += 1;
                    }
                }
            }
        }
        self.metrics.disk_hits.fetch_add(disk_hits, Relaxed);
        self.metrics
            .misses
            .fetch_add(keys.len() as u64 - ram_hits - disk_hits, Relaxed);
        Ok(found)
    }

    pub fn snapshot(&self) -> CacheSnapshot {
        let m = &self.metrics;
        CacheSnapshot {
            insert_commit: m.insert_commit.load(Relaxed),
            insert_applied: m.insert_applied.load(Relaxed),
            insert_proven: m.insert_proven.load(Relaxed),
            staged_keys: m.staged_keys.load(Relaxed),
            unproven_dropped: m.unproven_dropped.load(Relaxed),
            insert_warm: m.insert_warm.load(Relaxed),
            insert_owner_dedup: m.insert_owner_dedup.load(Relaxed),
            removed_rejected: m.removed_rejected.load(Relaxed),
            ram_hits: m.ram_hits.load(Relaxed),
            disk_hits: m.disk_hits.load(Relaxed),
            misses: m.misses.load(Relaxed),
            evictions: m.evictions.load(Relaxed),
            ram_keys: self.ram.lock().unwrap().map.len() as u64,
            disk_enabled: self.disk_enabled(),
            warm_rows: m.warm_rows.load(Relaxed),
            warm_bytes: m.warm_bytes.load(Relaxed),
            warm_fragments: m.warm_fragments.load(Relaxed),
            warm_complete_shards: m.warm_complete_shards.load(Relaxed),
            warm: m.warm.lock().unwrap().clone(),
        }
    }

    fn disk_enabled(&self) -> bool {
        #[cfg(feature = "applied-key-cache")]
        return self.disk.is_some();
        #[cfg(not(feature = "applied-key-cache"))]
        false
    }

    /// Last fragment id warmed for `shard`, persisted in the disk tier. A
    /// RAM-only cache always restarts its warm from the beginning.
    pub fn warmed_through(&self, shard: u32) -> Result<Option<u64>> {
        #[cfg(feature = "applied-key-cache")]
        if let Some(db) = &self.disk {
            return Ok(db.get(warm_key(shard))?.map(|v| {
                let mut bytes = [0u8; 8];
                bytes.copy_from_slice(&v[..8]);
                u64::from_be_bytes(bytes)
            }));
        }
        let _ = shard;
        Ok(None)
    }

    fn set_warmed_through(&self, shard: u32, fragment: u64) -> Result<()> {
        #[cfg(feature = "applied-key-cache")]
        if let Some(db) = &self.disk {
            // Key inserts above skip the RocksDB WAL; flush them first so the
            // cursor never runs ahead of durable keys.
            db.flush()?;
            db.put(warm_key(shard), fragment.to_be_bytes())?;
        }
        let _ = (shard, fragment);
        Ok(())
    }

    fn warm_progress(&self, shard: u32, update: impl FnOnce(&mut WarmProgress)) {
        update(self.metrics.warm.lock().unwrap().entry(shard).or_default());
    }
}

#[cfg(feature = "applied-key-cache")]
fn warm_key(shard: u32) -> Vec<u8> {
    let mut key = WARM_PREFIX.to_vec();
    key.extend_from_slice(&shard.to_be_bytes());
    key
}

#[cfg(feature = "applied-key-cache")]
fn open_disk(dir: &std::path::Path, config: &Config, scope: &[u8]) -> Result<rocksdb::DB> {
    std::fs::create_dir_all(dir)?;
    let db = open_db(dir, config)?;
    match db.get(SCOPE_KEY)? {
        Some(stored) if stored == scope => return Ok(db),
        None if db.iterator(rocksdb::IteratorMode::Start).next().is_none() => {
            db.put(SCOPE_KEY, scope)?;
            db.flush()?;
            return Ok(db);
        }
        // Another (or no recorded) table identity: its keys and warm
        // cursors must never answer for this scope. Wipe and start empty.
        _ => {}
    }
    drop(db);
    rocksdb::DB::destroy(&rocksdb::Options::default(), dir)?;
    std::fs::create_dir_all(dir)?;
    let db = open_db(dir, config)?;
    db.put(SCOPE_KEY, scope)?;
    db.flush()?;
    Ok(db)
}

#[cfg(feature = "applied-key-cache")]
fn open_db(dir: &std::path::Path, config: &Config) -> Result<rocksdb::DB> {
    let mut table = rocksdb::BlockBasedOptions::default();
    table.set_bloom_filter(config.bloom_bits_per_key, false);
    table.set_block_cache(&rocksdb::Cache::new_lru_cache(config.block_cache_bytes));
    table.set_cache_index_and_filter_blocks(true);
    table.set_pin_l0_filter_and_index_blocks_in_cache(true);
    let mut options = rocksdb::Options::default();
    options.create_if_missing(true);
    options.set_block_based_table_factory(&table);
    // Keys are random hashes: point lookups only.
    options.optimize_for_point_lookup(config.block_cache_bytes as u64 >> 20);
    options.set_max_open_files(256);
    Ok(rocksdb::DB::open(&options, dir)?)
}

/// IO bounds for the background warm.
#[derive(Clone, Copy, Debug)]
pub struct WarmLimits {
    pub batch_rows: usize,
    pub io_buffer_bytes: u64,
    /// Sustained source bytes per second (0 = unbounded).
    pub bytes_per_second: u64,
}

impl AppliedKeyCache {
    /// Warm one shard from its canonical Turns dataset's key columns, in
    /// ascending fragment order, resuming after the persisted warmed-through
    /// fragment. Fragments are read one at a time with a single readahead
    /// batch and a byte-rate limit. Run it as a background task; it stops at
    /// the end of the snapshot it opened.
    ///
    /// `selected` is the owner's verified selected full body for `shard` and
    /// `dataset` must be opened at exactly its manifest version; both must
    /// match this cache's scope for the shard, or nothing is warmed.
    pub async fn warm(
        &self,
        shard: u32,
        selected: &crate::annotations::single_body::SelectedBody,
        dataset: &lance::Dataset,
        key_columns: &[String],
        limits: WarmLimits,
        stop: &std::sync::atomic::AtomicBool,
    ) -> Result<()> {
        use futures::TryStreamExt;
        let (scope, digest) = self.scope(shard)?.clone();
        anyhow::ensure!(
            selected.table == scope.table
                && selected.incarnation == scope.incarnation
                && selected.source.shard == shard
                && dataset.uri() == selected.uri
                && dataset.version().version == selected.manifest.version,
            "applied-key warm source is not this shard's selected identity"
        );
        let transaction = dataset
            .read_transaction()
            .await?
            .context("applied-key warm source has no transaction")?;
        anyhow::ensure!(
            Uuid::parse_str(&transaction.uuid)? == selected.manifest.transaction,
            "applied-key warm source transaction differs from selection"
        );
        let resume = self.warmed_through(shard)?;
        let mut fragments: Vec<_> = dataset
            .get_fragments()
            .into_iter()
            .map(|fragment| fragment.metadata().clone())
            .collect();
        fragments.sort_by_key(|fragment| fragment.id);
        let total = fragments.len() as u64;
        let done_before = resume.map_or(0, |through| {
            fragments.iter().filter(|f| f.id <= through).count() as u64
        });
        self.warm_progress(shard, |p| {
            p.fragments_total = total;
            p.fragments_done = done_before;
            p.warmed_through_fragment = resume;
        });
        let started = std::time::Instant::now();
        let mut read_bytes = 0u64;
        for fragment in fragments {
            if resume.is_some_and(|through| fragment.id <= through) {
                continue;
            }
            if stop.load(Relaxed) {
                return Ok(());
            }
            let id = fragment.id;
            let mut scanner = dataset.scan();
            scanner.project(key_columns)?;
            scanner.with_fragments(vec![fragment]);
            scanner.batch_size(limits.batch_rows);
            scanner.batch_readahead(1);
            scanner.fragment_readahead(1);
            scanner.io_buffer_size(limits.io_buffer_bytes);
            scanner.use_scalar_index(false);
            let mut stream = scanner.try_into_stream().await?;
            let mut rows = 0u64;
            while let Some(batch) = stream.try_next().await? {
                if stop.load(Relaxed) {
                    return Ok(());
                }
                let bytes = batch.get_array_memory_size() as u64;
                let keys = batch_keys(&digest, &batch, key_columns)?;
                self.insert(&keys, Reason::Warm)?;
                rows += keys.len() as u64;
                read_bytes += bytes;
                self.metrics.warm_rows.fetch_add(keys.len() as u64, Relaxed);
                self.metrics.warm_bytes.fetch_add(bytes, Relaxed);
                if limits.bytes_per_second > 0 {
                    let due = std::time::Duration::from_secs_f64(
                        read_bytes as f64 / limits.bytes_per_second as f64,
                    );
                    if let Some(wait) = due.checked_sub(started.elapsed()) {
                        tokio::time::sleep(wait).await;
                    }
                }
            }
            self.set_warmed_through(shard, id)?;
            self.metrics.warm_fragments.fetch_add(1, Relaxed);
            self.warm_progress(shard, |p| {
                p.warmed_through_fragment = Some(id);
                p.fragments_done += 1;
                p.rows += rows;
            });
        }
        self.metrics.warm_complete_shards.fetch_add(1, Relaxed);
        self.warm_progress(shard, |p| p.complete = true);
        Ok(())
    }
}
