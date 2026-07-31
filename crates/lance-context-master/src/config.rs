//! Command-line / environment configuration for the master control-plane.

use clap::Parser;

/// Control-plane (master) process for lance-context rollout stores.
#[derive(Debug, Clone, Parser)]
#[command(name = "lance-context-master", version)]
pub struct MasterConfig {
    /// Data directory / object-store prefix shared with the data-plane server.
    #[arg(long, env = "DATA_DIR", default_value = "./data")]
    pub data_dir: String,

    /// Host to bind the admin API + UI to.
    #[arg(long, env = "MASTER_HOST", default_value = "0.0.0.0")]
    pub host: String,

    /// Port to bind the admin API + UI to.
    #[arg(long, env = "MASTER_PORT", default_value_t = 8090)]
    pub port: u16,

    /// Interval, in seconds, between full stats scans of every experiment.
    /// `0` disables the background scanner.
    #[arg(long, env = "STATS_SCAN_INTERVAL_SECS", default_value_t = 300)]
    pub stats_scan_interval_secs: u64,

    /// Bounded concurrency when scanning experiments each round.
    #[arg(long, env = "SCAN_CONCURRENCY", default_value_t = 8)]
    pub scan_concurrency: usize,

    /// Total byte budget for the Lance metadata/index caches, shared across
    /// every rollout store opened by this master process.
    ///
    /// Without an explicit shared session, Lance gives each store a fresh
    /// session with caches defaulting to 6 GiB index + 1 GiB metadata. The
    /// master opens stores for scans, browsing, compaction, and indexing, so
    /// per-store sessions make RSS grow with the experiments touched. This
    /// budget is split internally 6:1 (index:metadata), matching Lance's
    /// default ratio. `0` restores Lance's per-store default sessions.
    #[arg(long, env = "ROLLOUT_CACHE_BYTES", default_value = "2147483648")]
    pub rollout_cache_bytes: usize,

    /// Run maintenance (compaction + old-version cleanup) on the `_stats`
    /// dataset every Nth stats-scan round. `_stats` is written delete-then-
    /// append, so each scan adds versions and fragments per experiment; without
    /// maintenance the manifest chain grows without bound and slows cold start
    /// and `/experiments`. `0` disables stats maintenance.
    #[arg(long, env = "STATS_MAINTENANCE_EVERY_N_SCANS", default_value_t = 12)]
    pub stats_maintenance_every_n_scans: u64,

    /// Grace window, in seconds, for `_stats` old-version cleanup. Versions
    /// newer than this are never removed, so in-flight readers on another
    /// replica keep working.
    #[arg(long, env = "STATS_HISTORY_TTL_SECS", default_value_t = 3_600)]
    pub stats_history_ttl_secs: u64,

    /// Age, in seconds, after which an experiment with no writes is retired
    /// from the stats table. `0` disables retirement (every known experiment
    /// stays in the table forever).
    ///
    /// Retirement is what keeps the stats table proportional to *active* work
    /// rather than to everything ever created. A retired experiment is still
    /// listed in the registry and is observed on demand; it simply stops
    /// costing a row and an open on every scan round.
    ///
    /// Before an experiment is retired its MemWAL generations are merged and
    /// its fragments compacted, so it is left in a state that needs no further
    /// maintenance — which is what makes it safe for the sweeps to stop looking
    /// at it. See `scanner::retire_cold_experiments`.
    #[arg(long, env = "STATS_COLD_RETIRE_SECS", default_value_t = 604_800)]
    pub stats_cold_retire_secs: u64,

    /// Interval, in seconds, between automatic compaction sweeps. `0` disables
    /// automatic compaction (manual triggers still work).
    #[arg(long, env = "COMPACTION_INTERVAL_SECS", default_value_t = 600)]
    pub compaction_interval_secs: u64,

    /// Minimum fragment count before an experiment is auto-compacted.
    #[arg(long, env = "MIN_FRAGMENTS", default_value_t = 16)]
    pub min_fragments: usize,

    /// Target rows per fragment passed to compaction.
    #[arg(long, env = "TARGET_ROWS_PER_FRAGMENT", default_value_t = 1_048_576)]
    pub target_rows_per_fragment: usize,

    /// Maximum number of compactions executing in this master process.
    #[arg(long, env = "COMPACTION_CONCURRENCY", default_value_t = 1)]
    pub compaction_concurrency: usize,

    /// Lance rewrite tasks executing inside one compaction.
    #[arg(long, env = "COMPACTION_THREADS", default_value_t = 1)]
    pub compaction_threads: usize,

    /// Rows per input batch when compaction must decode and re-encode data.
    #[arg(long, env = "COMPACTION_BATCH_SIZE", default_value_t = 8)]
    pub compaction_batch_size: usize,

    /// Maximum source fragments rewritten by one compaction task. `0` disables
    /// the incremental limit.
    #[arg(long, env = "COMPACTION_MAX_SOURCE_FRAGMENTS", default_value_t = 32)]
    pub compaction_max_source_fragments: usize,

    /// Maximum bytes per compacted output file. `0` uses Lance's default.
    #[arg(
        long,
        env = "COMPACTION_MAX_BYTES_PER_FILE",
        default_value_t = 1_073_741_824
    )]
    pub compaction_max_bytes_per_file: usize,

    /// Interval, in seconds, between automatic WAL-merge sweeps. Each sweep
    /// enqueues a `MergeWal` task for every experiment whose pending MemWAL
    /// generation count crosses `merge_wal_min_generations`; the task fans out
    /// to the configured worker endpoints. `0` disables automatic WAL merge
    /// (manual triggers still work).
    #[arg(long, env = "MERGE_WAL_INTERVAL_SECS", default_value_t = 600)]
    pub merge_wal_interval_secs: u64,

    /// Minimum pending MemWAL generations (per experiment) before an automatic
    /// `MergeWal` is enqueued. Read from the periodically-scanned stats table.
    #[arg(long, env = "MERGE_WAL_MIN_GENERATIONS", default_value_t = 8)]
    pub merge_wal_min_generations: i64,

    /// Data-plane worker base URLs (comma-separated), e.g.
    /// `http://rollout-0:3000,http://rollout-1:3000`. A `MergeWal` task fans out
    /// to every endpoint so each worker merges its own MemWAL shard (the master
    /// cannot merge a shard it does not own without fencing the live writer).
    /// Empty (the default) means WAL-merge tasks have nowhere to go and fail
    /// fast with a clear message.
    #[arg(long, env = "WORKER_ENDPOINTS", value_delimiter = ',')]
    pub worker_endpoints: Vec<String>,

    /// Maximum number of scheduler tasks executing concurrently. Compaction of
    /// the *same* experiment is always serialized regardless of this value
    /// (two `Rewrite`s on one dataset conflict); this only bounds how many
    /// *distinct* experiments/tasks run at once.
    #[arg(long, env = "TASK_CONCURRENCY", default_value_t = 4)]
    pub task_concurrency: usize,

    /// Comma-separated etcd v3 endpoints. Scheduler state (task queue,
    /// lease-based claims, per-experiment write locks) lives in etcd so several
    /// stateless master replicas can share one queue. Required.
    #[arg(long, env = "ETCD_ENDPOINTS", value_delimiter = ',')]
    pub etcd_endpoints: Vec<String>,

    /// Namespace for all lance-context master keys in etcd.
    #[arg(long, env = "ETCD_PREFIX", default_value = "/lance-context/master")]
    pub etcd_prefix: String,

    /// Optional etcd username. `ETCD_PASSWORD` must also be set.
    #[arg(long, env = "ETCD_USERNAME")]
    pub etcd_username: Option<String>,

    /// Optional etcd password. `ETCD_USERNAME` must also be set.
    #[arg(long, env = "ETCD_PASSWORD")]
    pub etcd_password: Option<String>,

    /// Optional PEM CA certificate path for etcd TLS.
    #[arg(long, env = "ETCD_CA_CERT")]
    pub etcd_ca_cert: Option<String>,

    /// Optional PEM client certificate path for etcd mutual TLS.
    #[arg(long, env = "ETCD_CLIENT_CERT")]
    pub etcd_client_cert: Option<String>,

    /// Optional PEM client private-key path for etcd mutual TLS.
    #[arg(long, env = "ETCD_CLIENT_KEY")]
    pub etcd_client_key: Option<String>,

    /// TTL for etcd task claims and distributed locks. The master renews leases
    /// while work is running; orphaned tasks are requeued after expiry.
    #[arg(long, env = "ETCD_LEASE_TTL_SECS", default_value_t = 30)]
    pub etcd_lease_ttl_secs: i64,

    /// Maximum number of terminal scheduler tasks retained in etcd and the
    /// queue UI. Queued and running tasks are never pruned, and at least one
    /// terminal task is retained for status polling.
    #[arg(long, env = "TASK_HISTORY_LIMIT", default_value_t = 1_000)]
    pub task_history_limit: usize,

    /// Maximum age of terminal (Done/Failed) scheduler tasks before they are
    /// pruned, in seconds. `0` disables age-based pruning (the count cap in
    /// `TASK_HISTORY_LIMIT` still applies). Whichever of the two removes a task
    /// first wins. Queued/running tasks and terminal tasks that a live task
    /// still depends on are never pruned regardless of age.
    #[arg(long, env = "TASK_HISTORY_TTL_SECS", default_value_t = 86_400)]
    pub task_history_ttl_secs: u64,

    /// Directory of built UI assets to serve. When unset, only the JSON API is
    /// exposed.
    #[arg(long, env = "UI_DIR")]
    pub ui_dir: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn every_config_field_is_an_optional_flag() {
        let command = MasterConfig::command();

        let positionals: Vec<_> = command
            .get_positionals()
            .map(|arg| arg.get_id().to_string())
            .collect();
        assert!(
            positionals.is_empty(),
            "config fields must be flags, not positional args; found {positionals:?}"
        );

        for arg in command.get_arguments() {
            if arg.get_id() == "help" || arg.get_id() == "version" {
                continue;
            }
            assert!(
                arg.get_long().is_some(),
                "'{}' has no long flag",
                arg.get_id()
            );
            assert!(
                arg.get_default_values().len() == 1 || !arg.is_required_set(),
                "'{}' must have a default or be optional",
                arg.get_id()
            );
        }
    }

    #[test]
    fn rollout_cache_defaults_to_two_gib_and_can_be_disabled() {
        let default = MasterConfig::try_parse_from(["lance-context-master"]).unwrap();
        assert_eq!(default.rollout_cache_bytes, 2 * 1024 * 1024 * 1024);

        let disabled =
            MasterConfig::try_parse_from(["lance-context-master", "--rollout-cache-bytes", "0"])
                .unwrap();
        assert_eq!(disabled.rollout_cache_bytes, 0);
    }

    #[test]
    fn compaction_defaults_bound_parallelism_and_rewrite_size() {
        let config = MasterConfig::try_parse_from(["lance-context-master"]).unwrap();
        assert_eq!(config.compaction_concurrency, 1);
        assert_eq!(config.compaction_threads, 1);
        assert_eq!(config.compaction_batch_size, 8);
        assert_eq!(config.compaction_max_source_fragments, 32);
        assert_eq!(config.compaction_max_bytes_per_file, 1024 * 1024 * 1024);
    }
}
