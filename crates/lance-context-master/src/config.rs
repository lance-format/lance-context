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
