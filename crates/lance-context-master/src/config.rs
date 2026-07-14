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

    /// Directory of built UI assets to serve. When unset, only the JSON API is
    /// exposed.
    #[arg(long, env = "UI_DIR")]
    pub ui_dir: Option<String>,
}
