use clap::Parser;

#[derive(Debug, Clone, Parser)]
#[command(name = "lance-context-server")]
#[command(about = "REST API server for lance-context")]
pub struct ServerConfig {
    #[arg(long, default_value = "0.0.0.0")]
    pub host: String,

    #[arg(long, default_value = "3000")]
    pub port: u16,

    #[arg(long, default_value = "./lance-data")]
    pub data_dir: String,

    /// Stable identity of this server instance, used as the MemWAL shard key for
    /// rollout writes. Each instance must present a distinct, stable value so it
    /// owns exactly one shard and never contends with peers (see
    /// `specs/rollout-deployment.md`). In Kubernetes, set this to the
    /// StatefulSet pod ordinal hostname (e.g. `rollout-0`). Defaults to the
    /// `INSTANCE_ID` env var, then the pod/host `HOSTNAME`; if neither is set,
    /// rollout writes fall back to a single shared `default` shard, which is
    /// only safe for a single-instance deployment.
    #[arg(long, env = "INSTANCE_ID")]
    pub instance_id: Option<String>,

    /// Count-triggered self-merge threshold for rollout MemWAL shards. After an
    /// append flushes a new generation, if this instance's shard has at least
    /// this many un-merged flushed generations, the append synchronously folds
    /// them into the base table and drains the shard (see
    /// `specs/rollout-deployment.md`). This bounds read amplification. `0`
    /// (the default) disables self-merge — generations accumulate and are
    /// unioned at read time.
    #[arg(long, env = "ROLLOUT_MERGE_AFTER_GENERATIONS", default_value = "0")]
    pub rollout_merge_after_generations: usize,

    /// Interval, in seconds, for the periodic per-shard WAL cleanup task. When
    /// non-zero, each rollout store spawns a background timer that folds this
    /// instance's flushed MemWAL generations into the base table on a schedule —
    /// the *time* half of the "time OR count" trigger, complementing
    /// `--rollout-merge-after-generations`. Whichever fires first merges: the
    /// timer reclaims whatever is pending regardless of count, so stale
    /// generations are folded in even on low-traffic shards that never cross the
    /// count threshold. `0` (the default) disables the timer.
    #[arg(long, env = "ROLLOUT_CLEANUP_INTERVAL_SECS", default_value = "0")]
    pub rollout_cleanup_interval_secs: u64,

    /// Upper bound on the number of resident rollout-store handles kept in
    /// memory (an LRU). With one physical dataset per experiment a deployment
    /// may hold hundreds of thousands of stores; this bounds how many stay open
    /// at once. Size it for peak *concurrent* experiments, not the total count.
    /// Evicted stores are transparently reopened on the next request. `0` falls
    /// back to the built-in default.
    #[arg(long, env = "ROLLOUT_CACHE_CAPACITY", default_value = "2000")]
    pub rollout_cache_capacity: usize,
}

impl ServerConfig {
    /// Resolve the instance id used for rollout MemWAL sharding: the explicit
    /// `--instance-id`/`INSTANCE_ID` if provided, otherwise the `HOSTNAME`
    /// environment variable (stable per-pod under a StatefulSet).
    #[must_use]
    pub fn resolved_instance_id(&self) -> Option<String> {
        self.instance_id
            .clone()
            .or_else(|| std::env::var("HOSTNAME").ok())
            .filter(|value| !value.is_empty())
    }
}
