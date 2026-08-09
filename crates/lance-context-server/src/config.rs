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
    /// context, rollout, datagen, and generic writes. Each instance must present
    /// a distinct, stable value so it owns exactly one shard per store and never
    /// contends with peers (see
    /// `docs/src/specs/rollout-deployment.md`). In Kubernetes, set this to the
    /// StatefulSet pod ordinal hostname (e.g. `rollout-0`). Defaults to the
    /// `INSTANCE_ID` env var, then the pod/host `HOSTNAME`; if neither is set,
    /// writes fall back to a single shared `default` shard per store, which is
    /// only safe for a single-instance deployment.
    #[arg(long, env = "INSTANCE_ID")]
    pub instance_id: Option<String>,

    /// Count-triggered self-merge threshold for rollout MemWAL shards. When this
    /// instance's shard has at least this many un-merged flushed generations, the
    /// periodic sweeper folds them into the base table and drains the shard (see
    /// `docs/src/specs/rollout-deployment.md`). This bounds read amplification. `0`
    /// (the default) disables the count trigger — generations accumulate and are
    /// unioned at read time (or reclaimed by the time-based cleanup below).
    ///
    /// Note: this merge runs on the periodic sweeper, not on the append path.
    /// Appends are durable (WAL-persisted) but do not merge inline.
    #[arg(long, env = "ROLLOUT_MERGE_AFTER_GENERATIONS", default_value = "0")]
    pub rollout_merge_after_generations: usize,

    /// Interval, in seconds, for the periodic per-shard WAL cleanup task. When
    /// non-zero, the global sweeper folds this instance's flushed MemWAL
    /// generations into the base table on a schedule — the *time* half of the
    /// "time OR count" trigger, complementing `--rollout-merge-after-generations`.
    /// Whichever fires first merges: the timer reclaims whatever is pending
    /// regardless of count, so stale generations are folded in even on
    /// low-traffic shards that never cross the count threshold. `0` (the default)
    /// disables the timer.
    ///
    /// # Applies to every store kind, despite the name
    ///
    /// The sweeper visits rollout, datagen and generic stores alike — every
    /// MemWAL-backed store accumulates generations, and only a merge reclaims
    /// them. The `ROLLOUT_` prefix is historical: these variables predate the
    /// other store kinds, and renaming them would silently break existing
    /// deployment manifests for no functional gain.
    #[arg(long, env = "ROLLOUT_CLEANUP_INTERVAL_SECS", default_value = "0")]
    pub rollout_cleanup_interval_secs: u64,

    /// Interval, in seconds, at which the sweeper flushes each resident store's
    /// active MemWAL memtable into a queryable generation. A deferred-seal
    /// append is durable on return (the WAL entry is persisted to object
    /// storage) but is not visible to reads until the memtable is flushed, so
    /// this interval bounds read-after-write latency. Decoupling flush from the
    /// append path is what lets concurrent appends run without serializing
    /// behind a per-append seal. Default `30`.
    ///
    /// Like the cleanup interval, this applies to **every** store kind despite
    /// the `ROLLOUT_` prefix. Rollout and generic stores default to a deferred
    /// seal and depend on it; datagen seals on each append, so its pass is a
    /// no-op in steady state.
    ///
    /// `0` disables periodic flush, leaving the cleanup sweeper
    /// (`ROLLOUT_CLEANUP_INTERVAL_SECS`) as the only thing that seals memtables
    /// — it flushes before merging, so it is a sufficient fallback, but
    /// read-after-write latency is then bounded by the *cleanup* interval
    /// instead. Setting **both** to `0` means nothing ever seals: appends stay
    /// durable but invisible until the process restarts and replays the WAL.
    /// The server warns at startup in that configuration.
    #[arg(long, env = "ROLLOUT_FLUSH_INTERVAL_SECS", default_value = "30")]
    pub rollout_flush_interval_secs: u64,

    /// Upper bound on the number of resident rollout-store handles kept in
    /// memory (an LRU). With one physical dataset per experiment a deployment
    /// may hold hundreds of thousands of stores; this bounds how many stay open
    /// at once. Size it for peak *concurrent* experiments, not the total count.
    /// Evicted stores are transparently reopened on the next request. `0` falls
    /// back to the built-in default.
    #[arg(long, env = "ROLLOUT_CACHE_CAPACITY", default_value = "2000")]
    pub rollout_cache_capacity: usize,

    /// Ceiling, in bytes, on the total artifact-blob payload held in memory
    /// across all *concurrent* rollout uploads and downloads on this instance.
    ///
    /// Each blob request materializes its full payload as an in-memory buffer
    /// (uploads buffer the request body; downloads materialize the row's
    /// `binary_payload`). Without a global cap, N concurrent 1 GiB requests
    /// would need N GiB and can OOM the process. This budget admits a request
    /// only while enough of the budget is free — otherwise it is rejected with
    /// `503 Service Unavailable` and a `Retry-After`, applying backpressure at
    /// the edge instead of the allocator. A single request larger than the whole
    /// budget is still admitted when the instance is otherwise idle (it reserves
    /// the entire budget), so this bounds concurrency, not maximum blob size.
    /// `0` (the default) disables the budget.
    #[arg(long, env = "ROLLOUT_MAX_INFLIGHT_BLOB_BYTES", default_value = "0")]
    pub rollout_max_inflight_blob_bytes: usize,

    /// Total byte budget for the Lance metadata/index caches, shared across
    /// **all** resident rollout stores on this instance.
    ///
    /// Every rollout store opens a Lance dataset; without an explicit session
    /// Lance gives each store its own caches defaulting to **6 GiB index + 1 GiB
    /// metadata**, keyed by dataset URI. Because each flushed MemWAL generation
    /// is a distinct URI, a busy store's read path feeds an ever-growing key set
    /// into that per-store cache until it approaches 6 GiB — worker RSS then
    /// grows linearly with cumulative appends and never releases across
    /// merge/compact cycles. Attaching one shared, capacity-bounded session caps
    /// the process's *total* Lance cache at this budget instead of 6 GiB per
    /// store. Split internally 6:1 (index:metadata), matching Lance's own
    /// default ratio. `0` disables sharing and restores Lance's per-store
    /// default session (the pre-fix, leak-prone behavior).
    #[arg(long, env = "ROLLOUT_CACHE_BYTES", default_value = "2147483648")]
    pub rollout_cache_bytes: usize,
}

impl ServerConfig {
    /// Resolve the instance id used for server-managed MemWAL sharding: the
    /// explicit `--instance-id`/`INSTANCE_ID` if provided, otherwise the
    /// `HOSTNAME` environment variable (stable per-pod under a StatefulSet).
    #[must_use]
    pub fn resolved_instance_id(&self) -> Option<String> {
        self.instance_id
            .clone()
            .or_else(|| std::env::var("HOSTNAME").ok())
            .filter(|value| !value.is_empty())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    /// Every tunable must stay an *optional flag* with a default.
    ///
    /// Regression guard: dropping a field's `#[arg(..)]` attribute silently
    /// turns it into a required positional argument, so the binary still
    /// compiles and every unit test still passes — it just refuses to start
    /// with the documented flags. That is only caught by actually running the
    /// server, which nothing in the Rust suite does.
    #[test]
    fn every_config_field_is_an_optional_flag() {
        let command = ServerConfig::command();

        let positionals: Vec<_> = command
            .get_positionals()
            .map(|arg| arg.get_id().to_string())
            .collect();
        assert!(
            positionals.is_empty(),
            "config fields must be flags, not positional args; found {positionals:?} \
             (a missing #[arg(..)] attribute does this)"
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

    /// The server must start with no arguments beyond a data directory — the
    /// documented defaults have to actually be defaults.
    #[test]
    fn parses_with_only_a_data_dir() {
        let config = ServerConfig::try_parse_from(["lance-context-server", "--data-dir", "/tmp/x"])
            .expect("the documented minimal invocation must parse");
        assert_eq!(config.rollout_flush_interval_secs, 30);
        assert_eq!(config.rollout_cleanup_interval_secs, 0);
    }
}
