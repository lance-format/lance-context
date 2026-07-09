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
