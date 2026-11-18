//! Fresh-store benchmark launcher. Campaigns only during startup; use the normal
//! node binary for restarts. A local bootstrap receipt is not a quorum read or a
//! durability receipt. The driver must verify every replica before sending work.
use anyhow::{ensure, Context, Result};
use clap::Parser;
use lance_context_wal::{
    replication::runtime::Handle,
    service::{Config, Service},
};
use std::{collections::BTreeMap, fs, path::PathBuf, time::Duration};
use tokio::time::Instant;

#[derive(Parser)]
#[command(about = "Fresh-store WAL benchmark node with bounded startup campaigns")]
struct Args {
    #[arg(long)]
    config: PathBuf,
    /// JSON map of every layout group to its intended leader node.
    #[arg(long)]
    leader_plan: PathBuf,
    /// Created once, after local observations match the plan for three seconds.
    #[arg(long)]
    ready_file: PathBuf,
    #[arg(long,default_value_t=2,value_parser=clap::value_parser!(u16).range(1..=64))]
    io_threads: u16,
    #[arg(long,default_value_t=4,value_parser=clap::value_parser!(u16).range(1..=64))]
    blocking_threads: u16,
    #[arg(long,default_value_t=30_000,value_parser=clap::value_parser!(u64).range(1..=300_000))]
    bootstrap_timeout_ms: u64,
}

async fn bootstrap(
    handle: &Handle,
    config: &Config,
    plan: &BTreeMap<u64, u64>,
) -> Result<BTreeMap<u64, (u64, u64)>> {
    let hosted: Vec<_> = config
        .layout
        .groups
        .iter()
        .filter(|(_, voters)| voters.contains(&config.node))
        .map(|(&group, _)| group)
        .collect();
    let mut previous = BTreeMap::new();
    let mut stable_since = None;
    loop {
        let mut current = BTreeMap::new();
        for &group in &hosted {
            let observation = handle.observation(group)?;
            ensure!(
                !observation.stopped && observation.failure.is_none(),
                "bootstrap worker failed: {:?}",
                observation.failure
            );
            let status = observation.status;
            ensure!(
                status.leader == 0 || status.leader == plan[&group],
                "bootstrap leader mismatch for group {group}: expected {}, observed {}",
                plan[&group],
                status.leader
            );
            if status.leader == 0 && plan[&group] == config.node {
                // This only starts an election. Its completion says nothing
                // about leadership; subsequent observations decide readiness.
                handle.campaign(group)?.await?;
            }
            if status.leader == plan[&group]
                && status.committed > 0
                && status.applied == status.committed
                && observation.observed_at.elapsed() < Duration::from_secs(1)
            {
                current.insert(group, (status.leader, status.term));
            }
        }
        if current.len() == hosted.len() && current == previous {
            let since = stable_since.get_or_insert_with(Instant::now);
            if since.elapsed() >= Duration::from_secs(3) {
                return Ok(current);
            }
        } else {
            stable_since = None;
        }
        previous = current;
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

async fn signal() -> Result<()> {
    #[cfg(unix)]
    {
        let mut terminate =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?;
        tokio::select! { result=tokio::signal::ctrl_c()=>result?, _=terminate.recv()=>{} }
    }
    #[cfg(not(unix))]
    tokio::signal::ctrl_c().await?;
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    let config = Config::read(&args.config)?;
    let plan: BTreeMap<u64, u64> = serde_json::from_slice(&fs::read(&args.leader_plan)?)?;
    ensure!(
        plan.len() == config.layout.groups.len()
            && config
                .layout
                .groups
                .iter()
                .all(|(group, voters)| plan.get(group).is_some_and(|node| voters.contains(node))),
        "leader plan must name a voter for every layout group"
    );
    ensure!(
        !args.ready_file.try_exists()?,
        "bootstrap receipt already exists"
    );
    // Atomically claim a new directory. Existing stores, including failed
    // attempts, are never reopened by this launcher.
    fs::create_dir(&config.directory).context("benchmark requires a fresh WAL directory")?;
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .worker_threads(usize::from(args.io_threads))
        .max_blocking_threads(usize::from(args.blocking_threads))
        .build()?
        .block_on(async {
            let mut service = Service::start(config.clone()).await?;
            let handle = service.handle();
            let startup = async {
                let observed = tokio::time::timeout(
                    Duration::from_millis(args.bootstrap_timeout_ms),
                    bootstrap(&handle, &config, &plan),
                )
                .await
                .context("bootstrap deadline exceeded; do not seed this attempt")??;
                let mut receipt = fs::File::create_new(&args.ready_file)?;
                serde_json::to_writer(
                    &mut receipt,
                    &serde_json::json!({"node": config.node, "observed_leader_terms": observed}),
                )?;
                receipt.sync_all()?;
                Ok::<_, anyhow::Error>(())
            };
            let started = tokio::select! {
                result=startup=>result,
                result=signal()=>return result.and(service.shutdown().await),
                result=service.wait()=>return result.and(Err(anyhow::anyhow!("service stopped during bootstrap"))),
            };
            if let Err(error) = started {
                let stopped = service.shutdown().await;
                stopped.context(format!("bootstrap failed: {error:#}"))?;
                return Err(error);
            }
            // The bootstrap future is gone. Nothing campaigns or enforces the
            // plan during the workload; ordinary Raft elections remain enabled.
            tokio::select! {
                result=signal()=>result.and(service.shutdown().await),
                result=service.wait()=>result,
            }
        })
}
