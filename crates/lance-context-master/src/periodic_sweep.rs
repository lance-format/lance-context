//! Elected periodic discovery, independent of task dispatch and storage work.
//!
//! Identical policies share one persistent owner. Different policies preserve
//! their previous scheduling semantics. Followers watch only their owner key.
use std::{sync::Arc, time::Duration};

use tokio::task::JoinHandle;

use crate::{
    demand_controller::wait_for_vacancy, scheduler, state::MasterState,
    task_store::CoordinationGuard,
};

#[derive(Clone, Copy, Debug)]
enum Sweep {
    Merge,
    Compact,
}

impl Sweep {
    fn label(self) -> &'static str {
        match self {
            Self::Merge => "merge_wal",
            Self::Compact => "compact",
        }
    }

    fn interval(self, state: &MasterState) -> u64 {
        match self {
            Self::Merge => state.config.merge_wal_interval_secs,
            Self::Compact => state.config.compaction_interval_secs,
        }
    }

    fn policy_name(self, state: &MasterState) -> String {
        let policy = match self {
            Self::Merge => serde_json::json!({
                "pending": state.config.merge_wal_min_generations,
            }),
            Self::Compact => {
                let config = state.compaction_config();
                serde_json::json!({
                    "fragments": config.min_fragments,
                    "quiet_hours": config.quiet_hours,
                    "options": scheduler::compact_options_key(&config),
                })
            }
        };
        // Exact, stable JSON avoids hash collisions and excludes Pod identity,
        // mounts, and execution capacity. The task store encodes key segments.
        format!(
            "periodic-discovery-v1/{}",
            serde_json::json!({
                "kind": self.label(),
                "data": state.config.data_dir,
                "interval": self.interval(state),
                "policy": policy,
            })
        )
    }
}

pub(crate) fn spawn(state: &Arc<MasterState>) -> Vec<JoinHandle<()>> {
    [Sweep::Merge, Sweep::Compact]
        .into_iter()
        .filter(|kind| kind.interval(state) > 0)
        .map(|kind| {
            let state = state.clone();
            tokio::spawn(async move { run(state, kind).await })
        })
        .collect()
}

async fn run(state: Arc<MasterState>, kind: Sweep) {
    let name = kind.policy_name(&state);
    let key = state.task_store.coordination_key(&name);
    loop {
        if !state.admission.is_accepting() {
            return;
        }
        let election = async {
            // A healthy owner means one quiet Watch, not a lease/Txn per tick.
            wait_for_vacancy(state.task_store.etcd_client(), &key, None).await?;
            state
                .task_store
                .try_coordination_lock(&name)
                .await
                .map_err(|error| error.to_string())
        };
        let elected = tokio::select! {
            _ = state.admission.wait_closed() => return,
            result = election => result,
        };
        match elected {
            Ok(Some(owner)) => serve(&state, kind, owner).await,
            Ok(None) => continue, // Election race; watch the winner next.
            Err(error) => {
                tracing::warn!(kind = kind.label(), %error, "periodic discovery election deferred");
            }
        }
        tokio::select! {
            _ = state.admission.wait_closed() => return,
            _ = tokio::time::sleep(Duration::from_secs(5)) => {},
        }
    }
}

async fn serve(state: &Arc<MasterState>, kind: Sweep, owner: CoordinationGuard) {
    let publisher = state.task_store.discovery_publisher(&owner);
    let work = async {
        let mut ticker = tokio::time::interval(Duration::from_secs(kind.interval(state)));
        // A slow scan must not accumulate a burst of overdue duplicate scans.
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            ticker.tick().await;
            let Some(operation) = state.admission.try_admit_named("periodic_discovery") else {
                return;
            };
            operation.phase(kind.label());
            metrics::counter!("master_periodic_sweep_started_total", "kind" => kind.label())
                .increment(1);
            // Retain the old short sweep lock for manual calls and mixed-version
            // mutual exclusion. Enqueues additionally check this persistent owner.
            let result = match kind {
                Sweep::Merge => {
                    scheduler::sweep_merge_wal_with_publisher(state, Some(&publisher)).await
                }
                Sweep::Compact => {
                    scheduler::sweep_candidates_with_publisher(state, Some(&publisher)).await
                }
            };
            match result {
                Ok(queued) => {
                    operation.observed_work();
                    metrics::counter!("master_periodic_sweep_enqueued_total", "kind" => kind.label()).increment(queued as u64);
                    if queued > 0 {
                        tracing::info!(
                            kind = kind.label(),
                            queued,
                            "elected periodic sweep queued tasks"
                        );
                    }
                }
                Err(error) => {
                    metrics::counter!("master_periodic_sweep_errors_total", "kind" => kind.label())
                        .increment(1);
                    tracing::warn!(kind = kind.label(), %error, "elected periodic sweep failed");
                }
            }
        }
    };
    let lease_lost = async {
        while owner.is_live() {
            // Local keepalive state only: this timer issues no etcd operation.
            tokio::time::sleep(Duration::from_secs(1)).await;
        }
    };
    tokio::select! {
        _ = state.admission.wait_closed() => {},
        _ = lease_lost => {},
        result = wait_for_vacancy(state.task_store.etcd_client(), owner.key(), Some(owner.token())) => {
            if let Err(error) = result {
                tracing::warn!(kind = kind.label(), %error, "periodic discovery owner watch ended");
            }
        },
        _ = work => {},
    }
    // A cancelled enqueue may already be durable; normal task dedupe handles
    // it. This owner authorizes metadata only, never a merge storage commit.
    let _ = tokio::time::timeout(
        Duration::from_secs(2),
        state.task_store.release_coordination_lock(owner),
    )
    .await;
}
