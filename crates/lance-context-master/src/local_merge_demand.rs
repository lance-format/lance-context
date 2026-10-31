//! Optional successor suppression after joined, successful native storage work.
//! This deliberately rechecks manifests: zero reclaimed generations can also
//! mean skipped admission or a partial pass, and is never an emptiness proof.
use crate::{state::MasterState, task_store::TaskClaim};
use futures::future::join_all;
use lance_context_core::rollout_append::SealedWalProbe;
use std::{sync::Arc, time::Duration};

pub(crate) async fn acknowledge_empty_demand(state: &Arc<MasterState>, claim: &TaskClaim) {
    let target = &claim.task.target;
    if !state
        .config
        .local_merge
        .master_local_merge_empty_demand_targets
        .iter()
        .any(|t| t == "*" || t == target)
        || !super::selected(&state.config, target)
        || state.config.catchup.shards.is_empty()
        || super::drain_requested(state, target)
    {
        return;
    }
    let Some(local) = &state.local_append else {
        return;
    };
    let started = tokio::time::Instant::now();
    // Bound extra read work separately from productive merge admission. Only
    // read-only futures are cancelled by this timeout, never a mutating RPC.
    let observed = tokio::time::timeout(Duration::from_secs(2), async {
        let coordinator = state.task_store.merge_coordinator();
        let Some(request) = coordinator.get_merge_request(target).await? else {
            return Ok::<_, String>((None, "no_request"));
        };
        let uri = match target.strip_prefix("generic:") {
            Some(name) => state.generic_uri(name),
            None => state.rollout_uri(target),
        };
        let probe = SealedWalProbe::open(&uri, Some(local.session.clone()))
            .await
            .map_err(|e| e.to_string())?;
        for shards in state.config.catchup.shards.chunks(4) {
            let results = join_all(shards.iter().map(|shard| probe.has_sealed_wal(shard))).await;
            for result in results {
                if result.map_err(|e| e.to_string())? {
                    return Ok((None, "pending"));
                }
            }
            #[cfg(test)]
            testing::after_batch(shards.last().expect("nonempty chunk")).await;
        }
        Ok((Some(request), "empty"))
    })
    .await;
    let outcome = match observed {
        Ok(Ok((Some(request), _))) if !super::drain_requested(state, target) => {
            // An unknown response is not retried. The exact revision CAS may
            // have succeeded; either outcome leaves later writer demand intact.
            match state
                .task_store
                .acknowledge_empty_merge_request(claim, &request)
                .await
            {
                Ok(true) => "acknowledged",
                Ok(false) => "ownership_or_request_changed",
                Err(_) => "acknowledgement_unavailable",
            }
        }
        Ok(Ok((Some(_), _))) => "draining",
        Ok(Ok((None, reason))) => reason,
        Ok(Err(_)) => "observation_unavailable",
        Err(_) => "observation_timeout",
    };
    metrics::counter!("master_local_merge_empty_demand_total", "outcome" => outcome).increment(1);
    metrics::histogram!("master_local_merge_empty_demand_seconds", "outcome" => outcome)
        .record(started.elapsed().as_secs_f64());
}

/// A task-scoped barrier lets integration tests flush real WAL after a chosen
/// shard was observed empty. It is absent from production builds and cannot
/// pause unrelated tests or background tasks.
#[cfg(test)]
pub(crate) mod testing {
    use std::sync::Arc;
    use tokio::sync::Notify;

    #[derive(Clone)]
    pub(crate) struct Pause {
        pub last_shard: String,
        pub reached: Arc<Notify>,
        pub resume: Arc<Notify>,
    }

    tokio::task_local! {
        pub(crate) static PAUSE: Pause;
    }

    pub(super) async fn after_batch(last_shard: &str) {
        let pause = PAUSE
            .try_with(|p| (p.last_shard == last_shard).then(|| p.clone()))
            .ok()
            .flatten();
        if let Some(pause) = pause {
            pause.reached.notify_one();
            pause.resume.notified().await;
        }
    }
}
