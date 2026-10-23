//! Shared change notification, never task ownership. Every wake still performs
//! the existing authoritative bounded claim. A finite fallback covers watch
//! interruption, time-based eligibility, and changes during subscription setup.
use etcd_client::{Client, WatchOptions};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{watch, Semaphore};

pub(crate) const RECONCILE_INTERVAL: Duration = Duration::from_secs(5);

// Do not watch progress/work-report keys: those change every heartbeat without
// changing eligibility. Only queue, dependency and ownership transitions wake
// the pools. All ranges share one gRPC stream per master, not one per pool.
const RANGES: &[&str] = &[
    "queue",
    "tasks",
    "running",
    "claims",
    "target-locks",
    "merge-executions",
    "merge-claims",
    "catchup-active",
    "compact-preparations",
    "compact-commit-ready",
];

pub(crate) fn subscribe(client: Client, prefix: String) -> watch::Receiver<u64> {
    let (changed, receiver) = watch::channel(0_u64);
    tokio::spawn(async move {
        loop {
            let result = tokio::select! {
                _ = changed.closed() => return,
                result = observe(client.clone(), &prefix, &changed) => result,
            };
            if let Err(error) = result {
                metrics::counter!("master_scheduler_watch_errors_total").increment(1);
                tracing::warn!(%error, "scheduler change stream interrupted; bounded reconciliation continues");
            }
            tokio::select! {
                _ = changed.closed() => return,
                _ = tokio::time::sleep(RECONCILE_INTERVAL) => {},
            }
        }
    });
    receiver
}

async fn observe(
    mut client: Client,
    prefix: &str,
    changed: &watch::Sender<u64>,
) -> Result<(), String> {
    let key = |range: &str| format!("{}/{range}/", prefix.trim_end_matches('/'));
    let mut stream = client
        .watch(key(RANGES[0]), Some(WatchOptions::new().with_prefix()))
        .await
        .map_err(|error| error.to_string())?;
    for range in &RANGES[1..] {
        stream
            .watch(key(range), Some(WatchOptions::new().with_prefix()))
            .await
            .map_err(|error| error.to_string())?;
    }
    while let Some(response) = stream.message().await.map_err(|error| error.to_string())? {
        if response.canceled() {
            return Err("scheduler watch canceled; resubscribing from current state".into());
        }
        if response.created() || !response.events().is_empty() {
            // Coalesce arbitrary event bursts into one latest in-memory value.
            // Subscription acknowledgements also reconcile any setup gap.
            changed.send_modify(|generation| *generation = generation.wrapping_add(1));
            metrics::counter!("master_scheduler_watch_notifications_total").increment(1);
        }
    }
    Err("scheduler watch stream ended".into())
}

pub(crate) async fn wait(
    receiver: &mut watch::Receiver<u64>,
    pool: &Arc<Semaphore>,
    more_pages: bool,
) {
    // Continue a bounded multi-page sweep promptly even without new events.
    let delay = if more_pages {
        Duration::from_millis(500)
    } else {
        RECONCILE_INTERVAL
    };
    // A busy fleet can emit many unrelated transitions. Notifications must
    // never increase per-pool scan frequency beyond the previous 500ms cadence.
    let not_before = tokio::time::Instant::now() + Duration::from_millis(500);
    let saturated = pool.available_permits() == 0;
    tokio::select! {
        result = receiver.changed() => {
            // A closed observer must not turn the dispatcher into a hot loop.
            if result.is_err() { tokio::time::sleep(delay).await; }
        },
        permit = pool.clone().acquire_owned(), if saturated => { drop(permit); },
        _ = tokio::time::sleep(delay) => {},
    }
    tokio::time::sleep_until(not_before).await;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn notification_during_scan_and_capacity_release_do_not_wait_for_fallback() {
        let (tx, mut rx) = watch::channel(0_u64);
        let pool = Arc::new(Semaphore::new(1));
        rx.borrow_and_update(); // beginning of authoritative scan
        tx.send_modify(|n| *n += 1); // arrives while scan is in flight
        tokio::time::timeout(Duration::from_secs(2), wait(&mut rx, &pool, false))
            .await
            .unwrap();
        let held = pool.clone().acquire_owned().await.unwrap();
        let waiter = tokio::spawn(async move {
            wait(&mut rx, &pool, false).await;
        });
        tokio::task::yield_now().await;
        drop(held);
        tokio::time::timeout(Duration::from_secs(2), waiter)
            .await
            .unwrap()
            .unwrap();
    }

    #[tokio::test]
    async fn quiet_watch_reconciles_and_partial_sweep_does_not_starve() {
        let (_tx, mut rx) = watch::channel(0_u64);
        let pool = Arc::new(Semaphore::new(1));
        let started = tokio::time::Instant::now();
        wait(&mut rx, &pool, true).await;
        assert!(started.elapsed() < Duration::from_secs(2));
        let started = tokio::time::Instant::now();
        wait(&mut rx, &pool, false).await;
        assert!(started.elapsed() >= RECONCILE_INTERVAL);
        assert!(started.elapsed() < RECONCILE_INTERVAL + Duration::from_secs(2));
    }

    #[tokio::test]
    #[ignore = "requires isolated ETCD_TEST_ENDPOINTS"]
    async fn real_etcd_watches_admission_changes_but_not_progress_heartbeats() {
        let endpoints = std::env::var("ETCD_TEST_ENDPOINTS").unwrap();
        let mut client = Client::connect(endpoints.split(',').collect::<Vec<_>>(), None)
            .await
            .unwrap();
        let prefix = format!(
            "/scheduler-notification-test/{}",
            lance_context_core::generate_id()
        );
        let mut rx = subscribe(client.clone(), prefix.clone());
        tokio::time::timeout(Duration::from_secs(5), async {
            while *rx.borrow_and_update() < RANGES.len() as u64 {
                rx.changed().await.unwrap();
            }
        })
        .await
        .unwrap();
        // Routine unchanged progress does not change claim eligibility and must
        // not wake all scheduling pools on every writer heartbeat.
        for range in ["merge-progress", "merge-work", "executors", "demand-events"] {
            client
                .put(format!("{prefix}/{range}/fixture"), "heartbeat", None)
                .await
                .unwrap();
        }
        assert!(
            tokio::time::timeout(Duration::from_millis(300), rx.changed())
                .await
                .is_err()
        );
        for range in RANGES {
            client
                .put(format!("{prefix}/{range}/fixture"), "changed", None)
                .await
                .unwrap();
            tokio::time::timeout(Duration::from_secs(2), rx.changed())
                .await
                .unwrap()
                .unwrap();
            rx.borrow_and_update();
            client
                .delete(format!("{prefix}/{range}/fixture"), None)
                .await
                .unwrap();
            tokio::time::timeout(Duration::from_secs(2), rx.changed())
                .await
                .unwrap()
                .unwrap();
            rx.borrow_and_update();
        }
        drop(rx);
        client
            .delete(
                prefix,
                Some(etcd_client::DeleteOptions::new().with_prefix()),
            )
            .await
            .unwrap();
    }
}
