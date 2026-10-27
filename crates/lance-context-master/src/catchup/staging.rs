//! Bounded immutable staging with a single ordered publisher.
//!
//! Reserve in input order before opening readers. Extra admissions never wait
//! for memory while holding earlier prepared data. All immutable operations in
//! a wave are joined, including on failure, before publication or return.
use super::{executor::MERGE_BUDGET_BUSY, Result};
use futures::future::join_all;
use lance_context_core::{MergeMemoryBudget, MergeReservation};
use std::{collections::VecDeque, future::Future, sync::Arc};

pub(super) async fn run<I, P, F, C>(
    items: I,
    concurrency: usize,
    budget: Arc<MergeMemoryBudget>,
    max_bytes: usize,
    mut prepare: P,
) -> Result<usize>
where
    I: IntoIterator,
    I::Item: Clone,
    P: FnMut(I::Item, MergeReservation) -> F,
    F: Future<Output = Result<C>>,
    C: Future<Output = Result<usize>>,
{
    if !(1..=8).contains(&concurrency) {
        return Err("immutable WAL staging concurrency must be between 1 and 8".into());
    }
    let allowance = if max_bytes == 0 {
        budget.limit()
    } else {
        max_bytes.min(budget.limit())
    };
    let mut remaining: VecDeque<_> = items.into_iter().collect();
    let mut total = 0;
    while let Some(first) = remaining.pop_front() {
        // No earlier result remains held at this await.
        let first_reservation = budget.reserve(allowance).await;
        let mut wave = vec![(first, first_reservation)];
        while wave.len() < concurrency && !remaining.is_empty() {
            let Some(reservation) = budget.try_reserve(allowance) else {
                break;
            };
            wave.push((remaining.pop_front().unwrap(), reservation));
        }
        metrics::histogram!("master_catchup_staging_wave_size").record(wave.len() as f64);
        let retry_items: Vec<_> = wave.iter().map(|(item, _)| item.clone()).collect();
        // Parent-polled futures inherit the maintenance fence/progress scope.
        // Do not use try_join_all: an error must not abandon admitted encoders.
        let prepared = join_all(
            wave.into_iter()
                .map(|(item, reservation)| prepare(item, reservation)),
        )
        .await;
        let non_memory_error = prepared.iter().enumerate().find_map(|(index, result)| {
            result
                .as_ref()
                .err()
                .filter(|e| !e.contains(MERGE_BUDGET_BUSY))
                .map(|e| (index, e.clone()))
        });
        if prepared
            .iter()
            .any(|r| r.as_ref().is_err_and(|e| e.contains(MERGE_BUDGET_BUSY)))
        {
            // A complete first generation may exceed its initial allowance.
            // Release every speculative result before retrying alone; later
            // results must not hold memory needed by the first publisher.
            drop(prepared);
            metrics::counter!("master_catchup_staging_memory_retries_total").increment(1);
            for (index, item) in retry_items.into_iter().enumerate() {
                // Preserve a known storage failure instead of retrying it. Only
                // preceding memory-contended stages may be prepared again.
                if let Some((failed, error)) = &non_memory_error {
                    if index == *failed {
                        return Err(error.clone());
                    }
                }
                let reservation = budget.reserve(allowance).await;
                total += prepare(item, reservation).await?.await?;
            }
        } else {
            // Files are immutable; deletion masks are prepared against the
            // latest base only when this one ordered committer is polled.
            for commit in prepared {
                total += commit?.await?;
            }
        }
    }
    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::future::BoxFuture;
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Mutex,
    };
    use tokio::sync::{Barrier, Notify};
    const MIB: usize = 1024 * 1024;

    #[tokio::test]
    async fn encoding_overlaps_but_publication_is_ordered() {
        let budget = MergeMemoryBudget::new(2 * MIB);
        let barrier = Arc::new(Barrier::new(2));
        let second_ready = Arc::new(Notify::new());
        let stages = Arc::new(Mutex::new(Vec::new()));
        let commits = Arc::new(Mutex::new(Vec::new()));
        let active = Arc::new(AtomicUsize::new(0));
        let run = run(0..2, 2, budget.clone(), MIB, |i, reservation| {
            let (barrier, second_ready, stages, commits, active) = (
                barrier.clone(),
                second_ready.clone(),
                stages.clone(),
                commits.clone(),
                active.clone(),
            );
            async move {
                // Neither encoder can reach completion unless both are polled.
                barrier.wait().await;
                if i == 0 {
                    second_ready.notified().await;
                }
                stages.lock().unwrap().push(i);
                if i == 1 {
                    second_ready.notify_one();
                }
                Ok(Box::pin(async move {
                    let _held = reservation;
                    assert_eq!(active.fetch_add(1, Ordering::SeqCst), 0);
                    tokio::task::yield_now().await;
                    commits.lock().unwrap().push(i);
                    assert_eq!(active.fetch_sub(1, Ordering::SeqCst), 1);
                    Ok(1usize)
                }) as BoxFuture<'static, Result<usize>>)
            }
        });
        assert_eq!(
            tokio::time::timeout(std::time::Duration::from_secs(2), run)
                .await
                .unwrap()
                .unwrap(),
            2
        );
        assert_eq!(*stages.lock().unwrap(), [1, 0]);
        assert_eq!(*commits.lock().unwrap(), [0, 1]);
        assert_eq!(budget.reserved(), 0);
    }

    #[tokio::test]
    async fn failure_joins_later_encoder_before_releasing_ownership() {
        let budget = MergeMemoryBudget::new(2 * MIB);
        let started = Arc::new(Notify::new());
        let release = Arc::new(Notify::new());
        let finished = Arc::new(AtomicUsize::new(0));
        let published = Arc::new(AtomicUsize::new(0));
        let task = run(0..3, 2, budget.clone(), MIB, |i, reservation| {
            let (started, release, finished, published) = (
                started.clone(),
                release.clone(),
                finished.clone(),
                published.clone(),
            );
            async move {
                if i == 0 {
                    return Err("encoding failed".to_string());
                }
                assert_eq!(i, 1, "failure must close later wave admission");
                started.notify_one();
                release.notified().await;
                finished.fetch_add(1, Ordering::SeqCst);
                Ok(Box::pin(async move {
                    let _held = reservation;
                    published.fetch_add(1, Ordering::SeqCst);
                    Ok(1usize)
                }) as BoxFuture<'static, Result<usize>>)
            }
        });
        tokio::pin!(task);
        tokio::select! {
            result = &mut task => panic!("returned before encoder joined: {result:?}"),
            _ = started.notified() => {}
        }
        assert_eq!(finished.load(Ordering::SeqCst), 0);
        assert_eq!(budget.reserved(), MIB);
        release.notify_one();
        assert_eq!(task.await.unwrap_err(), "encoding failed");
        assert_eq!(finished.load(Ordering::SeqCst), 1);
        assert_eq!(published.load(Ordering::SeqCst), 0);
        assert_eq!(budget.reserved(), 0);
    }

    #[tokio::test]
    async fn oversized_first_generation_retries_without_later_reservations() {
        let budget = MergeMemoryBudget::new(2 * MIB);
        let attempts = Arc::new(AtomicUsize::new(0));
        let commits = Arc::new(Mutex::new(Vec::new()));
        let task = run(0..2, 2, budget.clone(), MIB, |i, mut reservation| {
            let (attempts, commits) = (attempts.clone(), commits.clone());
            async move {
                attempts.fetch_add(1, Ordering::SeqCst);
                if !reservation.try_grow_to(3 * MIB) {
                    return Err(MERGE_BUDGET_BUSY.into());
                }
                Ok(Box::pin(async move {
                    let _held = reservation;
                    commits.lock().unwrap().push(i);
                    Ok(1usize)
                }) as BoxFuture<'static, Result<usize>>)
            }
        });
        assert_eq!(
            tokio::time::timeout(std::time::Duration::from_secs(2), task)
                .await
                .unwrap()
                .unwrap(),
            2
        );
        assert_eq!(*commits.lock().unwrap(), [0, 1]);
        assert_eq!(attempts.load(Ordering::SeqCst), 4);
        assert_eq!(budget.reserved(), 0);
    }

    #[tokio::test]
    async fn one_batch_budget_makes_progress_with_large_concurrency() {
        let budget = MergeMemoryBudget::new(MIB);
        let active = Arc::new(AtomicUsize::new(0));
        let task = run(0..5, 8, budget.clone(), MIB, |_, reservation| {
            let active = active.clone();
            async move {
                assert_eq!(active.fetch_add(1, Ordering::SeqCst), 0);
                tokio::task::yield_now().await;
                Ok(Box::pin(async move {
                    let _held = reservation;
                    assert_eq!(active.fetch_sub(1, Ordering::SeqCst), 1);
                    Ok(1usize)
                }) as BoxFuture<'static, Result<usize>>)
            }
        });
        assert_eq!(
            tokio::time::timeout(std::time::Duration::from_secs(2), task)
                .await
                .unwrap()
                .unwrap(),
            5
        );
        assert_eq!(budget.reserved(), 0);
    }

    #[tokio::test]
    async fn commit_failure_drops_staged_successors_without_publishing_or_admitting_more() {
        let budget = MergeMemoryBudget::new(2 * MIB);
        let staged = Arc::new(AtomicUsize::new(0));
        let published = Arc::new(AtomicUsize::new(0));
        let task = run(0..4, 2, budget.clone(), MIB, |i, reservation| {
            let (staged, published) = (staged.clone(), published.clone());
            async move {
                staged.fetch_add(1, Ordering::SeqCst);
                Ok(Box::pin(async move {
                    let _held = reservation;
                    assert_eq!(i, 0, "later stage must not publish after commit failure");
                    published.fetch_add(1, Ordering::SeqCst);
                    Err("unknown commit outcome".to_string())
                }) as BoxFuture<'static, Result<usize>>)
            }
        });
        assert_eq!(task.await.unwrap_err(), "unknown commit outcome");
        assert_eq!(staged.load(Ordering::SeqCst), 2);
        assert_eq!(published.load(Ordering::SeqCst), 1);
        assert_eq!(budget.reserved(), 0);
    }
    #[tokio::test]
    async fn later_read_failure_preserves_completed_prefix_progress() {
        let budget = MergeMemoryBudget::new(3 * MIB);
        let published = Arc::new(Mutex::new(Vec::new()));
        let result = run(0..4, 3, budget.clone(), MIB, |i, reservation| {
            let published = published.clone();
            async move {
                if i == 1 {
                    return Err("unreadable generation".to_string());
                }
                assert!(i < 3, "failure closes next wave admission");
                Ok(Box::pin(async move {
                    let _held = reservation;
                    published.lock().unwrap().push(i);
                    Ok(1usize)
                }) as BoxFuture<'static, Result<usize>>)
            }
        })
        .await;
        assert_eq!(result.unwrap_err(), "unreadable generation");
        assert_eq!(*published.lock().unwrap(), [0]);
        assert_eq!(budget.reserved(), 0);
    }
    #[test]
    fn staging_is_opt_in_and_concurrency_is_bounded_even_for_resident_masters() {
        let mut config = super::super::CatchupConfig::default();
        assert!(config.staging_targets.is_empty());
        assert!(config.validate().is_ok());
        config.staging_targets = vec!["bench".into()];
        for width in [0, 9] {
            config.staging_concurrency = width;
            assert!(config.validate().is_err());
        }
        config.staging_concurrency = 2;
        assert!(config.validate().is_ok());
        config.staging_targets = vec!["*".into()];
        assert!(config.validate().is_err());
    }
}
