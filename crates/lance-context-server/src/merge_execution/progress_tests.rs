use super::*;
use lance_context_core::merge_write_scope::{checkpoint, MergeWriteScope};
use std::collections::VecDeque;
use tokio::{sync::mpsc, task::JoinHandle, time::Instant};

type Publication = (Instant, u64);

async fn start_watch(
    scope: Arc<MergeWriteScope>,
    idle_seconds: u64,
    replies: Vec<Result<bool, String>>,
) -> (JoinHandle<String>, mpsc::UnboundedReceiver<Publication>) {
    let (tx, rx) = mpsc::unbounded_channel();
    let handle = tokio::spawn(async move {
        let mut execution = Execution::new("table", "worker", "test", 3600);
        execution.idle_timeout_secs = idle_seconds;
        let mut replies: VecDeque<_> = replies.into();
        watch_progress_with(&execution, &scope, move |sequence| {
            tx.send((Instant::now(), sequence)).unwrap();
            std::future::ready(replies.pop_front().unwrap_or(Ok(true)))
        })
        .await
    });
    tokio::task::yield_now().await;
    (handle, rx)
}

async fn work(scope: &Arc<MergeWriteScope>) {
    scope.run(async { checkpoint() }).await;
}

async fn tick() {
    tokio::time::advance(Duration::from_secs(1)).await;
    tokio::task::yield_now().await;
}

async fn stop(handle: JoinHandle<String>) {
    handle.abort();
    assert!(handle.await.unwrap_err().is_cancelled());
}

#[tokio::test(start_paused = true)]
async fn continuous_work_coalesces_to_latest_sequence_without_stopping_work() {
    let scope = MergeWriteScope::new();
    let started = Instant::now();
    let (handle, mut publications) = start_watch(scope.clone(), 120, vec![]).await;
    let mut observed = Vec::new();
    for second in 1..=30 {
        work(&scope).await;
        tick().await;
        assert!(!handle.is_finished());
        while let Ok((at, sequence)) = publications.try_recv() {
            assert_eq!(sequence, scope.completed_steps());
            observed.push((at.duration_since(started).as_secs(), second));
        }
    }
    assert_eq!(
        observed,
        vec![(1, 1), (6, 6), (11, 11), (16, 16), (21, 21), (26, 26)]
    );
    stop(handle).await;
}

#[tokio::test(start_paused = true)]
async fn short_idle_budgets_still_observe_continuous_work_and_stop_real_stalls() {
    for idle in [1, 2, 4, 8] {
        let scope = MergeWriteScope::new();
        let (handle, mut publications) = start_watch(scope.clone(), idle, vec![]).await;
        for _ in 0..idle * 2 {
            work(&scope).await;
            tick().await;
            assert!(!handle.is_finished(), "idle budget {idle}");
        }
        assert!(
            publications.try_recv().is_ok(),
            "first work must be reported"
        );
        for _ in 1..idle {
            tick().await;
            assert!(!handle.is_finished());
        }
        tick().await;
        assert!(
            handle.is_finished(),
            "publication must not extend idle budget {idle}"
        );
        assert!(handle.await.unwrap().contains("no-progress"));
    }
}

#[tokio::test(start_paused = true)]
async fn no_work_emits_no_repeated_heartbeat_and_preserves_idle_deadline() {
    let scope = MergeWriteScope::new();
    let (handle, mut publications) = start_watch(scope, 8, vec![]).await;
    for _ in 0..8 {
        tick().await;
    }
    assert!(handle.is_finished());
    assert!(handle.await.unwrap().contains("no-progress"));
    assert!(publications.try_recv().is_err());
}

#[tokio::test(start_paused = true)]
async fn resumed_work_bypasses_a_recent_publication_of_buffered_progress() {
    let scope = MergeWriteScope::new();
    let (handle, mut publications) = start_watch(scope.clone(), 120, vec![]).await;
    work(&scope).await;
    tick().await;
    assert_eq!(publications.try_recv().unwrap().1, 1);
    work(&scope).await;
    tick().await;
    for _ in 3..=6 {
        tick().await;
    }
    assert_eq!(publications.try_recv().unwrap().1, 2);
    // The publication gate now closes until t=11, but actual work last moved
    // at t=2. Resuming at t=7 must not wait behind that gate.
    work(&scope).await;
    tick().await;
    assert_eq!(publications.try_recv().unwrap().1, 3);
    stop(handle).await;
}

#[tokio::test(start_paused = true)]
async fn publication_retries_next_tick_with_latest_work_and_honors_revocation() {
    let scope = MergeWriteScope::new();
    let (handle, mut publications) = start_watch(
        scope.clone(),
        120,
        vec![
            Ok(true),
            Err("etcd unavailable".into()),
            Ok(true),
            Ok(false),
        ],
    )
    .await;
    let mut sequences = Vec::new();
    for _ in 0..12 {
        work(&scope).await;
        tick().await;
        while let Ok((_, sequence)) = publications.try_recv() {
            sequences.push(sequence);
        }
    }
    assert!(handle.is_finished());
    assert!(handle.await.unwrap().contains("ownership revoked"));
    assert_eq!(sequences, vec![1, 6, 7, 12]);
}

#[tokio::test(start_paused = true)]
async fn buffered_progress_and_failed_writes_do_not_extend_local_idle_time() {
    for fail in [false, true] {
        let scope = MergeWriteScope::new();
        let replies = if fail {
            vec![Err("unavailable".into()); 20]
        } else {
            vec![]
        };
        let (handle, _publications) = start_watch(scope.clone(), 8, replies).await;
        for _ in 0..2 {
            work(&scope).await;
            tick().await;
        }
        for _ in 3..10 {
            tick().await;
            assert!(!handle.is_finished());
        }
        tick().await;
        assert!(handle.is_finished());
        assert!(handle.await.unwrap().contains("no-progress"));
    }
}

#[tokio::test(start_paused = true)]
async fn delayed_acknowledgement_neither_hides_work_nor_renews_idle_deadline() {
    for keep_working in [false, true] {
        let scope = MergeWriteScope::new();
        let watched_scope = scope.clone();
        let (tx, mut publications) = mpsc::unbounded_channel();
        let (ack, receive_ack) = tokio::sync::oneshot::channel();
        let handle = tokio::spawn(async move {
            let mut execution = Execution::new("table", "worker", "test", 3600);
            execution.idle_timeout_secs = 8;
            let mut receive_ack = Some(receive_ack);
            watch_progress_with(&execution, &watched_scope, move |sequence| {
                tx.send(sequence).unwrap();
                let wait = receive_ack.take();
                async move {
                    if let Some(wait) = wait {
                        wait.await.unwrap();
                    }
                    Ok(true)
                }
            })
            .await
        });
        tokio::task::yield_now().await;
        work(&scope).await;
        tick().await;
        assert_eq!(publications.try_recv().unwrap(), 1);
        for _ in 2..=7 {
            if keep_working {
                work(&scope).await;
            }
            tick().await;
        }
        assert!(publications.try_recv().is_err());
        ack.send(()).unwrap();
        tokio::task::yield_now().await;
        tick().await;
        if keep_working {
            assert_eq!(publications.try_recv().unwrap(), 7);
            assert!(!handle.is_finished());
            stop(handle).await;
        } else {
            assert!(!handle.is_finished());
            tick().await;
            assert!(handle.is_finished());
            assert!(handle.await.unwrap().contains("no-progress"));
            assert!(publications.try_recv().is_err());
        }
    }
}
