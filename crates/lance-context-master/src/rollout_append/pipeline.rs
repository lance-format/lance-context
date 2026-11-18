//! Poll admitted staging during a commit without admitting more generations.
use futures::{future::BoxFuture, stream, Stream, StreamExt};
use std::{
    future::Future,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    task::Poll,
};

pub(super) fn staging<'a, T: 'a>(
    futures: Vec<BoxFuture<'a, T>>,
    concurrency: usize,
) -> (Arc<AtomicBool>, impl Stream<Item = T> + Unpin + 'a) {
    let admission = Arc::new(AtomicBool::new(true));
    let gate = admission.clone();
    let mut pending = futures.into_iter();
    let work = stream::poll_fn(move |_| {
        if gate.load(Ordering::Relaxed) {
            Poll::Ready(pending.next())
        } else {
            // The commit wakes the caller, which restores admission before
            // polling this stream again. Buffered staging still gets polled.
            Poll::Pending
        }
    })
    .buffer_unordered(concurrency);
    (admission, work)
}

pub(super) async fn commit_while_polling<C: Future, S: Stream + Unpin>(
    admission: &AtomicBool,
    commit: C,
    work: &mut S,
    mut collect: impl FnMut(S::Item),
) -> C::Output {
    struct Paused<'a>(&'a AtomicBool);
    impl Drop for Paused<'_> {
        fn drop(&mut self) {
            self.0.store(true, Ordering::Relaxed);
        }
    }
    admission.store(false, Ordering::Relaxed);
    let _paused = Paused(admission);
    tokio::pin!(commit);
    let mut unfinished = true;
    loop {
        tokio::select! {
            result = &mut commit => return result,
            next = work.next(), if unfinished => match next {
                Some(result) => collect(result),
                None => unfinished = false,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::FutureExt;
    use std::{sync::atomic::AtomicUsize, time::Duration};
    use tokio::sync::oneshot;

    #[tokio::test]
    async fn commit_wait_polls_admitted_staging_without_starting_the_next_generation() {
        for outcome in [Ok(1), Err("publication outcome unknown")] {
            let started = Arc::new(AtomicUsize::new(0));
            let (release, released) = oneshot::channel();
            let (finished, staged) = oneshot::channel();
            let stages: Vec<BoxFuture<'_, usize>> = vec![
                {
                    let started = started.clone();
                    async move {
                        started.fetch_add(1, Ordering::Relaxed);
                        0
                    }
                    .boxed()
                },
                {
                    let started = started.clone();
                    async move {
                        started.fetch_add(1, Ordering::Relaxed);
                        released.await.unwrap();
                        finished.send(()).unwrap();
                        1
                    }
                    .boxed()
                },
                {
                    let started = started.clone();
                    async move {
                        started.fetch_add(1, Ordering::Relaxed);
                        2
                    }
                    .boxed()
                },
            ];
            let (admission, mut work) = staging(stages, 2);
            assert_eq!(work.next().await, Some(0));
            let commit = async {
                release.send(()).unwrap();
                // Awaiting this commit without polling work cannot complete.
                staged.await.unwrap();
                tokio::task::yield_now().await;
                assert_eq!(started.load(Ordering::Relaxed), 2);
                outcome
            };
            let mut ready = Vec::new();
            let result = tokio::time::timeout(
                Duration::from_secs(2),
                commit_while_polling(&admission, commit, &mut work, |part| ready.push(part)),
            )
            .await
            .expect("admitted staging stalled behind the commit");
            assert_eq!(result, outcome);
            assert_eq!(ready, [1]);
            assert_eq!(started.load(Ordering::Relaxed), 2);
            // A failed commit returns to the caller unchanged. The production
            // caller exits its pass; no commit is silently retried here.
            if result.is_ok() {
                assert_eq!(work.next().await, Some(2));
                assert_eq!(work.next().await, None);
            }
        }
    }
}
