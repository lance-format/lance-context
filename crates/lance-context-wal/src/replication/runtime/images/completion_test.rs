use super::*;
use crate::replication::tests::{batch, layout, pump};
use crate::replication::Limits;
use std::task::{Context, Wake, Waker};

struct AdmissionAtWake {
    owner: Arc<crate::replication::images::Owner>,
    woke: AtomicBool,
    admitted: AtomicBool,
}

impl Wake for AdmissionAtWake {
    fn wake(self: Arc<Self>) {
        self.wake_by_ref();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        // A caller can run on a sibling worker as soon as the reply wakes it.
        // Inspect physical-node admission synchronously at that exact boundary.
        let mut admission = Box::pin(self.owner.acquire_queued());
        let ready = matches!(
            admission
                .as_mut()
                .poll(&mut Context::from_waker(Waker::noop())),
            Poll::Ready(Ok(_))
        );
        self.admitted.store(ready, Ordering::Release);
        self.woke.store(true, Ordering::Release);
    }
}

#[test]
fn checkpoint_completion_releases_node_permit_before_waking_caller() -> Result<()> {
    for success in [true, false] {
        let root = tempfile::tempdir()?;
        let mut nodes = (1..=3)
            .map(|id| {
                Node::open(
                    &root.path().join(id.to_string()),
                    layout(),
                    id,
                    Limits::default(),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        nodes[0].campaign(10)?;
        pump(&mut nodes)?;
        nodes[0].propose(10, 1, batch(1))?;
        pump(&mut nodes)?;
        let image = nodes[0]
            .prepare_checkpoint(10, uuid::Uuid::new_v4())?
            .run()?;
        let observed = Arc::new(AdmissionAtWake {
            owner: nodes[0].images.clone(),
            woke: AtomicBool::new(false),
            admitted: AtomicBool::new(false),
        });
        let waker = Waker::from(observed.clone());
        let (reply, mut receive) = oneshot::channel();
        let credits = Credits::new(1, 8192);
        let work = Box::new(CheckpointWork {
            image,
            retained_bytes: 0,
            started: true,
            previous: None,
            reply: Completion {
                reply,
                _credit: credits.reserve(4096)?,
            },
        });
        assert!(Pin::new(&mut receive)
            .poll(&mut Context::from_waker(&waker))
            .is_pending());
        work.finish(if success {
            Ok(())
        } else {
            Err(anyhow::anyhow!("checkpoint failed"))
        });
        assert!(observed.woke.load(Ordering::Acquire));
        assert!(
            observed.admitted.load(Ordering::Acquire),
            "completion woke caller before releasing the node image permit"
        );
        let Poll::Ready(Ok(completed)) =
            Pin::new(&mut receive).poll(&mut Context::from_waker(Waker::noop()))
        else {
            anyhow::bail!("completion lost its result");
        };
        assert_eq!(completed.result.is_ok(), success);
    }
    Ok(())
}
