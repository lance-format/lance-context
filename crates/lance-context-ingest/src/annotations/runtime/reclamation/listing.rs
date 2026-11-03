//! Finish an already-pending LIST response while the publisher is busy.
//! Own at most one next-item result per listing. Do not enumerate ahead, start
//! another request, or classify/delete candidates outside the publisher.
use super::*;
use futures::Stream;
use std::task::{Context, Poll};
use tokio::task::JoinHandle;

type Listing = BoxStream<'static, object_store::Result<ObjectMeta>>;
type Item = Option<object_store::Result<ObjectMeta>>;
type PendingItem = JoinHandle<(Item, Listing)>;

pub(super) fn drive(stream: Listing) -> Listing {
    Driven {
        idle: Some(stream),
        pending: None,
    }
    .boxed()
}

struct Driven {
    idle: Option<Listing>,
    pending: Option<PendingItem>,
}

impl Stream for Driven {
    type Item = object_store::Result<ObjectMeta>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Item> {
        if self.pending.is_none() {
            // Preserve immediate ready-data/EOF/error handling under pressure.
            let stream = self.idle.as_mut().expect("listing stream owner");
            if let Poll::Ready(item) = stream.as_mut().poll_next(cx) {
                return Poll::Ready(item);
            }
            // Keeping the future alone is insufficient: its HTTP body deadline
            // still runs when publication stops polling. Drive only this item.
            let mut stream = self.idle.take().unwrap();
            self.pending = Some(tokio::spawn(async move {
                let item = stream.next().await;
                (item, stream)
            }));
        }
        let outcome = match Pin::new(self.pending.as_mut().unwrap()).poll(cx) {
            Poll::Pending => return Poll::Pending,
            Poll::Ready(outcome) => outcome,
        };
        self.pending = None;
        match outcome {
            Ok((item, stream)) => {
                self.idle = Some(stream);
                Poll::Ready(item)
            }
            Err(error) => {
                self.idle = Some(stream::empty().boxed());
                Poll::Ready(Some(Err(object_store::Error::Generic {
                    store: "annotation LIST",
                    source: Box::new(error),
                })))
            }
        }
    }
}

impl Drop for Driven {
    fn drop(&mut self) {
        if let Some(pending) = &self.pending {
            pending.abort();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::sync::oneshot;

    fn item_error() -> object_store::Error {
        object_store::Error::Generic {
            store: "listing test",
            source: std::io::Error::other("injected listing outcome").into(),
        }
    }

    #[tokio::test]
    async fn pending_response_does_not_enumerate_ahead() -> Result<()> {
        let count = Arc::new(AtomicUsize::new(0));
        let produced = count.clone();
        let (release, ready) = oneshot::channel();
        let (resolved, complete) = oneshot::channel();
        let source = async_stream::stream! {
            ready.await.unwrap();
            produced.fetch_add(1, Ordering::SeqCst);
            resolved.send(()).unwrap();
            yield Err(item_error());
            produced.fetch_add(1, Ordering::SeqCst);
            yield Err(item_error());
        };
        let mut listing = drive(source.boxed());
        assert!(futures::poll!(listing.next()).is_pending());
        release.send(()).unwrap();
        tokio::time::timeout(Duration::from_secs(1), complete).await??;
        assert_eq!(count.load(Ordering::SeqCst), 1);
        assert!(listing.next().await.unwrap().is_err());
        assert_eq!(count.load(Ordering::SeqCst), 1);
        assert!(listing.next().await.unwrap().is_err());
        assert_eq!(count.load(Ordering::SeqCst), 2);
        Ok(())
    }

    #[tokio::test]
    async fn dropping_listing_cancels_its_original_pending_response() -> Result<()> {
        let (mut release, ready) = oneshot::channel::<()>();
        let source = async_stream::stream! {
            let _ = ready.await;
            yield Err(item_error());
        };
        let mut listing = drive(source.boxed());
        assert!(futures::poll!(listing.next()).is_pending());
        drop(listing);
        tokio::time::timeout(Duration::from_secs(1), release.closed()).await?;
        Ok(())
    }
}
