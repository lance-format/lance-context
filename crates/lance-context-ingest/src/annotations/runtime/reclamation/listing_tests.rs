use super::*;
use object_store::{local::LocalFileSystem, ObjectStore, ObjectStoreExt};
use std::task::Poll;
use tokio::sync::oneshot;

mod http_polling;

fn progress(backpressured: bool) -> (watch::Sender<Progress>, watch::Receiver<Progress>) {
    watch::channel(Progress {
        through: 0,
        stopped: false,
        backpressure: backpressured.then_some(ApplyBackpressure::Staging),
    })
}

// Hold a real listing response inside the stream, as a storage SDK does while
// awaiting a page. Dropping only try_next must not discard this receiver.
async fn held_listing() -> Result<(
    BoxStream<'static, object_store::Result<ObjectMeta>>,
    oneshot::Sender<()>,
)> {
    let root = tempfile::tempdir()?;
    let store = LocalFileSystem::new_with_prefix(root.path())?;
    store
        .put(
            &Path::from("data/orphan.lance"),
            bytes::Bytes::from_static(b"object").into(),
        )
        .await?;
    let objects: Vec<_> = store.list(None).try_collect().await?;
    let (release, response) = oneshot::channel();
    let stream = async_stream::stream! {
        let _root = root;
        response.await.expect("listing response was abandoned");
        for object in objects {
            yield Ok(object);
        }
    };
    Ok((stream.boxed(), release))
}

#[tokio::test]
async fn backpressure_interrupts_pending_listing_and_retains_response() -> Result<()> {
    let (mut listing, release) = held_listing().await?;
    let (sender, mut progress) = progress(false);
    // Poll deterministically: yielding must not depend on expiry of this timer.
    let until = tokio::time::Instant::now() + Duration::from_secs(30);
    {
        let next = next_listing(&mut listing, &mut progress, until);
        tokio::pin!(next);
        assert!(futures::poll!(&mut next).is_pending());
        sender.send_modify(|p| p.through += 1);
        assert!(futures::poll!(&mut next).is_pending());
        sender.send_modify(|p| p.backpressure = Some(ApplyBackpressure::Staging));
        assert!(matches!(futures::poll!(&mut next), Poll::Ready(None)));
    }
    assert!(release.send(()).is_ok(), "original response was discarded");
    // Ready results must progress even while pressure remains asserted.
    let meta = next_listing(&mut listing, &mut progress, until)
        .await
        .expect("ready response yielded to pressure")?
        .expect("original response lost");
    assert_eq!(meta.location, Path::from("data/orphan.lance"));
    assert_eq!(meta.size, 6);
    assert!(matches!(
        next_listing(&mut listing, &mut progress, until).await,
        Some(Ok(None))
    ));
    Ok(())
}

#[tokio::test]
async fn existing_backpressure_yields_only_pending_listing() -> Result<()> {
    let (mut listing, release) = held_listing().await?;
    let (_sender, mut progress) = progress(true);
    let until = tokio::time::Instant::now() + Duration::from_secs(30);
    {
        let next = next_listing(&mut listing, &mut progress, until);
        tokio::pin!(next);
        assert!(matches!(futures::poll!(&mut next), Poll::Ready(None)));
    }
    assert!(release.send(()).is_ok());
    assert!(matches!(
        next_listing(&mut listing, &mut progress, until).await,
        Some(Ok(Some(_)))
    ));
    Ok(())
}

#[tokio::test]
async fn ready_listing_error_is_not_hidden_by_backpressure() {
    let (_sender, mut progress) = progress(true);
    let error = object_store::Error::Generic {
        store: "listing test",
        source: std::io::Error::other("failed listing response").into(),
    };
    let mut listing = stream::iter([Err(error)]).boxed();
    let until = tokio::time::Instant::now() + Duration::from_secs(30);
    let Some(Err(error)) = next_listing(&mut listing, &mut progress, until).await else {
        panic!("ready error was hidden");
    };
    assert!(error.to_string().contains("failed listing response"));
}
