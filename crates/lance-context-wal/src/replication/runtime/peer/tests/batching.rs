use super::*;
use std::{
    io::{self, IoSlice},
    pin::Pin,
    task::{Context, Poll},
};

struct Writer {
    bytes: Vec<u8>,
    writes: usize,
    flushes: usize,
    vectored: bool,
    max_write: usize,
    fail_flush: bool,
    pointers: Vec<usize>,
}
impl Writer {
    fn new(vectored: bool, max_write: usize) -> Self {
        Self {
            bytes: Vec::new(),
            writes: 0,
            flushes: 0,
            vectored,
            max_write,
            fail_flush: false,
            pointers: Vec::new(),
        }
    }
    fn accept(&mut self, slices: &[IoSlice<'_>]) -> Poll<io::Result<usize>> {
        self.writes += 1;
        let mut left = self.max_write;
        let mut written = 0;
        for slice in slices {
            self.pointers.push(slice.as_ptr() as usize);
            let size = left.min(slice.len());
            self.bytes.extend_from_slice(&slice[..size]);
            written += size;
            left -= size;
            if left == 0 {
                break;
            }
        }
        Poll::Ready(Ok(written))
    }
}
impl AsyncWrite for Writer {
    fn poll_write(
        mut self: Pin<&mut Self>,
        _: &mut Context<'_>,
        bytes: &[u8],
    ) -> Poll<io::Result<usize>> {
        self.accept(&[IoSlice::new(bytes)])
    }
    fn poll_write_vectored(
        mut self: Pin<&mut Self>,
        _: &mut Context<'_>,
        slices: &[IoSlice<'_>],
    ) -> Poll<io::Result<usize>> {
        self.accept(slices)
    }
    fn is_write_vectored(&self) -> bool {
        self.vectored
    }
    fn poll_flush(mut self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<io::Result<()>> {
        self.flushes += 1;
        if self.fail_flush {
            Poll::Ready(Err(io::ErrorKind::BrokenPipe.into()))
        } else {
            Poll::Ready(Ok(()))
        }
    }
    fn poll_shutdown(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<io::Result<()>> {
        Poll::Ready(Ok(()))
    }
}

#[tokio::test]
async fn queued_frames_share_one_tls_send_and_release_all_credits() -> Result<()> {
    let tls = certificates()?;
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let peer = Peer {
        address: listener.local_addr()?.to_string(),
        server_name: tls[&2].name.clone(),
        certificate: tls[&2].certificate,
    };
    let (send, receive) = mpsc::channel(4);
    let mut owners = Vec::new();
    let mut expected = Writer::new(true, usize::MAX);
    for _ in 0..3 {
        let (queued, credit, bytes, _) = timing::queued()?;
        write_packet(&mut expected, queued.packet.packet()).await?;
        send.try_send(queued)?;
        owners.push((credit, bytes));
    }
    drop(send);
    let acceptor = TlsAcceptor::from(tls[&2].server.clone());
    let receiver = async {
        let (tcp, _) = listener.accept().await?;
        let mut stream = acceptor.accept(tcp).await?;
        let mut actual = vec![0; expected.bytes.len()];
        stream.read_exact(&mut actual).await?;
        assert_eq!(actual, expected.bytes);
        Ok::<_, anyhow::Error>(())
    };
    let statistics = Arc::new(Statistics::default());
    let destination = Arc::new(DestinationStatistics::default());
    let (_stop, stopped) = watch::channel(false);
    let sender = send_loop(
        peer,
        receive,
        Arc::new(AtomicBool::new(false)),
        TlsConnector::from(tls[&1].client.clone()),
        Options {
            send_batch_frames: 8,
            ..Options::default()
        },
        stopped,
        (statistics.clone(), destination.clone()),
    );
    timeout(Duration::from_secs(5), async {
        tokio::try_join!(sender, receiver)
    })
    .await??;
    assert_eq!(destination.sent_batches.load(Ordering::Relaxed), 1);
    assert_eq!(destination.sent_frames.load(Ordering::Relaxed), 3);
    assert_eq!(statistics.sent_frames.load(Ordering::Relaxed), 3);
    for (credit, bytes) in owners {
        assert_eq!(credit.observation().used_requests, 0);
        assert!(bytes.available_permits() > 0);
    }
    Ok(())
}

#[tokio::test]
async fn coalescing_preserves_frame_order_and_payload_owners_with_one_flush() -> Result<()> {
    let mut frames = Vec::new();
    let mut owners = Vec::new();
    let mut expected = Writer::new(true, usize::MAX);
    for group in [10, 20, 10] {
        let (mut queued, credits, bytes, body) = timing::queued()?;
        queued.packet.packet.group = group;
        write_packet(&mut expected, queued.packet.packet()).await?;
        frames.push(queued);
        owners.push((credits, bytes, body));
    }
    for (vectored, max_write) in [(true, usize::MAX), (true, 17), (false, 17)] {
        let mut writer = Writer::new(vectored, max_write);
        let statistics = DestinationStatistics::default();
        send_queued(&mut writer, &frames, &statistics).await?;
        assert_eq!(writer.bytes, expected.bytes);
        assert_eq!(writer.flushes, 1);
        assert_eq!(statistics.sent_batches.load(Ordering::Relaxed), 1);
        assert_eq!(statistics.sent_frames.load(Ordering::Relaxed), 3);
        assert_eq!(statistics.queue_wait.snapshot().finished, 3);
        assert_eq!(statistics.send.snapshot().finished, 1);
        if max_write == usize::MAX {
            assert_eq!(writer.writes, 1);
            assert_eq!(expected.writes, 3);
            for (_, _, body) in &owners {
                assert!(writer.pointers.contains(&(body.as_ptr() as usize)));
            }
        }
        for (credits, bytes, _) in &owners {
            assert_eq!(credits.observation().used_requests, 1);
            assert_eq!(bytes.available_permits(), 0);
        }
    }
    drop(frames);
    for (credits, bytes, _) in owners {
        assert_eq!(credits.observation().used_requests, 0);
        assert!(bytes.available_permits() > 0);
    }
    Ok(())
}

#[test]
fn collection_preserves_fifo_at_count_and_byte_limits_and_allows_a_large_first_frame() -> Result<()>
{
    let (send, mut receive) = mpsc::channel(4);
    let mut frames = Vec::new();
    for group in 1..=4 {
        let (mut queued, _, _, _) = timing::queued()?;
        queued.packet.packet.group = group;
        if group == 1 {
            frames.push(queued);
        } else {
            send.try_send(queued)?;
        }
    }
    let size = frames[0]._bytes.num_permits();
    let mut options = Options {
        send_batch_frames: 2,
        send_batch_bytes: size * 4,
        ..Options::default()
    };
    assert!(fill_batch(&mut frames, &mut receive, &options).is_none());
    assert_eq!(frames.len(), 2);
    frames.clear();
    frames.push(receive.try_recv()?);
    assert_eq!(frames[0].packet.packet().group, 3);
    options.send_batch_bytes = size * 2 - 1;
    let deferred = fill_batch(&mut frames, &mut receive, &options).unwrap();
    assert_eq!(frames.len(), 1);
    assert_eq!(deferred.packet.packet().group, 4);
    assert_eq!(deferred._bytes.num_permits(), size);

    send.try_send(deferred)?;
    options.send_batch_bytes = size - 1;
    assert!(fill_batch(&mut frames, &mut receive, &options).is_none());
    assert_eq!(frames.len(), 1);
    assert_eq!(receive.try_recv()?.packet.packet().group, 4);
    // A sparse queue returns immediately, even when both targets have space.
    options.send_batch_bytes = size * 4;
    assert!(fill_batch(&mut frames, &mut receive, &options).is_none());
    Ok(())
}

#[tokio::test]
async fn failed_flush_and_cancelled_partial_batch_never_report_success_or_leak_credits(
) -> Result<()> {
    let (first, first_credit, first_bytes, _) = timing::queued()?;
    let (second, second_credit, second_bytes, _) = timing::queued()?;
    let frames = vec![first, second];
    let statistics = DestinationStatistics::default();
    let mut writer = Writer::new(true, usize::MAX);
    writer.fail_flush = true;
    assert!(send_queued(&mut writer, &frames, &statistics)
        .await
        .is_err());
    assert_eq!(writer.flushes, 1);
    assert_eq!(statistics.sent_frames.load(Ordering::Relaxed), 0);
    assert_eq!(statistics.sent_batches.load(Ordering::Relaxed), 0);
    assert_eq!(statistics.sent_payload_bytes.load(Ordering::Relaxed), 0);

    let (mut blocked, _read) = tokio::io::duplex(64);
    assert!(timeout(
        Duration::from_millis(10),
        send_queued(&mut blocked, &frames, &statistics),
    )
    .await
    .is_err());
    assert_eq!(statistics.send.snapshot().finished, 2);
    assert_eq!(statistics.sent_frames.load(Ordering::Relaxed), 0);
    for (credits, bytes) in [
        (&first_credit, &first_bytes),
        (&second_credit, &second_bytes),
    ] {
        assert_eq!(credits.observation().used_requests, 1);
        assert_eq!(bytes.available_permits(), 0);
    }
    drop(blocked);
    drop(frames);
    for (credits, bytes) in [(first_credit, first_bytes), (second_credit, second_bytes)] {
        assert_eq!(credits.observation().used_requests, 0);
        assert_eq!(credits.observation().used_bytes, 0);
        assert!(bytes.available_permits() > 0);
    }
    Ok(())
}
