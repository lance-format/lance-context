use super::*;
use bytes::BytesMut;
use http_body_util::BodyExt;
use tokio::time::{timeout_at, Instant};

pub(super) fn length(request: &Request, maximum: usize, content_type: &str) -> HttpResult<usize> {
    let headers = request.headers();
    if headers.contains_key(header::TRANSFER_ENCODING)
        || headers.get_all(header::CONTENT_LENGTH).iter().count() != 1
        || headers.get_all(header::CONTENT_TYPE).iter().count() != 1
        || headers
            .get(header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            != Some(content_type)
    {
        return Err(Failure::bad("invalid_body_headers"));
    }
    headers
        .get(header::CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|n| *n > 0 && *n <= maximum)
        .ok_or_else(|| Failure::bad("invalid_content_length"))
}

pub(super) async fn read(
    request: Request,
    maximum: usize,
    content_type: &str,
    options: &Options,
) -> HttpResult<Bytes> {
    let length = length(&request, maximum, content_type)?;
    read_exact(request, length, options).await
}
pub(super) async fn read_exact(
    request: Request,
    length: usize,
    options: &Options,
) -> HttpResult<Bytes> {
    Ok(read_exact_accounted(request, length, options).await?.0)
}

pub(super) async fn read_exact_accounted(
    request: Request,
    length: usize,
    options: &Options,
) -> HttpResult<(Bytes, Option<usize>)> {
    read_exact_admitted(request, length, options, None).await
}

pub(super) async fn read_exact_admitted(
    request: Request,
    length: usize,
    options: &Options,
    mut reservation: Option<
        Box<dyn lance_context_wal::replication::runtime::client::payload_memory::Reservation>,
    >,
) -> HttpResult<(Bytes, Option<usize>)> {
    let deadline = Instant::now() + options.body_total_timeout;
    let mut body = request.into_body();
    let mut first: Option<Bytes> = None;
    // Hyper frames may retain an unknown-size connection allocation. With a
    // shared pool, receive into one owned allocation whose capacity we know.
    // Fragmented uploads already require this single coalescing copy.
    let mut combined = if let Some(reservation) = &mut reservation {
        if reservation.bytes() < length {
            return Err(Failure::bad("undersized_upload_admission"));
        }
        let buffer = BytesMut::with_capacity(length);
        if buffer.capacity() > reservation.bytes() {
            return Err(Failure::bad("upload_allocation_exceeds_admission"));
        }
        reservation.set_backed_bytes(buffer.capacity());
        Some(buffer)
    } else {
        None
    };
    let mut received = 0usize;
    loop {
        let frame = timeout_at(
            deadline.min(Instant::now() + options.body_idle_timeout),
            body.frame(),
        )
        .await
        .map_err(|_| Failure::bad("body_deadline"))?;
        let Some(frame) = frame else {
            break;
        };
        let frame = frame.map_err(|_| Failure::bad("body_disconnected"))?;
        let data = frame
            .into_data()
            .map_err(|_| Failure::bad("unexpected_body_trailers"))?;
        if data.is_empty() {
            continue;
        }
        if data.len() > length - received {
            return Err(Failure::bad("body_length_mismatch"));
        }
        received += data.len();
        if let Some(buffer) = combined.as_mut() {
            buffer.extend_from_slice(&data);
        } else if let Some(previous) = first.take() {
            let mut buffer = BytesMut::with_capacity(length);
            buffer.extend_from_slice(&previous);
            buffer.extend_from_slice(&data);
            combined = Some(buffer);
        } else {
            first = Some(data);
        }
    }
    if received != length {
        return Err(Failure::bad("body_length_mismatch"));
    }
    // Without a shared pool, a single frame keeps its existing buffer. Owned
    // reception and fragmented bodies coalesce once; validation, WAL transport
    // and local application then share the same retained Bytes.
    Ok(match combined {
        Some(buffer) => {
            let capacity = buffer.capacity();
            let body = crate::payload_memory::retain(buffer.freeze(), capacity, reservation)
                .map_err(|_| Failure::bad("upload_allocation_exceeds_admission"))?;
            (body, Some(capacity))
        }
        // HTTP frames can be slices of a larger connection buffer. Keep the
        // zero-copy path, but do not claim their length is the allocation size.
        None => (first.unwrap_or_default(), None),
    })
}

#[cfg(all(test, feature = "materializer"))]
mod tests {
    use super::*;
    use lance_context_wal::replication::runtime::client::payload_memory::Reservation;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct Charge(Arc<AtomicUsize>, usize);
    impl Reservation for Charge {
        fn bytes(&self) -> usize {
            self.1
        }
        fn set_backed_bytes(&mut self, bytes: usize) {
            assert!(bytes <= self.1);
        }
    }
    impl Drop for Charge {
        fn drop(&mut self) {
            self.0.fetch_sub(self.1, Ordering::SeqCst);
        }
    }
    fn charge(held: &Arc<AtomicUsize>, bytes: usize) -> Option<Box<dyn Reservation>> {
        held.fetch_add(bytes, Ordering::SeqCst);
        Some(Box::new(Charge(held.clone(), bytes)))
    }

    #[tokio::test]
    async fn admitted_http_backing_follows_slices_and_releases_cancelled_reception() {
        let held = Arc::new(AtomicUsize::new(0));
        let parent = Bytes::from(vec![7; 8192]);
        let (body, capacity) = read_exact_admitted(
            Request::new(axum::body::Body::from(parent.slice(4096..5120))),
            1024,
            &Options::default(),
            charge(&held, 1024),
        )
        .await
        .unwrap_or_else(|_| panic!("valid admitted frame"));
        assert_eq!(capacity, Some(1024));
        assert_eq!(held.load(Ordering::SeqCst), 1024);
        assert_ne!(body.as_ptr(), parent.slice(4096..).as_ptr());
        let slice = body.slice(2..10);
        drop(body);
        drop(parent);
        assert_eq!(held.load(Ordering::SeqCst), 1024);
        assert_eq!(slice.as_ref(), &[7; 8]);
        drop(slice);
        assert_eq!(held.load(Ordering::SeqCst), 0);

        let (send, mut receive) = tokio::sync::mpsc::channel::<Bytes>(1);
        send.send(Bytes::from_static(&[7; 64])).await.unwrap();
        let stream = async_stream::stream! {
            while let Some(bytes) = receive.recv().await {
                yield Ok::<_, std::io::Error>(bytes);
            }
        };
        let options = Options::default();
        let pending = read_exact_admitted(
            Request::new(axum::body::Body::from_stream(stream)),
            1024,
            &options,
            charge(&held, 1024),
        );
        assert!(tokio::time::timeout(Duration::from_millis(20), pending)
            .await
            .is_err());
        assert_eq!(held.load(Ordering::SeqCst), 0);
        drop(send);
        let truncated = read_exact_admitted(
            Request::new(axum::body::Body::from(Bytes::from_static(&[7; 64]))),
            1024,
            &Options::default(),
            charge(&held, 1024),
        )
        .await;
        assert!(truncated.is_err());
        assert_eq!(held.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn receive_keeps_shared_frames_unknown_and_counts_coalesced_allocations() {
        let parent = Bytes::from(vec![7; 8192]);
        let slice = parent.slice(4096..5120);
        let pointer = slice.as_ptr();
        let (received, capacity) = read_exact_accounted(
            Request::new(axum::body::Body::from(slice)),
            1024,
            &Options::default(),
        )
        .await
        .unwrap_or_else(|_| panic!("valid single frame"));
        assert_eq!(received.as_ptr(), pointer);
        assert_eq!(capacity, None);
        let chunks = [parent.slice(0..512), parent.slice(512..1024)].map(Ok::<_, std::io::Error>);
        let (received, capacity) = read_exact_accounted(
            Request::new(axum::body::Body::from_stream(futures::stream::iter(chunks))),
            1024,
            &Options::default(),
        )
        .await
        .unwrap_or_else(|_| panic!("valid fragmented body"));
        assert_eq!(received.as_ref(), &[7; 1024]);
        assert_eq!(capacity, Some(1024));
        assert_ne!(received.as_ptr(), parent.as_ptr());
    }
}
