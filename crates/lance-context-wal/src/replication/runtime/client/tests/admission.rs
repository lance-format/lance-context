use super::*;

struct ReadObserved {
    stream: tokio::io::DuplexStream,
    reads: Arc<AtomicU64>,
}
impl AsyncRead for ReadObserved {
    fn poll_read(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
        buffer: &mut tokio::io::ReadBuf<'_>,
    ) -> std::task::Poll<std::io::Result<()>> {
        self.reads.fetch_add(1, Ordering::Relaxed);
        std::pin::Pin::new(&mut self.stream).poll_read(cx, buffer)
    }
}
impl AsyncWrite for ReadObserved {
    fn poll_write(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
        buffer: &[u8],
    ) -> std::task::Poll<std::io::Result<usize>> {
        std::pin::Pin::new(&mut self.stream).poll_write(cx, buffer)
    }
    fn poll_flush(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<std::io::Result<()>> {
        std::pin::Pin::new(&mut self.stream).poll_flush(cx)
    }
    fn poll_shutdown(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<std::io::Result<()>> {
        std::pin::Pin::new(&mut self.stream).poll_shutdown(cx)
    }
}

/// Exercise the server handler with bounded streams and real runtime handles.
/// Withhold bodies to hold twelve reservations concurrently without proposing a
/// mutation. Appends have no response payload to reserve in addition to those
/// bodies or a generic control reply; twelve 4 MiB requests must fit the default
/// 64 MiB data budget while leaving room for the next unvalidated header.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn partial_appends_share_data_credit_without_reserving_fetch_bodies() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let c = RpcCluster::open(directory.path(), Options::default()).await?;
    let leader = c.leader().await?;
    let handle = c.net.nodes[&leader].handle.clone();
    let budgets = Arc::new(Budgets::new(&Options::default()));
    let available = budgets.data.available_permits();
    let before = handle.append_observation()["started"];
    let original = Batch::new(
        batch(1).meta.stream,
        0,
        batch(1).meta.producer,
        1,
        Bytes::from(vec![1; 4 << 20]),
    );
    let metadata = codec().serialize(&(
        scope(&handle),
        WireRequest::Append {
            group: 10,
            meta: original.meta.clone(),
        },
    ))?;
    let mut tasks = JoinSet::new();
    let mut clients = Vec::new();
    for id in 1..=12 {
        // A second payload byte cannot be written until the server has read
        // the complete metadata and begun receiving the admitted body.
        let (client, mut server) = tokio::io::duplex(1);
        clients.push(client);
        let handle = handle.clone();
        let budgets = budgets.clone();
        let meta_len = metadata.len();
        let body_len = original.body.len();
        tasks.spawn(async move {
            serve_one(
                &mut server,
                true,
                &handle,
                &budgets,
                (id, meta_len, body_len),
            )
            .await
        });
    }
    // Any handler may win admission first. Supply all metadata concurrently so
    // an admitted handler can validate and release its generic reply reserve.
    let mut writers = JoinSet::new();
    for mut client in clients.drain(..) {
        let metadata = metadata.clone();
        let prefix = original.body.slice(..2);
        writers.spawn(async move {
            client.write_all(&metadata).await?;
            client.write_all(&prefix).await?;
            Ok::<_, anyhow::Error>(client)
        });
    }
    let admitted = timeout(Duration::from_secs(3), async {
        while let Some(result) = writers.join_next().await {
            clients.push(result??);
        }
        Ok::<_, anyhow::Error>(())
    })
    .await;
    let held = budgets.data.available_permits();
    let submitted = handle.append_observation()["started"];
    writers.abort_all();
    while writers.join_next().await.is_some() {}
    tasks.abort_all();
    while tasks.join_next().await.is_some() {}
    let released = budgets.data.available_permits();
    drop(clients);
    c.shutdown().await?;

    admitted.context("twelve partial appends were blocked by response admission")??;
    assert!(held < available);
    assert_eq!(
        released, available,
        "cancellation leaked server data credit"
    );
    assert_eq!(submitted, before, "a partial body reached append execution");
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn fetch_reserves_reply_credit_and_rejects_nonempty_request_bodies() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let c = RpcCluster::open(directory.path(), Options::default()).await?;
    let leader = c.leader().await?;
    let original = batch(1);
    let response = c
        .client
        .request(
            leader,
            Request::Append {
                group: 10,
                batch: original.clone(),
            },
        )
        .await?;
    let index = accepted(&response);
    drop(response);
    let handle = &c.net.nodes[&leader].handle;
    let budgets = Budgets::new(&Options::default());
    let available = budgets.data.available_permits();
    let metadata = codec().serialize(&(scope(handle), WireRequest::Fetch { group: 10, index }))?;

    // An attacker cannot avoid fetch reply admission by declaring a payload:
    // validation must reject this before reading its body or executing Fetch.
    let (mut client, mut server) = tokio::io::duplex(MAX_META);
    client.write_all(&metadata).await?;
    let malformed = timeout(
        Duration::from_secs(3),
        serve_one(&mut server, true, handle, &budgets, (1, metadata.len(), 1)),
    )
    .await?;
    assert_eq!(
        malformed.unwrap_err().to_string(),
        "unexpected request payload"
    );
    assert_eq!(budgets.data.available_permits(), available);

    // Leave enough capacity for request/reply metadata, but not a maximum
    // fetched body. A valid Fetch must wait before it reads any metadata.
    let held = budgets
        .reserve(true, available - handle.max_record + 1)
        .await?;
    let (mut client, stream) = tokio::io::duplex(MAX_META);
    let reads = Arc::new(AtomicU64::new(0));
    let mut server = ReadObserved {
        stream,
        reads: reads.clone(),
    };
    client.write_all(&metadata).await?;
    let mut fetch = Box::pin(serve_one(
        &mut server,
        true,
        handle,
        &budgets,
        (2, metadata.len(), 0),
    ));
    let waiting = std::future::poll_fn(|cx| {
        std::task::Poll::Ready(std::future::Future::poll(fetch.as_mut(), cx).is_pending())
    })
    .await;
    let reads_before_admission = reads.load(Ordering::Relaxed);
    drop(held);
    let result = timeout(Duration::from_secs(3), fetch).await?;
    let (id, meta_len, body_len) = prefix(&mut client, handle.max_record).await?;
    let reply: (Scope, Reply) = codec().deserialize(&receive(&mut client, meta_len).await?)?;
    let body = receive(&mut client, body_len).await?;
    let released = budgets.data.available_permits();
    c.shutdown().await?;

    assert!(
        waiting,
        "fetch executed without reserving its possible reply body"
    );
    assert_eq!(
        reads_before_admission, 0,
        "fetch read before reply admission"
    );
    result?;
    assert_eq!(id, 2);
    assert!(matches!(reply.1, Reply::Record(Some(meta)) if meta == original.meta));
    assert_eq!(body, original.body);
    assert_eq!(released, available);
    Ok(())
}
