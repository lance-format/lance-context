use super::*;

const FRAME_TIMEOUT: Duration = Duration::from_millis(250);

struct Receiver {
    runtime: Runtime,
    task: JoinHandle<Result<()>>,
    stop: watch::Sender<bool>,
    handle: Handle,
    stream: tokio_rustls::client::TlsStream<TcpStream>,
    statistics: Arc<Statistics>,
    _directory: tempfile::TempDir,
}

impl Receiver {
    async fn open() -> Result<Self> {
        let directory = tempfile::tempdir()?;
        let node = Node::open(directory.path(), layout(), 1, Limits::default())?;
        let (runtime, _outbox) = Runtime::spawn(node, super::super::super::Options::default())?;
        let handle = runtime.handle();
        let tls = certificates()?;
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let address = listener.local_addr()?;
        let (stop, stopped) = watch::channel(false);
        let statistics = Arc::new(Statistics::default());
        // Isolate this authenticated inbound connection from the ordinary
        // cluster's heartbeat traffic and outgoing reconnect counters.
        let task = tokio::spawn(listen(
            handle.clone(),
            listener,
            TlsAcceptor::from(tls[&1].server.clone()),
            BTreeMap::from([(tls[&2].certificate, 2)]),
            Options {
                io_timeout: FRAME_TIMEOUT,
                ..Options::default()
            },
            stopped,
            statistics.clone(),
        ));
        let tcp = TcpStream::connect(address).await?;
        tcp.set_nodelay(true)?;
        let stream = TlsConnector::from(tls[&2].client.clone())
            .connect(ServerName::try_from(tls[&1].name.clone())?, tcp)
            .await?;
        Ok(Self {
            runtime,
            task,
            stop,
            handle,
            stream,
            statistics,
            _directory: directory,
        })
    }

    async fn heartbeat(&mut self, received: u64) -> Result<()> {
        let mut frame = packet(&self.handle, 2, Bytes::new());
        frame.message.set_msg_type(MessageType::MsgHeartbeat);
        frame.message.entries.clear();
        write_packet(&mut self.stream, &frame).await?;
        timeout(Duration::from_secs(2), async {
            while self.statistics.received_frames.load(Ordering::Relaxed) < received {
                sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .context("frame after idle did not arrive on the original TLS connection")?;
        Ok(())
    }

    async fn close(self) -> Result<()> {
        // An idle read must not hold shutdown open indefinitely.
        self.stop.send(true)?;
        timeout(Duration::from_secs(2), self.task).await???;
        self.runtime.shutdown().await
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn authenticated_idle_peer_delivers_the_next_frame_without_reconnecting() -> Result<()> {
    let mut receiver = Receiver::open().await?;
    let result = async {
        receiver.heartbeat(1).await?;
        sleep(FRAME_TIMEOUT * 3).await;
        receiver.heartbeat(2).await?;
        ensure!(
            receiver.statistics.failed_exchanges.load(Ordering::Relaxed) == 0
                && receiver.statistics.rejected_packets.load(Ordering::Relaxed) == 0,
            "healthy idle peer was disconnected or rejected"
        );
        Ok(())
    }
    .await;
    receiver.close().await?;
    result
}

#[tokio::test]
async fn partial_peer_frames_keep_the_deadline_and_release_receive_credits() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let node = Node::open(directory.path(), layout(), 1, Limits::default())?;
    let (runtime, _outbox) = Runtime::spawn(node, super::super::super::Options::default())?;
    let handle = runtime.handle();
    let result = async {
        let frame = packet(&handle, 2, Bytes::from(vec![0xa5; 1024]));
        // Capture the actual wire bytes, including write_frame's length prefix.
        let (mut output, mut input) = tokio::io::duplex(4096);
        write_packet(&mut output, &frame).await?;
        drop(output);
        let mut wire = Vec::new();
        input.read_to_end(&mut wire).await?;
        let header_end = 4 + u32::from_be_bytes(wire[..4].try_into()?) as usize;
        // Partial length, header and payload must all expire, even when the
        // sender keeps its connection open. Header/body credits cannot leak.
        for prefix in [1, 3, 4, header_end - 1, header_end, wire.len() - 1] {
            let (mut write, mut read) = tokio::io::duplex(4096);
            write.write_all(&wire[..prefix]).await?;
            let credits = Arc::new(Semaphore::new(4096));
            let error = timeout(
                Duration::from_secs(2),
                read_next_packet(&mut read, &handle, 2, &credits, FRAME_TIMEOUT),
            )
            .await?
            .err()
            .context("partial peer frame unexpectedly completed")?;
            ensure!(
                error.to_string() == "peer frame receive deadline",
                "partial frame at byte {prefix} failed for a different reason: {error:#}"
            );
            ensure!(
                credits.available_permits() == 4096,
                "receive credits leaked"
            );
        }
        Ok(())
    }
    .await;
    runtime.shutdown().await?;
    result
}
