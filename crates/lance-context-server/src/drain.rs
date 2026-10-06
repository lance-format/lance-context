//! Join accepted handlers before retiring stalled response connections. A slow
//! reader must not pin shutdown, and a disconnected client must not cancel a
//! handler halfway through publishing a storage mutation. Response bodies remain
//! cancellable: blob bodies own their data, and staged-append streams only write
//! unreachable immutable files (their separate coordinator publishes them).
use tracing::Instrument;

use axum::{
    extract::{Request, State},
    http::StatusCode,
    middleware::Next,
    response::{IntoResponse, Response},
    serve::Listener,
    Router,
};
use std::{
    future::Future,
    io,
    net::{Shutdown, SocketAddr},
    pin::Pin,
    sync::{Arc, Mutex, Weak},
    task::{Context, Poll},
    time::Duration,
};
use tokio::{
    io::{AsyncRead, AsyncWrite, ReadBuf},
    net::{TcpListener, TcpStream},
    sync::Notify,
    time::Instant,
};

/// Serve until shutdown, joining storage handlers before retiring blocked
/// response sockets. Returning also joins handlers detached by client disconnect.
pub(crate) async fn serve(
    listener: TcpListener,
    app: Router,
    response_idle: Duration,
    shutdown: impl Future<Output = ()> + Send + 'static,
) -> io::Result<()> {
    let drain = Arc::new(Drain::default());
    let app = app.layer(axum::middleware::from_fn_with_state(
        drain.clone(),
        track_handler,
    ));
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();
    let response_drain = {
        let drain = drain.clone();
        tokio::spawn(async move {
            if shutdown_rx.await.is_ok() {
                drain.retire_idle_responses(response_idle).await;
            }
        })
    };
    let admission = drain.clone();
    let result = axum::serve(DrainListener::new(listener, drain.clone()), app)
        .with_graceful_shutdown(async move {
            shutdown.await;
            admission.close_admission();
            let _ = shutdown_tx.send(());
        })
        .await;
    response_drain.abort();
    let _ = response_drain.await;
    drain.close_admission();
    drain.joined().await;
    result
}

#[derive(Default)]
struct Admission {
    closed: bool,
    active: usize,
}
#[derive(Default)]
pub(crate) struct Drain {
    admission: Mutex<Admission>,
    changed: Notify,
    connections: Mutex<Vec<Weak<Connection>>>,
}
struct Handler(Arc<Drain>);
impl Drop for Handler {
    fn drop(&mut self) {
        let mut admission = self.0.admission.lock().unwrap();
        admission.active -= 1;
        self.0.changed.notify_waiters();
    }
}
impl Drain {
    fn admit(self: &Arc<Self>) -> Option<Handler> {
        let mut admission = self.admission.lock().unwrap();
        if admission.closed {
            return None;
        }
        admission.active += 1;
        Some(Handler(self.clone()))
    }
    pub(crate) fn close_admission(&self) {
        let mut admission = self.admission.lock().unwrap();
        if !admission.closed {
            tracing::info!(
                active_http_handlers = admission.active,
                "closed HTTP admission; joining accepted work"
            );
            admission.closed = true;
            self.changed.notify_waiters();
        }
    }
    pub(crate) async fn joined(&self) {
        loop {
            let changed = self.changed.notified();
            tokio::pin!(changed);
            changed.as_mut().enable();
            if self.admission.lock().unwrap().active == 0 {
                return;
            }
            changed.await;
        }
    }
    /// Invoke only after admission closes. No handler time limit: all accepted
    /// writes finish before any response connection can be forcibly closed.
    pub(crate) async fn retire_idle_responses(&self, idle: Duration) {
        self.joined().await;
        tracing::info!("accepted HTTP handlers joined; draining response connections");
        loop {
            {
                let mut connections = self.connections.lock().unwrap();
                connections.retain(|weak| {
                    let Some(connection) = weak.upgrade() else {
                        return false;
                    };
                    // Serialize the idle check/close with actual socket writes,
                    // so a concurrent successful write cannot race retirement.
                    let blocked = connection.blocked_since.lock().unwrap();
                    if blocked.is_some_and(|since| since.elapsed() >= idle) {
                        // The duplicate descriptor refers to this exact socket;
                        // no FD-number reuse or Pod/network administration needed.
                        match connection.control.shutdown(Shutdown::Both) {
                            Ok(()) => {
                                metrics::counter!("http_shutdown_idle_responses_total")
                                    .increment(1);
                                tracing::warn!(
                                    idle_seconds = idle.as_secs(),
                                    "closed stalled HTTP response after handlers joined"
                                );
                                return false;
                            }
                            Err(error) => {
                                tracing::warn!(%error, "failed to close stalled response")
                            }
                        }
                    }
                    true
                });
            }
            tokio::time::sleep(idle.min(Duration::from_secs(1))).await;
        }
    }
}

pub(crate) async fn track_handler(
    State(drain): State<Arc<Drain>>,
    request: Request,
    next: Next,
) -> Response {
    let Some(handler) = drain.admit() else {
        return (StatusCode::SERVICE_UNAVAILABLE, "worker is draining").into_response();
    };
    // Dropping a JoinHandle detaches it. A client disconnect therefore cannot
    // drop an accepted mutation halfway through a storage commit. The guard
    // remains in the task until success, error or panic.
    match tokio::spawn(
        async move {
            let _handler = handler;
            next.run(request).await
        }
        .in_current_span(),
    )
    .await
    {
        Ok(response) => response,
        Err(error) => {
            tracing::error!(%error, "HTTP handler task failed");
            StatusCode::INTERNAL_SERVER_ERROR.into_response()
        }
    }
}

struct Connection {
    control: std::net::TcpStream,
    blocked_since: Mutex<Option<Instant>>,
}
impl Connection {
    fn wrote(blocked: &mut Option<Instant>, result: &Poll<io::Result<usize>>) {
        match result {
            Poll::Pending => {
                blocked.get_or_insert_with(Instant::now);
            }
            Poll::Ready(Ok(n)) if *n > 0 => *blocked = None,
            _ => {}
        }
    }
}
pub(crate) struct DrainListener {
    listener: TcpListener,
    drain: Arc<Drain>,
}
impl DrainListener {
    pub(crate) fn new(listener: TcpListener, drain: Arc<Drain>) -> Self {
        Self { listener, drain }
    }
}
pub(crate) struct ConnectionIo {
    stream: TcpStream,
    connection: Arc<Connection>,
}
impl Listener for DrainListener {
    type Io = ConnectionIo;
    type Addr = SocketAddr;
    async fn accept(&mut self) -> (Self::Io, Self::Addr) {
        loop {
            let (stream, address) = Listener::accept(&mut self.listener).await;
            let wrapped = (|| {
                let stream = stream.into_std()?;
                let connection = Arc::new(Connection {
                    control: stream.try_clone()?,
                    blocked_since: Mutex::new(None),
                });
                let stream = TcpStream::from_std(stream)?;
                let mut connections = self.drain.connections.lock().unwrap();
                // Amortize pruning instead of scanning every live connection
                // for each accepted request. Dead weak entries retain no socket.
                if connections.len().is_multiple_of(256) {
                    connections.retain(|c| c.strong_count() != 0);
                }
                connections.push(Arc::downgrade(&connection));
                Ok::<_, io::Error>(ConnectionIo { stream, connection })
            })();
            match wrapped {
                Ok(io) => return (io, address),
                Err(error) => {
                    tracing::warn!(%error, "failed to track accepted socket");
                    tokio::time::sleep(Duration::from_secs(1)).await;
                }
            }
        }
    }
    fn local_addr(&self) -> io::Result<Self::Addr> {
        self.listener.local_addr()
    }
}
impl AsyncRead for ConnectionIo {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        Pin::new(&mut self.stream).poll_read(cx, buf)
    }
}
impl AsyncWrite for ConnectionIo {
    fn poll_write(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        let this = &mut *self;
        let mut blocked = this.connection.blocked_since.lock().unwrap();
        let result = Pin::new(&mut this.stream).poll_write(cx, buf);
        Connection::wrote(&mut blocked, &result);
        result
    }
    fn poll_write_vectored(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        bufs: &[io::IoSlice<'_>],
    ) -> Poll<io::Result<usize>> {
        let this = &mut *self;
        let mut blocked = this.connection.blocked_since.lock().unwrap();
        let result = Pin::new(&mut this.stream).poll_write_vectored(cx, bufs);
        Connection::wrote(&mut blocked, &result);
        result
    }
    fn is_write_vectored(&self) -> bool {
        self.stream.is_write_vectored()
    }
    fn poll_flush(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.stream).poll_flush(cx)
    }
    fn poll_shutdown(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.stream).poll_shutdown(cx)
    }
}

#[cfg(test)]
mod tests;
