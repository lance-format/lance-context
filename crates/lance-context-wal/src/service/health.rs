use super::*;
use bytes::Bytes;
use http_body_util::Full;
use hyper::{body::Incoming, service::service_fn, Request, Response, StatusCode};
use serde::Serialize;
use std::{
    collections::BTreeMap,
    convert::Infallible,
    sync::atomic::{AtomicBool, Ordering},
};

pub(super) struct State {
    pub node: u64,
    pub handle: runtime::Handle,
    pub groups: Vec<u64>,
    pub draining: AtomicBool,
    pub maintenance: maintenance::Shared,
    pub peers: Arc<peer::Statistics>,
    pub snapshots: Arc<snapshot::Statistics>,
}
#[derive(Serialize)]
struct Group {
    status: crate::replication::Status,
    checkpoint_index: u64,
    observed_age_ms: u64,
    polling: bool,
    stopped: bool,
}
#[derive(Serialize)]
struct View {
    node: u64,
    ready: bool,
    groups: BTreeMap<u64, Group>,
    disk: Option<crate::admission::Snapshot>,
    maintenance: maintenance::Progress,
    snapshots_sent: u64,
    snapshots_installed: u64,
    snapshots_failed: u64,
    peer_drops: (u64, u64),
    peer_sent_frames: u64,
    peer_received_frames: u64,
}
impl State {
    fn view(&self) -> Result<View> {
        let mut ready = !self.draining.load(Ordering::Acquire);
        let mut groups = BTreeMap::new();
        for &group in &self.groups {
            let o = self.handle.observation(group)?;
            let age = o.observed_at.elapsed();
            ready &= !o.stopped && o.failure.is_none() && age <= Duration::from_secs(15);
            groups.insert(
                group,
                Group {
                    status: o.status,
                    checkpoint_index: o.checkpoint_index,
                    observed_age_ms: age.as_millis().min(u64::MAX as u128) as u64,
                    polling: o.polling,
                    stopped: o.stopped,
                },
            );
        }
        Ok(View {
            node: self.node,
            ready,
            groups,
            disk: self.handle.disk_budget(),
            maintenance: self
                .maintenance
                .lock()
                .map_err(|_| anyhow::anyhow!("maintenance observation unavailable"))?
                .clone(),
            snapshots_sent: self.snapshots.sent.load(Ordering::Relaxed),
            snapshots_installed: self.snapshots.installed.load(Ordering::Relaxed),
            snapshots_failed: self.snapshots.failed.load(Ordering::Relaxed),
            peer_drops: self.handle.peer_drops(),
            peer_sent_frames: self.peers.sent_frames.load(Ordering::Relaxed),
            peer_received_frames: self.peers.received_frames.load(Ordering::Relaxed),
        })
    }
}
async fn respond(
    state: Arc<State>,
    request: Request<Incoming>,
) -> Result<Response<Full<Bytes>>, Infallible> {
    let (code, body) = if request.method() != hyper::Method::GET {
        (StatusCode::METHOD_NOT_ALLOWED, Bytes::new())
    } else {
        match request.uri().path() {
            "/health/live" => (StatusCode::OK, Bytes::from_static(b"{\"live\":true}\n")),
            "/health/ready" | "/status" => match state.view() {
                Ok(view) => (
                    if view.ready {
                        StatusCode::OK
                    } else {
                        StatusCode::SERVICE_UNAVAILABLE
                    },
                    Bytes::from(serde_json::to_vec(&view).unwrap_or_default()),
                ),
                Err(_) => (StatusCode::SERVICE_UNAVAILABLE, Bytes::new()),
            },
            _ => (StatusCode::NOT_FOUND, Bytes::new()),
        }
    };
    Ok(Response::builder()
        .status(code)
        .header("content-type", "application/json")
        .body(Full::new(body))
        .expect("fixed HTTP response"))
}
/// Bound even slow header readers independently of the WAL's data/control pools.
pub(super) async fn serve(
    listener: TcpListener,
    state: Arc<State>,
    mut stop: watch::Receiver<bool>,
) -> Result<()> {
    let slots = Arc::new(tokio::sync::Semaphore::new(16));
    let mut tasks = tokio::task::JoinSet::new();
    loop {
        tokio::select! {
            _ = stop.changed() => break,
            _ = tasks.join_next(), if !tasks.is_empty() => {},
            incoming = listener.accept() => {
                let (tcp,_)=incoming?;
                let Ok(permit)=slots.clone().try_acquire_owned() else { drop(tcp); continue; };
                let state=state.clone();
                tasks.spawn(async move {
                    let _permit=permit;
                    let service=service_fn(move |request| respond(state.clone(),request));
                    let mut http=hyper::server::conn::http1::Builder::new();
                    http.keep_alive(false).max_buf_size(8192);
                    let _=tokio::time::timeout(Duration::from_secs(2),http.serve_connection(hyper_util::rt::TokioIo::new(tcp),service)).await;
                });
            }
        }
    }
    tasks.abort_all();
    while tasks.join_next().await.is_some() {}
    Ok(())
}
