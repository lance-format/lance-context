//! Independently deployable, authenticated WAL node with joined maintenance.
//! Data ACKs still come only from the durable replicated runtime. Health reports
//! describe local serving state, never quorum freshness or Lance visibility.
use crate::replication::runtime::{
    self, client,
    peer::{self, snapshot},
    Runtime,
};
use anyhow::{ensure, Context, Result};
use std::{
    sync::{atomic::Ordering, Arc},
    time::Duration,
};
use tokio::{net::TcpListener, sync::watch, task::JoinHandle};

pub mod config;
mod health;
pub mod maintenance;
mod tls;
pub use config::Config;

pub(crate) mod duration_millis {
    use serde::{Deserialize, Deserializer, Serializer};
    use std::time::Duration;
    pub fn serialize<S: Serializer>(value: &Duration, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_u64(u64::try_from(value.as_millis()).map_err(serde::ser::Error::custom)?)
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Duration, D::Error> {
        Ok(Duration::from_millis(u64::deserialize(d)?))
    }
}

#[derive(Default)]
struct Parts {
    runtime: Option<Runtime>,
    peers: Option<peer::Transport>,
    images: Option<snapshot::Transport>,
    rpc: Option<client::Server>,
}
fn retain(error: &mut Option<anyhow::Error>, result: Result<()>) {
    if let Err(e) = result {
        if error.is_none() {
            *error = Some(e);
        }
    }
}
impl Parts {
    async fn stop_clients(&mut self, error: &mut Option<anyhow::Error>) {
        if let Some(server) = self.rpc.take() {
            retain(error, server.shutdown().await);
        }
    }
    async fn drain(mut self) -> Result<()> {
        let mut error = None;
        self.stop_clients(&mut error).await;
        if let Some(images) = self.images.take() {
            retain(&mut error, images.shutdown().await);
        }
        if let Some(peers) = self.peers.take() {
            retain(&mut error, peers.shutdown().await);
        }
        if let Some(runtime) = self.runtime.take() {
            retain(&mut error, runtime.shutdown().await);
        }
        error.map_or(Ok(()), Err)
    }
}

/// Dropping the service signals its supervisor without aborting it. The
/// supervisor retains and joins maintenance, file workers and Raft owners.
/// Await shutdown before reopening the same directory in this process.
pub struct Service {
    stop: watch::Sender<bool>,
    supervisor: Option<JoinHandle<Result<()>>>,
    handle: runtime::Handle,
}
impl Service {
    pub async fn start(config: Config) -> Result<Self> {
        config.validate()?;
        let (peer_listener, snapshot_listener, data, control, health_listener) = tokio::try_join!(
            TcpListener::bind(config.listen.peer),
            TcpListener::bind(config.listen.snapshot),
            TcpListener::bind(config.listen.data),
            TcpListener::bind(config.listen.control),
            TcpListener::bind(config.listen.health)
        )?;
        let opening = config.clone();
        // No worker starts until credentials, immutable policy and all listeners
        // are validated. A startup failure never silently creates new identity.
        let (node, identity) = tokio::task::spawn_blocking(move || opening.open()).await??;
        let mut parts = Parts::default();
        let startup: Result<runtime::Handle> = async {
            let (runtime, outbox) = Runtime::spawn(node, config.runtime.clone())?;
            let handle = runtime.handle();
            parts.runtime = Some(runtime);
            let (client_tls, server_tls) = identity.configs(peer::ALPN);
            parts.peers = Some(peer::Transport::spawn(
                handle.clone(),
                outbox,
                peer_listener,
                config.peers(false),
                client_tls,
                server_tls,
                config.peer.clone(),
            )?);
            let (client_tls, server_tls) = identity.configs(snapshot::ALPN);
            parts.images = Some(snapshot::Transport::spawn(
                handle.clone(),
                snapshot_listener,
                snapshot::Settings {
                    peers: config.peers(true),
                    client_tls,
                    server_tls,
                    options: config.snapshot.clone(),
                    admission: None,
                },
            )?);
            parts
                .peers
                .as_mut()
                .unwrap()
                .attach_snapshots(parts.images.as_ref().unwrap().route())?;
            let (_, server_tls) = identity.configs(client::ALPN);
            parts.rpc = Some(client::Server::spawn(
                handle.clone(),
                data,
                control,
                config.trusted_clients.clone(),
                server_tls,
                config.client.clone(),
            )?);
            Ok(handle)
        }
        .await;
        let handle = match startup {
            Ok(handle) => handle,
            Err(error) => {
                let cleanup = parts.drain().await;
                return Err(error.context(format!(
                    "startup failed; cleanup: {}",
                    cleanup.map_or_else(|e| e.to_string(), |_| "joined".into())
                )));
            }
        };
        let (stop, mut shutdown) = watch::channel(false);
        let state = Arc::new(health::State {
            node: config.node,
            handle: handle.clone(),
            groups: config.groups(),
            draining: std::sync::atomic::AtomicBool::new(false),
            maintenance: Arc::new(std::sync::Mutex::new(maintenance::Progress::default())),
            peers: parts.peers.as_ref().unwrap().statistics().clone(),
            snapshots: parts.images.as_ref().unwrap().statistics().clone(),
        });
        let mut maintenance = tokio::spawn(maintenance::run(
            handle.clone(),
            config.groups(),
            config.maintenance.clone(),
            state.maintenance.clone(),
            stop.subscribe(),
        ));
        let mut health = tokio::spawn(health::serve(
            health_listener,
            state.clone(),
            stop.subscribe(),
        ));
        let signal = stop.clone();
        let supervisor = tokio::spawn(async move {
            let mut error = None;
            let mut tick = tokio::time::interval(Duration::from_millis(200));
            tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            loop {
                tokio::select! { _=shutdown.changed()=>break, _=tick.tick()=>{} }
                if *shutdown.borrow() {
                    break;
                }
                if parts.peers.as_ref().unwrap().is_finished()
                    || parts.images.as_ref().unwrap().is_finished()
                    || parts.rpc.as_ref().unwrap().is_finished()
                    || maintenance.is_finished()
                    || health.is_finished()
                {
                    error = Some(anyhow::anyhow!("WAL service task stopped unexpectedly"));
                    break;
                }
                for &group in &state.groups {
                    match state.handle.observation(group) {
                        Ok(observed) if !observed.stopped && observed.failure.is_none() => {}
                        _ => {
                            error = Some(anyhow::anyhow!("WAL replica worker failed"));
                            break;
                        }
                    }
                }
                if error.is_some() {
                    break;
                }
            }
            state.draining.store(true, Ordering::Release);
            let _ = signal.send(true);
            // Stop client admission first, but keep owners and peer traffic
            // alive until every already-running maintenance operation is joined.
            parts.stop_clients(&mut error).await;
            retain(
                &mut error,
                (&mut maintenance)
                    .await
                    .map_err(anyhow::Error::from)
                    .and_then(|r| r),
            );
            retain(
                &mut error,
                (&mut health)
                    .await
                    .map_err(anyhow::Error::from)
                    .and_then(|r| r),
            );
            retain(&mut error, parts.drain().await);
            error.map_or(Ok(()), Err)
        });
        Ok(Self {
            stop,
            supervisor: Some(supervisor),
            handle,
        })
    }
    pub fn handle(&self) -> runtime::Handle {
        self.handle.clone()
    }
    pub async fn wait(&mut self) -> Result<()> {
        let result = self
            .supervisor
            .as_mut()
            .context("service already joined")?
            .await;
        self.supervisor.take();
        result?
    }
    pub async fn shutdown(mut self) -> Result<()> {
        let _ = self.stop.send(true);
        if self.supervisor.is_some() {
            self.wait().await
        } else {
            Ok(())
        }
    }
}
impl Drop for Service {
    fn drop(&mut self) {
        let _ = self.stop.send(true);
    }
}
