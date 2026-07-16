//! Single-master enforcement for the RocksDB scheduler backend.
//!
//! The RocksDB task store is an embedded, single-process database. Its on-disk
//! LOCK file already stops two processes from opening the *same* directory, but
//! that guard is local: two master pods mounting distinct PVCs can still point
//! at one shared `data_dir` object-store prefix and both drive compaction /
//! index / merge tasks against the same rollout datasets. That is exactly the
//! split-brain the RocksDB backend is not designed to tolerate (it has no
//! compare-and-swap enqueue or lease-based task claims).
//!
//! [`SingletonLease`] closes that gap with a heartbeat lease written into the
//! shared `data_dir`. On startup a master refuses to run if a live lease
//! (heartbeat newer than `ttl`) already exists; otherwise it claims the lease
//! and refreshes it from a background task. A crashed master's lease goes stale
//! after `ttl` and the next master takes over. On graceful shutdown the lease is
//! best-effort deleted so a replacement starts immediately.
//!
//! This is a cooperative guard, not a distributed lock: object stores do not all
//! offer atomic put-if-absent, so a narrow startup race is possible if two
//! masters boot within the same instant against an empty `data_dir`. It reliably
//! catches the real operational mistake — leaving an old master running while
//! starting a new one — which is what "no multi-master with RocksDB" requires.

use std::sync::Arc;
use std::time::Duration;

use lance::io::{ObjectStore, ObjectStoreParams, ObjectStoreRegistry};
use lance_context_core::{join_uri, new_uuid};
use object_store::path::Path as ObjectPath;
use serde::{Deserialize, Serialize};
use tokio::task::JoinHandle;

/// Relative name of the lease object inside `data_dir`.
const LEASE_OBJECT: &str = "_master.singleton.json";

/// How long a lease is considered live after its last heartbeat. A master that
/// stops refreshing (crash, OOM-kill, network partition) is treated as gone
/// after this window and a new master may take over.
const LEASE_TTL: Duration = Duration::from_secs(30);

/// Heartbeat cadence. Comfortably below `LEASE_TTL / 2` so a single missed tick
/// never expires a healthy lease.
const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(10);

/// Serialized lease document stored at `{data_dir}/_master.singleton.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct LeaseDoc {
    /// Random per-process owner id, used to recognize our own lease on renewal
    /// and to avoid deleting a lease another master has since taken over.
    owner: String,
    /// Best-effort human hint for operators reading the object directly.
    host: String,
    /// Unix seconds of the most recent heartbeat.
    heartbeat_at: i64,
}

/// An acquired single-master lease. Holds a background heartbeat task; dropping
/// it aborts the heartbeat and best-effort deletes the lease object.
pub struct SingletonLease {
    heartbeat: JoinHandle<()>,
    store: Arc<ObjectStore>,
    path: ObjectPath,
    owner: String,
}

impl SingletonLease {
    /// Claim the single-master lease under `data_dir`, or fail if another master
    /// currently holds it.
    pub async fn acquire(data_dir: &str) -> lance::Result<Self> {
        let uri = join_uri(data_dir, LEASE_OBJECT);
        let registry = Arc::new(ObjectStoreRegistry::default());
        let (store, path) =
            ObjectStore::from_uri_and_params(registry, &uri, &ObjectStoreParams::default()).await?;

        if let Some(existing) = read_lease(&store, &path).await? {
            let age = now_secs() - existing.heartbeat_at;
            if age < LEASE_TTL.as_secs() as i64 {
                return Err(lance::Error::io(format!(
                    "another master already holds the single-master lease at '{uri}' \
                     (owner '{}', host '{}', last heartbeat {age}s ago). The RocksDB scheduler \
                     backend permits only one active master per data directory. Stop the other \
                     master, or switch both to a shared backend, before starting this one.",
                    existing.owner, existing.host
                )));
            }
            tracing::warn!(
                previous_owner = %existing.owner,
                previous_host = %existing.host,
                stale_secs = age,
                "taking over stale single-master lease"
            );
        }

        let owner = new_uuid().to_string();
        write_lease(&store, &path, &owner).await?;
        tracing::info!(uri = %uri, owner = %owner, "acquired single-master lease");

        let heartbeat = spawn_heartbeat(store.clone(), path.clone(), owner.clone());
        Ok(Self {
            heartbeat,
            store,
            path,
            owner,
        })
    }

    /// Stop the heartbeat and best-effort release the lease so a replacement
    /// master can start immediately instead of waiting out the TTL.
    pub async fn release(self) {
        self.heartbeat.abort();
        // Only delete if we still own the lease: if it went stale and another
        // master took over, deleting would clobber their claim.
        match read_lease(&self.store, &self.path).await {
            Ok(Some(doc)) if doc.owner == self.owner => {
                if let Err(err) = self.store.delete(&self.path).await {
                    tracing::warn!(error = %err, "failed to release single-master lease");
                }
            }
            _ => {}
        }
    }
}

fn spawn_heartbeat(store: Arc<ObjectStore>, path: ObjectPath, owner: String) -> JoinHandle<()> {
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(HEARTBEAT_INTERVAL);
        ticker.tick().await; // first tick is immediate; we just wrote the lease.
        loop {
            ticker.tick().await;
            if let Err(err) = write_lease(&store, &path, &owner).await {
                tracing::error!(error = %err, "failed to refresh single-master lease");
            }
        }
    })
}

async fn read_lease(store: &ObjectStore, path: &ObjectPath) -> lance::Result<Option<LeaseDoc>> {
    if !store.exists(path).await? {
        return Ok(None);
    }
    let bytes = store.read_one_all(path).await?;
    match serde_json::from_slice::<LeaseDoc>(&bytes) {
        Ok(doc) => Ok(Some(doc)),
        // A corrupt/partial lease is treated as absent so a master can recover
        // rather than wedge forever on unreadable bytes.
        Err(err) => {
            tracing::warn!(error = %err, "ignoring unreadable single-master lease");
            Ok(None)
        }
    }
}

async fn write_lease(store: &ObjectStore, path: &ObjectPath, owner: &str) -> lance::Result<()> {
    let doc = LeaseDoc {
        owner: owner.to_string(),
        host: hostname(),
        heartbeat_at: now_secs(),
    };
    let bytes = serde_json::to_vec(&doc)
        .map_err(|err| lance::Error::io(format!("failed to encode single-master lease: {err}")))?;
    store.put(path, &bytes).await?;
    Ok(())
}

fn now_secs() -> i64 {
    chrono::Utc::now().timestamp()
}

fn hostname() -> String {
    std::env::var("HOSTNAME")
        .ok()
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| format!("pid-{}", std::process::id()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn second_master_is_rejected_while_first_holds_lease() {
        let dir = TempDir::new().unwrap();
        let data_dir = dir.path().to_string_lossy().to_string();

        let first = SingletonLease::acquire(&data_dir).await.unwrap();

        let error = match SingletonLease::acquire(&data_dir).await {
            Ok(_) => panic!("second master must be rejected"),
            Err(error) => error,
        };
        assert!(
            error.to_string().contains("another master already holds"),
            "unexpected error: {error}"
        );

        // After the holder releases, a new master may start.
        first.release().await;
        let second = SingletonLease::acquire(&data_dir).await.unwrap();
        second.release().await;
    }

    #[tokio::test]
    async fn stale_lease_is_taken_over() {
        let dir = TempDir::new().unwrap();
        let data_dir = dir.path().to_string_lossy().to_string();
        let uri = join_uri(&data_dir, LEASE_OBJECT);
        let registry = Arc::new(ObjectStoreRegistry::default());
        let (store, path) =
            ObjectStore::from_uri_and_params(registry, &uri, &ObjectStoreParams::default())
                .await
                .unwrap();

        // Write a lease whose heartbeat is well past the TTL.
        let stale = LeaseDoc {
            owner: "dead-master".to_string(),
            host: "old-host".to_string(),
            heartbeat_at: now_secs() - (LEASE_TTL.as_secs() as i64) - 100,
        };
        store
            .put(&path, &serde_json::to_vec(&stale).unwrap())
            .await
            .unwrap();

        let lease = SingletonLease::acquire(&data_dir)
            .await
            .expect("stale lease must be reclaimable");
        lease.release().await;
    }
}
