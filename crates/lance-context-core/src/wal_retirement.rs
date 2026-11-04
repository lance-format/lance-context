//! Durable WAL retirement intents, written before manifest drain.
//!
//! Recovery trusts neither a remembered successful drain nor the intent alone:
//! it reloads the shard manifest and honors active reader registrations before
//! deleting immutable generation directories. Runtime-wide bounded scheduling
//! and periodic discovery of cold tables still precede production activation.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

use futures::TryStreamExt;
use lance::dataset::mem_wal::ShardManifestStore;
use lance::{Error, Result};
use lance_io::object_store::ObjectStore;
use object_store::{path::Path, GetOptions, GetRange, ObjectStoreExt, PutMode, PutOptions};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::rollout_append::Generation;

const MAX_GENERATIONS: usize = 256;
const MAX_INTENT_BYTES: usize = 128 * 1024;
const IO_TIMEOUT: Duration = Duration::from_secs(5);
// Intent publication is on the merge's commit-to-drain path. Allow brief
// object-store throttling to recover without abandoning an already committed
// merge, while keeping background discovery/collection metadata I/O short.
const PUBLICATION_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Intent {
    schema: u32,
    shard: Uuid,
    generations: Vec<Generation>,
}

fn prefix(root: &Path) -> Path {
    root.clone().join("_wal_retirements_v1")
}

async fn io<T>(future: impl std::future::Future<Output = Result<T>>) -> Result<T> {
    tokio::time::timeout(IO_TIMEOUT, future)
        .await
        .map_err(|_| Error::io("WAL retirement metadata operation timed out"))?
}

impl Intent {
    fn validate(&self) -> Result<()> {
        let mut numbers = HashSet::new();
        let mut paths = HashSet::new();
        if self.schema != 1
            || self.generations.is_empty()
            || self.generations.len() > MAX_GENERATIONS
        {
            return Err(Error::io("Invalid WAL retirement intent header"));
        }
        for generation in &self.generations {
            // Never let an intent address a shard manifest, active WAL log,
            // another shard, or an arbitrary path. Legacy gen_N is allowed.
            let suffix = generation.path.strip_prefix("gen_").or_else(|| {
                generation
                    .path
                    .rsplit_once("_gen_")
                    .and_then(|(hash, number)| {
                        (!hash.is_empty() && hash.bytes().all(|c| c.is_ascii_hexdigit()))
                            .then_some(number)
                    })
            });
            if generation.path.len() > 256
                || suffix.and_then(|s| s.parse::<u64>().ok()) != Some(generation.number)
                || !numbers.insert(generation.number)
                || !paths.insert(&generation.path)
            {
                return Err(Error::io("Invalid WAL retirement generation identity"));
            }
        }
        Ok(())
    }
}

/// Publish durable intent(s) before attempting the separately fenced drain.
/// Partial/ambiguous publication is safe: referenced generations are deferred.
pub(crate) async fn record(
    store: Arc<ObjectStore>,
    root: &Path,
    shard: Uuid,
    generations: Vec<Generation>,
) -> Result<Vec<Path>> {
    let mut tickets = Vec::new();
    for chunk in generations.chunks(MAX_GENERATIONS) {
        let intent = Intent {
            schema: 1,
            shard,
            generations: chunk.to_vec(),
        };
        intent.validate()?;
        let bytes = serde_json::to_vec(&intent).map_err(|e| Error::io(e.to_string()))?;
        if bytes.len() > MAX_INTENT_BYTES {
            return Err(Error::io("WAL retirement intent exceeds size bound"));
        }
        let key = prefix(root).join(Uuid::new_v4().simple().to_string());
        tokio::time::timeout(PUBLICATION_TIMEOUT, async {
            store
                .inner
                .put_opts(
                    &key,
                    bytes.into(),
                    PutOptions {
                        mode: PutMode::Create,
                        ..Default::default()
                    },
                )
                .await?;
            Ok::<(), Error>(())
        })
        .await
        .map_err(|_| Error::io("WAL retirement intent publication timed out"))??;
        tickets.push(key);
    }
    Ok(tickets)
}

async fn load(store: &ObjectStore, key: &Path) -> Result<Option<Intent>> {
    let response = match store
        .inner
        .get_opts(
            key,
            GetOptions {
                range: Some(GetRange::Bounded(0..(MAX_INTENT_BYTES as u64 + 1))),
                ..Default::default()
            },
        )
        .await
    {
        Ok(response) => response,
        Err(object_store::Error::NotFound { .. }) => return Ok(None),
        Err(error) => return Err(error.into()),
    };
    let bytes = response.bytes().await?;
    if bytes.len() > MAX_INTENT_BYTES {
        return Err(Error::io("Oversized WAL retirement intent; retain files"));
    }
    let intent: Intent = serde_json::from_slice(&bytes).map_err(|e| Error::io(e.to_string()))?;
    intent.validate()?;
    Ok(Some(intent))
}

/// True means the intent was fully collected (or already absent); false means
/// it still references live data. Errors retain the intent for another pass.
pub(crate) async fn recover_one(store: Arc<ObjectStore>, root: &Path, key: &Path) -> Result<bool> {
    let expected = prefix(root).to_string() + "/";
    let text = key.to_string();
    let suffix = text.strip_prefix(&expected).unwrap_or("");
    if suffix.len() != 32 || !suffix.bytes().all(|c| c.is_ascii_hexdigit()) {
        return Err(Error::io("WAL retirement key outside this table"));
    }
    let Some(intent) = io(load(&store, key)).await? else {
        return Ok(true);
    };
    let manifests = ShardManifestStore::new(
        store.clone(),
        root,
        intent.shard,
        crate::store_base::DEFAULT_MANIFEST_SCAN_BATCH_SIZE,
    );
    let manifest = io(manifests.refresh_latest())
        .await?
        .ok_or_else(|| Error::io("Missing shard manifest; retain retirement intent"))?;
    if intent.generations.iter().any(|g| {
        g.number >= manifest.current_generation
            || manifest
                .sstables
                .iter()
                .any(|live| live.generation == g.number || live.path == g.path)
    }) {
        return Ok(false);
    }
    let shard_root = root.clone().join("_mem_wal").join(intent.shard.to_string());
    let directories = intent
        .generations
        .into_iter()
        .map(|g| shard_root.clone().join(g.path))
        .collect();
    // Bound a recovery attempt; cancellation retains its durable intent. A
    // later pass tolerates any partial directory deletion and captures readers
    // anew only after revalidating the current manifest.
    tokio::time::timeout(
        Duration::from_secs(120),
        crate::wal_read_lifetime::reclaim(store.clone(), root.clone(), directories),
    )
    .await
    .map_err(|_| Error::io("WAL retirement reader wait or deletion timed out"))??;
    io(async {
        match store.inner.delete(key).await {
            Ok(()) | Err(object_store::Error::NotFound { .. }) => Ok(()),
            Err(error) => Err(error.into()),
        }
    })
    .await?;
    Ok(true)
}

/// Stream discovery with bounded memory after reopening a table. Callers must
/// provide bounded process-wide scheduling and periodically revisit cold tables.
/// One malformed/failed intent must not prevent unrelated intents from progressing.
#[allow(dead_code)] // Lifecycle scheduling is deliberately not enabled yet.
pub(crate) async fn recover_pending(
    store: Arc<ObjectStore>,
    root: &Path,
) -> Result<(usize, usize)> {
    let listing_prefix = prefix(root);
    let mut listing = store.inner.list(Some(&listing_prefix));
    let (mut collected, mut deferred) = (0, 0);
    let mut failure = None;
    loop {
        let next = tokio::time::timeout(IO_TIMEOUT, listing.try_next())
            .await
            .map_err(|_| Error::io("WAL retirement listing timed out"))??;
        let Some(meta) = next else {
            break;
        };
        match recover_one(store.clone(), root, &meta.location).await {
            Ok(true) => collected += 1,
            Ok(false) => deferred += 1,
            Err(error) => {
                if failure.is_none() {
                    failure = Some(error);
                }
            }
        }
    }
    match failure {
        Some(error) => Err(error),
        None => Ok((collected, deferred)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lance_index::mem_wal::SsTable;

    async fn publication_fixture() -> (Arc<ObjectStore>, Path, Uuid, Generation) {
        let uri = format!("memory:///retirement-publication-{}", Uuid::new_v4());
        let (store, root) = ObjectStore::from_uri(&uri).await.unwrap();
        let shard = Uuid::new_v4();
        let generation = Generation {
            number: 1,
            path: "abcdef12_gen_1".into(),
        };
        let manifests = ShardManifestStore::new(store.clone(), &root, shard, 16);
        manifests
            .initialize_shard(0, Default::default())
            .await
            .unwrap();
        manifests
            .commit_update(0, |current| {
                let mut next = current.clone();
                next.version += 1;
                next.current_generation = 2;
                next.sstables.push(SsTable {
                    generation: 1,
                    path: generation.path.clone(),
                    in_memory_bytes: Some(1),
                    physical_rows: Some(1),
                    primary_key_bytes: Some(1),
                });
                next
            })
            .await
            .unwrap();
        store
            .put(&data_path(&root, shard, &generation.path), b"payload")
            .await
            .unwrap();
        (store, root, shard, generation)
    }

    fn delayed(store: &ObjectStore, put_secs: u64, get_secs: u64) -> Arc<ObjectStore> {
        use object_store::throttle::{ThrottleConfig, ThrottledStore};
        let mut wrapped = store.clone();
        wrapped.inner = Arc::new(ThrottledStore::new(
            store.inner.clone(),
            ThrottleConfig {
                wait_put_per_call: Duration::from_secs(put_secs),
                wait_get_per_call: Duration::from_secs(get_secs),
                ..Default::default()
            },
        ));
        Arc::new(wrapped)
    }

    #[tokio::test(start_paused = true)]
    async fn publication_survives_slow_put_without_relaxing_collection_reads() {
        let (store, root, shard, generation) = publication_fixture().await;
        let tickets = record(
            delayed(&store, 6, 0),
            &root,
            shard,
            vec![generation.clone()],
        )
        .await
        .unwrap();
        assert_eq!(tickets.len(), 1);
        // A slower publication must not extend background metadata read limits.
        let error = recover_one(delayed(&store, 0, 6), &root, &tickets[0])
            .await
            .unwrap_err();
        assert!(error.to_string().contains("metadata operation timed out"));
        assert!(!recover_one(store.clone(), &root, &tickets[0])
            .await
            .unwrap());
        assert!(store
            .inner
            .head(&data_path(&root, shard, &generation.path))
            .await
            .is_ok());
        drain(store.clone(), &root, shard).await;
        assert!(recover_one(store.clone(), &root, &tickets[0])
            .await
            .unwrap());
        assert!(matches!(
            store.inner.head(&tickets[0]).await,
            Err(object_store::Error::NotFound { .. })
        ));
        assert!(matches!(
            store
                .inner
                .head(&data_path(&root, shard, &generation.path))
                .await,
            Err(object_store::Error::NotFound { .. })
        ));
    }

    #[tokio::test(start_paused = true)]
    async fn publication_deadline_preserves_manifest_and_allows_later_retry() {
        let (store, root, shard, generation) = publication_fixture().await;
        let manifests = ShardManifestStore::new(store.clone(), &root, shard, 16);
        let before = manifests.refresh_latest().await.unwrap().unwrap();
        let started = tokio::time::Instant::now();
        let error = record(
            delayed(&store, 31, 0),
            &root,
            shard,
            vec![generation.clone()],
        )
        .await
        .unwrap_err();
        assert!(error.to_string().contains("intent publication timed out"));
        assert_eq!(started.elapsed(), Duration::from_secs(30));
        let after = manifests.refresh_latest().await.unwrap().unwrap();
        assert_eq!(after.version, before.version);
        assert_eq!(after.sstables.len(), 1);
        assert_eq!(after.sstables[0].path, generation.path);
        assert!(store
            .inner
            .head(&data_path(&root, shard, &generation.path))
            .await
            .is_ok());
        assert!(store
            .inner
            .list(Some(&prefix(&root)))
            .try_collect::<Vec<_>>()
            .await
            .unwrap()
            .is_empty());
        let tickets = record(store.clone(), &root, shard, vec![generation])
            .await
            .unwrap();
        assert!(!recover_one(store.clone(), &root, &tickets[0])
            .await
            .unwrap());
        drain(store.clone(), &root, shard).await;
        assert!(recover_one(store, &root, &tickets[0]).await.unwrap());
    }

    // Metadata/physical-file fixture: tests the recovery protocol independently
    // of base publication. Real merge publication remains covered by the paused
    // self-merge and grouped-coordinator reader tests in rollout_store.
    async fn fixture() -> (tempfile::TempDir, Arc<ObjectStore>, Path, Uuid, Generation) {
        let dir = tempfile::tempdir().unwrap();
        let (store, root) = ObjectStore::from_uri(dir.path().to_str().unwrap())
            .await
            .unwrap();
        let shard = Uuid::new_v4();
        let generation = Generation {
            number: 1,
            path: "abcdef12_gen_1".into(),
        };
        let manifests = ShardManifestStore::new(store.clone(), &root, shard, 16);
        manifests
            .initialize_shard(0, Default::default())
            .await
            .unwrap();
        manifests
            .commit_update(0, |current| {
                let mut next = current.clone();
                next.version += 1;
                next.current_generation = 2;
                next.sstables.push(SsTable {
                    generation: 1,
                    path: generation.path.clone(),
                    in_memory_bytes: Some(1),
                    physical_rows: Some(1),
                    primary_key_bytes: Some(1),
                });
                next
            })
            .await
            .unwrap();
        store
            .put(&data_path(&root, shard, &generation.path), b"payload")
            .await
            .unwrap();
        (dir, store, root, shard, generation)
    }

    fn data_path(root: &Path, shard: Uuid, generation: &str) -> Path {
        root.clone()
            .join("_mem_wal")
            .join(shard.to_string())
            .join(generation)
            .join("data/payload")
    }

    async fn drain(store: Arc<ObjectStore>, root: &Path, shard: Uuid) {
        let manifests = ShardManifestStore::new(store, root, shard, 16);
        manifests
            .commit_update(0, |current| {
                let mut next = current.clone();
                next.version += 1;
                next.sstables.clear();
                next
            })
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn restart_discovers_intent_and_preserves_a_new_live_generation() {
        let (dir, store, root, shard, generation) = fixture().await;
        let tickets = record(store.clone(), &root, shard, vec![generation.clone()])
            .await
            .unwrap();
        assert!(!recover_one(store.clone(), &root, &tickets[0])
            .await
            .unwrap());
        assert!(store
            .inner
            .head(&data_path(&root, shard, &generation.path))
            .await
            .is_ok());
        drain(store.clone(), &root, shard).await;
        let new_path = "fedcba98_gen_2";
        let manifests = ShardManifestStore::new(store.clone(), &root, shard, 16);
        manifests
            .commit_update(0, |current| {
                let mut next = current.clone();
                next.version += 1;
                next.current_generation = 3;
                next.sstables.push(SsTable {
                    generation: 2,
                    path: new_path.into(),
                    in_memory_bytes: Some(1),
                    physical_rows: Some(1),
                    primary_key_bytes: Some(1),
                });
                next
            })
            .await
            .unwrap();
        store
            .put(&data_path(&root, shard, new_path), b"new payload")
            .await
            .unwrap();
        drop(manifests);
        drop(store);
        drop(tickets);
        // Recreate the store and discover persisted intents without remembered
        // keys or any in-memory merge/GC state.
        let (reopened, root) = ObjectStore::from_uri(dir.path().to_str().unwrap())
            .await
            .unwrap();
        assert_eq!(
            recover_pending(reopened.clone(), &root).await.unwrap(),
            (1, 0)
        );
        assert!(reopened
            .inner
            .head(&data_path(&root, shard, &generation.path))
            .await
            .is_err());
        assert!(reopened
            .inner
            .head(&data_path(&root, shard, new_path))
            .await
            .is_ok());
        assert_eq!(recover_pending(reopened, &root).await.unwrap(), (0, 0));
    }

    #[tokio::test]
    async fn absent_manifest_or_malformed_intent_never_authorizes_deletion() {
        let (_dir, store, root, shard, generation) = fixture().await;
        for path in ["manifest", "../abcdef12_gen_1", "abcdef12_gen_2", "wal"] {
            assert!(record(
                store.clone(),
                &root,
                shard,
                vec![Generation {
                    number: 1,
                    path: path.into()
                }]
            )
            .await
            .is_err());
        }
        let unknown = Uuid::new_v4();
        let tickets = record(store.clone(), &root, unknown, vec![generation.clone()])
            .await
            .unwrap();
        let data = data_path(&root, unknown, &generation.path);
        store.put(&data, b"unverified").await.unwrap();
        assert!(recover_one(store.clone(), &root, &tickets[0])
            .await
            .is_err());
        assert!(store.inner.head(&data).await.is_ok());
        assert!(store.inner.head(&tickets[0]).await.is_ok());
    }

    #[tokio::test]
    async fn interrupted_deletion_is_idempotent_and_bad_intent_does_not_block_good_one() {
        let (_dir, store, root, shard, generation) = fixture().await;
        let tickets = record(store.clone(), &root, shard, vec![generation.clone()])
            .await
            .unwrap();
        drain(store.clone(), &root, shard).await;
        // The data delete succeeded before a process died while leaving intent.
        store
            .remove_dir_all(
                root.clone()
                    .join("_mem_wal")
                    .join(shard.to_string())
                    .join(generation.path),
            )
            .await
            .unwrap();
        let bad = prefix(&root).join(Uuid::new_v4().simple().to_string());
        store.put(&bad, b"{broken json").await.unwrap();
        assert!(recover_pending(store.clone(), &root).await.is_err());
        assert!(store.inner.head(&tickets[0]).await.is_err());
        assert!(store.inner.head(&bad).await.is_ok());
        assert!(recover_one(store, &root, &tickets[0]).await.unwrap());
    }
}
