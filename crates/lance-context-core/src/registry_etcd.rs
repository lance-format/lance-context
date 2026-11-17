//! [`StoreRegistry`] backed by etcd.
//!
//! One key per store: `<prefix>/registry/<kind>/<name>` holding
//! `{"uri": ..., "created_at": ...}`. A point lookup is one round trip, an
//! upsert is one `put`, and nothing is retried against a shared commit point.
//! Compare `RolloutRegistry`, where every read is a manifest fetch from object
//! storage and every write is two Lance commits contended by every worker.
//!
//! Store names are validated to be portable path segments before they reach a
//! registry, so they are used as key segments unescaped.

use std::sync::Arc;

use etcd_client::{Client, Compare, CompareOp, DeleteOptions, GetOptions, Txn, TxnOp};
use lance::{Error as LanceError, Result as LanceResult};
use serde::{Deserialize, Serialize};

use crate::etcd::etcd_error;
use crate::registry::{RegistryEntry, StoreRegistry};

#[derive(Serialize, Deserialize)]
struct Value {
    uri: String,
    created_at: i64,
    #[serde(default)]
    deleted: bool,
    #[serde(default)]
    source_version: Option<u64>,
}

#[derive(Serialize, Deserialize, Default)]
struct Migration {
    source: String,
    floor: u64,
    active: bool,
}

/// etcd-backed store directory for one store kind.
pub struct EtcdRegistry {
    client: Client,
    /// `<prefix>/registry/<kind>` with no trailing slash.
    prefix: String,
    migration_key: String,
}

impl EtcdRegistry {
    /// `kind` is `rollout`, `generic` or `datagen`; `prefix` is the
    /// deployment-wide etcd namespace (see `EtcdConfig::prefix`).
    pub fn new(client: Client, prefix: &str, kind: &str) -> Arc<Self> {
        Arc::new(Self {
            client,
            prefix: format!("{}/registry/{kind}", prefix.trim_end_matches('/')),
            migration_key: format!(
                "{}/registry-migrations/{kind}",
                prefix.trim_end_matches('/')
            ),
        })
    }

    fn key(&self, name: &str) -> String {
        format!("{}/{name}", self.prefix)
    }

    fn entry(&self, key: &[u8], value: &[u8]) -> LanceResult<Option<RegistryEntry>> {
        let key = String::from_utf8_lossy(key);
        let name = key
            .strip_prefix(&format!("{}/", self.prefix))
            .unwrap_or(&key)
            .to_string();
        let value: Value = serde_json::from_slice(value)
            .map_err(|e| LanceError::io(format!("decode registry entry '{name}': {e}")))?;
        if value.deleted {
            return Ok(None);
        }
        Ok(Some(RegistryEntry {
            name,
            uri: value.uri,
            created_at: value.created_at,
        }))
    }

    fn encode(uri: &str, created_at: i64) -> LanceResult<Vec<u8>> {
        serde_json::to_vec(&Value {
            uri: uri.to_string(),
            created_at,
            deleted: false,
            source_version: None,
        })
        .map_err(|e| LanceError::io(format!("encode registry entry: {e}")))
    }

    async fn migration(&self) -> LanceResult<(Migration, i64)> {
        let response = self
            .client
            .clone()
            .get(self.migration_key.clone(), None)
            .await
            .map_err(etcd_error("registry migration read"))?;
        match response.kvs().first() {
            Some(kv) => Ok((
                serde_json::from_slice(kv.value())
                    .map_err(|e| LanceError::io(format!("decode registry migration: {e}")))?,
                kv.mod_revision(),
            )),
            None => Ok((Migration::default(), 0)),
        }
    }

    /// Seal this destination before opening it as the authoritative backend.
    /// Operators must first quiesce all Lance registry writers and verify all
    /// three kinds. Sealing rejects every delayed mirror request afterwards.
    async fn activate(&self) -> LanceResult<()> {
        for _ in 0..32 {
            let (mut state, revision) = self.migration().await?;
            if state.active {
                return Ok(());
            }
            state.active = true;
            if self.put_migration(&state, revision).await? {
                return Ok(());
            }
        }
        Err(LanceError::io("registry activation contention"))
    }

    pub(crate) async fn activate_after_validation(&self, lance_uri: &str) -> LanceResult<()> {
        let (state, _) = self.migration().await?;
        if state.active {
            return Ok(());
        }
        if !state.source.is_empty() && state.source != lance_uri {
            return Err(LanceError::io(
                "registry migration source differs from configured data directory",
            ));
        }
        let source = crate::LanceRegistry::new(
            crate::RolloutRegistry::open_or_create(lance_uri, None).await?,
        );
        let diff = crate::diff_registries(&source, self).await?;
        if !diff.is_empty() {
            return Err(LanceError::io(format!("registry cutover refused: reconcile names, URIs and timestamps from Lance first (missing={}, extra={}, mismatched={})", diff.only_in_primary.len(), diff.only_in_mirror.len(), diff.mismatched.len())));
        }
        self.activate().await
    }

    async fn ensure_writable(&self) -> LanceResult<()> {
        for _ in 0..32 {
            let (mut state, revision) = self.migration().await?;
            if state.active {
                return Ok(());
            }
            if !state.source.is_empty() {
                return Err(LanceError::io(
                    "registry is a migration mirror; activate through validated backend startup first",
                ));
            }
            state.active = true;
            // Do not seal a mirror that appeared after our initial read.
            if self.put_migration(&state, revision).await? {
                return Ok(());
            }
        }
        Err(LanceError::io("registry activation contention"))
    }

    async fn put_migration(&self, state: &Migration, revision: i64) -> LanceResult<bool> {
        let value = serde_json::to_vec(state).map_err(|e| LanceError::io(e.to_string()))?;
        Ok(self
            .client
            .clone()
            .txn(
                Txn::new()
                    .when([Compare::mod_revision(
                        self.migration_key.clone(),
                        CompareOp::Equal,
                        revision,
                    )])
                    .and_then([TxnOp::put(self.migration_key.clone(), value, None)]),
            )
            .await
            .map_err(etcd_error("registry migration CAS"))?
            .succeeded())
    }

    async fn source_fence(
        &self,
        source: &str,
        version: u64,
        advance: bool,
    ) -> LanceResult<Option<i64>> {
        for _ in 0..32 {
            let (mut state, revision) = self.migration().await?;
            if state.active {
                return Err(LanceError::io(
                    "registry destination is active; mirror writes are sealed",
                ));
            }
            if !state.source.is_empty() && state.source != source {
                return Err(LanceError::io(
                    "registry migration source changed; use a fresh namespace",
                ));
            }
            if version < state.floor {
                return Ok(None);
            }
            if revision != 0 && (!advance || version == state.floor) {
                return Ok(Some(revision));
            }
            state.source = source.to_owned();
            if advance {
                state.floor = version;
            }
            if self.put_migration(&state, revision).await? {
                continue;
            }
        }
        Err(LanceError::io("registry mirror fence contention"))
    }

    /// Compare both the per-name watermark and the full-snapshot floor in
    /// the same transaction as the write. Tombstones retain delete ordering.
    pub(crate) async fn apply_source(
        &self,
        source: &str,
        version: u64,
        name: &str,
        entry: Option<&RegistryEntry>,
    ) -> LanceResult<bool> {
        for _ in 0..32 {
            let Some(fence) = self.source_fence(source, version, false).await? else {
                return Ok(false);
            };
            let key = self.key(name);
            let response = self
                .client
                .clone()
                .get(key.clone(), None)
                .await
                .map_err(etcd_error("registry mirror read"))?;
            let revision = match response.kvs().first() {
                Some(kv) => {
                    let old: Value = serde_json::from_slice(kv.value())
                        .map_err(|e| LanceError::io(e.to_string()))?;
                    if old.source_version.is_some_and(|v| v >= version) {
                        return Ok(false);
                    }
                    kv.mod_revision()
                }
                None => 0,
            };
            let value = Value {
                uri: entry.map_or_else(String::new, |e| e.uri.clone()),
                created_at: entry.map_or(0, |e| e.created_at),
                deleted: entry.is_none(),
                source_version: Some(version),
            };
            let value = serde_json::to_vec(&value).map_err(|e| LanceError::io(e.to_string()))?;
            let response = self
                .client
                .clone()
                .txn(
                    Txn::new()
                        .when([
                            Compare::mod_revision(
                                self.migration_key.clone(),
                                CompareOp::Equal,
                                fence,
                            ),
                            Compare::mod_revision(key.clone(), CompareOp::Equal, revision),
                        ])
                        .and_then([TxnOp::put(key, value, None)]),
                )
                .await
                .map_err(etcd_error("registry mirror apply"))?;
            if response.succeeded() {
                return Ok(true);
            }
        }
        Err(LanceError::io("registry mirror entry contention"))
    }

    pub(crate) async fn reconcile_source(
        &self,
        source: &str,
        version: u64,
        entries: &[RegistryEntry],
    ) -> LanceResult<usize> {
        // Advance BEFORE listing destination names. This also fences an old
        // snapshot inserting a name the newer snapshot never saw (e.g. a
        // create/delete that happened before initial backfill).
        if self.source_fence(source, version, true).await?.is_none() {
            return Ok(0);
        }
        let names: std::collections::HashSet<_> = entries.iter().map(|e| e.name.as_str()).collect();
        let mut changed = 0;
        for old in self.list().await? {
            if !names.contains(old.name.as_str()) {
                changed += usize::from(self.apply_source(source, version, &old.name, None).await?);
            }
        }
        for entry in entries {
            changed += usize::from(
                self.apply_source(source, version, &entry.name, Some(entry))
                    .await?,
            );
        }
        Ok(changed)
    }
}

#[async_trait::async_trait]
impl StoreRegistry for EtcdRegistry {
    async fn contains(&self, name: &str) -> LanceResult<bool> {
        Ok(self.get(name).await?.is_some())
    }

    async fn get(&self, name: &str) -> LanceResult<Option<RegistryEntry>> {
        let response = self
            .client
            .clone()
            .get(self.key(name), None)
            .await
            .map_err(etcd_error("registry get"))?;
        response
            .kvs()
            .first()
            .map(|kv| self.entry(kv.key(), kv.value()))
            .transpose()
            .map(Option::flatten)
    }

    async fn list(&self) -> LanceResult<Vec<RegistryEntry>> {
        let response = self
            .client
            .clone()
            .get(
                format!("{}/", self.prefix),
                Some(GetOptions::new().with_prefix()),
            )
            .await
            .map_err(etcd_error("registry list"))?;
        response
            .kvs()
            .iter()
            .map(|kv| self.entry(kv.key(), kv.value()))
            .collect::<LanceResult<Vec<_>>>()
            .map(|entries| entries.into_iter().flatten().collect())
    }

    async fn upsert(&self, name: &str, uri: &str) -> LanceResult<()> {
        self.ensure_writable().await?;
        // Keep the original created_at when the entry already exists: a
        // retried create must not look like a newer store.
        let created_at = match self.get(name).await? {
            Some(existing) => existing.created_at,
            None => chrono::Utc::now().timestamp_millis(),
        };
        self.client
            .clone()
            .put(self.key(name), Self::encode(uri, created_at)?, None)
            .await
            .map_err(etcd_error("registry put"))?;
        Ok(())
    }

    async fn remove(&self, name: &str) -> LanceResult<()> {
        self.ensure_writable().await?;
        self.client
            .clone()
            .delete(self.key(name), Some(DeleteOptions::new()))
            .await
            .map_err(etcd_error("registry delete"))?;
        Ok(())
    }

    async fn insert_missing(&self, entries: &[(String, String)]) -> LanceResult<usize> {
        self.ensure_writable().await?;
        // Each entry has its own compare-and-put transaction: live entries
        // are preserved, and retained migration tombstones count as absent.
        const BATCH: usize = 128;
        let now = chrono::Utc::now().timestamp_millis();
        let mut inserted = 0usize;
        let mut seen = std::collections::HashSet::new();
        for chunk in entries.chunks(BATCH) {
            let mut client = self.client.clone();
            for (name, uri) in chunk {
                if !seen.insert(name.clone()) {
                    continue;
                }
                let key = self.key(name);
                let existing = client
                    .get(key.clone(), None)
                    .await
                    .map_err(etcd_error("registry insert_missing read"))?;
                let revision = match existing.kvs().first() {
                    Some(kv) => {
                        if self.entry(kv.key(), kv.value())?.is_some() {
                            continue;
                        }
                        kv.mod_revision()
                    }
                    None => 0,
                };
                let txn = Txn::new()
                    .when([Compare::mod_revision(
                        key.as_str(),
                        CompareOp::Equal,
                        revision,
                    )])
                    .and_then([TxnOp::put(key.as_str(), Self::encode(uri, now)?, None)]);
                if client
                    .txn(txn)
                    .await
                    .map_err(etcd_error("registry insert_missing"))?
                    .succeeded()
                {
                    inserted += 1;
                }
            }
        }
        Ok(inserted)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn registry() -> Option<Arc<EtcdRegistry>> {
        let endpoints = std::env::var("ETCD_TEST_ENDPOINTS").ok()?;
        let client = Client::connect(endpoints.split(',').collect::<Vec<_>>(), None)
            .await
            .expect("connect to test etcd");
        let prefix = format!("/lance-context-test/{}", uuid::Uuid::new_v4());
        Some(EtcdRegistry::new(client, &prefix, "rollout"))
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn upsert_get_contains_remove() {
        let r = registry().await.unwrap();
        assert!(!r.contains("a").await.unwrap());
        assert!(r.get("a").await.unwrap().is_none());

        r.upsert("a", "/data/a.lance").await.unwrap();
        assert!(r.contains("a").await.unwrap());
        let first = r.get("a").await.unwrap().unwrap();
        assert_eq!(first.uri, "/data/a.lance");

        // A retried create keeps the original created_at.
        r.upsert("a", "/data/a2.lance").await.unwrap();
        let second = r.get("a").await.unwrap().unwrap();
        assert_eq!(second.uri, "/data/a2.lance");
        assert_eq!(second.created_at, first.created_at);

        r.remove("a").await.unwrap();
        assert!(!r.contains("a").await.unwrap());
        // Removing an absent name is a no-op.
        r.remove("a").await.unwrap();
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn list_and_insert_missing() {
        let r = registry().await.unwrap();
        r.upsert("keep", "/old").await.unwrap();
        let n = r
            .insert_missing(&[
                ("keep".into(), "/new".into()),
                ("x".into(), "/x".into()),
                ("y".into(), "/y".into()),
                ("y".into(), "/dup".into()),
            ])
            .await
            .unwrap();
        assert_eq!(n, 2, "only x and y were missing");
        let mut names: Vec<String> = r
            .list()
            .await
            .unwrap()
            .into_iter()
            .map(|e| e.name)
            .collect();
        names.sort();
        assert_eq!(names, vec!["keep", "x", "y"]);
        assert_eq!(r.get("keep").await.unwrap().unwrap().uri, "/old");
    }

    /// Migration step 1: Lance primary, etcd mirror. Writes reach both, reads
    /// come from Lance, `diff` is empty after a backfill of pre-existing rows.
    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn mirrored_registry_keeps_both_backends_in_step() {
        use crate::registry::{
            backfill_registry, diff_registries, LanceRegistry, MirroredRegistry,
        };
        let Some(etcd) = registry().await else { return };
        let dir = tempfile::TempDir::new().unwrap();
        let uri = dir.path().join("_registry.lance");
        let lance = crate::registry::RolloutRegistry::open_or_create(uri.to_str().unwrap(), None)
            .await
            .unwrap();
        let lance: Arc<dyn StoreRegistry> = Arc::new(LanceRegistry::new(lance));
        // Rows that existed before mirroring started.
        lance.upsert("pre-1", "/p1").await.unwrap();
        lance.upsert("pre-2", "/p2").await.unwrap();

        let etcd_dyn: Arc<dyn StoreRegistry> = etcd.clone();
        let diff = diff_registries(&*lance, &*etcd_dyn).await.unwrap();
        assert_eq!(diff.only_in_primary, vec!["pre-1", "pre-2"]);
        assert!(diff.only_in_mirror.is_empty());
        assert_eq!(backfill_registry(&*lance, &*etcd_dyn).await.unwrap(), 2);

        let mirrored = MirroredRegistry {
            primary: lance.clone(),
            mirror: etcd_dyn.clone(),
            label: "rollout".into(),
        };
        mirrored.upsert("new", "/n").await.unwrap();
        mirrored.remove("pre-1").await.unwrap();
        assert!(mirrored.contains("new").await.unwrap());

        let diff = diff_registries(&*lance, &*etcd_dyn).await.unwrap();
        assert!(diff.is_empty(), "{diff:?}");
        let mut names: Vec<String> = etcd
            .list()
            .await
            .unwrap()
            .into_iter()
            .map(|e| e.name)
            .collect();
        names.sort();
        assert_eq!(names, vec!["new", "pre-2"]);
    }

    /// Names that are prefixes of other names must not collide: `list` is a
    /// prefix scan on `<kind>/`, so `a` and `ab` are distinct keys and a
    /// lookup of `a` must not see `ab`.
    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn prefix_names_do_not_collide() {
        let r = registry().await.unwrap();
        r.upsert("ab", "/ab").await.unwrap();
        assert!(!r.contains("a").await.unwrap());
        assert!(r.get("a").await.unwrap().is_none());
        r.upsert("a", "/a").await.unwrap();
        assert_eq!(r.list().await.unwrap().len(), 2);
        r.remove("a").await.unwrap();
        assert!(r.contains("ab").await.unwrap());
    }

    fn source_entry(name: &str, uri: &str) -> RegistryEntry {
        RegistryEntry {
            name: name.into(),
            uri: uri.into(),
            created_at: 123456,
        }
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn reconciliation_repairs_missed_deletes_and_uri_changes_preserving_timestamps() {
        use crate::{backfill_registry, diff_registries, LanceRegistry, RolloutRegistry};
        let mirror = registry().await.unwrap();
        let dir = tempfile::TempDir::new().unwrap();
        let source = LanceRegistry::new(
            RolloutRegistry::open_or_create(
                dir.path().join("registry.lance").to_str().unwrap(),
                None,
            )
            .await
            .unwrap(),
        );
        source.upsert("deleted", "/deleted").await.unwrap();
        source.upsert("changed", "/old").await.unwrap();
        backfill_registry(&source, &*mirror).await.unwrap();
        // Primary commits succeeded while mirror writes were unavailable.
        source.remove("deleted").await.unwrap();
        source.upsert("changed", "/current").await.unwrap();
        let diff = diff_registries(&source, &*mirror).await.unwrap();
        assert_eq!(diff.only_in_mirror, ["deleted"]);
        assert_eq!(diff.mismatched.len(), 1);
        assert_eq!(diff.mismatched[0].primary.uri, "/current");
        assert_eq!(diff.mismatched[0].mirror.uri, "/old");
        assert_eq!(backfill_registry(&source, &*mirror).await.unwrap(), 2);
        assert!(diff_registries(&source, &*mirror).await.unwrap().is_empty());
        assert_eq!(
            source.get("changed").await.unwrap(),
            mirror.get("changed").await.unwrap()
        );
        assert!(!mirror.contains("deleted").await.unwrap());
        assert_eq!(backfill_registry(&source, &*mirror).await.unwrap(), 0);
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn stale_snapshots_and_point_updates_cannot_resurrect_or_remove_newer_entries() {
        let r = registry().await.unwrap();
        let old = source_entry("x", "/old");
        let new = source_entry("x", "/new");
        // New snapshot never saw x: a per-name watermark alone is insufficient.
        r.reconcile_source("source", 20, &[]).await.unwrap();
        assert_eq!(
            r.reconcile_source("source", 10, std::slice::from_ref(&old))
                .await
                .unwrap(),
            0
        );
        assert!(!r.apply_source("source", 10, "x", Some(&old)).await.unwrap());
        assert!(r.get("x").await.unwrap().is_none());
        r.apply_source("source", 30, "x", Some(&new)).await.unwrap();
        r.reconcile_source("source", 25, &[]).await.unwrap();
        assert_eq!(r.get("x").await.unwrap(), Some(new.clone()));
        r.apply_source("source", 40, "x", None).await.unwrap();
        assert!(!r.apply_source("source", 39, "x", Some(&old)).await.unwrap());
        assert!(r.get("x").await.unwrap().is_none());
        r.apply_source("source", 41, "x", Some(&new)).await.unwrap();
        assert_eq!(r.get("x").await.unwrap(), Some(new));
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn interrupted_reconciliation_retries_same_snapshot_and_seals_at_cutover() {
        let r = registry().await.unwrap();
        let entries = [source_entry("a", "/a"), source_entry("b", "/b")];
        // Failure after the floor and one entry committed, before the rest.
        r.source_fence("source", 10, true).await.unwrap();
        r.apply_source("source", 10, "a", Some(&entries[0]))
            .await
            .unwrap();
        assert_eq!(r.reconcile_source("source", 10, &entries).await.unwrap(), 1);
        assert_eq!(r.list().await.unwrap().len(), 2);
        // A mirror must not silently accept authoritative writes before cutover.
        assert!(r.upsert("a", "/unsafe").await.is_err());
        r.activate().await.unwrap();
        r.upsert("a", "/authoritative").await.unwrap();
        assert!(r.reconcile_source("source", 100, &[]).await.is_err());
        assert!(r.apply_source("source", 100, "a", None).await.is_err());
        assert_eq!(r.get("a").await.unwrap().unwrap().uri, "/authoritative");
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn cutover_rejects_missing_or_mismatched_entries_and_timestamp_only_differences() {
        use crate::{backfill_registry, diff_registries, LanceRegistry, RolloutRegistry};
        let r = registry().await.unwrap();
        let dir = tempfile::TempDir::new().unwrap();
        let uri = dir
            .path()
            .join("registry.lance")
            .to_string_lossy()
            .to_string();
        let source = LanceRegistry::new(RolloutRegistry::open_or_create(&uri, None).await.unwrap());
        source.upsert("a", "/a").await.unwrap();
        assert!(r.activate_after_validation(&uri).await.is_err());
        backfill_registry(&source, &*r).await.unwrap();
        let original = source.get("a").await.unwrap().unwrap();
        // Inject the unversioned value produced by the previous mirror implementation.
        r.client
            .clone()
            .put(
                r.key("a"),
                EtcdRegistry::encode("/a", original.created_at + 1).unwrap(),
                None,
            )
            .await
            .unwrap();
        let diff = diff_registries(&source, &*r).await.unwrap();
        assert_eq!(diff.mismatched.len(), 1);
        assert!(r.activate_after_validation(&uri).await.is_err());
        backfill_registry(&source, &*r).await.unwrap();
        r.activate_after_validation(&uri).await.unwrap();
        assert_eq!(r.get("a").await.unwrap(), Some(original));
    }

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn authoritative_discovery_can_replace_a_migration_tombstone() {
        let r = registry().await.unwrap();
        r.apply_source("source", 1, "a", None).await.unwrap();
        r.activate().await.unwrap();
        assert_eq!(
            r.insert_missing(&[("a".into(), "/recreated".into())])
                .await
                .unwrap(),
            1
        );
        assert_eq!(r.get("a").await.unwrap().unwrap().uri, "/recreated");
    }

    #[tokio::test]
    async fn reverse_mirroring_is_rejected_before_opening_backends() {
        use crate::etcd::{open_registry, RegistryBackend, RegistryConfig};
        let config = RegistryConfig {
            registry_backend: RegistryBackend::Etcd,
            registry_mirror: Some(RegistryBackend::Lance),
        };
        let err = open_registry("datagen", "/not-opened", None, &config)
            .await
            .err()
            .unwrap();
        assert!(err.to_string().contains("reverse mirroring"));
    }
}
