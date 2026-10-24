//! Startup discovery for rollout datasets created before the registry existed.

use std::collections::HashSet;
use std::time::Duration;

use lance::io::ObjectStore;
use lance_context_core::{join_uri, StoreRegistry};
use lance_table::io::commit::commit_handler_from_url;

const ROLLOUT_SUFFIX: &str = ".rollout.lance";

/// Discover top-level rollout datasets under `data_dir` and add any missing
/// registry rows in one batch. Returns the number of rows inserted.
pub async fn backfill_registry(
    data_dir: &str,
    registry: &dyn StoreRegistry,
) -> lance::Result<usize> {
    let (store, base_path) = ObjectStore::from_uri(data_dir).await?;
    let mut children = store.read_dir(base_path.clone()).await?;
    children.sort();

    // A retained WAL/attempt prefix is not a committed dataset. Re-registering
    // one after an intentional metadata-only cleanup poisons every fleet scan.
    // Read registry membership once so normal startup never opens all existing
    // datasets or adds a point request per table to etcd.
    let known: HashSet<_> = registry.list().await?.into_iter().map(|e| e.name).collect();
    let candidates = children.into_iter().filter_map(|child| {
        let name = child.strip_suffix(ROLLOUT_SUFFIX)?;
        if name.is_empty() || matches!(name, "_registry" | "_stats") || known.contains(name) {
            return None;
        }
        Some((name.to_string(), join_uri(data_dir, &child)))
    });

    let mut entries = Vec::new();
    let commit_handler = commit_handler_from_url(data_dir, &None).await?;
    // Resolve committed manifest locations without decoding manifests, indexes
    // or payloads. Use Lance's resolver for both version-naming schemes. Keep
    // one request in flight and never probe already registered stores.
    for (name, uri) in candidates {
        let path = base_path.clone().join(format!("{name}{ROLLOUT_SUFFIX}"));
        match tokio::time::timeout(
            Duration::from_secs(30),
            commit_handler.resolve_latest_location(&path, &store),
        )
        .await
        {
            Ok(Ok(_)) => entries.push((name, uri)),
            Ok(Err(lance::Error::NotFound { .. })) => {
                tracing::warn!(store = %name, "startup discovery skipped prefix without a committed manifest");
            }
            Ok(Err(error)) => return Err(error),
            Err(_) => {
                return Err(lance::Error::io(format!(
                    "startup discovery manifest lookup timed out for '{name}'"
                )));
            }
        }
    }

    registry.insert_missing(&entries).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use lance_context_core::RolloutRegistry;
    use lance_context_core::RolloutStore;
    use tempfile::TempDir;

    #[tokio::test]
    async fn residual_prefix_without_dataset_is_not_registered() {
        let dir = TempDir::new().unwrap();
        let residue = dir.path().join("retained.rollout.lance");
        tokio::fs::create_dir_all(residue.join("_mem_wal"))
            .await
            .unwrap();
        let retained = residue.join("_mem_wal/retained-attempt");
        tokio::fs::write(&retained, b"retain this evidence")
            .await
            .unwrap();
        let registry = lance_context_core::LanceRegistry::new(
            RolloutRegistry::open_or_create(
                dir.path().join("_registry.rollout.lance").to_str().unwrap(),
                None,
            )
            .await
            .unwrap(),
        );
        for _ in 0..2 {
            assert_eq!(
                backfill_registry(dir.path().to_str().unwrap(), &registry)
                    .await
                    .unwrap(),
                0
            );
            assert!(!registry.contains("retained").await.unwrap());
        }
        assert_eq!(
            tokio::fs::read(&retained).await.unwrap(),
            b"retain this evidence"
        );
    }

    #[tokio::test]
    async fn known_entry_is_not_reopened_or_removed_by_discovery() {
        let dir = TempDir::new().unwrap();
        let residue = dir.path().join("known.rollout.lance");
        tokio::fs::create_dir_all(&residue).await.unwrap();
        let registry = lance_context_core::LanceRegistry::new(
            RolloutRegistry::open_or_create(
                dir.path().join("_registry.rollout.lance").to_str().unwrap(),
                None,
            )
            .await
            .unwrap(),
        );
        registry
            .upsert("known", residue.to_str().unwrap())
            .await
            .unwrap();
        let before = registry.get("known").await.unwrap();
        assert_eq!(
            backfill_registry(dir.path().to_str().unwrap(), &registry)
                .await
                .unwrap(),
            0
        );
        assert_eq!(registry.get("known").await.unwrap(), before);
    }

    #[tokio::test]
    async fn mixed_manifest_schemes_are_not_treated_as_missing_datasets() {
        let dir = TempDir::new().unwrap();
        let residue = dir.path().join("invalid.rollout.lance");
        let versions = residue.join("_versions");
        tokio::fs::create_dir_all(&versions).await.unwrap();
        tokio::fs::write(versions.join("1.manifest"), b"retained V1 manifest")
            .await
            .unwrap();
        tokio::fs::write(
            versions.join(format!("{:020}.manifest", u64::MAX - 2)),
            b"retained V2 manifest",
        )
        .await
        .unwrap();
        let registry = lance_context_core::LanceRegistry::new(
            RolloutRegistry::open_or_create(
                dir.path().join("_registry.rollout.lance").to_str().unwrap(),
                None,
            )
            .await
            .unwrap(),
        );
        assert!(backfill_registry(dir.path().to_str().unwrap(), &registry)
            .await
            .is_err());
        assert!(!registry.contains("invalid").await.unwrap());
        assert_eq!(
            tokio::fs::read(versions.join("1.manifest")).await.unwrap(),
            b"retained V1 manifest"
        );
    }

    #[tokio::test]
    async fn discovers_only_unregistered_rollout_datasets() {
        let dir = TempDir::new().unwrap();
        let existing_uri = dir.path().join("existing.rollout.lance");
        let legacy_uri = dir.path().join("legacy.rollout.lance");
        RolloutStore::open(existing_uri.to_str().unwrap())
            .await
            .unwrap();
        RolloutStore::open(legacy_uri.to_str().unwrap())
            .await
            .unwrap();
        tokio::fs::create_dir(dir.path().join("context.lance"))
            .await
            .unwrap();
        tokio::fs::create_dir(dir.path().join("_stats.rollout.lance"))
            .await
            .unwrap();

        let registry_uri = dir.path().join("_registry.rollout.lance");
        let registry = lance_context_core::LanceRegistry::new(
            RolloutRegistry::open_or_create(registry_uri.to_str().unwrap(), None)
                .await
                .unwrap(),
        );
        registry
            .upsert("existing", existing_uri.to_str().unwrap())
            .await
            .unwrap();

        assert_eq!(
            backfill_registry(dir.path().to_str().unwrap(), &registry)
                .await
                .unwrap(),
            1
        );
        assert_eq!(
            backfill_registry(dir.path().to_str().unwrap(), &registry)
                .await
                .unwrap(),
            0
        );

        let mut names: Vec<String> = registry
            .list()
            .await
            .unwrap()
            .into_iter()
            .map(|entry| entry.name)
            .collect();
        names.sort();
        assert_eq!(names, vec!["existing", "legacy"]);
    }
}
