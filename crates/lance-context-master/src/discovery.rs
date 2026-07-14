//! Startup discovery for rollout datasets created before the registry existed.

use std::path::Path;

use lance::io::ObjectStore;
use lance_context_core::RolloutRegistry;

const ROLLOUT_SUFFIX: &str = ".rollout.lance";

/// Discover top-level rollout datasets under `data_dir` and add any missing
/// registry rows in one batch. Returns the number of rows inserted.
pub async fn backfill_registry(
    data_dir: &str,
    registry: &mut RolloutRegistry,
) -> lance::Result<usize> {
    let (store, base_path) = ObjectStore::from_uri(data_dir).await?;
    let mut children = store.read_dir(base_path).await?;
    children.sort();

    let entries: Vec<(String, String)> = children
        .into_iter()
        .filter_map(|child| {
            let name = child.strip_suffix(ROLLOUT_SUFFIX)?;
            if name.is_empty() || matches!(name, "_registry" | "_stats") {
                return None;
            }
            Some((name.to_string(), join_uri(data_dir, &child)))
        })
        .collect();

    registry.insert_missing(&entries).await
}

fn join_uri(base: &str, child: &str) -> String {
    if base.contains("://") {
        format!("{}/{}", base.trim_end_matches('/'), child)
    } else {
        Path::new(base).join(child).to_string_lossy().to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lance_context_core::RolloutStore;
    use tempfile::TempDir;

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
        let mut registry = RolloutRegistry::open_or_create(registry_uri.to_str().unwrap(), None)
            .await
            .unwrap();
        registry
            .upsert("existing", existing_uri.to_str().unwrap())
            .await
            .unwrap();

        assert_eq!(
            backfill_registry(dir.path().to_str().unwrap(), &mut registry)
                .await
                .unwrap(),
            1
        );
        assert_eq!(
            backfill_registry(dir.path().to_str().unwrap(), &mut registry)
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

    #[test]
    fn joins_local_paths_and_object_store_uris() {
        assert_eq!(
            join_uri("/data/rollouts", "exp.rollout.lance"),
            "/data/rollouts/exp.rollout.lance"
        );
        assert_eq!(
            join_uri("s3://bucket/prefix/", "exp.rollout.lance"),
            "s3://bucket/prefix/exp.rollout.lance"
        );
    }
}
