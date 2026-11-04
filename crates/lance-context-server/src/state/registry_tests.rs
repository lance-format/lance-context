use super::*;
use axum::extract::{Path, State};
use lance::Result as LanceResult;
use lance_context_core::RegistryEntry;
use tempfile::TempDir;

use crate::routes::rollouts::get_rollout_store;

async fn create_at(uri: &str) {
    RolloutStore::open_with_options(uri, RolloutStoreOptions::default())
        .await
        .unwrap()
        .close()
        .await
        .unwrap();
}

#[tokio::test]
async fn registered_uri_controls_open_metadata_and_uncached_deletion() {
    let dir = TempDir::new().unwrap();
    let state = Arc::new(AppState::new_for_test(dir.path().to_path_buf()).await);
    let default_uri = state.rollout_uri("logical");
    let registered_uri = dir.path().join("physical-generation.lance");
    let registered_uri = registered_uri.to_str().unwrap();
    // A valid dataset at the derived path must not shadow the registered one.
    create_at(&default_uri).await;
    create_at(registered_uri).await;
    state
        .rollout_registry
        .upsert("logical", registered_uri)
        .await
        .unwrap();

    let opened = state.get_or_open_rollout_store("logical").await.unwrap();
    assert_eq!(opened.read().await.uri(), registered_uri);
    let info = get_rollout_store(State(state.clone()), Path("logical".into()))
        .await
        .unwrap()
        .0;
    assert_eq!(info.uri, registered_uri);
    opened.write().await.close().await.unwrap();
    drop(opened);
    state.rollout_stores.lock().await.pop("logical");
    assert!(state
        .rollout_handles
        .lock("logical")
        .await
        .upgrade()
        .is_none());

    assert!(state.unregister_rollout("logical").await.unwrap());
    assert!(!std::path::Path::new(registered_uri).exists());
    assert!(std::path::Path::new(&default_uri).exists());
    assert!(!state.rollout_registry.contains("logical").await.unwrap());
    assert!(matches!(
        state.get_or_open_rollout_store("logical").await,
        Err(AppError::NotFound(_))
    ));
}

#[tokio::test]
async fn registered_uri_mismatch_does_not_delete_live_or_replacement_data() {
    let dir = TempDir::new().unwrap();
    let state = AppState::new_for_test(dir.path().to_path_buf()).await;
    let first = dir.path().join("first.lance").to_str().unwrap().to_string();
    let second = dir
        .path()
        .join("second.lance")
        .to_str()
        .unwrap()
        .to_string();
    create_at(&first).await;
    create_at(&second).await;
    state
        .rollout_registry
        .upsert("logical", &first)
        .await
        .unwrap();
    let opened = state.get_or_open_rollout_store("logical").await.unwrap();
    state
        .rollout_registry
        .upsert("logical", &second)
        .await
        .unwrap();
    assert!(matches!(
        state.unregister_rollout("logical").await,
        Err(AppError::Overloaded(_))
    ));
    assert!(std::path::Path::new(&first).exists());
    assert!(std::path::Path::new(&second).exists());
    assert_eq!(
        state
            .rollout_registry
            .get("logical")
            .await
            .unwrap()
            .unwrap()
            .uri,
        second
    );
    assert!(Arc::ptr_eq(
        state.rollout_stores.lock().await.peek("logical").unwrap(),
        &opened
    ));
    state.shutdown().await;
}

// Serve a changed registry entry at the post-open observation. Only registry
// timing is controlled; each open still uses a real Lance dataset.
struct ChangingRegistry {
    initial: RegistryEntry,
    current: Option<RegistryEntry>,
    reads: std::sync::atomic::AtomicUsize,
}

#[async_trait::async_trait]
impl StoreRegistry for ChangingRegistry {
    async fn get(&self, _: &str) -> LanceResult<Option<RegistryEntry>> {
        if self.reads.fetch_add(1, std::sync::atomic::Ordering::SeqCst) == 0 {
            Ok(Some(self.initial.clone()))
        } else {
            Ok(self.current.clone())
        }
    }
    async fn contains(&self, _: &str) -> LanceResult<bool> {
        unreachable!()
    }
    async fn list(&self) -> LanceResult<Vec<RegistryEntry>> {
        unreachable!()
    }
    async fn upsert(&self, _: &str, _: &str) -> LanceResult<()> {
        unreachable!()
    }
    async fn remove(&self, _: &str) -> LanceResult<()> {
        unreachable!()
    }
    async fn insert_missing(&self, _: &[(String, String)]) -> LanceResult<usize> {
        unreachable!()
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[tokio::test]
async fn registry_change_during_open_never_installs_a_stale_handle() {
    let dir = TempDir::new().unwrap();
    let mut state = AppState::new_for_test(dir.path().to_path_buf()).await;
    let uri = dir.path().join("first.lance").to_str().unwrap().to_string();
    create_at(&uri).await;
    let initial = RegistryEntry {
        name: "logical".into(),
        uri,
        created_at: 1,
    };
    let moved = RegistryEntry {
        uri: dir
            .path()
            .join("second.lance")
            .to_str()
            .unwrap()
            .to_string(),
        ..initial.clone()
    };
    create_at(&moved.uri).await;
    let recreated = RegistryEntry {
        created_at: 2,
        ..initial.clone()
    };
    for current in [None, Some(moved), Some(recreated)] {
        let removed = current.is_none();
        state.rollout_registry = Arc::new(ChangingRegistry {
            initial: initial.clone(),
            current,
            reads: Default::default(),
        });
        let result = state.get_or_open_rollout_store("logical").await;
        if removed {
            assert!(matches!(result, Err(AppError::NotFound(_))));
        } else {
            assert!(matches!(result, Err(AppError::Overloaded(_))));
        }
        assert!(state.rollout_stores.lock().await.peek("logical").is_none());
        assert!(state
            .rollout_handles
            .lock("logical")
            .await
            .upgrade()
            .is_none());
    }
}

#[tokio::test]
async fn store_info_timing_preserves_lock_wait_and_metadata() {
    use metrics_util::debugging::{DebugValue, DebuggingRecorder};
    use std::time::Duration;

    let dir = TempDir::new().unwrap();
    let state = Arc::new(AppState::new_for_test(dir.path().to_path_buf()).await);
    let uri = dir
        .path()
        .join("registered.lance")
        .to_str()
        .unwrap()
        .to_string();
    create_at(&uri).await;
    state
        .rollout_registry
        .upsert("logical", &uri)
        .await
        .unwrap();
    let opened = state.get_or_open_rollout_store("logical").await.unwrap();
    let held = opened.write().await;
    let expected_version = held.version();
    let recorder = DebuggingRecorder::new();
    let snapshots = recorder.snapshotter();
    let _recorder = metrics::set_default_local_recorder(&recorder);
    tokio::time::pause();

    let mut request = Box::pin(get_rollout_store(
        State(state.clone()),
        Path("logical".into()),
    ));
    assert!(futures::poll!(&mut request).is_pending());
    tokio::time::advance(Duration::from_secs(6)).await;
    assert!(futures::poll!(&mut request).is_pending());
    drop(held);
    let info = request.await.unwrap().0;
    assert_eq!(info.uri, uri);
    assert_eq!(info.version, Some(expected_version));

    let series = snapshots.snapshot().into_vec();
    let stages: Vec<_> = series
        .iter()
        .filter(|(key, _, _, _)| key.key().name() == "rollout_store_stage_duration_seconds")
        .collect();
    assert_eq!(stages.len(), 3);
    for (key, _, _, value) in stages {
        let labels: std::collections::BTreeMap<_, _> =
            key.key().labels().map(|l| (l.key(), l.value())).collect();
        assert_eq!(labels["outcome"], "completed");
        let DebugValue::Histogram(values) = value else {
            panic!("expected histogram")
        };
        assert_eq!(values.len(), 1);
        assert_eq!(
            values[0].into_inner(),
            match labels["stage"] {
                "info_resolve" | "cache_lock" => 0.0,
                "info_lock" => 6.0,
                other => panic!("unexpected stage: {other}"),
            }
        );
    }
    tokio::time::resume();
    state.shutdown().await;
}
