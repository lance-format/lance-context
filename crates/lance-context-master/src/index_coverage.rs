//! On-demand base-manifest/index metadata only. Never observes WAL or row data.
use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::Json;
use serde_json::{json, Value};

use crate::error::MasterError;
use crate::state::MasterState;

// Bound concurrent metadata opens independently of normal reads and scheduling.
// No waiting queue, fleet scan, stats-table update or periodic background work.
static LOOKUP: tokio::sync::Semaphore = tokio::sync::Semaphore::const_new(1);

pub(crate) async fn handler(
    State(state): State<Arc<MasterState>>,
    Path(name): Path<String>,
) -> Result<(StatusCode, Json<Value>), MasterError> {
    lance_context_core::validate_store_name(&name).map_err(MasterError::InvalidRequest)?;
    let Ok(_permit) = LOOKUP.try_acquire() else {
        return Ok((
            StatusCode::TOO_MANY_REQUESTS,
            Json(json!({"error":"index metadata lookup busy"})),
        ));
    };
    let result = tokio::time::timeout(Duration::from_secs(10), read(&state.rollout_uri(&name)))
        .await
        .map_err(|_| MasterError::Internal("index metadata lookup deadline exceeded".into()))?
        .map_err(|_| MasterError::Internal("index metadata lookup failed".into()))?;
    Ok((
        StatusCode::OK,
        Json(json!({"target":name,"snapshot":result})),
    ))
}

#[derive(Default)]
struct Coverage {
    covered: HashSet<usize>,
    unknown: bool,
    deltas: usize,
}

async fn read(uri: &str) -> lance::Result<Value> {
    // Opening a Dataset and loading its index metadata reads manifests only.
    // Do not open a RolloutStore writer, call observe_cold, scan rows, count rows,
    // or inspect WAL manifests/payloads here. The handle is never used to write.
    let dataset = lance::Dataset::open(uri).await?;
    let fragments = dataset.get_fragments();
    let active: HashSet<usize> = fragments.iter().map(|f| f.id()).collect();
    let id = dataset
        .schema()
        .field("id")
        .ok_or_else(|| lance::Error::invalid_input("missing id column"))?
        .id;
    // Do not use DatasetIndexExt::load_indices: it may open a fragment-reuse
    // mapping or legacy vector index files. Read just this manifest's section.
    let object_store = dataset.object_store(None).await?;
    let indices = lance_table::io::manifest::read_manifest_indexes(
        object_store.as_ref(),
        dataset.manifest_location(),
        dataset.manifest(),
    )
    .await?;
    let needs_translation = indices
        .iter()
        .any(|i| i.name == lance_table::system_index::frag_reuse::FRAG_REUSE_INDEX_NAME);
    // Union deltas of the same named index; summing counts could double count.
    let mut groups: BTreeMap<(String, &'static str), Coverage> = BTreeMap::new();
    for index in indices.iter().filter(|index| {
        index
            .fields
            .iter()
            .copied()
            .filter(|id| !index.covering_fields.contains(id))
            .collect::<Vec<_>>()
            == [id]
    }) {
        let kind = match index.index_details.as_ref().map(|d| d.type_url.as_str()) {
            Some(t) if t.ends_with("BTreeIndexDetails") => "btree",
            Some(t) if t.ends_with("ZoneMapIndexDetails") => "zonemap",
            _ => "unknown",
        };
        let key = (index.name.clone(), kind);
        if groups.len() >= 64 && !groups.contains_key(&key) {
            return Err(lance::Error::invalid_input(
                "too many key indexes for bounded metadata report",
            ));
        }
        let coverage = groups.entry(key).or_default();
        coverage.deltas += 1;
        coverage.unknown |= needs_translation;
        match &index.fragment_bitmap {
            Some(bitmap) => {
                for fragment in &active {
                    match u32::try_from(*fragment) {
                        Ok(id) if bitmap.contains(id) => {
                            coverage.covered.insert(*fragment);
                        }
                        Ok(_) => {}
                        Err(_) => coverage.unknown = true,
                    }
                }
            }
            None => coverage.unknown = true,
        }
    }
    let indices: Vec<_> = groups.into_iter().map(|((name, kind), coverage)| json!({
        "name":name,"kind":kind,"deltas":coverage.deltas,"coverage_known":!coverage.unknown,
        "covered_fragments":(!coverage.unknown).then_some(coverage.covered.len()),
        "uncovered_fragments":(!coverage.unknown).then_some(active.len()-coverage.covered.len()),
    })).collect();
    Ok(
        json!({"base_version":dataset.version().version,"fragment_count":active.len(),"fragment_reuse_translation_required":needs_translation,"key_indices":indices,"scope":"base manifest and index metadata only; no row or WAL scan"}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use lance_context_core::{KeyIndexType, RolloutStore, RolloutStoreOptions};

    #[tokio::test]
    #[ignore = "requires ETCD_TEST_ENDPOINTS"]
    async fn metadata_coverage_route_is_bounded_and_validates_names() {
        use axum::{body::Body, http::Request};
        use clap::Parser;
        use tower::ServiceExt;
        let dir = tempfile::TempDir::new().unwrap();
        let mut config = crate::config::MasterConfig::parse_from(["test"]);
        config.data_dir = dir.path().to_string_lossy().into_owned();
        config.etcd.etcd_endpoints = std::env::var("ETCD_TEST_ENDPOINTS")
            .unwrap()
            .split(',')
            .map(str::to_owned)
            .collect();
        config.etcd.etcd_prefix =
            format!("/index-coverage-test/{}", lance_context_core::generate_id());
        config.worker_endpoints.clear();
        config.stats_scan_interval_secs = 0;
        let state = MasterState::new(config).await.unwrap();
        let mut store = RolloutStore::open(&state.rollout_uri("hot")).await.unwrap();
        let router = crate::routes::api_router().with_state(state);
        let request = |name: &str| {
            Request::builder()
                .uri(format!("/experiments/{name}/key-index-coverage"))
                .body(Body::empty())
                .unwrap()
        };
        let permit = LOOKUP.acquire().await.unwrap();
        let response = router.clone().oneshot(request("hot")).await.unwrap();
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
        drop(permit);
        let response = router.clone().oneshot(request("hot")).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), 16 * 1024)
            .await
            .unwrap();
        let value: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(value["target"], "hot");
        assert_eq!(value["snapshot"]["fragment_count"], 0);
        assert_eq!(value["snapshot"]["key_indices"], json!([]));
        let response = router.oneshot(request("_registry")).await.unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        store.close().await.unwrap();
    }

    #[tokio::test]
    async fn metadata_coverage_reports_partial_indexes_without_writes() {
        for kind in [KeyIndexType::Btree, KeyIndexType::Zonemap] {
            let dir = tempfile::TempDir::new().unwrap();
            let uri = dir.path().to_str().unwrap();
            let mut store = RolloutStore::open_with_options(
                uri,
                RolloutStoreOptions {
                    merge_after_generations: Some(0),
                    key_index_type: kind,
                    ..Default::default()
                },
            )
            .await
            .unwrap();
            for i in 0..2 {
                let dto = serde_json::from_value(
                    json!({"id":format!("row-{i}"),"rollout_id":"r","content":"payload"}),
                )
                .unwrap();
                store
                    .add(&[lance_context_core::rollout_record_from_add_request(&dto)])
                    .await
                    .unwrap();
                store.cleanup_own_shard().await.unwrap();
                if i == 0 {
                    store.create_id_key_index().await.unwrap();
                }
            }
            let before = lance::Dataset::open(uri).await.unwrap().version().version;
            let report = read(uri).await.unwrap();
            assert_eq!(report["base_version"], before);
            assert_eq!(report["fragment_count"], 2);
            let indices = report["key_indices"].as_array().unwrap();
            assert_eq!(indices.len(), 1);
            assert_eq!(indices[0]["kind"], kind.as_str());
            assert_eq!(indices[0]["coverage_known"], true);
            assert_eq!(indices[0]["covered_fragments"], 1);
            assert_eq!(indices[0]["uncovered_fragments"], 1);
            assert_eq!(
                lance::Dataset::open(uri).await.unwrap().version().version,
                before
            );
            assert_eq!(store.pending_wal_generations().await.unwrap(), 0);
            assert_eq!(store.list(None, None).await.unwrap().len(), 2);
            store.close().await.unwrap();

            // Metadata-only negative fixture: deliberately publish an FRI entry
            // with no mapping file. A query-facing load_indices would try to
            // open the mapping; this diagnostic must report unknown without it.
            let dataset = Arc::new(lance::Dataset::open(uri).await.unwrap());
            let object_store = dataset.object_store(None).await.unwrap();
            let raw = lance_table::io::manifest::read_manifest_indexes(
                object_store.as_ref(),
                dataset.manifest_location(),
                dataset.manifest(),
            )
            .await
            .unwrap();
            let mut mapping = raw.iter().find(|i| !i.fields.is_empty()).unwrap().clone();
            mapping.uuid = lance_context_core::new_uuid();
            mapping.name = lance_table::system_index::frag_reuse::FRAG_REUSE_INDEX_NAME.into();
            mapping.fields.clear();
            mapping.covering_fields.clear();
            mapping.fragment_bitmap = None;
            mapping.index_details = None;
            mapping.index_version = 1;
            mapping.files = None;
            let transaction = lance::dataset::transaction::Transaction::new(
                dataset.version().version,
                lance::dataset::transaction::Operation::CreateIndex {
                    new_indices: vec![mapping],
                    removed_indices: vec![],
                },
                None,
            );
            let changed = lance::dataset::CommitBuilder::new(dataset)
                .execute(transaction)
                .await
                .unwrap();
            let report = read(uri).await.unwrap();
            assert_eq!(report["fragment_reuse_translation_required"], true);
            assert_eq!(report["key_indices"][0]["coverage_known"], false);
            assert!(report["key_indices"][0]["covered_fragments"].is_null());
            assert!(
                lance::index::DatasetIndexExt::load_indices(&changed)
                    .await
                    .is_err(),
                "the missing FRI mapping must be needed by the query-facing loader"
            );
        }
    }
}
