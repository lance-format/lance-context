//! Actual large native preparations, RF3 mutation ACKs and owner recovery.
use super::*;
use crate::annotations::runtime::{Configuration, Runtime};
use crate::materializer::{Acquisition, Options};
use crate::tests::annotations::{producer, request, run, wire};
use crate::{annotations::Service, tests::annotation_runtime as runtime};
use lance::index::DatasetIndexExt;
use object_store::ObjectStoreExt;

fn large_binding() -> TableBinding {
    TableBinding {
        max_rows: 2048,
        ..binding()
    }
}

async fn owner(
    c: &network::Cluster,
    storage: &Storage,
    limits: &Limits,
) -> Result<(Service, Runtime)> {
    Runtime::connect(
        c.client.clone(),
        one_shard(),
        large_binding(),
        storage.clone(),
        Configuration {
            limits: limits.clone(),
            upload_budget: MergeMemoryBudget::new(32 << 20),
            apply_budget: MergeMemoryBudget::new(16 << 20),
            publication_budget: MergeMemoryBudget::new(8 << 20),
            publication: Options {
                max_source_bytes: 1 << 20,
                min_source_bytes: 1 << 20,
                max_records: 2,
                max_batch_wait: Duration::from_millis(20),
                ..Default::default()
            },
        },
        Acquisition {
            owner: Uuid::new_v4(),
            request: Uuid::new_v4(),
            expected_epoch: state(c).await?.authority.epoch,
        },
    )
    .await
}

async fn seed_large(
    c: &network::Cluster,
    storage: &Storage,
    limits: &Limits,
    fragments: usize,
) -> Result<()> {
    let rows = fragments * 2;
    let ids: Vec<_> = (0..rows).map(|n| format!("t{n:08}")).collect();
    let mut labels = ListBuilder::new(StringBuilder::new());
    for _ in 0..rows {
        labels.append(true);
    }
    let labels = labels.finish();
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("body", DataType::LargeBinary, false),
        Field::new("labels", labels.data_type().clone(), false),
        Field::new("status", DataType::Utf8, true),
        Field::new("score", DataType::Float32, true),
    ]));
    let body = vec![0x5a; 1024];
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(StringArray::from(ids)),
            Arc::new(LargeBinaryArray::from_iter_values(
                (0..rows).map(|_| body.as_slice()),
            )),
            Arc::new(labels),
            Arc::new(StringArray::from(vec!["pending"; rows])),
            Arc::new(Float32Array::from(vec![None; rows])),
        ],
    )?;
    let mut base = Dataset::write(
        RecordBatchIterator::new(vec![Ok(batch)], schema),
        &storage.uri,
        Some(WriteParams {
            commit_handler: Some(Arc::new(ConditionalPutCommitHandler)),
            enable_stable_row_ids: true,
            max_rows_per_file: 2,
            max_rows_per_group: 2,
            ..Default::default()
        }),
    )
    .await?;
    assert_eq!(base.manifest().fragments.len(), fragments);
    base.create_index_builder(
        &["id"],
        lance_index::IndexType::BTree,
        &lance_index::scalar::ScalarIndexParams::default(),
    )
    .name("annotation_point_id".into())
    .await?;
    seed_base(c, Service::contract(&large_binding(), limits)?, &base).await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn large_annotation_preparation_recovers_original_files_and_rejects_missing_metadata(
) -> Result<()> {
    recover(Stop::Prepared, 4096).await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn large_annotation_preparation_resolves_lost_publication_reply_and_drains() -> Result<()> {
    recover(Stop::Selected, 4096).await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn large_annotation_preparation_recovers_confirmed_cut_before_gc() -> Result<()> {
    recover(Stop::Confirmed, 4096).await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn large_annotation_preparation_respects_configured_catalog_limit() -> Result<()> {
    recover(Stop::Prepared, 1024).await
}

async fn recover(stage: Stop, fragments: usize) -> Result<()> {
    tokio::time::timeout(Duration::from_secs(180), recover_inner(stage, fragments)).await?
}

async fn recover_inner(stage: Stop, fragments: usize) -> Result<()> {
    let root = tempfile::tempdir()?;
    let storage = Storage::resolve(
        root.path().join("turns.lance").to_str().unwrap().into(),
        Default::default(),
        Arc::default(),
    )
    .await?;
    let limits = Limits {
        retained_bytes: 3 << 20,
        retained_entries: 8192,
        value_bytes: 4096,
        readers: 16,
    };
    let c = network::Cluster::open_signed(root.path(), one_shard()).await?;
    seed_large(&c, &storage, &limits, fragments).await?;
    let (service, driver) = Service::connect(
        c.client.clone(),
        one_shard(),
        large_binding(),
        &storage,
        limits.clone(),
        MergeMemoryBudget::new(32 << 20),
        MergeMemoryBudget::new(16 << 20),
    )
    .await?;
    let (stop, task) = run(driver);
    let a = producer(&service, 1).await?;
    let b = producer(&service, 2).await?;
    let ids: Vec<_> = (0..fragments * 2)
        .step_by(2)
        .map(|n| format!("t{n:08}"))
        .collect();
    let mut last = None;
    for (sequence, chunk) in ids.chunks(2048).enumerate() {
        // Keep only the latest exact request. Two producers' retry requests
        // later coexist, each charged for the configured 2048-row maximum.
        drop(last.take());
        let keys: Vec<_> = chunk.iter().map(String::as_str).collect();
        let req = request(
            &service,
            &a,
            sequence as u64 + 1,
            wire(&limits, &keys, 1, Some("reviewed"), 0, None)?,
        )
        .await?;
        let result = service.submit(&req, Duration::from_secs(30)).await?;
        assert_eq!(result.result, crate::annotations::MutationResult::Applied);
        last = Some((req, result));
    }
    let (last_request, last_result) = last.unwrap();
    let frozen = service.freeze(Uuid::new_v4()).await?.unwrap();
    let selection = select_range(
        &c,
        frozen.sources().records.clone(),
        frozen.through(),
        Uuid::parse_str(frozen.identity())?,
    )
    .await?;
    let prepared = frozen
        .prepare(&MergeMemoryBudget::new(8 << 20), 8192)
        .await?;
    let original = prepared.annotations().merge().transaction.clone();
    let base = prepared.annotations().base().clone();
    let native_bytes = pb::Transaction::from(&original).encoded_len();
    assert!(native_bytes as u64 > c.client.max_descriptor_bytes());
    let inline = annotations::descriptors(prepared.annotations(), &selection);
    if fragments == 4096 {
        assert!(native_bytes > p::MAX_DESCRIPTOR_BYTES && inline.is_err());
    } else {
        assert!(native_bytes <= p::MAX_DESCRIPTOR_BYTES);
        assert!(inline?.validate(c.client.max_descriptor_bytes()).is_err());
    }
    let descriptors = annotations::persist_descriptors(
        prepared.annotations(),
        &selection,
        &storage,
        c.client.max_descriptor_bytes(),
    )
    .await?;
    assert_eq!(descriptors.format, 5);
    assert!(descriptors.bytes.len() < 1024);
    let commit = save(&c, &selection, descriptors.clone()).await?;
    let suffix = request(
        &service,
        &b,
        1,
        wire(&limits, &["t00000000"], 0, None, 1, Some(0.9))?,
    )
    .await?;
    let suffix_result = service.submit(&suffix, Duration::from_secs(30)).await?;
    let current = service
        .get("t00000000", Duration::from_secs(30))
        .await?
        .unwrap();
    assert_eq!(current.status.as_deref(), Some("reviewed"));
    assert_eq!(current.score, Some(0.9));
    if stage != Stop::Prepared {
        assert!(annotations::publish(
            Arc::new(LostReply(Arc::new(c.client.manifest_catalog()))),
            &one_shard(),
            &storage,
            commit.clone(),
            &selection,
            &descriptors,
            base.clone()
        )
        .await
        .is_err());
    }
    if stage == Stop::Confirmed {
        confirm(&c).await?;
    }
    stop.send(()).unwrap();
    task.await??;
    drop((service, prepared, frozen));
    c.shutdown().await?;
    let c = network::Cluster::open_signed(root.path(), one_shard()).await?;
    assert_eq!(
        state(&c)
            .await?
            .progress
            .prepared
            .as_ref()
            .unwrap()
            .descriptors,
        descriptors
    );
    if stage == Stop::Prepared {
        rejects_broken_metadata(
            &c,
            &storage,
            &limits,
            &base,
            &selection,
            &descriptors,
            &commit,
        )
        .await?;
    }
    let (service, runtime) = owner(&c, &storage, &limits).await?;
    let (stop, task) = runtime::run_owner(runtime);
    assert_eq!(
        service
            .submit(&last_request, Duration::from_secs(30))
            .await?,
        last_result
    );
    assert_eq!(
        service.submit(&suffix, Duration::from_secs(30)).await?,
        suffix_result
    );
    assert_eq!(
        service
            .get("t00000000", Duration::from_secs(30))
            .await?
            .unwrap(),
        current
    );
    runtime::drained(&service, suffix_result.index).await?;
    let published = read(&c, &storage, Some(selection.pending.plan.target_version()?)).await?;
    assert_eq!(published.read_transaction().await?, Some(original));
    let selected = read(&c, &storage, None).await?;
    assert_eq!(
        selected
            .count_rows(Some("status = 'reviewed'".into()))
            .await?,
        fragments
    );
    assert_eq!(
        selected
            .count_rows(Some("status = 'pending'".into()))
            .await?,
        fragments
    );
    assert_eq!(selected.count_rows(Some("score = 0.9".into())).await?, 1);
    let body_id = base.schema().field_id("body")?;
    let body_files = |dataset: &Dataset| {
        dataset
            .manifest()
            .fragments
            .iter()
            .flat_map(|f| &f.files)
            .filter(|f| f.fields.contains(&body_id))
            .map(|f| f.path.clone())
            .collect::<Vec<_>>()
    };
    assert_eq!(body_files(&base), body_files(&selected));
    if stage == Stop::Prepared && fragments == 4096 {
        // Start each next batch immediately after its ACK and point read. The
        // publisher runs concurrently; do not wait for a drain between batches.
        let mut through = suffix_result.index;
        for round in 0..3 {
            let status = format!("round-{}", round + 1);
            for (batch, chunk) in ids.chunks(2048).enumerate() {
                let keys: Vec<_> = chunk.iter().map(String::as_str).collect();
                let req = request(
                    &service,
                    &a,
                    3 + (round * 2 + batch) as u64,
                    wire(&limits, &keys, 1, Some(&status), 0, None)?,
                )
                .await?;
                let result = service.submit(&req, Duration::from_secs(30)).await?;
                assert_eq!(result.result, crate::annotations::MutationResult::Applied);
                through = result.index;
                assert_eq!(
                    service
                        .get(keys[0], Duration::from_secs(30))
                        .await?
                        .unwrap()
                        .status
                        .as_deref(),
                    Some(status.as_str())
                );
                assert!(service.observation().await.0.retained_bytes <= limits.retained_bytes);
            }
        }
        runtime::drained(&service, through).await?;
        let selected = read(&c, &storage, None).await?;
        assert_eq!(
            selected
                .count_rows(Some("status = 'round-3'".into()))
                .await?,
            fragments
        );
        assert_eq!(
            selected
                .count_rows(Some("status = 'pending'".into()))
                .await?,
            fragments
        );
        assert_eq!(body_files(&base), body_files(&selected));
        let value = service
            .get("t00000000", Duration::from_secs(30))
            .await?
            .unwrap();
        assert_eq!(value.status.as_deref(), Some("round-3"));
        assert_eq!(value.score, Some(0.9));
        assert!(matches!(
            c.client
                .request_routed(Request::Fetch {
                    group: 10,
                    index: through
                })
                .await?
                .reply,
            Reply::Record(None)
        ));
    }
    for index in [last_result.index, suffix_result.index] {
        assert!(matches!(
            c.client
                .request_routed(Request::Fetch { group: 10, index })
                .await?
                .reply,
            Reply::Record(None)
        ));
    }
    stop.send(()).unwrap();
    task.await??;
    c.shutdown().await
}

async fn rejects_broken_metadata(
    c: &network::Cluster,
    storage: &Storage,
    limits: &Limits,
    base: &Arc<Dataset>,
    selection: &annotations::Selection,
    descriptors: &p::Descriptors,
    commit: &p::lance::ManifestCommit,
) -> Result<()> {
    let store = base.object_store(None).await?;
    let prefix = storage.base.clone().join("_annotation_preparations");
    let objects: Vec<_> = store.inner.list(Some(&prefix)).try_collect().await?;
    assert_eq!(objects.len(), 1);
    let path = &objects[0].location;
    let bytes = store.inner.get(path).await?.bytes().await?;
    store.inner.delete(path).await?;
    assert!(owner(c, storage, limits).await.is_err());
    let mut corrupt = bytes.to_vec();
    corrupt[0] ^= 1;
    store.inner.put(path, Bytes::from(corrupt).into()).await?;
    assert!(owner(c, storage, limits).await.is_err());
    // A reference may not redirect recovery into another table/plan or bypass
    // the size bound, even with a correctly recomputed descriptor digest.
    for field in ["table", "incarnation", "transaction", "bytes"] {
        let mut reference: serde_json::Value = serde_json::from_slice(&descriptors.bytes)?;
        reference[field] = if field == "bytes" {
            (u64::MAX).into()
        } else {
            Uuid::new_v4().to_string().into()
        };
        let invalid =
            p::Descriptors::lance_annotation_reference(serde_json::to_vec(&reference)?.into())?;
        let mut changed = commit.clone();
        changed.descriptor_digest = invalid.digest;
        assert!(annotations::publish(
            Arc::new(c.client.manifest_catalog()),
            &one_shard(),
            storage,
            changed,
            selection,
            &invalid,
            base.clone()
        )
        .await
        .is_err());
    }
    let state = state(c).await?;
    assert!(state.progress.published.is_none());
    assert!(state.progress.checkpoint.is_empty());
    for (index, _) in &selection.pending.plan.sources[&10].records {
        assert!(matches!(
            c.client
                .request_routed(Request::Fetch {
                    group: 10,
                    index: *index
                })
                .await?
                .reply,
            Reply::Record(Some(_))
        ));
    }
    store.inner.put(path, bytes.into()).await?;
    Ok(())
}
