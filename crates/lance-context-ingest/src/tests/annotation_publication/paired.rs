//! The paired adapter preserves a prepared body identity and the later WAL suffix.
//! HTTP ingestion and automatic collection are separate integrations.
use super::*;
use lance::dataset::{
    transaction::{Operation, Transaction},
    CommitBuilder, InsertBuilder, WriteMode,
};
use lance_context_api::annotation::TurnAnnotation;
use lance_context_core::annotation_staging::{BodySnapshot, Registration, BODY_SNAPSHOT_PROPERTY};
use std::collections::HashMap;

mod maintenance;

fn scope() -> AnnotationScope {
    AnnotationScope {
        table_instance: binding().incarnation.to_string(),
        source_group: 10,
        shard: 0,
    }
}

async fn seed_pair(c: &network::Cluster, narrow: &Storage, body: &Storage) -> Result<Dataset> {
    let body_schema = Arc::new(rollout_schema());
    let body = Dataset::write(
        RecordBatchIterator::new(
            vec![Ok(RecordBatch::new_empty(body_schema.clone()))],
            body_schema,
        ),
        &body.uri,
        Some(WriteParams {
            enable_stable_row_ids: true,
            skip_auto_cleanup: true,
            ..Default::default()
        }),
    )
    .await?;
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("rollout_id", DataType::Utf8, false),
        Field::new(
            "labels",
            DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
            false,
        ),
        Field::new("status", DataType::Utf8, true),
        Field::new("score", DataType::Float32, true),
    ]));
    let base = Dataset::write(
        RecordBatchIterator::new(vec![Ok(RecordBatch::new_empty(schema.clone()))], schema),
        &narrow.uri,
        Some(WriteParams {
            enable_stable_row_ids: true,
            skip_auto_cleanup: true,
            ..Default::default()
        }),
    )
    .await?;
    let reference = BodySnapshot::capture(binding().incarnation.to_string(), &body).await?;
    let mut transaction = Transaction::new_from_version(
        base.version_id(),
        Operation::UpdateConfig {
            config_updates: None,
            table_metadata_updates: None,
            schema_metadata_updates: None,
            field_metadata_updates: Default::default(),
        },
    );
    transaction.transaction_properties = Some(Arc::new(HashMap::from([
        (
            BODY_SNAPSHOT_PROPERTY.into(),
            serde_json::to_string(&reference)?,
        ),
        (
            "lance-context.annotation-cut".into(),
            serde_json::to_string(&(scope(), 0u64))?,
        ),
    ])));
    let base = CommitBuilder::new(Arc::new(base))
        .with_max_retries(0)
        .with_skip_auto_cleanup(true)
        .execute(transaction)
        .await?;
    // This fixture's opaque source schema is shared with the existing RF3
    // publication tests; it is not a new public ingestion wire format.
    seed_base(
        c,
        IngestContract {
            schema: arrow_schema_fingerprint(&patch_schema())?,
            policy: WritePolicy::TurnIngestion,
            ..binding().contract()?
        },
        &base,
    )
    .await?;
    Ok(body)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn paired_preparation_survives_restart_before_publication() -> Result<()> {
    tokio::time::timeout(Duration::from_secs(60), recover(false, None)).await?
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn paired_unknown_publication_preserves_concurrent_updates() -> Result<()> {
    tokio::time::timeout(Duration::from_secs(60), recover(true, None)).await?
}

async fn recover(lose_publication_reply: bool, maintenance_stage: Option<u8>) -> Result<()> {
    let root = if let Ok(parent) = std::env::var("INGEST_TEST_ARTIFACTS") {
        tempfile::Builder::new()
            .prefix("paired-maintenance-")
            .disable_cleanup(true)
            .tempdir_in(parent)?
    } else {
        tempfile::tempdir()?
    };
    let narrow = Storage::resolve(
        root.path()
            .join("annotations.lance")
            .to_str()
            .unwrap()
            .into(),
        Default::default(),
        Arc::default(),
    )
    .await?;
    let mut body_path = root.path().to_path_buf();
    if maintenance_stage.is_some() {
        // The body URI alone must exceed the 512-byte inline descriptor cap.
        // Otherwise short CI temporary paths silently select the inline format.
        for _ in 0..5 {
            body_path.push("body-reference-".repeat(8));
        }
    }
    let body = Storage::resolve(
        body_path.join("body.lance").to_str().unwrap().into(),
        Default::default(),
        narrow.session.clone(),
    )
    .await?;
    let c = network::Cluster::open_signed(root.path(), one_shard()).await?;
    let body_base = seed_pair(&c, &narrow, &body).await?;
    let base = read(&c, &narrow, None).await?;
    let original_version = base.version_id();
    let staged = staging(base.clone(), 0)?;
    let first = append(&c, 1).await?;
    let initial = TurnAnnotation {
        labels: ["original".to_owned()].into(),
        status: Some("reviewed".into()),
        score: None,
    };
    let registered = staged
        .register_committed(
            0,
            first.0,
            &[
                Registration {
                    turn: "T",
                    rollout_id: "s",
                    annotation: &initial,
                },
                Registration {
                    turn: "U",
                    rollout_id: "s",
                    annotation: &initial,
                },
            ],
        )
        .await?;
    assert!(matches!(
        registered,
        lance_context_core::annotation_staging::ApplyOutcome::Applied
    ));
    let selection = select(&c, &first).await?;
    let frozen = staged
        .freeze_with_results(
            selection.pending.plan.transaction,
            Some(serde_json::json!({"version":1,"slots":1,"rejected":[]}).to_string()),
        )
        .await?
        .unwrap();
    let rows = crate::tests::batch(&["T", "U"], &["s", "s"])?;
    let mut columns = rows.columns().to_vec();
    columns[rows.schema().index_of("binary_payload")?] =
        Arc::new(LargeBinaryArray::from_iter_values([
            vec![7u8; 16 << 10],
            vec![9u8; 16 << 10],
        ]));
    let rows = RecordBatch::try_new(rows.schema(), columns)?;
    let expected_bodies = rows.column_by_name("binary_payload").unwrap().clone();
    let body_base = Arc::new(body_base);
    let mut transaction = InsertBuilder::new(body_base.clone())
        .with_params(&WriteParams {
            mode: WriteMode::Append,
            max_rows_per_file: if maintenance_stage.is_some() {
                1
            } else {
                WriteParams::default().max_rows_per_file
            },
            skip_auto_cleanup: true,
            ..Default::default()
        })
        .execute_uncommitted(vec![rows])
        .await?;
    transaction.uuid = frozen.identity().into();
    transaction.transaction_properties = Some(Arc::new(HashMap::from([(
        "lance-context.annotation-cut".into(),
        serde_json::to_string(&(scope(), first.0))?,
    )])));
    let prepared_body = CommitBuilder::new(body_base)
        .with_detached(true)
        .with_max_retries(1)
        .with_skip_auto_cleanup(true)
        .execute(transaction)
        .await?;
    let reference =
        BodySnapshot::capture(binding().incarnation.to_string(), &prepared_body).await?;
    let prepared = frozen
        .prepare_with_body(&MergeMemoryBudget::new(8 << 20), 64, reference.clone())
        .await?;
    assert!(annotations::persist_descriptors(
        &prepared,
        &selection,
        &narrow,
        p::MAX_DESCRIPTOR_BYTES as u64
    )
    .await
    .is_err());
    assert!(
        annotations::mixed::persist_descriptors(
            &prepared,
            &selection,
            &narrow,
            &narrow,
            p::MAX_DESCRIPTOR_BYTES as u64
        )
        .await
        .is_err(),
        "foreign body destination must be rejected"
    );
    let descriptors = annotations::mixed::persist_descriptors(
        &prepared,
        &selection,
        &narrow,
        &body,
        p::MAX_DESCRIPTOR_BYTES as u64,
    )
    .await?;
    let commit = save(&c, &selection, descriptors.clone()).await?;
    let second = append(&c, 2).await?;
    apply(&c, &staged, &second).await?;
    let latest = staged.capture("T").await?.resolve().await?.unwrap();
    assert_eq!(latest.status.as_deref(), Some("reviewed"));
    assert_eq!(latest.score, Some(0.9));
    if lose_publication_reply {
        assert!(annotations::mixed::publish(
            Arc::new(LostReply::new(Arc::new(c.client.manifest_catalog()))),
            &one_shard(),
            &narrow,
            &body,
            commit.clone(),
            &selection,
            &descriptors,
            base.clone(),
        )
        .await
        .is_err());
        let selected = read(&c, &narrow, None).await?;
        let observed = state(&c).await?;
        assert_eq!(
            annotations::mixed::selected_cut(
                &selected,
                &base,
                &descriptors,
                &selection,
                &observed.manifests.as_ref().unwrap().versions[&selected.version_id()],
                &narrow,
                &body
            )
            .await?,
            first.0
        );
        staged.install_published(&prepared, selected).await?;
        assert_eq!(
            staged.capture("T").await?.resolve().await?.unwrap(),
            latest,
            "retiring the frozen create must preserve the later score update"
        );
    }
    assert!(state(&c).await?.progress.checkpoint.is_empty());
    drop((staged, frozen, prepared, base, prepared_body));
    c.shutdown().await?;

    let c = network::Cluster::open_signed(root.path(), one_shard()).await?;
    let recovered = state(&c).await?;
    assert_eq!(
        recovered.progress.prepared.as_ref().unwrap().descriptors,
        descriptors
    );
    let original = read(&c, &narrow, Some(original_version)).await?;
    let selected = if lose_publication_reply {
        read(&c, &narrow, None).await?
    } else {
        Arc::new(
            annotations::mixed::publish(
                Arc::new(c.client.manifest_catalog()),
                &one_shard(),
                &narrow,
                &body,
                commit,
                &selection,
                &descriptors,
                original.clone(),
            )
            .await?,
        )
    };
    let state_after = state(&c).await?;
    let cut = annotations::mixed::selected_cut(
        &selected,
        &original,
        &descriptors,
        &selection,
        &state_after.manifests.as_ref().unwrap().versions[&selected.version_id()],
        &narrow,
        &body,
    )
    .await?;
    assert_eq!(cut, first.0);
    assert_eq!(
        BodySnapshot::read(&selected).await?,
        Some(reference.clone())
    );
    let staged = staging(selected.clone(), cut)?;
    apply(&c, &staged, &second).await?;
    assert_eq!(staged.capture("T").await?.resolve().await?.unwrap(), latest);
    let retained_body = DatasetBuilder::from_uri(&body.uri)
        .with_version(reference.version)
        .load()
        .await?;
    assert_eq!(retained_body.count_rows(None).await?, 2);
    let batches: Vec<RecordBatch> = retained_body
        .scan()
        .project(&["binary_payload"])?
        .try_into_stream()
        .await?
        .try_collect()
        .await?;
    let actual = batches
        .iter()
        .flat_map(|batch| {
            batch
                .column(0)
                .as_any()
                .downcast_ref::<LargeBinaryArray>()
                .unwrap()
                .iter()
        })
        .collect::<Vec<_>>();
    let expected = expected_bodies
        .as_any()
        .downcast_ref::<LargeBinaryArray>()
        .unwrap()
        .iter()
        .collect::<Vec<_>>();
    assert_eq!(actual, expected);
    finish(&c).await?;
    assert_eq!(
        state(&c).await?.progress.checkpoint.get(&10),
        Some(&first.0)
    );
    let suffix = c
        .client
        .request_routed(Request::Fetch {
            group: 10,
            index: second.0,
        })
        .await?;
    assert!(matches!(suffix.reply, Reply::Record(Some(meta)) if meta == second.1));
    if let Some(stage) = maintenance_stage {
        maintenance::recover(root.path(), c, &narrow, &body, staged, &second, stage).await
    } else {
        c.shutdown().await
    }
}
