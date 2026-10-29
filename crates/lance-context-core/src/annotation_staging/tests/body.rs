//! Body references belong to the immutable frozen preparation.
use super::*;
use arrow_array::{
    builder::{Int32Builder, LargeBinaryBuilder, StringDictionaryBuilder},
    new_null_array,
    types::Int8Type,
    ArrayRef, Int32Array, LargeStringArray, TimestampMicrosecondArray,
};
use lance::dataset::{transaction::Operation, InsertBuilder, WriteMode};

const BODY_BYTES: usize = 64 << 10;

pub(super) fn imported(start: usize, count: usize) -> Result<RecordBatch> {
    let ids = (start..start + count)
        .map(|n| format!("turn-{n:06}"))
        .collect::<Vec<_>>();
    let groups = (start..start + count)
        .map(|n| format!("session-{}", n / 8))
        .collect::<Vec<_>>();
    let mut labels = ListBuilder::new(StringBuilder::new());
    for _ in 0..count {
        labels.values().append_value("imported");
        labels.append(true);
    }
    let labels = labels.finish();
    let mut bodies = LargeBinaryBuilder::new();
    for n in start..start + count {
        let mut bytes = vec![(n % 251) as u8; BODY_BYTES];
        bytes[..8].copy_from_slice(&(n as u64).to_le_bytes());
        bodies.append_value(bytes);
    }
    let mut roles = StringDictionaryBuilder::<Int8Type>::new();
    let mut tokens = ListBuilder::new(Int32Builder::new());
    for n in start..start + count {
        roles.append("assistant")?;
        for token in n..n + 3 {
            tokens.values().append_value(token as i32);
        }
        tokens.append(true);
    }
    let ids: ArrayRef = Arc::new(StringArray::from(ids));
    let groups: ArrayRef = Arc::new(StringArray::from(groups));
    let bodies: ArrayRef = Arc::new(bodies.finish());
    let roles: ArrayRef = Arc::new(roles.finish());
    let tokens: ArrayRef = Arc::new(tokens.finish());
    let rollout = crate::rollout_store::rollout_schema();
    let mut columns = rollout
        .fields()
        .iter()
        .map(|field| -> ArrayRef {
            match field.name().as_str() {
                "id" => ids.clone(),
                "rollout_id" => groups.clone(),
                "binary_payload" => bodies.clone(),
                "role" => roles.clone(),
                "input_tokens" => tokens.clone(),
                "problem_id" => Arc::new(StringArray::from(
                    (start..start + count)
                        .map(|n| format!("problem-{}", n / 32))
                        .collect::<Vec<_>>(),
                )),
                "sequence_order" => Arc::new(Int32Array::from(
                    (start..start + count)
                        .map(|n| (n % 8) as i32)
                        .collect::<Vec<_>>(),
                )),
                "created_at" => Arc::new(TimestampMicrosecondArray::from(
                    (start..start + count)
                        .map(|n| n as i64 * 1000)
                        .collect::<Vec<_>>(),
                )),
                "content_type" => Arc::new(StringArray::from(vec!["text/plain"; count])),
                "content" => Arc::new(LargeStringArray::from(
                    (start..start + count)
                        .map(|n| format!("content-{n}"))
                        .collect::<Vec<_>>(),
                )),
                _ => {
                    assert!(field.is_nullable(), "unfilled required rollout column");
                    new_null_array(field.data_type(), count)
                }
            }
        })
        .collect::<Vec<_>>();
    let mut fields = rollout.fields().to_vec();
    fields.extend([
        Arc::new(Field::new("labels", labels.data_type().clone(), false)),
        Arc::new(Field::new("status", DataType::Utf8, true)),
    ]);
    columns.push(Arc::new(labels));
    columns.push(Arc::new(StringArray::from(vec!["pending"; count])));
    RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).map_err(Into::into)
}

fn write_params() -> WriteParams {
    WriteParams {
        mode: WriteMode::Append,
        max_rows_per_file: 32,
        max_bytes_per_file: 8 << 20,
        enable_stable_row_ids: true,
        skip_auto_cleanup: true,
        file_writer_options: Some(lance_file::writer::FileWriterOptions {
            data_cache_bytes: Some(2 << 20),
            max_page_bytes: Some(2 << 20),
            ..Default::default()
        }),
        ..Default::default()
    }
}

pub(super) async fn detached(base: Arc<Dataset>, batch: RecordBatch) -> Result<Arc<Dataset>> {
    let transaction = InsertBuilder::new(base.clone())
        .with_params(&write_params())
        .execute_uncommitted(vec![batch])
        .await?;
    assert!(matches!(transaction.operation, Operation::Append { .. }));
    let result = CommitBuilder::new(base)
        .with_detached(true)
        // Lance's detached loop counts total attempts, unlike its mainline
        // commit loop. One here means a single attempt, with no retry.
        .with_max_retries(1)
        .with_skip_auto_cleanup(true)
        .execute(transaction.clone())
        .await?;
    assert!(lance_table::format::is_detached_version(
        result.version_id()
    ));
    assert_eq!(
        result.read_transaction().await?.as_ref(),
        Some(&transaction)
    );
    Ok(Arc::new(result))
}

async fn selected_body(narrow: &Dataset) -> Result<(BodySnapshot, Arc<Dataset>)> {
    let transaction = narrow.read_transaction().await?.unwrap();
    let properties = transaction.transaction_properties.as_ref().unwrap();
    let reference = BodySnapshot::read(narrow).await?.unwrap();
    let (selected_scope, cut): (Scope, u64) =
        serde_json::from_str(&properties["lance-context.annotation-cut"])?;
    assert_eq!(selected_scope, scope());
    assert!(cut > 0);
    assert_eq!(reference.table_instance, scope().table_instance);
    let body = lance::dataset::builder::DatasetBuilder::from_uri(&reference.uri)
        .with_version(reference.version)
        .load()
        .await?;
    assert_eq!(
        body.read_transaction().await?.unwrap().uuid,
        reference.transaction
    );
    Ok((reference, Arc::new(body)))
}

async fn verify_body(body: &Dataset, expected_rows: usize) -> Result<()> {
    let mut scan = body.scan();
    scan.project(&["id", "binary_payload"])?;
    scan.batch_size(16).batch_readahead(1);
    let mut stream = scan.try_into_stream().await?;
    let mut seen = BTreeSet::new();
    while let Some(batch) = stream.try_next().await? {
        let ids = batch
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let bodies = batch
            .column(1)
            .as_any()
            .downcast_ref::<LargeBinaryArray>()
            .unwrap();
        for i in 0..batch.num_rows() {
            let id = ids.value(i);
            let n = id.strip_prefix("turn-").unwrap().parse::<usize>().unwrap();
            assert!(n < expected_rows && seen.insert(n));
            let bytes = bodies.value(i);
            assert_eq!(bytes.len(), BODY_BYTES);
            assert_eq!(&bytes[..8], &(n as u64).to_le_bytes());
            assert!(bytes[8..].iter().all(|b| *b == (n % 251) as u8));
        }
    }
    assert_eq!(seen.len(), expected_rows);
    Ok(())
}

#[tokio::test]
async fn bound_body_reference_survives_publication_suffix_and_reopen() -> Result<()> {
    use lance_table::{format::pb, transaction::Transaction};

    let root = tempfile::tempdir()?;
    let body_uri = root.path().join("body.lance");
    let narrow_uri = root.path().join("annotations.lance");
    let initial = imported(0, 16)?;
    let indices = ["id", "rollout_id", "labels", "status", "score"]
        .map(|name| initial.schema().index_of(name).unwrap());
    let projected = initial.project(&indices)?;
    let canonical = Arc::new(
        Dataset::write(
            RecordBatchIterator::new(vec![Ok(initial.clone())], initial.schema()),
            body_uri.to_str().unwrap(),
            Some(write_params()),
        )
        .await?,
    );
    let mut narrow = Arc::new(
        Dataset::write(
            RecordBatchIterator::new(vec![Ok(projected.clone())], projected.schema()),
            narrow_uri.to_str().unwrap(),
            Some(write_params()),
        )
        .await?,
    );
    let mut body = canonical.clone();
    let mut cut = 0;
    let mut previous_suffix: Option<String> = None;
    let mut old_pairs = Vec::new();
    for round in 1..=2 {
        let staging = AnnotationStaging::new(narrow.clone(), scope(), cut, Limits::default())?;
        if let Some(turn) = previous_suffix.take() {
            staging
                .apply(
                    cut,
                    cut + 1,
                    &turn,
                    &TurnAnnotationPatch {
                        score: Set(0.9),
                        ..Default::default()
                    },
                )
                .await?;
            cut += 1;
        }
        let batch = imported(round * 16, 16)?;
        let ids = batch
            .column_by_name("id")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let groups = batch
            .column_by_name("rollout_id")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let annotation = TurnAnnotation {
            labels: BTreeSet::from(["imported".into()]),
            status: Some("pending".into()),
            score: None,
        };
        let registrations = (0..batch.num_rows())
            .map(|i| Registration {
                turn: ids.value(i),
                rollout_id: groups.value(i),
                annotation: &annotation,
            })
            .collect::<Vec<_>>();
        assert_eq!(
            staging
                .register_committed(cut, cut + 1, &registrations)
                .await?,
            ApplyOutcome::Applied
        );
        let turn = ids.value(0).to_owned();
        staging
            .apply(
                cut + 1,
                cut + 2,
                &turn,
                &TurnAnnotationPatch {
                    score: Set(0.4),
                    ..Default::default()
                },
            )
            .await?;
        cut += 2;
        let frozen = staging.freeze(uuid::Uuid::new_v4()).await?.unwrap();
        let old_read = staging.capture(&turn).await?;
        staging
            .apply(
                cut,
                cut + 1,
                &turn,
                &TurnAnnotationPatch {
                    score: Set(0.9),
                    ..Default::default()
                },
            )
            .await?;
        let next_body = detached(body.clone(), batch).await?;
        // A later abandoned branch has the same row count but different IDs.
        // Neither the latest URI nor listing order may select the body view.
        let abandoned = detached(body.clone(), imported(900, 16)?).await?;
        assert_ne!(next_body.version_id(), abandoned.version_id());
        let reference = BodySnapshot::capture(scope().table_instance, &next_body).await?;
        let budget = crate::MergeMemoryBudget::new(8 << 20);
        let prepared = frozen
            .prepare_with_body(&budget, 8192, reference.clone())
            .await?;
        let repeated = frozen
            .prepare_with_body(&budget, 8192, reference.clone())
            .await?;
        assert_eq!(prepared.merge().transaction, repeated.merge().transaction);
        assert!(frozen.prepare(&budget, 8192).await.is_err());
        let mut changed = reference.clone();
        changed.version ^= 1;
        assert!(frozen
            .prepare_with_body(&budget, 8192, changed)
            .await
            .is_err());
        drop(repeated);
        let transaction = prepared.merge().transaction.clone();
        let exact = Transaction::try_from(pb::Transaction::from(&transaction))?;
        assert_eq!(exact, transaction);
        let selected = Arc::new(
            CommitBuilder::new(narrow.clone())
                .with_max_retries(0)
                .with_skip_auto_cleanup(true)
                .execute(exact.clone())
                .await?,
        );
        assert_eq!(selected.read_transaction().await?.as_ref(), Some(&exact));
        staging.install_published(&prepared, selected).await?;
        assert_eq!(old_read.resolve().await?.unwrap().score, Some(0.4));
        assert_eq!(current(&staging, &turn).await?.score, Some(0.9));
        drop((old_read, prepared, frozen, staging, next_body, abandoned));
        narrow = Arc::new(Dataset::open(narrow_uri.to_str().unwrap()).await?);
        let (recovered_reference, recovered_body) = selected_body(&narrow).await?;
        assert_eq!(recovered_reference, reference);
        assert_eq!(narrow.count_rows(None).await?, (round + 1) * 16);
        verify_body(&recovered_body, (round + 1) * 16).await?;
        let recovered = AnnotationStaging::new(narrow.clone(), scope(), cut, Limits::default())?;
        assert_eq!(current(&recovered, &turn).await?.score, Some(0.4));
        recovered
            .apply(
                cut,
                cut + 1,
                &turn,
                &TurnAnnotationPatch {
                    score: Set(0.9),
                    ..Default::default()
                },
            )
            .await?;
        assert_eq!(current(&recovered, &turn).await?.score, Some(0.9));
        let latest_body = Dataset::open(body_uri.to_str().unwrap()).await?;
        assert_eq!(latest_body.version_id(), canonical.version_id());
        verify_body(&latest_body, 16).await?;
        old_pairs.push((narrow.clone(), (round + 1) * 16));
        body = recovered_body;
        previous_suffix = Some(turn);
    }
    assert_eq!(narrow.count_rows(None).await?, 48);
    for (old_narrow, rows) in old_pairs {
        let (_, old_body) = selected_body(&old_narrow).await?;
        verify_body(&old_body, rows).await?;
    }
    Ok(())
}

pub(super) async fn binding_fixture(
    root: &std::path::Path,
) -> Result<(AnnotationStaging, BodySnapshot)> {
    let batch = imported(0, 2)?;
    let body = Dataset::write(
        RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema()),
        root.join("body.lance").to_str().unwrap(),
        Some(write_params()),
    )
    .await?;
    let indices = ["id", "rollout_id", "labels", "status", "score"]
        .map(|name| batch.schema().index_of(name).unwrap());
    let narrow = batch.project(&indices)?;
    let narrow = Dataset::write(
        RecordBatchIterator::new(vec![Ok(narrow.clone())], narrow.schema()),
        root.join("narrow.lance").to_str().unwrap(),
        Some(WriteParams {
            max_rows_per_file: 1,
            ..write_params()
        }),
    )
    .await?;
    let reference = BodySnapshot::capture(scope().table_instance, &body).await?;
    let staging = AnnotationStaging::new(Arc::new(narrow), scope(), 0, Limits::default())?;
    staging
        .apply(
            0,
            1,
            "turn-000000",
            &TurnAnnotationPatch {
                score: Set(0.4),
                ..Default::default()
            },
        )
        .await?;
    Ok((staging, reference))
}

#[tokio::test]
async fn binding_rejects_foreign_identity_retained_overflow_and_later_loss() -> Result<()> {
    let root = tempfile::tempdir()?;
    let (staging, reference) = binding_fixture(root.path()).await?;
    let budget = crate::MergeMemoryBudget::new(8 << 20);
    let frozen = staging.freeze(uuid::Uuid::new_v4()).await?.unwrap();
    let mut invalid = reference.clone();
    invalid.table_instance = "foreign-table".into();
    assert!(frozen
        .prepare_with_body(&budget, 8192, invalid)
        .await
        .is_err());
    let mut invalid = reference.clone();
    invalid.version = 0;
    assert!(frozen
        .prepare_with_body(&budget, 8192, invalid)
        .await
        .is_err());
    let mut invalid = reference.clone();
    invalid.transaction = uuid::Uuid::nil().to_string();
    assert!(frozen
        .prepare_with_body(&budget, 8192, invalid)
        .await
        .is_err());
    let mut invalid = reference.clone();
    invalid.uri = String::with_capacity(budget.limit());
    invalid.uri.push_str(&reference.uri);
    assert!(frozen
        .prepare_with_body(&budget, 8192, invalid)
        .await
        .is_err());
    let prepared = frozen
        .prepare_with_body(&budget, 8192, reference.clone())
        .await?;
    let selected = commit(&prepared).await?;
    assert_eq!(
        BodySnapshot::read(&selected).await?,
        Some(reference.clone())
    );
    staging.install_published(&prepared, selected).await?;
    drop((prepared, frozen));
    staging
        .apply(
            1,
            2,
            "turn-000000",
            &TurnAnnotationPatch {
                score: Set(0.9),
                ..Default::default()
            },
        )
        .await?;
    let frozen = staging.freeze(uuid::Uuid::new_v4()).await?.unwrap();
    assert!(frozen.prepare(&budget, 8192).await.is_err());
    let mut foreign = reference.clone();
    foreign.uri.push_str("-other");
    assert!(frozen
        .prepare_with_body(&budget, 8192, foreign)
        .await
        .is_err());
    let mut foreign = reference.clone();
    foreign.transaction = uuid::Uuid::new_v4().to_string();
    assert!(frozen
        .prepare_with_body(&budget, 8192, foreign)
        .await
        .is_err());
    let prepared = frozen
        .prepare_with_body(&budget, 8192, reference.clone())
        .await?;
    let selected = commit(&prepared).await?;
    assert_eq!(BodySnapshot::read(&selected).await?, Some(reference));
    staging.install_published(&prepared, selected).await?;
    assert_eq!(current(&staging, "turn-000000").await?.score, Some(0.9));
    Ok(())
}

#[tokio::test]
async fn narrow_compaction_must_preserve_the_selected_body_reference() -> Result<()> {
    for mode in 0..3 {
        let root = tempfile::tempdir()?;
        let (staging, reference) = binding_fixture(root.path()).await?;
        let frozen = staging.freeze(uuid::Uuid::new_v4()).await?.unwrap();
        let prepared = frozen
            .prepare_with_body(
                &crate::MergeMemoryBudget::new(8 << 20),
                8192,
                reference.clone(),
            )
            .await?;
        let base = commit(&prepared).await?;
        staging.install_published(&prepared, base.clone()).await?;
        drop((prepared, frozen));
        let held = staging.capture("turn-000001").await?;
        staging
            .apply(
                1,
                2,
                "turn-000000",
                &TurnAnnotationPatch {
                    score: Set(0.9),
                    ..Default::default()
                },
            )
            .await?;
        let before = staging.observation().await;
        let mut transaction = super::compaction::compact(&base).await?;
        let properties = Arc::make_mut(transaction.transaction_properties.as_mut().unwrap());
        if mode == 1 {
            properties.remove(BODY_SNAPSHOT_PROPERTY);
        } else if mode == 2 {
            let mut other = reference.clone();
            other.version ^= 1;
            properties.insert(
                BODY_SNAPSHOT_PROPERTY.into(),
                serde_json::to_string(&other)?,
            );
        }
        let selected = super::compaction::select(base, &transaction).await?;
        let result = staging
            .install_compacted(&transaction, selected.clone())
            .await;
        if mode == 0 {
            result?;
            assert_eq!(BodySnapshot::read(&selected).await?, Some(reference));
        } else {
            assert!(result.is_err());
        }
        assert_eq!(staging.observation().await, before);
        assert_eq!(current(&staging, "turn-000000").await?.score, Some(0.9));
        assert_eq!(held.resolve().await?.unwrap().score, None);
    }
    Ok(())
}

#[tokio::test]
async fn trusted_bootstrap_pair_at_zero_requires_explicit_later_binding() -> Result<()> {
    use lance_table::transaction::Transaction;
    let root = tempfile::tempdir()?;
    let (staging, reference) = binding_fixture(root.path()).await?;
    let base = staging.state.read().await.base.dataset.clone();
    let mut transaction = Transaction::new_from_version(
        base.version_id(),
        Operation::UpdateConfig {
            config_updates: None,
            table_metadata_updates: None,
            schema_metadata_updates: None,
            field_metadata_updates: Default::default(),
        },
    );
    transaction.transaction_properties = Some(Arc::new(std::collections::HashMap::from([
        (
            "lance-context.annotation-cut".into(),
            serde_json::to_string(&(scope(), 0u64))?,
        ),
        (
            BODY_SNAPSHOT_PROPERTY.into(),
            serde_json::to_string(&reference)?,
        ),
    ])));
    let selected = CommitBuilder::new(base)
        .with_max_retries(0)
        .with_skip_auto_cleanup(true)
        .execute(transaction)
        .await?;
    assert_eq!(
        BodySnapshot::read(&selected).await?,
        Some(reference.clone())
    );
    let fresh = AnnotationStaging::new(Arc::new(selected), scope(), 0, Limits::default())?;
    fresh
        .apply(
            0,
            1,
            "turn-000000",
            &TurnAnnotationPatch {
                score: Set(0.9),
                ..Default::default()
            },
        )
        .await?;
    let frozen = fresh.freeze(uuid::Uuid::new_v4()).await?.unwrap();
    let budget = crate::MergeMemoryBudget::new(8 << 20);
    assert!(frozen.prepare(&budget, 8192).await.is_err());
    let prepared = frozen
        .prepare_with_body(&budget, 8192, reference.clone())
        .await?;
    let selected = commit(&prepared).await?;
    assert_eq!(BodySnapshot::read(&selected).await?, Some(reference));
    fresh.install_published(&prepared, selected).await?;
    assert_eq!(current(&fresh, "turn-000000").await?.score, Some(0.9));
    Ok(())
}
