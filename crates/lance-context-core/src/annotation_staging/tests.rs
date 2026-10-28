use super::*;
use arrow_array::{
    builder::{ListBuilder, StringBuilder},
    LargeBinaryArray, RecordBatch, RecordBatchIterator,
};
use arrow_schema::Schema;
use lance::dataset::{CommitBuilder, WriteParams};
use lance::index::DatasetIndexExt;
use lance_context_api::annotation::AnnotationFieldUpdate::{Clear, Set};
use lance_index::{scalar::ScalarIndexParams, IndexType};
use std::collections::BTreeSet;

mod compaction;
mod concurrent_reads;
mod published_values;
mod reclamation;
mod registration;
mod views;

#[tokio::test]
async fn cold_batches_keep_frozen_values_and_order_across_lookup_windows() -> Result<()> {
    let root = tempfile::tempdir()?;
    let base = fixture(&root.path().join("turns.lance")).await?;
    let staging = AnnotationStaging::new(base, scope(), 0, Limits::default())?;
    let status = TurnAnnotationPatch {
        status: Set("reviewed".into()),
        ..Default::default()
    };
    staging.apply(0, 1, "T", &status).await?;
    let frozen = staging.freeze(uuid::Uuid::new_v4()).await?.unwrap();
    let score = TurnAnnotationPatch {
        score: Set(0.9),
        ..Default::default()
    };
    let clear = TurnAnnotationPatch {
        status: Clear,
        ..Default::default()
    };
    let mut batch = (0..COLD_LOOKUP_KEYS)
        .map(|_| Mutation {
            turn: "T",
            patch: &score,
        })
        .collect::<Vec<_>>();
    batch[1] = Mutation {
        turn: "U",
        patch: &status,
    };
    batch.push(Mutation {
        turn: "U",
        patch: &score,
    });
    batch.push(Mutation {
        turn: "o'hara",
        patch: &clear,
    });
    staging.apply_batch(1, 2, &batch).await?;
    for key in ["T", "U"] {
        let value = current(&staging, key).await?;
        assert_eq!(value.status.as_deref(), Some("reviewed"));
        assert_eq!(value.score, Some(0.9));
        assert_eq!(value.labels, BTreeSet::from(["original".into()]));
    }
    assert_eq!(current(&staging, "o'hara").await?.status, None);
    let before = staging.observation().await;
    let expected = current(&staging, "T").await?;
    batch[0] = Mutation {
        turn: "T",
        patch: &clear,
    };
    batch.push(Mutation {
        turn: "missing",
        patch: &score,
    });
    assert_eq!(
        staging.apply_committed(2, 3, &batch).await?,
        ApplyOutcome::Rejected(Rejection {
            row: COLD_LOOKUP_KEYS + 2,
            reason: RejectionReason::MissingTurn,
        })
    );
    assert_eq!(current(&staging, "T").await?, expected);
    assert_eq!(
        staging.observation().await,
        Observation {
            applied: 3,
            ..before
        }
    );
    let prepared = frozen
        .prepare(&crate::MergeMemoryBudget::new(8 << 20), 8192)
        .await?;
    staging
        .install_published(&prepared, commit(&prepared).await?)
        .await?;
    assert_eq!(current(&staging, "T").await?, expected);
    assert_eq!(current(&staging, "U").await?.score, Some(0.9));
    Ok(())
}

#[tokio::test]
async fn cold_lookup_projects_narrow_rows_and_rejects_duplicate_keys() -> Result<()> {
    let root = tempfile::tempdir()?;
    let base = fixture(&root.path().join("turns.lance")).await?;
    let keys = BTreeSet::from(["o'hara", "U", "T", "missing"]);
    let rows = read_bases(&base, &keys).await?;
    assert_eq!(rows.len(), 3);
    for key in ["T", "U", "o'hara"] {
        let row = &rows[key];
        assert_eq!(row.num_rows(), 1);
        assert_eq!(row.num_columns(), 4);
        assert!(row.schema().field_with_name("body").is_err());
        assert_eq!(
            Some(decode_base_row(row, 0, key, 4096)?),
            read_base(&base, key, 4096).await?
        );
    }
    // A corrupt base must fail without installing any part of the WAL batch.
    let batches = base
        .scan()
        .try_into_stream()
        .await?
        .try_collect::<Vec<_>>()
        .await?;
    let duplicate = Arc::new(
        Dataset::write(
            RecordBatchIterator::new(
                batches.into_iter().map(Ok),
                Arc::new(Schema::from(base.schema())),
            ),
            base.uri(),
            Some(WriteParams {
                mode: lance::dataset::WriteMode::Append,
                ..Default::default()
            }),
        )
        .await?,
    );
    let staging = AnnotationStaging::new(duplicate, scope(), 0, Limits::default())?;
    assert!(staging.capture("T").await?.resolve().await.is_err());
    let before = staging.observation().await;
    let patch = TurnAnnotationPatch::default();
    assert!(staging
        .apply_batch(
            0,
            1,
            &[
                Mutation {
                    turn: "T",
                    patch: &patch
                },
                Mutation {
                    turn: "U",
                    patch: &patch
                },
            ]
        )
        .await
        .is_err());
    assert_eq!(staging.observation().await, before);
    Ok(())
}

#[tokio::test]
async fn result_only_generation_publishes_a_new_version_without_data_files() -> Result<()> {
    let root = tempfile::tempdir()?;
    let base = fixture(&root.path().join("turns.lance")).await?;
    let before = base.manifest().fragments.clone();
    let staging = AnnotationStaging::new(base, scope(), 0, Limits::default())?;
    let patch = TurnAnnotationPatch::default();
    assert!(matches!(
        staging
            .apply_committed(
                0,
                1,
                &[Mutation {
                    turn: "absent",
                    patch: &patch
                }]
            )
            .await?,
        ApplyOutcome::Rejected(_)
    ));
    assert_eq!(staging.observation().await.active_entries, 0);
    let frozen = staging
        .freeze_with_results(uuid::Uuid::new_v4(), Some("opaque-results".into()))
        .await?
        .unwrap();
    assert_eq!(frozen.rows(), 0);
    let prepared = frozen
        .prepare(&crate::MergeMemoryBudget::new(8 << 20), 8192)
        .await?;
    assert_eq!(prepared.merge().stats.bytes_written, 0);
    let selected = commit(&prepared).await?;
    assert_eq!(selected.version_id(), 2);
    assert_eq!(selected.manifest().fragments, before);
    assert_eq!(
        selected
            .read_transaction()
            .await?
            .unwrap()
            .transaction_properties
            .unwrap()[RESULTS_PROPERTY],
        "opaque-results"
    );
    staging.install_published(&prepared, selected).await?;
    assert_eq!(staging.observation().await.materialized, 1);
    assert!(staging.observation().await.retained_metadata_bytes > 0);
    assert_eq!(
        staging
            .protected_bases()
            .await?
            .datasets()
            .map(Dataset::version_id)
            .collect::<Vec<_>>(),
        vec![1, 2]
    );
    drop(prepared);
    drop(frozen);
    assert_eq!(staging.observation().await.retained_metadata_bytes, 0);
    assert_eq!(
        staging
            .protected_bases()
            .await?
            .datasets()
            .map(Dataset::version_id)
            .collect::<Vec<_>>(),
        vec![2]
    );
    Ok(())
}

#[tokio::test]
async fn committed_rejection_advances_without_losing_later_updates() -> Result<()> {
    let root = tempfile::tempdir()?;
    let base = fixture(&root.path().join("turns.lance")).await?;
    let staging = AnnotationStaging::new(base.clone(), scope(), 0, Limits::default())?;
    let status = TurnAnnotationPatch {
        status: Set("reviewed".into()),
        ..Default::default()
    };
    let batch = [
        Mutation {
            turn: "T",
            patch: &status,
        },
        Mutation {
            turn: "missing",
            patch: &status,
        },
    ];
    let result = staging.apply_committed(0, 2, &batch).await?;
    assert_eq!(
        result,
        ApplyOutcome::Rejected(Rejection {
            row: 1,
            reason: RejectionReason::MissingTurn,
        })
    );
    assert_eq!(
        current(&staging, "T").await?.status.as_deref(),
        Some("pending")
    );
    assert_eq!(staging.observation().await.applied, 2);
    assert_eq!(staging.observation().await.retained_bytes, 0);
    assert!(staging.apply_committed(0, 2, &batch).await.is_err());
    staging.advance_processed(2, 4).await?;
    assert_eq!(
        staging.apply_committed(4, 5, &batch[..1]).await?,
        ApplyOutcome::Applied
    );
    assert_eq!(
        current(&staging, "T").await?.status.as_deref(),
        Some("reviewed")
    );
    // A reconstructed applier produces the same rejection from base + WAL.
    let replay = AnnotationStaging::new(base, scope(), 0, Limits::default())?;
    assert_eq!(replay.apply_committed(0, 2, &batch).await?, result);
    Ok(())
}

#[tokio::test]
async fn pressure_keeps_the_committed_record_pending_until_publication_releases_space() -> Result<()>
{
    let root = tempfile::tempdir()?;
    let base = fixture(&root.path().join("turns.lance")).await?;
    let staging = AnnotationStaging::new(
        base,
        scope(),
        0,
        Limits {
            retained_entries: 1,
            ..Limits::default()
        },
    )?;
    let patch = TurnAnnotationPatch {
        score: Set(0.9),
        ..Default::default()
    };
    let first = [Mutation {
        turn: "T",
        patch: &patch,
    }];
    let next = [Mutation {
        turn: "U",
        patch: &patch,
    }];
    assert_eq!(
        staging.apply_committed(0, 1, &first).await?,
        ApplyOutcome::Applied
    );
    let frozen = staging.freeze(uuid::Uuid::new_v4()).await?.unwrap();
    let before = staging.observation().await;
    assert_eq!(
        staging.apply_committed(1, 2, &next).await?,
        ApplyOutcome::Backpressure
    );
    assert_eq!(staging.observation().await, before);
    assert_eq!(current(&staging, "U").await?.score, None);
    let prepared = frozen
        .prepare(&crate::MergeMemoryBudget::new(8 << 20), 8192)
        .await?;
    staging
        .install_published(&prepared, commit(&prepared).await?)
        .await?;
    // External holders still charge retired generations.
    assert_eq!(
        staging.apply_committed(1, 2, &next).await?,
        ApplyOutcome::Backpressure
    );
    drop(prepared);
    drop(frozen);
    assert_eq!(
        staging.apply_committed(1, 2, &next).await?,
        ApplyOutcome::Applied
    );
    assert_eq!(current(&staging, "T").await?.score, Some(0.9));
    assert_eq!(current(&staging, "U").await?.score, Some(0.9));
    Ok(())
}

#[tokio::test]
async fn impossible_atomic_batches_are_rejected_instead_of_waiting_forever() -> Result<()> {
    let root = tempfile::tempdir()?;
    let base = fixture(&root.path().join("turns.lance")).await?;
    let staging = AnnotationStaging::new(
        base,
        scope(),
        0,
        Limits {
            retained_entries: 1,
            ..Limits::default()
        },
    )?;
    let patch = TurnAnnotationPatch::default();
    let batch = [
        Mutation {
            turn: "T",
            patch: &patch,
        },
        Mutation {
            turn: "U",
            patch: &patch,
        },
    ];
    assert_eq!(
        staging.apply_committed(0, 1, &batch).await?,
        ApplyOutcome::Rejected(Rejection {
            row: 1,
            reason: RejectionReason::BatchTooLarge,
        })
    );
    assert_eq!(staging.observation().await.applied, 1);
    assert_eq!(staging.observation().await.retained_entries, 0);
    let invalid = TurnAnnotationPatch {
        score: Set(f32::NAN),
        ..Default::default()
    };
    assert_eq!(
        staging
            .apply_committed(
                1,
                2,
                &[Mutation {
                    turn: "T",
                    patch: &invalid
                }]
            )
            .await?,
        ApplyOutcome::Rejected(Rejection {
            row: 0,
            reason: RejectionReason::InvalidPatch
        })
    );
    assert_eq!(current(&staging, "T").await?.score, None);
    Ok(())
}

#[tokio::test]
async fn one_wal_batch_merges_repeated_keys_and_never_exposes_a_partial_failure() -> Result<()> {
    let root = tempfile::tempdir()?;
    let base = fixture(&root.path().join("turns.lance")).await?;
    let staging = AnnotationStaging::new(base, scope(), 0, Limits::default())?;
    let status = TurnAnnotationPatch {
        status: Set("reviewed".into()),
        ..Default::default()
    };
    let score = TurnAnnotationPatch {
        score: Set(0.9),
        ..Default::default()
    };
    staging
        .apply_batch(
            0,
            1,
            &[
                Mutation {
                    turn: "T",
                    patch: &status,
                },
                Mutation {
                    turn: "T",
                    patch: &score,
                },
            ],
        )
        .await?;
    let before = current(&staging, "T").await?;
    assert_eq!(before.status.as_deref(), Some("reviewed"));
    assert_eq!(before.score, Some(0.9));
    assert_eq!(staging.observation().await.active_entries, 1);
    let clear = TurnAnnotationPatch {
        score: Clear,
        ..Default::default()
    };
    let observed = staging.observation().await;
    assert!(staging
        .apply_batch(
            1,
            2,
            &[
                Mutation {
                    turn: "T",
                    patch: &clear
                },
                Mutation {
                    turn: "missing",
                    patch: &score
                },
            ]
        )
        .await
        .is_err());
    assert_eq!(staging.observation().await, observed);
    assert_eq!(current(&staging, "T").await?, before);
    staging
        .apply_batch(
            1,
            2,
            &[
                Mutation {
                    turn: "T",
                    patch: &clear,
                },
                Mutation {
                    turn: "U",
                    patch: &status,
                },
            ],
        )
        .await?;
    let frozen = staging.freeze(uuid::Uuid::new_v4()).await?.unwrap();
    assert_eq!((frozen.through(), frozen.rows()), (2, 2));
    let prepared = frozen
        .prepare(&crate::MergeMemoryBudget::new(8 << 20), 8192)
        .await?;
    staging
        .install_published(&prepared, commit(&prepared).await?)
        .await?;
    assert_eq!(current(&staging, "T").await?.score, None);
    assert_eq!(
        current(&staging, "U").await?.status.as_deref(),
        Some("reviewed")
    );
    Ok(())
}

fn scope() -> Scope {
    Scope {
        table_instance: "test-incarnation".into(),
        source_group: 10,
        shard: 0,
    }
}

async fn fixture(root: &std::path::Path) -> Result<Arc<Dataset>> {
    fixture_with_row_ids(root, true).await
}

async fn fixture_with_row_ids(root: &std::path::Path, stable: bool) -> Result<Arc<Dataset>> {
    let mut labels = ListBuilder::new(StringBuilder::new());
    for _ in 0..3 {
        labels.values().append_value("original");
        labels.append(true);
    }
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("body", DataType::LargeBinary, false),
        Field::new("labels", labels.finish().data_type().clone(), false),
        Field::new("status", DataType::Utf8, true),
        Field::new("score", DataType::Float32, true),
    ]));
    for _ in 0..3 {
        labels.values().append_value("original");
        labels.append(true);
    }
    let body: Vec<u8> = (0..1 << 20)
        .map(|n| ((n * 31 + n / 17) % 251) as u8)
        .collect();
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(StringArray::from(vec!["T", "U", "o'hara"])),
            Arc::new(LargeBinaryArray::from_iter_values(
                (0..3).map(|_| body.as_slice()),
            )),
            Arc::new(labels.finish()),
            Arc::new(StringArray::from(vec!["pending"; 3])),
            Arc::new(Float32Array::from(vec![None; 3])),
        ],
    )?;
    Ok(Arc::new(
        Dataset::write(
            RecordBatchIterator::new(vec![Ok(batch)], schema),
            root.to_str().unwrap(),
            Some(WriteParams {
                max_rows_per_file: 2,
                max_rows_per_group: 2,
                enable_stable_row_ids: stable,
                ..Default::default()
            }),
        )
        .await?,
    ))
}

fn body_paths(base: &Dataset) -> Result<BTreeSet<String>> {
    let field = base.schema().field_id("body")?;
    Ok(base
        .manifest()
        .fragments
        .iter()
        .flat_map(|f| &f.files)
        .filter(|f| f.fields.contains(&field))
        .map(|f| f.path.clone())
        .collect())
}

async fn commit(prepared: &PreparedAnnotations) -> Result<Arc<Dataset>> {
    let mut commit = CommitBuilder::new(prepared.base().clone())
        .with_max_retries(0)
        .with_skip_auto_cleanup(true);
    if let Some(rows) = prepared.merge().affected_rows.clone() {
        commit = commit.with_affected_rows(rows);
    }
    Ok(Arc::new(
        commit.execute(prepared.merge().transaction.clone()).await?,
    ))
}

async fn current(staging: &AnnotationStaging, turn: &str) -> Result<TurnAnnotation> {
    staging
        .capture(turn)
        .await?
        .resolve()
        .await?
        .ok_or_else(|| invalid("test turn missing"))
}

#[tokio::test]
async fn annotation_publication_preserves_id_index_and_invalidates_status_index() -> Result<()> {
    check_index_publication(false, true).await
}

#[tokio::test]
async fn annotation_publication_preserves_index_with_deleted_rows_and_both_row_id_modes(
) -> Result<()> {
    for stable in [false, true] {
        check_index_publication(true, stable).await?;
    }
    Ok(())
}

async fn check_index_publication(deleted: bool, stable: bool) -> Result<()> {
    let root = tempfile::tempdir()?;
    let mut base = fixture_with_row_ids(&root.path().join("turns.lance"), stable).await?;
    for field in ["id", "status"] {
        Arc::get_mut(&mut base)
            .unwrap()
            .create_index_builder(&[field], IndexType::BTree, &ScalarIndexParams::default())
            .name(format!("annotation_{field}"))
            .await?;
    }
    if deleted {
        Arc::get_mut(&mut base).unwrap().delete("id = 'U'").await?;
    }
    let original_indices = base.load_indices().await?;
    let original_id = original_indices
        .iter()
        .find(|i| i.name == "annotation_id")
        .unwrap();
    let id_field = base.schema().field_id("id")?;
    let original_body = body_paths(&base)?;
    let staging = AnnotationStaging::new(base.clone(), scope(), 0, Limits::default())?;
    let workspace = crate::MergeMemoryBudget::new(8 << 20);
    for (position, turn) in [(1, "T"), (2, "o'hara"), (3, "T")] {
        staging
            .apply(
                position - 1,
                position,
                turn,
                &TurnAnnotationPatch {
                    status: Set(format!("reviewed-{position}")),
                    score: Set(position as f32 / 10.0),
                    ..Default::default()
                },
            )
            .await?;
        let frozen = staging.freeze(uuid::Uuid::new_v4()).await?.unwrap();
        let prepared = frozen.prepare(&workspace, 2).await?;
        let selected = commit(&prepared).await?;
        let indices = selected.load_indices().await?;
        let id = indices.iter().find(|i| i.name == "annotation_id").unwrap();
        assert_eq!(
            id.uuid, original_id.uuid,
            "the ID index must not be rebuilt"
        );
        assert_eq!(
            id.fragment_bitmap, original_id.fragment_bitmap,
            "annotation publication withdrew unchanged ID index coverage"
        );
        let status = indices
            .iter()
            .find(|i| i.name == "annotation_status")
            .unwrap();
        assert_eq!(
            status.fragment_bitmap.as_ref().unwrap().len(),
            if position == 1 { 1 } else { 0 }
        );
        for (original, updated) in base
            .manifest()
            .fragments
            .iter()
            .zip(selected.manifest().fragments.iter())
        {
            assert_eq!(original.id, updated.id);
            assert_eq!(original.row_id_meta, updated.row_id_meta);
            let original_file = original
                .files
                .iter()
                .find(|f| f.fields.contains(&id_field))
                .unwrap();
            let updated_file = updated
                .files
                .iter()
                .find(|f| f.fields.contains(&id_field))
                .unwrap();
            assert_eq!(original_file.path, updated_file.path);
            assert_eq!(original_file.base_id, updated_file.base_id);
            assert_eq!(
                original_file.fields.iter().position(|f| *f == id_field),
                updated_file.fields.iter().position(|f| *f == id_field)
            );
            assert_eq!(original_file.column_indices, updated_file.column_indices);
        }
        assert_eq!(body_paths(&selected)?, original_body);
        assert_eq!(
            selected.count_rows(None).await?,
            if deleted { 2 } else { 3 }
        );
        // The changed annotation index must not return stale matches.
        assert_eq!(
            selected
                .count_rows(Some(format!("status = 'reviewed-{position}'")))
                .await?,
            1
        );
        staging.install_published(&prepared, selected).await?;
        assert_eq!(
            current(&staging, turn).await?.score,
            Some(position as f32 / 10.0)
        );
        if deleted {
            assert!(staging.capture("U").await?.resolve().await?.is_none());
        } else {
            assert_eq!(
                current(&staging, "U").await?.status.as_deref(),
                Some("pending")
            );
        }
    }
    let reopened = Arc::new(Dataset::open(base.uri()).await?);
    let mut scan = reopened.scan();
    scan.project(&["labels", "status", "score"])?;
    scan.filter("id = 'T'")?;
    let plan = scan.explain_plan(false).await?;
    assert!(
        plan.contains("annotation_id"),
        "point read lost ID index: {plan}"
    );
    let recovered = AnnotationStaging::new(reopened, scope(), 3, Limits::default())?;
    assert_eq!(
        current(&recovered, "T").await?.status.as_deref(),
        Some("reviewed-3")
    );
    assert_eq!(current(&recovered, "o'hara").await?.score, Some(0.2));
    Ok(())
}

#[tokio::test]
async fn later_field_update_survives_real_lance_publication_and_old_reader_handoff() -> Result<()> {
    let root = tempfile::tempdir()?;
    let base = fixture(&root.path().join("turns.lance")).await?;
    let original_paths = body_paths(&base)?;
    let staging = AnnotationStaging::new(base.clone(), scope(), 0, Limits::default())?;
    let old_base_read = staging.capture("T").await?;
    staging
        .apply(
            0,
            1,
            "T",
            &TurnAnnotationPatch {
                status: Set("reviewed".into()),
                ..Default::default()
            },
        )
        .await?;
    let frozen = staging.freeze(uuid::Uuid::new_v4()).await?.unwrap();
    let frozen_read = staging.capture("T").await?;
    let workspace = crate::MergeMemoryBudget::new(8 << 20);
    let prepared = frozen.prepare(&workspace, 8192).await?;
    let again = frozen.prepare(&workspace, 8192).await?;
    assert_eq!(prepared.merge().transaction, again.merge().transaction);
    assert!(
        staging.freeze(uuid::Uuid::new_v4()).await.is_err(),
        "a second frozen batch was admitted"
    );
    assert!(staging
        .install_published(&prepared, base.clone())
        .await
        .is_err());
    assert_eq!(
        Dataset::open(base.uri()).await?.version_id(),
        base.version_id()
    );
    staging
        .apply(
            1,
            2,
            "T",
            &TurnAnnotationPatch {
                score: Set(0.9),
                ..Default::default()
            },
        )
        .await?;
    let selected = commit(&prepared).await?;
    // Publication alone cannot retire or hide the active value.
    assert_eq!(current(&staging, "T").await?.score, Some(0.9));
    staging
        .install_published(&prepared, selected.clone())
        .await?;
    let actual = current(&staging, "T").await?;
    assert_eq!(actual.status.as_deref(), Some("reviewed"));
    assert_eq!(actual.score, Some(0.9));
    assert_eq!(actual.labels, BTreeSet::from(["original".into()]));
    assert_eq!(body_paths(&selected)?, original_paths);
    // Reads which started earlier keep a compatible pair, including base misses.
    assert_eq!(
        old_base_read.resolve().await?.unwrap().status.as_deref(),
        Some("pending")
    );
    assert_eq!(frozen_read.resolve().await?.unwrap().score, None);
    let observed = staging.observation().await;
    assert_eq!(
        (
            observed.applied,
            observed.materialized,
            observed.active_entries,
            observed.frozen_entries
        ),
        (2, 1, 1, 0)
    );
    drop((again, prepared, frozen, frozen_read, old_base_read));
    let next = staging.freeze(uuid::Uuid::new_v4()).await?.unwrap();
    let prepared = next.prepare(&workspace, 8192).await?;
    let selected = commit(&prepared).await?;
    staging
        .install_published(&prepared, selected.clone())
        .await?;
    let reopened = Arc::new(Dataset::open(base.uri()).await?);
    let reconstructed = AnnotationStaging::new(reopened, scope(), 2, Limits::default())?;
    assert_eq!(current(&reconstructed, "T").await?, actual);
    assert_eq!(body_paths(&selected)?, original_paths);
    assert_eq!(selected.count_rows(None).await?, 3);
    // Full content remains readable; annotation-only methods above project no body.
    let mut scan = selected.scan();
    scan.project(&["body"])?;
    let batches = scan
        .try_into_stream()
        .await?
        .try_collect::<Vec<_>>()
        .await?;
    for batch in batches {
        let bodies = batch
            .column(0)
            .as_any()
            .downcast_ref::<LargeBinaryArray>()
            .unwrap();
        for bytes in bodies.iter().flatten() {
            assert_eq!(bytes.len(), 1 << 20);
            assert!(bytes
                .iter()
                .enumerate()
                .all(|(n, &byte)| byte == ((n * 31 + n / 17) % 251) as u8));
        }
    }
    Ok(())
}

#[tokio::test]
async fn equal_values_do_not_retire_newer_application_progress_and_clear_is_preserved() -> Result<()>
{
    let root = tempfile::tempdir()?;
    let base = fixture(&root.path().join("turns.lance")).await?;
    let staging = AnnotationStaging::new(base, scope(), 0, Limits::default())?;
    let patch = TurnAnnotationPatch {
        status: Set("reviewed".into()),
        score: Set(0.9),
        ..Default::default()
    };
    staging.apply(0, 1, "T", &patch).await?;
    let frozen = staging.freeze(uuid::Uuid::new_v4()).await?.unwrap();
    staging
        .apply(
            1,
            2,
            "T",
            &TurnAnnotationPatch {
                status: Set("pending".into()),
                ..Default::default()
            },
        )
        .await?;
    staging.apply(2, 3, "T", &patch).await?;
    let workspace = crate::MergeMemoryBudget::new(8 << 20);
    let prepared = frozen.prepare(&workspace, 8192).await?;
    staging
        .install_published(&prepared, commit(&prepared).await?)
        .await?;
    assert_eq!(staging.observation().await.active_entries, 1);
    assert_eq!(staging.observation().await.applied, 3);
    assert_eq!(
        current(&staging, "T").await?.status.as_deref(),
        Some("reviewed")
    );
    staging
        .apply(
            3,
            4,
            "T",
            &TurnAnnotationPatch {
                status: Clear,
                score: Clear,
                remove_labels: BTreeSet::from(["original".into()]),
                ..Default::default()
            },
        )
        .await?;
    drop((prepared, frozen));
    let frozen = staging.freeze(uuid::Uuid::new_v4()).await?.unwrap();
    assert_eq!(frozen.through(), 4);
    let prepared = frozen.prepare(&workspace, 8192).await?;
    let selected = commit(&prepared).await?;
    staging
        .install_published(&prepared, selected.clone())
        .await?;
    assert_eq!(current(&staging, "T").await?, TurnAnnotation::default());
    let properties = selected
        .read_transaction()
        .await?
        .unwrap()
        .transaction_properties
        .unwrap();
    let (persisted_scope, persisted_cut): (Scope, u64) =
        serde_json::from_str(&properties["lance-context.annotation-cut"]).unwrap();
    assert_eq!(persisted_scope, scope());
    assert_eq!(persisted_cut, 4);
    Ok(())
}

#[tokio::test]
async fn retained_generation_and_reader_admission_bound_memory_until_released() -> Result<()> {
    let root = tempfile::tempdir()?;
    let base = fixture(&root.path().join("turns.lance")).await?;
    let limits = Limits {
        retained_entries: 2,
        retained_bytes: 8192,
        value_bytes: 2048,
        readers: 1,
    };
    let staging = AnnotationStaging::new(base, scope(), 0, limits)?;
    let patch = TurnAnnotationPatch {
        score: Set(0.9),
        ..Default::default()
    };
    staging.apply(0, 1, "T", &patch).await?;
    let frozen = staging.freeze(uuid::Uuid::new_v4()).await?.unwrap();
    let held = staging.capture("T").await?;
    assert!(staging.capture("U").await.is_err());
    staging.apply(1, 2, "U", &patch).await?;
    let before = staging.observation().await;
    assert!(staging.apply(2, 3, "o'hara", &patch).await.is_err());
    assert_eq!(staging.observation().await, before);
    let workspace = crate::MergeMemoryBudget::new(8 << 20);
    let prepared = frozen.prepare(&workspace, 8192).await?;
    staging
        .install_published(&prepared, commit(&prepared).await?)
        .await?;
    assert!(
        staging.apply(2, 3, "o'hara", &patch).await.is_err(),
        "retired-but-held map escaped accounting"
    );
    drop((prepared, frozen));
    assert!(
        staging.apply(2, 3, "o'hara", &patch).await.is_err(),
        "held point value escaped accounting"
    );
    drop(held);
    staging.apply(2, 3, "o'hara", &patch).await?;
    assert_eq!(current(&staging, "o'hara").await?.score, Some(0.9));
    let frozen = staging.freeze(uuid::Uuid::new_v4()).await?.unwrap();
    let prepared = frozen.prepare(&workspace, 8192).await?;
    staging
        .install_published(&prepared, commit(&prepared).await?)
        .await?;
    drop((prepared, frozen));
    let done = staging.observation().await;
    assert_eq!(
        (
            done.applied,
            done.materialized,
            done.retained_entries,
            done.retained_bytes
        ),
        (3, 3, 0, 0)
    );
    assert_eq!(workspace.reserved(), 0);
    Ok(())
}

#[tokio::test]
async fn invalid_or_out_of_order_apply_leaves_progress_and_values_unchanged() -> Result<()> {
    let root = tempfile::tempdir()?;
    let base = fixture(&root.path().join("turns.lance")).await?;
    let staging = AnnotationStaging::new(base, scope(), 0, Limits::default())?;
    let patch = TurnAnnotationPatch {
        score: Set(0.9),
        ..Default::default()
    };
    assert!(staging.apply(1, 2, "T", &patch).await.is_err());
    assert!(staging.apply(0, 1, "missing", &patch).await.is_err());
    assert_eq!(staging.observation().await.applied, 0);
    staging.apply(0, 1, "T", &patch).await?;
    assert!(staging.apply(0, 1, "T", &patch).await.is_err());
    assert!(staging
        .apply(
            1,
            2,
            "T",
            &TurnAnnotationPatch {
                score: Set(f32::NAN),
                ..Default::default()
            }
        )
        .await
        .is_err());
    assert_eq!(staging.observation().await.applied, 1);
    assert_eq!(current(&staging, "T").await?.score, Some(0.9));
    assert!(staging.capture("missing").await?.resolve().await?.is_none());
    Ok(())
}

#[tokio::test]
async fn held_hot_key_value_exerts_byte_backpressure_on_later_replacements() -> Result<()> {
    let root = tempfile::tempdir()?;
    let base = fixture(&root.path().join("turns.lance")).await?;
    let staging = AnnotationStaging::new(
        base,
        scope(),
        0,
        Limits {
            retained_bytes: 4096,
            retained_entries: 8,
            value_bytes: 2048,
            readers: 4,
        },
    )?;
    let first = TurnAnnotationPatch {
        status: Set("a".repeat(700)),
        ..Default::default()
    };
    let second = TurnAnnotationPatch {
        status: Set("b".repeat(700)),
        ..Default::default()
    };
    staging.apply(0, 1, "T", &first).await?;
    let held = staging.capture("T").await?;
    staging.apply(1, 2, "T", &second).await?;
    let before = staging.observation().await;
    assert!(staging.apply(2, 3, "T", &first).await.is_err());
    assert_eq!(staging.observation().await, before);
    assert_eq!(held.resolve().await?.unwrap().status, Some("a".repeat(700)));
    assert_eq!(current(&staging, "T").await?.status, Some("b".repeat(700)));
    drop(held);
    staging.apply(2, 3, "T", &first).await?;
    let after = staging.observation().await;
    assert_eq!(
        (after.applied, after.active_entries, after.retained_entries),
        (3, 1, 1)
    );
    assert!(after.retained_bytes <= 2048);
    Ok(())
}

#[tokio::test]
async fn repeated_narrow_publication_drains_staging_without_growing_retained_history() -> Result<()>
{
    let root = tempfile::tempdir()?;
    let base = fixture(&root.path().join("turns.lance")).await?;
    let staging = AnnotationStaging::new(
        base,
        scope(),
        0,
        Limits {
            retained_entries: 8,
            ..Default::default()
        },
    )?;
    let workspace = crate::MergeMemoryBudget::new(8 << 20);
    let mut index = 0;
    for round in 0..16 {
        for turn in ["T", "U", "o'hara"] {
            staging
                .apply(
                    index,
                    index + 1,
                    turn,
                    &TurnAnnotationPatch {
                        score: Set(round as f32),
                        ..Default::default()
                    },
                )
                .await?;
            index += 1;
        }
        let frozen = staging.freeze(uuid::Uuid::new_v4()).await?.unwrap();
        let prepared = frozen.prepare(&workspace, 8192).await?;
        staging
            .install_published(&prepared, commit(&prepared).await?)
            .await?;
        drop((prepared, frozen));
        let observed = staging.observation().await;
        assert_eq!(
            (
                observed.materialized,
                observed.retained_entries,
                observed.retained_bytes
            ),
            (index, 0, 0)
        );
        assert_eq!(workspace.reserved(), 0);
        for turn in ["T", "U", "o'hara"] {
            assert_eq!(current(&staging, turn).await?.score, Some(round as f32));
        }
    }
    Ok(())
}

mod body;
mod body_origin;
