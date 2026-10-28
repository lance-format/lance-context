//! Registration and point updates share one bounded staging publication.
use super::*;
use arrow_array::UInt64Array;
use lance::dataset::transaction::Operation;
use lance_core::{ROW_CREATED_AT_VERSION, ROW_ID, ROW_LAST_UPDATED_AT_VERSION};

fn imported(start: usize, count: usize) -> Result<RecordBatch> {
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
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("rollout_id", DataType::Utf8, false),
        Field::new("labels", labels.data_type().clone(), false),
        Field::new("status", DataType::Utf8, true),
        Field::new("score", DataType::Float32, true),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from(ids)),
            Arc::new(StringArray::from(groups)),
            Arc::new(labels),
            Arc::new(StringArray::from(vec!["pending"; count])),
            Arc::new(Float32Array::from(vec![None; count])),
        ],
    )
    .map_err(Into::into)
}

#[derive(Debug, PartialEq)]
struct Row {
    group: String,
    annotation: TurnAnnotation,
    row_id: u64,
    created: u64,
    updated: u64,
}

async fn rows(dataset: &Dataset) -> Result<BTreeMap<String, Row>> {
    let mut scan = dataset.scan();
    scan.project(&[
        "labels",
        "status",
        "score",
        "id",
        "rollout_id",
        ROW_ID,
        ROW_CREATED_AT_VERSION,
        ROW_LAST_UPDATED_AT_VERSION,
    ])?;
    let batches = scan
        .try_into_stream()
        .await?
        .try_collect::<Vec<_>>()
        .await?;
    let mut result = BTreeMap::new();
    for batch in batches {
        let strings = |name| {
            batch
                .column_by_name(name)
                .unwrap()
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap()
        };
        let numbers = |name| {
            batch
                .column_by_name(name)
                .unwrap()
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap()
        };
        for i in 0..batch.num_rows() {
            let id = strings("id").value(i);
            let old = result.insert(
                id.to_owned(),
                Row {
                    group: strings("rollout_id").value(i).to_owned(),
                    annotation: decode_base_row(&batch, i, id, 4096)?,
                    row_id: numbers(ROW_ID).value(i),
                    created: numbers(ROW_CREATED_AT_VERSION).value(i),
                    updated: numbers(ROW_LAST_UPDATED_AT_VERSION).value(i),
                },
            );
            assert!(old.is_none(), "duplicate turn after native publication");
        }
    }
    assert_eq!(
        result
            .values()
            .map(|r| r.row_id)
            .collect::<BTreeSet<_>>()
            .len(),
        result.len(),
        "native append reused a live stable row ID"
    );
    Ok(result)
}

#[tokio::test]
async fn registration_batches_publish_new_and_existing_turns_together() -> Result<()> {
    for deleted in [false, true] {
        mixed(deleted).await?;
    }
    Ok(())
}

async fn mixed(deleted: bool) -> Result<()> {
    let root = tempfile::tempdir()?;
    let uri = root.path().join("annotations.lance");
    let initial = imported(0, 256)?;
    let schema = initial.schema();
    let mut base = Arc::new(
        Dataset::write(
            RecordBatchIterator::new(vec![Ok(initial)], schema),
            uri.to_str().unwrap(),
            Some(WriteParams {
                max_rows_per_file: 32,
                enable_stable_row_ids: true,
                ..Default::default()
            }),
        )
        .await?,
    );
    for field in ["id", "status"] {
        Arc::get_mut(&mut base)
            .unwrap()
            .create_index_builder(&[field], IndexType::BTree, &ScalarIndexParams::default())
            .name(format!("mixed_{field}"))
            .await?;
    }
    if deleted {
        Arc::get_mut(&mut base)
            .unwrap()
            .delete("id = 'turn-000005'")
            .await?;
    }
    let workspace = crate::MergeMemoryBudget::new(8 << 20);
    for round in 0..3 {
        let before = rows(&base).await?;
        let old_index = base.load_indices().await?;
        let old_id = old_index.iter().find(|i| i.name == "mixed_id").unwrap();
        let keys = if round == 0 {
            vec!["turn-000000".to_owned(), "turn-000001".to_owned()]
        } else {
            vec![
                "turn-000000".to_owned(),
                format!("turn-{:06}", 256 + (round - 1) * 64),
            ]
        };
        let patch = TurnAnnotationPatch {
            add_labels: BTreeSet::from(["manual".into()]),
            status: Set(format!("reviewed-{round}")),
            score: Set(0.1 * (round + 1) as f32),
            ..Default::default()
        };
        let cut = round as u64 * 2;
        let staging = AnnotationStaging::new(base.clone(), scope(), cut, Limits::default())?;
        let initial = TurnAnnotation {
            labels: BTreeSet::from(["imported".into()]),
            status: Some("pending".into()),
            score: None,
        };
        let new_keys = (256 + round * 64..256 + (round + 1) * 64)
            .map(|n| (format!("turn-{n:06}"), format!("session-{}", n / 8)))
            .collect::<Vec<_>>();
        let registrations = new_keys
            .iter()
            .map(|(turn, rollout_id)| Registration {
                turn,
                rollout_id,
                annotation: &initial,
            })
            .collect::<Vec<_>>();
        assert_eq!(
            staging
                .register_committed(cut, cut + 1, &registrations)
                .await?,
            ApplyOutcome::Applied
        );
        assert_eq!(current(&staging, &new_keys[0].0).await?, initial);
        let mutations = keys
            .iter()
            .map(|turn| Mutation {
                turn,
                patch: &patch,
            })
            .collect::<Vec<_>>();
        staging.apply_batch(cut + 1, cut + 2, &mutations).await?;
        let frozen = staging.freeze(uuid::Uuid::new_v4()).await?.unwrap();
        let prepared = frozen.prepare(&workspace, 32).await?;
        assert_eq!(prepared.merge().stats.num_inserted_rows, 64);
        assert_eq!(prepared.merge().stats.num_updated_rows, 2);
        let transaction = prepared.merge().transaction.clone();
        // No intermediate publication: the canonical table still has exactly
        // its original rows and values after both sets of files were written.
        assert_eq!(
            rows(&Dataset::open(uri.to_str().unwrap()).await?).await?,
            before
        );
        let mut commit = CommitBuilder::new(base.clone())
            .with_max_retries(0)
            .with_skip_auto_cleanup(true);
        if let Some(affected) = prepared.merge().affected_rows.clone() {
            commit = commit.with_affected_rows(affected);
        }
        let selected = Arc::new(commit.execute(transaction.clone()).await?);
        assert_eq!(selected.version_id(), base.version_id() + 1);
        assert_eq!(selected.read_transaction().await?, Some(transaction));
        staging
            .install_published(&prepared, selected.clone())
            .await?;
        drop((prepared, frozen));
        assert_eq!(staging.observation().await.materialized, cut + 2);
        assert_eq!(staging.observation().await.retained_bytes, 0);
        let reopened = Arc::new(Dataset::open(uri.to_str().unwrap()).await?);
        let after = rows(&reopened).await?;
        assert_eq!(after.len(), before.len() + 64);
        for (id, old) in &before {
            let row = &after[id];
            assert_eq!(row.row_id, old.row_id);
            assert_eq!(row.created, old.created);
            assert_eq!(row.group, old.group);
            let mut annotation = old.annotation.clone();
            if keys.contains(id) {
                patch.apply_to(&mut annotation).unwrap();
            }
            assert_eq!(row.annotation, annotation);
            assert_eq!(
                row.updated,
                if keys.contains(id) {
                    selected.version_id()
                } else {
                    old.updated
                }
            );
        }
        for n in 256 + round * 64..256 + (round + 1) * 64 {
            let row = &after[&format!("turn-{n:06}")];
            assert!(row.row_id >= base.manifest().next_row_id);
            assert_eq!(row.group, format!("session-{}", n / 8));
            assert_eq!(row.annotation.status.as_deref(), Some("pending"));
            assert_eq!(row.annotation.score, None);
            assert_eq!(row.annotation.labels, BTreeSet::from(["imported".into()]));
            assert_eq!(row.created, selected.version_id());
            assert_eq!(row.updated, selected.version_id());
        }
        let indices = reopened.load_indices().await?;
        let id = indices.iter().find(|i| i.name == "mixed_id").unwrap();
        assert_eq!(id.uuid, old_id.uuid);
        assert_eq!(id.fragment_bitmap, old_id.fragment_bitmap);
        for field in ["id", "rollout_id"] {
            let field_id = base.schema().field_id(field)?;
            for original in base.manifest().fragments.iter() {
                let updated = reopened
                    .manifest()
                    .fragments
                    .iter()
                    .find(|f| f.id == original.id)
                    .unwrap();
                let files = |f: &lance_table::format::Fragment| {
                    f.files
                        .iter()
                        .filter(|file| file.fields.contains(&field_id))
                        .map(|file| file.path.clone())
                        .collect::<Vec<_>>()
                };
                assert_eq!(files(original), files(updated));
            }
        }
        let fresh = AnnotationStaging::new(reopened.clone(), scope(), cut + 2, Limits::default())?;
        for key in [&keys[0], &keys[1], &format!("turn-{:06}", 256 + round * 64)] {
            assert_eq!(current(&fresh, key).await?, after[key].annotation);
        }
        println!(
            "{}",
            serde_json::json!({"experiment":"registration_publication", "deleted_hole":deleted, "round":round, "rows":after.len(), "version":reopened.version_id(), "fragments":reopened.fragments().len(), "inserted":64, "updated":keys.len(), "id_index_covered_fragments":id.fragment_bitmap.as_ref().unwrap().len()})
        );
        if deleted {
            assert!(fresh
                .capture("turn-000005")
                .await?
                .resolve()
                .await?
                .is_none());
        }
        base = reopened;
    }
    Ok(())
}

async fn narrow(root: &std::path::Path) -> Result<Arc<Dataset>> {
    let batch = imported(0, 1)?;
    let schema = batch.schema();
    Ok(Arc::new(
        Dataset::write(
            RecordBatchIterator::new(vec![Ok(batch)], schema),
            root.to_str().unwrap(),
            Some(WriteParams {
                enable_stable_row_ids: true,
                ..Default::default()
            }),
        )
        .await?,
    ))
}

#[tokio::test]
async fn registration_repeated_keys_preserve_first_values_across_lookup_windows() -> Result<()> {
    let root = tempfile::tempdir()?;
    let base = narrow(&root.path().join("annotations.lance")).await?;
    let keys = BTreeSet::from(["turn-000000", "new"]);
    let projected = read_existing(&base, &keys, &["id"]).await?;
    assert_eq!(projected.len(), 1);
    assert_eq!(projected["turn-000000"].num_columns(), 1);
    assert_eq!(projected["turn-000000"].schema().field(0).name(), "id");
    let staging = AnnotationStaging::new(base, scope(), 0, Limits::default())?;
    let initial = TurnAnnotation::default();
    let stale = TurnAnnotation {
        status: Some("must-not-replace".into()),
        ..Default::default()
    };
    let mut batch = vec![
        Registration {
            turn: "turn-000000",
            rollout_id: "different-session",
            annotation: &stale,
        };
        COLD_LOOKUP_KEYS + 2
    ];
    batch[1] = Registration {
        turn: "new",
        rollout_id: "original-session",
        annotation: &initial,
    };
    batch[COLD_LOOKUP_KEYS] = Registration {
        turn: "new",
        rollout_id: "different-session",
        annotation: &stale,
    };
    assert_eq!(
        staging.register_committed(0, 1, &batch).await?,
        ApplyOutcome::Applied
    );
    assert_eq!(staging.observation().await.active_entries, 1);
    assert_eq!(
        current(&staging, "turn-000000").await?.status.as_deref(),
        Some("pending")
    );
    assert_eq!(current(&staging, "new").await?, initial);
    let frozen = staging.freeze(uuid::Uuid::new_v4()).await?.unwrap();
    let prepared = frozen
        .prepare(&crate::MergeMemoryBudget::new(8 << 20), 32)
        .await?;
    let selected = commit(&prepared).await?;
    staging
        .install_published(&prepared, selected.clone())
        .await?;
    assert_eq!(rows(&selected).await?["new"].group, "original-session");
    Ok(())
}

#[tokio::test]
async fn registration_reimport_preserves_active_frozen_and_published_annotations() -> Result<()> {
    let root = tempfile::tempdir()?;
    let base = narrow(&root.path().join("annotations.lance")).await?;
    let staging = AnnotationStaging::new(base, scope(), 0, Limits::default())?;
    let before_registration = staging.capture("new").await?;
    let initial = TurnAnnotation::default();
    let registration = [Registration {
        turn: "new",
        rollout_id: "session-original",
        annotation: &initial,
    }];
    assert_eq!(
        staging.register_committed(0, 1, &registration).await?,
        ApplyOutcome::Applied
    );
    let frozen = staging.freeze(uuid::Uuid::new_v4()).await?.unwrap();
    assert_eq!(
        staging.register_committed(1, 2, &registration).await?,
        ApplyOutcome::Applied
    );
    assert_eq!(staging.observation().await.active_entries, 0);
    staging
        .apply(
            2,
            3,
            "new",
            &TurnAnnotationPatch {
                score: Set(0.9),
                ..Default::default()
            },
        )
        .await?;
    let stale = TurnAnnotation {
        status: Some("reimported".into()),
        score: Some(0.0),
        ..Default::default()
    };
    let reimport = [Registration {
        turn: "new",
        rollout_id: "different-session",
        annotation: &stale,
    }];
    assert_eq!(
        staging.register_committed(3, 4, &reimport).await?,
        ApplyOutcome::Applied
    );
    assert_eq!(current(&staging, "new").await?.score, Some(0.9));
    let held = staging.capture("new").await?;
    let budget = crate::MergeMemoryBudget::new(8 << 20);
    let prepared = frozen.prepare(&budget, 32).await?;
    assert_eq!(prepared.merge().stats.num_inserted_rows, 1);
    assert_eq!(prepared.merge().stats.num_updated_rows, 0);
    staging
        .install_published(&prepared, commit(&prepared).await?)
        .await?;
    drop((prepared, frozen));
    assert!(before_registration.resolve().await?.is_none());
    drop(before_registration);
    assert_eq!(held.resolve().await?.unwrap().score, Some(0.9));
    assert_eq!(current(&staging, "new").await?.score, Some(0.9));
    let frozen = staging.freeze(uuid::Uuid::new_v4()).await?.unwrap();
    let prepared = frozen.prepare(&budget, 32).await?;
    assert_eq!(
        prepared.merge().stats.num_inserted_rows,
        0,
        "a suffix update must not reinsert the newly published row"
    );
    assert_eq!(prepared.merge().stats.num_updated_rows, 1);
    let selected = commit(&prepared).await?;
    staging
        .install_published(&prepared, selected.clone())
        .await?;
    drop((prepared, frozen));
    let after = rows(&selected).await?;
    assert_eq!(after.len(), 2);
    assert_eq!(after["new"].group, "session-original");
    assert_eq!(after["new"].annotation.score, Some(0.9));
    assert_eq!(after["new"].annotation.status, None);
    assert_eq!(
        staging.register_committed(4, 5, &reimport).await?,
        ApplyOutcome::Applied
    );
    assert_eq!(staging.observation().await.active_entries, 0);
    assert!(staging.observation().await.retained_bytes > 0);
    drop(held);
    assert_eq!(staging.observation().await.retained_bytes, 0);
    // Request results still need a checkpoint even when every imported ID
    // already exists. The service supplies its real encoded result snapshot.
    let frozen = staging
        .freeze_with_results(uuid::Uuid::new_v4(), Some("component-checkpoint".into()))
        .await?
        .unwrap();
    let prepared = frozen.prepare(&budget, 32).await?;
    assert!(matches!(
        prepared.merge().transaction.operation,
        Operation::UpdateConfig { .. }
    ));
    let next = commit(&prepared).await?;
    staging.install_published(&prepared, next.clone()).await?;
    assert_eq!(next.manifest().fragments, selected.manifest().fragments);
    assert_eq!(rows(&next).await?, after);
    Ok(())
}

#[tokio::test]
async fn registration_metadata_does_not_change_patch_rejection_after_publication() -> Result<()> {
    let root = tempfile::tempdir()?;
    let base = narrow(&root.path().join("annotations.lance")).await?;
    let limits = Limits {
        value_bytes: 4096,
        retained_bytes: 32768,
        ..Default::default()
    };
    let staging = AnnotationStaging::new(base, scope(), 0, limits.clone())?;
    let initial = TurnAnnotation::default();
    let group = "g".repeat(3500);
    let turns = ["new-a", "new-b", "new-c"];
    let registrations = turns.map(|turn| Registration {
        turn,
        rollout_id: &group,
        annotation: &initial,
    });
    assert_eq!(
        staging.register_committed(0, 1, &registrations).await?,
        ApplyOutcome::Applied
    );
    let frozen = staging.freeze(uuid::Uuid::new_v4()).await?.unwrap();
    let patch = TurnAnnotationPatch {
        status: Set("s".repeat(2500)),
        ..Default::default()
    };
    let mutations = turns.map(|turn| Mutation {
        turn,
        patch: &patch,
    });
    // Both individual values and the batch fit their permanent limits without
    // grouping metadata. Counting that transient metadata would reject both.
    assert_eq!(
        staging.apply_committed(1, 2, &mutations).await?,
        ApplyOutcome::Applied
    );
    let budget = crate::MergeMemoryBudget::new(32768);
    let prepared = frozen.prepare(&budget, 32).await?;
    let selected = commit(&prepared).await?;
    staging
        .install_published(&prepared, selected.clone())
        .await?;
    let recovered = AnnotationStaging::new(selected, scope(), 1, limits)?;
    assert_eq!(
        recovered.apply_committed(1, 2, &mutations).await?,
        ApplyOutcome::Applied
    );
    for turn in turns {
        assert_eq!(
            current(&recovered, turn).await?,
            current(&staging, turn).await?
        );
    }
    Ok(())
}

#[tokio::test]
async fn registration_rejection_and_reader_pressure_do_not_partially_install_rows() -> Result<()> {
    let root = tempfile::tempdir()?;
    let base = narrow(&root.path().join("annotations.lance")).await?;
    let limits = Limits {
        retained_entries: 2,
        retained_bytes: 8192,
        value_bytes: 2048,
        readers: 2,
    };
    let staging = AnnotationStaging::new(base, scope(), 0, limits)?;
    let initial = TurnAnnotation::default();
    let invalid = TurnAnnotation {
        score: Some(f32::NAN),
        ..Default::default()
    };
    assert_eq!(
        staging
            .register_committed(
                0,
                1,
                &[
                    Registration {
                        turn: "A",
                        rollout_id: "group",
                        annotation: &initial
                    },
                    Registration {
                        turn: "B",
                        rollout_id: "group",
                        annotation: &invalid
                    },
                ]
            )
            .await?,
        ApplyOutcome::Rejected(Rejection {
            row: 1,
            reason: RejectionReason::InvalidPatch
        })
    );
    assert_eq!(staging.observation().await.applied, 1);
    assert_eq!(staging.observation().await.retained_entries, 0);
    assert!(staging.capture("A").await?.resolve().await?.is_none());
    let registrations = [
        Registration {
            turn: "A",
            rollout_id: "group",
            annotation: &initial,
        },
        Registration {
            turn: "B",
            rollout_id: "group",
            annotation: &initial,
        },
    ];
    assert_eq!(
        staging.register_committed(1, 2, &registrations).await?,
        ApplyOutcome::Applied
    );
    let held = [staging.capture("A").await?, staging.capture("B").await?];
    let third = [Registration {
        turn: "C",
        rollout_id: "group",
        annotation: &initial,
    }];
    assert_eq!(
        staging.register_committed(2, 3, &third).await?,
        ApplyOutcome::Backpressure
    );
    assert_eq!(staging.observation().await.applied, 2);
    let budget = crate::MergeMemoryBudget::new(8 << 20);
    let frozen = staging.freeze(uuid::Uuid::new_v4()).await?.unwrap();
    let prepared = frozen.prepare(&budget, 32).await?;
    staging
        .install_published(&prepared, commit(&prepared).await?)
        .await?;
    drop((prepared, frozen));
    assert_eq!(
        staging.register_committed(2, 3, &third).await?,
        ApplyOutcome::Backpressure
    );
    drop(held);
    assert_eq!(staging.observation().await.retained_entries, 0);
    assert_eq!(
        staging.register_committed(2, 3, &third).await?,
        ApplyOutcome::Applied
    );
    assert_eq!(
        staging
            .apply_committed(
                3,
                4,
                &[Mutation {
                    turn: "missing",
                    patch: &TurnAnnotationPatch::default()
                }]
            )
            .await?,
        ApplyOutcome::Rejected(Rejection {
            row: 0,
            reason: RejectionReason::MissingTurn
        })
    );
    let body_base = fixture(&root.path().join("body.lance")).await?;
    let body_staging = AnnotationStaging::new(body_base, scope(), 0, Limits::default())?;
    assert!(body_staging
        .register_committed(0, 1, &registrations)
        .await
        .is_err());
    assert_eq!(body_staging.observation().await.applied, 0);
    Ok(())
}
