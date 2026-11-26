use super::*;
use arrow_array::{LargeBinaryArray, RecordBatchIterator};
use arrow_schema::{DataType, Field, Schema};
use lance::dataset::{WriteMode, WriteParams};
use lance::index::DatasetIndexExt;

fn spec() -> Result<TurnIngestionSpec> {
    TurnIngestionSpec::new(
        Arc::new(Schema::new(vec![
            Field::new("session id", DataType::Utf8, false),
            Field::new("turn_id", DataType::Utf8, false),
            Field::new("body", DataType::LargeBinary, false),
        ])),
        vec!["session id".into(), "turn_id".into()],
        "session id",
    )
}

fn batch(spec: &TurnIngestionSpec, keys: &[(&str, &str)]) -> Result<RecordBatch> {
    Ok(RecordBatch::try_new(
        spec.body_schema().clone(),
        vec![
            Arc::new(StringArray::from_iter_values(keys.iter().map(|key| key.0))),
            Arc::new(StringArray::from_iter_values(keys.iter().map(|key| key.1))),
            Arc::new(LargeBinaryArray::from_iter_values(
                keys.iter().map(|_| vec![7u8; 64 << 10]),
            )),
        ],
    )?)
}

async fn write(
    spec: &TurnIngestionSpec,
    uri: &str,
    keys: &[(&str, &str)],
    mode: WriteMode,
) -> Result<Dataset> {
    Ok(Dataset::write(
        RecordBatchIterator::new(vec![Ok(batch(spec, keys)?)], spec.body_schema().clone()),
        uri,
        Some(WriteParams {
            mode,
            max_rows_per_file: 2,
            ..Default::default()
        }),
    )
    .await?)
}

#[tokio::test]
async fn composite_lookup_checks_exact_pairs_and_captured_snapshot() -> Result<()> {
    for indexed in [false, true] {
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("turns.lance");
        let uri = path.to_str().unwrap();
        let spec = spec()?;
        let mut base = write(
            &spec,
            uri,
            &[("s'1", "a"), ("s'1", "c"), ("other", "a"), ("会话", "b\\x")],
            WriteMode::Create,
        )
        .await?;
        if indexed {
            base.create_index_builder(
                &["turn_id"],
                lance_index::IndexType::BTree,
                &lance_index::scalar::ScalarIndexParams::default(),
            )
            .name("turn_key".into())
            .await?;
        }
        let base = Arc::new(base);
        let hit = spec.business_key(&["s'1", "a"], 4096)?;
        let missing = spec.business_key(&["other", "c"], 4096)?;
        let escaped = spec.business_key(&["会话", "b\\x"], 4096)?;
        let unknown = spec.business_key(&["not-requested", "a"], 4096)?;
        let budget = MergeMemoryBudget::new(1 << 20);
        let lookup = spec
            .lookup_existing(base.clone(), &[&hit, &missing, &escaped, &hit], &budget)
            .await?
            .context("lookup should be admitted")?;
        assert_eq!(lookup.get(&hit), Some(true));
        assert_eq!(lookup.get(&missing), Some(false));
        assert_eq!(lookup.get(&escaped), Some(true));
        assert_eq!(lookup.get(&unknown), None);
        assert!(lookup.matches(&base));
        assert_eq!(lookup.observation().requested_keys, 3);
        assert_eq!(lookup.observation().scans_started, 1);
        // The per-column predicate also returns (s'1,c) and (other,a).
        // Neither can turn the missing composite pair (other,c) into a hit.
        assert_eq!(lookup.observation().projected_rows, 4);
        assert!(lookup.observation().projected_array_bytes < 64 << 10);
        assert!(budget.reserved() > 0);

        let newer = Arc::new(write(&spec, uri, &[("other", "c")], WriteMode::Append).await?);
        assert!(!lookup.matches(&newer));
        assert_eq!(lookup.get(&missing), Some(false));
        drop(lookup);
        assert_eq!(budget.reserved(), 0);
        let fresh = spec
            .lookup_existing(newer.clone(), &[&missing], &budget)
            .await?
            .context("new snapshot lookup should be admitted")?;
        assert!(fresh.matches(&newer));
        assert_eq!(fresh.get(&missing), Some(true));
        drop(fresh);
        assert_eq!(budget.reserved(), 0);
    }
    Ok(())
}

#[tokio::test]
async fn admission_and_scan_errors_never_become_missing_keys() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let path = directory.path().join("turns.lance");
    let uri = path.to_str().unwrap();
    let spec = spec()?;
    let base = Arc::new(
        write(
            &spec,
            uri,
            &[
                ("session", "turn-a"),
                ("session", "turn-b"),
                ("session", "turn-c"),
                ("session", "turn-d"),
            ],
            WriteMode::Create,
        )
        .await?,
    );
    let key = spec.business_key(&["session", "turn-a"], 4096)?;
    let unavailable = spec.business_key(&["session", "turn-c"], 4096)?;
    let budget = MergeMemoryBudget::new(1 << 20);
    // One fragment stays readable, but the other requested key's file is lost.
    // Partial results cannot become negative membership. A workspace refusal
    // must return before attempting this scan at all.
    assert_eq!(base.fragments().len(), 2);
    std::fs::remove_file(path.join("data").join(&base.fragments()[1].files[0].path))?;
    let held = budget.try_reserve(budget.limit()).unwrap();
    assert!(spec
        .lookup_existing(base.clone(), &[&key, &unavailable], &budget)
        .await?
        .is_none());
    drop(held);
    let large = spec.business_key(&[&"s".repeat(60 << 10), "turn"], MAX_KEY_BYTES)?;
    assert!(spec
        .lookup_existing(base.clone(), &[&large, &large, &large], &budget)
        .await?
        .is_none());
    assert_eq!(budget.reserved(), 0);
    let noncanonical = format!("{key} ");
    let error = spec
        .lookup_existing(base.clone(), &[&noncanonical], &budget)
        .await
        .err()
        .context("noncanonical key must fail")?;
    assert!(format!("{error:#}").contains("noncanonical turn key"));
    assert_eq!(budget.reserved(), 0);

    let error = spec
        .lookup_existing(base, &[&key, &unavailable], &budget)
        .await
        .err()
        .context("missing data must be an error, not a negative lookup")?;
    assert!(error.chain().any(|cause| cause.is::<lance::Error>()));
    assert_eq!(budget.reserved(), 0);
    Ok(())
}
